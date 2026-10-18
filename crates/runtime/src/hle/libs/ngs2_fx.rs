//! NGS2 voice processing beyond mixing: sampler envelopes, the I3DL2 reverb
//! rack, the multi-tap delay module of custom racks and the submixers'
//! equalizer bands (`Biquad`).
//!
//! The envelope follows the SOTC native port's ngs2.rs (MIT): forward points
//! ramp linearly to their heights over their durations (in samples), the
//! last height holds until STOP, then the release points run and the voice
//! ends. The reverb and the delay are written here for the parameters TLG
//! sends (docs/LOG.md: Audio):
//! - Reverb racks (0x2001) are effect returns: I3DL2 wet 1.0, dry 0.0, so
//!   they output the reverberation only. A Schroeder/Moorer network (six
//!   damped combs, two allpasses per side) with an energy-normalized impulse
//!   response, its decay from DecayTime, high-frequency damping from
//!   DecayHFRatio and RoomHF, pre-delay from ReverbDelay, scaled by Room x
//!   Reverb (millibels) and wet; early reflections as one tap at
//!   ReflectionsDelay scaled by Room x Reflections.
//! - Custom racks (0x4002) whose module 0x1d is set are a multi-tap delay
//!   (early reflections / echoes): eight (gain, milliseconds) taps per
//!   channel, a level, a feedback amount and a low-pass cutoff. Also effect
//!   returns: the taps only, no dry signal (every sampler voice has its dry
//!   path into a submixer on port 0).

/// A voice envelope (SOTC port, MIT): forward points, then release points
/// from STOP on. Every curve type is linear here.
#[derive(Default, Clone, Debug)]
pub struct Envelope {
    pub points: Vec<(u32, u32, f32)>,
    pub fwd: usize,
    idx: usize,
    t: u32,
    from: f32,
    pub height: f32,
    releasing: bool,
    finished: bool,
}

impl Envelope {
    pub fn new(points: Vec<(u32, u32, f32)>, fwd: usize) -> Envelope {
        let mut e = Envelope { points, fwd, ..Default::default() };
        e.restart();
        e
    }

    pub fn restart(&mut self) {
        self.idx = 0;
        self.t = 0;
        self.from = 0.0;
        self.height = if self.points.is_empty() { 1.0 } else { 0.0 };
        self.releasing = false;
        self.finished = false;
    }

    /// Enters the release points; false when there are none.
    pub fn release(&mut self) -> bool {
        if self.points.len() <= self.fwd || self.releasing {
            return false;
        }
        self.releasing = true;
        self.idx = self.fwd;
        self.t = 0;
        self.from = self.height;
        true
    }

    pub fn done(&self) -> bool {
        self.finished
    }

    /// The height for the next output sample.
    pub fn step(&mut self) -> f32 {
        if self.points.is_empty() {
            return 1.0;
        }
        let end = if self.releasing { self.points.len() } else { self.fwd.min(self.points.len()) };
        while self.idx < end {
            let (_curve, dur, target) = self.points[self.idx];
            if self.t < dur {
                self.height = self.from + (target - self.from) * (self.t as f32 / dur as f32);
                self.t += 1;
                return self.height;
            }
            self.height = target;
            self.from = target;
            self.idx += 1;
            self.t = 0;
        }
        if self.releasing {
            self.finished = true;
            self.height = 0.0;
            return 0.0;
        }
        self.height
    }
}

/// SceNgs2ReverbI3DL2Param as TLG fills it: {wet, dry, room (mB), roomHF
/// (mB), reflectionPattern, decayTime (s), decayHFRatio, reflections (mB),
/// reflectionsDelay (s), reverb (mB), reverbDelay (s), diffusion (%),
/// density (%), hfReference (Hz)}.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct I3dl2 {
    pub wet: f32,
    pub dry: f32,
    pub room: i32,
    pub room_hf: i32,
    pub decay: f32,
    pub decay_hf: f32,
    pub reflections: i32,
    pub reflections_delay: f32,
    pub reverb: i32,
    pub reverb_delay: f32,
}

impl Default for I3dl2 {
    fn default() -> Self {
        // (Until the title sets it: wet only, a medium room.)
        I3dl2 { wet: 1.0, dry: 0.0, room: -1000, room_hf: -100, decay: 1.5, decay_hf: 0.8, reflections: -2600, reflections_delay: 0.007, reverb: 200, reverb_delay: 0.011 }
    }
}

/// Millibels to an amplitude factor.
pub fn mb(v: i32) -> f32 {
    10f32.powf(v.clamp(-10000, 2000) as f32 / 2000.0)
}

struct Comb {
    buf: Vec<f32>,
    pos: usize,
    fb: f32,
    lp: f32,
}

struct Allpass {
    buf: Vec<f32>,
    pos: usize,
}

impl Allpass {
    /// `process` frame by frame over a block no longer than the delay, in
    /// place: every value read was written before the block, so the ring's
    /// contiguous runs go in vector form (the same results).
    fn process_block(&mut self, x: &mut [f32]) {
        let n = self.buf.len();
        debug_assert!(x.len() <= n);
        let mut i = 0;
        while i < x.len() {
            let run = (n - self.pos).min(x.len() - i);
            for (b, v) in self.buf[self.pos..self.pos + run].iter_mut().zip(&mut x[i..i + run]) {
                let bv = *b;
                *b = *v + bv * 0.5;
                *v = bv - *v * 0.5;
            }
            self.pos += run;
            if self.pos == n {
                self.pos = 0;
            }
            i += run;
        }
    }
}

/// The Walsh-Hadamard transform of eight values: h[k] = sum over j of
/// (-1)^popcount(k & j) x[j] (Sylvester's rows; row 0 the plain sum).
fn fwht8(mut h: [f32; 8]) -> [f32; 8] {
    for len in [1, 2, 4] {
        for i in (0..8).step_by(2 * len) {
            for j in i..i + len {
                let (a, b) = (h[j], h[j + len]);
                h[j] = a + b;
                h[j + len] = a - b;
            }
        }
    }
    h
}

/// `(i + 1) % n` without a division (`i < n`).
fn wrap_inc(i: usize, n: usize) -> usize {
    if i + 1 == n { 0 } else { i + 1 }
}

/// The largest |x| of `v` (NaNs passed over), eight lanes at a time.
fn max_abs(v: &[f32]) -> f32 {
    let mut lanes = [0f32; 8];
    let chunks = v.chunks_exact(8);
    let rest = chunks.remainder();
    for c in chunks {
        for k in 0..8 {
            let a = c[k].abs();
            if a > lanes[k] {
                lanes[k] = a;
            }
        }
    }
    let mut m = 0f32;
    for &x in rest.iter().chain(&lanes) {
        let a = x.abs();
        if a > m {
            m = a;
        }
    }
    m
}

/// `max_abs` over frames of eight channels.
fn max_abs_frames(v: &[[f32; 8]]) -> f32 {
    let mut lanes = [0f32; 8];
    for f in v {
        for k in 0..8 {
            let a = f[k].abs();
            if a > lanes[k] {
                lanes[k] = a;
            }
        }
    }
    max_abs(&lanes)
}

/// The reverb of one reverb rack voice.
pub struct Reverb {
    p: I3dl2,
    rate: u32,
    /// Output channels with their own late reverb (`multi`): every channel
    /// but the LFE, each a different sign mix of the combs (Sylvester-Hadamard
    /// rows 1-7: mutually uncorrelated, each at the plain sum's energy) through
    /// its own allpass pair. Otherwise FL and FR only, both the plain sum.
    multi: bool,
    combs: Vec<Comb>,
    /// Allpass pairs per output slot: FL, FR (then C, SL, SR, BL, BR).
    ap: Vec<[Allpass; 2]>,
    /// Pre-delay line (mono) and its read offsets for the reflections tap and
    /// the late reverb.
    pre: Vec<f32>,
    pre_pos: usize,
    refl_delay: usize,
    late_delay: usize,
    /// Comb damping (feedback low-pass) and the input's RoomHF low-pass.
    damp: f32,
    in_coef: f32,
    in_lp: f32,
    /// The input's DC-blocking high-pass (`REVERB_HP_HZ`): previous input
    /// and output, pole.
    hp_x: f32,
    hp_y: f32,
    hp_r: f32,
    /// Output scale: energy normalization x Room x Reverb x wet.
    late_gain: f32,
    refl_gain: f32,
    /// Scratch of `process_block`: per frame the combs' mix (the plain sum,
    /// or the Hadamard rows) and the reflections; one slot's samples.
    mix: Vec<[f32; 8]>,
    refl: Vec<f32>,
    slot: Vec<f32>,
}

/// The output channels of a multichannel reverb's slots (the LFE, channel 3,
/// has none).
const REVERB_SLOTS: [usize; 7] = [0, 1, 2, 4, 5, 6, 7];

impl Reverb {
    pub fn new(p: I3dl2, rate: u32, multi: bool) -> Reverb {
        let rate = rate.max(8000);
        let scale = rate as f32 / 48000.0;
        let decay = p.decay.clamp(0.1, 20.0);
        // (Freeverb's comb lengths; the two-channel form keeps its six.)
        let lens: &[usize] = if multi { &[1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116] } else { &[1557, 1617, 1491, 1422, 1277, 1356] };
        let combs: Vec<Comb> = lens
            .iter()
            .map(|&d| {
                let len = ((d as f32 * scale) as usize).max(1);
                let t = len as f32 / rate as f32;
                Comb { buf: vec![0.0; len], pos: 0, fb: 10f32.powf(-3.0 * t / decay), lp: 0.0 }
            })
            .collect();
        let ap = |d: usize| Allpass { buf: vec![0.0; ((d as f32 * scale) as usize).max(1)], pos: 0 };
        let slots = if multi { REVERB_SLOTS.len() } else { 2 };
        let aps = (0..slots).map(|k| [ap(556 + 23 * k), ap(441 + 23 * k)]).collect();
        // Energy of the summed combs' impulse responses (unit input):
        // sum over combs of 1 / (1 - fb^2); normalized to 1 (signs in the
        // sums change nothing: the combs' outputs are uncorrelated).
        let energy: f32 = combs.iter().map(|c| 1.0 / (1.0 - c.fb * c.fb).max(1e-4)).sum();
        let norm = 1.0 / energy.sqrt();
        // High frequencies decay faster by DecayHFRatio (1 = as fast as lows).
        let damp = (1.0 - p.decay_hf.clamp(0.1, 2.0)).clamp(0.0, 0.85) * 0.7;
        // RoomHF: a gentle input low-pass (a 5 kHz reference).
        let hf = mb(p.room_hf.min(0));
        let in_coef = (1.0 - hf).clamp(0.0, 0.9);
        let refl_delay = ((p.reflections_delay.clamp(0.0, 0.3)) * rate as f32) as usize;
        let late_delay = (((p.reflections_delay + p.reverb_delay).clamp(0.0, 0.4)) * rate as f32) as usize;
        let room = mb(p.room.min(0));
        Reverb {
            p,
            rate,
            multi,
            combs,
            ap: aps,
            pre: vec![0.0; late_delay.max(refl_delay) + 2],
            pre_pos: 0,
            refl_delay,
            late_delay,
            damp,
            in_coef,
            in_lp: 0.0,
            hp_x: 0.0,
            hp_y: 0.0,
            hp_r: one_pole_r(REVERB_HP_HZ, rate),
            late_gain: norm * room * mb(p.reverb) * p.wet,
            refl_gain: room * mb(p.reflections) * p.wet,
            mix: Vec::new(),
            refl: Vec::new(),
            slot: Vec::new(),
        }
    }

    /// Whether the parameters, rate or form changed enough to rebuild.
    pub fn matches(&self, p: &I3dl2, rate: u32, multi: bool) -> bool {
        self.p == *p && self.rate == rate.max(8000) && self.multi == multi
    }

    /// Whether the reverb has died away: every value of its state under
    /// `eps`. The state is then cleared, so that silence in gives silence
    /// out bit for bit (the mixer skips it while its input stays silent).
    pub fn settle(&mut self, eps: f32) -> bool {
        let mut m = max_abs(&[self.in_lp, self.hp_x, self.hp_y]).max(max_abs(&self.pre));
        for c in &self.combs {
            m = m.max(c.lp.abs()).max(max_abs(&c.buf));
        }
        for a in self.ap.iter().flatten() {
            m = m.max(max_abs(&a.buf));
        }
        if !(m < eps) {
            return false;
        }
        (self.in_lp, self.hp_x, self.hp_y) = (0.0, 0.0, 0.0);
        self.pre.fill(0.0);
        for c in &mut self.combs {
            c.lp = 0.0;
            c.buf.fill(0.0);
        }
        for a in self.ap.iter_mut().flatten() {
            a.buf.fill(0.0);
        }
        true
    }

    /// One frame: mono input, the output channels out (multichannel: all but
    /// the LFE; otherwise FL and FR).
    pub fn process(&mut self, x: f32) -> [f32; 8] {
        let mut out = Vec::with_capacity(1);
        self.process_block(&[x], &mut out);
        out[0]
    }

    /// `process` over a block of frames (into `out`, cleared first): the
    /// input stage and the combs frame by frame (the combs' damping
    /// recursions interleaved), then each output slot's mix and allpasses in
    /// runs no longer than the shortest allpass, in vector form.
    pub fn process_block(&mut self, x: &[f32], out: &mut Vec<[f32; 8]>) {
        out.clear();
        let run = self.ap.iter().flatten().map(|a| a.buf.len()).min().unwrap_or(1).max(1);
        for xs in x.chunks(run) {
            self.run(xs, out);
        }
    }

    fn run(&mut self, xs: &[f32], out: &mut Vec<[f32; 8]>) {
        let (mut mix, mut refl, mut slot) = (std::mem::take(&mut self.mix), std::mem::take(&mut self.refl), std::mem::take(&mut self.slot));
        mix.clear();
        refl.clear();
        for &x in xs {
            // (DC and the lowest rumble kept out: at DC every comb's gain is
            // 1 / (1 - fb), ~13 times the reverb's broadband level for a 3 s
            // decay, so a low thump came back as a DC swell of the decay's
            // length.)
            let hp = x - self.hp_x + self.hp_r * self.hp_y;
            self.hp_x = x;
            self.hp_y = if hp.abs() < 1e-20 { 0.0 } else { hp };
            self.in_lp = hp * (1.0 - self.in_coef) + self.in_lp * self.in_coef;
            let n = self.pre.len();
            self.pre[self.pre_pos] = self.in_lp;
            let at = |d: usize| {
                let d = d.min(n - 1);
                if self.pre_pos >= d { self.pre_pos - d } else { self.pre_pos + n - d }
            };
            let late_in = self.pre[at(self.late_delay)];
            refl.push(self.pre[at(self.refl_delay)] * self.refl_gain);
            self.pre_pos = wrap_inc(self.pre_pos, n);
            let mut y = [0f32; 8];
            for (c, yo) in self.combs.iter_mut().zip(y.iter_mut()) {
                let v = c.buf[c.pos];
                c.lp = v * (1.0 - self.damp) + c.lp * self.damp;
                c.buf[c.pos] = late_in + c.lp * c.fb;
                c.pos = wrap_inc(c.pos, c.buf.len());
                *yo = v;
            }
            // Multichannel: the Hadamard rows (slot k takes row k + 1);
            // otherwise the plain sum (both slots).
            mix.push(if self.multi {
                fwht8(y)
            } else {
                let mut m = [0f32; 8];
                m[0] = y.iter().sum();
                m
            });
        }
        let base = out.len();
        out.resize(base + xs.len(), [0.0; 8]);
        slot.resize(xs.len(), 0.0);
        for (k, aps) in self.ap.iter_mut().enumerate() {
            let (row, ch) = if self.multi { (k + 1, REVERB_SLOTS[k]) } else { (0, k) };
            for (s, m) in slot.iter_mut().zip(&mix) {
                *s = m[row];
            }
            for a in aps.iter_mut() {
                a.process_block(&mut slot);
            }
            for ((o, &s), &r) in out[base..].iter_mut().zip(&slot).zip(&refl) {
                o[ch] = s * self.late_gain + if ch < 2 { r } else { 0.0 };
            }
        }
        (self.mix, self.refl, self.slot) = (mix, refl, slot);
    }
}

/// Corner of the reverb input's DC-blocking high-pass (Hz).
const REVERB_HP_HZ: f32 = 40.0;
/// Corner of the mastering output's DC blocker (Hz): below hearing.
const MASTER_DC_HZ: f32 = 10.0;
/// Decay time constant of the mastering limiter's peak envelope (s).
const LIMITER_RELEASE_S: f32 = 0.1;

/// Pole of a one-pole filter with corner `hz` at `rate`.
fn one_pole_r(hz: f32, rate: u32) -> f32 {
    (-std::f32::consts::TAU * hz / rate.max(8000) as f32).exp()
}

/// The mastering voice's output stage: a DC blocker per channel, then the
/// limiter (parameter {enable, threshold}) as gain reduction - a peak
/// envelope that jumps to each new peak of the frame's loudest channel and
/// decays with `LIMITER_RELEASE_S`, the gain threshold / envelope above the
/// threshold - instead of clipping each sample at the threshold (which
/// distorted every loud sound). No sample leaves above the threshold, and
/// only the peaks themselves reach it.
#[derive(Clone, Debug)]
pub struct Limiter {
    env: f32,
    rel: f32,
    dc_r: f32,
    dc_x: [f32; 8],
    dc_y: [f32; 8],
    rate: u32,
}

impl Limiter {
    pub fn new(rate: u32) -> Limiter {
        let fs = rate.max(8000) as f32;
        Limiter { env: 0.0, rel: (-1.0 / (LIMITER_RELEASE_S * fs)).exp(), dc_r: one_pole_r(MASTER_DC_HZ, rate), dc_x: [0.0; 8], dc_y: [0.0; 8], rate }
    }

    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// The gain applied at this threshold now.
    pub fn gain(&self, threshold: f32) -> f32 {
        if threshold > 0.0 && self.env > threshold { threshold / self.env } else { 1.0 }
    }

    /// One frame of 8 channels in place; `threshold` 0 = no limiting.
    pub fn process(&mut self, fr: &mut [f32; 8], threshold: f32) {
        let mut peak = 0f32;
        for c in 0..8 {
            let y = fr[c] - self.dc_x[c] + self.dc_r * self.dc_y[c];
            self.dc_x[c] = fr[c];
            self.dc_y[c] = if y.abs() < 1e-20 { 0.0 } else { y };
            fr[c] = y;
            peak = peak.max(y.abs());
        }
        self.env = peak.max(self.env * self.rel);
        if self.env < 1e-20 {
            self.env = 0.0;
        }
        let g = self.gain(threshold);
        if g < 1.0 {
            for x in fr.iter_mut() {
                *x *= g;
            }
        }
    }
}

/// The multi-tap delay of a custom rack (module 0x1d): per channel, the
/// taps' sum of delayed input (no dry signal), fed back by `feedback`, low-
/// passed at `cutoff`, scaled by `level`.
pub struct Taps {
    pub level: f32,
    pub feedback: f32,
    pub cutoff: f32,
    /// (gain, delay in samples).
    pub taps: Vec<(f32, usize)>,
    buf: Vec<[f32; 8]>,
    /// The delay line's length these taps need.
    need: usize,
    pos: usize,
    lp: [f32; 8],
    coef: f32,
    /// The feedback applied (`loop_feedback`).
    fb: f32,
}

/// The longest tap delay kept (1.5 s).
const MAX_DELAY_S: f32 = 1.5;

impl Taps {
    /// From module 0x1d's words after the header: [1] -, [2] level,
    /// [3] feedback, [4] low-pass cutoff (Hz), [5] number of taps, then
    /// (gain, milliseconds) pairs.
    pub fn from_words(w: &[u32], rate: u32) -> Option<Taps> {
        let mut t = Taps::params_from_words(w, rate)?;
        t.buf = vec![[0.0; 8]; t.need];
        Some(t)
    }

    /// The parameters of `from_words` without the delay line (empty), for
    /// `update`: the title sets them often, and a new line was up to half
    /// a megabyte allocated and zeroed each time.
    pub fn params_from_words(w: &[u32], rate: u32) -> Option<Taps> {
        let f = |i: usize| w.get(i).map(|&x| f32::from_bits(x)).unwrap_or(0.0);
        let n = (*w.get(5)? as usize).min(8);
        let rate = rate.max(8000) as f32;
        let mut taps = Vec::new();
        for k in 0..n {
            let (g, ms) = (f(6 + 2 * k), f(7 + 2 * k));
            if g != 0.0 && g.is_finite() && ms.is_finite() && ms >= 0.0 {
                taps.push((g.clamp(-2.0, 2.0), ((ms / 1000.0).min(MAX_DELAY_S) * rate) as usize));
            }
        }
        let level = if f(2).is_finite() { f(2).clamp(0.0, 4.0) } else { 0.0 };
        let feedback = if f(3).is_finite() { f(3).clamp(0.0, 0.9) } else { 0.0 };
        let cutoff = if f(4).is_finite() && f(4) > 0.0 { f(4) } else { rate / 2.0 };
        let len = taps.iter().map(|t| t.1).max().unwrap_or(0) + 2;
        let coef = if cutoff >= rate / 2.0 { 0.0 } else { (-2.0 * std::f32::consts::PI * cutoff / rate).exp() };
        let fb = loop_feedback(&taps, feedback);
        Some(Taps { level, feedback, cutoff, taps, buf: Vec::new(), need: len, pos: 0, lp: [0.0; 8], coef, fb })
    }

    /// New parameters (`params_from_words`), keeping the delay line (the
    /// title glides between environments by small steps); a longer line
    /// keeps the delayed samples in order.
    pub fn update(&mut self, new: Taps) {
        if new.need > self.buf.len() {
            let mut b = vec![[0.0; 8]; new.need];
            let n = self.buf.len();
            for i in 0..n {
                b[i] = self.buf[(self.pos + i) % n];
            }
            self.buf = b;
            self.pos = n % self.buf.len();
        }
        self.level = new.level;
        self.feedback = new.feedback;
        self.cutoff = new.cutoff;
        self.taps = new.taps;
        self.coef = new.coef;
        self.fb = new.fb;
    }

    /// Whether the delay has died away (every value of its line and filter
    /// under `eps`; then cleared, as `Reverb::settle`).
    pub fn settle(&mut self, eps: f32) -> bool {
        if !(max_abs(&self.lp).max(max_abs_frames(&self.buf)) < eps) {
            return false;
        }
        self.lp = [0.0; 8];
        self.buf.fill([0.0; 8]);
        true
    }

    /// One frame of `ch` channels.
    pub fn process(&mut self, x: &[f32; 8], ch: usize) -> [f32; 8] {
        let n = self.buf.len();
        let mut out = [0f32; 8];
        for &(g, d) in &self.taps {
            // ((pos + n - d) mod n without a division: pos < n, d < n.)
            let i = self.pos + n - d.min(n - 1);
            let fr = &self.buf[if i >= n { i - n } else { i }];
            for c in 0..ch {
                out[c] += g * fr[c];
            }
        }
        let fb = self.fb;
        let mut w = [0f32; 8];
        for c in 0..ch {
            self.lp[c] = out[c] * (1.0 - self.coef) + self.lp[c] * self.coef;
            out[c] = self.lp[c] * self.level;
            w[c] = x[c] + self.lp[c] * fb;
        }
        self.buf[self.pos] = w;
        self.pos = wrap_inc(self.pos, n);
        out
    }
}

/// The feedback of a multi-tap delay: loop gain kept below 0.7 (the taps'
/// sum times the feedback).
fn loop_feedback(taps: &[(f32, usize)], feedback: f32) -> f32 {
    let sum: f32 = taps.iter().map(|t| t.0.abs()).sum();
    if sum > 0.0 { feedback.min(0.7 / sum) } else { 0.0 }
}

/// Filter shapes of the sampler and submixer filter parameters (their type
/// word), as TLG's settings read: every submixer carries a four-band
/// equalizer (band 0 type 6 at 80-150 Hz, bands 1-2 type 8 at 280-2000 Hz
/// with gains 0.72-1.19, band 3 type 7 at 1-24 kHz or type 1 at 4.5-7 kHz;
/// gains mostly 1); samplers also use type 2 at 50-100 Hz (gain 1: a rumble
/// high-pass).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FilterKind {
    LowPass,
    HighPass,
    LowShelf,
    HighShelf,
    Peaking,
}

impl FilterKind {
    pub fn from_type(t: u32) -> Option<FilterKind> {
        match t {
            1 => Some(FilterKind::LowPass),
            2 => Some(FilterKind::HighPass),
            6 => Some(FilterKind::LowShelf),
            7 => Some(FilterKind::HighShelf),
            8 => Some(FilterKind::Peaking),
            _ => None,
        }
    }
}

/// One equalizer band: a biquad (RBJ cookbook) with its state per channel.
/// `level` is a linear gain (the shelf's or peak's gain, the pass-band gain
/// of the low- and high-pass).
#[derive(Clone, Debug)]
pub struct Biquad {
    pub kind: FilterKind,
    pub fc: f32,
    pub q: f32,
    pub level: f32,
    b: [f32; 3],
    a: [f32; 2],
    z: [[f32; 2]; 8],
}

impl Biquad {
    pub fn new(kind: FilterKind, fc: f32, q: f32, level: f32, rate: u32) -> Biquad {
        let mut f = Biquad { kind, fc, q, level, b: [1.0, 0.0, 0.0], a: [0.0, 0.0], z: [[0.0; 2]; 8] };
        f.design(rate);
        f
    }

    /// New parameters, keeping the state (no click when they glide).
    pub fn set(&mut self, kind: FilterKind, fc: f32, q: f32, level: f32, rate: u32) {
        (self.kind, self.fc, self.q, self.level) = (kind, fc, q, level);
        self.design(rate);
    }

    /// No effect at these parameters (a shelf or peak at unity gain).
    pub fn is_flat(&self) -> bool {
        !matches!(self.kind, FilterKind::LowPass | FilterKind::HighPass) && (self.level - 1.0).abs() < 1e-3
    }

    /// No state left: silence in gives silence out.
    pub fn at_rest(&self) -> bool {
        self.z.iter().all(|z| z[0] == 0.0 && z[1] == 0.0)
    }

    fn design(&mut self, rate: u32) {
        let fs = rate.max(1) as f32;
        let fc = self.fc.clamp(10.0, fs * 0.49);
        let q = self.q.clamp(0.1, 20.0);
        let w0 = std::f32::consts::TAU * fc / fs;
        let (sn, cs) = w0.sin_cos();
        let alpha = sn / (2.0 * q);
        let lvl = self.level.clamp(1e-4, 100.0);
        let a = lvl.sqrt(); // RBJ A = 10^(dB/40)
        let (b, aa) = match self.kind {
            FilterKind::LowPass => {
                let k = (1.0 - cs) / 2.0 * lvl;
                ([k, 2.0 * k, k], [1.0 + alpha, -2.0 * cs, 1.0 - alpha])
            }
            FilterKind::HighPass => {
                let k = (1.0 + cs) / 2.0 * lvl;
                ([k, -2.0 * k, k], [1.0 + alpha, -2.0 * cs, 1.0 - alpha])
            }
            FilterKind::Peaking => ([1.0 + alpha * a, -2.0 * cs, 1.0 - alpha * a], [1.0 + alpha / a, -2.0 * cs, 1.0 - alpha / a]),
            FilterKind::LowShelf => {
                let r = 2.0 * a.sqrt() * alpha;
                (
                    [a * ((a + 1.0) - (a - 1.0) * cs + r), 2.0 * a * ((a - 1.0) - (a + 1.0) * cs), a * ((a + 1.0) - (a - 1.0) * cs - r)],
                    [(a + 1.0) + (a - 1.0) * cs + r, -2.0 * ((a - 1.0) + (a + 1.0) * cs), (a + 1.0) + (a - 1.0) * cs - r],
                )
            }
            FilterKind::HighShelf => {
                let r = 2.0 * a.sqrt() * alpha;
                (
                    [a * ((a + 1.0) + (a - 1.0) * cs + r), -2.0 * a * ((a - 1.0) + (a + 1.0) * cs), a * ((a + 1.0) + (a - 1.0) * cs - r)],
                    [(a + 1.0) - (a - 1.0) * cs + r, 2.0 * ((a - 1.0) - (a + 1.0) * cs), (a + 1.0) - (a - 1.0) * cs - r],
                )
            }
        };
        let a0 = aa[0];
        self.b = [b[0] / a0, b[1] / a0, b[2] / a0];
        self.a = [aa[1] / a0, aa[2] / a0];
    }

    /// Filters channels 0..ch of every frame in place (transposed direct
    /// form II). All eight lanes run together (vector operations: one
    /// recurrence chain per frame instead of one per channel); lanes from
    /// `ch` on keep their samples and their state. (The mixer runs a
    /// voice's bands together: `process_bands`.)
    #[cfg(test)]
    pub fn process(&mut self, frames: &mut [[f32; 8]], ch: usize) {
        Biquad::process_bands(&mut [self], frames, ch);
    }

    /// `process` as before `process_bands` (`tune ngs2_vec` 0): the first
    /// `ch` lanes copied in and out of each frame, one band per pass.
    pub fn process_one(&mut self, frames: &mut [[f32; 8]], ch: usize) {
        let ch = ch.min(8);
        if ch == 0 {
            return;
        }
        let [b0, b1, b2] = self.b;
        let [a1, a2] = self.a;
        let (mut z1, mut z2) = ([0f32; 8], [0f32; 8]);
        for c in 0..ch {
            (z1[c], z2[c]) = (self.z[c][0], self.z[c][1]);
        }
        for fr in frames.iter_mut() {
            let mut x = [0f32; 8];
            x[..ch].copy_from_slice(&fr[..ch]);
            let mut y = [0f32; 8];
            for c in 0..8 {
                y[c] = b0 * x[c] + z1[c];
                z1[c] = b1 * x[c] - a1 * y[c] + z2[c];
                z2[c] = b2 * x[c] - a2 * y[c];
            }
            fr[..ch].copy_from_slice(&y[..ch]);
        }
        for c in 0..ch {
            self.z[c] = [if z1[c].abs() < 1e-20 { 0.0 } else { z1[c] }, if z2[c].abs() < 1e-20 { 0.0 } else { z2[c] }];
        }
    }

    /// Filters channels 0..ch of every frame in place through `bands` in
    /// turn, bit for bit as `process` band after band, in one pass over the
    /// frames for up to four bands: each band's recurrence is a chain of
    /// dependent operations per frame (latency-bound), and the chains of
    /// bands run together overlap.
    pub fn process_bands(bands: &mut [&mut Biquad], frames: &mut [[f32; 8]], ch: usize) {
        let ch = ch.min(8);
        if ch == 0 {
            return;
        }
        for group in bands.chunks_mut(4) {
            match group.len() {
                1 => run_bands::<1>(group, frames, ch),
                2 => run_bands::<2>(group, frames, ch),
                3 => run_bands::<3>(group, frames, ch),
                _ => run_bands::<4>(group, frames, ch),
            }
        }
    }
}

/// Bit masks of the lanes below `n` of an eight-lane frame (all ones there,
/// zero above): lanes picked by masks keep vector form (a copy of the first
/// `n` lanes was a library call per frame, a loop over them ran lane by
/// lane).
pub fn lane_mask(n: usize) -> [u32; 8] {
    let mut m = [0u32; 8];
    for (c, x) in m.iter_mut().enumerate() {
        if c < n {
            *x = !0;
        }
    }
    m
}

/// `N` biquads (`group`, in order) over channels 0..ch of `frames`.
fn run_bands<const N: usize>(group: &mut [&mut Biquad], frames: &mut [[f32; 8]], ch: usize) {
    let mut coef = [([0f32; 3], [0f32; 2]); N];
    // (Per band: z1 and z2 by lane; unused lanes start at zero, as in a
    // band's own pass.)
    let mut z = [[[0f32; 8]; 2]; N];
    for k in 0..N {
        coef[k] = (group[k].b, group[k].a);
        for c in 0..ch {
            (z[k][0][c], z[k][1][c]) = (group[k].z[c][0], group[k].z[c][1]);
        }
    }
    let m = lane_mask(ch);
    for fr in frames.iter_mut() {
        // (Unused lanes as zeros: no denormal or NaN arithmetic there.)
        let mut x = [0f32; 8];
        for c in 0..8 {
            x[c] = f32::from_bits(fr[c].to_bits() & m[c]);
        }
        for k in 0..N {
            let ([b0, b1, b2], [a1, a2]) = coef[k];
            let mut y = [0f32; 8];
            for c in 0..8 {
                y[c] = b0 * x[c] + z[k][0][c];
                z[k][0][c] = b1 * x[c] - a1 * y[c] + z[k][1][c];
                z[k][1][c] = b2 * x[c] - a2 * y[c];
            }
            x = y;
        }
        for c in 0..8 {
            fr[c] = f32::from_bits((x[c].to_bits() & m[c]) | (fr[c].to_bits() & !m[c]));
        }
    }
    for k in 0..N {
        for c in 0..ch {
            // (Denormals after silence.)
            let (z1, z2) = (z[k][0][c], z[k][1][c]);
            group[k].z[c] = [if z1.abs() < 1e-20 { 0.0 } else { z1 }, if z2.abs() < 1e-20 { 0.0 } else { z2 }];
        }
    }
}

/// A submixer compressor (parameter 0x20000002, words {index, threshold,
/// slope, -, attack s, release s, gain, detector window ms}) as TLG's values
/// read: above the threshold (linear) the level rises by `slope` dB per dB
/// (0.5 = 2:1, 1 = none); the gain then applies (it matches the reduction
/// at the bus's typical level: e.g. threshold -21 dB, slope 0.87, gain +2 dB).
/// The detector follows the loudest channel (RMS over the window, peak when
/// it is 0) with the attack and release times.
#[derive(Clone, Debug)]
pub struct Compressor {
    thr: f32,
    slope: f32,
    gain: f32,
    att: f32,
    rel: f32,
    ms_coef: f32,
    ms: f32,
    env: f32,
}

impl Compressor {
    pub fn new(words: &[u32], rate: u32) -> Compressor {
        let mut c = Compressor { thr: 1.0, slope: 1.0, gain: 1.0, att: 0.0, rel: 0.0, ms_coef: 0.0, ms: 0.0, env: 0.0 };
        c.set(words, rate);
        c
    }

    /// New parameters, keeping the detector state.
    pub fn set(&mut self, w: &[u32], rate: u32) {
        let f = |i: usize| w.get(i).map_or(0.0, |&x| f32::from_bits(x));
        let fs = rate.max(1) as f32;
        let coef = |secs: f32| if secs > 0.0 { (-1.0 / (secs * fs)).exp() } else { 0.0 };
        self.thr = f(1).clamp(1e-6, 1.0);
        self.slope = f(2).clamp(0.0, 1.0);
        self.att = coef(f(4).clamp(0.0, 5.0));
        self.rel = coef(f(5).clamp(0.0, 10.0));
        self.gain = f(6).clamp(0.0, 16.0);
        let window_ms = w.get(7).copied().unwrap_or(0).min(10_000);
        self.ms_coef = coef(window_ms as f32 / 1000.0);
    }

    /// No effect at these parameters.
    pub fn is_flat(&self) -> bool {
        (self.slope >= 1.0 || self.thr >= 1.0) && (self.gain - 1.0).abs() < 1e-3
    }

    /// No detector state left (on silence its output is silence anyway; the
    /// state decides how it meets the next sound).
    pub fn at_rest(&self) -> bool {
        self.ms == 0.0 && self.env == 0.0
    }

    /// Compresses channels 0..ch of every frame in place.
    pub fn process(&mut self, frames: &mut [[f32; 8]], ch: usize) {
        let ch = ch.min(8);
        for fr in frames.iter_mut() {
            let mut sq = 0f32;
            for &x in &fr[..ch] {
                sq = sq.max(x * x);
            }
            let level = if self.ms_coef > 0.0 {
                self.ms = self.ms_coef * self.ms + (1.0 - self.ms_coef) * sq;
                self.ms.sqrt()
            } else {
                sq.sqrt()
            };
            let k = if level > self.env { self.att } else { self.rel };
            self.env = k * self.env + (1.0 - k) * level;
            if self.env < 1e-20 {
                self.env = 0.0;
            }
            let g = if self.env > self.thr { (self.thr / self.env).powf(1.0 - self.slope) } else { 1.0 } * self.gain;
            for x in &mut fr[..ch] {
                *x *= g;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compressor_slope_and_gain() {
        // Threshold 0.1 (-20 dB), 2:1, fast peak detector, gain 1: a steady
        // 1.0 (+20 dB over) comes out 10 dB over the threshold (0.316).
        let w = [0, 0.1f32.to_bits(), 0.5f32.to_bits(), 0, 0.001f32.to_bits(), 0.001f32.to_bits(), 1.0f32.to_bits(), 0];
        let mut c = Compressor::new(&w, 48000);
        let mut fr = vec![[1.0f32, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; 4800];
        c.process(&mut fr, 2);
        assert!((fr[4799][0] - 0.316).abs() < 0.005, "{}", fr[4799][0]);
        assert!((fr[4799][1] + 0.316).abs() < 0.005);
        // Below the threshold only the gain applies.
        let w = [0, 0.5f32.to_bits(), 0.5f32.to_bits(), 0, 0.001f32.to_bits(), 0.001f32.to_bits(), 2.0f32.to_bits(), 36];
        let mut c = Compressor::new(&w, 48000);
        let mut fr = vec![[0.1f32; 8]; 4800];
        c.process(&mut fr, 1);
        assert!((fr[4799][0] - 0.2).abs() < 1e-4);
        let off = [0, 1.0f32.to_bits(), 1.0f32.to_bits(), 0, 0.12f32.to_bits(), 0.3f32.to_bits(), 1.0f32.to_bits(), 0];
        assert!(Compressor::new(&off, 48000).is_flat());
    }

    /// Steady-state amplitude of a unit sine at `hz` through the filter
    /// (from the RMS of the second half second: whole periods at these
    /// frequencies).
    fn gain_at(f: &mut Biquad, hz: f32) -> f32 {
        let n = 48000;
        let mut fr: Vec<[f32; 8]> = (0..n).map(|i| {
            let mut x = [0f32; 8];
            x[0] = (std::f64::consts::TAU * hz as f64 * i as f64 / 48000.0).sin() as f32;
            x
        }).collect();
        f.process(&mut fr, 1);
        let ms = fr[n / 2..].iter().map(|x| (x[0] as f64).powi(2)).sum::<f64>() / (n / 2) as f64;
        (2.0 * ms).sqrt() as f32
    }

    #[test]
    fn equalizer_bands() {
        let lp = |fc| Biquad::new(FilterKind::LowPass, fc, 0.707, 1.0, 48000);
        assert!((gain_at(&mut lp(1000.0), 100.0) - 1.0).abs() < 0.02);
        assert!(gain_at(&mut lp(1000.0), 10000.0) < 0.02);
        let mut hp = Biquad::new(FilterKind::HighPass, 100.0, 0.5, 1.0, 48000);
        assert!(gain_at(&mut hp, 20.0) < 0.1);
        assert!((gain_at(&mut hp, 4000.0) - 1.0).abs() < 0.02);
        let mut hs = Biquad::new(FilterKind::HighShelf, 1000.0, 0.707, 0.25, 48000);
        assert!((gain_at(&mut hs, 100.0) - 1.0).abs() < 0.03);
        assert!((gain_at(&mut hs, 12000.0) - 0.25).abs() < 0.02);
        let mut ls = Biquad::new(FilterKind::LowShelf, 200.0, 0.707, 2.0, 48000);
        assert!((gain_at(&mut ls, 30.0) - 2.0).abs() < 0.05);
        assert!((gain_at(&mut ls, 8000.0) - 1.0).abs() < 0.02);
        let mut pk = Biquad::new(FilterKind::Peaking, 1000.0, 1.0, 0.5, 48000);
        assert!((gain_at(&mut pk, 1000.0) - 0.5).abs() < 0.02);
        assert!((gain_at(&mut pk, 50.0) - 1.0).abs() < 0.02);
        let flat = Biquad::new(FilterKind::Peaking, 2000.0, 0.5, 1.0, 48000);
        assert!(flat.is_flat());
        assert!(!lp(24000.0).is_flat());
    }

    #[test]
    fn biquad_lanes_match_per_channel() {
        // The all-lanes filter gives the per-channel recurrence's samples
        // bit for bit and leaves the lanes past `ch` alone.
        let mut seed = 0x1234_5678u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for ch in 1..=8 {
            let mut f = Biquad::new(FilterKind::Peaking, 800.0, 0.9, 1.7, 48000);
            let (b, a) = (f.b, f.a);
            let mut z = [[0f32; 2]; 8];
            for _block in 0..3 {
                let input: Vec<[f32; 8]> = (0..97).map(|_| std::array::from_fn(|_| rnd())).collect();
                let mut want = input.clone();
                for c in 0..ch {
                    let [mut z1, mut z2] = z[c];
                    for fr in want.iter_mut() {
                        let x = fr[c];
                        let y = b[0] * x + z1;
                        z1 = b[1] * x - a[0] * y + z2;
                        z2 = b[2] * x - a[1] * y;
                        fr[c] = y;
                    }
                    z[c] = [z1, z2];
                }
                let mut got = input.clone();
                f.process(&mut got, ch);
                for (g, w) in got.iter().zip(&want) {
                    for c in 0..8 {
                        assert_eq!(g[c].to_bits(), w[c].to_bits(), "ch {ch} lane {c}");
                    }
                }
            }
        }
    }

    #[test]
    fn bands_in_one_pass_match_band_by_band() {
        // Up to eight bands in one pass (groups of four) give the
        // per-channel recurrences band after band bit for bit, keep each
        // band's state across grains and leave the lanes past `ch` alone.
        let mut seed = 0x8765_4321u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let kinds = [FilterKind::HighPass, FilterKind::Peaking, FilterKind::LowShelf, FilterKind::LowPass, FilterKind::HighShelf];
        for n in 1..=8 {
            for ch in 1..=8 {
                let mut bands: Vec<Biquad> = (0..n).map(|k| Biquad::new(kinds[k % kinds.len()], 150.0 + 900.0 * k as f32, 0.8, 1.4, 48000)).collect();
                let mut one_by_one = bands.clone();
                let coef: Vec<([f32; 3], [f32; 2])> = bands.iter().map(|f| (f.b, f.a)).collect();
                let mut z = vec![[[0f32; 2]; 8]; n];
                for _grain in 0..3 {
                    let input: Vec<[f32; 8]> = (0..61).map(|_| std::array::from_fn(|_| rnd())).collect();
                    let mut want = input.clone();
                    for k in 0..n {
                        let (b, a) = coef[k];
                        for c in 0..ch {
                            let [mut z1, mut z2] = z[k][c];
                            for fr in want.iter_mut() {
                                let x = fr[c];
                                let y = b[0] * x + z1;
                                z1 = b[1] * x - a[0] * y + z2;
                                z2 = b[2] * x - a[1] * y;
                                fr[c] = y;
                            }
                            z[k][c] = [z1, z2];
                        }
                    }
                    let mut got = input.clone();
                    let mut refs: Vec<&mut Biquad> = bands.iter_mut().collect();
                    Biquad::process_bands(&mut refs, &mut got, ch);
                    // (And the band-by-band passes `tune ngs2_vec` 0 keeps.)
                    let mut old = input.clone();
                    for f in one_by_one.iter_mut() {
                        f.process_one(&mut old, ch);
                    }
                    for ((g, o), w) in got.iter().zip(&old).zip(&want) {
                        for c in 0..8 {
                            assert_eq!(g[c].to_bits(), w[c].to_bits(), "{n} bands, ch {ch}, lane {c}");
                            assert_eq!(o[c].to_bits(), w[c].to_bits(), "{n} bands one by one, ch {ch}, lane {c}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn envelope_attack_hold_release() {
        // Up to 1.0 over 4 samples, hold; release to 0 over 2.
        let mut e = Envelope::new(vec![(0, 4, 1.0), (0, 2, 0.0)], 1);
        let a: Vec<f32> = (0..6).map(|_| e.step()).collect();
        assert_eq!(a, vec![0.0, 0.25, 0.5, 0.75, 1.0, 1.0]);
        assert!(e.release());
        let r: Vec<f32> = (0..3).map(|_| e.step()).collect();
        assert_eq!(r, vec![1.0, 0.5, 0.0]);
        assert!(e.done());
        let mut none = Envelope::new(Vec::new(), 0);
        assert_eq!(none.step(), 1.0);
        assert!(!none.release());
    }

    #[test]
    fn reverb_is_wet_only_and_decays() {
        let p = I3dl2 { wet: 1.0, dry: 0.0, room: 0, room_hf: 0, decay: 1.0, decay_hf: 1.0, reflections: -10000, reflections_delay: 0.01, reverb: 0, reverb_delay: 0.0 };
        let mut r = Reverb::new(p, 48000, false);
        // An impulse: no output at once (wet only), energy near 1, and the
        // last 0.5 s far below the first.
        let first = r.process(1.0);
        assert!(first[0].abs() < 1e-6 && first[1].abs() < 1e-6);
        let mut e = [0f64; 2];
        let mut early = 0f64;
        let mut late = 0f64;
        for i in 1..96000 {
            let y = r.process(0.0);
            for k in 0..2 {
                e[k] += (y[k] as f64).powi(2);
            }
            if i < 24000 {
                early += (y[0] as f64).powi(2);
            } else if i >= 72000 {
                late += (y[0] as f64).powi(2);
            }
        }
        assert!(e[0] > 0.3 && e[0] < 3.0, "energy {e:?}");
        assert!(late < early * 1e-3, "early {early} late {late}");
    }

    #[test]
    fn reverb_blocks_match_frame_by_frame() {
        // Grains (and odd lengths, longer than an allpass) give the frame-by-
        // frame results bit for bit, in both forms.
        let p = I3dl2 { wet: 1.0, dry: 0.0, room: -200, room_hf: -300, decay: 2.7, decay_hf: 0.7, reflections: -1200, reflections_delay: 0.02, reverb: 500, reverb_delay: 0.03 };
        let mut x = 0x1357_2468u32;
        let input: Vec<f32> = (0..6000)
            .map(|i| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                if i % 1500 < 400 { (x as i32) as f32 / 2.2e9 } else { 0.0 }
            })
            .collect();
        for multi in [false, true] {
            let mut a = Reverb::new(p, 48000, multi);
            let want: Vec<[f32; 8]> = input.iter().map(|&v| a.process(v)).collect();
            let mut b = Reverb::new(p, 48000, multi);
            let mut got = Vec::new();
            let mut out = Vec::new();
            for chunk in input.chunks(700).flat_map(|c| c.chunks(256)) {
                b.process_block(chunk, &mut out);
                got.extend_from_slice(&out);
            }
            assert_eq!(got.len(), want.len());
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(g.iter().zip(w).all(|(p, q)| p.to_bits() == q.to_bits()), "multi {multi} frame {i}");
            }
        }
    }

    #[test]
    fn multichannel_reverb_fills_every_channel_but_the_lfe() {
        // Noise in: each output channel but the LFE carries the late reverb
        // at about the two-channel form's energy, the channels mutually
        // uncorrelated (|r| < 0.1), the LFE silent.
        let p = I3dl2 { wet: 1.0, dry: 0.0, room: 0, room_hf: 0, decay: 2.0, decay_hf: 0.8, reflections: -10000, reflections_delay: 0.01, reverb: 0, reverb_delay: 0.02 };
        let (mut multi, mut two) = (Reverb::new(p, 48000, true), Reverb::new(p, 48000, false));
        let mut x = 0x2468_ace1u32;
        let (mut e, mut e2, mut cross) = ([0f64; 8], 0f64, [[0f64; 8]; 8]);
        for i in 0..48000 * 4 {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let s = (x as i32) as f32 / 2.2e9;
            let (y, y2) = (multi.process(s), two.process(s));
            if i < 48000 {
                continue;
            }
            e2 += (y2[0] as f64).powi(2);
            for a in 0..8 {
                e[a] += (y[a] as f64).powi(2);
                for b in 0..8 {
                    cross[a][b] += y[a] as f64 * y[b] as f64;
                }
            }
        }
        assert_eq!(e[3], 0.0);
        for a in REVERB_SLOTS {
            let db = 10.0 * (e[a] / e2).log10();
            assert!(db.abs() < 1.5, "channel {a}: {db:.2} dB against the two-channel form");
            for b in REVERB_SLOTS.into_iter().filter(|&b| b > a) {
                let r = cross[a][b] / (e[a] * e[b]).sqrt();
                assert!(r.abs() < 0.1, "channels {a} and {b}: correlation {r:.3}");
            }
        }
    }

    #[test]
    fn reverb_passes_no_dc() {
        // A constant input (DC) leaves no lasting output: the combs used to
        // hold ~13 times the broadband gain at DC.
        let mut r = Reverb::new(I3dl2 { wet: 1.0, dry: 0.0, room: 0, room_hf: 0, decay: 3.0, decay_hf: 0.8, reflections: 0, reflections_delay: 0.02, reverb: 0, reverb_delay: 0.04 }, 48000, false);
        let mut last = 0f32;
        for i in 0..48000 * 3 {
            let l = r.process(1.0)[0];
            if i >= 48000 * 5 / 2 {
                last = last.max(l.abs());
            }
        }
        assert!(last < 0.01, "lasting output {last}");
        // (Speech-band content still reverberates.)
        let mut r = Reverb::new(I3dl2 { wet: 1.0, dry: 0.0, room: 0, room_hf: 0, decay: 3.0, decay_hf: 0.8, reflections: 0, reflections_delay: 0.02, reverb: 0, reverb_delay: 0.04 }, 48000, false);
        let mut e = 0f32;
        for i in 0..48000 {
            let x = if i < 4800 { (i as f32 * std::f32::consts::TAU * 440.0 / 48000.0).sin() } else { 0.0 };
            let l = r.process(x)[0];
            if i >= 9600 {
                e += l * l;
            }
        }
        assert!(e > 1.0, "tail energy {e}");
    }

    #[test]
    fn limiter_reduces_gain_instead_of_clipping() {
        let mut l = Limiter::new(48000);
        // A 1 kHz tone at twice the threshold: no sample over it, and the
        // waveform's shape kept - no flat tops (runs of 3 or more equal
        // samples; clipping at the threshold makes most of each cycle one).
        let (mut flat, mut run) = (0, 0);
        let mut prev = 0f32;
        let mut over = 0f32;
        for i in 0..4800 {
            let x = 2.0 * (i as f32 * std::f32::consts::TAU * 1000.0 / 48000.0).sin();
            let mut fr = [x, -x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            l.process(&mut fr, 0.5);
            over = over.max(fr[0].abs() - 0.5);
            run = if i > 480 && (fr[0] - prev).abs() < 1e-7 && fr[0].abs() > 0.4 { run + 1 } else { 0 };
            if run == 2 {
                flat += 1;
            }
            prev = fr[0];
        }
        assert!(over <= 1e-6, "over the threshold by {over}");
        assert!(flat == 0, "{flat} flat tops");
        // Below the threshold the gain recovers toward 1.
        for _ in 0..48000 {
            let mut fr = [0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            l.process(&mut fr, 0.5);
        }
        assert!(l.gain(0.5) > 0.99, "gain {}", l.gain(0.5));
        // DC is removed.
        let mut fr = [0.0; 8];
        for _ in 0..48000 {
            fr = [0.3; 8];
            l.process(&mut fr, 0.0);
        }
        assert!(fr[0].abs() < 0.01, "dc {}", fr[0]);
    }

    #[test]
    fn taps_delay_without_dry() {
        // Words: [0] -, [1] -, [2] level 1.0, [3] feedback 0, [4] cutoff 24 kHz,
        // [5] 2 taps, (0.5 @ 1 ms), (0.25 @ 2 ms).
        let w = [0, 1.0f32.to_bits(), 1.0f32.to_bits(), 0, 24000.0f32.to_bits(), 2, 0.5f32.to_bits(), 1.0f32.to_bits(), 0.25f32.to_bits(), 2.0f32.to_bits()];
        let mut t = Taps::from_words(&w, 48000).unwrap();
        assert_eq!(t.taps, vec![(0.5, 48), (0.25, 96)]);
        let mut x = [0f32; 8];
        x[0] = 1.0;
        let y0 = t.process(&x, 2);
        assert_eq!(y0[0], 0.0, "no dry signal");
        let z = [0f32; 8];
        let ys: Vec<f32> = (1..200).map(|_| t.process(&z, 2)[0]).collect();
        assert!((ys[47] - 0.5).abs() < 1e-6, "{}", ys[47]);
        assert!((ys[95] - 0.25).abs() < 1e-6, "{}", ys[95]);
        assert!(ys.iter().enumerate().all(|(i, &v)| i == 47 || i == 95 || v == 0.0));
    }

    #[test]
    fn taps_match_the_modulo_form() {
        // Feedback, a low-pass and three taps (one at the longest delay):
        // bit for bit what the per-frame modulo and feedback sum gave.
        let w = [0, 0, 0.8f32.to_bits(), 0.6f32.to_bits(), 3000.0f32.to_bits(), 3, 0.5f32.to_bits(), 1.0f32.to_bits(), (-0.3f32).to_bits(), 2.5f32.to_bits(), 0.25f32.to_bits(), 4.0f32.to_bits()];
        let mut t = Taps::from_words(&w, 48000).unwrap();
        let (n, coef, level) = (t.buf.len(), t.coef, t.level);
        let (mut buf, mut pos, mut lp) = (vec![[0f32; 8]; n], 0usize, [0f32; 8]);
        let mut x = 0x2545_f491u32;
        for _ in 0..2000 {
            let fr: [f32; 8] = std::array::from_fn(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (x as i32) as f32 / 2e9
            });
            let mut want = [0f32; 8];
            for &(g, d) in &t.taps {
                let f = &buf[(pos + n - d.min(n - 1)) % n];
                for c in 0..3 {
                    want[c] += g * f[c];
                }
            }
            let sum: f32 = t.taps.iter().map(|t| t.0.abs()).sum();
            let fb = if sum > 0.0 { t.feedback.min(0.7 / sum) } else { 0.0 };
            let mut wr = [0f32; 8];
            for c in 0..3 {
                lp[c] = want[c] * (1.0 - coef) + lp[c] * coef;
                want[c] = lp[c] * level;
                wr[c] = fr[c] + lp[c] * fb;
            }
            buf[pos] = wr;
            pos = (pos + 1) % n;
            let got = t.process(&fr, 3);
            assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    #[test]
    fn effects_settle_after_silence() {
        // A reverb and a delay after an impulse: not settled while their
        // tails ring, settled within seconds of silence and then cleared:
        // silence in, silence out.
        let mut r = Reverb::new(I3dl2::default(), 48000, false);
        r.process(1.0);
        for _ in 0..4800 {
            r.process(0.0);
        }
        assert!(!r.settle(1e-9));
        let mut n = 4800;
        while !r.settle(1e-9) {
            for _ in 0..4800 {
                r.process(0.0);
            }
            n += 4800;
            assert!(n < 48000 * 30, "the reverb never settled");
        }
        for _ in 0..48000 {
            assert_eq!(r.process(0.0), [0.0; 8]);
        }
        let words = [0, 0, 0.5f32.to_bits(), 0.3f32.to_bits(), 8000f32.to_bits(), 2, 0.6f32.to_bits(), 120f32.to_bits(), 0.4f32.to_bits(), 250f32.to_bits()];
        let mut t = Taps::from_words(&words, 48000).unwrap();
        t.process(&[1.0; 8], 2);
        for _ in 0..4800 {
            t.process(&[0.0; 8], 2);
        }
        assert!(!t.settle(1e-9));
        let mut n = 4800;
        while !t.settle(1e-9) {
            for _ in 0..4800 {
                t.process(&[0.0; 8], 2);
            }
            n += 4800;
            assert!(n < 48000 * 30, "the delay never settled");
        }
        for _ in 0..48000 {
            assert_eq!(t.process(&[0.0; 8], 2), [0.0; 8]);
        }
    }
}
