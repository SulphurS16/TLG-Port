//! NGS2 voice processing beyond mixing: sampler envelopes, the I3DL2 reverb
//! rack and the multi-tap delay module of custom racks.
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
    fn process(&mut self, x: f32) -> f32 {
        let b = self.buf[self.pos];
        self.buf[self.pos] = x + b * 0.5;
        self.pos = (self.pos + 1) % self.buf.len();
        b - x * 0.5
    }
}

/// The reverb of one reverb rack voice.
pub struct Reverb {
    p: I3dl2,
    rate: u32,
    combs: Vec<Comb>,
    ap: [Vec<Allpass>; 2],
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
    /// Output scale: energy normalization x Room x Reverb x wet.
    late_gain: f32,
    refl_gain: f32,
}

impl Reverb {
    pub fn new(p: I3dl2, rate: u32) -> Reverb {
        let rate = rate.max(8000);
        let scale = rate as f32 / 48000.0;
        let decay = p.decay.clamp(0.1, 20.0);
        let combs: Vec<Comb> = [1557usize, 1617, 1491, 1422, 1277, 1356]
            .iter()
            .map(|&d| {
                let len = ((d as f32 * scale) as usize).max(1);
                let t = len as f32 / rate as f32;
                Comb { buf: vec![0.0; len], pos: 0, fb: 10f32.powf(-3.0 * t / decay), lp: 0.0 }
            })
            .collect();
        let ap = |ds: &[usize]| ds.iter().map(|&d| Allpass { buf: vec![0.0; ((d as f32 * scale) as usize).max(1)], pos: 0 }).collect::<Vec<_>>();
        // Energy of the summed combs' impulse responses (unit input):
        // sum over combs of 1 / (1 - fb^2); normalized to 1.
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
            combs,
            ap: [ap(&[556, 441]), ap(&[579, 464])],
            pre: vec![0.0; late_delay.max(refl_delay) + 2],
            pre_pos: 0,
            refl_delay,
            late_delay,
            damp,
            in_coef,
            in_lp: 0.0,
            late_gain: norm * room * mb(p.reverb) * p.wet,
            refl_gain: room * mb(p.reflections) * p.wet,
        }
    }

    /// Whether the parameters or rate changed enough to rebuild.
    pub fn matches(&self, p: &I3dl2, rate: u32) -> bool {
        self.p == *p && self.rate == rate.max(8000)
    }

    /// One frame: mono input (the channels' mean), a stereo pair out.
    pub fn process(&mut self, x: f32) -> [f32; 2] {
        self.in_lp = x * (1.0 - self.in_coef) + self.in_lp * self.in_coef;
        let n = self.pre.len();
        self.pre[self.pre_pos] = self.in_lp;
        let at = |d: usize| (self.pre_pos + n - d.min(n - 1)) % n;
        let late_in = self.pre[at(self.late_delay)];
        let refl = self.pre[at(self.refl_delay)] * self.refl_gain;
        self.pre_pos = (self.pre_pos + 1) % n;
        let mut sum = 0.0;
        for c in &mut self.combs {
            let y = c.buf[c.pos];
            c.lp = y * (1.0 - self.damp) + c.lp * self.damp;
            c.buf[c.pos] = late_in + c.lp * c.fb;
            c.pos = (c.pos + 1) % c.buf.len();
            sum += y;
        }
        let mut out = [0.0; 2];
        for (k, aps) in self.ap.iter_mut().enumerate() {
            let mut s = sum;
            for a in aps.iter_mut() {
                s = a.process(s);
            }
            out[k] = s * self.late_gain + refl;
        }
        out
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
    pos: usize,
    lp: [f32; 8],
    coef: f32,
}

/// The longest tap delay kept (1.5 s).
const MAX_DELAY_S: f32 = 1.5;

impl Taps {
    /// From module 0x1d's words after the header: [1] -, [2] level,
    /// [3] feedback, [4] low-pass cutoff (Hz), [5] number of taps, then
    /// (gain, milliseconds) pairs.
    pub fn from_words(w: &[u32], rate: u32) -> Option<Taps> {
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
        Some(Taps { level, feedback, cutoff, taps, buf: vec![[0.0; 8]; len], pos: 0, lp: [0.0; 8], coef })
    }

    /// New parameters, keeping the delay line (the title glides between
    /// environments by small steps).
    pub fn update(&mut self, new: Taps) {
        if new.buf.len() > self.buf.len() {
            let mut b = vec![[0.0; 8]; new.buf.len()];
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
    }

    /// One frame of `ch` channels.
    pub fn process(&mut self, x: &[f32; 8], ch: usize) -> [f32; 8] {
        let n = self.buf.len();
        let mut out = [0f32; 8];
        for &(g, d) in &self.taps {
            let fr = &self.buf[(self.pos + n - d.min(n - 1)) % n];
            for c in 0..ch {
                out[c] += g * fr[c];
            }
        }
        // (Loop gain kept below 0.7: the taps' sum times the feedback.)
        let sum: f32 = self.taps.iter().map(|t| t.0.abs()).sum();
        let fb = if sum > 0.0 { self.feedback.min(0.7 / sum) } else { 0.0 };
        let mut w = [0f32; 8];
        for c in 0..ch {
            self.lp[c] = out[c] * (1.0 - self.coef) + self.lp[c] * self.coef;
            out[c] = self.lp[c] * self.level;
            w[c] = x[c] + self.lp[c] * fb;
        }
        self.buf[self.pos] = w;
        self.pos = (self.pos + 1) % n;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut r = Reverb::new(p, 48000);
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
}
