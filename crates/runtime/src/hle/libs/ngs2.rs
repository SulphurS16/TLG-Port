//! libSceNgs2: the system audio graph (systems of racks of voices: sampler voices
//! play waveform data, submixers and mastering voices mix it to output buffers).
//!
//! Stage 2 (this file): sampler voices decode their waveform blocks (ATRAC9 through
//! LibAtrac9, PCM), honour skip/repeat counts and pitch (linear-interpolation
//! resampling to the system rate) and report decoding progress through
//! sceNgs2VoiceGetState, which the title's streaming relies on. Voices are mixed
//! along their patches with port volumes and port matrices into submixers and the
//! mastering voice, which writes the render buffers.
//!
//! Sampler envelopes (with release on STOP), sampler and submixer filters
//! (equalizer bands: types 1, 2, 6, 7, 8 as low-pass, high-pass, low shelf,
//! high shelf, peaking), submixer compressors, reverb racks (I3DL2, wet only: effect
//! returns), the multi-tap delay module of custom racks (the taps only) and
//! the mastering gains and limiter are applied (`ngs2_fx`). Not implemented
//! yet (logged once each, audio passes through unchanged): other filter
//! types, submixer envelopes and distortion, other custom rack modules, port
//! delays, peak meters. Unknown parameter ids are logged with their payload.

use super::atrac9;
use super::ngs2_fx::{lane_mask, Biquad, Compressor, Envelope, FilterKind, I3dl2, Limiter, Reverb, Taps};
use crate::gpu::render::FastMap;
use crate::hle::Registry;
use crate::{reg, sysv};
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet, VecDeque};

const ERR_INVALID_HANDLE: i32 = 0x804A_0230u32 as i32; // SCE_NGS2_ERROR_INVALID_SYSTEM/RACK/VOICE_HANDLE class
const ERR_INVALID_OPTION: i32 = 0x804A_0081u32 as i32;
const ERR_INVALID_OUT_ADDRESS: i32 = 0x804A_0052u32 as i32;
const ERR_INVALID_VOICE_INDEX: i32 = 0x804A_0261u32 as i32;
const ERR_INVALID_BUFFER: i32 = 0x804A_0050u32 as i32;

// Voice state flags.
const STATE_IN_USE: u32 = 0x1;
const STATE_PLAYING: u32 = 0x2;
const STATE_PAUSED: u32 = 0x4;
const STATE_STOPPED: u32 = 0x8;
const STATE_EMPTY: u32 = 0x20;

// Common voice parameter ids.
const PARAM_MATRIX_LEVELS: u32 = 1;
const PARAM_PORT_VOLUME: u32 = 2;
const PARAM_PORT_MATRIX: u32 = 3;
const PARAM_PORT_DELAY: u32 = 4;
const PARAM_PATCH: u32 = 5;
const PARAM_EVENT: u32 = 6;
const PARAM_CALLBACK: u32 = 7;
// Sampler voice parameter ids (0x1000_0000 + n).
const SAMPLER_SETUP: u32 = 0x1000_0000;
const SAMPLER_ADD_BLOCKS: u32 = 0x1000_0001;
const SAMPLER_REPLACE_ADDRESS: u32 = 0x1000_0002;
const SAMPLER_FRAME_OFFSET: u32 = 0x1000_0003;
const SAMPLER_EXIT_LOOP: u32 = 0x1000_0004;
const SAMPLER_PITCH: u32 = 0x1000_0005;
const SAMPLER_ENVELOPE: u32 = 0x1000_0006;
const SAMPLER_DISTORTION: u32 = 0x1000_0007;
const SAMPLER_USER_FX: u32 = 0x1000_0008;
const SAMPLER_PEAKMETER: u32 = 0x1000_0009;
const SAMPLER_FILTER: u32 = 0x1000_000A;
// Submixer / reverb / mastering setup ids.
const SUBMIXER_SETUP: u32 = 0x2000_0000;
const REVERB_SETUP: u32 = 0x2001_0000;
const REVERB_I3DL2: u32 = 0x2001_0001;
const MASTERING_SETUP: u32 = 0x3000_0000;
const MASTERING_LIMITER: u32 = 0x3000_0003;
const MASTERING_GAIN: u32 = 0x3000_0004;
/// The multi-tap delay module of custom racks (module 0x1d's parameters).
const CUSTOM_DELAY: u32 = 0x4000_1D00;

// Voice events.
const EVENT_PLAY: u32 = 0;
const EVENT_STOP: u32 = 1;
const EVENT_STOP_IMMEDIATELY: u32 = 2;
const EVENT_KILL: u32 = 3;
const EVENT_PAUSE: u32 = 4;
const EVENT_RESUME: u32 = 5;

// Waveform types (SceNgs2WaveformType).
const WF_PCM_I16L: u32 = 0x12;
const WF_PCM_F32L: u32 = 0x18;
const WF_VAG: u32 = 0x1C;
const WF_ATRAC9: u32 = 0x40;

/// Decodes one 16-byte PS-ADPCM (VAG) frame into 28 samples; `hist` holds the
/// channel's previous two output samples (filter state, scaled to i16 range).
fn vag_frame(fr: &[u8], hist: &mut [f32; 2]) -> [f32; 28] {
    const COEF: [(f32, f32); 5] = [(0.0, 0.0), (60.0 / 64.0, 0.0), (115.0 / 64.0, -52.0 / 64.0), (98.0 / 64.0, -55.0 / 64.0), (122.0 / 64.0, -60.0 / 64.0)];
    let shift = (fr[0] & 0x0F) as i32;
    let (c1, c2) = COEF[((fr[0] >> 4) as usize).min(4)];
    let mut out = [0f32; 28];
    for (i, o) in out.iter_mut().enumerate() {
        let byte = fr[2 + i / 2];
        let nib = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        let raw = ((((nib as i32) << 28) >> 28) << 12) >> shift.min(12);
        let s = (raw as f32 + hist[0] * c1 + hist[1] * c2).clamp(-32768.0, 32767.0);
        hist[1] = hist[0];
        hist[0] = s;
        *o = s / 32768.0;
    }
    out
}

/// Channels of every mixing bus (output buffers have at most 8).
const MAX_CH: usize = 8;
/// Output ports per voice with their own volume and matrix. (TLG's sampler
/// voices patch ports 0-4: the dry path and four effect sends; with four
/// slots port 4 shared port 0's volume and matrix, so its reverb send ran at
/// the full dry level whatever the title set.)
const MAX_PORTS: usize = 8;

#[derive(Clone, Copy, Default)]
struct BufferInfo {
    buffer: u64,
    size: u64,
    user_data: u64,
    free_handler: u64,
}

struct System {
    buf: BufferInfo,
    sample_rate: u32,
    grain: u32,
    renders: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Sampler,
    Submixer,
    /// Reverb racks (0x2001): the reverberation of their input.
    Reverb,
    /// Custom racks (0x4xxx): their modules' output (the multi-tap delay's
    /// taps; other modules pass the input through).
    Custom,
    Mastering,
}

fn kind_of(rack_id: u32) -> Kind {
    match rack_id {
        0x1000..=0x1FFF => Kind::Sampler,
        0x2001 => Kind::Reverb,
        0x3000..=0x3FFF => Kind::Mastering,
        0x4000..=0x4FFF => Kind::Custom,
        _ => Kind::Submixer,
    }
}

struct Rack {
    system: u64,
    rack_id: u32,
    buf: BufferInfo,
    voices: Vec<u64>,
}

/// A queued waveform block (SceNgs2SamplerWaveformBlock).
#[derive(Clone, Copy, Debug)]
struct Block {
    addr: u64,
    size: u32,
    repeats: u32,
    skip: u32,
    samples: u32,
    user_data: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct Patch {
    port: u32,
    dest: u64,
    dest_input: u32,
}

struct Sampler {
    wf_type: u32,
    channels: u32,
    rate: u32,
    config: u32,
    blocks: VecDeque<Block>,
    /// Decoding position in the front block: bytes consumed, samples produced.
    byte_pos: u32,
    sample_pos: u32,
    /// Repeat passes left for the front block (taken from it when it starts).
    repeats_left: u32,
    started_block: bool,
    /// The front block's data, taken in whole when it starts (and again for
    /// each repeat pass); decoding reads this copy (`take_block`).
    copy: Vec<u8>,
    dec: Option<atrac9::Decoder>,
    /// Decoded frames not yet consumed by the resampler (interleaved by channel).
    pcm: VecDeque<[f32; MAX_CH]>,
    /// Resampler phase in [0, 1) between `cur` and the next decoded frame.
    phase: f64,
    cur: [f32; MAX_CH],
    pitch: f32,
    decoded_samples: u64,
    decoded_bytes: u64,
    user_data: u64,
    data: u64,
    error_logged: bool,
    vag_hist: [[f32; 2]; MAX_CH],
}

impl Default for Sampler {
    fn default() -> Self {
        Sampler {
            wf_type: 0,
            channels: 1,
            rate: 48000,
            config: 0,
            blocks: VecDeque::new(),
            byte_pos: 0,
            sample_pos: 0,
            repeats_left: 0,
            started_block: false,
            copy: Vec::new(),
            dec: None,
            pcm: VecDeque::new(),
            phase: 0.0,
            cur: [0.0; MAX_CH],
            pitch: 1.0,
            decoded_samples: 0,
            decoded_bytes: 0,
            user_data: 0,
            data: 0,
            error_logged: false,
            vag_hist: [[0.0; 2]; MAX_CH],
        }
    }
}

struct Voice {
    rack: u64,
    kind: Kind,
    index: u32,
    flags: u32,
    /// Bus channels (submixer/mastering setup), sampler: its waveform's.
    channels: u32,
    port_volume: [f32; MAX_PORTS],
    port_matrix: [i32; MAX_PORTS],
    matrices: FastMap<i32, Vec<f32>>,
    /// Where each matrix's levels came from (TLG_NGS2_TRACE: checked for
    /// later in-place changes).
    matrix_src: FastMap<i32, (u64, usize)>,
    patches: Vec<Patch>,
    sampler: Sampler,
    /// Input bus (submixers, mastering): `grain` frames of MAX_CH channels.
    input: Vec<[f32; MAX_CH]>,
    gain: f32,
    /// Peak level of the voice's last rendered grain (state peakHeight).
    peak: f32,
    /// Highest peak (samplers: output, mixers: input) since the last
    /// TLG_NGS2_TRACE voice report.
    peak_win: f32,
    /// Flags last returned to the title (for TLG_NGS2_TRACE transitions).
    reported: u32,
    /// Submixer user effect (the title's own DSP, `run_user_fx`): handler and
    /// user data; its calls so far; guest scratch for its context and
    /// planar channel buffers.
    user_fx: Option<(u64, u64)>,
    fx_calls: u64,
    fx_buf: u64,
    /// Sampler envelope (empty: full level).
    env: Envelope,
    /// Reverb racks: parameters and state.
    i3dl2: I3dl2,
    reverb: Option<Reverb>,
    /// Custom racks: the multi-tap delay (module 0x1d) when set.
    taps: Option<Taps>,
    /// Submixer filter slots 0-7 (parameter 0x20000006): shape, cutoff, Q,
    /// gain; their filters designed at the system rate (`eq_rate`; 0 = to
    /// be designed).
    eq_params: [Option<(FilterKind, f32, f32, f32)>; 8],
    eq: [Option<Biquad>; 8],
    eq_rate: u32,
    /// Submixer compressor (parameter 0x20000002): its words, and the
    /// compressor (made at the system rate with the filters).
    comp_words: Option<[u32; 8]>,
    comp: Option<Compressor>,
    /// Output channels of reverb and custom racks.
    out_channels: u32,
    /// Mastering: LFE gain (channel 3; `gain` for the others) and limiter
    /// threshold (0: off); the output stage (DC blocker and limiter).
    lfe_gain: f32,
    limit: f32,
    limiter: Option<Limiter>,
    /// Mixing voices: consecutive grains with silent input, and whether the
    /// reverb's or delay's state has died away and was cleared (`render`).
    quiet: u32,
    fx_idle: bool,
    /// Sampler voices: the current grain before routing (`sampler_grain`),
    /// its channels and whether it is silent.
    out: Vec<[f32; MAX_CH]>,
    out_ch: usize,
    out_silent: bool,
}

impl Voice {
    fn new(rack: u64, kind: Kind, index: u32) -> Voice {
        Voice {
            rack,
            kind,
            index,
            flags: 0,
            channels: if kind == Kind::Sampler { 1 } else { 2 },
            port_volume: [1.0; MAX_PORTS],
            port_matrix: [-1; MAX_PORTS],
            matrices: FastMap::default(),
            matrix_src: FastMap::default(),
            patches: Vec::new(),
            sampler: Sampler::default(),
            input: Vec::new(),
            gain: 1.0,
            peak: 0.0,
            peak_win: 0.0,
            reported: 0,
            user_fx: None,
            fx_calls: 0,
            fx_buf: 0,
            env: Envelope::default(),
            i3dl2: I3dl2::default(),
            reverb: None,
            taps: None,
            eq_params: [None; 8],
            eq: [const { None }; 8],
            eq_rate: 0,
            comp_words: None,
            comp: None,
            out_channels: 2,
            lfe_gain: 1.0,
            limit: 0.0,
            limiter: None,
            quiet: 0,
            fx_idle: false,
            out: Vec::new(),
            out_ch: 1,
            out_silent: false,
        }
    }
}

#[derive(Default)]
struct State {
    systems: FastMap<u64, System>,
    racks: FastMap<u64, Rack>,
    voices: FastMap<u64, Voice>,
    seen_params: HashSet<(u32, u16)>,
    noted: HashSet<&'static str>,
    /// Output peak since the last statistics line.
    peak: f32,
    /// The current `tune::REV_AB` window (logged at each start).
    rev_ab_window: Option<u64>,
    /// Measurement (a `tune::REV_AB` variant): ports 4-7 take the volume
    /// and matrix of port & 3, as before ports past 3 had their own.
    port_fold: bool,
    /// Grains skipped as idle and run since the last statistics line
    /// (`IDLE_FX_SKIPPED` ...).
    idle: [u64; 6],
}

static NGS2: Mutex<Option<State>> = Mutex::new(None);

fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
    let mut g = NGS2.lock();
    f(g.get_or_insert_with(State::default))
}

/// TLG_NGS2_TRACE=1: sampler voice block traffic, endings and state queries.
fn trace_voices() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("TLG_NGS2_TRACE").is_some())
}

/// Logs a not-implemented behaviour once.
fn note(s: &mut State, what: &'static str) {
    if s.noted.insert(what) {
        crate::warn!("NGS2: {what}: not implemented (audio passes through unchanged)");
    }
}

/// Buffer sizes this implementation requests (its own state lives on the host;
/// the buffers back the handles and keep the title's memory accounting realistic).
const SYSTEM_BUFFER: u64 = 0x4000;
fn rack_buffer(max_voices: u32) -> u64 {
    0x4000 + max_voices as u64 * 0x100
}

unsafe fn rack_option_voices(option: *const u8) -> u32 {
    // SceNgs2RackOption: {size_t size; char name[16]; u32 flags, maxGrainSamples, maxVoices, ...}
    if option.is_null() { 1 } else { unsafe { *(option.add(32) as *const u32) }.max(1) }
}

/// Calls the title's buffer allocator: {allocHandler, freeHandler, userData}.
unsafe fn allocate(allocator: *const u64, size: u64) -> Result<BufferInfo, i32> {
    if allocator.is_null() {
        return Err(ERR_INVALID_OPTION);
    }
    let (alloc, free, user) = unsafe { (*allocator, *allocator.add(1), *allocator.add(2)) };
    // SceNgs2ContextBufferInfo: {hostBuffer, hostBufferSize, reserved[5], userData}
    let info = crate::mem::arena_alloc(64, 8) as *mut u64;
    unsafe {
        std::ptr::write_bytes(info, 0, 8);
        *info.add(1) = size;
        *info.add(7) = user;
        let r = crate::hle::call_guest(alloc, &[info as u64]) as i32;
        if r < 0 || *info == 0 {
            crate::error!("NGS2: title allocator failed ({r:#x}) for {size:#x} bytes");
            return Err(if r < 0 { r } else { ERR_INVALID_BUFFER });
        }
        Ok(BufferInfo { buffer: *info, size, user_data: user, free_handler: free })
    }
}

unsafe fn release(b: &BufferInfo) {
    if b.free_handler == 0 {
        return;
    }
    let info = crate::mem::arena_alloc(64, 8) as *mut u64;
    unsafe {
        std::ptr::write_bytes(info, 0, 8);
        *info = b.buffer;
        *info.add(1) = b.size;
        *info.add(7) = b.user_data;
        crate::hle::call_guest(b.free_handler, &[info as u64]);
    }
}

unsafe fn create_rack(system: u64, rack_id: u32, option: *const u8, buf: BufferInfo) -> u64 {
    let max_voices = unsafe { rack_option_voices(option) };
    let handle = buf.buffer;
    let kind = kind_of(rack_id);
    with(|s| {
        let voices: Vec<u64> = (0..max_voices).map(|i| handle + 0x100 + i as u64 * 0x10).collect();
        for (i, &v) in voices.iter().enumerate() {
            s.voices.insert(v, Voice::new(handle, kind, i as u32));
        }
        s.racks.insert(handle, Rack { system, rack_id, buf, voices });
    });
    crate::info!("NGS2 rack {rack_id:#x} ({kind:?}) on system {system:#x}: {max_voices} voices -> handle {handle:#x}");
    handle
}

/// Applies one voice parameter chain ({u16 size; s16 next; u32 id} headers).
unsafe fn control(s: &mut State, vh: u64, mut p: *const u8) -> i32 {
    let mut n = 0;
    while !p.is_null() && n < 256 {
        n += 1;
        let (size, next, id) = unsafe { (*(p as *const u16), *(p.add(2) as *const i16), *(p.add(4) as *const u32)) };
        if s.seen_params.insert((id, size)) {
            let words: Vec<String> = (0..((size as usize).saturating_sub(8) / 4).min(12))
                .map(|i| format!("{:08x}", unsafe { *(p.add(8 + i * 4) as *const u32) }))
                .collect();
            crate::info!("NGS2 voice param id {id:#x} size {size}: {}", words.join(" "));
        }
        let u32_at = |off: usize| unsafe { *(p.add(off) as *const u32) };
        let u64_at = |off: usize| unsafe { *(p.add(off) as *const u64) };
        let f32_at = |off: usize| f32::from_bits(u32_at(off));
        let mut unimpl: Option<&'static str> = None;
        {
            let Some(v) = s.voices.get_mut(&vh) else { return ERR_INVALID_HANDLE };
            if id == PARAM_EVENT && trace_voices() && v.kind == Kind::Sampler {
                crate::info!("NGS2 voice {vh:#x}: event {} (flags {:#x}, {} blocks queued, {} samples decoded)", u32_at(8), v.flags, v.sampler.blocks.len(), v.sampler.decoded_samples);
            }
            match id {
                PARAM_EVENT => match u32_at(8) {
                    EVENT_PLAY => {
                        if v.flags & STATE_PLAYING == 0 {
                            v.env.restart();
                        }
                        v.flags = (v.flags | STATE_IN_USE | STATE_PLAYING) & !(STATE_STOPPED | STATE_PAUSED | STATE_EMPTY);
                    }
                    // STOP on a playing sampler with release points plays on
                    // through them (the voice ends with the envelope).
                    EVENT_STOP if v.kind == Kind::Sampler && v.flags & STATE_PLAYING != 0 && v.env.release() => {}
                    e @ (EVENT_STOP | EVENT_STOP_IMMEDIATELY | EVENT_KILL) => {
                        v.flags = (v.flags & !(STATE_PLAYING | STATE_PAUSED)) | STATE_STOPPED;
                        if e != EVENT_STOP || v.kind == Kind::Sampler {
                            v.flags &= !STATE_IN_USE;
                            let sm = &mut v.sampler;
                            sm.blocks.clear();
                            sm.pcm.clear();
                            sm.started_block = false;
                            sm.copy = Vec::new();
                        }
                        // A killed sampler voice keeps no patches: TLG kills
                        // each voice before reusing it and patches only the
                        // ports the next sound uses (port 0 for a cutscene
                        // stream, 0-3 for most effects). Kept patches sent
                        // the narration and the boy's voice, at full level,
                        // into the previous sound's reverb and delay sends,
                        // which the PS4 recording does not have.
                        if e == EVENT_KILL && v.kind == Kind::Sampler && crate::tune::get(crate::tune::NGS2_COMPAT) & 1 == 0 {
                            v.patches.clear();
                        }
                    }
                    EVENT_PAUSE => v.flags |= STATE_PAUSED,
                    EVENT_RESUME => v.flags &= !STATE_PAUSED,
                    e => crate::warn!("NGS2: unknown voice event {e}"),
                },
                PARAM_PORT_VOLUME => {
                    let port = u32_at(8) as usize;
                    if port >= MAX_PORTS {
                        unimpl = Some("voice ports above 7");
                    } else {
                        if trace_voices() && v.port_volume[port] != f32_at(12) {
                            crate::info!("NGS2 voice {vh:#x} ({:?} {}ch): port {port} volume {:.4}", v.kind, v.channels, f32_at(12));
                        }
                        v.port_volume[port] = f32_at(12);
                    }
                }
                PARAM_PORT_MATRIX => {
                    let port = u32_at(8) as usize;
                    if port >= MAX_PORTS {
                        unimpl = Some("voice ports above 7");
                    } else {
                        if trace_voices() && v.port_matrix[port] != u32_at(12) as i32 {
                            crate::info!("NGS2 voice {vh:#x} ({:?} {}ch): port {port} matrix {}", v.kind, v.channels, u32_at(12) as i32);
                        }
                        v.port_matrix[port] = u32_at(12) as i32;
                    }
                }
                PARAM_MATRIX_LEVELS => {
                    // {i32 matrixId; u32 numLevels; const float *aLevel}
                    let mid = u32_at(8) as i32;
                    let n = (u32_at(12) as usize).min(MAX_CH * MAX_CH);
                    let levels = u64_at(16) as *const f32;
                    if !levels.is_null() {
                        let l: Vec<f32> = (0..n).map(|i| unsafe { *levels.add(i) }).collect();
                        static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                        if l.iter().any(|&x| x != 0.0) && LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                            crate::info!("NGS2 matrix {mid} ({} ch voice, {n} levels): {l:?}", v.channels);
                        }
                        if trace_voices() && v.matrices.get(&mid) != Some(&l) {
                            let nz: Vec<String> = l.iter().enumerate().filter(|(_, x)| **x != 0.0).map(|(i, x)| format!("{i}:{x:.3}")).collect();
                            crate::info!("NGS2 voice {vh:#x} ({:?} {}ch): matrix {mid}, {n} levels, nonzero [{}]", v.kind, v.channels, nz.join(" "));
                        }
                        v.matrices.insert(mid, l);
                        v.matrix_src.insert(mid, (levels as u64, n));
                    }
                }
                PARAM_PATCH => {
                    // {u32 port; u32 destInputId; SceNgs2Handle destHandle}
                    let port = u32_at(8);
                    let dest_input = u32_at(12);
                    let dest = u64_at(16);
                    v.patches.retain(|pa| pa.port != port || pa.dest_input != dest_input);
                    if dest != 0 {
                        v.patches.push(Patch { port, dest, dest_input });
                    }
                    if trace_voices() {
                        crate::info!("NGS2 voice {vh:#x} ({:?} {}ch): patch port {port} -> {dest:#x} input {dest_input}", v.kind, v.channels);
                    }
                }
                PARAM_PORT_DELAY => unimpl = Some("port delay"),
                PARAM_CALLBACK => unimpl = Some("voice callbacks"),
                SAMPLER_SETUP => {
                    // {SceNgs2WaveformFormat {type, channels, rate, configData, frameOffset, frameMargin}; flags}
                    let sm = &mut v.sampler;
                    *sm = Sampler { pitch: sm.pitch, ..Sampler::default() };
                    sm.wf_type = u32_at(8);
                    sm.channels = u32_at(12).clamp(1, MAX_CH as u32);
                    sm.rate = u32_at(16).max(1);
                    sm.config = u32_at(20);
                    v.channels = sm.channels;
                    if sm.wf_type == WF_ATRAC9 {
                        match atrac9::Decoder::new(sm.config.to_be_bytes()) {
                            Ok(d) => {
                                if d.info.channels as u32 != sm.channels {
                                    crate::warn!("NGS2: ATRAC9 config {:08x}: {} channels, voice says {}", sm.config, d.info.channels, sm.channels);
                                }
                                sm.dec = Some(d);
                            }
                            Err(e) => crate::warn!("NGS2: {e}"),
                        }
                    } else if !matches!(sm.wf_type, WF_PCM_I16L | WF_PCM_F32L | WF_VAG) {
                        crate::warn!("NGS2: sampler waveform type {:#x} not supported (silent)", sm.wf_type);
                    }
                    v.flags = (v.flags | STATE_IN_USE) & !STATE_EMPTY;
                }
                SAMPLER_ADD_BLOCKS => {
                    // {const void *data; u32 flags; u32 numBlocks; const SceNgs2SamplerWaveformBlock *aBlock}
                    // block: {u32 offset, size, numRepeats, numSkipSamples, numSamples, reserved; uintptr userData}
                    let data = u64_at(8);
                    v.sampler.data = data;
                    let count = u32_at(20).min(256);
                    let blocks = u64_at(24) as *const u32;
                    for i in 0..count as usize {
                        if blocks.is_null() {
                            break;
                        }
                        let b = unsafe { blocks.add(i * 8) };
                        let rd = |k: usize| unsafe { *b.add(k) };
                        let blk = Block {
                            addr: data + rd(0) as u64,
                            size: rd(1),
                            repeats: rd(2),
                            skip: rd(3),
                            samples: rd(4),
                            user_data: unsafe { *(b.add(6) as *const u64) },
                        };
                        if trace_voices() {
                            crate::info!("NGS2 voice {:#x}: add block {i}/{count} (flags {:#x}): {:#x}+{:#x} bytes, {} samples (skip {}, repeats {:#x}), user {:#x}, type {:#x} {} ch, queued {}, voice flags {:#x}", vh, u32_at(16), blk.addr, blk.size, blk.samples, blk.skip, blk.repeats, blk.user_data, v.sampler.wf_type, v.sampler.channels, v.sampler.blocks.len(), v.flags);
                        }
                        v.sampler.blocks.push_back(blk);
                    }
                    v.flags &= !STATE_EMPTY;
                }
                SAMPLER_REPLACE_ADDRESS => {
                    // {const void *from; const void *to}: queued blocks move with the data.
                    let (from, to) = (u64_at(8), u64_at(16));
                    for b in v.sampler.blocks.iter_mut() {
                        if b.addr >= from {
                            b.addr = b.addr - from + to;
                        }
                    }
                    if v.sampler.data == from {
                        v.sampler.data = to;
                    }
                }
                SAMPLER_EXIT_LOOP => {
                    v.sampler.repeats_left = 0;
                    for b in v.sampler.blocks.iter_mut() {
                        b.repeats = 0;
                    }
                }
                SAMPLER_PITCH => v.sampler.pitch = f32_at(8),
                SAMPLER_FRAME_OFFSET => unimpl = Some("sampler waveform frame offset"),
                SAMPLER_ENVELOPE => {
                    // {u32 numForwardPoints, numReleasePoints; const SceNgs2EnvelopePoint *aPoint}
                    // with points {u32 curve, duration (samples); float height}.
                    let (fwd, rel, pts) = (u32_at(8), u32_at(12), u64_at(16));
                    let n = (fwd as u64 + rel as u64).min(32);
                    if n > 0 && pts != 0 && crate::mem::is_guest(pts, n * 12) {
                        let l: Vec<(u32, u32, f32)> = (0..n).map(|i| unsafe {
                            let q = (pts + i * 12) as *const u32;
                            (*q, *q.add(1), f32::from_bits(*q.add(2)))
                        }).collect();
                        static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                        if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 6 {
                            crate::info!("NGS2 envelope: {fwd} forward + {rel} release points {l:?}");
                        }
                        v.env = Envelope::new(l, (fwd as usize).min(n as usize));
                    } else {
                        v.env = Envelope::default();
                    }
                }
                SAMPLER_DISTORTION => unimpl = Some("sampler distortion"),
                SAMPLER_USER_FX => unimpl = Some("sampler user effects"),
                SAMPLER_PEAKMETER => unimpl = Some("sampler peak meters"),
                SAMPLER_FILTER | SUBMIXER_FILTER => {
                    // {index, ?, type, ?, fc, Q, level} (docs/LOG.md: Audio level).
                    trace_payload(vh, id, size, p);
                    let slot = u32_at(8) as usize;
                    let t = u32_at(16);
                    if slot < 8 && size >= 36 {
                        match FilterKind::from_type(t) {
                            Some(k) => v.eq_params[slot] = Some((k, f32_at(24), f32_at(28), f32_at(32))),
                            None => {
                                v.eq_params[slot] = None;
                                if t != 0 {
                                    unimpl = Some("filter types other than 1, 2, 6, 7, 8");
                                }
                            }
                        }
                        v.eq_rate = 0;
                    }
                }
                SUBMIXER_COMPRESSOR => {
                    trace_payload(vh, id, size, p);
                    if size >= 40 {
                        let mut w = [0u32; 8];
                        for (i, x) in w.iter_mut().enumerate() {
                            *x = u32_at(8 + i * 4);
                        }
                        v.comp_words = Some(w);
                        v.eq_rate = 0;
                    }
                }
                SUBMIXER_SETUP | REVERB_SETUP | MASTERING_SETUP | 0x4002_0000 | 0x4001_0000 | 0x4003_0000 => {
                    // {u32 numChannels; ...} (reverb/custom: numInputChannels, numOutputChannels)
                    v.channels = u32_at(8).clamp(1, MAX_CH as u32);
                    v.out_channels = if matches!(v.kind, Kind::Reverb | Kind::Custom) { u32_at(12).clamp(1, MAX_CH as u32) } else { v.channels };
                    v.flags |= STATE_IN_USE;
                }
                MASTERING_GAIN => {
                    // {float fbwLevel; float lfeLevel}
                    v.gain = f32_at(8);
                    v.lfe_gain = f32_at(12);
                    note_mastering("gain", vh, v.gain, v.lfe_gain);
                }
                MASTERING_LIMITER => {
                    // {u32 enable; float threshold}
                    v.limit = if u32_at(8) != 0 { f32_at(12).max(0.0) } else { 0.0 };
                    note_mastering("limiter", vh, u32_at(8) as f32, f32_at(12));
                }
                REVERB_I3DL2 if v.kind == Kind::Reverb => {
                    let i = |off: usize| u32_at(off) as i32;
                    v.i3dl2 = I3dl2 {
                        wet: f32_at(8),
                        dry: f32_at(12),
                        room: i(16),
                        room_hf: i(20),
                        decay: f32_at(28),
                        decay_hf: f32_at(32),
                        reflections: i(36),
                        reflections_delay: f32_at(40),
                        reverb: i(44),
                        reverb_delay: f32_at(48),
                    };
                    static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                        crate::info!("NGS2 reverb {vh:#x}: {:?}", v.i3dl2);
                    }
                }
                CUSTOM_DELAY if v.kind == Kind::Custom => {
                    let words: Vec<u32> = (0..((size as usize).saturating_sub(8) / 4).min(40)).map(|k| u32_at(8 + k * 4)).collect();
                    // (A delay line only for a new delay: updates keep theirs.)
                    let parsed = if v.taps.is_some() { Taps::params_from_words(&words, 48000) } else { Taps::from_words(&words, 48000) };
                    match (parsed, v.taps.as_mut()) {
                        (Some(t), Some(cur)) => cur.update(t),
                        (Some(t), None) => {
                            static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                            if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                                crate::info!("NGS2 custom rack {vh:#x}: multi-tap delay level {:.3} feedback {:.3} cutoff {:.0} Hz, taps {:?}", t.level, t.feedback, t.cutoff, t.taps);
                            }
                            v.taps = Some(t);
                        }
                        _ => {}
                    }
                }
                SUBMIXER_USER_FX => {
                    // {handler; userData; ...}: run per grain (`run_user_fx`).
                    let (h, u) = (u64_at(8), u64_at(16));
                    v.user_fx = (h != 0 && user_fx_on()).then_some((h, u));
                    v.fx_calls = 0;
                    if trace_voices() {
                        crate::info!("NGS2 voice {vh:#x} ({:?} {}ch): user effect {h:#x} (data {u:#x})", v.kind, v.channels);
                    }
                }
                0x2000_0001 | 0x2000_0003 | 0x2000_0005 => {
                    unimpl = Some("submixer envelope/distortion");
                    trace_payload(vh, id, size, p);
                }
                0x2001_0002..=0x2001_00FF => unimpl = Some("reverb parameters other than I3DL2"),
                0x3000_0001..=0x3000_0002 | 0x3000_0005..=0x3000_00FF => unimpl = Some("mastering matrix/output/peak meter"),
                0x4000_0000..=0x4FFF_FFFF => {
                    unimpl = Some("custom rack modules other than the multi-tap delay (passed through)");
                    if trace_voices() {
                        let words: Vec<String> = (0..((size as usize).saturating_sub(8) / 4).min(34)).map(|i| format!("{:08x}", u32_at(8 + i * 4))).collect();
                        crate::info!("NGS2 voice {vh:#x} ({:?} {}ch): custom module param {id:#x}: {}", v.kind, v.channels, words.join(" "));
                    }
                }
                _ => unimpl = Some("unknown voice parameters"),
            }
        }
        if let Some(what) = unimpl {
            note(s, what);
        }
        if next == 0 {
            break;
        }
        p = unsafe { p.offset(next as isize) };
    }
    0
}

const SUBMIXER_USER_FX: u32 = 0x2000_0004;
const SUBMIXER_FILTER: u32 = 0x2000_0006;
const SUBMIXER_COMPRESSOR: u32 = 0x2000_0002;

/// Runs a voice's filters (equalizer bands) and then its compressor over
/// `frames` (making them at the system rate first after a parameter change).
fn run_eq(v: &mut Voice, frames: &mut [[f32; MAX_CH]], ch: usize, rate: u32) {
    if v.eq_rate != rate {
        for i in 0..8 {
            v.eq[i] = match (v.eq_params[i], v.eq[i].take()) {
                (None, _) => None,
                (Some((k, fc, q, l)), Some(mut f)) => {
                    f.set(k, fc, q, l, rate);
                    Some(f)
                }
                (Some((k, fc, q, l)), None) => Some(Biquad::new(k, fc, q, l, rate)),
            };
        }
        v.comp = match (v.comp_words, v.comp.take()) {
            (None, _) => None,
            (Some(w), Some(mut c)) => {
                c.set(&w, rate);
                Some(c)
            }
            (Some(w), None) => Some(Compressor::new(&w, rate)),
        };
        v.eq_rate = rate;
    }
    // (The bands in one pass: `Biquad::process_bands`.)
    let mut bands: smallvec::SmallVec<[&mut Biquad; 8]> = v.eq.iter_mut().flatten().filter(|f| !f.is_flat()).collect();
    if crate::tune::get(crate::tune::NGS2_VEC) != 0 {
        Biquad::process_bands(&mut bands, frames, ch);
    } else {
        for f in bands {
            f.process_one(frames, ch);
        }
    }
    if let Some(c) = v.comp.as_mut().filter(|c| !c.is_flat()) {
        c.process(frames, ch);
    }
}

/// TLG_NGS2_TRACE: the distinct payloads of not-yet-applied parameters
/// (filters, compressors), the first 64 of each id.
fn trace_payload(vh: u64, id: u32, size: u16, p: *const u8) {
    if !trace_voices() {
        return;
    }
    static SEEN: parking_lot::Mutex<Option<HashMap<u32, HashSet<Vec<u32>>>>> = parking_lot::Mutex::new(None);
    let words: Vec<u32> = (0..((size as usize).saturating_sub(8) / 4).min(16)).map(|i| unsafe { *(p.add(8 + i * 4) as *const u32) }).collect();
    let mut g = SEEN.lock();
    let set = g.get_or_insert_with(HashMap::new).entry(id).or_default();
    if set.len() < 64 && set.insert(words.clone()) {
        let shown: Vec<String> = words.iter().map(|&w| {
            let f = f32::from_bits(w);
            if w != 0 && f.is_finite() && f.abs() > 1e-6 && f.abs() < 1e6 && w > 0x00800000 { format!("{f}") } else { format!("{w:#x}") }
        }).collect();
        crate::info!("NGS2 param {id:#x} on {vh:#x}: [{}]", shown.join(", "));
    }
}

/// TLG_NGS2_USER_FX=0: user effects not called (audio passes through).
fn user_fx_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TLG_NGS2_USER_FX").map_or(true, |v| v != "0"))
}

/// Runs a submixer's user effect (the title's DSP) on its mixed input, in
/// place. The handler takes a context: +0 the address of eight channel
/// pointers (planar float buffers of one grain), +8 the user data, +0x20
/// flags (bit 0 on the first call: reset; TLG's handler skips processing for
/// bits 2-3); we also fill +0x24 channels, +0x28 grain samples, +0x2c rate.
fn run_user_fx(v: &mut Voice, input: &mut [[f32; MAX_CH]], rate: u32) {
    let Some((handler, user)) = v.user_fx else { return };
    let grain = input.len();
    let need = 0x80 + (MAX_CH * grain * 4) as u64;
    if v.fx_buf == 0 {
        v.fx_buf = crate::mem::arena_alloc(need.max(0x80 + MAX_CH as u64 * 1024 * 4), 64);
    }
    let base = v.fx_buf;
    unsafe {
        let ctx = base as *mut u8;
        std::ptr::write_bytes(ctx, 0, 0x40);
        let ptrs = (base + 0x40) as *mut u64;
        for c in 0..MAX_CH {
            let ch = base + 0x80 + (c * grain * 4) as u64;
            *ptrs.add(c) = ch;
            let dst = ch as *mut f32;
            for (i, fr) in input.iter().enumerate() {
                *dst.add(i) = fr[c];
            }
        }
        *(ctx as *mut u64) = base + 0x40;
        *(ctx.add(8) as *mut u64) = user;
        *(ctx.add(0x20) as *mut u32) = u32::from(v.fx_calls == 0);
        *(ctx.add(0x24) as *mut u32) = v.channels;
        *(ctx.add(0x28) as *mut u32) = grain as u32;
        *(ctx.add(0x2c) as *mut u32) = rate;
        crate::hle::call_guest(handler, &[base]);
        for c in 0..MAX_CH {
            let src = (base + 0x80 + (c * grain * 4) as u64) as *const f32;
            for (i, fr) in input.iter_mut().enumerate() {
                fr[c] = *src.add(i);
            }
        }
    }
    v.fx_calls += 1;
}

/// Takes in a block's data whole when it starts playing (and again for each
/// repeat pass), counting it as decoded: the PS4 consumes a block well
/// before it has played it. TLG streams its cutscene ambience through a ring
/// one block long and writes the next block once the decoded byte count says
/// the ring is free; consumed just in time, the voice ran dry first and the
/// title gave the stream up (the opening scene had no cave ambience). The
/// decoder reads the copy, so the title may refill or move the data meanwhile.
fn take_block(sm: &mut Sampler, b: &Block) {
    // (A new allocation each time: a voice's copy does not keep the
    // capacity of the largest block it ever played.)
    sm.copy = if b.addr != 0 && b.size > 0 && crate::mem::is_guest(b.addr, b.size as u64) {
        unsafe { std::slice::from_raw_parts(b.addr as *const u8, b.size as usize) }.to_vec()
    } else {
        Vec::new()
    };
    sm.decoded_bytes += b.size as u64;
}

/// Decodes the next portion of the front block into `sm.pcm`; false when the
/// voice has no more data.
fn decode_more(sm: &mut Sampler) -> bool {
    // (Bounded: a looping block that decodes nothing must not spin under the lock.)
    for _ in 0..64 {
        let Some(&b) = sm.blocks.front() else { return false };
        if !sm.started_block {
            sm.started_block = true;
            sm.byte_pos = 0;
            sm.sample_pos = 0;
            sm.repeats_left = b.repeats;
            sm.user_data = b.user_data;
            take_block(sm, &b);
        }
        let ch = sm.channels as usize;
        let mut produced = false;
        if sm.sample_pos < b.samples + b.skip && sm.byte_pos < b.size && b.addr != 0 {
            match sm.wf_type {
                WF_ATRAC9 if sm.dec.is_some() => {
                    let d = sm.dec.as_mut().unwrap();
                    let sf = d.info.superframe_size.max(1) as usize;
                    let frames = d.info.frames_in_superframe.max(1) as usize;
                    let fs = d.info.frame_samples.max(1) as usize;
                    let dch = (d.info.channels as usize).max(1);
                    // Superframe into a zero-padded copy (the decoder reads bitwise past frames' ends).
                    let avail = (b.size - sm.byte_pos) as usize;
                    let take = sf.min(avail);
                    let mut buf = vec![0u8; sf + 64];
                    let p = sm.byte_pos as usize;
                    if let Some(src) = sm.copy.get(p..p + take) {
                        buf[..take].copy_from_slice(src);
                    }
                    let mut pcm16 = vec![0i16; fs * dch];
                    let mut off = 0usize;
                    for _ in 0..frames {
                        match d.decode_frame(buf[off..].as_ptr(), sf - off.min(sf), &mut pcm16) {
                            Ok(used) => off += used,
                            Err(e) => {
                                if !sm.error_logged {
                                    sm.error_logged = true;
                                    crate::warn!("NGS2: {e} (block at {:#x}+{:#x})", b.addr, sm.byte_pos);
                                }
                                break;
                            }
                        }
                        for f in 0..fs {
                            let s = sm.sample_pos;
                            sm.sample_pos += 1;
                            if s < b.skip || s >= b.skip + b.samples {
                                continue;
                            }
                            let mut fr = [0f32; MAX_CH];
                            for c in 0..dch.min(MAX_CH).min(ch.max(1)) {
                                fr[c] = pcm16[f * dch + c] as f32 / 32768.0;
                            }
                            sm.pcm.push_back(fr);
                            sm.decoded_samples += 1;
                        }
                        produced = true;
                    }
                    sm.byte_pos += take as u32;
                }
                WF_PCM_I16L | WF_PCM_F32L => {
                    let bps = if sm.wf_type == WF_PCM_I16L { 2 } else { 4 } * ch;
                    let frames_left = ((b.size - sm.byte_pos) as usize / bps).min(1024);
                    let p = sm.byte_pos as usize;
                    if frames_left > 0 && sm.copy.len() >= p + frames_left * bps {
                        for f in 0..frames_left {
                            let s = sm.sample_pos;
                            sm.sample_pos += 1;
                            if s < b.skip || s >= b.skip + b.samples {
                                continue;
                            }
                            let mut fr = [0f32; MAX_CH];
                            for c in 0..ch.min(MAX_CH) {
                                let at = &sm.copy[p + f * bps..];
                                fr[c] = if sm.wf_type == WF_PCM_I16L {
                                    i16::from_le_bytes([at[c * 2], at[c * 2 + 1]]) as f32 / 32768.0
                                } else {
                                    f32::from_le_bytes([at[c * 4], at[c * 4 + 1], at[c * 4 + 2], at[c * 4 + 3]])
                                };
                            }
                            sm.pcm.push_back(fr);
                            sm.decoded_samples += 1;
                        }
                        produced = true;
                    }
                    sm.byte_pos += (frames_left * bps) as u32;
                    if frames_left == 0 {
                        sm.byte_pos = b.size;
                    }
                }
                WF_VAG => {
                    // PS-ADPCM: per channel, 16-byte frames (shift/filter byte, flags
                    // byte, 28 4-bit samples); channels interleave frame by frame.
                    let frame_bytes = 16 * ch;
                    let frames_left = ((b.size - sm.byte_pos) as usize / frame_bytes).min(64);
                    let p = sm.byte_pos as usize;
                    if frames_left > 0 && sm.copy.len() >= p + frames_left * frame_bytes {
                        for f in 0..frames_left {
                            let mut out = [[0f32; MAX_CH]; 28];
                            for c in 0..ch.min(MAX_CH) {
                                let at = p + f * frame_bytes + c * 16;
                                let decoded = vag_frame(&sm.copy[at..at + 16], &mut sm.vag_hist[c]);
                                for (k, x) in decoded.iter().enumerate() {
                                    out[k][c] = *x;
                                }
                            }
                            for o in out {
                                let s = sm.sample_pos;
                                sm.sample_pos += 1;
                                if s < b.skip || s >= b.skip + b.samples {
                                    continue;
                                }
                                sm.pcm.push_back(o);
                                sm.decoded_samples += 1;
                            }
                        }
                        produced = true;
                    }
                    sm.byte_pos += (frames_left * frame_bytes) as u32;
                    if frames_left == 0 {
                        sm.byte_pos = b.size;
                    }
                }
                _ => {
                    // Unsupported formats play silence of the block's length.
                    for _ in 0..b.samples.min(4096) {
                        sm.pcm.push_back([0.0; MAX_CH]);
                    }
                    sm.sample_pos = b.skip + b.samples;
                    produced = true;
                }
            }
        }
        let block_done = sm.sample_pos >= b.skip + b.samples || sm.byte_pos >= b.size || b.addr == 0;
        if block_done {
            if sm.repeats_left > 0 {
                if sm.repeats_left != u32::MAX {
                    sm.repeats_left -= 1;
                }
                sm.byte_pos = 0;
                sm.sample_pos = 0;
                take_block(sm, &b);
            } else {
                sm.blocks.pop_front();
                sm.started_block = false;
                sm.copy = Vec::new();
            }
        }
        if produced {
            return true;
        }
    }
    if !sm.error_logged {
        sm.error_logged = true;
        crate::warn!("NGS2: a sampler voice's blocks produce no samples (stopped)");
    }
    sm.blocks.clear();
    sm.started_block = false;
    sm.copy = Vec::new();
    false
}

/// Renders `grain` frames of a sampler voice at the system rate.
fn render_sampler(v: &mut Voice, grain: usize, rate: u32, out: &mut Vec<[f32; MAX_CH]>) {
    out.clear();
    let v = &mut *v;
    let sm = &mut v.sampler;
    let step = (sm.rate as f64 / rate.max(1) as f64) * sm.pitch.max(0.0) as f64;
    let mut ended = false;
    for _ in 0..grain {
        // Next frame for interpolation (decoded on demand).
        if sm.pcm.is_empty() && !ended && !decode_more(sm) {
            ended = true;
        }
        let next = sm.pcm.front().copied().unwrap_or([0.0; MAX_CH]);
        let t = sm.phase as f32;
        let h = v.env.step();
        let mut fr = [0f32; MAX_CH];
        for c in 0..MAX_CH {
            fr[c] = (sm.cur[c] + (next[c] - sm.cur[c]) * t) * h;
        }
        out.push(fr);
        if v.env.done() {
            // (The release points ran out: the voice ends.)
            ended = true;
            sm.pcm.clear();
            sm.blocks.clear();
            sm.started_block = false;
            sm.copy = Vec::new();
            break;
        }
        sm.phase += step;
        while sm.phase >= 1.0 {
            sm.phase -= 1.0;
            if let Some(n) = sm.pcm.pop_front() {
                sm.cur = n;
            } else if !ended && decode_more(sm) {
                if let Some(n) = sm.pcm.pop_front() {
                    sm.cur = n;
                }
            } else {
                ended = true;
                sm.cur = [0.0; MAX_CH];
                break;
            }
        }
    }
    // (A grain cut short by the envelope's end: silence for the rest.)
    out.resize(grain, [0.0; MAX_CH]);
    if ended && sm.pcm.is_empty() && sm.blocks.is_empty() {
        if trace_voices() {
            crate::info!("NGS2 voice (rack {:#x} #{}) ran out of data after {} samples ({} bytes decoded)", v.rack, v.index, sm.decoded_samples, sm.decoded_bytes);
        }
        v.flags = (v.flags & !STATE_PLAYING) | STATE_STOPPED | STATE_EMPTY;
        v.flags &= !STATE_IN_USE;
    }
}

/// Mixes `src` (`src_ch` channels) of voice `vh` along its patches into the
/// destination voices' input buses.
fn route(s: &mut State, vh: u64, src: &[[f32; MAX_CH]], src_ch: usize) {
    let src_ch = src_ch.min(MAX_CH);
    let Some(v) = s.voices.get(&vh) else { return };
    // Each patch's destination, its channels, the port's gain and level
    // matrix (rows per destination channel), before the destinations are
    // borrowed for mixing.
    let mut jobs: smallvec::SmallVec<[(u64, usize, f32, Option<[[f32; MAX_CH]; MAX_CH]>); 8]> = smallvec::SmallVec::new();
    let fold = s.port_fold || crate::tune::get(crate::tune::NGS2_COMPAT) & 2 != 0;
    for p in &v.patches {
        // The title patches to rack handles of one-voice racks (mastering,
        // submixers): their first voice.
        let dest = if s.voices.contains_key(&p.dest) { p.dest } else { match s.racks.get(&p.dest).and_then(|r| r.voices.first().copied()) { Some(v) => v, None => continue } };
        let Some(d) = s.voices.get(&dest) else { continue };
        if d.kind == Kind::Sampler {
            continue;
        }
        let dch = (d.channels as usize).clamp(1, MAX_CH);
        let port = if fold { p.port as usize & 3 } else { p.port as usize };
        if port >= MAX_PORTS {
            continue;
        }
        // Levels per destination channel, source channels inner (aLevel[dst * src_ch + src]).
        let rows = v.matrices.get(&v.port_matrix[port]).filter(|m| m.len() >= src_ch * dch).map(|m| {
            let mut rows = [[0f32; MAX_CH]; MAX_CH];
            for (oc, row) in rows.iter_mut().enumerate().take(dch) {
                row[..src_ch].copy_from_slice(&m[oc * src_ch..oc * src_ch + src_ch]);
            }
            rows
        });
        jobs.push((dest, dch, v.port_volume[port], rows));
    }
    let vec = crate::tune::get(crate::tune::NGS2_VEC) != 0;
    for (dest, dch, g, rows) in jobs {
        let Some(d) = s.voices.get_mut(&dest) else { continue };
        if d.input.len() < src.len() {
            d.input.resize(src.len(), [0.0; MAX_CH]);
        }
        match &rows {
            Some(rows) if vec => mix_rows(&mut d.input, src, rows, src_ch, dch, g),
            Some(rows) => mix_rows_lanes(&mut d.input, src, rows, src_ch, dch, g),
            // No matrix: channel to channel; mono feeds the front pair.
            None if src_ch == 1 && dch >= 2 => {
                for (o, fr) in d.input.iter_mut().zip(src) {
                    o[0] += fr[0] * g;
                    o[1] += fr[0] * g;
                }
            }
            None if vec => mix_direct(&mut d.input, src, src_ch.min(dch), g),
            None => mix_direct_lanes(&mut d.input, src, src_ch.min(dch), g),
        }
    }
}

/// dst[oc] += g * sum over ic < src_ch of src[ic] * rows[oc][ic], for the
/// destination channels oc < dch (the rest untouched). By source channel
/// (the matrix transposed): each source sample scales every destination
/// channel at once (vector operations), each destination's sum in the same
/// order as row by row.
fn mix_rows(dst: &mut [[f32; MAX_CH]], src: &[[f32; MAX_CH]], rows: &[[f32; MAX_CH]; MAX_CH], src_ch: usize, dch: usize, g: f32) {
    let (src_ch, dch) = (src_ch.min(MAX_CH), dch.min(MAX_CH));
    let mut cols = [[0f32; MAX_CH]; MAX_CH];
    for (oc, row) in rows.iter().enumerate().take(dch) {
        for ic in 0..src_ch {
            cols[ic][oc] = row[ic];
        }
    }
    let m = lane_mask(dch);
    for (o, fr) in dst.iter_mut().zip(src) {
        let mut acc = [0f32; MAX_CH];
        for ic in 0..src_ch {
            let x = fr[ic];
            for oc in 0..MAX_CH {
                acc[oc] += x * cols[ic][oc];
            }
        }
        // (Every lane in vector form, those from dch on kept by the mask: a
        // loop over the first dch ran lane by lane, ~14% of the sound thread
        // in busy areas.)
        for c in 0..MAX_CH {
            let v = o[c] + acc[c] * g;
            o[c] = f32::from_bits((v.to_bits() & m[c]) | (o[c].to_bits() & !m[c]));
        }
    }
}

/// dst[c] += src[c] * g for the channels c < n (the rest untouched), every
/// lane in vector form and the lanes from n on kept by a mask. (The masked
/// lanes of a per-lane form went through closures the compiler kept: ~8% of
/// the sound thread.)
fn mix_direct(dst: &mut [[f32; MAX_CH]], src: &[[f32; MAX_CH]], n: usize, g: f32) {
    let m = lane_mask(n.min(MAX_CH));
    for (o, fr) in dst.iter_mut().zip(src) {
        for c in 0..MAX_CH {
            let v = o[c] + fr[c] * g;
            o[c] = f32::from_bits((v.to_bits() & m[c]) | (o[c].to_bits() & !m[c]));
        }
    }
}

/// `mix_rows` as before the masks (`tune ngs2_vec` 0): the destination's
/// first dch lanes in a loop.
fn mix_rows_lanes(dst: &mut [[f32; MAX_CH]], src: &[[f32; MAX_CH]], rows: &[[f32; MAX_CH]; MAX_CH], src_ch: usize, dch: usize, g: f32) {
    let (src_ch, dch) = (src_ch.min(MAX_CH), dch.min(MAX_CH));
    let mut cols = [[0f32; MAX_CH]; MAX_CH];
    for (oc, row) in rows.iter().enumerate().take(dch) {
        for ic in 0..src_ch {
            cols[ic][oc] = row[ic];
        }
    }
    for (o, fr) in dst.iter_mut().zip(src) {
        let mut acc = [0f32; MAX_CH];
        for ic in 0..src_ch {
            let x = fr[ic];
            for oc in 0..MAX_CH {
                acc[oc] += x * cols[ic][oc];
            }
        }
        for (oc, &a) in o[..dch].iter_mut().zip(&acc[..dch]) {
            *oc += a * g;
        }
    }
}

/// `mix_direct` as before the masks (`tune ngs2_vec` 0).
fn mix_direct_lanes(dst: &mut [[f32; MAX_CH]], src: &[[f32; MAX_CH]], n: usize, g: f32) {
    let n = n.min(MAX_CH);
    if n == MAX_CH {
        for (o, fr) in dst.iter_mut().zip(src) {
            for c in 0..MAX_CH {
                o[c] += fr[c] * g;
            }
        }
        return;
    }
    for (o, fr) in dst.iter_mut().zip(src) {
        for (oc, &x) in o[..n].iter_mut().zip(&fr[..n]) {
            *oc += x * g;
        }
    }
}

/// The largest |x| in the first `ch` channels of `frames` (what folding
/// `f32::max` gives: NaNs are passed over). Per lane over all frames, then
/// over the first `ch` lanes: the same comparisons per value, in vector
/// form.
fn peak_of(frames: &[[f32; MAX_CH]], ch: usize) -> f32 {
    let ch = ch.min(MAX_CH);
    let mut lanes = [0f32; MAX_CH];
    for f in frames {
        for c in 0..MAX_CH {
            let a = f[c].abs();
            if a > lanes[c] {
                lanes[c] = a;
            }
        }
    }
    let mut m = 0f32;
    for &x in &lanes[..ch] {
        if x > m {
            m = x;
        }
    }
    m
}

/// The measurement gain of the reverb racks' wet output (`tune::REV_DB`,
/// `tune::REV_AB`) for the grain starting at rendered sample `at`; logs the
/// start of each A/B window with its frame in the audio dump.
fn reverb_ab_gain(s: &mut State, at: u64, rate: u32) -> f32 {
    let mut db = crate::tune::get(crate::tune::REV_DB) as f32;
    let win = crate::tune::get(crate::tune::REV_AB);
    s.port_fold = false;
    if win > 0 {
        // (dB, ports 4-7 folded onto 0-3)
        const STEPS: [(f32, bool); 5] = [(0.0, false), (12.0, false), (18.0, false), (24.0, false), (0.0, true)];
        let w = at / (win as u64 * rate.max(1) as u64);
        let (d, fold) = STEPS[(w % STEPS.len() as u64) as usize];
        db += d;
        s.port_fold = fold;
        if s.rev_ab_window != Some(w) {
            s.rev_ab_window = Some(w);
            crate::info!("NGS2 reverb A/B: {db:+.0} dB{} from rendered sample {at} (port 1 dump frame {})", if fold { ", ports folded" } else { "" }, crate::audio::dumped_frames(0));
        }
    }
    if db == 0.0 { 1.0 } else { 10f32.powf(db / 20.0) }
}

/// MXCSR with flush-to-zero and denormals-are-zero (bits 15 and 6) for the
/// mixer's own arithmetic while it lives (`tune ngs2_ftz`), the caller's
/// restored after.
struct FtzGuard(Option<u32>);

impl FtzGuard {
    fn new() -> FtzGuard {
        if crate::tune::get(crate::tune::NGS2_FTZ) == 0 {
            return FtzGuard(None);
        }
        let mut old = 0u32;
        unsafe { std::arch::asm!("stmxcsr [{0}]", in(reg) &mut old, options(nostack)) };
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| crate::info!("NGS2: the rendering thread's MXCSR is {old:#06x}; the mixer computes with flush-to-zero and denormals-are-zero"));
        let new = old | 0x8040;
        if new == old {
            return FtzGuard(None);
        }
        unsafe { std::arch::asm!("ldmxcsr [{0}]", in(reg) &new, options(nostack)) };
        FtzGuard(Some(old))
    }
}

impl Drop for FtzGuard {
    fn drop(&mut self) {
        if let Some(old) = self.0 {
            unsafe { std::arch::asm!("ldmxcsr [{0}]", in(reg) &old, options(nostack)) };
        }
    }
}

/// A reverb's or delay's state under this (-180 dB) has died away (`tune
/// ngs2_idle`): it is cleared, and the effect skipped while its input stays
/// silent.
const IDLE_EPS: f32 = 1e-9;
/// Silent grains between checks of an effect's state (a scan of its delay
/// lines).
const IDLE_SCAN: u32 = 8;
/// `State::idle` counters: reverb and delay grains skipped / run, submixer
/// filter grains skipped / run, voice outputs not mixed (silent) / mixed.
const IDLE_FX_SKIPPED: usize = 0;
const IDLE_FX_RUN: usize = 1;
const IDLE_EQ_SKIPPED: usize = 2;
const IDLE_EQ_RUN: usize = 3;
const IDLE_UNMIXED: usize = 4;
const IDLE_MIXED: usize = 5;

/// One sampler voice's grain before routing (`Voice::out`): its samples, its
/// filters and compressor, its peak; then its channels and whether the
/// output is silent (`out_ch`, `out_silent`). Touches the voice alone.
fn sampler_grain(v: &mut Voice, grain: usize, rate: u32, idle: bool) {
    let mut out = std::mem::take(&mut v.out);
    render_sampler(v, grain, rate, &mut out);
    let ch = (v.sampler.channels as usize).clamp(1, MAX_CH);
    if trace_voices() {
        note_dc(v, &out, ch);
    }
    run_eq(v, &mut out, ch, rate);
    v.peak = peak_of(&out, ch);
    v.peak_win = v.peak_win.max(v.peak);
    v.out = out;
    v.out_ch = ch;
    v.out_silent = idle && v.peak == 0.0;
}

/// Fewer playing sampler voices than this are rendered without the helper
/// (handing a share over costs two thread wake-ups per grain).
const HELPER_MIN: usize = 8;

/// The mixer's helper thread (`tune ngs2_threads`): it renders a share of
/// each grain's sampler voices (`sampler_grain`) while the rendering thread
/// does the rest; the rendering thread waits for it before routing. It runs
/// only while the rendering thread holds the NGS2 lock and waits, so nothing
/// else touches those voices (ATRAC9 decoders are made only under the lock,
/// as the library's shared tables are written then).
struct Helper {
    /// Job number, voices (`*mut Voice` as usize), grain, rate, idle.
    job: Mutex<(u64, Vec<usize>, usize, u32, bool)>,
    job_cv: parking_lot::Condvar,
    done: Mutex<u64>,
    done_cv: parking_lot::Condvar,
}

impl Helper {
    /// Hands `voices` over; returns the job's number.
    fn give(&self, voices: Vec<usize>, grain: usize, rate: u32, idle: bool) -> u64 {
        let seq = {
            let mut j = self.job.lock();
            *j = (j.0 + 1, voices, grain, rate, idle);
            j.0
        };
        self.job_cv.notify_one();
        seq
    }

    /// Waits until job `seq` is done.
    fn wait(&self, seq: u64) {
        let mut d = self.done.lock();
        while *d < seq {
            self.done_cv.wait(&mut d);
        }
    }

    fn run(&'static self) {
        crate::log::set_thread_name("NGS2-helper");
        // (The title's sound thread runs raised: SceKernel priority 256.)
        crate::thread::raise_priority();
        let mut last = 0;
        loop {
            let (seq, voices, grain, rate, idle) = {
                let mut j = self.job.lock();
                while j.0 == last {
                    self.job_cv.wait(&mut j);
                }
                (j.0, std::mem::take(&mut j.1), j.2, j.3, j.4)
            };
            last = seq;
            {
                // (The same floating-point state as the rendering thread's.)
                let _ftz = FtzGuard::new();
                for v in voices {
                    // SAFETY: `render` hands over distinct voices and waits.
                    sampler_grain(unsafe { &mut *(v as *mut Voice) }, grain, rate, idle);
                }
            }
            *self.done.lock() = seq;
            self.done_cv.notify_one();
        }
    }
}

/// The helper thread, started at first use.
fn helper() -> &'static Helper {
    static H: std::sync::OnceLock<&'static Helper> = std::sync::OnceLock::new();
    H.get_or_init(|| {
        let h: &'static Helper = Box::leak(Box::new(Helper {
            job: Mutex::new((0, Vec::new(), 0, 0, false)),
            job_cv: parking_lot::Condvar::new(),
            done: Mutex::new(0),
            done_cv: parking_lot::Condvar::new(),
        }));
        crate::thread::spawn_host("NGS2-helper", 1 << 20, Box::new(move || h.run())).expect("NGS2 helper thread");
        h
    })
}

/// A voice's filters and compressor have no state left: on silent input
/// they give silence, bit for bit (flat ones are not run at all).
fn eq_at_rest(v: &Voice) -> bool {
    v.eq.iter().flatten().all(|f| f.is_flat() || f.at_rest()) && v.comp.as_ref().is_none_or(|c| c.is_flat() || c.at_rest())
}

/// One grain of the system: samplers, then mixing voices in patch order, then
/// the mastering voice into the render buffers. With `idle` (`tune
/// ngs2_idle`), what silence passes through unchanged is skipped: the mixing
/// of silent outputs, a submixer's filters on silent input while they are at
/// rest, reverbs and delays on silent input once they have died away.
fn render(s: &mut State, system: u64, grain: usize, rate: u32, idle: bool, threads: bool) -> Vec<[f32; MAX_CH]> {
    let _ftz = FtzGuard::new();
    let at = s.systems.get(&system).map_or(0, |y| y.renders.saturating_sub(1)) * grain as u64;
    let rev_gain = reverb_ab_gain(s, at, rate);
    let racks: Vec<(u64, u32)> = s.racks.iter().filter(|(_, r)| r.system == system).map(|(&h, r)| (h, r.rack_id)).collect();
    let mut all: Vec<u64> = Vec::new();
    for (rh, _) in &racks {
        all.extend(s.racks[rh].voices.iter().copied());
    }
    for vh in &all {
        if let Some(v) = s.voices.get_mut(vh) {
            if v.kind != Kind::Sampler {
                v.input.clear();
                v.input.resize(grain, [0.0; MAX_CH]);
            }
        }
    }
    // The playing sampler voices' grains (each voice's own state alone: its
    // decoding, envelope, filters, peak), with a share on the helper thread
    // (`tune ngs2_threads`), then their routing in the voices' order - the
    // same mix as voice by voice.
    let mut playing: Vec<(u64, *mut Voice)> = Vec::new();
    for vh in &all {
        let Some(v) = s.voices.get_mut(vh) else { continue };
        if v.kind == Kind::Sampler && v.flags & STATE_PLAYING != 0 && v.flags & STATE_PAUSED == 0 {
            playing.push((*vh, v as *mut Voice));
        } else {
            v.peak = 0.0;
        }
    }
    // (A voice listed twice - never seen - keeps the helper out.)
    let distinct = {
        let mut p: Vec<usize> = playing.iter().map(|&(_, v)| v as usize).collect();
        p.sort_unstable();
        p.windows(2).all(|w| w[0] != w[1])
    };
    let helper = (distinct && threads && playing.len() >= HELPER_MIN).then(helper);
    if let Some(h) = helper {
        // (Every other voice: the formats' costs spread evenly.)
        let theirs: Vec<usize> = playing.iter().skip(1).step_by(2).map(|&(_, v)| v as usize).collect();
        let seq = h.give(theirs, grain, rate, idle);
        for &(_, v) in playing.iter().step_by(2) {
            // SAFETY: the voices are distinct entries of `s.voices`, which
            // nothing changes until the helper is done with its share.
            sampler_grain(unsafe { &mut *v }, grain, rate, idle);
        }
        h.wait(seq);
    } else {
        for &(_, v) in &playing {
            sampler_grain(unsafe { &mut *v }, grain, rate, idle);
        }
    }
    for &(vh, _) in &playing {
        let (ch, silent, out) = {
            let v = s.voices.get_mut(&vh).unwrap();
            (v.out_ch, v.out_silent, std::mem::take(&mut v.out))
        };
        if silent {
            s.idle[IDLE_UNMIXED] += 1;
        } else {
            s.idle[IDLE_MIXED] += 1;
            route(s, vh, &out, ch);
        }
        s.voices.get_mut(&vh).unwrap().out = out;
    }
    // Mixing voices: deepest upstream first (a voice runs after everything that
    // patches into it; cycles are cut).
    let mixers: Vec<u64> = all.iter().copied().filter(|vh| s.voices.get(vh).is_some_and(|v| matches!(v.kind, Kind::Submixer | Kind::Reverb | Kind::Custom))).collect();
    let mut depth: FastMap<u64, u32> = FastMap::default();
    fn depth_of(s: &State, vh: u64, memo: &mut FastMap<u64, u32>, guard: u32) -> u32 {
        if let Some(&d) = memo.get(&vh) {
            return d;
        }
        if guard > 16 {
            return 0;
        }
        let d = s.voices.get(&vh).map(|v| v.patches.iter().map(|p| {
            let dest = s.racks.get(&p.dest).and_then(|r| r.voices.first().copied()).unwrap_or(p.dest);
            1 + depth_of(s, dest, memo, guard + 1)
        }).max().unwrap_or(0)).unwrap_or(0);
        memo.insert(vh, d);
        d
    }
    let mut order: Vec<(u32, u64)> = mixers.iter().map(|&vh| (depth_of(s, vh, &mut depth, 0), vh)).collect();
    order.sort_by(|a, b| b.0.cmp(&a.0));
    for (_, vh) in order {
        let (input, out, ch, silent) = {
            let v = s.voices.get_mut(&vh).unwrap();
            let ch = (v.channels as usize).clamp(1, MAX_CH);
            let mut input = std::mem::take(&mut v.input);
            let pk = peak_of(&input, ch);
            v.peak_win = v.peak_win.max(pk);
            // (Silent input: the voice's own state alone makes its output. A
            // reverb's or delay's is checked every IDLE_SCAN silent grains.)
            let quiet = idle && pk == 0.0;
            v.quiet = if quiet { v.quiet.saturating_add(1) } else { 0 };
            let scan = quiet && v.quiet % IDLE_SCAN == 0;
            match v.kind {
                Kind::Reverb | Kind::Custom if quiet && v.fx_idle => {
                    s.idle[IDLE_FX_SKIPPED] += 1;
                    (input, None, ch, true)
                }
                Kind::Reverb => {
                    let och = (v.out_channels as usize).clamp(1, MAX_CH);
                    let out = reverb_grain(v, &input, ch, och, rate, rev_gain);
                    v.fx_idle = scan && v.reverb.as_mut().is_some_and(|r| r.settle(IDLE_EPS));
                    s.idle[IDLE_FX_RUN] += 1;
                    (input, Some(out), och, false)
                }
                Kind::Custom if v.taps.is_some() => {
                    let och = (v.out_channels as usize).clamp(1, MAX_CH);
                    let t = v.taps.as_mut().unwrap();
                    let out: Vec<[f32; MAX_CH]> = input.iter().map(|fr| t.process(fr, och.min(ch))).collect();
                    v.fx_idle = scan && t.settle(IDLE_EPS);
                    s.idle[IDLE_FX_RUN] += 1;
                    (input, Some(out), och, false)
                }
                _ => {
                    let rest = quiet && eq_at_rest(v);
                    if v.kind == Kind::Submixer {
                        if rest {
                            s.idle[IDLE_EQ_SKIPPED] += 1;
                        } else {
                            s.idle[IDLE_EQ_RUN] += 1;
                            run_eq(v, &mut input, ch, rate);
                        }
                    }
                    run_user_fx(v, &mut input, rate);
                    (input, None, ch, rest && v.user_fx.is_none())
                }
            }
        };
        if silent {
            s.idle[IDLE_UNMIXED] += 1;
        } else {
            s.idle[IDLE_MIXED] += 1;
            route(s, vh, out.as_deref().unwrap_or(&input), ch);
        }
        if let Some(v) = s.voices.get_mut(&vh) {
            v.input = input;
        }
    }
    // Mastering voices: their inputs, scaled by their gain, summed.
    let mut out = vec![[0f32; MAX_CH]; grain];
    for vh in &all {
        let Some(v) = s.voices.get_mut(vh) else { continue };
        if v.kind != Kind::Mastering {
            continue;
        }
        let pk = peak_of(&v.input[..v.input.len().min(grain)], MAX_CH);
        v.peak_win = v.peak_win.max(pk);
        if v.limiter.as_ref().is_none_or(|l| l.rate() != rate) {
            v.limiter = Some(Limiter::new(rate));
        }
        let (gain, lfe, limit) = (v.gain, v.lfe_gain, v.limit);
        let Voice { input, limiter, .. } = v;
        let lim = limiter.as_mut().unwrap();
        for (i, fr) in input.iter().enumerate().take(grain) {
            let mut x: [f32; MAX_CH] = *fr;
            for (c, s) in x.iter_mut().enumerate() {
                *s *= if c == 3 { lfe } else { gain };
            }
            lim.process(&mut x, limit);
            for c in 0..MAX_CH {
                out[i][c] += x[c];
            }
        }
    }
    out
}

/// The mastering voices' gain and limiter settings (logged the first times).
fn note_mastering(what: &str, vh: u64, a: f32, b: f32) {
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
        crate::info!("NGS2 mastering {what} on {vh:#x}: {a} {b}");
    }
}

/// TLG_NGS2_TRACE: a sampler voice whose grain has a large DC level (a
/// mean far from 0: a thump or an offset in the waveform), named with what it
/// plays (the first 32).
fn note_dc(v: &Voice, buf: &[[f32; MAX_CH]], ch: usize) {
    if buf.is_empty() {
        return;
    }
    let n = buf.len() as f32;
    let dc = (0..ch).map(|c| buf.iter().map(|f| f[c]).sum::<f32>() / n).fold(0f32, |m, x| if x.abs() > m.abs() { x } else { m });
    if dc.abs() < 0.2 {
        return;
    }
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 32 {
        let sm = &v.sampler;
        crate::info!(
            "NGS2 voice (rack {:#x} #{}) grain DC {dc:.3}: waveform type {:#x}, {} ch at {} Hz, pitch {:.3}, data {:#x}, {} samples decoded, filters {:?}",
            v.rack,
            v.index,
            sm.wf_type,
            sm.channels,
            sm.rate,
            sm.pitch,
            sm.data,
            sm.decoded_samples,
            v.eq_params.iter().flatten().map(|p| (p.0, p.1 as u32)).collect::<Vec<_>>()
        );
    }
}

/// One grain of a reverb rack: the reverberation of the input's mean (front
/// pair; no centre or LFE) times wet, plus the input times dry.
fn reverb_grain(v: &mut Voice, input: &[[f32; MAX_CH]], ich: usize, och: usize, rate: u32, wet_gain: f32) -> Vec<[f32; MAX_CH]> {
    let p = v.i3dl2;
    if !v.reverb.as_ref().is_some_and(|r| r.matches(&p, rate)) {
        v.reverb = Some(Reverb::new(p, rate));
    }
    let r = v.reverb.as_mut().unwrap();
    let dry = p.dry;
    input
        .iter()
        .map(|fr| {
            let mono = fr[..ich].iter().sum::<f32>() / ich as f32;
            let [l, rr] = r.process(mono);
            let mut o = [0f32; MAX_CH];
            o[0] = l * wet_gain;
            if och > 1 {
                o[1] = rr * wet_gain;
            }
            if dry != 0.0 {
                for c in 0..och.min(ich) {
                    o[c] += fr[c] * dry;
                }
            }
            o
        })
        .collect()
}

sysv! {
    fn sceNgs2SystemQueryBufferSize(option: *const u8, out: *mut u64) -> i32 {
        if out.is_null() { return ERR_INVALID_OUT_ADDRESS; }
        *out.add(1) = SYSTEM_BUFFER;
        0
    }
    /// SceNgs2SystemOption: {size_t size; char name[16]; u32 flags, maxGrainSamples, numGrainSamples, sampleRate}
    fn sceNgs2SystemCreateWithAllocator(option: *const u8, allocator: *const u64, out: *mut u64) -> i32 {
        if out.is_null() { return ERR_INVALID_OUT_ADDRESS; }
        let (grain, rate) = if option.is_null() { (256, 48000) } else { (*(option.add(32) as *const u32), *(option.add(36) as *const u32)) };
        let buf = match allocate(allocator, SYSTEM_BUFFER) { Ok(b) => b, Err(e) => return e };
        with(|s| s.systems.insert(buf.buffer, System { buf, sample_rate: if rate == 0 { 48000 } else { rate }, grain: if grain == 0 { 256 } else { grain }, renders: 0 }));
        *out = buf.buffer;
        crate::info!("NGS2 system: grain {grain} rate {rate} -> handle {:#x}", buf.buffer);
        0
    }
    fn sceNgs2SystemCreate(option: *const u8, info: *const u64, out: *mut u64) -> i32 {
        if out.is_null() { return ERR_INVALID_OUT_ADDRESS; }
        if info.is_null() || *info == 0 { return ERR_INVALID_BUFFER; }
        let (grain, rate) = if option.is_null() { (256, 48000) } else { (*(option.add(32) as *const u32), *(option.add(36) as *const u32)) };
        let buf = BufferInfo { buffer: *info, size: *info.add(1), user_data: *info.add(7), free_handler: 0 };
        with(|s| s.systems.insert(buf.buffer, System { buf, sample_rate: if rate == 0 { 48000 } else { rate }, grain: if grain == 0 { 256 } else { grain }, renders: 0 }));
        *out = buf.buffer;
        0
    }
    fn sceNgs2SystemDestroy(h: u64, out: *mut u64) -> i32 {
        let Some(sys) = with(|s| s.systems.remove(&h)) else { return ERR_INVALID_HANDLE };
        if !out.is_null() { *out = sys.buf.buffer; *out.add(1) = sys.buf.size; *out.add(7) = sys.buf.user_data; }
        release(&sys.buf);
        0
    }
    fn sceNgs2RackQueryBufferSize(rack_id: u32, option: *const u8, out: *mut u64) -> i32 {
        if out.is_null() { return ERR_INVALID_OUT_ADDRESS; }
        *out.add(1) = rack_buffer(rack_option_voices(option));
        0
    }
    fn sceNgs2RackCreateWithAllocator(system: u64, rack_id: u32, option: *const u8, allocator: *const u64, out: *mut u64) -> i32 {
        if out.is_null() { return ERR_INVALID_OUT_ADDRESS; }
        if !with(|s| s.systems.contains_key(&system)) { return ERR_INVALID_HANDLE; }
        let buf = match allocate(allocator, rack_buffer(rack_option_voices(option))) { Ok(b) => b, Err(e) => return e };
        *out = create_rack(system, rack_id, option, buf);
        0
    }
    fn sceNgs2RackCreate(system: u64, rack_id: u32, option: *const u8, info: *const u64, out: *mut u64) -> i32 {
        if out.is_null() { return ERR_INVALID_OUT_ADDRESS; }
        if !with(|s| s.systems.contains_key(&system)) { return ERR_INVALID_HANDLE; }
        if info.is_null() || *info == 0 || *info.add(1) < rack_buffer(rack_option_voices(option)) { return ERR_INVALID_BUFFER; }
        let buf = BufferInfo { buffer: *info, size: *info.add(1), user_data: *info.add(7), free_handler: 0 };
        *out = create_rack(system, rack_id, option, buf);
        0
    }
    fn sceNgs2RackDestroy(h: u64, out: *mut u64) -> i32 {
        let Some(rack) = with(|s| {
            let r = s.racks.remove(&h)?;
            for v in &r.voices { s.voices.remove(v); }
            // Patches into the destroyed voices go nowhere.
            for v in s.voices.values_mut() { v.patches.retain(|p| !r.voices.contains(&p.dest)); }
            Some(r)
        }) else { return ERR_INVALID_HANDLE };
        if !out.is_null() { *out = rack.buf.buffer; *out.add(1) = rack.buf.size; *out.add(7) = rack.buf.user_data; }
        release(&rack.buf);
        0
    }
    fn sceNgs2RackGetVoiceHandle(rack: u64, index: u32, out: *mut u64) -> i32 {
        if out.is_null() { return ERR_INVALID_OUT_ADDRESS; }
        match with(|s| s.racks.get(&rack).map(|r| r.voices.get(index as usize).copied())) {
            None => ERR_INVALID_HANDLE,
            Some(None) => ERR_INVALID_VOICE_INDEX,
            Some(Some(v)) => { *out = v; 0 }
        }
    }
    fn sceNgs2VoiceControl(voice: u64, params: *const u8) -> i32 {
        let mut g = NGS2.lock();
        let s = g.get_or_insert_with(State::default);
        if !s.voices.contains_key(&voice) { return ERR_INVALID_HANDLE; }
        control(s, voice, params)
    }
    fn sceNgs2VoiceGetStateFlags(voice: u64, out: *mut u32) -> i32 {
        if out.is_null() { return ERR_INVALID_OUT_ADDRESS; }
        match with(|s| s.voices.get(&voice).map(|v| v.flags)) { Some(f) => { *out = f; 0 } None => ERR_INVALID_HANDLE }
    }
    /// SceNgs2SamplerVoiceState {u32 stateFlags; float envelopeHeight, peakHeight; u32 -;
    /// u64 numDecodedSamples, decodedDataSize; uintptr userData; const void *waveformData}.
    fn sceNgs2VoiceGetState(voice: u64, out: *mut u8, size: u64) -> i32 {
        if out.is_null() { return ERR_INVALID_OUT_ADDRESS; }
        let Some((flags, samples, bytes, user, data, peak, before, env)) = with(|s| s.voices.get_mut(&voice).map(|v| {
            let sm = &v.sampler;
            let data = sm.blocks.front().map(|b| b.addr).unwrap_or(sm.data);
            let before = std::mem::replace(&mut v.reported, v.flags);
            let env = if v.env.points.is_empty() { 1.0 } else { v.env.height };
            (v.flags, sm.decoded_samples, sm.decoded_bytes, sm.user_data, data, v.peak, before, env)
        })) else { return ERR_INVALID_HANDLE };
        std::ptr::write_bytes(out, 0, size as usize);
        if trace_voices() && flags != before {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4000 {
                crate::info!("NGS2 state of {voice:#x} ({size} bytes): flags {before:#x} -> {flags:#x}, decoded {samples} samples / {bytes} bytes, user {user:#x}, data {data:#x}");
            }
        }
        if size >= 4 { *(out as *mut u32) = flags; }
        if size >= 8 { *(out.add(4) as *mut f32) = if flags & STATE_PLAYING != 0 { env } else { 0.0 }; }
        if size >= 12 { *(out.add(8) as *mut f32) = peak; }
        if size >= 24 { *(out.add(16) as *mut u64) = samples; }
        if size >= 32 { *(out.add(24) as *mut u64) = bytes; }
        if size >= 40 { *(out.add(32) as *mut u64) = user; }
        if size >= 48 { *(out.add(40) as *mut u64) = data; }
        0
    }
    /// Renders one grain into each buffer ({buffer*, size, waveformType, numChannels}).
    fn sceNgs2SystemRender(system: u64, infos: *const u8, count: u32) -> i32 {
        let mut g = NGS2.lock();
        let s = g.get_or_insert_with(State::default);
        let Some((grain, rate)) = s.systems.get_mut(&system).map(|y| { y.renders += 1; (y.grain as usize, y.sample_rate) }) else { return ERR_INVALID_HANDLE };
        let mix = render(s, system, grain, rate, crate::tune::get(crate::tune::NGS2_IDLE) != 0, crate::tune::get(crate::tune::NGS2_THREADS) != 0);
        // Every ~10 s: what the mixer is doing.
        let renders = s.systems[&system].renders;
        let peak = mix.iter().flat_map(|f| f.iter()).fold(0f32, |m, &x| m.max(x.abs()));
        s.peak = s.peak.max(peak);
        if renders % (10 * rate as u64 / grain.max(1) as u64).max(1) == 0 {
            let mut by_type: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();
            let mut starved = 0;
            for v in s.voices.values() {
                if v.kind == Kind::Sampler && v.flags & STATE_PLAYING != 0 {
                    *by_type.entry(v.sampler.wf_type).or_default() += 1;
                    if v.sampler.blocks.is_empty() && v.sampler.pcm.is_empty() {
                        starved += 1;
                    }
                }
            }
            let st = std::mem::take(&mut s.idle);
            crate::info!(
                "NGS2: {} sampler voices playing (by waveform type {:x?}), {starved} without data; output peak {:.4} over the last 10 s; grains skipped as idle: reverbs and delays {} of {}, submixer filters {} of {}, voice outputs {} of {} (silent, not mixed)",
                by_type.values().sum::<u32>(),
                by_type,
                s.peak,
                st[IDLE_FX_SKIPPED],
                st[IDLE_FX_SKIPPED] + st[IDLE_FX_RUN],
                st[IDLE_EQ_SKIPPED],
                st[IDLE_EQ_SKIPPED] + st[IDLE_EQ_RUN],
                st[IDLE_UNMIXED],
                st[IDLE_UNMIXED] + st[IDLE_MIXED]
            );
            s.peak = 0.0;
            if trace_voices() {
                // The mix path's levels: each voice that carried sound, with
                // its routing (port volume, matrix id, destination).
                let mut lines = Vec::new();
                for (&vh, v) in s.voices.iter() {
                    if v.peak_win <= 0.0 && !(v.kind == Kind::Sampler && v.flags & STATE_PLAYING != 0) {
                        continue;
                    }
                    let route: Vec<String> = v
                        .patches
                        .iter()
                        .map(|pa| {
                            let dest = if s.voices.contains_key(&pa.dest) { Some(pa.dest) } else { s.racks.get(&pa.dest).and_then(|r| r.voices.first().copied()) };
                            let d = dest.and_then(|d| s.voices.get(&d).map(|dv| format!("{d:#x} {:?} {}ch", dv.kind, dv.channels))).unwrap_or_else(|| "?".into());
                            let port = if crate::tune::get(crate::tune::NGS2_COMPAT) & 2 != 0 { pa.port as usize & 3 } else { (pa.port as usize).min(MAX_PORTS - 1) };
                            format!("port {} vol {:.3} matrix {} -> {d} in {}", pa.port, v.port_volume[port], v.port_matrix[port], pa.dest_input)
                        })
                        .collect();
                    let rack = s.racks.get(&v.rack).map_or(0, |r| r.rack_id);
                    let what = match v.kind {
                        Kind::Sampler => format!("type {:#x} {}ch {} Hz pitch {:.3}", v.sampler.wf_type, v.sampler.channels, v.sampler.rate, v.sampler.pitch),
                        Kind::Mastering => format!("{}ch gain {:.3}", v.channels, v.gain),
                        Kind::Submixer => format!("rack {rack:#x} {}ch{}", v.channels, if v.user_fx.is_some() { " +user effect" } else { "" }),
                        Kind::Reverb => format!("reverb {}->{}ch wet {:.2} dry {:.2} decay {:.2}s", v.channels, v.out_channels, v.i3dl2.wet, v.i3dl2.dry, v.i3dl2.decay),
                        Kind::Custom => format!("custom rack {rack:#x} {}->{}ch{}", v.channels, v.out_channels, if v.taps.is_some() { " +delay" } else { "" }),
                    };
                    lines.push(format!("{vh:#x} {:?} {what} peak {:.4}: [{}]", v.kind, v.peak_win, route.join("; ")));
                }
                lines.sort();
                for l in lines {
                    crate::info!("NGS2 voice {l}");
                }
                // Matrix level arrays the title changed in place since it
                // passed them.
                for (&vh, v) in s.voices.iter() {
                    for (&mid, &(ptr, n)) in &v.matrix_src {
                        let mut raw = vec![0u8; n * 4];
                        crate::gpu::resources::copy_from_guest(&mut raw, ptr);
                        let now: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
                        if v.matrices.get(&mid) != Some(&now) {
                            let nz: Vec<String> = now.iter().enumerate().filter(|(_, x)| **x != 0.0).map(|(i, x)| format!("{i}:{x:.3}")).collect();
                            crate::info!("NGS2 voice {vh:#x}: matrix {mid} levels at {ptr:#x} changed in place: nonzero [{}]", nz.join(" "));
                        }
                    }
                }
                for v in s.voices.values_mut() {
                    v.peak_win = 0.0;
                }
            }
        }
        for i in 0..count as usize {
            let e = infos.add(i * 24);
            let buf = *(e as *const u64) as *mut u8;
            let size = *(e.add(8) as *const u64) as usize;
            let wf = *(e.add(16) as *const u32);
            let ch = (*(e.add(20) as *const u32) as usize).clamp(1, MAX_CH);
            if buf.is_null() { continue; }
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| crate::info!("NGS2 render: {count} buffer(s), first: type {wf:#x}, {ch} channels, {size} bytes per grain of {grain}"));
            match wf {
                WF_PCM_I16L => {
                    let n = (size / 2).min(grain * ch);
                    let o = std::slice::from_raw_parts_mut(buf as *mut i16, n);
                    for (k, x) in o.iter_mut().enumerate() {
                        *x = (mix[k / ch][k % ch].clamp(-1.0, 1.0) * 32767.0) as i16;
                    }
                }
                _ => {
                    let n = (size / 4).min(grain * ch);
                    let o = std::slice::from_raw_parts_mut(buf as *mut f32, n);
                    for (k, x) in o.iter_mut().enumerate() {
                        *x = mix[k / ch][k % ch];
                    }
                }
            }
        }
        0
    }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceNgs2", [
        sceNgs2SystemQueryBufferSize, sceNgs2SystemCreateWithAllocator, sceNgs2SystemCreate, sceNgs2SystemDestroy,
        sceNgs2RackQueryBufferSize, sceNgs2RackCreateWithAllocator, sceNgs2RackCreate, sceNgs2RackDestroy,
        sceNgs2RackGetVoiceHandle, sceNgs2VoiceControl, sceNgs2VoiceGetStateFlags, sceNgs2VoiceGetState,
        sceNgs2SystemRender,
    ]);
}


#[cfg(test)]
mod tests {
    use super::*;

    /// One voice parameter ({u16 size; s16 next; u32 id}, then the payload
    /// words), alone in its chain; u64 storage keeps handles aligned.
    fn param(id: u32, words: &[u32]) -> Vec<u64> {
        let size = 8 + 4 * words.len();
        let mut b = vec![0u64; size.div_ceil(8)];
        let p = b.as_mut_ptr() as *mut u8;
        unsafe {
            *(p as *mut u16) = size as u16;
            *(p.add(4) as *mut u32) = id;
            for (i, w) in words.iter().enumerate() {
                *(p.add(8 + 4 * i) as *mut u32) = *w;
            }
        }
        b
    }

    #[test]
    fn port_4_volume_and_kill_unpatches() {
        let mut s = State::default();
        let (sam, dry, send) = (0x1_0000u64, 0x2_0000u64, 0x3_0000u64);
        s.voices.insert(sam, Voice::new(0x100, Kind::Sampler, 0));
        s.voices.insert(dry, Voice::new(0x200, Kind::Submixer, 0));
        s.voices.insert(send, Voice::new(0x300, Kind::Submixer, 0));
        let patch = |port: u32, dest: u64| param(PARAM_PATCH, &[port, 0, dest as u32, (dest >> 32) as u32]);
        let set = |s: &mut State, b: Vec<u64>| assert_eq!(unsafe { control(s, sam, b.as_ptr() as *const u8) }, 0);
        set(&mut s, patch(0, dry));
        set(&mut s, patch(4, send));
        set(&mut s, param(PARAM_PORT_VOLUME, &[4, 0.25f32.to_bits()]));
        // Mono into a stereo bus without a matrix: the front pair, at each
        // port's own volume (port 4 is not port 0).
        route(&mut s, sam, &[[1.0; MAX_CH]; 4], 1);
        assert_eq!(s.voices[&dry].input[0][..2], [1.0, 1.0]);
        assert_eq!(s.voices[&send].input[0][..2], [0.25, 0.25]);
        // KILL drops the patches: nothing reaches the old destinations.
        set(&mut s, param(PARAM_EVENT, &[EVENT_KILL]));
        assert!(s.voices[&sam].patches.is_empty());
        for d in [dry, send] {
            s.voices.get_mut(&d).unwrap().input.clear();
        }
        route(&mut s, sam, &[[1.0; MAX_CH]; 4], 1);
        assert!(s.voices[&send].input.iter().chain(&s.voices[&dry].input).all(|f| f[0] == 0.0));
    }

    #[test]
    fn peak_matches_folding_max() {
        let mut frames = vec![[0f32; MAX_CH]; 64];
        let mut x = 0x1234_5678u32;
        for f in frames.iter_mut() {
            for c in f.iter_mut() {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *c = (x as i32) as f32 / 1e9;
            }
        }
        frames[7][2] = f32::NAN;
        frames[9][0] = -0.0;
        for ch in 0..=MAX_CH {
            let want = frames.iter().flat_map(|f| f[..ch].iter()).fold(0f32, |m, &x| m.max(x.abs()));
            assert_eq!(peak_of(&frames, ch).to_bits(), want.to_bits(), "{ch} channels");
        }
        assert_eq!(peak_of(&[], 2), 0.0);
    }

    #[test]
    fn mixing_matches_row_by_row() {
        // The vector forms give the row-by-row sums bit for bit and leave
        // the destination channels past dch (or n) alone.
        let mut x = 0x9e37_79b9u32;
        let mut rnd = || {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (x as i32) as f32 / 2e9
        };
        let src: Vec<[f32; MAX_CH]> = (0..37).map(|_| std::array::from_fn(|_| rnd())).collect();
        let dst0: Vec<[f32; MAX_CH]> = (0..37).map(|_| std::array::from_fn(|_| rnd())).collect();
        let rows: [[f32; MAX_CH]; MAX_CH] = std::array::from_fn(|_| std::array::from_fn(|_| rnd()));
        for src_ch in 1..=MAX_CH {
            for dch in 1..=MAX_CH {
                let mut want = dst0.clone();
                for (o, fr) in want.iter_mut().zip(&src) {
                    for (oc, row) in rows.iter().enumerate().take(dch) {
                        let mut acc = 0.0;
                        for ic in 0..src_ch {
                            acc += fr[ic] * row[ic];
                        }
                        o[oc] += acc * 0.7;
                    }
                }
                for (form, f) in [("masked", mix_rows as fn(&mut [[f32; MAX_CH]], &[[f32; MAX_CH]], &[[f32; MAX_CH]; MAX_CH], usize, usize, f32)), ("lanes", mix_rows_lanes)] {
                    let mut got = dst0.clone();
                    f(&mut got, &src, &rows, src_ch, dch, 0.7);
                    for (g, w) in got.iter().zip(&want) {
                        assert!(g.iter().zip(w).all(|(a, b)| a.to_bits() == b.to_bits()), "rows {src_ch}->{dch} ({form})");
                    }
                }
            }
            let mut want = dst0.clone();
            for (o, fr) in want.iter_mut().zip(&src) {
                for c in 0..src_ch {
                    o[c] += fr[c] * 0.7;
                }
            }
            for (form, f) in [("masked", mix_direct as fn(&mut [[f32; MAX_CH]], &[[f32; MAX_CH]], usize, f32)), ("lanes", mix_direct_lanes)] {
                let mut got = dst0.clone();
                f(&mut got, &src, src_ch, 0.7);
                for (g, w) in got.iter().zip(&want) {
                    assert!(g.iter().zip(w).all(|(a, b)| a.to_bits() == b.to_bits()), "direct {src_ch} ({form})");
                }
            }
        }
    }

    #[test]
    fn idle_skips_change_nothing_audible() {
        // A burst through a filtered bus into a reverb, a delay and the
        // mastering voice, then silence, with and without the idle skips:
        // the same output bit for bit until the effects have died away (their
        // tails are cut under -180 dB), and the skips happen.
        fn build() -> State {
            let mut s = State::default();
            let (sam, bus, rev, dly, mst) = (0x10u64, 0x20u64, 0x30u64, 0x40u64, 0x50u64);
            for (rack, vh, kind) in [(0x100u64, sam, Kind::Sampler), (0x200, bus, Kind::Submixer), (0x300, rev, Kind::Reverb), (0x400, dly, Kind::Custom), (0x500, mst, Kind::Mastering)] {
                s.racks.insert(rack, Rack { system: 1, rack_id: 0, buf: BufferInfo::default(), voices: vec![vh] });
                s.voices.insert(vh, Voice::new(rack, kind, 0));
            }
            let patch = |port: u32, dest: u64| Patch { port, dest, dest_input: 0 };
            let v = s.voices.get_mut(&sam).unwrap();
            v.flags = STATE_IN_USE | STATE_PLAYING;
            let mut x = 0x1357_9bdfu32;
            v.sampler.pcm = (0..4800)
                .map(|_| {
                    x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let mut fr = [0f32; MAX_CH];
                    fr[0] = (x as i32) as f32 / 4e9;
                    fr
                })
                .collect();
            v.patches = vec![patch(0, bus)];
            let v = s.voices.get_mut(&bus).unwrap();
            v.eq_params[0] = Some((FilterKind::Peaking, 1000.0, 1.0, 2.0));
            v.patches = vec![patch(0, mst), patch(1, rev), patch(2, dly)];
            s.voices.get_mut(&rev).unwrap().patches = vec![patch(0, mst)];
            let words = [0, 0, 0.5f32.to_bits(), 0.3f32.to_bits(), 8000f32.to_bits(), 2, 0.6f32.to_bits(), 120f32.to_bits(), 0.4f32.to_bits(), 250f32.to_bits()];
            let v = s.voices.get_mut(&dly).unwrap();
            v.taps = Taps::from_words(&words, 48000);
            v.patches = vec![patch(0, mst)];
            s
        }
        let (mut on, mut off) = (build(), build());
        let (mut first_diff, mut worst, mut peak) = (None, 0f32, 0f32);
        for g in 0..1600 {
            let a = render(&mut on, 1, 256, 48000, true, false);
            let b = render(&mut off, 1, 256, 48000, false, false);
            for (x, y) in a.iter().zip(&b) {
                for c in 0..MAX_CH {
                    peak = peak.max(y[c].abs());
                    if x[c].to_bits() != y[c].to_bits() {
                        first_diff.get_or_insert(g);
                        worst = worst.max((x[c] - y[c]).abs());
                    }
                }
            }
        }
        assert!(peak > 0.01, "the burst is heard ({peak})");
        assert!(worst < 1e-7, "{worst}");
        assert!(first_diff.is_none_or(|g| g >= 200), "{first_diff:?}");
        assert!(on.idle[IDLE_FX_SKIPPED] > 0 && on.idle[IDLE_EQ_SKIPPED] > 0 && on.idle[IDLE_UNMIXED] > 0, "{:?}", on.idle);
        assert_eq!(off.idle[IDLE_FX_SKIPPED] + off.idle[IDLE_EQ_SKIPPED] + off.idle[IDLE_UNMIXED], 0);
    }

    #[test]
    fn helper_thread_changes_nothing() {
        // Twelve sampler voices (pitches, filters, envelopes, mono and
        // stereo, some ending) into a bus and the mastering voice: the same
        // output bit for bit with the helper thread rendering every other
        // voice as without it, and the same voice states.
        fn build() -> State {
            let mut s = State::default();
            let (bus, mst) = (0x20u64, 0x50u64);
            let patch = |port: u32, dest: u64| Patch { port, dest, dest_input: 0 };
            s.racks.insert(0x100, Rack { system: 1, rack_id: 0, buf: BufferInfo::default(), voices: (0..12).map(|i| 0x1000 + i).collect() });
            for (rack, vh, kind) in [(0x200u64, bus, Kind::Submixer), (0x500, mst, Kind::Mastering)] {
                s.racks.insert(rack, Rack { system: 1, rack_id: 0, buf: BufferInfo::default(), voices: vec![vh] });
                s.voices.insert(vh, Voice::new(rack, kind, 0));
            }
            s.voices.get_mut(&bus).unwrap().patches = vec![patch(0, mst)];
            let mut x = 0x2468_ace0u32;
            for i in 0..12u64 {
                let mut v = Voice::new(0x100, Kind::Sampler, i as u32);
                v.flags = STATE_IN_USE | STATE_PLAYING;
                let ch = 1 + (i % 2) as usize;
                v.sampler.channels = ch as u32;
                v.channels = ch as u32;
                v.sampler.pitch = 0.75 + 0.1 * i as f32;
                // (Some voices run out within the test: their end and silence.)
                v.sampler.pcm = (0..(3000 + 4000 * (i % 3)))
                    .map(|_| {
                        let mut fr = [0f32; MAX_CH];
                        for c in fr.iter_mut().take(ch) {
                            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                            *c = (x as i32) as f32 / 4e9;
                        }
                        fr
                    })
                    .collect();
                if i % 3 != 0 {
                    v.eq_params[0] = Some((FilterKind::LowPass, 2000.0 + 500.0 * i as f32, 0.7, 1.0));
                }
                if i % 4 == 1 {
                    v.eq_params[1] = Some((FilterKind::HighPass, 80.0, 0.7, 1.0));
                }
                if i % 5 == 2 {
                    v.env = Envelope::new(vec![(0, 300, 1.0), (0, 2000, 0.0)], 1);
                }
                v.patches = vec![patch(0, bus)];
                s.voices.insert(0x1000 + i, v);
            }
            s
        }
        let (mut with, mut without) = (build(), build());
        for g in 0..120 {
            let a = render(&mut with, 1, 256, 48000, true, true);
            let b = render(&mut without, 1, 256, 48000, true, false);
            assert!(a.iter().zip(&b).all(|(x, y)| x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())), "grain {g}");
        }
        for i in 0..12u64 {
            let (p, q) = (&with.voices[&(0x1000 + i)], &without.voices[&(0x1000 + i)]);
            assert_eq!((p.flags, p.peak.to_bits(), p.sampler.pcm.len()), (q.flags, q.peak.to_bits(), q.sampler.pcm.len()), "voice {i}");
        }
    }
}
