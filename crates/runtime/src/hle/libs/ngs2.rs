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
//! Not implemented yet (logged once each, audio passes through unchanged):
//! envelopes, filters, distortion, compressors, reverb (I3DL2), custom rack
//! modules, user effect callbacks, port delays, peak meters, limiter. Unknown
//! parameter ids are logged with their payload.

use super::atrac9;
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
const MASTERING_SETUP: u32 = 0x3000_0000;
const MASTERING_GAIN: u32 = 0x3000_0004;

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
    Mastering,
}

fn kind_of(rack_id: u32) -> Kind {
    match rack_id & 0xF000 {
        0x1000 => Kind::Sampler,
        0x3000 => Kind::Mastering,
        // Submixers, reverbs (0x2001) and custom racks (0x4xxx) mix their
        // inputs to their outputs.
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
    port_volume: [f32; 4],
    port_matrix: [i32; 4],
    matrices: HashMap<i32, Vec<f32>>,
    patches: Vec<Patch>,
    sampler: Sampler,
    /// Input bus (submixers, mastering): `grain` frames of MAX_CH channels.
    input: Vec<[f32; MAX_CH]>,
    gain: f32,
}

impl Voice {
    fn new(rack: u64, kind: Kind, index: u32) -> Voice {
        Voice {
            rack,
            kind,
            index,
            flags: 0,
            channels: if kind == Kind::Sampler { 1 } else { 2 },
            port_volume: [1.0; 4],
            port_matrix: [-1; 4],
            matrices: HashMap::new(),
            patches: Vec::new(),
            sampler: Sampler::default(),
            input: Vec::new(),
            gain: 1.0,
        }
    }
}

#[derive(Default)]
struct State {
    systems: HashMap<u64, System>,
    racks: HashMap<u64, Rack>,
    voices: HashMap<u64, Voice>,
    seen_params: HashSet<(u32, u16)>,
    noted: HashSet<&'static str>,
}

static NGS2: Mutex<Option<State>> = Mutex::new(None);

fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
    let mut g = NGS2.lock();
    f(g.get_or_insert_with(State::default))
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
            match id {
                PARAM_EVENT => match u32_at(8) {
                    EVENT_PLAY => {
                        v.flags = (v.flags | STATE_IN_USE | STATE_PLAYING) & !(STATE_STOPPED | STATE_PAUSED | STATE_EMPTY);
                    }
                    e @ (EVENT_STOP | EVENT_STOP_IMMEDIATELY | EVENT_KILL) => {
                        // (STOP releases through the envelope on hardware; without
                        // envelopes every stop is immediate.)
                        v.flags = (v.flags & !(STATE_PLAYING | STATE_PAUSED)) | STATE_STOPPED;
                        if e != EVENT_STOP || v.kind == Kind::Sampler {
                            v.flags &= !STATE_IN_USE;
                            let sm = &mut v.sampler;
                            sm.blocks.clear();
                            sm.pcm.clear();
                            sm.started_block = false;
                        }
                    }
                    EVENT_PAUSE => v.flags |= STATE_PAUSED,
                    EVENT_RESUME => v.flags &= !STATE_PAUSED,
                    e => crate::warn!("NGS2: unknown voice event {e}"),
                },
                PARAM_PORT_VOLUME => {
                    let port = u32_at(8) as usize;
                    if port < 4 {
                        v.port_volume[port] = f32_at(12);
                    }
                }
                PARAM_PORT_MATRIX => {
                    let port = u32_at(8) as usize;
                    if port < 4 {
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
                        v.matrices.insert(mid, l);
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
                        v.sampler.blocks.push_back(Block {
                            addr: data + rd(0) as u64,
                            size: rd(1),
                            repeats: rd(2),
                            skip: rd(3),
                            samples: rd(4),
                            user_data: unsafe { *(b.add(6) as *const u64) },
                        });
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
                SAMPLER_ENVELOPE => unimpl = Some("sampler envelopes"),
                SAMPLER_DISTORTION => unimpl = Some("sampler distortion"),
                SAMPLER_USER_FX => unimpl = Some("sampler user effects"),
                SAMPLER_PEAKMETER => unimpl = Some("sampler peak meters"),
                SAMPLER_FILTER => unimpl = Some("sampler filters"),
                SUBMIXER_SETUP | REVERB_SETUP | MASTERING_SETUP | 0x4002_0000 | 0x4001_0000 | 0x4003_0000 => {
                    // {u32 numChannels; ...} (reverb/custom: numInputChannels, numOutputChannels)
                    v.channels = u32_at(8).clamp(1, MAX_CH as u32);
                    v.flags |= STATE_IN_USE;
                }
                MASTERING_GAIN => {
                    // {float fbwLevel; float lfeLevel}
                    v.gain = f32_at(8);
                }
                0x2000_0001..=0x2000_0006 => unimpl = Some("submixer envelope/compressor/distortion/user effect/filter"),
                0x2001_0001..=0x2001_00FF => unimpl = Some("reverb (I3DL2)"),
                0x3000_0001..=0x3000_0003 | 0x3000_0005..=0x3000_00FF => unimpl = Some("mastering matrix/LFE/limiter/output/peak meter"),
                0x4000_0000..=0x4FFF_FFFF => unimpl = Some("custom rack modules"),
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
                    if crate::mem::is_guest(b.addr + sm.byte_pos as u64, take as u64) {
                        unsafe { std::ptr::copy_nonoverlapping((b.addr + sm.byte_pos as u64) as *const u8, buf.as_mut_ptr(), take) };
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
                    sm.decoded_bytes += take as u64;
                }
                WF_PCM_I16L | WF_PCM_F32L => {
                    let bps = if sm.wf_type == WF_PCM_I16L { 2 } else { 4 } * ch;
                    let frames_left = ((b.size - sm.byte_pos) as usize / bps).min(1024);
                    let addr = b.addr + sm.byte_pos as u64;
                    if frames_left > 0 && crate::mem::is_guest(addr, (frames_left * bps) as u64) {
                        for f in 0..frames_left {
                            let s = sm.sample_pos;
                            sm.sample_pos += 1;
                            if s < b.skip || s >= b.skip + b.samples {
                                continue;
                            }
                            let mut fr = [0f32; MAX_CH];
                            for c in 0..ch.min(MAX_CH) {
                                fr[c] = unsafe {
                                    if sm.wf_type == WF_PCM_I16L {
                                        *((addr + (f * bps + c * 2) as u64) as *const i16) as f32 / 32768.0
                                    } else {
                                        *((addr + (f * bps + c * 4) as u64) as *const f32)
                                    }
                                };
                            }
                            sm.pcm.push_back(fr);
                            sm.decoded_samples += 1;
                        }
                        produced = true;
                    }
                    sm.byte_pos += (frames_left * bps) as u32;
                    sm.decoded_bytes += (frames_left * bps) as u64;
                    if frames_left == 0 {
                        sm.byte_pos = b.size;
                    }
                }
                WF_VAG => {
                    // PS-ADPCM: per channel, 16-byte frames (shift/filter byte, flags
                    // byte, 28 4-bit samples); channels interleave frame by frame.
                    let frame_bytes = 16 * ch;
                    let frames_left = ((b.size - sm.byte_pos) as usize / frame_bytes).min(64);
                    let addr = b.addr + sm.byte_pos as u64;
                    if frames_left > 0 && crate::mem::is_guest(addr, (frames_left * frame_bytes) as u64) {
                        for f in 0..frames_left {
                            let mut out = [[0f32; MAX_CH]; 28];
                            for c in 0..ch.min(MAX_CH) {
                                let fr = unsafe { std::slice::from_raw_parts((addr + (f * frame_bytes + c * 16) as u64) as *const u8, 16) };
                                let decoded = vag_frame(fr, &mut sm.vag_hist[c]);
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
                    sm.decoded_bytes += (frames_left * frame_bytes) as u64;
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
            } else {
                sm.blocks.pop_front();
                sm.started_block = false;
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
    false
}

/// Renders `grain` frames of a sampler voice at the system rate.
fn render_sampler(v: &mut Voice, grain: usize, rate: u32, out: &mut Vec<[f32; MAX_CH]>) {
    out.clear();
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
        let mut fr = [0f32; MAX_CH];
        for c in 0..MAX_CH {
            fr[c] = sm.cur[c] + (next[c] - sm.cur[c]) * t;
        }
        out.push(fr);
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
    if ended && sm.pcm.is_empty() && sm.blocks.is_empty() {
        v.flags = (v.flags & !STATE_PLAYING) | STATE_STOPPED | STATE_EMPTY;
        v.flags &= !STATE_IN_USE;
    }
}

/// Mixes `src` (`src_ch` channels) of voice `vh` along its patches into the
/// destination voices' input buses.
fn route(s: &mut State, vh: u64, src: &[[f32; MAX_CH]], src_ch: usize) {
    let Some(v) = s.voices.get(&vh) else { return };
    let patches = v.patches.clone();
    let vol = v.port_volume;
    let pm = v.port_matrix;
    let matrices: Vec<(i32, Vec<f32>)> = patches.iter().filter_map(|p| v.matrices.get(&pm[p.port as usize & 3]).map(|m| (pm[p.port as usize & 3], m.clone()))).collect();
    for p in patches {
        // The title patches to rack handles of one-voice racks (mastering,
        // submixers): their first voice.
        let dest = if s.voices.contains_key(&p.dest) { p.dest } else { match s.racks.get(&p.dest).and_then(|r| r.voices.first().copied()) { Some(v) => v, None => continue } };
        let Some(d) = s.voices.get_mut(&dest) else { continue };
        if d.kind == Kind::Sampler {
            continue;
        }
        let dch = (d.channels as usize).clamp(1, MAX_CH);
        if d.input.len() < src.len() {
            d.input.resize(src.len(), [0.0; MAX_CH]);
        }
        let g = vol[p.port as usize & 3];
        let mid = pm[p.port as usize & 3];
        let matrix = matrices.iter().find(|(id, _)| *id == mid).map(|(_, m)| m);
        for (i, fr) in src.iter().enumerate() {
            let o = &mut d.input[i];
            match matrix {
                // Levels per destination channel, source channels inner (aLevel[dst * src_ch + src]).
                Some(m) if m.len() >= src_ch * dch => {
                    for oc in 0..dch {
                        let mut acc = 0.0;
                        for ic in 0..src_ch {
                            acc += fr[ic] * m[oc * src_ch + ic];
                        }
                        o[oc] += acc * g;
                    }
                }
                _ => {
                    // No matrix: channel to channel; mono feeds the front pair.
                    if src_ch == 1 && dch >= 2 {
                        o[0] += fr[0] * g;
                        o[1] += fr[0] * g;
                    } else {
                        for c in 0..src_ch.min(dch) {
                            o[c] += fr[c] * g;
                        }
                    }
                }
            }
        }
    }
}

/// One grain of the system: samplers, then mixing voices in patch order, then
/// the mastering voice into the render buffers.
fn render(s: &mut State, system: u64, grain: usize, rate: u32) -> Vec<[f32; MAX_CH]> {
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
    let mut buf = Vec::with_capacity(grain);
    for vh in &all {
        let play = s.voices.get(vh).is_some_and(|v| v.kind == Kind::Sampler && v.flags & STATE_PLAYING != 0 && v.flags & STATE_PAUSED == 0);
        if !play {
            continue;
        }
        let ch = {
            let v = s.voices.get_mut(vh).unwrap();
            render_sampler(v, grain, rate, &mut buf);
            v.sampler.channels as usize
        };
        route(s, *vh, &buf, ch);
    }
    // Mixing voices: deepest upstream first (a voice runs after everything that
    // patches into it; cycles are cut).
    let mixers: Vec<u64> = all.iter().copied().filter(|vh| s.voices.get(vh).is_some_and(|v| v.kind == Kind::Submixer)).collect();
    let mut depth: HashMap<u64, u32> = HashMap::new();
    fn depth_of(s: &State, vh: u64, memo: &mut HashMap<u64, u32>, guard: u32) -> u32 {
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
        let (input, ch) = {
            let v = s.voices.get_mut(&vh).unwrap();
            (std::mem::take(&mut v.input), v.channels as usize)
        };
        route(s, vh, &input, ch.clamp(1, MAX_CH));
        if let Some(v) = s.voices.get_mut(&vh) {
            v.input = input;
        }
    }
    // Mastering voices: their inputs, scaled by their gain, summed.
    let mut out = vec![[0f32; MAX_CH]; grain];
    for vh in &all {
        let Some(v) = s.voices.get(vh) else { continue };
        if v.kind != Kind::Mastering {
            continue;
        }
        for (i, fr) in v.input.iter().enumerate().take(grain) {
            for c in 0..MAX_CH {
                out[i][c] += fr[c] * v.gain;
            }
        }
    }
    out
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
        let Some((flags, samples, bytes, user, data)) = with(|s| s.voices.get(&voice).map(|v| {
            let sm = &v.sampler;
            let data = sm.blocks.front().map(|b| b.addr).unwrap_or(sm.data);
            (v.flags, sm.decoded_samples, sm.decoded_bytes, sm.user_data, data)
        })) else { return ERR_INVALID_HANDLE };
        std::ptr::write_bytes(out, 0, size as usize);
        if size >= 4 { *(out as *mut u32) = flags; }
        if size >= 8 { *(out.add(4) as *mut f32) = if flags & STATE_PLAYING != 0 { 1.0 } else { 0.0 }; }
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
        let mix = render(s, system, grain, rate);
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
