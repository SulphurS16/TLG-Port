//! libSceNgs2: the system audio graph (systems of racks of voices: sampler voices
//! play waveform data, submixers and mastering voices mix it to output buffers).
//!
//! Stage 1 (this file): systems, racks and voices are created with the title's
//! memory (allocator callbacks / supplied buffers), voice parameter chains are
//! parsed and their effect on voice state is tracked (events, waveform blocks,
//! playback position per render), and `sceNgs2SystemRender` produces SILENCE: the
//! mixing itself is not implemented yet. Each distinct parameter id is logged with
//! its size and payload so the remaining semantics can be implemented from the
//! title's actual usage.

use crate::hle::Registry;
use crate::{reg, sysv};
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};

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
// Sampler voice parameter ids.
const SAMPLER_SETUP: u32 = 0x1000_0001;
const SAMPLER_ADD_BLOCKS: u32 = 0x1000_0002;
const SAMPLER_REPLACE_ADDRESS: u32 = 0x1000_0003;
const SAMPLER_FRAME_OFFSET: u32 = 0x1000_0004;
const SAMPLER_EXIT_LOOP: u32 = 0x1000_0005;
const SAMPLER_PITCH: u32 = 0x1000_0006;

// Voice events.
const EVENT_PLAY: u32 = 0;
const EVENT_STOP: u32 = 1;
const EVENT_STOP_IMMEDIATELY: u32 = 2;
const EVENT_KILL: u32 = 3;
const EVENT_PAUSE: u32 = 4;
const EVENT_RESUME: u32 = 5;

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

struct Rack {
    system: u64,
    rack_id: u32,
    buf: BufferInfo,
    voices: Vec<u64>,
}

#[derive(Default)]
struct Block {
    samples: u64,
    repeats: u32,
}

#[derive(Default)]
struct Voice {
    rack: u64,
    index: u32,
    flags: u32,
    sample_rate: u32,
    channels: u32,
    waveform_type: u32,
    blocks: Vec<Block>,
    block: usize,
    pos: f64,
    pitch: f32,
    data: u64,
}

#[derive(Default)]
struct State {
    systems: HashMap<u64, System>,
    racks: HashMap<u64, Rack>,
    voices: HashMap<u64, Voice>,
    seen_params: HashSet<(u32, u16)>,
}

static NGS2: Mutex<Option<State>> = Mutex::new(None);

fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
    let mut g = NGS2.lock();
    f(g.get_or_insert_with(State::default))
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
    with(|s| {
        let voices: Vec<u64> = (0..max_voices).map(|i| handle + 0x100 + i as u64 * 0x10).collect();
        for (i, &v) in voices.iter().enumerate() {
            s.voices.insert(v, Voice { rack: handle, index: i as u32, flags: 0, pitch: 1.0, ..Default::default() });
        }
        s.racks.insert(handle, Rack { system, rack_id, buf, voices });
    });
    crate::info!("NGS2 rack {rack_id:#x} on system {system:#x}: {max_voices} voices -> handle {handle:#x}");
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
        let Some(v) = s.voices.get_mut(&vh) else { return ERR_INVALID_HANDLE };
        let u32_at = |off: usize| unsafe { *(p.add(off) as *const u32) };
        match id {
            PARAM_EVENT => match u32_at(8) {
                EVENT_PLAY => v.flags = (v.flags | STATE_IN_USE | STATE_PLAYING) & !(STATE_STOPPED | STATE_PAUSED),
                EVENT_STOP | EVENT_STOP_IMMEDIATELY | EVENT_KILL => {
                    v.flags = (v.flags & !(STATE_PLAYING | STATE_PAUSED)) | STATE_STOPPED;
                    if u32_at(8) != EVENT_STOP {
                        v.flags &= !STATE_IN_USE;
                        v.blocks.clear();
                    }
                }
                EVENT_PAUSE => v.flags |= STATE_PAUSED,
                EVENT_RESUME => v.flags &= !STATE_PAUSED,
                e => crate::warn!("NGS2: unknown voice event {e}"),
            },
            SAMPLER_SETUP => {
                // format {type, channels, sampleRate, configData, frameOffset, frameMargin}, flags
                v.waveform_type = u32_at(8);
                v.channels = u32_at(12);
                v.sample_rate = u32_at(16);
                v.blocks.clear();
                v.block = 0;
                v.pos = 0.0;
                v.flags = (v.flags | STATE_IN_USE) & !STATE_EMPTY;
            }
            SAMPLER_ADD_BLOCKS => {
                // {data*, flags, numBlocks, aBlock*} with blocks {offset, size, repeats, skip, samples, -, userData}
                v.data = unsafe { *(p.add(8) as *const u64) };
                let count = u32_at(20).min(64);
                let blocks = unsafe { *(p.add(24) as *const u64) } as *const u32;
                for i in 0..count as usize {
                    if blocks.is_null() {
                        break;
                    }
                    let b = unsafe { blocks.add(i * 8) };
                    v.blocks.push(Block { samples: unsafe { *b.add(4) } as u64, repeats: unsafe { *b.add(2) } });
                }
            }
            SAMPLER_PITCH => v.pitch = f32::from_bits(u32_at(8)),
            SAMPLER_EXIT_LOOP => {
                for b in &mut v.blocks {
                    b.repeats = 0;
                }
            }
            PARAM_MATRIX_LEVELS | PARAM_PORT_VOLUME | PARAM_PORT_MATRIX | PARAM_PORT_DELAY | PARAM_PATCH
            | PARAM_CALLBACK | SAMPLER_REPLACE_ADDRESS | SAMPLER_FRAME_OFFSET => {}
            _ => {}
        }
        if next == 0 {
            break;
        }
        p = unsafe { p.offset(next as isize) };
    }
    0
}

/// Advances sampler voices by one grain: playback ends after the last block.
fn advance(s: &mut State, system: u64, grain: u32, rate: u32) {
    let racks: Vec<u64> = s.racks.iter().filter(|(_, r)| r.system == system && r.rack_id & 0xF000 == 0x1000).map(|(&h, _)| h).collect();
    for rh in racks {
        let voices = s.racks[&rh].voices.clone();
        for vh in voices {
            let Some(v) = s.voices.get_mut(&vh) else { continue };
            if v.flags & STATE_PLAYING == 0 || v.flags & STATE_PAUSED != 0 {
                continue;
            }
            let step = grain as f64 * v.pitch.max(0.0) as f64 * v.sample_rate.max(1) as f64 / rate.max(1) as f64;
            v.pos += step;
            while let Some(b) = v.blocks.get_mut(v.block) {
                if v.pos < b.samples as f64 || b.samples == 0 {
                    break;
                }
                v.pos -= b.samples as f64;
                if b.repeats > 0 {
                    b.repeats -= 1;
                } else {
                    v.block += 1;
                }
            }
            if v.block >= v.blocks.len() {
                v.flags = (v.flags & !STATE_PLAYING) | STATE_STOPPED | STATE_EMPTY;
                v.flags &= !STATE_IN_USE;
            }
        }
    }
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
        crate::info!("NGS2 system: grain {grain} rate {rate} -> handle {:#x} (mixing not implemented: renders silence)", buf.buffer);
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
    /// SceNgs2VoiceState {u32 stateFlags; ...} (sampler: envelope, peak, decoded counts, user data, data*).
    fn sceNgs2VoiceGetState(voice: u64, out: *mut u8, size: u64) -> i32 {
        if out.is_null() { return ERR_INVALID_OUT_ADDRESS; }
        let Some((flags, data)) = with(|s| s.voices.get(&voice).map(|v| (v.flags, v.data))) else { return ERR_INVALID_HANDLE };
        std::ptr::write_bytes(out, 0, size as usize);
        if size >= 4 { *(out as *mut u32) = flags; }
        if size >= 48 { *(out.add(40) as *mut u64) = data; }
        0
    }
    /// Renders one grain into each buffer ({buffer*, size, waveformType, numChannels}).
    fn sceNgs2SystemRender(system: u64, infos: *const u8, count: u32) -> i32 {
        let Some((grain, rate)) = with(|s| s.systems.get_mut(&system).map(|y| { y.renders += 1; (y.grain, y.sample_rate) })) else { return ERR_INVALID_HANDLE };
        for i in 0..count as usize {
            let e = infos.add(i * 24);
            let buf = *(e as *const u64) as *mut u8;
            let size = *(e.add(8) as *const u64) as usize;
            if !buf.is_null() { std::ptr::write_bytes(buf, 0, size); }
        }
        with(|s| advance(s, system, grain, rate));
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| crate::warn!("NGS2 render: output is SILENT (voice mixing not implemented yet); {count} buffers per grain of {grain}"));
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
