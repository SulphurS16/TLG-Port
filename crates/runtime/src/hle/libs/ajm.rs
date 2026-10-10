//! libSceAjm batch execution. Batch buffers use an internal job format (both
//! the builders and the executor are HLE). ATRAC9 is currently decoded as
//! frame-accurate silence: stream bookkeeping (input consumed, samples written,
//! sidebands) follows the codec configuration, but PCM output is zero. This is
//! logged once per instance; a real ATRAC9 decoder replaces it later.

use crate::hle::Registry;
use crate::{fixed, reg, sysv};
use parking_lot::Mutex;
use std::collections::HashMap;

const JOB_MAGIC: u32 = 0x4A4D_4A41; // "AJMJ"
const KIND_CONTROL: u32 = 1;
const KIND_RUN: u32 = 2;

const ERR_INVALID: i32 = 0x8093_0003u32 as i32;
const AT9_SAMPLE_RATES: [u32; 16] = [11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 44100, 48000, 64000, 88200, 96000, 128000, 176400, 192000];
const AT9_FRAME_SAMPLES_POW: [u32; 16] = [6, 6, 7, 7, 7, 8, 8, 8, 6, 6, 7, 7, 7, 8, 8, 8];
const AT9_CHANNELS: [u32; 6] = [1, 2, 2, 6, 8, 4];

#[repr(C)]
#[derive(Clone, Copy)]
struct Job {
    magic: u32,
    kind: u32,
    instance: u32,
    pad: u32,
    flags: u64,
    sb_in: u64,
    sb_in_size: u64,
    data_in: u64,
    data_in_size: u64,
    data_out: u64,
    data_out_size: u64,
    sb_out: u64,
    sb_out_size: u64,
}

#[derive(Default, Clone)]
struct Instance {
    codec: u32,
    format: u32, // 0 s16, 1 s32, 2 float
    channels: u32,
    sample_rate: u32,
    frame_bytes: u32,
    frame_samples: u32,
    total_samples: u64,
    configured: bool,
    warned: bool,
}

static INSTANCES: Mutex<Option<HashMap<u32, Instance>>> = Mutex::new(None);
static NEXT_BATCH: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

fn with_instances<R>(f: impl FnOnce(&mut HashMap<u32, Instance>) -> R) -> R {
    let mut g = INSTANCES.lock();
    f(g.get_or_insert_with(HashMap::new))
}

// Job records: tag = 0x4A41_0000 | kind << 8 | slot size. The title's inline
// batch code advances by fixed slot sizes (observed: control 0x40, run 0x50),
// so each record declares exactly that size.
const TAG_HI: u32 = 0x4A41;
const CONTROL_SLOT: usize = 0x40;
const RUN_SLOT: usize = 0x50;

#[repr(C)]
struct CtlRec {
    tag: u32,
    instance: u32,
    flags: u64,
    sb_in: u64,
    sb_in_size: u32,
    sb_out_size: u32,
    sb_out: u64,
}

#[repr(C)]
struct RunRec {
    tag: u32,
    instance: u32,
    flags: u64,
    data_in: u64,
    in_size: u32,
    out_size: u32,
    data_out: u64,
    sb_out: u64,
    sb_out_size: u32,
    pad: u32,
}

fn tag(kind: u32, size: usize) -> u32 {
    (TAG_HI << 16) | (kind << 8) | size as u32
}

unsafe fn parse_job(p: u64, end: u64) -> Option<(Job, u64)> {
    if p + 4 > end {
        return None;
    }
    let t = unsafe { std::ptr::read_unaligned(p as *const u32) };
    if t >> 16 != TAG_HI {
        return None;
    }
    let size = (t & 0xFF) as u64;
    if size == 0 || p + size > end {
        return None;
    }
    let job = match (t >> 8) & 0xFF {
        KIND_CONTROL => {
            let r = unsafe { std::ptr::read_unaligned(p as *const CtlRec) };
            Job { magic: JOB_MAGIC, kind: KIND_CONTROL, instance: r.instance, pad: 0, flags: r.flags, sb_in: r.sb_in, sb_in_size: r.sb_in_size as u64, data_in: 0, data_in_size: 0, data_out: 0, data_out_size: 0, sb_out: r.sb_out, sb_out_size: r.sb_out_size as u64 }
        }
        KIND_RUN => {
            let r = unsafe { std::ptr::read_unaligned(p as *const RunRec) };
            Job { magic: JOB_MAGIC, kind: KIND_RUN, instance: r.instance, pad: 0, flags: r.flags, sb_in: 0, sb_in_size: 0, data_in: r.data_in, data_in_size: r.in_size as u64, data_out: r.data_out, data_out_size: r.out_size as u64, sb_out: r.sb_out, sb_out_size: r.sb_out_size as u64 }
        }
        _ => return None,
    };
    Some((job, p + size))
}
fn configure_at9(inst: &mut Instance, cfg: &[u8]) -> bool {
    if cfg.len() < 4 || cfg[0] != 0xFE {
        return false;
    }
    let sr_index = (cfg[1] >> 4) as usize;
    let ch_index = ((cfg[1] >> 1) & 7) as usize;
    let frame_bytes = (((cfg[2] as u32) << 3) | ((cfg[3] as u32) >> 5)) + 1;
    if ch_index >= AT9_CHANNELS.len() {
        return false;
    }
    inst.sample_rate = AT9_SAMPLE_RATES[sr_index];
    inst.frame_samples = 1 << AT9_FRAME_SAMPLES_POW[sr_index];
    inst.channels = AT9_CHANNELS[ch_index];
    inst.frame_bytes = frame_bytes;
    inst.total_samples = 0;
    inst.configured = true;
    true
}

/// Writes sideband outputs: result, then stream/format/gapless/multi-frame as flagged.
unsafe fn write_sideband(job: &Job, inst: &Instance, result: i32, consumed: u32, written: u32, frames: u32) {
    if job.sb_out == 0 || job.sb_out_size < 8 {
        return;
    }
    let mut p = job.sb_out;
    let end = job.sb_out + job.sb_out_size;
    unsafe {
        *(p as *mut i32) = result;
        *((p + 4) as *mut i32) = 0;
        p += 8;
        let sb = job.flags >> 45;
        if sb & 0b100 != 0 && p + 16 <= end {
            *(p as *mut i32) = consumed as i32;
            *((p + 4) as *mut i32) = written as i32;
            *((p + 8) as *mut u64) = inst.total_samples;
            p += 16;
        }
        if sb & 0b010 != 0 && p + 24 <= end {
            let v = [inst.channels, 0, inst.sample_rate, inst.format, 0, 0];
            for (i, x) in v.iter().enumerate() {
                *((p + i as u64 * 4) as *mut u32) = *x;
            }
            p += 24;
        }
        if sb & 0b001 != 0 && p + 8 <= end {
            *(p as *mut u32) = 0;
            *((p + 4) as *mut u32) = 0;
            p += 8;
        }
        let run_flags = (job.flags >> 11) & 3;
        if run_flags & 0b10 != 0 && p + 8 <= end {
            *(p as *mut u32) = frames;
            *((p + 4) as *mut u32) = 0;
        }
    }
}

unsafe fn execute(job: &Job) {
    with_instances(|insts| {
        let inst = insts.entry(job.instance).or_default();
        match job.kind {
            KIND_CONTROL => {
                let control = (job.flags >> 13) & 7;
                let mut ok = true;
                if control & 0b010 != 0 && job.sb_in != 0 && job.sb_in_size >= 4 {
                    let cfg = unsafe { std::slice::from_raw_parts(job.sb_in as *const u8, 4) };
                    ok = inst.codec != 1 || configure_at9(inst, cfg);
                }
                if control & 0b001 != 0 {
                    inst.total_samples = 0;
                }
                unsafe { write_sideband(job, inst, if ok { 0 } else { ERR_INVALID }, 0, 0, 0) };
            }
            _ => {
                if !inst.warned {
                    inst.warned = true;
                    crate::warn!(
                        "Ajm instance {:#x} (codec {}, {} ch @ {} Hz): decoding as silence (ATRAC9 decoder not implemented yet)",
                        job.instance,
                        inst.codec,
                        inst.channels,
                        inst.sample_rate
                    );
                }
                if !inst.configured || inst.frame_bytes == 0 {
                    unsafe { write_sideband(job, inst, ERR_INVALID, 0, 0, 0) };
                    return;
                }
                let bps = if inst.format == 0 { 2 } else { 4 };
                let frame_out = inst.frame_samples * inst.channels * bps;
                let multi = (job.flags >> 11) & 0b10 != 0;
                let mut consumed = 0u64;
                let mut written = 0u64;
                let mut frames = 0u32;
                loop {
                    if consumed + inst.frame_bytes as u64 > job.data_in_size || written + frame_out as u64 > job.data_out_size {
                        break;
                    }
                    if job.data_out != 0 {
                        unsafe { std::ptr::write_bytes((job.data_out + written) as *mut u8, 0, frame_out as usize) };
                    }
                    consumed += inst.frame_bytes as u64;
                    written += frame_out as u64;
                    frames += 1;
                    inst.total_samples += inst.frame_samples as u64;
                    if !multi {
                        break;
                    }
                }
                unsafe { write_sideband(job, inst, 0, consumed as u32, written as u32, frames) };
            }
        }
    });
}

sysv! {
    // Builders write one job at `pos` and return the position after it.
    fn sceAjmBatchJobControlBufferRa(pos: u64, instance: u32, flags: u64, sb_in: u64, sb_in_size: u64, sb_out: u64, sb_out_size: u64, ra: u64) -> u64 {
        if pos == 0 { return 0; }
        let size = CONTROL_SLOT;
        let r = CtlRec { tag: tag(KIND_CONTROL, size), instance, flags, sb_in, sb_in_size: sb_in_size as u32, sb_out_size: sb_out_size as u32, sb_out };
        std::ptr::write_unaligned(pos as *mut CtlRec, r);
        pos + size as u64
    }
    fn sceAjmBatchJobRunBufferRa(pos: u64, instance: u32, flags: u64, data_in: u64, in_size: u64, data_out: u64, out_size: u64, sb_out: u64, sb_out_size: u64, ra: u64) -> u64 {
        if pos == 0 { return 0; }
        let size = RUN_SLOT;
        let r = RunRec { tag: tag(KIND_RUN, size), instance, flags, data_in, in_size: in_size as u32, out_size: out_size as u32, data_out, sb_out, sb_out_size: sb_out_size as u32, pad: 0 };
        std::ptr::write_unaligned(pos as *mut RunRec, r);
        pos + size as u64
    }
    fn sceAjmBatchStartBuffer(ctx: u32, buf: u64, size: u64, prio: i32, err: *mut u8, id: *mut u32) -> i32 {
        let end = buf + size;
        let mut p = buf;
        while p < end {
            match parse_job(p, end) {
                Some((job, next)) => {
                    execute(&job);
                    p = next;
                }
                None => {
                    let dump: Vec<String> = (0..(size.min(0xA0) / 8)).map(|i| format!("{:016x}", std::ptr::read_unaligned((buf + i * 8) as *const u64))).collect();
                    crate::error!("Ajm batch at {buf:#x}+{:#x} (size {size:#x}): unknown job record; batch qwords: {}", p - buf, dump.join(" "));
                    return ERR_INVALID;
                }
            }
        }
        if !id.is_null() { *id = NEXT_BATCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
        0
    }
    fn sceAjmInstanceCreate(ctx: u32, codec: u32, flags: u64, inst: *mut u32) -> i32 {
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);
        let id = (codec << 14) | NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if !inst.is_null() { *inst = id; }
        let format = ((flags >> 7) & 7) as u32;
        let channels = ((flags >> 3) & 0xF) as u32;
        with_instances(|m| { m.insert(id, Instance { codec, format, channels, ..Default::default() }); });
        crate::debug!("sceAjmInstanceCreate codec {codec} flags {flags:#x} -> {id:#x}");
        0
    }
    fn sceAjmInstanceDestroy(ctx: u32, inst: u32) -> i32 {
        with_instances(|m| { m.remove(&inst); });
        0
    }
}

fixed! {
    sceAjmBatchWait => 0,
    sceAjmBatchCancel => 0,
    sceAjmBatchErrorDump => 0,
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceAjm", [
        sceAjmBatchJobControlBufferRa, sceAjmBatchJobRunBufferRa, sceAjmBatchStartBuffer, sceAjmInstanceCreate,
        sceAjmInstanceDestroy, sceAjmBatchWait, sceAjmBatchCancel, sceAjmBatchErrorDump,
    ]);
}
