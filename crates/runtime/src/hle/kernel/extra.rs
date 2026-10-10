//! libkernel / libc-internal functions Shadow of the Colossus imports beyond the
//! imported Bloodborne set: heap trace info, mlock, time-zone conversion, UUIDs,
//! POSIX scheduling queries.

use super::errno::*;
use super::{posix_ret, sce};
use crate::hle::Registry;
use crate::{reg, sysv};
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::Ordering;

/// PS4 FIFO priority range (lower value = higher priority).
pub const PRIO_FIFO_HIGHEST: i32 = 256;
pub const PRIO_FIFO_LOWEST: i32 = 767;
const SCHED_FIFO: i32 = 1;
const SCHED_RR: i32 = 3;

/// Random 64-bit values (std's per-process random hash keys).
pub fn rand_u64() -> u64 {
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u64(crate::hle::kernel::time::process_time_us());
    h.finish()
}

/// libc's allocator asks the kernel where its heap-tracing switches live; with a
/// zero atomic-id mask and an empty mstate table tracing stays disabled (as on a
/// retail console without a profiler attached).
#[repr(C)]
struct HeapTraceInfo {
    size: u64,
    flag: u32,
    get_segment_info: u32,
    mspace_atomic_id_mask: u64,
    mstate_table: u64,
}

static HEAP_TRACE: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();

/// Host time-zone offsets for a UTC time: (seconds west of UTC excluding DST, DST seconds).
fn tz_offsets(utc: i64) -> (i64, i64) {
    use windows_sys::Win32::System::Time::{GetTimeZoneInformation, TIME_ZONE_INFORMATION};
    let mut tz: TIME_ZONE_INFORMATION = unsafe { std::mem::zeroed() };
    let r = unsafe { GetTimeZoneInformation(&mut tz) };
    let west = tz.Bias as i64 * 60;
    // 2 = TIME_ZONE_ID_DAYLIGHT for the current date; the title converts current times.
    let dst = if r == 2 { -(tz.DaylightBias as i64) * 60 } else { 0 };
    let _ = utc;
    (west, dst)
}

sysv! {
    fn sceLibcHeapGetTraceInfo(info: *mut HeapTraceInfo) {
        if info.is_null() { return; }
        let (mask, table) = *HEAP_TRACE.get_or_init(|| (crate::mem::arena_alloc(8, 8), crate::mem::arena_alloc(64 * 8, 8)));
        (*info).mspace_atomic_id_mask = mask;
        (*info).mstate_table = table;
        (*info).get_segment_info = 0;
    }

    // Guest memory is committed host memory; page locking has no observable effect.
    fn sceKernelMlock(addr: u64, len: u64) -> i32 { if addr == 0 || len == 0 { sce(EINVAL) } else { 0 } }
    fn sceKernelMunlock(addr: u64, len: u64) -> i32 { if addr == 0 || len == 0 { sce(EINVAL) } else { 0 } }
    fn mlock(addr: u64, len: u64) -> i32 { posix_ret(sceKernelMlock(addr, len)) }
    fn munlock(addr: u64, len: u64) -> i32 { posix_ret(sceKernelMunlock(addr, len)) }

    fn _nanosleep(req: *const i64, rem: *mut i64) -> i32 {
        if req.is_null() { crate::hle::kernel::set_errno(EINVAL); return -1; }
        let ns = (*req).max(0) as u64 * 1_000_000_000 + (*req.add(1)).max(0) as u64;
        crate::hle::kernel::time::sleep_us(ns.div_ceil(1000));
        if !rem.is_null() { *rem = 0; *rem.add(1) = 0; }
        0
    }

    /// (time_t utc, time_t* local, {time_t t; u32 west_sec; u32 dst_sec}* st, u64* dst_sec)
    fn sceKernelConvertUtcToLocaltime(utc: i64, local: *mut i64, st: *mut u8, dst_out: *mut u64) -> i32 {
        if local.is_null() { return sce(EINVAL); }
        let (west, dst) = tz_offsets(utc);
        *local = utc - west + dst;
        if !st.is_null() {
            *(st as *mut i64) = utc;
            *(st.add(8) as *mut u32) = west as u32;
            *(st.add(12) as *mut u32) = dst as u32;
        }
        if !dst_out.is_null() { *dst_out = dst as u64; }
        0
    }
    /// (time_t local, i64 unused, time_t* utc, {s32 minuteswest; s32 dsttime}* tz, s32* dst_sec)
    fn sceKernelConvertLocaltimeToUtc(local: i64, unused: i64, utc: *mut i64, tz: *mut i32, dst_out: *mut i32) -> i32 {
        if tz.is_null() { return sce(EINVAL); }
        let (west, dst) = tz_offsets(local);
        *tz = (west / 60) as i32;
        *tz.add(1) = (dst / 60) as i32;
        if !utc.is_null() { *utc = local + west - dst; }
        if !dst_out.is_null() { *dst_out = dst as i32; }
        0
    }

    /// Version-4 (random) UUID in the RFC 4122 byte layout.
    fn sceKernelUuidCreate(uuid: *mut u8) -> i32 {
        if uuid.is_null() { return sce(EINVAL); }
        let mut b = [0u8; 16];
        b[..8].copy_from_slice(&rand_u64().to_le_bytes());
        b[8..].copy_from_slice(&rand_u64().to_le_bytes());
        b[6] = (b[6] & 0x0F) | 0x40;
        b[8] = (b[8] & 0x3F) | 0x80;
        std::ptr::copy_nonoverlapping(b.as_ptr(), uuid, 16);
        0
    }

    fn sched_get_priority_max(policy: i32) -> i32 {
        if policy == SCHED_FIFO || policy == SCHED_RR { PRIO_FIFO_HIGHEST } else { crate::hle::kernel::set_errno(EINVAL); -1 }
    }
    fn sched_get_priority_min(policy: i32) -> i32 {
        if policy == SCHED_FIFO || policy == SCHED_RR { PRIO_FIFO_LOWEST } else { crate::hle::kernel::set_errno(EINVAL); -1 }
    }
    fn pthread_getschedparam(t: u64, policy: *mut i32, param: *mut i32) -> i32 {
        if policy.is_null() || param.is_null() { return EINVAL; }
        let th = if t == 0 { crate::thread::current() } else { crate::thread::from_handle(t) };
        let Some(th) = th else { return ESRCH };
        *policy = SCHED_FIFO;
        *param = th.priority.load(Ordering::Relaxed);
        0
    }
    fn pthread_setschedparam(t: u64, policy: i32, param: *const i32) -> i32 {
        if param.is_null() { return EINVAL; }
        let th = if t == 0 { crate::thread::current() } else { crate::thread::from_handle(t) };
        let Some(th) = th else { return ESRCH };
        th.priority.store(*param, Ordering::Relaxed);
        0
    }
    fn pthread_cancel(t: u64) -> i32 {
        // Asynchronous cancellation of another guest thread is not emulated: report it loudly.
        crate::error!("pthread_cancel({t:#x}) called: thread cancellation is not implemented");
        crate::fatal_exit(3, "pthread_cancel is not implemented");
    }
    fn sigreturn(ctx: u64) -> i32 {
        crate::error!("sigreturn({ctx:#x}) called: guest signal delivery is not implemented");
        crate::fatal_exit(3, "sigreturn without signal delivery");
    }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceLibcInternalExt", [sceLibcHeapGetTraceInfo]);
    reg!(r, "libkernel", [
        sceKernelMlock, sceKernelMunlock, _nanosleep, sceKernelConvertUtcToLocaltime,
        sceKernelConvertLocaltimeToUtc, sceKernelUuidCreate, sigreturn,
    ]);
    reg!(r, "libScePosix", [
        mlock, munlock, sched_get_priority_max, sched_get_priority_min, pthread_getschedparam,
        pthread_setschedparam, pthread_cancel,
    ]);
}
