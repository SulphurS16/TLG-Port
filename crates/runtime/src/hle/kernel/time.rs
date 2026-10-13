//! Time sources: process time, TSC, clocks and sleeping.

use super::errno::*;
use super::sce;
use crate::hle::Registry;
use crate::{reg, sysv};
use std::sync::OnceLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

struct Clock {
    start: Instant,
    tsc_hz: u64,
}

static CLOCK: OnceLock<Clock> = OnceLock::new();

/// Measures the TSC frequency against the monotonic clock (guest code may read
/// the TSC directly with rdtsc, so the reported frequency must be the host's).
pub fn init() {
    // 1 ms system timer resolution for this process: timed condition/event
    // waits (event queues, semaphores, the title's timed sleeps) otherwise
    // round up to the 15.6 ms default tick.
    let r = unsafe { windows_sys::Win32::Media::timeBeginPeriod(1) };
    crate::info!("timeBeginPeriod(1) -> {r}");
    // Windows 11 may ignore that request and run the process in its
    // efficiency mode (EcoQoS) while no window of it is visible (headless
    // runs, a minimized game): this process opts out of both
    // (TLG_POWER_THROTTLING=1 keeps the system's choice).
    if std::env::var("TLG_POWER_THROTTLING").map_or(true, |v| v != "1") {
        use windows_sys::Win32::System::Threading::{GetCurrentProcess, ProcessPowerThrottling, SetProcessInformation, PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_EXECUTION_SPEED, PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION, PROCESS_POWER_THROTTLING_STATE};
        let st = PROCESS_POWER_THROTTLING_STATE { Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION, ControlMask: PROCESS_POWER_THROTTLING_EXECUTION_SPEED | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION, StateMask: 0 };
        let ok = unsafe { SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &st as *const _ as *const core::ffi::c_void, std::mem::size_of::<PROCESS_POWER_THROTTLING_STATE>() as u32) };
        crate::info!("power throttling opt-out (execution speed, timer resolution) -> {ok}");
    }
    CLOCK.get_or_init(|| {
        let t0 = Instant::now();
        let c0 = unsafe { core::arch::x86_64::_rdtsc() };
        std::thread::sleep(Duration::from_millis(100));
        let c1 = unsafe { core::arch::x86_64::_rdtsc() };
        let dt = t0.elapsed().as_secs_f64();
        let hz = ((c1 - c0) as f64 / dt) as u64;
        crate::info!("TSC frequency calibrated at {:.3} MHz", hz as f64 / 1e6);
        Clock { start: Instant::now(), tsc_hz: hz }
    });
}

fn clock() -> &'static Clock {
    CLOCK.get().expect("time::init not called")
}

pub fn process_time_us() -> u64 {
    clock().start.elapsed().as_micros() as u64
}

pub fn tsc_hz() -> u64 {
    clock().tsc_hz
}

fn realtime() -> Duration {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default()
}

const CLOCK_REALTIME: i32 = 0;
const CLOCK_VIRTUAL: i32 = 1;
const CLOCK_PROF: i32 = 2;
const CLOCK_MONOTONIC: i32 = 4;
const CLOCK_UPTIME: i32 = 5;
const CLOCK_UPTIME_PRECISE: i32 = 7;
const CLOCK_UPTIME_FAST: i32 = 8;
const CLOCK_REALTIME_PRECISE: i32 = 9;
const CLOCK_REALTIME_FAST: i32 = 10;
const CLOCK_MONOTONIC_PRECISE: i32 = 11;
const CLOCK_MONOTONIC_FAST: i32 = 12;
const CLOCK_SECOND: i32 = 13;
const CLOCK_THREAD_CPUTIME_ID: i32 = 14;
const CLOCK_PROCESS_CPUTIME_ID: i32 = 15;

fn clock_value(id: i32) -> Option<Duration> {
    Some(match id {
        CLOCK_REALTIME | CLOCK_REALTIME_PRECISE | CLOCK_REALTIME_FAST => realtime(),
        CLOCK_SECOND => Duration::from_secs(realtime().as_secs()),
        CLOCK_MONOTONIC | CLOCK_MONOTONIC_PRECISE | CLOCK_MONOTONIC_FAST | CLOCK_UPTIME | CLOCK_UPTIME_PRECISE
        | CLOCK_UPTIME_FAST => clock().start.elapsed() + Duration::from_secs(1),
        CLOCK_VIRTUAL | CLOCK_PROF | CLOCK_THREAD_CPUTIME_ID | CLOCK_PROCESS_CPUTIME_ID => clock().start.elapsed(),
        _ => return None,
    })
}

unsafe fn write_ts(ts: *mut i64, d: Duration) {
    unsafe {
        *ts = d.as_secs() as i64;
        *ts.add(1) = d.subsec_nanos() as i64;
    }
}

pub fn sleep_us(us: u64) {
    if us == 0 {
        std::thread::yield_now();
    } else {
        std::thread::sleep(Duration::from_micros(us));
    }
}

sysv! {
    fn sceKernelGetProcessTime() -> u64 { process_time_us() }
    fn sceKernelGetProcessTimeCounter() -> u64 { unsafe { core::arch::x86_64::_rdtsc() } }
    fn sceKernelGetProcessTimeCounterFrequency() -> u64 { tsc_hz() }
    fn sceKernelReadTsc() -> u64 { unsafe { core::arch::x86_64::_rdtsc() } }
    fn sceKernelGetTscFrequency() -> u64 { tsc_hz() }
    fn sceKernelUsleep(us: u32) -> i32 { sleep_us(us as u64); 0 }
    fn sceKernelSleep(s: u32) -> i32 { std::thread::sleep(Duration::from_secs(s as u64)); 0 }
    fn sceKernelNanosleep(req: *const i64, rem: *mut i64) -> i32 {
        if req.is_null() { return sce(EINVAL); }
        std::thread::sleep(Duration::new((*req).max(0) as u64, (*req.add(1)).clamp(0, 999_999_999) as u32));
        if !rem.is_null() { *rem = 0; *rem.add(1) = 0; }
        0
    }
    fn sceKernelClockGettime(id: i32, ts: *mut i64) -> i32 {
        if ts.is_null() { return sce(EFAULT); }
        match clock_value(id) { Some(d) => { write_ts(ts, d); 0 } None => sce(EINVAL) }
    }
    fn sceKernelClockGetres(id: i32, ts: *mut i64) -> i32 {
        if ts.is_null() { return sce(EFAULT); }
        write_ts(ts, Duration::from_nanos(1));
        0
    }
    fn sceKernelGettimeofday(tv: *mut i64) -> i32 {
        if tv.is_null() { return sce(EFAULT); }
        let d = realtime();
        *tv = d.as_secs() as i64;
        *tv.add(1) = d.subsec_micros() as i64;
        0
    }
    fn sceKernelGettimezone(tz: *mut i32) -> i32 {
        if !tz.is_null() { *tz = 0; *tz.add(1) = 0; }
        0
    }
    fn clock_gettime(id: i32, ts: *mut i64) -> i32 {
        match clock_value(id) { Some(d) if !ts.is_null() => { write_ts(ts, d); 0 } _ => { crate::hle::kernel::set_errno(EINVAL); -1 } }
    }
    fn clock_getres(id: i32, ts: *mut i64) -> i32 {
        if !ts.is_null() { write_ts(ts, Duration::from_nanos(1)); }
        0
    }
    fn gettimeofday(tv: *mut i64, tz: *mut i32) -> i32 {
        if !tv.is_null() {
            let d = realtime();
            *tv = d.as_secs() as i64;
            *tv.add(1) = d.subsec_micros() as i64;
        }
        if !tz.is_null() { *tz = 0; *tz.add(1) = 0; }
        0
    }
    fn nanosleep(req: *const i64, rem: *mut i64) -> i32 {
        if req.is_null() { crate::hle::kernel::set_errno(EINVAL); return -1; }
        std::thread::sleep(Duration::new((*req).max(0) as u64, (*req.add(1)).clamp(0, 999_999_999) as u32));
        if !rem.is_null() { *rem = 0; *rem.add(1) = 0; }
        0
    }
    fn usleep(us: u32) -> i32 { sleep_us(us as u64); 0 }
    fn sleep(s: u32) -> u32 { std::thread::sleep(Duration::from_secs(s as u64)); 0 }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libkernel", [
        sceKernelGetProcessTime, sceKernelGetProcessTimeCounter, sceKernelGetProcessTimeCounterFrequency,
        sceKernelReadTsc, sceKernelGetTscFrequency, sceKernelUsleep, sceKernelSleep, sceKernelNanosleep,
        sceKernelClockGettime, sceKernelClockGetres, sceKernelGettimeofday, sceKernelGettimezone,
    ]);
    reg!(r, "libScePosix", [clock_gettime, clock_getres, gettimeofday, nanosleep, usleep, sleep]);
}
