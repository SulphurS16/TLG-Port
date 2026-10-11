//! Host environment monitor: every report, the machine's CPU load, the CPU's
//! effective performance relative to its nominal clock (laptop power plans and
//! thermal limits throttle it heavily on the development machine), and this
//! process's share of the CPU. Timing results are only comparable between runs
//! with similar figures.

use windows_sys::Win32::System::Performance::{PDH_FMT_COUNTERVALUE, PDH_FMT_DOUBLE, PDH_HCOUNTER, PDH_HQUERY, PdhAddEnglishCounterW, PdhCollectQueryData, PdhGetFormattedCounterValue, PdhOpenQueryW};

struct Pdh {
    query: PDH_HQUERY,
    load: PDH_HCOUNTER,
    perf: PDH_HCOUNTER,
}

// PDH handles are used from the host thread only (under the lock).
unsafe impl Send for Pdh {}

static STATE: parking_lot::Mutex<Option<(Pdh, u64, std::time::Instant)>> = parking_lot::Mutex::new(None);

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

fn process_cpu_100ns() -> u64 {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
    let z = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut c, mut e, mut k, mut u) = (z, z, z, z);
    unsafe { GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) };
    let f = |t: FILETIME| ((t.dwHighDateTime as u64) << 32) | t.dwLowDateTime as u64;
    f(k) + f(u)
}

/// Logs one environment line (the first call only primes the counters).
pub fn report() {
    let mut g = STATE.lock();
    if g.is_none() {
        let mut q: PDH_HQUERY = unsafe { std::mem::zeroed() };
        if unsafe { PdhOpenQueryW(std::ptr::null(), 0, &mut q) } != 0 {
            return;
        }
        let (mut load, mut perf): (PDH_HCOUNTER, PDH_HCOUNTER) = unsafe { (std::mem::zeroed(), std::mem::zeroed()) };
        unsafe {
            PdhAddEnglishCounterW(q, wide("\\Processor(_Total)\\% Processor Time").as_ptr(), 0, &mut load);
            PdhAddEnglishCounterW(q, wide("\\Processor Information(_Total)\\% Processor Performance").as_ptr(), 0, &mut perf);
            PdhCollectQueryData(q);
        }
        *g = Some((Pdh { query: q, load, perf }, process_cpu_100ns(), std::time::Instant::now()));
        return;
    }
    let (pdh, last_cpu, last_t) = g.as_mut().unwrap();
    unsafe { PdhCollectQueryData(pdh.query) };
    let read = |c: PDH_HCOUNTER| -> f64 {
        let mut v: PDH_FMT_COUNTERVALUE = unsafe { std::mem::zeroed() };
        if unsafe { PdhGetFormattedCounterValue(c, PDH_FMT_DOUBLE, std::ptr::null_mut(), &mut v) } == 0 { unsafe { v.Anonymous.doubleValue } } else { f64::NAN }
    };
    let (load, perf) = (read(pdh.load), read(pdh.perf));
    let cpu = process_cpu_100ns();
    let dt = last_t.elapsed().as_secs_f64().max(1e-3);
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) as f64;
    let ours = (cpu - *last_cpu) as f64 / 1e7 / dt / cores * 100.0;
    *last_cpu = cpu;
    *last_t = std::time::Instant::now();
    crate::info!("env: machine CPU load {load:.0}%, CPU performance {perf:.0}% of nominal, this process {ours:.0}% of {cores} logical CPUs");
}
