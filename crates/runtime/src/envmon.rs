//! Host environment monitor: every report, the machine's CPU load, the CPU's
//! effective performance relative to its nominal clock (laptop power plans and
//! thermal limits throttle it heavily on the development machine), this
//! process's share of the CPU, and the 3D engine utilization of the GPU this
//! process uses, ours and other processes' (another process rendering slows
//! our GPU work). Timing results are only comparable between runs with
//! similar figures.

use windows_sys::Win32::System::Performance::{
    PDH_FMT_COUNTERVALUE, PDH_FMT_COUNTERVALUE_ITEM_W, PDH_FMT_DOUBLE, PDH_HCOUNTER, PDH_HQUERY, PdhAddEnglishCounterW, PdhCollectQueryData, PdhGetFormattedCounterArrayW, PdhGetFormattedCounterValue,
    PdhOpenQueryW,
};

struct Pdh {
    query: PDH_HQUERY,
    load: PDH_HCOUNTER,
    perf: PDH_HCOUNTER,
    /// `GPU Engine(*engtype_3D)`: one instance per process and engine.
    gpu: PDH_HCOUNTER,
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

/// 3D engine utilization (percent) of the adapter where this process uses
/// the most, (ours, other processes'); None without the counters. Instance
/// names read `pid_<pid>_luid_<hi>_<lo>_phys_<n>_eng_<n>_engtype_3D`.
fn gpu_3d(c: PDH_HCOUNTER) -> Option<(f64, f64)> {
    let (mut size, mut count) = (0u32, 0u32);
    unsafe { PdhGetFormattedCounterArrayW(c, PDH_FMT_DOUBLE, &mut size, &mut count, std::ptr::null_mut()) };
    if size == 0 {
        return None;
    }
    let mut buf = vec![0u64; (size as usize).div_ceil(8)];
    let items = buf.as_mut_ptr() as *mut PDH_FMT_COUNTERVALUE_ITEM_W;
    if unsafe { PdhGetFormattedCounterArrayW(c, PDH_FMT_DOUBLE, &mut size, &mut count, items) } != 0 {
        return None;
    }
    let me = std::process::id();
    // Per adapter (luid): (ours, others).
    let mut by: std::collections::HashMap<String, (f64, f64)> = std::collections::HashMap::new();
    for i in 0..count as usize {
        let it = unsafe { &*items.add(i) };
        let name = unsafe {
            let mut n = 0;
            while *it.szName.add(n) != 0 {
                n += 1;
            }
            String::from_utf16_lossy(std::slice::from_raw_parts(it.szName, n))
        };
        let v = unsafe { it.FmtValue.Anonymous.doubleValue };
        let mut parts = name.split('_');
        let (Some("pid"), Some(pid)) = (parts.next(), parts.next().and_then(|p| p.parse::<u32>().ok())) else { continue };
        let luid = name.split("_luid_").nth(1).and_then(|r| r.split("_phys").next()).unwrap_or("?").to_string();
        let e = by.entry(luid).or_default();
        if pid == me {
            e.0 += v;
        } else {
            e.1 += v;
        }
    }
    by.into_values().max_by(|a, b| a.0.total_cmp(&b.0))
}

/// Logs one environment line (the first call only primes the counters).
pub fn report() {
    let mut g = STATE.lock();
    if g.is_none() {
        let mut q: PDH_HQUERY = unsafe { std::mem::zeroed() };
        if unsafe { PdhOpenQueryW(std::ptr::null(), 0, &mut q) } != 0 {
            return;
        }
        let (mut load, mut perf, mut gpu): (PDH_HCOUNTER, PDH_HCOUNTER, PDH_HCOUNTER) = unsafe { (std::mem::zeroed(), std::mem::zeroed(), std::mem::zeroed()) };
        unsafe {
            PdhAddEnglishCounterW(q, wide(r"\Processor(_Total)\% Processor Time").as_ptr(), 0, &mut load);
            PdhAddEnglishCounterW(q, wide(r"\Processor Information(_Total)\% Processor Performance").as_ptr(), 0, &mut perf);
            PdhAddEnglishCounterW(q, wide(r"\GPU Engine(*engtype_3D)\Utilization Percentage").as_ptr(), 0, &mut gpu);
            PdhCollectQueryData(q);
        }
        *g = Some((Pdh { query: q, load, perf, gpu }, process_cpu_100ns(), std::time::Instant::now()));
        return;
    }
    let (pdh, last_cpu, last_t) = g.as_mut().unwrap();
    unsafe { PdhCollectQueryData(pdh.query) };
    let read = |c: PDH_HCOUNTER| -> f64 {
        let mut v: PDH_FMT_COUNTERVALUE = unsafe { std::mem::zeroed() };
        if unsafe { PdhGetFormattedCounterValue(c, PDH_FMT_DOUBLE, std::ptr::null_mut(), &mut v) } == 0 { unsafe { v.Anonymous.doubleValue } } else { f64::NAN }
    };
    let (load, perf) = (read(pdh.load), read(pdh.perf));
    let gpu = gpu_3d(pdh.gpu).map(|(o, x)| format!(", GPU 3D engine: this process {o:.0}%, others {x:.0}%")).unwrap_or_default();
    let cpu = process_cpu_100ns();
    let dt = last_t.elapsed().as_secs_f64().max(1e-3);
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) as f64;
    let ours = (cpu - *last_cpu) as f64 / 1e7 / dt / cores * 100.0;
    *last_cpu = cpu;
    *last_t = std::time::Instant::now();
    crate::info!("env: machine CPU load {load:.0}%, CPU performance {perf:.0}% of nominal, this process {ours:.0}% of {cores} logical CPUs{gpu}");
}
