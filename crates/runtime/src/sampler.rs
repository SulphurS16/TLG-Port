//! Sampling profiler for host threads (TLG_SAMPLE=<thread name>[,...], e.g.
//! `GPU-gfx,GPU-complete`). About every millisecond the thread is suspended, its
//! instruction pointer and callers (x64 unwind data of any module) are
//! recorded, and it is resumed. Every 10 s the functions with the most
//! samples (self and inclusive) are logged, symbolized from tlg.pdb (and
//! exports elsewhere), plus time outside tlg (heap, driver, system calls)
//! by the tlg function that called out. TLG_SAMPLE_TOP=<n> lengthens the
//! inclusive list (40); TLG_SAMPLE_FOCUS=<name part>[,...] adds, for each
//! function whose name contains one, the callees its samples spent time in.
//! TLG_SAMPLE_LINES=<name part>[,...] adds, for each such function, its
//! samples by source line (self time, and time in callees at the line of the
//! call; lines of inlined code are the inlined function's).
//! Spikes: the last seconds of samples are kept; `spike(t0, t1, label)`
//! (the GPU thread calls it for submissions over TLG_SAMPLE_SPIKE_MS, default
//! 25) logs the functions sampled in that window alone, and the stacks of all
//! spike windows so far go to `spikes-<thread>.txt` (for `tools/profile.py`).
//! The PDB only has public symbols (the build's debug information is DWARF):
//! code in other functions is named after the public symbol before it. Exact
//! names, inlined frames and lines: every distinct stack with its sample
//! count and the modules it touches are written to `samples-<thread>.txt` in
//! the run directory (rewritten every 10 s), which `tools/profile.py`
//! symbolizes from the executable's DWARF with llvm-symbolizer.

use std::collections::{HashMap, HashSet};
use std::os::windows::ffi::OsStrExt;
use windows_sys::Win32::Foundation::{DuplicateHandle, DUPLICATE_SAME_ACCESS, HANDLE};
use windows_sys::Win32::System::Diagnostics::Debug::*;
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetCurrentThread, ResumeThread, SuspendThread};

const DEPTH: usize = 48;
/// Unwinding stops past this much stack above the sampled frame (host
/// threads have at most 8 MiB).
const STACK_SPAN: u64 = 8 << 20;

#[repr(C, align(16))]
struct AlignedContext(CONTEXT);

/// Pending spike reports: window and label.
static SPIKES: parking_lot::Mutex<Vec<(std::time::Instant, std::time::Instant, String)>> = parking_lot::Mutex::new(Vec::new());
/// Samples kept for spike reports (~4 s at one per millisecond).
const KEEP: usize = 4096;

/// Whether the sampler runs (spike reports are worth requesting).
pub fn active() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("TLG_SAMPLE").is_some())
}

/// Spike threshold for callers (TLG_SAMPLE_SPIKE_MS, default 25 ms).
pub fn spike_threshold() -> std::time::Duration {
    static T: std::sync::OnceLock<std::time::Duration> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::time::Duration::from_secs_f64(std::env::var("TLG_SAMPLE_SPIKE_MS").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(25.0) / 1000.0))
}

/// Requests a report of the samples taken in [t0, t1] (the sampled thread's
/// slow piece of work).
pub fn spike(t0: std::time::Instant, t1: std::time::Instant, label: String) {
    if active() {
        let mut g = SPIKES.lock();
        if g.len() < 64 {
            g.push((t0, t1, label));
        }
    }
}

/// Called by threads when they name themselves: starts sampling the thread
/// TLG_SAMPLE names.
pub fn register_current_thread(name: &str) {
    let Ok(want) = std::env::var("TLG_SAMPLE") else { return };
    if !want.split(',').any(|w| w.trim() == name) {
        return;
    }
    let mut h: HANDLE = std::ptr::null_mut();
    unsafe { DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &mut h, 0, 0, DUPLICATE_SAME_ACCESS) };
    let h = h as usize;
    let name = name.to_string();
    std::thread::Builder::new().name("sampler".into()).spawn(move || run(h as HANDLE, name)).ok();
}

/// [start, end) of tlg.exe in memory.
fn module_range() -> (u64, u64) {
    unsafe {
        let base = GetModuleHandleW(std::ptr::null()) as u64;
        let nt = base + std::ptr::read_unaligned((base + 0x3C) as *const u32) as u64;
        let size = std::ptr::read_unaligned((nt + 0x50) as *const u32) as u64;
        (base, base + size)
    }
}

fn run(h: HANDLE, name: String) {
    crate::info!("sampler: profiling thread '{name}'");
    let (lo, hi) = module_range();
    let mut leaf: HashMap<u64, u64> = HashMap::new();
    let mut incl: HashMap<u64, u64> = HashMap::new();
    // Samples outside tlg: (leaf, first tlg caller).
    let mut outside: HashMap<(u64, u64), u64> = HashMap::new();
    // (caller frame, callee frame) pairs, once per sample (TLG_SAMPLE_FOCUS).
    let focus = std::env::var("TLG_SAMPLE_FOCUS").is_ok();
    let mut edges: HashMap<(u64, u64), u64> = HashMap::new();
    // Samples per code address (callers: the call instruction), once per
    // sample (TLG_SAMPLE_LINES).
    let lines = std::env::var("TLG_SAMPLE_LINES").is_ok();
    let mut at: HashMap<u64, u64> = HashMap::new();
    let mut n = 0u64;
    let mut last = std::time::Instant::now();
    let mut names: HashMap<u64, String> = HashMap::new();
    let mut sym_ready = false;
    let mut recent: std::collections::VecDeque<(std::time::Instant, Vec<u64>)> = std::collections::VecDeque::with_capacity(KEEP);
    // Distinct stacks of the whole run and their sample counts.
    let mut stacks: HashMap<Vec<u64>, u64> = HashMap::new();
    let init_sym = || {
        // tlg.pdb sits next to the executable.
        let dir: Vec<u16> = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.as_os_str().to_owned())).unwrap_or_default().encode_wide().chain([0]).collect();
        unsafe {
            SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | if lines { SYMOPT_LOAD_LINES } else { 0 });
            SymInitializeW(GetCurrentProcess(), dir.as_ptr(), 1);
        }
    };
    loop {
        std::thread::sleep(std::time::Duration::from_millis(1));
        let mut frames = [0u64; DEPTH];
        let mut k = 0usize;
        unsafe {
            if SuspendThread(h) == u32::MAX {
                break;
            }
            let mut ctx: AlignedContext = std::mem::zeroed();
            ctx.0.ContextFlags = CONTEXT_FULL_AMD64;
            if GetThreadContext(h, &mut ctx.0) != 0 {
                let c = &mut ctx.0;
                let rsp0 = c.Rsp;
                while k < DEPTH {
                    let rip = c.Rip;
                    frames[k] = rip;
                    k += 1;
                    let mut image = 0u64;
                    let f = RtlLookupFunctionEntry(rip, &mut image, std::ptr::null_mut());
                    if f.is_null() && k > 1 {
                        // No unwind data past the leaf: guest code and the HLE
                        // entry trampolines keep an rbp frame chain.
                        let rbp = c.Rbp;
                        let mut pair = [0u64; 2];
                        // (Checked read: a stale rbp may point anywhere.)
                        if rbp < c.Rsp || rbp >= rsp0 + STACK_SPAN || rbp % 8 != 0 || ReadProcessMemory(GetCurrentProcess(), rbp as _, pair.as_mut_ptr() as _, 16, std::ptr::null_mut()) == 0 {
                            break;
                        }
                        c.Rip = pair[1];
                        c.Rsp = rbp + 16;
                        c.Rbp = pair[0];
                    } else if f.is_null() {
                        // A leaf function may lack unwind data.
                        c.Rip = std::ptr::read_unaligned(c.Rsp as *const u64);
                        c.Rsp += 8;
                    } else {
                        let mut hd = std::ptr::null_mut();
                        let mut est = 0u64;
                        RtlVirtualUnwind(UNW_FLAG_NHANDLER, image, rip, f, c, &mut hd, &mut est, std::ptr::null_mut());
                    }
                    if c.Rip == 0 || c.Rsp < rsp0 || c.Rsp >= rsp0 + STACK_SPAN {
                        break;
                    }
                }
            }
            ResumeThread(h);
        }
        if k == 0 {
            continue;
        }
        if recent.len() == KEEP {
            recent.pop_front();
        }
        recent.push_back((std::time::Instant::now(), frames[..k].to_vec()));
        let ready: Vec<(std::time::Instant, std::time::Instant, String)> = {
            let mut g = SPIKES.lock();
            let now = std::time::Instant::now();
            let (done, wait): (Vec<_>, Vec<_>) = g.drain(..).partition(|s| s.1 + std::time::Duration::from_millis(5) <= now);
            *g = wait;
            done
        };
        for (t0, t1, label) in ready {
            if !sym_ready {
                init_sym();
                sym_ready = true;
            }
            report_spike(&name, t0, t1, &label, &recent, (lo, hi), &mut names);
        }
        n += 1;
        *stacks.entry(frames[..k].to_vec()).or_default() += 1;
        *leaf.entry(frames[0]).or_default() += 1;
        if frames[0] < lo || frames[0] >= hi {
            let caller = frames[..k].iter().copied().find(|&f| f >= lo && f < hi).unwrap_or(0);
            *outside.entry((frames[0], caller)).or_default() += 1;
        }
        let mut seen = HashSet::new();
        for &f in &frames[..k] {
            if seen.insert(f) {
                *incl.entry(f).or_default() += 1;
            }
        }
        if focus {
            let mut seen_e = HashSet::new();
            for i in 0..k.saturating_sub(1) {
                let e = (frames[i + 1], frames[i]);
                if seen_e.insert(e) {
                    *edges.entry(e).or_default() += 1;
                }
            }
        }
        if lines {
            let mut seen_a = HashSet::new();
            for (i, &f) in frames[..k].iter().enumerate() {
                // (A return address follows the call: its line is the next
                // statement's.)
                let a = if i == 0 { f } else { f - 1 };
                if f >= lo && f < hi && seen_a.insert(a) {
                    *at.entry(a).or_default() += 1;
                }
            }
        }
        if last.elapsed() >= std::time::Duration::from_secs(10) {
            if !sym_ready {
                init_sym();
                sym_ready = true;
            }
            report(&name, n, &leaf, &incl, &outside, &mut names);
            write_stacks(&name, &stacks);
            if focus {
                report_focus(&name, n, &edges, &mut names);
            }
            if lines {
                report_lines(&name, n, &at, &mut names);
            }
            leaf.clear();
            incl.clear();
            outside.clear();
            edges.clear();
            at.clear();
            n = 0;
            last = std::time::Instant::now();
        }
    }
}

/// Writes `samples-<thread>.txt`: `# module <base> <size> <path>` lines for
/// the modules the stacks touch, then `<count> <frame>...` per distinct
/// stack (absolute addresses, leaf first; callers are return addresses).
fn write_stacks(thread: &str, stacks: &HashMap<Vec<u64>, u64>) {
    write_stacks_as("samples", thread, stacks);
}

/// `write_stacks` into `<prefix>-<thread>.txt`.
fn write_stacks_as(prefix: &str, thread: &str, stacks: &HashMap<Vec<u64>, u64>) {
    use windows_sys::Win32::System::LibraryLoader::{GetModuleFileNameW, GetModuleHandleExW, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT};
    let Some(dir) = crate::run_dir() else { return };
    let mut modules: std::collections::BTreeMap<u64, (u64, String)> = std::collections::BTreeMap::new();
    let mut seen: HashSet<u64> = HashSet::new();
    for f in stacks.keys().flatten() {
        if !seen.insert(*f) || modules.range(..=*f).next_back().is_some_and(|(b, (s, _))| f - b < *s) {
            continue;
        }
        let mut h = std::ptr::null_mut();
        if unsafe { GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, *f as *const u16, &mut h) } == 0 {
            continue;
        }
        let base = h as u64;
        let mut buf = [0u16; 520];
        let len = unsafe { GetModuleFileNameW(h, buf.as_mut_ptr(), buf.len() as u32) } as usize;
        let size = unsafe {
            let nt = base + std::ptr::read_unaligned((base + 0x3C) as *const u32) as u64;
            std::ptr::read_unaligned((nt + 0x50) as *const u32) as u64
        };
        modules.insert(base, (size, String::from_utf16_lossy(&buf[..len])));
    }
    let mut out = String::new();
    for (b, (s, p)) in &modules {
        out.push_str(&format!("# module {b:#x} {s:#x} {p}\n"));
    }
    let mut v: Vec<(&Vec<u64>, &u64)> = stacks.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1));
    for (frames, c) in v {
        out.push_str(&c.to_string());
        for f in frames {
            out.push_str(&format!(" {f:#x}"));
        }
        out.push('\n');
    }
    let name: String = thread.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).collect();
    let _ = std::fs::write(dir.join(format!("{prefix}-{name}.txt")), out);
}

/// The functions sampled in [t0, t1]: self time, inclusive time (tlg
/// functions only, the most frequent first) and calls out of tlg.
fn report_spike(thread: &str, t0: std::time::Instant, t1: std::time::Instant, label: &str, recent: &std::collections::VecDeque<(std::time::Instant, Vec<u64>)>, (lo, hi): (u64, u64), cache: &mut HashMap<u64, String>) {
    let window: Vec<&Vec<u64>> = recent.iter().filter(|(t, _)| *t >= t0 && *t <= t1).map(|(_, f)| f).collect();
    let n = window.len();
    if n == 0 {
        crate::info!("sampler '{thread}' spike {label}: {:.1} ms, no samples kept", (t1 - t0).as_secs_f64() * 1000.0);
        return;
    }
    // (Every spike's stacks so far, for `tools/profile.py spikes-<thread>.txt`.)
    {
        static ALL: parking_lot::Mutex<Option<HashMap<String, HashMap<Vec<u64>, u64>>>> = parking_lot::Mutex::new(None);
        let mut g = ALL.lock();
        let m = g.get_or_insert_with(HashMap::new).entry(thread.to_string()).or_default();
        for frames in &window {
            *m.entry((*frames).clone()).or_default() += 1;
        }
        write_stacks_as("spikes", thread, m);
    }
    let mut leaf: HashMap<String, u64> = HashMap::new();
    let mut incl: HashMap<String, u64> = HashMap::new();
    for frames in &window {
        *leaf.entry(symbol(frames[0], cache)).or_default() += 1;
        let mut seen = HashSet::new();
        for &f in frames.iter() {
            if f >= lo && f < hi {
                let s = symbol(f, cache);
                if seen.insert(s.clone()) {
                    *incl.entry(s).or_default() += 1;
                }
            }
        }
    }
    let top = |m: HashMap<String, u64>, k: usize| {
        let mut v: Vec<(String, u64)> = m.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v.into_iter().take(k).map(|(f, c)| format!("{:.0}% {}", 100.0 * c as f64 / n as f64, f.chars().take(90).collect::<String>())).collect::<Vec<_>>().join("; ")
    };
    crate::info!(
        "sampler '{thread}' spike {label}: {:.1} ms, {n} samples; self: {}; inclusive: {}",
        (t1 - t0).as_secs_f64() * 1000.0,
        top(leaf, 6),
        top(incl, 14)
    );
}

/// TLG_SAMPLE_FOCUS: per focused function, the callees its samples were in.
fn report_focus(thread: &str, n: u64, edges: &HashMap<(u64, u64), u64>, cache: &mut HashMap<u64, String>) {
    let Ok(want) = std::env::var("TLG_SAMPLE_FOCUS") else { return };
    let mut by: HashMap<(String, String), u64> = HashMap::new();
    for (&(caller, callee), &c) in edges {
        let (a, b) = (symbol(caller, cache), symbol(callee, cache));
        if a != b {
            *by.entry((a, b)).or_default() += c;
        }
    }
    for part in want.split([',', ';']).map(str::trim).filter(|p| !p.is_empty()) {
        let mut v: Vec<(&(String, String), &u64)> = by.iter().filter(|((a, _), _)| a.contains(part)).collect();
        v.sort_by(|x, y| y.1.cmp(x.1));
        crate::info!("sampler '{thread}': callees of '{part}':");
        for ((a, b), c) in v.into_iter().take(16) {
            crate::info!("  {:5.1}%  {} <- {}", 100.0 * *c as f64 / n as f64, b.chars().take(110).collect::<String>(), a.chars().take(60).collect::<String>());
        }
    }
}

/// TLG_SAMPLE_LINES: per matching function, its samples by source line.
fn report_lines(thread: &str, n: u64, at: &HashMap<u64, u64>, cache: &mut HashMap<u64, String>) {
    let Ok(want) = std::env::var("TLG_SAMPLE_LINES") else { return };
    for part in want.split([',', ';']).map(str::trim).filter(|p| !p.is_empty()) {
        let mut by: HashMap<String, u64> = HashMap::new();
        let mut total = 0u64;
        for (&a, &c) in at {
            let f = symbol(a, cache);
            if !f.contains(part) {
                continue;
            }
            let short = f.rsplit("::").next().unwrap_or(&f).chars().take(40).collect::<String>();
            *by.entry(format!("{short} {}", source_line(a).unwrap_or_else(|| "?".into()))).or_default() += c;
            total += c;
        }
        let mut v: Vec<(String, u64)> = by.into_iter().collect();
        v.sort_by(|x, y| y.1.cmp(&x.1));
        crate::info!("sampler '{thread}': lines of '{part}' ({:.1}% of samples at their lines; a sample counts once per line):", 100.0 * total as f64 / n.max(1) as f64);
        for (l, c) in v.into_iter().take(30) {
            crate::info!("  {:5.1}%  {}", 100.0 * c as f64 / n.max(1) as f64, l);
        }
    }
}

/// `file:line` of a code address in tlg (from tlg.pdb).
fn source_line(addr: u64) -> Option<String> {
    unsafe {
        let mut l: IMAGEHLP_LINEW64 = std::mem::zeroed();
        l.SizeOfStruct = std::mem::size_of::<IMAGEHLP_LINEW64>() as u32;
        let mut disp = 0u32;
        if SymGetLineFromAddrW64(GetCurrentProcess(), addr, &mut disp, &mut l) == 0 || l.FileName.is_null() {
            return None;
        }
        let mut len = 0usize;
        while *l.FileName.add(len) != 0 && len < 1024 {
            len += 1;
        }
        let file = String::from_utf16_lossy(std::slice::from_raw_parts(l.FileName, len));
        let file = file.rsplit(['\\', '/']).next().unwrap_or("").to_string();
        Some(format!("{file}:{}", l.LineNumber))
    }
}

fn symbol(addr: u64, cache: &mut HashMap<u64, String>) -> String {
    if let Some(s) = cache.get(&addr) {
        return s.clone();
    }
    const MAX: usize = 512;
    let mut buf = vec![0u64; (std::mem::size_of::<SYMBOL_INFOW>() + MAX * 2).div_ceil(8)];
    let info = buf.as_mut_ptr() as *mut SYMBOL_INFOW;
    let name = unsafe {
        (*info).SizeOfStruct = std::mem::size_of::<SYMBOL_INFOW>() as u32;
        (*info).MaxNameLen = MAX as u32;
        let mut disp = 0u64;
        if SymFromAddrW(GetCurrentProcess(), addr, &mut disp, info) != 0 {
            let len = ((*info).NameLen as usize).min(MAX);
            String::from_utf16_lossy(std::slice::from_raw_parts((*info).Name.as_ptr(), len))
        } else {
            // Guest code: module+offset of the address itself.
            crate::loader::describe_addr(addr)
        }
    };
    cache.insert(addr, name.clone());
    name
}

fn report(thread: &str, n: u64, leaf: &HashMap<u64, u64>, incl: &HashMap<u64, u64>, outside: &HashMap<(u64, u64), u64>, cache: &mut HashMap<u64, String>) {
    if n == 0 {
        return;
    }
    // Aggregate by function name.
    let mut by = |m: &HashMap<u64, u64>| {
        let mut agg: HashMap<String, u64> = HashMap::new();
        for (&a, &c) in m {
            *agg.entry(symbol(a, cache)).or_default() += c;
        }
        let mut v: Vec<(String, u64)> = agg.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    };
    let self_ = by(leaf);
    let incl = by(incl);
    crate::info!("sampler '{thread}': {n} samples; self time:");
    for (f, c) in self_.iter().take(25) {
        crate::info!("  {:5.1}%  {}", 100.0 * *c as f64 / n as f64, f.chars().take(150).collect::<String>());
    }
    crate::info!("sampler '{thread}': inclusive time:");
    let top = std::env::var("TLG_SAMPLE_TOP").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
    for (f, c) in incl.iter().take(top) {
        crate::info!("  {:5.1}%  {}", 100.0 * *c as f64 / n as f64, f.chars().take(150).collect::<String>());
    }
    let mut out: HashMap<(String, String), u64> = HashMap::new();
    for (&(l, c), &v) in outside {
        let callee = symbol(l, cache);
        let caller = if c == 0 { "?".to_string() } else { symbol(c, cache) };
        *out.entry((callee, caller)).or_default() += v;
    }
    let mut out: Vec<((String, String), u64)> = out.into_iter().collect();
    out.sort_by(|a, b| b.1.cmp(&a.1));
    crate::info!("sampler '{thread}': time outside tlg by caller:");
    for ((callee, caller), c) in out.iter().take(30) {
        crate::info!("  {:5.1}%  {} <- {}", 100.0 * *c as f64 / n as f64, callee.chars().take(60).collect::<String>(), caller.chars().take(120).collect::<String>());
    }
}
