//! Guest threads: each guest thread is a host thread running guest code natively
//! on its own stack, with its TLS slot pointing at the guest thread pointer.

use parking_lot::{Condvar, Mutex};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};

std::arch::global_asm!(
    // u64 tlg_guest_thread_call(entry, a0, a1, *mut saved_rsp)  [sysv64]
    ".globl tlg_guest_thread_call",
    "tlg_guest_thread_call:",
    "push rbp",
    "mov rbp, rsp",
    "push rbx",
    "push r12",
    "push r13",
    "push r14",
    "push r15",
    "sub rsp, 8",
    "mov [rcx], rsp",
    "mov rax, rdi",
    "mov rdi, rsi",
    "mov rsi, rdx",
    "call rax",
    "tlg_guest_thread_resume:",
    "add rsp, 8",
    "pop r15",
    "pop r14",
    "pop r13",
    "pop r12",
    "pop rbx",
    "pop rbp",
    "ret",
    // ! tlg_guest_thread_exit(saved_rsp, value)  [sysv64]
    ".globl tlg_guest_thread_exit",
    "tlg_guest_thread_exit:",
    "mov rsp, rdi",
    "mov rax, rsi",
    "jmp tlg_guest_thread_resume",
);

unsafe extern "sysv64" {
    fn tlg_guest_thread_call(entry: u64, a0: u64, a1: u64, saved_rsp: *mut u64) -> u64;
    fn tlg_guest_thread_exit(saved_rsp: u64, value: u64) -> !;
}

pub const THREAD_MAGIC: u64 = 0x4242_5448_5245_4144; // "BBTHREAD"
pub const PTHREAD_KEYS_MAX: usize = 256;

pub struct GuestThread {
    pub magic: u64,
    pub id: u64,
    pub name: Mutex<String>,
    pub tp: AtomicU64,
    pub entry: u64,
    pub arg: u64,
    pub stack_size: u64,
    pub detached: AtomicBool,
    pub exited: Mutex<Option<u64>>,
    pub exit_cv: Condvar,
    pub saved_rsp: AtomicU64,
    pub errno: AtomicI32,
    pub priority: AtomicI32,
    pub affinity: AtomicU64,
    pub specific: Mutex<Vec<u64>>,
    pub host_tid: AtomicU64,
    /// HLE function the thread is in (address of a `&'static str`, 0 = guest code) and
    /// since when (ms since start): what a stuck thread is waiting in (`threads` command).
    pub hle_call: AtomicU64,
    pub hle_since: AtomicU64,
}

/// Records entry into HLE function `name`; returns the previous record for `hle_exit`.
#[inline]
pub fn hle_enter(name: &'static &'static str) -> (u64, u64) {
    match current() {
        Some(t) => {
            let since = crate::log::elapsed_ms() as u64;
            (t.hle_call.swap(name as *const &str as u64, Ordering::Relaxed), t.hle_since.swap(since, Ordering::Relaxed))
        }
        None => (0, 0),
    }
}

#[inline]
pub fn hle_exit(prev: (u64, u64)) {
    if let Some(t) = current() {
        t.hle_call.store(prev.0, Ordering::Relaxed);
        t.hle_since.store(prev.1, Ordering::Relaxed);
    }
}

/// Logs every guest thread with the HLE call it is in (diagnosing hangs).
pub fn dump_threads(why: &str) {
    let now = crate::log::elapsed_ms() as u64;
    let mut s = format!("guest threads ({why}):");
    for t in THREADS.lock().iter() {
        let exited = t.exited.try_lock().map(|g| g.is_some()).unwrap_or(false);
        let call = t.hle_call.load(Ordering::Relaxed);
        let what = if exited {
            "exited".to_string()
        } else if call == 0 {
            "in guest code".to_string()
        } else {
            let name = unsafe { *(call as *const &'static str) };
            format!("in {name} for {} ms", now.saturating_sub(t.hle_since.load(Ordering::Relaxed)))
        };
        s.push_str(&format!("\n  {:>3} {:<32} {what}", t.id, t.name()));
    }
    crate::info!("{s}");
    crate::hle::kernel::equeue::dump();
    crate::hle::kernel::sync::dump_event_flags();
    crate::gpu::dump_state();
}

/// Every thread's CPU time (user + kernel) since the previous call, or since
/// it started, in ms per second of wall time, busiest first and summed by
/// name (`cpu` control command): who shares the cores and the power budget
/// with the command processors.
pub fn cpu_report() {
    use windows_sys::Win32::Foundation::{CloseHandle, FILETIME, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, THREADENTRY32, TH32CS_SNAPTHREAD, Thread32First, Thread32Next};
    use windows_sys::Win32::System::Threading::{GetCurrentProcessId, GetThreadDescription, GetThreadTimes, OpenThread, THREAD_QUERY_LIMITED_INFORMATION};
    static LAST: Mutex<Option<(std::time::Instant, std::collections::HashMap<u32, u64>)>> = Mutex::new(None);
    let ft = |f: &FILETIME| ((f.dwHighDateTime as u64) << 32) | f.dwLowDateTime as u64;
    let mut now: std::collections::HashMap<u32, (String, u64)> = std::collections::HashMap::new();
    unsafe {
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if snap == INVALID_HANDLE_VALUE {
            return;
        }
        let pid = GetCurrentProcessId();
        let mut e: THREADENTRY32 = std::mem::zeroed();
        e.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
        let mut ok = Thread32First(snap, &mut e) != 0;
        while ok {
            if e.th32OwnerProcessID == pid {
                let h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, e.th32ThreadID);
                if !h.is_null() {
                    let (mut c, mut x, mut k, mut u) = (std::mem::zeroed(), std::mem::zeroed(), std::mem::zeroed(), std::mem::zeroed());
                    if GetThreadTimes(h, &mut c, &mut x, &mut k, &mut u) != 0 {
                        let mut name = String::new();
                        let mut w: *mut u16 = std::ptr::null_mut();
                        if GetThreadDescription(h, &mut w) >= 0 && !w.is_null() {
                            let n = (0..).take_while(|&i| *w.add(i) != 0).count();
                            name = String::from_utf16_lossy(std::slice::from_raw_parts(w, n));
                            windows_sys::Win32::Foundation::LocalFree(w as _);
                        }
                        if name.is_empty() {
                            name = format!("tid {}", e.th32ThreadID);
                        }
                        now.insert(e.th32ThreadID, (name, ft(&k) + ft(&u)));
                    }
                    CloseHandle(h);
                }
            }
            ok = Thread32Next(snap, &mut e) != 0;
        }
        CloseHandle(snap);
    }
    let mut last = LAST.lock();
    let (since, prev) = last.take().unwrap_or_else(|| (crate::log::start_instant(), std::collections::HashMap::new()));
    let secs = since.elapsed().as_secs_f64().max(1e-3);
    let mut by_name: std::collections::HashMap<String, (f64, usize)> = std::collections::HashMap::new();
    for (tid, (name, t)) in &now {
        let ms = t.saturating_sub(prev.get(tid).copied().unwrap_or(0)) as f64 / 1e4 / secs;
        // Numbered siblings (job workers) summed under one name.
        let key = name.trim_end_matches(|c: char| c.is_ascii_digit()).to_string();
        let v = by_name.entry(key).or_default();
        v.0 += ms;
        v.1 += 1;
    }
    let total: f64 = by_name.values().map(|v| v.0).sum();
    let mut rows: Vec<(String, (f64, usize))> = by_name.into_iter().filter(|r| r.1.0 >= 5.0).collect();
    rows.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
    let list: Vec<String> = rows.iter().map(|(n, (ms, k))| if *k > 1 { format!("{n}* x{k} {ms:.0}") } else { format!("{n} {ms:.0}") }).collect();
    let kick = [crate::title_patches::KICK_STATS[0].swap(0, Ordering::Relaxed), crate::title_patches::KICK_STATS[1].swap(0, Ordering::Relaxed)];
    let hrt: [u64; 3] = std::array::from_fn(|i| crate::hle::kernel::equeue::HRT_STATS[i].swap(0, Ordering::Relaxed));
    crate::info!(
        "thread CPU over {secs:.1} s (ms per s; threads under 5 left out): total {total:.0}; {}; render-kick waits {:.0}/s ({:.0} ms per s); HR timers {:.0}/s ({:.0} us each, final spins {:.0} ms per s)",
        list.join(", "),
        kick[0] as f64 / secs,
        kick[1] as f64 / 1e6 / secs,
        hrt[0] as f64 / secs,
        hrt[1] as f64 / hrt[0].max(1) as f64,
        hrt[2] as f64 / 1e3 / secs
    );
    *last = Some((std::time::Instant::now(), now.into_iter().map(|(k, v)| (k, v.1)).collect()));
}

/// Spins until `done()` holds or `limit` has passed; returns `done()`.
/// Mostly pause instructions: the clock is read once per `tune spin_pauses`
/// of them.
#[inline]
pub fn spin_until(limit: std::time::Duration, mut done: impl FnMut() -> bool) -> bool {
    let t0 = std::time::Instant::now();
    let per = crate::tune::get(crate::tune::SPIN_PAUSES).max(1);
    loop {
        for _ in 0..per {
            if done() {
                return true;
            }
            std::hint::spin_loop();
        }
        if t0.elapsed() >= limit {
            return done();
        }
    }
}

/// Gives the calling host thread above-normal priority: the command
/// processors, the renderer and its completion thread are the frame's
/// critical path, while the title's job workers poll their queue lock on
/// every core they get.
pub fn raise_priority() {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL};
    unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL) };
}

/// The current thread at raised (`raise_priority`) or normal priority.
pub fn set_raised_priority(raised: bool) {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL, THREAD_PRIORITY_NORMAL};
    unsafe { SetThreadPriority(GetCurrentThread(), if raised { THREAD_PRIORITY_ABOVE_NORMAL } else { THREAD_PRIORITY_NORMAL }) };
}

/// Samples where every guest thread is for `ms` milliseconds (about once a
/// millisecond, on a thread of its own) and logs each thread's hottest
/// locations: guest code by module offset (64-byte buckets), otherwise the
/// HLE call it is in. No lock is taken while a thread is suspended.
pub fn sample_threads(ms: u64) {
    let _ = std::thread::Builder::new().name("sampler".into()).spawn(move || {
        use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, GetThreadContext};
        use windows_sys::Win32::System::Threading::{OpenThread, ResumeThread, SuspendThread, THREAD_GET_CONTEXT, THREAD_QUERY_INFORMATION, THREAD_SUSPEND_RESUME};
        #[repr(C, align(16))]
        struct Ctx(CONTEXT);
        const CONTEXT_CONTROL_AMD64: u32 = 0x0010_0001;
        let threads: Vec<(String, u64, &'static GuestThread)> = THREADS.lock().iter().map(|t| (t.name(), t.host_tid.load(Ordering::Relaxed), *t)).filter(|(_, tid, _)| *tid != 0).collect();
        let handles: Vec<_> = threads.iter().map(|(_, tid, _)| unsafe { OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, 0, *tid as u32) }).collect();
        let mut samples: Vec<Vec<(u64, u64)>> = vec![Vec::new(); threads.len()];
        let end = std::time::Instant::now() + std::time::Duration::from_millis(ms);
        while std::time::Instant::now() < end {
            for (k, &h) in handles.iter().enumerate() {
                if h.is_null() {
                    continue;
                }
                let mut c: Ctx = unsafe { std::mem::zeroed() };
                c.0.ContextFlags = CONTEXT_CONTROL_AMD64;
                unsafe {
                    if SuspendThread(h) == u32::MAX {
                        continue;
                    }
                    let ok = GetThreadContext(h, &mut c.0) != 0;
                    let call = threads[k].2.hle_call.load(Ordering::Relaxed);
                    ResumeThread(h);
                    if ok {
                        samples[k].push((c.0.Rip, call));
                    }
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        for h in handles {
            if !h.is_null() {
                unsafe { windows_sys::Win32::Foundation::CloseHandle(h) };
            }
        }
        let mut s = format!("thread samples over {ms} ms:");
        for (k, (name, _, _)) in threads.iter().enumerate() {
            let n = samples[k].len().max(1);
            let mut hist: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            for &(rip, call) in &samples[k] {
                let what = if crate::mem::is_guest(rip, 1) {
                    crate::loader::describe_addr(rip & !0x3F)
                } else if call != 0 {
                    format!("HLE {}", unsafe { *(call as *const &'static str) })
                } else {
                    "host".to_string()
                };
                *hist.entry(what).or_default() += 1;
            }
            let mut v: Vec<(String, usize)> = hist.into_iter().collect();
            v.sort_by(|a, b| b.1.cmp(&a.1));
            let top: Vec<String> = v.iter().take(6).map(|(w, c)| format!("{w} {}%", c * 100 / n)).collect();
            s.push_str(&format!("\n  {name:<32} {}", top.join(", ")));
        }
        crate::info!("{s}");
    });
}

impl GuestThread {
    pub fn handle(&'static self) -> u64 {
        self as *const _ as u64
    }
    pub fn name(&self) -> String {
        self.name.lock().clone()
    }
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
pub static THREADS: Mutex<Vec<&'static GuestThread>> = Mutex::new(Vec::new());

thread_local! {
    static CURRENT: Cell<Option<&'static GuestThread>> = const { Cell::new(None) };
}

pub fn current() -> Option<&'static GuestThread> {
    CURRENT.with(|c| c.get())
}

/// Validates a guest-supplied thread handle.
pub fn from_handle(h: u64) -> Option<&'static GuestThread> {
    if h == 0 {
        return None;
    }
    let threads = THREADS.lock();
    threads.iter().copied().find(|t| t.handle() == h && t.magic == THREAD_MAGIC)
}

pub fn new_thread(name: &str, entry: u64, arg: u64, stack_size: u64, priority: i32) -> &'static GuestThread {
    let t: &'static GuestThread = Box::leak(Box::new(GuestThread {
        magic: THREAD_MAGIC,
        id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        name: Mutex::new(name.to_string()),
        tp: AtomicU64::new(0),
        entry,
        arg,
        stack_size,
        detached: AtomicBool::new(false),
        exited: Mutex::new(None),
        exit_cv: Condvar::new(),
        saved_rsp: AtomicU64::new(0),
        errno: AtomicI32::new(0),
        priority: AtomicI32::new(priority),
        affinity: AtomicU64::new(0x7F),
        specific: Mutex::new(vec![0; PTHREAD_KEYS_MAX]),
        host_tid: AtomicU64::new(0),
        hle_call: AtomicU64::new(0),
        hle_since: AtomicU64::new(0),
    }));
    let tp = crate::tls::alloc_thread_area(t.handle());
    t.tp.store(tp, Ordering::SeqCst);
    THREADS.lock().push(t);
    t
}

/// Binds the calling host thread to a guest thread (TLS slot + current pointer).
pub fn enter(t: &'static GuestThread) {
    CURRENT.with(|c| c.set(Some(t)));
    crate::tls::set_current_tp(t.tp.load(Ordering::SeqCst));
    crate::log::set_thread_name(&t.name());
    t.host_tid.store(unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() } as u64, Ordering::SeqCst);
}

/// Runs guest code `entry(a0, a1)` on the current (already entered) thread.
/// Returns when the function returns or the thread calls pthread_exit.
pub fn run_entry(t: &'static GuestThread, entry: u64, a0: u64, a1: u64) -> u64 {
    unsafe { tlg_guest_thread_call(entry, a0, a1, t.saved_rsp.as_ptr()) }
}

/// Terminates the current guest thread with `value` (pthread_exit semantics).
pub fn exit_current(value: u64) -> ! {
    let t = current().expect("pthread_exit on a non-guest thread");
    run_key_destructors(t);
    let rsp = t.saved_rsp.load(Ordering::SeqCst);
    unsafe { tlg_guest_thread_exit(rsp, value) }
}

fn finish(t: &'static GuestThread, value: u64) {
    *t.exited.lock() = Some(value);
    t.exit_cv.notify_all();
    crate::debug!("thread '{}' exited with {:#x}", t.name(), value);
}

pub fn run_key_destructors(t: &'static GuestThread) {
    // POSIX: up to PTHREAD_DESTRUCTOR_ITERATIONS rounds.
    for _ in 0..4 {
        let mut any = false;
        for key in 0..PTHREAD_KEYS_MAX {
            let val = {
                let mut s = t.specific.lock();
                std::mem::replace(&mut s[key], 0)
            };
            if val == 0 {
                continue;
            }
            if let Some(dtor) = crate::hle::kernel::pthread::key_destructor(key) {
                any = true;
                unsafe {
                    crate::hle::call_guest(dtor, &[val]);
                }
            }
        }
        if !any {
            break;
        }
    }
}

/// Creates a host thread whose whole stack is committed up front. PS4 code does
/// not probe the stack, so lazily grown (guard-page) Windows stacks would fault
/// when a large frame is touched below the guard page.
pub fn spawn_host(name: &str, stack_size: usize, f: Box<dyn FnOnce() + Send>) -> std::io::Result<()> {
    use windows_sys::Win32::System::Threading::{CreateThread, SetThreadDescription};
    unsafe extern "system" fn proc_(param: *mut std::ffi::c_void) -> u32 {
        let f = unsafe { Box::from_raw(param as *mut Box<dyn FnOnce() + Send>) };
        f();
        0
    }
    let stack = (stack_size + 0xFFFF) & !0xFFFF;
    let param = Box::into_raw(Box::new(f)) as *mut std::ffi::c_void;
    let mut tid = 0u32;
    let h = unsafe { CreateThread(std::ptr::null(), stack, Some(proc_), param, 0, &mut tid) };
    if h.is_null() {
        drop(unsafe { Box::from_raw(param as *mut Box<dyn FnOnce() + Send>) });
        return Err(std::io::Error::last_os_error());
    }
    let wname: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
    unsafe {
        SetThreadDescription(h, wname.as_ptr());
        windows_sys::Win32::Foundation::CloseHandle(h);
    }
    Ok(())
}

/// Starts a host thread that runs a guest thread function.
pub fn spawn(t: &'static GuestThread) -> std::io::Result<()> {
    let host_stack = (t.stack_size as usize).max(256 << 10) + (1 << 20);
    spawn_host(
        &t.name(),
        host_stack,
        Box::new(move || {
            enter(t);
            crate::debug!("thread '{}' start entry={:#x} arg={:#x}", t.name(), t.entry, t.arg);
            let v = run_entry(t, t.entry, t.arg, 0);
            run_key_destructors(t);
            finish(t, v);
        }),
    )
}

pub fn join(t: &'static GuestThread) -> u64 {
    let mut g = t.exited.lock();
    while g.is_none() {
        t.exit_cv.wait(&mut g);
    }
    g.unwrap()
}
