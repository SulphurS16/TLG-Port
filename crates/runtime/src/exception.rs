//! Vectored exception handling: fatal faults in guest code or in our own image
//! are reported with guest module offsets, registers and a frame-pointer
//! backtrace, then the process terminates. Exceptions in other DLLs are left to
//! their own SEH handlers.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, CONTEXT, EXCEPTION_POINTERS};

const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

static HOST_IMAGE: OnceLock<(u64, u64)> = OnceLock::new();
static IN_HANDLER: AtomicBool = AtomicBool::new(false);
/// Host thread id of the thread producing the fatal report.
static REPORTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// [start, end) of our own executable image.
pub fn host_image() -> (u64, u64) {
    *HOST_IMAGE.get_or_init(|| unsafe {
        let base = windows_sys::Win32::System::LibraryLoader::GetModuleHandleW(std::ptr::null()) as u64;
        let nt = base + *((base + 0x3C) as *const u32) as u64;
        let size = *((nt + 0x50) as *const u32) as u64;
        (base, base + size)
    })
}

pub fn install() {
    host_image();
    let h = unsafe { AddVectoredExceptionHandler(1, Some(veh)) };
    crate::info!("vectored exception handler installed: {:?}", h);
}

fn code_name(code: u32) -> &'static str {
    match code {
        0xC0000005 => "ACCESS_VIOLATION",
        0xC000001D => "ILLEGAL_INSTRUCTION",
        0xC0000096 => "PRIVILEGED_INSTRUCTION",
        0xC0000094 => "INTEGER_DIVIDE_BY_ZERO",
        0xC0000095 => "INTEGER_OVERFLOW",
        0x80000003 => "BREAKPOINT",
        0xC00000FD => "STACK_OVERFLOW",
        0xC000008C => "ARRAY_BOUNDS_EXCEEDED",
        0xC000008E => "FLOAT_DIVIDE_BY_ZERO",
        0xC0000090 => "FLOAT_INVALID_OPERATION",
        0xC0000409 => "STACK_BUFFER_OVERRUN",
        0x80000001 => "GUARD_PAGE",
        _ => "?",
    }
}

fn in_guest(addr: u64) -> bool {
    match crate::loader::MODULES.try_read() {
        Some(m) => m.iter().any(|x| x.contains(addr)),
        None => false,
    }
}

fn describe(addr: u64) -> String {
    if crate::loader::MODULES.try_read().is_some() && crate::loader::STUBS.try_lock().is_some() {
        crate::loader::describe_addr(addr)
    } else {
        format!("{addr:#x}")
    }
}

/// Debug breakpoints (`TLG_BREAK=<eboot offsets>`): int3 patched over the
/// first byte of a guest instruction; each hit is logged. A replaced
/// `push rbp` is emulated; any other instruction is executed by restoring its
/// byte and single-stepping, then the int3 is re-armed (hits on other threads
/// during that step are missed). (address, hit count, original byte)
pub static BREAKPOINTS: parking_lot::Mutex<Vec<(u64, u64, u8)>> = parking_lot::Mutex::new(Vec::new());

/// `TLG_BREAK_STATS=1`: hits are counted per (breakpoint, caller chain) instead
/// of being logged one by one (logging slows the title enough to change its
/// behavior); `dump_break_stats` prints and resets the table.
static BREAK_STATS: parking_lot::Mutex<Option<std::collections::HashMap<(u64, [u64; 4]), u64>>> = parking_lot::Mutex::new(None);

pub fn break_stats_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("TLG_BREAK_STATS").is_some())
}

/// Logs the breakpoint hit histogram collected since the last call.
pub fn dump_break_stats() {
    // try_lock: also called from the fatal-fault path.
    let Some(map) = BREAK_STATS.try_lock_for(std::time::Duration::from_millis(100)).and_then(|mut g| g.take()) else { return };
    let mut v: Vec<_> = map.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    let total: u64 = v.iter().map(|e| e.1).sum();
    let mut s = format!("breakpoint stats: {total} hits");
    for ((bp, chain), n) in v.iter().take(40) {
        s.push_str(&format!("
  {n:>7} {}", describe(*bp).replace("Shadow_Shipping", "")));
        for &a in chain.iter().filter(|&&a| a != 0) {
            s.push_str(&format!(" <{}", describe(a).replace("Shadow_Shipping", "")));
        }
    }
    crate::log::write(crate::log::Level::Info, format_args!("{s}"));
}

thread_local! {
    /// Breakpoint to re-arm after this thread's single step (0 = none).
    static REARM: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Data-watch page to write-protect again after this thread's single
    /// step, and the watched bytes' address and contents before the write.
    static WATCH_STEP: std::cell::Cell<(u64, u64, [u8; 16])> = const { std::cell::Cell::new((0, 0, [0; 16])) };
}

/// Data watchpoint (`TLG_DATA_WATCH=<hex addr>[+<hex len>]`, e.g. to find who
/// overwrites a buffer): the pages holding the range are write-protected
/// whenever they are mapped or re-protected; each write into the range is
/// logged with the writing instruction (guest module offset or host address)
/// and caller chain, the bytes before and after, then executed with the page
/// writable for one single step. Writes elsewhere on the pages are let
/// through silently. GPU writes through imported memory are not seen.
pub fn data_watch() -> Option<(u64, u64)> {
    static W: OnceLock<Option<(u64, u64)>> = OnceLock::new();
    *W.get_or_init(|| {
        let v = std::env::var("TLG_DATA_WATCH").ok()?;
        let (a, n) = v.split_once('+').unwrap_or((&v, "8"));
        let a = u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()?;
        let n = u64::from_str_radix(n.trim().trim_start_matches("0x"), 16).ok()?.max(1);
        crate::info!("data watch on {a:#x}+{n:#x}");
        Some((a, a + n))
    })
}

const WATCH_PAGE: u64 = 0x1000;

fn watch_pages() -> Option<(u64, u64)> {
    data_watch().map(|(lo, hi)| (lo & !(WATCH_PAGE - 1), (hi + WATCH_PAGE - 1) & !(WATCH_PAGE - 1)))
}

/// Write-protects the watched pages inside a guest range just committed or
/// re-protected with protection `prot` (no-op without a watch).
pub fn arm_data_watch(start: u64, end: u64, prot: u32) {
    let Some((plo, phi)) = watch_pages() else { return };
    let (lo, hi) = (plo.max(start), phi.min(end));
    if lo >= hi || prot & 2 == 0 {
        return;
    }
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let mut old = 0u32;
    let ok = unsafe { VirtualProtect(lo as *const _, (hi - lo) as usize, PAGE_READONLY, &mut old) };
    crate::info!("data watch armed on {lo:#x}-{hi:#x} (ok {ok})");
}

/// A write fault on a watched page: logs it when it hits the watched range,
/// makes the page writable and single-steps the instruction.
unsafe fn handle_watch_fault(ctx: &mut CONTEXT, addr: u64) -> bool {
    let Some((plo, phi)) = watch_pages() else { return false };
    if addr < plo || addr >= phi {
        return false;
    }
    let (lo, hi) = data_watch().unwrap();
    let page = addr & !(WATCH_PAGE - 1);
    let mut before = [0u8; 16];
    let hit = addr + 16 > lo && addr < hi;
    if hit {
        unsafe { std::ptr::copy_nonoverlapping((addr & !7) as *const u8, before.as_mut_ptr(), 16) };
        // Caller chain (frame pointers within guest modules or our image).
        let mut chain = String::new();
        let mut fp = ctx.Rbp;
        for _ in 0..4 {
            if fp < 0x10000 || fp % 8 != 0 {
                break;
            }
            let ret = unsafe { std::ptr::read_volatile((fp + 8) as *const u64) };
            if ret == 0 {
                break;
            }
            chain.push_str(&format!(" <{}", describe(ret)));
            fp = unsafe { std::ptr::read_volatile(fp as *const u64) };
        }
        crate::log::write(
            crate::log::Level::Warn,
            format_args!(
                "data watch: write to {addr:#x} by {} [{}]{chain} | before {:02x?} | rax {:#x} rcx {:#x} rdx {:#x} rsi {:#x} rdi {:#x} r8 {:#x}",
                describe(ctx.Rip),
                crate::log::thread_name(),
                before,
                ctx.Rax,
                ctx.Rcx,
                ctx.Rdx,
                ctx.Rsi,
                ctx.Rdi,
                ctx.R8
            ),
        );
    }
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let mut old = 0u32;
    unsafe { VirtualProtect(page as *const _, WATCH_PAGE as usize, PAGE_READWRITE, &mut old) };
    WATCH_STEP.with(|w| w.set((page, if hit { addr & !7 } else { 0 }, before)));
    ctx.EFlags |= 0x100;
    true
}

/// After a watched write's single step: logs the new bytes, protects again.
unsafe fn finish_watch_step(ctx: &mut CONTEXT) -> bool {
    let (page, at, _before) = WATCH_STEP.with(|w| w.replace((0, 0, [0; 16])));
    if page == 0 {
        return false;
    }
    if at != 0 {
        let mut after = [0u8; 16];
        unsafe { std::ptr::copy_nonoverlapping(at as *const u8, after.as_mut_ptr(), 16) };
        crate::log::write(crate::log::Level::Warn, format_args!("data watch: {at:#x} now {after:02x?}"));
    }
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let mut old = 0u32;
    unsafe { VirtualProtect(page as *const _, WATCH_PAGE as usize, PAGE_READONLY, &mut old) };
    ctx.EFlags &= !0x100;
    true
}

/// Hardware data watchpoint (control `hwwatch <addr> [len] [w]`, `hwwatch
/// off`): debug register 0 of every guest thread existing at the time watches
/// `len` bytes (1, 2, 4 or 8, aligned) for reads and writes (`w`: writes
/// only). The CPU traps after each access: hits are counted per trap address
/// (the instruction after the access) and caller, the first ones of each site
/// logged with the watched value; `hwwatch off` disarms and lists the sites.
/// Threads created later are not watched.
static HW_WATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static HW_HITS: parking_lot::Mutex<Option<std::collections::HashMap<(u64, u64), u64>>> = parking_lot::Mutex::new(None);

pub fn hw_watch(addr: u64, len: u64, write_only: bool) -> Result<(), String> {
    let len_bits = match len {
        1 => 0u64,
        2 => 1,
        4 => 3,
        8 => 2,
        _ => return Err(format!("hwwatch: length {len} (1, 2, 4 or 8)")),
    };
    if addr == 0 || addr % len != 0 {
        return Err(format!("hwwatch: {addr:#x} is not {len}-byte aligned"));
    }
    let dr7 = 1 | (if write_only { 1u64 } else { 3 }) << 16 | len_bits << 18;
    *HW_HITS.lock() = Some(Default::default());
    HW_WATCH.store(addr, Ordering::SeqCst);
    let _ = std::thread::Builder::new().name("hwwatch".into()).spawn(move || {
        let n = set_debug_regs(addr, dr7);
        crate::info!("hwwatch: {addr:#x}+{len} ({}) armed on {n} threads", if write_only { "writes" } else { "reads and writes" });
    });
    Ok(())
}

pub fn hw_watch_off() {
    let _ = std::thread::Builder::new().name("hwwatch".into()).spawn(|| {
        let n = set_debug_regs(0, 0);
        let addr = HW_WATCH.swap(0, Ordering::SeqCst);
        let mut v: Vec<_> = HW_HITS.lock().take().unwrap_or_default().into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        let mut s = format!("hwwatch {addr:#x}: disarmed on {n} threads; {} sites (trap address = the instruction after the access)", v.len());
        for ((rip, ret), c) in v.iter().take(40) {
            s.push_str(&format!("\n  {c:>8} after {} <{}", describe(*rip), describe(*ret)));
        }
        crate::info!("{s}");
    });
}

/// Sets debug registers 0 and 7 (slot 0 bits only) of every guest thread,
/// each suspended meanwhile; returns how many were set.
fn set_debug_regs(dr0: u64, dr7: u64) -> usize {
    use windows_sys::Win32::System::Diagnostics::Debug::{GetThreadContext, SetThreadContext};
    use windows_sys::Win32::System::Threading::{OpenThread, ResumeThread, SuspendThread, THREAD_GET_CONTEXT, THREAD_SET_CONTEXT, THREAD_SUSPEND_RESUME};
    #[repr(C, align(16))]
    struct Ctx(CONTEXT);
    const CONTEXT_DEBUG_REGISTERS_AMD64: u32 = 0x0010_0010;
    let tids: Vec<u32> = crate::thread::THREADS.lock().iter().map(|t| t.host_tid.load(Ordering::Relaxed) as u32).filter(|&t| t != 0).collect();
    let mut n = 0;
    for tid in tids {
        unsafe {
            let h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_SET_CONTEXT, 0, tid);
            if h.is_null() {
                continue;
            }
            if SuspendThread(h) != u32::MAX {
                let mut c: Ctx = std::mem::zeroed();
                c.0.ContextFlags = CONTEXT_DEBUG_REGISTERS_AMD64;
                if GetThreadContext(h, &mut c.0) != 0 {
                    c.0.Dr0 = dr0;
                    c.0.Dr7 = (c.0.Dr7 & !0xF_0003) | dr7;
                    c.0.Dr6 = 0;
                    if SetThreadContext(h, &c.0) != 0 {
                        n += 1;
                    }
                }
                ResumeThread(h);
            }
            windows_sys::Win32::Foundation::CloseHandle(h);
        }
    }
    n
}

/// Counts (and the first times logs) a hardware watch hit.
unsafe fn note_hw_watch(ctx: &CONTEXT) {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, MEM_COMMIT, VirtualQuery};
    let fp = ctx.Rbp;
    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let ret = if fp >= 0x10000 && fp % 8 == 0 && unsafe { VirtualQuery((fp + 8) as *const _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } != 0 && mbi.State == MEM_COMMIT {
        unsafe { std::ptr::read_volatile((fp + 8) as *const u64) }
    } else {
        0
    };
    let n = {
        let mut g = HW_HITS.lock();
        let c = g.get_or_insert_with(Default::default).entry((ctx.Rip, ret)).or_insert(0);
        *c += 1;
        *c
    };
    if n <= 2 {
        let a = HW_WATCH.load(Ordering::Relaxed);
        let v = if a != 0 { unsafe { std::ptr::read_volatile((a & !7) as *const u64) } } else { 0 };
        crate::log::write(
            crate::log::Level::Info,
            format_args!("hwwatch hit #{n} after {} <{} [{}]: [{:#x}] {v:#018x}", describe(ctx.Rip), describe(ret), crate::log::thread_name(), a & !7),
        );
    }
}

/// Writes one code byte. The page is left writable: restoring its protection
/// would race with other threads patching the same page.
unsafe fn patch_code_byte(addr: u64, byte: u8) {
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    static LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());
    let _g = LOCK.lock();
    unsafe {
        let mut old = 0u32;
        if VirtualProtect(addr as *const _, 1, PAGE_EXECUTE_READWRITE, &mut old) == 0 {
            return;
        }
        std::ptr::write_volatile(addr as *mut u8, byte);
        windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache(windows_sys::Win32::System::Threading::GetCurrentProcess(), addr as *const _, 1);
    }
}
/// Remaining breakpoint hits to log unconditionally (control `breaklog N`).
pub static BREAK_VERBOSE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Expressions logged with each breakpoint hit (control `breakpeek`;
/// overrides TLG_BREAK_PEEK).
pub static BREAK_PEEK: parking_lot::Mutex<Option<String>> = parking_lot::Mutex::new(None);

/// Installs a logging breakpoint at `addr`, which must be the first byte of
/// a guest instruction.
pub fn install_breakpoint(addr: u64, module_base: u64) -> Result<(), String> {
    let orig = unsafe { *(addr as *const u8) };
    if orig == 0xCC {
        return Err(format!("{addr:#x} already holds int3"));
    }
    let back = (addr - module_base).min(0x80) as usize;
    // SAFETY: the module's code is mapped from its base to past `addr`.
    let code = unsafe { std::slice::from_raw_parts((addr - back as u64) as *const u8, back + 16) };
    if !instruction_start(code, addr - back as u64, addr) {
        return Err(format!("{addr:#x} is not the start of an instruction (an int3 there would change the instruction instead of trapping)"));
    }
    BREAKPOINTS.lock().push((addr, 0, orig));
    unsafe { patch_code_byte(addr, 0xCC) };
    if unsafe { *(addr as *const u8) } != 0xCC {
        return Err("could not patch code".into());
    }
    Ok(())
}

/// Whether `addr` starts an instruction in `code` (mapped at `base`): decoding
/// from several earlier points must land on it for most of them (x86
/// decoding resynchronizes within a few instructions).
pub fn instruction_start(code: &[u8], base: u64, addr: u64) -> bool {
    use iced_x86::{Decoder, DecoderOptions};
    let off = (addr - base) as usize;
    let backs: Vec<usize> = [0x80usize, 0x70, 0x61, 0x50].into_iter().filter(|&b| b <= off).collect();
    if backs.is_empty() {
        return true;
    }
    let hits = backs
        .iter()
        .filter(|&&b| {
            let mut d = Decoder::with_ip(64, &code[off - b..], addr - b as u64, DecoderOptions::NONE);
            while d.can_decode() && d.ip() < addr {
                let _ = d.decode();
            }
            d.ip() == addr
        })
        .count();
    hits * 4 >= backs.len() * 3
}

/// Evaluates a peek expression `base(+hex|*)*` against register values;
/// `*` loads a qword (when readable).
unsafe fn eval_peek(expr: &str, regs: &[(&str, u64)]) -> Option<u64> {
    let readable = |a: u64| -> bool {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery};
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { VirtualQuery(a as *const _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
            return false;
        }
        mbi.State == MEM_COMMIT && mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0 && mbi.Protect != 0 && a + 8 <= mbi.BaseAddress as u64 + mbi.RegionSize as u64
    };
    let base_end = expr.find(['+', '-', '*']).unwrap_or(expr.len());
    let base = &expr[..base_end];
    let mut v = regs.iter().find(|(n, _)| *n == base).map(|(_, v)| *v).or_else(|| u64::from_str_radix(base.trim_start_matches("0x"), 16).ok())?;
    let mut rest = &expr[base_end..];
    while !rest.is_empty() {
        if let Some(r) = rest.strip_prefix('*') {
            if !readable(v & !7) {
                return None;
            }
            v = unsafe { *((v & !7) as *const u64) };
            rest = r;
        } else if let Some(r) = rest.strip_prefix('+') {
            let end = r.find(['+', '-', '*']).unwrap_or(r.len());
            v = v.wrapping_add(u64::from_str_radix(&r[..end], 16).ok()?);
            rest = &r[end..];
        } else if let Some(r) = rest.strip_prefix('-') {
            let end = r.find(['+', '-', '*']).unwrap_or(r.len());
            v = v.wrapping_sub(u64::from_str_radix(&r[..end], 16).ok()?);
            rest = &r[end..];
        } else {
            return None;
        }
    }
    Some(v)
}

unsafe fn handle_breakpoint(ctx: &mut CONTEXT) -> bool {
    unsafe {
        let rip = ctx.Rip;
        let hit = {
            let mut bps = BREAKPOINTS.lock();
            bps.iter_mut().find(|b| b.0 == rip || b.0 + 1 == rip).map(|b| {
                b.1 += 1;
                (b.0, b.1, b.2)
            })
        };
        let Some((bp, n, orig)) = hit else { return false };
        if *(bp as *const u8) != 0xCC && rip == bp {
            // Restored by another thread's step: just execute the instruction.
            return true;
        }
        if break_stats_enabled() {
            // Return address (meaningful at function entries) and rbp chain.
            let mut key = [0u64; 4];
            key[0] = *(ctx.Rsp as *const u64);
            let depth: usize = std::env::var("TLG_BREAK_DEPTH").ok().and_then(|v| v.parse().ok()).unwrap_or(0).min(3);
            let mut fp = ctx.Rbp;
            for slot in key.iter_mut().skip(1).take(depth) {
                use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, MEM_COMMIT, VirtualQuery};
                let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                if fp < 0x1000 || fp & 7 != 0 || VirtualQuery(fp as *const _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) == 0 || mbi.State != MEM_COMMIT {
                    break;
                }
                *slot = *((fp + 8) as *const u64);
                fp = *(fp as *const u64);
            }
            *BREAK_STATS.lock().get_or_insert_with(Default::default).entry((bp, key)).or_insert(0) += 1;
        }
        let verbose = BREAK_VERBOSE.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_sub(1)).is_ok();
        if !break_stats_enabled() && (n <= 64 || n.is_power_of_two() || verbose) {
            let ret = *(ctx.Rsp as *const u64);
            // TLG_BREAK_PEEK: comma-separated expressions logged with each hit.
            let mut extra = String::new();
            // TLG_BREAK_DEPTH: caller frames (rbp chain) appended to each hit.
            let depth: usize = std::env::var("TLG_BREAK_DEPTH").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            if depth > 0 {
                let mut bp_ = ctx.Rbp;
                extra.push_str(" chain");
                for _ in 0..depth {
                    if bp_ < 0x1000 || bp_ & 7 != 0 {
                        break;
                    }
                    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, MEM_COMMIT, VirtualQuery};
                    let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                    if VirtualQuery(bp_ as *const _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) == 0 || mbi.State != MEM_COMMIT {
                        break;
                    }
                    let r = *((bp_ + 8) as *const u64);
                    extra.push_str(&format!(" <{}", describe(r).replace("Shadow_Shipping", "")));
                    bp_ = *(bp_ as *const u64);
                }
            }
            if let Some(spec) = BREAK_PEEK.lock().clone().or_else(|| std::env::var("TLG_BREAK_PEEK").ok()) {
                let regs = [
                    ("rdi", ctx.Rdi),
                    ("rsi", ctx.Rsi),
                    ("rdx", ctx.Rdx),
                    ("rcx", ctx.Rcx),
                    ("r8", ctx.R8),
                    ("r9", ctx.R9),
                    ("rsp", ctx.Rsp),
                    ("rax", ctx.Rax),
                    ("rbx", ctx.Rbx),
                    ("rbp", ctx.Rbp),
                    ("r12", ctx.R12),
                    ("r13", ctx.R13),
                    ("r14", ctx.R14),
                    ("r15", ctx.R15),
                ];
                for e in spec.split(',').map(str::trim).filter(|e| !e.is_empty()) {
                    match eval_peek(e, &regs) {
                        Some(v) => extra.push_str(&format!(" {e}={v:#x}")),
                        None => extra.push_str(&format!(" {e}=?")),
                    }
                }
            }
            crate::log::write(
                crate::log::Level::Info,
                format_args!(
                    "break {} #{n}: rdi={:#x} rsi={:#x} rdx={:#x} rcx={:#x} r8={:#x} r9={:#x} ret {}{}",
                    describe(bp),
                    ctx.Rdi,
                    ctx.Rsi,
                    ctx.Rdx,
                    ctx.Rcx,
                    ctx.R8,
                    ctx.R9,
                    describe(ret),
                    extra
                ),
            );
        }
        if orig == 0x55 {
            // Emulate the replaced `push rbp`.
            ctx.Rsp -= 8;
            *(ctx.Rsp as *mut u64) = ctx.Rbp;
            ctx.Rip = bp + 1;
        } else {
            // Execute the original instruction with the trap flag set; the
            // single-step exception re-arms the breakpoint.
            patch_code_byte(bp, orig);
            ctx.Rip = bp;
            ctx.EFlags |= 0x100;
            REARM.with(|r| r.set(bp));
        }
        true
    }
}

/// Single-step after a non-`push rbp` breakpoint: re-arms it.
unsafe fn handle_single_step(ctx: &mut CONTEXT) -> bool {
    let bp = REARM.with(|r| r.replace(0));
    if bp == 0 {
        return false;
    }
    unsafe { patch_code_byte(bp, 0xCC) };
    ctx.EFlags &= !0x100;
    true
}

unsafe extern "system" fn veh(info: *mut EXCEPTION_POINTERS) -> i32 {
    unsafe {
        let rec = &*(*info).ExceptionRecord;
        let code = rec.ExceptionCode as u32;
        if code == 0x80000003 && handle_breakpoint(&mut *(*info).ContextRecord) {
            return -1; // EXCEPTION_CONTINUE_EXECUTION
        }
        if code == 0x80000004 {
            let ctx = &mut *(*info).ContextRecord;
            // Hardware watch: Dr6.B0, or (should the context lack the debug
            // registers) any single step no other mechanism claims.
            let armed = HW_WATCH.load(Ordering::Relaxed) != 0;
            let hw = armed && ctx.Dr6 & 1 != 0;
            if hw {
                note_hw_watch(ctx);
                ctx.Dr6 = 0;
            }
            if finish_watch_step(ctx) || handle_single_step(ctx) {
                return -1;
            }
            if armed {
                if !hw {
                    note_hw_watch(ctx);
                }
                return -1;
            }
        }
        if code == 0xC000001D && crate::sse4a::handle_trap(&mut *(*info).ContextRecord) {
            return -1;
        }
        if code == 0xC0000005 && rec.NumberParameters >= 2 && rec.ExceptionInformation[0] == 1 && handle_watch_fault(&mut *(*info).ContextRecord, rec.ExceptionInformation[1] as u64) {
            return -1;
        }
        if code == 0xC0000005 && rec.NumberParameters >= 2 {
            let rip = (*(*info).ContextRecord).Rip;
            if crate::gpu::lazywb::fault(rec.ExceptionInformation[1] as u64, rec.ExceptionInformation[0] == 8) {
                return -1;
            }
            if rec.ExceptionInformation[0] == 1 && crate::gpu::cpuwatch::fault(rec.ExceptionInformation[1] as u64) {
                return -1;
            }
            if crate::gpu::wbprobe::fault(rec.ExceptionInformation[1] as u64, rec.ExceptionInformation[0] == 1, rip, || format!("{} [{}]", describe(rip), crate::log::thread_name())) {
                return -1;
            }
        }
        let ctx = &*(*info).ContextRecord;
        let interesting = matches!(
            code,
            0xC0000005
                | 0xC000001D
                | 0xC0000096
                | 0xC0000094
                | 0xC0000095
                | 0x80000003
                | 0xC00000FD
                | 0xC000008C
                | 0xC000008E
                | 0xC0000090
                | 0xC0000409
        );
        if !interesting {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let rip = ctx.Rip;
        let host = host_image();
        let in_host = rip >= host.0 && rip < host.1;
        let stub = crate::loader::STUBS.try_lock().and_then(|s| s.describe(rip)).is_some();
        if !in_guest(rip) && !in_host && !stub {
            // Elsewhere (system or driver DLLs): passed on, but noted (a few
            // times) so a process that then dies leaves a trace.
            static NOTED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if NOTED.fetch_add(1, Ordering::Relaxed) < 16 {
                use windows_sys::Win32::System::LibraryLoader::{GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, GetModuleFileNameW, GetModuleHandleExW};
                let mut m = std::ptr::null_mut();
                let mut name = String::from("?");
                let mut base = 0u64;
                if GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, rip as *const u16, &mut m) != 0 {
                    let mut buf = [0u16; 260];
                    let n = GetModuleFileNameW(m, buf.as_mut_ptr(), buf.len() as u32) as usize;
                    name = String::from_utf16_lossy(&buf[..n]).rsplit('\\').next().unwrap_or("?").to_string();
                    base = m as u64;
                }
                let what = if code == 0xC0000005 && rec.NumberParameters >= 2 { format!(" ({} of {:#x})", if rec.ExceptionInformation[0] == 1 { "write" } else { "read" }, rec.ExceptionInformation[1]) } else { String::new() };
                crate::log::write(
                    crate::log::Level::Warn,
                    format_args!("first-chance exception {:#010x} at {name}+{:#x}{what} [{}] (passed on)", code, rip.wrapping_sub(base), crate::log::thread_name()),
                );
            }
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let me = windows_sys::Win32::System::Threading::GetCurrentThreadId();
        if IN_HANDLER.swap(true, Ordering::SeqCst) {
            if REPORTER.load(Ordering::SeqCst) == me {
                // A fault inside the report itself: stop immediately.
                windows_sys::Win32::System::Threading::TerminateProcess(
                    windows_sys::Win32::System::Threading::GetCurrentProcess(),
                    0xBB0F,
                );
            }
            // Another thread is already reporting a fatal fault: note this one
            // and park until that report terminates the process.
            crate::log::write(
                crate::log::Level::Error,
                format_args!("concurrent fatal fault {:#010x} at {} [{}]", code, describe(rip), crate::log::thread_name()),
            );
            loop {
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        }
        REPORTER.store(me, Ordering::SeqCst);
        if break_stats_enabled() {
            dump_break_stats();
        }
        let mut r = String::new();
        use std::fmt::Write;
        let _ = writeln!(r, "FATAL {} ({:#010x}) at {} [{}]", code_name(code), code, describe(rip), crate::log::thread_name());
        if code == 0xC0000005 && rec.NumberParameters >= 2 {
            let kind = match rec.ExceptionInformation[0] {
                0 => "read",
                1 => "write",
                8 => "execute",
                _ => "?",
            };
            let addr = rec.ExceptionInformation[1] as u64;
            let _ = writeln!(r, "  {kind} of {:#x} ({})", addr, describe(addr));
        }
        let _ = writeln!(
            r,
            "  rax={:#018x} rbx={:#018x} rcx={:#018x} rdx={:#018x}\n  rsi={:#018x} rdi={:#018x} rbp={:#018x} rsp={:#018x}\n  r8 ={:#018x} r9 ={:#018x} r10={:#018x} r11={:#018x}\n  r12={:#018x} r13={:#018x} r14={:#018x} r15={:#018x} efl={:#x}",
            ctx.Rax, ctx.Rbx, ctx.Rcx, ctx.Rdx, ctx.Rsi, ctx.Rdi, ctx.Rbp, ctx.Rsp, ctx.R8, ctx.R9, ctx.R10, ctx.R11, ctx.R12,
            ctx.R13, ctx.R14, ctx.R15, ctx.EFlags
        );
        // Memory at registers that point to readable memory (checked with
        // VirtualQuery: the handler must not take runtime locks).
        {
            use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_NOACCESS, PAGE_GUARD, VirtualQuery};
            let readable = |a: u64| -> bool {
                let mut mbi: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                if VirtualQuery(a as *const _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) == 0 {
                    return false;
                }
                let end = mbi.BaseAddress as u64 + mbi.RegionSize as u64;
                mbi.State == MEM_COMMIT && mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0 && mbi.Protect != 0 && a + 64 <= end
            };
            let regs = [
                ("rax", ctx.Rax),
                ("rbx", ctx.Rbx),
                ("rcx", ctx.Rcx),
                ("rdx", ctx.Rdx),
                ("rsi", ctx.Rsi),
                ("rdi", ctx.Rdi),
                ("r8", ctx.R8),
                ("r9", ctx.R9),
                ("r10", ctx.R10),
                ("r11", ctx.R11),
                ("r12", ctx.R12),
                ("r13", ctx.R13),
                ("r14", ctx.R14),
                ("r15", ctx.R15),
            ];
            for (name, v) in regs {
                let a = v & !7;
                if crate::mem::is_guest(a, 64) && readable(a) {
                    let q: Vec<String> = (0..8).map(|i| format!("{:016x}", *((a + i * 8) as *const u64))).collect();
                    let _ = writeln!(r, "  [{name}] {a:#x}: {}", q.join(" "));
                }
            }
            // TLG_CRASH_PEEK: comma-separated `base(+hex|*)*[:count]` expressions,
            // e.g. `rbx+b4c8*:8` (load the pointer at rbx+0xb4c8, dump 8 qwords).
            if let Ok(spec) = std::env::var("TLG_CRASH_PEEK") {
                for item in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                    let (expr, count) = match item.split_once(':') {
                        Some((e, c)) => (e, u64::from_str_radix(c, 16).unwrap_or(8).min(64)),
                        None => (item, 8),
                    };
                    let mut chars = expr;
                    let base_end = chars.find(['+', '*']).unwrap_or(chars.len());
                    let base = &chars[..base_end];
                    let mut v = regs.iter().find(|(n, _)| *n == base).map(|(_, v)| *v).or_else(|| u64::from_str_radix(base.trim_start_matches("0x"), 16).ok());
                    chars = &chars[base_end..];
                    while let (Some(cur), false) = (v, chars.is_empty()) {
                        if let Some(rest) = chars.strip_prefix('*') {
                            v = if readable(cur & !7) { Some(*((cur & !7) as *const u64)) } else { None };
                            chars = rest;
                        } else if let Some(rest) = chars.strip_prefix('+') {
                            let end = rest.find(['+', '*']).unwrap_or(rest.len());
                            v = u64::from_str_radix(&rest[..end], 16).ok().map(|o| cur.wrapping_add(o));
                            chars = &rest[end..];
                        } else {
                            v = None;
                        }
                    }
                    match v {
                        Some(a) if readable(a & !7) => {
                            let a = a & !7;
                            let q: Vec<String> = (0..count).map(|i| format!("{:016x}", *((a + i * 8) as *const u64))).collect();
                            let _ = writeln!(r, "  peek {item} = {a:#x}: {}", q.join(" "));
                        }
                        _ => {
                            let _ = writeln!(r, "  peek {item}: unreadable");
                        }
                    }
                }
            }
        }
        if code != 0xC00000FD {
            let mut lo = 0usize;
            let mut hi = 0usize;
            windows_sys::Win32::System::Threading::GetCurrentThreadStackLimits(&mut lo, &mut hi);
            let (lo, hi) = (lo as u64, hi as u64);
            // Frame-pointer chain.
            let _ = writeln!(r, "  backtrace (rbp chain):");
            let mut bp = ctx.Rbp;
            for i in 0..32 {
                if bp < lo || bp + 16 > hi || bp & 7 != 0 {
                    break;
                }
                let ret = *((bp + 8) as *const u64);
                let _ = writeln!(r, "    #{i:<2} {}", describe(ret));
                bp = *(bp as *const u64);
            }
            // Stack words that point into guest code.
            let _ = writeln!(r, "  stack scan (guest return candidates):");
            let mut sp = ctx.Rsp & !7;
            let mut n = 0;
            while sp + 8 <= hi && n < 24 && sp < ctx.Rsp + 0x2000 {
                let v = *(sp as *const u64);
                if in_guest(v) {
                    let _ = writeln!(r, "    [rsp+{:#06x}] {}", sp - ctx.Rsp, describe(v));
                    n += 1;
                }
                sp += 8;
            }
        }
        crate::fatal_report(&r);
        crate::fatal_exit(0xBB01, "fatal exception");
    }
}

#[cfg(test)]
mod tests {
    use super::instruction_start;

    #[test]
    fn breakpoints_only_at_instruction_starts() {
        // 16 x `mov qword ptr [rbp-38h], 0` (8 bytes), then
        // `mov dword ptr [rbp-48h], 3D088889h` (whose disp8 byte 0xB8 starts
        // `mov eax, imm32` when decoded on its own), `mov rax, [r13+0C8h]`.
        let mut code = Vec::new();
        for _ in 0..16 {
            code.extend([0x48, 0xC7, 0x45, 0xC8, 0, 0, 0, 0]);
        }
        let site = code.len() as u64;
        code.extend([0xC7, 0x45, 0xB8, 0x89, 0x88, 0x08, 0x3D]);
        code.extend([0x49, 0x8B, 0x85, 0xC8, 0, 0, 0]);
        code.extend([0xC3; 16]);
        let base = 0x8_0000_0000u64;
        assert!(instruction_start(&code, base, base + site));
        assert!(instruction_start(&code, base, base + site + 7));
        assert!(!instruction_start(&code, base, base + site + 1));
        assert!(!instruction_start(&code, base, base + site + 2));
        // Too close to the start of the code to check: allowed.
        assert!(instruction_start(&code, base, base + 0x10));
    }
}
