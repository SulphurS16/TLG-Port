//! libSceFiber: user-mode fibers (cooperative contexts with their own stacks).
//!
//! A switch saves the callee-saved registers, MXCSR and the x87 control word on the
//! current stack, records the stack pointer and the TEB stack bounds in the outgoing
//! context, installs the incoming context's bounds (as Windows' own fibers do, so the
//! OS exception dispatcher accepts frames on fiber stacks) and resumes it. A fiber
//! that has never run starts in `tlg_fiber_start`, which calls its guest entry with
//! (argOnInitialize, argOnRun); an entry must not return (fatal, as on hardware).
//!
//! The SceFiber object (256 bytes, title-allocated) holds this module's state:
//! signatures, state, entry, arguments, context range, name, saved context, flags.

use crate::hle::{Registry, cstr};
use crate::{reg, sysv};
use std::cell::UnsafeCell;

const ERR_NULL: i32 = 0x8059_0001u32 as i32;
const ERR_ALIGNMENT: i32 = 0x8059_0002u32 as i32;
const ERR_RANGE: i32 = 0x8059_0003u32 as i32;
const ERR_INVALID: i32 = 0x8059_0004u32 as i32;
const ERR_PERMISSION: i32 = 0x8059_0005u32 as i32;
const ERR_STATE: i32 = 0x8059_0006u32 as i32;

const SIG0: u32 = 0xDEF1_649C;
const SIG1: u32 = 0xB375_92A0;
const OPT_SIG: u32 = 0xBB40_E64D;
const STACK_SIG: u64 = 0x7149_F2CA_7149_F2CA;
const CONTEXT_MIN: u64 = 512;
const FLAG_SET_FPU_REGS: u32 = 0x100;

const STATE_RUN: u32 = 1;
const STATE_IDLE: u32 = 2;
const STATE_TERMINATED: u32 = 3;

/// Saved execution context: stack pointer and the TEB stack bounds.
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct Ctx {
    rsp: u64,
    stack_base: u64,
    stack_limit: u64,
}

#[repr(C)]
struct Fiber {
    sig0: u32,
    state: u32,
    entry: u64,
    arg_init: u64,
    addr_context: u64,
    size_context: u64,
    name: [u8; 32],
    ctx: Ctx,
    flags: u32,
    started: u32,
    sig1: u32,
}
const _: () = assert!(std::mem::size_of::<Fiber>() <= 256);

/// Per-thread fiber state.
#[derive(Default)]
struct ThreadFibers {
    current: u64,
    thread_ctx: Ctx,
    /// argOnRunTo of the latest Run/Switch (read by the resumed side).
    arg_run_to: u64,
    /// argOnReturn of the latest ReturnToThread.
    arg_return: u64,
}

thread_local! {
    static TF: UnsafeCell<ThreadFibers> = UnsafeCell::new(ThreadFibers::default());
}

fn tf() -> *mut ThreadFibers {
    TF.with(|t| t.get())
}

std::arch::global_asm!(
    ".globl tlg_fiber_swap",
    "tlg_fiber_swap:", // rdi = outgoing Ctx*, rsi = incoming Ctx* (sysv64)
    "push rbp",
    "push rbx",
    "push r12",
    "push r13",
    "push r14",
    "push r15",
    "sub rsp, 8",
    "stmxcsr [rsp]",
    "fnstcw [rsp+4]",
    "mov [rdi], rsp",
    "mov rax, gs:[8]",
    "mov [rdi+8], rax",
    "mov rax, gs:[16]",
    "mov [rdi+16], rax",
    "mov rax, [rsi+8]",
    "mov gs:[8], rax",
    "mov rax, [rsi+16]",
    "mov gs:[16], rax",
    "mov rsp, [rsi]",
    "ldmxcsr [rsp]",
    "fldcw [rsp+4]",
    "add rsp, 8",
    "pop r15",
    "pop r14",
    "pop r13",
    "pop r12",
    "pop rbx",
    "pop rbp",
    "ret",
    ".globl tlg_fiber_start",
    "tlg_fiber_start:",
    "xor ebp, ebp",
    "push rbp",
    "call {main}",
    "ud2",
    main = sym fiber_main,
);

unsafe extern "sysv64" {
    fn tlg_fiber_swap(out: *mut Ctx, inc: *const Ctx);
    fn tlg_fiber_start();
}

/// Entry of a fiber that runs for the first time (on its own stack).
extern "sysv64" fn fiber_main() -> ! {
    let t = tf();
    let (entry, arg_init, arg_run, f) = unsafe {
        let f = (*t).current as *mut Fiber;
        ((*f).entry, (*f).arg_init, (*t).arg_run_to, f)
    };
    unsafe { crate::hle::call_guest(entry, &[arg_init, arg_run]) };
    let name = unsafe { name_of(f) };
    crate::error!("fiber '{name}' returned from its entry {}: fibers must switch away instead", crate::loader::describe_addr(entry));
    crate::fatal_exit(3, "fiber entry returned");
}

unsafe fn name_of(f: *const Fiber) -> String {
    let n = unsafe { &(*f).name };
    String::from_utf8_lossy(&n[..n.iter().position(|&c| c == 0).unwrap_or(32)]).into_owned()
}

unsafe fn valid(f: *const Fiber) -> Result<(), i32> {
    if f.is_null() {
        return Err(ERR_NULL);
    }
    if f as u64 & 7 != 0 {
        return Err(ERR_ALIGNMENT);
    }
    unsafe {
        if (*f).sig0 != SIG0 || (*f).sig1 != SIG1 {
            return Err(ERR_INVALID);
        }
    }
    Ok(())
}

/// Prepares the first switch into a fiber: an initial frame on its stack that
/// `tlg_fiber_swap` "returns" from into `tlg_fiber_start`.
unsafe fn prepare_start(f: *mut Fiber) -> Result<(), i32> {
    unsafe {
        if (*f).addr_context == 0 {
            crate::error!("fiber '{}' has no context of its own: run-time contexts are not implemented", name_of(f));
            return Err(ERR_INVALID);
        }
        let base = (*f).addr_context;
        let top = (base + (*f).size_context) & !15;
        *(base as *mut u64) = STACK_SIG;
        let (mut mxcsr, mut fcw): (u32, u16) = (0, 0);
        std::arch::asm!("stmxcsr [{0}]", "fnstcw [{1}]", in(reg) &mut mxcsr, in(reg) &mut fcw);
        if (*f).flags & FLAG_SET_FPU_REGS != 0 {
            mxcsr = 0x9FC0;
            fcw = 0x037F;
        }
        let w = |off: u64, v: u64| *((top - off) as *mut u64) = v;
        w(8, 0); // terminating return address
        w(16, tlg_fiber_start as usize as u64);
        for off in [24, 32, 40, 48, 56, 64] {
            w(off, 0); // rbp, rbx, r12..r15
        }
        w(72, mxcsr as u64 | (fcw as u64) << 32);
        (*f).ctx = Ctx { rsp: top - 72, stack_base: top, stack_limit: base };
        (*f).started = 1;
    }
    Ok(())
}

/// Verifies the stack-bottom signature of a fiber leaving the CPU.
unsafe fn check_overflow(f: *const Fiber) {
    unsafe {
        if (*f).addr_context != 0 && *((*f).addr_context as *const u64) != STACK_SIG {
            crate::error!("fiber '{}' overflowed its {:#x}-byte context", name_of(f), (*f).size_context);
            crate::fatal_exit(3, "fiber stack overflow");
        }
    }
}

sysv! {
    fn sceFiberOptParamInitialize(opt: *mut u32) -> i32 {
        if opt.is_null() { return ERR_NULL; }
        if opt as u64 & 7 != 0 { return ERR_ALIGNMENT; }
        std::ptr::write_bytes(opt as *mut u8, 0, 128);
        *opt = OPT_SIG;
        0
    }
    fn _sceFiberInitializeImpl(f: *mut Fiber, name: *const u8, entry: u64, arg_init: u64, addr_context: u64, size_context: u64, opt: *const u32, build: u32) -> i32 {
        if f.is_null() || name.is_null() || entry == 0 { return ERR_NULL; }
        if f as u64 & 7 != 0 || addr_context & 15 != 0 { return ERR_ALIGNMENT; }
        if !opt.is_null() && opt as u64 & 7 != 0 { return ERR_ALIGNMENT; }
        if size_context != 0 && size_context < CONTEXT_MIN { return ERR_RANGE; }
        if size_context & 15 != 0 || (addr_context == 0) != (size_context == 0) { return ERR_INVALID; }
        if !opt.is_null() && *opt != OPT_SIG { return ERR_INVALID; }
        std::ptr::write_bytes(f as *mut u8, 0, std::mem::size_of::<Fiber>());
        (*f).sig0 = SIG0;
        (*f).sig1 = SIG1;
        (*f).state = STATE_IDLE;
        (*f).entry = entry;
        (*f).arg_init = arg_init;
        (*f).addr_context = addr_context;
        (*f).size_context = size_context;
        crate::hle::write_cstr((*f).name.as_mut_ptr(), 32, cstr(name));
        crate::debug!("fiber '{}' at {f:p}: entry {}, context {addr_context:#x}+{size_context:#x}", cstr(name), crate::loader::describe_addr(entry));
        0
    }
    fn sceFiberFinalize(f: *mut Fiber) -> i32 {
        if let Err(e) = valid(f) { return e; }
        let st = std::sync::atomic::AtomicU32::from_ptr(&mut (*f).state);
        if st.compare_exchange(STATE_IDLE, STATE_TERMINATED, std::sync::atomic::Ordering::AcqRel, std::sync::atomic::Ordering::Acquire).is_err() { return ERR_STATE; }
        0
    }
    /// Runs `f` from thread context until it returns to the thread.
    fn sceFiberRun(f: *mut Fiber, arg_run_to: u64, arg_on_return: *mut u64) -> i32 {
        if let Err(e) = valid(f) { return e; }
        let t = tf();
        if (*t).current != 0 { return ERR_PERMISSION; }
        let st = std::sync::atomic::AtomicU32::from_ptr(&mut (*f).state);
        if st.compare_exchange(STATE_IDLE, STATE_RUN, std::sync::atomic::Ordering::AcqRel, std::sync::atomic::Ordering::Acquire).is_err() { return ERR_STATE; }
        if (*f).started == 0 {
            if let Err(e) = prepare_start(f) { st.store(STATE_IDLE, std::sync::atomic::Ordering::Release); return e; }
        }
        (*t).current = f as u64;
        (*t).arg_run_to = arg_run_to;
        tlg_fiber_swap(&mut (*t).thread_ctx, &(*f).ctx);
        // Back in thread context: the fiber returned to the thread.
        if !arg_on_return.is_null() { *arg_on_return = (*t).arg_return; }
        0
    }
    /// Switches from the running fiber to `f`.
    fn sceFiberSwitch(f: *mut Fiber, arg_run_to: u64, arg_on_run: *mut u64) -> i32 {
        if let Err(e) = valid(f) { return e; }
        let t = tf();
        let cur = (*t).current as *mut Fiber;
        if cur.is_null() { return ERR_PERMISSION; }
        let st = std::sync::atomic::AtomicU32::from_ptr(&mut (*f).state);
        if st.compare_exchange(STATE_IDLE, STATE_RUN, std::sync::atomic::Ordering::AcqRel, std::sync::atomic::Ordering::Acquire).is_err() { return ERR_STATE; }
        if (*f).started == 0 {
            if let Err(e) = prepare_start(f) { st.store(STATE_IDLE, std::sync::atomic::Ordering::Release); return e; }
        }
        check_overflow(cur);
        (*cur).state = STATE_IDLE;
        (*t).current = f as u64;
        (*t).arg_run_to = arg_run_to;
        tlg_fiber_swap(&mut (*cur).ctx, &(*f).ctx);
        // Resumed (possibly on another thread): read that thread's state.
        let t = tf();
        if !arg_on_run.is_null() { *arg_on_run = (*t).arg_run_to; }
        0
    }
    fn sceFiberReturnToThread(arg_on_return: u64, arg_on_run: *mut u64) -> i32 {
        let t = tf();
        let cur = (*t).current as *mut Fiber;
        if cur.is_null() { return ERR_PERMISSION; }
        check_overflow(cur);
        (*cur).state = STATE_IDLE;
        (*t).arg_return = arg_on_return;
        (*t).current = 0;
        tlg_fiber_swap(&mut (*cur).ctx, &(*t).thread_ctx);
        let t = tf();
        if !arg_on_run.is_null() { *arg_on_run = (*t).arg_run_to; }
        0
    }
    fn sceFiberGetSelf(out: *mut u64) -> i32 {
        if out.is_null() { return ERR_NULL; }
        let cur = (*tf()).current;
        if cur == 0 { return ERR_PERMISSION; }
        *out = cur;
        0
    }
    /// SceFiberInfo: {size, entry, argOnInitialize, addrContext, sizeContext, name[32], sizeContextMargin}
    fn sceFiberGetInfo(f: *mut Fiber, info: *mut u64) -> i32 {
        if let Err(e) = valid(f) { return e; }
        if info.is_null() { return ERR_NULL; }
        if *info != 128 { return ERR_INVALID; }
        *info.add(1) = (*f).entry;
        *info.add(2) = (*f).arg_init;
        *info.add(3) = (*f).addr_context;
        *info.add(4) = (*f).size_context;
        std::ptr::copy_nonoverlapping((*f).name.as_ptr(), info.add(5) as *mut u8, 32);
        *info.add(9) = u64::MAX; // context-size checking is not active
        0
    }
    fn sceFiberRename(f: *mut Fiber, name: *const u8) -> i32 {
        if let Err(e) = valid(f) { return e; }
        if name.is_null() { return ERR_NULL; }
        crate::hle::write_cstr((*f).name.as_mut_ptr(), 32, cstr(name));
        0
    }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceFiber", [
        sceFiberOptParamInitialize, _sceFiberInitializeImpl, sceFiberFinalize, sceFiberRun, sceFiberSwitch,
        sceFiberReturnToThread, sceFiberGetSelf, sceFiberGetInfo, sceFiberRename,
    ]);
}
