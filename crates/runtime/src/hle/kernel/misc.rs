//! Process/module information, errno, stack protector, signals, sysctl, exit.

use super::errno::*;
use super::sce;
use crate::hle::{Registry, write_cstr};
use crate::loader::{self, LoadedModule};
use crate::{reg, sysv};
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};

static FALLBACK_ERRNO: AtomicI32 = AtomicI32::new(0);
pub static HEAP_API: AtomicU64 = AtomicU64::new(0);
pub static SDK_VERSION: AtomicU64 = AtomicU64::new(0);

/// SceKernelModuleInfoEx (0x1A8 bytes).
#[repr(C)]
struct ModuleInfoEx {
    size: u64,
    name: [u8; 256],
    id: i32,
    tls_index: u32,
    tls_init_addr: u64,
    tls_init_size: u32,
    tls_size: u32,
    tls_offset: u32,
    tls_align: u32,
    init_proc_addr: u64,
    fini_proc_addr: u64,
    reserved1: u64,
    reserved2: u64,
    eh_frame_hdr_addr: u64,
    eh_frame_addr: u64,
    eh_frame_hdr_size: u32,
    eh_frame_size: u32,
    segments: [(u64, u32, i32); 4],
    segment_count: u32,
    ref_count: u32,
}

/// SceKernelModuleInfo (0x160 bytes).
#[repr(C)]
struct ModuleInfo {
    size: u64,
    name: [u8; 256],
    segments: [(u64, u32, i32); 4],
    segment_count: u32,
    fingerprint: [u8; 20],
}

fn segments(m: &LoadedModule) -> ([(u64, u32, i32); 4], u32) {
    let mut s = [(0u64, 0u32, 0i32); 4];
    let mut n = 0;
    for ph in &m.phdrs {
        if ph.p_type == sce_elf::PT_LOAD && n < 4 {
            let mut prot = 0;
            if ph.p_flags & sce_elf::PF_R != 0 {
                prot |= 1;
            }
            if ph.p_flags & sce_elf::PF_W != 0 {
                prot |= 2;
            }
            if ph.p_flags & sce_elf::PF_X != 0 {
                prot |= 4;
            }
            s[n] = (m.base + ph.p_vaddr, crate::mem::align_up(ph.p_memsz, crate::mem::PAGE) as u32, prot);
            n += 1;
        }
    }
    (s, n as u32)
}

fn fill_info_ex(m: &LoadedModule, out: *mut ModuleInfoEx) {
    let mut info: ModuleInfoEx = unsafe { std::mem::zeroed() };
    info.size = std::mem::size_of::<ModuleInfoEx>() as u64;
    let n = m.name.len().min(255);
    info.name[..n].copy_from_slice(&m.name.as_bytes()[..n]);
    info.id = m.handle;
    if let Some(t) = m.tls {
        info.tls_index = t.index as u32;
        info.tls_init_addr = t.init_image;
        info.tls_init_size = t.init_size as u32;
        info.tls_size = t.mem_size as u32;
        info.tls_offset = t.offset as u32;
        info.tls_align = t.align as u32;
    }
    info.init_proc_addr = m.init.unwrap_or(0);
    info.fini_proc_addr = m.fini.unwrap_or(0);
    if let Some((a, s)) = m.eh_frame_hdr {
        info.eh_frame_hdr_addr = a;
        info.eh_frame_hdr_size = s as u32;
    }
    if let Some((a, s)) = m.eh_frame {
        info.eh_frame_addr = a;
        info.eh_frame_size = s as u32;
    }
    let (segs, n) = segments(m);
    info.segments = segs;
    info.segment_count = n;
    info.ref_count = 1;
    // Honor the caller-declared structure size.
    let want = unsafe { *(out as *const u64) } as usize;
    let n = want.min(std::mem::size_of::<ModuleInfoEx>());
    unsafe { std::ptr::copy_nonoverlapping(&info as *const ModuleInfoEx as *const u8, out as *mut u8, n) };
}

fn exit_process(code: i32) -> ! {
    crate::info!("guest requested process exit with status {code}");
    crate::shutdown(code);
}

extern "sysv64" fn rtld_exit_callback() {
    crate::info!("guest called the rtld exit callback");
}

pub fn rtld_exit_fn() -> u64 {
    rtld_exit_callback as usize as u64
}

sysv! {
    fn __error() -> *mut i32 {
        match crate::thread::current() {
            Some(t) => t.errno.as_ptr(),
            None => FALLBACK_ERRNO.as_ptr(),
        }
    }
    fn __tls_get_addr(ti: *const u64) -> u64 { crate::tls::get_addr(*ti, *ti.add(1)) }
    fn __stack_chk_fail() {
        crate::error!("guest stack protector failure (__stack_chk_fail)");
        crate::fatal_exit(0xBB10, "stack smashing detected");
    }
    fn sceKernelGetProcParam() -> u64 {
        loader::main_module().and_then(|m| m.proc_param).unwrap_or(0)
    }
    fn _sceKernelRtldSetApplicationHeapAPI(api: u64) -> i32 {
        HEAP_API.store(api, Ordering::SeqCst);
        crate::debug!("application heap API registered at {api:#x}");
        0
    }
    fn sceKernelGetModuleInfoFromAddr(addr: u64, flags: i32, info: *mut ModuleInfoEx) -> i32 {
        if info.is_null() { return sce(EFAULT); }
        match loader::module_at(addr) { Some(m) => { fill_info_ex(&m, info); 0 } None => sce(ESRCH) }
    }
    fn sceKernelGetModuleInfoForUnwind(addr: u64, flags: i32, info: *mut u8) -> i32 {
        if info.is_null() { return sce(EFAULT); }
        let Some(m) = loader::module_at(addr) else { return sce(ESRCH) };
        let size = *(info as *const u64);
        if size < 0x130 { return sce(EINVAL); }
        write_cstr(info.add(8), 256, &m.name);
        let p = info.add(8 + 256) as *mut u64;
        let (hdr, _) = m.eh_frame_hdr.unwrap_or((0, 0));
        let (ef, efs) = m.eh_frame.unwrap_or((0, 0));
        *p = hdr;
        *p.add(1) = ef;
        *p.add(2) = efs;
        *p.add(3) = m.code.0;
        *p.add(4) = m.code.1 - m.code.0;
        0
    }
    fn sceKernelGetModuleInfo(handle: i32, info: *mut ModuleInfo) -> i32 {
        let mods = loader::MODULES.read();
        let Some(m) = mods.iter().find(|m| m.handle == handle) else { return sce(ESRCH) };
        let mut mi: ModuleInfo = std::mem::zeroed();
        mi.size = std::mem::size_of::<ModuleInfo>() as u64;
        let n = m.name.len().min(255);
        mi.name[..n].copy_from_slice(&m.name.as_bytes()[..n]);
        let (segs, cnt) = segments(m);
        mi.segments = segs;
        mi.segment_count = cnt;
        mi.fingerprint = m.fingerprint;
        std::ptr::write_unaligned(info, mi);
        0
    }
    fn sceKernelGetModuleList(handles: *mut i32, num: u64, actual: *mut u64) -> i32 {
        let mods = loader::MODULES.read();
        let n = mods.len().min(num as usize);
        for (i, m) in mods.iter().take(n).enumerate() { *handles.add(i) = m.handle; }
        if !actual.is_null() { *actual = n as u64; }
        0
    }
    fn __elf_phdr_match_addr(info: *const u64, addr: u64) -> i32 {
        let base = *info;
        let phdr = *info.add(2) as *const u8;
        let phnum = *(info.add(3) as *const u16);
        for i in 0..phnum as usize {
            let p = phdr.add(i * 56);
            let ptype = *(p as *const u32);
            let vaddr = *(p.add(16) as *const u64);
            let memsz = *(p.add(40) as *const u64);
            if ptype == sce_elf::PT_LOAD && addr >= base + vaddr && addr < base + vaddr + memsz { return 1; }
        }
        0
    }
    fn __pthread_cxa_finalize(info: u64) {}
    fn _is_signal_return(addr: u64) -> i32 { 0 }
    fn sceKernelIsNeoMode() -> i32 { 0 }
    fn sceKernelHasNeoMode() -> i32 { 0 }
    fn sceKernelGetCompiledSdkVersion(ver: *mut u32) -> i32 {
        if ver.is_null() { return sce(EINVAL); }
        *ver = SDK_VERSION.load(Ordering::Relaxed) as u32;
        0
    }
    fn sceKernelGetCpumode() -> i32 { 0 }
    fn sceKernelDebugRaiseException(code: u32, param: u64) {
        crate::error!("guest sceKernelDebugRaiseException code={code:#x} param={param:#x}");
        crate::fatal_exit(0xBB11, "guest raised a debug exception");
    }
    fn sceKernelDebugRaiseExceptionOnReleaseMode(code: u32, param: u64) {
        crate::error!("guest sceKernelDebugRaiseExceptionOnReleaseMode code={code:#x} param={param:#x}");
        crate::fatal_exit(0xBB12, "guest raised a debug exception");
    }
    fn _exit(code: i32) { exit_process(code) }
    fn getpid() -> i32 { 0x42 }
    fn getppid() -> i32 { 1 }
    fn signal(sig: i32, handler: u64) -> u64 {
        crate::debug!("signal({sig}, {handler:#x}) ignored");
        0
    }
    fn sigaction(sig: i32, act: u64, old: *mut u8) -> i32 {
        if !old.is_null() { std::ptr::write_bytes(old, 0, 32); }
        0
    }
    fn sigfillset(set: *mut u32) -> i32 { if !set.is_null() { for i in 0..4 { *set.add(i) = !0; } } 0 }
    fn sigemptyset(set: *mut u32) -> i32 { if !set.is_null() { for i in 0..4 { *set.add(i) = 0; } } 0 }
    fn sigaddset(set: *mut u32, sig: i32) -> i32 { 0 }
    fn sigdelset(set: *mut u32, sig: i32) -> i32 { 0 }
    fn sigprocmask(how: i32, set: *const u32, old: *mut u32) -> i32 {
        if !old.is_null() { for i in 0..4 { *old.add(i) = 0; } }
        0
    }
    fn _sigprocmask(how: i32, set: *const u32, old: *mut u32) -> i32 {
        if !old.is_null() { for i in 0..4 { *old.add(i) = 0; } }
        0
    }
    fn getrusage(who: i32, ru: *mut i64) -> i32 {
        if ru.is_null() { return -1; }
        std::ptr::write_bytes(ru as *mut u8, 0, 144);
        let us = crate::hle::kernel::time::process_time_us();
        *ru = (us / 1_000_000) as i64;
        *ru.add(1) = (us % 1_000_000) as i64;
        0
    }
    fn sysctl(name: *const i32, namelen: u32, oldp: *mut u8, oldlenp: *mut u64, newp: *const u8, newlen: u64) -> i32 {
        let mib: Vec<i32> = (0..namelen as usize).map(|i| *name.add(i)).collect();
        match mib.as_slice() {
            [6, 7] => {
                // hw.pagesize
                if !oldp.is_null() { *(oldp as *mut i32) = crate::mem::PAGE as i32; }
                if !oldlenp.is_null() { *oldlenp = 4; }
                0
            }
            [1, 37] => {
                // kern.arandom
                if !oldp.is_null() && !oldlenp.is_null() {
                    let n = *oldlenp as usize;
                    let mut x = crate::hle::kernel::time::process_time_us() ^ 0x9E37_79B9_7F4A_7C15;
                    for i in 0..n {
                        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                        *oldp.add(i) = x as u8;
                    }
                }
                0
            }
            _ => {
                crate::warn!("sysctl {:?} not supported", mib);
                crate::hle::kernel::set_errno(ENOENT);
                -1
            }
        }
    }
    fn sceKernelGetSanitizerMallocReplaceExternal() -> u64 { 0 }
    fn sceKernelGetSanitizerNewReplaceExternal() -> u64 { 0 }
    fn sceKernelIsAddressSanitizerEnabled() -> i32 { 0 }
    fn sceKernelSetThreadDtors(f: u64) -> i32 { 0 }
    fn _sceKernelSetThreadDtors(f: u64) -> i32 { 0 }
    fn _sceKernelSetThreadAtexitCount(f: u64) -> i32 { 0 }
    fn _sceKernelSetThreadAtexitReport(f: u64) -> i32 { 0 }
    fn sceKernelGetFsSandboxRandomWord() -> u64 { crate::mem::arena_cstr("tlg0000") }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libkernel", [
        __error, __tls_get_addr, __stack_chk_fail, sceKernelGetProcParam, _sceKernelRtldSetApplicationHeapAPI,
        sceKernelGetModuleInfoFromAddr, sceKernelGetModuleInfoForUnwind, sceKernelGetModuleInfo,
        sceKernelGetModuleList, __elf_phdr_match_addr, __pthread_cxa_finalize, _is_signal_return, sceKernelIsNeoMode,
        sceKernelHasNeoMode, sceKernelGetCompiledSdkVersion, sceKernelGetCpumode, sceKernelDebugRaiseException,
        sceKernelDebugRaiseExceptionOnReleaseMode, _exit, getpid, getppid, signal, sigaction, sigfillset,
        sigemptyset, sigaddset, sigdelset, sigprocmask, _sigprocmask, getrusage, sysctl,
        sceKernelGetSanitizerMallocReplaceExternal, sceKernelGetSanitizerNewReplaceExternal,
        sceKernelIsAddressSanitizerEnabled, sceKernelSetThreadDtors, _sceKernelSetThreadDtors,
        _sceKernelSetThreadAtexitCount, _sceKernelSetThreadAtexitReport, sceKernelGetFsSandboxRandomWord,
    ]);
    // Data exports.
    let guard = crate::mem::arena_alloc(8, 8);
    unsafe { *(guard as *mut u64) = 0x2B99_2DDF_A232_49D6 ^ (crate::hle::kernel::time::process_time_us() << 17) };
    r.data("libkernel", "__stack_chk_guard", guard);
    let name = crate::mem::arena_cstr("eboot.bin");
    let progname = crate::mem::arena_alloc(8, 8);
    unsafe { *(progname as *mut u64) = name };
    r.data("libkernel", "__progname", progname);
    let env = crate::mem::arena_alloc(16, 8);
    let environ = crate::mem::arena_alloc(8, 8);
    unsafe { *(environ as *mut u64) = env };
    r.data("libkernel", "environ", environ);
    r.data("libkernel", "_environ", environ);
}
