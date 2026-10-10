//! Direct/flexible memory, mmap family and memory queries.

use super::errno::*;
use super::{posix_ret, sce};
use crate::hle::{Registry, cstr};
use crate::mem::{self, MEM, MemState, PAGE, VmaKind};
use crate::{reg, sysv};

const MAP_FIXED: i32 = 0x10;
const MAP_NO_OVERWRITE: i32 = 0x80;
const MAP_ANON: i32 = 0x1000;
const MAP_FAILED: u64 = u64::MAX;

const FLEXIBLE_TOTAL: u64 = 448 << 20;

/// Picks a virtual address for a new mapping following PS4 hint/fixed rules.
fn choose_addr(st: &MemState, hint: u64, size: u64, align: u64, flags: i32) -> Result<u64, i32> {
    let align = align.max(PAGE);
    if flags & MAP_FIXED != 0 {
        if hint % PAGE != 0 || hint < mem::GUEST_BASE || hint + size > mem::GUEST_END {
            return Err(EINVAL);
        }
        if flags & MAP_NO_OVERWRITE != 0 {
            let mut a = hint;
            while a < hint + size {
                if let Some(v) = st.find(a) {
                    if v.kind != VmaKind::Reserved {
                        return Err(ENOMEM);
                    }
                    a = v.end();
                } else {
                    a += PAGE;
                }
            }
        }
        return Ok(hint);
    }
    let start = if hint >= mem::GUEST_BASE { hint } else { mem::MMAP_BASE };
    st.find_free(start, size, align, mem::GUEST_END).ok_or(ENOMEM)
}

fn map_common(addr_io: *mut u64, len: u64, prot: i32, flags: i32, kind: VmaKind, phys: u64, align: u64, name: &str) -> i32 {
    if addr_io.is_null() || len == 0 {
        return sce(EINVAL);
    }
    let len = mem::align_up(len, PAGE);
    let hint = unsafe { *addr_io };
    let mut st = MEM.lock();
    let addr = match choose_addr(&st, hint, len, align, flags) {
        Ok(a) => a,
        Err(e) => return sce(e),
    };
    match st.map(addr, len, kind, prot as u32, phys, 0, name) {
        Ok(a) => {
            unsafe { *addr_io = a };
            crate::debug!("map {:?} {:#x}+{:#x} prot={:#x} flags={:#x} phys={:#x} '{}'", kind, a, len, prot, flags, phys, name);
            0
        }
        Err(e) => {
            crate::error!("map {:?} at {:#x}+{:#x} failed: {:?}", kind, addr, len, e);
            sce(ENOMEM)
        }
    }
}

fn flexible(addr_io: *mut u64, len: u64, prot: i32, flags: i32, name: &str) -> i32 {
    {
        let st = MEM.lock();
        if st.flexible_used + mem::align_up(len, PAGE) > FLEXIBLE_TOTAL {
            crate::warn!("flexible memory exhausted ({:#x} used, {:#x} requested)", st.flexible_used, len);
            return sce(ENOMEM);
        }
    }
    map_common(addr_io, len, prot, flags, VmaKind::Flexible, 0, 0, name)
}

#[repr(C)]
struct BatchEntry {
    start: u64,
    offset: u64,
    length: u64,
    protection: u8,
    mtype: u8,
    reserved: u16,
    operation: i32,
}

fn batch_map(entries: *const BatchEntry, n: i32, out: *mut i32, flags: i32) -> i32 {
    let mut done = 0;
    let mut result = 0;
    for i in 0..n.max(0) as usize {
        let e = unsafe { &*entries.add(i) };
        let mut a = e.start;
        let r = match e.operation {
            0 => map_common(&mut a, e.length, e.protection as i32, flags | MAP_FIXED, VmaKind::Direct, e.offset, 0, "batch"),
            1 => {
                MEM.lock().unmap_range(e.start, e.start + e.length);
                0
            }
            2 => match MEM.lock().protect(e.start, e.start + e.length, e.protection as u32) {
                Ok(()) => 0,
                Err(_) => sce(EINVAL),
            },
            3 => flexible(&mut a, e.length, e.protection as i32, flags | MAP_FIXED, "batch-flex"),
            4 => match MEM.lock().protect(e.start, e.start + e.length, e.protection as u32) {
                Ok(()) => 0,
                Err(_) => sce(EINVAL),
            },
            op => {
                crate::error!("sceKernelBatchMap: unsupported operation {op}");
                sce(EINVAL)
            }
        };
        if r != 0 {
            result = r;
            break;
        }
        done += 1;
    }
    if !out.is_null() {
        unsafe { *out = done };
    }
    result
}

#[repr(C)]
struct VirtualQueryInfo {
    start: u64,
    end: u64,
    offset: u64,
    protection: i32,
    memory_type: i32,
    bits: u32,
    name: [u8; 32],
}

sysv! {
    fn sceKernelGetDirectMemorySize() -> u64 { mem::DMEM_SIZE }
    fn sceKernelAllocateDirectMemory(start: i64, end: i64, len: u64, align: u64, mtype: i32, out: *mut i64) -> i32 {
        if out.is_null() || len == 0 || len % PAGE != 0 || (align != 0 && !align.is_power_of_two()) { return sce(EINVAL); }
        match MEM.lock().dmem_allocate(start.max(0) as u64, end.max(0) as u64, len, align, mtype) {
            Some(p) => {
                *out = p as i64;
                crate::debug!("dmem alloc {:#x}+{:#x} align {:#x} type {}", p, len, align, mtype);
                0
            }
            None => {
                crate::warn!("direct memory allocation failed: {:#x} bytes in [{:#x}, {:#x})", len, start, end);
                sce(EAGAIN)
            }
        }
    }
    fn sceKernelAllocateMainDirectMemory(len: u64, align: u64, mtype: i32, out: *mut i64) -> i32 {
        sceKernelAllocateDirectMemory(0, mem::DMEM_SIZE as i64, len, align, mtype, out)
    }
    fn sceKernelReleaseDirectMemory(start: i64, len: u64) -> i32 {
        MEM.lock().dmem_release(start as u64, len);
        0
    }
    fn sceKernelCheckedReleaseDirectMemory(start: i64, len: u64) -> i32 {
        MEM.lock().dmem_release(start as u64, len);
        0
    }
    fn sceKernelAvailableDirectMemorySize(start: i64, end: i64, align: u64, out_start: *mut i64, out_size: *mut u64) -> i32 {
        let (s, n) = MEM.lock().dmem_largest_free(start.max(0) as u64, end.max(0) as u64);
        let a = align.max(PAGE);
        let s2 = mem::align_up(s, a);
        let n2 = n.saturating_sub(s2 - s);
        if !out_start.is_null() { *out_start = s2 as i64; }
        if !out_size.is_null() { *out_size = n2; }
        if n2 == 0 { sce(ENOMEM) } else { 0 }
    }
    fn sceKernelGetDirectMemoryType(start: i64, mtype: *mut i32, out_start: *mut i64, out_end: *mut i64) -> i32 {
        let st = MEM.lock();
        let s = start as u64;
        for (&b, blk) in st.dmem.range(..=s).rev().take(1) {
            if s < b + blk.size {
                if !mtype.is_null() { *mtype = blk.mtype; }
                if !out_start.is_null() { *out_start = b as i64; }
                if !out_end.is_null() { *out_end = (b + blk.size) as i64; }
                return 0;
            }
        }
        sce(ENOENT)
    }
    fn sceKernelMapDirectMemory(addr: *mut u64, len: u64, prot: i32, flags: i32, phys: i64, align: u64) -> i32 {
        map_common(addr, len, prot, flags, VmaKind::Direct, phys as u64, align, "dmem")
    }
    fn sceKernelMapNamedDirectMemory(addr: *mut u64, len: u64, prot: i32, flags: i32, phys: i64, align: u64, name: *const u8) -> i32 {
        map_common(addr, len, prot, flags, VmaKind::Direct, phys as u64, align, cstr(name))
    }
    fn sceKernelMapFlexibleMemory(addr: *mut u64, len: u64, prot: i32, flags: i32) -> i32 {
        flexible(addr, len, prot, flags, "flexible")
    }
    fn sceKernelMapNamedFlexibleMemory(addr: *mut u64, len: u64, prot: i32, flags: i32, name: *const u8) -> i32 {
        flexible(addr, len, prot, flags, cstr(name))
    }
    fn sceKernelMapNamedSystemFlexibleMemory(addr: *mut u64, len: u64, prot: i32, flags: i32, name: *const u8) -> i32 {
        flexible(addr, len, prot, flags, cstr(name))
    }
    fn sceKernelReleaseFlexibleMemory(addr: u64, len: u64) -> i32 {
        MEM.lock().unmap_range(addr, addr + mem::align_up(len, PAGE));
        0
    }
    fn sceKernelAvailableFlexibleMemorySize(out: *mut u64) -> i32 {
        if out.is_null() { return sce(EINVAL); }
        *out = FLEXIBLE_TOTAL.saturating_sub(MEM.lock().flexible_used);
        0
    }
    fn sceKernelMunmap(addr: u64, len: u64) -> i32 {
        if addr % PAGE != 0 || len == 0 { return sce(EINVAL); }
        MEM.lock().unmap_range(addr, addr + mem::align_up(len, PAGE));
        0
    }
    fn sceKernelMprotect(addr: u64, len: u64, prot: i32) -> i32 {
        let start = mem::align_down(addr, PAGE);
        let end = mem::align_up(addr + len, PAGE);
        match MEM.lock().protect(start, end, prot as u32) { Ok(()) => 0, Err(_) => sce(EINVAL) }
    }
    fn sceKernelMtypeprotect(addr: u64, len: u64, mtype: i32, prot: i32) -> i32 {
        sceKernelMprotect(addr, len, prot)
    }
    fn sceKernelReserveVirtualRange(addr: *mut u64, len: u64, flags: i32, align: u64) -> i32 {
        map_common(addr, len, 0, flags, VmaKind::Reserved, 0, align, "reserved")
    }
    fn sceKernelQueryMemoryProtection(addr: u64, start: *mut u64, end: *mut u64, prot: *mut u32) -> i32 {
        let st = MEM.lock();
        match st.find(addr) {
            Some(v) if v.kind != VmaKind::Reserved => {
                if !start.is_null() { *start = v.start; }
                if !end.is_null() { *end = v.end() - 1; }
                if !prot.is_null() { *prot = v.prot; }
                0
            }
            _ => sce(EACCES),
        }
    }
    fn sceKernelVirtualQuery(addr: u64, flags: i32, info: *mut u8, size: u64) -> i32 {
        if info.is_null() || size < std::mem::size_of::<VirtualQueryInfo>() as u64 { return sce(EINVAL); }
        let st = MEM.lock();
        let v = match st.find(addr) {
            Some(v) => Some(v.clone()),
            None if flags & 1 != 0 => st.vmas.range(addr..).next().map(|(_, v)| v.clone()),
            None => None,
        };
        let Some(v) = v else { return sce(EACCES) };
        let mut q = VirtualQueryInfo { start: v.start, end: v.end(), offset: v.phys, protection: v.prot as i32, memory_type: v.mtype, bits: 0, name: [0; 32] };
        q.bits = match v.kind {
            VmaKind::Flexible => 1 | 0x10,
            VmaKind::Direct => 2 | 0x10,
            VmaKind::Reserved => 0,
            _ => 0x10,
        };
        let n = v.name.len().min(31);
        q.name[..n].copy_from_slice(&v.name.as_bytes()[..n]);
        std::ptr::write_unaligned(info as *mut VirtualQueryInfo, q);
        0
    }
    fn sceKernelBatchMap(entries: *const BatchEntry, n: i32, out: *mut i32) -> i32 {
        batch_map(entries, n, out, MAP_FIXED)
    }
    fn sceKernelBatchMap2(entries: *const BatchEntry, n: i32, out: *mut i32, flags: i32) -> i32 {
        batch_map(entries, n, out, flags)
    }
    fn sceKernelMmap(addr: u64, len: u64, prot: i32, flags: i32, fd: i32, offset: i64, out: *mut u64) -> i32 {
        let r = mmap(addr, len, prot, flags, fd, offset);
        if r == MAP_FAILED { return sce(ENOMEM); }
        *out = r;
        0
    }
    fn mmap(addr: u64, len: u64, prot: i32, flags: i32, fd: i32, offset: i64) -> u64 {
        let mut a = addr;
        if flags & MAP_ANON != 0 || fd < 0 {
            let r = map_common(&mut a, len, prot, flags, VmaKind::Flexible, 0, 0, "mmap");
            if r != 0 { posix_ret(r); return MAP_FAILED; }
            return a;
        }
        // File-backed: private copy of the file contents.
        let r = map_common(&mut a, len, prot | 3, flags, VmaKind::Flexible, 0, 0, "mmap-file");
        if r != 0 { posix_ret(r); return MAP_FAILED; }
        let buf = std::slice::from_raw_parts_mut(a as *mut u8, len as usize);
        match crate::hle::kernel::fs::pread_fd(fd, buf, offset as u64) {
            Ok(_) => {
                crate::warn!("mmap of fd {fd} at {a:#x}+{len:#x}: private copy (not shared)");
                a
            }
            Err(e) => { crate::hle::kernel::set_errno(e); MAP_FAILED }
        }
    }
    fn munmap(addr: u64, len: u64) -> i32 {
        posix_ret(sceKernelMunmap(addr, len))
    }
    fn mprotect(addr: u64, len: u64, prot: i32) -> i32 { posix_ret(sceKernelMprotect(addr, len, prot)) }
    fn msync(addr: u64, len: u64, flags: i32) -> i32 { 0 }
    fn madvise(addr: u64, len: u64, advice: i32) -> i32 { 0 }
    fn getpagesize() -> i32 { PAGE as i32 }
    fn sceKernelIsStack(addr: u64, start: *mut u64, end: *mut u64) -> i32 { sce(EINVAL) }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libkernel", [
        sceKernelGetDirectMemorySize, sceKernelAllocateDirectMemory, sceKernelAllocateMainDirectMemory,
        sceKernelReleaseDirectMemory, sceKernelCheckedReleaseDirectMemory, sceKernelAvailableDirectMemorySize,
        sceKernelGetDirectMemoryType, sceKernelMapDirectMemory, sceKernelMapNamedDirectMemory,
        sceKernelMapFlexibleMemory, sceKernelMapNamedFlexibleMemory, sceKernelMapNamedSystemFlexibleMemory,
        sceKernelReleaseFlexibleMemory, sceKernelAvailableFlexibleMemorySize, sceKernelMunmap, sceKernelMprotect,
        sceKernelMtypeprotect, sceKernelReserveVirtualRange, sceKernelQueryMemoryProtection, sceKernelVirtualQuery,
        sceKernelBatchMap, sceKernelBatchMap2, sceKernelMmap, mmap, munmap, mprotect, msync, madvise, getpagesize,
        sceKernelIsStack,
    ]);
    reg!(r, "libScePosix", [mmap, munmap, mprotect, msync, madvise, getpagesize]);
}
