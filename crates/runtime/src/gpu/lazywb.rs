//! Lazy write-back of GPU-written data held in VRAM mirror windows (`tune
//! lazy_wb`, default on; 0 for every write-back after each batch).
//!
//! Every batch end copies the GPU's mirror-resident writes back to guest
//! memory. In Trico's cave most of those bytes are never looked at again:
//! the write-back probe (wbprobe.rs) found every high-traffic range untouched
//! by the title and by the runtime between write-backs. Here a range is
//! written back eagerly at first; after each write-back its pages (those it
//! covers entirely) are made inaccessible. A range that went `LAZY_AFTER`
//! write-backs without anything touching it is no longer copied out: its
//! pages stay inaccessible and the mirror holds the newest data. The first
//! access then faults; the exception handler reads the range back from the
//! mirror (its own command buffer, fence and staging buffer: it may run on
//! any thread, the GPU threads included, without the renderer lock), copies
//! it into guest memory, and the range is eager again (with a longer quiet
//! period before it may go lazy once more). A range touched while eager is
//! simply made accessible; after `MAX_STRIKES` touches it stays eager.
//!
//! A skip is decided while its batch is being submitted; a readback waits
//! until the batches of the range's skips have been submitted (`submitted`),
//! so it always includes their writes (it runs after all submitted work).
//! The renderer reads lazy data back before a mirror window that holds it
//! goes out of service (`release_buffer`) and before an eager write-back
//! overlapping it lands (`before_write`); the OS's own accesses (file I/O,
//! which fail instead of faulting) are preceded by `touch`, unmapping and
//! protection changes of guest memory by `settle`.
//!
//! Locks: callers may hold the memory map lock (faults in guest copies,
//! `settle`), `mem::WRITE_WATCH_LOCK` (the completion thread) or the
//! renderer lock; `STATE` holders take neither of them (a readback's copy
//! and write-watch reset are atomic for harvests through
//! `mem::HARVEST_LOCK`).

use ash::vk;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READWRITE, PAGE_GUARD, PAGE_NOACCESS, PAGE_READWRITE, VirtualProtect, VirtualQuery};

/// Untouched write-backs after which a range goes lazy.
const LAZY_AFTER: u32 = 4;
/// Touches (eager or lazy) after which a range stays eager.
const MAX_STRIKES: u32 = 4;

struct Entry {
    /// Page-aligned end (the key is the page-aligned start).
    end: u64,
    /// Untouched write-backs since the range was last touched.
    quiet: u32,
    strikes: u32,
    /// Lazy: where the newest data is (mirror buffer, offset of the start).
    lazy: Option<(vk::Buffer, u64)>,
    /// The newest skip decided for the range (see `SUBMITTED`).
    skip: u64,
    protected: bool,
}

#[derive(Default)]
struct Stats {
    lazy_writebacks: u64,
    lazy_bytes: u64,
    faults: u64,
    io_touches: u64,
    readbacks: u64,
    readback_bytes: u64,
    readback_ns: u64,
    submit_waits: u64,
    protects: u64,
}

struct State {
    entries: BTreeMap<u64, Entry>,
    skips: u64,
    stats: Stats,
}

static STATE: parking_lot::Mutex<State> = parking_lot::Mutex::new(State {
    entries: BTreeMap::new(),
    skips: 0,
    stats: Stats { lazy_writebacks: 0, lazy_bytes: 0, faults: 0, io_touches: 0, readbacks: 0, readback_bytes: 0, readback_ns: 0, submit_waits: 0, protects: 0 },
});
/// Some range is tracked (its pages may be inaccessible).
static ACTIVE: AtomicBool = AtomicBool::new(false);
/// Skips are numbered as they are decided; the newest whose batch has been
/// submitted.
static SUBMITTED: AtomicU64 = AtomicU64::new(0);

pub fn enabled() -> bool {
    crate::tune::get(crate::tune::LAZY_WB) != 0 && super::vk::ctx().ext_host.is_none() && !super::wbprobe::enabled()
}

/// 64 KiB blocks of the guest range entirely inside protected ranges, a bit
/// each: the CPU cannot write them without faulting into `fault` first, so
/// the write tracker leaves them out of its write-watch queries once it has
/// queried them after they were covered (writes.rs, `tune lazy_skip`).
/// Changed under STATE; each change is listed for the tracker
/// (`take_cover_events`), which queries the block again.
const COVER_SHIFT: u32 = 16;
const COVER_WORDS: usize = ((crate::mem::GUEST_END - crate::mem::GUEST_BASE) >> COVER_SHIFT >> 6) as usize;
static COVERED: [AtomicU64; COVER_WORDS] = [const { AtomicU64::new(0) }; COVER_WORDS];
static COVER_EVENTS: parking_lot::Mutex<Vec<u64>> = parking_lot::Mutex::new(Vec::new());
static COVER_PENDING: AtomicBool = AtomicBool::new(false);

/// Marks the blocks entirely inside [lo, hi) covered or not (STATE held).
fn cover(lo: u64, hi: u64, on: bool) {
    let base = crate::mem::GUEST_BASE >> COVER_SHIFT;
    let (b0, b1) = ((lo + (1 << COVER_SHIFT) - 1) >> COVER_SHIFT, hi >> COVER_SHIFT);
    if b0 >= b1 || b0 < base || (b1 - base) as usize > COVER_WORDS * 64 {
        return;
    }
    let mut ev = COVER_EVENTS.lock();
    for b in b0..b1 {
        let i = (b - base) as usize;
        let bit = 1u64 << (i & 63);
        let old = if on { COVERED[i >> 6].fetch_or(bit, Ordering::AcqRel) } else { COVERED[i >> 6].fetch_and(!bit, Ordering::AcqRel) };
        if (old & bit != 0) != on {
            ev.push(b);
        }
    }
    if !ev.is_empty() {
        COVER_PENDING.store(true, Ordering::Release);
    }
}

/// Whether 64 KiB block `block` (address >> 16) is covered (see `COVERED`).
pub fn covered(block: u64) -> bool {
    let i = block.wrapping_sub(crate::mem::GUEST_BASE >> COVER_SHIFT) as usize;
    i < COVER_WORDS * 64 && COVERED[i >> 6].load(Ordering::Acquire) >> (i & 63) & 1 != 0
}

/// The blocks covered or uncovered since the last call (appended to `out`).
pub fn take_cover_events(out: &mut Vec<u64>) {
    if COVER_PENDING.load(Ordering::Acquire) && COVER_PENDING.swap(false, Ordering::AcqRel) {
        out.append(&mut COVER_EVENTS.lock());
    }
}

fn active() -> bool {
    ACTIVE.load(Ordering::SeqCst)
}

/// The pages entirely inside [addr, addr+size).
pub fn inner_pages(addr: u64, size: u64) -> Option<(u64, u64)> {
    let (lo, hi) = ((addr + 0xFFF) & !0xFFF, (addr + size) & !0xFFF);
    (lo < hi).then_some((lo, hi))
}

fn protect(st: &mut Stats, lo: u64, hi: u64, prot: u32) -> bool {
    let mut old = 0u32;
    st.protects += 1;
    unsafe { VirtualProtect(lo as _, (hi - lo) as usize, prot, &mut old) != 0 }
}

/// Runs `f` with the state locked when no tracked range overlaps [lo, hi):
/// the write protection of cold blocks (cpuwatch.rs) arms blocks under it,
/// so its protection changes and ours never interleave. None when one does.
pub fn without_ranges<R>(lo: u64, hi: u64, f: impl FnOnce() -> R) -> Option<R> {
    let g = STATE.lock();
    if !overlapping(&g, lo, hi).is_empty() {
        return None;
    }
    Some(f())
}

/// Keys of the entries overlapping [lo, hi) (entries are disjoint).
fn overlapping(g: &State, lo: u64, hi: u64) -> Vec<u64> {
    g.entries.range(..hi).rev().take_while(|(_, e)| e.end > lo).map(|(k, _)| *k).collect()
}

/// Untouched write-backs after which a range of `bytes` with `strikes`
/// touches goes lazy: `LAZY_AFTER`, doubled per strike; with `tune
/// lazy_big_kib` (off) ranges that large never touched, after one (the
/// first eager write-back makes guest memory current and protects it, as
/// for any range).
fn lazy_after(bytes: u64, strikes: u32) -> u32 {
    let big = crate::tune::get(crate::tune::LAZY_BIG_KIB);
    if strikes == 0 && big > 0 && bytes >= (big as u64) << 10 {
        return 1;
    }
    LAZY_AFTER << strikes.min(4)
}

/// At submission: whether the mirror write-back of the inner pages
/// [lo, hi) can be skipped (the range is lazy; its newest data at
/// `buf`+`off`, `off` the mirror offset of `lo`); then the skip's number,
/// for `submitted` once the batch has been submitted. None: copy it out.
pub fn skip_writeback(lo: u64, hi: u64, buf: vk::Buffer, off: u64) -> Option<u64> {
    let mut guard = STATE.lock();
    let State { entries, skips, stats } = &mut *guard;
    let e = entries.get_mut(&lo)?;
    if e.end != hi || !e.protected || e.strikes >= MAX_STRIKES || e.quiet < lazy_after(hi - lo, e.strikes) {
        return None;
    }
    *skips += 1;
    e.lazy = Some((buf, off));
    e.skip = *skips;
    stats.lazy_writebacks += 1;
    stats.lazy_bytes += hi - lo;
    Some(*skips)
}

/// The batch with skips up to `skip` has been submitted.
pub fn submitted(skip: u64) {
    SUBMITTED.fetch_max(skip, Ordering::SeqCst);
}

/// Readback service: mirror -> host staging on its own command buffer.
struct Readback {
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    staging: Option<(vk::Buffer, vk::DeviceMemory, *mut u8, u64)>,
}

unsafe impl Send for Readback {}

static READBACK: parking_lot::Mutex<Option<Readback>> = parking_lot::Mutex::new(None);

/// Copies [off, off+size) of `buf` into guest memory at `dst`, after all work
/// submitted so far; `unprotect` makes the pages accessible right before the
/// copy (`guarded`: they were inaccessible until then).
fn read_back(buf: vk::Buffer, off: u64, size: u64, dst: u64, guarded: bool, unprotect: impl FnOnce()) -> Result<(), String> {
    let c = super::vk::ctx();
    let mut g = READBACK.lock();
    if g.is_none() {
        unsafe {
            let pool = c.device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None).map_err(|e| e.to_string())?;
            let cmd = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1)).map_err(|e| e.to_string())?[0];
            let fence = c.device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| e.to_string())?;
            *g = Some(Readback { pool, cmd, fence, staging: None });
        }
    }
    let rb = g.as_mut().unwrap();
    let _ = rb.pool;
    if rb.staging.as_ref().is_none_or(|s| s.3 < size) {
        if let Some((b, m, _, _)) = rb.staging.take() {
            unsafe { c.device.destroy_buffer(b, None) };
            c.free(m);
        }
        let n = size.next_multiple_of(1 << 20);
        let (b, m, p) = c.host_buffer(n, vk::BufferUsageFlags::TRANSFER_DST)?;
        rb.staging = Some((b, m, p, n));
    }
    let (sb, _, sp, _) = *rb.staging.as_ref().unwrap();
    unsafe {
        c.device.reset_fences(&[rb.fence]).map_err(|e| e.to_string())?;
        c.device.begin_command_buffer(rb.cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).map_err(|e| e.to_string())?;
        let before = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ);
        c.device.cmd_pipeline_barrier(rb.cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[before], &[], &[]);
        c.device.cmd_copy_buffer(rb.cmd, buf, sb, &[vk::BufferCopy { src_offset: off, dst_offset: 0, size }]);
        let after = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
        c.device.cmd_pipeline_barrier(rb.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[after], &[], &[]);
        c.device.end_command_buffer(rb.cmd).map_err(|e| e.to_string())?;
    }
    c.submit(rb.cmd, &[], &[], rb.fence)?;
    unsafe { c.device.wait_for_fences(&[rb.fence], true, 10_000_000_000) }.map_err(|e| format!("lazy write-back readback: {e}"))?;
    // (Directly: the memory map lock may be held by this thread. Our copy is
    // not a CPU write of the range: forgotten before a harvest can see it.)
    let _h = crate::mem::HARVEST_LOCK.lock();
    super::wbprobe::note_range(dst, size, 4);
    // (`unprotect` makes the pages accessible: from then until the copy is
    // in, no guest thread runs - `tune lazy_hold`.)
    let install = || {
        unprotect();
        unsafe { std::ptr::copy_nonoverlapping(sp, dst as *mut u8, size as usize) };
        crate::mem::reset_write_watch(dst, size);
    };
    // (Pages that were accessible all along: nothing to hold the threads for.)
    if guarded && crate::tune::get(crate::tune::LAZY_HOLD) != 0 {
        crate::thread::hold_guest_threads(install);
    } else {
        install();
    }
    Ok(())
}

/// Locks the state once the batches of the skips of the entries overlapping
/// [lo, hi) have been submitted (a readback then includes their writes).
/// The decision and the submission happen in one renderer call that takes
/// none of the locks a caller here may hold.
fn lock_submitted(lo: u64, hi: u64) -> parking_lot::MutexGuard<'static, State> {
    let t0 = std::time::Instant::now();
    let mut waited = false;
    loop {
        let mut g = STATE.lock();
        let need = overlapping(&g, lo, hi).iter().filter_map(|k| g.entries[k].lazy.map(|_| g.entries[k].skip)).max().unwrap_or(0);
        if need <= SUBMITTED.load(Ordering::SeqCst) {
            if waited {
                g.stats.submit_waits += 1;
            }
            return g;
        }
        drop(g);
        waited = true;
        if t0.elapsed() > std::time::Duration::from_secs(2) {
            crate::error!("lazy write-back: skip {need} of {lo:#x}-{hi:#x} not submitted after 2 s; reading back without it");
            let mut g = STATE.lock();
            g.stats.submit_waits += 1;
            return g;
        }
        std::thread::yield_now();
    }
}

/// Makes entry `lo` accessible, reading it back first when lazy; it is
/// eager afterwards. `touched`: a CPU access (a strike).
fn resolve(g: &mut State, lo: u64, touched: bool) {
    let State { entries, stats, .. } = g;
    let Some(e) = entries.get_mut(&lo) else { return };
    let (hi, lazy) = (e.end, e.lazy.take());
    let protected = std::mem::replace(&mut e.protected, false);
    if protected {
        // (Before the pages turn accessible: the write tracker queries them
        // again from now on.)
        cover(lo, hi, false);
    }
    if touched {
        e.strikes += 1;
        e.quiet = 0;
    }
    match lazy {
        // (The GPU's data is staged while the pages are still inaccessible,
        // then copied in with the title's threads held: a title thread
        // writing there between the pages turning accessible and the copy
        // would have its write buried under it - at a stage's load the
        // title puts new objects in memory a lazy range held, from several
        // threads at once.)
        Some((buf, off)) => {
            let t0 = std::time::Instant::now();
            let unprotect = || {
                if protected {
                    protect(stats, lo, hi, PAGE_READWRITE);
                }
            };
            if let Err(e) = read_back(buf, off, hi - lo, lo, protected, unprotect) {
                crate::error!("lazy write-back of {lo:#x}-{hi:#x}: {e}");
                // (The pages accessible regardless: nothing else will.)
                if protected {
                    protect(stats, lo, hi, PAGE_READWRITE);
                }
            }
            stats.readbacks += 1;
            stats.readback_bytes += hi - lo;
            stats.readback_ns += t0.elapsed().as_nanos() as u64;
        }
        None => {
            if protected {
                protect(stats, lo, hi, PAGE_READWRITE);
            }
        }
    }
}

fn update_active(g: &State) {
    if g.entries.is_empty() {
        ACTIVE.store(false, Ordering::SeqCst);
    }
}

/// Completion thread, before an eager write-back into [addr, addr+size):
/// ranges there accessible (lazy ones read back first: the write-back is
/// newer only where it lands).
pub fn before_write(addr: u64, size: u64) {
    if !active() {
        return;
    }
    let mut g = lock_submitted(addr, addr + size);
    for k in overlapping(&g, addr, addr + size) {
        resolve(&mut g, k, false);
    }
}

/// Completion thread, after an eager write-back into [addr, addr+size)
/// (`mirror`: of data the VRAM mirror holds): its inner pages protected, the
/// range one write-back quieter.
pub fn after_write(addr: u64, size: u64, mirror: bool) {
    let on = enabled() && mirror;
    if !on && !active() {
        return;
    }
    let Some((lo, hi)) = inner_pages(addr, size) else { return };
    let mut g = lock_submitted(lo, hi);
    // (Overlapping entries of other extents give way; without tracking, all.)
    for k in overlapping(&g, lo, hi) {
        if !on || k != lo || g.entries[&k].end != hi {
            resolve(&mut g, k, false);
            g.entries.remove(&k);
        }
    }
    if !on {
        update_active(&g);
        return;
    }
    ACTIVE.store(true, Ordering::SeqCst);
    let State { entries, stats, .. } = &mut *g;
    let e = entries.entry(lo).or_insert(Entry { end: hi, quiet: 0, strikes: 0, lazy: None, skip: 0, protected: false });
    if e.strikes >= MAX_STRIKES {
        return;
    }
    e.quiet = e.quiet.saturating_add(1);
    if !e.protected {
        // (Armed cold blocks there writable first: cpuwatch.rs.)
        super::cpuwatch::touch(lo, hi - lo);
        e.protected = protect(stats, lo, hi, PAGE_NOACCESS);
        if e.protected {
            cover(lo, hi, true);
        }
    }
}

/// Before the mirror window buffer `buf` goes out of service: its lazy data
/// read back to guest memory. (Renderer, never while it submits a batch.)
pub fn release_buffer(buf: vk::Buffer) {
    if !active() {
        return;
    }
    let mut g = STATE.lock();
    let keys: Vec<u64> = g.entries.iter().filter(|(_, e)| e.lazy.is_some_and(|(b, _)| b == buf)).map(|(k, _)| *k).collect();
    for k in keys {
        resolve(&mut g, k, false);
    }
}

/// Before the OS itself reads or writes guest memory [addr, addr+len) (file
/// I/O: it fails on inaccessible pages instead of faulting): ranges there
/// current and accessible.
pub fn touch(addr: u64, len: u64) {
    if !active() || len == 0 {
        return;
    }
    let mut g = lock_submitted(addr, addr + len);
    for k in overlapping(&g, addr, addr + len) {
        if g.entries[&k].protected || g.entries[&k].lazy.is_some() {
            g.stats.io_touches += 1;
            resolve(&mut g, k, true);
        }
    }
}

/// Memory map, before [lo, hi) is unmapped or its protection changes (its
/// lock held): ranges there current and no longer tracked.
pub fn settle(lo: u64, hi: u64) {
    if !active() {
        return;
    }
    let mut g = lock_submitted(lo, hi);
    for k in overlapping(&g, lo, hi) {
        resolve(&mut g, k, false);
        g.entries.remove(&k);
    }
    update_active(&g);
}

/// Exception handler: an access violation at `addr` (not an execution).
/// True when it was a tracked range (now current and accessible) or one
/// another thread made accessible meanwhile: the access is retried.
pub fn fault(addr: u64, exec: bool) -> bool {
    if !active() || exec {
        return false;
    }
    {
        let mut g = lock_submitted(addr, addr + 1);
        if let Some((&lo, e)) = g.entries.range(..=addr).next_back() {
            if addr < e.end && e.protected {
                g.stats.faults += 1;
                resolve(&mut g, lo, true);
                return true;
            }
        }
    }
    // (Raced with another thread resolving or dropping the range.)
    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let n = unsafe { VirtualQuery(addr as _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) };
    n != 0 && mbi.State == MEM_COMMIT && mbi.Protect & PAGE_GUARD == 0 && mbi.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE) != 0
}

/// Report line.
pub fn report() {
    let g = STATE.lock();
    if g.entries.is_empty() && g.stats.lazy_writebacks == 0 {
        return;
    }
    let s = &g.stats;
    let lazy = g.entries.values().filter(|e| e.lazy.is_some()).count();
    let eager_for_good = g.entries.values().filter(|e| e.strikes >= MAX_STRIKES).count();
    crate::info!(
        "lazy write-backs: {} skipped ({} MiB), {} faults, {} file I/O touches, {} readbacks ({} MiB, {} ms), {} waits for submission, {} protection changes; {} ranges tracked, {} lazy now, {} eager for good",
        s.lazy_writebacks,
        s.lazy_bytes >> 20,
        s.faults,
        s.io_touches,
        s.readbacks,
        s.readback_bytes >> 20,
        s.readback_ns / 1_000_000,
        s.submit_waits,
        s.protects,
        g.entries.len(),
        lazy,
        eager_for_good
    );
}
