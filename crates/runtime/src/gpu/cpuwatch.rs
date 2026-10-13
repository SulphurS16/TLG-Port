//! Write protection of cold guest memory blocks (`tune cpu_watch`).
//!
//! The write tracker (writes.rs) asks the OS which pages were written
//! (GetWriteWatch) for every 64 KiB block a command reads, once per CPU epoch
//! (a few per frame): in Trico's cave ~175,000 pages per frame, nearly all
//! of them static (geometry pools, textures), ~7% of the graphics CP thread.
//! A block the tracker found unwritten in a number of consecutive queries is
//! armed instead: its pages are made read-only, and it needs no query until
//! the CPU writes it. Such a write faults; the exception handler makes the
//! block writable again and lists it as disarmed, the write proceeds (the OS
//! records it as usual), and the tracker queries the block again from then
//! on (arming it again only after a longer quiet period).
//!
//! Only blocks whose pages are all committed read-write are armed, never
//! ones overlapping ranges the lazy write-back keeps inaccessible (decided
//! under its lock, which its protection changes hold too: `before_protect`).
//! Before the guest unmaps memory or changes its protection (`settle`) and
//! before the OS itself writes guest memory (file reads, `touch`), armed
//! blocks there are made writable again.
//!
//! Locks: STATE is held only around set updates and protection changes,
//! never while guest memory is accessed (a fault there would deadlock).
//! Callers may hold the lazy write-back's lock (taken before this one).

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_READONLY, PAGE_READWRITE, VirtualProtect, VirtualQuery};

/// Block size (as the write tracker's query blocks).
pub const BLOCK_SHIFT: u32 = 16;
const BLOCK: u64 = 1 << BLOCK_SHIFT;

struct State {
    /// Armed blocks (address >> BLOCK_SHIFT): read-only now, read-write for
    /// the guest.
    armed: BTreeSet<u64>,
    /// Blocks made writable again since the tracker last took them.
    disarmed: Vec<u64>,
    faults: u64,
    arms: u64,
    restores: u64,
}

static STATE: parking_lot::Mutex<State> = parking_lot::Mutex::new(State { armed: BTreeSet::new(), disarmed: Vec::new(), faults: 0, arms: 0, restores: 0 });
/// Some block is armed.
static ACTIVE: AtomicBool = AtomicBool::new(false);

pub fn enabled() -> bool {
    crate::tune::get(crate::tune::CPU_WATCH) != 0 && crate::mem::WRITE_WATCH.load(Ordering::Relaxed) && super::vk::ctx().ext_host.is_none() && !super::wbprobe::enabled()
}

fn protect(lo: u64, hi: u64, prot: u32) -> bool {
    let mut old = 0u32;
    unsafe { VirtualProtect(lo as _, (hi - lo) as usize, prot, &mut old) != 0 }
}

/// Whether every page of [lo, hi) is committed and exactly read-write.
fn all_read_write(lo: u64, hi: u64) -> bool {
    let mut a = lo;
    while a < hi {
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { VirtualQuery(a as _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 || mbi.State != MEM_COMMIT || mbi.Protect != PAGE_READWRITE {
            return false;
        }
        a = mbi.BaseAddress as u64 + mbi.RegionSize as u64;
    }
    true
}

/// Arms the blocks [lo, hi) (block-aligned); false when they cannot be (not
/// all committed read-write, or lazy write-back ranges overlap). The caller
/// queries the write watch for them once more afterwards: writes made
/// before the protection took effect are recorded there, later ones fault.
pub fn arm(lo: u64, hi: u64) -> bool {
    debug_assert!(lo % BLOCK == 0 && hi % BLOCK == 0 && lo < hi);
    super::lazywb::without_ranges(lo, hi, || {
        if !all_read_write(lo, hi) {
            return false;
        }
        let mut g = STATE.lock();
        if !protect(lo, hi, PAGE_READONLY) {
            return false;
        }
        g.armed.extend((lo >> BLOCK_SHIFT)..(hi >> BLOCK_SHIFT));
        g.arms += (hi - lo) >> BLOCK_SHIFT;
        ACTIVE.store(true, Ordering::SeqCst);
        true
    })
    .unwrap_or(false)
}

/// The blocks made writable again since the last call.
pub fn take_disarmed(out: &mut Vec<u64>) {
    if !ACTIVE.load(Ordering::Relaxed) {
        return;
    }
    let mut g = STATE.lock();
    out.append(&mut g.disarmed);
}

/// Makes the armed blocks overlapping [lo, hi) writable (holding STATE).
fn restore(g: &mut State, lo: u64, hi: u64) {
    if g.armed.is_empty() || hi <= lo {
        return;
    }
    let blocks: Vec<u64> = g.armed.range((lo >> BLOCK_SHIFT)..=((hi - 1) >> BLOCK_SHIFT)).copied().collect();
    for b in blocks {
        g.armed.remove(&b);
        protect(b << BLOCK_SHIFT, (b + 1) << BLOCK_SHIFT, PAGE_READWRITE);
        g.disarmed.push(b);
        g.restores += 1;
    }
}

/// Before the OS itself writes guest memory [addr, addr+len) (it fails on
/// read-only pages instead of faulting), before the guest unmaps it or
/// changes its protection, and before the lazy write-back makes it
/// inaccessible: armed blocks there writable again.
pub fn touch(addr: u64, len: u64) {
    if !ACTIVE.load(Ordering::SeqCst) || len == 0 {
        return;
    }
    restore(&mut STATE.lock(), addr, addr + len);
}

/// Exception handler: a write access violation at `addr`. True when it hit
/// an armed block (now writable) or one another thread made writable
/// meanwhile: the access is retried.
pub fn fault(addr: u64) -> bool {
    if !ACTIVE.load(Ordering::SeqCst) {
        return false;
    }
    let b = addr >> BLOCK_SHIFT;
    {
        let mut g = STATE.lock();
        if g.armed.contains(&b) {
            g.faults += 1;
            restore(&mut g, b << BLOCK_SHIFT, (b + 1) << BLOCK_SHIFT);
            return true;
        }
    }
    // (Raced with another thread restoring the block.)
    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let n = unsafe { VirtualQuery(addr as _, &mut mbi, std::mem::size_of::<MEMORY_BASIC_INFORMATION>()) };
    n != 0 && mbi.State == MEM_COMMIT && mbi.Protect == PAGE_READWRITE
}

/// Report line.
pub fn report() {
    let g = STATE.lock();
    if g.arms == 0 {
        return;
    }
    crate::info!(
        "write protection of cold blocks: {} armed now ({} MiB); {} armings, {} write faults, {} made writable again in all",
        g.armed.len(),
        (g.armed.len() as u64 * BLOCK) >> 20,
        g.arms,
        g.faults,
        g.restores
    );
}
