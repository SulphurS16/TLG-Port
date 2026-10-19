//! Debug (`TLG_WB_PROBE=1`): which GPU-written ranges does anything on the CPU
//! touch after their write-back? Once a mirror write-back has landed in guest
//! memory, the pages it covers entirely are made inaccessible; the first
//! access (by the title or by the runtime) faults, is logged with the code
//! that made it, and ends probing of that range (its pages are restored, and
//! it is not protected again). Before a later write-back of a probed range
//! its pages are restored for the copy. The data is always current, so the
//! probe changes timing only; ranges never touched are candidates for
//! write-backs on demand instead of after every batch.

use std::collections::BTreeMap;
use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READWRITE, VirtualProtect};

struct Range {
    end: u64,
    /// Write-backs while probed.
    writes: u64,
    protected: bool,
}

struct State {
    /// Probed ranges (page-aligned start -> range).
    ranges: BTreeMap<u64, Range>,
    /// Ranges something touched: (start, end) -> (first accessor, write-backs before it).
    touched: BTreeMap<(u64, u64), (String, u64)>,
}

static STATE: parking_lot::Mutex<State> = parking_lot::Mutex::new(State { ranges: BTreeMap::new(), touched: BTreeMap::new() });

pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TLG_WB_PROBE").is_ok_and(|v| v == "1"))
}

/// The pages entirely inside [addr, addr+size).
fn inner_pages(addr: u64, size: u64) -> Option<(u64, u64)> {
    let (lo, hi) = ((addr + 0xFFF) & !0xFFF, (addr + size) & !0xFFF);
    (lo < hi).then_some((lo, hi))
}

fn protect(lo: u64, hi: u64, prot: u32) {
    let mut old = 0u32;
    unsafe { VirtualProtect(lo as _, (hi - lo) as usize, prot, &mut old) };
}

/// Before the write-back copy into [addr, addr+size): probed ranges there
/// accessible again.
pub fn before_write(addr: u64, size: u64) {
    let mut g = STATE.lock();
    let end = addr + size;
    let keys: Vec<u64> = g.ranges.range(..end).filter(|(_, r)| r.end > addr && r.protected).map(|(k, _)| *k).collect();
    for k in keys {
        let r = g.ranges.get_mut(&k).unwrap();
        protect(k, r.end, PAGE_READWRITE);
        r.protected = false;
    }
}

/// After the write-back copy into [addr, addr+size): its pages probed (unless
/// something touched the range before).
pub fn after_write(addr: u64, size: u64) {
    let Some((lo, hi)) = inner_pages(addr, size) else { return };
    let mut g = STATE.lock();
    if g.touched.keys().any(|&(a, b)| a < hi && lo < b) {
        return;
    }
    // (Overlapping probed ranges of other extents: replaced by this one.)
    let stale: Vec<u64> = g.ranges.range(..hi).filter(|(k, r)| r.end > lo && **k != lo).map(|(k, _)| *k).collect();
    for k in stale {
        let r = g.ranges.remove(&k).unwrap();
        if r.protected {
            protect(k, r.end, PAGE_READWRITE);
        }
    }
    let r = g.ranges.entry(lo).or_insert(Range { end: hi, writes: 0, protected: false });
    r.end = hi;
    r.writes += 1;
    if !r.protected {
        protect(lo, hi, PAGE_NOACCESS);
        r.protected = true;
    }
}

/// Before the OS itself reads or writes guest memory [addr, addr+len) (file
/// I/O: it fails on inaccessible pages instead of faulting): probed ranges
/// there accessible again, recorded as touched.
pub fn touch(addr: u64, len: u64) {
    if !enabled() || len == 0 {
        return;
    }
    let mut g = STATE.lock();
    let end = addr + len;
    let keys: Vec<u64> = g.ranges.range(..end).filter(|(_, r)| r.end > addr && r.protected).map(|(k, _)| *k).collect();
    for lo in keys {
        let r = g.ranges.remove(&lo).unwrap();
        protect(lo, r.end, PAGE_READWRITE);
        crate::info!("wb probe: GPU-written {lo:#x}-{:#x} ({} KiB, {} write-backs) touched: file I/O", r.end, (r.end - lo) >> 10, r.writes);
        g.touched.insert((lo, r.end), ("file I/O".to_string(), r.writes));
    }
}

/// Exception handler: an access violation at `addr` (`write`) by code at
/// `rip`. True when it was a probed range (now accessible again).
pub fn fault(addr: u64, write: bool, rip: u64, who: impl FnOnce() -> String) -> bool {
    if !enabled() {
        return false;
    }
    let mut g = STATE.lock();
    let Some((&lo, r)) = g.ranges.range(..=addr).next_back() else { return false };
    if addr >= r.end || !r.protected {
        return false;
    }
    let (hi, writes) = (r.end, r.writes);
    protect(lo, hi, PAGE_READWRITE);
    g.ranges.remove(&lo);
    let accessor = format!("{} of {addr:#x} by {}", if write { "write" } else { "read" }, who());
    crate::info!("wb probe: GPU-written {lo:#x}-{hi:#x} ({} KiB, {writes} write-backs) touched: {accessor} (rip {rip:#x})", (hi - lo) >> 10);
    g.touched.insert((lo, hi), (accessor, writes));
    true
}

/// Report lines: ranges probed and never touched (largest first), ranges touched.
pub fn report() {
    if !enabled() {
        return;
    }
    let g = STATE.lock();
    let mut quiet: Vec<(u64, u64, u64)> = g.ranges.iter().map(|(k, r)| (*k, r.end, r.writes)).collect();
    quiet.sort_by(|a, b| (b.1 - b.0).saturating_mul(b.2).cmp(&(a.1 - a.0).saturating_mul(a.2)));
    let q: Vec<String> = quiet.iter().take(12).map(|(a, b, w)| format!("{a:#x}+{:#x} x{w}", b - a)).collect();
    crate::info!("wb probe: {} ranges untouched since their write-backs (largest traffic first): {}", quiet.len(), q.join(", "));
    let t: Vec<String> = g.touched.iter().take(16).map(|((a, b), (who, w))| format!("{a:#x}+{:#x} after {w}: {who}", b - a)).collect();
    crate::info!("wb probe: {} ranges touched: {}", g.touched.len(), t.join("; "));
}

/// Debug (`TLG_WB_RANGES=1`): every distinct guest range the runtime writes
/// GPU results into (write-backs by kind: 1 marked pages of a mirror range,
/// 2 a large mirror range's pieces, 3 a whole range, 4 a lazy write-back's
/// readback, 5 a render target's tiled write-back) is logged the first time.
pub fn note_range(addr: u64, len: u64, kind: u8) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("TLG_WB_RANGES").is_ok_and(|v| v == "1")) {
        return;
    }
    static SEEN: parking_lot::Mutex<Option<std::collections::HashSet<(u64, u64, u8)>>> = parking_lot::Mutex::new(None);
    let mut g = SEEN.lock();
    let s = g.get_or_insert_with(Default::default);
    if s.len() < 1 << 18 && s.insert((addr, len, kind)) {
        crate::info!("wb range {kind}: {addr:#x}-{:#x} ({len:#x} bytes)", addr + len);
    }
}
