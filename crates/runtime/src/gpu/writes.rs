//! Write tracking for texture validation.
//!
//! Textures sourced from guest memory must be re-uploaded when the title
//! changes their memory. Hashing every texture whenever it might have
//! changed (once per submission) cost a third of the GPU thread in gameplay.
//! The guest range is reserved with `MEM_WRITE_WATCH`, so the OS records
//! which pages are written; a texture is hashed again only when pages in its
//! range were written after it was last validated. Writes the GPU makes into
//! imported guest memory (in-place shader stores) bypass the CPU's page
//! tables and are registered when recorded: they stamp the same pages (whole
//! 4 MiB chunks at once for large ranges).
//!
//! Validation queries (`changed_since`, `cpu_changed_since`) collect the
//! OS's records for whole 64 KiB blocks, each at most once per CPU epoch:
//! within an epoch the renderer treats CPU-written memory as a snapshot (a
//! texture validated in it is not checked again), so later checks of other
//! ranges in a block read the page stamps instead of calling into the
//! kernel (~0.43 us per call; textures and vertex data share blocks).

use super::cpuwatch;
use super::render::FastMap;
use crate::mem;

const PAGE_SHIFT: u32 = 12;
/// Pages are stamped in 4 MiB chunks (1024 host pages).
const CHUNK_SHIFT: u32 = 22;
const CHUNK_PAGES: usize = 1 << (CHUNK_SHIFT - PAGE_SHIFT);
/// Write-watch queries cover 64 KiB blocks (64 per chunk).
const BLOCK_SHIFT: u32 = 16;
const CHUNK_BLOCKS: usize = 1 << (CHUNK_SHIFT - BLOCK_SHIFT);
/// Consecutive unwritten queries after which a block is armed (made
/// read-only, cpuwatch.rs), doubled for each time it was written while armed.
const ARM_AFTER: u32 = 8;

/// Per chunk: blocks armed (one bit each), and per block its consecutive
/// unwritten queries and how often it was written while armed.
struct Watch {
    armed: u64,
    clean: [u16; CHUNK_BLOCKS],
    strikes: [u8; CHUNK_BLOCKS],
}

struct Chunk {
    /// Highest stamp of any page in the chunk.
    max: u64,
    /// Stamp of the last write covering the whole chunk.
    whole: u64,
    /// Stamp of the last harvest that found each page written, or of the last
    /// GPU write to it (0 = never).
    pages: Box<[u64; CHUNK_PAGES]>,
    /// Stamp of the last harvest that found each page written (CPU writes
    /// only) and the highest of them.
    cpu: Box<[u64; CHUNK_PAGES]>,
    cpu_max: u64,
}

impl Chunk {
    fn new() -> Chunk {
        Chunk { max: 0, whole: 0, pages: Box::new([0; CHUNK_PAGES]), cpu: Box::new([0; CHUNK_PAGES]), cpu_max: 0 }
    }
}

pub struct WriteTracker {
    /// Stamp counter: advances with every harvest that finds written pages
    /// and every registered GPU write.
    seq: u64,
    chunks: FastMap<u64, Chunk>,
    buf: Vec<usize>,
    pages: Vec<u64>,
    /// Validations that found the range unwritten (no hashing needed) and
    /// that had to hash.
    pub clean: u64,
    pub dirty: u64,
    /// CPU epoch of each 64 KiB block's last write-watch query, per chunk.
    queried: FastMap<u64, [u64; CHUNK_BLOCKS]>,
    /// The current CPU epoch and the oldest one whose block queries still
    /// answer for the current command (the renderer's `required_epoch`).
    pub epoch: u64,
    pub required: u64,
    /// Validation queries answered from this epoch's block queries alone.
    pub cached: u64,
    /// Write protection of cold blocks (cpuwatch.rs).
    watch: FastMap<u64, Watch>,
    disarmed: Vec<u64>,
    /// Block queries left out because the block was armed (TLG_PERF).
    pub armed_skips: u64,
}

impl WriteTracker {
    pub fn new() -> WriteTracker {
        WriteTracker { seq: 1, chunks: FastMap::default(), buf: Vec::new(), pages: Vec::new(), clean: 0, dirty: 0, queried: FastMap::default(), epoch: 0, required: 0, cached: 0, watch: FastMap::default(), disarmed: Vec::new(), armed_skips: 0 }
    }

    /// Collects the OS's records of [addr, addr+len) for a validation: the
    /// 64 KiB blocks around it not yet queried in a current epoch, in runs
    /// (`tune ww_blocks 0`: exactly the range). False without write watching.
    fn harvest_for_validation(&mut self, addr: u64, len: u64) -> bool {
        let hi = addr.saturating_add(len.max(1));
        if self.epoch == 0 || crate::tune::get(crate::tune::WW_BLOCKS) == 0 || hi < addr || !mem::is_guest(addr, hi - addr) {
            self.pages.clear();
            if !mem::harvest_writes(addr, len, &mut self.buf, &mut self.pages) {
                return false;
            }
            self.stamp_harvest();
            return true;
        }
        let runs = self.blocks_to_query(addr >> BLOCK_SHIFT, (hi - 1) >> BLOCK_SHIFT);
        if runs.is_empty() {
            self.cached += 1;
            return true;
        }
        self.pages.clear();
        for (i, &(lo, hi)) in runs.iter().enumerate() {
            if !mem::harvest_writes(lo << BLOCK_SHIFT, (hi - lo) << BLOCK_SHIFT, &mut self.buf, &mut self.pages) {
                // (Not expected inside the guest range.) The blocks not
                // queried count as unqueried again.
                for &(lo, hi) in &runs[i..] {
                    for b in lo..hi {
                        if let Some(q) = self.queried.get_mut(&(b >> (CHUNK_SHIFT - BLOCK_SHIFT))) {
                            q[b as usize & (CHUNK_BLOCKS - 1)] = 0;
                        }
                    }
                }
                self.stamp_harvest();
                return false;
            }
        }
        if cpuwatch::enabled() {
            self.arm_cold(&runs);
        }
        self.stamp_harvest();
        true
    }

    /// After the queries of `runs` (blocks [first, end)): counts the blocks
    /// found unwritten and arms those quiet long enough (in runs); their
    /// writes since these queries are collected once more after the
    /// protection (later ones fault).
    fn arm_cold(&mut self, runs: &[(u64, u64)]) {
        let mut dirty: smallvec::SmallVec<[u64; 16]> = self.pages.iter().map(|&p| p >> BLOCK_SHIFT).collect();
        dirty.sort_unstable();
        dirty.dedup();
        let mut arm: smallvec::SmallVec<[(u64, u64); 4]> = smallvec::SmallVec::new();
        for &(lo, hi) in runs {
            for b in lo..hi {
                let i = b as usize & (CHUNK_BLOCKS - 1);
                let w = self.watch.entry(b >> (CHUNK_SHIFT - BLOCK_SHIFT)).or_insert_with(|| Watch { armed: 0, clean: [0; CHUNK_BLOCKS], strikes: [0; CHUNK_BLOCKS] });
                if dirty.binary_search(&b).is_ok() {
                    w.clean[i] = 0;
                    continue;
                }
                w.clean[i] = w.clean[i].saturating_add(1);
                if w.armed >> i & 1 == 0 && w.clean[i] as u32 >= ARM_AFTER << w.strikes[i] {
                    match arm.last_mut() {
                        Some(r) if r.1 == b => r.1 = b + 1,
                        _ => arm.push((b, b + 1)),
                    }
                }
            }
        }
        for (lo, hi) in arm {
            if !cpuwatch::arm(lo << BLOCK_SHIFT, hi << BLOCK_SHIFT) {
                // (Not armable: counted from zero again.)
                for b in lo..hi {
                    if let Some(w) = self.watch.get_mut(&(b >> (CHUNK_SHIFT - BLOCK_SHIFT))) {
                        w.clean[b as usize & (CHUNK_BLOCKS - 1)] = 0;
                    }
                }
                continue;
            }
            for b in lo..hi {
                if let Some(w) = self.watch.get_mut(&(b >> (CHUNK_SHIFT - BLOCK_SHIFT))) {
                    w.armed |= 1 << (b as usize & (CHUNK_BLOCKS - 1));
                }
            }
            mem::harvest_writes(lo << BLOCK_SHIFT, (hi - lo) << BLOCK_SHIFT, &mut self.buf, &mut self.pages);
        }
    }

    /// Blocks made writable again by writes or protection changes since
    /// the last look: queried again (and armed later only after a longer
    /// quiet period).
    fn take_disarmed(&mut self) {
        let mut d = std::mem::take(&mut self.disarmed);
        cpuwatch::take_disarmed(&mut d);
        for &b in &d {
            let (c, i) = (b >> (CHUNK_SHIFT - BLOCK_SHIFT), b as usize & (CHUNK_BLOCKS - 1));
            if let Some(w) = self.watch.get_mut(&c) {
                if w.armed >> i & 1 != 0 {
                    w.strikes[i] = (w.strikes[i] + 1).min(6);
                }
                w.armed &= !(1 << i);
                w.clean[i] = 0;
            }
            if let Some(q) = self.queried.get_mut(&c) {
                q[i] = 0;
            }
        }
        d.clear();
        self.disarmed = d;
    }

    /// Runs [first, end) of the blocks first..=last not queried in a current
    /// epoch, marked as queried now.
    fn blocks_to_query(&mut self, first: u64, last: u64) -> smallvec::SmallVec<[(u64, u64); 4]> {
        let watching = cpuwatch::enabled();
        if watching {
            self.take_disarmed();
        }
        let (epoch, required) = (self.epoch, self.required.max(1));
        let mut runs: smallvec::SmallVec<[(u64, u64); 4]> = smallvec::SmallVec::new();
        let mut b = first;
        while b <= last {
            let c = b >> (CHUNK_SHIFT - BLOCK_SHIFT);
            let q = self.queried.entry(c).or_insert([0; CHUNK_BLOCKS]);
            // (Armed blocks: unwritten until a write faults.)
            let armed = if watching { self.watch.get(&c).map_or(0, |w| w.armed) } else { 0 };
            let stop = last.min(((c + 1) << (CHUNK_SHIFT - BLOCK_SHIFT)) - 1);
            while b <= stop {
                let i = b as usize & (CHUNK_BLOCKS - 1);
                if q[i] < required && armed >> i & 1 != 0 {
                    q[i] = epoch;
                    self.armed_skips += 1;
                } else if q[i] < required {
                    q[i] = epoch;
                    match runs.last_mut() {
                        Some(r) if r.1 == b => r.1 = b + 1,
                        _ => runs.push((b, b + 1)),
                    }
                }
                b += 1;
            }
        }
        runs
    }

    /// Records a GPU write to guest memory [lo, lo+len) (recorded before it
    /// executes; readers synchronize with in-flight GPU writes before they
    /// validate, so the stamp is never older than the data they see).
    pub fn note_gpu_write(&mut self, lo: u64, len: u64) {
        self.seq += 1;
        let s = self.seq;
        let hi = lo + len.max(1);
        let mut c = lo >> CHUNK_SHIFT;
        while (c << CHUNK_SHIFT) < hi {
            let base = c << CHUNK_SHIFT;
            let end = base + (1 << CHUNK_SHIFT);
            let ch = self.chunks.entry(c).or_insert_with(Chunk::new);
            if lo <= base && hi >= end {
                ch.whole = s;
            } else {
                let first = ((lo.max(base) - base) >> PAGE_SHIFT) as usize;
                let last = ((hi.min(end) - base + (1 << PAGE_SHIFT) - 1) >> PAGE_SHIFT) as usize;
                for p in &mut ch.pages[first..last.min(CHUNK_PAGES)] {
                    *p = s;
                }
            }
            ch.max = s;
            c += 1;
        }
    }

    /// Stamps the pages a harvest found written.
    fn stamp_harvest(&mut self) {
        if self.pages.is_empty() {
            return;
        }
        self.seq += 1;
        let s = self.seq;
        for &p in &self.pages {
            let c = self.chunks.entry(p >> CHUNK_SHIFT).or_insert_with(Chunk::new);
            let i = ((p >> PAGE_SHIFT) as usize) & (CHUNK_PAGES - 1);
            c.pages[i] = s;
            c.max = s;
            c.cpu[i] = s;
            c.cpu_max = s;
        }
    }

    /// Whether the CPU (title or host code, not the GPU) wrote pages of
    /// [addr, addr+len) after stamp `since`, collecting the OS's records for
    /// the range first; also returns the stamp to compare later checks with.
    pub fn cpu_written_since(&mut self, addr: u64, len: u64, since: u64) -> (bool, u64) {
        self.pages.clear();
        if !mem::harvest_writes(addr, len, &mut self.buf, &mut self.pages) {
            return (false, self.seq);
        }
        self.stamp_harvest();
        let hi = addr + len.max(1);
        let mut c = addr >> CHUNK_SHIFT;
        while (c << CHUNK_SHIFT) < hi {
            if let Some(ch) = self.chunks.get(&c).filter(|ch| ch.cpu_max > since) {
                let base = c << CHUNK_SHIFT;
                let first = ((addr.max(base) - base) >> PAGE_SHIFT) as usize;
                let last = ((hi.min(base + (1 << CHUNK_SHIFT)) - base + (1 << PAGE_SHIFT) - 1) >> PAGE_SHIFT) as usize;
                if ch.cpu[first..last.min(CHUNK_PAGES)].iter().any(|&s| s > since) {
                    return (true, self.seq);
                }
            }
            c += 1;
        }
        (false, self.seq)
    }

    /// Collects the pages of [addr, addr+len) the OS saw written since they
    /// were last collected and reports whether anything in the range may have
    /// been written after stamp `since`. Returns (written, stamp): `stamp` is
    /// what to remember once the range has been validated (hashed) again.
    /// Without write watching every range counts as written.
    pub fn changed_since(&mut self, addr: u64, len: u64, since: u64) -> (bool, u64) {
        if !self.harvest_for_validation(addr, len) {
            self.seq += 1;
            self.dirty += 1;
            return (true, self.seq);
        }
        let written = self.pages_after(addr, addr + len, since);
        if written {
            self.dirty += 1;
        } else {
            self.clean += 1;
        }
        (written, self.seq)
    }

    /// Like `changed_since`, but for CPU writes only (the pages the OS saw
    /// written; registered GPU writes do not count: callers track those
    /// themselves, e.g. the VRAM mirror, which the GPU's own writes keep
    /// current). Without write watching every range counts as written.
    /// (`with_gpu`: registered GPU writes count too.)
    pub fn cpu_changed_since(&mut self, addr: u64, len: u64, since: u64, with_gpu: bool) -> (bool, u64) {
        if !self.harvest_for_validation(addr, len) {
            self.seq += 1;
            return (true, self.seq);
        }
        let written = if with_gpu { self.pages_after(addr, addr + len, since) } else { self.cpu_pages_after(addr, addr + len, since) };
        (written, self.seq)
    }

    /// The current stamp: writes registered or harvested later have higher
    /// ones (`gpu_written_since`).
    pub fn stamp(&self) -> u64 {
        self.seq
    }

    /// Whether GPU writes (and CPU writes registered like them) recorded
    /// after stamp `since` overlap [addr, addr+len), without a write-watch
    /// query: within one CPU epoch nothing else changes guest memory.
    pub fn gpu_written_since(&self, addr: u64, len: u64, since: u64) -> bool {
        self.pages_after(addr, addr + len, since)
    }

    /// Merged runs of [lo, hi) with pages written after stamp `since`: by
    /// the CPU (the OS's records) or by registered GPU writes.
    pub fn dirty_runs(&self, lo: u64, hi: u64, since: u64) -> Vec<(u64, u64)> {
        self.runs_after(lo, hi, since, false)
    }

    /// Merged runs of [lo, hi) with pages the OS saw written after stamp
    /// `since` (CPU writes only, see `cpu_changed_since`).
    pub fn cpu_dirty_runs(&self, lo: u64, hi: u64, since: u64) -> Vec<(u64, u64)> {
        self.runs_after(lo, hi, since, true)
    }

    fn runs_after(&self, lo: u64, hi: u64, since: u64, cpu_only: bool) -> Vec<(u64, u64)> {
        let mut runs: Vec<(u64, u64)> = Vec::new();
        let mut c = lo >> CHUNK_SHIFT;
        while (c << CHUNK_SHIFT) < hi {
            if let Some(ch) = self.chunks.get(&c).filter(|ch| if cpu_only { ch.cpu_max > since } else { ch.max > since }) {
                let base = c << CHUNK_SHIFT;
                if !cpu_only && ch.whole > since {
                    runs.push((lo.max(base), hi.min(base + (1 << CHUNK_SHIFT))));
                    c += 1;
                    continue;
                }
                let stamps = if cpu_only { &ch.cpu } else { &ch.pages };
                let first = ((lo.max(base) - base) >> PAGE_SHIFT) as usize;
                let last = (((hi.min(base + (1 << CHUNK_SHIFT)) - base) + (1 << PAGE_SHIFT) - 1) >> PAGE_SHIFT) as usize;
                for p in first..last.min(CHUNK_PAGES) {
                    if stamps[p] > since {
                        let a = base + ((p as u64) << PAGE_SHIFT);
                        runs.push((a.max(lo), (a + (1 << PAGE_SHIFT)).min(hi)));
                    }
                }
            }
            c += 1;
        }
        runs.sort_unstable();
        let mut out: Vec<(u64, u64)> = Vec::with_capacity(runs.len());
        for (a, b) in runs {
            match out.last_mut() {
                Some(m) if a <= m.1 => m.1 = m.1.max(b),
                _ => out.push((a, b)),
            }
        }
        out
    }

    /// Whether the OS saw any page of [lo, hi) written after stamp `since`.
    fn cpu_pages_after(&self, lo: u64, hi: u64, since: u64) -> bool {
        let mut c = lo >> CHUNK_SHIFT;
        while (c << CHUNK_SHIFT) < hi {
            if let Some(ch) = self.chunks.get(&c).filter(|ch| ch.cpu_max > since) {
                let base = c << CHUNK_SHIFT;
                let first = ((lo.max(base) - base) >> PAGE_SHIFT) as usize;
                let last = ((hi.min(base + (1 << CHUNK_SHIFT)) - base + (1 << PAGE_SHIFT) - 1) >> PAGE_SHIFT) as usize;
                if ch.cpu[first..last.min(CHUNK_PAGES)].iter().any(|&s| s > since) {
                    return true;
                }
            }
            c += 1;
        }
        false
    }

    /// Whether any page of [lo, hi) carries a stamp after `since`.
    fn pages_after(&self, lo: u64, hi: u64, since: u64) -> bool {
        let mut c = lo >> CHUNK_SHIFT;
        while (c << CHUNK_SHIFT) < hi {
            if let Some(ch) = self.chunks.get(&c) {
                if ch.whole > since {
                    return true;
                }
                if ch.max > since {
                    let base = c << CHUNK_SHIFT;
                    let first = ((lo.max(base) - base) >> PAGE_SHIFT) as usize;
                    let last = ((hi.min(base + (1 << CHUNK_SHIFT)) - base + (1 << PAGE_SHIFT) - 1) >> PAGE_SHIFT) as usize;
                    if ch.pages[first..last.min(CHUNK_PAGES)].iter().any(|&s| s > since) {
                        return true;
                    }
                }
            }
            c += 1;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_hash_detects_changes() {
        use super::super::resources::hash_slice;
        let mut d: Vec<u8> = (0..4099u32).map(|i| (i * 31 + 7) as u8).collect();
        let h = hash_slice(0, &d);
        assert_eq!(h, hash_slice(0, &d));
        assert_ne!(h, hash_slice(1, &d));
        for &at in &[0usize, 31, 32, 4095, 4098] {
            d[at] ^= 1;
            assert_ne!(h, hash_slice(0, &d), "bit flip at {at}");
            d[at] ^= 1;
        }
        assert_ne!(hash_slice(0, &d[..4098]), hash_slice(0, &d[..4097]));
        assert_ne!(hash_slice(0, &[]), hash_slice(0, &[0]));
    }

    #[test]
    fn blocks_queried_once_per_epoch() {
        let mut t = WriteTracker::new();
        t.epoch = 5;
        t.required = 5;
        let r = |v: &[(u64, u64)]| smallvec::SmallVec::<[(u64, u64); 4]>::from_slice(v);
        assert_eq!(t.blocks_to_query(10, 12), r(&[(10, 13)]));
        // Only the blocks not queried yet, as runs.
        assert_eq!(t.blocks_to_query(11, 14), r(&[(13, 15)]));
        assert_eq!(t.blocks_to_query(9, 15), r(&[(9, 10), (15, 16)]));
        assert_eq!(t.blocks_to_query(10, 15), r(&[]));
        // Across a chunk boundary (64 blocks per chunk).
        assert_eq!(t.blocks_to_query(62, 66), r(&[(62, 67)]));
        // A new epoch: queried again. A compute-queue command whose required
        // epoch is older still accepts the epoch-6 queries.
        t.epoch = 6;
        t.required = 6;
        assert_eq!(t.blocks_to_query(10, 11), r(&[(10, 12)]));
        t.required = 5;
        assert_eq!(t.blocks_to_query(9, 12), r(&[]));
        t.required = 6;
        assert_eq!(t.blocks_to_query(9, 12), r(&[(9, 10), (12, 13)]));
    }

    #[test]
    fn gpu_writes_and_page_stamps() {
        let mut t = WriteTracker::new();
        // Page stamps as a harvest would record them.
        t.seq = 10;
        let base = 0x1000_0000_0000u64;
        let c = t.chunks.entry(base >> CHUNK_SHIFT).or_insert_with(Chunk::new);
        c.pages[3] = 10;
        c.max = 10;
        assert!(t.pages_after(base, base + 0x4000, 9));
        assert!(!t.pages_after(base, base + 0x4000, 10));
        assert!(!t.pages_after(base + 0x4000, base + 0x8000, 0));
        // A page that straddles the end of a range counts.
        assert!(t.pages_after(base + 0x3800, base + 0x3900, 0));
        // GPU writes stamp the pages they touch.
        t.note_gpu_write(base + 0x10400, 0x10);
        let s = t.seq;
        assert!(t.gpu_written_since(base + 0x10000, 0x1000, s - 1));
        assert!(!t.gpu_written_since(base + 0x10000, 0x1000, s));
        assert!(!t.gpu_written_since(base + 0x11000, 0x1000, s - 1));
        // Whole chunks at once; the dirty runs cover them.
        t.note_gpu_write(base + (1 << CHUNK_SHIFT), 3 << CHUNK_SHIFT);
        let s2 = t.seq;
        let mid = base + (2 << CHUNK_SHIFT);
        assert!(t.gpu_written_since(mid + 0x5000, 4, s2 - 1));
        assert!(!t.gpu_written_since(mid + 0x5000, 4, s2));
        assert_eq!(t.dirty_runs(mid, mid + 0x3000, s2 - 1), vec![(mid, mid + 0x3000)]);
        assert!(!t.gpu_written_since(base + (4 << CHUNK_SHIFT), 4, s2 - 1));
        // CPU-only views ignore the registered GPU writes.
        assert!(t.cpu_dirty_runs(mid, mid + 0x3000, 0).is_empty());
        assert!(!t.cpu_pages_after(base + 0x10000, base + 0x11000, 0));
        // A harvested CPU write shows in both.
        t.pages = vec![mid + 0x1000];
        t.stamp_harvest();
        let s3 = t.seq;
        assert_eq!(t.cpu_dirty_runs(mid, mid + 0x3000, s3 - 1), vec![(mid + 0x1000, mid + 0x2000)]);
        assert!(t.cpu_pages_after(mid, mid + 0x3000, s3 - 1));
        assert!(!t.cpu_pages_after(mid, mid + 0x3000, s3));
        assert_eq!(t.dirty_runs(mid, mid + 0x3000, s3 - 1), vec![(mid + 0x1000, mid + 0x2000)]);
    }
}
