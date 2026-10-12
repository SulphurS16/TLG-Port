//! Command processor: executes PM4 streams. Rendering hooks are called for
//! draws/dispatches; memory/synchronization packets are executed exactly.

use super::pm4::{self, op};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub const CE_RAM_DWORDS: usize = 0x3000;

pub struct Regs {
    pub ctx: Vec<u32>,
    pub sh: Vec<u32>,
    pub uconfig: Vec<u32>,
    pub config: Vec<u32>,
}

impl Regs {
    pub fn new() -> Regs {
        Regs { ctx: vec![0; 0x400], sh: vec![0; 0x400], uconfig: vec![0; 0x400], config: vec![0; 0xC00] }
    }
    pub fn ctx(&self, reg: u32) -> u32 {
        self.ctx[(reg - pm4::CONTEXT_REG_BASE) as usize]
    }
    pub fn sh(&self, reg: u32) -> u32 {
        self.sh[(reg - pm4::SH_REG_BASE) as usize]
    }
}

#[derive(Default, Clone, Debug)]
pub struct Stats {
    pub packets: u64,
    pub draws: u64,
    pub dispatches: u64,
    pub eops: u64,
    pub waits: u64,
    /// Memory waits met by our own in-flight end-of-pipe writes (no stall).
    pub virtual_waits: u64,
    /// Memory waits met by this CP's own queued (not yet committed)
    /// end-of-pipe writes: ordered by a barrier within the current batch.
    pub inbatch_waits: u64,
    pub unknown: u64,
    /// Memory-to-memory DMA_DATA recorded as GPU copies (see `Hooks::dma_on_gpu`).
    pub dma_gpu: u64,
    /// DISPATCH_INDIRECT with arguments read on the GPU (`Hooks::dispatch_indirect`).
    pub indirect_on_gpu: u64,
}

/// Command words fetched at submission time, keyed by guest start address.
/// The emulated CP fetches a submitted command buffer (and the indirect
/// buffers it calls) when it is submitted, so later CPU reuse of that memory
/// cannot alter queued work however far the GPU thread lags behind.
pub type Captured = HashMap<u64, Arc<Vec<u32>>>;

struct Frame {
    cur: u64,
    end: u64,
    base: u64,
    data: Option<Arc<Vec<u32>>>,
}

/// An end-of-pipe/shader event: memory write and/or interrupt, performed
/// once all GPU work submitted before it has completed.
pub struct EopAction {
    pub addr: u64,
    pub data_sel: u32,
    pub lo: u32,
    pub hi: u32,
    pub interrupt: bool,
    /// Raised by the graphics queue (compute-queue interrupts are not
    /// graphics EOP events).
    pub gfx: bool,
    /// `OCCL_DUMP` actions: the occlusion queries counted since the previous
    /// dump (see `render::occlusion_write`).
    pub occl: Option<Box<[u32]>>,
}

/// `EopAction::data_sel` of a PIXEL_PIPE_STAT_DUMP: the occlusion counters
/// of the 8 depth blocks, 64-bit each with the valid bit 63, at a 16-byte
/// stride (a query's begin dump at +0, its end dump at +8).
pub const OCCL_DUMP: u32 = 0x100;

impl EopAction {
    /// Guest bytes the action writes.
    pub fn span(&self) -> Option<(u64, u64)> {
        match self.data_sel {
            1 => Some((self.addr, 4)),
            2..=4 => Some((self.addr, 8)),
            OCCL_DUMP => Some((self.addr, 8 * 16)),
            _ => None,
        }
    }

    pub fn overlaps(&self, addr: u64, len: u64) -> bool {
        self.span().is_some_and(|(a, n)| a < addr + len && addr < a + n)
    }

    /// Performs the write and interrupt (completion time).
    pub fn perform(&self) {
        if self.data_sel == OCCL_DUMP {
            super::render::occlusion_write(self.addr, self.occl.as_deref().unwrap_or(&[]));
            return;
        }
        let (n, val) = match self.data_sel {
            0 => (0, 0),
            1 => (4, self.lo as u64),
            2 => (8, (self.lo as u64) | ((self.hi as u64) << 32)),
            3 | 4 => (8, unsafe { core::arch::x86_64::_rdtsc() }),
            other => {
                crate::warn!("EOP data_sel {other} unsupported");
                (0, 0)
            }
        };
        if n != 0 {
            if guest_ok(self.addr, n) {
                unsafe {
                    if n == 4 {
                        std::ptr::write_volatile(self.addr as *mut u32, val as u32);
                    } else {
                        std::ptr::write_volatile(self.addr as *mut u64, val);
                    }
                }
            } else {
                crate::error!("EOP write to invalid address {:#x}", self.addr);
            }
        }
        if self.interrupt && self.gfx {
            super::graphics_eop_interrupt();
        }
    }
}

/// One PM4 instruction stream with an indirect-buffer call stack, followed
/// by further top-level buffers (the DCBs or CCBs of one submission run as
/// one stream per engine, as the hardware queues them on its ring).
pub struct Stream {
    frames: Vec<Frame>,
    queue: std::collections::VecDeque<(u64, u32)>,
    captured: Arc<Captured>,
}

impl Stream {
    /// Stream read live from guest memory.
    pub fn new(addr: u64, dwords: u32) -> Stream {
        Stream::with_capture(addr, dwords, Arc::new(Captured::new()))
    }
    /// Stream executing captured command words where available.
    pub fn with_capture(addr: u64, dwords: u32, captured: Arc<Captured>) -> Stream {
        let mut s = Stream::chained(captured);
        s.push(addr, dwords);
        s
    }
    /// Empty stream; top-level buffers are added with `chain`.
    pub fn chained(captured: Arc<Captured>) -> Stream {
        Stream { frames: Vec::new(), queue: Default::default(), captured }
    }
    pub fn empty() -> Stream {
        Stream::chained(Arc::new(Captured::new()))
    }
    /// Appends a top-level buffer, run after everything before it.
    pub fn chain(&mut self, addr: u64, dwords: u32) {
        if addr != 0 && dwords != 0 {
            self.queue.push_back((addr, dwords));
        }
    }
    /// Start of the next queued top-level buffer (diagnostics).
    fn next_root(&self) -> Option<u64> {
        self.queue.front().map(|q| q.0)
    }
    fn push(&mut self, addr: u64, dwords: u32) {
        let data = self.captured.get(&addr).filter(|d| d.len() >= dwords as usize).cloned();
        self.frames.push(Frame { cur: addr, end: addr + dwords as u64 * 4, base: addr, data });
    }
    fn top(&self) -> (u64, u64) {
        let f = self.frames.last().unwrap();
        (f.cur, f.end)
    }
    fn set_cur(&mut self, cur: u64) {
        self.frames.last_mut().unwrap().cur = cur;
    }
    /// Command word at `addr` in the current buffer.
    fn word(&self, addr: u64) -> u32 {
        if let Some(f) = self.frames.last() {
            if let Some(d) = &f.data {
                let i = ((addr - f.base) / 4) as usize;
                if addr >= f.base && i < d.len() {
                    return d[i];
                }
            }
        }
        unsafe { rd(addr) }
    }
    fn done(&mut self) -> bool {
        loop {
            while let Some(f) = self.frames.last() {
                if f.cur < f.end {
                    return false;
                }
                self.frames.pop();
            }
            match self.queue.pop_front() {
                Some((a, n)) => self.push(a, n),
                None => return true,
            }
        }
    }
}

/// Fetches the command words of `roots` and, transitively, of the indirect
/// buffers they call.
pub fn capture(roots: &[(u64, u32)]) -> Captured {
    const LIMIT_DWORDS: usize = 16 << 20;
    let mut out = Captured::new();
    let mut work: Vec<(u64, u32, u32)> = roots.iter().map(|&(a, n)| (a, n, 0)).collect();
    let mut total = 0usize;
    while let Some((addr, dw, depth)) = work.pop() {
        if dw == 0 || out.contains_key(&addr) || !guest_ok(addr, dw as u64 * 4) || total + dw as usize > LIMIT_DWORDS {
            continue;
        }
        let mut bytes = vec![0u8; dw as usize * 4];
        super::resources::copy_from_guest(&mut bytes, addr);
        let words: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let mut i = 0usize;
        while i < words.len() {
            let h = words[i];
            match h >> 30 {
                0 => i += 2 + ((h >> 16) & 0x3FFF) as usize,
                2 => i += 1,
                3 => {
                    let n = ((h >> 16) & 0x3FFF) as usize + 1;
                    let opcode = (h >> 8) & 0xFF;
                    if (opcode == op::INDIRECT_BUFFER || opcode == op::INDIRECT_BUFFER_CONST) && i + 3 < words.len() && depth < 8 {
                        let a = (words[i + 1] as u64 & !3) | ((words[i + 2] as u64 & 0xFFFF) << 32);
                        work.push((a, words[i + 3] & 0xFFFFF, depth + 1));
                    }
                    i += 1 + n;
                }
                _ => break,
            }
        }
        total += words.len();
        out.insert(addr, Arc::new(words));
    }
    out
}

#[derive(PartialEq, Eq, Debug)]
pub enum Engine {
    Draw,
    Constant,
    Compute,
}

enum Step {
    Ok,
    Blocked,
    Done,
}

/// Transfers to or from the global data share, performed in order with the
/// GPU work recorded before them.
#[derive(Clone, Copy, Debug)]
pub enum GdsOp {
    /// GDS [offset, offset+bytes) = value (per dword).
    Fill { offset: u32, bytes: u32, value: u32 },
    /// Guest memory [src, src+bytes) -> GDS at offset.
    Load { offset: u32, src: u64, bytes: u32 },
    /// GDS [offset, offset+bytes) -> guest memory at dst.
    Store { offset: u32, dst: u64, bytes: u32 },
}

pub trait Hooks {
    fn draw(&mut self, cp: &CommandProcessor, kind: &'static str, count: u32, index_addr: u64);
    fn dispatch(&mut self, cp: &CommandProcessor, x: u32, y: u32, z: u32);
    /// DISPATCH_INDIRECT whose arguments at guest `addr` the GPU reads in
    /// queue order (no CP writes of them pending); false when the renderer
    /// cannot (the CP then waits for them and dispatches directly).
    fn dispatch_indirect(&mut self, _cp: &CommandProcessor, _addr: u64) -> bool {
        false
    }
    /// Completes all outstanding GPU work and end-of-pipe actions (blocking).
    fn flush(&mut self, cp: &CommandProcessor);
    /// Hands end-of-pipe actions over to be performed after all work recorded
    /// so far has completed (may return before that happens).
    fn commit(&mut self, cp: &CommandProcessor, actions: Vec<EopAction>);
    /// Waits for submitted work whose completion writes overlap guest memory
    /// [addr, addr+len) (end-of-pipe writes, buffer write-backs).
    fn sync_range(&mut self, cp: &CommandProcessor, addr: u64, len: u64);
    /// Before the CP writes guest memory [addr, addr+len): waits for recorded
    /// GPU work that still reads that memory in place.
    fn sync_write(&mut self, cp: &CommandProcessor, addr: u64, len: u64) {
        let _ = (cp, addr, len);
    }
    /// DMA_DATA memory -> memory: records the copy on the GPU, in order with
    /// the work recorded so far, when that work wrote the source (instead of
    /// the CP waiting for it). False: the CP copies.
    fn dma_on_gpu(&mut self, cp: &CommandProcessor, src: u64, dst: u64, bytes: u64) -> bool {
        let _ = (cp, src, dst, bytes);
        false
    }
    /// DMA_DATA memory -> GDS: records the load on the GPU when work
    /// recorded so far wrote the source (see `dma_on_gpu`). False: the CP
    /// synchronizes and the load reads guest memory as usual.
    fn gds_load_on_gpu(&mut self, cp: &CommandProcessor, offset: u32, src: u64, bytes: u32) -> bool {
        let _ = (cp, offset, src, bytes);
        false
    }
    /// A memory wait met by this CP's own queued end-of-pipe writes: work
    /// recorded after this point must execute after all work recorded before
    /// it (a full barrier within the current batch).
    fn order_barrier(&mut self, cp: &CommandProcessor) {
        let _ = cp;
    }
    /// Whether guest memory [addr, addr+len) is touched only by end-of-pipe
    /// writes (no shader or copy work recorded or in flight reads or writes
    /// it): a CP write ordered behind those can be queued behind them.
    fn label_only(&mut self, cp: &CommandProcessor, addr: u64, len: u64) -> bool {
        let _ = (cp, addr, len);
        false
    }
    /// GDS transfer (DMA_DATA to/from GDS, end-of-shader GDS store).
    fn gds(&mut self, cp: &CommandProcessor, op: GdsOp) {
        let _ = cp;
        crate::warn!("GDS transfer without a renderer ignored: {op:?}");
    }
    /// PIXEL_PIPE_STAT_DUMP at this point of the command stream (`begin`:
    /// a query's begin sample): the occlusion queries counted since the
    /// previous dump, whose results its write adds up.
    fn occlusion_dump(&mut self, cp: &CommandProcessor, begin: bool) -> Option<Vec<u32>> {
        let _ = (cp, begin);
        None
    }
}

pub struct CommandProcessor {
    pub name: String,
    /// Report activity to the GPU stall watchdog (graphics queue only).
    pub watch: bool,
    /// Compute virtual queue id (0 = graphics), for the stall watchdog.
    pub queue: usize,
    /// End-of-pipe writes/interrupts not yet handed to the renderer (see `commit`).
    pending: Vec<EopAction>,
    pub regs: Regs,
    pub ce_ram: Vec<u32>,
    pub ce_count: u64,
    pub de_count: u64,
    /// Let the next draw/constant engine counter wait pass (it cannot be
    /// satisfied within the submission).
    skip_de_wait: bool,
    skip_ce_wait: bool,
    /// Address and engine of the packet being executed (diagnostics).
    cur_packet: (u64, bool),
    pub index_base: u64,
    pub index_type: u32,
    pub index_size: u32,
    /// Indices the current indexed draw may fetch (its packet's MAX_SIZE,
    /// counted from the draw's first index; u32::MAX = no limit): beyond it
    /// the VGT fetches index 0 instead of reading past the buffer.
    pub draw_index_valid: u32,
    pub num_instances: u32,
    pub indirect_base: u64,
    pub stats: Stats,
    pub markers: Vec<String>,
    pub trace: bool,
}

fn guest_ok(addr: u64, len: u64) -> bool {
    crate::mem::is_guest(addr, len)
}

unsafe fn rd(addr: u64) -> u32 {
    unsafe { std::ptr::read_volatile(addr as *const u32) }
}

/// Type-3 packets executed per opcode (graphics queue; +256 compute queues).
pub static OPCODES: [std::sync::atomic::AtomicU64; 512] = [const { std::sync::atomic::AtomicU64::new(0) }; 512];

/// EVENT_WRITE packets by VGT event type. They are not acted on: caches are
/// coherent and partial flushes are implied by our ordering. Counter samples
/// would need results: occlusion queries (ZPASS_DONE 0x15; on this GPU the
/// title uses PIXEL_PIPE_STAT_CONTROL/DUMP 0x38/0x39, a dump writing 8 DB
/// counters at a 16-byte stride) and pipeline statistics (0x1E, 0x20).
pub static EVENT_TYPES: [std::sync::atomic::AtomicU64; 64] = [const { std::sync::atomic::AtomicU64::new(0) }; 64];

/// CP memory accesses that waited for GPU work (or its write-backs), by
/// cp.rs line of the packet handler: (count, microseconds) (TLG_PERF report).
pub static SYNC_SITES: parking_lot::Mutex<Option<HashMap<u32, (u64, u64)>>> = parking_lot::Mutex::new(None);

/// WRITE_DATA packets queued behind end-of-pipe writes (`defer_label_write`).
pub static DEFERRED_WRITES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// WAIT_REG_MEM polls over 0.5 ms (TLG_PERF report): (queue, function,
/// reference, value at the start, value when met) -> (count, microseconds).
pub type PollKey = (usize, u32, u32, u32, u32);
pub static POLLS: parking_lot::Mutex<Option<std::collections::HashMap<PollKey, (u64, u64)>>> = parking_lot::Mutex::new(None);

/// TLG_WD_TRACE=1: WRITE_DATA packets whose ordering with earlier GPU work
/// made the CP wait (address, data, the end-of-pipe writes involved).
fn wd_trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("TLG_WD_TRACE").is_some())
}

fn note_sync(line: u32, t0: Instant) {
    let us = t0.elapsed().as_micros() as u64;
    if us >= 10 {
        let mut g = SYNC_SITES.lock();
        let e = g.get_or_insert_with(HashMap::new).entry(line).or_default();
        e.0 += 1;
        e.1 += us;
    }
}

/// Nonzero `EVENT_TYPES` as "type:count" pairs.
pub fn event_type_counts() -> String {
    let v: Vec<String> = EVENT_TYPES
        .iter()
        .enumerate()
        .filter_map(|(i, n)| {
            let n = n.load(std::sync::atomic::Ordering::Relaxed);
            (n != 0).then(|| format!("{i:#04x}:{n}"))
        })
        .collect();
    v.join(" ")
}

/// Logs the dwords around a malformed packet (first few occurrences only).
/// Whether register writes are logged (TLG_WATCH_REG); cached.
#[inline]
fn reg_watching() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static STATE: AtomicU8 = AtomicU8::new(0);
    match STATE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = std::env::var_os("TLG_WATCH_REG").is_some();
            STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

fn dump_context(cur: u64, end: u64) {
    static COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= 4 {
        return;
    }
    let start = if crate::mem::is_guest(cur.saturating_sub(64), 64) { cur.saturating_sub(64) } else { cur };
    let mut s = String::new();
    let mut a = start;
    while a < (cur + 64).min(end + 32) {
        if guest_ok(a, 4) {
            s.push_str(&format!("{}{:08x} ", if a == cur { ">" } else { "" }, unsafe { rd(a) }));
        }
        a += 4;
    }
    crate::error!("  context from {start:#x}: {s}");
}

impl CommandProcessor {
    pub fn new(name: &str) -> CommandProcessor {
        CommandProcessor {
            name: name.to_string(),
            watch: name == "gfx",
            queue: 0,
            pending: Vec::new(),
            regs: Regs::new(),
            ce_ram: vec![0; CE_RAM_DWORDS],
            ce_count: 0,
            de_count: 0,
            skip_de_wait: false,
            skip_ce_wait: false,
            cur_packet: (0, false),
            index_base: 0,
            index_type: 0,
            index_size: 0,
            draw_index_valid: u32::MAX,
            num_instances: 1,
            indirect_base: 0,
            stats: Stats::default(),
            markers: Vec::new(),
            trace: false,
        }
    }

    /// Runs a draw stream and an optional constant stream to completion,
    /// interleaving them at CE/DE counter synchronization points.
    pub fn run(&mut self, de: &mut Stream, ce: &mut Stream, hooks: &mut dyn Hooks) {
        loop {
            // Run the constant engine until it blocks or finishes.
            let ce_blocked = loop {
                match self.step(ce, Engine::Constant, hooks) {
                    Step::Ok => {}
                    Step::Blocked => break true,
                    Step::Done => break false,
                }
            };
            let ce_done = ce.done();
            // Run the draw engine until it blocks or finishes.
            let de_blocked = loop {
                match self.step(de, Engine::Draw, hooks) {
                    Step::Ok => {}
                    Step::Blocked => break true,
                    Step::Done => break false,
                }
                // Let the CE make progress after each DE step.
                if ce_blocked && !ce_done {
                    break false;
                }
            };
            let de_done = de.done();
            if de_done && ce_done {
                return;
            }
            if de_blocked && (ce_blocked || ce_done) {
                // The other engine cannot make progress within this
                // submission: release the draw engine's wait (constants it
                // waits for may be stale) instead of dropping its commands.
                static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 10 {
                    crate::warn!(
                        "{}: CE/DE counter wait cannot be satisfied in this submission (ce_count {}, de_count {}, CE {}); releasing the DE wait",
                        self.name,
                        self.ce_count,
                        self.de_count,
                        if ce_done { "finished" } else { "blocked" }
                    );
                }
                self.skip_de_wait = true;
                continue;
            }
            if de_done && ce_blocked {
                static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 10 {
                    crate::warn!("{}: CE blocked after DE finished (ce_count {}, de_count {}); releasing the CE wait", self.name, self.ce_count, self.de_count);
                }
                self.skip_ce_wait = true;
                continue;
            }
        }
    }

    /// Runs a single stream (compute rings).
    pub fn run_single(&mut self, s: &mut Stream, engine: Engine, hooks: &mut dyn Hooks) {
        loop {
            match self.step(s, engine_clone(&engine), hooks) {
                Step::Ok => {}
                Step::Blocked => {
                    crate::error!("{}: stream blocked on CE/DE counter in single-stream mode", self.name);
                    return;
                }
                Step::Done => return,
            }
        }
    }

    fn step(&mut self, s: &mut Stream, engine: Engine, hooks: &mut dyn Hooks) -> Step {
        if s.done() {
            return Step::Done;
        }
        let (cur, end) = s.top();
        if !guest_ok(cur, 4) {
            crate::error!("{}: command pointer {:#x} outside guest memory", self.name, cur);
            s.frames.clear();
            return Step::Done;
        }
        if self.watch {
            super::set_phase(super::phase::PACKET, cur);
        } else if self.queue != 0 {
            super::set_qphase(self.queue, super::phase::PACKET, cur);
        }
        let header = s.word(cur);
        let ptype = header >> 30;
        self.cur_packet = (cur, engine == Engine::Constant);
        match ptype {
            2 => {
                s.set_cur(cur + 4);
                return Step::Ok;
            }
            0 => {
                let base = header & 0xFFFF;
                let count = ((header >> 16) & 0x3FFF) + 1;
                for i in 0..count {
                    let v = s.word(cur + 4 + i as u64 * 4);
                    self.write_reg(base + i, v);
                }
                s.set_cur(cur + 4 + count as u64 * 4);
                self.stats.packets += 1;
                return Step::Ok;
            }
            3 => {}
            _ => {
                crate::error!("{}: unsupported PM4 packet type {} ({:#010x}) at {:#x} (frame end {:#x}, depth {})", self.name, ptype, header, cur, end, s.frames.len());
                dump_context(cur, end);
                s.set_cur(end);
                return Step::Ok;
            }
        }
        let count = ((header >> 16) & 0x3FFF) + 1;
        let opcode = (header >> 8) & 0xFF;
        if header & 1 != 0 {
            // PREDICATE bit: executed only while predication passes (ignored).
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 10 {
                crate::info!("{}: predicated packet {:#04x} at {:#x} (predication not emulated: executed)", self.name, opcode, cur);
            }
        }
        let next = cur + 4 + count as u64 * 4;
        if next > end {
            crate::error!(
                "{}: packet {:#x} at {:#x} overruns its buffer (frame end {:#x}, depth {}, next top-level buffer {:x?})",
                self.name,
                opcode,
                cur,
                end,
                s.frames.len(),
                s.next_root()
            );
            dump_context(cur, end);
            s.set_cur(end);
            return Step::Ok;
        }
        // (Most packets are short: no allocation, no initialization.)
        let mut small = [const { std::mem::MaybeUninit::<u32>::uninit() }; 128];
        let big: Vec<u32>;
        let body: &[u32] = if count as usize <= small.len() {
            for (i, w) in small[..count as usize].iter_mut().enumerate() {
                w.write(s.word(cur + 4 + i as u64 * 4));
            }
            // SAFETY: the first `count` elements were just written.
            unsafe { std::slice::from_raw_parts(small.as_ptr() as *const u32, count as usize) }
        } else {
            big = (0..count).map(|i| s.word(cur + 4 + i as u64 * 4)).collect();
            &big
        };
        // Counter waits do not advance the stream when blocked.
        let blocked = match opcode {
            op::WAIT_ON_CE_COUNTER => self.ce_count <= self.de_count,
            op::WAIT_ON_DE_COUNTER_DIFF => self.ce_count.saturating_sub(self.de_count) >= (body[0] as u64).max(1),
            _ => false,
        };
        if blocked {
            let skip = if engine == Engine::Draw { &mut self.skip_de_wait } else { &mut self.skip_ce_wait };
            if !*skip {
                return Step::Blocked;
            }
            *skip = false;
        }
        s.set_cur(next);
        self.stats.packets += 1;
        OPCODES[(opcode as usize) | if self.watch { 0 } else { 256 }].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if self.trace {
            crate::trace!("{} {:?} op {:#04x} len {} at {:#x}", self.name, engine, opcode, count, cur);
        }
        match opcode {
            op::NOP => self.nop(&body),
            op::SET_CONTEXT_REG => self.set_regs(pm4::CONTEXT_REG_BASE, &body),
            op::SET_SH_REG => self.set_regs(pm4::SH_REG_BASE, &body),
            op::SET_UCONFIG_REG => self.set_regs(pm4::UCONFIG_REG_BASE, &body),
            op::SET_CONFIG_REG => self.set_regs(pm4::CONFIG_REG_BASE, &body),
            op::INDIRECT_BUFFER | op::INDIRECT_BUFFER_CONST => {
                let addr = (body[0] as u64 & !3) | ((body[1] as u64 & 0xFFFF) << 32);
                let size = body[2] & 0xFFFFF;
                if !(guest_ok(addr, size as u64 * 4) && size > 0) {
                    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                        crate::warn!("{}: indirect buffer packet {:#x} not executed: body {:08x?} at {:#x}", self.name, opcode, body, cur);
                    }
                }
                if guest_ok(addr, size as u64 * 4) && size > 0 {
                    if s.frames.len() > 16 {
                        crate::error!("{}: indirect buffer nesting too deep", self.name);
                    } else {
                        super::note_exec(addr, !self.watch);
                        s.push(addr, size);
                    }
                }
            }
            // End-of-pipe/shader writes happen once all prior work completes:
            // they are queued and performed by `commit` after the renderer has
            // finished that work (one GPU synchronization for many events).
            op::EVENT_WRITE_EOP => {
                let addr = (body[1] as u64) | ((body[2] as u64 & 0xFFFF) << 32);
                let int_sel = (body[2] >> 24) & 3;
                let data_sel = (body[2] >> 29) & 7;
                self.queue_eop(addr, data_sel, body[3], body[4], int_sel != 0);
            }
            op::RELEASE_MEM => {
                let int_sel = (body[1] >> 24) & 7;
                let data_sel = (body[1] >> 29) & 7;
                let addr = (body[2] as u64) | ((body[3] as u64) << 32);
                self.queue_eop(addr, data_sel, body[4], body[5], int_sel != 0);
            }
            op::EVENT_WRITE_EOS => {
                let addr = (body[1] as u64) | ((body[2] as u64 & 0xFFFF) << 32);
                let cmd = (body[2] >> 29) & 7;
                match cmd {
                    2 => self.queue_eop(addr, 1, body[3], 0, false),
                    1 => {
                        // Store GDS dwords [index, index+size) at the end of prior shader work.
                        let (index, size) = (body[3] & 0xFFFF, body[3] >> 16);
                        if guest_ok(addr, size as u64 * 4) {
                            self.commit(hooks);
                            hooks.gds(self, GdsOp::Store { offset: index * 4, dst: addr, bytes: size * 4 });
                        } else {
                            crate::error!("{}: EVENT_WRITE_EOS GDS store to invalid address {:#x}", self.name, addr);
                        }
                    }
                    other => crate::warn!("{}: EVENT_WRITE_EOS command {} unsupported", self.name, other),
                }
            }
            op::WRITE_DATA => {
                let dst_sel = (body[0] >> 8) & 0xF;
                let one_addr = body[0] & (1 << 16) != 0;
                let addr = (body[1] as u64) | ((body[2] as u64) << 32);
                let data = &body[3..];
                if dst_sel != 0 {
                    self.mem_log("WRITE_DATA", addr, data.len() as u64 * 4, data);
                }
                if matches!(dst_sel, 1 | 2 | 5) && self.defer_label_write(hooks, addr, data, one_addr) {
                    return Step::Ok;
                }
                if dst_sel != 0 {
                    // Keep program order with earlier end-of-pipe writes.
                    let t0 = Instant::now();
                    let (own, inflight) = if wd_trace() { (self.own_pending_value(addr), super::render::pending_label_value(addr)) } else { (None, None) };
                    self.order_write(hooks, addr, data.len() as u64 * 4);
                    if wd_trace() && t0.elapsed() > Duration::from_micros(50) {
                        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 300 {
                            crate::info!(
                                "{}: WRITE_DATA {:#x}+{:#x} = {:#x}.. waited {} us (own queued EOP value {:x?}, in-flight EOP value {:x?}) at packet {:#x}",
                                self.name,
                                addr,
                                data.len() * 4,
                                data.first().copied().unwrap_or(0),
                                t0.elapsed().as_micros(),
                                own,
                                inflight,
                                self.cur_packet.0
                            );
                        }
                    }
                }
                match dst_sel {
                    0 => {
                        for (i, &v) in data.iter().enumerate() {
                            self.write_reg(body[1] + if one_addr { 0 } else { i as u32 }, v);
                        }
                    }
                    1 | 2 | 5 => {
                        if guest_ok(addr, data.len() as u64 * 4) {
                            for (i, &v) in data.iter().enumerate() {
                                let a = if one_addr { addr } else { addr + i as u64 * 4 };
                                unsafe { std::ptr::write_volatile(a as *mut u32, v) };
                            }
                        } else {
                            crate::error!("{}: WRITE_DATA to invalid address {:#x}", self.name, addr);
                        }
                    }
                    _ => crate::warn!("{}: WRITE_DATA dst_sel {} unsupported", self.name, dst_sel),
                }
            }
            op::WAIT_REG_MEM => self.wait_reg_mem(&body, hooks),
            op::DMA_DATA => {
                let src = (body[1] as u64) | ((body[2] as u64) << 32);
                let dst = (body[3] as u64) | ((body[4] as u64) << 32);
                let bytes = (body[5] & 0x1F_FFFF) as u64;
                let (src_sel, dst_sel) = ((body[0] >> 29) & 3, (body[0] >> 20) & 3);
                if (src_sel == 1 || dst_sel == 1) && body[5] & (3 << 26) == 0 {
                    self.dma_gds(&body, hooks);
                    return Step::Ok;
                }
                // (Memory to memory with both addresses incrementing, no queued
                // end-of-pipe writes of either range that would land later.)
                if src_sel == 0 && dst_sel == 0 && body[5] & (0xF << 26) == 0 && !self.pending_overlaps(src, bytes) && !self.pending_overlaps(dst, bytes) && hooks.dma_on_gpu(self, src, dst, bytes) {
                    self.stats.dma_gpu += 1;
                    self.mem_log("DMA_DATA (on the GPU)", dst, bytes, &[src as u32, (src >> 32) as u32, src_sel]);
                    return Step::Ok;
                }
                if src_sel == 0 {
                    let t0 = Instant::now();
                    self.order_access(hooks, src, bytes.max(4));
                    // (Diagnostics: what the copies that wait for GPU writes move.)
                    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    if t0.elapsed() >= Duration::from_micros(10) && (super::render::trace_frame() || N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16) {
                        let head: Vec<u32> = (0..(bytes / 4).min(8)).filter(|&i| guest_ok(src + i * 4, 4)).map(|i| unsafe { rd(src + i * 4) }).collect();
                        let to = if body[5] & (1 << 27) != 0 { format!("register {:#x}", (dst as u32) >> 2) } else { format!("{dst:#x}") };
                        crate::info!("{}: DMA_DATA waited {}us for GPU writes of its source {src:#x} -> {to} ({bytes} bytes, saic {} daic {}, packet at {:#x}); data {head:x?}", self.name, t0.elapsed().as_micros(), (body[5] >> 28) & 1, (body[5] >> 29) & 1, self.cur_packet.0);
                    }
                }
                if dst_sel == 0 && body[5] & (1 << 27) == 0 {
                    self.order_write(hooks, dst, bytes.max(4));
                    self.mem_log("DMA_DATA", dst, bytes, &[src as u32, (src >> 32) as u32, src_sel]);
                }
                self.dma_data(&body);
            }
            op::COPY_DATA => {
                // Order with queued end-of-pipe writes to the same memory.
                let src = (body[1] as u64) | ((body[2] as u64) << 32);
                let dst = (body[3] as u64) | ((body[4] as u64) << 32);
                if matches!(body[0] & 0xF, 1 | 2) {
                    self.order_access(hooks, src, 8);
                }
                if matches!((body[0] >> 8) & 0xF, 1 | 2 | 5) {
                    self.order_write(hooks, dst, 8);
                }
                self.copy_data(&body);
            }
            op::WRITE_CONST_RAM => {
                let off = (body[0] & 0xFFFF) as usize / 4;
                self.ce_log("WRITE_CONST_RAM", off * 4, &body[1..]);
                for (i, &v) in body[1..].iter().enumerate() {
                    if let Some(d) = self.ce_ram.get_mut(off + i) {
                        *d = v;
                    }
                }
            }
            op::DUMP_CONST_RAM => {
                let off = (body[0] & 0xFFFF) as usize / 4;
                let n = (body[1] & 0x7FFF) as usize;
                let addr = (body[2] as u64) | ((body[3] as u64) << 32);
                self.order_write(hooks, addr, n as u64 * 4);
                if off + n <= CE_RAM_DWORDS {
                    self.mem_log("DUMP_CONST_RAM", addr, n as u64 * 4, &self.ce_ram[off..off + n.min(8)]);
                    self.ce_log(&format!("DUMP_CONST_RAM {n} dwords to {addr:#x} from"), off * 4, &self.ce_ram[off..off + n.min(80)]);
                }
                if guest_ok(addr, n as u64 * 4) && off + n <= CE_RAM_DWORDS {
                    unsafe { std::ptr::copy_nonoverlapping(self.ce_ram[off..].as_ptr(), addr as *mut u32, n) };
                }
            }
            op::LOAD_CONST_RAM => {
                let addr = (body[0] as u64) | ((body[1] as u64) << 32);
                let n = (body[2] & 0x7FFF) as usize;
                self.order_access(hooks, addr, n as u64 * 4);
                let off = (body[3] & 0xFFFF) as usize / 4;
                self.mem_log("LOAD_CONST_RAM (source)", addr, n as u64 * 4, &[off as u32 * 4]);
                if guest_ok(addr, n as u64 * 4) {
                    let head: Vec<u32> = (0..n.min(80) as u64).map(|i| unsafe { std::ptr::read_volatile((addr + i * 4) as *const u32) }).collect();
                    self.ce_log(&format!("LOAD_CONST_RAM {n} dwords from {addr:#x} to"), off * 4, &head);
                }
                if guest_ok(addr, n as u64 * 4) && off + n <= CE_RAM_DWORDS {
                    unsafe { std::ptr::copy_nonoverlapping(addr as *const u32, self.ce_ram[off..].as_mut_ptr(), n) };
                }
            }
            op::INCREMENT_CE_COUNTER => {
                self.ce_count += 1;
                self.cede_log("INCREMENT_CE_COUNTER", &body);
            }
            op::INCREMENT_DE_COUNTER => {
                self.de_count += 1;
                self.cede_log("INCREMENT_DE_COUNTER", &body);
            }
            op::WAIT_ON_CE_COUNTER => self.cede_log("WAIT_ON_CE_COUNTER", &body),
            op::WAIT_ON_DE_COUNTER_DIFF => self.cede_log("WAIT_ON_DE_COUNTER_DIFF", &body),
            op::INDEX_TYPE => self.index_type = body[0] & 3,
            op::INDEX_BASE => self.index_base = (body[0] as u64 & !1) | ((body[1] as u64 & 0xFFFF) << 32),
            op::INDEX_BUFFER_SIZE => self.index_size = body[0],
            op::NUM_INSTANCES => self.num_instances = body[0].max(1),
            op::SET_BASE => {
                if body[0] & 0xF == 1 {
                    self.indirect_base = (body[1] as u64 & !7) | ((body[2] as u64 & 0xFFFF) << 32);
                }
            }
            op::DRAW_INDEX_AUTO => {
                self.stats.draws += 1;
                hooks.draw(self, "auto", body[0], 0);
            }
            op::DRAW_INDEX_2 => {
                self.stats.draws += 1;
                let addr = (body[1] as u64) | ((body[2] as u64 & 0xFFFF) << 32);
                // MAX_SIZE: indices in the buffer.
                self.draw_index_valid = body[0];
                hooks.draw(self, "index2", body[3], addr);
                self.draw_index_valid = u32::MAX;
            }
            op::DRAW_INDEX_OFFSET_2 => {
                self.stats.draws += 1;
                // MAX_SIZE bounds the indices fetched for this draw, counted
                // from its start (the title's UI draws pass MAX_SIZE = count
                // with nonzero offsets).
                self.draw_index_valid = body[0];
                if body[0] < body[2] {
                    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 24 {
                        crate::info!("{}: DRAW_INDEX_OFFSET_2 max {} offset {} count {} (index base {:#x})", self.name, body[0], body[1], body[2], self.index_base);
                    }
                }
                hooks.draw(self, "index_offset2", body[2], self.index_base + body[1] as u64 * if self.index_type == 1 { 4 } else { 2 });
                self.draw_index_valid = u32::MAX;
            }
            op::DRAW_INDIRECT | op::DRAW_INDEX_INDIRECT | op::DRAW_INDIRECT_MULTI | op::DRAW_INDEX_INDIRECT_MULTI => {
                let indexed = matches!(opcode, op::DRAW_INDEX_INDIRECT | op::DRAW_INDEX_INDIRECT_MULTI);
                let multi = matches!(opcode, op::DRAW_INDIRECT_MULTI | op::DRAW_INDEX_INDIRECT_MULTI);
                let args = if indexed { 20u64 } else { 16 };
                let (n, stride) = if multi { (body.get(3).copied().unwrap_or(0), body.get(4).copied().unwrap_or(0) as u64) } else { (1, 0) };
                for k in 0..n.min(4096) as u64 {
                    let stride = if stride == 0 { args } else { stride };
                    self.draw_indirect(hooks, self.indirect_base + body[0] as u64 + k * stride, indexed, body[1] & 0xFFFF, body[2] & 0xFFFF);
                }
            }
            op::DRAW_INDEX_INDIRECT_COUNT_MULTI => {
                // [data offset, base-vertex SGPR, start-instance SGPR, count-enable (bit 30),
                //  max count, count address lo/hi, stride, draw initiator]
                let max = body.get(4).copied().unwrap_or(0);
                let stride = body.get(7).copied().unwrap_or(20).max(20) as u64;
                let mut n = max;
                if body.get(3).is_some_and(|f| f & (1 << 30) != 0) {
                    let ca = body[5] as u64 | (body[6] as u64 & 0xFFFF) << 32;
                    self.order_access(hooks, ca, 4);
                    n = if guest_ok(ca, 4) { unsafe { rd(ca) }.min(max) } else { 0 };
                }
                for k in 0..n.min(4096) as u64 {
                    self.draw_indirect(hooks, self.indirect_base + body[0] as u64 + k * stride, true, body[1] & 0xFFFF, body[2] & 0xFFFF);
                }
            }
            op::DISPATCH_DIRECT => {
                self.stats.dispatches += 1;
                hooks.dispatch(self, body[0], body[1], body[2]);
            }
            op::DISPATCH_INDIRECT => {
                self.stats.dispatches += 1;
                // Graphics form: [offset from SET_BASE, initiator]; compute-queue form
                // (sceGnmDispatchIndirectOnMec): [address lo, address hi, initiator].
                let a = if body.len() >= 3 { body[0] as u64 | (body[1] as u64 & 0xFFFF) << 32 } else { self.indirect_base + body[0] as u64 };
                // Arguments written by earlier GPU work are read on the GPU in
                // queue order when the renderer can (no wait for that work);
                // otherwise the CP waits for them and reads them itself.
                if guest_ok(a, 12) && !self.pending_overlaps(a, 12) && hooks.dispatch_indirect(self, a) {
                    self.stats.indirect_on_gpu += 1;
                } else {
                    self.order_access(hooks, a, 12);
                    if guest_ok(a, 12) {
                        let (x, y, z) = unsafe { (rd(a), rd(a + 4), rd(a + 8)) };
                        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                        let groups = x as u64 * y as u64 * z as u64;
                        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 24 || groups > 1 << 22 {
                            crate::info!("{}: DISPATCH_INDIRECT {}x{}x{} (args at {:#x})", self.name, x, y, z, a);
                        }
                        if groups > 1 << 22 {
                            crate::error!("{}: DISPATCH_INDIRECT of {} groups skipped (implausible arguments)", self.name, groups);
                        } else {
                            hooks.dispatch(self, x, y, z);
                        }
                    }
                }
            }
            op::MEM_SEMAPHORE => self.mem_semaphore(&body, hooks),
            op::SET_PREDICATION => {
                static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 10 {
                    crate::info!("{}: SET_PREDICATION {:08x?} at {:#x} (not emulated)", self.name, body, cur);
                }
            }
            op::EVENT_WRITE => {
                let ev = body.first().copied().unwrap_or(0) & 0x3F;
                let n = EVENT_TYPES[ev as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // Counter samples other than occlusion dumps (pipeline
                // statistics, ZPASS_DONE) are not emulated: the results stay
                // unwritten (logged with the target). PIXEL_PIPE_STAT_CONTROL
                // (0x38) only configures the counters.
                let occl = crate::tune::get(crate::tune::OCCL) != 0;
                if (matches!(ev, 0x15 | 0x1E | 0x20) || (!occl && matches!(ev, 0x38 | 0x39))) && n < 24 {
                    let addr = body.get(1).map_or(0, |&lo| lo as u64 | (body.get(2).map_or(0, |&hi| hi as u64 & 0xFFFF) << 32));
                    let mut cur = [0u8; 32];
                    if addr != 0 {
                        super::resources::copy_from_guest(&mut cur, addr & !7);
                    }
                    crate::warn!("{}: EVENT_WRITE type {ev:#x} body {:08x?} (counter sample to {addr:#x}, holding {:02x?}) not emulated", self.name, body, cur);
                }
                // PIXEL_PIPE_STAT_DUMP: the occlusion counters at this point,
                // written (like an end-of-pipe write) once the work before it
                // completed. Begin samples sit at the start of a 16-byte
                // counter entry, end samples 8 bytes in.
                if ev == 0x39 && occl {
                    let addr = body.get(1).map_or(0, |&lo| (lo as u64 & !7) | (body.get(2).map_or(0, |&hi| hi as u64 & 0xFFFF) << 32));
                    if guest_ok(addr, 8 * 16) {
                        if let Some(queries) = hooks.occlusion_dump(self, addr % 16 == 0) {
                            self.stats.eops += 1;
                            let gfx = self.watch;
                            self.pending.push(EopAction { addr, data_sel: OCCL_DUMP, lo: 0, hi: 0, interrupt: false, gfx, occl: Some(queries.into_boxed_slice()) });
                        }
                    }
                }
            }
            op::GET_LOD_STATS => {
                // Texture mip-usage statistics for the title's streaming: not gathered.
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| crate::info!("{}: GET_LOD_STATS ignored (mip usage statistics are not gathered)", self.name));
            }
            op::CONTEXT_CONTROL | op::CLEAR_STATE | op::SURFACE_SYNC | op::ACQUIRE_MEM
            | op::PFP_SYNC_ME | op::DRAW_PREAMBLE | op::PREAMBLE_CNTL | op::PRED_EXEC
            | op::SWITCH_BUFFER | op::OCCLUSION_QUERY => {}
            other => {
                self.stats.unknown += 1;
                if self.stats.unknown <= 20 {
                    crate::warn!("{}: unhandled PM4 opcode {:#04x} ({} dwords) at {:#x}", self.name, other, count, cur);
                }
            }
        }
        Step::Ok
    }

    fn nop(&mut self, body: &[u32]) {
        let Some(&tag) = body.first() else { return };
        match tag {
            pm4::marker::PUSH => {
                let s = pm4::decode_string(&body[1..]);
                if self.trace {
                    crate::debug!("{}: push marker '{}'", self.name, s);
                }
                self.markers.push(s);
            }
            pm4::marker::POP => {
                self.markers.pop();
            }
            pm4::marker::SET => {
                if self.trace {
                    crate::debug!("{}: marker '{}'", self.name, pm4::decode_string(&body[1..]));
                }
            }
            pm4::marker::WAIT_FLIP => {
                let handle = body.get(1).copied().unwrap_or(0) as i32;
                let index = body.get(2).copied().unwrap_or(0) as i32;
                crate::hle::libs::videoout::wait_flip_done(handle, index);
            }
            _ => {}
        }
    }

    fn set_regs(&mut self, base: u32, body: &[u32]) {
        let start = base + (body[0] & 0xFFFF);
        let vals = &body[1..];
        // Bulk copy into one register file unless writes are being watched.
        if !reg_watching() {
            let (s, e) = (start as usize, start as usize + vals.len());
            let file = match s {
                0xA000..0xA400 if e <= 0xA400 => Some(&mut self.regs.ctx[s - 0xA000..e - 0xA000]),
                0x2C00..0x3000 if e <= 0x3000 => Some(&mut self.regs.sh[s - 0x2C00..e - 0x2C00]),
                0xC000..0xC400 if e <= 0xC400 => Some(&mut self.regs.uconfig[s - 0xC000..e - 0xC000]),
                0x2000..0x2C00 if e <= 0x2C00 => Some(&mut self.regs.config[s - 0x2000..e - 0x2000]),
                _ => None,
            };
            if let Some(f) = file {
                f.copy_from_slice(vals);
                return;
            }
        }
        for (i, &v) in vals.iter().enumerate() {
            self.write_reg(start + i as u32, v);
        }
    }

    #[inline]
    pub fn write_reg(&mut self, reg: u32, v: u32) {
        if reg_watching() {
            self.write_reg_watched(reg, v);
            return;
        }
        let r = reg as usize;
        match r {
            0xA000..0xA400 => self.regs.ctx[r - 0xA000] = v,
            0x2C00..0x3000 => self.regs.sh[r - 0x2C00] = v,
            0xC000..0xC400 => self.regs.uconfig[r - 0xC000] = v,
            0x2000..0x2C00 => self.regs.config[r - 0x2000] = v,
            _ => {}
        }
    }

    #[cold]
    fn write_reg_watched(&mut self, reg: u32, v: u32) {
        // TLG_WATCH_REG=<reg>[,<reg>...] (hex): log writes to those registers
        // (only zero values with TLG_WATCH_ZERO=1).
        static WATCH: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
        let watch = WATCH.get_or_init(|| std::env::var("TLG_WATCH_REG").map(|v| v.split(',').filter_map(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok()).collect()).unwrap_or_default());
        static ZERO: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !watch.is_empty() && watch.contains(&reg) && (v == 0 || !*ZERO.get_or_init(|| std::env::var_os("TLG_WATCH_ZERO").is_some())) {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 2000 {
                crate::info!("{}: reg {:#x} = {:#x} by packet at {:#x}{}", self.name, reg, v, self.cur_packet.0, if self.cur_packet.1 { " (CE)" } else { "" });
            }
        }
        let r = reg as usize;
        if (0xA000..0xA400).contains(&r) {
            self.regs.ctx[r - 0xA000] = v;
        } else if (0x2C00..0x3000).contains(&r) {
            self.regs.sh[r - 0x2C00] = v;
        } else if (0xC000..0xC400).contains(&r) {
            self.regs.uconfig[r - 0xC000] = v;
        } else if (0x2000..0x2C00).contains(&r) {
            self.regs.config[r - 0x2000] = v;
        }
    }

    /// WRITE_DATA to memory that only end-of-pipe writes still to be
    /// performed touch: queued behind them (performed in order once the work
    /// recorded so far has completed) instead of the CP draining the GPU to
    /// keep program order. TLG's compute queues signal each other this way:
    /// RELEASE_MEM label = 1, WAIT_REG_MEM label == 1 (met in GPU order), then
    /// WRITE_DATA label = 0x10 for the queues waiting on it; draining there
    /// cost up to 90 ms per write. The write becomes visible when that work is
    /// done, as on the hardware after its wait. TLG_DEFER_LABEL_WRITES=0: drain.
    fn defer_label_write(&mut self, hooks: &mut dyn Hooks, addr: u64, data: &[u32], one_addr: bool) -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let len = data.len() as u64 * 4;
        if !*ON.get_or_init(|| std::env::var("TLG_DEFER_LABEL_WRITES").map(|v| v != "0").unwrap_or(true)) || data.is_empty() || addr % 4 != 0 || !guest_ok(addr, len) {
            return false;
        }
        if !(self.pending_overlaps(addr, len) || super::render::actions_in_flight(addr, len)) || !hooks.label_only(self, addr, len) {
            return false;
        }
        for (i, &v) in data.iter().enumerate() {
            let a = if one_addr { addr } else { addr + i as u64 * 4 };
            self.pending.push(EopAction { addr: a, data_sel: 1, lo: v, hi: 0, interrupt: false, gfx: false, occl: None });
        }
        DEFERRED_WRITES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        true
    }

    fn queue_eop(&mut self, addr: u64, data_sel: u32, lo: u32, hi: u32, interrupt: bool) {
        self.stats.eops += 1;
        let gfx = self.watch;
        self.pending.push(EopAction { addr, data_sel, lo, hi, interrupt, gfx, occl: None });
    }

    /// Orders a direct memory access by the CP with end-of-pipe writes that
    /// are queued here or still in flight.
    #[track_caller]
    fn order_access(&mut self, hooks: &mut dyn Hooks, addr: u64, len: u64) {
        let t0 = Instant::now();
        if self.pending_overlaps(addr, len) {
            self.commit(hooks);
        }
        hooks.sync_range(self, addr, len);
        note_sync(std::panic::Location::caller().line(), t0);
    }

    /// `order_access` for a CP write: also waits until GPU work recorded
    /// earlier no longer reads the memory in place.
    #[track_caller]
    fn order_write(&mut self, hooks: &mut dyn Hooks, addr: u64, len: u64) {
        self.order_access(hooks, addr, len);
        let t0 = Instant::now();
        hooks.sync_write(self, addr, len);
        note_sync(std::panic::Location::caller().line(), t0);
    }

    /// The value this CP's queued end-of-pipe writes leave at `addr` (the
    /// last one covering it), if any.
    fn own_pending_value(&self, addr: u64) -> Option<u32> {
        let mut val = None;
        for p in &self.pending {
            let n = match p.data_sel {
                1 => 4,
                2..=4 => 8,
                _ => continue,
            };
            if addr < p.addr || addr + 4 > p.addr + n {
                continue;
            }
            val = match p.data_sel {
                1 => Some(p.lo),
                2 => Some(if addr == p.addr { p.lo } else { p.hi }),
                _ => None,
            };
        }
        val
    }

    fn pending_overlaps(&self, addr: u64, len: u64) -> bool {
        self.pending.iter().any(|p| p.overlaps(addr, len))
    }

    /// Hands the queued end-of-pipe events to the renderer, which performs
    /// them in order once the work recorded before them has completed.
    pub fn commit(&mut self, hooks: &mut dyn Hooks) {
        if self.pending.is_empty() {
            return;
        }
        let actions = std::mem::take(&mut self.pending);
        hooks.commit(self, actions);
    }

    fn wait_reg_mem(&mut self, body: &[u32], hooks: &mut dyn Hooks) {
        let func = body[0] & 7;
        let mem = body[0] & (1 << 4) != 0;
        let addr = (body[1] as u64 & !3) | ((body[2] as u64 & 0xFFFF) << 32);
        let reference = body[3];
        let mask = body[4];
        self.stats.waits += 1;
        if !mem {
            return; // register polls: our registers never change asynchronously
        }
        if !guest_ok(addr, 4) {
            crate::error!("{}: WAIT_REG_MEM on invalid address {:#x}", self.name, addr);
            return;
        }
        let test = |v: u32| -> bool {
            let v = v & mask;
            match func {
                0 => true,
                1 => v < reference,
                2 => v <= reference,
                3 => v == reference,
                4 => v != reference,
                5 => v >= reference,
                6 => v > reference,
                _ => true,
            }
        };
        // (Time not met at once: per queue, TLG_PERF.)
        struct Timed(usize, Instant);
        impl Drop for Timed {
            fn drop(&mut self) {
                super::render::QUEUE_WAIT_REG_NS[self.0 & 63].fetch_add(self.1.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            }
        }
        let _timed = (!test(unsafe { rd(addr) })).then(|| Timed(self.queue, Instant::now()));
        if !test(unsafe { rd(addr) }) {
            static VIRTUAL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let virtual_waits = *VIRTUAL.get_or_init(|| std::env::var("TLG_VIRTUAL_WAITS").map(|v| v != "0").unwrap_or(true));
            let inbatch = crate::tune::get(crate::tune::INBATCH_WAITS) != 0;
            // The title's "end-of-pipe label, then wait for it" drain, with
            // the label write still queued here: met in GPU order by a full
            // barrier in the current batch; the queued writes stay queued
            // (committed at the next commit point, at the latest when this
            // submission ends) instead of closing the batch per drain.
            // Every path that can block commits first, so nothing waits on
            // writes held here (TLG_INBATCH_WAITS=0: commit at every drain).
            if virtual_waits && inbatch && self.own_pending_value(addr).is_some_and(|v| test(v)) {
                self.stats.inbatch_waits += 1;
                hooks.order_barrier(self);
                return;
            }
            // The awaited value may come from our own queued end-of-pipe
            // writes or from shader writes not yet written back.
            self.commit(hooks);
            // When our in-flight end-of-pipe writes will produce a satisfying
            // value, the wait is met in GPU order: everything recorded after
            // it executes after them on the queue. Memory the CP itself reads
            // later is synchronized where it is read (TLG_VIRTUAL_WAITS=0:
            // always stall until the write happened).
            if virtual_waits && super::render::pending_label_value(addr).is_some_and(|v| test(v)) {
                self.stats.virtual_waits += 1;
                return;
            }
            let t0 = Instant::now();
            hooks.sync_range(self, addr, 4);
            note_sync(line!(), t0);
        }
        if self.watch {
            super::set_phase(super::phase::WAIT_REG_MEM, addr);
        } else if self.queue != 0 {
            super::set_qphase(self.queue, super::phase::WAIT_REG_MEM, addr);
        }
        let start = Instant::now();
        let mut warned = false;
        let mut spins = 0u32;
        static VIRTUAL_POLL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let virtual_waits = *VIRTUAL_POLL.get_or_init(|| std::env::var("TLG_VIRTUAL_WAITS").map(|v| v != "0").unwrap_or(true));
        // (TLG_PERF: polls over 0.5 ms by queue, condition and values.)
        struct Poll {
            queue: usize,
            func: u32,
            reference: u32,
            before: u32,
            addr: u64,
            start: Instant,
            polled: std::cell::Cell<bool>,
        }
        impl Drop for Poll {
            fn drop(&mut self) {
                let us = self.start.elapsed().as_micros() as u64;
                if self.polled.get() && us >= 500 && super::render::perf_on() {
                    let after = unsafe { rd(self.addr) };
                    let mut g = POLLS.lock();
                    let e = g.get_or_insert_with(Default::default).entry((self.queue, self.func, self.reference, self.before, after)).or_default();
                    e.0 += 1;
                    e.1 += us;
                }
            }
        }
        let poll = Poll { queue: self.queue, func, reference, before: unsafe { rd(addr) }, addr, start, polled: std::cell::Cell::new(false) };
        while !test(unsafe { rd(addr) }) {
            poll.polled.set(true);
            spins += 1;
            if spins < 64 {
                std::hint::spin_loop();
            } else {
                std::thread::sleep(Duration::from_micros(100));
                // Another queue may meanwhile have recorded the end-of-pipe
                // write that satisfies the wait: met in GPU order from then
                // on (as above), without waiting for that work to execute.
                if virtual_waits && super::render::pending_label_value(addr).is_some_and(|v| test(v)) {
                    self.stats.virtual_waits += 1;
                    return;
                }
            }
            if !warned && start.elapsed() > Duration::from_millis(250) {
                warned = true;
                crate::warn!(
                    "{}: WAIT_REG_MEM still waiting on {:#x} (value {:#x}, func {}, ref {:#x}, mask {:#x}) markers {:?}",
                    self.name,
                    addr,
                    unsafe { rd(addr) },
                    func,
                    reference,
                    mask,
                    self.markers
                );
            }
        }
    }

    /// MEM_SEMAPHORE: 64-bit memory semaphore. Signal increments it (or sets
    /// it to 1 for write-type signals); wait blocks until it is non-zero and
    /// then decrements it.
    fn mem_semaphore(&mut self, body: &[u32], hooks: &mut dyn Hooks) {
        use std::sync::atomic::{AtomicU64, Ordering};
        let addr = (body[0] as u64 & !7) | ((body[1] as u64 & 0xFFFF) << 32);
        let sel = body[1] >> 29;
        let write_signal = (body[1] >> 20) & 1 != 0;
        if !guest_ok(addr, 8) {
            crate::error!("{}: MEM_SEMAPHORE on invalid address {:#x}", self.name, addr);
            return;
        }
        self.order_write(hooks, addr, 8);
        let sem = unsafe { &*(addr as *const AtomicU64) };
        match sel {
            6 => {
                if write_signal {
                    sem.store(1, Ordering::SeqCst);
                } else {
                    sem.fetch_add(1, Ordering::SeqCst);
                }
            }
            7 => {
                // Our own queued end-of-pipe work may be what signals it.
                if sem.load(Ordering::SeqCst) == 0 {
                    self.commit(hooks);
                }
                let start = Instant::now();
                let mut warned = false;
                loop {
                    let v = sem.load(Ordering::SeqCst);
                    if v != 0 && sem.compare_exchange(v, v - 1, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                        break;
                    }
                    if v == 0 {
                        std::thread::sleep(Duration::from_micros(100));
                    }
                    if !warned && start.elapsed() > Duration::from_millis(250) {
                        warned = true;
                        crate::warn!("{}: MEM_SEMAPHORE wait on {:#x} still blocked", self.name, addr);
                    }
                }
            }
            other => crate::warn!("{}: MEM_SEMAPHORE select {} unsupported", self.name, other),
        }
    }

    fn dma_data(&mut self, body: &[u32]) {
        let src_sel = (body[0] >> 29) & 3;
        let dst_sel = (body[0] >> 20) & 3;
        let src = (body[1] as u64) | ((body[2] as u64) << 32);
        let dst = (body[3] as u64) | ((body[4] as u64) << 32);
        let bytes = (body[5] & 0x1F_FFFF) as u64;
        let sas = body[5] & (1 << 26) != 0;
        let das = body[5] & (1 << 27) != 0;
        let saic = body[5] & (1 << 28) != 0;
        let daic = body[5] & (1 << 29) != 0;
        if das {
            // Destination is register space (byte offset).
            let reg = (dst as u32) >> 2;
            let words = (bytes / 4) as u32;
            for i in 0..words {
                let v = match src_sel {
                    2 => body[1],
                    0 if !sas && guest_ok(src, bytes) => unsafe { rd(src + if saic { 0 } else { i as u64 * 4 }) },
                    _ => 0,
                };
                self.write_reg(reg + if daic { 0 } else { i }, v);
            }
            return;
        }
        if dst_sel != 0 {
            crate::warn!("{}: DMA_DATA dst_sel {} unsupported", self.name, dst_sel);
            return;
        }
        if !guest_ok(dst, bytes) {
            crate::error!("{}: DMA_DATA to invalid address {:#x}+{:#x}", self.name, dst, bytes);
            return;
        }
        match src_sel {
            0 if guest_ok(src, bytes) => unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, bytes as usize) },
            2 => {
                let words = bytes / 4;
                for i in 0..words {
                    unsafe { std::ptr::write_volatile((dst + i * 4) as *mut u32, body[1]) };
                }
            }
            _ => crate::warn!("{}: DMA_DATA src_sel {} unsupported", self.name, src_sel),
        }
    }

    /// TLG_CEDE_LOG=<from seconds>:<n>: logs n CE/DE counter packets executed
    /// after that much process time (diagnostics).
    /// Debug: CP-side memory writes (and constant-RAM loads) overlapping
    /// TLG_CP_WATCH=<lo>-<hi> (hex), logged while a frame is traced (all the
    /// time with TLG_CP_WATCH_ALL=1; at most 20000 lines).
    fn mem_log(&self, what: &str, addr: u64, len: u64, head: &[u32]) {
        static CFG: std::sync::OnceLock<Option<(u64, u64, bool)>> = std::sync::OnceLock::new();
        let Some((lo, hi, all)) = *CFG.get_or_init(|| {
            let v = std::env::var("TLG_CP_WATCH").ok()?;
            let (a, b) = v.split_once('-')?;
            let p = |s: &str| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok();
            Some((p(a)?, p(b)?, std::env::var_os("TLG_CP_WATCH_ALL").is_some()))
        }) else {
            return;
        };
        if addr >= hi || addr + len.max(1) <= lo || !(all || super::render::trace_frame()) {
            return;
        }
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 20000 {
            crate::info!("{}: CPWATCH {} {:#x}+{:#x} at {:#x}{} [{}]", self.name, what, addr, len, self.cur_packet.0, if self.cur_packet.1 { " CE" } else { "" }, head.iter().map(|w| format!("{w:08x}")).collect::<Vec<_>>().join(" "));
        }
    }

    /// Debug (TLG_CE_TRACE=1): constant-RAM writes, loads and dumps of the
    /// traced frame (byte offsets into CE RAM).
    fn ce_log(&self, what: &str, off: usize, data: &[u32]) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("TLG_CE_TRACE").is_some()) || !super::render::trace_frame() {
            return;
        }
        crate::info!("{}: CE {} ram+{:#x} at {:#x} [{}]", self.name, what, off, self.cur_packet.0, data.iter().map(|w| format!("{w:x}")).collect::<Vec<_>>().join(" "));
    }

    fn cede_log(&self, what: &str, body: &[u32]) {
        static CFG: std::sync::OnceLock<Option<(f64, u32)>> = std::sync::OnceLock::new();
        let Some((from, limit)) = *CFG.get_or_init(|| {
            let v = std::env::var("TLG_CEDE_LOG").ok()?;
            let (a, b) = v.split_once(':')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        }) else {
            return;
        };
        if crate::log::elapsed_ms() < from * 1000.0 {
            return;
        }
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < limit {
            crate::info!("{}: {} {:x?} at {:#x}{} (ce {} de {})", self.name, what, body, self.cur_packet.0, if self.cur_packet.1 { " CE" } else { "" }, self.ce_count, self.de_count);
        }
    }

    /// One indirect draw: arguments {count, instances, start (index or
    /// vertex), [base vertex,] start instance} from guest memory (possibly
    /// written by earlier GPU work). The CP passes the vertex and instance
    /// offsets to the shaders through the user-data registers at
    /// `vtx_loc` / `inst_loc` (offsets from the SH register base).
    fn draw_indirect(&mut self, hooks: &mut dyn Hooks, addr: u64, indexed: bool, vtx_loc: u32, inst_loc: u32) {
        self.stats.draws += 1;
        let len = if indexed { 20 } else { 16 };
        self.order_access(hooks, addr, len);
        if !guest_ok(addr, len) {
            crate::error!("{}: indirect draw arguments at invalid address {:#x}", self.name, addr);
            return;
        }
        let w: Vec<u32> = (0..len / 4).map(|i| unsafe { rd(addr + i * 4) }).collect();
        let (count, instances) = (w[0], w[1]);
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let work = count as u64 * instances as u64;
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 24 || work > 1 << 26 {
            crate::info!("{}: {} indirect draw args {:x?} at {:#x}", self.name, if indexed { "indexed" } else { "auto" }, w, addr);
        }
        if work > 1 << 26 || instances > 1 << 20 {
            crate::error!("{}: indirect draw of {} vertices x {} instances skipped (implausible arguments)", self.name, count, instances);
            return;
        }
        let (start, vtx_offset, start_inst) = if indexed { (w[2], w[3], w[4]) } else { (w[2], w[2], w[3]) };
        if vtx_loc != 0 {
            self.write_reg(pm4::SH_REG_BASE + vtx_loc, vtx_offset);
        }
        if inst_loc != 0 {
            self.write_reg(pm4::SH_REG_BASE + inst_loc, start_inst);
        }
        if count == 0 || instances == 0 {
            return;
        }
        // The instance count applies to this draw only (a lasting garbage
        // count from GPU-written arguments would multiply later draws).
        let saved = self.num_instances;
        self.num_instances = instances;
        if indexed {
            let isz = if self.index_type == 1 { 4 } else { 2 };
            hooks.draw(self, "index2", count, self.index_base + start as u64 * isz);
        } else {
            hooks.draw(self, "auto", count, 0);
        }
        self.num_instances = saved;
    }

    /// DMA_DATA with GDS as source or destination (GDS addresses are byte
    /// offsets). Recorded in order with the GPU work.
    fn dma_gds(&mut self, body: &[u32], hooks: &mut dyn Hooks) {
        let src_sel = (body[0] >> 29) & 3;
        let dst_sel = (body[0] >> 20) & 3;
        let src = (body[1] as u64) | ((body[2] as u64) << 32);
        let dst = (body[3] as u64) | ((body[4] as u64) << 32);
        let bytes = body[5] & 0x1F_FFFF;
        let op = match (src_sel, dst_sel) {
            (2, 1) => GdsOp::Fill { offset: dst as u32, bytes, value: body[1] },
            (0, 1) if guest_ok(src, bytes as u64) => {
                if !self.pending_overlaps(src, bytes as u64) && hooks.gds_load_on_gpu(self, dst as u32, src, bytes) {
                    self.stats.dma_gpu += 1;
                    return;
                }
                self.order_access(hooks, src, bytes.max(4) as u64);
                GdsOp::Load { offset: dst as u32, src, bytes }
            }
            (1, 0) if guest_ok(dst, bytes as u64) => {
                // The store is recorded in queue order and lands after earlier
                // GPU writes of the range (in place, or as a write-back after
                // theirs); the commit below orders this CP's own end-of-pipe
                // writes first (`tune dma_gpu 0`: wait for the range as before).
                if crate::tune::get(crate::tune::DMA_GPU) == 0 {
                    self.order_access(hooks, dst, bytes.max(4) as u64);
                }
                GdsOp::Store { offset: src as u32, dst, bytes }
            }
            _ => {
                crate::warn!("{}: DMA_DATA src_sel {} dst_sel {} ({:#x} -> {:#x}, {} bytes) unsupported", self.name, src_sel, dst_sel, src, dst, bytes);
                return;
            }
        };
        // Earlier end-of-pipe writes precede the transfer.
        self.commit(hooks);
        hooks.gds(self, op);
    }

    fn copy_data(&mut self, body: &[u32]) {
        let src_sel = body[0] & 0xF;
        let dst_sel = (body[0] >> 8) & 0xF;
        let count64 = body[0] & (1 << 16) != 0;
        let src = (body[1] as u64) | ((body[2] as u64) << 32);
        let dst = (body[3] as u64) | ((body[4] as u64) << 32);
        let n = if count64 { 8 } else { 4 };
        let value: u64 = match src_sel {
            1 | 2 if guest_ok(src, n) => unsafe {
                if count64 { std::ptr::read_volatile(src as *const u64) } else { rd(src) as u64 }
            },
            5 => (body[1] as u64) | if count64 { (body[2] as u64) << 32 } else { 0 },
            9 => unsafe { core::arch::x86_64::_rdtsc() },
            _ => {
                crate::warn!("{}: COPY_DATA src_sel {} unsupported", self.name, src_sel);
                return;
            }
        };
        match dst_sel {
            1 | 2 | 5 if guest_ok(dst, n) => unsafe {
                if count64 {
                    std::ptr::write_volatile(dst as *mut u64, value)
                } else {
                    std::ptr::write_volatile(dst as *mut u32, value as u32)
                }
            },
            0 => self.write_reg(body[3], value as u32),
            _ => crate::warn!("{}: COPY_DATA dst_sel {} unsupported", self.name, dst_sel),
        }
    }
}

fn engine_clone(e: &Engine) -> Engine {
    match e {
        Engine::Draw => Engine::Draw,
        Engine::Constant => Engine::Constant,
        Engine::Compute => Engine::Compute,
    }
}
