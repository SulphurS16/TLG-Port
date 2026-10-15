//! Shader preparation ahead of the graphics CP (`tune prep_ahead`): the
//! `GPU-prep` thread runs each graphics submission through a second command
//! processor in dry mode (register state, indirect buffers, constant RAM and
//! the draw/dispatch hooks only; the CP's memory writes - constant RAM dumps,
//! WRITE_DATA - go to a private overlay) and prepares every draw's and
//! dispatch's shader stages (`shaders::prepare*`) before the graphics CP gets
//! there. The graphics CP takes a prepared result for a command when the
//! registers preparation reads hash the same, the shader code is unchanged
//! and every guest dword preparation read still has the value it read
//! (`shaders::dry_valid`); otherwise it prepares itself as before. Shader
//! preparation was ~8% of the graphics CP (profile 2026-10-14).
//!
//! Decoded submissions (`tune decoded_cp`): the dry run also hands the
//! graphics CP its submission decoded (`Decoded`): register writes and every
//! packet with effects, in execution order - indirect buffers followed, the
//! constant and draw engines interleaved as `CommandProcessor::run` does -
//! so the graphics CP executes them without parsing PM4
//! (`CommandProcessor::run_decoded`). Command words that a CP write earlier
//! in the submission changes (none seen in the title: its CP writes into
//! command memory land in data embedded there) are decoded only once the
//! graphics CP has executed that write (`dry_sync`), as when it parses them.

use super::cp::{CommandProcessor, Captured, EopAction, Hooks, Regs, Stream};
use super::shaders::{self, Prepared, StageRegs};
use gcn::translate::Stage;
use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::hash::Hasher;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub type DrawPrep = (Result<Prepared, String>, Option<Result<Prepared, String>>);

enum Prep {
    Draw(DrawPrep),
    Dispatch(Result<Prepared, String>),
}

struct Entry {
    sub: u64,
    /// The graphics epoch the shader code was checked in (`shaders::program`).
    epoch: u64,
    key: u64,
    prep: Prep,
    log: shaders::DryLog,
}

struct Job {
    sub: u64,
    dcb: Vec<(u64, u32)>,
    ccb: Vec<(u64, u32)>,
    captured: Arc<Captured>,
    decoded: Option<Arc<Decoded>>,
}

/// A submission decoded for the graphics CP, in chunks as the prepare thread
/// produces them. Operations: `[1 << 31 | n, reg, values...]` (registers
/// [reg, reg+n)) or `[opcode | constant << 8 | n << 16, address low, address
/// high, body...]` (a type-3 packet with an n-dword body).
pub struct Decoded {
    /// Chunks with the operation count at their end; whether all are in.
    chunks: Mutex<(VecDeque<(Vec<u32>, u64)>, bool)>,
    cv: Condvar,
    /// Operations the graphics CP has executed (`dry_sync`).
    executed: AtomicU64,
    spare: Mutex<Vec<Vec<u32>>>,
    /// Chunks taken (TLG_PERF: waits for the first one).
    taken: AtomicU64,
}

impl Decoded {
    fn new() -> Decoded {
        Decoded { chunks: Mutex::new((VecDeque::new(), false)), cv: Condvar::new(), executed: AtomicU64::new(0), spare: Mutex::new(Vec::new()), taken: AtomicU64::new(0) }
    }

    /// The next chunk and the operation count at its end; None once the
    /// submission is through.
    pub fn next(&self) -> Option<(Vec<u32>, u64)> {
        let mut g = self.chunks.lock();
        let mut waited: Option<std::time::Instant> = None;
        let c = loop {
            if let Some(c) = g.0.pop_front() {
                break Some(c);
            }
            if g.1 {
                break None;
            }
            waited.get_or_insert_with(std::time::Instant::now);
            self.cv.wait(&mut g);
        };
        drop(g);
        // (TLG_PERF: the graphics CP waiting for the prepare thread, and
        // for the first chunk of a submission.)
        let first = self.taken.fetch_add(1, Ordering::Relaxed) == 0;
        if let Some(t) = waited {
            let ns = t.elapsed().as_nanos() as u64;
            STATS[8].fetch_add(1, Ordering::Relaxed);
            STATS[9].fetch_add(ns, Ordering::Relaxed);
            if first {
                STATS[10].fetch_add(1, Ordering::Relaxed);
                STATS[11].fetch_add(ns, Ordering::Relaxed);
            }
        }
        c
    }

    /// The graphics CP has executed the first `n` operations.
    #[inline]
    pub fn executed_through(&self, n: u64) {
        self.executed.store(n, Ordering::Release);
    }

    /// A chunk's buffer back for reuse.
    pub fn recycle(&self, mut c: Vec<u32>) {
        c.clear();
        let mut s = self.spare.lock();
        if s.len() < 4 {
            s.push(c);
        }
    }

    fn publish(&self, words: &mut Vec<u32>, ops: u64, last: bool) {
        if !words.is_empty() {
            let fresh = self.spare.lock().pop().unwrap_or_else(|| Vec::with_capacity(CHUNK_WORDS + 512));
            let c = std::mem::replace(words, fresh);
            self.chunks.lock().0.push_back((c, ops));
        }
        if last {
            self.chunks.lock().1 = true;
        }
        self.cv.notify_one();
    }
}

/// Words per published chunk at most (~100 commands); the first chunk of a
/// submission is smaller, so the graphics CP starts early.
const CHUNK_WORDS: usize = 4096;
const FIRST_CHUNK_WORDS: usize = 256;

static JOBS: Mutex<VecDeque<Job>> = Mutex::new(VecDeque::new());
static JOBS_CV: Condvar = Condvar::new();
static RESULTS: Mutex<VecDeque<Entry>> = Mutex::new(VecDeque::new());
/// The graphics CP's current submission (results of older ones are stale).
static SUB: AtomicU64 = AtomicU64::new(0);
static STARTED: std::sync::Once = std::sync::Once::new();
/// Commands served from prepared results, and declined ones (no result,
/// or it did not validate); CP writes into command memory; decoded
/// submissions: prepare-thread waits for the graphics CP (`dry_sync`),
/// graphics-CP waits for chunks and their time (ns), those for a
/// submission's first chunk; preparations over 200 us and their time (ns);
/// the prepare thread's time on submissions (ns) (TLG_PERF).
pub static STATS: [AtomicU64; 15] = [const { AtomicU64::new(0) }; 15];

/// A preparation of `what` at `addr` took `t0.elapsed()` (TLG_PERF: slow ones).
fn note_prep(t0: std::time::Instant, what: &str, addr: u64) {
    let ns = t0.elapsed().as_nanos() as u64;
    if ns >= 200_000 {
        let n = STATS[12].fetch_add(1, Ordering::Relaxed);
        STATS[13].fetch_add(ns, Ordering::Relaxed);
        if n < 24 {
            crate::info!("prepare thread: {what} {addr:#x} took {} us", ns / 1000);
        }
    }
}

/// Results kept at most (a submission has ~2,000 commands).
const MAX_RESULTS: usize = 8192;

/// Whether the prepare thread runs (it then follows every submission).
pub fn started() -> bool {
    STARTED.is_completed()
}

/// The graphics CP starts a submission: the prepare thread runs it too;
/// `decoded`: and decodes it for the graphics CP (returned).
pub fn submit(dcb: &[(u64, u32)], ccb: &[(u64, u32)], captured: Arc<Captured>, decoded: bool) -> Option<Arc<Decoded>> {
    STARTED.call_once(|| {
        crate::thread::spawn_host("GPU-prep", 8 << 20, Box::new(run)).expect("prepare thread");
    });
    let sub = SUB.fetch_add(1, Ordering::AcqRel) + 1;
    let d = decoded.then(|| Arc::new(Decoded::new()));
    JOBS.lock().push_back(Job { sub, dcb: dcb.to_vec(), ccb: ccb.to_vec(), captured, decoded: d.clone() });
    JOBS_CV.notify_one();
    d
}

fn run() {
    crate::log::set_thread_name("GPU-prep");
    let mut cp = CommandProcessor::new("gfx-prep");
    cp.dry = true;
    let mut raised = None;
    loop {
        let job = {
            let mut q = JOBS.lock();
            while q.is_empty() {
                JOBS_CV.wait(&mut q);
            }
            // (Only the newest submission matters: the CP has moved on from
            // the others. Their register writes still apply, in order.)
            q.pop_front().unwrap()
        };
        // (`tune prep_priority`: the graphics CP's priority, which waits for
        // this thread's decoding.)
        let want = crate::tune::get(crate::tune::PREP_PRIORITY) != 0;
        if raised != Some(want) {
            crate::thread::set_raised_priority(want);
            raised = Some(want);
        }
        // CP writes of the previous submission are in guest memory by now.
        shaders::dry_clear();
        // Shader code is checked for changes once per graphics epoch, which
        // the CP started for this submission before handing it over.
        let epoch = super::GFX_EPOCH.load(Ordering::Acquire);
        super::REQUIRED_EPOCH.with(|r| r.set(epoch));
        let mut de = Stream::chained(job.captured.clone());
        let mut ce = Stream::chained(job.captured.clone());
        for (i, &(addr, dw)) in job.dcb.iter().enumerate() {
            de.chain(addr, dw);
            if let Some(&(a, n)) = job.ccb.get(i) {
                ce.chain(a, n);
            }
        }
        let mut hooks = PrepHooks {
            sub: job.sub,
            epoch,
            commands: job.dcb.iter().chain(job.ccb.iter()).map(|&(a, n)| (a, a + n as u64 * 4)).collect(),
            out: job.decoded.clone(),
            words: Vec::with_capacity(CHUNK_WORDS + 512),
            limit: FIRST_CHUNK_WORDS,
            ops: 0,
            written: crate::gpu::render::FastMap::default(),
        };
        let t0 = std::time::Instant::now();
        cp.run(&mut de, &mut ce, &mut hooks);
        if let Some(d) = hooks.out.take() {
            d.publish(&mut hooks.words, hooks.ops, true);
        }
        STATS[14].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        // (Diagnostics: CP writes into command memory of the submission.)
        if !hooks.written.is_empty() {
            let mut hits = 0;
            for &(c, d) in &hooks.commands {
                for p in (c >> 12)..=((d.max(c + 1) - 1) >> 12) {
                    if let Some(ws) = hooks.written.get(&p) {
                        hits += ws.iter().filter(|&&(lo, hi, _)| lo < d && c < hi && lo.max(c) >> 12 == p).count() as u64;
                    }
                }
            }
            STATS[5].fetch_add(hits, Ordering::Relaxed);
        }
    }
}

fn hash_regs(h: &mut crate::gpu::render::FxHasher, regs: &[u32]) {
    // (Pairs of words in four independent lanes, folded in at the end: one
    // multiply chain per word was ~1% of the graphics CP's time.)
    let mut lanes = [0u64; 4];
    let mut chunks = regs.chunks_exact(8);
    for c in &mut chunks {
        for (l, lane) in lanes.iter_mut().enumerate() {
            let v = c[2 * l] as u64 | (c[2 * l + 1] as u64) << 32;
            *lane = (lane.rotate_left(5) ^ v).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
        }
    }
    for &r in chunks.remainder() {
        h.write_u32(r);
    }
    for l in lanes {
        h.write_u64(l);
    }
}

/// The registers a draw's preparation reads (VS and PS stages), hashed.
fn draw_key(r: &Regs) -> u64 {
    let mut h = crate::gpu::render::FxHasher::default();
    hash_regs(&mut h, &r.sh[0..0x80]);
    hash_regs(&mut h, &r.ctx[0x191..0x1B1]);
    for reg in [0xA1B4u32, 0xA1B6, 0xA1C5, 0xA207, 0xA2A8, 0xA2A9] {
        h.write_u32(r.ctx(reg));
    }
    for m in 0..8 {
        h.write_u32(r.ctx(0xA31C + m * 0xF));
    }
    h.write_u32(r.uconfig[0x242]);
    h.finish()
}

/// The registers a dispatch's preparation reads, hashed.
fn dispatch_key(r: &Regs) -> u64 {
    let mut h = crate::gpu::render::FxHasher::default();
    hash_regs(&mut h, &r.sh[0x200..0x280]);
    h.finish() ^ 0x5bd1_e995
}

fn push(e: Entry) {
    let mut q = RESULTS.lock();
    if q.len() < MAX_RESULTS {
        q.push_back(e);
    }
}

/// The prepared result for the next command with `key`, if one was made for
/// it in the current submission (results ahead of it are kept; older ones and
/// those skipped over are dropped).
fn take(key: u64, draw: bool) -> Option<Entry> {
    let sub = SUB.load(Ordering::Acquire);
    let mut q = RESULTS.lock();
    while q.front().is_some_and(|e| e.sub < sub) {
        q.pop_front();
    }
    // (TLG_PERF: no result yet - the prepare thread behind -, or none
    // matching among the next ones.)
    let Some(i) = q.iter().take(8).position(|e| e.key == key && matches!(e.prep, Prep::Draw(_)) == draw) else {
        STATS[if q.is_empty() { 3 } else { 4 }].fetch_add(1, Ordering::Relaxed);
        return None;
    };
    q.drain(..i);
    q.pop_front()
}

/// The stage's code is unchanged: checked in the current graphics epoch
/// (code changes only between epochs), else compared now.
fn code_ok(p: &Result<Prepared, String>, epoch: u64, stage: Stage, regs: &Regs) -> bool {
    match p {
        Ok(_) if epoch == super::GFX_EPOCH.load(Ordering::Acquire) => true,
        Ok(p) => shaders::code_hash_now(&StageRegs { stage, regs }) == Some(p.code_hash),
        Err(_) => false,
    }
}

/// A draw's prepared shader stages (VS, PS), when valid for these registers
/// and the memory as it is now.
pub fn take_draw(regs: &Regs) -> Option<DrawPrep> {
    let Some(e) = take(draw_key(regs), true) else {
        STATS[1].fetch_add(1, Ordering::Relaxed);
        return None;
    };
    let epoch = e.epoch;
    let Prep::Draw((vs, ps)) = e.prep else { return None };
    let ok = code_ok(&vs, epoch, Stage::Vertex, regs) && ps.as_ref().is_none_or(|ps| code_ok(ps, epoch, Stage::Fragment, regs)) && shaders::dry_valid(&e.log);
    if !ok {
        STATS[2].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    STATS[0].fetch_add(1, Ordering::Relaxed);
    Some((vs, ps))
}

/// A dispatch's prepared compute stage (`take_draw`).
pub fn take_dispatch(regs: &Regs) -> Option<Result<Prepared, String>> {
    let Some(e) = take(dispatch_key(regs), false) else {
        STATS[1].fetch_add(1, Ordering::Relaxed);
        return None;
    };
    let epoch = e.epoch;
    let Prep::Dispatch(cs) = e.prep else { return None };
    if !(code_ok(&cs, epoch, Stage::Compute, regs) && shaders::dry_valid(&e.log)) {
        STATS[2].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    STATS[0].fetch_add(1, Ordering::Relaxed);
    Some(cs)
}

/// The dry-run CP's hooks: preparation of every draw and dispatch.
struct PrepHooks {
    sub: u64,
    epoch: u64,
    /// Command memory of the submission (DCBs, CCBs, indirect buffers;
    /// diagnostics).
    commands: Vec<(u64, u64)>,
    /// Decoded submission under way: the chunk being filled (published at
    /// `limit` words), operations so far, and the guest memory the CP writes
    /// among them change (by 4 KiB page: [lo, hi) with the operation count
    /// through the write; kept without decoding too, for diagnostics).
    out: Option<Arc<Decoded>>,
    words: Vec<u32>,
    limit: usize,
    ops: u64,
    written: crate::gpu::render::FastMap<u64, Vec<(u64, u64, u64)>>,
}

impl PrepHooks {
    fn op_done(&mut self) {
        self.ops += 1;
        if self.words.len() >= self.limit {
            if let Some(d) = &self.out {
                d.publish(&mut self.words, self.ops, false);
            }
            self.limit = (self.limit * 2).min(CHUNK_WORDS);
        }
    }

    /// The operations so far end with a CP write of guest [a, a+n).
    fn note_write(&mut self, a: u64, n: u64) {
        let (ops, end) = (self.ops, a.saturating_add(n.max(1)));
        for p in (a >> 12)..=((end - 1) >> 12) {
            self.written.entry(p).or_default().push((a, end, ops));
        }
    }
}

/// Packets the graphics CP has nothing to do for in a decoded submission:
/// counter waits (resolved by the dry run's engine interleaving), packets it
/// ignores, NOPs other than the runtime's markers.
fn decoded_skips(opcode: u32, body: &[u32]) -> bool {
    use super::pm4::{marker, op};
    match opcode {
        op::WAIT_ON_CE_COUNTER | op::WAIT_ON_DE_COUNTER_DIFF => true,
        // (SURFACE_SYNC and ACQUIRE_MEM stay: the CP counts them, TLG_PERF.)
        op::CONTEXT_CONTROL | op::CLEAR_STATE | op::PFP_SYNC_ME | op::DRAW_PREAMBLE | op::PREAMBLE_CNTL | op::PRED_EXEC | op::SWITCH_BUFFER | op::OCCLUSION_QUERY => true,
        op::NOP => !matches!(body.first(), Some(&(marker::PUSH | marker::POP | marker::SET | marker::WAIT_FLIP))),
        _ => false,
    }
}

impl Hooks for PrepHooks {
    fn draw(&mut self, cp: &CommandProcessor, _kind: &'static str, _count: u32, _index_addr: u64) {
        let regs = &cp.regs;
        // (Patches: their stages are prepared under the lock, see `VkHooks::draw`.)
        if regs.uconfig[0x242] == 9 {
            return;
        }
        let key = draw_key(regs);
        let ps_addr = ((regs.sh(0x2C08) as u64) | ((regs.sh(0x2C09) as u64) << 32)) << 8;
        let t0 = std::time::Instant::now();
        shaders::dry_begin();
        let vs = shaders::prepare_tess(&StageRegs { stage: Stage::Vertex, regs }, Default::default());
        let ps = (ps_addr != 0).then(|| shaders::prepare(&StageRegs { stage: Stage::Fragment, regs }));
        let log = shaders::dry_end();
        note_prep(t0, "draw with pixel shader", ps_addr);
        push(Entry { sub: self.sub, epoch: self.epoch, key, prep: Prep::Draw((vs, ps)), log });
    }
    fn dispatch(&mut self, cp: &CommandProcessor, _x: u32, _y: u32, _z: u32) {
        let regs = &cp.regs;
        let key = dispatch_key(regs);
        let t0 = std::time::Instant::now();
        shaders::dry_begin();
        let cs = shaders::prepare(&StageRegs { stage: Stage::Compute, regs });
        let log = shaders::dry_end();
        note_prep(t0, "dispatch of compute shader", ((regs.sh(0x2E0C) as u64) | ((regs.sh(0x2E0D) as u64) << 32)) << 8);
        push(Entry { sub: self.sub, epoch: self.epoch, key, prep: Prep::Dispatch(cs), log });
    }
    fn dry_write(&mut self, addr: u64, data: &[u32]) {
        shaders::dry_write(addr, data);
    }
    fn dry_sync(&mut self, addr: u64, len: u64) {
        if self.written.is_empty() {
            return;
        }
        let end = addr.saturating_add(len.max(1));
        let mut need = None;
        for p in (addr >> 12)..=((end - 1) >> 12) {
            if let Some(ws) = self.written.get(&p) {
                for &(lo, hi, n) in ws {
                    if lo < end && addr < hi {
                        need = Some(need.map_or(n, |m: u64| m.max(n)));
                    }
                }
            }
        }
        let Some(need) = need else { return };
        // (Diagnostics: command words read after a CP write into them.)
        let k = STATS[6].fetch_add(1, Ordering::Relaxed);
        if k < 16 {
            crate::info!("prepare thread: command words at {addr:#x}+{len:#x} read after a CP write into them");
        }
        let Some(d) = self.out.clone() else { return };
        if d.executed.load(Ordering::Acquire) >= need {
            return;
        }
        // The graphics CP gets everything decoded so far, then performs the
        // write before these words are read.
        STATS[7].fetch_add(1, Ordering::Relaxed);
        d.publish(&mut self.words, self.ops, false);
        let mut spins = 0u32;
        while d.executed.load(Ordering::Acquire) < need {
            spins += 1;
            if spins < 64 {
                std::thread::yield_now();
            } else {
                std::thread::sleep(std::time::Duration::from_micros(50));
            }
        }
    }
    fn dry_emit(&mut self, opcode: u32, body: &[u32], addr: u64, constant: bool) {
        use super::pm4::op;
        if self.out.is_some() && !decoded_skips(opcode, body) {
            self.words.push(opcode | (constant as u32) << 8 | (body.len() as u32) << 16);
            self.words.push(addr as u32);
            self.words.push((addr >> 32) as u32);
            self.words.extend_from_slice(body);
            self.op_done();
        }
        // CP memory writes (what they change is decoded after they ran;
        // end-of-pipe writes land later in any case).
        let at = |lo: usize| (body[lo] as u64) | ((body[lo + 1] as u64) << 32);
        match opcode {
            op::WRITE_DATA if matches!((body[0] >> 8) & 0xF, 1 | 2 | 5) => {
                let n = if body[0] & (1 << 16) != 0 { 4 } else { 4 * (body.len() as u64).saturating_sub(3) };
                self.note_write(at(1), n);
            }
            op::DMA_DATA if body[5] & (1 << 27) == 0 && (body[0] >> 20) & 3 == 0 => self.note_write(at(3), (body[5] & 0x1F_FFFF) as u64),
            op::COPY_DATA if matches!((body[0] >> 8) & 0xF, 1 | 2 | 5) => self.note_write(at(3), 8),
            op::DUMP_CONST_RAM => self.note_write(at(2), 4 * (body[1] & 0x7FFF) as u64),
            _ => {}
        }
    }
    fn dry_emit_regs(&mut self, reg: u32, vals: &[u32]) {
        if self.out.is_none() {
            return;
        }
        self.words.push(1 << 31 | vals.len() as u32);
        self.words.push(reg);
        self.words.extend_from_slice(vals);
        self.op_done();
    }
    fn dry_commands(&mut self, addr: u64, len: u64) {
        self.commands.push((addr, addr + len));
    }
    fn flush(&mut self, _cp: &CommandProcessor) {}
    fn commit(&mut self, _cp: &CommandProcessor, _actions: Vec<EopAction>) {}
    fn sync_range(&mut self, _cp: &CommandProcessor, _addr: u64, _len: u64) {}
}
