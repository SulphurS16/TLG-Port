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
    log: Vec<(u64, u32, u64)>,
}

struct Job {
    sub: u64,
    dcb: Vec<(u64, u32)>,
    ccb: Vec<(u64, u32)>,
    captured: Arc<Captured>,
}

static JOBS: Mutex<VecDeque<Job>> = Mutex::new(VecDeque::new());
static JOBS_CV: Condvar = Condvar::new();
static RESULTS: Mutex<VecDeque<Entry>> = Mutex::new(VecDeque::new());
/// The graphics CP's current submission (results of older ones are stale).
static SUB: AtomicU64 = AtomicU64::new(0);
static STARTED: std::sync::Once = std::sync::Once::new();
/// Commands served from prepared results, and declined ones (no result,
/// or it did not validate) (TLG_PERF).
pub static STATS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Results kept at most (a submission has ~2,000 commands).
const MAX_RESULTS: usize = 8192;

/// Whether the prepare thread runs (it then follows every submission).
pub fn started() -> bool {
    STARTED.is_completed()
}

/// The graphics CP starts a submission: the prepare thread runs it too.
pub fn submit(dcb: &[(u64, u32)], ccb: &[(u64, u32)], captured: Arc<Captured>) {
    STARTED.call_once(|| {
        crate::thread::spawn_host("GPU-prep", 8 << 20, Box::new(run)).expect("prepare thread");
    });
    let sub = SUB.fetch_add(1, Ordering::AcqRel) + 1;
    JOBS.lock().push_back(Job { sub, dcb: dcb.to_vec(), ccb: ccb.to_vec(), captured });
    JOBS_CV.notify_one();
}

fn run() {
    crate::log::set_thread_name("GPU-prep");
    let mut cp = CommandProcessor::new("gfx-prep");
    cp.dry = true;
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
        let mut hooks = PrepHooks { sub: job.sub, epoch };
        cp.run(&mut de, &mut ce, &mut hooks);
    }
}

fn hash_regs(h: &mut crate::gpu::render::FxHasher, regs: &[u32]) {
    for &r in regs {
        h.write_u32(r);
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
    let i = q.iter().take(8).position(|e| e.key == key && matches!(e.prep, Prep::Draw(_)) == draw)?;
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
        shaders::dry_begin();
        let vs = shaders::prepare_tess(&StageRegs { stage: Stage::Vertex, regs }, Default::default());
        let ps = (ps_addr != 0).then(|| shaders::prepare(&StageRegs { stage: Stage::Fragment, regs }));
        let log = shaders::dry_end();
        push(Entry { sub: self.sub, epoch: self.epoch, key, prep: Prep::Draw((vs, ps)), log });
    }
    fn dispatch(&mut self, cp: &CommandProcessor, _x: u32, _y: u32, _z: u32) {
        let regs = &cp.regs;
        let key = dispatch_key(regs);
        shaders::dry_begin();
        let cs = shaders::prepare(&StageRegs { stage: Stage::Compute, regs });
        let log = shaders::dry_end();
        push(Entry { sub: self.sub, epoch: self.epoch, key, prep: Prep::Dispatch(cs), log });
    }
    fn dry_write(&mut self, addr: u64, data: &[u32]) {
        shaders::dry_write(addr, data);
    }
    fn flush(&mut self, _cp: &CommandProcessor) {}
    fn commit(&mut self, _cp: &CommandProcessor, _actions: Vec<EopAction>) {}
    fn sync_range(&mut self, _cp: &CommandProcessor, _addr: u64, _len: u64) {}
}
