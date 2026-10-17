//! The pipeline manifest (`tune pipe_manifest`): what a session translates and
//! compiles, for later sessions to compile again before the game needs it.
//!
//! Each new pipeline's stages - their translation inputs: the program's code
//! and its fetch subroutine's, where it was, the user SGPR count, the
//! specialization - and its render state are appended to
//! `pipeline-manifest.bin` beside the pipeline cache. At the start of a
//! session the recorded pipelines are translated and compiled again on the
//! compile workers, behind the session's own compiles
//! (`pcompile::spawn_low`), into the pipeline cache, then dropped: when the
//! game first needs one, the driver takes it from the cache in milliseconds.
//! After an update that changes the translator - which makes every pipeline
//! new - the places the player has been then do not stall on first sight
//! again. (Nothing is gained on a first run; with the pipelines still in the
//! cache the replay's compiles come from it too.) The replayed translations
//! stay in the translation cache, so the prepare thread does not make them
//! again. The replay runs once per build of tlg (and translation settings)
//! - another build may translate differently - or when the pipeline cache
//! starts empty; a replay cut short by the session's end runs again.

use super::render::FastMap;
use gcn::translate::Spec;
use gcn::translate::spec_io::{Reader, SPEC_IO_VERSION, Writer};
use parking_lot::Mutex;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

const MAGIC: &[u8; 8] = b"TLGPMAN1";
/// Record kinds: a translation's inputs, a graphics pipeline (its stages'
/// translations and render state), a compute pipeline.
const STAGE: u8 = 1;
const GFX: u8 = 2;
const CS: u8 = 3;
/// The largest program kept (dwords).
const MAX_CODE: usize = 1 << 16;

/// The inputs of one translation (`shaders::prepare`).
pub struct StageRec {
    pub code_hash: u64,
    pub key_hash: u64,
    pub addr: u64,
    pub user_sgprs: u16,
    pub main: Vec<u32>,
    pub fetch: Vec<u32>,
    pub spec: Spec,
}

impl StageRec {
    fn encode(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(64 + 4 * (self.main.len() + self.fetch.len()));
        let mut w = Writer(&mut b);
        w.u64(self.code_hash);
        w.u64(self.key_hash);
        w.u64(self.addr);
        w.u16(self.user_sgprs);
        w.words(&self.main);
        w.words(&self.fetch);
        self.spec.encode(&mut b);
        b
    }

    fn decode(b: &[u8]) -> Option<StageRec> {
        let mut r = Reader(b);
        let (code_hash, key_hash, addr, user_sgprs) = (r.u64()?, r.u64()?, r.u64()?, r.u16()?);
        let main = r.words(MAX_CODE)?;
        let fetch = r.words(MAX_CODE)?;
        let spec = Spec::decode(&mut r)?;
        r.0.is_empty().then_some(StageRec { code_hash, key_hash, addr, user_sgprs, main, fetch, spec })
    }
}

struct State {
    path: std::path::PathBuf,
    /// Translations and pipelines in the file or appended this session.
    stages_known: HashSet<(u64, u64)>,
    pipes_known: HashSet<Vec<u8>>,
    /// This session's translations not in the file (until a pipeline uses
    /// them).
    stages: FastMap<(u64, u64), Vec<u8>>,
    /// Records to append.
    pending: Vec<u8>,
    /// The file is missing or of another layout: written anew.
    fresh: bool,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);
static ON: AtomicBool = AtomicBool::new(false);
/// Replay: pipelines queued, done, compiled, skipped (not replayable), failed;
/// the workers' time on them, and when it started.
static REPLAY: [AtomicUsize; 5] = [const { AtomicUsize::new(0) }; 5];
static REPLAY_NS: AtomicU64 = AtomicU64::new(0);
static REPLAY_T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
/// Where a finished replay notes the build it was for, and that build.
static REPLAYED: std::sync::OnceLock<(std::path::PathBuf, String)> = std::sync::OnceLock::new();

/// This build of tlg and the settings its translations depend on: the
/// executable's size and time, the page marks, `TLG_SPIRV_NONCE`.
fn build_identity() -> String {
    let exe = std::env::current_exe().ok().and_then(|p| std::fs::metadata(p).ok()).map(|m| {
        let t = m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos());
        format!("{} {t}", m.len())
    });
    let Some(exe) = exe else { return String::new() };
    format!("{exe} marks {} nonce {}", crate::tune::get(crate::tune::PAGE_MARKS), std::env::var("TLG_SPIRV_NONCE").unwrap_or_default())
}

/// Notes that this build's replay is complete.
fn replay_complete() {
    if let Some((side, id)) = REPLAYED.get() {
        let _ = std::fs::write(side, id);
    }
}

/// Whether translations and pipelines are recorded this session.
pub fn on() -> bool {
    ON.load(Ordering::Relaxed)
}

fn record(out: &mut Vec<u8>, kind: u8, payload: &[u8]) {
    out.push(kind);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
}

/// The file's records: translations by key, pipelines in order; None when it
/// is not a manifest of this layout.
#[allow(clippy::type_complexity)]
fn parse(data: &[u8]) -> Option<(FastMap<(u64, u64), Vec<u8>>, Vec<(u8, Vec<u8>)>)> {
    let mut r = Reader(data);
    if r.bytes(8)? != MAGIC || r.u32()? != SPEC_IO_VERSION {
        return None;
    }
    let mut stages = FastMap::default();
    let mut pipes = Vec::new();
    // (A record cut short at the end - a session that ended while writing -
    // ends the list.)
    while let (Some(kind), Some(len)) = (r.u8(), r.u32()) {
        let Some(p) = r.bytes(len as usize) else { break };
        match kind {
            STAGE if p.len() >= 16 => {
                let k = (u64::from_le_bytes(p[..8].try_into().unwrap()), u64::from_le_bytes(p[8..16].try_into().unwrap()));
                stages.insert(k, p.to_vec());
            }
            GFX | CS => pipes.push((kind, p.to_vec())),
            _ => {}
        }
    }
    Some((stages, pipes))
}

/// A pipeline record's stages: (code hash, translation key) pairs, then the
/// render state (graphics).
fn pipe_payload(stages: &[(u64, u64)], key: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(1 + 16 * stages.len() + key.len());
    let mut w = Writer(&mut b);
    w.u8(stages.len() as u8);
    for &(c, k) in stages {
        w.u64(c);
        w.u64(k);
    }
    b.extend_from_slice(key);
    b
}

fn pipe_stages(p: &[u8]) -> Option<(Vec<(u64, u64)>, &[u8])> {
    let mut r = Reader(p);
    let n = r.u8()? as usize;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push((r.u64()?, r.u64()?));
    }
    Some((v, r.0))
}

/// Starts the manifest at `path` (with the pipeline cache): reads it and
/// queues the recorded pipelines' compiles behind the session's own.
pub fn init(path: std::path::PathBuf) {
    if crate::tune::get(crate::tune::PIPE_MANIFEST) == 0 {
        return;
    }
    let data = std::fs::read(&path).unwrap_or_default();
    let parsed = parse(&data);
    let fresh = parsed.is_none();
    if fresh && !data.is_empty() {
        crate::info!("pipeline manifest {}: another layout, started anew", path.display());
    }
    let (stages, pipes) = parsed.unwrap_or_default();
    // (Not again for the build that replayed last, unless the pipeline
    // cache lost its pipelines.)
    let side = path.with_extension("built");
    let id = build_identity();
    let empty = super::render::pipeline_cache_started_empty();
    let replay = id.is_empty() || std::fs::read_to_string(&side).map_or(true, |last| last != id) || empty;
    if !id.is_empty() {
        let _ = REPLAYED.set((side, id));
    }
    let mut state = State { path, stages_known: stages.keys().copied().collect(), pipes_known: HashSet::new(), stages: FastMap::default(), pending: Vec::new(), fresh };
    // Translations shared by pipelines are decoded once.
    let mut decoded: FastMap<(u64, u64), Option<Arc<StageRec>>> = FastMap::default();
    let mut jobs = Vec::new();
    for (kind, p) in pipes {
        // (Known even when it cannot be replayed: not recorded twice.)
        if !state.pipes_known.insert(p.clone()) {
            continue;
        }
        if !replay {
            continue;
        }
        let Some((keys, key)) = pipe_stages(&p) else { continue };
        let recs: Option<Vec<Arc<StageRec>>> = keys.iter().map(|k| decoded.entry(*k).or_insert_with(|| stages.get(k).and_then(|b| StageRec::decode(b)).map(Arc::new)).clone()).collect();
        if let Some(recs) = recs.filter(|r| !r.is_empty()) {
            jobs.push((kind, recs, key.to_vec()));
        }
    }
    let queued = jobs.len();
    REPLAY[0].store(queued, Ordering::Relaxed);
    let _ = REPLAY_T0.set(std::time::Instant::now());
    for (kind, recs, key) in jobs {
        super::pcompile::spawn_low(Box::new(move || replay_one(kind, &recs, &key)));
    }
    if queued > 0 {
        crate::info!("pipeline manifest: {queued} pipelines of earlier sessions ({} translations) queued to compile again behind this session's{}", decoded.len(), if empty { " (the pipeline cache started empty)" } else { "" });
    } else {
        if !replay {
            crate::info!("pipeline manifest: {} pipelines of earlier sessions, compiled before by this build", state.pipes_known.len());
        }
        replay_complete();
    }
    *STATE.lock() = Some(state);
    ON.store(true, Ordering::Relaxed);
}

/// Translates a recorded stage now (or takes this session's translation of
/// it): with this session's page marks.
fn translated(rec: &StageRec) -> Result<Arc<gcn::translate::Output>, String> {
    let mut spec = rec.spec.clone();
    spec.page_marks = crate::tune::get(crate::tune::PAGE_MARKS).clamp(0, 3) as u8;
    let key_hash = super::shaders::spec_key(&spec);
    if let Some(out) = super::shaders::cached_translation(rec.code_hash, key_hash) {
        return Ok(out);
    }
    let main = gcn::decode_program(&rec.main).map_err(|e| e.0)?;
    let code = if rec.fetch.is_empty() { main } else { gcn::translate::inline_fetch(&main, &gcn::decode_program(&rec.fetch).map_err(|e| e.0)?) };
    let plan = gcn::analysis::analyze(&code, rec.user_sgprs, rec.addr);
    if !plan.errors.is_empty() {
        return Err(plan.errors.join("; "));
    }
    let out = gcn::translate::translate(&code, &plan, &spec)?;
    Ok(super::shaders::insert_translation(rec.code_hash, key_hash, out, spec.page_marks))
}

/// One recorded pipeline: translated and compiled into the pipeline cache.
fn replay_one(kind: u8, recs: &[Arc<StageRec>], key: &[u8]) {
    let t0 = std::time::Instant::now();
    let r = (|| -> Result<bool, String> {
        let outs: Vec<Arc<gcn::translate::Output>> = recs.iter().map(|r| translated(r)).collect::<Result<_, _>>()?;
        match kind {
            GFX => super::render::replay_gfx(key, &outs[0], recs[0].spec.tess.role, outs.get(1).map(|o| &**o)),
            _ => super::render::replay_cs(&outs[0], recs[0].spec.tess.role),
        }
    })();
    REPLAY_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    match r {
        Ok(true) => REPLAY[2].fetch_add(1, Ordering::Relaxed),
        Ok(false) => REPLAY[3].fetch_add(1, Ordering::Relaxed),
        Err(e) => {
            static LOGGED: AtomicUsize = AtomicUsize::new(0);
            if LOGGED.fetch_add(1, Ordering::Relaxed) < 8 {
                crate::info!("pipeline manifest: a recorded pipeline could not be made again: {e}");
            }
            REPLAY[4].fetch_add(1, Ordering::Relaxed)
        }
    };
    let done = REPLAY[1].fetch_add(1, Ordering::Relaxed) + 1;
    if done == REPLAY[0].load(Ordering::Relaxed) {
        replay_complete();
        let wall = REPLAY_T0.get().map_or(0.0, |t| t.elapsed().as_secs_f64());
        crate::info!(
            "pipeline manifest: {done} pipelines of earlier sessions done in {wall:.1} s ({:.1} s of worker time): {} compiled, {} skipped (tessellation), {} failed",
            REPLAY_NS.load(Ordering::Relaxed) as f64 / 1e9,
            REPLAY[2].load(Ordering::Relaxed),
            REPLAY[3].load(Ordering::Relaxed),
            REPLAY[4].load(Ordering::Relaxed)
        );
    }
}

/// A translation made this session (`shaders::prepare`): kept until a
/// pipeline uses it. Translations with debug captures are not recorded.
#[allow(clippy::too_many_arguments)]
pub fn note_stage(code_hash: u64, key_hash: u64, addr: u64, user_sgprs: u16, main: Vec<u32>, fetch: Vec<u32>, spec: &Spec) {
    if !on() || !spec.debug_capture.is_empty() || spec.debug_pos_store || main.len() > MAX_CODE || fetch.len() > MAX_CODE {
        return;
    }
    let mut g = STATE.lock();
    let Some(s) = g.as_mut() else { return };
    let k = (code_hash, key_hash);
    if s.stages_known.contains(&k) || s.stages.contains_key(&k) {
        return;
    }
    let rec = StageRec { code_hash, key_hash, addr, user_sgprs, main, fetch, spec: spec.clone() };
    s.stages.insert(k, rec.encode());
}

/// A pipeline the renderer makes for the first time this session: recorded
/// with its stages' translations (once).
fn note_pipeline(kind: u8, stages: &[(u64, u64)], key: &[u8]) {
    if !on() {
        return;
    }
    let payload = pipe_payload(stages, key);
    let mut g = STATE.lock();
    let Some(s) = g.as_mut() else { return };
    if s.pipes_known.contains(&payload) {
        return;
    }
    // (Stages first: the file is read in order. A stage this session did not
    // record - debug captures - leaves the pipeline out.)
    if stages.iter().any(|k| !s.stages_known.contains(k) && !s.stages.contains_key(k)) {
        return;
    }
    for k in stages {
        if s.stages_known.insert(*k) {
            if let Some(b) = s.stages.remove(k) {
                record(&mut s.pending, STAGE, &b);
            }
        }
    }
    record(&mut s.pending, kind, &payload);
    s.pipes_known.insert(payload);
}

/// A new graphics pipeline: its stages' translations (vertex, pixel) and its
/// render state (`render::GfxKey::encode`).
pub fn note_gfx(key: &[u8], vs: (u64, u64), ps: Option<(u64, u64)>) {
    let stages: smallvec::SmallVec<[(u64, u64); 2]> = std::iter::once(vs).chain(ps).collect();
    note_pipeline(GFX, &stages, key);
}

/// A new compute pipeline: its translation.
pub fn note_cs(cs: (u64, u64)) {
    note_pipeline(CS, &[cs], &[]);
}

/// Appends the records made since the last call to the file (with the
/// pipeline cache's saves).
pub fn flush() {
    let (path, data, fresh) = {
        let mut g = STATE.lock();
        let Some(s) = g.as_mut() else { return };
        if s.pending.is_empty() {
            return;
        }
        let fresh = std::mem::replace(&mut s.fresh, false);
        (s.path.clone(), std::mem::take(&mut s.pending), fresh)
    };
    use std::io::Write;
    let r = if fresh {
        let mut all = Vec::with_capacity(12 + data.len());
        all.extend_from_slice(MAGIC);
        all.extend_from_slice(&SPEC_IO_VERSION.to_le_bytes());
        all.extend_from_slice(&data);
        std::fs::write(&path, all)
    } else {
        std::fs::OpenOptions::new().append(true).open(&path).and_then(|mut f| f.write_all(&data))
    };
    if let Err(e) = r {
        crate::warn!("pipeline manifest {}: {e}", path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_round_trip() {
        let spec = Spec {
            stage: gcn::translate::Stage::Compute,
            user_sgprs: 4,
            vs_vgpr_comp_cnt: 0,
            vs_step_rates: [1, 1],
            vs_out_cntl: 0,
            denorm_flush32: false,
            debug_pos_store: false,
            tess: Default::default(),
            ps_input_addr: 0,
            ps_inputs: Vec::new(),
            mrt: Default::default(),
            cs_local_size: [64, 1, 1],
            cs_rsrc2: 0x8C,
            resources: vec![gcn::translate::ResSpec::Sampler],
            buffer_strides: vec![0],
            binding_base: 0,
            debug_capture: Vec::new(),
            page_marks: 3,
        };
        let rec = StageRec { code_hash: 7, key_hash: 9, addr: 0x1130_0000, user_sgprs: 4, main: vec![0xBF81_0000], fetch: Vec::new(), spec: spec.clone() };
        let mut file = Vec::new();
        file.extend_from_slice(MAGIC);
        file.extend_from_slice(&SPEC_IO_VERSION.to_le_bytes());
        record(&mut file, STAGE, &rec.encode());
        record(&mut file, CS, &pipe_payload(&[(7, 9)], &[]));
        record(&mut file, GFX, &pipe_payload(&[(7, 9), (7, 9)], &[1, 2, 3]));
        // (A record cut short at the end is left out.)
        file.extend_from_slice(&[CS, 40, 0, 0, 0, 1]);
        let (stages, pipes) = parse(&file).unwrap();
        let back = StageRec::decode(&stages[&(7, 9)]).unwrap();
        assert_eq!((back.code_hash, back.key_hash, back.addr, back.user_sgprs), (7, 9, 0x1130_0000, 4));
        assert_eq!((back.main.as_slice(), back.fetch.len()), (&[0xBF81_0000u32][..], 0));
        assert_eq!(back.spec, spec);
        assert_eq!(pipes.len(), 2);
        let (keys, key) = pipe_stages(&pipes[1].1).unwrap();
        assert_eq!((pipes[1].0, keys.as_slice(), key), (GFX, &[(7, 9), (7, 9)][..], &[1u8, 2, 3][..]));
        assert!(parse(b"TLGPMAN0\x01\x00\x00\x00").is_none());
    }
}
