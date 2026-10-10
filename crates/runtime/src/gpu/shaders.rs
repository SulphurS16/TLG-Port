//! Shader front end: program decoding/caching, fetch-shader inlining,
//! descriptor resolution against live guest state, specialization and
//! translation to SPIR-V.

use super::cp::Regs;
use super::desc::{self, Ssharp, Tsharp, Vsharp};
use gcn::analysis::{self, Plan, Prov, Ptr, ResKind};
use gcn::translate::{self, ImageDim, MrtSpec, Output, PsInput, ResSpec, Spec, Stage, TessRole, TessSpec};
use parking_lot::Mutex;
use super::render::FastMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

pub const PS_BINDING_BASE: u32 = 32;

struct Program {
    len_words: usize,
    hash: u64,
    insts: Arc<Vec<gcn::Inst>>,
    /// SGPR holding the fetch subroutine's address (the first S_SWAPPC_B64's
    /// source), found once at decode instead of per draw.
    fetch_sgpr: Option<u16>,
    /// CPU epoch in which the code was last checked against `hash`.
    checked: u64,
}

static PROGRAMS: Mutex<Option<FastMap<u64, Program>>> = Mutex::new(None);

#[derive(Clone)]
pub struct Translated {
    pub out: Arc<Output>,
    pub key_hash: u64,
}

static CACHE: Mutex<Option<FastMap<(u64, u64), Translated>>> = Mutex::new(None);

/// Content hash of shader code (checked on every use of a cached program).
fn hash_words(w: &[u32]) -> u64 {
    let bytes = unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * 4) };
    super::resources::hash_slice(w.len() as u64, bytes)
}

thread_local! {
    /// Guest addresses read while preparing a shader (see `Prepared::mem_reads`).
    static READS: std::cell::RefCell<Vec<u64>> = const { std::cell::RefCell::new(Vec::new()) };
    /// Last mapping a read was validated against: (start, end, MAP_GEN).
    static LAST_VMA: std::cell::Cell<(u64, u64, u64)> = const { std::cell::Cell::new((0, 0, u64::MAX)) };
}

/// Records a guest read for `Prepared::mem_reads`, whose users only look at
/// pages: consecutive reads in one page are recorded once.
fn note_read(addr: u64) {
    READS.with(|r| {
        let mut r = r.borrow_mut();
        if r.last().is_none_or(|&l| l >> 12 != addr >> 12) {
            r.push(addr);
        }
    });
}

/// Whether guest [lo, hi) lies in one committed mapping. Descriptor tables of
/// consecutive draws sit in the same mappings: the lookup is skipped while
/// the mappings are unchanged.
fn mapped(lo: u64, hi: u64) -> bool {
    if lo >= hi || !crate::mem::is_guest(lo, hi - lo) {
        return false;
    }
    let generation = crate::mem::MAP_GEN.load(std::sync::atomic::Ordering::Acquire);
    let (vlo, vhi, g) = LAST_VMA.get();
    if g != generation || lo < vlo || hi > vhi {
        let st = crate::mem::MEM.lock();
        let Some(v) = st.find(lo) else { return false };
        if v.kind == crate::mem::VmaKind::Reserved || hi > v.end() {
            return false;
        }
        LAST_VMA.set((v.start, v.end(), generation));
    }
    true
}

fn guest_u32(addr: u64) -> Option<u32> {
    if addr % 4 != 0 || !mapped(addr, addr + 4) {
        return None;
    }
    note_read(addr);
    Some(unsafe { std::ptr::read_volatile(addr as *const u32) })
}

/// Reads `out.len()` consecutive guest dwords at `addr` with one mapping
/// check; false when unaligned or not mapped.
fn guest_read_words(addr: u64, out: &mut [u32]) -> bool {
    let end = addr + 4 * out.len() as u64;
    if addr % 4 != 0 || out.is_empty() || !mapped(addr, end) {
        return false;
    }
    note_read(addr);
    note_read(end - 4);
    for (k, w) in out.iter_mut().enumerate() {
        *w = unsafe { std::ptr::read_volatile((addr + 4 * k as u64) as *const u32) };
    }
    true
}

/// Appends `n` guest dwords from `addr` to `out` (0 for unreadable ones).
fn guest_read_into(addr: Option<u64>, n: u32, out: &mut Vec<u32>) {
    match addr {
        Some(a) if a % 4 == 0 && mapped(a, a + 4 * n as u64) => {
            note_read(a);
            note_read(a + 4 * n.max(1) as u64 - 4);
            out.extend((0..n as u64).map(|k| unsafe { std::ptr::read_volatile((a + 4 * k) as *const u32) }));
        }
        Some(a) => out.extend((0..n as u64).map(|k| guest_u32(a + 4 * k).unwrap_or(0))),
        None => out.extend(std::iter::repeat_n(0, n as usize)),
    }
}

/// Decodes (and caches) the program at `addr`: instructions, code hash and
/// the fetch-call SGPR (see `Program::fetch_sgpr`).
fn program(addr: u64) -> Result<(Arc<Vec<gcn::Inst>>, u64, Option<u16>), String> {
    // Code changes only through CPU writes (new CPU epoch) or the CP's
    // direct memory accesses (which start one too): checked once per epoch
    // that applies to the command (see `super::CPU_EPOCH`).
    let epoch = super::CPU_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
    let required = super::REQUIRED_EPOCH.with(|r| r.get());
    if let Some(p) = PROGRAMS.lock().as_ref().and_then(|m| m.get(&addr)).filter(|p| p.checked >= required) {
        return Ok((p.insts.clone(), p.hash, p.fetch_sgpr));
    }
    let code = super::shader_dump::code_slice(addr).ok_or_else(|| format!("shader {addr:#x} not mapped"))?;
    let mut g = PROGRAMS.lock();
    let map = g.get_or_insert_with(FastMap::default);
    if let Some(p) = map.get_mut(&addr) {
        if p.len_words <= code.len() && hash_words(&code[..p.len_words]) == p.hash {
            p.checked = epoch;
            return Ok((p.insts.clone(), p.hash, p.fetch_sgpr));
        }
    }
    let insts = gcn::decode_program(code).map_err(|e| format!("decode {addr:#x}: {}", e.0))?;
    let last = insts.last().unwrap();
    let len = (last.pc / 4 + last.len as u32) as usize;
    let hash = hash_words(&code[..len]);
    let fetch_sgpr = insts.iter().find(|i| i.enc == gcn::Enc::Sop1 && i.op == 33).map(|i| i.ssrc0());
    let insts = Arc::new(insts);
    map.insert(addr, Program { len_words: len, hash, insts: insts.clone(), checked: epoch, fetch_sgpr });
    Ok((insts, hash, fetch_sgpr))
}

fn eval_ptr<'a>(p: &'a Ptr, user: &[u32], memo: &mut PtrMemo<'a>) -> Option<u64> {
    match p {
        Ptr::User(i) => Some((*user.get(*i as usize)? as u64) | ((*user.get(*i as usize + 1)? as u64) << 32)),
        Ptr::Mem(inner, off) => {
            let base = eval_ptr_m(inner, user, memo)? + *off as u64;
            Some((guest_u32(base)? as u64) | ((guest_u32(base + 4)? as u64) << 32))
        }
        Ptr::Abs(a) => Some(*a),
        Ptr::Pair(lh) => Some(eval(&lh.0, user, memo)? as u64 | (eval(&lh.1, user, memo)? as u64) << 32),
    }
}

/// Pointers evaluated during one preparation: the dwords of a descriptor
/// (and scalar loads) share their table pointer, which would otherwise be
/// read from guest memory again for every dword.
type PtrMemo<'a> = smallvec::SmallVec<[(&'a Ptr, Option<u64>); 8]>;

fn eval_ptr_m<'a>(p: &'a Ptr, user: &[u32], memo: &mut PtrMemo<'a>) -> Option<u64> {
    if !matches!(p, Ptr::Mem(..) | Ptr::Pair(..)) {
        return eval_ptr(p, user, memo);
    }
    if let Some((_, v)) = memo.iter().find(|(q, _)| *q == p) {
        return *v;
    }
    let v = eval_ptr(p, user, memo);
    memo.push((p, v));
    v
}

/// Descriptor of a 1x1 opaque white RGBA8 texture in runtime-owned guest
/// memory (linear), standing in for image descriptors shaders load at run
/// time (see `prepare_tess`).
fn placeholder_tsharp() -> [u32; 8] {
    static T: std::sync::OnceLock<[u32; 8]> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        let a = crate::mem::arena_alloc(4096, 256);
        super::resources::copy_to_guest(&[0xFF; 4096], a);
        // DATA_FORMAT 8_8_8_8 UNORM; DST_SEL xyzw, TILING_INDEX 8 (linear), TYPE 2D.
        [(a >> 8) as u32, ((a >> 40) & 0xFF) as u32 | (10 << 20), 0, 0xFAC | (8 << 20) | ((desc::T_2D as u32) << 28), 0, 0, 0, 0]
    })
}

fn eval<'a>(p: &'a Prov, user: &[u32], memo: &mut PtrMemo<'a>) -> Option<u32> {
    match p {
        Prov::User(i) => user.get(*i as usize).copied(),
        Prov::Const(c) => Some(*c),
        Prov::Mem(ptr, off) => guest_u32(eval_ptr_m(ptr, user, memo)? + *off as u64),
        Prov::PtrLo(ptr, b) => Some((eval_ptr_m(ptr, user, memo)? + *b as u64) as u32),
        Prov::PtrHi(ptr, b) => Some(((eval_ptr_m(ptr, user, memo)? + *b as u64) >> 32) as u32),
        Prov::Op(o, ab) => Some(o.apply(eval(&ab.0, user, memo)?, eval(&ab.1, user, memo)?)),
        // (Run-time values: the shader computes them.)
        Prov::DynLo(_) | Prov::DynHi(_) | Prov::Unknown => None,
    }
}

#[derive(Clone, Debug)]
pub enum ResolvedRes {
    Buffer(Vsharp),
    Image(Tsharp),
    Sampler(Ssharp),
}

/// Everything needed to bind one shader stage for a draw/dispatch.
pub struct Prepared {
    pub stage: Stage,
    pub addr: u64,
    pub translated: Translated,
    pub resources: Vec<ResolvedRes>,
    pub sgpr_data: Vec<u32>,
    /// Buffer resources written by the shader (need write-back to guest memory).
    pub written_buffers: Arc<Vec<usize>>,
    /// Buffer resources read by the shader (need current guest contents).
    pub read_buffers: Arc<Vec<usize>>,
    /// Debug: instruction pcs using each resource (frame traces).
    pub res_pcs: Arc<Vec<Vec<u32>>>,
    /// Guest addresses read while resolving descriptors and scalar loads
    /// (record time): they must not have pending GPU writes.
    pub mem_reads: Vec<u64>,
    /// The engine's scatter kernel: (records resource, destination resource).
    pub scatter: Option<(usize, usize)>,
    /// Destination dwords the scatter records write, as byte ranges (set by
    /// the renderer from the records before the dispatch is recorded).
    pub sparse_writes: Option<Vec<(u64, u64)>>,
    /// The engine's buffer copy / fill kernels (see `Utility`).
    pub utility: Option<Utility>,
    /// Role in a tessellated draw (LS/HS: compute, DS: evaluation stage).
    pub tess: TessRole,
}

/// Engine utility kernels the renderer can perform as image operations when
/// they cover a whole render-target surface.
#[derive(Clone, Copy, Debug)]
pub enum Utility {
    /// Thread t (64 per workgroup) copies dword t of `src` to `dst`
    /// (bounded by the destination's records).
    Copy { src: usize, dst: usize },
    /// Every element of `dst` covered by the dispatch (`per_group` elements
    /// per workgroup) gets the 4-dword value read from `value` (offset 0),
    /// stored through the destination's format.
    Fill { value: usize, dst: usize, per_group: u64 },
    /// SOTC's pattern copy: thread t (64 per workgroup) below `count` and the
    /// destination's records stores element `t & m` (`modulo`: `t % m`) of
    /// `src` to element t of `dst` (format x loads and stores); count and m
    /// are dwords 0 and 1 of `params`. m = 0 (1) is a fill, a period at
    /// least as long as the dispatch a copy.
    Pattern { src: usize, dst: usize, params: usize, modulo: bool },
}

fn match_utility(code: &[gcn::Inst], plan: &Plan) -> Option<Utility> {
    let insts: Vec<&gcn::Inst> = code.iter().filter(|i| i.name() != "s_waitcnt").collect();
    let names: Vec<String> = insts.iter().map(|i| i.name()).collect();
    let is = |p: &[&str]| names.len() == p.len() && names.iter().zip(p).all(|(a, b)| a == b);
    let res = |k: usize| plan.inst_res.get(&insts[k].pc).map(|r| r.0);
    // SOTC: [markers] count check, m, (t & m | t % m), load, bound check, store.
    const PATTERN_HEAD: [&str; 9] = ["s_mov_b32", "s_mov_b32", "s_ttracedata", "s_mul_i32", "s_buffer_load_dword", "v_add_i32", "v_cmpx_gt_u32", "s_cbranch_execz", "s_buffer_load_dword"];
    const PATTERN_TAIL: [&str; 4] = ["buffer_load_format_x", "v_cmpx_gt_u32", "buffer_store_format_x", "s_endpgm"];
    let n = names.len();
    if n > 13 && names[..9].iter().zip(PATTERN_HEAD).all(|(a, b)| a == b) && names[n - 4..].iter().zip(PATTERN_TAIL).all(|(a, b)| a == b) {
        let mid = &names[9..n - 4];
        let modulo = match mid.len() {
            1 if mid[0] == "v_and_b32" => Some(false),
            24 if mid.iter().any(|m| m == "v_rcp_iflag_f32") && mid.iter().filter(|m| *m == "v_mul_hi_u32").count() == 2 => Some(true),
            _ => None,
        };
        let offs = (insts[4].smrd_const_offset(), insts[8].smrd_const_offset());
        if let (Some(modulo), (Some(0), Some(4))) = (modulo, offs) {
            let params = res(4)?;
            if res(8)? == params {
                return Some(Utility::Pattern { src: res(n - 4)?, dst: res(n - 2)?, params, modulo });
            }
        }
    }
    const COPY: [&str; 8] = ["s_mov_b32", "s_lshl_b32", "v_add_i32", "buffer_load_format_x", "v_cmp_gt_u32", "s_and_saveexec_b64", "buffer_store_format_x", "s_endpgm"];
    const FILL1: [&str; 10] = ["s_mov_b32", "s_lshl_b32", "v_add_i32", "s_buffer_load_dwordx4", "v_mov_b32", "v_mov_b32", "v_mov_b32", "v_mov_b32", "buffer_store_format_xyzw", "s_endpgm"];
    const FILL2: [&str; 22] = [
        "s_mov_b32",
        "s_lshl_b32",
        "v_add_i32",
        "s_buffer_load_dwordx4",
        "v_lshrrev_b32",
        "v_sub_i32",
        "v_lshlrev_b32",
        "v_add_i32",
        "v_lshlrev_b32",
        "v_add_i32",
        "v_mov_b32",
        "v_mov_b32",
        "v_mov_b32",
        "v_mov_b32",
        "v_add_i32",
        "v_mov_b32",
        "v_mov_b32",
        "v_mov_b32",
        "v_mov_b32",
        "buffer_store_format_xyzw",
        "buffer_store_format_xyzw",
        "s_endpgm",
    ];
    if is(&COPY) {
        return Some(Utility::Copy { src: res(3)?, dst: res(6)? });
    }
    if is(&FILL1) {
        return Some(Utility::Fill { value: res(3)?, dst: res(8)?, per_group: 64 });
    }
    if is(&FILL2) {
        return Some(Utility::Fill { value: res(3)?, dst: res(19)?, per_group: 128 });
    }
    None
}

/// Per-program analysis (independent of user-data values), cached by
/// (code hash, address, user SGPR count).
struct Analyzed {
    code: Arc<Vec<gcn::Inst>>,
    plan: Plan,
    res_pcs: Arc<Vec<Vec<u32>>>,
    written_buffers: Arc<Vec<usize>>,
    read_buffers: Arc<Vec<usize>>,
    scatter: Option<(usize, usize)>,
    utility: Option<Utility>,
    /// Exports to pos1..3 (vertex shaders): their use depends on
    /// PA_CL_VS_OUT_CNTL, which then becomes part of the translation key.
    extra_pos: bool,
    /// Per resource: its descriptor is consecutive dwords behind one pointer.
    res_block: Vec<bool>,
}

static ANALYZED: Mutex<Option<FastMap<(u64, u64, u16), Arc<Analyzed>>>> = Mutex::new(None);

/// Recognizes the engine's scatter kernel: per workgroup a record {dst dword
/// index, src dword index, count - 1} is loaded from buffer R; lanes
/// 0..=count copy src[src + lane] to dst[dst + lane] (bounded by the
/// destination size). Its destination binding spans gigabytes while each
/// dispatch writes a few kilobytes; the records say which.
fn match_scatter(code: &[gcn::Inst], plan: &Plan) -> Option<(usize, usize)> {
    const PATTERN: [&str; 13] = [
        "s_mov_b32",
        "v_mov_b32",
        "buffer_load_format_xyz",
        "v_cmp_ge_u32",
        "s_and_saveexec_b64",
        "v_add_i32",
        "s_cbranch_execz",
        "buffer_load_format_x",
        "v_add_i32",
        "v_cmp_gt_u32",
        "s_and_saveexec_b64",
        "buffer_store_format_x",
        "s_endpgm",
    ];
    let insts: Vec<&gcn::Inst> = code.iter().filter(|i| i.name() != "s_waitcnt").collect();
    if insts.len() != PATTERN.len() || insts.iter().zip(PATTERN).any(|(i, p)| i.name() != p) {
        return None;
    }
    let records = plan.inst_res.get(&insts[2].pc)?.0;
    let dst = plan.inst_res.get(&insts[11].pc)?.0;
    (records != dst).then_some((records, dst))
}

/// Register state for a stage, captured at draw time.
pub struct StageRegs<'a> {
    pub stage: Stage,
    pub regs: &'a Regs,
}

const VS_PGM_LO: u32 = 0x2C48;
const LS_PGM_LO: u32 = 0x2D48;
const HS_PGM_LO: u32 = 0x2D08;
const PS_PGM_LO: u32 = 0x2C08;
const CS_PGM_LO: u32 = 0x2E0C;

fn image_dim(t: &Tsharp) -> ImageDim {
    match t.ty {
        desc::T_1D => ImageDim::D1,
        desc::T_3D => ImageDim::D3,
        desc::T_CUBE => {
            if t.last_array > t.base_array { ImageDim::CubeArray } else { ImageDim::Cube }
        }
        desc::T_1D_ARRAY => ImageDim::D1Array,
        desc::T_2D_ARRAY | desc::T_2D_MSAA_ARRAY => ImageDim::D2Array,
        _ => ImageDim::D2,
    }
}

/// Debug (`TLG_PS_DEBUG=<shader addr>:<pc>:v<N>[:<pc2>:v<M>][;<addr>:...]`, `u<N>` converts
/// unsigned integers to floats divided by 255):
/// export v[N..N+2] captured after the instruction at `pc` instead of the
/// shader's outputs (first capture to MRT0, second to MRT1 and beyond).
fn debug_capture(addr: u64) -> Vec<(u32, u32, bool)> {
    static SPEC: std::sync::OnceLock<Vec<(u64, Vec<(u32, u32, bool)>)>> = std::sync::OnceLock::new();
    let specs = SPEC.get_or_init(|| {
        let Ok(v) = std::env::var("TLG_PS_DEBUG") else { return Vec::new() };
        v.split(';')
            .filter_map(|e| {
                let mut it = e.trim().split(':');
                let a = u64::from_str_radix(it.next()?.trim_start_matches("0x"), 16).ok()?;
                let mut caps = Vec::new();
                while let (Some(pc), Some(r)) = (it.next(), it.next()) {
                    // vN: raw bits; uN: unsigned integer converted to float.
                    let int = r.starts_with('u');
                    caps.push((u32::from_str_radix(pc.trim_start_matches("0x"), 16).ok()?, r.trim_start_matches(['v', 'u']).parse().ok()?, int));
                }
                Some((a, caps))
            })
            .collect()
    });
    specs.iter().find(|(a, _)| *a == addr).map(|(_, c)| c.clone()).unwrap_or_default()
}

/// TLG_VS_CAPTURE=<pc>:v<N>,... (with TLG_VS_POS): registers of the debugged
/// vertex shader captured after those instructions (up to 12).
pub fn debug_vs_capture() -> Vec<(u32, u32, bool)> {
    std::env::var("TLG_VS_CAPTURE")
        .map(|v| {
            v.split(',')
                .filter_map(|e| {
                    let (pc, r) = e.trim().split_once(':')?;
                    Some((u32::from_str_radix(pc.trim_start_matches("0x"), 16).ok()?, r.trim_start_matches('v').parse().ok()?, false))
                })
                .take(12)
                .collect()
        })
        .unwrap_or_default()
}

/// TLG_VS_POS=<vs addr>[,<vs addr>...]: whether `addr` is one of them.
pub fn debug_vs_pos(addr: u64) -> bool {
    static A: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    A.get_or_init(|| std::env::var("TLG_VS_POS").map(|v| v.split(',').filter_map(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok()).collect()).unwrap_or_default())
        .contains(&addr)
}

pub fn prepare(sr: &StageRegs) -> Result<Prepared, String> {
    prepare_tess(sr, TessSpec::default())
}

/// `prepare` for a stage of a tessellated draw: `tess.role` LS or HS (with
/// `Stage::Compute`) takes that stage's program and user data, DS (with
/// `Stage::Vertex`) the VS stage's (see `TessRole`).
pub fn prepare_tess(sr: &StageRegs, mut tess: TessSpec) -> Result<Prepared, String> {
    let r = sr.regs;
    let (pgm_lo, user_base, rsrc1, rsrc2) = match (sr.stage, tess.role) {
        (Stage::Compute, TessRole::Ls) => (LS_PGM_LO, 0x2D4C, r.sh(0x2D4A), r.sh(0x2D4B)),
        (Stage::Compute, TessRole::Hs) => (HS_PGM_LO, 0x2D0C, r.sh(0x2D0A), r.sh(0x2D0B)),
        (Stage::Vertex, _) => (VS_PGM_LO, 0x2C4C, r.sh(0x2C4A), r.sh(0x2C4B)),
        (Stage::Fragment, _) => (PS_PGM_LO, 0x2C0C, r.sh(0x2C0A), r.sh(0x2C0B)),
        (Stage::Compute, _) => (CS_PGM_LO, 0x2E40, r.sh(0x2E12), r.sh(0x2E13)),
    };
    tess.rsrc2 = rsrc2;
    let addr = ((r.sh(pgm_lo) as u64) | ((r.sh(pgm_lo + 1) as u64) << 32)) << 8;
    let ub = (user_base - 0x2C00) as usize;
    let user: [u32; 16] = r.sh[ub..ub + 16].try_into().unwrap();
    let user_sgprs = ((rsrc2 >> 1) & 0x1F) as u16;
    let (main, main_hash, fetch_sgpr) = program(addr)?;
    // The fetch subroutine (VS) is inlined.
    let fetch = match fetch_sgpr {
        Some(s) if sr.stage == Stage::Vertex || tess.role == TessRole::Ls => {
            let s = s as usize;
            let faddr = (*user.get(s).ok_or("fetch pointer outside user data")? as u64) | ((*user.get(s + 1).ok_or("fetch pointer outside user data")? as u64) << 32);
            let (f, fh, _) = program(faddr)?;
            Some((f, fh))
        }
        _ => None,
    };
    let code_hash = match &fetch {
        Some((_, fh)) => main_hash ^ fh.rotate_left(17),
        None => main_hash,
    };
    let akey = (code_hash, addr, user_sgprs);
    let cached = ANALYZED.lock().as_ref().and_then(|m| m.get(&akey).cloned());
    let analyzed = match cached {
        Some(a) => a,
        None => {
            // (Diagnostics, once per shader: the RSRC1 FLOAT_MODE [19:12],
            // DX10_CLAMP [21] and IEEE_MODE [23] combinations in use.)
            static SEEN: Mutex<Option<std::collections::HashSet<(u8, u32)>>> = Mutex::new(None);
            let mode = (rsrc1 >> 12) & 0xAFF;
            if SEEN.lock().get_or_insert_with(Default::default).insert((sr.stage as u8, mode)) {
                crate::info!("shader float mode: {:?} FLOAT_MODE {:#04x} DX10_CLAMP {} IEEE_MODE {} (first at {addr:#x})", sr.stage, mode & 0xFF, (rsrc1 >> 21) & 1, (rsrc1 >> 23) & 1);
            }
            let code = match &fetch {
                Some((f, _)) => Arc::new(translate::inline_fetch(&main, f)),
                None => main.clone(),
            };
            let plan = analysis::analyze(&code, user_sgprs, addr);
            let mut res_pcs = vec![Vec::new(); plan.resources.len()];
            for (&pc, &(r, s)) in &plan.inst_res {
                if let Some(v) = res_pcs.get_mut(r) {
                    v.push(pc);
                }
                if let Some(v) = s.and_then(|s| res_pcs.get_mut(s)) {
                    v.push(pc);
                }
            }
            for v in &mut res_pcs {
                v.sort_unstable();
            }
            let written_buffers: Vec<usize> = plan.resources.iter().enumerate().filter(|(_, r)| r.kind == ResKind::Buffer && r.written).map(|(i, _)| i).collect();
            let read_buffers: Vec<usize> = plan.resources.iter().enumerate().filter(|(_, r)| r.kind == ResKind::Buffer && r.read).map(|(i, _)| i).collect();
            let scatter = if sr.stage == Stage::Compute { match_scatter(&code, &plan) } else { None };
            let utility = if sr.stage == Stage::Compute { match_utility(&code, &plan) } else { None };
            let extra_pos = code.iter().any(|i| i.enc == gcn::Enc::Exp && (13..=15).contains(&i.exp_target()));
            // Descriptors whose dwords are consecutive dwords behind one
            // pointer (the usual table entry) are read as a block.
            let res_block = plan
                .resources
                .iter()
                .map(|r| match r.dwords.first() {
                    Some(Prov::Mem(p0, o0)) => r.dwords.len() <= 8 && r.dwords.iter().enumerate().all(|(k, d)| matches!(d, Prov::Mem(p, o) if p == p0 && *o == o0 + 4 * k as u32)),
                    _ => false,
                })
                .collect();
            let a = Arc::new(Analyzed { code, plan, res_pcs: Arc::new(res_pcs), written_buffers: Arc::new(written_buffers), read_buffers: Arc::new(read_buffers), scatter, utility, extra_pos, res_block });
            ANALYZED.lock().get_or_insert_with(FastMap::default).insert(akey, a.clone());
            a
        }
    };
    let code = &analyzed.code;
    let plan = &analyzed.plan;
    READS.with(|r| r.borrow_mut().clear());
    let stage_name = match (sr.stage, tess.role) {
        (_, TessRole::Ls) => "ls",
        (_, TessRole::Hs) => "hs",
        (_, TessRole::Ds) => "ds",
        (Stage::Vertex, _) => "vs",
        (Stage::Fragment, _) => "ps",
        (Stage::Compute, _) => "cs",
    };
    if !plan.errors.is_empty() {
        super::shader_dump::dump(stage_name, addr, &user);
        return Err(format!("{:?} shader {addr:#x}: {}", sr.stage, plan.errors.join("; ")));
    }
    if !plan.inst_dyn.is_empty() && crate::tune::get(crate::tune::DYN_LOADS) == 0 {
        return Err(format!("{:?} shader {addr:#x}: scalar loads through run-time addresses disabled (tune dyn_loads)", sr.stage));
    }
    // Resolve descriptors.
    let mut memo = PtrMemo::new();
    let mut resources = Vec::with_capacity(plan.resources.len());
    let mut specs = Vec::with_capacity(plan.resources.len());
    let mut strides = Vec::with_capacity(plan.resources.len());
    for (i, res) in plan.resources.iter().enumerate() {
        let mut w = [0u32; 8];
        let block = analyzed.res_block[i]
            && match &res.dwords[0] {
                Prov::Mem(ptr, off) => eval_ptr_m(ptr, &user, &mut memo).is_some_and(|b| guest_read_words(b + *off as u64, &mut w[..res.dwords.len()])),
                _ => false,
            };
        for (k, p) in res.dwords.iter().enumerate().filter(|_| !block) {
            w[k] = match eval(p, &user, &mut memo) {
                Some(v) => v,
                // Sampled images and samplers the shader loads at run time
                // from memory the CPU cannot resolve (e.g. per-light textures
                // in the tiled lighting pass's light loop): stand-ins, so the
                // rest of the shader runs.
                None if matches!(res.kind, ResKind::Image | ResKind::Sampler) && !res.written => {
                    static LOGGED: Mutex<Option<std::collections::HashSet<(u64, usize)>>> = Mutex::new(None);
                    if LOGGED.lock().get_or_insert_with(Default::default).insert((addr, i)) {
                        crate::info!("{:?} shader {addr:#x}: {:?} resource {i} loaded at run time: bound to a stand-in ({p:?})", sr.stage, res.kind);
                    }
                    w = if res.kind == ResKind::Image { placeholder_tsharp() } else { [0; 8] };
                    break;
                }
                None => {
                    super::shader_dump::dump(stage_name, addr, &user);
                    static LOGGED: Mutex<Option<std::collections::HashSet<u64>>> = Mutex::new(None);
                    if LOGGED.lock().get_or_insert_with(Default::default).insert(addr) {
                        crate::info!("{:?} shader {addr:#x}: resource {i} dword {k} ({p:?}) unresolvable with user data {:x?}", sr.stage, &user[..(user_sgprs as usize).min(16)]);
                    }
                    return Err(format!("{:?} shader {addr:#x}: resource {i} dword {k} unresolvable ({p:?})", sr.stage));
                }
            };
        }
        match res.kind {
            ResKind::Buffer => {
                let v = Vsharp::decode(&w);
                specs.push(ResSpec::Buffer { dfmt: v.dfmt, nfmt: v.nfmt, dst_sel: v.dst_sel });
                strides.push(v.stride);
                resources.push(ResolvedRes::Buffer(v));
            }
            ResKind::Image => {
                let t = Tsharp::decode(&w);
                specs.push(ResSpec::Image { dim: image_dim(&t), class: desc::num_class(t.nfmt) });
                strides.push(0);
                resources.push(ResolvedRes::Image(t));
            }
            ResKind::Sampler => {
                specs.push(ResSpec::Sampler);
                strides.push(0);
                resources.push(ResolvedRes::Sampler(Ssharp([w[0], w[1], w[2], w[3]])));
            }
        }
    }
    // Stage specialization.
    let mut spec = Spec {
        stage: sr.stage,
        user_sgprs,
        vs_vgpr_comp_cnt: (rsrc1 >> 24) & 3,
        vs_step_rates: if sr.stage == Stage::Vertex { [r.ctx(0xA2A8).max(1), r.ctx(0xA2A9).max(1)] } else { [1, 1] },
        vs_out_cntl: 0,
        // RSRC1 FLOAT_MODE [17:16]: FP32 denormal mode (0 = flush input and
        // output; 1 and 2 flush one side: also flushed here).
        denorm_flush32: (rsrc1 >> 16) & 3 != 3 && super::vk::ctx().denorm_flush32 && crate::tune::get(crate::tune::FTZ) != 0,
        // TLG_VS_POS=<vs addr>: that vertex shader also stores its position
        // exports (traced frames analyse them, see `render::check_vs_positions`).
        debug_pos_store: sr.stage == Stage::Vertex && tess.role == TessRole::None && debug_vs_pos(addr),
        tess,
        ps_input_addr: 0,
        ps_inputs: Vec::new(),
        mrt: [MrtSpec::default(); 8],
        cs_local_size: [1, 1, 1],
        cs_rsrc2: 0,
        resources: specs,
        buffer_strides: strides,
        binding_base: if sr.stage == Stage::Fragment { PS_BINDING_BASE } else { 0 },
        debug_capture: if sr.stage == Stage::Vertex && debug_vs_pos(addr) { debug_vs_capture() } else { debug_capture(addr) },
    };
    match sr.stage {
        Stage::Fragment => {
            spec.ps_input_addr = r.ctx(0xA1B4);
            let num_interp = r.ctx(0xA1B6) & 0x3F;
            spec.ps_inputs.reserve_exact(num_interp.min(32) as usize);
            for n in 0..num_interp.min(32) {
                let c = r.ctx(0xA191 + n);
                let off = c & 0x3F;
                spec.ps_inputs.push(PsInput {
                    param: (off & 0x1F) as u8,
                    flat: (c >> 10) & 1 != 0,
                    default: if off & 0x20 != 0 { Some(((c >> 8) & 3) as u8) } else { None },
                });
            }
            let col = r.ctx(0xA1C5);
            for m in 0..8 {
                let info = r.ctx(0xA31C + m * 0xF);
                let nt = (info >> 8) & 7;
                spec.mrt[m as usize] = MrtSpec { export_format: ((col >> (m * 4)) & 0xF) as u8, class: match nt { 4 => 2, 5 => 1, _ => 0 } };
            }
        }
        Stage::Compute => match tess.role {
            // LS: one invocation per control point; HS: a workgroup per
            // threadgroup (see `TessRole`).
            TessRole::Ls => spec.cs_local_size = [64, 1, 1],
            TessRole::Hs => spec.cs_local_size = [(tess.patches * tess.out_cp).max(1), 1, 1],
            _ => {
                spec.cs_local_size = [r.sh(0x2E07) & 0xFFFF, r.sh(0x2E08) & 0xFFFF, r.sh(0x2E09) & 0xFFFF];
                spec.cs_rsrc2 = rsrc2;
            }
        },
        Stage::Vertex => {
            if analyzed.extra_pos && crate::tune::get(crate::tune::CLIP_DIST) != 0 {
                // PA_CL_VS_OUT_CNTL: CLIP/CULL_DIST_ENA_0..7 and the MISC /
                // CCDIST0 / CCDIST1 vector enables (distances the device
                // cannot honour are dropped).
                let (clip, cull) = super::vk::ctx().clip_cull;
                let mut m = 0x00E0_0000;
                if clip {
                    m |= 0xFF;
                }
                if cull {
                    m |= 0xFF00;
                }
                spec.vs_out_cntl = r.ctx(0xA207) & m;
            }
        }
    }
    let key_hash = {
        let mut h = super::render::FxHasher::default();
        spec.hash(&mut h);
        h.finish()
    };
    let translated = {
        let mut g = CACHE.lock();
        let map = g.get_or_insert_with(FastMap::default);
        match map.get(&(code_hash, key_hash)) {
            Some(t) => t.clone(),
            None => {
                super::shader_dump::dump(stage_name, addr, &user);
                let out = translate::translate(code, plan, &spec).map_err(|e| format!("{:?} shader {addr:#x}: {e}", sr.stage))?;
                for w in &out.warnings {
                    crate::warn!("{:?} shader {addr:#x}: {w}", sr.stage);
                }
                if super::shader_dump::DIR.get().is_some() {
                    let bytes: Vec<u8> = out.spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
                    super::shader_dump::write_file(format!("{stage_name}_{addr:x}_{key_hash:016x}.spv"), bytes);
                }
                let t = Translated { out: Arc::new(out), key_hash };
                map.insert((code_hash, key_hash), t.clone());
                t
            }
        }
    };
    // Scalar-load data (resolved now, per draw).
    let mut sgpr_data = Vec::with_capacity(translate::SGPR_DATA_USER as usize + plan.loads.iter().map(|l| l.dwords as usize).sum::<usize>() + resources.len());
    sgpr_data.extend_from_slice(&user);
    sgpr_data.resize(translate::SGPR_DATA_USER as usize, 0);
    for l in &plan.loads {
        let base = eval_ptr_m(&l.ptr, &user, &mut memo);
        guest_read_into(base.map(|b| b + l.offset as u64), l.dwords, &mut sgpr_data);
    }
    // Per-resource binding biases (see translate::bias_offset): read-only
    // buffers are bound at their 16-byte-aligned address.
    for (i, res) in resources.iter().enumerate() {
        let written = plan.resources[i].written;
        sgpr_data.push(match res {
            ResolvedRes::Buffer(v) if !written => (v.base & 15) as u32,
            _ => 0,
        });
    }
    // Values the shader needs (see translate::values_offset).
    for v in &plan.values {
        sgpr_data.push(eval(v, &user, &mut memo).unwrap_or(0));
    }
    let written_buffers = analyzed.written_buffers.clone();
    let read_buffers = analyzed.read_buffers.clone();
    let res_pcs = analyzed.res_pcs.clone();
    let scatter = analyzed.scatter;
    let utility = analyzed.utility;
    let mem_reads = READS.with(|r| std::mem::take(&mut *r.borrow_mut()));
    Ok(Prepared { stage: sr.stage, addr, translated, resources, sgpr_data, written_buffers, read_buffers, res_pcs, mem_reads, scatter, sparse_writes: None, utility, tess: tess.role })
}
