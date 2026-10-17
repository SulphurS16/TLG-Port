//! GCN -> SPIR-V translation.
//!
//! Execution model: each SPIR-V invocation emulates one GCN lane as a wave of
//! one. Scalar registers are per-invocation copies (all inputs are uniform, so
//! values agree across lanes); EXEC/VCC masks keep this lane's bit in bit 0.
//! VALU results are written only when the lane is active. Branchy programs are
//! structured with a loop + switch dispatcher over basic blocks.

mod alu;
mod cfg;
pub mod gs;
pub mod kernels;
mod mem;
pub mod spec_io;
pub mod tess;
mod wave;

use crate::analysis::{Plan, ResKind};
use crate::{Enc, Inst};
use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{self, BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel, StorageClass, Word};
use crate::DetMap as HashMap;
use std::collections::BTreeSet;

thread_local! {
    /// Measurements only (the runtime's `TLG_COMPILE_PROBE`): 1 = buffer
    /// stores leave out their page marks (`mem::mark_dirty`; the bindings
    /// stay); 2 = they mark with a plain atomic per active lane (no subgroup
    /// operations); 3 = they set a word per page with a plain store (a page
    /// word map), so the driver's compile time can be compared. Such
    /// shaders are never used.
    pub static PROBE_MARKS: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    Vertex,
    Fragment,
    Compute,
}

/// A program's role in a tessellated draw (VGT_SHADER_STAGES_EN: LS and HS
/// on, the VS stage running the domain shader). LS and HS programs run as
/// compute (`Stage::Compute`, one dispatch each), the DS as a tessellation
/// evaluation shader (`Stage::Vertex`). The LDS the three share is a storage
/// buffer (`BindingKind::Lds`) with one `lds_dwords` region per threadgroup
/// of `patches` patches, as the hardware allocates it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TessRole {
    #[default]
    None,
    Ls,
    Hs,
    Ds,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TessSpec {
    pub role: TessRole,
    /// VGT_LS_HS_CONFIG: patches per threadgroup, input and output control
    /// points per patch.
    pub patches: u32,
    pub in_cp: u32,
    pub out_cp: u32,
    /// LDS dwords per threadgroup (SPI_SHADER_PGM_RSRC2_LS.LDS_SIZE).
    pub lds_dwords: u32,
    /// VGT_TF_PARAM: domain [1:0], partitioning [4:2], output topology [7:5].
    pub tf_param: u32,
    /// The program's SPI_SHADER_PGM_RSRC2 (HS: OC_LDS_EN [7] and TG_SIZE_EN
    /// [8] add system SGPRs after the user SGPRs).
    pub rsrc2: u32,
}

/// Tessellation factors per patch in the factor ring by VGT_TF_PARAM's
/// domain: isoline 2 outer; triangle 3 outer, 1 inner; quad 4 outer, 2 inner.
pub fn tf_dwords(tf_param: u32) -> u32 {
    match tf_param & 3 {
        0 => 2,
        1 => 4,
        _ => 6,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageDim {
    D1,
    D2,
    D3,
    Cube,
    D1Array,
    D2Array,
    CubeArray,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NumClass {
    Float,
    Sint,
    Uint,
}

/// Fragment output / export conversion for one MRT.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct MrtSpec {
    /// SPI_SHADER_COL_FORMAT value for this target (0 = none).
    pub export_format: u8,
    /// 0 float, 1 sint, 2 uint (render target number type).
    pub class: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PsInput {
    /// VS parameter index feeding this attribute (SPI_PS_INPUT_CNTL.OFFSET).
    pub param: u8,
    pub flat: bool,
    /// Use a constant default instead of a VS output (0..3 per DEFAULT_VAL).
    pub default: Option<u8>,
}

/// Per-resource specialization resolved from descriptors at draw time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResSpec {
    Buffer { dfmt: u8, nfmt: u8, dst_sel: u16 },
    Image { dim: ImageDim, class: NumClass },
    Sampler,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Spec {
    pub stage: Stage,
    pub user_sgprs: u16,
    pub vs_vgpr_comp_cnt: u32,
    /// VGT_INSTANCE_STEP_RATE_0/1: VS input v1/v2 = instance id / rate.
    pub vs_step_rates: [u32; 2],
    /// PA_CL_VS_OUT_CNTL for vertex shaders exporting pos1..3 (else 0):
    /// which of those exports carry clip/cull distances and which distances
    /// clip or cull (see `dist_layout`).
    pub vs_out_cntl: u32,
    /// 32-bit float denormals are flushed to zero (RSRC1 FLOAT_MODE FP32
    /// denorm mode 0, which the title's shaders use; needs the device's
    /// shaderDenormFlushToZeroFloat32).
    pub denorm_flush32: bool,
    /// Debug (vertex shaders): also store each vertex's position export in
    /// a `DebugOut` buffer.
    pub debug_pos_store: bool,
    /// Tessellation role and configuration (`TessRole::None` otherwise).
    pub tess: TessSpec,
    pub ps_input_addr: u32,
    pub ps_inputs: Vec<PsInput>,
    pub mrt: [MrtSpec; 8],
    pub cs_local_size: [u32; 3],
    /// COMPUTE_PGM_RSRC2 (TGID enables, TG size, TIDIG count, LDS size).
    pub cs_rsrc2: u32,
    pub resources: Vec<ResSpec>,
    /// Buffer stride (V# word1 [29:16]) per resource index (0 for non-buffers).
    pub buffer_strides: Vec<u32>,
    /// Binding index of the first binding (stages share one descriptor set).
    pub binding_base: u32,
    /// Debug: after the instruction at pc, capture v[N..N+3] and export them
    /// (as floats, alpha 1) instead of the shader's values. Capture k goes to
    /// MRT k (the last capture to any further MRTs). The flag converts the
    /// captured registers from unsigned integers to floats.
    pub debug_capture: Vec<(u32, u32, bool)>,
    /// How buffer stores mark their pages (`mem::mark_dirty`): 1 = kept per
    /// invocation and flushed when a store moves on to another bitmap word
    /// and before the program ends; 0 = one subgroup-coalesced mark per store
    /// (its subgroup operations made such shaders compile ~1.8x slower);
    /// 2 = a plain atomic per lane and store; 3 = a word per page set by a
    /// plain store (the binding's `DirtyInfo` names a word map; the
    /// write-back's copy turns it into the bitmap): no atomic and no branch
    /// of its own, ~1.2x the unmarked shaders' compile time against ~2x
    /// with 1.
    pub page_marks: u8,
}

/// Clip/cull distances of a vertex shader's extra position exports
/// (PA_CL_VS_OUT_CNTL): after pos0 come, in this order and only when
/// enabled, the misc vector (VS_OUT_MISC_VEC_ENA: point size, edge flag,
/// layer, viewport index), distances 0-3 (VS_OUT_CCDIST0_VEC_ENA) and 4-7
/// (VS_OUT_CCDIST1_VEC_ENA). Distance d clips with CLIP_DIST_ENA_d and culls
/// with CULL_DIST_ENA_d; enabled ones get consecutive ClipDistance /
/// CullDistance array elements.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DistLayout {
    /// Export targets (13..=15) carrying distances 0-3 and 4-7.
    pub vec_target: [Option<u32>; 2],
    pub misc_target: Option<u32>,
    /// Per distance: its ClipDistance / CullDistance element.
    pub clip: [Option<u32>; 8],
    pub cull: [Option<u32>; 8],
    pub clip_count: u32,
    pub cull_count: u32,
}

pub fn dist_layout(cntl: u32) -> DistLayout {
    let mut next = 13;
    let mut slot = |on: bool| {
        on.then(|| {
            next += 1;
            next - 1
        })
    };
    let misc_target = slot(cntl & (1 << 21) != 0);
    let vec_target = [slot(cntl & (1 << 22) != 0), slot(cntl & (1 << 23) != 0)];
    let mut l = DistLayout { vec_target, misc_target, ..Default::default() };
    for d in 0..8 {
        if vec_target[d / 4].is_none() {
            continue;
        }
        if cntl & (1 << d) != 0 {
            l.clip[d] = Some(l.clip_count);
            l.clip_count += 1;
        }
        if cntl & (1 << (8 + d)) != 0 {
            l.cull[d] = Some(l.cull_count);
            l.cull_count += 1;
        }
    }
    l
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingKind {
    /// u32 array: user data (16 dwords) followed by resolved scalar loads.
    SgprData,
    Buffer(usize),
    Image(usize),
    StorageImage(usize),
    Sampler(usize),
    /// u32 array: the global data share (GDS_BYTES), shared by all shaders.
    Gds,
    /// u32 array (debug, `Spec::debug_pos_store`): per vertex index 16
    /// dwords, its pos0 export and up to 12 captured registers (the first
    /// register of each `Spec::debug_capture`).
    DebugOut,
    /// u32 array (tessellation stages): the LDS of every threadgroup of the
    /// draw, `TessSpec::lds_dwords` each.
    Lds,
    /// u32 array, 3 dwords per plan resource (written buffers): the dword
    /// index in `DirtyBits` of the bitmap of the pages the shader writes, the
    /// binding's byte offset within its first page, and the highest bitmap
    /// dword (page indices beyond it mark that dword).
    DirtyInfo,
    /// u32 array: page bitmaps (`DirtyInfo`); stores to written buffers set
    /// the bit of every 4 KiB page they touch.
    DirtyBits,
}

/// Vertices `DebugOut` holds (indices are clamped below it) and dwords per vertex.
pub const DEBUG_OUT_VERTICES: u32 = 1 << 18;
pub const DEBUG_OUT_STRIDE: u32 = 16;

/// Size of the global data share.
pub const GDS_BYTES: u32 = 64 << 10;

#[derive(Clone, Debug)]
pub struct Binding {
    pub binding: u32,
    pub kind: BindingKind,
}

pub struct Output {
    pub spirv: Vec<u32>,
    pub bindings: Vec<Binding>,
    /// Number of VS parameters exported (VS) / highest attribute read (PS).
    pub params: u32,
    /// Bit mask of MRTs written (PS).
    pub mrt_mask: u32,
    pub writes_depth: bool,
    /// Bit mask of position exports (VS): bit k = pos k; and of those whose
    /// contents are dropped (the misc vector, or no use per the spec).
    pub pos_exports: u8,
    pub pos_ignored: u8,
    pub warnings: Vec<String>,
}

pub const SGPR_DATA_USER: u32 = 16;

/// Dword index in the SGPR data buffer of the first per-resource byte bias:
/// after the user data and the resolved scalar loads, one dword per plan
/// resource holds the distance (0..15 bytes) from the 16-byte-aligned address
/// the host binds a read-only buffer at to the V# base; shaders add it to
/// their buffer byte offsets (0 for other resources).
pub fn bias_offset(plan: &Plan) -> u32 {
    SGPR_DATA_USER + plan.loads.iter().map(|l| l.dwords).sum::<u32>()
}

/// SGPR data index of the plan's CPU-evaluated values (`Plan::values`),
/// after the biases.
pub fn values_offset(plan: &Plan) -> u32 {
    bias_offset(plan) + plan.resources.len() as u32
}

/// SGPR data index of the per-draw values the renderer appends for LS and
/// HS programs (after the plan's values): LS `[vertex count, vertex id per
/// control point...]`, HS `[patch count]`.
pub fn tess_info_offset(plan: &Plan) -> u32 {
    values_offset(plan) + plan.values.len() as u32
}

/// Offsets (in dwords) of each scalar load in the SGPR data buffer.
pub fn load_offsets(plan: &Plan) -> Vec<u32> {
    let mut v = Vec::with_capacity(plan.loads.len());
    let mut off = SGPR_DATA_USER;
    for l in &plan.loads {
        v.push(off);
        off += l.dwords;
    }
    v
}

pub(crate) struct Types {
    pub void: Word,
    pub bool_: Word,
    pub u32: Word,
    pub i32: Word,
    pub f32: Word,
    pub u64: Word,
    pub v2f: Word,
    pub v3f: Word,
    pub v4f: Word,
    pub v2u: Word,
    pub v3u: Word,
    pub v4u: Word,
    pub v4i: Word,
    pub v2i: Word,
    pub v3i: Word,
    pub fn_u32: Word,
    pub fn_bool: Word,
    pub sb_u32: Word,
    pub in_u32: Word,
    pub in_f32: Word,
    pub in_v4f: Word,
    pub in_v3u: Word,
    pub in_bool: Word,
    pub out_v4f: Word,
    pub out_f32: Word,
}

pub(crate) struct ImageRes {
    pub var: Word,
    pub img_ty: Word,
    pub dim: ImageDim,
    pub class: NumClass,
    pub storage: bool,
}

pub(crate) struct T<'a> {
    pub b: Builder,
    pub t: Types,
    pub glsl: Word,
    pub spec: &'a Spec,
    pub plan: &'a Plan,
    /// One Function variable per scalar / vector register (no register-file
    /// arrays: access chains into function-local arrays inside the dispatcher
    /// loop kept drivers from promoting registers to SSA values).
    pub sgpr: Vec<Word>,
    pub vgpr: Vec<Word>,
    pub scc: Word,
    pub consts: HashMap<(Word, u32), Word>,
    pub interface: Vec<Word>,
    pub sgpr_data: Word,
    pub load_offsets: Vec<u32>,
    pub buffers: HashMap<usize, Word>,
    pub images: HashMap<usize, ImageRes>,
    pub samplers: HashMap<usize, Word>,
    pub sampler_ty: Word,
    pub bindings: Vec<Binding>,
    pub next_binding: u32,
    // stage interface
    pub out_pos: Option<Word>,
    pub out_params: HashMap<u32, Word>,
    pub out_mrt: HashMap<u32, (Word, Word)>,
    pub out_depth: Option<Word>,
    pub in_attrs: HashMap<u32, Word>,
    pub frag_coord: Option<Word>,
    pub lds: Option<Word>,
    /// Compute workgroup LDS: dwords, and the invocation's index in the
    /// workgroup (for zeroing it at entry, `zero_lds`).
    pub lds_len: u32,
    pub local_index: Option<Word>,
    /// Per-invocation LDS stand-in of graphics stages (`mem::lane_lds_index`).
    pub lane_lds: Option<Word>,
    /// Tessellation stages: `lds` is the `Lds` storage buffer and this is
    /// the dword index of this invocation's threadgroup region in it.
    pub lds_base: Option<Word>,
    /// GDS storage buffer (when the program uses GDS instructions).
    pub gds: Option<Word>,
    /// Page tracking of buffer writes: (`DirtyInfo`, `DirtyBits`) variables.
    pub dirty: Option<(Word, Word)>,
    /// Per written buffer: its DirtyInfo (bitmap base, head, limit).
    pub dirty_res: HashMap<usize, (Word, Word, Word)>,
    /// Per written buffer (`Spec::page_marks` 1): function variables holding
    /// this invocation's pending bitmap word and its bits (`mem::mark_dirty`,
    /// `mem::flush_dirty`).
    pub dirty_pend: HashMap<usize, (Word, Word)>,
    pub lane_id: Option<Word>,
    /// SubgroupLocalInvocationId input (created on first use).
    pub sg_lane_var: Option<Word>,
    /// (vgpr, lane) slots written by v_writelane with constant lanes (SGPR
    /// spills): per-invocation copies of those lanes' values.
    pub lane_slots: HashMap<(u32, u32), Word>,
    /// `active()` computed in the current block (block index, value):
    /// reused until EXEC is written or another block is selected.
    active_cache: Option<(usize, Word)>,
    /// Per buffer resource: byte bias of its binding (see `bias_offset`).
    pub buf_bias: HashMap<usize, Word>,
    pub warnings: Vec<String>,
    pub params: u32,
    pub mrt_mask: u32,
    pub pos_exports: u8,
    pub pos_ignored: u8,
    /// Clip/cull distance layout (vertex shaders) and the ClipDistance /
    /// CullDistance outputs (declared when an export carries enabled ones).
    pub dist: DistLayout,
    pub out_clip: Option<Word>,
    pub out_cull: Option<Word>,
    /// `Spec::debug_pos_store`: the DebugOut buffer and this vertex's index.
    pub dbg_out: Option<(Word, Word)>,
    pub pc_var: Word,
    pub exit_block: Option<Word>,
    pub kill_used: bool,
    /// Whether any instruction of the program may write EXEC (`may_write_exec`):
    /// without one every lane stays active, and valid-mask exports kill none.
    pub exec_written: bool,
    /// Workgroup LDS accesses since the last barrier (bit 0 reads, bit 1
    /// writes) and the nesting depth of structured constructs being emitted:
    /// GCN lanes of a wave run in lockstep, so a wave-sized workgroup may
    /// write LDS and read other lanes' entries without s_barrier; here lanes
    /// are invocations, so a barrier goes where accesses switch between
    /// reads and writes -- only at depth 0, in uniform control flow (also
    /// inside constructs when `wave_sync` makes all control flow uniform).
    pub lds_phase: u8,
    pub cf_depth: u32,
    /// LDS hazards found inside control flow (left without a barrier).
    pub lds_hazards_nested: u32,
    /// Wave-synchronous translation (`wave`): VCCZ/EXECZ branches vote
    /// across the workgroup, LDS barriers also inside control flow.
    pub wave_sync: bool,
    /// The vote's workgroup slots (three u32) and this invocation's index of
    /// the next slot (`wave_any`).
    pub vote: Option<(Word, Word)>,
    pub sb_struct: Option<Word>,
    /// Function variables holding the debug capture (see Spec::debug_capture).
    pub dbg_vars: Vec<Word>,
}

pub type R<T> = Result<T, String>;

pub(crate) fn e<E: std::fmt::Debug>(r: Result<Word, E>) -> R<Word> {
    r.map_err(|x| format!("spirv builder: {x:?}"))
}

impl<'a> T<'a> {
    pub fn u(&mut self, v: u32) -> Word {
        let ty = self.t.u32;
        *self.consts.entry((ty, v)).or_insert_with(|| self.b.constant_bit32(ty, v))
    }
    pub fn i(&mut self, v: i32) -> Word {
        let ty = self.t.i32;
        *self.consts.entry((ty, v as u32)).or_insert_with(|| self.b.constant_bit32(ty, v as u32))
    }
    pub fn f(&mut self, v: f32) -> Word {
        let ty = self.t.f32;
        *self.consts.entry((ty, v.to_bits())).or_insert_with(|| self.b.constant_bit32(ty, v.to_bits()))
    }
    pub fn bool_c(&mut self, v: bool) -> Word {
        let ty = self.t.bool_;
        *self.consts.entry((ty, v as u32)).or_insert_with(|| if v { self.b.constant_true(ty) } else { self.b.constant_false(ty) })
    }

    /// Raw 32-bit value of a scalar register slot (0..127).
    pub fn sreg(&mut self, idx: u32) -> R<Word> {
        let p = self.sgpr[(idx & 127) as usize];
        e(self.b.load(self.t.u32, None, p, None, []))
    }
    pub fn set_sreg(&mut self, idx: u32, v: Word) -> R<()> {
        if idx >= 128 {
            return Ok(()); // writes to constants / null are ignored
        }
        if idx == crate::OP_EXEC_LO as u32 || idx == crate::OP_EXEC_LO as u32 + 1 {
            self.active_cache = None;
        }
        let p = self.sgpr[idx as usize];
        self.b.store(p, v, None, []).map_err(|x| format!("{x:?}"))
    }
    pub fn vreg(&mut self, idx: u32) -> R<Word> {
        let p = self.vgpr[(idx & 255) as usize];
        e(self.b.load(self.t.u32, None, p, None, []))
    }
    /// Unconditional VGPR write (used for initialization).
    pub fn set_vreg_raw(&mut self, idx: u32, v: Word) -> R<()> {
        let p = self.vgpr[(idx & 255) as usize];
        self.b.store(p, v, None, []).map_err(|x| format!("{x:?}"))
    }
    /// VGPR write predicated on this lane being active.
    pub fn set_vreg(&mut self, idx: u32, v: Word) -> R<()> {
        let act = self.active()?;
        let old = self.vreg(idx)?;
        let nv = e(self.b.select(self.t.u32, None, act, v, old))?;
        self.set_vreg_raw(idx, nv)
    }
    /// Whether this lane is active (EXEC bit 0). The value is reused within
    /// a block until EXEC is written: most instructions write a VGPR under it
    /// (thousands of recomputations made the largest shaders take ~35 s to
    /// compile).
    pub fn active(&mut self) -> R<Word> {
        let blk = self.b.selected_block();
        if let (Some((cb, w)), Some(b)) = (self.active_cache, blk) {
            if cb == b {
                return Ok(w);
            }
        }
        let ex = self.sreg(crate::OP_EXEC_LO as u32)?;
        let one = self.u(1);
        let zero = self.u(0);
        let b = e(self.b.bitwise_and(self.t.u32, None, ex, one))?;
        let w = e(self.b.i_not_equal(self.t.bool_, None, b, zero))?;
        self.active_cache = blk.map(|b| (b, w));
        Ok(w)
    }
    /// OpControlBarrier at workgroup scope for the workgroup's LDS (compute).
    pub fn workgroup_barrier(&mut self) -> R<()> {
        let wg = self.u(spirv::Scope::Workgroup as u32);
        let mem = if self.lds_base.is_some() { spirv::MemorySemantics::UNIFORM_MEMORY } else { spirv::MemorySemantics::WORKGROUP_MEMORY };
        let sem = self.u((spirv::MemorySemantics::ACQUIRE_RELEASE | mem).bits());
        self.lds_phase = 0;
        self.b.control_barrier(wg, wg, sem).map_err(|x| format!("{x:?}"))
    }
    /// Before an LDS access (`read`/`write`) in a compute shader's workgroup
    /// LDS: a barrier when it would see a lockstep hazard (see `lds_phase`).
    pub fn lds_access(&mut self, read: bool, write: bool) -> R<()> {
        if self.spec.stage != Stage::Compute || self.lds.is_none() || self.lds_base.is_some() {
            return Ok(());
        }
        let hazard = (read && self.lds_phase & 2 != 0) || (write && self.lds_phase & 1 != 0);
        if hazard {
            if self.cf_depth == 0 || self.wave_sync {
                self.workgroup_barrier()?;
            } else {
                self.lds_hazards_nested += 1;
            }
        }
        self.lds_phase |= read as u8 | ((write as u8) << 1);
        Ok(())
    }
    /// Workgroup vote (`wave_sync`): whether `bit` holds in any invocation,
    /// the same answer in all of them (a VCCZ/EXECZ test of the whole wave).
    /// Three slots rotate: vote k ORs into slot k % 3, reads it after the
    /// barrier and clears slot (k + 2) % 3 - whose last reads (vote k - 1)
    /// precede this barrier and whose next writes (vote k + 2) follow the
    /// next one.
    pub fn wave_any(&mut self, bit: Word) -> R<Word> {
        let (slots, idx) = self.vote.ok_or("wave vote without slots")?;
        let (u, bt) = (self.t.u32, self.t.bool_);
        let (zero, one, two) = (self.u(0), self.u(1), self.u(2));
        let ef = |x: rspirv::dr::Error| format!("{x:?}");
        let k = e(self.b.load(u, None, idx, None, []))?;
        let last = e(self.b.i_equal(bt, None, k, two))?;
        let k1 = e(self.b.i_add(u, None, k, one))?;
        let next = e(self.b.select(u, None, last, zero, k1))?;
        self.b.store(idx, next, None, []).map_err(ef)?;
        let p_wg = self.b.type_pointer(None, StorageClass::Workgroup, u);
        let scope = self.u(spirv::Scope::Workgroup as u32);
        let relaxed = self.u(0);
        let p = e(self.b.access_chain(p_wg, None, slots, [k]))?;
        let v = e(self.b.select(u, None, bit, one, zero))?;
        e(self.b.atomic_or(u, None, p, scope, relaxed, v))?;
        self.workgroup_barrier()?;
        let r = e(self.b.atomic_load(u, None, p, scope, relaxed))?;
        let first = e(self.b.i_equal(bt, None, k, zero))?;
        let km1 = e(self.b.i_sub(u, None, k, one))?;
        let clear = e(self.b.select(u, None, first, two, km1))?;
        let pc = e(self.b.access_chain(p_wg, None, slots, [clear]))?;
        self.b.atomic_store(pc, scope, relaxed, zero).map_err(ef)?;
        e(self.b.i_not_equal(bt, None, r, zero))
    }
    pub fn get_scc(&mut self) -> R<Word> {
        e(self.b.load(self.t.bool_, None, self.scc, None, []))
    }
    pub fn set_scc(&mut self, v: Word) -> R<()> {
        self.b.store(self.scc, v, None, []).map_err(|x| format!("{x:?}"))
    }

    /// 32-bit value of an operand (SGPR, VGPR, inline constant, literal, special).
    pub fn operand(&mut self, ins: &Inst, op: u16) -> R<Word> {
        Ok(match op {
            0..=127 => self.sreg(op as u32)?,
            128..=192 => self.u((op - 128) as u32),
            193..=208 => self.u((-((op - 192) as i32)) as u32),
            240 => self.u(0.5f32.to_bits()),
            241 => self.u((-0.5f32).to_bits()),
            242 => self.u(1.0f32.to_bits()),
            243 => self.u((-1.0f32).to_bits()),
            244 => self.u(2.0f32.to_bits()),
            245 => self.u((-2.0f32).to_bits()),
            246 => self.u(4.0f32.to_bits()),
            247 => self.u((-4.0f32).to_bits()),
            crate::OP_VCCZ => {
                let lo = self.sreg(crate::OP_VCC_LO as u32)?;
                let hi = self.sreg(crate::OP_VCC_HI as u32)?;
                let o = e(self.b.bitwise_or(self.t.u32, None, lo, hi))?;
                let z = self.u(0);
                let c = e(self.b.i_equal(self.t.bool_, None, o, z))?;
                let one = self.u(1);
                e(self.b.select(self.t.u32, None, c, one, z))?
            }
            crate::OP_EXECZ => {
                let lo = self.sreg(crate::OP_EXEC_LO as u32)?;
                let hi = self.sreg(crate::OP_EXEC_HI as u32)?;
                let o = e(self.b.bitwise_or(self.t.u32, None, lo, hi))?;
                let z = self.u(0);
                let c = e(self.b.i_equal(self.t.bool_, None, o, z))?;
                let one = self.u(1);
                e(self.b.select(self.t.u32, None, c, one, z))?
            }
            crate::OP_SCC => {
                let s = self.get_scc()?;
                let one = self.u(1);
                let z = self.u(0);
                e(self.b.select(self.t.u32, None, s, one, z))?
            }
            crate::OP_LITERAL => self.u(ins.lit.unwrap_or(0)),
            256..=511 => self.vreg((op - 256) as u32)?,
            _ => {
                self.warn(format!("{:#x}: unsupported operand {op}", ins.pc));
                self.u(0)
            }
        })
    }

    pub fn to_f(&mut self, v: Word) -> R<Word> {
        e(self.b.bitcast(self.t.f32, None, v))
    }
    pub fn to_u(&mut self, v: Word) -> R<Word> {
        e(self.b.bitcast(self.t.u32, None, v))
    }
    pub fn to_i(&mut self, v: Word) -> R<Word> {
        e(self.b.bitcast(self.t.i32, None, v))
    }
    pub fn glsl1(&mut self, ty: Word, op: spirv::GlslStd450Op, a: Word) -> R<Word> {
        e(self.b.ext_inst(ty, None, self.glsl, op as u32, [Operand::IdRef(a)]))
    }
    pub fn glsl2(&mut self, ty: Word, op: spirv::GlslStd450Op, a: Word, c: Word) -> R<Word> {
        e(self.b.ext_inst(ty, None, self.glsl, op as u32, [Operand::IdRef(a), Operand::IdRef(c)]))
    }
    pub fn glsl3(&mut self, ty: Word, op: spirv::GlslStd450Op, a: Word, c: Word, d: Word) -> R<Word> {
        e(self.b.ext_inst(ty, None, self.glsl, op as u32, [Operand::IdRef(a), Operand::IdRef(c), Operand::IdRef(d)]))
    }
    pub fn bool_to_u(&mut self, c: Word) -> R<Word> {
        let one = self.u(1);
        let z = self.u(0);
        e(self.b.select(self.t.u32, None, c, one, z))
    }
    pub fn warn(&mut self, s: String) {
        if !self.warnings.contains(&s) {
            self.warnings.push(s);
        }
    }

    /// Reads dword `idx` of the SGPR data buffer.
    pub fn sgpr_data(&mut self, idx: u32) -> R<Word> {
        let z = self.u(0);
        let i = self.u(idx);
        let p = e(self.b.access_chain(self.t.sb_u32, None, self.sgpr_data, [z, i]))?;
        e(self.b.load(self.t.u32, None, p, None, []))
    }

    /// Creates a module-scope OpVariable (never inside the current block).
    /// The per-invocation LDS stand-in of graphics stages (created on first use).
    pub fn lane_lds_var(&mut self) -> Word {
        if let Some(v) = self.lane_lds {
            return v;
        }
        let len = self.u(mem::LANE_LDS_DWORDS);
        let arr = self.b.type_array(self.t.u32, len);
        let pt = self.b.type_pointer(None, StorageClass::Private, arr);
        let v = self.global_var(pt, StorageClass::Private);
        self.lane_lds = Some(v);
        v
    }

    pub fn global_var(&mut self, ptr_ty: Word, sc: StorageClass) -> Word {
        let id = self.b.id();
        let inst = rspirv::dr::Instruction::new(spirv::Op::Variable, Some(ptr_ty), Some(id), vec![Operand::StorageClass(sc)]);
        self.b.module_mut().types_global_values.push(inst);
        id
    }

    fn alloc_binding(&mut self, kind: BindingKind) -> u32 {
        let n = self.spec.binding_base + self.next_binding;
        self.next_binding += 1;
        self.bindings.push(Binding { binding: n, kind });
        n
    }

    fn decorate_binding(&mut self, var: Word, binding: u32) {
        self.b.decorate(var, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
        self.b.decorate(var, Decoration::Binding, [Operand::LiteralBit32(binding)]);
    }

    /// Storage buffer `struct { uint data[]; }` variable.
    fn sb_var(&mut self, _readonly: bool) -> Word {
        // rspirv deduplicates identical types: create and decorate the block type once.
        let st = match self.sb_struct {
            Some(s) => s,
            None => {
                let rta = self.b.type_runtime_array(self.t.u32);
                self.b.decorate(rta, Decoration::ArrayStride, [Operand::LiteralBit32(4)]);
                let st = self.b.type_struct([rta]);
                self.b.decorate(st, Decoration::Block, []);
                self.b.member_decorate(st, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
                self.sb_struct = Some(st);
                st
            }
        };
        let pt = self.b.type_pointer(None, StorageClass::StorageBuffer, st);
        self.global_var(pt, StorageClass::StorageBuffer)
    }

    /// This invocation's index in its subgroup (for cross-lane operations).
    pub fn subgroup_lane(&mut self) -> R<Word> {
        self.b.capability(Capability::GroupNonUniform);
        let var = match self.sg_lane_var {
            Some(v) => v,
            None => {
                let v = self.input(self.t.u32, Some(BuiltIn::SubgroupLocalInvocationId), None);
                if self.spec.stage == Stage::Fragment {
                    self.b.decorate(v, Decoration::Flat, []);
                }
                self.sg_lane_var = Some(v);
                v
            }
        };
        e(self.b.load(self.t.u32, None, var, None, []))
    }

    fn input(&mut self, ty: Word, builtin: Option<BuiltIn>, location: Option<u32>) -> Word {
        let pt = self.b.type_pointer(None, StorageClass::Input, ty);
        let v = self.global_var(pt, StorageClass::Input);
        if let Some(bi) = builtin {
            self.b.decorate(v, Decoration::BuiltIn, [Operand::BuiltIn(bi)]);
        }
        if let Some(l) = location {
            self.b.decorate(v, Decoration::Location, [Operand::LiteralBit32(l)]);
        }
        self.interface.push(v);
        v
    }

    /// A float[n] output decorated `bi`, every element set to 1.0 (inside):
    /// a distance export that is not reached must not clip.
    fn dist_array(&mut self, n: u32, bi: BuiltIn) -> R<Word> {
        let len = self.u(n);
        let arr = self.b.type_array(self.t.f32, len);
        let v = self.output(arr, Some(bi), None);
        let one = self.f(1.0);
        for i in 0..n {
            let idx = self.u(i);
            let p = e(self.b.access_chain(self.t.out_f32, None, v, [idx]))?;
            self.b.store(p, one, None, []).map_err(|x| format!("{x:?}"))?;
        }
        Ok(v)
    }

    fn output(&mut self, ty: Word, builtin: Option<BuiltIn>, location: Option<u32>) -> Word {
        let pt = self.b.type_pointer(None, StorageClass::Output, ty);
        let v = self.global_var(pt, StorageClass::Output);
        if let Some(bi) = builtin {
            self.b.decorate(v, Decoration::BuiltIn, [Operand::BuiltIn(bi)]);
        }
        if let Some(l) = location {
            self.b.decorate(v, Decoration::Location, [Operand::LiteralBit32(l)]);
        }
        self.interface.push(v);
        v
    }
}

/// Inlines a fetch subroutine at the `s_swappc_b64` of a vertex shader.
pub fn inline_fetch(main: &[Inst], fetch: &[Inst]) -> Vec<Inst> {
    let mut out = Vec::with_capacity(main.len() + fetch.len());
    for ins in main {
        if ins.enc == Enc::Sop1 && ins.op == 33 {
            for f in fetch {
                if f.enc == Enc::Sop1 && f.op == 32 {
                    continue; // s_setpc_b64 return
                }
                let mut g = *f;
                g.pc = f.pc + 0x10_0000; // separate pc space for fetch code
                out.push(g);
            }
        } else {
            out.push(*ins);
        }
    }
    out
}

/// Splits the program into basic blocks: returns block start pcs.
fn block_starts(code: &[Inst]) -> BTreeSet<u32> {
    let mut s = BTreeSet::new();
    s.insert(code[0].pc);
    for (i, ins) in code.iter().enumerate() {
        if let Some(t) = ins.branch_target() {
            s.insert(t);
            if let Some(n) = code.get(i + 1) {
                s.insert(n.pc);
            }
        }
        if ins.is_endpgm() {
            if let Some(n) = code.get(i + 1) {
                s.insert(n.pc);
            }
        }
    }
    s
}

/// The instructions (indices into `code`) of each block in `starts` order: a
/// block runs in program order from its start to the next start, so an
/// inlined fetch subroutine, which keeps its own pc space (`inline_fetch`),
/// belongs to the block holding its call site.
fn block_ranges(code: &[Inst], starts: &BTreeSet<u32>) -> Vec<std::ops::Range<usize>> {
    let index: HashMap<u32, usize> = code.iter().enumerate().map(|(k, i)| (i.pc, k)).collect();
    starts
        .iter()
        .map(|s| {
            let first = index.get(s).copied().unwrap_or(code.len());
            let end = (first + 1..code.len()).find(|&k| starts.contains(&code[k].pc)).unwrap_or(code.len());
            first..end.max(first)
        })
        .collect()
}

/// Whether `ins` may write EXEC (conservatively: any scalar destination
/// field naming exec_lo/exec_hi counts, also where the encoding has none).
pub fn may_write_exec(ins: &Inst) -> bool {
    let exec_dst = |r: u16| r == crate::OP_EXEC_LO || r == crate::OP_EXEC_LO + 1;
    match ins.enc {
        // (SOP1 36..=43: s_*_saveexec_b64.)
        Enc::Sop1 => (36..=43).contains(&ins.op) || exec_dst(ins.sdst()),
        Enc::Sop2 | Enc::Sopk | Enc::Smrd => exec_dst(ins.sdst()),
        // v_cmpx_* (odd high nibble).
        Enc::Vopc => (ins.op >> 4) & 1 == 1,
        // VOPC in VOP3 form, VOP3b carry-outs, v_readlane into exec.
        Enc::Vop3 => (ins.op < 0x100 && (ins.op >> 4) & 1 == 1) || exec_dst(ins.sdst()) || exec_dst(ins.vdst()),
        // v_readfirstlane / v_readlane name an SGPR destination.
        Enc::Vop1 | Enc::Vop2 => exec_dst(ins.vdst()),
        _ => false,
    }
}

pub fn translate(code: &[Inst], plan: &Plan, spec: &Spec) -> R<Output> {
    let (out, nested) = translate_mode(code, plan, spec, false)?;
    // A one-wave workgroup whose lanes share LDS inside control flow without
    // s_barrier: again wave-synchronously (`wave`), when that is safe.
    let [x, y, z] = spec.cs_local_size;
    let one_wave = x.max(1) * y.max(1) * z.max(1) <= 64;
    if nested == 0 || spec.stage != Stage::Compute || spec.tess.role != TessRole::None || !one_wave {
        return Ok(out);
    }
    if !wave::scc_branches_uniform(code) {
        let mut out = out;
        out.warnings.push("not wave-synchronous: an SCC branch tests a per-lane value".into());
        return Ok(out);
    }
    match translate_mode(code, plan, spec, true) {
        Ok((mut w, _)) => {
            w.warnings.push(format!("wave-synchronous: {nested} LDS read/write switch(es) inside control flow get barriers, VCCZ/EXECZ branches a workgroup vote"));
            Ok(w)
        }
        Err(why) => {
            let mut out = out;
            out.warnings.push(format!("not wave-synchronous: {why}"));
            Ok(out)
        }
    }
}

/// The translation; with `wave_sync` (`wave`) branches on VCCZ/EXECZ vote
/// across the workgroup and LDS barriers also go inside control flow. Also
/// returns the LDS hazards left without a barrier inside control flow.
fn translate_mode(code: &[Inst], plan: &Plan, spec: &Spec, wave_sync: bool) -> R<(Output, u32)> {
    if code.is_empty() {
        return Err("empty program".into());
    }
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(Capability::Shader);
    // TLG_SPIRV_NONCE=<text> (measurements): every module differs from the
    // driver's cached ones, so pipelines compile afresh.
    if let Ok(n) = std::env::var("TLG_SPIRV_NONCE") {
        b.source_extension(format!("tlg-nonce-{n}"));
    }
    b.capability(Capability::ImageQuery);
    b.capability(Capability::StorageImageWriteWithoutFormat);
    b.capability(Capability::StorageImageReadWithoutFormat);
    b.capability(Capability::Int64);
    b.capability(Capability::Sampled1D);
    b.capability(Capability::Image1D);
    b.capability(Capability::SampledCubeArray);
    b.capability(Capability::ImageCubeArray);
    // Texel offsets are register values (not constants) in GCN sample instructions.
    b.capability(Capability::ImageGatherExtended);
    b.extension("SPV_KHR_storage_buffer_storage_class");
    let glsl = b.ext_inst_import("GLSL.std.450");
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let bool_ = b.type_bool();
    let u32t = b.type_int(32, 0);
    let i32t = b.type_int(32, 1);
    let f32t = b.type_float(32, None);
    let u64t = b.type_int(64, 0);
    let v2f = b.type_vector(f32t, 2);
    let v3f = b.type_vector(f32t, 3);
    let v4f = b.type_vector(f32t, 4);
    let v2u = b.type_vector(u32t, 2);
    let v3u = b.type_vector(u32t, 3);
    let v4u = b.type_vector(u32t, 4);
    let v2i = b.type_vector(i32t, 2);
    let v3i = b.type_vector(i32t, 3);
    let v4i = b.type_vector(i32t, 4);
    let t = Types {
        void,
        bool_,
        u32: u32t,
        i32: i32t,
        f32: f32t,
        u64: u64t,
        v2f,
        v3f,
        v4f,
        v2u,
        v3u,
        v4u,
        v2i,
        v3i,
        v4i,
        fn_u32: b.type_pointer(None, StorageClass::Function, u32t),
        fn_bool: b.type_pointer(None, StorageClass::Function, bool_),
        sb_u32: b.type_pointer(None, StorageClass::StorageBuffer, u32t),
        in_u32: b.type_pointer(None, StorageClass::Input, u32t),
        in_f32: b.type_pointer(None, StorageClass::Input, f32t),
        in_v4f: b.type_pointer(None, StorageClass::Input, v4f),
        in_v3u: b.type_pointer(None, StorageClass::Input, v3u),
        in_bool: b.type_pointer(None, StorageClass::Input, bool_),
        out_v4f: b.type_pointer(None, StorageClass::Output, v4f),
        out_f32: b.type_pointer(None, StorageClass::Output, f32t),
    };
    let sampler_ty = b.type_sampler();
    let mut tr = T {
        b,
        t,
        glsl,
        spec,
        plan,
        sgpr: Vec::new(),
        vgpr: Vec::new(),
        scc: 0,
        consts: HashMap::default(),
        interface: Vec::new(),
        sgpr_data: 0,
        load_offsets: load_offsets(plan),
        buffers: HashMap::default(),
        images: HashMap::default(),
        samplers: HashMap::default(),
        sampler_ty,
        bindings: Vec::new(),
        next_binding: 0,
        out_pos: None,
        out_params: HashMap::default(),
        out_mrt: HashMap::default(),
        out_depth: None,
        in_attrs: HashMap::default(),
        frag_coord: None,
        lds: None,
        lds_len: 0,
        local_index: None,
        lane_lds: None,
        lds_base: None,
        gds: None,
        dirty: None,
        dirty_res: HashMap::default(),
        dirty_pend: HashMap::default(),
        lane_id: None,
        sg_lane_var: None,
        lane_slots: HashMap::default(),
        active_cache: None,
        buf_bias: HashMap::default(),
        warnings: Vec::new(),
        params: 0,
        mrt_mask: 0,
        pos_exports: 0,
        pos_ignored: 0,
        dist: if spec.stage == Stage::Vertex { dist_layout(spec.vs_out_cntl) } else { DistLayout::default() },
        out_clip: None,
        out_cull: None,
        dbg_out: None,
        pc_var: 0,
        exit_block: None,
        kill_used: false,
        exec_written: code.iter().any(may_write_exec),
        lds_phase: 0,
        cf_depth: 0,
        lds_hazards_nested: 0,
        wave_sync,
        vote: None,
        sb_struct: None,
        dbg_vars: Vec::new(),
    };
    // SGPR data buffer (binding base + 0).
    let sd = tr.sb_var(true);
    let bnd = tr.alloc_binding(BindingKind::SgprData);
    tr.decorate_binding(sd, bnd);
    tr.sgpr_data = sd;
    if code.iter().any(|i| i.enc == Enc::Ds && i.ds_gds()) {
        let v = tr.sb_var(false);
        let bnd = tr.alloc_binding(BindingKind::Gds);
        tr.decorate_binding(v, bnd);
        tr.gds = Some(v);
    }
    // Resource bindings in plan order.
    for (i, r) in plan.resources.iter().enumerate() {
        match (r.kind, spec.resources.get(i)) {
            (ResKind::Buffer, _) => {
                let v = tr.sb_var(!r.written);
                let bnd = tr.alloc_binding(BindingKind::Buffer(i));
                tr.decorate_binding(v, bnd);
                tr.buffers.insert(i, v);
            }
            (ResKind::Image, Some(ResSpec::Image { dim, class })) => {
                let (dim, class) = (*dim, *class);
                let sampled_ty = match class {
                    NumClass::Float => tr.t.f32,
                    NumClass::Sint => tr.t.i32,
                    NumClass::Uint => tr.t.u32,
                };
                // 1D images are 2D images of height 1 on the host (so they can
                // alias other images); coordinates get a zero y component.
                let (sd, arrayed) = match dim {
                    ImageDim::D1 => (spirv::Dim::Dim2D, 0),
                    ImageDim::D2 => (spirv::Dim::Dim2D, 0),
                    ImageDim::D3 => (spirv::Dim::Dim3D, 0),
                    ImageDim::Cube => (spirv::Dim::DimCube, 0),
                    ImageDim::D1Array => (spirv::Dim::Dim2D, 1),
                    ImageDim::D2Array => (spirv::Dim::Dim2D, 1),
                    ImageDim::CubeArray => (spirv::Dim::DimCube, 1),
                };
                let storage = r.written;
                let img_ty = tr.b.type_image(sampled_ty, sd, 0, arrayed, 0, if storage { 2 } else { 1 }, spirv::ImageFormat::Unknown, None);
                let pt = tr.b.type_pointer(None, StorageClass::UniformConstant, img_ty);
                let v = tr.global_var(pt, StorageClass::UniformConstant);
                let bnd = tr.alloc_binding(if storage { BindingKind::StorageImage(i) } else { BindingKind::Image(i) });
                tr.decorate_binding(v, bnd);
                tr.images.insert(i, ImageRes { var: v, img_ty, dim, class, storage });
            }
            (ResKind::Sampler, _) => {
                let pt = tr.b.type_pointer(None, StorageClass::UniformConstant, tr.sampler_ty);
                let v = tr.global_var(pt, StorageClass::UniformConstant);
                let bnd = tr.alloc_binding(BindingKind::Sampler(i));
                tr.decorate_binding(v, bnd);
                tr.samplers.insert(i, v);
            }
            (ResKind::Image, _) => return Err(format!("image resource {i} has no specialization")),
        }
    }
    // Page tracking of buffer writes (see `BindingKind::DirtyInfo`).
    if plan.resources.iter().any(|r| r.kind == ResKind::Buffer && r.written) {
        let info = tr.sb_var(true);
        let bnd = tr.alloc_binding(BindingKind::DirtyInfo);
        tr.decorate_binding(info, bnd);
        let bits = tr.sb_var(false);
        let bnd = tr.alloc_binding(BindingKind::DirtyBits);
        tr.decorate_binding(bits, bnd);
        tr.dirty = Some((info, bits));
    }
    // Function.
    let fn_ty = tr.b.type_function(void, []);
    let func = e(tr.b.begin_function(void, None, spirv::FunctionControl::NONE, fn_ty))?;
    e(tr.b.begin_block(None))?;
    for _ in 0..128 {
        let v = tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, None);
        tr.sgpr.push(v);
    }
    for _ in 0..256 {
        let v = tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, None);
        tr.vgpr.push(v);
    }
    tr.scc = tr.b.variable(tr.t.fn_bool, None, StorageClass::Function, None);
    tr.pc_var = tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, None);
    let lds_i = tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, None);
    if wave_sync {
        let idx = tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, None);
        let three = tr.u(3);
        let arr = tr.b.type_array(tr.t.u32, three);
        let pt = tr.b.type_pointer(None, StorageClass::Workgroup, arr);
        let slots = tr.global_var(pt, StorageClass::Workgroup);
        tr.vote = Some((slots, idx));
    }
    for ins in code {
        if let Some((true, vgpr, lane)) = ins.const_lane_op() {
            if !tr.lane_slots.contains_key(&(vgpr as u32, lane)) {
                let v = tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, None);
                tr.lane_slots.insert((vgpr as u32, lane), v);
            }
        }
    }
    for _ in 0..spec.debug_capture.len() * 3 {
        let v = tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, None);
        tr.dbg_vars.push(v);
    }
    // Page marks of written buffers pending in this invocation (word, bits).
    if tr.dirty.is_some() && spec.page_marks == 1 && PROBE_MARKS.with(|s| s.get()) == 0 {
        for (i, r) in plan.resources.iter().enumerate() {
            if r.kind == ResKind::Buffer && r.written {
                let w = tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, None);
                let b = tr.b.variable(tr.t.fn_u32, None, StorageClass::Function, None);
                tr.dirty_pend.insert(i, (w, b));
            }
        }
    }
    // Initial state.
    let f = tr.bool_c(false);
    tr.set_scc(f)?;
    let one = tr.u(1);
    let zero = tr.u(0);
    let pend: Vec<(Word, Word)> = tr.dirty_pend.values().copied().collect();
    if !pend.is_empty() {
        let none = tr.u(u32::MAX);
        for (w, b) in pend {
            tr.b.store(w, none, None, []).map_err(|x| format!("{x:?}"))?;
            tr.b.store(b, zero, None, []).map_err(|x| format!("{x:?}"))?;
        }
    }
    tr.set_sreg(crate::OP_EXEC_LO as u32, one)?;
    tr.set_sreg(crate::OP_EXEC_HI as u32, zero)?;
    tr.set_sreg(crate::OP_VCC_LO as u32, zero)?;
    tr.set_sreg(crate::OP_VCC_HI as u32, zero)?;
    tr.set_sreg(crate::OP_M0 as u32, zero)?;
    for i in 0..spec.user_sgprs.min(16) as u32 {
        let v = tr.sgpr_data(i)?;
        tr.set_sreg(i, v)?;
    }
    mem::stage_inputs(&mut tr)?;
    zero_lds(&mut tr, lds_i)?;
    init_votes(&mut tr)?;
    if spec.debug_pos_store && spec.stage == Stage::Vertex && spec.tess.role == TessRole::None {
        let v = tr.sb_var(false);
        let bnd = tr.alloc_binding(BindingKind::DebugOut);
        tr.decorate_binding(v, bnd);
        // v0 holds the vertex index at entry.
        let vid = tr.vreg(0)?;
        let max = tr.u(DEBUG_OUT_VERTICES - 1);
        let glsl = tr.glsl;
        let idx = e(tr.b.ext_inst(tr.t.u32, None, glsl, spirv::GlslStd450Op::UMin as u32, [Operand::IdRef(vid), Operand::IdRef(max)]))?;
        tr.dbg_out = Some((v, idx));
    }
    if spec.stage == Stage::Vertex {
        let exported = code.iter().filter(|i| i.enc == Enc::Exp).fold(0u64, |m, i| m | 1 << i.exp_target().min(63));
        if tr.dist.vec_target.iter().flatten().any(|&t| exported & (1 << t) != 0) {
            if tr.dist.clip_count > 0 {
                tr.b.capability(Capability::ClipDistance);
                let n = tr.dist.clip_count;
                tr.out_clip = Some(tr.dist_array(n, BuiltIn::ClipDistance)?);
            }
            if tr.dist.cull_count > 0 {
                tr.b.capability(Capability::CullDistance);
                let n = tr.dist.cull_count;
                tr.out_cull = Some(tr.dist_array(n, BuiltIn::CullDistance)?);
            }
        }
    }
    // Page bitmaps of written buffers (`DirtyInfo`), loaded once in the entry block.
    if let Some((info, _)) = tr.dirty {
        let written: Vec<usize> = plan.resources.iter().enumerate().filter(|(_, r)| r.kind == ResKind::Buffer && r.written).map(|(i, _)| i).collect();
        for r in written {
            let mut v = [0; 3];
            for (k, slot) in v.iter_mut().enumerate() {
                let z = tr.u(0);
                let i = tr.u(3 * r as u32 + k as u32);
                let p = e(tr.b.access_chain(tr.t.sb_u32, None, info, [z, i]))?;
                *slot = e(tr.b.load(tr.t.u32, None, p, None, []))?;
            }
            tr.dirty_res.insert(r, (v[0], v[1], v[2]));
        }
    }
    // Buffer binding biases, loaded once in the entry block.
    let bias_base = bias_offset(plan);
    let buffer_res: Vec<usize> = tr.buffers.keys().copied().collect();
    for r in buffer_res {
        let v = tr.sgpr_data(bias_base + r as u32)?;
        tr.buf_bias.insert(r, v);
    }

    // Body.
    let starts = block_starts(code);
    if starts.len() == 1 {
        for ins in code {
            if ins.is_endpgm() {
                break;
            }
            translate_inst(&mut tr, ins)?;
        }
        mem::flush_dirty(&mut tr)?;
        tr.b.ret().map_err(|x| format!("{x:?}"))?;
    } else {
        // Structured control flow when the CFG allows it (TLG_STRUCTURED=0
        // keeps the dispatcher for every program, for A/B comparisons).
        let ranges = block_ranges(code, &starts);
        let want = std::env::var("TLG_STRUCTURED").map(|v| v != "0").unwrap_or(true);
        match want.then(|| cfg::plan(code, &starts, &ranges)) {
            Some(Ok(p)) => cfg::emit(&mut tr, code, &ranges, &p.nodes, None, None)?,
            other => {
                if wave_sync {
                    return Err("control flow not structured".into());
                }
                if let Some(Err(why)) = other {
                    tr.warnings.push(format!("control flow kept as a dispatcher loop: {why}"));
                }
                // (No uniform points inside the dispatcher: no LDS barriers.)
                tr.cf_depth = 1;
                emit_dispatch_loop(&mut tr, code, &starts)?;
            }
        }
    }
    let nested = tr.lds_hazards_nested;
    if nested > 0 {
        tr.warnings.push(format!("{nested} LDS read/write switch(es) inside control flow left without a barrier (wave lockstep assumed)"));
    }
    tr.b.end_function().map_err(|x| format!("{x:?}"))?;

    let model = match spec.stage {
        Stage::Vertex if spec.tess.role == TessRole::Ds => ExecutionModel::TessellationEvaluation,
        Stage::Vertex => ExecutionModel::Vertex,
        Stage::Fragment => ExecutionModel::Fragment,
        Stage::Compute => ExecutionModel::GLCompute,
    };
    let iface = tr.interface.clone();
    tr.b.entry_point(model, func, "main", iface);
    match spec.stage {
        Stage::Fragment => {
            tr.b.execution_mode(func, ExecutionMode::OriginUpperLeft, []);
            if tr.out_depth.is_some() {
                tr.b.execution_mode(func, ExecutionMode::DepthReplacing, []);
            }
        }
        Stage::Compute => {
            let l = spec.cs_local_size;
            tr.b.execution_mode(func, ExecutionMode::LocalSize, [l[0].max(1), l[1].max(1), l[2].max(1)]);
        }
        Stage::Vertex if spec.tess.role == TessRole::Ds => {
            // VGT_TF_PARAM: domain, partitioning, output topology.
            let p = spec.tess.tf_param;
            let domain = match p & 3 {
                0 => ExecutionMode::Isolines,
                1 => ExecutionMode::Triangles,
                _ => ExecutionMode::Quads,
            };
            let spacing = match (p >> 2) & 7 {
                2 => ExecutionMode::SpacingFractionalOdd,
                3 => ExecutionMode::SpacingFractionalEven,
                _ => ExecutionMode::SpacingEqual, // integer (and pow2)
            };
            tr.b.execution_mode(func, domain, []);
            tr.b.execution_mode(func, spacing, []);
            match (p >> 5) & 7 {
                0 => tr.b.execution_mode(func, ExecutionMode::PointMode, []),
                3 => tr.b.execution_mode(func, ExecutionMode::VertexOrderCcw, []),
                _ => tr.b.execution_mode(func, ExecutionMode::VertexOrderCw, []),
            }
        }
        Stage::Vertex => {}
    }
    if spec.denorm_flush32 {
        tr.b.extension("SPV_KHR_float_controls");
        tr.b.capability(Capability::DenormFlushToZero);
        tr.b.execution_mode(func, ExecutionMode::DenormFlushToZero, [32]);
    }
    let spirv = tr.b.module().assemble();
    let out = Output {
        spirv,
        bindings: tr.bindings,
        params: tr.params,
        mrt_mask: tr.mrt_mask,
        writes_depth: tr.out_depth.is_some(),
        pos_exports: tr.pos_exports,
        pos_ignored: tr.pos_ignored,
        warnings: tr.warnings,
    };
    Ok((out, nested))
}

const PC_EXIT: u32 = 0xFFFF_FFFF;

/// Zeroes a compute shader's workgroup LDS at entry (the invocations share
/// the dwords; then a workgroup barrier, in the entry block's uniform control
/// flow). Vulkan workgroup memory starts undefined, and lanes of a GCN wave
/// run here as independent invocations: a lane can read LDS before another
/// lane's initializing write that the hardware's lockstep would have ordered
/// first (cs 0x105fcedd00 reads a light count lane 0 resets; a stale count as
/// the loop bound hung the GPU).
fn zero_lds(tr: &mut T, var: Word) -> R<()> {
    let (Some(lds), None, Some(idx)) = (tr.lds, tr.lds_base, tr.local_index) else { return Ok(()) };
    if tr.lds_len == 0 {
        return Ok(());
    }
    let u = tr.t.u32;
    let [x, y, z] = tr.spec.cs_local_size;
    let stride = tr.u((x.max(1) * y.max(1) * z.max(1)).max(1));
    let n = tr.u(tr.lds_len);
    let zero = tr.u(0);
    tr.b.store(var, idx, None, []).map_err(|x| format!("{x:?}"))?;
    let (header, check, body, cont, merge) = (tr.b.id(), tr.b.id(), tr.b.id(), tr.b.id(), tr.b.id());
    tr.b.branch(header).map_err(|x| format!("{x:?}"))?;
    e(tr.b.begin_block(Some(header)))?;
    tr.b.loop_merge(merge, cont, spirv::LoopControl::NONE, []).map_err(|x| format!("{x:?}"))?;
    tr.b.branch(check).map_err(|x| format!("{x:?}"))?;
    e(tr.b.begin_block(Some(check)))?;
    let i = e(tr.b.load(u, None, var, None, []))?;
    let c = e(tr.b.u_less_than(tr.t.bool_, None, i, n))?;
    tr.b.branch_conditional(c, body, merge, []).map_err(|x| format!("{x:?}"))?;
    e(tr.b.begin_block(Some(body)))?;
    let p_wg = tr.b.type_pointer(None, StorageClass::Workgroup, u);
    let p = e(tr.b.access_chain(p_wg, None, lds, [i]))?;
    tr.b.store(p, zero, None, []).map_err(|x| format!("{x:?}"))?;
    tr.b.branch(cont).map_err(|x| format!("{x:?}"))?;
    e(tr.b.begin_block(Some(cont)))?;
    let i2 = e(tr.b.i_add(u, None, i, stride))?;
    tr.b.store(var, i2, None, []).map_err(|x| format!("{x:?}"))?;
    tr.b.branch(header).map_err(|x| format!("{x:?}"))?;
    e(tr.b.begin_block(Some(merge)))?;
    let wg = tr.u(spirv::Scope::Workgroup as u32);
    let sem = tr.u((spirv::MemorySemantics::ACQUIRE_RELEASE | spirv::MemorySemantics::WORKGROUP_MEMORY).bits());
    tr.b.control_barrier(wg, wg, sem).map_err(|x| format!("{x:?}"))?;
    Ok(())
}

/// The workgroup vote's slots cleared and its index at 0 (`T::wave_any`),
/// then a barrier before any vote, in the entry block.
fn init_votes(tr: &mut T) -> R<()> {
    let Some((slots, idx)) = tr.vote else { return Ok(()) };
    let u = tr.t.u32;
    let zero = tr.u(0);
    tr.b.store(idx, zero, None, []).map_err(|x| format!("{x:?}"))?;
    let p_wg = tr.b.type_pointer(None, StorageClass::Workgroup, u);
    let scope = tr.u(spirv::Scope::Workgroup as u32);
    let relaxed = tr.u(0);
    for k in 0..3 {
        let kk = tr.u(k);
        let p = e(tr.b.access_chain(p_wg, None, slots, [kk]))?;
        tr.b.atomic_store(p, scope, relaxed, zero).map_err(|x| format!("{x:?}"))?;
    }
    tr.workgroup_barrier()
}

fn emit_dispatch_loop(tr: &mut T, code: &[Inst], starts: &BTreeSet<u32>) -> R<()> {
    let first = code[0].pc;
    let s0 = tr.u(first);
    tr.b.store(tr.pc_var, s0, None, []).map_err(|x| format!("{x:?}"))?;
    let header = tr.b.id();
    let dispatch = tr.b.id();
    let sw_merge = tr.b.id();
    let cont = tr.b.id();
    let exit = tr.b.id();
    tr.exit_block = Some(exit);
    tr.b.branch(header).map_err(|x| format!("{x:?}"))?;
    e(tr.b.begin_block(Some(header)))?;
    tr.b.loop_merge(exit, cont, spirv::LoopControl::NONE, []).map_err(|x| format!("{x:?}"))?;
    tr.b.branch(dispatch).map_err(|x| format!("{x:?}"))?;
    e(tr.b.begin_block(Some(dispatch)))?;
    let pc = e(tr.b.load(tr.t.u32, None, tr.pc_var, None, []))?;
    let block_ids: Vec<(u32, Word)> = starts.iter().map(|&s| (s, tr.b.id())).collect();
    tr.b.selection_merge(sw_merge, spirv::SelectionControl::NONE).map_err(|x| format!("{x:?}"))?;
    let targets: Vec<(Operand, Word)> = block_ids.iter().map(|&(s, id)| (Operand::LiteralBit32(s), id)).collect();
    tr.b.switch(pc, sw_merge, targets).map_err(|x| format!("{x:?}"))?;
    for (&(_, id), range) in block_ids.iter().zip(block_ranges(code, starts)) {
        e(tr.b.begin_block(Some(id)))?;
        let mut next_pc = PC_EXIT;
        let mut terminated = false;
        for k in range {
            let ins = &code[k];
            next_pc = code.get(k + 1).map(|n| n.pc).unwrap_or(PC_EXIT);
            if ins.is_endpgm() {
                let x = tr.u(PC_EXIT);
                tr.b.store(tr.pc_var, x, None, []).map_err(|x| format!("{x:?}"))?;
                terminated = true;
                break;
            }
            if let Some(target) = ins.branch_target() {
                let fall = next_pc;
                let cond = match ins.op {
                    2 => None,
                    4 => {
                        let s = tr.get_scc()?;
                        Some(e(tr.b.logical_not(tr.t.bool_, None, s))?)
                    }
                    5 => Some(tr.get_scc()?),
                    6 | 7 | 8 | 9 => {
                        let src = if ins.op <= 7 { crate::OP_VCC_LO } else { crate::OP_EXEC_LO };
                        let lo = tr.sreg(src as u32)?;
                        let hi = tr.sreg(src as u32 + 1)?;
                        let o = e(tr.b.bitwise_or(tr.t.u32, None, lo, hi))?;
                        let z = tr.u(0);
                        let zero = e(tr.b.i_equal(tr.t.bool_, None, o, z))?;
                        Some(if ins.op == 6 || ins.op == 8 { zero } else { e(tr.b.logical_not(tr.t.bool_, None, zero))? })
                    }
                    _ => None,
                };
                let tv = tr.u(target);
                let val = match cond {
                    None => tv,
                    Some(c) => {
                        let fv = tr.u(fall);
                        e(tr.b.select(tr.t.u32, None, c, tv, fv))?
                    }
                };
                tr.b.store(tr.pc_var, val, None, []).map_err(|x| format!("{x:?}"))?;
                terminated = true;
                break;
            }
            translate_inst(tr, ins)?;
        }
        if !terminated {
            let x = tr.u(next_pc);
            tr.b.store(tr.pc_var, x, None, []).map_err(|x| format!("{x:?}"))?;
        }
        tr.b.branch(sw_merge).map_err(|x| format!("{x:?}"))?;
    }
    e(tr.b.begin_block(Some(sw_merge)))?;
    tr.b.branch(cont).map_err(|x| format!("{x:?}"))?;
    e(tr.b.begin_block(Some(cont)))?;
    let pc2 = e(tr.b.load(tr.t.u32, None, tr.pc_var, None, []))?;
    let ex = tr.u(PC_EXIT);
    let done = e(tr.b.i_equal(tr.t.bool_, None, pc2, ex))?;
    tr.b.branch_conditional(done, exit, header, []).map_err(|x| format!("{x:?}"))?;
    e(tr.b.begin_block(Some(exit)))?;
    mem::flush_dirty(tr)?;
    tr.b.ret().map_err(|x| format!("{x:?}"))?;
    Ok(())
}

fn translate_inst(tr: &mut T, ins: &Inst) -> R<()> {
    translate_inst_inner(tr, ins)?;
    for c in 0..tr.spec.debug_capture.len() {
        let (pc, v, int) = tr.spec.debug_capture[c];
        if ins.pc == pc {
            for k in 0..3 {
                let mut x = tr.vreg(v + k)?;
                if int {
                    // Scaled by 1/255: on UNORM8 targets the stored byte equals the integer.
                    let f = e(tr.b.convert_u_to_f(tr.t.f32, None, x))?;
                    let k = tr.f(1.0 / 255.0);
                    let f = e(tr.b.f_mul(tr.t.f32, None, f, k))?;
                    x = e(tr.b.bitcast(tr.t.u32, None, f))?;
                }
                let var = tr.dbg_vars[c * 3 + k as usize];
                tr.b.store(var, x, None, []).map_err(|x| format!("{x:?}"))?;
            }
        }
    }
    Ok(())
}

fn translate_inst_inner(tr: &mut T, ins: &Inst) -> R<()> {
    match ins.enc {
        Enc::Sop2 | Enc::Sop1 | Enc::Sopc | Enc::Sopk | Enc::Sopp => alu::scalar(tr, ins),
        Enc::Vop1 | Enc::Vop2 | Enc::Vopc | Enc::Vop3 => alu::vector(tr, ins),
        Enc::Smrd => mem::smrd(tr, ins),
        Enc::Vintrp => mem::interp(tr, ins),
        Enc::Mubuf | Enc::Mtbuf => mem::buffer(tr, ins),
        Enc::Mimg => mem::image(tr, ins),
        Enc::Exp => mem::export(tr, ins),
        Enc::Ds => mem::ds(tr, ins),
        Enc::Flat => {
            tr.warn(format!("{:#x}: FLAT instructions unsupported", ins.pc));
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_distance_layout() {
        // Bloodborne's vertex shaders: CCDIST0 vector, clip distances 0 and 1.
        let l = dist_layout(0x0040_0003);
        assert_eq!(l.vec_target, [Some(13), None]);
        assert_eq!((l.clip[0], l.clip[1], l.clip[2]), (Some(0), Some(1), None));
        assert_eq!((l.clip_count, l.cull_count, l.misc_target), (2, 0, None));
        // Misc vector first, then both distance vectors; clip 0 and 5, cull 6.
        let l = dist_layout((1 << 21) | (1 << 22) | (1 << 23) | 0x21 | (0x40 << 8));
        assert_eq!((l.misc_target, l.vec_target), (Some(13), [Some(14), Some(15)]));
        assert_eq!((l.clip[0], l.clip[5], l.cull[6]), (Some(0), Some(1), Some(0)));
        assert_eq!((l.clip_count, l.cull_count), (2, 1));
        // Enables without their vector have no source.
        assert_eq!(dist_layout(0x3).clip_count, 0);
    }

    fn vertex_spec(vs_out_cntl: u32) -> Spec {
        Spec {
            stage: Stage::Vertex,
            user_sgprs: 0,
            vs_vgpr_comp_cnt: 0,
            vs_step_rates: [1, 1],
            vs_out_cntl,
            denorm_flush32: false,
            debug_pos_store: false,
            tess: TessSpec::default(),
            ps_input_addr: 0,
            ps_inputs: Vec::new(),
            mrt: [MrtSpec::default(); 8],
            cs_local_size: [1, 1, 1],
            cs_rsrc2: 0,
            resources: Vec::new(),
            buffer_strides: Vec::new(),
            binding_base: 0,
            debug_capture: Vec::new(),
            page_marks: 1,
        }
    }

    #[test]
    fn vertex_shader_exports_clip_distances() {
        // exp pos0 v0..v3; exp pos1 v4 v5 v4 v4 done; s_endpgm
        let exp = |tgt: u32, done: bool, v: [u32; 4]| [0xF800_000F | (u32::from(done) << 11) | (tgt << 4), v[0] | v[1] << 8 | v[2] << 16 | v[3] << 24];
        let mut words = Vec::new();
        words.extend(exp(12, false, [0, 1, 2, 3]));
        words.extend(exp(13, true, [4, 5, 4, 4]));
        words.push(0xBF81_0000);
        let code = crate::decode_program(&words).unwrap();
        let plan = crate::analysis::analyze(&code, 0, 0);
        let builtins = |spirv: &[u32]| -> Vec<BuiltIn> {
            let m = rspirv::dr::load_words(spirv).unwrap();
            m.annotations.iter().filter_map(|i| match i.operands.as_slice() {
                [_, Operand::Decoration(Decoration::BuiltIn), Operand::BuiltIn(b)] => Some(*b),
                _ => None,
            }).collect()
        };
        let out = translate(&code, &plan, &vertex_spec(0x0040_0003)).unwrap();
        assert!(builtins(&out.spirv).contains(&BuiltIn::ClipDistance));
        assert_eq!(out.pos_ignored, 0, "{:?}", out.warnings);
        // Without the vector enable the export has no use: no ClipDistance,
        // the export is reported.
        let out = translate(&code, &plan, &vertex_spec(0x3)).unwrap();
        assert!(!builtins(&out.spirv).contains(&BuiltIn::ClipDistance));
        assert_eq!(out.pos_ignored, 0b10);
    }

    #[test]
    fn inlined_fetch_runs_at_its_call_site() {
        // main: s_mov_b32 s0, 0; s_swappc_b64 s[0:1], s[0:1];
        //       s_cbranch_scc0 0x10; v_mov_b32 v1, v0; 0x10: s_endpgm
        let main = crate::decode_program(&[0xBE80_0380, 0xBE80_2100, 0xBF84_0001, 0x7E02_0300, 0xBF81_0000]).unwrap();
        // fetch: v_mov_b32 v2, v0; s_setpc_b64 s[0:1]
        let fetch = crate::decode_program(&[0x7E04_0300, 0xBE80_2000]).unwrap();
        let code = inline_fetch(&main, &fetch);
        assert_eq!(code.iter().map(|i| i.pc).collect::<Vec<_>>(), [0x0, 0x10_0000, 0x8, 0xc, 0x10]);
        let starts = block_starts(&code);
        assert_eq!(starts.iter().copied().collect::<Vec<_>>(), [0x0, 0xc, 0x10]);
        // The fetch code belongs to the entry block (a pc-range split put it
        // in the last block, after the code using its results).
        assert_eq!(block_ranges(&code, &starts), [0..3, 3..4, 4..5]);
        let plan = crate::analysis::analyze(&code, 0, 0);
        translate(&code, &plan, &vertex_spec(0)).unwrap();
    }

    #[test]
    fn valid_mask_export_kills_only_after_exec_writes() {
        let kills = |words: &[u32]| -> usize {
            let code = crate::decode_program(words).unwrap();
            let plan = crate::analysis::analyze(&code, 0, 0);
            let spec = Spec { stage: Stage::Fragment, ..vertex_spec(0) };
            let out = translate(&code, &plan, &spec).unwrap();
            let m = rspirv::dr::load_words(&out.spirv).unwrap();
            m.functions.iter().flat_map(|f| &f.blocks).flat_map(|b| &b.instructions).filter(|i| i.class.opcode == spirv::Op::Kill).count()
        };
        // TLG's depth-only ps 0x1130fedf00: s_mov_b32 vcc_hi, 3;
        // exp null done vm; s_endpgm. Every lane stays active.
        assert_eq!(kills(&[0xBEEB_0383, 0xF800_1890, 0, 0xBF81_0000]), 0);
        // v_cmpx_gt_u32 vcc, v0, v1 first (an alpha test): lanes may die.
        assert_eq!(kills(&[0x7DA8_0300, 0xF800_1890, 0, 0xBF81_0000]), 1);
    }

    #[test]
    fn page_marks_are_flushed_once_at_the_end() {
        // buffer_store_dword v0, v1, s[0:3], 0 offen; buffer_store_dwordx2
        // v[2:3], v1, s[0:3], 0 offen; s_endpgm
        let code = crate::decode_program(&[0xE070_1000, 0x8000_0001, 0xE074_1000, 0x8000_0201, 0xBF81_0000]).unwrap();
        let plan = crate::analysis::analyze(&code, 4, 0);
        assert!(plan.resources.iter().any(|r| r.kind == ResKind::Buffer && r.written));
        let spec = Spec { stage: Stage::Compute, user_sgprs: 4, resources: vec![ResSpec::Buffer { dfmt: 4, nfmt: 4, dst_sel: 0 }], buffer_strides: vec![0], ..vertex_spec(0) };
        let ops_of = |page_marks: u8| -> Vec<spirv::Op> {
            let out = translate(&code, &plan, &Spec { page_marks, ..spec.clone() }).unwrap();
            let m = rspirv::dr::load_words(&out.spirv).unwrap();
            m.functions.iter().flat_map(|f| &f.blocks).flat_map(|b| &b.instructions).map(|i| i.class.opcode).collect()
        };
        let ops = ops_of(1);
        let count = |ops: &[spirv::Op], op: spirv::Op| ops.iter().filter(|&&o| o == op).count();
        // Per store an atomic for the word left behind (the dword store's
        // bytes share a page; the dwordx2's last byte may be in another
        // word: one more), one subgroup-coalesced atomic at the end.
        assert_eq!(count(&ops, spirv::Op::GroupNonUniformAllEqual), 1);
        assert_eq!(count(&ops, spirv::Op::GroupNonUniformElect), 1);
        assert_eq!(count(&ops, spirv::Op::AtomicOr), 1 + 2 + 1);
        // The flush comes before the return.
        let last_atomic = ops.iter().rposition(|&o| o == spirv::Op::AtomicOr).unwrap();
        assert!(ops[last_atomic..].contains(&spirv::Op::Return));
        // The per-store marks: a subgroup reduction per page (1 + 2); plain
        // atomics: one per page, no subgroup operation.
        let old = ops_of(0);
        assert_eq!(count(&old, spirv::Op::GroupNonUniformAllEqual), 3);
        let plain = ops_of(2);
        assert_eq!(count(&plain, spirv::Op::GroupNonUniformAllEqual), 0);
        assert_eq!(count(&plain, spirv::Op::AtomicOr), 3);
        // A word per page (3): a plain store per page (1 + 2), no atomic,
        // no subgroup operation, nothing pending to flush.
        let map = ops_of(3);
        assert_eq!(count(&map, spirv::Op::AtomicOr), 0);
        assert_eq!(count(&map, spirv::Op::GroupNonUniformAllEqual), 0);
        PROBE_MARKS.with(|s| s.set(1));
        let none = ops_of(1);
        PROBE_MARKS.with(|s| s.set(0));
        assert_eq!(count(&map, spirv::Op::Store), count(&none, spirv::Op::Store) + 3);
    }

    #[test]
    fn bit_moves_apply_vop3_sign_modifiers() {
        // VOP3 v_cndmask_b32 v0, v1, -v1, vcc; VOP3 v_mov_b32 v2, |v3|; s_endpgm
        let vop3 = |op: u32, vdst: u32, abs: u32, src: [u32; 3], neg: u32| [0xD000_0000 | (op << 17) | (abs << 8) | vdst, src[0] | (src[1] << 9) | (src[2] << 18) | (neg << 29)];
        let mut words = Vec::new();
        words.extend(vop3(0x100, 0, 0, [257, 257, 106], 0b010));
        words.extend(vop3(0x181, 2, 0b001, [259, 0, 0], 0));
        words.push(0xBF81_0000);
        let code = crate::decode_program(&words).unwrap();
        let plan = crate::analysis::analyze(&code, 0, 0);
        let out = translate(&code, &plan, &vertex_spec(0)).unwrap();
        let m = rspirv::dr::load_words(&out.spirv).unwrap();
        let consts: HashMap<Word, u32> = m.types_global_values.iter().filter_map(|i| match (i.class.opcode, i.result_id, i.operands.as_slice()) {
            (spirv::Op::Constant, Some(id), [Operand::LiteralBit32(v)]) => Some((id, *v)),
            _ => None,
        }).collect();
        let uses = |op: spirv::Op, k: u32| {
            m.functions.iter().flat_map(|f| &f.blocks).flat_map(|b| &b.instructions).any(|i| i.class.opcode == op && i.operands.iter().any(|o| matches!(o, Operand::IdRef(id) if consts.get(id) == Some(&k))))
        };
        assert!(uses(spirv::Op::BitwiseXor, 0x8000_0000), "the negation of v_cndmask's src1");
        assert!(uses(spirv::Op::BitwiseAnd, 0x7FFF_FFFF), "the abs of v_mov's src0");
    }
}
