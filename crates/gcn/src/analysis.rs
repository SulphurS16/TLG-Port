//! Scalar provenance analysis.
//!
//! Tracks where each SGPR dword comes from (user data, a scalar memory load
//! through a pointer derived from user data, or a constant) so that resource
//! descriptors and scalar-loaded data can be resolved on the CPU at draw time.
//! Flow-sensitive: states are propagated along the basic-block graph and
//! merged at joins (values that differ between paths become unknown), so a
//! register clobbered on one branch does not affect the others.

use crate::{Enc, Inst};
use crate::DetMap as HashMap;
use std::collections::{BTreeSet, VecDeque};

/// Source of a 64-bit pointer used by scalar loads.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ptr {
    /// User-data SGPR pair (lo index).
    User(u16),
    /// Pointer loaded from memory: dword pair at `offset` from another pointer.
    Mem(Box<Ptr>, u32),
    /// Absolute address (e.g. from s_getpc_b64).
    Abs(u64),
    /// Pointer computed from two known dwords (low, high), e.g. a masked or
    /// offset table pointer.
    Pair(Box<(Prov, Prov)>),
}

/// Provenance of one SGPR dword.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Prov {
    Unknown,
    User(u16),
    Const(u32),
    /// dword at byte `offset` from pointer `ptr` (plus a constant pointer bias).
    Mem(Ptr, u32),
    /// Low/high half of `ptr + bias` (pointer arithmetic).
    PtrLo(Ptr, u32),
    PtrHi(Ptr, u32),
    /// Scalar ALU result of two known dwords (descriptor words built in the
    /// shader, such as a table address with the stride or'ed in).
    Op(SOp, Box<(Prov, Prov)>),
    /// Low dword of a pointer whose low dword is `L` plus a run-time
    /// (wave-uniform) offset, computed with vector ALU ops and moved back to
    /// an SGPR with v_readfirstlane (e.g. a light record indexed from LDS).
    DynLo(Box<Prov>),
    /// High dword of that pointer: (`L`, high dword `H` plus the carry).
    DynHi(Box<(Prov, Prov)>),
}

/// Scalar ALU operations tracked by `Prov::Op` (evaluated by `SOp::apply`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SOp {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    AndN2,
    OrN2,
    Nand,
    Nor,
    Xnor,
    Shl,
    Shr,
    Ashr,
    /// s_bfe_u32/s_bfe_i32: field of `a` at `b[4:0]`, `b[22:16]` bits wide.
    BfeU,
    BfeI,
    /// s_bfm_b32: `((1 << a) - 1) << b`.
    Bfm,
    MinI,
    MinU,
    MaxI,
    MaxU,
    /// Carry out of `a + b` (SCC of s_add_u32).
    Carry,
    /// Borrow out of `a - b` (SCC of s_sub_u32).
    Borrow,
}

impl SOp {
    pub fn apply(self, a: u32, b: u32) -> u32 {
        let field = |b: u32| (b & 31, (b >> 16) & 0x7F);
        match self {
            SOp::Add => a.wrapping_add(b),
            SOp::Sub => a.wrapping_sub(b),
            SOp::Mul => a.wrapping_mul(b),
            SOp::And => a & b,
            SOp::Or => a | b,
            SOp::Xor => a ^ b,
            SOp::AndN2 => a & !b,
            SOp::OrN2 => a | !b,
            SOp::Nand => !(a & b),
            SOp::Nor => !(a | b),
            SOp::Xnor => !(a ^ b),
            SOp::Shl => a << (b & 31),
            SOp::Shr => a >> (b & 31),
            SOp::Ashr => ((a as i32) >> (b & 31)) as u32,
            SOp::BfeU => {
                let (off, w) = field(b);
                if w == 0 { 0 } else if w >= 32 { a >> off } else { (a >> off) & ((1 << w) - 1) }
            }
            SOp::BfeI => {
                let (off, w) = field(b);
                if w == 0 {
                    0
                } else if w >= 32 {
                    ((a as i32) >> off) as u32
                } else {
                    let v = (a >> off) << (32 - w);
                    ((v as i32) >> (32 - w)) as u32
                }
            }
            SOp::Bfm => (((1u64 << (a & 31)) - 1) as u32) << (b & 31),
            SOp::MinI => (a as i32).min(b as i32) as u32,
            SOp::MinU => a.min(b),
            SOp::MaxI => (a as i32).max(b as i32) as u32,
            SOp::MaxU => a.max(b),
            SOp::Carry => ((a as u64 + b as u64) >> 32) as u32,
            SOp::Borrow => (b > a) as u32,
        }
    }
}

/// Expressions larger than this (nodes) are not tracked.
const MAX_NODES: usize = 40;

impl Prov {
    fn nodes(&self) -> usize {
        match self {
            Prov::Op(_, ab) | Prov::DynHi(ab) => 1 + ab.0.nodes() + ab.1.nodes(),
            Prov::Mem(p, _) | Prov::PtrLo(p, _) | Prov::PtrHi(p, _) => 1 + p.nodes(),
            Prov::DynLo(l) => 1 + l.nodes(),
            _ => 1,
        }
    }

    /// A value known only at run time (`DynLo`/`DynHi`): not resolvable on
    /// the CPU.
    fn dynamic(&self) -> bool {
        matches!(self, Prov::DynLo(_) | Prov::DynHi(_))
    }
}

impl Ptr {
    fn nodes(&self) -> usize {
        match self {
            Ptr::Mem(p, _) => 1 + p.nodes(),
            Ptr::Pair(ab) => 1 + ab.0.nodes() + ab.1.nodes(),
            _ => 1,
        }
    }
}

/// `o(a, b)`: folded for constants and identities, unknown when an operand is
/// (or the expression would grow too large).
fn op(o: SOp, a: Prov, b: Prov) -> Prov {
    use Prov::{Const, Unknown};
    if a.dynamic() || b.dynamic() {
        return Unknown;
    }
    match (o, &a, &b) {
        (_, Unknown, _) | (_, _, Unknown) => Unknown,
        (_, Const(x), Const(y)) => Const(o.apply(*x, *y)),
        (SOp::Add | SOp::Sub | SOp::Or | SOp::Xor | SOp::AndN2 | SOp::Shl | SOp::Shr | SOp::Ashr, _, Const(0)) => a,
        (SOp::Add | SOp::Or | SOp::Xor, Const(0), _) => b,
        (SOp::And, _, Const(u32::MAX)) => a,
        (SOp::And, Const(u32::MAX), _) => b,
        (SOp::And | SOp::Mul | SOp::Carry, _, Const(0)) | (SOp::And | SOp::Mul | SOp::Carry, Const(0), _) | (SOp::Borrow, _, Const(0)) => Const(0),
        _ if a.nodes() + b.nodes() >= MAX_NODES => Unknown,
        _ => Prov::Op(o, Box::new((a, b))),
    }
}

/// V# dword 3 of the windows set up for scalar loads with run-time offsets:
/// X/Y/Z/W selects, 32-bit uint (raw loads ignore the format).
pub const WINDOW_DWORD3: u32 = 0x0002_4FAC;
/// Bytes readable through such a window (from the pointer on).
pub const WINDOW_BYTES: u32 = 0x1_0000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResKind {
    /// V# used by buffer instructions (MUBUF/MTBUF/s_buffer_load).
    Buffer,
    /// T# used by image instructions.
    Image,
    /// S# used by sampling instructions.
    Sampler,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Resource {
    pub kind: ResKind,
    /// Provenance of each descriptor dword (4 for V#/S#, 8 for T#).
    pub dwords: Vec<Prov>,
    /// Buffer is written by the shader / image is a storage image.
    pub written: bool,
    /// Buffer is read by the shader (loads, atomics, scalar buffer loads).
    pub read: bool,
    /// Buffer is read with format conversion (buffer_load_format/tbuffer).
    pub formatted: bool,
}

/// A scalar memory load whose dwords are resolved on the CPU at draw time.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ScalarLoad {
    pub ptr: Ptr,
    pub offset: u32,
    pub dwords: u32,
}

#[derive(Default, Clone, Debug)]
pub struct Plan {
    pub resources: Vec<Resource>,
    /// Instruction pc -> resource indices (descriptor, sampler) it uses.
    pub inst_res: HashMap<u32, (usize, Option<usize>)>,
    pub loads: Vec<ScalarLoad>,
    /// Instruction pc -> index into `loads` for s_load_dword* instructions.
    pub inst_load: HashMap<u32, usize>,
    /// Dwords the CPU evaluates per draw for the shader (SGPR data after the
    /// resource biases, see `translate::values_offset`).
    pub values: Vec<Prov>,
    /// s_load_dword* through a `DynLo`/`DynHi` address -> index into
    /// `values` of the window pointer's low dword (the shader subtracts it
    /// from the address to get its offset into the window, `inst_res`).
    pub inst_dyn: HashMap<u32, usize>,
    /// s_getpc_b64 -> indices into `values` of the address it returns.
    pub inst_pc: HashMap<u32, (usize, usize)>,
    /// Problems found (unresolvable descriptors etc.).
    pub errors: Vec<String>,
}

impl Plan {
    fn value(&mut self, p: Prov) -> usize {
        if let Some(i) = self.values.iter().position(|v| *v == p) {
            return i;
        }
        self.values.push(p);
        self.values.len() - 1
    }

    fn resource(&mut self, r: Resource) -> usize {
        if let Some(i) = self.resources.iter().position(|x| x.kind == r.kind && x.dwords == r.dwords) {
            let x = &mut self.resources[i];
            x.written |= r.written;
            x.read |= r.read;
            x.formatted |= r.formatted;
            return i;
        }
        self.resources.push(r);
        self.resources.len() - 1
    }
}

#[derive(Clone)]
struct State {
    s: Vec<Prov>,
    /// SGPR values spilled into VGPR lanes (v_writelane with constant lanes).
    lanes: Vec<((u16, u32), Prov)>,
    scc: Prov,
    /// Set by an `s_add_u32` that offsets the low half of a pointer pair by a
    /// constant, for the `s_addc_u32` right after it.
    ptr_add: Option<PtrAdd>,
    /// VGPRs holding a tracked value: copies of scalar provenance
    /// (v_mov_b32 from an SGPR) and `DynLo`/`DynHi` pointer halves.
    v: Vec<(u16, Prov)>,
    /// VCC holds the carry of a v_add_i32 of a pointer's low dword (this
    /// provenance) and a run-time value, for the v_addc_u32 of its high dword.
    vcc_lo_of: Option<Prov>,
}

#[derive(Clone, PartialEq)]
struct PtrAdd {
    /// SGPR holding the pointer's high half, and where the sum's goes.
    hi_src: u16,
    hi_dst: u16,
    ptr: Ptr,
    bias: u32,
}

impl State {
    /// Merges another path's state into this one; returns whether it changed.
    fn merge(&mut self, o: &State) -> bool {
        let mut changed = false;
        for (a, b) in self.s.iter_mut().zip(o.s.iter()).chain(std::iter::once((&mut self.scc, &o.scc))) {
            if a != b && *a != Prov::Unknown {
                *a = Prov::Unknown;
                changed = true;
            }
        }
        if self.ptr_add.is_some() && self.ptr_add != o.ptr_add {
            self.ptr_add = None;
            changed = true;
        }
        if self.vcc_lo_of.is_some() && self.vcc_lo_of != o.vcc_lo_of {
            self.vcc_lo_of = None;
            changed = true;
        }
        let (before, vbefore) = (self.lanes.len(), self.v.len());
        self.lanes.retain(|(k, v)| o.lanes.iter().any(|(k2, v2)| k2 == k && v2 == v));
        self.v.retain(|(k, v)| o.v.iter().any(|(k2, v2)| k2 == k && v2 == v));
        changed || self.lanes.len() != before || self.v.len() != vbefore
    }
    /// Tracked value of VGPR `r` (Unknown when not tracked).
    fn vget(&self, r: u16) -> Prov {
        self.v.iter().find(|(k, _)| *k == r).map(|(_, p)| p.clone()).unwrap_or(Prov::Unknown)
    }
    fn vset(&mut self, r: u16, p: Prov) {
        self.v.retain(|(k, _)| *k != r);
        if p != Prov::Unknown && self.v.len() < 64 {
            self.v.push((r, p));
        }
    }
    fn get(&self, r: u16) -> Prov {
        match r {
            0..=127 => self.s[r as usize].clone(),
            128 => Prov::Const(0),
            129..=192 => Prov::Const((r - 128) as u32),
            193..=208 => Prov::Const((-((r - 192) as i32)) as u32),
            240..=247 => Prov::Const([0.5f32, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0][(r - 240) as usize].to_bits()),
            253 => self.scc.clone(),
            _ => Prov::Unknown,
        }
    }
    fn set(&mut self, r: u16, p: Prov) {
        if r == crate::OP_VCC_LO || r == crate::OP_VCC_HI {
            self.vcc_lo_of = None;
        }
        if (r as usize) < self.s.len() {
            self.s[r as usize] = p;
        }
    }
    /// Interprets SGPR pair `r` as a pointer.
    fn ptr(&self, r: u16) -> Option<(Ptr, u32)> {
        match (self.get(r), self.get(r + 1)) {
            (Prov::User(a), Prov::User(b)) if b == a + 1 => Some((Ptr::User(a), 0)),
            (Prov::Mem(p, o), Prov::Mem(q, o2)) if p == q && o2 == o + 4 => Some((Ptr::Mem(Box::new(p), o), 0)),
            (Prov::PtrLo(p, b), Prov::PtrHi(q, b2)) if p == q && b == b2 => Some((p, b)),
            (Prov::Unknown, _) | (_, Prov::Unknown) => None,
            (lo, hi) if lo.dynamic() || hi.dynamic() => None,
            (lo, hi) => Some((Ptr::Pair(Box::new((lo, hi))), 0)),
        }
    }
}

fn src_operand(st: &State, ins: &Inst, op: u16) -> Prov {
    if op == crate::OP_LITERAL {
        return ins.lit.map(Prov::Const).unwrap_or(Prov::Unknown);
    }
    st.get(op)
}

/// Both halves of a 64-bit scalar source (inline integer constants are
/// sign-extended, literals zero-extended).
fn src_operand64(st: &State, ins: &Inst, op: u16) -> (Prov, Prov) {
    match op {
        0..=127 => (st.get(op), st.get(op + 1)),
        193..=208 => (st.get(op), Prov::Const(u32::MAX)),
        128..=192 | crate::OP_LITERAL => {
            let lo = src_operand(st, ins, op);
            let hi = if lo == Prov::Unknown { Prov::Unknown } else { Prov::Const(0) };
            (lo, hi)
        }
        _ => (Prov::Unknown, Prov::Unknown),
    }
}

/// Runs the analysis. `user_sgprs` is the number of user-data SGPRs (s0..).
pub fn analyze(code: &[Inst], user_sgprs: u16, code_base: u64) -> Plan {
    let mut init = State { s: vec![Prov::Unknown; 128], lanes: Vec::new(), scc: Prov::Unknown, ptr_add: None, v: Vec::new(), vcc_lo_of: None };
    for i in 0..user_sgprs.min(16) {
        init.s[i as usize] = Prov::User(i);
    }
    let n = code.len();
    if n == 0 {
        return Plan::default();
    }
    // Basic blocks.
    let index: HashMap<u32, usize> = code.iter().enumerate().map(|(i, ins)| (ins.pc, i)).collect();
    let mut starts = BTreeSet::new();
    starts.insert(0usize);
    for (i, ins) in code.iter().enumerate() {
        if let Some(t) = ins.branch_target() {
            if let Some(&j) = index.get(&t) {
                starts.insert(j);
            }
            if i + 1 < n {
                starts.insert(i + 1);
            }
        }
        if ins.is_endpgm() && i + 1 < n {
            starts.insert(i + 1);
        }
    }
    let starts: Vec<usize> = starts.into_iter().collect();
    let block_at: HashMap<usize, usize> = starts.iter().enumerate().map(|(b, &s)| (s, b)).collect();
    let end = |b: usize| starts.get(b + 1).copied().unwrap_or(n);
    let succ: Vec<Vec<usize>> = (0..starts.len())
        .map(|b| {
            let e = end(b);
            let last = &code[e - 1];
            let mut v = Vec::new();
            if last.is_endpgm() {
                return v;
            }
            match last.branch_target() {
                Some(t) => {
                    if let Some(j) = index.get(&t) {
                        v.push(block_at[j]);
                    }
                    if last.op != 2 && e < n {
                        v.push(block_at[&e]);
                    }
                }
                None if e < n => v.push(b + 1),
                None => {}
            }
            v
        })
        .collect();
    // Fixpoint over block entry states (records are discarded here).
    let mut entry: Vec<Option<State>> = vec![None; starts.len()];
    entry[0] = Some(init);
    let mut work: VecDeque<usize> = VecDeque::from([0]);
    let mut queued = vec![false; starts.len()];
    queued[0] = true;
    let mut scratch = Plan::default();
    while let Some(b) = work.pop_front() {
        queued[b] = false;
        let mut st = entry[b].clone().unwrap();
        for ins in &code[starts[b]..end(b)] {
            step(&mut st, ins, code_base, &mut scratch);
        }
        scratch = Plan::default();
        for &sb in &succ[b] {
            let changed = match &mut entry[sb] {
                None => {
                    entry[sb] = Some(st.clone());
                    true
                }
                Some(old) => old.merge(&st),
            };
            if changed && !queued[sb] {
                queued[sb] = true;
                work.push_back(sb);
            }
        }
    }
    // Final pass over reachable blocks with their converged entry states.
    let mut plan = Plan::default();
    for b in 0..starts.len() {
        let Some(mut st) = entry[b].clone() else { continue };
        for ins in &code[starts[b]..end(b)] {
            step(&mut st, ins, code_base, &mut plan);
        }
    }
    for (i, r) in plan.resources.iter().enumerate() {
        // (Sampled images and samplers loaded at run time get stand-ins when
        // the shader is prepared; anything else cannot be bound.)
        let stand_in = matches!(r.kind, ResKind::Image | ResKind::Sampler) && !r.written;
        if !stand_in && r.dwords.iter().any(|p| matches!(p, Prov::Unknown) || p.dynamic()) {
            plan.errors.push(format!("resource {i} ({:?}) has unresolvable descriptor dwords {:?}", r.kind, r.dwords));
        }
    }
    plan
}

/// Vector pointer arithmetic whose result returns to SGPRs (v_readfirstlane)
/// as a scalar-load address: VOP2 v_add_i32 of a pointer's low dword `L`
/// (scalar provenance, not a constant) and a run-time value is `DynLo(L)`,
/// its carry in VCC; v_addc_u32 of the high dword `H` (or of `DynHi`), 0
/// and that carry is `DynHi(L, H)`.
fn vector_ptr_add(st: &mut State, op: u16, vdst: u16, a: Prov, b: Prov, carry_in: Option<Prov>) {
    let pointerish = |p: &Prov| !matches!(p, Prov::Unknown | Prov::Const(_)) && !p.dynamic();
    let runtime = |p: &Prov| matches!(p, Prov::Unknown | Prov::Const(_));
    match op {
        // v_add_i32
        37 => {
            let lo = match (&a, &b) {
                (Prov::DynLo(l), _) | (_, Prov::DynLo(l)) => Some((**l).clone()),
                (x, y) if pointerish(x) && runtime(y) => Some(x.clone()),
                (y, x) if pointerish(x) && runtime(y) => Some(x.clone()),
                _ => None,
            };
            if let Some(l) = lo {
                st.vset(vdst, Prov::DynLo(Box::new(l.clone())));
                st.vcc_lo_of = Some(l);
            }
        }
        // v_addc_u32
        40 => {
            let Some(l) = carry_in else { return };
            let hi = match (&a, &b) {
                (Prov::DynHi(lh), Prov::Const(0)) | (Prov::Const(0), Prov::DynHi(lh)) if lh.0 == l => Some(lh.1.clone()),
                (h, Prov::Const(0)) | (Prov::Const(0), h) if pointerish(h) => Some(h.clone()),
                _ => None,
            };
            if let Some(h) = hi {
                st.vset(vdst, Prov::DynHi(Box::new((l, h))));
            }
        }
        _ => {}
    }
}

/// Transfer function of one instruction (records resources/loads in `plan`).
fn step(st: &mut State, ins: &Inst, code_base: u64, plan: &mut Plan) {
    // Operands of the tracked vector ops, read before the destination dies.
    let vsrc = |st: &State, op: u16| if op >= crate::OP_VGPR0 { st.vget(op - crate::OP_VGPR0) } else { src_operand(st, ins, op) };
    let (va, vb) = match ins.enc {
        Enc::Vop1 | Enc::Vop2 | Enc::Vop3 => (vsrc(st, ins.src(0)), vsrc(st, ins.src(1))),
        _ => (Prov::Unknown, Prov::Unknown),
    };
    // VGPR destinations lose their tracked values (memory loads: all of them).
    match ins.enc {
        Enc::Vop1 if !matches!(ins.op, 0 | 2) => {
            st.vset(ins.vdst(), Prov::Unknown);
            // (f64 results and conversions to f64 write a pair.)
            if matches!(ins.op, 4 | 16 | 22..=26 | 37..=41 | 47 | 48 | 61..=63) {
                st.vset(ins.vdst() + 1, Prov::Unknown);
            }
        }
        // (VOP2 results are one VGPR; v_readlane writes an SGPR.)
        Enc::Vop2 if ins.op != 1 => st.vset(ins.vdst(), Prov::Unknown),
        Enc::Vop3 if ins.op >= 0x100 && !matches!(ins.op, 0x101 | 0x182) => {
            st.vset(ins.vdst(), Prov::Unknown);
            // 64-bit results (f64 arithmetic, 64-bit shifts, v_mad_u64_u32,
            // the SAD ops; VOP1 f64 ops in VOP3 form) write a pair or more.
            let wide = match ins.op {
                0x14C | 0x15F..=0x168 | 0x16E | 0x170 | 0x172..=0x177 => 2,
                o if o >= 0x180 && matches!(o - 0x180, 4 | 16 | 22..=26 | 37..=41 | 47 | 48 | 61..=63) => 2,
                _ => 1,
            };
            for k in 1..(if ins.op == 0x175 { 4 } else { wide }) {
                st.vset(ins.vdst() + k, Prov::Unknown);
            }
        }
        Enc::Vintrp => st.vset(ins.vdst(), Prov::Unknown),
        // Loads (and atomics returning data) write at most 4 dwords, +1 for
        // TFE/LWE.
        Enc::Mubuf | Enc::Mtbuf | Enc::Mimg => {
            let d = ins.mem_vdata();
            st.v.retain(|(r, _)| *r < d || *r > d + 4);
        }
        Enc::Ds => {
            let d = ins.vdst();
            st.v.retain(|(r, _)| *r < d || *r > d + 4);
        }
        Enc::Flat => st.v.clear(),
        _ => {}
    }
    // SGPR spills through VGPR lanes.
    if let Some((write, vgpr, lane)) = ins.const_lane_op() {
        if write {
            let v = src_operand(st, ins, ins.src(0));
            st.lanes.retain(|(k, _)| *k != (vgpr, lane));
            st.lanes.push(((vgpr, lane), v));
        } else {
            let v = st.lanes.iter().find(|(k, _)| *k == (vgpr, lane)).map(|(_, p)| p.clone()).unwrap_or(Prov::Unknown);
            st.set(ins.vdst(), v);
        }
        return;
    }
    let ptr_add = st.ptr_add.take();
    {
        match ins.enc {
            Enc::Sop1 => {
                let d = ins.sdst();
                let s = ins.ssrc0();
                // Everything but moves and s_getpc_b64 writes SCC (or is not tracked).
                if !matches!(ins.op, 3 | 4 | 31) {
                    st.scc = Prov::Unknown;
                }
                match ins.op {
                    31 => {
                        let pc = code_base + (ins.pc as u64 & 0xF_FFFF) + 4;
                        let (lo, hi) = (Prov::PtrLo(Ptr::Abs(pc), 0), Prov::PtrHi(Ptr::Abs(pc), 0));
                        let l = plan.value(lo.clone());
                        let h = plan.value(hi.clone());
                        plan.inst_pc.insert(ins.pc, (l, h));
                        st.set(d, lo);
                        st.set(d + 1, hi);
                    }
                    3 => st.set(d, src_operand(st, ins, s)),
                    4 => {
                        let (lo, hi) = src_operand64(st, ins, s);
                        st.set(d, lo);
                        st.set(d + 1, hi);
                    }
                    // s_not_b32 / s_not_b64
                    7 => st.set(d, op(SOp::Xor, src_operand(st, ins, s), Prov::Const(u32::MAX))),
                    8 => {
                        let (lo, hi) = src_operand64(st, ins, s);
                        st.set(d, op(SOp::Xor, lo, Prov::Const(u32::MAX)));
                        st.set(d + 1, op(SOp::Xor, hi, Prov::Const(u32::MAX)));
                    }
                    // s_sext_i32_i8 / s_sext_i32_i16
                    25 => st.set(d, op(SOp::BfeI, src_operand(st, ins, s), Prov::Const(8 << 16))),
                    26 => st.set(d, op(SOp::BfeI, src_operand(st, ins, s), Prov::Const(16 << 16))),
                    _ => {
                        st.set(d, Prov::Unknown);
                        if crate::sop_is_64(ins.enc, ins.op) {
                            st.set(d + 1, Prov::Unknown);
                        }
                    }
                }
            }
            Enc::Sop2 => {
                let d = ins.sdst();
                let (s0, s1) = (ins.ssrc0(), ins.ssrc1());
                let a = src_operand(st, ins, s0);
                let b = src_operand(st, ins, s1);
                let bit32 = |o: SOp| Some(op(o, a.clone(), b.clone()));
                let result = match ins.op {
                    0 => {
                        // s_add_u32 lo, ptr_lo, const (either order): pointer
                        // arithmetic completed by the s_addc_u32 that follows.
                        let ptr = match (&a, &b) {
                            (_, Prov::Const(c)) if s0 < 127 => st.ptr(s0).map(|(p, bias)| (p, bias, *c, s0 + 1)),
                            (Prov::Const(c), _) if s1 < 127 => st.ptr(s1).map(|(p, bias)| (p, bias, *c, s1 + 1)),
                            _ => None,
                        };
                        let sum = op(SOp::Add, a.clone(), b.clone());
                        st.scc = op(SOp::Carry, a.clone(), b.clone());
                        match ptr {
                            Some((p, bias, c, hi_src)) if !matches!(p, Ptr::Pair(_)) => {
                                let bias = bias.wrapping_add(c);
                                st.ptr_add = Some(PtrAdd { hi_src, hi_dst: d + 1, ptr: p.clone(), bias });
                                Some(Prov::PtrLo(p, bias))
                            }
                            _ => Some(sum),
                        }
                    }
                    1 => {
                        st.scc = op(SOp::Borrow, a.clone(), b.clone());
                        bit32(SOp::Sub)
                    }
                    2 | 3 => {
                        st.scc = Prov::Unknown;
                        bit32(if ins.op == 2 { SOp::Add } else { SOp::Sub })
                    }
                    4 | 5 => {
                        // s_addc_u32 / s_subb_u32: the carry/borrow comes from SCC.
                        let scc = std::mem::replace(&mut st.scc, Prov::Unknown);
                        let matched = ins.op == 4
                            && ptr_add.as_ref().is_some_and(|pa| {
                                d == pa.hi_dst && ((s0 == pa.hi_src && b == Prov::Const(0)) || (s1 == pa.hi_src && a == Prov::Const(0)))
                            });
                        if matched {
                            let pa = ptr_add.unwrap();
                            Some(Prov::PtrHi(pa.ptr, pa.bias))
                        } else if ins.op == 4 {
                            Some(op(SOp::Add, op(SOp::Add, a.clone(), b.clone()), scc))
                        } else {
                            Some(op(SOp::Sub, op(SOp::Sub, a.clone(), b.clone()), scc))
                        }
                    }
                    6..=9 | 14 | 16 | 18 | 20 | 22 | 24 | 26 | 28 | 30 | 32 | 34 | 36 | 38 | 39 | 40 => {
                        // s_bfm_b32 and s_mul_i32 leave SCC alone.
                        if !matches!(ins.op, 36 | 38) {
                            st.scc = Prov::Unknown;
                        }
                        bit32(match ins.op {
                            6 => SOp::MinI,
                            7 => SOp::MinU,
                            8 => SOp::MaxI,
                            9 => SOp::MaxU,
                            14 => SOp::And,
                            16 => SOp::Or,
                            18 => SOp::Xor,
                            20 => SOp::AndN2,
                            22 => SOp::OrN2,
                            24 => SOp::Nand,
                            26 => SOp::Nor,
                            28 => SOp::Xnor,
                            30 => SOp::Shl,
                            32 => SOp::Shr,
                            34 => SOp::Ashr,
                            36 => SOp::Bfm,
                            38 => SOp::Mul,
                            39 => SOp::BfeU,
                            _ => SOp::BfeI,
                        })
                    }
                    15 | 17 | 19 | 21 | 23 | 25 | 27 | 29 => {
                        // 64-bit bitwise operations act on each half.
                        st.scc = Prov::Unknown;
                        let o = match ins.op {
                            15 => SOp::And,
                            17 => SOp::Or,
                            19 => SOp::Xor,
                            21 => SOp::AndN2,
                            23 => SOp::OrN2,
                            25 => SOp::Nand,
                            27 => SOp::Nor,
                            _ => SOp::Xnor,
                        };
                        let (alo, ahi) = src_operand64(st, ins, s0);
                        let (blo, bhi) = src_operand64(st, ins, s1);
                        st.set(d, op(o, alo, blo));
                        st.set(d + 1, op(o, ahi, bhi));
                        None
                    }
                    _ => {
                        // s_cselect (reads SCC), 64-bit shifts and fields, ...
                        if !matches!(ins.op, 10 | 11 | 37) {
                            st.scc = Prov::Unknown;
                        }
                        Some(Prov::Unknown)
                    }
                };
                if let Some(r) = result {
                    st.set(d, r);
                    if crate::sop_is_64(ins.enc, ins.op) {
                        st.set(d + 1, Prov::Unknown);
                    }
                }
            }
            Enc::Sopc => st.scc = Prov::Unknown,
            Enc::Sopk => {
                let d = ins.sdst();
                let k = Prov::Const(ins.simm16() as i16 as i32 as u32);
                match ins.op {
                    0 => st.set(d, k),
                    3..=14 => st.scc = Prov::Unknown,
                    // s_addk_i32 (SCC = overflow) / s_mulk_i32
                    15 => {
                        st.set(d, op(SOp::Add, st.get(d), k));
                        st.scc = Prov::Unknown;
                    }
                    16 => st.set(d, op(SOp::Mul, st.get(d), k)),
                    // s_cbranch_i_fork / s_setreg*: the field names a source.
                    17 | 19 | 21 => {}
                    _ => st.set(d, Prov::Unknown),
                }
            }
            Enc::Smrd => {
                let d = ins.sdst();
                let n = match ins.op & 7 {
                    0 => 1,
                    1 => 2,
                    2 => 4,
                    3 => 8,
                    4 => 16,
                    _ => 0,
                };
                if ins.op == 30 {
                    // s_memtime
                    st.set(d, Prov::Unknown);
                    st.set(d + 1, Prov::Unknown);
                }
                if ins.op >= 29 || n == 0 {
                    return;
                }
                let off = match ins.smrd_const_offset() {
                    Some(b) => Some(b),
                    None => match st.get(ins.smrd_offset() as u16) {
                        Prov::Const(c) => Some(c),
                        _ => None,
                    },
                };
                if ins.op < 8 {
                    // s_load_dword* through an address computed at run time
                    // from a known pointer (`DynLo`/`DynHi`): read in the
                    // shader through a window at the pointer, at the address's
                    // distance from it.
                    if let (Prov::DynLo(l), Prov::DynHi(lh)) = (st.get(ins.smrd_sbase()), st.get(ins.smrd_sbase() + 1)) {
                        if *l == lh.0 {
                            let dwords = vec![(*l).clone(), lh.1.clone(), Prov::Const(WINDOW_BYTES), Prov::Const(WINDOW_DWORD3)];
                            let r = plan.resource(Resource { kind: ResKind::Buffer, dwords, written: false, read: true, formatted: false });
                            plan.inst_res.insert(ins.pc, (r, None));
                            let vi = plan.value(*l);
                            plan.inst_dyn.insert(ins.pc, vi);
                            for k in 0..n {
                                st.set(d + k as u16, Prov::Unknown);
                            }
                            return;
                        }
                    }
                    // s_load_dword*: memory through a pointer.
                    match (st.ptr(ins.smrd_sbase()), off) {
                        (Some((p, bias)), Some(o)) => {
                            let o = o.wrapping_add(bias);
                            let idx = plan.loads.iter().position(|l| l.ptr == p && l.offset == o && l.dwords == n).unwrap_or_else(|| {
                                plan.loads.push(ScalarLoad { ptr: p.clone(), offset: o, dwords: n });
                                plan.loads.len() - 1
                            });
                            plan.inst_load.insert(ins.pc, idx);
                            for k in 0..n {
                                st.set(d + k as u16, Prov::Mem(p.clone(), o + 4 * k));
                            }
                        }
                        (Some((p, bias)), None) => {
                            // Offset computed at run time (an index into a
                            // table): read in the shader through a buffer
                            // window from the pointer on.
                            let dwords = vec![Prov::PtrLo(p.clone(), bias), Prov::PtrHi(p, bias), Prov::Const(WINDOW_BYTES), Prov::Const(WINDOW_DWORD3)];
                            let r = plan.resource(Resource { kind: ResKind::Buffer, dwords, written: false, read: true, formatted: false });
                            plan.inst_res.insert(ins.pc, (r, None));
                            for k in 0..n {
                                st.set(d + k as u16, Prov::Unknown);
                            }
                        }
                        _ => {
                            plan.errors.push(format!("{:#x}: s_load with unresolvable address", ins.pc));
                            for k in 0..n {
                                st.set(d + k as u16, Prov::Unknown);
                            }
                        }
                    }
                } else {
                    // s_buffer_load_dword*: V# in s[sbase:sbase+3].
                    let dwords = (0..4).map(|k| st.get(ins.smrd_sbase() + k)).collect();
                    let r = plan.resource(Resource { kind: ResKind::Buffer, dwords, written: false, read: true, formatted: false });
                    plan.inst_res.insert(ins.pc, (r, None));
                    for k in 0..n {
                        st.set(d + k as u16, Prov::Unknown);
                    }
                }
            }
            Enc::Mubuf | Enc::Mtbuf => {
                let base = ins.mem_srsrc();
                let dwords = (0..4).map(|k| st.get(base + k)).collect();
                let store = if ins.enc == Enc::Mubuf { matches!(ins.op, 4..=7 | 24..=31 | 48..=99) } else { ins.op >= 4 };
                let atomic = ins.enc == Enc::Mubuf && ins.op >= 48;
                let formatted = (ins.enc == Enc::Mubuf && ins.op <= 7) || ins.enc == Enc::Mtbuf;
                let r = plan.resource(Resource { kind: ResKind::Buffer, dwords, written: store, read: !store || atomic, formatted });
                plan.inst_res.insert(ins.pc, (r, None));
            }
            Enc::Mimg => {
                let base = ins.mem_srsrc();
                let dwords = (0..8).map(|k| st.get(base + k)).collect();
                let store = matches!(ins.op, 8..=11 | 15..=28);
                let img = plan.resource(Resource { kind: ResKind::Image, dwords, written: store, read: !store, formatted: false });
                let samp = if ins.op >= 32 {
                    let sb = ins.mimg_ssamp();
                    let sd = (0..4).map(|k| st.get(sb + k)).collect();
                    Some(plan.resource(Resource { kind: ResKind::Sampler, dwords: sd, written: false, read: false, formatted: false }))
                } else {
                    None
                };
                plan.inst_res.insert(ins.pc, (img, samp));
            }
            // v_readfirstlane to SGPR: tracked values carry over (copies of
            // scalar provenance, run-time pointer halves).
            Enc::Vop1 if ins.op == 2 => st.set(ins.vdst(), va.clone()),
            // v_mov_b32: scalar provenance and tracked values move into VGPRs.
            Enc::Vop1 if ins.op == 1 => st.vset(ins.vdst(), va.clone()),
            Enc::Vop3 => {
                // VOPC in VOP3 form and VOP3b write SGPRs.
                if ins.op < 0x100 {
                    let d = ins.vdst();
                    st.set(d, Prov::Unknown);
                    st.set(d + 1, Prov::Unknown);
                } else if ins.is_vop3b() {
                    let d = ins.sdst();
                    st.set(d, Prov::Unknown);
                    st.set(d + 1, Prov::Unknown);
                } else if ins.op == 0x180 + 2 || ins.op == 0x101 {
                    st.set(ins.vdst(), Prov::Unknown);
                }
            }
            Enc::Vopc => {
                st.set(crate::OP_VCC_LO, Prov::Unknown);
                st.set(crate::OP_VCC_HI, Prov::Unknown);
            }
            Enc::Vop2 if ins.op == 1 => st.set(ins.vdst(), Prov::Unknown), // v_readlane
            Enc::Vop2 if (37..=42).contains(&ins.op) => {
                // v_add_i32 .. v_subbrev_u32 write their carry to VCC.
                let carry_in = st.vcc_lo_of.clone();
                st.set(crate::OP_VCC_LO, Prov::Unknown);
                st.set(crate::OP_VCC_HI, Prov::Unknown);
                vector_ptr_add(st, ins.op, ins.vdst(), va, vb, carry_in);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A register pair clobbered on one branch stays a user-data pointer on
    /// the other (pattern from Bloodborne vertex shaders).
    #[test]
    fn branch_local_clobber_does_not_leak() {
        let code = [
            0xBF00_8004, // 0x00 s_cmp_eq_i32 s4, 0
            0xBF85_0002, // 0x04 s_cbranch_scc1 0x10
            0xC2C0_0900, // 0x08 s_buffer_load_dwordx8 s[0:7], s[8:11], 0x0
            0xBF82_0002, // 0x0c s_branch 0x18
            0xC086_070C, // 0x10 s_load_dwordx4 s[12:15], s[6:7], 0x30
            0xBF81_0000, // 0x14 s_endpgm
            0xBF81_0000, // 0x18 s_endpgm
        ];
        let p = crate::decode_program(&code).unwrap();
        let plan = analyze(&p, 12, 0);
        assert!(plan.errors.is_empty(), "{:?}", plan.errors);
        assert!(plan.loads.iter().any(|l| l.ptr == Ptr::User(6) && l.offset == 0x30 && l.dwords == 4));
    }

    /// A record address computed in VGPRs from a loaded table pointer and
    /// moved back with v_readfirstlane (SOTC's tiled light loop, cs
    /// 0x105fcedd00): read through a window at the pointer.
    #[test]
    fn readfirstlane_pointer_load() {
        let code = [
            0xC08A_0100, // s_load_dwordx4 s[20:23], s[0:1], 0x0
            0x4A0A_0016, // v_add_i32 v5, s22, v0
            0x7E00_0217, // v_mov_b32 v0, s23
            0x500C_0080, // v_addc_u32 v6, 0, v0
            0x7E12_0506, // v_readfirstlane_b32 s9, v6
            0x7E10_0505, // v_readfirstlane_b32 s8, v5
            0xC086_0900, // s_load_dwordx4 s[12:15], s[8:9], 0x0
            0xBF81_0000, // s_endpgm
        ];
        let p = crate::decode_program(&code).unwrap();
        let plan = analyze(&p, 2, 0);
        assert!(plan.errors.is_empty(), "{:?}", plan.errors);
        let lo = Prov::Mem(Ptr::User(0), 8);
        assert_eq!(plan.values, vec![lo.clone()]);
        let pc = p[6].pc;
        assert_eq!(plan.inst_dyn.get(&pc), Some(&0));
        let (r, _) = plan.inst_res[&pc];
        assert_eq!(plan.resources[r].dwords, vec![lo, Prov::Mem(Ptr::User(0), 12), Prov::Const(WINDOW_BYTES), Prov::Const(WINDOW_DWORD3)]);
        // Two addresses from one copy of the high dword (cs 0x105fca3f00).
        let code3 = [
            0xC092_0100, // s_load_dwordx4 s[36:39], s[0:1], 0x0
            0x7E0A_0227, // v_mov_b32 v5, s39
            0xD282_0004, 0x040E_0501, // v_mad_f32 v4, v1, v2, v3 (writes v4 only)
            0x4A04_0426, // v_add_i32 v2, s38, v2
            0x5008_0A80, // v_addc_u32 v4, 0, v5
            0x4A02_0226, // v_add_i32 v1, s38, v1
            0x500C_0A80, // v_addc_u32 v6, 0, v5
            0x7E06_0506, // v_readfirstlane_b32 s3, v6
            0x7E04_0501, // v_readfirstlane_b32 s2, v1
            0xC006_0300, // s_load_dword s12, s[2:3], 0x0
            0xBF81_0000, // s_endpgm
        ];
        let p3 = crate::decode_program(&code3).unwrap();
        let plan3 = analyze(&p3, 2, 0);
        assert!(plan3.errors.is_empty(), "{:?}", plan3.errors);
        assert_eq!(plan3.inst_dyn.get(&p3[9].pc), Some(&0));
        assert_eq!(plan3.values, vec![Prov::Mem(Ptr::User(0), 8)]);
        // A constant added to the pointer instead of the run-time value: not
        // a pointer pattern (0x50 + v1 is just an offset).
        let code2 = [
            0x4A0A_02FF, 0x0000_0050, // v_add_i32 v5, 0x50, v1
            0x7E10_0505, // v_readfirstlane_b32 s8, v5
            0xBF81_0000,
        ];
        let p2 = crate::decode_program(&code2).unwrap();
        let plan2 = analyze(&p2, 2, 0);
        assert!(plan2.values.is_empty());
    }

    /// A V# spilled into VGPR lanes (v_writelane) and restored (v_readlane)
    /// keeps its provenance (pattern from Bloodborne's culling shaders).
    #[test]
    fn sgpr_spill_through_vgpr_lanes() {
        let code = [
            0x0405_0004, // v_writelane_b32 v2, s4, 0
            0x0405_0205, // v_writelane_b32 v2, s5, 1
            0x0405_0406, // v_writelane_b32 v2, s6, 2
            0x0405_0607, // v_writelane_b32 v2, s7, 3
            0x0221_0102, // v_readlane_b32 s16, v2, 0
            0x0223_0302, // v_readlane_b32 s17, v2, 1
            0x0225_0502, // v_readlane_b32 s18, v2, 2
            0x0227_0702, // v_readlane_b32 s19, v2, 3
            0xC20A_1100, // s_buffer_load_dword s20, s[16:19], 0x0
            0xBF81_0000, // s_endpgm
        ];
        let p = crate::decode_program(&code).unwrap();
        assert_eq!(p[0].const_lane_op(), Some((true, 2, 0)));
        assert_eq!(p[7].const_lane_op(), Some((false, 2, 3)));
        let plan = analyze(&p, 8, 0);
        assert!(plan.errors.is_empty(), "{:?}", plan.errors);
        assert_eq!(plan.resources[0].dwords, vec![Prov::User(4), Prov::User(5), Prov::User(6), Prov::User(7)]);
    }

    /// A V# whose stride is or'ed into the loaded table pointer (SOTC compute
    /// shaders) resolves to an expression.
    #[test]
    fn descriptor_built_with_salu() {
        let code = [
            0xC08C_0100, // s_load_dwordx4 s[24:27], s[0:1], 0x0
            0x8821_FF19, // s_or_b32 s33, s25, 0x400000
            0x0040_0000,
            0xBEA0_0318, // s_mov_b32 s32, s24
            0xBEA2_03C1, // s_mov_b32 s34, -1
            0xBEA3_03FF, // s_mov_b32 s35, 0x2000c004
            0x2000_C004,
            0xE030_2000, // buffer_load_dword v0, v0, s[32:35], 0 idxen
            0x8008_0000,
            0xBF81_0000, // s_endpgm
        ];
        let p = crate::decode_program(&code).unwrap();
        let plan = analyze(&p, 2, 0);
        assert!(plan.errors.is_empty(), "{:?}", plan.errors);
        let m = |o| Prov::Mem(Ptr::User(0), o);
        assert_eq!(
            plan.resources[0].dwords,
            vec![m(0), Prov::Op(SOp::Or, Box::new((m(4), Prov::Const(0x40_0000)))), Prov::Const(u32::MAX), Prov::Const(0x2000_C004)]
        );
    }

    /// 64-bit pointer + constant with the constant as the first operand.
    #[test]
    fn pointer_add_constant_first() {
        let code = [
            0xC0C2_0100, // s_load_dwordx8 s[4:11], s[0:1], 0x0
            0x8004_04FF, // s_add_u32 s4, 0x244, s4
            0x0000_0244,
            0x8205_0580, // s_addc_u32 s5, 0, s5
            0xC006_0500, // s_load_dword s12, s[4:5], 0x0
            0xBF81_0000, // s_endpgm
        ];
        let p = crate::decode_program(&code).unwrap();
        let plan = analyze(&p, 2, 0);
        assert!(plan.errors.is_empty(), "{:?}", plan.errors);
        assert!(plan.loads.iter().any(|l| l.ptr == Ptr::Mem(Box::new(Ptr::User(0)), 0) && l.offset == 0x244 && l.dwords == 1), "{:?}", plan.loads);
    }

    /// A scalar load at a run-time offset reads through a buffer window.
    #[test]
    fn dynamic_offset_load_uses_window() {
        let code = [
            0xC082_0003, // s_load_dwordx4 s[4:7], s[0:1], s3
            0xBF81_0000, // s_endpgm
        ];
        let p = crate::decode_program(&code).unwrap();
        let plan = analyze(&p, 2, 0);
        assert!(plan.errors.is_empty(), "{:?}", plan.errors);
        let (r, _) = plan.inst_res[&0];
        assert_eq!(
            plan.resources[r].dwords,
            vec![Prov::PtrLo(Ptr::User(0), 0), Prov::PtrHi(Ptr::User(0), 0), Prov::Const(WINDOW_BYTES), Prov::Const(WINDOW_DWORD3)]
        );
    }

    #[test]
    fn sop_semantics() {
        assert_eq!(SOp::BfeU.apply(0xABCD_1234, (8 << 16) | 4), 0x23);
        assert_eq!(SOp::BfeI.apply(0xF0, (4 << 16) | 4), u32::MAX);
        assert_eq!(SOp::Bfm.apply(4, 8), 0xF00);
        assert_eq!(SOp::Carry.apply(u32::MAX, 1), 1);
        assert_eq!(SOp::Borrow.apply(0, 1), 1);
        // Pointer arithmetic through SCC: (lo, hi) + 0x10 with carry.
        let lo = op(SOp::Add, Prov::User(0), Prov::Const(0x10));
        let c = op(SOp::Carry, Prov::User(0), Prov::Const(0x10));
        let hi = op(SOp::Add, Prov::User(1), c);
        assert!(matches!(hi, Prov::Op(SOp::Add, _)) && matches!(lo, Prov::Op(SOp::Add, _)));
    }

    /// Values that differ between paths become unknown at the join.
    #[test]
    fn join_merges_to_unknown() {
        let code = [
            0xBF00_8004, // 0x00 s_cmp_eq_i32 s4, 0
            0xBF85_0001, // 0x04 s_cbranch_scc1 0x0c
            0xC2C0_0900, // 0x08 s_buffer_load_dwordx8 s[0:7], s[8:11], 0x0
            0xC086_070C, // 0x0c s_load_dwordx4 s[12:15], s[6:7], 0x30 (s6/s7 differ by path)
            0xBF81_0000, // 0x10 s_endpgm
        ];
        let p = crate::decode_program(&code).unwrap();
        let plan = analyze(&p, 12, 0);
        assert!(plan.errors.iter().any(|e| e.contains("0xc: s_load with unresolvable address")), "{:?}", plan.errors);
    }
}
