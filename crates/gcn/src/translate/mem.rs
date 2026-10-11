//! Memory, image, interpolation, export and stage-interface semantics.

use super::{ImageDim, NumClass, R, ResSpec, Stage, T, e};
use crate::Inst;
use rspirv::dr::Operand;
use rspirv::spirv::{self, BuiltIn, Capability, Decoration, GlslStd450Op as GLOp, ImageOperands, StorageClass, Word};

// ---------------------------------------------------------------------------
// Stage inputs

/// PS input VGPR counts per SPI_PS_INPUT_ADDR bit.
const PS_INPUT_VGPRS: [u32; 16] = [2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1];

pub fn stage_inputs(tr: &mut T) -> R<()> {
    if tr.spec.tess.role != super::TessRole::None {
        return tess_inputs(tr);
    }
    match tr.spec.stage {
        Stage::Vertex => {
            let vid = tr.input(tr.t.i32, Some(BuiltIn::VertexIndex), None);
            let iid = tr.input(tr.t.i32, Some(BuiltIn::InstanceIndex), None);
            let pi = tr.b.type_pointer(None, StorageClass::Input, tr.t.i32);
            let _ = pi;
            let v = e(tr.b.load(tr.t.i32, None, vid, None, []))?;
            let v = tr.to_u(v)?;
            tr.set_vreg_raw(0, v)?;
            // v1/v2: instance id divided by the step rates; v3: instance id.
            let i = e(tr.b.load(tr.t.i32, None, iid, None, []))?;
            let i = tr.to_u(i)?;
            for (k, rate) in tr.spec.vs_step_rates.into_iter().enumerate() {
                let v = if rate <= 1 {
                    i
                } else {
                    let r = tr.u(rate);
                    e(tr.b.u_div(tr.t.u32, None, i, r))?
                };
                tr.set_vreg_raw(1 + k as u32, v)?;
            }
            tr.set_vreg_raw(3, i)?;
        }
        Stage::Fragment => {
            let fc = tr.input(tr.t.v4f, Some(BuiltIn::FragCoord), None);
            tr.frag_coord = Some(fc);
            let ff = tr.input(tr.t.bool_, Some(BuiltIn::FrontFacing), None);
            let fcv = e(tr.b.load(tr.t.v4f, None, fc, None, []))?;
            let mut vg = 0u32;
            let addr = if tr.spec.ps_input_addr == 0 { 0x2 } else { tr.spec.ps_input_addr };
            for bit in 0..16 {
                if addr & (1 << bit) == 0 {
                    continue;
                }
                let n = PS_INPUT_VGPRS[bit];
                for k in 0..n {
                    let val = match bit {
                        8..=11 => {
                            let c = e(tr.b.composite_extract(tr.t.f32, None, fcv, [(bit - 8) as u32]))?;
                            tr.to_u(c)?
                        }
                        12 => {
                            let f = e(tr.b.load(tr.t.bool_, None, ff, None, []))?;
                            let one = tr.f(1.0);
                            let m1 = tr.f(-1.0);
                            let s = e(tr.b.select(tr.t.f32, None, f, one, m1))?;
                            tr.to_u(s)?
                        }
                        _ => tr.u(0),
                    };
                    tr.set_vreg_raw(vg + k, val)?;
                }
                vg += n;
            }
        }
        Stage::Compute => {
            let lid = tr.input(tr.t.v3u, Some(BuiltIn::LocalInvocationId), None);
            let wid = tr.input(tr.t.v3u, Some(BuiltIn::WorkgroupId), None);
            let lidx = tr.input(tr.t.u32, Some(BuiltIn::LocalInvocationIndex), None);
            let l = e(tr.b.load(tr.t.v3u, None, lid, None, []))?;
            let tid_cnt = (tr.spec.cs_rsrc2 >> 11) & 3;
            for k in 0..=tid_cnt.min(2) {
                let c = e(tr.b.composite_extract(tr.t.u32, None, l, [k]))?;
                tr.set_vreg_raw(k, c)?;
            }
            let w = e(tr.b.load(tr.t.v3u, None, wid, None, []))?;
            let mut s = tr.spec.user_sgprs as u32;
            for k in 0..3 {
                if tr.spec.cs_rsrc2 & (1 << (7 + k)) != 0 {
                    let c = e(tr.b.composite_extract(tr.t.u32, None, w, [k]))?;
                    tr.set_sreg(s, c)?;
                    s += 1;
                }
            }
            if tr.spec.cs_rsrc2 & (1 << 10) != 0 {
                // TG_SIZE: wave id in group [5:0], wave count [11:6] (single-lane waves: report 1 wave)
                let v = tr.u(1 << 6);
                tr.set_sreg(s, v)?;
            }
            let idx = e(tr.b.load(tr.t.u32, None, lidx, None, []))?;
            tr.local_index = Some(idx);
            let k63 = tr.u(63);
            tr.lane_id = Some(e(tr.b.bitwise_and(tr.t.u32, None, idx, k63))?);
            let lds_dwords = ((tr.spec.cs_rsrc2 >> 15) & 0x1FF) * 128;
            if lds_dwords > 0 {
                let n = lds_dwords.min(12288);
                if n < lds_dwords {
                    tr.warn(format!("LDS of {} dwords clamped to {n}", lds_dwords));
                }
                let len = tr.u(n);
                let arr = tr.b.type_array(tr.t.u32, len);
                let pt = tr.b.type_pointer(None, StorageClass::Workgroup, arr);
                let v = tr.global_var(pt, StorageClass::Workgroup);
                tr.lds = Some(v);
                tr.lds_len = n;
            }
        }
    }
    Ok(())
}

/// Initial state of the tessellation stages (see `TessRole`), as the
/// hardware sets it up per threadgroup of `patches` patches:
/// - LS (compute, one invocation per control point i): v0 vertex id, v1 the
///   control point's index in its threadgroup, v2 instance id 0;
/// - HS (compute, one workgroup per threadgroup, an invocation per output
///   control point): v0 patch id, v1 relative patch id [7:0] | control point
///   [12:8]; after the user SGPRs the off-chip offset (OC_LDS_EN, 0) and the
///   threadgroup's factor offset in the factor ring (dense, `tf_dwords` per
///   patch);
/// - DS (tessellation evaluation): v0/v1 the domain coordinates, v2 relative
///   patch id, v3 patch id.
/// Invocations past the draw's control points (LS) or patches (HS) run with
/// EXEC clear, so the threadgroup's barriers still see every invocation.
fn tess_inputs(tr: &mut T) -> R<()> {
    use super::TessRole;
    let ts = tr.spec.tess;
    let u = tr.t.u32;
    let info = super::tess_info_offset(tr.plan);
    let lds_dw = tr.u(ts.lds_dwords);
    let np = tr.u(ts.patches.max(1));
    let tg = match ts.role {
        TessRole::Ls => {
            let gid = tr.input(tr.t.v3u, Some(BuiltIn::GlobalInvocationId), None);
            let g = e(tr.b.load(tr.t.v3u, None, gid, None, []))?;
            let i = e(tr.b.composite_extract(u, None, g, [0]))?;
            let k63 = tr.u(63);
            tr.lane_id = Some(e(tr.b.bitwise_and(u, None, i, k63))?);
            let n = tr.sgpr_data(info)?;
            let inside = e(tr.b.u_less_than(tr.t.bool_, None, i, n))?;
            let one = tr.u(1);
            let zero = tr.u(0);
            let exec = e(tr.b.select(u, None, inside, one, zero))?;
            tr.set_sreg(crate::OP_EXEC_LO as u32, exec)?;
            let k1 = tr.u(info + 1);
            let at = e(tr.b.i_add(u, None, k1, i))?;
            let z = tr.u(0);
            let p = e(tr.b.access_chain(tr.t.sb_u32, None, tr.sgpr_data, [z, at]))?;
            let vid = e(tr.b.load(u, None, p, None, []))?;
            tr.set_vreg_raw(0, vid)?;
            let per_tg = tr.u((ts.patches * ts.in_cp).max(1));
            let rel = e(tr.b.u_mod(u, None, i, per_tg))?;
            tr.set_vreg_raw(1, rel)?;
            tr.set_vreg_raw(2, zero)?;
            tr.set_vreg_raw(3, zero)?;
            e(tr.b.u_div(u, None, i, per_tg))?
        }
        TessRole::Hs => {
            let wid = tr.input(tr.t.v3u, Some(BuiltIn::WorkgroupId), None);
            let w = e(tr.b.load(tr.t.v3u, None, wid, None, []))?;
            let tg = e(tr.b.composite_extract(u, None, w, [0]))?;
            let lidx = tr.input(tr.t.u32, Some(BuiltIn::LocalInvocationIndex), None);
            let l = e(tr.b.load(u, None, lidx, None, []))?;
            let k63 = tr.u(63);
            tr.lane_id = Some(e(tr.b.bitwise_and(u, None, l, k63))?);
            let ocp = tr.u(ts.out_cp.max(1));
            let rel = e(tr.b.u_div(u, None, l, ocp))?;
            let cp = e(tr.b.u_mod(u, None, l, ocp))?;
            let first = e(tr.b.i_mul(u, None, tg, np))?;
            let patch = e(tr.b.i_add(u, None, first, rel))?;
            let n = tr.sgpr_data(info)?;
            let inside = e(tr.b.u_less_than(tr.t.bool_, None, patch, n))?;
            let one = tr.u(1);
            let zero = tr.u(0);
            let exec = e(tr.b.select(u, None, inside, one, zero))?;
            tr.set_sreg(crate::OP_EXEC_LO as u32, exec)?;
            tr.set_vreg_raw(0, patch)?;
            let eight = tr.u(8);
            let cps = e(tr.b.shift_left_logical(u, None, cp, eight))?;
            let ids = e(tr.b.bitwise_or(u, None, rel, cps))?;
            tr.set_vreg_raw(1, ids)?;
            let mut s = tr.spec.user_sgprs as u32;
            if ts.rsrc2 & (1 << 7) != 0 {
                tr.set_sreg(s, zero)?;
                s += 1;
            }
            let tf_bytes = tr.u(ts.patches * super::tf_dwords(ts.tf_param) * 4);
            let tf_off = e(tr.b.i_mul(u, None, tg, tf_bytes))?;
            tr.set_sreg(s, tf_off)?;
            s += 1;
            if ts.rsrc2 & (1 << 8) != 0 {
                // TG_SIZE: wave id [5:0], wave count [11:6] (one-lane waves).
                let v = tr.u(1 << 6);
                tr.set_sreg(s, v)?;
            }
            tg
        }
        TessRole::Ds | TessRole::None => {
            let tc = tr.input(tr.t.v3f, Some(BuiltIn::TessCoord), None);
            let pid = tr.input(tr.t.i32, Some(BuiltIn::PrimitiveId), None);
            tr.b.capability(spirv::Capability::Tessellation);
            let c = e(tr.b.load(tr.t.v3f, None, tc, None, []))?;
            for k in 0..2 {
                let x = e(tr.b.composite_extract(tr.t.f32, None, c, [k]))?;
                let bits = tr.to_u(x)?;
                tr.set_vreg_raw(k, bits)?;
            }
            let p = e(tr.b.load(tr.t.i32, None, pid, None, []))?;
            let patch = tr.to_u(p)?;
            let rel = e(tr.b.u_mod(u, None, patch, np))?;
            tr.set_vreg_raw(2, rel)?;
            tr.set_vreg_raw(3, patch)?;
            e(tr.b.u_div(u, None, patch, np))?
        }
    };
    let v = tr.sb_var(false);
    let bnd = tr.alloc_binding(super::BindingKind::Lds);
    tr.decorate_binding(v, bnd);
    tr.lds = Some(v);
    tr.lds_base = Some(e(tr.b.i_mul(u, None, tg, lds_dw))?);
    Ok(())
}

// ---------------------------------------------------------------------------
// Scalar memory

pub fn smrd(tr: &mut T, ins: &Inst) -> R<()> {
    let n = match ins.op & 7 {
        0 => 1,
        1 => 2,
        2 => 4,
        3 => 8,
        4 => 16,
        _ => 0,
    };
    if ins.op >= 29 {
        if ins.op == 30 {
            let z = tr.u(0);
            tr.set_sreg(ins.sdst() as u32, z)?;
            tr.set_sreg(ins.sdst() as u32 + 1, z)?;
        }
        return Ok(());
    }
    let d = ins.sdst() as u32;
    if ins.op < 8 {
        if let Some(&k) = tr.plan.inst_load.get(&ins.pc) {
            let base = tr.load_offsets[k];
            for j in 0..n {
                let v = tr.sgpr_data(base + j)?;
                tr.set_sreg(d + j, v)?;
            }
            return Ok(());
        }
        if !tr.plan.inst_res.contains_key(&ins.pc) {
            tr.warn(format!("{:#x}: s_load without resolved address", ins.pc));
            return Ok(());
        }
        // Run-time offset: read through the buffer window the analysis set
        // up at the pointer (below, like s_buffer_load).
    }
    let Some(&(r, _)) = tr.plan.inst_res.get(&ins.pc) else { return Err(format!("{:#x}: s_buffer_load without resource", ins.pc)) };
    let var = tr.buffers[&r];
    let byte_off = match ins.smrd_const_offset() {
        Some(b) => tr.u(b),
        None => tr.sreg(ins.smrd_offset())?,
    };
    // An address computed at run time from the window's pointer: its
    // distance from the pointer (low dwords) is the offset into the window.
    let byte_off = match tr.plan.inst_dyn.get(&ins.pc) {
        Some(&vi) => {
            let lo = tr.sreg(ins.smrd_sbase() as u32)?;
            let base = tr.sgpr_data(super::values_offset(tr.plan) + vi as u32)?;
            let dist = e(tr.b.i_sub(tr.t.u32, None, lo, base))?;
            e(tr.b.i_add(tr.t.u32, None, byte_off, dist))?
        }
        None => byte_off,
    };
    let byte_off = match tr.buf_bias.get(&r) {
        Some(&b) => e(tr.b.i_add(tr.t.u32, None, byte_off, b))?,
        None => byte_off,
    };
    let two = tr.u(2);
    let idx0 = e(tr.b.shift_right_logical(tr.t.u32, None, byte_off, two))?;
    for j in 0..n {
        let kj = tr.u(j);
        let idx = e(tr.b.i_add(tr.t.u32, None, idx0, kj))?;
        let v = load_dword(tr, var, idx)?;
        tr.set_sreg(d + j, v)?;
    }
    Ok(())
}

fn load_dword(tr: &mut T, var: Word, idx: Word) -> R<Word> {
    let z = tr.u(0);
    let p = e(tr.b.access_chain(tr.t.sb_u32, None, var, [z, idx]))?;
    e(tr.b.load(tr.t.u32, None, p, None, []))
}

fn store_dword(tr: &mut T, var: Word, idx: Word, v: Word) -> R<()> {
    let z = tr.u(0);
    let p = e(tr.b.access_chain(tr.t.sb_u32, None, var, [z, idx]))?;
    tr.b.store(p, v, None, []).map_err(|x| format!("{x:?}"))
}

/// Loads `nbytes` (1..=4) little-endian bytes at an arbitrary byte address.
fn load_bytes(tr: &mut T, var: Word, addr: Word, nbytes: u32) -> R<Word> {
    let u = tr.t.u32;
    let two = tr.u(2);
    let idx = e(tr.b.shift_right_logical(u, None, addr, two))?;
    let lo = load_dword(tr, var, idx)?;
    let one = tr.u(1);
    let idx1 = e(tr.b.i_add(u, None, idx, one))?;
    let hi = load_dword(tr, var, idx1)?;
    let wide = {
        let v = e(tr.b.composite_construct(tr.t.v2u, None, [lo, hi]))?;
        e(tr.b.bitcast(tr.t.u64, None, v))?
    };
    let k3 = tr.u(3);
    let sub = e(tr.b.bitwise_and(u, None, addr, k3))?;
    let sh = e(tr.b.shift_left_logical(u, None, sub, k3))?;
    let shifted = e(tr.b.shift_right_logical(tr.t.u64, None, wide, sh))?;
    let r = e(tr.b.u_convert(u, None, shifted))?;
    if nbytes >= 4 {
        return Ok(r);
    }
    let m = tr.u((1u32 << (nbytes * 8)) - 1);
    e(tr.b.bitwise_and(u, None, r, m))
}

// ---------------------------------------------------------------------------
// Attribute interpolation

fn attr_input(tr: &mut T, attr: u32) -> R<Option<Word>> {
    let spec_in = tr.spec.ps_inputs.get(attr as usize).copied();
    if let Some(pi) = spec_in {
        if pi.default.is_some() {
            return Ok(None);
        }
    }
    if let Some(&v) = tr.in_attrs.get(&attr) {
        return Ok(Some(v));
    }
    let (loc, flat) = match spec_in {
        Some(pi) => (pi.param as u32, pi.flat),
        None => (attr, false),
    };
    let v = tr.input(tr.t.v4f, None, Some(loc));
    if flat {
        tr.b.decorate(v, Decoration::Flat, []);
    }
    tr.in_attrs.insert(attr, v);
    tr.params = tr.params.max(attr + 1);
    Ok(Some(v))
}

pub fn interp(tr: &mut T, ins: &Inst) -> R<()> {
    let attr = ins.interp_attr();
    let chan = ins.interp_chan();
    let val = match attr_input(tr, attr)? {
        Some(var) => {
            let v = e(tr.b.load(tr.t.v4f, None, var, None, []))?;
            e(tr.b.composite_extract(tr.t.f32, None, v, [chan]))?
        }
        None => {
            let d = tr.spec.ps_inputs[attr as usize].default.unwrap_or(0);
            let c = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]][d as usize & 3][chan as usize];
            tr.f(c)
        }
    };
    let u = tr.to_u(val)?;
    tr.set_vreg(ins.vdst() as u32, u)
}

// ---------------------------------------------------------------------------
// Buffer formats

/// Bit layout (offset, width) of up to 4 components for a buffer data format.
fn dfmt_layout(dfmt: u8) -> Option<(&'static [(u32, u32)], u32)> {
    Some(match dfmt {
        1 => (&[(0, 8)], 1),
        2 => (&[(0, 16)], 2),
        3 => (&[(0, 8), (8, 8)], 2),
        4 => (&[(0, 32)], 4),
        5 => (&[(0, 16), (16, 16)], 4),
        6 => (&[(0, 11), (11, 11), (22, 10)], 4),
        7 => (&[(0, 10), (10, 11), (21, 11)], 4),
        8 => (&[(0, 2), (2, 10), (12, 10), (22, 10)], 4),
        9 => (&[(0, 10), (10, 10), (20, 10), (30, 2)], 4),
        10 => (&[(0, 8), (8, 8), (16, 8), (24, 8)], 4),
        11 => (&[(0, 32), (32, 32)], 8),
        12 => (&[(0, 16), (16, 16), (32, 16), (48, 16)], 8),
        13 => (&[(0, 32), (32, 32), (64, 32)], 12),
        14 => (&[(0, 32), (32, 32), (64, 32), (96, 32)], 16),
        _ => return None,
    })
}

fn float_class(nfmt: u8) -> bool {
    !matches!(nfmt, 4 | 5)
}

/// Converts a raw component (already extracted, `bits` wide) to a 32-bit register value.
fn convert_component(tr: &mut T, raw: Word, bits: u32, nfmt: u8, dfmt: u8) -> R<Word> {
    let u = tr.t.u32;
    let f = tr.t.f32;
    let z = tr.u(0);
    let w = tr.u(bits);
    let sext = |tr: &mut T| -> R<Word> {
        if bits == 32 { Ok(raw) } else { e(tr.b.bit_field_s_extract(u, None, raw, z, w)) }
    };
    Ok(match nfmt {
        0 => {
            let x = e(tr.b.convert_u_to_f(f, None, raw))?;
            let k = tr.f(((1u64 << bits) - 1) as f32);
            let r = e(tr.b.f_div(f, None, x, k))?;
            tr.to_u(r)?
        }
        1 | 6 => {
            let s = sext(tr)?;
            let si = tr.to_i(s)?;
            let x = e(tr.b.convert_s_to_f(f, None, si))?;
            let k = tr.f(((1u64 << (bits - 1)) - 1) as f32);
            let r = e(tr.b.f_div(f, None, x, k))?;
            let m1 = tr.f(-1.0);
            let r = tr.glsl2(f, GLOp::FMax, r, m1)?;
            tr.to_u(r)?
        }
        2 => {
            let x = e(tr.b.convert_u_to_f(f, None, raw))?;
            tr.to_u(x)?
        }
        3 => {
            let s = sext(tr)?;
            let si = tr.to_i(s)?;
            let x = e(tr.b.convert_s_to_f(f, None, si))?;
            tr.to_u(x)?
        }
        4 => raw,
        5 => sext(tr)?,
        _ => match bits {
            32 => raw,
            16 => {
                let v = tr.glsl1(tr.t.v2f, GLOp::UnpackHalf2x16, raw)?;
                let x = e(tr.b.composite_extract(f, None, v, [0]))?;
                tr.to_u(x)?
            }
            11 | 10 => {
                // Unsigned small float (5-bit exponent, 6/5-bit mantissa) -> f32 bits.
                let mbits = bits - 5;
                let mmask = tr.u((1 << mbits) - 1);
                let man = e(tr.b.bitwise_and(u, None, raw, mmask))?;
                let ms = tr.u(mbits);
                let ex = e(tr.b.shift_right_logical(u, None, raw, ms))?;
                // f16-like rebias: build an f16 then convert (exponent bias 15 in both).
                let sh = tr.u(10 - mbits);
                let man16 = e(tr.b.shift_left_logical(u, None, man, sh))?;
                let k10 = tr.u(10);
                let ex16 = e(tr.b.shift_left_logical(u, None, ex, k10))?;
                let h = e(tr.b.bitwise_or(u, None, ex16, man16))?;
                let v = tr.glsl1(tr.t.v2f, GLOp::UnpackHalf2x16, h)?;
                let x = e(tr.b.composite_extract(f, None, v, [0]))?;
                let _ = dfmt;
                tr.to_u(x)?
            }
            _ => {
                tr.warn(format!("float buffer component of {bits} bits unsupported"));
                raw
            }
        },
    })
}

/// Loads and converts a formatted element: returns 4 register values.
fn load_formatted(tr: &mut T, var: Word, addr: Word, dfmt: u8, nfmt: u8) -> R<[Word; 4]> {
    let one_f = tr.u(1.0f32.to_bits());
    let one_i = tr.u(1);
    let z = tr.u(0);
    let def = [z, z, z, if float_class(nfmt) { one_f } else { one_i }];
    let Some((layout, size)) = dfmt_layout(dfmt) else {
        if dfmt != 0 {
            tr.warn(format!("buffer data format {dfmt} unsupported"));
        }
        // BUF_DATA_FORMAT_INVALID reads return zero.
        let z = tr.u(0);
        return Ok([z, z, z, z]);
    };
    let u = tr.t.u32;
    // Load the element's dwords.
    let ndw = size.div_ceil(4);
    let mut dws = Vec::new();
    for k in 0..ndw {
        let off = tr.u(k * 4);
        let a = e(tr.b.i_add(u, None, addr, off))?;
        dws.push(load_bytes(tr, var, a, if size < 4 { size } else { 4 })?);
    }
    let mut out = def;
    for (c, &(bo, bw)) in layout.iter().enumerate() {
        let dw = dws[(bo / 32) as usize];
        let raw = if bw == 32 {
            dw
        } else {
            let o = tr.u(bo % 32);
            let w = tr.u(bw);
            e(tr.b.bit_field_u_extract(u, None, dw, o, w))?
        };
        out[c] = convert_component(tr, raw, bw, nfmt, dfmt)?;
    }
    Ok(out)
}

fn apply_dst_sel(tr: &mut T, v: [Word; 4], dst_sel: u16, float: bool) -> [Word; 4] {
    if dst_sel == 0 {
        return v;
    }
    let one = if float { tr.u(1.0f32.to_bits()) } else { tr.u(1) };
    let z = tr.u(0);
    let mut out = v;
    for (c, o) in out.iter_mut().enumerate() {
        let sel = (dst_sel >> (c * 3)) & 7;
        *o = match sel {
            0 => z,
            1 => one,
            4..=7 => v[(sel - 4) as usize],
            _ => v[c],
        };
    }
    out
}

pub fn buffer(tr: &mut T, ins: &Inst) -> R<()> {
    let u = tr.t.u32;
    let Some(&(r, _)) = tr.plan.inst_res.get(&ins.pc) else { return Err(format!("{:#x}: buffer op without resource", ins.pc)) };
    let var = tr.buffers[&r];
    let (dfmt_v, nfmt_v, dst_sel, stride) = match tr.spec.resources.get(r) {
        Some(ResSpec::Buffer { dfmt, nfmt, dst_sel }) => (*dfmt, *nfmt, *dst_sel, stride_of(tr, r)),
        _ => (0, 0, 0, 0),
    };
    if ins.buf_lds() || ins.buf_tfe() || ins.buf_addr64() {
        tr.warn(format!("{:#x}: buffer lds/tfe/addr64 unsupported", ins.pc));
    }
    // Address (plus the binding's byte bias).
    let mut addr = tr.u(ins.buf_offset());
    if let Some(&b) = tr.buf_bias.get(&r) {
        addr = e(tr.b.i_add(u, None, addr, b))?;
    }
    let mut vreg = ins.mem_vaddr() as u32;
    if ins.buf_idxen() {
        let idx = tr.vreg(vreg)?;
        vreg += 1;
        let st = tr.sgpr_data_stride(r)?.unwrap_or_else(|| tr.u(stride));
        let m = e(tr.b.i_mul(u, None, idx, st))?;
        addr = e(tr.b.i_add(u, None, addr, m))?;
    }
    if ins.buf_offen() {
        let off = tr.vreg(vreg)?;
        addr = e(tr.b.i_add(u, None, addr, off))?;
    }
    let so = tr.operand(ins, ins.buf_soffset())?;
    addr = e(tr.b.i_add(u, None, addr, so))?;
    let vd = ins.mem_vdata() as u32;
    let mtbuf = ins.enc == crate::Enc::Mtbuf;
    let (dfmt, nfmt) = if mtbuf { (ins.mtbuf_dfmt() as u8, ins.mtbuf_nfmt() as u8) } else { (dfmt_v, nfmt_v) };
    let op = ins.op;
    let is_format_load = (mtbuf && op <= 3) || (!mtbuf && op <= 3);
    let is_format_store = (mtbuf && (4..=7).contains(&op)) || (!mtbuf && (4..=7).contains(&op));
    if is_format_load {
        let n = (op & 3) + 1;
        let v = load_formatted(tr, var, addr, dfmt, nfmt)?;
        let v = apply_dst_sel(tr, v, dst_sel, float_class(nfmt));
        for k in 0..n as usize {
            tr.set_vreg(vd + k as u32, v[k])?;
        }
        return Ok(());
    }
    // Page tracking of the bytes stores and atomics write (format stores:
    // up to 16 bytes).
    let written_bytes = match op {
        _ if is_format_store => Some(16),
        24 => Some(1),
        26 => Some(2),
        28 => Some(4),
        29 => Some(8),
        31 => Some(12),
        30 => Some(16),
        48..=61 => Some(4),
        _ => None,
    };
    if let Some(nb) = written_bytes {
        let k = tr.u(nb - 1);
        let last = e(tr.b.i_add(u, None, addr, k))?;
        mark_dirty(tr, r, addr, last)?;
    }
    if is_format_store {
        let n = (op & 3) + 1;
        return store_formatted(tr, var, addr, vd, n as u32, dfmt, nfmt);
    }
    let two = tr.u(2);
    match op {
        8..=11 => {
            let nb = if op <= 9 { 1 } else { 2 };
            let raw = load_bytes(tr, var, addr, nb)?;
            let r = if op == 9 || op == 11 {
                let z = tr.u(0);
                let w = tr.u(nb * 8);
                e(tr.b.bit_field_s_extract(u, None, raw, z, w))?
            } else {
                raw
            };
            tr.set_vreg(vd, r)?;
        }
        12 | 13 | 14 | 15 => {
            let n = match op {
                12 => 1,
                13 => 2,
                15 => 3,
                _ => 4,
            };
            let idx = e(tr.b.shift_right_logical(u, None, addr, two))?;
            for k in 0..n {
                let kk = tr.u(k);
                let i = e(tr.b.i_add(u, None, idx, kk))?;
                let v = load_dword(tr, var, i)?;
                tr.set_vreg(vd + k, v)?;
            }
        }
        24 | 26 | 28..=31 => {
            let act = tr.active()?;
            let n = match op {
                28 => 1,
                29 => 2,
                31 => 3,
                30 => 4,
                _ => 0,
            };
            let idx = e(tr.b.shift_right_logical(u, None, addr, two))?;
            let merge = tr.b.id();
            let body = tr.b.id();
            tr.b.selection_merge(merge, spirv::SelectionControl::NONE).map_err(|x| format!("{x:?}"))?;
            tr.b.branch_conditional(act, body, merge, []).map_err(|x| format!("{x:?}"))?;
            e(tr.b.begin_block(Some(body)))?;
            if n > 0 {
                for k in 0..n {
                    let kk = tr.u(k);
                    let i = e(tr.b.i_add(u, None, idx, kk))?;
                    let v = tr.vreg(vd + k)?;
                    store_dword(tr, var, i, v)?;
                }
            } else {
                // byte / short: atomic read-modify-write within the dword
                let bits = if op == 24 { 8 } else { 16 };
                let k3 = tr.u(3);
                let sub = e(tr.b.bitwise_and(u, None, addr, k3))?;
                let sh = e(tr.b.shift_left_logical(u, None, sub, k3))?;
                let m = tr.u((1u32 << bits) - 1);
                let mask = e(tr.b.shift_left_logical(u, None, m, sh))?;
                let v = tr.vreg(vd)?;
                let vm = e(tr.b.bitwise_and(u, None, v, m))?;
                let vs = e(tr.b.shift_left_logical(u, None, vm, sh))?;
                let z = tr.u(0);
                let p = e(tr.b.access_chain(tr.t.sb_u32, None, var, [z, idx]))?;
                let scope = tr.u(spirv::Scope::Device as u32);
                let sem = tr.u(0);
                let nm = e(tr.b.not(u, None, mask))?;
                e(tr.b.atomic_and(u, None, p, scope, sem, nm))?;
                e(tr.b.atomic_or(u, None, p, scope, sem, vs))?;
            }
            tr.b.branch(merge).map_err(|x| format!("{x:?}"))?;
            e(tr.b.begin_block(Some(merge)))?;
        }
        48..=61 => {
            let idx = e(tr.b.shift_right_logical(u, None, addr, two))?;
            let z = tr.u(0);
            let p = e(tr.b.access_chain(tr.t.sb_u32, None, var, [z, idx]))?;
            let scope = tr.u(spirv::Scope::Device as u32);
            let sem = tr.u(0);
            let v = tr.vreg(vd)?;
            let act = tr.active()?;
            let merge = tr.b.id();
            let body = tr.b.id();
            let pre_var = tr.b.id();
            let _ = pre_var;
            tr.b.selection_merge(merge, spirv::SelectionControl::NONE).map_err(|x| format!("{x:?}"))?;
            tr.b.branch_conditional(act, body, merge, []).map_err(|x| format!("{x:?}"))?;
            e(tr.b.begin_block(Some(body)))?;
            let old = match op {
                48 => e(tr.b.atomic_exchange(u, None, p, scope, sem, v))?,
                49 => {
                    let cmp = tr.vreg(vd + 1)?;
                    e(tr.b.atomic_compare_exchange(u, None, p, scope, sem, sem, v, cmp))?
                }
                50 => e(tr.b.atomic_i_add(u, None, p, scope, sem, v))?,
                51 => e(tr.b.atomic_i_sub(u, None, p, scope, sem, v))?,
                53 => e(tr.b.atomic_s_min(u, None, p, scope, sem, v))?,
                54 => e(tr.b.atomic_u_min(u, None, p, scope, sem, v))?,
                55 => e(tr.b.atomic_s_max(u, None, p, scope, sem, v))?,
                56 => e(tr.b.atomic_u_max(u, None, p, scope, sem, v))?,
                57 => e(tr.b.atomic_and(u, None, p, scope, sem, v))?,
                58 => e(tr.b.atomic_or(u, None, p, scope, sem, v))?,
                59 => e(tr.b.atomic_xor(u, None, p, scope, sem, v))?,
                60 => e(tr.b.atomic_i_increment(u, None, p, scope, sem))?,
                _ => e(tr.b.atomic_i_decrement(u, None, p, scope, sem))?,
            };
            if ins.buf_glc() {
                tr.set_vreg_raw(vd, old)?;
            }
            tr.b.branch(merge).map_err(|x| format!("{x:?}"))?;
            e(tr.b.begin_block(Some(merge)))?;
        }
        112 | 113 => {}
        _ => tr.warn(format!("{:#x}: unsupported {}", ins.pc, ins.name())),
    }
    Ok(())
}

/// Sets the `DirtyBits` bit of the pages of written buffer `r` holding byte
/// offsets `first` and `last` (one store; at most two pages) for the active
/// lanes. When a subgroup's lanes all hit one bitmap dword, one lane sets
/// their bits together (stores of a wave usually share pages).
fn mark_dirty(tr: &mut T, r: usize, first: Word, last: Word) -> R<()> {
    let Some((_, bits)) = tr.dirty else { return Ok(()) };
    let Some(&(base, head, limit)) = tr.dirty_res.get(&r) else { return Ok(()) };
    let u = tr.t.u32;
    let bt = tr.t.bool_;
    let sg = tr.u(spirv::Scope::Subgroup as u32);
    let dev = tr.u(spirv::Scope::Device as u32);
    let sem = tr.u(0);
    tr.b.capability(spirv::Capability::GroupNonUniform);
    tr.b.capability(spirv::Capability::GroupNonUniformVote);
    tr.b.capability(spirv::Capability::GroupNonUniformArithmetic);
    let act = tr.active()?;
    let merge = tr.b.id();
    let body = tr.b.id();
    tr.b.selection_merge(merge, spirv::SelectionControl::NONE).map_err(|x| format!("{x:?}"))?;
    tr.b.branch_conditional(act, body, merge, []).map_err(|x| format!("{x:?}"))?;
    e(tr.b.begin_block(Some(body)))?;
    for at in [first, last] {
        let (k12, k5, k31, one) = (tr.u(12), tr.u(5), tr.u(31), tr.u(1));
        let a = e(tr.b.i_add(u, None, at, head))?;
        let page = e(tr.b.shift_right_logical(u, None, a, k12))?;
        let word = e(tr.b.shift_right_logical(u, None, page, k5))?;
        let word = tr.glsl2(u, GLOp::UMin, word, limit)?;
        let word = e(tr.b.i_add(u, None, word, base))?;
        let bit_i = e(tr.b.bitwise_and(u, None, page, k31))?;
        let bit = e(tr.b.shift_left_logical(u, None, one, bit_i))?;
        let same = e(tr.b.group_non_uniform_all_equal(bt, None, sg, word))?;
        let all = e(tr.b.group_non_uniform_bitwise_or(u, None, sg, spirv::GroupOperation::Reduce, bit, None))?;
        let elected = e(tr.b.group_non_uniform_elect(bt, None, sg))?;
        let lead = e(tr.b.logical_and(bt, None, same, elected))?;
        let alone = e(tr.b.logical_not(bt, None, same))?;
        let go = e(tr.b.logical_or(bt, None, lead, alone))?;
        let val = e(tr.b.select(u, None, same, all, bit))?;
        let m2 = tr.b.id();
        let b2 = tr.b.id();
        tr.b.selection_merge(m2, spirv::SelectionControl::NONE).map_err(|x| format!("{x:?}"))?;
        tr.b.branch_conditional(go, b2, m2, []).map_err(|x| format!("{x:?}"))?;
        e(tr.b.begin_block(Some(b2)))?;
        let z = tr.u(0);
        let p = e(tr.b.access_chain(tr.t.sb_u32, None, bits, [z, word]))?;
        e(tr.b.atomic_or(u, None, p, dev, sem, val))?;
        tr.b.branch(m2).map_err(|x| format!("{x:?}"))?;
        e(tr.b.begin_block(Some(m2)))?;
    }
    tr.b.branch(merge).map_err(|x| format!("{x:?}"))?;
    e(tr.b.begin_block(Some(merge)))?;
    Ok(())
}

fn stride_of(tr: &T, r: usize) -> u32 {
    tr.spec.buffer_strides.get(r).copied().unwrap_or(0)
}

impl T<'_> {
    /// Stride is specialized per resource; no dynamic stride source.
    fn sgpr_data_stride(&mut self, _r: usize) -> R<Option<Word>> {
        Ok(None)
    }
}

fn store_formatted(tr: &mut T, var: Word, addr: Word, vd: u32, n: u32, dfmt: u8, nfmt: u8) -> R<()> {
    let u = tr.t.u32;
    let f = tr.t.f32;
    let act = tr.active()?;
    let merge = tr.b.id();
    let body = tr.b.id();
    tr.b.selection_merge(merge, spirv::SelectionControl::NONE).map_err(|x| format!("{x:?}"))?;
    tr.b.branch_conditional(act, body, merge, []).map_err(|x| format!("{x:?}"))?;
    e(tr.b.begin_block(Some(body)))?;
    let two = tr.u(2);
    match (dfmt, nfmt) {
        (4 | 11 | 13 | 14, _) => {
            let idx = e(tr.b.shift_right_logical(u, None, addr, two))?;
            let comps = match dfmt {
                4 => 1,
                11 => 2,
                13 => 3,
                _ => 4,
            };
            for k in 0..n.min(comps) {
                let kk = tr.u(k);
                let i = e(tr.b.i_add(u, None, idx, kk))?;
                let v = tr.vreg(vd + k)?;
                store_dword(tr, var, i, v)?;
            }
        }
        (10, 0) => {
            // RGBA8 unorm
            let mut vals = Vec::new();
            for k in 0..4 {
                let v = if k < n { tr.vreg(vd + k)? } else { tr.u(0) };
                vals.push(tr.to_f(v)?);
            }
            let vv = e(tr.b.composite_construct(tr.t.v4f, None, vals))?;
            let packed = tr.glsl1(u, GLOp::PackUnorm4x8, vv)?;
            let idx = e(tr.b.shift_right_logical(u, None, addr, two))?;
            store_dword(tr, var, idx, packed)?;
        }
        (5 | 12, 7) => {
            let idx = e(tr.b.shift_right_logical(u, None, addr, two))?;
            let words = if dfmt == 5 { 1 } else { 2 };
            for w in 0..words {
                let a = if 2 * w < n { tr.vreg(vd + 2 * w)? } else { tr.u(0) };
                let b = if 2 * w + 1 < n { tr.vreg(vd + 2 * w + 1)? } else { tr.u(0) };
                let (af, bf) = (tr.to_f(a)?, tr.to_f(b)?);
                let vv = e(tr.b.composite_construct(tr.t.v2f, None, [af, bf]))?;
                let p = tr.glsl1(u, GLOp::PackHalf2x16, vv)?;
                let kk = tr.u(w);
                let i = e(tr.b.i_add(u, None, idx, kk))?;
                store_dword(tr, var, i, p)?;
            }
        }
        (6, 7) => {
            // 10_11_11 float (sampled as B10G11R11_UFLOAT_PACK32): R 11 bits
            // at [10:0], G 11 bits at [21:11], B 10 bits at [31:22]; unsigned
            // 5-bit-exponent floats, i.e. the top bits of the half encoding
            // (negative values clamp to zero).
            let comp = |tr: &mut T, k: u32| -> R<Word> { if k < n { tr.vreg(vd + k) } else { Ok(tr.u(0)) } };
            let (r, g, b) = (comp(tr, 0)?, comp(tr, 1)?, comp(tr, 2)?);
            let (rf, gf, bf, zf) = (tr.to_f(r)?, tr.to_f(g)?, tr.to_f(b)?, tr.f(0.0));
            let rg = e(tr.b.composite_construct(tr.t.v2f, None, [rf, gf]))?;
            let rg = tr.glsl1(u, GLOp::PackHalf2x16, rg)?;
            let b0 = e(tr.b.composite_construct(tr.t.v2f, None, [bf, zf]))?;
            let bh = tr.glsl1(u, GLOp::PackHalf2x16, b0)?;
            let sixteen = tr.u(16);
            let gh = e(tr.b.shift_right_logical(u, None, rg, sixteen))?;
            // (half, mantissa bits dropped, width) -> field value
            let mut field = |tr: &mut T, h: Word, drop: u32, mask: u32| -> R<Word> {
                let sign_bit = tr.u(0x8000);
                let s = e(tr.b.bitwise_and(u, None, h, sign_bit))?;
                let zero = tr.u(0);
                let neg = e(tr.b.i_not_equal(tr.t.bool_, None, s, zero))?;
                let sh = tr.u(drop);
                let v = e(tr.b.shift_right_logical(u, None, h, sh))?;
                let m = tr.u(mask);
                let v = e(tr.b.bitwise_and(u, None, v, m))?;
                e(tr.b.select(u, None, neg, zero, v))
            };
            let r11 = field(tr, rg, 4, 0x7FF)?;
            let g11 = field(tr, gh, 4, 0x7FF)?;
            let b10 = field(tr, bh, 5, 0x3FF)?;
            let (s11, s22) = (tr.u(11), tr.u(22));
            let g_s = e(tr.b.shift_left_logical(u, None, g11, s11))?;
            let b_s = e(tr.b.shift_left_logical(u, None, b10, s22))?;
            let rg_or = e(tr.b.bitwise_or(u, None, r11, g_s))?;
            let packed = e(tr.b.bitwise_or(u, None, rg_or, b_s))?;
            let idx = e(tr.b.shift_right_logical(u, None, addr, two))?;
            store_dword(tr, var, idx, packed)?;
        }
        // Whole-dword layouts (16_16, 2_10_10_10, 10_10_10_2, 8_8_8_8, 16_16_16_16) in
        // UNORM / SNORM / UINT / SINT: each component converted and packed at its offset.
        (5 | 8 | 9 | 10 | 12, 0 | 1 | 4 | 5) => {
            let (layout, bytes) = dfmt_layout(dfmt).expect("dfmt layout");
            let idx = e(tr.b.shift_right_logical(u, None, addr, two))?;
            let mut words = vec![tr.u(0); (bytes / 4) as usize];
            for (k, &(off, w)) in layout.iter().enumerate() {
                let v = if (k as u32) < n { tr.vreg(vd + k as u32)? } else { tr.u(0) };
                let field = pack_field(tr, v, w, nfmt)?;
                let s = tr.u(off % 32);
                let shifted = e(tr.b.shift_left_logical(u, None, field, s))?;
                let wi = (off / 32) as usize;
                words[wi] = e(tr.b.bitwise_or(u, None, words[wi], shifted))?;
            }
            for (wi, &wv) in words.iter().enumerate() {
                let k = tr.u(wi as u32);
                let i = e(tr.b.i_add(u, None, idx, k))?;
                store_dword(tr, var, i, wv)?;
            }
        }
        // Sub-dword layouts (8, 16, 8_8; 16-bit float too): the element's
        // bytes are merged into their dword with atomics, as buffer_store_byte
        // / _short do (lanes store neighbouring elements of one dword).
        (1 | 3, 0 | 1 | 4 | 5) | (2, 0 | 1 | 4 | 5 | 7) => {
            let (layout, bytes) = dfmt_layout(dfmt).expect("dfmt layout");
            let mut val = tr.u(0);
            for (k, &(off, w)) in layout.iter().enumerate() {
                let v = if (k as u32) < n { tr.vreg(vd + k as u32)? } else { tr.u(0) };
                let field = if nfmt == 7 {
                    let (x, z) = (tr.to_f(v)?, tr.f(0.0));
                    let vv = e(tr.b.composite_construct(tr.t.v2f, None, [x, z]))?;
                    tr.glsl1(u, GLOp::PackHalf2x16, vv)?
                } else {
                    pack_field(tr, v, w, nfmt)?
                };
                let s = tr.u(off);
                let shifted = e(tr.b.shift_left_logical(u, None, field, s))?;
                val = e(tr.b.bitwise_or(u, None, val, shifted))?;
            }
            let k3 = tr.u(3);
            let sub = e(tr.b.bitwise_and(u, None, addr, k3))?;
            let sh = e(tr.b.shift_left_logical(u, None, sub, k3))?;
            let m = tr.u((1u32 << (bytes * 8)) - 1);
            let vm = e(tr.b.bitwise_and(u, None, val, m))?;
            let mask = e(tr.b.shift_left_logical(u, None, m, sh))?;
            let vs = e(tr.b.shift_left_logical(u, None, vm, sh))?;
            let idx = e(tr.b.shift_right_logical(u, None, addr, two))?;
            let z = tr.u(0);
            let p = e(tr.b.access_chain(tr.t.sb_u32, None, var, [z, idx]))?;
            let scope = tr.u(spirv::Scope::Device as u32);
            let sem = tr.u(0);
            let nm = e(tr.b.not(u, None, mask))?;
            e(tr.b.atomic_and(u, None, p, scope, sem, nm))?;
            e(tr.b.atomic_or(u, None, p, scope, sem, vs))?;
        }
        _ => {
            let _ = f;
            tr.warn(format!("formatted store dfmt {dfmt} nfmt {nfmt} unsupported"));
        }
    }
    tr.b.branch(merge).map_err(|x| format!("{x:?}"))?;
    e(tr.b.begin_block(Some(merge)))?;
    Ok(())
}

/// Converts a register value to a `w`-bit field of a formatted store (low bits).
fn pack_field(tr: &mut T, v: Word, w: u32, nfmt: u8) -> R<Word> {
    let u = tr.t.u32;
    let i = tr.t.i32;
    let f = tr.t.f32;
    let max_u = if w >= 32 { u32::MAX } else { (1u32 << w) - 1 };
    let mask = tr.u(max_u);
    let r = match nfmt {
        0 => {
            // UNORM: clamp to [0, 1], scale, round to nearest even.
            let x = tr.to_f(v)?;
            let (z, one) = (tr.f(0.0), tr.f(1.0));
            let c = tr.glsl3(f, GLOp::FClamp, x, z, one)?;
            let scale = tr.f(max_u as f32);
            let s = e(tr.b.f_mul(f, None, c, scale))?;
            let r = tr.glsl1(f, GLOp::RoundEven, s)?;
            e(tr.b.convert_f_to_u(u, None, r))?
        }
        1 => {
            // SNORM: clamp to [-1, 1], scale by 2^(w-1)-1, round, two's complement field.
            let x = tr.to_f(v)?;
            let (lo, hi) = (tr.f(-1.0), tr.f(1.0));
            let c = tr.glsl3(f, GLOp::FClamp, x, lo, hi)?;
            let scale = tr.f(((1u64 << (w - 1)) - 1) as f32);
            let s = e(tr.b.f_mul(f, None, c, scale))?;
            let r = tr.glsl1(f, GLOp::RoundEven, s)?;
            let si = e(tr.b.convert_f_to_s(i, None, r))?;
            tr.to_u(si)?
        }
        4 => tr.glsl2(u, GLOp::UMin, v, mask)?,
        _ => {
            // SINT: saturate to the field's signed range.
            let vi = tr.to_i(v)?;
            let lo = tr.i(-(1i64 << (w - 1)) as i32);
            let hi = tr.i(((1i64 << (w - 1)) - 1) as i32);
            let c = tr.glsl3(i, GLOp::SClamp, vi, lo, hi)?;
            tr.to_u(c)?
        }
    };
    e(tr.b.bitwise_and(u, None, r, mask))
}

// ---------------------------------------------------------------------------
// Images

fn coord_count(dim: ImageDim) -> (u32, bool) {
    match dim {
        ImageDim::D1 => (1, false),
        ImageDim::D2 => (2, false),
        ImageDim::D3 => (3, false),
        ImageDim::Cube => (3, false),
        ImageDim::D1Array => (2, true),
        ImageDim::D2Array => (3, true),
        ImageDim::CubeArray => (4, true),
    }
}

/// Builds a host integer coordinate from GCN coordinates (1D images are
/// 2D on the host: insert y = 0 after x).
fn int_coord(tr: &mut T, dim: ImageDim, mut cs: Vec<Word>) -> R<Word> {
    if matches!(dim, ImageDim::D1 | ImageDim::D1Array) {
        let z = tr.i(0);
        cs.insert(1, z);
    }
    let cty = match cs.len() {
        1 => tr.t.i32,
        2 => tr.t.v2i,
        3 => tr.t.v3i,
        _ => tr.t.v4i,
    };
    if cs.len() == 1 { Ok(cs[0]) } else { e(tr.b.composite_construct(cty, None, cs)) }
}

/// GCN cube coordinates (s+1.5, t+1.5, face) -> direction vector.
fn cube_dir(tr: &mut T, s: Word, t: Word, face: Word) -> R<Word> {
    let f = tr.t.f32;
    let bt = tr.t.bool_;
    let k15 = tr.f(1.5);
    let two = tr.f(2.0);
    let sc0 = e(tr.b.f_sub(f, None, s, k15))?;
    let tc0 = e(tr.b.f_sub(f, None, t, k15))?;
    let sc = e(tr.b.f_mul(f, None, sc0, two))?;
    let tc = e(tr.b.f_mul(f, None, tc0, two))?;
    let one = tr.f(1.0);
    let m1 = tr.f(-1.0);
    let nsc = e(tr.b.f_negate(f, None, sc))?;
    let ntc = e(tr.b.f_negate(f, None, tc))?;
    let fr = tr.glsl1(f, GLOp::RoundEven, face)?;
    // face: 0 +X, 1 -X, 2 +Y, 3 -Y, 4 +Z, 5 -Z
    let faces = [
        [one, ntc, nsc],
        [m1, ntc, sc],
        [sc, one, tc],
        [sc, m1, ntc],
        [sc, ntc, one],
        [nsc, ntc, m1],
    ];
    let mut out = Vec::new();
    for c in 0..3 {
        let mut v = faces[5][c];
        for k in (0..5).rev() {
            let kf = tr.f(k as f32);
            let is = e(tr.b.f_ord_equal(bt, None, fr, kf))?;
            v = e(tr.b.select(f, None, is, faces[k][c], v))?;
        }
        out.push(v);
    }
    e(tr.b.composite_construct(tr.t.v3f, None, out))
}

pub fn image(tr: &mut T, ins: &Inst) -> R<()> {
    let u = tr.t.u32;
    let f = tr.t.f32;
    let Some(&(ri, si)) = tr.plan.inst_res.get(&ins.pc) else { return Err(format!("{:#x}: image op without resource", ins.pc)) };
    let (var, img_ty, dim, class, storage) = {
        let ir = &tr.images[&ri];
        (ir.var, ir.img_ty, ir.dim, ir.class, ir.storage)
    };
    let op = ins.op;
    let vaddr = ins.mem_vaddr() as u32;
    let vd = ins.mem_vdata() as u32;
    let dmask = ins.mimg_dmask();
    let (ncoord, _arrayed) = coord_count(dim);
    let res_ty = match class {
        NumClass::Float => tr.t.v4f,
        NumClass::Sint => tr.t.v4i,
        NumClass::Uint => tr.t.v4u,
    };
    let comp_ty = match class {
        NumClass::Float => f,
        NumClass::Sint => tr.t.i32,
        NumClass::Uint => u,
    };
    let image = e(tr.b.load(img_ty, None, var, None, []))?;
    // Writes the `mask`-selected components of a vec4 result to consecutive
    // VGPRs (the dmask; all four for gathers).
    let write_result = |tr: &mut T, res: Word, mask: u32| -> R<()> {
        let mut k = 0;
        for c in 0..4 {
            if mask & (1 << c) != 0 {
                let x = e(tr.b.composite_extract(comp_ty, None, res, [c]))?;
                let xu = tr.to_u(x)?;
                tr.set_vreg(vd + k, xu)?;
                k += 1;
            }
        }
        Ok(())
    };
    match op {
        0 | 1 => {
            // image_load / image_load_mip: integer coordinates
            let mut cs = Vec::new();
            for k in 0..ncoord {
                let v = tr.vreg(vaddr + k)?;
                cs.push(tr.to_i(v)?);
            }
            let coord = int_coord(tr, dim, cs)?;
            let res = if storage {
                e(tr.b.image_read(res_ty, None, image, coord, None, []))?
            } else {
                let lod = if op == 1 {
                    let v = tr.vreg(vaddr + ncoord)?;
                    tr.to_i(v)?
                } else {
                    tr.i(0)
                };
                e(tr.b.image_fetch(res_ty, None, image, coord, Some(ImageOperands::LOD), [Operand::IdRef(lod)]))?
            };
            write_result(tr, res, dmask)
        }
        8 | 9 => {
            let mut cs = Vec::new();
            for k in 0..ncoord {
                let v = tr.vreg(vaddr + k)?;
                cs.push(tr.to_i(v)?);
            }
            let coord = int_coord(tr, dim, cs)?;
            let mut comps = Vec::new();
            let mut k = 0;
            for c in 0..4 {
                let v = if dmask & (1 << c) != 0 {
                    let x = tr.vreg(vd + k)?;
                    k += 1;
                    x
                } else {
                    tr.u(0)
                };
                comps.push(match class {
                    NumClass::Float => tr.to_f(v)?,
                    NumClass::Sint => tr.to_i(v)?,
                    NumClass::Uint => v,
                });
            }
            let texel = e(tr.b.composite_construct(res_ty, None, comps))?;
            let act = tr.active()?;
            let merge = tr.b.id();
            let body = tr.b.id();
            tr.b.selection_merge(merge, spirv::SelectionControl::NONE).map_err(|x| format!("{x:?}"))?;
            tr.b.branch_conditional(act, body, merge, []).map_err(|x| format!("{x:?}"))?;
            e(tr.b.begin_block(Some(body)))?;
            tr.b.image_write(image, coord, texel, None, []).map_err(|x| format!("{x:?}"))?;
            tr.b.branch(merge).map_err(|x| format!("{x:?}"))?;
            e(tr.b.begin_block(Some(merge)))?;
            Ok(())
        }
        14 => {
            // image_get_resinfo: [w, h, d/layers, levels]
            let lodv = tr.vreg(vaddr)?;
            let lod = tr.to_i(lodv)?;
            let (n, sty) = match dim {
                ImageDim::D1 | ImageDim::D2 | ImageDim::Cube => (2, tr.t.v2i),
                _ => (3, tr.t.v3i),
            };
            let size = if storage { e(tr.b.image_query_size(sty, None, image))? } else { e(tr.b.image_query_size_lod(sty, None, image, lod))? };
            let levels = if storage { tr.i(1) } else { e(tr.b.image_query_levels(tr.t.i32, None, image))? };
            // Host component order; 1D arrays report (width, layers).
            let order: [u32; 3] = if dim == ImageDim::D1Array { [0, 2, 3] } else { [0, 1, 2] };
            let mut vals = Vec::new();
            for c in order {
                if c < n {
                    vals.push(e(tr.b.composite_extract(tr.t.i32, None, size, [c]))?);
                } else {
                    vals.push(tr.i(1));
                }
            }
            vals.push(levels);
            let mut k = 0;
            for (c, v) in vals.iter().enumerate() {
                if dmask & (1 << c) != 0 {
                    let x = tr.to_u(*v)?;
                    tr.set_vreg(vd + k, x)?;
                    k += 1;
                }
            }
            Ok(())
        }
        32..=63 | 64..=95 => sample(tr, ins, image, img_ty, si, dim, res_ty, &write_result),
        _ => {
            tr.warn(format!("{:#x}: unsupported {}", ins.pc, ins.name()));
            Ok(())
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sample(tr: &mut T, ins: &Inst, image: Word, img_ty: Word, si: Option<usize>, dim: ImageDim, res_ty: Word, write_result: &dyn Fn(&mut T, Word, u32) -> R<()>) -> R<()> {
    let f = tr.t.f32;
    let op = ins.op;
    let gather = op >= 64;
    let base = if gather { op - 64 } else { op - 32 };
    let has_o = base >= 16;
    let b = base & 15;
    let has_c = b >= 8;
    let k = b & 7;
    // k: 0 plain, 1 cl, 2 d, 3 d_cl, 4 l, 5 b, 6 b_cl, 7 lz
    let has_d = k == 2 || k == 3;
    let has_l = k == 4;
    let has_b = k == 5 || k == 6;
    let has_cl = k == 1 || k == 3 || k == 6;
    let has_lz = k == 7;
    let Some(sidx) = si else { return Err(format!("{:#x}: sample without sampler", ins.pc)) };
    let svar = tr.samplers[&sidx];
    let sampler = e(tr.b.load(tr.sampler_ty, None, svar, None, []))?;
    let si_ty = tr.b.type_sampled_image(img_ty);
    let simg = e(tr.b.sampled_image(si_ty, None, image, sampler))?;
    let mut a = ins.mem_vaddr() as u32;
    let mut next = |tr: &mut T| -> R<Word> {
        let v = tr.vreg(a)?;
        a += 1;
        Ok(v)
    };
    let offset = if has_o { Some(next(tr)?) } else { None };
    let bias = if has_b { Some(next(tr)?) } else { None };
    let zcmp = if has_c { Some(next(tr)?) } else { None };
    let (ncoord, _) = coord_count(dim);
    let gdims = match dim {
        ImageDim::D1 | ImageDim::D1Array => 1,
        ImageDim::D3 | ImageDim::Cube | ImageDim::CubeArray => 3,
        _ => 2,
    };
    let derivs = if has_d {
        let mut dx = Vec::new();
        let mut dy = Vec::new();
        for _ in 0..gdims {
            let v = next(tr)?;
            dx.push(tr.to_f(v)?);
        }
        for _ in 0..gdims {
            let v = next(tr)?;
            dy.push(tr.to_f(v)?);
        }
        Some((dx, dy))
    } else {
        None
    };
    let mut cs = Vec::new();
    let gcn_coords = match dim {
        ImageDim::Cube => 3,
        ImageDim::CubeArray => 3,
        _ => ncoord,
    };
    for _ in 0..gcn_coords {
        let v = next(tr)?;
        cs.push(tr.to_f(v)?);
    }
    let lod_or_clamp = if has_l || has_cl { Some(next(tr)?) } else { None };
    // Offsets of non-gather 2D samples move the coordinates by that many
    // texels of the sampled level (Vulkan allows a non-constant Offset image
    // operand only on gathers). Exact for _lz; _l uses floor(lod), implicit
    // and gradient levels are taken as 0.
    if let (Some(ov), false, ImageDim::D2 | ImageDim::D2Array) = (offset, gather, dim) {
        let (u, i) = (tr.t.u32, tr.t.i32);
        let (z, six, k8) = (tr.u(0), tr.u(6), tr.u(8));
        let ox = e(tr.b.bit_field_s_extract(u, None, ov, z, six))?;
        let oy = e(tr.b.bit_field_s_extract(u, None, ov, k8, six))?;
        let zero_i = tr.to_i(z)?;
        let level = match lod_or_clamp.filter(|_| has_l) {
            Some(l) => {
                let lf = tr.to_f(l)?;
                let fl = tr.glsl1(f, GLOp::Floor, lf)?;
                let li = e(tr.b.convert_f_to_s(i, None, fl))?;
                tr.glsl2(i, GLOp::SMax, li, zero_i)?
            }
            None => zero_i,
        };
        tr.b.capability(Capability::ImageQuery);
        let size_ty = if dim == ImageDim::D2 { tr.t.v2i } else { tr.b.type_vector(i, 3) };
        let size = e(tr.b.image_query_size_lod(size_ty, None, image, level))?;
        for (c, o) in [(0usize, ox), (1, oy)] {
            let n = e(tr.b.composite_extract(i, None, size, [c as u32]))?;
            let nf = e(tr.b.convert_s_to_f(f, None, n))?;
            let oi = tr.to_i(o)?;
            let of = e(tr.b.convert_s_to_f(f, None, oi))?;
            let d = e(tr.b.f_div(f, None, of, nf))?;
            cs[c] = e(tr.b.f_add(f, None, cs[c], d))?;
        }
    }
    let coord = match dim {
        ImageDim::Cube | ImageDim::CubeArray => {
            if dim == ImageDim::CubeArray {
                // face = slice * 8 + face
                let k8 = tr.f(8.0);
                let q = e(tr.b.f_div(f, None, cs[2], k8))?;
                let slice = tr.glsl1(f, GLOp::Floor, q)?;
                let s8 = e(tr.b.f_mul(f, None, slice, k8))?;
                let face = e(tr.b.f_sub(f, None, cs[2], s8))?;
                let d = cube_dir(tr, cs[0], cs[1], face)?;
                let x = e(tr.b.composite_extract(f, None, d, [0]))?;
                let y = e(tr.b.composite_extract(f, None, d, [1]))?;
                let z = e(tr.b.composite_extract(f, None, d, [2]))?;
                e(tr.b.composite_construct(tr.t.v4f, None, [x, y, z, slice]))?
            } else {
                cube_dir(tr, cs[0], cs[1], cs[2])?
            }
        }
        _ => {
            let mut cs = cs.clone();
            if matches!(dim, ImageDim::D1 | ImageDim::D1Array) {
                // Row centre of the height-1 host image.
                let half = tr.f(0.5);
                cs.insert(1, half);
            }
            match cs.len() {
                1 => cs[0],
                2 => e(tr.b.composite_construct(tr.t.v2f, None, cs))?,
                3 => e(tr.b.composite_construct(tr.t.v3f, None, cs))?,
                _ => e(tr.b.composite_construct(tr.t.v4f, None, cs))?,
            }
        }
    };
    let implicit_ok = tr.spec.stage == Stage::Fragment;
    let mut operands: Vec<Operand> = Vec::new();
    let mut mask = ImageOperands::NONE;
    let explicit;
    if let Some((dx, dy)) = derivs {
        let vty = match gdims {
            1 => f,
            2 => tr.t.v2f,
            _ => tr.t.v3f,
        };
        // 1D images are 2D on the host: zero y gradient.
        let (vty, dx, dy) = if gdims == 1 {
            let z = tr.f(0.0);
            (tr.t.v2f, vec![dx[0], z], vec![dy[0], z])
        } else {
            (vty, dx, dy)
        };
        let gx = e(tr.b.composite_construct(vty, None, dx))?;
        let gy = e(tr.b.composite_construct(vty, None, dy))?;
        mask |= ImageOperands::GRAD;
        operands.push(Operand::IdRef(gx));
        operands.push(Operand::IdRef(gy));
        explicit = true;
    } else if has_l || has_lz || !implicit_ok {
        let lod = if has_l {
            let v = lod_or_clamp.unwrap();
            tr.to_f(v)?
        } else {
            tr.f(0.0)
        };
        mask |= ImageOperands::LOD;
        operands.push(Operand::IdRef(lod));
        explicit = true;
    } else {
        if let Some(bv) = bias {
            let bf = tr.to_f(bv)?;
            mask |= ImageOperands::BIAS;
            operands.insert(0, Operand::IdRef(bf));
        }
        explicit = false;
    }
    if let Some(ov) = offset.filter(|_| gather) {
        if dim == ImageDim::D2 || dim == ImageDim::D2Array {
            let u = tr.t.u32;
            let z = tr.u(0);
            let six = tr.u(6);
            let k8 = tr.u(8);
            let x = e(tr.b.bit_field_s_extract(u, None, ov, z, six))?;
            let y = e(tr.b.bit_field_s_extract(u, None, ov, k8, six))?;
            let xi = tr.to_i(x)?;
            let yi = tr.to_i(y)?;
            let off = e(tr.b.composite_construct(tr.t.v2i, None, [xi, yi]))?;
            mask |= ImageOperands::OFFSET;
            operands.push(Operand::IdRef(off));
        } else {
            tr.warn(format!("{:#x}: sample offset on non-2D image ignored", ins.pc));
        }
    } else if offset.is_some() && !matches!(dim, ImageDim::D2 | ImageDim::D2Array) {
        tr.warn(format!("{:#x}: sample offset on non-2D image ignored", ins.pc));
    }
    if has_cl && !has_l && implicit_ok && !explicit {
        tr.warn(format!("{:#x}: min-lod clamp ignored", ins.pc));
    }
    let res = if gather {
        let comp = tr.u(match ins.mimg_dmask() {
            2 => 1,
            4 => 2,
            8 => 3,
            _ => 0,
        });
        // Core SPIR-V gathers read the base level and take no Lod/Grad/Bias operand:
        // exact for image_gather4*_lz and implicit-LOD gathers; an explicit nonzero
        // level (gather4_l) is approximated by the base level.
        if mask.intersects(ImageOperands::LOD | ImageOperands::GRAD | ImageOperands::BIAS) {
            if has_l || mask.contains(ImageOperands::GRAD) {
                tr.warn(format!("{:#x}: gather with explicit level/gradients reads the base level", ins.pc));
            }
            let lead = (mask.contains(ImageOperands::BIAS) as usize)
                + (mask.contains(ImageOperands::LOD) as usize)
                + 2 * (mask.contains(ImageOperands::GRAD) as usize);
            operands.drain(..lead);
            mask &= !(ImageOperands::LOD | ImageOperands::GRAD | ImageOperands::BIAS);
        }
        let m = if mask == ImageOperands::NONE { None } else { Some(mask) };
        if let Some(zc) = zcmp {
            let zf = tr.to_f(zc)?;
            e(tr.b.image_dref_gather(res_ty, None, simg, coord, zf, m, operands))?
        } else {
            e(tr.b.image_gather(res_ty, None, simg, coord, comp, m, operands))?
        }
    } else if let Some(zc) = zcmp {
        let zf = tr.to_f(zc)?;
        let r = if explicit {
            e(tr.b.image_sample_dref_explicit_lod(f, None, simg, coord, zf, mask, operands))?
        } else {
            e(tr.b.image_sample_dref_implicit_lod(f, None, simg, coord, zf, if mask == ImageOperands::NONE { None } else { Some(mask) }, operands))?
        };
        let z = tr.f(0.0);
        let one = tr.f(1.0);
        e(tr.b.composite_construct(tr.t.v4f, None, [r, z, z, one]))?
    } else if explicit {
        e(tr.b.image_sample_explicit_lod(res_ty, None, simg, coord, mask, operands))?
    } else {
        e(tr.b.image_sample_implicit_lod(res_ty, None, simg, coord, if mask == ImageOperands::NONE { None } else { Some(mask) }, operands))?
    };
    // A gather's dmask selects the component; its four texels' values
    // always fill four VGPRs.
    write_result(tr, res, if gather { 0xF } else { ins.mimg_dmask() })
}

// ---------------------------------------------------------------------------
// Exports

pub fn export(tr: &mut T, ins: &Inst) -> R<()> {
    let tgt = ins.exp_target();
    let en = ins.exp_en();
    let f = tr.t.f32;
    let mut vals = Vec::new();
    for k in 0..4 {
        let v = tr.vreg(ins.exp_vsrc(k) as u32)?;
        vals.push(v);
    }
    let act = tr.active()?;
    // Kill inactive fragments at the valid-mask export.
    if tr.spec.stage == Stage::Fragment && ins.exp_vm() {
        let nact = e(tr.b.logical_not(tr.t.bool_, None, act))?;
        let merge = tr.b.id();
        let kill = tr.b.id();
        tr.b.selection_merge(merge, spirv::SelectionControl::NONE).map_err(|x| format!("{x:?}"))?;
        tr.b.branch_conditional(nact, kill, merge, []).map_err(|x| format!("{x:?}"))?;
        e(tr.b.begin_block(Some(kill)))?;
        tr.b.kill().map_err(|x| format!("{x:?}"))?;
        e(tr.b.begin_block(Some(merge)))?;
        tr.kill_used = true;
    }
    match tgt {
        0..=7 => {
            let mrt = tr.spec.mrt[tgt as usize];
            let comps: Vec<Word> = if !tr.spec.debug_capture.is_empty() {
                let mut c = Vec::new();
                let cap = (tgt as usize).min(tr.spec.debug_capture.len() - 1);
                for k in 0..3 {
                    let var = tr.dbg_vars[cap * 3 + k];
                    let u = e(tr.b.load(tr.t.u32, None, var, None, []))?;
                    c.push(tr.to_f(u)?);
                }
                c.push(tr.f(1.0));
                c
            } else if ins.exp_compr() {
                let mut c = Vec::new();
                for k in 0..2 {
                    let w = vals[k];
                    let (op, int) = match mrt.export_format {
                        5 => (Some(GLOp::UnpackUnorm2x16), false),
                        6 => (Some(GLOp::UnpackSnorm2x16), false),
                        7 | 8 => (None, true),
                        _ => (Some(GLOp::UnpackHalf2x16), false),
                    };
                    if let Some(op) = op {
                        let v = tr.glsl1(tr.t.v2f, op, w)?;
                        c.push(e(tr.b.composite_extract(f, None, v, [0]))?);
                        c.push(e(tr.b.composite_extract(f, None, v, [1]))?);
                    } else {
                        let _ = int;
                        let u = tr.t.u32;
                        let z = tr.u(0);
                        let k16 = tr.u(16);
                        let lo = e(tr.b.bit_field_u_extract(u, None, w, z, k16))?;
                        let hi = e(tr.b.bit_field_u_extract(u, None, w, k16, k16))?;
                        c.push(tr.to_f(lo)?);
                        c.push(tr.to_f(hi)?);
                    }
                }
                c
            } else {
                let mut c = Vec::new();
                for v in vals.iter().take(4) {
                    c.push(tr.to_f(*v)?);
                }
                c
            };
            let (ty, var) = match tr.out_mrt.get(&tgt) {
                Some(&x) => x,
                None => {
                    let ty = match mrt.class {
                        1 => tr.t.v4i,
                        2 => tr.t.v4u,
                        _ => tr.t.v4f,
                    };
                    let v = tr.output(ty, None, Some(tgt));
                    tr.out_mrt.insert(tgt, (ty, v));
                    (ty, v)
                }
            };
            tr.mrt_mask |= 1 << tgt;
            let comps: Vec<Word> = if ty == tr.t.v4f {
                comps
            } else {
                let mut out = Vec::new();
                for c in comps {
                    let u = tr.to_u(c)?;
                    out.push(if ty == tr.t.v4i { tr.to_i(u)? } else { u });
                }
                out
            };
            let mut cs = comps;
            while cs.len() < 4 {
                cs.push(tr.f(0.0));
            }
            let v = e(tr.b.composite_construct(ty, None, cs))?;
            tr.b.store(var, v, None, []).map_err(|x| format!("{x:?}"))?;
        }
        8 => {
            if en & 1 != 0 {
                let var = match tr.out_depth {
                    Some(v) => v,
                    None => {
                        let v = tr.output(f, Some(BuiltIn::FragDepth), None);
                        tr.out_depth = Some(v);
                        v
                    }
                };
                let d = tr.to_f(vals[0])?;
                tr.b.store(var, d, None, []).map_err(|x| format!("{x:?}"))?;
            }
        }
        9 => {}
        12 => {
            let var = match tr.out_pos {
                Some(v) => v,
                None => {
                    let v = tr.output(tr.t.v4f, Some(BuiltIn::Position), None);
                    tr.out_pos = Some(v);
                    v
                }
            };
            let mut cs = Vec::new();
            for v in vals.iter() {
                cs.push(tr.to_f(*v)?);
            }
            let p = e(tr.b.composite_construct(tr.t.v4f, None, cs.clone()))?;
            tr.b.store(var, p, None, []).map_err(|x| format!("{x:?}"))?;
            if let Some((buf, idx)) = tr.dbg_out {
                let stride = tr.u(super::DEBUG_OUT_STRIDE);
                let base = e(tr.b.i_mul(tr.t.u32, None, idx, stride))?;
                let zero = tr.u(0);
                let mut vals = Vec::new();
                for c in cs {
                    vals.push(tr.to_u(c)?);
                }
                for c in 0..tr.spec.debug_capture.len().min(12) {
                    let var = tr.dbg_vars[c * 3];
                    vals.push(e(tr.b.load(tr.t.u32, None, var, None, []))?);
                }
                for (k, bits) in vals.into_iter().enumerate() {
                    let off = tr.u(k as u32);
                    let at = e(tr.b.i_add(tr.t.u32, None, base, off))?;
                    let ptr_ty = tr.b.type_pointer(None, StorageClass::StorageBuffer, tr.t.u32);
                    let ptr = e(tr.b.access_chain(ptr_ty, None, buf, [zero, at]))?;
                    tr.b.store(ptr, bits, None, []).map_err(|x| format!("{x:?}"))?;
                }
            }
        }
        13..=15 => {
            tr.pos_exports |= 1 << (tgt - 12);
            if let Some(v) = tr.dist.vec_target.iter().position(|t| *t == Some(tgt)) {
                for k in 0..4 {
                    if en & (1 << k) == 0 {
                        continue;
                    }
                    let d = v * 4 + k;
                    let x = tr.to_f(vals[k])?;
                    for (slot, var) in [(tr.dist.clip[d], tr.out_clip), (tr.dist.cull[d], tr.out_cull)] {
                        if let (Some(i), Some(var)) = (slot, var) {
                            let idx = tr.u(i);
                            let p = e(tr.b.access_chain(tr.t.out_f32, None, var, [idx]))?;
                            tr.b.store(p, x, None, []).map_err(|x| format!("{x:?}"))?;
                        }
                    }
                }
            } else {
                tr.pos_ignored |= 1 << (tgt - 12);
                let what = if tr.dist.misc_target == Some(tgt) { "the misc vector (point size, edge flag, layer, viewport index)" } else { "no use" };
                tr.warn(format!("export to pos{} ignored: {what} per PA_CL_VS_OUT_CNTL {:#x}", tgt - 12, tr.spec.vs_out_cntl));
            }
        }
        32..=63 => {
            let p = tgt - 32;
            let var = match tr.out_params.get(&p) {
                Some(&v) => v,
                None => {
                    let v = tr.output(tr.t.v4f, None, Some(p));
                    tr.out_params.insert(p, v);
                    v
                }
            };
            tr.params = tr.params.max(p + 1);
            let mut cs = Vec::new();
            for v in vals.iter() {
                cs.push(tr.to_f(*v)?);
            }
            let pv = e(tr.b.composite_construct(tr.t.v4f, None, cs))?;
            tr.b.store(var, pv, None, []).map_err(|x| format!("{x:?}"))?;
        }
        _ => tr.warn(format!("export target {tgt} ignored")),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// LDS / GDS

/// Emits `f` in a block executed only when `act` holds.
fn guarded(tr: &mut T, act: Word, f: &mut dyn FnMut(&mut T) -> R<()>) -> R<()> {
    let merge = tr.b.id();
    let body = tr.b.id();
    tr.b.selection_merge(merge, spirv::SelectionControl::NONE).map_err(|x| format!("{x:?}"))?;
    tr.b.branch_conditional(act, body, merge, []).map_err(|x| format!("{x:?}"))?;
    e(tr.b.begin_block(Some(body)))?;
    f(tr)?;
    tr.b.branch(merge).map_err(|x| format!("{x:?}"))?;
    e(tr.b.begin_block(Some(merge)))?;
    Ok(())
}

/// DS atomic `base` (0 add, 1 sub, 5..11 min/max/and/or/xor) on `p`;
/// returns the pre-op value, or None for unsupported operations.
fn ds_atomic(tr: &mut T, base: u16, p: Word, scope: spirv::Scope, v: Word) -> R<Option<Word>> {
    let u = tr.t.u32;
    let scope = tr.u(scope as u32);
    let sem = tr.u(0);
    Ok(Some(match base {
        0 => e(tr.b.atomic_i_add(u, None, p, scope, sem, v))?,
        1 => e(tr.b.atomic_i_sub(u, None, p, scope, sem, v))?,
        5 => e(tr.b.atomic_s_min(u, None, p, scope, sem, v))?,
        6 => e(tr.b.atomic_s_max(u, None, p, scope, sem, v))?,
        7 => e(tr.b.atomic_u_min(u, None, p, scope, sem, v))?,
        8 => e(tr.b.atomic_u_max(u, None, p, scope, sem, v))?,
        9 => e(tr.b.atomic_and(u, None, p, scope, sem, v))?,
        10 => e(tr.b.atomic_or(u, None, p, scope, sem, v))?,
        11 => e(tr.b.atomic_xor(u, None, p, scope, sem, v))?,
        _ => return Ok(None),
    }))
}

/// GDS instructions on the shared GDS storage buffer. Address: M0[15:0]
/// (allocation base) + instruction offset (+ VGPR address except for
/// append/consume/ordered count). Lanes are waves of one, so ds_append /
/// ds_consume add / subtract 1 per active lane and return the pre-op value
/// (v_mbcnt of a lane mask is 0 in this model, so per-lane slots come out
/// unique and dense as on hardware).
fn gds(tr: &mut T, ins: &Inst) -> R<()> {
    let u = tr.t.u32;
    let Some(gds) = tr.gds else { return Err(format!("{:#x}: GDS binding missing", ins.pc)) };
    let op = ins.op;
    match op {
        0..=12 | 32..=44 | 13 | 54 | 61..=63 => {}
        24..=29 => {
            tr.warn(format!("{:#x}: {} (global wave sync) ignored", ins.pc, ins.name()));
            return Ok(());
        }
        _ => {
            tr.warn(format!("{:#x}: unsupported GDS {}", ins.pc, ins.name()));
            return Ok(());
        }
    }
    if op == 63 {
        tr.warn(format!("{:#x}: ds_ordered_count approximated by an unordered atomic add", ins.pc));
    }
    let off16 = ins.ds_offset0() | (ins.ds_offset1() << 8);
    let m0 = tr.sreg(crate::OP_M0 as u32)?;
    let lo16 = tr.u(0xFFFF);
    let base = e(tr.b.bitwise_and(u, None, m0, lo16))?;
    let k = tr.u(off16);
    let mut addr = e(tr.b.i_add(u, None, base, k))?;
    if !matches!(op, 61..=63) {
        let va = tr.vreg(ins.ds_addr() as u32)?;
        addr = e(tr.b.i_add(u, None, addr, va))?;
    }
    let two = tr.u(2);
    let dw = e(tr.b.shift_right_logical(u, None, addr, two))?;
    let lim = tr.u(super::GDS_BYTES / 4 - 1);
    let dw = e(tr.b.bitwise_and(u, None, dw, lim))?;
    let z = tr.u(0);
    let p = e(tr.b.access_chain(tr.t.sb_u32, None, gds, [z, dw]))?;
    let vd = ins.vdst() as u32;
    let d0 = ins.ds_data0() as u32;
    let act = tr.active()?;
    let mut unsupported = false;
    guarded(tr, act, &mut |tr| {
        let one = tr.u(1);
        let ret = match op {
            62 => ds_atomic(tr, 0, p, spirv::Scope::Device, one)?,
            61 => ds_atomic(tr, 1, p, spirv::Scope::Device, one)?,
            63 => {
                let v = tr.vreg(d0)?;
                ds_atomic(tr, 0, p, spirv::Scope::Device, v)?
            }
            13 => {
                let v = tr.vreg(d0)?;
                tr.b.store(p, v, None, []).map_err(|x| format!("{x:?}"))?;
                None
            }
            54 => Some(e(tr.b.load(u, None, p, None, []))?),
            _ => {
                let rtn = op >= 32;
                let v = tr.vreg(d0)?;
                match ds_atomic(tr, if rtn { op - 32 } else { op }, p, spirv::Scope::Device, v)? {
                    Some(old) => rtn.then_some(old),
                    None => {
                        unsupported = true;
                        None
                    }
                }
            }
        };
        if let Some(r) = ret {
            tr.set_vreg_raw(vd, r)?;
        }
        Ok(())
    })?;
    if unsupported {
        tr.warn(format!("{:#x}: unsupported GDS {}", ins.pc, ins.name()));
    }
    Ok(())
}

/// ds_swizzle_b32 (reads no LDS memory): offset bit 15 selects quad
/// permute mode (lane i of each quad reads lane offset[2i+1:2i]); otherwise
/// lanes read lane ((id & and) | or) ^ xor within groups of 32.
fn swizzle(tr: &mut T, ins: &Inst) -> R<()> {
    let u = tr.t.u32;
    let off = ins.ds_offset0() | (ins.ds_offset1() << 8);
    let v = tr.vreg(ins.ds_data0() as u32)?;
    let sg = tr.u(spirv::Scope::Subgroup as u32);
    tr.b.capability(Capability::GroupNonUniform);
    let r = if off & 0x8000 != 0 {
        tr.b.capability(Capability::GroupNonUniformQuad);
        let sel: [u32; 4] = std::array::from_fn(|i| (off >> (2 * i)) & 3);
        let mut bc: [Option<Word>; 4] = [None; 4];
        for &s in &sel {
            if bc[s as usize].is_none() {
                let idx = tr.u(s);
                bc[s as usize] = Some(e(tr.b.group_non_uniform_quad_broadcast(u, None, sg, v, idx))?);
            }
        }
        if sel.iter().all(|&s| s == sel[0]) {
            bc[sel[0] as usize].unwrap()
        } else {
            let lane = tr.subgroup_lane()?;
            let three = tr.u(3);
            let q = e(tr.b.bitwise_and(u, None, lane, three))?;
            let mut r = bc[sel[3] as usize].unwrap();
            for i in (0..3).rev() {
                let ki = tr.u(i as u32);
                let c = e(tr.b.i_equal(tr.t.bool_, None, q, ki))?;
                r = e(tr.b.select(u, None, c, bc[sel[i] as usize].unwrap(), r))?;
            }
            r
        }
    } else {
        tr.b.capability(Capability::GroupNonUniformShuffle);
        let (and, or, xor) = (off & 0x1F, (off >> 5) & 0x1F, (off >> 10) & 0x1F);
        let lane = tr.subgroup_lane()?;
        let m31 = tr.u(31);
        let l = e(tr.b.bitwise_and(u, None, lane, m31))?;
        let hi_mask = tr.u(!31);
        let hi = e(tr.b.bitwise_and(u, None, lane, hi_mask))?;
        let ka = tr.u(and);
        let ko = tr.u(or);
        let kx = tr.u(xor);
        let s = e(tr.b.bitwise_and(u, None, l, ka))?;
        let s = e(tr.b.bitwise_or(u, None, s, ko))?;
        let s = e(tr.b.bitwise_xor(u, None, s, kx))?;
        let src = e(tr.b.bitwise_or(u, None, hi, s))?;
        e(tr.b.group_non_uniform_shuffle(u, None, sg, v, src))?
    };
    tr.set_vreg(ins.vdst() as u32, r)
}

/// Dwords of the per-invocation stand-in for a graphics wave's LDS.
pub const LANE_LDS_DWORDS: u32 = 256;

/// Private-array index of LDS byte address `byte` for one lane: 256-byte
/// rows (one dword per lane of a 64-lane wave) map to two entries each, the
/// second for the odd dword (64-bit spills: slot k at k*512 + l*8). Distinct
/// for the addresses a single lane uses in these layouts.
fn lane_lds_index(tr: &mut T, byte: Word) -> R<Word> {
    let u = tr.t.u32;
    let (k8, k2, k1, mask) = (tr.u(8), tr.u(2), tr.u(1), tr.u(LANE_LDS_DWORDS - 1));
    let row = e(tr.b.shift_right_logical(u, None, byte, k8))?;
    let row2 = e(tr.b.shift_left_logical(u, None, row, k1))?;
    let dw = e(tr.b.shift_right_logical(u, None, byte, k2))?;
    let odd = e(tr.b.bitwise_and(u, None, dw, k1))?;
    let i = e(tr.b.bitwise_or(u, None, row2, odd))?;
    e(tr.b.bitwise_and(u, None, i, mask))
}

/// Whether a DS opcode reads and/or writes LDS memory (atomics, exchanges,
/// append/consume: both; ds_swizzle: neither, it moves VGPRs).
pub fn ds_access_kind(op: u16) -> (bool, bool) {
    match op {
        13 | 14 | 15 | 30 | 31 | 77 | 78 | 79 | 222 | 223 => (false, true),
        54..=60 | 118..=120 | 254 | 255 => (true, false),
        53 => (false, false),
        _ => (true, true),
    }
}

pub fn ds(tr: &mut T, ins: &Inst) -> R<()> {
    if ins.ds_gds() {
        return gds(tr, ins);
    }
    if ins.op == 53 {
        return swizzle(tr, ins);
    }
    let (r, w) = ds_access_kind(ins.op);
    tr.lds_access(r, w)?;
    let u = tr.t.u32;
    // Graphics stages use their LDS as per-lane scratch (the shader
    // compiler's VGPR spills: slot k of lane l at k*256 + l*4): a private
    // array per invocation, indexed by slot (see `lane_lds_index`).
    let lane_lds = tr.lds.is_none() && tr.spec.stage != super::Stage::Compute && matches!(ins.op, 13 | 14 | 15 | 54..=56 | 77..=79 | 118..=120 | 222 | 223 | 254 | 255);
    let lds = match tr.lds {
        Some(l) => l,
        None if lane_lds => tr.lane_lds_var(),
        None => {
            tr.warn(format!("{:#x}: {} without LDS", ins.pc, ins.name()));
            return Ok(());
        }
    };
    let p_wg = tr.b.type_pointer(None, StorageClass::Workgroup, u);
    let p_priv = tr.b.type_pointer(None, StorageClass::Private, u);
    let addr = tr.vreg(ins.ds_addr() as u32)?;
    let off16 = ins.ds_offset0() | (ins.ds_offset1() << 8);
    let two = tr.u(2);
    // Tessellation stages: this threadgroup's region of the LDS buffer.
    let lds_base = tr.lds_base;
    let elem = |tr: &mut T, byte: Word| -> R<Word> {
        if lane_lds {
            let i = lane_lds_index(tr, byte)?;
            return e(tr.b.access_chain(p_priv, None, lds, [i]));
        }
        let i = e(tr.b.shift_right_logical(u, None, byte, two))?;
        match lds_base {
            Some(base) => {
                let i = e(tr.b.i_add(u, None, base, i))?;
                let z = tr.u(0);
                e(tr.b.access_chain(tr.t.sb_u32, None, lds, [z, i]))
            }
            None => e(tr.b.access_chain(p_wg, None, lds, [i])),
        }
    };
    let atomic_scope = if lds_base.is_some() { spirv::Scope::Device } else { spirv::Scope::Workgroup };
    let add = |tr: &mut T, a: Word, k: u32| -> R<Word> {
        let kk = tr.u(k);
        e(tr.b.i_add(u, None, a, kk))
    };
    let vd = ins.vdst() as u32;
    let d0 = ins.ds_data0() as u32;
    let d1 = ins.ds_data1() as u32;
    let act = tr.active()?;
    match ins.op {
        13 | 77 | 223 | 222 => {
            let n = match ins.op {
                13 => 1,
                77 => 2,
                222 => 3,
                _ => 4,
            };
            guarded(tr, act, &mut |tr| {
                let a = add(tr, addr, off16)?;
                for k in 0..n {
                    let ak = add(tr, a, k * 4)?;
                    let p = elem(tr, ak)?;
                    let v = tr.vreg(d0 + k)?;
                    tr.b.store(p, v, None, []).map_err(|x| format!("{x:?}"))?;
                }
                Ok(())
            })?;
        }
        14 | 15 | 78 | 79 => {
            let st = if ins.op == 15 || ins.op == 79 { 64 } else { 1 };
            let wide = ins.op >= 78;
            let esz = if wide { 8 } else { 4 };
            guarded(tr, act, &mut |tr| {
                for (k, (off, src)) in [(ins.ds_offset0(), d0), (ins.ds_offset1(), d1)].into_iter().enumerate() {
                    let _ = k;
                    let a = add(tr, addr, off * esz * st)?;
                    for w in 0..(esz / 4) {
                        let aw = add(tr, a, w * 4)?;
                        let p = elem(tr, aw)?;
                        let v = tr.vreg(src + w)?;
                        tr.b.store(p, v, None, []).map_err(|x| format!("{x:?}"))?;
                    }
                }
                Ok(())
            })?;
        }
        54 | 118 | 254 | 255 => {
            let n = match ins.op {
                54 => 1,
                118 => 2,
                254 => 3,
                _ => 4,
            };
            let a = add(tr, addr, off16)?;
            for k in 0..n {
                let ak = add(tr, a, k * 4)?;
                let p = elem(tr, ak)?;
                let v = e(tr.b.load(u, None, p, None, []))?;
                tr.set_vreg(vd + k, v)?;
            }
        }
        55 | 56 | 119 | 120 => {
            let st = if ins.op == 56 || ins.op == 120 { 64 } else { 1 };
            let wide = ins.op >= 119;
            let esz = if wide { 8 } else { 4 };
            let mut dst = vd;
            for off in [ins.ds_offset0(), ins.ds_offset1()] {
                let a = add(tr, addr, off * esz * st)?;
                for w in 0..(esz / 4) {
                    let aw = add(tr, a, w * 4)?;
                    let p = elem(tr, aw)?;
                    let v = e(tr.b.load(u, None, p, None, []))?;
                    tr.set_vreg(dst, v)?;
                    dst += 1;
                }
            }
        }
        0..=12 | 32..=44 => {
            // Only active lanes perform the atomic.
            let rtn = ins.op >= 32;
            let base = if rtn { ins.op - 32 } else { ins.op };
            let mut unsupported = false;
            guarded(tr, act, &mut |tr| {
                let a = add(tr, addr, off16)?;
                let p = elem(tr, a)?;
                let v = tr.vreg(d0)?;
                match ds_atomic(tr, base, p, atomic_scope, v)? {
                    Some(old) if rtn => tr.set_vreg_raw(vd, old)?,
                    Some(_) => {}
                    None => unsupported = true,
                }
                Ok(())
            })?;
            if unsupported {
                tr.warn(format!("{:#x}: unsupported {}", ins.pc, ins.name()));
            }
        }
        _ => tr.warn(format!("{:#x}: unsupported {}", ins.pc, ins.name())),
    }
    Ok(())
}
