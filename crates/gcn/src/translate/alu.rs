//! Scalar and vector ALU semantics.

use super::{R, T, e};
use crate::{Enc, Inst};
use rspirv::spirv::{GlslStd450Op as GLOp, Word};

// ---------------------------------------------------------------------------
// Scalar

fn s64(tr: &mut T, ins: &Inst, op: u16) -> R<(Word, Word)> {
    if op < 128 {
        return Ok((tr.sreg(op as u32)?, tr.sreg(op as u32 + 1)?));
    }
    let lo = tr.operand(ins, op)?;
    // Inline integer constants are sign-extended; literals/float constants zero-extended.
    let hi = if (193..=208).contains(&op) { tr.u(0xFFFF_FFFF) } else { tr.u(0) };
    Ok((lo, hi))
}

fn set_s64(tr: &mut T, d: u16, lo: Word, hi: Word) -> R<()> {
    tr.set_sreg(d as u32, lo)?;
    tr.set_sreg(d as u32 + 1, hi)
}

fn to64(tr: &mut T, lo: Word, hi: Word) -> R<Word> {
    let v = e(tr.b.composite_construct(tr.t.v2u, None, [lo, hi]))?;
    e(tr.b.bitcast(tr.t.u64, None, v))
}

fn from64(tr: &mut T, v: Word) -> R<(Word, Word)> {
    let p = e(tr.b.bitcast(tr.t.v2u, None, v))?;
    Ok((e(tr.b.composite_extract(tr.t.u32, None, p, [0]))?, e(tr.b.composite_extract(tr.t.u32, None, p, [1]))?))
}

fn nonzero(tr: &mut T, v: Word) -> R<Word> {
    let z = tr.u(0);
    e(tr.b.i_not_equal(tr.t.bool_, None, v, z))
}

fn nonzero64(tr: &mut T, lo: Word, hi: Word) -> R<Word> {
    let o = e(tr.b.bitwise_or(tr.t.u32, None, lo, hi))?;
    nonzero(tr, o)
}

/// Whole-quad mode: each 4-bit group becomes 0xF if any bit is set.
fn wqm32(tr: &mut T, v: Word) -> R<Word> {
    let mut acc = tr.u(0);
    for q in 0..8 {
        let m = tr.u(0xF << (q * 4));
        let g = e(tr.b.bitwise_and(tr.t.u32, None, v, m))?;
        let nz = nonzero(tr, g)?;
        let z = tr.u(0);
        let sel = e(tr.b.select(tr.t.u32, None, nz, m, z))?;
        acc = e(tr.b.bitwise_or(tr.t.u32, None, acc, sel))?;
    }
    Ok(acc)
}

pub fn scalar(tr: &mut T, ins: &Inst) -> R<()> {
    let u = tr.t.u32;
    let bt = tr.t.bool_;
    match ins.enc {
        Enc::Sopp => {
            match ins.op {
                0 | 12..=15 | 16 | 17 | 19..=22 => {} // nop, waitcnt, sethalt, sleep, setprio, sendmsg, cache/perf hints
                10 => {
                    // s_barrier
                    if tr.spec.stage == super::Stage::Compute {
                        let wg = tr.u(rspirv::spirv::Scope::Workgroup as u32);
                        // (Tessellation stages keep their LDS in a storage buffer.)
                        let mem = if tr.lds_base.is_some() { rspirv::spirv::MemorySemantics::UNIFORM_MEMORY } else { rspirv::spirv::MemorySemantics::WORKGROUP_MEMORY };
                        let sem = tr.u((rspirv::spirv::MemorySemantics::ACQUIRE_RELEASE | mem).bits());
                        tr.b.control_barrier(wg, wg, sem).map_err(|x| format!("{x:?}"))?;
                    }
                }
                _ => tr.warn(format!("{:#x}: unsupported {}", ins.pc, ins.name())),
            }
            Ok(())
        }
        Enc::Sopk => {
            let d = ins.sdst();
            let imm_s = ins.simm16() as i16 as i32 as u32;
            let imm_u = ins.simm16() as u32;
            match ins.op {
                0 => {
                    let v = tr.u(imm_s);
                    tr.set_sreg(d as u32, v)?;
                }
                2 => {
                    let c = tr.get_scc()?;
                    let old = tr.sreg(d as u32)?;
                    let v = tr.u(imm_s);
                    let r = e(tr.b.select(u, None, c, v, old))?;
                    tr.set_sreg(d as u32, r)?;
                }
                3..=14 => {
                    let a = tr.sreg(d as u32)?;
                    let signed = ins.op <= 8;
                    let k = if signed { tr.u(imm_s) } else { tr.u(imm_u) };
                    let c = int_cmp(tr, a, k, (ins.op - if signed { 3 } else { 9 }) as u8, signed)?;
                    tr.set_scc(c)?;
                }
                15 => {
                    let a = tr.sreg(d as u32)?;
                    let k = tr.u(imm_s);
                    let r = e(tr.b.i_add(u, None, a, k))?;
                    let ov = add_overflow(tr, a, k, r)?;
                    tr.set_scc(ov)?;
                    tr.set_sreg(d as u32, r)?;
                }
                16 => {
                    let a = tr.sreg(d as u32)?;
                    let k = tr.u(imm_s);
                    let r = e(tr.b.i_mul(u, None, a, k))?;
                    tr.set_sreg(d as u32, r)?;
                }
                18 => {
                    tr.warn(format!("{:#x}: s_getreg_b32 returns 0", ins.pc));
                    let z = tr.u(0);
                    tr.set_sreg(d as u32, z)?;
                }
                19 | 21 => {} // s_setreg: mode/trap registers are not modeled
                _ => tr.warn(format!("{:#x}: unsupported {}", ins.pc, ins.name())),
            }
            Ok(())
        }
        Enc::Sopc => {
            let a = tr.operand(ins, ins.ssrc0())?;
            let b = tr.operand(ins, ins.ssrc1())?;
            let c = match ins.op {
                0..=5 => int_cmp(tr, a, b, ins.op as u8, true)?,
                6..=11 => int_cmp(tr, a, b, (ins.op - 6) as u8, false)?,
                12 | 13 => {
                    let one = tr.u(1);
                    let sh = e(tr.b.shift_left_logical(u, None, one, b))?;
                    let m = e(tr.b.bitwise_and(u, None, a, sh))?;
                    let nz = nonzero(tr, m)?;
                    if ins.op == 12 { e(tr.b.logical_not(bt, None, nz))? } else { nz }
                }
                _ => {
                    tr.warn(format!("{:#x}: unsupported {}", ins.pc, ins.name()));
                    tr.bool_c(false)
                }
            };
            tr.set_scc(c)
        }
        Enc::Sop1 => sop1(tr, ins),
        Enc::Sop2 => sop2(tr, ins),
        _ => Ok(()),
    }
}

/// cmp: 0 eq, 1 ne(lg), 2 gt, 3 ge, 4 lt, 5 le.
fn int_cmp(tr: &mut T, a: Word, b: Word, cmp: u8, signed: bool) -> R<Word> {
    let bt = tr.t.bool_;
    Ok(match (cmp, signed) {
        (0, _) => e(tr.b.i_equal(bt, None, a, b))?,
        (1, _) => e(tr.b.i_not_equal(bt, None, a, b))?,
        (2, true) => e(tr.b.s_greater_than(bt, None, a, b))?,
        (3, true) => e(tr.b.s_greater_than_equal(bt, None, a, b))?,
        (4, true) => e(tr.b.s_less_than(bt, None, a, b))?,
        (5, true) => e(tr.b.s_less_than_equal(bt, None, a, b))?,
        (2, false) => e(tr.b.u_greater_than(bt, None, a, b))?,
        (3, false) => e(tr.b.u_greater_than_equal(bt, None, a, b))?,
        (4, false) => e(tr.b.u_less_than(bt, None, a, b))?,
        _ => e(tr.b.u_less_than_equal(bt, None, a, b))?,
    })
}

/// Signed overflow of r = a + b.
fn add_overflow(tr: &mut T, a: Word, b: Word, r: Word) -> R<Word> {
    let u = tr.t.u32;
    let x1 = e(tr.b.bitwise_xor(u, None, a, r))?;
    let x2 = e(tr.b.bitwise_xor(u, None, b, r))?;
    let m = e(tr.b.bitwise_and(u, None, x1, x2))?;
    let s = tr.u(31);
    let top = e(tr.b.shift_right_logical(u, None, m, s))?;
    nonzero(tr, top)
}

fn sop1(tr: &mut T, ins: &Inst) -> R<()> {
    let u = tr.t.u32;
    let bt = tr.t.bool_;
    let d = ins.sdst();
    let s = ins.ssrc0();
    match ins.op {
        3 => {
            let v = tr.operand(ins, s)?;
            tr.set_sreg(d as u32, v)?;
        }
        4 => {
            let (lo, hi) = s64(tr, ins, s)?;
            set_s64(tr, d, lo, hi)?;
        }
        5 | 6 => {
            let c = tr.get_scc()?;
            if ins.op == 5 {
                let v = tr.operand(ins, s)?;
                let old = tr.sreg(d as u32)?;
                let r = e(tr.b.select(u, None, c, v, old))?;
                tr.set_sreg(d as u32, r)?;
            } else {
                let (lo, hi) = s64(tr, ins, s)?;
                let (olo, ohi) = (tr.sreg(d as u32)?, tr.sreg(d as u32 + 1)?);
                let rl = e(tr.b.select(u, None, c, lo, olo))?;
                let rh = e(tr.b.select(u, None, c, hi, ohi))?;
                set_s64(tr, d, rl, rh)?;
            }
        }
        7 => {
            let v = tr.operand(ins, s)?;
            let r = e(tr.b.not(u, None, v))?;
            let nz = nonzero(tr, r)?;
            tr.set_scc(nz)?;
            tr.set_sreg(d as u32, r)?;
        }
        8 => {
            let (lo, hi) = s64(tr, ins, s)?;
            let rl = e(tr.b.not(u, None, lo))?;
            let rh = e(tr.b.not(u, None, hi))?;
            let nz = nonzero64(tr, rl, rh)?;
            tr.set_scc(nz)?;
            set_s64(tr, d, rl, rh)?;
        }
        9 | 10 => {
            let (lo, hi) = s64(tr, ins, s)?;
            let rl = wqm32(tr, lo)?;
            let rh = if ins.op == 10 { wqm32(tr, hi)? } else { tr.u(0) };
            let nz = nonzero64(tr, rl, rh)?;
            tr.set_scc(nz)?;
            if ins.op == 10 { set_s64(tr, d, rl, rh)? } else { tr.set_sreg(d as u32, rl)? }
        }
        11 => {
            let v = tr.operand(ins, s)?;
            let r = e(tr.b.bit_reverse(u, None, v))?;
            tr.set_sreg(d as u32, r)?;
        }
        13 | 15 => {
            let v = tr.operand(ins, s)?;
            let x = if ins.op == 13 { e(tr.b.not(u, None, v))? } else { v };
            let r = e(tr.b.bit_count(u, None, x))?;
            let nz = nonzero(tr, r)?;
            tr.set_scc(nz)?;
            tr.set_sreg(d as u32, r)?;
        }
        14 | 16 => {
            let (lo, hi) = s64(tr, ins, s)?;
            let (lo, hi) = if ins.op == 14 { (e(tr.b.not(u, None, lo))?, e(tr.b.not(u, None, hi))?) } else { (lo, hi) };
            let a = e(tr.b.bit_count(u, None, lo))?;
            let b = e(tr.b.bit_count(u, None, hi))?;
            let r = e(tr.b.i_add(u, None, a, b))?;
            let nz = nonzero(tr, r)?;
            tr.set_scc(nz)?;
            tr.set_sreg(d as u32, r)?;
        }
        17 | 19 => {
            // s_ff0/ff1_i32_b32: index of first 0/1 bit (or -1)
            let v = tr.operand(ins, s)?;
            let x = if ins.op == 17 { e(tr.b.not(u, None, v))? } else { v };
            let r = tr.glsl1(tr.t.i32, GLOp::FindILsb, x)?;
            let r = tr.to_u(r)?;
            tr.set_sreg(d as u32, r)?;
        }
        18 | 20 => {
            let (lo, hi) = s64(tr, ins, s)?;
            let (lo, hi) = if ins.op == 18 { (e(tr.b.not(u, None, lo))?, e(tr.b.not(u, None, hi))?) } else { (lo, hi) };
            let a = tr.glsl1(tr.t.i32, GLOp::FindILsb, lo)?;
            let b = tr.glsl1(tr.t.i32, GLOp::FindILsb, hi)?;
            let a = tr.to_u(a)?;
            let b = tr.to_u(b)?;
            let k32 = tr.u(32);
            let b32 = e(tr.b.i_add(u, None, b, k32))?;
            let neg1 = tr.u(u32::MAX);
            let bvalid = e(tr.b.i_not_equal(bt, None, b, neg1))?;
            let bres = e(tr.b.select(u, None, bvalid, b32, neg1))?;
            let avalid = e(tr.b.i_not_equal(bt, None, a, neg1))?;
            let r = e(tr.b.select(u, None, avalid, a, bres))?;
            tr.set_sreg(d as u32, r)?;
        }
        21 => {
            // s_flbit_i32_b32: leading zeros count from MSB, -1 if 0
            let v = tr.operand(ins, s)?;
            let m = tr.glsl1(tr.t.i32, GLOp::FindUMsb, v)?;
            let m = tr.to_u(m)?;
            let k31 = tr.u(31);
            let r = e(tr.b.i_sub(u, None, k31, m))?;
            let z = tr.u(0);
            let isz = e(tr.b.i_equal(bt, None, v, z))?;
            let neg1 = tr.u(u32::MAX);
            let r = e(tr.b.select(u, None, isz, neg1, r))?;
            tr.set_sreg(d as u32, r)?;
        }
        25 | 26 => {
            let v = tr.operand(ins, s)?;
            let z = tr.u(0);
            let w = tr.u(if ins.op == 25 { 8 } else { 16 });
            let r = e(tr.b.bit_field_s_extract(u, None, v, z, w))?;
            tr.set_sreg(d as u32, r)?;
        }
        27 | 29 => {
            let v = tr.operand(ins, s)?;
            let one = tr.u(1);
            let k = tr.u(31);
            let idx = e(tr.b.bitwise_and(u, None, v, k))?;
            let bit = e(tr.b.shift_left_logical(u, None, one, idx))?;
            let old = tr.sreg(d as u32)?;
            let r = if ins.op == 27 {
                let nb = e(tr.b.not(u, None, bit))?;
                e(tr.b.bitwise_and(u, None, old, nb))?
            } else {
                e(tr.b.bitwise_or(u, None, old, bit))?
            };
            tr.set_sreg(d as u32, r)?;
        }
        31 => {
            // s_getpc_b64: address of the next instruction, supplied per draw
            // (`Plan::inst_pc`): the program's SPIR-V does not depend on
            // where the title loaded it (pipelines stay cached across runs).
            let Some(&(l, h)) = tr.plan.inst_pc.get(&ins.pc) else { return Err(format!("{:#x}: s_getpc_b64 without its value", ins.pc)) };
            let base = super::values_offset(tr.plan);
            let lo = tr.sgpr_data(base + l as u32)?;
            let hi = tr.sgpr_data(base + h as u32)?;
            set_s64(tr, d, lo, hi)?;
        }
        36..=43 => {
            // s_*_saveexec_b64: d = exec; exec = op(src, exec); scc = exec != 0
            let (slo, shi) = s64(tr, ins, s)?;
            let elo = tr.sreg(crate::OP_EXEC_LO as u32)?;
            let ehi = tr.sreg(crate::OP_EXEC_HI as u32)?;
            let f = |tr: &mut T, a: Word, x: Word| -> R<Word> {
                Ok(match ins.op {
                    36 => e(tr.b.bitwise_and(u, None, a, x))?,
                    37 => e(tr.b.bitwise_or(u, None, a, x))?,
                    38 => e(tr.b.bitwise_xor(u, None, a, x))?,
                    39 => {
                        let nx = e(tr.b.not(u, None, x))?;
                        e(tr.b.bitwise_and(u, None, a, nx))?
                    }
                    40 => {
                        let nx = e(tr.b.not(u, None, x))?;
                        e(tr.b.bitwise_or(u, None, a, nx))?
                    }
                    41 => {
                        let t = e(tr.b.bitwise_and(u, None, a, x))?;
                        e(tr.b.not(u, None, t))?
                    }
                    42 => {
                        let t = e(tr.b.bitwise_or(u, None, a, x))?;
                        e(tr.b.not(u, None, t))?
                    }
                    _ => {
                        let t = e(tr.b.bitwise_xor(u, None, a, x))?;
                        e(tr.b.not(u, None, t))?
                    }
                })
            };
            let nlo = f(tr, slo, elo)?;
            let nhi = f(tr, shi, ehi)?;
            set_s64(tr, d, elo, ehi)?;
            tr.set_sreg(crate::OP_EXEC_LO as u32, nlo)?;
            tr.set_sreg(crate::OP_EXEC_HI as u32, nhi)?;
            let nz = nonzero64(tr, nlo, nhi)?;
            tr.set_scc(nz)?;
        }
        52 => {
            let v = tr.operand(ins, s)?;
            let i = tr.to_i(v)?;
            let a = tr.glsl1(tr.t.i32, GLOp::SAbs, i)?;
            let r = tr.to_u(a)?;
            let nz = nonzero(tr, r)?;
            tr.set_scc(nz)?;
            tr.set_sreg(d as u32, r)?;
        }
        _ => tr.warn(format!("{:#x}: unsupported {}", ins.pc, ins.name())),
    }
    Ok(())
}

fn sop2(tr: &mut T, ins: &Inst) -> R<()> {
    let u = tr.t.u32;
    let bt = tr.t.bool_;
    let d = ins.sdst();
    if crate::sop_is_64(Enc::Sop2, ins.op) {
        let (alo, ahi) = s64(tr, ins, ins.ssrc0())?;
        let shift_src = matches!(ins.op, 31 | 33 | 35 | 41 | 42);
        let (blo, bhi) = if shift_src { (tr.operand(ins, ins.ssrc1())?, tr.u(0)) } else { s64(tr, ins, ins.ssrc1())? };
        let (rl, rh, set_scc) = match ins.op {
            11 => {
                let c = tr.get_scc()?;
                (e(tr.b.select(u, None, c, alo, blo))?, e(tr.b.select(u, None, c, ahi, bhi))?, false)
            }
            15 | 17 | 19 | 21 | 23 | 25 | 27 | 29 => {
                let f = |tr: &mut T, a: Word, b: Word| -> R<Word> {
                    Ok(match ins.op {
                        15 => e(tr.b.bitwise_and(u, None, a, b))?,
                        17 => e(tr.b.bitwise_or(u, None, a, b))?,
                        19 => e(tr.b.bitwise_xor(u, None, a, b))?,
                        21 => {
                            let nb = e(tr.b.not(u, None, b))?;
                            e(tr.b.bitwise_and(u, None, a, nb))?
                        }
                        23 => {
                            let nb = e(tr.b.not(u, None, b))?;
                            e(tr.b.bitwise_or(u, None, a, nb))?
                        }
                        25 => {
                            let t = e(tr.b.bitwise_and(u, None, a, b))?;
                            e(tr.b.not(u, None, t))?
                        }
                        27 => {
                            let t = e(tr.b.bitwise_or(u, None, a, b))?;
                            e(tr.b.not(u, None, t))?
                        }
                        _ => {
                            let t = e(tr.b.bitwise_xor(u, None, a, b))?;
                            e(tr.b.not(u, None, t))?
                        }
                    })
                };
                (f(tr, alo, blo)?, f(tr, ahi, bhi)?, true)
            }
            31 | 33 | 35 => {
                let a = to64(tr, alo, ahi)?;
                let k = tr.u(63);
                let sh = e(tr.b.bitwise_and(u, None, blo, k))?;
                let r = match ins.op {
                    31 => e(tr.b.shift_left_logical(tr.t.u64, None, a, sh))?,
                    33 => e(tr.b.shift_right_logical(tr.t.u64, None, a, sh))?,
                    _ => e(tr.b.shift_right_arithmetic(tr.t.u64, None, a, sh))?,
                };
                let (l, h) = from64(tr, r)?;
                (l, h, true)
            }
            37 => {
                // s_bfm_b64: ((1 << size) - 1) << offset
                let a = to64(tr, alo, ahi)?;
                let one64 = {
                    let o = tr.u(1);
                    let z = tr.u(0);
                    to64(tr, o, z)?
                };
                let k = tr.u(63);
                let size = e(tr.b.bitwise_and(u, None, alo, k))?;
                let off = e(tr.b.bitwise_and(u, None, blo, k))?;
                let m = e(tr.b.shift_left_logical(tr.t.u64, None, one64, size))?;
                let m = e(tr.b.i_sub(tr.t.u64, None, m, one64))?;
                let r = e(tr.b.shift_left_logical(tr.t.u64, None, m, off))?;
                let _ = a;
                let (l, h) = from64(tr, r)?;
                (l, h, false)
            }
            41 | 42 => {
                let a = to64(tr, alo, ahi)?;
                let k63 = tr.u(63);
                let k127 = tr.u(0x7F);
                let off = e(tr.b.bitwise_and(u, None, blo, k63))?;
                let s16 = tr.u(16);
                let wsh = e(tr.b.shift_right_logical(u, None, blo, s16))?;
                let w = e(tr.b.bitwise_and(u, None, wsh, k127))?;
                let r = if ins.op == 41 {
                    e(tr.b.bit_field_u_extract(tr.t.u64, None, a, off, w))?
                } else {
                    e(tr.b.bit_field_s_extract(tr.t.u64, None, a, off, w))?
                };
                let (l, h) = from64(tr, r)?;
                (l, h, true)
            }
            _ => {
                tr.warn(format!("{:#x}: unsupported {}", ins.pc, ins.name()));
                (tr.u(0), tr.u(0), false)
            }
        };
        if set_scc {
            let nz = nonzero64(tr, rl, rh)?;
            tr.set_scc(nz)?;
        }
        return set_s64(tr, d, rl, rh);
    }
    let a = tr.operand(ins, ins.ssrc0())?;
    let b = tr.operand(ins, ins.ssrc1())?;
    let k31 = tr.u(31);
    let r: Word = match ins.op {
        0 => {
            let r = e(tr.b.i_add(u, None, a, b))?;
            let c = e(tr.b.u_less_than(bt, None, r, a))?;
            tr.set_scc(c)?;
            r
        }
        1 => {
            let r = e(tr.b.i_sub(u, None, a, b))?;
            let c = e(tr.b.u_less_than(bt, None, a, b))?;
            tr.set_scc(c)?;
            r
        }
        2 => {
            let r = e(tr.b.i_add(u, None, a, b))?;
            let ov = add_overflow(tr, a, b, r)?;
            tr.set_scc(ov)?;
            r
        }
        3 => {
            let r = e(tr.b.i_sub(u, None, a, b))?;
            let nb = e(tr.b.not(u, None, b))?;
            let one = tr.u(1);
            let nb1 = e(tr.b.i_add(u, None, nb, one))?;
            let ov = add_overflow(tr, a, nb1, r)?;
            tr.set_scc(ov)?;
            r
        }
        4 | 5 => {
            let c = tr.get_scc()?;
            let ci = tr.bool_to_u(c)?;
            if ins.op == 4 {
                let t = e(tr.b.i_add(u, None, a, b))?;
                let r = e(tr.b.i_add(u, None, t, ci))?;
                let c1 = e(tr.b.u_less_than(bt, None, t, a))?;
                let c2 = e(tr.b.u_less_than(bt, None, r, t))?;
                let co = e(tr.b.logical_or(bt, None, c1, c2))?;
                tr.set_scc(co)?;
                r
            } else {
                let t = e(tr.b.i_sub(u, None, a, b))?;
                let r = e(tr.b.i_sub(u, None, t, ci))?;
                let c1 = e(tr.b.u_less_than(bt, None, a, b))?;
                let c2 = e(tr.b.u_less_than(bt, None, t, ci))?;
                let co = e(tr.b.logical_or(bt, None, c1, c2))?;
                tr.set_scc(co)?;
                r
            }
        }
        6..=9 => {
            let signed = ins.op == 6 || ins.op == 8;
            let lt = int_cmp(tr, a, b, 4, signed)?;
            let gt = int_cmp(tr, a, b, 2, signed)?;
            let pick_a = if ins.op <= 7 { lt } else { gt };
            tr.set_scc(pick_a)?;
            e(tr.b.select(u, None, pick_a, a, b))?
        }
        10 => {
            let c = tr.get_scc()?;
            e(tr.b.select(u, None, c, a, b))?
        }
        14 | 16 | 18 | 20 | 22 | 24 | 26 | 28 => {
            let r = match ins.op {
                14 => e(tr.b.bitwise_and(u, None, a, b))?,
                16 => e(tr.b.bitwise_or(u, None, a, b))?,
                18 => e(tr.b.bitwise_xor(u, None, a, b))?,
                20 => {
                    let nb = e(tr.b.not(u, None, b))?;
                    e(tr.b.bitwise_and(u, None, a, nb))?
                }
                22 => {
                    let nb = e(tr.b.not(u, None, b))?;
                    e(tr.b.bitwise_or(u, None, a, nb))?
                }
                24 => {
                    let t = e(tr.b.bitwise_and(u, None, a, b))?;
                    e(tr.b.not(u, None, t))?
                }
                26 => {
                    let t = e(tr.b.bitwise_or(u, None, a, b))?;
                    e(tr.b.not(u, None, t))?
                }
                _ => {
                    let t = e(tr.b.bitwise_xor(u, None, a, b))?;
                    e(tr.b.not(u, None, t))?
                }
            };
            let nz = nonzero(tr, r)?;
            tr.set_scc(nz)?;
            r
        }
        30 | 32 | 34 => {
            let sh = e(tr.b.bitwise_and(u, None, b, k31))?;
            let r = match ins.op {
                30 => e(tr.b.shift_left_logical(u, None, a, sh))?,
                32 => e(tr.b.shift_right_logical(u, None, a, sh))?,
                _ => e(tr.b.shift_right_arithmetic(u, None, a, sh))?,
            };
            let nz = nonzero(tr, r)?;
            tr.set_scc(nz)?;
            r
        }
        36 => {
            // s_bfm_b32: ((1 << a) - 1) << b
            let one = tr.u(1);
            let sa = e(tr.b.bitwise_and(u, None, a, k31))?;
            let sb = e(tr.b.bitwise_and(u, None, b, k31))?;
            let m = e(tr.b.shift_left_logical(u, None, one, sa))?;
            let m = e(tr.b.i_sub(u, None, m, one))?;
            e(tr.b.shift_left_logical(u, None, m, sb))?
        }
        38 => e(tr.b.i_mul(u, None, a, b))?,
        39 | 40 => {
            let off = e(tr.b.bitwise_and(u, None, b, k31))?;
            let s16 = tr.u(16);
            let wsh = e(tr.b.shift_right_logical(u, None, b, s16))?;
            let k7f = tr.u(0x7F);
            let w = e(tr.b.bitwise_and(u, None, wsh, k7f))?;
            let r = if ins.op == 39 { e(tr.b.bit_field_u_extract(u, None, a, off, w))? } else { e(tr.b.bit_field_s_extract(u, None, a, off, w))? };
            let nz = nonzero(tr, r)?;
            tr.set_scc(nz)?;
            r
        }
        44 => {
            let ai = tr.to_i(a)?;
            let bi = tr.to_i(b)?;
            let df = e(tr.b.i_sub(tr.t.i32, None, ai, bi))?;
            let ab = tr.glsl1(tr.t.i32, GLOp::SAbs, df)?;
            let r = tr.to_u(ab)?;
            let nz = nonzero(tr, r)?;
            tr.set_scc(nz)?;
            r
        }
        _ => {
            tr.warn(format!("{:#x}: unsupported {}", ins.pc, ins.name()));
            return Ok(());
        }
    };
    tr.set_sreg(d as u32, r)
}

// ---------------------------------------------------------------------------
// Vector

struct V {
    op: u16,
    dst: u16,
    src: [u16; 3],
    abs: u32,
    neg: u32,
    clamp: bool,
    omod: u32,
    vop3: bool,
}

fn norm(ins: &Inst) -> V {
    match ins.enc {
        Enc::Vop3 => V {
            op: ins.op,
            dst: ins.vdst(),
            src: [ins.src(0), ins.src(1), ins.src(2)],
            abs: if ins.is_vop3b() { 0 } else { ins.vop3_abs() },
            neg: ins.vop3_neg(),
            clamp: !ins.is_vop3b() && ins.vop3_clamp(),
            omod: ins.vop3_omod(),
            vop3: true,
        },
        _ => V { op: ins.vop3_equiv().unwrap(), dst: if ins.enc == Enc::Vopc { crate::OP_VCC_LO } else { ins.vdst() }, src: [ins.src(0), ins.src(1), 0], abs: 0, neg: 0, clamp: false, omod: 0, vop3: false },
    }
}

fn fsrc(tr: &mut T, ins: &Inst, v: &V, k: usize) -> R<Word> {
    let raw = tr.operand(ins, v.src[k])?;
    let mut x = tr.to_f(raw)?;
    if v.abs >> k & 1 != 0 {
        x = tr.glsl1(tr.t.f32, GLOp::FAbs, x)?;
    }
    if v.neg >> k & 1 != 0 {
        x = e(tr.b.f_negate(tr.t.f32, None, x))?;
    }
    Ok(x)
}

fn usrc(tr: &mut T, ins: &Inst, v: &V, k: usize) -> R<Word> {
    tr.operand(ins, v.src[k])
}

/// 64-bit vector-instruction source (VGPR pair, SGPR pair or constant).
fn vsrc64(tr: &mut T, ins: &Inst, op: u16) -> R<(Word, Word)> {
    if op >= 256 {
        let r = (op - 256) as u32;
        return Ok((tr.vreg(r)?, tr.vreg(r + 1)?));
    }
    s64(tr, ins, op)
}

fn fout(tr: &mut T, v: &V, x: Word) -> R<()> {
    let mut x = x;
    match v.omod {
        1 => {
            let k = tr.f(2.0);
            x = e(tr.b.f_mul(tr.t.f32, None, x, k))?;
        }
        2 => {
            let k = tr.f(4.0);
            x = e(tr.b.f_mul(tr.t.f32, None, x, k))?;
        }
        3 => {
            let k = tr.f(0.5);
            x = e(tr.b.f_mul(tr.t.f32, None, x, k))?;
        }
        _ => {}
    }
    if v.clamp {
        let z = tr.f(0.0);
        let o = tr.f(1.0);
        x = tr.glsl3(tr.t.f32, GLOp::FClamp, x, z, o)?;
    }
    let u = tr.to_u(x)?;
    tr.set_vreg(v.dst as u32, u)
}

fn uout(tr: &mut T, v: &V, x: Word) -> R<()> {
    tr.set_vreg(v.dst as u32, x)
}

/// Writes a per-lane boolean to an SGPR pair (VCC or VOP3 sdst).
fn write_mask(tr: &mut T, sdst: u16, c: Word) -> R<()> {
    let act = tr.active()?;
    let c = e(tr.b.logical_and(tr.t.bool_, None, c, act))?;
    let v = tr.bool_to_u(c)?;
    let z = tr.u(0);
    tr.set_sreg(sdst as u32, v)?;
    tr.set_sreg(sdst as u32 + 1, z)
}

fn read_mask(tr: &mut T, ins: &Inst, op: u16) -> R<Word> {
    let lo = tr.operand(ins, op)?;
    let one = tr.u(1);
    let b = e(tr.b.bitwise_and(tr.t.u32, None, lo, one))?;
    nonzero(tr, b)
}

fn fcmp(tr: &mut T, a: Word, b: Word, c: u16) -> R<Word> {
    let bt = tr.t.bool_;
    Ok(match c & 15 {
        0 => tr.bool_c(false),
        1 => e(tr.b.f_ord_less_than(bt, None, a, b))?,
        2 => e(tr.b.f_ord_equal(bt, None, a, b))?,
        3 => e(tr.b.f_ord_less_than_equal(bt, None, a, b))?,
        4 => e(tr.b.f_ord_greater_than(bt, None, a, b))?,
        5 => e(tr.b.f_ord_not_equal(bt, None, a, b))?,
        6 => e(tr.b.f_ord_greater_than_equal(bt, None, a, b))?,
        7 => e(tr.b.ordered(bt, None, a, b))?,
        8 => e(tr.b.unordered(bt, None, a, b))?,
        9 => e(tr.b.f_unord_less_than(bt, None, a, b))?,
        10 => e(tr.b.f_unord_equal(bt, None, a, b))?,
        11 => e(tr.b.f_unord_less_than_equal(bt, None, a, b))?,
        12 => e(tr.b.f_unord_greater_than(bt, None, a, b))?,
        13 => e(tr.b.f_unord_not_equal(bt, None, a, b))?,
        14 => e(tr.b.f_unord_greater_than_equal(bt, None, a, b))?,
        _ => tr.bool_c(true),
    })
}

/// v_cmp_class_f32: tests floating-point class bits.
fn fclass(tr: &mut T, x: Word, mask: Word) -> R<Word> {
    let bt = tr.t.bool_;
    let u = tr.t.u32;
    let bits = tr.to_u(x)?;
    let abs = tr.glsl1(tr.t.f32, GLOp::FAbs, x)?;
    let inf = tr.f(f32::INFINITY);
    let minn = tr.f(f32::MIN_POSITIVE);
    let zero = tr.f(0.0);
    let s31 = tr.u(31);
    let sign_b = e(tr.b.shift_right_logical(u, None, bits, s31))?;
    let one = tr.u(1);
    let neg = e(tr.b.i_equal(bt, None, sign_b, one))?;
    let pos = e(tr.b.logical_not(bt, None, neg))?;
    let nan = e(tr.b.is_nan(bt, None, x))?;
    let qbit = tr.u(1 << 22);
    let qm = e(tr.b.bitwise_and(u, None, bits, qbit))?;
    let quiet = nonzero(tr, qm)?;
    let snan = {
        let nq = e(tr.b.logical_not(bt, None, quiet))?;
        e(tr.b.logical_and(bt, None, nan, nq))?
    };
    let qnan = e(tr.b.logical_and(bt, None, nan, quiet))?;
    let isinf = e(tr.b.f_ord_equal(bt, None, abs, inf))?;
    let iszero = e(tr.b.f_ord_equal(bt, None, abs, zero))?;
    let normal = {
        let ge = e(tr.b.f_ord_greater_than_equal(bt, None, abs, minn))?;
        let ni = e(tr.b.logical_not(bt, None, isinf))?;
        e(tr.b.logical_and(bt, None, ge, ni))?
    };
    let denorm = {
        let lt = e(tr.b.f_ord_less_than(bt, None, abs, minn))?;
        let nz = e(tr.b.logical_not(bt, None, iszero))?;
        e(tr.b.logical_and(bt, None, lt, nz))?
    };
    let classes = [
        snan,
        qnan,
        e(tr.b.logical_and(bt, None, neg, isinf))?,
        e(tr.b.logical_and(bt, None, neg, normal))?,
        e(tr.b.logical_and(bt, None, neg, denorm))?,
        e(tr.b.logical_and(bt, None, neg, iszero))?,
        e(tr.b.logical_and(bt, None, pos, iszero))?,
        e(tr.b.logical_and(bt, None, pos, denorm))?,
        e(tr.b.logical_and(bt, None, pos, normal))?,
        e(tr.b.logical_and(bt, None, pos, isinf))?,
    ];
    let mut acc = tr.bool_c(false);
    for (i, c) in classes.iter().enumerate() {
        let bit = tr.u(1 << i);
        let mb = e(tr.b.bitwise_and(u, None, mask, bit))?;
        let on = nonzero(tr, mb)?;
        let t = e(tr.b.logical_and(bt, None, on, *c))?;
        acc = e(tr.b.logical_or(bt, None, acc, t))?;
    }
    Ok(acc)
}

fn lane_masks(tr: &mut T) -> R<(Word, Word)> {
    // Masks of lanes below this one (lane id from LocalInvocationIndex for compute).
    let u = tr.t.u32;
    let lane = match tr.lane_id {
        Some(l) => l,
        None => tr.u(0),
    };
    let one = tr.u(1);
    let k32 = tr.u(32);
    let bt = tr.t.bool_;
    let lo_full = e(tr.b.u_greater_than_equal(bt, None, lane, k32))?;
    let sh_lo = tr.glsl2(u, GLOp::UMin, lane, k32)?;
    let k31 = tr.u(31);
    let shl = e(tr.b.bitwise_and(u, None, sh_lo, k31))?;
    let m = e(tr.b.shift_left_logical(u, None, one, shl))?;
    let m = e(tr.b.i_sub(u, None, m, one))?;
    let all = tr.u(u32::MAX);
    let lo = e(tr.b.select(u, None, lo_full, all, m))?;
    let hi_sh = e(tr.b.i_sub(u, None, lane, k32))?;
    let hi_sh = e(tr.b.bitwise_and(u, None, hi_sh, k31))?;
    let hm = e(tr.b.shift_left_logical(u, None, one, hi_sh))?;
    let hm = e(tr.b.i_sub(u, None, hm, one))?;
    let z = tr.u(0);
    let hi = e(tr.b.select(u, None, lo_full, hm, z))?;
    Ok((lo, hi))
}

pub fn vector(tr: &mut T, ins: &Inst) -> R<()> {
    let v = norm(ins);
    let f = tr.t.f32;
    let u = tr.t.u32;
    let i = tr.t.i32;
    let bt = tr.t.bool_;
    // Compares.
    if v.op < 0x100 {
        let sdst = if v.vop3 { v.dst } else { crate::OP_VCC_LO };
        let hi = v.op >> 4;
        let lo = v.op & 15;
        let c = match hi {
            0 | 1 | 4 | 5 => {
                let a = fsrc(tr, ins, &v, 0)?;
                let b = fsrc(tr, ins, &v, 1)?;
                fcmp(tr, a, b, lo)?
            }
            8..=0xF if lo == 8 && hi <= 9 => {
                let a = fsrc(tr, ins, &v, 0)?;
                let m = usrc(tr, ins, &v, 1)?;
                fclass(tr, a, m)?
            }
            8 | 9 | 0xC | 0xD if lo < 8 => {
                let a = usrc(tr, ins, &v, 0)?;
                let b = usrc(tr, ins, &v, 1)?;
                let signed = hi <= 9;
                match lo {
                    0 => tr.bool_c(false),
                    7 => tr.bool_c(true),
                    1 => int_cmp(tr, a, b, 4, signed)?,
                    2 => int_cmp(tr, a, b, 0, signed)?,
                    3 => int_cmp(tr, a, b, 5, signed)?,
                    4 => int_cmp(tr, a, b, 2, signed)?,
                    5 => int_cmp(tr, a, b, 1, signed)?,
                    _ => int_cmp(tr, a, b, 3, signed)?,
                }
            }
            _ => {
                tr.warn(format!("{:#x}: unsupported compare {}", ins.pc, ins.name()));
                tr.bool_c(false)
            }
        };
        write_mask(tr, sdst, c)?;
        let is_x = matches!(hi, 1 | 3 | 5 | 7 | 9 | 0xB | 0xD | 0xF);
        if is_x {
            let act = tr.active()?;
            let c2 = e(tr.b.logical_and(bt, None, c, act))?;
            let nv = tr.bool_to_u(c2)?;
            let z = tr.u(0);
            tr.set_sreg(crate::OP_EXEC_LO as u32, nv)?;
            tr.set_sreg(crate::OP_EXEC_HI as u32, z)?;
        }
        return Ok(());
    }
    match v.op {
        // ---- VOP2-derived (0x100 + op)
        0x100 => {
            // v_cndmask_b32
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let c = if v.vop3 { read_mask(tr, ins, v.src[2])? } else { read_mask(tr, ins, crate::OP_VCC_LO)? };
            let r = e(tr.b.select(u, None, c, b, a))?;
            uout(tr, &v, r)?;
        }
        0x101 => {
            // v_readlane_b32 sdst, vsrc, lane: a lane written by v_writelane
            // (SGPR spill) reads its slot, otherwise this lane's value.
            let slot = ins.const_lane_op().and_then(|(_, g, l)| tr.lane_slots.get(&(g as u32, l)).copied());
            let a = match slot {
                Some(var) => e(tr.b.load(u, None, var, None, []))?,
                None => usrc(tr, ins, &v, 0)?,
            };
            tr.set_sreg(v.dst as u32, a)?;
        }
        0x102 => {
            // v_writelane_b32 vdst, ssrc, lane: writes only that lane, so the
            // VGPR itself changes only in the invocation emulating that lane.
            let a = usrc(tr, ins, &v, 0)?;
            match ins.const_lane_op() {
                Some((_, g, l)) => {
                    if let Some(&var) = tr.lane_slots.get(&(g as u32, l)) {
                        tr.b.store(var, a, None, []).map_err(|x| format!("{x:?}"))?;
                    }
                    if let Some(lid) = tr.lane_id {
                        let kl = tr.u(l);
                        let here = e(tr.b.i_equal(tr.t.bool_, None, lid, kl))?;
                        let old = tr.vreg(v.dst as u32)?;
                        let nv = e(tr.b.select(u, None, here, a, old))?;
                        tr.set_vreg_raw(v.dst as u32, nv)?;
                    }
                }
                None => tr.set_vreg_raw(v.dst as u32, a)?,
            }
        }
        0x103 | 0x104 | 0x105 | 0x108 | 0x10F | 0x110 | 0x10D | 0x10E | 0x107 => {
            let a = fsrc(tr, ins, &v, 0)?;
            let b = fsrc(tr, ins, &v, 1)?;
            let r = match v.op {
                0x103 => e(tr.b.f_add(f, None, a, b))?,
                0x104 => e(tr.b.f_sub(f, None, a, b))?,
                0x105 => e(tr.b.f_sub(f, None, b, a))?,
                0x108 => e(tr.b.f_mul(f, None, a, b))?,
                0x107 => legacy_mul(tr, a, b)?,
                0x10F | 0x10D => tr.glsl2(f, GLOp::FMin, a, b)?,
                _ => tr.glsl2(f, GLOp::FMax, a, b)?,
            };
            fout(tr, &v, r)?;
        }
        0x106 | 0x11F => {
            // v_mac_legacy_f32 / v_mac_f32: d = a*b + d
            let a = fsrc(tr, ins, &v, 0)?;
            let b = fsrc(tr, ins, &v, 1)?;
            let dv = tr.vreg(v.dst as u32)?;
            let d = tr.to_f(dv)?;
            let m = if v.op == 0x106 { legacy_mul(tr, a, b)? } else { e(tr.b.f_mul(f, None, a, b))? };
            let r = e(tr.b.f_add(f, None, m, d))?;
            fout(tr, &v, r)?;
        }
        0x120 | 0x121 => {
            // v_madmk_f32 d = a*K + b ; v_madak_f32 d = a*b + K
            let a = fsrc(tr, ins, &v, 0)?;
            let b = fsrc(tr, ins, &v, 1)?;
            let k = tr.f(f32::from_bits(ins.lit.unwrap_or(0)));
            let r = if v.op == 0x120 {
                let m = e(tr.b.f_mul(f, None, a, k))?;
                e(tr.b.f_add(f, None, m, b))?
            } else {
                let m = e(tr.b.f_mul(f, None, a, b))?;
                e(tr.b.f_add(f, None, m, k))?
            };
            fout(tr, &v, r)?;
        }
        0x109..=0x10C => {
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let z = tr.u(0);
            let w = tr.u(24);
            let (ea, eb) = if v.op <= 0x10A {
                (e(tr.b.bit_field_s_extract(u, None, a, z, w))?, e(tr.b.bit_field_s_extract(u, None, b, z, w))?)
            } else {
                (e(tr.b.bit_field_u_extract(u, None, a, z, w))?, e(tr.b.bit_field_u_extract(u, None, b, z, w))?)
            };
            let r = if v.op == 0x109 || v.op == 0x10B {
                e(tr.b.i_mul(u, None, ea, eb))?
            } else {
                mul_hi(tr, ea, eb, v.op == 0x10A)?
            };
            uout(tr, &v, r)?;
        }
        0x111..=0x114 => {
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let r = match v.op {
                0x111 => {
                    let (x, y) = (tr.to_i(a)?, tr.to_i(b)?);
                    let m = tr.glsl2(i, GLOp::SMin, x, y)?;
                    tr.to_u(m)?
                }
                0x112 => {
                    let (x, y) = (tr.to_i(a)?, tr.to_i(b)?);
                    let m = tr.glsl2(i, GLOp::SMax, x, y)?;
                    tr.to_u(m)?
                }
                0x113 => tr.glsl2(u, GLOp::UMin, a, b)?,
                _ => tr.glsl2(u, GLOp::UMax, a, b)?,
            };
            uout(tr, &v, r)?;
        }
        0x115..=0x11E => {
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let k31 = tr.u(31);
            let r = match v.op {
                0x115 => {
                    let s = e(tr.b.bitwise_and(u, None, b, k31))?;
                    e(tr.b.shift_right_logical(u, None, a, s))?
                }
                0x116 => {
                    let s = e(tr.b.bitwise_and(u, None, a, k31))?;
                    e(tr.b.shift_right_logical(u, None, b, s))?
                }
                0x117 => {
                    let s = e(tr.b.bitwise_and(u, None, b, k31))?;
                    e(tr.b.shift_right_arithmetic(u, None, a, s))?
                }
                0x118 => {
                    let s = e(tr.b.bitwise_and(u, None, a, k31))?;
                    e(tr.b.shift_right_arithmetic(u, None, b, s))?
                }
                0x119 => {
                    let s = e(tr.b.bitwise_and(u, None, b, k31))?;
                    e(tr.b.shift_left_logical(u, None, a, s))?
                }
                0x11A => {
                    let s = e(tr.b.bitwise_and(u, None, a, k31))?;
                    e(tr.b.shift_left_logical(u, None, b, s))?
                }
                0x11B => e(tr.b.bitwise_and(u, None, a, b))?,
                0x11C => e(tr.b.bitwise_or(u, None, a, b))?,
                0x11D => e(tr.b.bitwise_xor(u, None, a, b))?,
                _ => {
                    // v_bfm_b32
                    let one = tr.u(1);
                    let sa = e(tr.b.bitwise_and(u, None, a, k31))?;
                    let sb = e(tr.b.bitwise_and(u, None, b, k31))?;
                    let m = e(tr.b.shift_left_logical(u, None, one, sa))?;
                    let m = e(tr.b.i_sub(u, None, m, one))?;
                    e(tr.b.shift_left_logical(u, None, m, sb))?
                }
            };
            uout(tr, &v, r)?;
        }
        0x122 => {
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let c = e(tr.b.bit_count(u, None, a))?;
            let r = e(tr.b.i_add(u, None, c, b))?;
            uout(tr, &v, r)?;
        }
        0x123 | 0x124 => {
            // v_mbcnt_lo/hi_u32_b32: popcount(mask & lanes_below) + src1.
            // Lane masks (EXEC/VCC copies) hold only this lane's bit: in a
            // wave of one no active lane is below, so they count 0. Other
            // masks (-1 for the lane id) use the hardware lane index.
            let m = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let one = tr.u(1);
            let lane_mask = e(tr.b.u_less_than_equal(tr.t.bool_, None, m, one))?;
            let zero = tr.u(0);
            let m = e(tr.b.select(u, None, lane_mask, zero, m))?;
            let (lo, hi) = lane_masks(tr)?;
            let lm = if v.op == 0x123 { lo } else { hi };
            let x = e(tr.b.bitwise_and(u, None, m, lm))?;
            let c = e(tr.b.bit_count(u, None, x))?;
            let r = e(tr.b.i_add(u, None, c, b))?;
            uout(tr, &v, r)?;
        }
        0x125..=0x12A => {
            // add/sub with carry: VOP2 uses VCC, VOP3b uses sdst/src2
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let carry_out = if v.vop3 { ins.sdst() } else { crate::OP_VCC_LO };
            let cin = if v.op >= 0x128 {
                let c = if v.vop3 { read_mask(tr, ins, v.src[2])? } else { read_mask(tr, ins, crate::OP_VCC_LO)? };
                Some(tr.bool_to_u(c)?)
            } else {
                None
            };
            let (x, y) = match v.op {
                0x125 | 0x128 => (a, b),
                0x126 | 0x129 => (a, b),
                _ => (b, a),
            };
            let (r, co) = if matches!(v.op, 0x125 | 0x128) {
                let t = e(tr.b.i_add(u, None, x, y))?;
                let c1 = e(tr.b.u_less_than(bt, None, t, x))?;
                match cin {
                    Some(ci) => {
                        let r = e(tr.b.i_add(u, None, t, ci))?;
                        let c2 = e(tr.b.u_less_than(bt, None, r, t))?;
                        (r, e(tr.b.logical_or(bt, None, c1, c2))?)
                    }
                    None => (t, c1),
                }
            } else {
                let t = e(tr.b.i_sub(u, None, x, y))?;
                let c1 = e(tr.b.u_less_than(bt, None, x, y))?;
                match cin {
                    Some(ci) => {
                        let r = e(tr.b.i_sub(u, None, t, ci))?;
                        let c2 = e(tr.b.u_less_than(bt, None, t, ci))?;
                        (r, e(tr.b.logical_or(bt, None, c1, c2))?)
                    }
                    None => (t, c1),
                }
            };
            uout(tr, &v, r)?;
            write_mask(tr, carry_out, co)?;
        }
        0x12B => {
            let a = fsrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let bi = tr.to_i(b)?;
            let r = tr.glsl2(f, GLOp::Ldexp, a, bi)?;
            fout(tr, &v, r)?;
        }
        0x12D | 0x12E | 0x12F => {
            let a = fsrc(tr, ins, &v, 0)?;
            let b = fsrc(tr, ins, &v, 1)?;
            let vv = e(tr.b.composite_construct(tr.t.v2f, None, [a, b]))?;
            let op = match v.op {
                0x12D => GLOp::PackSnorm2x16,
                0x12E => GLOp::PackUnorm2x16,
                _ => GLOp::PackHalf2x16,
            };
            let r = tr.glsl1(u, op, vv)?;
            uout(tr, &v, r)?;
        }
        0x130 | 0x131 => {
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let m = tr.u(0xFFFF);
            let s16 = tr.u(16);
            let (a, b) = if v.op == 0x130 {
                (tr.glsl2(u, GLOp::UMin, a, m)?, tr.glsl2(u, GLOp::UMin, b, m)?)
            } else {
                let lo = tr.i(-32768);
                let hi = tr.i(32767);
                let (ai, bi) = (tr.to_i(a)?, tr.to_i(b)?);
                let ca = tr.glsl3(i, GLOp::SClamp, ai, lo, hi)?;
                let cb = tr.glsl3(i, GLOp::SClamp, bi, lo, hi)?;
                (tr.to_u(ca)?, tr.to_u(cb)?)
            };
            let al = e(tr.b.bitwise_and(u, None, a, m))?;
            let bl = e(tr.b.bitwise_and(u, None, b, m))?;
            let bh = e(tr.b.shift_left_logical(u, None, bl, s16))?;
            let r = e(tr.b.bitwise_or(u, None, al, bh))?;
            uout(tr, &v, r)?;
        }
        // ---- VOP3-only
        0x140 | 0x141 | 0x14B => {
            let a = fsrc(tr, ins, &v, 0)?;
            let b = fsrc(tr, ins, &v, 1)?;
            let c = fsrc(tr, ins, &v, 2)?;
            let r = match v.op {
                0x140 => {
                    let m = legacy_mul(tr, a, b)?;
                    e(tr.b.f_add(f, None, m, c))?
                }
                0x141 => {
                    let m = e(tr.b.f_mul(f, None, a, b))?;
                    e(tr.b.f_add(f, None, m, c))?
                }
                _ => tr.glsl3(f, GLOp::Fma, a, b, c)?,
            };
            fout(tr, &v, r)?;
        }
        0x142 | 0x143 => {
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let c = usrc(tr, ins, &v, 2)?;
            let z = tr.u(0);
            let w = tr.u(24);
            let (ea, eb) = if v.op == 0x142 {
                (e(tr.b.bit_field_s_extract(u, None, a, z, w))?, e(tr.b.bit_field_s_extract(u, None, b, z, w))?)
            } else {
                (e(tr.b.bit_field_u_extract(u, None, a, z, w))?, e(tr.b.bit_field_u_extract(u, None, b, z, w))?)
            };
            let m = e(tr.b.i_mul(u, None, ea, eb))?;
            let r = e(tr.b.i_add(u, None, m, c))?;
            uout(tr, &v, r)?;
        }
        0x144..=0x147 => {
            let x = fsrc(tr, ins, &v, 0)?;
            let y = fsrc(tr, ins, &v, 1)?;
            let z = fsrc(tr, ins, &v, 2)?;
            let r = cube_op(tr, v.op, x, y, z)?;
            fout(tr, &v, r)?;
        }
        0x148 | 0x149 => {
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let c = usrc(tr, ins, &v, 2)?;
            let k31 = tr.u(31);
            let off = e(tr.b.bitwise_and(u, None, b, k31))?;
            let w = e(tr.b.bitwise_and(u, None, c, k31))?;
            let r = if v.op == 0x148 { e(tr.b.bit_field_u_extract(u, None, a, off, w))? } else { e(tr.b.bit_field_s_extract(u, None, a, off, w))? };
            uout(tr, &v, r)?;
        }
        0x14A => {
            let m = usrc(tr, ins, &v, 0)?;
            let a = usrc(tr, ins, &v, 1)?;
            let b = usrc(tr, ins, &v, 2)?;
            let x = e(tr.b.bitwise_and(u, None, m, a))?;
            let nm = e(tr.b.not(u, None, m))?;
            let y = e(tr.b.bitwise_and(u, None, nm, b))?;
            let r = e(tr.b.bitwise_or(u, None, x, y))?;
            uout(tr, &v, r)?;
        }
        0x14E | 0x14F => {
            // v_alignbit_b32 / v_alignbyte_b32: ({a,b} >> shift)[31:0]
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let c = usrc(tr, ins, &v, 2)?;
            let wide = to64(tr, b, a)?;
            let k = tr.u(if v.op == 0x14E { 31 } else { 3 });
            let mut sh = e(tr.b.bitwise_and(u, None, c, k))?;
            if v.op == 0x14F {
                let k3 = tr.u(3);
                sh = e(tr.b.shift_left_logical(u, None, sh, k3))?;
            }
            let r64 = e(tr.b.shift_right_logical(tr.t.u64, None, wide, sh))?;
            let r = e(tr.b.u_convert(u, None, r64))?;
            uout(tr, &v, r)?;
        }
        0x151..=0x159 => {
            let kind = (v.op - 0x151) % 3; // 0 f32, 1 i32, 2 u32
            let which = (v.op - 0x151) / 3; // 0 min3, 1 max3, 2 med3
            let r = if kind == 0 {
                let a = fsrc(tr, ins, &v, 0)?;
                let b = fsrc(tr, ins, &v, 1)?;
                let c = fsrc(tr, ins, &v, 2)?;
                let r = match which {
                    0 => {
                        let m = tr.glsl2(f, GLOp::FMin, a, b)?;
                        tr.glsl2(f, GLOp::FMin, m, c)?
                    }
                    1 => {
                        let m = tr.glsl2(f, GLOp::FMax, a, b)?;
                        tr.glsl2(f, GLOp::FMax, m, c)?
                    }
                    _ => {
                        let mn = tr.glsl2(f, GLOp::FMin, a, b)?;
                        let mx = tr.glsl2(f, GLOp::FMax, a, b)?;
                        let m2 = tr.glsl2(f, GLOp::FMin, mx, c)?;
                        tr.glsl2(f, GLOp::FMax, mn, m2)?
                    }
                };
                return fout(tr, &v, r);
            } else {
                let a = usrc(tr, ins, &v, 0)?;
                let b = usrc(tr, ins, &v, 1)?;
                let c = usrc(tr, ins, &v, 2)?;
                let (mn, mx, ty) = if kind == 1 { (GLOp::SMin, GLOp::SMax, i) } else { (GLOp::UMin, GLOp::UMax, u) };
                let (a, b, c) = if kind == 1 { (tr.to_i(a)?, tr.to_i(b)?, tr.to_i(c)?) } else { (a, b, c) };
                let r = match which {
                    0 => {
                        let m = tr.glsl2(ty, mn, a, b)?;
                        tr.glsl2(ty, mn, m, c)?
                    }
                    1 => {
                        let m = tr.glsl2(ty, mx, a, b)?;
                        tr.glsl2(ty, mx, m, c)?
                    }
                    _ => {
                        let lo = tr.glsl2(ty, mn, a, b)?;
                        let hi = tr.glsl2(ty, mx, a, b)?;
                        let m2 = tr.glsl2(ty, mn, hi, c)?;
                        tr.glsl2(ty, mx, lo, m2)?
                    }
                };
                if kind == 1 { tr.to_u(r)? } else { r }
            };
            uout(tr, &v, r)?;
        }
        0x15A..=0x15D => {
            // v_sad_u8 / v_sad_hi_u8 / v_sad_u16 / v_sad_u32: sum of absolute
            // differences of the bytes / halves / words of src0 and src1, plus src2
            // (v_sad_hi_u8: the byte sum shifted into the high half).
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let c = usrc(tr, ins, &v, 2)?;
            let absdiff = |tr: &mut T, x: Word, y: Word| -> R<Word> {
                let mx = tr.glsl2(u, GLOp::UMax, x, y)?;
                let mn = tr.glsl2(u, GLOp::UMin, x, y)?;
                e(tr.b.i_sub(u, None, mx, mn))
            };
            let sum = if v.op == 0x15D {
                absdiff(tr, a, b)?
            } else {
                let (n, w) = if v.op == 0x15C { (2u32, 16u32) } else { (4, 8) };
                let mut sum = tr.u(0);
                for k in 0..n {
                    let off = tr.u(k * w);
                    let width = tr.u(w);
                    let xa = e(tr.b.bit_field_u_extract(u, None, a, off, width))?;
                    let xb = e(tr.b.bit_field_u_extract(u, None, b, off, width))?;
                    let d = absdiff(tr, xa, xb)?;
                    sum = e(tr.b.i_add(u, None, sum, d))?;
                }
                if v.op == 0x15B {
                    let s16 = tr.u(16);
                    sum = e(tr.b.shift_left_logical(u, None, sum, s16))?;
                }
                sum
            };
            let r = e(tr.b.i_add(u, None, sum, c))?;
            uout(tr, &v, r)?;
        }
        0x176 | 0x177 => {
            // v_mad_u64_u32 / v_mad_i64_i32: {sdst carry, D.u64} = S0 * S1 (32x32->64) + S2.u64
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let (clo, chi) = vsrc64(tr, ins, v.src[2])?;
            let c = to64(tr, clo, chi)?;
            let widen = |tr: &mut T, x: Word| -> R<Word> {
                if v.op == 0x176 {
                    e(tr.b.u_convert(tr.t.u64, None, x))
                } else {
                    let xi = tr.to_i(x)?;
                    let k31 = tr.u(31);
                    let sign = e(tr.b.shift_right_arithmetic(i, None, xi, k31))?;
                    let hi = tr.to_u(sign)?;
                    to64(tr, x, hi)
                }
            };
            let (a64, b64) = (widen(tr, a)?, widen(tr, b)?);
            let m = e(tr.b.i_mul(tr.t.u64, None, a64, b64))?;
            let r = e(tr.b.i_add(tr.t.u64, None, m, c))?;
            // Carry out of the unsigned 64-bit addition.
            let carry = e(tr.b.u_less_than(bt, None, r, c))?;
            let (l, h) = from64(tr, r)?;
            tr.set_vreg(v.dst as u32, l)?;
            tr.set_vreg(v.dst as u32 + 1, h)?;
            write_mask(tr, ins.sdst(), carry)?;
        }
        0x15E => {
            // v_cvt_pk_u8_f32 d = (src2 & ~(0xff << 8*byte)) | (u8(src0) << 8*byte)
            let x = fsrc(tr, ins, &v, 0)?;
            let by = usrc(tr, ins, &v, 1)?;
            let base = usrc(tr, ins, &v, 2)?;
            let z = tr.f(0.0);
            let m255 = tr.f(255.0);
            let c = tr.glsl3(f, GLOp::FClamp, x, z, m255)?;
            let rn = tr.glsl1(f, GLOp::RoundEven, c)?;
            let q = e(tr.b.convert_f_to_u(u, None, rn))?;
            let k3 = tr.u(3);
            let k8 = tr.u(8);
            let bi = e(tr.b.bitwise_and(u, None, by, k3))?;
            let sh = e(tr.b.i_mul(u, None, bi, k8))?;
            let kff = tr.u(0xFF);
            let mask = e(tr.b.shift_left_logical(u, None, kff, sh))?;
            let nm = e(tr.b.not(u, None, mask))?;
            let kept = e(tr.b.bitwise_and(u, None, base, nm))?;
            let ins_ = e(tr.b.shift_left_logical(u, None, q, sh))?;
            let r = e(tr.b.bitwise_or(u, None, kept, ins_))?;
            uout(tr, &v, r)?;
        }
        0x15F => {
            // v_div_fixup_f32: approximate as src2 / src1 result already computed in src0
            let a = fsrc(tr, ins, &v, 0)?;
            fout(tr, &v, a)?;
        }
        0x161..=0x163 => {
            // v_lshl_b64 / v_lshr_b64 / v_ashr_i64
            let (alo, ahi) = vsrc64(tr, ins, v.src[0])?;
            let a = to64(tr, alo, ahi)?;
            let s = usrc(tr, ins, &v, 1)?;
            let k = tr.u(63);
            let sh = e(tr.b.bitwise_and(u, None, s, k))?;
            let r = match v.op {
                0x161 => e(tr.b.shift_left_logical(tr.t.u64, None, a, sh))?,
                0x162 => e(tr.b.shift_right_logical(tr.t.u64, None, a, sh))?,
                _ => e(tr.b.shift_right_arithmetic(tr.t.u64, None, a, sh))?,
            };
            let (l, h) = from64(tr, r)?;
            tr.set_vreg(v.dst as u32, l)?;
            tr.set_vreg(v.dst as u32 + 1, h)?;
        }
        0x16D => {
            // v_div_scale_f32: no scaling needed in our model; pass src0 through, vcc = 0
            let a = fsrc(tr, ins, &v, 0)?;
            fout(tr, &v, a)?;
            let fl = tr.bool_c(false);
            write_mask(tr, ins.sdst(), fl)?;
        }
        0x16F => {
            // v_div_fmas_f32: fma(a, b, c) (scale factor from vcc ignored)
            let a = fsrc(tr, ins, &v, 0)?;
            let b = fsrc(tr, ins, &v, 1)?;
            let c = fsrc(tr, ins, &v, 2)?;
            let r = tr.glsl3(f, GLOp::Fma, a, b, c)?;
            fout(tr, &v, r)?;
        }
        0x169 | 0x16B => {
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let r = e(tr.b.i_mul(u, None, a, b))?;
            uout(tr, &v, r)?;
        }
        0x16A | 0x16C => {
            let a = usrc(tr, ins, &v, 0)?;
            let b = usrc(tr, ins, &v, 1)?;
            let r = mul_hi(tr, a, b, v.op == 0x16C)?;
            uout(tr, &v, r)?;
        }
        // ---- VOP1-derived (0x180 + op)
        0x180 => {}
        0x181 | 0x189 => {
            let a = usrc(tr, ins, &v, 0)?;
            uout(tr, &v, a)?;
        }
        0x182 => {
            // v_readfirstlane_b32 sdst: this lane's value (values are uniform where used)
            let a = usrc(tr, ins, &v, 0)?;
            tr.set_sreg(v.dst as u32, a)?;
        }
        0x185 | 0x186 => {
            let a = usrc(tr, ins, &v, 0)?;
            let r = if v.op == 0x185 {
                let ai = tr.to_i(a)?;
                e(tr.b.convert_s_to_f(f, None, ai))?
            } else {
                e(tr.b.convert_u_to_f(f, None, a))?
            };
            fout(tr, &v, r)?;
        }
        0x187 | 0x188 | 0x18C | 0x18D => {
            let a = fsrc(tr, ins, &v, 0)?;
            let r = match v.op {
                0x187 => {
                    let z = tr.f(0.0);
                    let mx = tr.f(4294967040.0);
                    let c = tr.glsl3(f, GLOp::FClamp, a, z, mx)?;
                    e(tr.b.convert_f_to_u(u, None, c))?
                }
                0x188 => {
                    let lo = tr.f(-2147483648.0);
                    let hi = tr.f(2147483520.0);
                    let c = tr.glsl3(f, GLOp::FClamp, a, lo, hi)?;
                    let x = e(tr.b.convert_f_to_s(i, None, c))?;
                    tr.to_u(x)?
                }
                0x18C => {
                    let h = tr.f(0.5);
                    let s = e(tr.b.f_add(f, None, a, h))?;
                    let fl = tr.glsl1(f, GLOp::Floor, s)?;
                    let x = e(tr.b.convert_f_to_s(i, None, fl))?;
                    tr.to_u(x)?
                }
                _ => {
                    let fl = tr.glsl1(f, GLOp::Floor, a)?;
                    let x = e(tr.b.convert_f_to_s(i, None, fl))?;
                    tr.to_u(x)?
                }
            };
            uout(tr, &v, r)?;
        }
        0x18A => {
            let a = fsrc(tr, ins, &v, 0)?;
            let z = tr.f(0.0);
            let vv = e(tr.b.composite_construct(tr.t.v2f, None, [a, z]))?;
            let r = tr.glsl1(u, GLOp::PackHalf2x16, vv)?;
            uout(tr, &v, r)?;
        }
        0x18B => {
            let a = usrc(tr, ins, &v, 0)?;
            let vv = tr.glsl1(tr.t.v2f, GLOp::UnpackHalf2x16, a)?;
            let r = e(tr.b.composite_extract(f, None, vv, [0]))?;
            fout(tr, &v, r)?;
        }
        0x18E => {
            // v_cvt_off_f32_i4: low 4 bits as a signed integer, in units of 1/16.
            let a = usrc(tr, ins, &v, 0)?;
            let (z, w4) = (tr.u(0), tr.u(4));
            let x = e(tr.b.bit_field_s_extract(u, None, a, z, w4))?;
            let xi = tr.to_i(x)?;
            let xf = e(tr.b.convert_s_to_f(f, None, xi))?;
            let k = tr.f(1.0 / 16.0);
            let r = e(tr.b.f_mul(f, None, xf, k))?;
            fout(tr, &v, r)?;
        }
        0x191..=0x194 => {
            let a = usrc(tr, ins, &v, 0)?;
            let off = tr.u(((v.op - 0x191) * 8) as u32);
            let w = tr.u(8);
            let x = e(tr.b.bit_field_u_extract(u, None, a, off, w))?;
            let r = e(tr.b.convert_u_to_f(f, None, x))?;
            fout(tr, &v, r)?;
        }
        0x1A0..=0x1AA | 0x1AB | 0x1AC | 0x1AD | 0x1AE | 0x1B3 | 0x1B5 | 0x1B6 | 0x1C5 | 0x1C6 => {
            let a = fsrc(tr, ins, &v, 0)?;
            let r = match v.op {
                0x1A0 => tr.glsl1(f, GLOp::Fract, a)?,
                0x1A1 => tr.glsl1(f, GLOp::Trunc, a)?,
                0x1A2 => tr.glsl1(f, GLOp::Ceil, a)?,
                0x1A3 => tr.glsl1(f, GLOp::RoundEven, a)?,
                0x1A4 => tr.glsl1(f, GLOp::Floor, a)?,
                0x1A5 | 0x1C6 => tr.glsl1(f, GLOp::Exp2, a)?,
                0x1A7 | 0x1C5 => tr.glsl1(f, GLOp::Log2, a)?,
                // v_log_clamp_f32: -inf (log of 0) clamped to -max_float.
                0x1A6 => {
                    let r = tr.glsl1(f, GLOp::Log2, a)?;
                    clamp_inf(tr, r)?
                }
                0x1A8 | 0x1A9 | 0x1AA | 0x1AB => {
                    let one = tr.f(1.0);
                    let r = e(tr.b.f_div(f, None, one, a))?;
                    // v_rcp_clamp_f32: +-inf clamped to +-max_float.
                    if v.op == 0x1A8 { clamp_inf(tr, r)? } else { r }
                }
                0x1AC | 0x1AD | 0x1AE => {
                    let r = tr.glsl1(f, GLOp::InverseSqrt, a)?;
                    // v_rsq_clamp_f32: rsq(0) is max_float, not inf (an
                    // infinity times 0 later would be NaN).
                    if v.op == 0x1AC { clamp_inf(tr, r)? } else { r }
                }
                0x1B3 => tr.glsl1(f, GLOp::Sqrt, a)?,
                0x1B5 | 0x1B6 => {
                    // Input in revolutions; outside [-256, 256] the result is 0.
                    let two_pi = tr.f(std::f32::consts::TAU);
                    let x = e(tr.b.f_mul(f, None, a, two_pi))?;
                    let r = tr.glsl1(f, if v.op == 0x1B5 { GLOp::Sin } else { GLOp::Cos }, x)?;
                    let bt = tr.t.bool_;
                    let abs = tr.glsl1(f, GLOp::FAbs, a)?;
                    let lim = tr.f(256.0);
                    let out = e(tr.b.f_ord_greater_than(bt, None, abs, lim))?;
                    let z = tr.f(0.0);
                    e(tr.b.select(f, None, out, z, r))?
                }
                _ => a,
            };
            fout(tr, &v, r)?;
        }
        0x1B7 => {
            let a = usrc(tr, ins, &v, 0)?;
            let r = e(tr.b.not(u, None, a))?;
            uout(tr, &v, r)?;
        }
        0x1B8 => {
            let a = usrc(tr, ins, &v, 0)?;
            let r = e(tr.b.bit_reverse(u, None, a))?;
            uout(tr, &v, r)?;
        }
        0x1B9 | 0x1BB => {
            // v_ffbh_u32 / v_ffbh_i32: count from MSB, -1 if none
            let a = usrc(tr, ins, &v, 0)?;
            let m = if v.op == 0x1B9 {
                tr.glsl1(i, GLOp::FindUMsb, a)?
            } else {
                let ai = tr.to_i(a)?;
                tr.glsl1(i, GLOp::FindSMsb, ai)?
            };
            let m = tr.to_u(m)?;
            let k31 = tr.u(31);
            let r = e(tr.b.i_sub(u, None, k31, m))?;
            let neg1 = tr.u(u32::MAX);
            let none = e(tr.b.i_equal(bt, None, m, neg1))?;
            let r = e(tr.b.select(u, None, none, neg1, r))?;
            uout(tr, &v, r)?;
        }
        0x1BA => {
            let a = usrc(tr, ins, &v, 0)?;
            let r = tr.glsl1(i, GLOp::FindILsb, a)?;
            let r = tr.to_u(r)?;
            uout(tr, &v, r)?;
        }
        0x1BF | 0x1C0 => {
            let a = fsrc(tr, ins, &v, 0)?;
            let st = tr.b.type_struct([f, i]);
            let fr = e(tr.b.ext_inst(st, None, tr.glsl, GLOp::FrexpStruct as u32, [rspirv::dr::Operand::IdRef(a)]))?;
            if v.op == 0x1BF {
                let ex = e(tr.b.composite_extract(i, None, fr, [1]))?;
                let r = tr.to_u(ex)?;
                uout(tr, &v, r)?;
            } else {
                let m = e(tr.b.composite_extract(f, None, fr, [0]))?;
                fout(tr, &v, m)?;
            }
        }
        _ => tr.warn(format!("{:#x}: unsupported {} (vop3 op {:#x})", ins.pc, ins.name(), v.op)),
    }
    Ok(())
}

/// +-inf -> +-max_float (the clamping transcendental ops); NaN stays NaN.
fn clamp_inf(tr: &mut T, x: Word) -> R<Word> {
    let f = tr.t.f32;
    let bt = tr.t.bool_;
    let inf = e(tr.b.is_inf(bt, None, x))?;
    let sign = tr.glsl1(f, GLOp::FSign, x)?;
    let max = tr.f(f32::MAX);
    let big = e(tr.b.f_mul(f, None, sign, max))?;
    e(tr.b.select(f, None, inf, big, x))
}

/// DX9-style multiply: 0 * anything = 0.
fn legacy_mul(tr: &mut T, a: Word, b: Word) -> R<Word> {
    let f = tr.t.f32;
    let bt = tr.t.bool_;
    let z = tr.f(0.0);
    let az = e(tr.b.f_ord_equal(bt, None, a, z))?;
    let bz = e(tr.b.f_ord_equal(bt, None, b, z))?;
    let any = e(tr.b.logical_or(bt, None, az, bz))?;
    let m = e(tr.b.f_mul(f, None, a, b))?;
    e(tr.b.select(f, None, any, z, m))
}

fn mul_hi(tr: &mut T, a: Word, b: Word, signed: bool) -> R<Word> {
    let u = tr.t.u32;
    if signed {
        let i = tr.t.i32;
        let st = tr.b.type_struct([i, i]);
        let (ai, bi) = (tr.to_i(a)?, tr.to_i(b)?);
        let r = e(tr.b.s_mul_extended(st, None, ai, bi))?;
        let hi = e(tr.b.composite_extract(i, None, r, [1]))?;
        tr.to_u(hi)
    } else {
        let st = tr.b.type_struct([u, u]);
        let r = e(tr.b.u_mul_extended(st, None, a, b))?;
        e(tr.b.composite_extract(u, None, r, [1]))
    }
}

/// v_cube{id,sc,tc,ma}_f32 per the GCN definition.
fn cube_op(tr: &mut T, op: u16, x: Word, y: Word, z: Word) -> R<Word> {
    let f = tr.t.f32;
    let bt = tr.t.bool_;
    let ax = tr.glsl1(f, GLOp::FAbs, x)?;
    let ay = tr.glsl1(f, GLOp::FAbs, y)?;
    let az = tr.glsl1(f, GLOp::FAbs, z)?;
    let zero = tr.f(0.0);
    let zmaj = {
        let a = e(tr.b.f_ord_greater_than_equal(bt, None, az, ax))?;
        let b = e(tr.b.f_ord_greater_than_equal(bt, None, az, ay))?;
        e(tr.b.logical_and(bt, None, a, b))?
    };
    let ymaj = {
        let a = e(tr.b.f_ord_greater_than_equal(bt, None, ay, ax))?;
        let nz = e(tr.b.logical_not(bt, None, zmaj))?;
        e(tr.b.logical_and(bt, None, a, nz))?
    };
    let zneg = e(tr.b.f_ord_less_than(bt, None, z, zero))?;
    let yneg = e(tr.b.f_ord_less_than(bt, None, y, zero))?;
    let xneg = e(tr.b.f_ord_less_than(bt, None, x, zero))?;
    let nx = e(tr.b.f_negate(f, None, x))?;
    let ny = e(tr.b.f_negate(f, None, y))?;
    let nz = e(tr.b.f_negate(f, None, z))?;
    match op {
        0x144 => {
            let (c0, c1, c2, c3, c4, c5) = (tr.f(0.0), tr.f(1.0), tr.f(2.0), tr.f(3.0), tr.f(4.0), tr.f(5.0));
            let zid = e(tr.b.select(f, None, zneg, c5, c4))?;
            let yid = e(tr.b.select(f, None, yneg, c3, c2))?;
            let xid = e(tr.b.select(f, None, xneg, c1, c0))?;
            let t = e(tr.b.select(f, None, ymaj, yid, xid))?;
            e(tr.b.select(f, None, zmaj, zid, t))
        }
        0x145 => {
            let zsc = e(tr.b.select(f, None, zneg, nx, x))?;
            let xsc = e(tr.b.select(f, None, xneg, z, nz))?;
            let t = e(tr.b.select(f, None, ymaj, x, xsc))?;
            e(tr.b.select(f, None, zmaj, zsc, t))
        }
        0x146 => {
            let ytc = e(tr.b.select(f, None, yneg, nz, z))?;
            let t = e(tr.b.select(f, None, ymaj, ytc, ny))?;
            e(tr.b.select(f, None, zmaj, ny, t))
        }
        _ => {
            let two = tr.f(2.0);
            let m = e(tr.b.select(f, None, ymaj, ay, ax))?;
            let m = e(tr.b.select(f, None, zmaj, az, m))?;
            e(tr.b.f_mul(f, None, m, two))
        }
    }
}
