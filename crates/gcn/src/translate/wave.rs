//! Wave-synchronous translation (`T::wave_sync`) of compute shaders whose
//! workgroup is one GCN wave whose lanes share LDS without `s_barrier`.
//!
//! A wave's 64 lanes run in lockstep on GCN, so such a shader may write LDS in
//! one instruction and read other lanes' entries in the next. Here each lane is
//! an invocation, and the host may run a 64-invocation workgroup as two
//! 32-wide subgroups (NVIDIA): unless a barrier separates such steps, one half
//! can read entries the other half has not written yet. TLG's character
//! occlusion (cs 0x1130f77800: per 32x32-pixel tile a depth min/max reduction
//! and a bitonic sort of up to 128 occluders in LDS, the sort's steps inside
//! loops) sorted garbage in some tiles: square blocks of wrong ambient
//! occlusion around the boy.
//!
//! Barriers must be in uniform control flow, and in this mode it is: the
//! wave-wide branch tests (VCCZ, EXECZ; an invocation holds only its own
//! lane's bit) take a workgroup vote (`T::wave_any`), so every invocation
//! follows the wave's path while its own EXEC bit still masks its writes;
//! branches on SCC must test values computed from registers that are equal in
//! every invocation (`scc_branches_uniform`), else the mode is not used. The
//! LDS barriers (`T::lds_access`) then also go inside loops and branches.

use super::{block_ranges, block_starts};
use crate::{sop_is_64, Enc, Inst, OP_EXEC_LO, OP_SCC, OP_VCCZ, OP_VCC_LO};

/// Slots tracked: SGPRs 0..=127 (VCC, M0 and EXEC among them), then SCC.
const SCC: usize = 128;
type State = [bool; 129];

/// Whether every s_cbranch_scc0/scc1 of `code` tests an SCC computed from
/// registers holding the same value in every invocation that emulates one of
/// the wave's lanes - not derived from EXEC or from vector results (compare
/// masks, carries, readlane), whose bits differ per invocation here. A
/// forward dataflow over the blocks (a register is varying at a point if it
/// is on some path to it). Programs with indexed or computed scalar control
/// (s_movreld, s_setpc, forks) do not qualify.
pub fn scc_branches_uniform(code: &[Inst]) -> bool {
    if code.is_empty() {
        return true;
    }
    let computed = |i: &Inst| match i.enc {
        Enc::Sop1 => matches!(i.op, 32..=34 | 48..=50),
        Enc::Sop2 => i.op == 43,
        Enc::Sopk => i.op == 17,
        _ => false,
    };
    if code.iter().any(computed) {
        return false;
    }
    let starts = block_starts(code);
    let ranges = block_ranges(code, &starts);
    let index: std::collections::HashMap<u32, usize> = starts.iter().enumerate().map(|(k, &pc)| (pc, k)).collect();
    let n = ranges.len();
    let mut entry: Vec<Option<State>> = vec![None; n];
    let mut init = [false; 129];
    init[OP_EXEC_LO as usize] = true;
    init[OP_EXEC_LO as usize + 1] = true;
    entry[0] = Some(init);
    let mut work = vec![0usize];
    while let Some(b) = work.pop() {
        let Some(mut s) = entry[b] else { continue };
        let r = ranges[b].clone();
        for k in r.clone() {
            step(&code[k], &mut s);
        }
        let last = r.end.checked_sub(1).filter(|&k| k >= r.start).map(|k| &code[k]);
        let falls = !last.is_some_and(|i| i.is_endpgm() || (i.enc == Enc::Sopp && i.op == 2));
        let mut succ = Vec::with_capacity(2);
        if let Some(t) = last.and_then(|i| i.branch_target()) {
            match index.get(&t) {
                Some(&k) => succ.push(k),
                None => return false,
            }
        }
        if falls && b + 1 < n {
            succ.push(b + 1);
        }
        for k in succ {
            let merged = match entry[k] {
                None => s,
                Some(old) => std::array::from_fn(|j| old[j] | s[j]),
            };
            if entry[k] != Some(merged) {
                entry[k] = Some(merged);
                work.push(k);
            }
        }
    }
    for (b, r) in ranges.iter().enumerate() {
        let Some(mut s) = entry[b] else { continue };
        for k in r.clone() {
            let i = &code[k];
            if i.enc == Enc::Sopp && matches!(i.op, 4 | 5) && s[SCC] {
                return false;
            }
            step(i, &mut s);
        }
    }
    true
}

/// Whether scalar operand `r` is varying in `s` (constants are not; VCCZ and
/// EXECZ follow their masks).
fn varying(s: &State, r: u16) -> bool {
    match r {
        0..=127 => s[r as usize],
        OP_VCCZ => s[OP_VCC_LO as usize] || s[OP_VCC_LO as usize + 1],
        252 => true, // EXECZ
        OP_SCC => s[SCC],
        _ => false,
    }
}

fn varying2(s: &State, r: u16, wide: bool) -> bool {
    varying(s, r) || (wide && r < 127 && varying(s, r + 1))
}

fn set(s: &mut State, r: u16, wide: bool, x: bool) {
    if r < 128 {
        s[r as usize] = x;
    }
    if wide && r < 127 {
        s[r as usize + 1] = x;
    }
}

/// `s` after `i`: a scalar destination is varying when a source is, or when
/// the instruction makes per-invocation values (vector results into SGPRs,
/// saved EXEC); EXEC itself always is.
fn step(i: &Inst, s: &mut State) {
    match i.enc {
        Enc::Sop2 => {
            let w = sop_is_64(Enc::Sop2, i.op);
            let mut x = varying2(s, i.ssrc0(), w) || varying2(s, i.ssrc1(), w);
            // s_addc, s_subb and s_cselect read SCC.
            if matches!(i.op, 4 | 5 | 10 | 11) {
                x |= s[SCC];
            }
            set(s, i.sdst(), w, x);
            // All but s_cselect, s_bfm, s_mul_i32 and the fork write SCC.
            if !matches!(i.op, 10 | 11 | 36 | 37 | 38 | 43) {
                s[SCC] = x;
            }
        }
        Enc::Sopk => {
            let d = i.sdst();
            match i.op {
                0 | 18 => set(s, d, false, false), // s_movk, s_getreg
                2 => {
                    let x = varying(s, d) || s[SCC]; // s_cmovk
                    set(s, d, false, x);
                }
                3..=14 => s[SCC] = varying(s, d), // s_cmpk
                15 => {
                    let x = varying(s, d); // s_addk
                    s[SCC] = x;
                }
                _ => {} // s_mulk keeps its register's state; s_setreg writes none
            }
        }
        Enc::Sop1 => {
            let w = sop_is_64(Enc::Sop1, i.op);
            let src = varying2(s, i.ssrc0(), w);
            let d = i.sdst();
            match i.op {
                5 | 6 => {
                    let x = src || varying2(s, d, w) || s[SCC]; // s_cmov
                    set(s, d, w, x);
                }
                27..=30 => {
                    let x = varying(s, i.ssrc0()) || varying2(s, d, w); // s_bitset: modify the destination
                    set(s, d, w, x);
                }
                31 => set(s, d, true, false), // s_getpc
                36..=43 => {
                    set(s, d, true, true); // s_*_saveexec: the old EXEC
                    s[SCC] = true;
                }
                46 | 47 => set(s, d, w, true), // s_movrels: an indexed source
                _ => {
                    set(s, d, w, src);
                    // SCC writers: s_not, s_wqm, s_bcnt0/1, s_quadmask, s_abs.
                    if matches!(i.op, 7..=10 | 13..=16 | 44 | 45 | 52) {
                        s[SCC] = src;
                    }
                }
            }
        }
        Enc::Sopc => {
            // All but s_setvskip compare into SCC (s_bitcmp*_b64: a pair).
            if i.op != 16 {
                s[SCC] = varying2(s, i.ssrc0(), matches!(i.op, 14 | 15)) || varying(s, i.ssrc1());
            }
        }
        Enc::Smrd => {
            if !matches!(i.op, 29 | 31) {
                let base = i.smrd_sbase();
                let x = i.op == 30 // s_memtime: a clock each invocation reads
                    || (0..if i.op >= 8 { 4 } else { 2 }).any(|k| varying(s, base + k))
                    || (!i.smrd_imm() && i.smrd_offset() != 255 && varying(s, i.smrd_offset() as u16));
                let count = match i.op {
                    30 => 2,
                    op => 1u16 << (op & 7).min(4),
                };
                for k in 0..count {
                    set(s, i.sdst() + k, false, x);
                }
            }
        }
        // Compare masks and carries (VCC), readlane results.
        Enc::Vopc => set(s, OP_VCC_LO, true, true),
        Enc::Vop2 => {
            if matches!(i.op, 37..=42) {
                set(s, OP_VCC_LO, true, true);
            }
            if i.op == 1 {
                set(s, i.vdst(), false, true);
            }
        }
        Enc::Vop1 => {
            if i.op == 2 {
                set(s, i.vdst(), false, true);
            }
        }
        Enc::Vop3 => {
            if i.op < 0x100 {
                set(s, i.vdst(), true, true);
            } else if i.is_vop3b() {
                set(s, i.sdst(), true, true);
            } else if matches!(i.op, 0x101 | 0x182) {
                set(s, i.vdst(), false, true);
            }
        }
        _ => {}
    }
    s[OP_EXEC_LO as usize] = true;
    s[OP_EXEC_LO as usize + 1] = true;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prog(words: &[u32]) -> Vec<Inst> {
        crate::decode_program(words).unwrap()
    }
    // SOP encodings: SOPK 1011 dddd ddd op5 imm16; SOPC 1011 1111 0 op7 s1 s0;
    // SOP2 10 op7 sdst7 s1 s0; SOPP 1011 1111 1 op7 simm16.
    fn sopk(op: u32, d: u32, imm: u16) -> u32 {
        0xB000_0000 | (op << 23) | (d << 16) | imm as u32
    }
    fn sopc(op: u32, s0: u32, s1: u32) -> u32 {
        0xBF00_0000 | (op << 16) | (s1 << 8) | s0
    }
    fn sop2(op: u32, d: u32, s0: u32, s1: u32) -> u32 {
        0x8000_0000 | (op << 23) | (d << 16) | (s1 << 8) | s0
    }
    fn sopp(op: u32, simm: i16) -> u32 {
        0xBF80_0000 | (op << 16) | simm as u16 as u32
    }
    // VOPC v_cmp_lt_i32 (0x81) of v0 with s6: 0111 110 op8 vsrc1 src0.
    fn vopc_lt(s0: u32, v1: u32) -> u32 {
        0x7C00_0000 | (0x81 << 17) | (v1 << 9) | s0
    }

    #[test]
    fn counter_loops_are_uniform() {
        // s_movk s6, 7; loop: s_add_i32 s6, s6, -1 (inline -1 = 193);
        // s_cmp_ge_i32 s6, 0 (inline 0 = 128); s_cbranch_scc1 loop; s_endpgm.
        let p = prog(&[sopk(0, 6, 7), sop2(2, 6, 6, 193), sopc(3, 6, 128), sopp(5, -3), sopp(1, 0)]);
        assert!(scc_branches_uniform(&p));
    }

    #[test]
    fn scc_from_a_compare_mask_is_varying() {
        // v_cmp_lt_i32 vcc, s6, v0; s_and_b64 s[0:1], vcc, exec (SCC = any);
        // s_cbranch_scc1 +0; s_endpgm.
        let p = prog(&[vopc_lt(6, 0), sop2(15, 0, 106, 126), sopp(5, 0), sopp(1, 0)]);
        assert!(!scc_branches_uniform(&p));
    }

    #[test]
    fn overwritten_mask_register_becomes_uniform() {
        // v_cmp into vcc, then s_movk vcc_lo, 1 and s_cmpk_eq vcc_lo: uniform again.
        let p = prog(&[vopc_lt(6, 0), sopk(0, 106, 1), sopk(3, 106, 1), sopp(4, 0), sopp(1, 0)]);
        assert!(scc_branches_uniform(&p));
    }
}
