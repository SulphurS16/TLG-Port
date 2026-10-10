//! AMD GCN gfx7 (PS4 "Liverpool") shader instruction decoder and disassembler.

/// Hash map with a fixed hasher: its iteration order depends only on the
/// insertions, so translations are byte-identical between runs (driver shader
/// caches key on the SPIR-V).
pub type DetMap<K, V> = std::collections::HashMap<K, V, std::hash::BuildHasherDefault<std::collections::hash_map::DefaultHasher>>;

pub mod analysis;
pub mod ops;
pub mod translate;

use std::fmt::Write;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Enc {
    Sop2,
    Sopk,
    Sop1,
    Sopc,
    Sopp,
    Smrd,
    Vop2,
    Vop1,
    Vopc,
    Vop3,
    Vintrp,
    Ds,
    Flat,
    Mubuf,
    Mtbuf,
    Mimg,
    Exp,
}

#[derive(Clone, Copy, Debug)]
pub struct Inst {
    pub enc: Enc,
    /// Encoding-local opcode. VOP3 keeps the 9-bit value (VOPC 0..0xFF,
    /// VOP2 0x100.., VOP3-only 0x140.., VOP1 0x180..).
    pub op: u16,
    /// Byte offset from the start of the program.
    pub pc: u32,
    /// Length in dwords including any literal.
    pub len: u8,
    pub w0: u32,
    pub w1: u32,
    pub lit: Option<u32>,
}

pub const OP_LITERAL: u16 = 255;
pub const OP_VCC_LO: u16 = 106;
pub const OP_VCC_HI: u16 = 107;
pub const OP_M0: u16 = 124;
pub const OP_EXEC_LO: u16 = 126;
pub const OP_EXEC_HI: u16 = 127;
pub const OP_VCCZ: u16 = 251;
pub const OP_EXECZ: u16 = 252;
pub const OP_SCC: u16 = 253;
pub const OP_VGPR0: u16 = 256;

/// Ops that take a 64-bit (register pair) scalar source/destination.
pub fn sop_is_64(enc: Enc, op: u16) -> bool {
    match enc {
        Enc::Sop2 => matches!(op, 11 | 15 | 17 | 19 | 21 | 23 | 25 | 27 | 29 | 31 | 33 | 35 | 37 | 41 | 42),
        Enc::Sop1 => matches!(op, 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 24 | 28 | 30 | 31..=34 | 36..=43 | 45 | 47 | 49),
        _ => false,
    }
}

#[derive(Debug)]
pub struct DecodeError(pub String);

fn bits(v: u32, lo: u32, n: u32) -> u32 {
    (v >> lo) & ((1u32 << n) - 1)
}

impl Inst {
    // Scalar fields
    pub fn sdst(&self) -> u16 {
        match self.enc {
            Enc::Smrd => bits(self.w0, 15, 7) as u16,
            Enc::Vop3 => bits(self.w0, 8, 7) as u16,
            _ => bits(self.w0, 16, 7) as u16,
        }
    }
    pub fn ssrc0(&self) -> u16 {
        bits(self.w0, 0, 8) as u16
    }
    pub fn ssrc1(&self) -> u16 {
        bits(self.w0, 8, 8) as u16
    }
    pub fn simm16(&self) -> u16 {
        (self.w0 & 0xFFFF) as u16
    }
    pub fn smrd_sbase(&self) -> u16 {
        (bits(self.w0, 9, 6) * 2) as u16
    }
    pub fn smrd_imm(&self) -> bool {
        bits(self.w0, 8, 1) != 0
    }
    pub fn smrd_offset(&self) -> u32 {
        bits(self.w0, 0, 8)
    }
    /// Constant byte offset of an SMRD access: the 8-bit immediate and the
    /// CI 32-bit literal (OFFSET = 255 with IMM = 0) both count dwords;
    /// None when the offset comes from an SGPR (which holds bytes).
    pub fn smrd_const_offset(&self) -> Option<u32> {
        if self.smrd_imm() {
            Some(self.smrd_offset() * 4)
        } else if self.smrd_offset() == 255 {
            Some(self.lit.unwrap_or(0).wrapping_mul(4))
        } else {
            None
        }
    }
    // Vector ALU fields
    pub fn vdst(&self) -> u16 {
        match self.enc {
            Enc::Vop3 => bits(self.w0, 0, 8) as u16,
            Enc::Vintrp => bits(self.w0, 18, 8) as u16,
            Enc::Ds => bits(self.w1, 24, 8) as u16,
            _ => bits(self.w0, 17, 8) as u16,
        }
    }
    /// 9-bit source operand `i` (VOP1/VOP2/VOPC/VOP3).
    pub fn src(&self, i: usize) -> u16 {
        match self.enc {
            Enc::Vop3 => bits(self.w1, 9 * i as u32, 9) as u16,
            _ => {
                if i == 0 {
                    bits(self.w0, 0, 9) as u16
                } else {
                    OP_VGPR0 + bits(self.w0, 9, 8) as u16
                }
            }
        }
    }
    pub fn vop3_abs(&self) -> u32 {
        bits(self.w0, 8, 3)
    }
    pub fn vop3_clamp(&self) -> bool {
        bits(self.w0, 11, 1) != 0
    }
    pub fn vop3_omod(&self) -> u32 {
        bits(self.w1, 27, 2)
    }
    pub fn vop3_neg(&self) -> u32 {
        bits(self.w1, 29, 3)
    }
    /// VOP3b ops carry an SGPR destination instead of abs/clamp.
    pub fn is_vop3b(&self) -> bool {
        self.enc == Enc::Vop3 && matches!(self.op, 0x125..=0x12A | 0x16D | 0x16E | 0x176 | 0x177)
    }
    // VINTRP
    pub fn interp_attr(&self) -> u32 {
        bits(self.w0, 10, 6)
    }
    pub fn interp_chan(&self) -> u32 {
        bits(self.w0, 8, 2)
    }
    pub fn interp_vsrc(&self) -> u16 {
        bits(self.w0, 0, 8) as u16
    }
    // DS
    pub fn ds_offset0(&self) -> u32 {
        bits(self.w0, 0, 8)
    }
    pub fn ds_offset1(&self) -> u32 {
        bits(self.w0, 8, 8)
    }
    pub fn ds_gds(&self) -> bool {
        bits(self.w0, 17, 1) != 0
    }
    pub fn ds_addr(&self) -> u16 {
        bits(self.w1, 0, 8) as u16
    }
    pub fn ds_data0(&self) -> u16 {
        bits(self.w1, 8, 8) as u16
    }
    pub fn ds_data1(&self) -> u16 {
        bits(self.w1, 16, 8) as u16
    }
    // MUBUF / MTBUF
    pub fn buf_offset(&self) -> u32 {
        bits(self.w0, 0, 12)
    }
    pub fn buf_offen(&self) -> bool {
        bits(self.w0, 12, 1) != 0
    }
    pub fn buf_idxen(&self) -> bool {
        bits(self.w0, 13, 1) != 0
    }
    pub fn buf_glc(&self) -> bool {
        bits(self.w0, 14, 1) != 0
    }
    pub fn buf_addr64(&self) -> bool {
        bits(self.w0, 15, 1) != 0
    }
    pub fn buf_lds(&self) -> bool {
        self.enc == Enc::Mubuf && bits(self.w0, 16, 1) != 0
    }
    pub fn mtbuf_dfmt(&self) -> u32 {
        bits(self.w0, 19, 4)
    }
    pub fn mtbuf_nfmt(&self) -> u32 {
        bits(self.w0, 23, 3)
    }
    pub fn mem_vaddr(&self) -> u16 {
        bits(self.w1, 0, 8) as u16
    }
    pub fn mem_vdata(&self) -> u16 {
        bits(self.w1, 8, 8) as u16
    }
    pub fn mem_srsrc(&self) -> u16 {
        (bits(self.w1, 16, 5) * 4) as u16
    }
    pub fn buf_slc(&self) -> bool {
        bits(self.w1, 22, 1) != 0
    }
    pub fn buf_tfe(&self) -> bool {
        bits(self.w1, 23, 1) != 0
    }
    pub fn buf_soffset(&self) -> u16 {
        bits(self.w1, 24, 8) as u16
    }
    // MIMG
    pub fn mimg_dmask(&self) -> u32 {
        bits(self.w0, 8, 4)
    }
    pub fn mimg_unorm(&self) -> bool {
        bits(self.w0, 12, 1) != 0
    }
    pub fn mimg_da(&self) -> bool {
        bits(self.w0, 14, 1) != 0
    }
    pub fn mimg_ssamp(&self) -> u16 {
        (bits(self.w1, 21, 5) * 4) as u16
    }
    // EXP
    pub fn exp_en(&self) -> u32 {
        bits(self.w0, 0, 4)
    }
    pub fn exp_target(&self) -> u32 {
        bits(self.w0, 4, 6)
    }
    pub fn exp_compr(&self) -> bool {
        bits(self.w0, 10, 1) != 0
    }
    pub fn exp_done(&self) -> bool {
        bits(self.w0, 11, 1) != 0
    }
    pub fn exp_vm(&self) -> bool {
        bits(self.w0, 12, 1) != 0
    }
    pub fn exp_vsrc(&self, i: usize) -> u16 {
        bits(self.w1, 8 * i as u32, 8) as u16
    }

    /// Branch target byte offset for SOPP branches.
    pub fn branch_target(&self) -> Option<u32> {
        if self.enc == Enc::Sopp && matches!(self.op, 2 | 4..=9) {
            let off = self.simm16() as i16 as i32;
            Some((self.pc as i32 + 4 + off * 4) as u32)
        } else {
            None
        }
    }

    /// v_readlane_b32 / v_writelane_b32 (VOP2 or VOP3 form) with a constant
    /// lane select: (is_write, vgpr, lane). The VOP2 form encodes the lane as a
    /// scalar operand in the VSRC1 field.
    pub fn const_lane_op(&self) -> Option<(bool, u16, u32)> {
        let (write, lane_op) = match (self.enc, self.op) {
            (Enc::Vop2, 1) | (Enc::Vop2, 2) => (self.op == 2, self.src(1) - OP_VGPR0),
            (Enc::Vop3, 0x101) | (Enc::Vop3, 0x102) => (self.op == 0x102, self.src(1)),
            _ => return None,
        };
        if !(128..=192).contains(&lane_op) {
            return None;
        }
        let vgpr = if write { self.vdst() } else { self.src(0).checked_sub(OP_VGPR0)? };
        Some((write, vgpr, (lane_op - 128) as u32))
    }

    pub fn is_endpgm(&self) -> bool {
        self.enc == Enc::Sopp && self.op == 1
    }

    pub fn name(&self) -> String {
        let n = match self.enc {
            Enc::Sop2 => ops::lookup(ops::SOP2, self.op),
            Enc::Sopk => ops::lookup(ops::SOPK, self.op),
            Enc::Sop1 => ops::lookup(ops::SOP1, self.op),
            Enc::Sopc => ops::lookup(ops::SOPC, self.op),
            Enc::Sopp => ops::lookup(ops::SOPP, self.op),
            Enc::Smrd => ops::lookup(ops::SMRD, self.op),
            Enc::Vop2 => ops::lookup(ops::VOP2, self.op),
            Enc::Vop1 => ops::lookup(ops::VOP1, self.op),
            Enc::Vopc => return ops::vopc_name(self.op),
            Enc::Vop3 => {
                return match self.op {
                    0..=0xFF => ops::vopc_name(self.op),
                    0x100..=0x13F => ops::lookup(ops::VOP2, self.op - 0x100).map(|s| s.to_string()).unwrap_or(format!("vop3_{:#x}", self.op)),
                    0x140..=0x17F => ops::lookup(ops::VOP3, self.op).map(|s| s.to_string()).unwrap_or(format!("vop3_{:#x}", self.op)),
                    _ => ops::lookup(ops::VOP1, self.op - 0x180).map(|s| s.to_string()).unwrap_or(format!("vop3_{:#x}", self.op)),
                };
            }
            Enc::Vintrp => ops::lookup(ops::VINTRP, self.op),
            Enc::Ds => ops::lookup(ops::DS, self.op),
            Enc::Flat => None,
            Enc::Mubuf => ops::lookup(ops::MUBUF, self.op),
            Enc::Mtbuf => ops::lookup(ops::MTBUF, self.op),
            Enc::Mimg => ops::lookup(ops::MIMG, self.op),
            Enc::Exp => Some("exp"),
        };
        n.map(|s| s.to_string()).unwrap_or_else(|| format!("{:?}_{:#x}", self.enc, self.op).to_lowercase())
    }

    /// Opcode identity independent of encoding (VOP1/VOP2/VOPC ops mapped to their VOP3 numbers).
    pub fn vop3_equiv(&self) -> Option<u16> {
        match self.enc {
            Enc::Vopc => Some(self.op),
            Enc::Vop2 => Some(self.op + 0x100),
            Enc::Vop1 => Some(self.op + 0x180),
            Enc::Vop3 => Some(self.op),
            _ => None,
        }
    }
}

pub fn operand_name(op: u16, lit: Option<u32>) -> String {
    match op {
        0..=103 => format!("s{op}"),
        104 => "flat_scr_lo".into(),
        105 => "flat_scr_hi".into(),
        OP_VCC_LO => "vcc_lo".into(),
        OP_VCC_HI => "vcc_hi".into(),
        108 => "tba_lo".into(),
        109 => "tba_hi".into(),
        110 => "tma_lo".into(),
        111 => "tma_hi".into(),
        112..=123 => format!("ttmp{}", op - 112),
        OP_M0 => "m0".into(),
        OP_EXEC_LO => "exec_lo".into(),
        OP_EXEC_HI => "exec_hi".into(),
        128 => "0".into(),
        129..=192 => format!("{}", op - 128),
        193..=208 => format!("-{}", op - 192),
        240 => "0.5".into(),
        241 => "-0.5".into(),
        242 => "1.0".into(),
        243 => "-1.0".into(),
        244 => "2.0".into(),
        245 => "-2.0".into(),
        246 => "4.0".into(),
        247 => "-4.0".into(),
        OP_VCCZ => "vccz".into(),
        OP_EXECZ => "execz".into(),
        OP_SCC => "scc".into(),
        254 => "lds_direct".into(),
        OP_LITERAL => match lit {
            Some(v) => format!("{v:#x}"),
            None => "lit?".into(),
        },
        256..=511 => format!("v{}", op - 256),
        _ => format!("op{op}"),
    }
}

fn needs_literal(enc: Enc, w0: u32) -> bool {
    match enc {
        Enc::Sop2 | Enc::Sopc => bits(w0, 0, 8) == 255 || bits(w0, 8, 8) == 255,
        Enc::Sop1 => bits(w0, 0, 8) == 255,
        Enc::Sopk => bits(w0, 23, 5) == 21, // s_setreg_imm32_b32
        Enc::Vop1 | Enc::Vopc => bits(w0, 0, 9) == 255,
        Enc::Vop2 => bits(w0, 0, 9) == 255 || matches!(bits(w0, 25, 6), 32 | 33),
        Enc::Smrd => bits(w0, 8, 1) == 0 && bits(w0, 0, 8) == 255,
        _ => false,
    }
}

/// Decodes one instruction at word index `i`.
pub fn decode(code: &[u32], i: usize) -> Result<Inst, DecodeError> {
    let w0 = *code.get(i).ok_or_else(|| DecodeError(format!("truncated at word {i}")))?;
    let (enc, op, two) = if w0 >> 31 == 0 {
        match bits(w0, 25, 7) {
            0x3F => (Enc::Vop1, bits(w0, 9, 8) as u16, false),
            0x3E => (Enc::Vopc, bits(w0, 17, 8) as u16, false),
            _ => (Enc::Vop2, bits(w0, 25, 6) as u16, false),
        }
    } else if bits(w0, 30, 2) == 0b10 {
        match bits(w0, 23, 9) {
            0x17D => (Enc::Sop1, bits(w0, 8, 8) as u16, false),
            0x17E => (Enc::Sopc, bits(w0, 16, 7) as u16, false),
            0x17F => (Enc::Sopp, bits(w0, 16, 7) as u16, false),
            _ if bits(w0, 28, 4) == 0xB => (Enc::Sopk, bits(w0, 23, 5) as u16, false),
            _ => (Enc::Sop2, bits(w0, 23, 7) as u16, false),
        }
    } else if bits(w0, 27, 5) == 0b11000 {
        (Enc::Smrd, bits(w0, 22, 5) as u16, false)
    } else {
        match bits(w0, 26, 6) {
            0x32 => (Enc::Vintrp, bits(w0, 16, 2) as u16, false),
            0x34 => (Enc::Vop3, bits(w0, 17, 9) as u16, true),
            0x36 => (Enc::Ds, bits(w0, 18, 8) as u16, true),
            0x37 => (Enc::Flat, bits(w0, 18, 7) as u16, true),
            0x38 => (Enc::Mubuf, bits(w0, 18, 7) as u16, true),
            0x3A => (Enc::Mtbuf, bits(w0, 16, 3) as u16, true),
            0x3C => (Enc::Mimg, bits(w0, 18, 7) as u16, true),
            0x3E => (Enc::Exp, 0, true),
            other => return Err(DecodeError(format!("unknown encoding {other:#x} in {w0:#010x} at word {i}"))),
        }
    };
    let w1 = if two { *code.get(i + 1).ok_or_else(|| DecodeError("truncated 64-bit instruction".into()))? } else { 0 };
    let mut len = if two { 2 } else { 1 };
    let lit = if needs_literal(enc, w0) {
        let v = *code.get(i + len).ok_or_else(|| DecodeError("truncated literal".into()))?;
        len += 1;
        Some(v)
    } else {
        None
    };
    Ok(Inst { enc, op, pc: (i * 4) as u32, len: len as u8, w0, w1, lit })
}

/// Decodes a program starting at word 0. Decoding continues past `s_endpgm`
/// (and `s_setpc_b64` returns) while branch targets beyond it are known.
pub fn decode_program(code: &[u32]) -> Result<Vec<Inst>, DecodeError> {
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut max_target = 0u32;
    while i < code.len() {
        let ins = decode(code, i)?;
        if let Some(t) = ins.branch_target() {
            max_target = max_target.max(t);
        }
        i += ins.len as usize;
        let end = ins.is_endpgm() || (ins.enc == Enc::Sop1 && ins.op == 32);
        out.push(ins);
        if end && (i as u32 * 4) > max_target {
            return Ok(out);
        }
        if out.len() > 1 << 16 {
            return Err(DecodeError("program too long".into()));
        }
    }
    Err(DecodeError("program has no s_endpgm".into()))
}

/// One-line disassembly.
pub fn disasm(ins: &Inst) -> String {
    let mut s = format!("{:<24}", ins.name());
    let o = |x: u16| operand_name(x, ins.lit);
    let v = |x: u16| format!("v{x}");
    let _ = match ins.enc {
        Enc::Sop2 => write!(s, "{}, {}, {}", o(ins.sdst()), o(ins.ssrc0()), o(ins.ssrc1())),
        Enc::Sopk => write!(s, "{}, {:#x}", o(ins.sdst()), ins.simm16()),
        Enc::Sop1 => write!(s, "{}, {}", o(ins.sdst()), o(ins.ssrc0())),
        Enc::Sopc => write!(s, "{}, {}", o(ins.ssrc0()), o(ins.ssrc1())),
        Enc::Sopp => match ins.branch_target() {
            Some(t) => write!(s, "{t:#x}"),
            None => write!(s, "{:#x}", ins.simm16()),
        },
        Enc::Smrd => {
            let off = match ins.smrd_const_offset() {
                Some(b) => format!("{b:#x}"),
                None => o(ins.smrd_offset() as u16),
            };
            write!(s, "{}, s[{}:{}], {}", o(ins.sdst()), ins.smrd_sbase(), ins.smrd_sbase() + 1, off)
        }
        Enc::Vop2 => match ins.op {
            32 => write!(s, "{}, {}, {}, {}", v(ins.vdst()), o(ins.src(0)), o(OP_LITERAL), o(ins.src(1))),
            _ => write!(s, "{}, {}, {}", v(ins.vdst()), o(ins.src(0)), o(ins.src(1))),
        },
        Enc::Vop1 => write!(s, "{}, {}", v(ins.vdst()), o(ins.src(0))),
        Enc::Vopc => write!(s, "vcc, {}, {}", o(ins.src(0)), o(ins.src(1))),
        Enc::Vop3 => {
            let dst = if ins.op < 0x100 { o(ins.vdst()) } else { v(ins.vdst()) };
            let mut t = dst;
            if ins.is_vop3b() {
                let _ = write!(t, ", {}", o(ins.sdst()));
            }
            for k in 0..3 {
                let x = ins.src(k);
                let neg = ins.vop3_neg() >> k & 1 != 0;
                let abs = !ins.is_vop3b() && ins.vop3_abs() >> k & 1 != 0;
                let mut n = o(x);
                if abs {
                    n = format!("|{n}|");
                }
                if neg {
                    n = format!("-{n}");
                }
                let _ = write!(t, ", {n}");
            }
            if !ins.is_vop3b() && ins.vop3_clamp() {
                t.push_str(" clamp");
            }
            if ins.vop3_omod() != 0 {
                let _ = write!(t, " omod:{}", ins.vop3_omod());
            }
            write!(s, "{t}")
        }
        Enc::Vintrp => write!(s, "{}, {}, attr{}.{}", v(ins.vdst()), v(ins.interp_vsrc()), ins.interp_attr(), ["x", "y", "z", "w"][ins.interp_chan() as usize]),
        Enc::Ds => write!(
            s,
            "{}, {}, {}, {} off0:{} off1:{}{}",
            v(ins.vdst()),
            v(ins.ds_addr()),
            v(ins.ds_data0()),
            v(ins.ds_data1()),
            ins.ds_offset0(),
            ins.ds_offset1(),
            if ins.ds_gds() { " gds" } else { "" }
        ),
        Enc::Mubuf | Enc::Mtbuf => {
            let mut t = format!("{}, {}, s[{}:{}], {} off:{:#x}", v(ins.mem_vdata()), v(ins.mem_vaddr()), ins.mem_srsrc(), ins.mem_srsrc() + 3, o(ins.buf_soffset()), ins.buf_offset());
            if ins.buf_offen() {
                t.push_str(" offen");
            }
            if ins.buf_idxen() {
                t.push_str(" idxen");
            }
            if ins.buf_glc() {
                t.push_str(" glc");
            }
            if ins.enc == Enc::Mtbuf {
                let _ = write!(t, " dfmt:{} nfmt:{}", ins.mtbuf_dfmt(), ins.mtbuf_nfmt());
            }
            write!(s, "{t}")
        }
        Enc::Mimg => write!(
            s,
            "{}, {}, s[{}:{}], s[{}:{}] dmask:{:#x}{}{}",
            v(ins.mem_vdata()),
            v(ins.mem_vaddr()),
            ins.mem_srsrc(),
            ins.mem_srsrc() + 7,
            ins.mimg_ssamp(),
            ins.mimg_ssamp() + 3,
            ins.mimg_dmask(),
            if ins.mimg_unorm() { " unorm" } else { "" },
            if ins.mimg_da() { " da" } else { "" }
        ),
        Enc::Exp => {
            let tgt = match ins.exp_target() {
                t @ 0..=7 => format!("mrt{t}"),
                8 => "mrtz".into(),
                9 => "null".into(),
                t @ 12..=15 => format!("pos{}", t - 12),
                t @ 32..=63 => format!("param{}", t - 32),
                t => format!("tgt{t}"),
            };
            write!(
                s,
                "{tgt} en:{:#x} {}, {}, {}, {}{}{}{}",
                ins.exp_en(),
                v(ins.exp_vsrc(0)),
                v(ins.exp_vsrc(1)),
                v(ins.exp_vsrc(2)),
                v(ins.exp_vsrc(3)),
                if ins.exp_compr() { " compr" } else { "" },
                if ins.exp_done() { " done" } else { "" },
                if ins.exp_vm() { " vm" } else { "" }
            )
        }
        Enc::Flat => write!(s, "{:#010x} {:#010x}", ins.w0, ins.w1),
    };
    s
}

/// Multi-line listing with byte offsets.
pub fn listing(insts: &[Inst]) -> String {
    let mut out = String::new();
    for i in insts {
        let _ = writeln!(out, "  {:05x}: {}", i.pc, disasm(i));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_basic_forms() {
        // s_mov_b32 s0, 0x3f800000 (literal)
        let code = [0xBE80_03FF, 0x3F80_0000, 0xBF81_0000];
        let p = decode_program(&code).unwrap();
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].name(), "s_mov_b32");
        assert_eq!(p[0].lit, Some(0x3F80_0000));
        assert!(p[1].is_endpgm());
        // v_mov_b32 v1, s2  (VOP1: 0x7E02_0202)
        let i = decode(&[0x7E02_0202], 0).unwrap();
        assert_eq!(i.enc, Enc::Vop1);
        assert_eq!(i.name(), "v_mov_b32");
        assert_eq!(i.vdst(), 1);
        assert_eq!(i.src(0), 2);
        // v_add_f32 v0, v1, v2 (VOP2 op 3): 0x06000501
        let i = decode(&[0x0600_0501], 0).unwrap();
        assert_eq!(i.name(), "v_add_f32");
        assert_eq!(i.src(0), 256 + 1);
        assert_eq!(i.src(1), 256 + 2);
    }

    #[test]
    fn smrd_offsets_count_dwords() {
        // s_buffer_load_dwordx4 s0, s[36:39], imm 0xbc dwords
        let w = (0b11000 << 27) | (10 << 22) | (0 << 15) | (18 << 9) | (1 << 8) | 0xBC;
        let i = decode(&[w], 0).unwrap();
        assert_eq!(i.name(), "s_buffer_load_dwordx4");
        assert_eq!(i.smrd_const_offset(), Some(0x2F0));
        // CI 32-bit literal offset (OFFSET = 255, IMM = 0) is in dwords too:
        // Bloodborne's tone-mapping constants at byte 0xca0 are addressed
        // with literal 0x328.
        let w = (0b11000 << 27) | (10 << 22) | (4 << 15) | (18 << 9) | 0xFF;
        let i = decode(&[w, 0x328], 0).unwrap();
        assert_eq!(i.len, 2);
        assert_eq!(i.smrd_const_offset(), Some(0xCA0));
        // SGPR offset: not a constant.
        let w = (0b11000 << 27) | (10 << 22) | (18 << 9) | 7;
        assert_eq!(decode(&[w], 0).unwrap().smrd_const_offset(), None);
    }

    #[test]
    fn branch_targets_extend_decoding() {
        // s_branch +1 ; s_endpgm ; s_endpgm
        let code = [0xBF82_0001, 0xBF81_0000, 0xBF81_0000];
        let p = decode_program(&code).unwrap();
        assert_eq!(p.len(), 3);
        assert_eq!(p[0].branch_target(), Some(8));
    }
}
