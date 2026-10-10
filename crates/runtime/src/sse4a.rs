//! AMD SSE4a (EXTRQ/INSERTQ) on hosts without it.
//!
//! The PS4's Jaguar cores implement SSE4a; Intel CPUs raise #UD for these opcodes
//! (Shadow of the Colossus' main module has 98 sites, mostly in one vectorized
//! routine). At load time each site in FDE-described code becomes a jump to a
//! trampoline that emulates it with integer and SSE4.1 instructions -- flags, general
//! registers and the System V red zone below `rsp` are preserved -- and jumps back.
//! A 4-byte encoding is too short for the jump: the trampoline then also takes the
//! following instruction(s), relocated, when nothing branches between them. Sites
//! that cannot be patched safely keep their bytes and are emulated from the thread
//! context by the vectored exception handler (`handle_trap`), which is slower but
//! exact. On hosts that implement SSE4a nothing is patched.
//!
//! Semantics (AMD64 APM vol. 4): only the low quadword of the destination is
//! defined; this implementation leaves the upper quadword (and the upper YMM half)
//! unchanged, as the legacy-SSE encoding of the original instruction does for YMM.

use iced_x86::code_asm::*;
use iced_x86::{Instruction, Mnemonic, OpKind};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// EXTRQ xmm, imm8 (length), imm8 (index)
    ExtrqImm { dst: u8, len: u8, idx: u8 },
    /// EXTRQ xmm1, xmm2: length = xmm2[5:0], index = xmm2[13:8]
    ExtrqReg { dst: u8, src: u8 },
    /// INSERTQ xmm1, xmm2, imm8 (length), imm8 (index)
    InsertqImm { dst: u8, src: u8, len: u8, idx: u8 },
    /// INSERTQ xmm1, xmm2: length = xmm2[69:64], index = xmm2[77:72]
    InsertqReg { dst: u8, src: u8 },
}

pub fn host_has_sse4a() -> bool {
    let r = unsafe { std::arch::x86_64::__cpuid(0x8000_0001) };
    r.ecx & (1 << 6) != 0
}

fn xmm_no(ins: &Instruction, op: u32) -> Option<u8> {
    let r = ins.op_register(op);
    r.is_xmm().then(|| r.number() as u8)
}

pub fn decode(ins: &Instruction) -> Option<Op> {
    match (ins.mnemonic(), ins.op_count()) {
        (Mnemonic::Extrq, 3) if ins.op1_kind() == OpKind::Immediate8 => Some(Op::ExtrqImm {
            dst: xmm_no(ins, 0)?,
            len: ins.immediate8() & 63,
            idx: ins.immediate8_2nd() & 63,
        }),
        (Mnemonic::Extrq, 2) => Some(Op::ExtrqReg { dst: xmm_no(ins, 0)?, src: xmm_no(ins, 1)? }),
        (Mnemonic::Insertq, 4) => Some(Op::InsertqImm {
            dst: xmm_no(ins, 0)?,
            src: xmm_no(ins, 1)?,
            len: ins.immediate8() & 63,
            idx: ins.immediate8_2nd() & 63,
        }),
        (Mnemonic::Insertq, 2) => Some(Op::InsertqReg { dst: xmm_no(ins, 0)?, src: xmm_no(ins, 1)? }),
        _ => None,
    }
}

fn mask(len: u64) -> u64 {
    if len == 0 { !0 } else { (1u64 << len) - 1 }
}

/// Applies `op` to XMM register values (`[low, high]` quadwords).
pub fn apply(op: Op, x: &mut [[u64; 2]; 16]) {
    match op {
        Op::ExtrqImm { dst, len, idx } => {
            let d = dst as usize;
            x[d][0] = (x[d][0] >> idx) & mask(len as u64);
        }
        Op::ExtrqReg { dst, src } => {
            let c = x[src as usize][0];
            let d = dst as usize;
            x[d][0] = (x[d][0] >> ((c >> 8) & 63)) & mask(c & 63);
        }
        Op::InsertqImm { dst, src, len, idx } => {
            let m = mask(len as u64);
            let d = dst as usize;
            x[d][0] = (x[d][0] & !(m << idx)) | ((x[src as usize][0] & m) << idx);
        }
        Op::InsertqReg { dst, src } => {
            let c = x[src as usize][1];
            let (len, idx) = (c & 63, (c >> 8) & 63);
            let m = mask(len);
            let d = dst as usize;
            x[d][0] = (x[d][0] & !(m << idx)) | ((x[src as usize][0] & m) << idx);
        }
    }
}

const XMM: [AsmRegisterXmm; 16] = [
    xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15,
];

/// Appends code emulating `op` (state-preserving apart from the destination's low quadword).
pub fn emit(op: Op, a: &mut CodeAssembler) -> Result<(), IcedError> {
    a.lea(rsp, ptr(rsp - 128))?; // keep the red zone intact
    a.pushfq()?;
    a.push(rax)?;
    a.push(rcx)?;
    a.push(rdx)?;
    a.push(r8)?;
    a.push(r9)?;
    let dst = |d: u8| XMM[d as usize];
    match op {
        Op::ExtrqImm { dst: d, len, idx } => {
            a.vmovq(rax, dst(d))?;
            if idx != 0 {
                a.shr(rax, idx as u32)?;
            }
            if len != 0 {
                a.mov(rdx, mask(len as u64))?;
                a.and(rax, rdx)?;
            }
            a.pinsrq(dst(d), rax, 0)?;
        }
        Op::ExtrqReg { dst: d, src: s } => {
            let mut full = a.create_label();
            a.vmovq(rax, dst(d))?;
            a.vmovq(rdx, dst(s))?;
            a.mov(ecx, edx)?;
            a.shr(ecx, 8)?;
            a.and(ecx, 63)?;
            a.shr(rax, cl)?;
            a.mov(ecx, edx)?;
            a.and(ecx, 63)?;
            a.jz(full)?;
            a.mov(edx, 1)?;
            a.shl(rdx, cl)?;
            a.dec(rdx)?;
            a.and(rax, rdx)?;
            a.set_label(&mut full)?;
            a.pinsrq(dst(d), rax, 0)?;
        }
        Op::InsertqImm { dst: d, src: s, len, idx } => {
            let m = mask(len as u64);
            a.vmovq(rax, dst(d))?;
            a.vmovq(rdx, dst(s))?;
            a.mov(rcx, m)?;
            a.and(rdx, rcx)?;
            if idx != 0 {
                a.shl(rdx, idx as u32)?;
            }
            a.mov(rcx, !(m << idx))?;
            a.and(rax, rcx)?;
            a.or(rax, rdx)?;
            a.pinsrq(dst(d), rax, 0)?;
        }
        Op::InsertqReg { dst: d, src: s } => {
            let mut full = a.create_label();
            a.vmovq(rax, dst(d))?;
            a.vmovq(rdx, dst(s))?;
            a.vpextrq(r8, dst(s), 1)?;
            a.mov(r9, -1i64)?;
            a.mov(ecx, r8d)?;
            a.and(ecx, 63)?;
            a.jz(full)?;
            a.mov(r9d, 1)?;
            a.shl(r9, cl)?;
            a.dec(r9)?;
            a.set_label(&mut full)?;
            a.and(rdx, r9)?;
            a.mov(ecx, r8d)?;
            a.shr(ecx, 8)?;
            a.and(ecx, 63)?;
            a.shl(rdx, cl)?;
            a.shl(r9, cl)?;
            a.not(r9)?;
            a.and(rax, r9)?;
            a.or(rax, rdx)?;
            a.pinsrq(dst(d), rax, 0)?;
        }
    }
    a.pop(r9)?;
    a.pop(r8)?;
    a.pop(rdx)?;
    a.pop(rcx)?;
    a.pop(rax)?;
    a.popfq()?;
    a.lea(rsp, ptr(rsp + 128))?;
    Ok(())
}

/// Sites left for exception-time emulation: address -> (instruction length, op).
static TRAP_SITES: RwLock<Option<HashMap<u64, (u8, Op)>>> = RwLock::new(None);
pub static TRAPS: AtomicU64 = AtomicU64::new(0);

pub fn register_trap(addr: u64, len: u8, op: Op) {
    TRAP_SITES.write().get_or_insert_with(HashMap::new).insert(addr, (len, op));
}

/// Emulates a registered site that raised #UD; returns false if `ctx.Rip` is not one.
pub unsafe fn handle_trap(ctx: &mut windows_sys::Win32::System::Diagnostics::Debug::CONTEXT) -> bool {
    let Some(g) = TRAP_SITES.try_read() else { return false };
    let Some(&(len, op)) = g.as_ref().and_then(|m| m.get(&ctx.Rip)) else { return false };
    drop(g);
    unsafe {
        let regs = &mut ctx.Anonymous.FltSave.XmmRegisters;
        let mut x = [[0u64; 2]; 16];
        for (i, r) in regs.iter().enumerate() {
            x[i] = [r.Low, r.High as u64];
        }
        apply(op, &mut x);
        for (i, r) in regs.iter_mut().enumerate() {
            r.Low = x[i][0];
            r.High = x[i][1] as i64;
        }
    }
    ctx.Rip += len as u64;
    let n = TRAPS.fetch_add(1, Ordering::Relaxed);
    if n.is_power_of_two() && n >= 1024 {
        crate::warn!("SSE4a trap emulation: {n} executions so far (a hot site was not patched)");
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs the emitted trampoline code natively against `apply`.
    fn run(op: Op, regs: &mut [[u64; 2]; 16]) {
        let mut a = CodeAssembler::new(64).unwrap();
        // Load xmm0..15 from [rdi], emulate, store back, return.
        for i in 0..16 {
            a.movdqu(XMM[i], ptr(rdi + (i * 16) as i32)).unwrap();
        }
        emit(op, &mut a).unwrap();
        for i in 0..16 {
            a.movdqu(ptr(rdi + (i * 16) as i32), XMM[i]).unwrap();
        }
        a.ret().unwrap();
        let mem = unsafe {
            windows_sys::Win32::System::Memory::VirtualAlloc(
                std::ptr::null(),
                0x1000,
                windows_sys::Win32::System::Memory::MEM_COMMIT | windows_sys::Win32::System::Memory::MEM_RESERVE,
                windows_sys::Win32::System::Memory::PAGE_EXECUTE_READWRITE,
            )
        } as u64;
        let code = a.assemble(mem).unwrap();
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), mem as *mut u8, code.len()) };
        let f: extern "sysv64" fn(*mut [[u64; 2]; 16]) = unsafe { std::mem::transmute(mem as usize) };
        // xmm6..15 are callee-saved on Windows; the test function is called with the
        // sysv64 ABI, where all XMM registers are caller-saved.
        f(regs);
        unsafe { windows_sys::Win32::System::Memory::VirtualFree(mem as *mut _, 0, windows_sys::Win32::System::Memory::MEM_RELEASE) };
    }

    #[test]
    fn trampolines_match_semantics() {
        let mut seed = 0x1234_5678_9ABC_DEF0u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for case in 0..2000 {
            let mut regs = [[0u64; 2]; 16];
            for r in regs.iter_mut() {
                *r = [rnd(), rnd()];
            }
            let d = (rnd() % 16) as u8;
            let s = (rnd() % 16) as u8;
            let len = (rnd() % 64) as u8;
            let idx = (rnd() % 64) as u8;
            // Keep len + idx within the defined range for the immediate forms.
            let idx = if len as u32 + idx as u32 > 64 { 64 - len } else { idx } & 63;
            let op = match case % 4 {
                0 => Op::ExtrqImm { dst: d, len, idx },
                1 => {
                    regs[s as usize][0] = (regs[s as usize][0] & !0x3F3F) | len as u64 | (idx as u64) << 8;
                    Op::ExtrqReg { dst: d, src: s }
                }
                2 => Op::InsertqImm { dst: d, src: s, len, idx },
                _ => {
                    regs[s as usize][1] = (regs[s as usize][1] & !0x3F3F) | len as u64 | (idx as u64) << 8;
                    Op::InsertqReg { dst: d, src: s }
                }
            };
            let mut want = regs;
            apply(op, &mut want);
            let mut got = regs;
            run(op, &mut got);
            assert_eq!(got, want, "{op:?}");
        }
    }

    #[test]
    fn known_values() {
        let mut x = [[0u64; 2]; 16];
        x[1] = [0xFEDC_BA98_7654_3210, 7];
        apply(Op::ExtrqImm { dst: 1, len: 8, idx: 4 }, &mut x);
        assert_eq!(x[1], [0x21, 7]);
        let mut x = [[0u64; 2]; 16];
        x[0] = [0xFFFF_FFFF_FFFF_FFFF, 1];
        x[2] = [0xAB, 0];
        apply(Op::InsertqImm { dst: 0, src: 2, len: 8, idx: 16 }, &mut x);
        assert_eq!(x[0], [0xFFFF_FFFF_FFAB_FFFF, 1]);
    }
}
