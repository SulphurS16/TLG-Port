//! libSceGnmDriver: command-buffer builder functions (emit PM4 register writes
//! into guest command buffers, NOP-padded to the reserved size), submission,
//! graphics events and compute queues.

use crate::gpu::pm4::{self, Writer, marker};
use crate::gpu::{self, FlipRequest, Submission};
use crate::hle::{Registry, cstr};
use crate::{fixed, reg, sysv};

use crate::gpu::render::TESS_FACTOR_RING;
const ERR_INVALID: i32 = 0x8025_0001u32 as i32;

// Shader stage registers (gfx7 SH register offsets).
const SPI_SHADER_PGM_LO_PS: u32 = 0x2C08;
const SPI_SHADER_PGM_RSRC1_PS: u32 = 0x2C0A;
const SPI_SHADER_PGM_LO_VS: u32 = 0x2C48;
const SPI_SHADER_PGM_RSRC1_VS: u32 = 0x2C4A;
const SPI_SHADER_PGM_LO_GS: u32 = 0x2C88;
const SPI_SHADER_PGM_RSRC1_GS: u32 = 0x2C8A;
const SPI_SHADER_PGM_LO_ES: u32 = 0x2CC8;
const SPI_SHADER_PGM_RSRC1_ES: u32 = 0x2CCA;
const SPI_SHADER_PGM_LO_HS: u32 = 0x2D08;
const SPI_SHADER_PGM_RSRC1_HS: u32 = 0x2D0A;
const SPI_SHADER_PGM_LO_LS: u32 = 0x2D48;
const SPI_SHADER_PGM_RSRC1_LS: u32 = 0x2D4A;
const COMPUTE_NUM_THREAD_X: u32 = 0x2E07;
const COMPUTE_PGM_LO: u32 = 0x2E0C;
const COMPUTE_PGM_RSRC1: u32 = 0x2E12;
// Context registers.
const CB_SHADER_MASK: u32 = 0xA08F;
const SPI_VS_OUT_CONFIG: u32 = 0xA1B1;
const SPI_PS_INPUT_ENA: u32 = 0xA1B3;
const SPI_PS_IN_CONTROL: u32 = 0xA1B6;
const SPI_BARYC_CNTL: u32 = 0xA1B8;
const SPI_SHADER_POS_FORMAT: u32 = 0xA1C3;
const SPI_SHADER_Z_FORMAT: u32 = 0xA1C4;
const DB_SHADER_CONTROL: u32 = 0xA203;
const PA_CL_VS_OUT_CNTL: u32 = 0xA207;
const VGT_GS_OUT_PRIM_TYPE: u32 = 0xA29B;
const VGT_HOS_MAX_TESS_LEVEL: u32 = 0xA286;
const VGT_LS_HS_CONFIG: u32 = 0xA2D6;
const VGT_TF_PARAM: u32 = 0xA2DB;

fn modified_rsrc1(rsrc1: u32, modifier: u32) -> u32 {
    if modifier == 0 { rsrc1 } else { (rsrc1 & 0xFCFF_FC3F) | modifier }
}

unsafe fn regs<'a>(p: *const u32, n: usize) -> Option<&'a [u32]> {
    if p.is_null() { None } else { Some(unsafe { std::slice::from_raw_parts(p, n) }) }
}

fn fill_default_state(cmd: *mut u32, n: u32, tag: u32) -> u32 {
    let size = n.min(0x100);
    if cmd.is_null() || size < 4 {
        return 0;
    }
    let mut w = unsafe { Writer::new(cmd, size) };
    w.packet(pm4::op::CONTEXT_CONTROL, &[0x8000_0000, 0x8000_0000]);
    w.marker(tag, "default-state");
    w.finish()
}

sysv! {
    fn sceGnmDrawInitDefaultHardwareState200(cmd: *mut u32, n: u32) -> u32 { fill_default_state(cmd, n, marker::INIT_DEFAULT) }
    fn sceGnmDispatchInitDefaultHardwareState(cmd: *mut u32, n: u32) -> u32 { fill_default_state(cmd, n, marker::INIT_DEFAULT) }

    fn sceGnmSetVsShader(cmd: *mut u32, n: u32, vs: *const u32, modifier: u32) -> i32 {
        let Some(v) = regs(vs, 7) else { return ERR_INVALID };
        let mut w = Writer::new(cmd, n);
        let ok = w.set_sh(SPI_SHADER_PGM_LO_VS, &[v[0], v[1]])
            && w.set_sh(SPI_SHADER_PGM_RSRC1_VS, &[modified_rsrc1(v[2], modifier), v[3]])
            && w.set_ctx(SPI_VS_OUT_CONFIG, &[v[4]])
            && w.set_ctx(SPI_SHADER_POS_FORMAT, &[v[5]])
            && w.set_ctx(PA_CL_VS_OUT_CNTL, &[v[6]]);
        if !ok { return ERR_INVALID; }
        w.finish();
        0
    }
    fn sceGnmUpdateVsShader(cmd: *mut u32, n: u32, vs: *const u32, modifier: u32) -> i32 { sceGnmSetVsShader(cmd, n, vs, modifier) }
    fn sceGnmSetPsShader(cmd: *mut u32, n: u32, ps: *const u32) -> i32 {
        let mut w = Writer::new(cmd, n);
        let ok = match regs(ps, 12) {
            Some(p) => w.set_sh(SPI_SHADER_PGM_LO_PS, &[p[0], p[1]])
                && w.set_sh(SPI_SHADER_PGM_RSRC1_PS, &[p[2], p[3]])
                && w.set_ctx(SPI_SHADER_Z_FORMAT, &[p[4], p[5]])
                && w.set_ctx(SPI_PS_INPUT_ENA, &[p[6], p[7]])
                && w.set_ctx(SPI_PS_IN_CONTROL, &[p[8]])
                && w.set_ctx(SPI_BARYC_CNTL, &[p[9]])
                && w.set_ctx(DB_SHADER_CONTROL, &[p[10]])
                && w.set_ctx(CB_SHADER_MASK, &[p[11]]),
            None => w.set_sh(SPI_SHADER_PGM_LO_PS, &[0, 0])
                && w.set_ctx(SPI_SHADER_Z_FORMAT, &[0, 0])
                && w.set_ctx(CB_SHADER_MASK, &[0]),
        };
        if !ok { return ERR_INVALID; }
        w.finish();
        0
    }
    fn sceGnmUpdatePsShader(cmd: *mut u32, n: u32, ps: *const u32) -> i32 { sceGnmSetPsShader(cmd, n, ps) }
    fn sceGnmSetCsShader(cmd: *mut u32, n: u32, cs: *const u32) -> i32 {
        let Some(c) = regs(cs, 7) else { return ERR_INVALID };
        let mut w = Writer::new(cmd, n);
        let ok = w.set_sh(COMPUTE_PGM_LO, &[c[0], c[1]])
            && w.set_sh(COMPUTE_PGM_RSRC1, &[c[2], c[3]])
            && w.set_sh(COMPUTE_NUM_THREAD_X, &[c[4], c[5], c[6]]);
        if !ok { return ERR_INVALID; }
        w.finish();
        0
    }
    fn sceGnmSetEsShader(cmd: *mut u32, n: u32, es: *const u32, modifier: u32) -> i32 {
        let Some(e) = regs(es, 4) else { return ERR_INVALID };
        let mut w = Writer::new(cmd, n);
        let ok = w.set_sh(SPI_SHADER_PGM_LO_ES, &[e[0], e[1]]) && w.set_sh(SPI_SHADER_PGM_RSRC1_ES, &[modified_rsrc1(e[2], modifier), e[3]]);
        if !ok { return ERR_INVALID; }
        w.finish();
        0
    }
    fn sceGnmSetLsShader(cmd: *mut u32, n: u32, ls: *const u32, modifier: u32) -> i32 {
        let Some(l) = regs(ls, 4) else { return ERR_INVALID };
        let mut w = Writer::new(cmd, n);
        let ok = w.set_sh(SPI_SHADER_PGM_LO_LS, &[l[0], l[1]]) && w.set_sh(SPI_SHADER_PGM_RSRC1_LS, &[modified_rsrc1(l[2], modifier), l[3]]);
        if !ok { return ERR_INVALID; }
        w.finish();
        0
    }
    // hs: PGM_LO/HI, RSRC1/2, VGT_TF_PARAM, VGT_HOS_MAX/MIN_TESS_LEVEL; the
    // fourth argument is the VGT_LS_HS_CONFIG value (the title passes the
    // first dword of its tessellation registers by value).
    fn sceGnmSetHsShader(cmd: *mut u32, n: u32, hs: *const u32, ls_hs_config: u32) -> i32 {
        let Some(h) = regs(hs, 7) else { return ERR_INVALID };
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
            crate::info!("sceGnmSetHsShader: regs {h:08x?}, VGT_LS_HS_CONFIG {ls_hs_config:#x}");
        }
        let mut w = Writer::new(cmd, n);
        let ok = w.set_sh(SPI_SHADER_PGM_LO_HS, &[h[0], h[1]])
            && w.set_sh(SPI_SHADER_PGM_RSRC1_HS, &[h[2], h[3]])
            && w.set_ctx(VGT_HOS_MAX_TESS_LEVEL, &[h[5], h[6]])
            && w.set_ctx(VGT_TF_PARAM, &[h[4]])
            && w.set_ctx(VGT_LS_HS_CONFIG, &[ls_hs_config]);
        if !ok { return ERR_INVALID; }
        w.finish();
        0
    }
    fn sceGnmUpdateHsShader(cmd: *mut u32, n: u32, hs: *const u32, ls_hs_config: u32) -> i32 { sceGnmSetHsShader(cmd, n, hs, ls_hs_config) }
    fn sceGnmSetGsShader(cmd: *mut u32, n: u32, gs: *const u32) -> i32 {
        let Some(g) = regs(gs, 7) else { return ERR_INVALID };
        let mut w = Writer::new(cmd, n);
        let ok = w.set_sh(SPI_SHADER_PGM_LO_GS, &[g[0], g[1]])
            && w.set_sh(SPI_SHADER_PGM_RSRC1_GS, &[g[2], g[3]])
            && w.set_ctx(VGT_GS_OUT_PRIM_TYPE, &[g[5]]);
        if !ok { return ERR_INVALID; }
        w.finish();
        0
    }
    fn sceGnmUpdateGsShader(cmd: *mut u32, n: u32, gs: *const u32) -> i32 { sceGnmSetGsShader(cmd, n, gs) }

    fn sceGnmInsertPushMarker(cmd: *mut u32, n: u32, text: *const u8) -> i32 {
        let mut w = Writer::new(cmd, n);
        w.marker(marker::PUSH, cstr(text));
        w.finish();
        0
    }
    fn sceGnmInsertPushColorMarker(cmd: *mut u32, n: u32, text: *const u8, color: u32) -> i32 {
        let mut w = Writer::new(cmd, n);
        w.marker(marker::PUSH, cstr(text));
        w.finish();
        0
    }
    fn sceGnmInsertPopMarker(cmd: *mut u32, n: u32) -> i32 {
        let mut w = Writer::new(cmd, n);
        w.packet(pm4::op::NOP, &[marker::POP]);
        w.finish();
        0
    }
    fn sceGnmInsertSetMarker(cmd: *mut u32, n: u32, text: *const u8) -> i32 {
        let mut w = Writer::new(cmd, n);
        w.marker(marker::SET, cstr(text));
        w.finish();
        0
    }
    fn sceGnmInsertSetColorMarker(cmd: *mut u32, n: u32, text: *const u8, color: u32) -> i32 {
        let mut w = Writer::new(cmd, n);
        w.marker(marker::SET, cstr(text));
        w.finish();
        0
    }
    fn sceGnmInsertWaitFlipDone(cmd: *mut u32, n: u32, handle: i32, index: i32) -> i32 {
        let mut w = Writer::new(cmd, n);
        if !w.packet(pm4::op::NOP, &[marker::WAIT_FLIP, handle as u32, index as u32]) { return ERR_INVALID; }
        w.finish();
        0
    }

    fn sceGnmSubmitCommandBuffers(count: u32, dcb: *const u64, dcb_sizes: *const u32, ccb: *const u64, ccb_sizes: *const u32) -> i32 {
        submit(count, dcb, dcb_sizes, ccb, ccb_sizes, None)
    }
    fn sceGnmSubmitAndFlipCommandBuffers(count: u32, dcb: *const u64, dcb_sizes: *const u32, ccb: *const u64, ccb_sizes: *const u32, handle: i32, index: i32, mode: u32, arg: i64) -> i32 {
        submit(count, dcb, dcb_sizes, ccb, ccb_sizes, Some(FlipRequest { handle, index, mode, arg }))
    }
    fn sceGnmAddEqEvent(eq: u64, id: u64, udata: u64) -> i32 {
        gpu::add_eop_event(eq, id);
        crate::info!("sceGnmAddEqEvent(eq {eq:#x}, id {id:#x})");
        0
    }
    fn sceGnmDeleteEqEvent(eq: u64, id: u64) -> i32 { gpu::remove_eop_event(eq, id); 0 }
    fn sceGnmGetEqEventType(ev: *const u64) -> u32 { if ev.is_null() { 0 } else { *ev as u32 } }
    fn sceGnmGetEqTimeStamp(ev: *const u64) -> u64 { if ev.is_null() { 0 } else { *ev.add(2) } }
    fn sceGnmMapComputeQueue(pipe: u32, queue: u32, base: u64, size_dw: u32, rptr: u64) -> i32 {
        gpu::map_compute_queue(pipe, queue, base, size_dw, rptr)
    }
    fn sceGnmMapComputeQueueWithPriority(pipe: u32, queue: u32, base: u64, size_dw: u32, rptr: u64, prio: u32) -> i32 {
        gpu::map_compute_queue(pipe, queue, base, size_dw, rptr)
    }
    fn sceGnmDingDong(vq: u32, next_dw: u32) { gpu::ding_dong(vq, next_dw) }
    fn sceGnmLogicalCuMaskToPhysicalCuMask(mask: u32) -> u32 { mask }
    fn sceGnmGetTheTessellationFactorRingBufferBaseAddress() -> u64 { TESS_FACTOR_RING }
}

/// Defensive bound on DCBs per submission (guards against garbage counts).
/// Titles submit one DCB per recorded command-list segment; Bloodborne's
/// end-of-frame submitter does not split its list, so the bound must be well
/// above typical counts (an earlier cap of 32 silently dropped whole frames'
/// worth of command lists, leaking the title's command-memory pool).
const MAX_SUBMIT_DCBS: u32 = 4096;

fn submit(count: u32, dcb: *const u64, dcb_sizes: *const u32, ccb: *const u64, ccb_sizes: *const u32, flip: Option<FlipRequest>) -> i32 {
    if dcb.is_null() || dcb_sizes.is_null() || count == 0 || count > MAX_SUBMIT_DCBS {
        crate::error!("sceGnmSubmit*CommandBuffers rejected: count {count}, dcb {dcb:?}, sizes {dcb_sizes:?}");
        return ERR_INVALID;
    }
    for i in 0..count as usize {
        let (d, c) = unsafe { (*dcb_sizes.add(i), if ccb_sizes.is_null() { 0 } else { *ccb_sizes.add(i) }) };
        if d == 0 || d > 0x3F_FFFC || c > 0x3F_FFFC {
            crate::error!("sceGnmSubmit*CommandBuffers rejected: DCB {i} of {d} bytes, CCB {c} bytes");
            return 0x80D1_1000u32 as i32;
        }
    }
    let mut s = Submission { dcb: Vec::new(), ccb: Vec::new(), flip, captured: Default::default(), submitted: std::time::Instant::now() };
    if INIT_PENDING.swap(false, std::sync::atomic::Ordering::Relaxed) {
        s.dcb.push(hw_init_buffer());
        s.ccb.push((0, 0));
    }
    unsafe {
        for i in 0..count as usize {
            s.dcb.push((*dcb.add(i), *dcb_sizes.add(i) / 4));
            if !ccb.is_null() && !ccb_sizes.is_null() {
                s.ccb.push((*ccb.add(i), *ccb_sizes.add(i) / 4));
            } else {
                s.ccb.push((0, 0));
            }
        }
        if s.flip.is_some() {
            let (last, size) = *s.dcb.last().unwrap();
            if let Err(e) = patch_prepare_flip(last, size) {
                crate::error!("flip submission: {e}");
                return 0x80D1_1000u32 as i32;
            }
        }
    }
    if std::env::var_os("SOTC_SUBMIT_LOG").is_some() {
        let total: u64 = s.dcb.iter().map(|d| d.1 as u64 * 4).sum();
        crate::info!("submit {} dcb(s) {} bytes{}: {:x?}", s.dcb.len(), total, if s.flip.is_some() { " +flip" } else { "" }, s.dcb.iter().take(8).map(|d| d.0).collect::<Vec<_>>());

    }
    if let Some(f) = &s.flip {
        crate::hle::libs::videoout::gpu_flip_requested(f.handle);
    }
    gpu::submit(s);
    0
}

// ---------------------------------------------------------------------------
// Builder functions Shadow of the Colossus calls (SDK 4.5). Each requires the
// exact reserved size the real driver checks and fills it completely: its
// packet(s), then a trailing NOP over the rest.

const ERR_FAILURE: i32 = 0x8D00_0001u32 as i32; // retail firmware: feature unavailable
/// SPI_SHADER_USER_DATA_*_0 offsets (SH register space) of the stages an
/// indirect draw can patch base vertex / instance SGPRs into, by stage id
/// (cs, ps, vs, gs, es, hs, ls).
const INDIRECT_SGPR_BASE: [u32; 7] = [0, 0, 0x4C, 0, 0xCC, 0, 0x14C];

fn pred(flags: u32) -> u32 {
    flags & 1
}

/// Type-3 header with the predicate bit and the shader-type (compute) bit.
fn header(op: u32, payload: u32, predicate: u32, compute: bool) -> u32 {
    pm4::type3(op, payload) | predicate | if compute { 2 } else { 0 }
}

unsafe fn fill(cmd: *mut u32, words: &[u32], total: u32) {
    let buf = unsafe { std::slice::from_raw_parts_mut(cmd, total as usize) };
    buf[..words.len()].copy_from_slice(words);
    pm4::write_nop(&mut buf[words.len()..], total as usize - words.len());
}

fn sgpr_slot(stage: u32, sgpr: u32) -> u32 {
    if sgpr == 0 { 0 } else { (sgpr & 0xFFFF) + INDIRECT_SGPR_BASE[stage as usize] }
}

/// Default register state of SDK 3.50+ titles (hardware init and context init),
/// as (packet, register offset, values); written as real SET_*_REG packets so
/// the command processor's register file matches the hardware's.
const SH: u32 = pm4::op::SET_SH_REG;
const CTX: u32 = pm4::op::SET_CONTEXT_REG;
const UCFG: u32 = pm4::op::SET_UCONFIG_REG;
const DEFAULT_CONTEXT: &[(u32, u32, &[u32])] = &[
    (CTX, 0x2F9, &[0x2D]),        // PA_SU_VTX_CNTL
    (CTX, 0x282, &[8]),           // PA_SU_LINE_CNTL
    (CTX, 0x280, &[0x0008_0008]), // PA_SU_POINT_SIZE
    (CTX, 0x281, &[0xFFFF_0000]), // PA_SU_POINT_MINMAX
    (CTX, 0x204, &[0]),           // PA_CL_CLIP_CNTL
    (CTX, 0x206, &[0x43F]),       // PA_CL_VTE_CNTL
    (CTX, 0x083, &[0xFFFF]),      // PA_SC_CLIPRECT_RULE
    (CTX, 0x317, &[0x10]),        // VGT_OUT_DEALLOC_CNTL
    (CTX, 0x2FA, &[0x3F80_0000]), // PA_CL_GB_VERT_CLIP_ADJ = 1.0
    (CTX, 0x2FC, &[0x3F80_0000]), // PA_CL_GB_HORZ_CLIP_ADJ
    (CTX, 0x2FB, &[0x3F80_0000]), // PA_CL_GB_VERT_DISC_ADJ
    (CTX, 0x2FD, &[0x3F80_0000]), // PA_CL_GB_HORZ_DISC_ADJ
    (CTX, 0x202, &[0x00CC_0010]), // CB_COLOR_CONTROL
    (CTX, 0x30E, &[0xFFFF_FFFF]), // PA_SC_AA_MASK_X0Y0_X1Y0
    (CTX, 0x30F, &[0xFFFF_FFFF]), // PA_SC_AA_MASK_X0Y1_X1Y1
];
const DEFAULT_CONTEXT_2: &[(u32, u32, &[u32])] = &[
    (CTX, 0x1B1, &[2]),                 // SPI_VS_OUT_CONFIG
    (CTX, 0x101, &[0]),                 // VGT_MIN_VTX_INDX
    (CTX, 0x100, &[0xFFFF_FFFF]),       // VGT_MAX_VTX_INDX
    (CTX, 0x103, &[0]),                 // VGT_MULTI_PRIM_IB_RESET_INDX
    (CTX, 0x284, &[0]),
    (CTX, 0x290, &[0]),                 // VGT_GS_MODE
    (CTX, 0x2AE, &[0]),
    (CTX, 0x102, &[0]),                 // VGT_INDX_OFFSET
    (CTX, 0x292, &[0]),                 // PA_SC_MODE_CNTL_0
    (CTX, 0x293, &[0x0602_0000]),       // PA_SC_MODE_CNTL_1
    (CTX, 0x2F8, &[0]),                 // PA_SC_AA_CONFIG
    (CTX, 0x2DE, &[0x1E9]),
    (CTX, 0x295, &[0x100, 0x100, 4]),   // VGT_GS_PER_ES, VGT_ES_PER_GS, VGT_GS_PER_VS
];
const DEFAULT_SHADER_350: &[(u32, u32, &[u32])] = &[
    (SH, 0x007, &[0x0017_01FF]), // SPI_SHADER_PGM_RSRC3_PS
    (SH, 0x046, &[0x0017_01FD]), // SPI_SHADER_PGM_RSRC3_VS
    (SH, 0x087, &[0x0017_01FF]), // SPI_SHADER_PGM_RSRC3_GS
    (SH, 0x0C7, &[0x0017_01FD]), // SPI_SHADER_PGM_RSRC3_ES
    (SH, 0x107, &[0x17]),        // SPI_SHADER_PGM_RSRC3_HS
    (SH, 0x147, &[0x0017_01FD]), // SPI_SHADER_PGM_RSRC3_LS
    (SH, 0x047, &[0x1C]),        // SPI_SHADER_LATE_ALLOC_VS
];

fn write_regs(w: &mut Writer, list: &[(u32, u32, &[u32])]) -> bool {
    list.iter().all(|&(op, reg, vals)| {
        let mut p = vec![reg];
        p.extend_from_slice(vals);
        w.packet(op, &p)
    })
}

/// SDK 3.50+ hardware default state (also injected before the first submission
/// of every frame, as the driver does after sceGnmSubmitDone).
fn hw_init_350(w: &mut Writer) -> bool {
    w.packet(pm4::op::CLEAR_STATE, &[0])
        && w.packet(SH, &[0x216, 0xFFFF_FFFF]) // COMPUTE_STATIC_THREAD_MGMT_SE0
        && w.packet(SH, &[0x217, 0xFFFF_FFFF]) // COMPUTE_STATIC_THREAD_MGMT_SE1
        && w.packet(SH, &[0x215, 0]) // COMPUTE_RESOURCE_LIMITS
        && write_regs(w, DEFAULT_CONTEXT)
        && w.packet(pm4::op::NUM_INSTANCES, &[1])
        && write_regs(w, DEFAULT_SHADER_350)
        && write_regs(w, DEFAULT_CONTEXT_2)
        && w.packet(UCFG, &[0x200, 0xE000_0000]) // GRBM_GFX_INDEX: broadcast
        && w.packet(CTX, &[0x2AA, 0xFF]) // IA_MULTI_VGT_PARAM
}

const HW_INIT_SIZE: u32 = 0x100;

/// Guest buffer holding the per-frame hardware init (built once).
fn hw_init_buffer() -> (u64, u32) {
    static BUF: std::sync::OnceLock<(u64, u32)> = std::sync::OnceLock::new();
    *BUF.get_or_init(|| {
        let addr = crate::mem::arena_alloc(HW_INIT_SIZE as u64 * 4, 256);
        let mut w = unsafe { Writer::new(addr as *mut u32, HW_INIT_SIZE) };
        assert!(hw_init_350(&mut w), "hardware init sequence exceeds its buffer");
        w.finish();
        (addr, HW_INIT_SIZE)
    })
}

/// Set by sceGnmSubmitDone: the next submission starts from the default hardware state.
static INIT_PENDING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Embedded driver shaders (sceGnmSetEmbedded{Vs,Ps}Shader): GCN programs in
/// guest memory, encoded here: VS 0 = full-screen triangle with one parameter;
/// PS 0 = zero colour (FP16 export), PS 1 = zero colour (32-bit export).
struct Embedded {
    vs0: u64,
    ps0: u64,
    ps1: u64,
}

fn embedded() -> &'static Embedded {
    static E: std::sync::OnceLock<Embedded> = std::sync::OnceLock::new();
    E.get_or_init(|| {
        let put = |code: &[u32]| -> u64 {
            let a = crate::mem::arena_alloc(code.len() as u64 * 4, 256);
            unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), a as *mut u32, code.len()) };
            a
        };
        // s_mov_b32 vcc_hi, 7 | v_and_b32 v1, 1, v0 | v_lshlrev_b32 v1, 1, v1 | v_and_b32 v0, -2, v0
        // v_add_i32 v1, vcc, -1, v1 | v_add_i32 v0, vcc, -1, v0 | v_cvt_f32_i32 v1, v1 | v_cvt_f32_i32 v0, v0
        // v_mov_b32 v2, 0 | v_mov_b32 v3, 1.0 | exp pos0 v1 v0 v2 v3 done | exp param0 v3 v3 v3 v3 | s_endpgm
        let vs0 = put(&[
            0xBEEB_03FF, 7, 0x3602_0081, 0x3402_0281, 0x3600_00C2, 0x4A02_02C1, 0x4A00_00C1, 0x7E02_0B01, 0x7E00_0B00,
            0x7E04_0280, 0x7E06_02F2, 0xF800_08CF, 0x0302_0001, 0xF800_020F, 0x0303_0303, 0xBF81_0000,
        ]);
        // s_mov_b32 vcc_hi, 3 | v_mov_b32 v0, 0 | v_cvt_pkrtz_f16_f32 v0, v0, v0 | s_nop 0
        // exp mrt0 v0 v0 compr vm done | s_endpgm
        let ps0 = put(&[0xBEEB_03FF, 3, 0x7E00_0280, 0x5E00_0100, 0xBF80_0000, 0xF800_1C0F, 0, 0xBF81_0000]);
        // s_mov_b32 vcc_hi, 3 | v_mov_b32 v2, 0 | exp mrt0 v2 v2 off off vm done | s_endpgm
        let ps1 = put(&[0xBEEB_03FF, 3, 0x7E04_0280, 0xF800_1803, 0x0202_0202, 0xBF81_0000]);
        Embedded { vs0, ps0, ps1 }
    })
}

/// Rewrites the 64-dword "prepare flip" NOP the title's Gnm library leaves at the
/// end of the last DCB of a flip submission into what it requests: nothing
/// (plain), a label write, or an end-of-pipe interrupt (with or without data).
/// The flip itself is executed after the submission (`Submission::flip`).
unsafe fn patch_prepare_flip(dcb: u64, size_dw: u32) -> Result<(), String> {
    const PREPARE_FLIP: u32 = 0x6875_0777;
    const PREPARE_FLIP_LABEL: u32 = 0x6875_0778;
    const PREPARE_FLIP_INTERRUPT: u32 = 0x6875_0780;
    const PREPARE_FLIP_INTERRUPT_LABEL: u32 = 0x6875_0781;
    const PATCHED_FLIP: u32 = 0x6875_0776;
    if size_dw < 64 {
        return Err(format!("last DCB has {size_dw} dwords, no room for a prepare-flip packet"));
    }
    let p = (dcb + (size_dw as u64 - 64) * 4) as *mut u32;
    let b: Vec<u32> = (0..7).map(|i| unsafe { *p.add(i) }).collect();
    if b[0] != pm4::type3(pm4::op::NOP, 63) {
        return Err(format!("no prepare-flip packet at the end of the DCB (found {:#010x})", b[0]));
    }
    let tail: Vec<u32> = match b[1] {
        PATCHED_FLIP | PREPARE_FLIP => Vec::new(),
        PREPARE_FLIP_LABEL => vec![pm4::type3(pm4::op::WRITE_DATA, 4), 0x500, b[2] & !3, b[3], b[4]],
        PREPARE_FLIP_INTERRUPT_LABEL => vec![
            pm4::type3(pm4::op::EVENT_WRITE_EOP, 4),
            (b[5] & 0x3F) + 0x500 + (b[6] & 0x3F) * 0x1000,
            b[2] & !3,
            (b[3] & 0xFFFF) | 0x2200_0000,
            b[4],
            0,
        ],
        PREPARE_FLIP_INTERRUPT => vec![pm4::type3(pm4::op::EVENT_WRITE_EOP, 4), (b[5] & 0x3F) + 0x500 + (b[6] & 0x3F) * 0x1000, 0, 0x0100_0000, 0, 0],
        other => return Err(format!("unknown prepare-flip payload {other:#010x}")),
    };
    let buf = unsafe { std::slice::from_raw_parts_mut(p, 64) };
    let nop = 64 - tail.len();
    pm4::write_nop(&mut buf[..nop], nop);
    if nop >= 2 {
        buf[1] = PATCHED_FLIP;
    }
    buf[nop..].copy_from_slice(&tail);
    Ok(())
}

sysv! {
    fn sceGnmDrawInitDefaultHardwareState350(cmd: *mut u32, n: u32) -> u32 {
        if cmd.is_null() || n < HW_INIT_SIZE { return 0; }
        let mut w = Writer::new(cmd, HW_INIT_SIZE);
        if !hw_init_350(&mut w) { return 0; }
        w.finish()
    }
    fn sceGnmDrawInitToDefaultContextState400(cmd: *mut u32, n: u32) -> u32 {
        if cmd.is_null() || n != 0x100 { return 0; }
        let mut w = Writer::new(cmd, n);
        let ok = w.packet(pm4::op::CONTEXT_CONTROL, &[0x8000_0000, 0x8000_0000])
            && w.packet(pm4::op::CLEAR_STATE, &[0])
            && write_regs(&mut w, DEFAULT_CONTEXT)
            && w.packet(pm4::op::NUM_INSTANCES, &[1])
            && write_regs(&mut w, DEFAULT_CONTEXT_2)
            && w.packet(CTX, &[0x2AA, 0xFF]);
        if !ok { return 0; }
        w.finish()
    }
    fn sceGnmDrawIndex(cmd: *mut u32, n: u32, count: u32, addr: u64, flags: u32, ty: u32) -> i32 {
        if cmd.is_null() || n != 10 || addr == 0 || addr & 1 != 0 || flags & 0x1FFF_FFFE != 0 { return -1; }
        fill(cmd, &[header(pm4::op::DRAW_INDEX_2, 5, 0, false), count, addr as u32, (addr >> 32) as u32, count, 0], n);
        0
    }
    fn sceGnmDrawIndexAuto(cmd: *mut u32, n: u32, count: u32, flags: u32) -> i32 {
        if cmd.is_null() || n != 7 || flags & 0x1FFF_FFFE != 0 { return -1; }
        fill(cmd, &[header(pm4::op::DRAW_INDEX_AUTO, 2, 0, false), count, 2], n);
        0
    }
    fn sceGnmDrawIndexOffset(cmd: *mut u32, n: u32, offset: u32, count: u32, flags: u32) -> i32 {
        if cmd.is_null() || n != 9 { return -1; }
        fill(cmd, &[header(pm4::op::DRAW_INDEX_OFFSET_2, 4, pred(flags), false), count, offset, count, 0], n);
        0
    }
    fn sceGnmDrawIndexIndirect(cmd: *mut u32, n: u32, data_offset: u32, stage: u32, vsgpr: u32, isgpr: u32, flags: u32) -> i32 {
        if cmd.is_null() || n != 9 || stage >= 7 || vsgpr >= 16 || isgpr >= 16 { return -1; }
        fill(cmd, &[header(pm4::op::DRAW_INDEX_INDIRECT, 4, pred(flags), false), data_offset, sgpr_slot(stage, vsgpr), sgpr_slot(stage, isgpr), 0], n);
        0
    }
    fn sceGnmDrawIndirect(cmd: *mut u32, n: u32, data_offset: u32, stage: u32, vsgpr: u32, isgpr: u32, flags: u32) -> i32 {
        if cmd.is_null() || n != 9 || stage >= 7 || vsgpr >= 16 || isgpr >= 16 { return -1; }
        fill(cmd, &[header(pm4::op::DRAW_INDIRECT, 4, pred(flags), false), data_offset, sgpr_slot(stage, vsgpr), sgpr_slot(stage, isgpr), 2], n);
        0
    }
    fn sceGnmDrawIndexIndirectCountMulti(cmd: *mut u32, n: u32, data_offset: u32, max_count: u32, count_addr: u64, stage: u32, vsgpr: u32, isgpr: u32, flags: u32) -> i32 {
        if cmd.is_null() || n != 16 || vsgpr >= 16 || isgpr >= 16 || !matches!(stage, 2 | 4 | 6) { return -1; }
        fill(cmd, &[
            pm4::type3(pm4::op::NOP, 2), 0, 0,
            header(pm4::op::DRAW_INDEX_INDIRECT_COUNT_MULTI, 9, pred(flags), false),
            data_offset, sgpr_slot(stage, vsgpr), sgpr_slot(stage, isgpr), ((count_addr != 0) as u32) << 30,
            max_count, count_addr as u32, (count_addr >> 32) as u32, 20, 0,
        ], n);
        0
    }
    fn sceGnmDispatchDirect(cmd: *mut u32, n: u32, x: u32, y: u32, z: u32, flags: u32) -> i32 {
        if cmd.is_null() || n != 9 || ((x | y | z) as i32) < 0 { return -1; }
        fill(cmd, &[header(pm4::op::DISPATCH_DIRECT, 4, pred(flags), true), x, y, z, (flags & 0x18) + 1], n);
        0
    }
    fn sceGnmDispatchIndirect(cmd: *mut u32, n: u32, data_offset: u32, flags: u32) -> i32 {
        if cmd.is_null() || n != 7 { return -1; }
        fill(cmd, &[header(pm4::op::DISPATCH_INDIRECT, 2, pred(flags), true), data_offset, (flags & 0x18) + 1], n);
        0
    }
    // Compute-queue form: the argument address is absolute (three payload dwords).
    fn sceGnmDispatchIndirectOnMec(cmd: *mut u32, n: u32, args: u64, modifier: u32) -> i32 {
        if cmd.is_null() || n != 8 || args == 0 || args & 3 != 0 { return -1; }
        fill(cmd, &[header(pm4::op::DISPATCH_INDIRECT, 3, modifier & 1, true), args as u32, (args >> 32) as u32, (modifier & 0x18) | 1], n);
        0
    }
    fn sceGnmSetCsShaderWithModifier(cmd: *mut u32, n: u32, cs: *const u32, modifier: u32) -> i32 {
        if cmd.is_null() || n <= 0x18 || modifier & 0xFFFF_FC3F != 0 { return -1; }
        let Some(c) = regs(cs, 7) else { return -1 };
        if c[1] != 0 { return -1; }
        let rsrc1 = if modifier == 0 { c[2] } else { (c[2] & 0xFFFF_FC3F) | modifier };
        let mut w = Writer::new(cmd, n);
        let ok = w.set_sh(COMPUTE_PGM_LO, &[c[0], 0]) && w.set_sh(COMPUTE_PGM_RSRC1, &[rsrc1, c[3]]) && w.set_sh(COMPUTE_NUM_THREAD_X, &[c[4], c[5], c[6]]);
        if !ok { return -1; }
        w.finish();
        0
    }
    fn sceGnmSetPsShader350(cmd: *mut u32, n: u32, ps: *const u32) -> i32 {
        if cmd.is_null() || n <= 0x27 { return -1; }
        let mut w = Writer::new(cmd, n);
        let ok = match regs(ps, 12) {
            Some(p) => p[1] == 0
                && w.set_sh(SPI_SHADER_PGM_LO_PS, &[p[0], 0])
                && w.set_sh(SPI_SHADER_PGM_RSRC1_PS, &[p[2], p[3]])
                && w.set_ctx(SPI_SHADER_Z_FORMAT, &[p[4], p[5]])
                && w.set_ctx(SPI_PS_INPUT_ENA, &[p[6], p[7]])
                && w.set_ctx(SPI_PS_IN_CONTROL, &[p[8]])
                && w.set_ctx(SPI_BARYC_CNTL, &[p[9]])
                && w.set_ctx(DB_SHADER_CONTROL, &[p[10]])
                && w.set_ctx(CB_SHADER_MASK, &[p[11]]),
            None => w.set_sh(SPI_SHADER_PGM_LO_PS, &[0, 0]) && w.set_ctx(DB_SHADER_CONTROL, &[0]) && w.set_ctx(CB_SHADER_MASK, &[0xF]),
        };
        if !ok { return -1; }
        w.finish();
        0
    }
    fn sceGnmUpdatePsShader350(cmd: *mut u32, n: u32, ps: *const u32) -> i32 { sceGnmSetPsShader350(cmd, n, ps) }
    fn sceGnmSetEmbeddedVsShader(cmd: *mut u32, n: u32, id: u32, modifier: u32) -> i32 {
        if cmd.is_null() || id != 0 { return ERR_FAILURE; }
        let a = embedded().vs0;
        let mut w = Writer::new(cmd, n);
        let ok = w.set_sh(SPI_SHADER_PGM_LO_VS, &[(a >> 8) as u32, (a >> 40) as u32])
            && w.set_sh(SPI_SHADER_PGM_RSRC1_VS, &[modified_rsrc1(0x000C_0000, modifier), 4])
            && w.set_ctx(PA_CL_VS_OUT_CNTL, &[0])
            && w.set_ctx(SPI_VS_OUT_CONFIG, &[0])
            && w.set_ctx(SPI_SHADER_POS_FORMAT, &[4]);
        if !ok { return -1; }
        w.finish();
        0
    }
    fn sceGnmSetEmbeddedPsShader(cmd: *mut u32, n: u32, id: u32, modifier: u32) -> i32 {
        if cmd.is_null() || id > 1 { return ERR_FAILURE; }
        let e = embedded();
        // PGM, RSRC1/2, Z/COL_FORMAT, PS_INPUT_ENA/ADDR, PS_IN_CONTROL, BARYC_CNTL, DB_SHADER_CONTROL, CB_SHADER_MASK
        let (a, r): (u64, [u32; 11]) = if id == 0 {
            (e.ps0, [0x000C_0000, 4, 0, 4, 2, 2, 0, 0, 0x10, 0xF, 0])
        } else {
            (e.ps1, [0x2000, 0, 0, 2, 2, 2, 0, 0, 0x10, 3, 0])
        };
        let mut w = Writer::new(cmd, n);
        let ok = w.set_sh(SPI_SHADER_PGM_LO_PS, &[(a >> 8) as u32, (a >> 40) as u32])
            && w.set_sh(SPI_SHADER_PGM_RSRC1_PS, &[r[0], r[1]])
            && w.set_ctx(SPI_SHADER_Z_FORMAT, &[r[2], r[3]])
            && w.set_ctx(SPI_PS_INPUT_ENA, &[r[4], r[5]])
            && w.set_ctx(SPI_PS_IN_CONTROL, &[r[6]])
            && w.set_ctx(SPI_BARYC_CNTL, &[r[7]])
            && w.set_ctx(DB_SHADER_CONTROL, &[r[8]])
            && w.set_ctx(CB_SHADER_MASK, &[r[9]]);
        if !ok { return -1; }
        w.finish();
        0
    }
    fn sceGnmSetVgtControl(cmd: *mut u32, n: u32, prim_group_minus_one: u32, partial_vs_wave: u32, wd_switch_eop: u32) -> i32 {
        if cmd.is_null() || n != 3 || prim_group_minus_one >= 0x100 || (wd_switch_eop | partial_vs_wave) >= 2 { return -1; }
        fill(cmd, &[pm4::type3(CTX, 2), 0x2AA, ((partial_vs_wave & 1) << 16) | prim_group_minus_one], n); // IA_MULTI_VGT_PARAM
        0
    }
    fn sceGnmResetVgtControl(cmd: *mut u32, n: u32) -> i32 {
        if cmd.is_null() || n != 3 { return -1; }
        fill(cmd, &[pm4::type3(CTX, 2), 0x2AA, 0xFF], n);
        0
    }
    fn sceGnmComputeWaitOnAddress(cmd: *mut u32, n: u32, addr: u64, mask: u32, func: u32, reference: u32) -> i32 {
        if cmd.is_null() || n != 0xE { return -1; }
        let mem = addr > 0xFFFF;
        let lo = if mem { addr as u32 & !3 } else { addr as u32 & 0xFFFF };
        fill(cmd, &[
            pm4::type3(pm4::op::NOP, 3), 0, 0, 0,
            pm4::type3(pm4::op::WAIT_REG_MEM, 6), ((mem as u32) << 4) | (func & 7), lo, (addr >> 32) as u32, reference, mask, 10,
        ], n);
        0
    }
    fn sceGnmInsertDingDongMarker(cmd: *mut u32, n: u32) -> i32 {
        if cmd.is_null() || n != 4 { return -1; }
        fill(cmd, &[pm4::type3(pm4::op::NOP, 3), 0, 0, 0], n);
        0
    }
    fn sceGnmSubmitCommandBuffersForWorkload(workload: u64, count: u32, dcb: *const u64, dcb_sizes: *const u32, ccb: *const u64, ccb_sizes: *const u32) -> i32 {
        submit(count, dcb, dcb_sizes, ccb, ccb_sizes, None)
    }
    fn sceGnmSubmitAndFlipCommandBuffersForWorkload(workload: u64, count: u32, dcb: *const u64, dcb_sizes: *const u32, ccb: *const u64, ccb_sizes: *const u32, handle: i32, index: i32, mode: u32, arg: i64) -> i32 {
        submit(count, dcb, dcb_sizes, ccb, ccb_sizes, Some(FlipRequest { handle, index, mode, arg }))
    }
    fn sceGnmCreateWorkloadStream(name: u64, stream: *mut u32) -> i32 {
        if name == 0 || stream.is_null() { return 3; }
        *stream = 1;
        0
    }
    fn sceGnmDestroyWorkloadStream(stream: u32) -> i32 { 0 }
    fn sceGnmBeginWorkload(stream: u32, workload: *mut u64) -> i32 {
        if workload.is_null() { return 3; }
        *workload = (stream < 16) as u64;
        (stream > 15) as i32
    }
    fn sceGnmEndWorkload(workload: u64) -> i32 {
        if workload == 0 { 2 } else { (((workload >> 56) & 0xFF) > 15) as i32 * 2 }
    }
    fn sceGnmDingDongForWorkload(vq: u32, next_dw: u32, workload: u64) { gpu::ding_dong(vq, next_dw) }
    fn sceGnmGetOffChipTessellationBufferSize() -> u32 { 0x80_0000 }
    fn sceGnmAreSubmitsAllowed() -> i32 { 1 }
    fn sceGnmLogicalCuIndexToPhysicalCuIndex(index: u32) -> u32 { index }
}

fixed! {
    // Debug/profiling facilities a retail console does not provide.
    sceGnmDriverTraceInProgress => 0,
    sceGnmGetGpuBlockStatus => 0,
    sceGnmGetGpuInfoStatus => 0,
    sceGnmGetLastWaitedAddress => 0,
    sceGnmGetNumTcaUnits => 0,
    sceGnmLogicalTcaUnitToPhysical => 0,
    sceGnmGetProtectionFaultTimeStamp => 0,
    sceGnmGetShaderProgramBaseAddress => 0,
    sceGnmGetShaderStatus => 0,
    sceGnmInsertThreadTraceMarker => 0,
    sceGnmRequestMipStatsReportAndReset => 0,
    sceGnmSetWaveLimitMultipliers => 0,
    sceGnmComputeWaitSemaphore => 0,
    sceGnmFindResourcesPublic => ERR_FAILURE,
    sceGnmGetOwnerName => ERR_FAILURE,
    sceGnmGetPhysicalCounterFromVirtualized => ERR_FAILURE,
    sceGnmGetResourceBaseAddressAndSizeInBytes => ERR_FAILURE,
    sceGnmGetResourceName => ERR_FAILURE,
    sceGnmGetResourceShaderGuid => ERR_FAILURE,
    sceGnmGetResourceType => ERR_FAILURE,
    sceGnmGetResourceUserData => ERR_FAILURE,
    sceGnmQueryResourceRegistrationUserMemoryRequirements => ERR_FAILURE,
    sceGnmRegisterGdsResource => ERR_FAILURE,
    sceGnmRegisterOwner => ERR_FAILURE,
    sceGnmRegisterResource => ERR_FAILURE,
    sceGnmSetResourceRegistrationUserMemory => ERR_FAILURE,
    sceGnmSetResourceUserData => ERR_FAILURE,
    sceGnmUnregisterAllResourcesForOwner => ERR_FAILURE,
    sceGnmUnregisterOwnerAndResources => ERR_FAILURE,
    sceGnmUnregisterResource => ERR_FAILURE,
}

sysv! {
    /// End of the title's frame: the next submission starts from the default
    /// hardware state again (the driver re-sends its init sequence).
    fn sceGnmSubmitDone() -> i32 {
        INIT_PENDING.store(true, std::sync::atomic::Ordering::Relaxed);
        0
    }
}

fixed! {
    sceGnmFlushGarlic => 0,
    sceGnmValidateCommandBuffers => 0,
    sceGnmDebugHardwareStatus => 0,
    sceGnmUnmapComputeQueue => 0,
    sceGnmSetupMipStatsReport => 0,
    sceGnmDisableMipStatsReport => 0,
    sceGnmSetGsRingSizes => 0,
    sceGnmIsUserPaEnabled => 0,
    sceGnmDriverCaptureInProgress => 0,
    sceGnmDriverTriggerCapture => 0,
    sceRazorIsLoaded => 0,
    sceRazorCaptureImmediate => 0,
    sceRazorCaptureCommandBuffersOnlyImmediate => 0,
    sceRazorCaptureSinceLastFlip => 0,
    sceRazorCaptureCommandBuffersOnlySinceLastFlip => 0,
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceGnmDriver", [
        sceGnmDrawInitDefaultHardwareState200, sceGnmDispatchInitDefaultHardwareState, sceGnmSetVsShader,
        sceGnmUpdateVsShader, sceGnmSetPsShader, sceGnmUpdatePsShader, sceGnmSetCsShader, sceGnmSetEsShader,
        sceGnmSetLsShader, sceGnmSetHsShader, sceGnmUpdateHsShader, sceGnmSetGsShader, sceGnmUpdateGsShader,
        sceGnmInsertPushMarker, sceGnmInsertPushColorMarker, sceGnmInsertPopMarker, sceGnmInsertSetMarker,
        sceGnmInsertSetColorMarker, sceGnmInsertWaitFlipDone, sceGnmSubmitCommandBuffers,
        sceGnmSubmitAndFlipCommandBuffers, sceGnmAddEqEvent, sceGnmDeleteEqEvent, sceGnmGetEqEventType,
        sceGnmGetEqTimeStamp, sceGnmMapComputeQueue, sceGnmMapComputeQueueWithPriority, sceGnmDingDong,
        sceGnmLogicalCuMaskToPhysicalCuMask, sceGnmGetTheTessellationFactorRingBufferBaseAddress, sceGnmSubmitDone,
        sceGnmFlushGarlic, sceGnmValidateCommandBuffers, sceGnmDebugHardwareStatus, sceGnmUnmapComputeQueue,
        sceGnmSetupMipStatsReport, sceGnmDisableMipStatsReport, sceGnmSetGsRingSizes, sceGnmIsUserPaEnabled,
        sceGnmDriverCaptureInProgress, sceGnmDriverTriggerCapture, sceRazorIsLoaded, sceRazorCaptureImmediate,
        sceRazorCaptureCommandBuffersOnlyImmediate, sceRazorCaptureSinceLastFlip,
        sceRazorCaptureCommandBuffersOnlySinceLastFlip,
        // Shadow of the Colossus additions.
        sceGnmDrawInitDefaultHardwareState350, sceGnmDrawInitToDefaultContextState400, sceGnmDrawIndex,
        sceGnmDrawIndexAuto, sceGnmDrawIndexOffset, sceGnmDrawIndexIndirect, sceGnmDrawIndirect,
        sceGnmDrawIndexIndirectCountMulti, sceGnmDispatchDirect, sceGnmDispatchIndirect, sceGnmDispatchIndirectOnMec,
        sceGnmSetCsShaderWithModifier, sceGnmSetPsShader350, sceGnmUpdatePsShader350, sceGnmSetEmbeddedVsShader,
        sceGnmSetEmbeddedPsShader, sceGnmSetVgtControl, sceGnmResetVgtControl, sceGnmComputeWaitOnAddress,
        sceGnmInsertDingDongMarker, sceGnmSubmitCommandBuffersForWorkload, sceGnmSubmitAndFlipCommandBuffersForWorkload,
        sceGnmCreateWorkloadStream, sceGnmDestroyWorkloadStream, sceGnmBeginWorkload, sceGnmEndWorkload,
        sceGnmDingDongForWorkload, sceGnmGetOffChipTessellationBufferSize, sceGnmAreSubmitsAllowed,
        sceGnmLogicalCuIndexToPhysicalCuIndex, sceGnmDriverTraceInProgress, sceGnmGetGpuBlockStatus,
        sceGnmGetGpuInfoStatus, sceGnmGetLastWaitedAddress, sceGnmGetNumTcaUnits, sceGnmLogicalTcaUnitToPhysical,
        sceGnmGetProtectionFaultTimeStamp, sceGnmGetShaderProgramBaseAddress, sceGnmGetShaderStatus,
        sceGnmInsertThreadTraceMarker, sceGnmRequestMipStatsReportAndReset, sceGnmSetWaveLimitMultipliers,
        sceGnmComputeWaitSemaphore, sceGnmFindResourcesPublic, sceGnmGetOwnerName,
        sceGnmGetPhysicalCounterFromVirtualized, sceGnmGetResourceBaseAddressAndSizeInBytes, sceGnmGetResourceName,
        sceGnmGetResourceShaderGuid, sceGnmGetResourceType, sceGnmGetResourceUserData,
        sceGnmQueryResourceRegistrationUserMemoryRequirements, sceGnmRegisterGdsResource, sceGnmRegisterOwner,
        sceGnmRegisterResource, sceGnmSetResourceRegistrationUserMemory, sceGnmSetResourceUserData,
        sceGnmUnregisterAllResourcesForOwner, sceGnmUnregisterOwnerAndResources, sceGnmUnregisterResource,
    ]);
}
