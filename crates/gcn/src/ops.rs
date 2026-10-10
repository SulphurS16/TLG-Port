//! Opcode name tables for gfx7 (Sea Islands / PS4 Liverpool). Names follow
//! AMD's ISA documentation; values are the encoding-local opcode numbers.

pub type Table = &'static [(u16, &'static str)];

pub fn lookup(t: Table, op: u16) -> Option<&'static str> {
    t.iter().find(|(o, _)| *o == op).map(|(_, n)| *n)
}

pub const SOP2: Table = &[
    (0, "s_add_u32"), (1, "s_sub_u32"), (2, "s_add_i32"), (3, "s_sub_i32"), (4, "s_addc_u32"), (5, "s_subb_u32"),
    (6, "s_min_i32"), (7, "s_min_u32"), (8, "s_max_i32"), (9, "s_max_u32"), (10, "s_cselect_b32"),
    (11, "s_cselect_b64"), (14, "s_and_b32"), (15, "s_and_b64"), (16, "s_or_b32"), (17, "s_or_b64"),
    (18, "s_xor_b32"), (19, "s_xor_b64"), (20, "s_andn2_b32"), (21, "s_andn2_b64"), (22, "s_orn2_b32"),
    (23, "s_orn2_b64"), (24, "s_nand_b32"), (25, "s_nand_b64"), (26, "s_nor_b32"), (27, "s_nor_b64"),
    (28, "s_xnor_b32"), (29, "s_xnor_b64"), (30, "s_lshl_b32"), (31, "s_lshl_b64"), (32, "s_lshr_b32"),
    (33, "s_lshr_b64"), (34, "s_ashr_i32"), (35, "s_ashr_i64"), (36, "s_bfm_b32"), (37, "s_bfm_b64"),
    (38, "s_mul_i32"), (39, "s_bfe_u32"), (40, "s_bfe_i32"), (41, "s_bfe_u64"), (42, "s_bfe_i64"),
    (43, "s_cbranch_g_fork"), (44, "s_absdiff_i32"),
];

pub const SOPK: Table = &[
    (0, "s_movk_i32"), (2, "s_cmovk_i32"), (3, "s_cmpk_eq_i32"), (4, "s_cmpk_lg_i32"), (5, "s_cmpk_gt_i32"),
    (6, "s_cmpk_ge_i32"), (7, "s_cmpk_lt_i32"), (8, "s_cmpk_le_i32"), (9, "s_cmpk_eq_u32"), (10, "s_cmpk_lg_u32"),
    (11, "s_cmpk_gt_u32"), (12, "s_cmpk_ge_u32"), (13, "s_cmpk_lt_u32"), (14, "s_cmpk_le_u32"), (15, "s_addk_i32"),
    (16, "s_mulk_i32"), (17, "s_cbranch_i_fork"), (18, "s_getreg_b32"), (19, "s_setreg_b32"),
    (21, "s_setreg_imm32_b32"),
];

pub const SOP1: Table = &[
    (3, "s_mov_b32"), (4, "s_mov_b64"), (5, "s_cmov_b32"), (6, "s_cmov_b64"), (7, "s_not_b32"), (8, "s_not_b64"),
    (9, "s_wqm_b32"), (10, "s_wqm_b64"), (11, "s_brev_b32"), (12, "s_brev_b64"), (13, "s_bcnt0_i32_b32"),
    (14, "s_bcnt0_i32_b64"), (15, "s_bcnt1_i32_b32"), (16, "s_bcnt1_i32_b64"), (17, "s_ff0_i32_b32"),
    (18, "s_ff0_i32_b64"), (19, "s_ff1_i32_b32"), (20, "s_ff1_i32_b64"), (21, "s_flbit_i32_b32"),
    (22, "s_flbit_i32_b64"), (23, "s_flbit_i32"), (24, "s_flbit_i32_i64"), (25, "s_sext_i32_i8"),
    (26, "s_sext_i32_i16"), (27, "s_bitset0_b32"), (28, "s_bitset0_b64"), (29, "s_bitset1_b32"),
    (30, "s_bitset1_b64"), (31, "s_getpc_b64"), (32, "s_setpc_b64"), (33, "s_swappc_b64"), (34, "s_rfe_b64"),
    (36, "s_and_saveexec_b64"), (37, "s_or_saveexec_b64"), (38, "s_xor_saveexec_b64"),
    (39, "s_andn2_saveexec_b64"), (40, "s_orn2_saveexec_b64"), (41, "s_nand_saveexec_b64"),
    (42, "s_nor_saveexec_b64"), (43, "s_xnor_saveexec_b64"), (44, "s_quadmask_b32"), (45, "s_quadmask_b64"),
    (46, "s_movrels_b32"), (47, "s_movrels_b64"), (48, "s_movreld_b32"), (49, "s_movreld_b64"),
    (50, "s_cbranch_join"), (52, "s_abs_i32"), (53, "s_mov_fed_b32"),
];

pub const SOPC: Table = &[
    (0, "s_cmp_eq_i32"), (1, "s_cmp_lg_i32"), (2, "s_cmp_gt_i32"), (3, "s_cmp_ge_i32"), (4, "s_cmp_lt_i32"),
    (5, "s_cmp_le_i32"), (6, "s_cmp_eq_u32"), (7, "s_cmp_lg_u32"), (8, "s_cmp_gt_u32"), (9, "s_cmp_ge_u32"),
    (10, "s_cmp_lt_u32"), (11, "s_cmp_le_u32"), (12, "s_bitcmp0_b32"), (13, "s_bitcmp1_b32"),
    (14, "s_bitcmp0_b64"), (15, "s_bitcmp1_b64"), (16, "s_setvskip"),
];

pub const SOPP: Table = &[
    (0, "s_nop"), (1, "s_endpgm"), (2, "s_branch"), (4, "s_cbranch_scc0"), (5, "s_cbranch_scc1"),
    (6, "s_cbranch_vccz"), (7, "s_cbranch_vccnz"), (8, "s_cbranch_execz"), (9, "s_cbranch_execnz"),
    (10, "s_barrier"), (12, "s_waitcnt"), (13, "s_sethalt"), (14, "s_sleep"), (15, "s_setprio"),
    (16, "s_sendmsg"), (17, "s_sendmsghalt"), (18, "s_trap"), (19, "s_icache_inv"), (20, "s_incperflevel"),
    (21, "s_decperflevel"), (22, "s_ttracedata"), (23, "s_cbranch_cdbgsys"), (24, "s_cbranch_cdbguser"),
    (25, "s_cbranch_cdbgsys_or_user"), (26, "s_cbranch_cdbgsys_and_user"),
];

pub const SMRD: Table = &[
    (0, "s_load_dword"), (1, "s_load_dwordx2"), (2, "s_load_dwordx4"), (3, "s_load_dwordx8"),
    (4, "s_load_dwordx16"), (8, "s_buffer_load_dword"), (9, "s_buffer_load_dwordx2"),
    (10, "s_buffer_load_dwordx4"), (11, "s_buffer_load_dwordx8"), (12, "s_buffer_load_dwordx16"),
    (29, "s_dcache_inv_vol"), (30, "s_memtime"), (31, "s_dcache_inv"),
];

pub const VOP2: Table = &[
    (0, "v_cndmask_b32"), (1, "v_readlane_b32"), (2, "v_writelane_b32"), (3, "v_add_f32"), (4, "v_sub_f32"),
    (5, "v_subrev_f32"), (6, "v_mac_legacy_f32"), (7, "v_mul_legacy_f32"), (8, "v_mul_f32"), (9, "v_mul_i32_i24"),
    (10, "v_mul_hi_i32_i24"), (11, "v_mul_u32_u24"), (12, "v_mul_hi_u32_u24"), (13, "v_min_legacy_f32"),
    (14, "v_max_legacy_f32"), (15, "v_min_f32"), (16, "v_max_f32"), (17, "v_min_i32"), (18, "v_max_i32"),
    (19, "v_min_u32"), (20, "v_max_u32"), (21, "v_lshr_b32"), (22, "v_lshrrev_b32"), (23, "v_ashr_i32"),
    (24, "v_ashrrev_i32"), (25, "v_lshl_b32"), (26, "v_lshlrev_b32"), (27, "v_and_b32"), (28, "v_or_b32"),
    (29, "v_xor_b32"), (30, "v_bfm_b32"), (31, "v_mac_f32"), (32, "v_madmk_f32"), (33, "v_madak_f32"),
    (34, "v_bcnt_u32_b32"), (35, "v_mbcnt_lo_u32_b32"), (36, "v_mbcnt_hi_u32_b32"), (37, "v_add_i32"),
    (38, "v_sub_i32"), (39, "v_subrev_i32"), (40, "v_addc_u32"), (41, "v_subb_u32"), (42, "v_subbrev_u32"),
    (43, "v_ldexp_f32"), (44, "v_cvt_pkaccum_u8_f32"), (45, "v_cvt_pknorm_i16_f32"),
    (46, "v_cvt_pknorm_u16_f32"), (47, "v_cvt_pkrtz_f16_f32"), (48, "v_cvt_pk_u16_u32"),
    (49, "v_cvt_pk_i16_i32"),
];

pub const VOP1: Table = &[
    (0, "v_nop"), (1, "v_mov_b32"), (2, "v_readfirstlane_b32"), (3, "v_cvt_i32_f64"), (4, "v_cvt_f64_i32"),
    (5, "v_cvt_f32_i32"), (6, "v_cvt_f32_u32"), (7, "v_cvt_u32_f32"), (8, "v_cvt_i32_f32"), (9, "v_mov_fed_b32"),
    (10, "v_cvt_f16_f32"), (11, "v_cvt_f32_f16"), (12, "v_cvt_rpi_i32_f32"), (13, "v_cvt_flr_i32_f32"),
    (14, "v_cvt_off_f32_i4"), (15, "v_cvt_f32_f64"), (16, "v_cvt_f64_f32"), (17, "v_cvt_f32_ubyte0"),
    (18, "v_cvt_f32_ubyte1"), (19, "v_cvt_f32_ubyte2"), (20, "v_cvt_f32_ubyte3"), (21, "v_cvt_u32_f64"),
    (22, "v_cvt_f64_u32"), (23, "v_trunc_f64"), (24, "v_ceil_f64"), (25, "v_rndne_f64"), (26, "v_floor_f64"),
    (32, "v_fract_f32"), (33, "v_trunc_f32"), (34, "v_ceil_f32"), (35, "v_rndne_f32"), (36, "v_floor_f32"),
    (37, "v_exp_f32"), (38, "v_log_clamp_f32"), (39, "v_log_f32"), (40, "v_rcp_clamp_f32"),
    (41, "v_rcp_legacy_f32"), (42, "v_rcp_f32"), (43, "v_rcp_iflag_f32"), (44, "v_rsq_clamp_f32"),
    (45, "v_rsq_legacy_f32"), (46, "v_rsq_f32"), (47, "v_rcp_f64"), (48, "v_rcp_clamp_f64"), (49, "v_rsq_f64"),
    (50, "v_rsq_clamp_f64"), (51, "v_sqrt_f32"), (52, "v_sqrt_f64"), (53, "v_sin_f32"), (54, "v_cos_f32"),
    (55, "v_not_b32"), (56, "v_bfrev_b32"), (57, "v_ffbh_u32"), (58, "v_ffbl_b32"), (59, "v_ffbh_i32"),
    (60, "v_frexp_exp_i32_f64"), (61, "v_frexp_mant_f64"), (62, "v_fract_f64"), (63, "v_frexp_exp_i32_f32"),
    (64, "v_frexp_mant_f32"), (65, "v_clrexcp"), (66, "v_movreld_b32"), (67, "v_movrels_b32"),
    (68, "v_movrelsd_b32"), (69, "v_log_legacy_f32"), (70, "v_exp_legacy_f32"),
];

/// VOP3-only opcodes (0x140..0x17F).
pub const VOP3: Table = &[
    (0x140, "v_mad_legacy_f32"), (0x141, "v_mad_f32"), (0x142, "v_mad_i32_i24"), (0x143, "v_mad_u32_u24"),
    (0x144, "v_cubeid_f32"), (0x145, "v_cubesc_f32"), (0x146, "v_cubetc_f32"), (0x147, "v_cubema_f32"),
    (0x148, "v_bfe_u32"), (0x149, "v_bfe_i32"), (0x14A, "v_bfi_b32"), (0x14B, "v_fma_f32"), (0x14C, "v_fma_f64"),
    (0x14D, "v_lerp_u8"), (0x14E, "v_alignbit_b32"), (0x14F, "v_alignbyte_b32"), (0x150, "v_mullit_f32"),
    (0x151, "v_min3_f32"), (0x152, "v_min3_i32"), (0x153, "v_min3_u32"), (0x154, "v_max3_f32"),
    (0x155, "v_max3_i32"), (0x156, "v_max3_u32"), (0x157, "v_med3_f32"), (0x158, "v_med3_i32"),
    (0x159, "v_med3_u32"), (0x15A, "v_sad_u8"), (0x15B, "v_sad_hi_u8"), (0x15C, "v_sad_u16"), (0x15D, "v_sad_u32"),
    (0x15E, "v_cvt_pk_u8_f32"), (0x15F, "v_div_fixup_f32"), (0x160, "v_div_fixup_f64"), (0x161, "v_lshl_b64"),
    (0x162, "v_lshr_b64"), (0x163, "v_ashr_i64"), (0x164, "v_add_f64"), (0x165, "v_mul_f64"), (0x166, "v_min_f64"),
    (0x167, "v_max_f64"), (0x168, "v_ldexp_f64"), (0x169, "v_mul_lo_u32"), (0x16A, "v_mul_hi_u32"),
    (0x16B, "v_mul_lo_i32"), (0x16C, "v_mul_hi_i32"), (0x16D, "v_div_scale_f32"), (0x16E, "v_div_scale_f64"),
    (0x16F, "v_div_fmas_f32"), (0x170, "v_div_fmas_f64"), (0x171, "v_msad_u8"), (0x172, "v_qsad_pk_u16_u8"),
    (0x173, "v_mqsad_pk_u16_u8"), (0x174, "v_trig_preop_f64"), (0x175, "v_mqsad_u32_u8"), (0x176, "v_mad_u64_u32"),
    (0x177, "v_mad_i64_i32"),
];

const CMP_F: [&str; 16] = ["f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "tru"];
const CMP_I: [&str; 8] = ["f", "lt", "eq", "le", "gt", "ne", "ge", "t"];

/// Name of a VOPC opcode (0..0xFF).
pub fn vopc_name(op: u16) -> String {
    let hi = op >> 4;
    let lo = (op & 15) as usize;
    match hi {
        0x0 => format!("v_cmp_{}_f32", CMP_F[lo]),
        0x1 => format!("v_cmpx_{}_f32", CMP_F[lo]),
        0x2 => format!("v_cmp_{}_f64", CMP_F[lo]),
        0x3 => format!("v_cmpx_{}_f64", CMP_F[lo]),
        0x4 => format!("v_cmps_{}_f32", CMP_F[lo]),
        0x5 => format!("v_cmpsx_{}_f32", CMP_F[lo]),
        0x6 => format!("v_cmps_{}_f64", CMP_F[lo]),
        0x7 => format!("v_cmpsx_{}_f64", CMP_F[lo]),
        0x8..=0xF => {
            let x = if hi & 1 == 1 { "x" } else { "" };
            let ty = match hi & !1 {
                0x8 => "i32",
                0xA => "i64",
                0xC => "u32",
                _ => "u64",
            };
            if lo == 8 && (hi == 0x8 || hi == 0x9) {
                format!("v_cmp{x}_class_f32")
            } else if lo == 8 && (hi == 0xA || hi == 0xB) {
                format!("v_cmp{x}_class_f64")
            } else if lo < 8 {
                format!("v_cmp{x}_{}_{ty}", CMP_I[lo])
            } else {
                format!("vopc_{op:#x}")
            }
        }
        _ => format!("vopc_{op:#x}"),
    }
}

pub const VINTRP: Table = &[(0, "v_interp_p1_f32"), (1, "v_interp_p2_f32"), (2, "v_interp_mov_f32")];

pub const DS: Table = &[
    (0, "ds_add_u32"), (1, "ds_sub_u32"), (2, "ds_rsub_u32"), (3, "ds_inc_u32"), (4, "ds_dec_u32"),
    (5, "ds_min_i32"), (6, "ds_max_i32"), (7, "ds_min_u32"), (8, "ds_max_u32"), (9, "ds_and_b32"), (10, "ds_or_b32"),
    (11, "ds_xor_b32"), (12, "ds_mskor_b32"), (13, "ds_write_b32"), (14, "ds_write2_b32"),
    (15, "ds_write2st64_b32"), (16, "ds_cmpst_b32"), (17, "ds_cmpst_f32"), (18, "ds_min_f32"), (19, "ds_max_f32"),
    (20, "ds_nop"), (24, "ds_gws_sema_release_all"), (25, "ds_gws_init"), (26, "ds_gws_sema_v"),
    (27, "ds_gws_sema_br"), (28, "ds_gws_sema_p"), (29, "ds_gws_barrier"), (30, "ds_write_b8"),
    (31, "ds_write_b16"), (32, "ds_add_rtn_u32"), (33, "ds_sub_rtn_u32"), (34, "ds_rsub_rtn_u32"),
    (35, "ds_inc_rtn_u32"), (36, "ds_dec_rtn_u32"), (37, "ds_min_rtn_i32"), (38, "ds_max_rtn_i32"),
    (39, "ds_min_rtn_u32"), (40, "ds_max_rtn_u32"), (41, "ds_and_rtn_b32"), (42, "ds_or_rtn_b32"),
    (43, "ds_xor_rtn_b32"), (44, "ds_mskor_rtn_b32"), (45, "ds_wrxchg_rtn_b32"), (46, "ds_wrxchg2_rtn_b32"),
    (47, "ds_wrxchg2st64_rtn_b32"), (48, "ds_cmpst_rtn_b32"), (49, "ds_cmpst_rtn_f32"), (50, "ds_min_rtn_f32"),
    (51, "ds_max_rtn_f32"), (52, "ds_wrap_rtn_b32"), (53, "ds_swizzle_b32"), (54, "ds_read_b32"),
    (55, "ds_read2_b32"), (56, "ds_read2st64_b32"), (57, "ds_read_i8"), (58, "ds_read_u8"), (59, "ds_read_i16"),
    (60, "ds_read_u16"), (61, "ds_consume"), (62, "ds_append"), (63, "ds_ordered_count"), (64, "ds_add_u64"),
    (65, "ds_sub_u64"), (66, "ds_rsub_u64"), (67, "ds_inc_u64"), (68, "ds_dec_u64"), (69, "ds_min_i64"),
    (70, "ds_max_i64"), (71, "ds_min_u64"), (72, "ds_max_u64"), (73, "ds_and_b64"), (74, "ds_or_b64"),
    (75, "ds_xor_b64"), (76, "ds_mskor_b64"), (77, "ds_write_b64"), (78, "ds_write2_b64"),
    (79, "ds_write2st64_b64"), (80, "ds_cmpst_b64"), (81, "ds_cmpst_f64"), (82, "ds_min_f64"), (83, "ds_max_f64"),
    (96, "ds_add_rtn_u64"), (97, "ds_sub_rtn_u64"), (98, "ds_rsub_rtn_u64"), (99, "ds_inc_rtn_u64"),
    (100, "ds_dec_rtn_u64"), (101, "ds_min_rtn_i64"), (102, "ds_max_rtn_i64"), (103, "ds_min_rtn_u64"),
    (104, "ds_max_rtn_u64"), (105, "ds_and_rtn_b64"), (106, "ds_or_rtn_b64"), (107, "ds_xor_rtn_b64"),
    (108, "ds_mskor_rtn_b64"), (109, "ds_wrxchg_rtn_b64"), (110, "ds_wrxchg2_rtn_b64"),
    (111, "ds_wrxchg2st64_rtn_b64"), (112, "ds_cmpst_rtn_b64"), (113, "ds_cmpst_rtn_f64"), (114, "ds_min_rtn_f64"),
    (115, "ds_max_rtn_f64"), (118, "ds_read_b64"), (119, "ds_read2_b64"), (120, "ds_read2st64_b64"),
    (126, "ds_condxchg32_rtn_b64"), (128, "ds_add_src2_u32"), (222, "ds_write_b96"), (223, "ds_write_b128"),
    (254, "ds_read_b96"), (255, "ds_read_b128"),
];

pub const MUBUF: Table = &[
    (0, "buffer_load_format_x"), (1, "buffer_load_format_xy"), (2, "buffer_load_format_xyz"),
    (3, "buffer_load_format_xyzw"), (4, "buffer_store_format_x"), (5, "buffer_store_format_xy"),
    (6, "buffer_store_format_xyz"), (7, "buffer_store_format_xyzw"), (8, "buffer_load_ubyte"),
    (9, "buffer_load_sbyte"), (10, "buffer_load_ushort"), (11, "buffer_load_sshort"), (12, "buffer_load_dword"),
    (13, "buffer_load_dwordx2"), (14, "buffer_load_dwordx4"), (15, "buffer_load_dwordx3"), (24, "buffer_store_byte"),
    (26, "buffer_store_short"), (28, "buffer_store_dword"), (29, "buffer_store_dwordx2"),
    (30, "buffer_store_dwordx4"), (31, "buffer_store_dwordx3"), (48, "buffer_atomic_swap"),
    (49, "buffer_atomic_cmpswap"), (50, "buffer_atomic_add"), (51, "buffer_atomic_sub"),
    (53, "buffer_atomic_smin"), (54, "buffer_atomic_umin"), (55, "buffer_atomic_smax"),
    (56, "buffer_atomic_umax"), (57, "buffer_atomic_and"), (58, "buffer_atomic_or"), (59, "buffer_atomic_xor"),
    (60, "buffer_atomic_inc"), (61, "buffer_atomic_dec"), (62, "buffer_atomic_fcmpswap"),
    (63, "buffer_atomic_fmin"), (64, "buffer_atomic_fmax"), (80, "buffer_atomic_swap_x2"),
    (81, "buffer_atomic_cmpswap_x2"), (82, "buffer_atomic_add_x2"), (112, "buffer_wbinvl1_sc"),
    (113, "buffer_wbinvl1"),
];

pub const MTBUF: Table = &[
    (0, "tbuffer_load_format_x"), (1, "tbuffer_load_format_xy"), (2, "tbuffer_load_format_xyz"),
    (3, "tbuffer_load_format_xyzw"), (4, "tbuffer_store_format_x"), (5, "tbuffer_store_format_xy"),
    (6, "tbuffer_store_format_xyz"), (7, "tbuffer_store_format_xyzw"),
];

pub const MIMG: Table = &[
    (0, "image_load"), (1, "image_load_mip"), (2, "image_load_pck"), (3, "image_load_pck_sgn"),
    (4, "image_load_mip_pck"), (5, "image_load_mip_pck_sgn"), (8, "image_store"), (9, "image_store_mip"),
    (10, "image_store_pck"), (11, "image_store_mip_pck"), (14, "image_get_resinfo"), (15, "image_atomic_swap"),
    (16, "image_atomic_cmpswap"), (17, "image_atomic_add"), (18, "image_atomic_sub"), (20, "image_atomic_smin"),
    (21, "image_atomic_umin"), (22, "image_atomic_smax"), (23, "image_atomic_umax"), (24, "image_atomic_and"),
    (25, "image_atomic_or"), (26, "image_atomic_xor"), (27, "image_atomic_inc"), (28, "image_atomic_dec"),
    (32, "image_sample"), (33, "image_sample_cl"), (34, "image_sample_d"), (35, "image_sample_d_cl"),
    (36, "image_sample_l"), (37, "image_sample_b"), (38, "image_sample_b_cl"), (39, "image_sample_lz"),
    (40, "image_sample_c"), (41, "image_sample_c_cl"), (42, "image_sample_c_d"), (43, "image_sample_c_d_cl"),
    (44, "image_sample_c_l"), (45, "image_sample_c_b"), (46, "image_sample_c_b_cl"), (47, "image_sample_c_lz"),
    (48, "image_sample_o"), (49, "image_sample_cl_o"), (50, "image_sample_d_o"), (51, "image_sample_d_cl_o"),
    (52, "image_sample_l_o"), (53, "image_sample_b_o"), (54, "image_sample_b_cl_o"), (55, "image_sample_lz_o"),
    (56, "image_sample_c_o"), (57, "image_sample_c_cl_o"), (58, "image_sample_c_d_o"),
    (59, "image_sample_c_d_cl_o"), (60, "image_sample_c_l_o"), (61, "image_sample_c_b_o"),
    (62, "image_sample_c_b_cl_o"), (63, "image_sample_c_lz_o"), (64, "image_gather4"), (65, "image_gather4_cl"),
    (68, "image_gather4_l"), (69, "image_gather4_b"), (70, "image_gather4_b_cl"), (71, "image_gather4_lz"),
    (72, "image_gather4_c"), (73, "image_gather4_c_cl"), (76, "image_gather4_c_l"), (77, "image_gather4_c_b"),
    (78, "image_gather4_c_b_cl"), (79, "image_gather4_c_lz"), (80, "image_gather4_o"), (81, "image_gather4_cl_o"),
    (84, "image_gather4_l_o"), (85, "image_gather4_b_o"), (86, "image_gather4_b_cl_o"), (87, "image_gather4_lz_o"),
    (88, "image_gather4_c_o"), (89, "image_gather4_c_cl_o"), (92, "image_gather4_c_l_o"),
    (93, "image_gather4_c_b_o"), (94, "image_gather4_c_b_cl_o"), (95, "image_gather4_c_lz_o"),
    (96, "image_get_lod"), (104, "image_sample_cd"), (105, "image_sample_cd_cl"), (106, "image_sample_c_cd"),
    (107, "image_sample_c_cd_cl"), (108, "image_sample_cd_o"), (109, "image_sample_cd_cl_o"),
    (110, "image_sample_c_cd_o"), (111, "image_sample_c_cd_cl_o"),
];
