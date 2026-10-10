//! Small host-side compute kernels (hand-built SPIR-V) used by the renderer.

use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{self, BuiltIn, Decoration, ExecutionMode, ExecutionModel, StorageClass};

/// Marks a tiled element with no linear counterpart (surface padding).
pub const NO_ELEMENT: u32 = 0xFFFF_FFFF;

/// Element copy between a tiled surface (in guest memory, accessed in its
/// own memory order so host-memory traffic stays contiguous) and a linear
/// element array (device memory), for (de)tiling on the GPU.
///
/// Bindings (set 0): 0 = tiled surface dwords, 1 = map (linear element index
/// of each tiled element, or NO_ELEMENT), 2 = linear dwords. Push constants
/// (4 x u32): tiled element count, dwords per element (1, 2 or 4), tiled and
/// linear base (dwords). One invocation per tiled element, 64 per workgroup.
/// `tile == false` (detile): `lin[map[t]*k + j] = tiled[t*k + j]`;
/// `tile == true`: `tiled[t*k + j] = lin[map[t]*k + j]`.
pub fn remap_cs(tile: bool) -> Vec<u32> {
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(spirv::Capability::Shader);
    b.extension("SPV_KHR_storage_buffer_storage_class");
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let boolt = b.type_bool();
    let u32t = b.type_int(32, 0);
    let v3u = b.type_vector(u32t, 3);
    let rta = b.type_runtime_array(u32t);
    b.decorate(rta, Decoration::ArrayStride, [Operand::LiteralBit32(4)]);
    let st = b.type_struct([rta]);
    b.decorate(st, Decoration::Block, []);
    b.member_decorate(st, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
    let p_st = b.type_pointer(None, StorageClass::StorageBuffer, st);
    let p_sb_u = b.type_pointer(None, StorageClass::StorageBuffer, u32t);
    let bufs: Vec<u32> = (0..3)
        .map(|i| {
            let v = global(&mut b, p_st, StorageClass::StorageBuffer);
            b.decorate(v, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
            b.decorate(v, Decoration::Binding, [Operand::LiteralBit32(i)]);
            v
        })
        .collect();
    let (tiled, map, lin) = (bufs[0], bufs[1], bufs[2]);
    let pcs = b.type_struct([u32t, u32t, u32t, u32t]);
    b.decorate(pcs, Decoration::Block, []);
    for m in 0..4 {
        b.member_decorate(pcs, m, Decoration::Offset, [Operand::LiteralBit32(m * 4)]);
    }
    let p_pcs = b.type_pointer(None, StorageClass::PushConstant, pcs);
    let p_pc_u = b.type_pointer(None, StorageClass::PushConstant, u32t);
    let pc = global(&mut b, p_pcs, StorageClass::PushConstant);
    let p_in_v3 = b.type_pointer(None, StorageClass::Input, v3u);
    let gid = global(&mut b, p_in_v3, StorageClass::Input);
    b.decorate(gid, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::GlobalInvocationId)]);

    let fn_ty = b.type_function(void, []);
    let func = b.begin_function(void, None, spirv::FunctionControl::NONE, fn_ty).unwrap();
    b.begin_block(None).unwrap();
    let c0 = b.constant_bit32(u32t, 0);
    let c1 = b.constant_bit32(u32t, 1);
    let c2 = b.constant_bit32(u32t, 2);
    let c3 = b.constant_bit32(u32t, 3);
    let none = b.constant_bit32(u32t, NO_ELEMENT);
    let pc_field = |b: &mut Builder, m: u32| {
        let idx = b.constant_bit32(u32t, m);
        let p = b.access_chain(p_pc_u, None, pc, [idx]).unwrap();
        b.load(u32t, None, p, None, []).unwrap()
    };
    let n = pc_field(&mut b, 0);
    let kk = pc_field(&mut b, 1);
    let tiled_base = pc_field(&mut b, 2);
    let lin_base = pc_field(&mut b, 3);
    let g = b.load(v3u, None, gid, None, []).unwrap();
    let t = b.composite_extract(u32t, None, g, [0]).unwrap();
    let in_range = b.u_less_than(boolt, None, t, n).unwrap();
    let body = b.id();
    let merge = b.id();
    b.selection_merge(merge, spirv::SelectionControl::NONE).unwrap();
    b.branch_conditional(in_range, body, merge, []).unwrap();
    b.begin_block(Some(body)).unwrap();
    let pm = b.access_chain(p_sb_u, None, map, [c0, t]).unwrap();
    let e = b.load(u32t, None, pm, None, []).unwrap();
    let real = b.i_not_equal(boolt, None, e, none).unwrap();
    let elem_body = b.id();
    let elem_merge = b.id();
    b.selection_merge(elem_merge, spirv::SelectionControl::NONE).unwrap();
    b.branch_conditional(real, elem_body, elem_merge, []).unwrap();
    b.begin_block(Some(elem_body)).unwrap();
    let tt = b.i_mul(u32t, None, t, kk).unwrap();
    let tt = b.i_add(u32t, None, tt, tiled_base).unwrap();
    let ll = b.i_mul(u32t, None, e, kk).unwrap();
    let ll = b.i_add(u32t, None, ll, lin_base).unwrap();
    let (src, s0, dst, d0) = if tile { (lin, ll, tiled, tt) } else { (tiled, tt, lin, ll) };
    let copy = |b: &mut Builder, j: u32| {
        let s = b.i_add(u32t, None, s0, j).unwrap();
        let d = b.i_add(u32t, None, d0, j).unwrap();
        let ps = b.access_chain(p_sb_u, None, src, [c0, s]).unwrap();
        let v = b.load(u32t, None, ps, None, []).unwrap();
        let pd = b.access_chain(p_sb_u, None, dst, [c0, d]).unwrap();
        b.store(pd, v, None, []).unwrap();
    };
    copy(&mut b, c0);
    // Further dwords of 64/128-bit elements.
    for (j, need) in [(c1, c1), (c2, c2), (c3, c2)] {
        let more = b.u_greater_than(boolt, None, kk, need).unwrap();
        let yes = b.id();
        let done = b.id();
        b.selection_merge(done, spirv::SelectionControl::NONE).unwrap();
        b.branch_conditional(more, yes, done, []).unwrap();
        b.begin_block(Some(yes)).unwrap();
        copy(&mut b, j);
        b.branch(done).unwrap();
        b.begin_block(Some(done)).unwrap();
    }
    b.branch(elem_merge).unwrap();
    b.begin_block(Some(elem_merge)).unwrap();
    b.branch(merge).unwrap();
    b.begin_block(Some(merge)).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    b.entry_point(ExecutionModel::GLCompute, func, "main", [gid]);
    b.execution_mode(func, ExecutionMode::LocalSize, [64, 1, 1]);
    b.module().assemble()
}

/// Copies the pages of a written range whose bit is set in its page bitmap
/// (see `BindingKind::DirtyBits`) from the device to the host ring.
///
/// Bindings (set 0): 0 = source dwords, 1 = destination dwords, 2 = bitmap
/// dwords (whole buffers). Push constants (6 x u32): page count, byte
/// offset of the range within its first page, range bytes, source,
/// destination and bitmap base (dwords). One workgroup of 64 per page; the
/// range's bytes [p*4096 - head, (p+1)*4096 - head) ∩ [0, bytes) of page p.
pub fn dirty_copy_cs() -> Vec<u32> {
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(spirv::Capability::Shader);
    b.extension("SPV_KHR_storage_buffer_storage_class");
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let boolt = b.type_bool();
    let u32t = b.type_int(32, 0);
    let v3u = b.type_vector(u32t, 3);
    let rta = b.type_runtime_array(u32t);
    b.decorate(rta, Decoration::ArrayStride, [Operand::LiteralBit32(4)]);
    let st = b.type_struct([rta]);
    b.decorate(st, Decoration::Block, []);
    b.member_decorate(st, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
    let p_st = b.type_pointer(None, StorageClass::StorageBuffer, st);
    let p_sb_u = b.type_pointer(None, StorageClass::StorageBuffer, u32t);
    let bufs: Vec<u32> = (0..3)
        .map(|i| {
            let v = global(&mut b, p_st, StorageClass::StorageBuffer);
            b.decorate(v, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
            b.decorate(v, Decoration::Binding, [Operand::LiteralBit32(i)]);
            v
        })
        .collect();
    let (src, dst, bits) = (bufs[0], bufs[1], bufs[2]);
    let pcs = b.type_struct([u32t, u32t, u32t, u32t, u32t, u32t]);
    b.decorate(pcs, Decoration::Block, []);
    for m in 0..6 {
        b.member_decorate(pcs, m, Decoration::Offset, [Operand::LiteralBit32(m * 4)]);
    }
    let p_pcs = b.type_pointer(None, StorageClass::PushConstant, pcs);
    let p_pc_u = b.type_pointer(None, StorageClass::PushConstant, u32t);
    let pc = global(&mut b, p_pcs, StorageClass::PushConstant);
    let p_in_v3 = b.type_pointer(None, StorageClass::Input, v3u);
    let wg = global(&mut b, p_in_v3, StorageClass::Input);
    b.decorate(wg, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::WorkgroupId)]);
    let lid = global(&mut b, p_in_v3, StorageClass::Input);
    b.decorate(lid, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::LocalInvocationId)]);

    let fn_ty = b.type_function(void, []);
    let func = b.begin_function(void, None, spirv::FunctionControl::NONE, fn_ty).unwrap();
    b.begin_block(None).unwrap();
    let c0 = b.constant_bit32(u32t, 0);
    let c1 = b.constant_bit32(u32t, 1);
    let c2 = b.constant_bit32(u32t, 2);
    let c5 = b.constant_bit32(u32t, 5);
    let c12 = b.constant_bit32(u32t, 12);
    let c31 = b.constant_bit32(u32t, 31);
    let c4096 = b.constant_bit32(u32t, 4096);
    let pc_field = |b: &mut Builder, m: u32| {
        let idx = b.constant_bit32(u32t, m);
        let p = b.access_chain(p_pc_u, None, pc, [idx]).unwrap();
        b.load(u32t, None, p, None, []).unwrap()
    };
    let (npages, head, bytes) = (pc_field(&mut b, 0), pc_field(&mut b, 1), pc_field(&mut b, 2));
    let (src_dw, dst_dw, bits_dw) = (pc_field(&mut b, 3), pc_field(&mut b, 4), pc_field(&mut b, 5));
    let g = b.load(v3u, None, wg, None, []).unwrap();
    let page = b.composite_extract(u32t, None, g, [0]).unwrap();
    let l = b.load(v3u, None, lid, None, []).unwrap();
    let lane = b.composite_extract(u32t, None, l, [0]).unwrap();
    // Dirty page in range?
    let wi = b.shift_right_logical(u32t, None, page, c5).unwrap();
    let wi = b.i_add(u32t, None, wi, bits_dw).unwrap();
    let pw = b.access_chain(p_sb_u, None, bits, [c0, wi]).unwrap();
    let w = b.load(u32t, None, pw, None, []).unwrap();
    let bi = b.bitwise_and(u32t, None, page, c31).unwrap();
    let bit = b.shift_right_logical(u32t, None, w, bi).unwrap();
    let bit = b.bitwise_and(u32t, None, bit, c1).unwrap();
    let set = b.i_not_equal(boolt, None, bit, c0).unwrap();
    let in_range = b.u_less_than(boolt, None, page, npages).unwrap();
    let go = b.logical_and(boolt, None, set, in_range).unwrap();
    let body = b.id();
    let merge = b.id();
    b.selection_merge(merge, spirv::SelectionControl::NONE).unwrap();
    b.branch_conditional(go, body, merge, []).unwrap();
    b.begin_block(Some(body)).unwrap();
    // [start, end) in range bytes.
    let pstart = b.shift_left_logical(u32t, None, page, c12).unwrap();
    let past = b.u_greater_than(boolt, None, pstart, head).unwrap();
    let s0 = b.i_sub(u32t, None, pstart, head).unwrap();
    let start = b.select(u32t, None, past, s0, c0).unwrap();
    let e0 = b.i_add(u32t, None, pstart, c4096).unwrap();
    let e0 = b.i_sub(u32t, None, e0, head).unwrap();
    let short = b.u_less_than(boolt, None, e0, bytes).unwrap();
    let end = b.select(u32t, None, short, e0, bytes).unwrap();
    let d0 = b.shift_right_logical(u32t, None, start, c2).unwrap();
    let dend = b.shift_right_logical(u32t, None, end, c2).unwrap();
    let d0 = b.i_add(u32t, None, d0, lane).unwrap();
    for j in 0..16u32 {
        let off = b.constant_bit32(u32t, j * 64);
        let d = b.i_add(u32t, None, d0, off).unwrap();
        let ok = b.u_less_than(boolt, None, d, dend).unwrap();
        let yes = b.id();
        let done = b.id();
        b.selection_merge(done, spirv::SelectionControl::NONE).unwrap();
        b.branch_conditional(ok, yes, done, []).unwrap();
        b.begin_block(Some(yes)).unwrap();
        let s = b.i_add(u32t, None, d, src_dw).unwrap();
        let ps = b.access_chain(p_sb_u, None, src, [c0, s]).unwrap();
        let v = b.load(u32t, None, ps, None, []).unwrap();
        let t = b.i_add(u32t, None, d, dst_dw).unwrap();
        let pd = b.access_chain(p_sb_u, None, dst, [c0, t]).unwrap();
        b.store(pd, v, None, []).unwrap();
        b.branch(done).unwrap();
        b.begin_block(Some(done)).unwrap();
    }
    b.branch(merge).unwrap();
    b.begin_block(Some(merge)).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    b.entry_point(ExecutionModel::GLCompute, func, "main", [wg, lid]);
    b.execution_mode(func, ExecutionMode::LocalSize, [64, 1, 1]);
    b.module().assemble()
}

/// Validates the arguments of an indirect dispatch (three dwords in guest
/// memory, written by earlier GPU work) into a device buffer the dispatch
/// reads (vkCmdDispatchIndirect). Counts over the device's per-dimension
/// limits or more than `max_groups` workgroups in total become 0 x 0 x 0 (an
/// implausible count could keep the GPU busy for seconds; the CP rejected
/// such arguments the same way when it read them itself), and each rejection
/// is counted with its values for the log.
///
/// Bindings (set 0): 0 = source dwords, 1 = destination (x, y, z, 0),
/// 2 = diagnostics (rejections, last rejected x, y, z). Push constants
/// (5 x u32): source dword index, maximum x, y, z, maximum total groups.
/// One invocation.
pub fn indirect_args_cs() -> Vec<u32> {
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(spirv::Capability::Shader);
    b.extension("SPV_KHR_storage_buffer_storage_class");
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let boolt = b.type_bool();
    let u32t = b.type_int(32, 0);
    let rta = b.type_runtime_array(u32t);
    b.decorate(rta, Decoration::ArrayStride, [Operand::LiteralBit32(4)]);
    let st = b.type_struct([rta]);
    b.decorate(st, Decoration::Block, []);
    b.member_decorate(st, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
    let p_st = b.type_pointer(None, StorageClass::StorageBuffer, st);
    let p_sb_u = b.type_pointer(None, StorageClass::StorageBuffer, u32t);
    let bufs: Vec<u32> = (0..3)
        .map(|i| {
            let v = global(&mut b, p_st, StorageClass::StorageBuffer);
            b.decorate(v, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
            b.decorate(v, Decoration::Binding, [Operand::LiteralBit32(i)]);
            v
        })
        .collect();
    let (src, dst, diag) = (bufs[0], bufs[1], bufs[2]);
    let pcs = b.type_struct([u32t, u32t, u32t, u32t, u32t]);
    b.decorate(pcs, Decoration::Block, []);
    for m in 0..5 {
        b.member_decorate(pcs, m, Decoration::Offset, [Operand::LiteralBit32(m * 4)]);
    }
    let p_pcs = b.type_pointer(None, StorageClass::PushConstant, pcs);
    let p_pc_u = b.type_pointer(None, StorageClass::PushConstant, u32t);
    let pc = global(&mut b, p_pcs, StorageClass::PushConstant);

    let fn_ty = b.type_function(void, []);
    let func = b.begin_function(void, None, spirv::FunctionControl::NONE, fn_ty).unwrap();
    b.begin_block(None).unwrap();
    let c0 = b.constant_bit32(u32t, 0);
    let c1 = b.constant_bit32(u32t, 1);
    let c2 = b.constant_bit32(u32t, 2);
    let c3 = b.constant_bit32(u32t, 3);
    // (Scope Device, relaxed semantics.)
    let scope = b.constant_bit32(u32t, 1);
    let relaxed = b.constant_bit32(u32t, 0);
    let pc_field = |b: &mut Builder, m: u32| {
        let idx = b.constant_bit32(u32t, m);
        let p = b.access_chain(p_pc_u, None, pc, [idx]).unwrap();
        b.load(u32t, None, p, None, []).unwrap()
    };
    let at = pc_field(&mut b, 0);
    let (mx, my, mz, mg) = (pc_field(&mut b, 1), pc_field(&mut b, 2), pc_field(&mut b, 3), pc_field(&mut b, 4));
    let load = |b: &mut Builder, k: u32| {
        let i = b.i_add(u32t, None, at, k).unwrap();
        let p = b.access_chain(p_sb_u, None, src, [c0, i]).unwrap();
        b.load(u32t, None, p, None, []).unwrap()
    };
    let (x, y, z) = (load(&mut b, c0), load(&mut b, c1), load(&mut b, c2));
    let okx = b.u_less_than_equal(boolt, None, x, mx).unwrap();
    let oky = b.u_less_than_equal(boolt, None, y, my).unwrap();
    let okz = b.u_less_than_equal(boolt, None, z, mz).unwrap();
    // x * y * z <= max_groups without overflow: y <= max / x, z <= max / xy.
    let total_ok = |b: &mut Builder, f: u32, g: u32| {
        let f0 = b.i_equal(boolt, None, f, c0).unwrap();
        let d = b.select(u32t, None, f0, c1, f).unwrap();
        let q = b.u_div(u32t, None, mg, d).unwrap();
        let le = b.u_less_than_equal(boolt, None, g, q).unwrap();
        b.logical_or(boolt, None, f0, le).unwrap()
    };
    let ok_xy = total_ok(&mut b, x, y);
    let xy = b.i_mul(u32t, None, x, y).unwrap();
    let ok_z = total_ok(&mut b, xy, z);
    let mut ok = b.logical_and(boolt, None, okx, oky).unwrap();
    for t in [okz, ok_xy, ok_z] {
        ok = b.logical_and(boolt, None, ok, t).unwrap();
    }
    for (k, v) in [(c0, x), (c1, y), (c2, z)] {
        let s = b.select(u32t, None, ok, v, c0).unwrap();
        let p = b.access_chain(p_sb_u, None, dst, [c0, k]).unwrap();
        b.store(p, s, None, []).unwrap();
    }
    let p = b.access_chain(p_sb_u, None, dst, [c0, c3]).unwrap();
    b.store(p, c0, None, []).unwrap();
    let rejected = b.id();
    let merge = b.id();
    b.selection_merge(merge, spirv::SelectionControl::NONE).unwrap();
    b.branch_conditional(ok, merge, rejected, []).unwrap();
    b.begin_block(Some(rejected)).unwrap();
    let p = b.access_chain(p_sb_u, None, diag, [c0, c0]).unwrap();
    b.atomic_i_add(u32t, None, p, scope, relaxed, c1).unwrap();
    for (k, v) in [(c1, x), (c2, y), (c3, z)] {
        let p = b.access_chain(p_sb_u, None, diag, [c0, k]).unwrap();
        b.store(p, v, None, []).unwrap();
    }
    b.branch(merge).unwrap();
    b.begin_block(Some(merge)).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    b.entry_point(ExecutionModel::GLCompute, func, "main", []);
    b.execution_mode(func, ExecutionMode::LocalSize, [1, 1, 1]);
    b.module().assemble()
}

/// Common setup of the metadata kernels: two storage buffers (0 = metadata
/// dwords, 1 = predicates) and four u32 push constants, loaded at the start
/// of `main` (the builder is left in its first block).
struct MetaKernel {
    b: Builder,
    u32t: u32,
    boolt: u32,
    p_sb_u: u32,
    meta: u32,
    pred: u32,
    pc: [u32; 4],
    func: u32,
}

fn meta_kernel() -> MetaKernel {
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(spirv::Capability::Shader);
    b.extension("SPV_KHR_storage_buffer_storage_class");
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let boolt = b.type_bool();
    let u32t = b.type_int(32, 0);
    let rta = b.type_runtime_array(u32t);
    b.decorate(rta, Decoration::ArrayStride, [Operand::LiteralBit32(4)]);
    let st = b.type_struct([rta]);
    b.decorate(st, Decoration::Block, []);
    b.member_decorate(st, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
    let p_st = b.type_pointer(None, StorageClass::StorageBuffer, st);
    let p_sb_u = b.type_pointer(None, StorageClass::StorageBuffer, u32t);
    let bufs: Vec<u32> = (0..2)
        .map(|i| {
            let v = global(&mut b, p_st, StorageClass::StorageBuffer);
            b.decorate(v, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
            b.decorate(v, Decoration::Binding, [Operand::LiteralBit32(i)]);
            v
        })
        .collect();
    let pcs = b.type_struct([u32t, u32t, u32t, u32t]);
    b.decorate(pcs, Decoration::Block, []);
    for m in 0..4 {
        b.member_decorate(pcs, m, Decoration::Offset, [Operand::LiteralBit32(m * 4)]);
    }
    let p_pcs = b.type_pointer(None, StorageClass::PushConstant, pcs);
    let p_pc_u = b.type_pointer(None, StorageClass::PushConstant, u32t);
    let pcv = global(&mut b, p_pcs, StorageClass::PushConstant);
    let fn_ty = b.type_function(void, []);
    let func = b.begin_function(void, None, spirv::FunctionControl::NONE, fn_ty).unwrap();
    b.begin_block(None).unwrap();
    let mut pc = [0u32; 4];
    for (m, slot) in pc.iter_mut().enumerate() {
        let idx = b.constant_bit32(u32t, m as u32);
        let p = b.access_chain(p_pc_u, None, pcv, [idx]).unwrap();
        *slot = b.load(u32t, None, p, None, []).unwrap();
    }
    MetaKernel { b, u32t, boolt, p_sb_u, meta: bufs[0], pred: bufs[1], pc, func }
}

/// Samples a render target's compression metadata on the GPU -- the 48
/// dwords the renderer's CPU probe reads: the first, middle and last 16 --
/// into clear predicates for VK_EXT_conditional_rendering. HTILE (mode 0):
/// depth cleared when every sample's ZMask [3:0] is 0, stencil (with stencil
/// tiles) when every SMem [9:8] is 0; CMASK (mode 1): cleared when every
/// sample is 0. Bindings (set 0): 0 = metadata dwords, 1 = predicates
/// (depth, stencil, color, any; 1 or 0). Push constants (4 x u32): first
/// dword index, dword count (at least 32), mode, stencil tiles. One
/// invocation.
pub fn meta_probe_cs() -> Vec<u32> {
    let MetaKernel { mut b, u32t, boolt, p_sb_u, meta, pred, pc, func } = meta_kernel();
    let [at, n, mode, st] = pc;
    let c0 = b.constant_bit32(u32t, 0);
    let c1 = b.constant_bit32(u32t, 1);
    let c3 = b.constant_bit32(u32t, 3);
    let c8 = b.constant_bit32(u32t, 8);
    let c15 = b.constant_bit32(u32t, 15);
    let c16 = b.constant_bit32(u32t, 16);
    let half = b.shift_right_logical(u32t, None, n, c1).unwrap();
    let tail = b.i_sub(u32t, None, n, c16).unwrap();
    let (mut z, mut s, mut col) = (c0, c0, c0);
    for k in 0..48u32 {
        let (start, j) = match k / 16 {
            0 => (c0, k),
            1 => (half, k - 16),
            _ => (tail, k - 32),
        };
        let jc = b.constant_bit32(u32t, j);
        let idx = b.i_add(u32t, None, start, jc).unwrap();
        let i = b.i_add(u32t, None, at, idx).unwrap();
        let p = b.access_chain(p_sb_u, None, meta, [c0, i]).unwrap();
        let w = b.load(u32t, None, p, None, []).unwrap();
        let zm = b.bitwise_and(u32t, None, w, c15).unwrap();
        z = b.bitwise_or(u32t, None, z, zm).unwrap();
        let sh = b.shift_right_logical(u32t, None, w, c8).unwrap();
        let sm = b.bitwise_and(u32t, None, sh, c3).unwrap();
        s = b.bitwise_or(u32t, None, s, sm).unwrap();
        col = b.bitwise_or(u32t, None, col, w).unwrap();
    }
    let is_h = b.i_equal(boolt, None, mode, c0).unwrap();
    let is_c = b.i_equal(boolt, None, mode, c1).unwrap();
    let z0 = b.i_equal(boolt, None, z, c0).unwrap();
    let s0 = b.i_equal(boolt, None, s, c0).unwrap();
    let col0 = b.i_equal(boolt, None, col, c0).unwrap();
    let st_on = b.i_not_equal(boolt, None, st, c0).unwrap();
    let depth = b.logical_and(boolt, None, is_h, z0).unwrap();
    let stencil = b.logical_and(boolt, None, is_h, s0).unwrap();
    let stencil = b.logical_and(boolt, None, stencil, st_on).unwrap();
    let color = b.logical_and(boolt, None, is_c, col0).unwrap();
    let any = b.logical_or(boolt, None, depth, stencil).unwrap();
    let any = b.logical_or(boolt, None, any, color).unwrap();
    for (k, v) in [depth, stencil, color, any].into_iter().enumerate() {
        let kc = b.constant_bit32(u32t, k as u32);
        let x = b.select(u32t, None, v, c1, c0).unwrap();
        let p = b.access_chain(p_sb_u, None, pred, [c0, kc]).unwrap();
        b.store(p, x, None, []).unwrap();
    }
    b.ret().unwrap();
    b.end_function().unwrap();
    b.entry_point(ExecutionModel::GLCompute, func, "main", []);
    b.execution_mode(func, ExecutionMode::LocalSize, [1, 1, 1]);
    b.module().assemble()
}

/// After `meta_probe_cs`: when it found a clear (predicate `any`), writes the
/// expanded state (all bits set) over the whole metadata surface, as the
/// hardware leaves tiles once rendered (a later write near the metadata must
/// not make the old cleared state apply again). Bindings and push constants
/// as `meta_probe_cs` (first dword index, dword count). 64 invocations per
/// workgroup, one dword each.
pub fn meta_expand_cs() -> Vec<u32> {
    let MetaKernel { mut b, u32t, boolt, p_sb_u, meta, pred, pc, func } = meta_kernel();
    let [at, n, _, _] = pc;
    let v3u = b.type_vector(u32t, 3);
    let p_in_v3 = b.type_pointer(None, StorageClass::Input, v3u);
    let gid = global(&mut b, p_in_v3, StorageClass::Input);
    b.decorate(gid, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::GlobalInvocationId)]);
    let c0 = b.constant_bit32(u32t, 0);
    let c3 = b.constant_bit32(u32t, 3);
    let all = b.constant_bit32(u32t, 0xFFFF_FFFF);
    let g = b.load(v3u, None, gid, None, []).unwrap();
    let t = b.composite_extract(u32t, None, g, [0]).unwrap();
    let pa = b.access_chain(p_sb_u, None, pred, [c0, c3]).unwrap();
    let any = b.load(u32t, None, pa, None, []).unwrap();
    let in_range = b.u_less_than(boolt, None, t, n).unwrap();
    let applied = b.i_not_equal(boolt, None, any, c0).unwrap();
    let go = b.logical_and(boolt, None, in_range, applied).unwrap();
    let body = b.id();
    let merge = b.id();
    b.selection_merge(merge, spirv::SelectionControl::NONE).unwrap();
    b.branch_conditional(go, body, merge, []).unwrap();
    b.begin_block(Some(body)).unwrap();
    let i = b.i_add(u32t, None, at, t).unwrap();
    let p = b.access_chain(p_sb_u, None, meta, [c0, i]).unwrap();
    b.store(p, all, None, []).unwrap();
    b.branch(merge).unwrap();
    b.begin_block(Some(merge)).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    b.entry_point(ExecutionModel::GLCompute, func, "main", [gid]);
    b.execution_mode(func, ExecutionMode::LocalSize, [64, 1, 1]);
    b.module().assemble()
}

fn global(b: &mut Builder, ptr: u32, sc: StorageClass) -> u32 {
    let id = b.id();
    let inst = rspirv::dr::Instruction::new(spirv::Op::Variable, Some(ptr), Some(id), vec![Operand::StorageClass(sc)]);
    b.module_mut().types_global_values.push(inst);
    id
}

#[cfg(test)]
mod tests {
    #[test]
    fn remap_kernels_assemble() {
        for tile in [false, true] {
            let w = super::remap_cs(tile);
            assert_eq!(w[0], 0x0723_0203); // SPIR-V magic
            assert!(w.len() > 100);
            // TLG_DUMP_KERNELS=<dir>: write the binaries for spirv-val.
            if let Ok(dir) = std::env::var("TLG_DUMP_KERNELS") {
                let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(format!("{dir}/remap_{}.spv", if tile { "tile" } else { "detile" }), bytes).unwrap();
            }
        }
    }

    /// The binary parses back (well-formed instructions and operands) and has
    /// one GLCompute entry point whose blocks all end in a terminator.
    fn check_module(w: &[u32], name: &str) {
        let m = rspirv::dr::load_words(w).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        assert_eq!(m.entry_points.len(), 1, "{name}");
        for f in &m.functions {
            for b in &f.blocks {
                let last = b.instructions.last().map(|i| i.class.opcode);
                assert!(matches!(last, Some(rspirv::spirv::Op::Return | rspirv::spirv::Op::Branch | rspirv::spirv::Op::BranchConditional)), "{name}: block without terminator");
            }
        }
    }

    #[test]
    fn meta_kernels_assemble() {
        for (name, w) in [("meta_probe", super::meta_probe_cs()), ("meta_expand", super::meta_expand_cs())] {
            assert_eq!(w[0], 0x0723_0203);
            assert!(w.len() > 100, "{name}");
            check_module(&w, name);
            if let Ok(dir) = std::env::var("TLG_DUMP_KERNELS") {
                let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(format!("{dir}/{name}.spv"), bytes).unwrap();
            }
        }
    }

    #[test]
    fn indirect_args_kernel_assembles() {
        let w = super::indirect_args_cs();
        assert_eq!(w[0], 0x0723_0203);
        assert!(w.len() > 100);
        check_module(&w, "indirect_args");
        check_module(&super::remap_cs(false), "remap");
        if let Ok(dir) = std::env::var("TLG_DUMP_KERNELS") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(format!("{dir}/indirect_args.spv"), bytes).unwrap();
        }
    }
}
