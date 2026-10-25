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
/// (see `BindingKind::DirtyBits`) from the device to the host ring. With a
/// word map (`Spec::page_marks` 3: a word per page, nonzero = written): the
/// pages whose word is set, and their bits are set in the bitmap (zeroed
/// before) for the host's write-back.
///
/// Bindings (set 0): 0 = source dwords, 1 = destination dwords, 2 = bitmap
/// and word-map dwords (whole buffers). Push constants (7 x u32): page
/// count, byte offset of the range within its first page, range bytes,
/// source, destination, bitmap and word-map base (dwords; u32::MAX: no word
/// map). One workgroup of 64 per page; the range's bytes [p*4096 - head,
/// (p+1)*4096 - head) ∩ [0, bytes) of page p.
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
    let pcs = b.type_struct([u32t, u32t, u32t, u32t, u32t, u32t, u32t]);
    b.decorate(pcs, Decoration::Block, []);
    for m in 0..7 {
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
    let map_dw = pc_field(&mut b, 6);
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
    let bit_set = b.i_not_equal(boolt, None, bit, c0).unwrap();
    // (With a word map: the page's word; without, the bitmap's first word
    // is read in its place and not used.)
    let cmax = b.constant_bit32(u32t, u32::MAX);
    let has_map = b.i_not_equal(boolt, None, map_dw, cmax).unwrap();
    let mi = b.i_add(u32t, None, page, map_dw).unwrap();
    let mi = b.select(u32t, None, has_map, mi, bits_dw).unwrap();
    let pm = b.access_chain(p_sb_u, None, bits, [c0, mi]).unwrap();
    let mw = b.load(u32t, None, pm, None, []).unwrap();
    let map_set = b.i_not_equal(boolt, None, mw, c0).unwrap();
    let set = b.select(boolt, None, has_map, map_set, bit_set).unwrap();
    let in_range = b.u_less_than(boolt, None, page, npages).unwrap();
    let go = b.logical_and(boolt, None, set, in_range).unwrap();
    let body = b.id();
    let merge = b.id();
    b.selection_merge(merge, spirv::SelectionControl::NONE).unwrap();
    b.branch_conditional(go, body, merge, []).unwrap();
    b.begin_block(Some(body)).unwrap();
    // A word-mapped page's bit for the host (one lane).
    let lane0 = b.i_equal(boolt, None, lane, c0).unwrap();
    let set_bit = b.logical_and(boolt, None, has_map, lane0).unwrap();
    let sb = b.id();
    let sm = b.id();
    b.selection_merge(sm, spirv::SelectionControl::NONE).unwrap();
    b.branch_conditional(set_bit, sb, sm, []).unwrap();
    b.begin_block(Some(sb)).unwrap();
    let one_bit = b.shift_left_logical(u32t, None, c1, bi).unwrap();
    let dev = b.constant_bit32(u32t, spirv::Scope::Device as u32);
    b.atomic_or(u32t, None, pw, dev, c0, one_bit).unwrap();
    b.branch(sm).unwrap();
    b.begin_block(Some(sm)).unwrap();
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

/// Prepares an indirect draw whose arguments earlier GPU work wrote (their
/// device copy) for vkCmdDraw(Indexed)Indirect without the CP reading them:
/// the title's {count, instances, start, [base vertex,] start instance}
/// become the Vulkan command {count, instances, first index (indexed; first
/// vertex 0 otherwise), vertex offset (VGT_INDX_OFFSET, indexed), first
/// instance 0}, and the base vertex / start instance go to the vertex stage's user-data dwords the CP
/// passes them in (what the CP does when it reads the arguments itself).
/// Implausible counts (more than the maximum instances, or count x instances
/// over the maximum work) and indexed draws past the index buffer draw
/// nothing (instance count 0) and are counted for the log.
///
/// Bindings (set 0): 0 = source dwords, 1 = destination command (5 dwords),
/// 2 = the vertex stage's SGPR data, 3 = diagnostics (rejections, last
/// rejected count, instances, start). Push constants (8 x u32): source dword
/// index, indexed (0/1), SGPR data dword of the base vertex and of the start
/// instance (u32::MAX: none), index buffer size in indices, maximum count x
/// instances, maximum instances, vertex offset of indexed draws. One
/// invocation.
pub fn indirect_draw_cs() -> Vec<u32> {
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
    let bufs: Vec<u32> = (0..4)
        .map(|i| {
            let v = global(&mut b, p_st, StorageClass::StorageBuffer);
            b.decorate(v, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
            b.decorate(v, Decoration::Binding, [Operand::LiteralBit32(i)]);
            v
        })
        .collect();
    let (src, dst, sgpr, diag) = (bufs[0], bufs[1], bufs[2], bufs[3]);
    let pcs = b.type_struct([u32t; 8]);
    b.decorate(pcs, Decoration::Block, []);
    for m in 0..8 {
        b.member_decorate(pcs, m, Decoration::Offset, [Operand::LiteralBit32(m * 4)]);
    }
    let p_pcs = b.type_pointer(None, StorageClass::PushConstant, pcs);
    let p_pc_u = b.type_pointer(None, StorageClass::PushConstant, u32t);
    let pc = global(&mut b, p_pcs, StorageClass::PushConstant);

    let fn_ty = b.type_function(void, []);
    let func = b.begin_function(void, None, spirv::FunctionControl::NONE, fn_ty).unwrap();
    b.begin_block(None).unwrap();
    let c: Vec<u32> = (0..5).map(|k| b.constant_bit32(u32t, k)).collect();
    let none = b.constant_bit32(u32t, u32::MAX);
    // (Scope Device, relaxed semantics.)
    let scope = b.constant_bit32(u32t, 1);
    let relaxed = b.constant_bit32(u32t, 0);
    let pc_field = |b: &mut Builder, m: u32| {
        let idx = b.constant_bit32(u32t, m);
        let p = b.access_chain(p_pc_u, None, pc, [idx]).unwrap();
        b.load(u32t, None, p, None, []).unwrap()
    };
    let at = pc_field(&mut b, 0);
    let indexed_u = pc_field(&mut b, 1);
    let (vk, ik, nidx, maxw, maxi) = (pc_field(&mut b, 2), pc_field(&mut b, 3), pc_field(&mut b, 4), pc_field(&mut b, 5), pc_field(&mut b, 6));
    let vofs = pc_field(&mut b, 7);
    let indexed = b.i_not_equal(boolt, None, indexed_u, c[0]).unwrap();
    let load_at = |b: &mut Builder, k: u32| {
        let i = b.i_add(u32t, None, at, k).unwrap();
        let p = b.access_chain(p_sb_u, None, src, [c[0], i]).unwrap();
        b.load(u32t, None, p, None, []).unwrap()
    };
    let (count, inst, w2, w3) = (load_at(&mut b, c[0]), load_at(&mut b, c[1]), load_at(&mut b, c[2]), load_at(&mut b, c[3]));
    // (Non-indexed arguments are 4 dwords: their fifth is not read.)
    let k4 = b.select(u32t, None, indexed, c[4], c[3]).unwrap();
    let w4 = load_at(&mut b, k4);
    let base_vertex = b.select(u32t, None, indexed, w3, w2).unwrap();
    let start_inst = b.select(u32t, None, indexed, w4, w3).unwrap();
    // instances <= max, count x instances <= max work (without overflow).
    let ok_i = b.u_less_than_equal(boolt, None, inst, maxi).unwrap();
    let count0 = b.i_equal(boolt, None, count, c[0]).unwrap();
    let d = b.select(u32t, None, count0, c[1], count).unwrap();
    let q = b.u_div(u32t, None, maxw, d).unwrap();
    let le = b.u_less_than_equal(boolt, None, inst, q).unwrap();
    let ok_w = b.logical_or(boolt, None, count0, le).unwrap();
    // Indexed: start + count within the index buffer.
    let in1 = b.u_less_than_equal(boolt, None, w2, nidx).unwrap();
    let rem = b.i_sub(u32t, None, nidx, w2).unwrap();
    let in2 = b.u_less_than_equal(boolt, None, count, rem).unwrap();
    let in_ib = b.logical_and(boolt, None, in1, in2).unwrap();
    let not_indexed = b.logical_not(boolt, None, indexed).unwrap();
    let ok_ib = b.logical_or(boolt, None, not_indexed, in_ib).unwrap();
    let ok_iw = b.logical_and(boolt, None, ok_i, ok_w).unwrap();
    let ok = b.logical_and(boolt, None, ok_iw, ok_ib).unwrap();
    let out_inst = b.select(u32t, None, ok, inst, c[0]).unwrap();
    let first = b.select(u32t, None, indexed, w2, c[0]).unwrap();
    // (Non-indexed: dword 3 is the first instance.)
    let d3 = b.select(u32t, None, indexed, vofs, c[0]).unwrap();
    for (k, v) in [(c[0], count), (c[1], out_inst), (c[2], first), (c[3], d3), (c[4], c[0])] {
        let p = b.access_chain(p_sb_u, None, dst, [c[0], k]).unwrap();
        b.store(p, v, None, []).unwrap();
    }
    // The user-data dwords (when the draw passes them).
    for (k, v) in [(vk, base_vertex), (ik, start_inst)] {
        let has = b.i_not_equal(boolt, None, k, none).unwrap();
        let then = b.id();
        let merge = b.id();
        b.selection_merge(merge, spirv::SelectionControl::NONE).unwrap();
        b.branch_conditional(has, then, merge, []).unwrap();
        b.begin_block(Some(then)).unwrap();
        let p = b.access_chain(p_sb_u, None, sgpr, [c[0], k]).unwrap();
        b.store(p, v, None, []).unwrap();
        b.branch(merge).unwrap();
        b.begin_block(Some(merge)).unwrap();
    }
    let rejected = b.id();
    let merge = b.id();
    b.selection_merge(merge, spirv::SelectionControl::NONE).unwrap();
    b.branch_conditional(ok, merge, rejected, []).unwrap();
    b.begin_block(Some(rejected)).unwrap();
    let p = b.access_chain(p_sb_u, None, diag, [c[0], c[0]]).unwrap();
    b.atomic_i_add(u32t, None, p, scope, relaxed, c[1]).unwrap();
    for (k, v) in [(c[1], count), (c[2], inst), (c[3], w2)] {
        let p = b.access_chain(p_sb_u, None, diag, [c[0], k]).unwrap();
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

/// Copies the texels of a depth image (a depth view, fetched by integer
/// coordinates) into a color storage image (R32/R16, written without a
/// format) in one pass, for sampling through a smaller T# than the depth
/// target (instead of depth -> buffer -> color transfers). Bindings (set 0):
/// 0 = the depth view (sampled image), 1 = the color image (storage image).
/// Push constants (2 x u32): width, height. 8x8 invocations per workgroup.
pub fn depth_copy_cs() -> Vec<u32> {
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(spirv::Capability::Shader);
    b.capability(spirv::Capability::StorageImageWriteWithoutFormat);
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let boolt = b.type_bool();
    let u32t = b.type_int(32, 0);
    let i32t = b.type_int(32, 1);
    let f32t = b.type_float(32, None);
    let v2i = b.type_vector(i32t, 2);
    let v3u = b.type_vector(u32t, 3);
    let v4f = b.type_vector(f32t, 4);
    let img_t = b.type_image(f32t, spirv::Dim::Dim2D, 0, 0, 0, 1, spirv::ImageFormat::Unknown, None);
    let st_t = b.type_image(f32t, spirv::Dim::Dim2D, 0, 0, 0, 2, spirv::ImageFormat::Unknown, None);
    let p_img = b.type_pointer(None, StorageClass::UniformConstant, img_t);
    let p_st = b.type_pointer(None, StorageClass::UniformConstant, st_t);
    let src = global(&mut b, p_img, StorageClass::UniformConstant);
    let dst = global(&mut b, p_st, StorageClass::UniformConstant);
    for (v, binding) in [(src, 0), (dst, 1)] {
        b.decorate(v, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
        b.decorate(v, Decoration::Binding, [Operand::LiteralBit32(binding)]);
    }
    b.decorate(dst, Decoration::NonReadable, []);
    let pcs = b.type_struct([u32t, u32t]);
    b.decorate(pcs, Decoration::Block, []);
    for m in 0..2 {
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
    let lod0 = b.constant_bit32(i32t, 0);
    let pw = b.access_chain(p_pc_u, None, pc, [c0]).unwrap();
    let w = b.load(u32t, None, pw, None, []).unwrap();
    let ph = b.access_chain(p_pc_u, None, pc, [c1]).unwrap();
    let h = b.load(u32t, None, ph, None, []).unwrap();
    let g = b.load(v3u, None, gid, None, []).unwrap();
    let x = b.composite_extract(u32t, None, g, [0]).unwrap();
    let y = b.composite_extract(u32t, None, g, [1]).unwrap();
    let in_x = b.u_less_than(boolt, None, x, w).unwrap();
    let in_y = b.u_less_than(boolt, None, y, h).unwrap();
    let inside = b.logical_and(boolt, None, in_x, in_y).unwrap();
    let body = b.id();
    let merge = b.id();
    b.selection_merge(merge, spirv::SelectionControl::NONE).unwrap();
    b.branch_conditional(inside, body, merge, []).unwrap();
    b.begin_block(Some(body)).unwrap();
    let xi = b.bitcast(i32t, None, x).unwrap();
    let yi = b.bitcast(i32t, None, y).unwrap();
    let at = b.composite_construct(v2i, None, [xi, yi]).unwrap();
    let si = b.load(img_t, None, src, None, []).unwrap();
    let v = b.image_fetch(v4f, None, si, at, Some(spirv::ImageOperands::LOD), [Operand::IdRef(lod0)]).unwrap();
    let di = b.load(st_t, None, dst, None, []).unwrap();
    b.image_write(di, at, v, None, []).unwrap();
    b.branch(merge).unwrap();
    b.begin_block(Some(merge)).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    b.entry_point(ExecutionModel::GLCompute, func, "main", [gid]);
    b.execution_mode(func, ExecutionMode::LocalSize, [8, 8, 1]);
    b.module().assemble()
}

/// The frame for an HDR10 (ST 2084) swapchain: samples the game's frame
/// (sRGB-encoded values) at each output pixel inside the frame's rectangle
/// (black outside), decodes it (gamma 2.2, or the piecewise sRGB curve),
/// scales it to the paper-white level, converts BT.709 primaries to BT.2020
/// and PQ-encodes it, packed as A2B10G10R10 (or A2R10G10B10) into a 32-bit
/// unsigned storage image that is then copied into the swapchain image.
/// Bindings (set 0): 0 = the frame (combined image sampler), 1 = the output
/// (R32 uint storage image). Push constants (8 x 32 bits): the frame's
/// rectangle x0, y0, x1, y1 (f32, in output pixels), output width and height
/// (u32), the scale (f32: for SDR frames the paper white in nits / 10000, for
/// PQ frames a gain), flags (u32: bit 0 = the piecewise sRGB curve, bit 1 =
/// red in the high bits, bit 2 = the frame is already PQ-encoded BT.2020 -
/// the game's own HDR output - decoded, scaled by the gain and encoded
/// again, bit 3 = 8-bit sRGB output instead, for an SDR swapchain: BT.709,
/// the scale taking white to 1). 8x8 per workgroup.
pub fn hdr10_cs() -> Vec<u32> {
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(spirv::Capability::Shader);
    let glsl = b.ext_inst_import("GLSL.std.450");
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let boolt = b.type_bool();
    let u32t = b.type_int(32, 0);
    let i32t = b.type_int(32, 1);
    let f32t = b.type_float(32, None);
    let v2f = b.type_vector(f32t, 2);
    let v2i = b.type_vector(i32t, 2);
    let v3u = b.type_vector(u32t, 3);
    let v3f = b.type_vector(f32t, 3);
    let v3b = b.type_vector(boolt, 3);
    let v4f = b.type_vector(f32t, 4);
    let v4u = b.type_vector(u32t, 4);
    let img_t = b.type_image(f32t, spirv::Dim::Dim2D, 0, 0, 0, 1, spirv::ImageFormat::Unknown, None);
    let simg_t = b.type_sampled_image(img_t);
    let st_t = b.type_image(u32t, spirv::Dim::Dim2D, 0, 0, 0, 2, spirv::ImageFormat::R32ui, None);
    let p_simg = b.type_pointer(None, StorageClass::UniformConstant, simg_t);
    let p_st = b.type_pointer(None, StorageClass::UniformConstant, st_t);
    let src = global(&mut b, p_simg, StorageClass::UniformConstant);
    let dst = global(&mut b, p_st, StorageClass::UniformConstant);
    for (v, binding) in [(src, 0), (dst, 1)] {
        b.decorate(v, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
        b.decorate(v, Decoration::Binding, [Operand::LiteralBit32(binding)]);
    }
    b.decorate(dst, Decoration::NonReadable, []);
    let pcs = b.type_struct([f32t, f32t, f32t, f32t, u32t, u32t, f32t, u32t]);
    b.decorate(pcs, Decoration::Block, []);
    for m in 0..8 {
        b.member_decorate(pcs, m, Decoration::Offset, [Operand::LiteralBit32(m * 4)]);
    }
    let p_pcs = b.type_pointer(None, StorageClass::PushConstant, pcs);
    let p_pc_u = b.type_pointer(None, StorageClass::PushConstant, u32t);
    let p_pc_f = b.type_pointer(None, StorageClass::PushConstant, f32t);
    let pc = global(&mut b, p_pcs, StorageClass::PushConstant);
    let p_in_v3 = b.type_pointer(None, StorageClass::Input, v3u);
    let gid = global(&mut b, p_in_v3, StorageClass::Input);
    b.decorate(gid, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::GlobalInvocationId)]);

    let fn_ty = b.type_function(void, []);
    let func = b.begin_function(void, None, spirv::FunctionControl::NONE, fn_ty).unwrap();
    b.begin_block(None).unwrap();
    let ku = |b: &mut Builder, v: u32| b.constant_bit32(u32t, v);
    let kf = |b: &mut Builder, v: f32| b.constant_bit32(f32t, v.to_bits());
    let kf3 = |b: &mut Builder, v: [f32; 3]| {
        let c: Vec<u32> = v.iter().map(|&x| b.constant_bit32(f32t, x.to_bits())).collect();
        b.constant_composite(v3f, c)
    };
    let pcf = |b: &mut Builder, m: u32| {
        let i = b.constant_bit32(u32t, m);
        let p = b.access_chain(p_pc_f, None, pc, [i]).unwrap();
        b.load(f32t, None, p, None, []).unwrap()
    };
    let pcu = |b: &mut Builder, m: u32| {
        let i = b.constant_bit32(u32t, m);
        let p = b.access_chain(p_pc_u, None, pc, [i]).unwrap();
        b.load(u32t, None, p, None, []).unwrap()
    };
    let x0 = pcf(&mut b, 0);
    let y0 = pcf(&mut b, 1);
    let x1 = pcf(&mut b, 2);
    let y1 = pcf(&mut b, 3);
    let w = pcu(&mut b, 4);
    let h = pcu(&mut b, 5);
    let scale = pcf(&mut b, 6);
    let flags = pcu(&mut b, 7);
    let g = b.load(v3u, None, gid, None, []).unwrap();
    let x = b.composite_extract(u32t, None, g, [0]).unwrap();
    let y = b.composite_extract(u32t, None, g, [1]).unwrap();
    let in_x = b.u_less_than(boolt, None, x, w).unwrap();
    let in_y = b.u_less_than(boolt, None, y, h).unwrap();
    let in_img = b.logical_and(boolt, None, in_x, in_y).unwrap();
    let body = b.id();
    let merge = b.id();
    b.selection_merge(merge, spirv::SelectionControl::NONE).unwrap();
    b.branch_conditional(in_img, body, merge, []).unwrap();
    b.begin_block(Some(body)).unwrap();
    // The output pixel's centre in the frame's rectangle.
    let half = kf(&mut b, 0.5);
    let fx = b.convert_u_to_f(f32t, None, x).unwrap();
    let fx = b.f_add(f32t, None, fx, half).unwrap();
    let fy = b.convert_u_to_f(f32t, None, y).unwrap();
    let fy = b.f_add(f32t, None, fy, half).unwrap();
    let ge_x = b.f_ord_greater_than_equal(boolt, None, fx, x0).unwrap();
    let lt_x = b.f_ord_less_than(boolt, None, fx, x1).unwrap();
    let ge_y = b.f_ord_greater_than_equal(boolt, None, fy, y0).unwrap();
    let lt_y = b.f_ord_less_than(boolt, None, fy, y1).unwrap();
    let ix = b.logical_and(boolt, None, ge_x, lt_x).unwrap();
    let iy = b.logical_and(boolt, None, ge_y, lt_y).unwrap();
    let inside = b.logical_and(boolt, None, ix, iy).unwrap();
    let dx = b.f_sub(f32t, None, fx, x0).unwrap();
    let wx = b.f_sub(f32t, None, x1, x0).unwrap();
    let u = b.f_div(f32t, None, dx, wx).unwrap();
    let dy = b.f_sub(f32t, None, fy, y0).unwrap();
    let wy = b.f_sub(f32t, None, y1, y0).unwrap();
    let v = b.f_div(f32t, None, dy, wy).unwrap();
    let uv = b.composite_construct(v2f, None, [u, v]).unwrap();
    let si = b.load(simg_t, None, src, None, []).unwrap();
    let lod0 = kf(&mut b, 0.0);
    let c4 = b.image_sample_explicit_lod(v4f, None, si, uv, spirv::ImageOperands::LOD, [Operand::IdRef(lod0)]).unwrap();
    let c = b.vector_shuffle(v3f, None, c4, c4, [0, 1, 2]).unwrap();
    let zero3 = kf3(&mut b, [0.0; 3]);
    let one3 = kf3(&mut b, [1.0; 3]);
    let c = b.ext_inst(v3f, None, glsl, 43, [Operand::IdRef(c), Operand::IdRef(zero3), Operand::IdRef(one3)]).unwrap(); // FClamp
    // Gamma 2.2, or the piecewise sRGB curve.
    let g22 = kf3(&mut b, [2.2; 3]);
    let lin_g = b.ext_inst(v3f, None, glsl, 26, [Operand::IdRef(c), Operand::IdRef(g22)]).unwrap(); // Pow
    let k1292 = kf3(&mut b, [12.92; 3]);
    let lo = b.f_div(v3f, None, c, k1292).unwrap();
    let k055 = kf3(&mut b, [0.055; 3]);
    let k1055 = kf3(&mut b, [1.055; 3]);
    let k24 = kf3(&mut b, [2.4; 3]);
    let t = b.f_add(v3f, None, c, k055).unwrap();
    let t = b.f_div(v3f, None, t, k1055).unwrap();
    let hi = b.ext_inst(v3f, None, glsl, 26, [Operand::IdRef(t), Operand::IdRef(k24)]).unwrap();
    let kcut = kf3(&mut b, [0.04045; 3]);
    let low = b.f_ord_less_than_equal(v3b, None, c, kcut).unwrap();
    let lin_s = b.select(v3f, None, low, lo, hi).unwrap();
    let f1 = ku(&mut b, 1);
    let f2 = ku(&mut b, 2);
    let z = ku(&mut b, 0);
    let pw = b.bitwise_and(u32t, None, flags, f1).unwrap();
    let pw = b.i_not_equal(boolt, None, pw, z).unwrap();
    let pw3 = b.composite_construct(v3b, None, [pw, pw, pw]).unwrap();
    let lin = b.select(v3f, None, pw3, lin_s, lin_g).unwrap();
    // BT.709 -> BT.2020 primaries.
    let rows = [[0.627_404, 0.329_282, 0.043_313_6], [0.069_097, 0.919_54, 0.011_361_2], [0.016_391_6, 0.088_013_2, 0.895_595]];
    let comps: Vec<u32> = rows
        .iter()
        .map(|r| {
            let k = kf3(&mut b, *r);
            b.dot(f32t, None, k, lin).unwrap()
        })
        .collect();
    let sdr2020 = b.composite_construct(v3f, None, comps).unwrap();
    // SMPTE ST 2084 (PQ) constants.
    let m1 = kf3(&mut b, [0.159_301_76; 3]);
    let m2 = kf3(&mut b, [78.843_75; 3]);
    let c1 = kf3(&mut b, [0.835_937_5; 3]);
    let c2 = kf3(&mut b, [18.851_562; 3]);
    let c3 = kf3(&mut b, [18.687_5; 3]);
    // A PQ frame decoded: e = c^(1/m2), ((e - c1)+ / (c2 - c3 e))^(1/m1).
    let inv_m1 = kf3(&mut b, [1.0 / 0.159_301_76; 3]);
    let inv_m2 = kf3(&mut b, [1.0 / 78.843_75; 3]);
    let e = b.ext_inst(v3f, None, glsl, 26, [Operand::IdRef(c), Operand::IdRef(inv_m2)]).unwrap();
    let en = b.f_sub(v3f, None, e, c1).unwrap();
    let en = b.ext_inst(v3f, None, glsl, 40, [Operand::IdRef(en), Operand::IdRef(zero3)]).unwrap(); // FMax
    let ed = b.f_mul(v3f, None, c3, e).unwrap();
    let ed = b.f_sub(v3f, None, c2, ed).unwrap();
    let eq = b.f_div(v3f, None, en, ed).unwrap();
    let pq_lin = b.ext_inst(v3f, None, glsl, 26, [Operand::IdRef(eq), Operand::IdRef(inv_m1)]).unwrap();
    let f4 = ku(&mut b, 4);
    let is_pq = b.bitwise_and(u32t, None, flags, f4).unwrap();
    let is_pq = b.i_not_equal(boolt, None, is_pq, z).unwrap();
    let pq3 = b.composite_construct(v3b, None, [is_pq, is_pq, is_pq]).unwrap();
    let y2020 = b.select(v3f, None, pq3, pq_lin, sdr2020).unwrap();
    let in3 = b.composite_construct(v3b, None, [inside, inside, inside]).unwrap();
    let y2020 = b.select(v3f, None, in3, y2020, zero3).unwrap();
    let y_scaled = b.vector_times_scalar(v3f, None, y2020, scale).unwrap();
    let y2020 = b.ext_inst(v3f, None, glsl, 43, [Operand::IdRef(y_scaled), Operand::IdRef(zero3), Operand::IdRef(one3)]).unwrap();
    let ym = b.ext_inst(v3f, None, glsl, 26, [Operand::IdRef(y2020), Operand::IdRef(m1)]).unwrap();
    let num = b.f_mul(v3f, None, c2, ym).unwrap();
    let num = b.f_add(v3f, None, c1, num).unwrap();
    let den = b.f_mul(v3f, None, c3, ym).unwrap();
    let den = b.f_add(v3f, None, one3, den).unwrap();
    let q = b.f_div(v3f, None, num, den).unwrap();
    let pq = b.ext_inst(v3f, None, glsl, 26, [Operand::IdRef(q), Operand::IdRef(m2)]).unwrap();
    let k1023 = kf(&mut b, 1023.0);
    let s = b.vector_times_scalar(v3f, None, pq, k1023).unwrap();
    let half3 = kf3(&mut b, [0.5; 3]);
    let s = b.f_add(v3f, None, s, half3).unwrap();
    let qu = b.convert_f_to_u(v3u, None, s).unwrap();
    let r = b.composite_extract(u32t, None, qu, [0]).unwrap();
    let gq = b.composite_extract(u32t, None, qu, [1]).unwrap();
    let bq = b.composite_extract(u32t, None, qu, [2]).unwrap();
    let sw = b.bitwise_and(u32t, None, flags, f2).unwrap();
    let sw = b.i_not_equal(boolt, None, sw, z).unwrap();
    let low_c = b.select(u32t, None, sw, bq, r).unwrap();
    let high_c = b.select(u32t, None, sw, r, bq).unwrap();
    let s10 = ku(&mut b, 10);
    let s20 = ku(&mut b, 20);
    let alpha = ku(&mut b, 3 << 30);
    let gs = b.shift_left_logical(u32t, None, gq, s10).unwrap();
    let hs = b.shift_left_logical(u32t, None, high_c, s20).unwrap();
    let packed = b.bitwise_or(u32t, None, low_c, gs).unwrap();
    let packed = b.bitwise_or(u32t, None, packed, hs).unwrap();
    let packed = b.bitwise_or(u32t, None, packed, alpha).unwrap();
    // 8-bit sRGB output (bit 3: an SDR swapchain showing PQ frames; the
    // scale makes white 1): BT.2020 -> BT.709, the sRGB curve.
    let rows7 = [[1.660_491, -0.587_641, -0.072_850], [-0.124_550, 1.132_9, -0.008_349], [-0.018_151, -0.100_579, 1.118_73]];
    let comps7: Vec<u32> = rows7
        .iter()
        .map(|r| {
            let k = kf3(&mut b, *r);
            b.dot(f32t, None, k, y_scaled).unwrap()
        })
        .collect();
    let s7 = b.composite_construct(v3f, None, comps7).unwrap();
    let s7 = b.ext_inst(v3f, None, glsl, 43, [Operand::IdRef(s7), Operand::IdRef(zero3), Operand::IdRef(one3)]).unwrap();
    let kcut_o = kf3(&mut b, [0.003_130_8; 3]);
    let lo_o = b.f_mul(v3f, None, s7, k1292).unwrap();
    let inv24 = kf3(&mut b, [1.0 / 2.4; 3]);
    let hi_o = b.ext_inst(v3f, None, glsl, 26, [Operand::IdRef(s7), Operand::IdRef(inv24)]).unwrap();
    let hi_o = b.f_mul(v3f, None, hi_o, k1055).unwrap();
    let hi_o = b.f_sub(v3f, None, hi_o, k055).unwrap();
    let low_o = b.f_ord_less_than_equal(v3b, None, s7, kcut_o).unwrap();
    let enc = b.select(v3f, None, low_o, lo_o, hi_o).unwrap();
    let k255 = kf(&mut b, 255.0);
    let e8 = b.vector_times_scalar(v3f, None, enc, k255).unwrap();
    let e8 = b.f_add(v3f, None, e8, half3).unwrap();
    let q8 = b.convert_f_to_u(v3u, None, e8).unwrap();
    let r8 = b.composite_extract(u32t, None, q8, [0]).unwrap();
    let g8 = b.composite_extract(u32t, None, q8, [1]).unwrap();
    let b8 = b.composite_extract(u32t, None, q8, [2]).unwrap();
    let lo8 = b.select(u32t, None, sw, b8, r8).unwrap();
    let hi8 = b.select(u32t, None, sw, r8, b8).unwrap();
    let s8 = ku(&mut b, 8);
    let s16 = ku(&mut b, 16);
    let alpha8 = ku(&mut b, 0xFF00_0000);
    let g8s = b.shift_left_logical(u32t, None, g8, s8).unwrap();
    let h8s = b.shift_left_logical(u32t, None, hi8, s16).unwrap();
    let p8 = b.bitwise_or(u32t, None, lo8, g8s).unwrap();
    let p8 = b.bitwise_or(u32t, None, p8, h8s).unwrap();
    let p8 = b.bitwise_or(u32t, None, p8, alpha8).unwrap();
    let f8 = ku(&mut b, 8);
    let is8 = b.bitwise_and(u32t, None, flags, f8).unwrap();
    let is8 = b.i_not_equal(boolt, None, is8, z).unwrap();
    let packed = b.select(u32t, None, is8, p8, packed).unwrap();
    let texel = b.composite_construct(v4u, None, [packed, z, z, z]).unwrap();
    let xi = b.bitcast(i32t, None, x).unwrap();
    let yi = b.bitcast(i32t, None, y).unwrap();
    let at = b.composite_construct(v2i, None, [xi, yi]).unwrap();
    let di = b.load(st_t, None, dst, None, []).unwrap();
    b.image_write(di, at, texel, None, []).unwrap();
    b.branch(merge).unwrap();
    b.begin_block(Some(merge)).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    b.entry_point(ExecutionModel::GLCompute, func, "main", [gid]);
    b.execution_mode(func, ExecutionMode::LocalSize, [8, 8, 1]);
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
    fn dirty_copy_kernel_assembles() {
        let w = super::dirty_copy_cs();
        check_module(&w, "dirty_copy");
        if let Ok(dir) = std::env::var("TLG_DUMP_KERNELS") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(format!("{dir}/dirty_copy.spv"), bytes).unwrap();
        }
    }

    #[test]
    fn depth_copy_kernel_assembles() {
        let w = super::depth_copy_cs();
        check_module(&w, "depth_copy");
        if let Ok(dir) = std::env::var("TLG_DUMP_KERNELS") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(format!("{dir}/depth_copy.spv"), bytes).unwrap();
        }
    }

    #[test]
    fn hdr10_kernel_assembles() {
        let w = super::hdr10_cs();
        check_module(&w, "hdr10");
        if let Ok(dir) = std::env::var("TLG_DUMP_KERNELS") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(format!("{dir}/hdr10.spv"), bytes).unwrap();
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
        check_module(&super::indirect_draw_cs(), "indirect_draw");
        check_module(&super::remap_cs(false), "remap");
        if let Ok(dir) = std::env::var("TLG_DUMP_KERNELS") {
            let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(format!("{dir}/indirect_args.spv"), bytes).unwrap();
            let bytes: Vec<u8> = super::indirect_draw_cs().iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(format!("{dir}/indirect_draw.spv"), bytes).unwrap();
        }
    }
}
