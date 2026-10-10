//! Generated stages of emulated tessellation (see `TessRole`): the LS and HS
//! programs run as compute, the domain shader as the tessellation evaluation
//! stage; the pipeline's vertex stage does nothing and its control stage only
//! hands the factors the HS wrote to the fixed-function tessellator.

use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{self, BuiltIn, Decoration, ExecutionMode, ExecutionModel, StorageClass};

/// Tessellation control shader with one output control point per patch:
/// patch `PrimitiveId`'s factors come from the factor buffer at `binding`
/// (u32 array, `tf_dwords(tf_param)` per patch: outer factors, then inner),
/// positive ones clamped to [min, max] (VGT_HOS_MIN/MAX_TESS_LEVEL; factors
/// of 0 or NaN discard the patch as on the hardware).
pub fn control_shader(tf_param: u32, min: f32, max: f32, binding: u32) -> Vec<u32> {
    let (outer_n, inner_n) = match tf_param & 3 {
        0 => (2, 0),
        1 => (3, 1),
        _ => (4, 2),
    };
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(spirv::Capability::Shader);
    b.capability(spirv::Capability::Tessellation);
    b.extension("SPV_KHR_storage_buffer_storage_class");
    let glsl = b.ext_inst_import("GLSL.std.450");
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let boolt = b.type_bool();
    let u32t = b.type_int(32, 0);
    let i32t = b.type_int(32, 1);
    let f32t = b.type_float(32, None);
    let c = |b: &mut Builder, v: u32| b.constant_bit32(u32t, v);
    let n4 = c(&mut b, 4);
    let n2 = c(&mut b, 2);
    let arr4 = b.type_array(f32t, n4);
    let arr2 = b.type_array(f32t, n2);
    let p_out4 = b.type_pointer(None, StorageClass::Output, arr4);
    let p_out2 = b.type_pointer(None, StorageClass::Output, arr2);
    let p_out_f = b.type_pointer(None, StorageClass::Output, f32t);
    let outer = global(&mut b, p_out4, StorageClass::Output);
    b.decorate(outer, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::TessLevelOuter)]);
    b.decorate(outer, Decoration::Patch, []);
    let inner = global(&mut b, p_out2, StorageClass::Output);
    b.decorate(inner, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::TessLevelInner)]);
    b.decorate(inner, Decoration::Patch, []);
    let p_in_i = b.type_pointer(None, StorageClass::Input, i32t);
    let prim = global(&mut b, p_in_i, StorageClass::Input);
    b.decorate(prim, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::PrimitiveId)]);
    let rta = b.type_runtime_array(u32t);
    b.decorate(rta, Decoration::ArrayStride, [Operand::LiteralBit32(4)]);
    let st = b.type_struct([rta]);
    b.decorate(st, Decoration::Block, []);
    b.member_decorate(st, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
    b.member_decorate(st, 0, Decoration::NonWritable, []);
    let p_sb = b.type_pointer(None, StorageClass::StorageBuffer, st);
    let p_sb_u = b.type_pointer(None, StorageClass::StorageBuffer, u32t);
    let tf = global(&mut b, p_sb, StorageClass::StorageBuffer);
    b.decorate(tf, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
    b.decorate(tf, Decoration::Binding, [Operand::LiteralBit32(binding)]);
    let clamp = min > 0.0 && min <= max;
    let fmin = b.constant_bit32(f32t, min.to_bits());
    let fmax = b.constant_bit32(f32t, max.to_bits());
    let fzero = b.constant_bit32(f32t, 0);
    let k0 = c(&mut b, 0);
    let stride = c(&mut b, crate::translate::tf_dwords(tf_param));
    let idx: Vec<u32> = (0..6).map(|k| c(&mut b, k)).collect();
    let fn_ty = b.type_function(void, []);
    let func = b.begin_function(void, None, spirv::FunctionControl::NONE, fn_ty).unwrap();
    b.begin_block(None).unwrap();
    let p = b.load(i32t, None, prim, None, []).unwrap();
    let pu = b.bitcast(u32t, None, p).unwrap();
    let base = b.i_mul(u32t, None, pu, stride).unwrap();
    let factor = |b: &mut Builder, k: u32| {
        let at = b.i_add(u32t, None, base, idx[k as usize]).unwrap();
        let ptr = b.access_chain(p_sb_u, None, tf, [k0, at]).unwrap();
        let bits = b.load(u32t, None, ptr, None, []).unwrap();
        let f = b.bitcast(f32t, None, bits).unwrap();
        if !clamp {
            return f;
        }
        let cl = b.ext_inst(f32t, None, glsl, spirv::GlslStd450Op::FClamp as u32, [Operand::IdRef(f), Operand::IdRef(fmin), Operand::IdRef(fmax)]).unwrap();
        let pos = b.f_ord_greater_than(boolt, None, f, fzero).unwrap();
        b.select(f32t, None, pos, cl, f).unwrap()
    };
    for k in 0..outer_n {
        let f = factor(&mut b, k);
        let dst = b.access_chain(p_out_f, None, outer, [idx[k as usize]]).unwrap();
        b.store(dst, f, None, []).unwrap();
    }
    for k in 0..inner_n {
        let f = factor(&mut b, outer_n + k);
        let dst = b.access_chain(p_out_f, None, inner, [idx[k as usize]]).unwrap();
        b.store(dst, f, None, []).unwrap();
    }
    b.ret().unwrap();
    b.end_function().unwrap();
    b.entry_point(ExecutionModel::TessellationControl, func, "main", [prim, outer, inner]);
    b.execution_mode(func, ExecutionMode::OutputVertices, [1]);
    b.module().assemble()
}

/// Vertex stage of an emulated tessellation pipeline: one vertex per patch,
/// no inputs or outputs (the control and evaluation stages use PrimitiveId).
pub fn null_vertex_shader() -> Vec<u32> {
    let mut b = Builder::new();
    b.set_version(1, 3);
    b.capability(spirv::Capability::Shader);
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();
    let fn_ty = b.type_function(void, []);
    let func = b.begin_function(void, None, spirv::FunctionControl::NONE, fn_ty).unwrap();
    b.begin_block(None).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    b.entry_point(ExecutionModel::Vertex, func, "main", []);
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
    use super::*;

    #[test]
    fn generated_tessellation_stages_parse() {
        for tf_param in [0x42, 0x41, 0x40, 0x22] {
            let words = control_shader(tf_param, 1.0, 3.0, 96);
            let m = rspirv::dr::load_words(&words).unwrap();
            assert_eq!(m.entry_points.len(), 1);
        }
        let m = rspirv::dr::load_words(&null_vertex_shader()).unwrap();
        assert_eq!(m.entry_points.len(), 1);
    }
}
