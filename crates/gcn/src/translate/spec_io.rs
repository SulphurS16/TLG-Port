//! `Spec` as bytes: the runtime's pipeline manifest keeps the inputs of the
//! session's translations for the next session (`gpu::manifest`).

use super::{ImageDim, MrtSpec, NumClass, PsInput, ResSpec, Spec, Stage, TessRole, TessSpec};

/// The layout of `Spec::encode`; files written with another are not read.
pub const SPEC_IO_VERSION: u32 = 1;

/// Appends little-endian values to a byte vector.
pub struct Writer<'a>(pub &'a mut Vec<u8>);

impl Writer<'_> {
    pub fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    pub fn u16(&mut self, v: u16) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn bool(&mut self, v: bool) {
        self.u8(v as u8);
    }
    /// A count, then the dwords.
    pub fn words(&mut self, w: &[u32]) {
        self.u32(w.len() as u32);
        for &x in w {
            self.u32(x);
        }
    }
}

/// Reads what `Writer` wrote; None past the end or on an invalid value.
pub struct Reader<'a>(pub &'a [u8]);

impl<'a> Reader<'a> {
    pub fn bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        if self.0.len() < n {
            return None;
        }
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Some(a)
    }
    pub fn u8(&mut self) -> Option<u8> {
        Some(self.bytes(1)?[0])
    }
    pub fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.bytes(2)?.try_into().ok()?))
    }
    pub fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.bytes(4)?.try_into().ok()?))
    }
    pub fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.bytes(8)?.try_into().ok()?))
    }
    pub fn bool(&mut self) -> Option<bool> {
        match self.u8()? {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }
    /// A count (at most `max`), then that many dwords.
    pub fn words(&mut self, max: usize) -> Option<Vec<u32>> {
        let n = self.u32()? as usize;
        if n > max || n * 4 > self.0.len() {
            return None;
        }
        (0..n).map(|_| self.u32()).collect()
    }
    /// A count of items (at most `max`).
    fn count(&mut self, max: usize) -> Option<usize> {
        let n = self.u32()? as usize;
        (n <= max).then_some(n)
    }
}

fn dim_code(d: ImageDim) -> u8 {
    match d {
        ImageDim::D1 => 0,
        ImageDim::D2 => 1,
        ImageDim::D3 => 2,
        ImageDim::Cube => 3,
        ImageDim::D1Array => 4,
        ImageDim::D2Array => 5,
        ImageDim::CubeArray => 6,
    }
}

fn dim_of(c: u8) -> Option<ImageDim> {
    Some(match c {
        0 => ImageDim::D1,
        1 => ImageDim::D2,
        2 => ImageDim::D3,
        3 => ImageDim::Cube,
        4 => ImageDim::D1Array,
        5 => ImageDim::D2Array,
        6 => ImageDim::CubeArray,
        _ => return None,
    })
}

impl Spec {
    /// Appends the specialization (`SPEC_IO_VERSION`).
    pub fn encode(&self, out: &mut Vec<u8>) {
        let mut w = Writer(out);
        w.u8(match self.stage {
            Stage::Vertex => 0,
            Stage::Fragment => 1,
            Stage::Compute => 2,
        });
        w.u16(self.user_sgprs);
        w.u32(self.vs_vgpr_comp_cnt);
        for v in self.vs_step_rates {
            w.u32(v);
        }
        w.u32(self.vs_out_cntl);
        w.bool(self.denorm_flush32);
        w.bool(self.debug_pos_store);
        let t = &self.tess;
        w.u8(match t.role {
            TessRole::None => 0,
            TessRole::Ls => 1,
            TessRole::Hs => 2,
            TessRole::Ds => 3,
        });
        for v in [t.patches, t.in_cp, t.out_cp, t.lds_dwords, t.tf_param, t.rsrc2] {
            w.u32(v);
        }
        w.u32(self.ps_input_addr);
        w.u32(self.ps_inputs.len() as u32);
        for p in &self.ps_inputs {
            w.u8(p.param);
            w.bool(p.flat);
            w.bool(p.default.is_some());
            w.u8(p.default.unwrap_or(0));
        }
        for m in &self.mrt {
            w.u8(m.export_format);
            w.u8(m.class);
        }
        for v in self.cs_local_size {
            w.u32(v);
        }
        w.u32(self.cs_rsrc2);
        w.u32(self.resources.len() as u32);
        for r in &self.resources {
            match *r {
                ResSpec::Buffer { dfmt, nfmt, dst_sel } => {
                    w.u8(0);
                    w.u8(dfmt);
                    w.u8(nfmt);
                    w.u16(dst_sel);
                }
                ResSpec::Image { dim, class } => {
                    w.u8(1);
                    w.u8(dim_code(dim));
                    w.u8(match class {
                        NumClass::Float => 0,
                        NumClass::Sint => 1,
                        NumClass::Uint => 2,
                    });
                }
                ResSpec::Sampler => w.u8(2),
            }
        }
        w.words(&self.buffer_strides);
        w.u32(self.binding_base);
        w.u32(self.debug_capture.len() as u32);
        for &(a, b, f) in &self.debug_capture {
            w.u32(a);
            w.u32(b);
            w.bool(f);
        }
        w.u8(self.page_marks);
    }

    /// The specialization `encode` wrote, or None for bytes it did not.
    pub fn decode(r: &mut Reader) -> Option<Spec> {
        let stage = match r.u8()? {
            0 => Stage::Vertex,
            1 => Stage::Fragment,
            2 => Stage::Compute,
            _ => return None,
        };
        let user_sgprs = r.u16()?;
        let vs_vgpr_comp_cnt = r.u32()?;
        let vs_step_rates = [r.u32()?, r.u32()?];
        let vs_out_cntl = r.u32()?;
        let denorm_flush32 = r.bool()?;
        let debug_pos_store = r.bool()?;
        let role = match r.u8()? {
            0 => TessRole::None,
            1 => TessRole::Ls,
            2 => TessRole::Hs,
            3 => TessRole::Ds,
            _ => return None,
        };
        let tess = TessSpec { role, patches: r.u32()?, in_cp: r.u32()?, out_cp: r.u32()?, lds_dwords: r.u32()?, tf_param: r.u32()?, rsrc2: r.u32()? };
        let ps_input_addr = r.u32()?;
        let n = r.count(256)?;
        let mut ps_inputs = Vec::with_capacity(n);
        for _ in 0..n {
            let param = r.u8()?;
            let flat = r.bool()?;
            let has = r.bool()?;
            let d = r.u8()?;
            ps_inputs.push(PsInput { param, flat, default: has.then_some(d) });
        }
        let mut mrt = [MrtSpec::default(); 8];
        for m in &mut mrt {
            *m = MrtSpec { export_format: r.u8()?, class: r.u8()? };
        }
        let cs_local_size = [r.u32()?, r.u32()?, r.u32()?];
        let cs_rsrc2 = r.u32()?;
        let n = r.count(1024)?;
        let mut resources = Vec::with_capacity(n);
        for _ in 0..n {
            resources.push(match r.u8()? {
                0 => ResSpec::Buffer { dfmt: r.u8()?, nfmt: r.u8()?, dst_sel: r.u16()? },
                1 => {
                    let dim = dim_of(r.u8()?)?;
                    let class = match r.u8()? {
                        0 => NumClass::Float,
                        1 => NumClass::Sint,
                        2 => NumClass::Uint,
                        _ => return None,
                    };
                    ResSpec::Image { dim, class }
                }
                2 => ResSpec::Sampler,
                _ => return None,
            });
        }
        let buffer_strides = r.words(1024)?;
        let binding_base = r.u32()?;
        let n = r.count(256)?;
        let mut debug_capture = Vec::with_capacity(n);
        for _ in 0..n {
            debug_capture.push((r.u32()?, r.u32()?, r.bool()?));
        }
        let page_marks = r.u8()?;
        Some(Spec {
            stage,
            user_sgprs,
            vs_vgpr_comp_cnt,
            vs_step_rates,
            vs_out_cntl,
            denorm_flush32,
            debug_pos_store,
            tess,
            ps_input_addr,
            ps_inputs,
            mrt,
            cs_local_size,
            cs_rsrc2,
            resources,
            buffer_strides,
            binding_base,
            debug_capture,
            page_marks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_round_trips() {
        let spec = Spec {
            stage: Stage::Fragment,
            user_sgprs: 12,
            vs_vgpr_comp_cnt: 3,
            vs_step_rates: [2, 5],
            vs_out_cntl: 0x00E0_00FF,
            denorm_flush32: true,
            debug_pos_store: false,
            tess: TessSpec { role: TessRole::Hs, patches: 4, in_cp: 3, out_cp: 3, lds_dwords: 1024, tf_param: 0x25, rsrc2: 0x180 },
            ps_input_addr: 0x802,
            ps_inputs: vec![PsInput { param: 1, flat: true, default: None }, PsInput { param: 7, flat: false, default: Some(2) }],
            mrt: [MrtSpec { export_format: 4, class: 1 }; 8],
            cs_local_size: [64, 2, 1],
            cs_rsrc2: 0x9C,
            resources: vec![ResSpec::Buffer { dfmt: 14, nfmt: 7, dst_sel: 0xFAC }, ResSpec::Image { dim: ImageDim::CubeArray, class: NumClass::Sint }, ResSpec::Sampler],
            buffer_strides: vec![16, 0, 0],
            binding_base: 32,
            debug_capture: vec![(5, 2, true)],
            page_marks: 3,
        };
        let mut b = Vec::new();
        spec.encode(&mut b);
        let mut r = Reader(&b);
        assert_eq!(Spec::decode(&mut r).as_ref(), Some(&spec));
        assert!(r.0.is_empty());
        // Cut short or with a bad tag: no specialization.
        assert!(Spec::decode(&mut Reader(&b[..b.len() - 1])).is_none());
        let mut bad = b.clone();
        bad[0] = 9;
        assert!(Spec::decode(&mut Reader(&bad)).is_none());
    }
}
