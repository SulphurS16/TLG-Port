//! GCN resource descriptors: V# (buffer), T# (image), S# (sampler).

use ash::vk;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Vsharp {
    pub base: u64,
    pub stride: u32,
    pub num_records: u32,
    pub dst_sel: u16,
    pub nfmt: u8,
    pub dfmt: u8,
    pub add_tid: bool,
    pub swizzle: bool,
}

impl Vsharp {
    pub fn decode(w: &[u32]) -> Vsharp {
        Vsharp {
            // PS4 GPU addresses are 40-bit; word1[15:8] carries other bits.
            base: (w[0] as u64) | (((w[1] & 0xFF) as u64) << 32),
            stride: (w[1] >> 16) & 0x3FFF,
            num_records: w[2],
            dst_sel: (w[3] & 0xFFF) as u16,
            nfmt: ((w[3] >> 12) & 7) as u8,
            dfmt: ((w[3] >> 15) & 0xF) as u8,
            add_tid: (w[3] >> 23) & 1 != 0,
            swizzle: (w[1] >> 31) & 1 != 0,
        }
    }
    /// Byte size of the addressable range.
    pub fn size(&self) -> u64 {
        if self.stride == 0 { self.num_records as u64 } else { self.num_records as u64 * self.stride as u64 }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Tsharp {
    pub base: u64,
    pub dfmt: u8,
    pub nfmt: u8,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub pitch: u32,
    pub dst_sel: u16,
    pub base_level: u32,
    pub last_level: u32,
    pub tiling_index: u8,
    pub pow2_pad: bool,
    pub ty: u8,
    pub base_array: u32,
    pub last_array: u32,
}

pub const T_1D: u8 = 8;
pub const T_2D: u8 = 9;
pub const T_3D: u8 = 10;
pub const T_CUBE: u8 = 11;
pub const T_1D_ARRAY: u8 = 12;
pub const T_2D_ARRAY: u8 = 13;
pub const T_2D_MSAA: u8 = 14;
pub const T_2D_MSAA_ARRAY: u8 = 15;

impl Tsharp {
    pub fn decode(w: &[u32]) -> Tsharp {
        Tsharp {
            // BASE_ADDRESS_HI (word1[7:0]) would extend the address beyond
            // 40 bits; titles set bit 46 there (e.g. 0x40_1056eb3900) for
            // ordinary textures, so only the 40-bit address is used.
            base: (((w[0] as u64) | (((w[1] & 0xFF) as u64) << 32)) << 8) & ((1u64 << 40) - 1),
            dfmt: ((w[1] >> 20) & 0x3F) as u8,
            nfmt: ((w[1] >> 26) & 0xF) as u8,
            width: (w[2] & 0x3FFF) + 1,
            height: ((w[2] >> 14) & 0x3FFF) + 1,
            dst_sel: (w[3] & 0xFFF) as u16,
            base_level: (w[3] >> 12) & 0xF,
            last_level: (w[3] >> 16) & 0xF,
            tiling_index: ((w[3] >> 20) & 0x1F) as u8,
            pow2_pad: (w[3] >> 25) & 1 != 0,
            ty: ((w[3] >> 28) & 0xF) as u8,
            depth: (w.get(4).copied().unwrap_or(0) & 0x1FFF) + 1,
            pitch: ((w.get(4).copied().unwrap_or(0) >> 13) & 0x3FFF) + 1,
            base_array: w.get(5).copied().unwrap_or(0) & 0x1FFF,
            last_array: (w.get(5).copied().unwrap_or(0) >> 13) & 0x1FFF,
        }
    }
    pub fn layers(&self) -> u32 {
        match self.ty {
            // Cube slices count faces (Bloodborne's shadow cube arrays use
            // BASE/LAST_ARRAY 0..35 over six 6-face render targets); a range
            // that is not a whole number of cubes is taken as cube units.
            T_CUBE => {
                let n = self.last_array.saturating_sub(self.base_array) + 1;
                if n % 6 == 0 { n } else { 6 * n }
            }
            T_1D_ARRAY | T_2D_ARRAY | T_2D_MSAA_ARRAY => self.last_array.saturating_sub(self.base_array) + 1,
            _ => 1,
        }
    }
    /// Slices of the underlying resource. For arrays the DEPTH field holds
    /// the total; BASE_ARRAY/LAST_ARRAY select the viewed range and the
    /// shader's slice coordinate is relative to BASE_ARRAY.
    pub fn total_layers(&self) -> u32 {
        match self.ty {
            T_CUBE | T_3D => self.layers(),
            T_1D_ARRAY | T_2D_ARRAY | T_2D_MSAA_ARRAY => self.depth.max(self.last_array + 1),
            _ if self.base_array > 0 => self.depth.max(self.base_array + 1),
            _ => 1,
        }
    }
    /// First slice of the view within the resource.
    pub fn view_base_layer(&self) -> u32 {
        match self.ty {
            T_CUBE | T_3D => 0,
            _ => self.base_array.min(self.total_layers() - 1),
        }
    }
    pub fn levels(&self) -> u32 {
        self.last_level.saturating_sub(self.base_level) + 1
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Ssharp(pub [u32; 4]);

impl Ssharp {
    pub fn clamp(&self, axis: u32) -> u32 {
        (self.0[0] >> (axis * 3)) & 7
    }
    pub fn compare_func(&self) -> u32 {
        (self.0[0] >> 12) & 7
    }
    pub fn force_unnormalized(&self) -> bool {
        (self.0[0] >> 15) & 1 != 0
    }
    pub fn max_aniso(&self) -> u32 {
        (self.0[0] >> 9) & 7
    }
    pub fn min_lod(&self) -> f32 {
        (self.0[1] & 0xFFF) as f32 / 256.0
    }
    pub fn max_lod(&self) -> f32 {
        ((self.0[1] >> 12) & 0xFFF) as f32 / 256.0
    }
    pub fn lod_bias(&self) -> f32 {
        let v = (self.0[2] & 0x3FFF) as i32;
        let v = (v << 18) >> 18; // sign-extend 14 bits
        v as f32 / 256.0
    }
    pub fn mag_filter(&self) -> u32 {
        (self.0[2] >> 20) & 3
    }
    pub fn min_filter(&self) -> u32 {
        (self.0[2] >> 22) & 3
    }
    pub fn mip_filter(&self) -> u32 {
        (self.0[2] >> 26) & 3
    }
    pub fn border_type(&self) -> u32 {
        (self.0[3] >> 30) & 3
    }
}

pub fn vk_address_mode(m: u32) -> vk::SamplerAddressMode {
    match m {
        0 => vk::SamplerAddressMode::REPEAT,
        1 => vk::SamplerAddressMode::MIRRORED_REPEAT,
        2 => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        // (Mirror once; where the device lacks it, mirrored repeat - the
        // same within one mirrored repeat of the texture.)
        3 | 5 | 7 if super::vk::ctx().mirror_clamp => vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE,
        3 | 5 | 7 => vk::SamplerAddressMode::MIRRORED_REPEAT,
        _ => vk::SamplerAddressMode::CLAMP_TO_BORDER,
    }
}

pub fn vk_compare(f: u32) -> vk::CompareOp {
    match f {
        0 => vk::CompareOp::NEVER,
        1 => vk::CompareOp::LESS,
        2 => vk::CompareOp::EQUAL,
        3 => vk::CompareOp::LESS_OR_EQUAL,
        4 => vk::CompareOp::GREATER,
        5 => vk::CompareOp::NOT_EQUAL,
        6 => vk::CompareOp::GREATER_OR_EQUAL,
        _ => vk::CompareOp::ALWAYS,
    }
}

/// Number class of an image number format.
pub fn num_class(nfmt: u8) -> gcn::translate::NumClass {
    match nfmt {
        4 | 12 => gcn::translate::NumClass::Uint,
        5 => gcn::translate::NumClass::Sint,
        _ => gcn::translate::NumClass::Float,
    }
}

/// Vulkan format for an image (texture or render target) data/number format pair.
pub fn vk_format(dfmt: u8, nfmt: u8) -> Option<vk::Format> {
    use vk::Format as F;
    let srgb = nfmt == 9;
    Some(match (dfmt, nfmt) {
        (1, 0) => F::R8_UNORM,
        (1, 1) => F::R8_SNORM,
        (1, 4) => F::R8_UINT,
        (1, 5) => F::R8_SINT,
        (1, 9) => F::R8_SRGB,
        (2, 0) => F::R16_UNORM,
        (2, 1) => F::R16_SNORM,
        (2, 4) => F::R16_UINT,
        (2, 5) => F::R16_SINT,
        (2, 7) => F::R16_SFLOAT,
        (3, 0) => F::R8G8_UNORM,
        (3, 1) => F::R8G8_SNORM,
        (3, 4) => F::R8G8_UINT,
        (3, 5) => F::R8G8_SINT,
        (4, 4) => F::R32_UINT,
        (4, 5) => F::R32_SINT,
        (4, 7) => F::R32_SFLOAT,
        (5, 0) => F::R16G16_UNORM,
        (5, 1) => F::R16G16_SNORM,
        (5, 4) => F::R16G16_UINT,
        (5, 5) => F::R16G16_SINT,
        (5, 7) => F::R16G16_SFLOAT,
        (6, 7) => F::B10G11R11_UFLOAT_PACK32,
        (7, 7) => F::B10G11R11_UFLOAT_PACK32,
        (9, 0) => F::A2B10G10R10_UNORM_PACK32,
        (9, 4) => F::A2B10G10R10_UINT_PACK32,
        (10, 0) => F::R8G8B8A8_UNORM,
        (10, 1) => F::R8G8B8A8_SNORM,
        (10, 4) => F::R8G8B8A8_UINT,
        (10, 5) => F::R8G8B8A8_SINT,
        (10, _) if srgb => F::R8G8B8A8_SRGB,
        (11, 4) => F::R32G32_UINT,
        (11, 5) => F::R32G32_SINT,
        (11, 7) => F::R32G32_SFLOAT,
        (12, 0) => F::R16G16B16A16_UNORM,
        (12, 1) => F::R16G16B16A16_SNORM,
        (12, 4) => F::R16G16B16A16_UINT,
        (12, 5) => F::R16G16B16A16_SINT,
        (12, 7) => F::R16G16B16A16_SFLOAT,
        (13, 4) => F::R32G32B32_UINT,
        (13, 5) => F::R32G32B32_SINT,
        (13, 7) => F::R32G32B32_SFLOAT,
        (14, 4) => F::R32G32B32A32_UINT,
        (14, 5) => F::R32G32B32A32_SINT,
        (14, 7) => F::R32G32B32A32_SFLOAT,
        (16, 0) => F::B5G6R5_UNORM_PACK16,
        (17, 0) => F::A1R5G5B5_UNORM_PACK16,
        (18, 0) => F::R5G5B5A1_UNORM_PACK16,
        (19, 0) => F::R4G4B4A4_UNORM_PACK16,
        (34, 7) => F::E5B9G9R9_UFLOAT_PACK32,
        (35, _) => if srgb { F::BC1_RGBA_SRGB_BLOCK } else { F::BC1_RGBA_UNORM_BLOCK },
        (36, _) => if srgb { F::BC2_SRGB_BLOCK } else { F::BC2_UNORM_BLOCK },
        (37, _) => if srgb { F::BC3_SRGB_BLOCK } else { F::BC3_UNORM_BLOCK },
        (38, 1) => F::BC4_SNORM_BLOCK,
        (38, _) => F::BC4_UNORM_BLOCK,
        (39, 1) => F::BC5_SNORM_BLOCK,
        (39, _) => F::BC5_UNORM_BLOCK,
        (40, 1) => F::BC6H_SFLOAT_BLOCK,
        (40, _) => F::BC6H_UFLOAT_BLOCK,
        (41, _) => if srgb { F::BC7_SRGB_BLOCK } else { F::BC7_UNORM_BLOCK },
        _ => return None,
    })
}

/// Bytes per element (or per 4x4 block for BC formats) and block dimension.
pub fn format_bytes(dfmt: u8) -> (u32, u32) {
    match dfmt {
        1 => (1, 1),
        2 | 3 | 16..=19 => (2, 1),
        4..=10 | 20 | 21 | 34 => (4, 1),
        11 | 12 | 22 => (8, 1),
        13 => (12, 1),
        14 => (16, 1),
        35 | 38 => (8, 4),
        36 | 37 | 39..=41 => (16, 4),
        _ => (4, 1),
    }
}
