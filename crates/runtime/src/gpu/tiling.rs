//! GCN surface tiling (detiling for uploads).
//!
//! Elements are pixels or 4x4 blocks for BC formats. Micro tiles are 8x8
//! elements. 2D (macro) tiling follows the AMD address-library algorithm with
//! explicit pipe/bank parameters (calibrated for the PS4 configuration).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MicroType {
    Display,
    Thin,
    Depth,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArrayMode {
    Linear,
    Tiled1D,
    Tiled2D,
}

#[derive(Clone, Copy, Debug)]
pub struct TileMode {
    pub array: ArrayMode,
    pub micro: MicroType,
}

/// PS4 Gnm tile mode index -> layout.
pub fn tile_mode(index: u8) -> TileMode {
    use ArrayMode::*;
    use MicroType::*;
    let (array, micro) = match index {
        0..=4 | 6 | 7 => (Tiled2D, Depth),
        5 => (Tiled1D, Depth),
        8 | 31 => (Linear, Display),
        9 => (Tiled1D, Display),
        10 | 11 | 12 => (Tiled2D, Display),
        13 => (Tiled1D, Thin),
        14..=18 => (Tiled2D, Thin),
        19 => (Tiled1D, Thin),
        _ => (Tiled2D, Thin),
    };
    TileMode { array, micro }
}

#[inline]
fn bit(v: u32, b: u32) -> u32 {
    (v >> b) & 1
}

/// Index of element (x, y) within its 8x8 micro tile.
pub fn micro_index(x: u32, y: u32, bpp: u32, micro: MicroType) -> u32 {
    let (x0, x1, x2) = (bit(x, 0), bit(x, 1), bit(x, 2));
    let (y0, y1, y2) = (bit(y, 0), bit(y, 1), bit(y, 2));
    let b: [u32; 6] = match micro {
        MicroType::Display => match bpp {
            8 => [x0, x1, x2, y1, y0, y2],
            16 => [x0, x1, x2, y0, y1, y2],
            32 => [x0, x1, y0, x2, y1, y2],
            64 => [x0, y0, x1, x2, y1, y2],
            _ => [y0, x0, x1, x2, y1, y2],
        },
        _ => [x0, y0, x1, y1, x2, y2],
    };
    b[0] | (b[1] << 1) | (b[2] << 2) | (b[3] << 3) | (b[4] << 4) | (b[5] << 5)
}

#[derive(Clone, Copy, Debug)]
pub struct MacroParams {
    pub num_pipes: u32,
    pub num_banks: u32,
    pub bank_width: u32,
    pub bank_height: u32,
    pub macro_aspect: u32,
    pub pipe_interleave: u32,
}

/// Macro-tile parameters per element size for the PS4 (8 pipes, P8_32x32_16x16).
pub fn ps4_macro_params(bpp: u32) -> MacroParams {
    let (bw, bh, aspect, banks) = match bpp {
        8 => (1, 4, 4, 16),
        16 => (1, 2, 4, 16),
        32 => (1, 1, 2, 16),
        64 => (1, 1, 2, 8),
        _ => (1, 1, 1, 8),
    };
    MacroParams { num_pipes: 8, num_banks: banks, bank_width: bw, bank_height: bh, macro_aspect: aspect, pipe_interleave: 256 }
}

fn pipe_p8_32x32_16x16(x: u32, y: u32) -> u32 {
    let p0 = bit(x, 3) ^ bit(y, 3) ^ bit(x, 4);
    let p1 = bit(x, 4) ^ bit(y, 4);
    let p2 = bit(x, 5) ^ bit(y, 5);
    p0 | (p1 << 1) | (p2 << 2)
}

fn bank(x: u32, y: u32, p: &MacroParams) -> u32 {
    let tx = x / 8 / (p.bank_width * p.num_pipes);
    let ty = y / 8 / p.bank_height;
    let (x3, x4, x5, x6) = (bit(tx, 0), bit(tx, 1), bit(tx, 2), bit(tx, 3));
    let (y3, y4, y5, y6) = (bit(ty, 0), bit(ty, 1), bit(ty, 2), bit(ty, 3));
    match p.num_banks {
        16 => (x3 ^ y6) | ((x4 ^ y5 ^ y6) << 1) | ((x5 ^ y4) << 2) | ((x6 ^ y3) << 3),
        8 => (x3 ^ y5) | ((x4 ^ y4 ^ y5) << 1) | ((x5 ^ y3) << 2),
        4 => (x3 ^ y4) | ((x4 ^ y3) << 1),
        2 => x3 ^ y3,
        _ => 0,
    }
}

/// Byte offset of element (x, y) in slice 0 of a 2D-thin surface.
pub fn macro_offset(x: u32, y: u32, pitch: u32, height: u32, bpp: u32, micro: MicroType, p: &MacroParams) -> u64 {
    let micro_bytes = 64 * bpp / 8;
    let mt_pitch = 8 * p.bank_width * p.num_pipes * p.macro_aspect;
    let mt_height = 8 * p.bank_height * p.num_banks / p.macro_aspect;
    let mt_bytes = (mt_pitch * mt_height * bpp / 8) as u64;
    let mt_per_row = pitch.div_ceil(mt_pitch).max(1);
    let mt_x = x / mt_pitch;
    let mt_y = y / mt_height;
    let mt_offset = (mt_y as u64 * mt_per_row as u64 + mt_x as u64) * mt_bytes;
    let tile_row = (y / 8) % p.bank_height;
    let tile_col = ((x / 8) / p.num_pipes) % p.bank_width;
    let tile_offset = ((tile_row * p.bank_width + tile_col) * micro_bytes) as u64;
    let elem_offset = (micro_index(x, y, bpp, micro) * bpp / 8) as u64;
    let pipe_bits = p.num_pipes.trailing_zeros();
    let bank_bits = p.num_banks.trailing_zeros();
    let group_bits = p.pipe_interleave.trailing_zeros();
    let total = elem_offset + tile_offset + (mt_offset >> (pipe_bits + bank_bits));
    let pipe = pipe_p8_32x32_16x16(x, y) & (p.num_pipes - 1);
    let bk = bank(x, y, p);
    let group_mask = (1u64 << group_bits) - 1;
    let _ = height;
    ((total & !group_mask) << (pipe_bits + bank_bits)) | ((bk as u64) << (group_bits + pipe_bits)) | ((pipe as u64) << group_bits) | (total & group_mask)
}

/// Byte offset of element (x, y) in slice 0 of a 1D-thin surface.
pub fn micro_offset(x: u32, y: u32, pitch: u32, bpp: u32, micro: MicroType) -> u64 {
    let micro_bytes = (64 * bpp / 8) as u64;
    let tiles_per_row = (pitch / 8).max(1) as u64;
    let tile = (y / 8) as u64 * tiles_per_row + (x / 8) as u64;
    tile * micro_bytes + (micro_index(x, y, bpp, micro) * bpp / 8) as u64
}

/// Surface pitch and padded height in elements. `pitch` is the descriptor's
/// PITCH (already aligned by the title's Gnm library and used as-is by the
/// texture unit for addressing); only the height is padded to the tile
/// height, which determines the slice size.
pub fn padded(pitch: u32, height: u32, bpp: u32, mode: TileMode) -> (u32, u32) {
    match mode.array {
        ArrayMode::Linear => (pitch, height),
        ArrayMode::Tiled1D => (pitch, height.next_multiple_of(8)),
        ArrayMode::Tiled2D => {
            let p = ps4_macro_params(bpp);
            let mt_height = 8 * p.bank_height * p.num_banks / p.macro_aspect;
            (pitch, height.next_multiple_of(mt_height))
        }
    }
}

/// Byte offsets of the 64 elements of the 8x8 micro tile whose top-left
/// element is (x0, y0) (both multiples of 8). Within a micro tile the pipe
/// and bank bits are constant, so the macro-tiled address is computed once
/// per tile instead of once per element.
#[allow(clippy::too_many_arguments)]
fn micro_tile_offsets(x0: u32, y0: u32, pitch: u32, bpp: u32, mode: TileMode, p: &MacroParams, lut: &[u32; 64], out: &mut [u64; 64]) {
    let eb = (bpp / 8) as u64;
    if mode.array == ArrayMode::Tiled1D {
        let base = micro_offset(x0, y0, pitch, bpp, mode.micro);
        for (i, o) in out.iter_mut().enumerate() {
            *o = base + lut[i] as u64 * eb;
        }
        return;
    }
    let pipe_bits = p.num_pipes.trailing_zeros();
    let bank_bits = p.num_banks.trailing_zeros();
    let group_bits = p.pipe_interleave.trailing_zeros();
    let group_mask = (1u64 << group_bits) - 1;
    // Same terms as `macro_offset` for the tile's first element (micro index 0).
    let micro_bytes = 64 * bpp / 8;
    let mt_pitch = 8 * p.bank_width * p.num_pipes * p.macro_aspect;
    let mt_height = 8 * p.bank_height * p.num_banks / p.macro_aspect;
    let mt_bytes = (mt_pitch * mt_height * bpp / 8) as u64;
    let mt_per_row = pitch.div_ceil(mt_pitch).max(1);
    let mt_offset = ((y0 / mt_height) as u64 * mt_per_row as u64 + (x0 / mt_pitch) as u64) * mt_bytes;
    let tile_row = (y0 / 8) % p.bank_height;
    let tile_col = ((x0 / 8) / p.num_pipes) % p.bank_width;
    let tile_total = ((tile_row * p.bank_width + tile_col) * micro_bytes) as u64 + (mt_offset >> (pipe_bits + bank_bits));
    let pipe = (pipe_p8_32x32_16x16(x0, y0) & (p.num_pipes - 1)) as u64;
    let swz = ((bank(x0, y0, p) as u64) << (group_bits + pipe_bits)) | (pipe << group_bits);
    for (i, o) in out.iter_mut().enumerate() {
        let total = tile_total + lut[i] as u64 * eb;
        *o = ((total & !group_mask) << (pipe_bits + bank_bits)) | swz | (total & group_mask);
    }
}

/// Calls `f(tiled_offset, linear_offset)` for every element of a tiled
/// (1D or 2D) slice of `width` x `height` elements, a micro tile at a time.
fn for_each_tiled(width: u32, height: u32, pitch: u32, bpp: u32, mode: TileMode, mut f: impl FnMut(usize, usize)) {
    let eb = (bpp / 8) as usize;
    let p = ps4_macro_params(bpp);
    let mut lut = [0u32; 64];
    for (i, l) in lut.iter_mut().enumerate() {
        *l = micro_index(i as u32 % 8, i as u32 / 8, bpp, mode.micro);
    }
    let mut offs = [0u64; 64];
    for ty in (0..height).step_by(8) {
        for tx in (0..width).step_by(8) {
            micro_tile_offsets(tx, ty, pitch, bpp, mode, &p, &lut, &mut offs);
            for ly in 0..8u32.min(height - ty) {
                for lx in 0..8u32.min(width - tx) {
                    let l = ((ty + ly) as usize * width as usize + (tx + lx) as usize) * eb;
                    f(offs[(ly * 8 + lx) as usize] as usize, l);
                }
            }
        }
    }
}

/// For each element of a tiled slice (`slice_elems` elements, surface order):
/// the index of the linear element stored there, or `none` for padding.
pub fn linear_index_map(width: u32, height: u32, pitch: u32, bpp: u32, mode: TileMode, slice_elems: usize, none: u32) -> Vec<u32> {
    let eb = (bpp / 8) as usize;
    let mut map = vec![none; slice_elems];
    match mode.array {
        ArrayMode::Linear => {
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let t = y * pitch as usize + x;
                    if t < slice_elems {
                        map[t] = (y * width as usize + x) as u32;
                    }
                }
            }
        }
        ArrayMode::Tiled1D | ArrayMode::Tiled2D => {
            for_each_tiled(width, height, pitch, bpp, mode, |t, l| {
                if t / eb < slice_elems {
                    map[t / eb] = (l / eb) as u32;
                }
            });
        }
    }
    map
}

/// Detiles one slice of `width` x `height` elements of `bpp` bits from `src`
/// (surface pitch/height in elements) into a tightly packed linear buffer.
#[allow(clippy::too_many_arguments)]
pub fn detile_slice(src: &[u8], dst: &mut [u8], width: u32, height: u32, pitch: u32, pheight: u32, bpp: u32, mode: TileMode) {
    let eb = (bpp / 8) as usize;
    match mode.array {
        ArrayMode::Linear => {
            for y in 0..height as usize {
                let s = y * pitch as usize * eb;
                let d = y * width as usize * eb;
                let n = width as usize * eb;
                if s + n <= src.len() && d + n <= dst.len() {
                    dst[d..d + n].copy_from_slice(&src[s..s + n]);
                }
            }
        }
        ArrayMode::Tiled1D | ArrayMode::Tiled2D => {
            let _ = pheight;
            for_each_tiled(width, height, pitch, bpp, mode, |t, l| {
                if t + eb <= src.len() && l + eb <= dst.len() {
                    dst[l..l + eb].copy_from_slice(&src[t..t + eb]);
                }
            });
        }
    }
}

/// Tiles a tightly packed linear slice of `width` x `height` elements into
/// `dst` (surface pitch/height in elements); inverse of `detile_slice`.
#[allow(clippy::too_many_arguments)]
pub fn tile_slice(src: &[u8], dst: &mut [u8], width: u32, height: u32, pitch: u32, pheight: u32, bpp: u32, mode: TileMode) {
    let eb = (bpp / 8) as usize;
    let _ = pheight;
    if mode.array == ArrayMode::Linear {
        for y in 0..height as usize {
            let d = y * pitch as usize * eb;
            let s = y * width as usize * eb;
            let n = width as usize * eb;
            if d + n <= dst.len() && s + n <= src.len() {
                dst[d..d + n].copy_from_slice(&src[s..s + n]);
            }
        }
        return;
    }
    for_each_tiled(width, height, pitch, bpp, mode, |t, l| {
        if t + eb <= dst.len() && l + eb <= src.len() {
            dst[t..t + eb].copy_from_slice(&src[l..l + eb]);
        }
    });
}

/// Slices per micro tile of a Gnm tile mode: thick modes (19..=24: 1D/2D/3D
/// thick and their PRT variants) interleave 4 slices in each 8x8 micro tile,
/// XThick (25, 26) 8; all others 1. Volume textures (e.g. colour-grading
/// LUTs) use them.
pub fn thickness(index: u8) -> u32 {
    match index {
        19..=24 => 4,
        25 | 26 => 8,
        _ => 1,
    }
}

/// Index of element (x, y, z) (each within the tile: x, y < 8, z <
/// thickness) in a thick micro tile, in the AMD address library's order
/// for thick micro tiling.
pub fn thick_micro_index(x: u32, y: u32, z: u32, bpp: u32, thickness: u32) -> u32 {
    let (x0, x1, x2) = (bit(x, 0), bit(x, 1), bit(x, 2));
    let (y0, y1, y2) = (bit(y, 0), bit(y, 1), bit(y, 2));
    let (z0, z1, z2) = (bit(z, 0), bit(z, 1), bit(z, 2));
    let b: [u32; 6] = match bpp {
        8 | 16 => [x0, y0, x1, y1, z0, z1],
        32 => [x0, y0, x1, z0, y1, z1],
        _ => [x0, y0, z0, x1, y1, z1],
    };
    let mut v = b[0] | (b[1] << 1) | (b[2] << 2) | (b[3] << 3) | (b[4] << 4) | (b[5] << 5) | (x2 << 6) | (y2 << 7);
    if thickness == 8 {
        v |= z2 << 8;
    }
    v
}

/// Byte offset of element (x, y, z) of a 1D-thick surface (micro tiles of
/// 8x8xthickness elements in row order; each group of `thickness` slices is
/// `pitch * pheight * thickness` elements).
pub fn thick_1d_offset(x: u32, y: u32, z: u32, pitch: u32, pheight: u32, bpp: u32, thickness: u32) -> u64 {
    let eb = (bpp / 8) as u64;
    let mt_bytes = 64 * thickness as u64 * eb;
    let tiles_per_row = (pitch / 8).max(1) as u64;
    let group_bytes = pitch as u64 * pheight as u64 * thickness as u64 * eb;
    (z / thickness) as u64 * group_bytes
        + ((y / 8) as u64 * tiles_per_row + (x / 8) as u64) * mt_bytes
        + thick_micro_index(x % 8, y % 8, z % thickness, bpp, thickness) as u64 * eb
}

/// Detiles a 1D-thick volume of `depth` slices from `src` (the surface from
/// its first slice) into tightly packed slices of `width` x `height`.
#[allow(clippy::too_many_arguments)]
pub fn detile_thick_1d(src: &[u8], dst: &mut [u8], width: u32, height: u32, depth: u32, pitch: u32, pheight: u32, bpp: u32, thickness: u32) {
    let eb = (bpp / 8) as usize;
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let t = thick_1d_offset(x, y, z, pitch, pheight, bpp, thickness) as usize;
                let l = ((z as usize * height as usize + y as usize) * width as usize + x as usize) * eb;
                if t + eb <= src.len() && l + eb <= dst.len() {
                    dst[l..l + eb].copy_from_slice(&src[t..t + eb]);
                }
            }
        }
    }
}

/// For each element of a 1D-thick volume's surface (`elems` elements from
/// its first slice): the index of the linear element (packed slices of
/// `width` x `height`) stored there, or `none` for padding - the GPU
/// detiler's map (`kernels::remap_cs`), the inverse of `thick_1d_offset`.
#[allow(clippy::too_many_arguments)]
pub fn thick_1d_index_map(width: u32, height: u32, depth: u32, pitch: u32, pheight: u32, bpp: u32, thickness: u32, elems: usize, none: u32) -> Vec<u32> {
    let eb = (bpp / 8) as u64;
    let mut map = vec![none; elems];
    let mut l = 0u32;
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let t = (thick_1d_offset(x, y, z, pitch, pheight, bpp, thickness) / eb) as usize;
                if t < elems {
                    map[t] = l;
                }
                l += 1;
            }
        }
    }
    map
}

/// Size in bytes of one slice of a surface with the given padded dimensions.
pub fn slice_bytes(pitch: u32, pheight: u32, bpp: u32) -> u64 {
    pitch as u64 * pheight as u64 * (bpp / 8) as u64
}

/// Macro tile width and height in elements (2D tiling) for an element size.
pub fn macro_tile_dims(bpp: u32) -> (u32, u32) {
    let p = ps4_macro_params(bpp);
    (8 * p.bank_width * p.num_pipes * p.macro_aspect, 8 * p.bank_height * p.num_banks / p.macro_aspect)
}

/// One mip level of a texture in guest memory (dimensions in elements).
#[derive(Clone, Copy, Debug)]
pub struct MipLevel {
    /// Byte offset of the level's first slice from the texture's base.
    pub offset: u64,
    /// The level's elements and its surface's (padded) pitch and height.
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub pheight: u32,
    pub mode: TileMode,
    pub slice_bytes: u64,
    /// Slices allocated for the level (all of them follow each other).
    pub slices: u32,
}

/// Mip chain layout (levels 0..`levels`) of a tiled 2D/cube/array texture
/// of `width` x `height` texels with descriptor PITCH `pitch` texels,
/// `block` x `block`-texel elements of `bpp` bits and `slices` layers, by
/// the GCN texture addressing rules (AMD address library, SI/CI):
/// - level 0 is the descriptor's surface; POW2_PAD pads its height to a
///   power of two;
/// - level l > 0 allocates power-of-two dimensions: max(1, pitch >> l) and
///   max(1, height >> l) rounded up;
/// - every level allocates a power-of-two slice count with POW2_PAD (cube
///   maps too: 6 faces take 8), without it levels after the first do
///   except cube maps;
/// - a macro-tiled (2D) level smaller than one macro tile in either
///   dimension, and every level after it, is tiled 1D (same micro tiling);
/// - levels follow each other from level 0, each with all of its slices.
#[allow(clippy::too_many_arguments)]
pub fn mip_levels(width: u32, height: u32, pitch: u32, block: u32, bpp: u32, tiling_index: u8, pow2_pad: bool, levels: u32, slices: u32, cube: bool) -> Vec<MipLevel> {
    let mut out = Vec::with_capacity(levels.max(1) as usize);
    for_each_mip_level(width, height, pitch, block, bpp, tiling_index, pow2_pad, levels, slices, cube, |l| out.push(l));
    out
}

/// `mip_levels` without the vector (per draw: chain sizes for validation).
#[allow(clippy::too_many_arguments)]
pub fn for_each_mip_level(width: u32, height: u32, pitch: u32, block: u32, bpp: u32, tiling_index: u8, pow2_pad: bool, levels: u32, slices: u32, cube: bool, mut f: impl FnMut(MipLevel)) {
    let (mtw, mth) = macro_tile_dims(bpp);
    let elems = |px: u32| px.div_ceil(block).max(1);
    let base_pitch = pitch.max(width);
    let mut mode = tile_mode(tiling_index);
    let mut offset = 0u64;
    for l in 0..levels.max(1) {
        let (pitch_e, ph_e, n) = if l == 0 {
            let h = if pow2_pad { height.next_power_of_two() } else { height };
            let (p, ph) = padded(elems(base_pitch), elems(h), bpp, mode);
            (p, ph, if pow2_pad { slices.max(1).next_power_of_two() } else { slices.max(1) })
        } else {
            let pw = elems((base_pitch >> l).max(1).next_power_of_two());
            let ph = elems((height >> l).max(1).next_power_of_two());
            if mode.array == ArrayMode::Tiled2D && (pw < mtw || ph < mth) {
                mode = TileMode { array: ArrayMode::Tiled1D, micro: mode.micro };
            }
            let (p, h) = match mode.array {
                ArrayMode::Tiled2D => (pw.next_multiple_of(mtw), ph.next_multiple_of(mth)),
                _ => (pw.next_multiple_of(8), ph.next_multiple_of(8)),
            };
            (p, h, if cube && !pow2_pad { slices.max(1) } else { slices.max(1).next_power_of_two() })
        };
        let sb = slice_bytes(pitch_e, ph_e, bpp);
        f(MipLevel { offset, width: elems((width >> l).max(1)), height: elems((height >> l).max(1)), pitch: pitch_e, pheight: ph_e, mode, slice_bytes: sb, slices: n });
        offset += sb * n as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn micro_index_is_a_permutation() {
        for &bpp in &[8, 16, 32, 64, 128] {
            for m in [MicroType::Display, MicroType::Thin] {
                let mut seen = [false; 64];
                for y in 0..8 {
                    for x in 0..8 {
                        let i = micro_index(x, y, bpp, m) as usize;
                        assert!(!seen[i]);
                        seen[i] = true;
                    }
                }
            }
        }
    }
    #[test]
    fn thick_micro_index_is_a_permutation() {
        for &bpp in &[8, 16, 32, 64, 128] {
            for &th in &[4u32, 8] {
                let n = 64 * th as usize;
                let mut seen = vec![false; n];
                for z in 0..th {
                    for y in 0..8 {
                        for x in 0..8 {
                            let i = thick_micro_index(x, y, z, bpp, th) as usize;
                            assert!(i < n && !seen[i], "bpp {bpp} thickness {th} at {x},{y},{z}");
                            seen[i] = true;
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn thick_volume_detile_inverts_offsets() {
        // A 16x16x8 RGBA8 volume, 1D thick: each element's tiled offset holds its index.
        let (w, h, d, bpp) = (16u32, 16u32, 8u32, 32u32);
        let mut tiled = vec![0u8; (w * h * d * 4) as usize];
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    let o = thick_1d_offset(x, y, z, w, h, bpp, 4) as usize;
                    tiled[o..o + 4].copy_from_slice(&((z * h + y) * w + x).to_le_bytes());
                }
            }
        }
        let mut lin = vec![0u8; tiled.len()];
        detile_thick_1d(&tiled, &mut lin, w, h, d, w, h, bpp, 4);
        for (i, c) in lin.chunks(4).enumerate() {
            assert_eq!(u32::from_le_bytes(c.try_into().unwrap()), i as u32);
        }
        // Micro tiles are 8x8x4 elements: (0,0,1) is the 9th element (x0 y0 x1 z0 order).
        assert_eq!(thick_1d_offset(0, 0, 1, w, h, bpp, 4), 8 * 4);
        assert_eq!(thick_1d_offset(8, 0, 0, w, h, bpp, 4), 256 * 4);
        assert_eq!(thick_1d_offset(0, 0, 4, w, h, bpp, 4), (16 * 16 * 4 * 4) as u64);
    }
    #[test]
    fn thick_index_map_matches_cpu_detile() {
        // First Antenna's light volumes in small: 36x36 RGBA16F (pitch and
        // height padded to 40), a depth that leaves padding slices.
        for (bpp, d) in [(64u32, 13u32), (32, 6), (128, 4)] {
            let (w, h, pitch, ph, th) = (36u32, 36u32, 40u32, 40u32, 4u32);
            let eb = (bpp / 8) as usize;
            let total = slice_bytes(pitch, ph, bpp) as usize * th as usize * d.div_ceil(th) as usize;
            let mut seed = 0x1234_5678u32;
            let tiled: Vec<u8> = (0..total)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (seed >> 24) as u8
                })
                .collect();
            let mut cpu = vec![0u8; (w * h * d) as usize * eb];
            detile_thick_1d(&tiled, &mut cpu, w, h, d, pitch, ph, bpp, th);
            // The remap kernel: linear element map[t] takes surface element t.
            let map = thick_1d_index_map(w, h, d, pitch, ph, bpp, th, total / eb, u32::MAX);
            let mut gpu = vec![0u8; cpu.len()];
            for (t, &l) in map.iter().enumerate() {
                if l != u32::MAX {
                    gpu[l as usize * eb..(l as usize + 1) * eb].copy_from_slice(&tiled[t * eb..(t + 1) * eb]);
                }
            }
            assert_eq!(gpu, cpu, "bpp {bpp} depth {d}");
            assert_eq!(map.iter().filter(|&&l| l != u32::MAX).count(), (w * h * d) as usize);
        }
    }
    #[test]
    fn fast_offsets_match_reference() {
        for (idx, bpp) in [(14u8, 32u32), (14, 64), (14, 8), (14, 128), (0, 32), (10, 32), (13, 32), (5, 16)] {
            let mode = tile_mode(idx);
            let p = ps4_macro_params(bpp);
            let mut lut = [0u32; 64];
            for (i, l) in lut.iter_mut().enumerate() {
                *l = micro_index(i as u32 % 8, i as u32 / 8, bpp, mode.micro);
            }
            let (pitch, ph) = padded(256, 128, bpp, mode);
            let mut offs = [0u64; 64];
            for ty in (0..ph).step_by(8) {
                for tx in (0..pitch).step_by(8) {
                    micro_tile_offsets(tx, ty, pitch, bpp, mode, &p, &lut, &mut offs);
                    for i in 0..64u32 {
                        let (x, y) = (tx + i % 8, ty + i / 8);
                        let r = match mode.array {
                            ArrayMode::Tiled1D => micro_offset(x, y, pitch, bpp, mode.micro),
                            _ => macro_offset(x, y, pitch, ph, bpp, mode.micro, &p),
                        };
                        assert_eq!(offs[i as usize], r, "mode {idx} bpp {bpp} at {x},{y}");
                    }
                }
            }
        }
    }
    #[test]
    fn tile_round_trip() {
        for idx in [0u8, 5, 8, 13, 14] {
            let mode = tile_mode(idx);
            let (w, h) = (96, 40);
            let (pitch, ph) = padded(128, h, 32, mode);
            let lin: Vec<u8> = (0..w * h * 4).map(|i| (i * 7 + idx as u32) as u8).collect();
            let mut tiled = vec![0u8; slice_bytes(pitch, ph, 32) as usize];
            tile_slice(&lin, &mut tiled, w, h, pitch, ph, 32, mode);
            let mut back = vec![0u8; lin.len()];
            detile_slice(&tiled, &mut back, w, h, pitch, ph, 32, mode);
            assert_eq!(lin, back, "tile mode {idx}");
        }
    }
    /// The element map used by the GPU (de)tiling kernels reproduces the CPU
    /// detile/tile byte for byte (render target layouts: pitch == width).
    #[test]
    fn linear_index_map_matches_cpu_paths() {
        const NONE: u32 = u32::MAX;
        // (w, requested pitch): textures may have a pitch wider than the
        // padded width (e.g. 1920 wide, pitch 2048).
        for (idx, bpp, (w, pw)) in [0u8, 2, 5, 8, 10, 13, 14].into_iter().flat_map(|i| [32u32, 64, 128].into_iter().flat_map(move |b| [(i, b, (256u32, 256u32)), (i, b, (200, 200)), (i, b, (200, 384))])) {
            {
                let mode = tile_mode(idx);
                let h = 72;
                let (pitch, ph) = padded(pw, h, bpp, mode);
                let eb = (bpp / 8) as usize;
                let slice = slice_bytes(pitch, ph, bpp) as usize;
                let tiled: Vec<u8> = (0..slice).map(|i| (i * 13 + i / 7) as u8).collect();
                let map = linear_index_map(w, h, pitch, bpp, mode, slice / eb, NONE);
                // Detile: lin[map[t]] = tiled[t].
                let mut cpu = vec![0u8; (w * h) as usize * eb];
                detile_slice(&tiled, &mut cpu, w, h, pitch, ph, bpp, mode);
                let mut gpu = vec![0u8; cpu.len()];
                let mut covered = 0;
                for (t, &e) in map.iter().enumerate() {
                    if e != NONE {
                        let e = e as usize;
                        gpu[e * eb..(e + 1) * eb].copy_from_slice(&tiled[t * eb..(t + 1) * eb]);
                        covered += 1;
                    }
                }
                // (Pitches that are not a whole number of macro tiles put
                // some elements past the slice; both paths drop them.)
                if w == 256 {
                    assert_eq!(covered, (w * h) as usize, "mode {idx} bpp {bpp}: every element mapped once");
                }
                assert!(cpu == gpu, "detile mismatch: mode {idx} bpp {bpp}");
                // Tile: tiled[t] = lin[map[t]] (padding untouched).
                let lin: Vec<u8> = (0..cpu.len()).map(|i| (i * 7 + 3) as u8).collect();
                let mut cpu_t = tiled.clone();
                tile_slice(&lin, &mut cpu_t, w, h, pitch, ph, bpp, mode);
                let mut gpu_t = tiled.clone();
                for (t, &e) in map.iter().enumerate() {
                    if e != NONE {
                        let e = e as usize;
                        gpu_t[t * eb..(t + 1) * eb].copy_from_slice(&lin[e * eb..(e + 1) * eb]);
                    }
                }
                assert!(cpu_t == gpu_t, "tile mismatch: mode {idx} bpp {bpp}");
            }
        }
    }
    #[test]
    fn mip_chain_layouts() {
        // BC6H cube map, 1D thin, POW2_PAD (as the title's reflection probes):
        // 8 slices per level (faces 6 and 7 are zero padding in guest memory).
        let l = mip_levels(128, 128, 128, 4, 128, 13, true, 8, 6, true);
        let offs: Vec<u64> = l.iter().map(|l| l.offset).collect();
        assert_eq!(&offs[..4], &[0, 8 * 16384, 8 * 16384 + 8 * 4096, 8 * 16384 + 8 * 4096 + 8 * 1024]);
        assert!(l.iter().all(|l| l.slices == 8));
        assert_eq!((l[1].width, l[1].pitch, l[3].width, l[3].pitch, l[3].pheight), (16, 16, 4, 8, 8));
        // BC1 2D thin: macro tiles are 128x32 elements; levels under that
        // switch to 1D tiling.
        let l = mip_levels(2048, 2048, 2048, 4, 64, 14, true, 12, 1, false);
        let arrays: Vec<ArrayMode> = l.iter().take(5).map(|l| l.mode.array).collect();
        assert_eq!(arrays, [ArrayMode::Tiled2D, ArrayMode::Tiled2D, ArrayMode::Tiled2D, ArrayMode::Tiled1D, ArrayMode::Tiled1D]);
        assert_eq!(&l.iter().take(4).map(|l| l.offset).collect::<Vec<_>>()[..], &[0, 2 << 20, (2 << 20) + (512 << 10), (2 << 20) + (640 << 10)]);
        // A lone base level is the descriptor's surface.
        let l = mip_levels(100, 60, 128, 1, 32, 13, false, 1, 3, false);
        assert_eq!((l.len(), l[0].pitch, l[0].pheight, l[0].slices), (1, 128, 64, 3));
    }
    #[test]
    fn macro_offsets_are_unique() {
        let (w, h) = padded(512, 256, 32, tile_mode(14));
        let p = ps4_macro_params(32);
        let mut seen = std::collections::HashSet::new();
        for y in 0..h {
            for x in 0..w {
                let o = macro_offset(x, y, w, h, 32, MicroType::Thin, &p);
                assert!(o < (w * h * 4) as u64, "offset {o} out of range at {x},{y}");
                assert!(seen.insert(o), "duplicate offset at {x},{y}");
            }
        }
    }
}
