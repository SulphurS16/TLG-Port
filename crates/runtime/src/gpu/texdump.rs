//! Debug aid: writes the first uploaded textures as PNG files (SOTC_DUMP_TEX=N).

use super::desc::Tsharp;
use std::sync::atomic::{AtomicU32, Ordering};

static DUMPED: AtomicU32 = AtomicU32::new(0);

fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
    }
    !c
}

/// PNG encoder (RGBA8, zlib-compressed).
pub fn png(w: u32, h: u32, rgba: &[u8]) -> Vec<u8> {
    let mut raw = Vec::with_capacity((w * 4 + 1) as usize * h as usize);
    for y in 0..h as usize {
        raw.push(0);
        raw.extend_from_slice(&rgba[y * w as usize * 4..(y + 1) * w as usize * 4]);
    }
    let z = miniz_oxide::deflate::compress_to_vec_zlib(&raw, 3);
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut chunk = |ty: &[u8; 4], data: &[u8]| {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let mut c = ty.to_vec();
        c.extend_from_slice(data);
        out.extend_from_slice(&c);
        out.extend_from_slice(&crc32(&c).to_be_bytes());
    };
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
    chunk(b"IHDR", &ihdr);
    chunk(b"IDAT", &z);
    chunk(b"IEND", &[]);
    out
}

fn rgb565(c: u16) -> [u8; 3] {
    let r = ((c >> 11) & 31) as u32;
    let g = ((c >> 5) & 63) as u32;
    let b = (c & 31) as u32;
    [(r * 255 / 31) as u8, (g * 255 / 63) as u8, (b * 255 / 31) as u8]
}

fn bc1_block(b: &[u8], out: &mut [[u8; 4]; 16], alpha_mode: bool) {
    let c0 = u16::from_le_bytes([b[0], b[1]]);
    let c1 = u16::from_le_bytes([b[2], b[3]]);
    let (p0, p1) = (rgb565(c0), rgb565(c1));
    let mut pal = [[p0[0], p0[1], p0[2], 255], [p1[0], p1[1], p1[2], 255], [0; 4], [0; 4]];
    if c0 > c1 || alpha_mode {
        for k in 0..3 {
            pal[2][k] = ((2 * p0[k] as u32 + p1[k] as u32) / 3) as u8;
            pal[3][k] = ((p0[k] as u32 + 2 * p1[k] as u32) / 3) as u8;
        }
        pal[2][3] = 255;
        pal[3][3] = 255;
    } else {
        for k in 0..3 {
            pal[2][k] = ((p0[k] as u32 + p1[k] as u32) / 2) as u8;
        }
        pal[2][3] = 255;
        pal[3] = [0, 0, 0, 0];
    }
    let idx = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    for (i, px) in out.iter_mut().enumerate() {
        *px = pal[((idx >> (2 * i)) & 3) as usize];
    }
}

fn bc4_alpha(b: &[u8]) -> [u8; 16] {
    let (a0, a1) = (b[0] as u32, b[1] as u32);
    let mut pal = [a0, a1, 0, 0, 0, 0, 0, 0];
    if a0 > a1 {
        for i in 1..7 {
            pal[i + 1] = ((7 - i) as u32 * a0 + i as u32 * a1) / 7;
        }
    } else {
        for i in 1..5 {
            pal[i + 1] = ((5 - i) as u32 * a0 + i as u32 * a1) / 5;
        }
        pal[6] = 0;
        pal[7] = 255;
    }
    let mut bits = 0u64;
    for k in 0..6 {
        bits |= (b[2 + k] as u64) << (8 * k);
    }
    let mut out = [0u8; 16];
    for (i, o) in out.iter_mut().enumerate() {
        *o = pal[((bits >> (3 * i)) & 7) as usize] as u8;
    }
    out
}

/// Converts linear texture data to RGBA8 for viewing.
pub fn to_rgba(t: &Tsharp, data: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    let (w, h) = (t.width, t.height);
    let mut out = vec![0u8; (w * h * 4) as usize];
    match t.dfmt {
        10 => out.copy_from_slice(&data[..(w * h * 4) as usize]),
        1 => {
            for i in 0..(w * h) as usize {
                out[i * 4..i * 4 + 4].copy_from_slice(&[data[i], data[i], data[i], 255]);
            }
        }
        3 => {
            for i in 0..(w * h) as usize {
                out[i * 4..i * 4 + 4].copy_from_slice(&[data[i * 2], data[i * 2 + 1], 0, 255]);
            }
        }
        4 if t.nfmt == 7 => {
            for i in 0..(w * h) as usize {
                let f = f32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap());
                let v = (f.clamp(0.0, 1.0) * 255.0) as u8;
                out[i * 4..i * 4 + 4].copy_from_slice(&[v, v, v, 255]);
            }
        }
        35 | 37 | 38 | 39 => {
            let bw = w.div_ceil(4);
            let bh = h.div_ceil(4);
            let bsz = if t.dfmt == 35 || t.dfmt == 38 { 8 } else { 16 };
            for by in 0..bh {
                for bx in 0..bw {
                    let o = ((by * bw + bx) * bsz) as usize;
                    if o + bsz as usize > data.len() {
                        continue;
                    }
                    let b = &data[o..o + bsz as usize];
                    let mut px = [[0u8; 4]; 16];
                    match t.dfmt {
                        35 => bc1_block(b, &mut px, false),
                        37 => {
                            bc1_block(&b[8..], &mut px, true);
                            let a = bc4_alpha(&b[..8]);
                            for i in 0..16 {
                                px[i][3] = a[i];
                            }
                        }
                        38 => {
                            let a = bc4_alpha(b);
                            for i in 0..16 {
                                px[i] = [a[i], a[i], a[i], 255];
                            }
                        }
                        _ => {
                            let r = bc4_alpha(&b[..8]);
                            let g = bc4_alpha(&b[8..]);
                            for i in 0..16 {
                                px[i] = [r[i], g[i], 0, 255];
                            }
                        }
                    }
                    for i in 0..16u32 {
                        let (x, y) = (bx * 4 + i % 4, by * 4 + i / 4);
                        if x < w && y < h {
                            let d = ((y * w + x) * 4) as usize;
                            out[d..d + 4].copy_from_slice(&px[i as usize]);
                        }
                    }
                }
            }
        }
        _ => return None,
    }
    Some((w, h, out))
}

pub fn maybe_dump(t: &Tsharp, data: &[u8]) {
    let limit: u32 = std::env::var("SOTC_DUMP_TEX").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    if limit == 0 || DUMPED.load(Ordering::Relaxed) >= limit || t.width < 16 || t.height < 16 {
        return;
    }
    let Some(dir) = super::shader_dump::DIR.get() else { return };
    let Some((w, h, rgba)) = to_rgba(t, data) else { return };
    let n = DUMPED.fetch_add(1, Ordering::Relaxed);
    let dir = dir.parent().unwrap().join("textures");
    let _ = std::fs::create_dir_all(&dir);
    let name = format!("tex{n:03}_{:x}_{}x{}_d{}_n{}_t{}.png", t.base, w, h, t.dfmt, t.nfmt, t.tiling_index);
    let _ = std::fs::write(dir.join(name), png(w, h, &rgba));
}

/// Debug: PNGs of the first `n` levels of a chain (`resources::read_texture`
/// data, first slice) as `textures/mip_<base>_l<level>.png`.
pub fn dump_levels(t: &Tsharp, data: &[u8], layers: u32, n: usize) {
    let Some(dir) = super::shader_dump::DIR.get() else { return };
    let dir = dir.parent().unwrap().join("textures");
    let _ = std::fs::create_dir_all(&dir);
    let lv = super::resources::texture_layout(t, layers);
    let (bytes, _) = super::desc::format_bytes(t.dfmt);
    let mut pos = 0usize;
    for (l, lev) in lv.iter().enumerate().take(n) {
        let mut tl = *t;
        tl.width = (t.width >> l).max(1);
        tl.height = (t.height >> l).max(1);
        if let Some((w, h, rgba)) = data.get(pos..).and_then(|d| to_rgba(&tl, d)) {
            let _ = std::fs::write(dir.join(format!("mip_{:x}_l{l}.png", t.base)), png(w, h, &rgba));
        }
        pos += lev.width as usize * lev.height as usize * bytes as usize * layers as usize;
    }
}

/// One scalar per element for comparing mip levels: the mean of a BC
/// block's colour endpoints (green of BC1-3, red of BC4/5), or a texel's
/// first channel; None for formats it does not read (BC6/7, others).
fn element_value(dfmt: u8, e: &[u8]) -> Option<f32> {
    let g565 = |v: u16| ((v >> 5) & 0x3F) as f32 / 63.0;
    Some(match dfmt {
        35 => (g565(u16::from_le_bytes([e[0], e[1]])) + g565(u16::from_le_bytes([e[2], e[3]]))) / 2.0,
        36 | 37 => (g565(u16::from_le_bytes([e[8], e[9]])) + g565(u16::from_le_bytes([e[10], e[11]]))) / 2.0,
        38 | 39 => (e[0] as f32 + e[1] as f32) / 510.0,
        1 | 3 | 10 => e[0] as f32 / 255.0,
        _ => return None,
    })
}

/// Debug (SOTC_MIP_CHECK): for each level of an uploaded mip chain (`data` as
/// `resources::read_texture` returns it), the correlation of its first slice
/// with the previous level's first slice averaged 2x2: near 1 when the
/// levels were read from where the hardware stores them, near 0 otherwise.
/// None for levels under 4x4 elements or unread formats.
pub fn mip_correlations(t: &Tsharp, data: &[u8], layers: u32) -> Vec<Option<f32>> {
    let lv = super::resources::texture_layout(t, layers);
    let (bytes, _) = super::desc::format_bytes(t.dfmt);
    let eb = bytes as usize;
    let mut starts = Vec::new();
    let mut pos = 0usize;
    for l in &lv {
        starts.push(pos);
        pos += l.width as usize * l.height as usize * eb * layers as usize;
    }
    let value = |l: usize, x: u32, y: u32| -> Option<f32> {
        let i = starts[l] + (y as usize * lv[l].width as usize + x as usize) * eb;
        element_value(t.dfmt, data.get(i..i + eb)?)
    };
    let mut out = Vec::new();
    for l in 1..lv.len() {
        let (w, h) = (lv[l].width.min(lv[l - 1].width / 2), lv[l].height.min(lv[l - 1].height / 2));
        if w < 4 || h < 4 {
            out.push(None);
            continue;
        }
        let (mut sa, mut sb, mut saa, mut sbb, mut sab, mut n) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
        let mut ok = true;
        'px: for y in 0..h {
            for x in 0..w {
                let (Some(a), Some(b0), Some(b1), Some(b2), Some(b3)) = (value(l, x, y), value(l - 1, 2 * x, 2 * y), value(l - 1, 2 * x + 1, 2 * y), value(l - 1, 2 * x, 2 * y + 1), value(l - 1, 2 * x + 1, 2 * y + 1)) else {
                    ok = false;
                    break 'px;
                };
                let (a, b) = (a as f64, ((b0 + b1 + b2 + b3) / 4.0) as f64);
                sa += a;
                sb += b;
                saa += a * a;
                sbb += b * b;
                sab += a * b;
                n += 1.0;
            }
        }
        let var = (saa - sa * sa / n) * (sbb - sb * sb / n);
        out.push((ok && var > 1e-12).then(|| ((sab - sa * sb / n) / var.sqrt()) as f32));
    }
    out
}
