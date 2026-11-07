//! The port's own game files, mounted by the game itself: the title looks
//! for an optional archive `/app0/trico_other.psarc` after its own (eboot
//! 0xf4852b, sceFiosFileExistsSync) and mounts it when it exists. The port
//! builds that archive in memory at start-up (`init`) and the VFS serves it
//! (`fs::add_virtual`); nothing is written to the game's folders. It holds
//! the textures of the in-game PC settings: the text atlas the port's
//! script draws its words from, filled at run time (`text`), and the label
//! of the Options list's "PC Settings" item.
//!
//! The archive has the form of the game's patch archives (PSARC 1.4, flags
//! 13: the manifest first under a zero hash, the other entries sorted by
//! the MD5 of their upper-case names, names NUL-separated, 64 KiB blocks -
//! here all stored, none compressed). Textures are GNF files: a 256-byte
//! header with the texture's descriptor (T#, its base address the game's),
//! then the texels - RGBA8 in tiling index 13 (1D thin), as every UI
//! texture of the game: its texture creation (CreateGnmTexture) asserted on
//! a linear one and on one-byte texels.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// MD5 (RFC 1321), for the archive's name hashes.

pub fn md5(data: &[u8]) -> [u8; 16] {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    let k: Vec<u32> = (0..64).map(|i| ((i as f64 + 1.0).sin().abs() * 4294967296.0) as u32).collect();
    let (mut a0, mut b0, mut c0, mut d0) = (0x67452301u32, 0xefcdab89u32, 0x98badcfeu32, 0x10325476u32);
    let mut msg = data.to_vec();
    let bits = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bits.to_le_bytes());
    for chunk in msg.chunks(64) {
        let m: Vec<u32> = chunk.chunks(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect();
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let f = f.wrapping_add(a).wrapping_add(k[i]).wrapping_add(m[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(f.rotate_left(S[i]));
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }
    let mut out = [0u8; 16];
    for (i, v) in [a0, b0, c0, d0].iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// PSARC 1.4 in the game's patch-archive form (module docs).

const BLOCK: usize = 0x10000;

pub fn psarc(files: &[(String, Vec<u8>)]) -> Vec<u8> {
    let manifest = files.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join("\0").into_bytes();
    // (Entry 0, the manifest, keeps its place; the rest by name hash.)
    let mut order: Vec<usize> = (0..files.len()).collect();
    let hashes: Vec<[u8; 16]> = files.iter().map(|(n, _)| md5(n.to_ascii_uppercase().as_bytes())).collect();
    order.sort_by_key(|&i| hashes[i]);
    let mut entries: Vec<([u8; 16], &[u8])> = vec![([0u8; 16], &manifest)];
    entries.extend(order.iter().map(|&i| (hashes[i], files[i].1.as_slice())));
    let blocks: Vec<usize> = entries.iter().map(|(_, d)| d.len().div_ceil(BLOCK).max(1)).collect();
    let nblocks: usize = blocks.iter().sum();
    let toc_len = 32 + entries.len() * 30 + nblocks * 2;
    let mut out = Vec::with_capacity(toc_len + entries.iter().map(|(_, d)| d.len()).sum::<usize>());
    out.extend_from_slice(b"PSAR");
    out.extend_from_slice(&1u16.to_be_bytes());
    out.extend_from_slice(&4u16.to_be_bytes());
    out.extend_from_slice(b"zlib");
    for v in [toc_len as u32, 30, entries.len() as u32, BLOCK as u32, 13] {
        out.extend_from_slice(&v.to_be_bytes());
    }
    let (mut block, mut offset) = (0u32, toc_len as u64);
    for (i, (hash, data)) in entries.iter().enumerate() {
        out.extend_from_slice(hash);
        out.extend_from_slice(&block.to_be_bytes());
        out.extend_from_slice(&(data.len() as u64).to_be_bytes()[3..]);
        out.extend_from_slice(&offset.to_be_bytes()[3..]);
        block += blocks[i] as u32;
        offset += data.len() as u64;
    }
    // (Stored blocks: a full one is written as 0, a short one as its size.)
    for (_, data) in &entries {
        let n = data.len().div_ceil(BLOCK).max(1);
        for b in 0..n {
            let len = (data.len() - b * BLOCK).min(BLOCK);
            out.extend_from_slice(&(if len == BLOCK { 0 } else { len as u16 }).to_be_bytes());
        }
    }
    for (_, data) in &entries {
        out.extend_from_slice(data);
    }
    out
}

// ---------------------------------------------------------------------------
// GNF textures (RGBA8, 1D thin, one level).

const GNF_HEADER: usize = 0x100;
/// Tiling index 13 (1D thin: 8x8 micro tiles, row by row).
const TILE: u8 = 13;

/// `rgba` (`width` x `height` texels, row by row) in the texture's tiled
/// order (`width` and `height` multiples of 8).
pub fn tiled(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; rgba.len()];
    crate::gpu::tiling::tile_slice(rgba, &mut out, width, height, width, height, 32, crate::gpu::tiling::tile_mode(TILE));
    out
}

/// A GNF file for a `width` x `height` RGBA8 texture (`rgba`: row by row;
/// `width` and `height` multiples of 8).
pub fn gnf(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    let size = (width * height * 4) as usize;
    assert!(width % 8 == 0 && height % 8 == 0 && rgba.len() == size, "gnf: {width}x{height} with {} bytes", rgba.len());
    // (The words of the game's own UI textures, e.g. its boot logos.)
    let t: [u32; 8] = [
        0,
        // base address high byte as in the game's files, IMG_DATA_FORMAT_8_8_8_8, UNORM
        0x08 | 10 << 20,
        (width - 1) | (height - 1) << 14 | 7 << 28,
        // selects X Y Z W, level 0 only, tiling index 13, bit 26 as in the
        // game's files, 2D
        0xfac | (TILE as u32) << 20 | 1 << 26 | 9 << 28,
        // depth 1, pitch
        (width - 1) << 13,
        0,
        0,
        // (the texels' size, as in the game's files)
        size as u32,
    ];
    let data = tiled(width, height, rgba);
    let mut out = Vec::with_capacity(GNF_HEADER + size);
    out.extend_from_slice(b"GNF ");
    out.extend_from_slice(&((GNF_HEADER - 8) as u32).to_le_bytes());
    out.extend_from_slice(&[2, 1, 8, 0]);
    out.extend_from_slice(&((GNF_HEADER + size) as u32).to_le_bytes());
    for w in t {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out.resize(GNF_HEADER, 0);
    out.extend_from_slice(&data);
    out
}

// ---------------------------------------------------------------------------
// Text: GDI rasterization (grayscale antialiasing) into coverage masks.

/// The face the game's menus resemble most among Windows' fonts (a
/// Helvetica-style sans serif).
const FACE: &str = "Arial";

/// The menus' font at `px` pixels (em height) in `weight`.
fn create_font(px: u32, weight: i32) -> windows_sys::Win32::Graphics::Gdi::HFONT {
    use windows_sys::Win32::Graphics::Gdi::*;
    let face: Vec<u16> = FACE.encode_utf16().chain(std::iter::once(0)).collect();
    // SAFETY: a plain GDI call with a NUL-terminated face name.
    unsafe { CreateFontW(-(px as i32), 0, 0, 0, weight, 0, 0, 0, DEFAULT_CHARSET as u32, OUT_TT_PRECIS as u32, CLIP_DEFAULT_PRECIS as u32, ANTIALIASED_QUALITY as u32, (DEFAULT_PITCH | FF_SWISS) as u32, face.as_ptr()) }
}

/// `text` split at spaces into lines at most `max_w` pixels wide at `px` in
/// `weight` (measured as `raster` draws them; a word wider than a line stays
/// whole). Text that fits stays one line as it is.
pub fn wrap(text: &str, px: u32, weight: i32, max_w: u32) -> Vec<String> {
    use windows_sys::Win32::Graphics::Gdi::*;
    // SAFETY: a memory DC with the font selected while it measures; both
    // released before returning.
    unsafe {
        let dc = CreateCompatibleDC(std::ptr::null_mut());
        if dc.is_null() {
            return vec![text.to_string()];
        }
        let font = create_font(px, weight);
        let old_font = SelectObject(dc, font as _);
        let width = |s: &str| {
            let w: Vec<u16> = s.encode_utf16().collect();
            let mut size = std::mem::zeroed::<windows_sys::Win32::Foundation::SIZE>();
            GetTextExtentPoint32W(dc, w.as_ptr(), w.len() as i32, &mut size);
            size.cx.max(0) as u32
        };
        let mut lines = Vec::new();
        if width(text) <= max_w {
            lines.push(text.to_string());
        } else {
            let mut cur = String::new();
            for word in text.split(' ').filter(|w| !w.is_empty()) {
                let longer = if cur.is_empty() { word.to_string() } else { format!("{cur} {word}") };
                if !cur.is_empty() && width(&longer) > max_w {
                    lines.push(std::mem::replace(&mut cur, word.to_string()));
                } else {
                    cur = longer;
                }
            }
            if !cur.is_empty() {
                lines.push(cur);
            }
        }
        SelectObject(dc, old_font);
        DeleteObject(font as _);
        DeleteDC(dc);
        lines
    }
}

/// `text` at `px` pixels (em height) in `weight` (400 regular, 700 bold):
/// its width, height (one line) and coverage, row by row.
pub fn raster(text: &str, px: u32, weight: i32) -> Option<(u32, u32, Vec<u8>)> {
    use windows_sys::Win32::Graphics::Gdi::*;
    // (The menus' mouse pointer: its white face and its dark rim, drawn
    // under it.)
    if text == "\u{1}pointer" || text == "\u{1}pointer_rim" {
        return Some(pointer(px, text.ends_with("rim")));
    }
    let wide: Vec<u16> = text.encode_utf16().collect();
    unsafe {
        let dc = CreateCompatibleDC(std::ptr::null_mut());
        if dc.is_null() {
            return None;
        }
        let font = create_font(px, weight);
        let old_font = SelectObject(dc, font as _);
        let mut size = std::mem::zeroed::<windows_sys::Win32::Foundation::SIZE>();
        GetTextExtentPoint32W(dc, wide.as_ptr(), wide.len() as i32, &mut size);
        let mut tm = std::mem::zeroed::<TEXTMETRICW>();
        GetTextMetricsW(dc, &mut tm);
        // (A little room for overhangs of italic or accented glyphs.)
        let pad = (px / 8).max(2) as i32;
        let w = (size.cx + 2 * pad).max(1) as u32;
        let h = (tm.tmHeight).max(1) as u32;
        let mut bmi = std::mem::zeroed::<BITMAPINFO>();
        bmi.bmiHeader.biSize = std::mem::size_of::<BITMAPINFOHEADER>() as u32;
        bmi.bmiHeader.biWidth = w as i32;
        bmi.bmiHeader.biHeight = -(h as i32);
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = BI_RGB;
        let mut bits: *mut core::ffi::c_void = std::ptr::null_mut();
        let dib = CreateDIBSection(dc, &bmi, DIB_RGB_COLORS, &mut bits, std::ptr::null_mut(), 0);
        if dib.is_null() || bits.is_null() {
            SelectObject(dc, old_font);
            DeleteObject(font as _);
            DeleteDC(dc);
            return None;
        }
        let old_bmp = SelectObject(dc, dib as _);
        std::ptr::write_bytes(bits as *mut u8, 0, (w * h * 4) as usize);
        SetBkMode(dc, TRANSPARENT as i32);
        SetTextColor(dc, 0x00FF_FFFF);
        TextOutW(dc, pad, 0, wide.as_ptr(), wide.len() as i32);
        GdiFlush();
        let px32 = std::slice::from_raw_parts(bits as *const u8, (w * h * 4) as usize);
        let cov: Vec<u8> = px32.chunks(4).map(|p| p[1].max(p[0]).max(p[2])).collect();
        SelectObject(dc, old_bmp);
        SelectObject(dc, old_font);
        DeleteObject(dib as _);
        DeleteObject(font as _);
        DeleteDC(dc);
        Some((w, h, cov))
    }
}

/// The menus' mouse pointer, `px` tall: an arrow's coverage - its face, or
/// (`rim`) the face grown by its rim - with its tip near the texel origin.
fn pointer(px: u32, rim: bool) -> (u32, u32, Vec<u8>) {
    let h = px.max(12);
    let w = h * 3 / 4;
    // (The arrow's outline in units of its height: tip, left edge down,
    // notch, tail, back up to the wing.)
    let poly: [(f32, f32); 7] = [(0.0, 0.0), (0.0, 0.78), (0.2, 0.6), (0.34, 0.92), (0.45, 0.87), (0.31, 0.56), (0.56, 0.56)];
    let pts: Vec<(f32, f32)> = poly.iter().map(|&(x, y)| (x * h as f32 + 1.0, y * h as f32 + 1.0)).collect();
    let inside = |x: f32, y: f32| {
        let mut c = false;
        let n = pts.len();
        for i in 0..n {
            let (xi, yi) = pts[i];
            let (xj, yj) = pts[(i + n - 1) % n];
            if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
                c = !c;
            }
        }
        c
    };
    let dist = |x: f32, y: f32| {
        let n = pts.len();
        let mut best = f32::MAX;
        for i in 0..n {
            let (ax, ay) = pts[i];
            let (bx, by) = pts[(i + 1) % n];
            let (dx, dy) = (bx - ax, by - ay);
            let t = (((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
            let (qx, qy) = (ax + t * dx - x, ay + t * dy - y);
            best = best.min((qx * qx + qy * qy).sqrt());
        }
        best
    };
    let (tw, th) = (w + 3, h + 3);
    let mut out = vec![0u8; (tw * th) as usize];
    for y in 0..th {
        for x in 0..tw {
            // (4x4 samples per texel.)
            let mut n = 0u32;
            for sy in 0..4 {
                for sx in 0..4 {
                    let (fx, fy) = (x as f32 + (sx as f32 + 0.5) / 4.0, y as f32 + (sy as f32 + 0.5) / 4.0);
                    let (ins, d) = (inside(fx, fy), dist(fx, fy));
                    if if rim { ins || d <= 1.6 } else { ins && d > 0.6 } {
                        n += 1;
                    }
                }
            }
            out[(y * tw + x) as usize] = (n * 255 / 16) as u8;
        }
    }
    (tw, th, out)
}

// ---------------------------------------------------------------------------
// The text atlas: strings rendered on demand into one alpha texture of the
// archive, written into the game's copy of it once that is known.

pub const ATLAS_W: u32 = 2048;
/// (An unusual height: the texture cache recognizes the atlas by its
/// shape - 2048x1016 RGBA8 - among the game's textures.)
pub const ATLAS_H: u32 = 1016;
/// The atlas texture's name for the game's `Load` (its file:
/// `<name>.gnf` in the archive).
pub const ATLAS_NAME: &str = "game/textures/ui/tlgpc/text0";

#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

struct Atlas {
    pixels: Vec<u8>,
    /// Shelves: (y, height, x of the next free texel).
    shelves: Vec<(u32, u32, u32)>,
    map: HashMap<(String, u32, i32), Rect>,
    /// Rows changed since the last copy into the game's texture.
    dirty: Option<(u32, u32)>,
    /// Bumped when the atlas is emptied (callers' cached places are stale).
    generation: u64,
}

static ATLAS: Mutex<Option<Atlas>> = Mutex::new(None);
/// The game's copy of the atlas texels (0: not seen yet), from the texture
/// cache (`note_texture`).
static ATLAS_ADDR: AtomicU64 = AtomicU64::new(0);

fn atlas_place(a: &mut Atlas, w: u32, h: u32) -> Option<(u32, u32)> {
    const GAP: u32 = 2;
    if w + GAP > ATLAS_W || h + GAP > ATLAS_H {
        return None;
    }
    for s in a.shelves.iter_mut() {
        if h <= s.1 && s.1 <= h + h / 4 + 4 && s.2 + w + GAP <= ATLAS_W {
            let x = s.2;
            s.2 += w + GAP;
            return Some((x, s.0));
        }
    }
    let y = a.shelves.last().map_or(0, |s| s.0 + s.1 + GAP);
    if y + h > ATLAS_H {
        return None;
    }
    a.shelves.push((y, h, w + GAP));
    Some((0, y))
}

/// `text` in the atlas at `px` pixels and `weight` (rendered now if new):
/// its place and the atlas generation. None: it cannot be rendered.
pub fn text(text: &str, px: u32, weight: i32) -> Option<(Rect, u64)> {
    let mut g = ATLAS.lock();
    let a = g.get_or_insert_with(|| Atlas { pixels: vec![0; (ATLAS_W * ATLAS_H) as usize], shelves: Vec::new(), map: HashMap::new(), dirty: None, generation: 1 });
    let key = (text.to_string(), px, weight);
    if let Some(r) = a.map.get(&key) {
        return Some((*r, a.generation));
    }
    let (w, h, cov) = raster(text, px, weight)?;
    let (x, y) = match atlas_place(a, w, h) {
        Some(p) => p,
        None => {
            // Full: start over (the script asks again for what it shows).
            a.pixels.fill(0);
            a.shelves.clear();
            a.map.clear();
            a.generation += 1;
            a.dirty = Some((0, ATLAS_H));
            atlas_place(a, w, h)?
        }
    };
    for row in 0..h {
        let src = &cov[(row * w) as usize..((row + 1) * w) as usize];
        let at = ((y + row) * ATLAS_W + x) as usize;
        a.pixels[at..at + w as usize].copy_from_slice(src);
    }
    a.dirty = Some(a.dirty.map_or((y, y + h), |(y0, y1)| (y0.min(y), y1.max(y + h))));
    let r = Rect { x, y, w, h };
    a.map.insert(key, r);
    let gen_ = a.generation;
    flush(a);
    Some((r, gen_))
}

/// Copies the changed rows into the game's copy of the atlas (when known):
/// whole bands of micro tiles (8 rows each, contiguous in 1D thin), white
/// texels with the coverage as alpha.
fn flush(a: &mut Atlas) {
    let base = ATLAS_ADDR.load(Ordering::Acquire);
    let Some((y0, y1)) = a.dirty else { return };
    if base == 0 || !crate::mem::is_guest(base, (ATLAS_W * ATLAS_H * 4) as u64) {
        return;
    }
    let (b0, b1) = (y0 & !7, y1.min(ATLAS_H).next_multiple_of(8));
    let rows = b1 - b0;
    let mut rgba = vec![0u8; (ATLAS_W * rows * 4) as usize];
    for (i, &c) in a.pixels[(b0 * ATLAS_W) as usize..(b1 * ATLAS_W) as usize].iter().enumerate() {
        rgba[i * 4..i * 4 + 4].copy_from_slice(&[255, 255, 255, c]);
    }
    let band = tiled(ATLAS_W, rows, &rgba);
    let at = (b0 * ATLAS_W * 4) as usize;
    // SAFETY: the game's texture memory for the atlas (its descriptor's
    // base, the whole texture checked as guest memory); only our texels.
    unsafe { std::ptr::copy_nonoverlapping(band.as_ptr(), (base as usize + at) as *mut u8, band.len()) };
    a.dirty = None;
}

/// The texture cache made an RGBA8 image: if it is the atlas (by its
/// shape), its texels are at `base` from now on.
pub fn note_image(base: u64, width: u32, height: u32) {
    if width != ATLAS_W || height != ATLAS_H {
        return;
    }
    if ATLAS_ADDR.swap(base, Ordering::AcqRel) != base {
        crate::info!("pc settings: the text atlas is at {base:#x}");
        if let Some(a) = ATLAS.lock().as_mut() {
            a.dirty = Some((0, ATLAS_H));
            flush(a);
        }
    }
}

// ---------------------------------------------------------------------------
// Reading the game's own archives (the user's files, read-only), for
// copies of its menu parts under other names.

struct Archive {
    file: std::fs::File,
    /// Per entry: first block, size, offset.
    entries: Vec<(u32, u64, u64)>,
    zsizes: Vec<u32>,
    block: u64,
    by_md5: HashMap<[u8; 16], usize>,
}

impl Archive {
    fn open(path: &std::path::Path) -> Option<Archive> {
        use std::os::windows::fs::FileExt;
        let file = std::fs::File::open(path).ok()?;
        let mut h = [0u8; 32];
        file.seek_read(&mut h, 0).ok()?;
        if &h[0..4] != b"PSAR" {
            return None;
        }
        let be = |o: usize| u32::from_be_bytes([h[o], h[o + 1], h[o + 2], h[o + 3]]);
        let (toc_len, esz, count, block) = (be(12) as usize, be(16) as usize, be(20) as usize, be(24) as u64);
        let mut toc = vec![0u8; toc_len.checked_sub(32)?];
        file.seek_read(&mut toc, 32).ok()?;
        let mut entries = Vec::with_capacity(count);
        let mut by_md5 = HashMap::new();
        for i in 0..count {
            let e = toc.get(i * esz..i * esz + 30)?;
            let n40 = |o: usize| e[o..o + 5].iter().fold(0u64, |v, &b| v << 8 | b as u64);
            entries.push((u32::from_be_bytes([e[16], e[17], e[18], e[19]]), n40(20), n40(25)));
            by_md5.insert(<[u8; 16]>::try_from(&e[0..16]).ok()?, i);
        }
        let width = if block <= 0x10000 { 2 } else if block <= 0x100_0000 { 3 } else { 4 };
        let tab = &toc[count * esz..];
        let zsizes = tab.chunks_exact(width).map(|c| c.iter().fold(0u32, |v, &b| v << 8 | b as u32)).collect();
        Some(Archive { file, entries, zsizes, block, by_md5 })
    }

    fn entry(&self, i: usize) -> Option<Vec<u8>> {
        use std::os::windows::fs::FileExt;
        let (mut idx, size, mut off) = *self.entries.get(i)?;
        let mut out = Vec::with_capacity(size as usize);
        while (out.len() as u64) < size {
            let want = self.block.min(size - out.len() as u64) as usize;
            let z = match *self.zsizes.get(idx as usize)? {
                0 => self.block as usize,
                z => z as usize,
            };
            let mut data = vec![0u8; z];
            self.file.seek_read(&mut data, off).ok()?;
            if z == want {
                out.extend_from_slice(&data);
            } else {
                out.extend_from_slice(&miniz_oxide::inflate::decompress_to_vec_zlib(&data).ok()?);
            }
            off += z as u64;
            idx += 1;
        }
        Some(out)
    }

    /// A file by name (the base archives hash names as they are, the patch
    /// archives upper-case).
    fn read(&self, name: &str) -> Option<Vec<u8>> {
        [name.to_ascii_uppercase(), name.to_string(), format!("/{name}"), format!("/{name}").to_ascii_uppercase()]
            .iter()
            .find_map(|v| self.by_md5.get(&md5(v.as_bytes())))
            .and_then(|&i| self.entry(i))
    }
}

/// A file of the game's archives (patch over base; the language's first,
/// then the boot archives).
pub fn game_file(name: &str, lang: &str) -> Option<Vec<u8>> {
    let mut names = Vec::new();
    if !lang.is_empty() {
        names.push(format!("trico_{lang}p.psarc"));
        names.push(format!("trico_{lang}.psarc"));
    }
    names.push("trico_bootp.psarc".into());
    names.push("trico_boot.psarc".into());
    for n in names {
        let Some(p) = crate::hle::kernel::fs::host_path_existing(&format!("/app0/{n}")) else { continue };
        if let Some(d) = Archive::open(&p).and_then(|a| a.read(name)) {
            return Some(d);
        }
    }
    None
}

/// `data` with every `from` replaced by `to` (same lengths: the game's
/// object files keep their string tables' offsets).
fn renamed(data: &[u8], pairs: &[(&str, &str)]) -> Vec<u8> {
    let mut out = data.to_vec();
    for (from, to) in pairs {
        assert_eq!(from.len(), to.len());
        let (f, t) = (from.as_bytes(), to.as_bytes());
        let mut i = 0;
        while i + f.len() <= out.len() {
            if &out[i..i + f.len()] == f {
                out[i..i + f.len()].copy_from_slice(t);
                i += f.len();
            } else {
                i += 1;
            }
        }
    }
    out
}

/// The Options lists' "PC Settings" item's part: the game's "Exit Game"
/// label object (a title-menu item of the Windows editor build: in no menu
/// table or code of the PS4 game), placed by layouts of its own; the game's
/// list code moves its cursor onto it and chooses it like any item
/// (`menu_patches`). The game's code draws only the parts of its own lists'
/// arrays, so this part stays undrawn (copies of the Copyright item under a
/// new name did not draw either); its label is drawn by the port's script
/// with the game's UI drawing, where the row is (`pc_menu`). Its texture
/// is made transparent all the same (the port's archive is read before the
/// game's own: an override of a game texture there showed instead). The
/// part table entries: `script_fixes::ALWAYS`.

fn pc_item_files(lang: &str) -> Result<Vec<(String, Vec<u8>)>, String> {
    let mut out = Vec::new();
    let clear = gnf(1024, 128, &vec![0u8; 1024 * 128 * 4]);
    out.push(("game/textures/ui/gamemenu/sys_exitgame_1.gnf".into(), clear.clone()));
    if !lang.is_empty() {
        out.push((format!("game_{lang}/textures/ui/gamemenu/sys_exitgame_1.gnf"), clear));
    }
    // Its places (the menus' layouts: normalized positions, float4 records
    // x y 0 1): in the pause menu the Copyright item's own layout, a row
    // the pause lists leave free above Back; on the title screen the title
    // Copyright item's, one row lower (half way to Back, two rows below).
    // (Layouts copied under new names work: their objects get new tokens.)
    let layout_pairs = |token: &'static str| [("COPYRIGHT", "PCSETTING"), ("copyright", "pcsetting"), (token, "7c5ec0d1")];
    for side in ["objects", "assets"] {
        let from = format!("game/{side}/ui/gamemenu/model/sys_option_item_copyright_model_pos.bod");
        let d = game_file(&from, "").ok_or_else(|| format!("{from} not in the game files"))?;
        out.push((format!("game/{side}/ui/gamemenu/model/sys_option_item_pcsetting_model_pos.bod"), renamed(&d, &layout_pairs("19f555e3"))));
    }
    // (The pause menu without Treasure - before it is found: the PC item in
    // Treasure's row, no gap above it.)
    for side in ["objects", "assets"] {
        let from = format!("game/{side}/ui/gamemenu/model/sys_option_item_copyright_model_pos.bod");
        let treasure = format!("game/{side}/ui/gamemenu/model/sys_option_item_treasure_model_pos.bod");
        let d = game_file(&from, "").ok_or_else(|| format!("{from} not in the game files"))?;
        let tr = game_file(&treasure, "").ok_or_else(|| format!("{treasure} not in the game files"))?;
        let (Some(y0), Some(y1)) = (layout_y(&d), layout_y(&tr)) else { return Err(format!("{from}: no position")) };
        let pairs = [("COPYRIGHT", "PCSETTNG3"), ("copyright", "pcsettng3"), ("19f555e3", "7c5ec0d3")];
        out.push((format!("game/{side}/ui/gamemenu/model/sys_option_item_pcsettng3_model_pos.bod"), layout_moved(&renamed(&d, &pairs), y0, y1)));
    }
    for side in ["objects", "assets"] {
        let from = format!("game/{side}/ui/gamemenu/model/sys_option2_item_copyright_model_pos.bod");
        let back = format!("game/{side}/ui/gamemenu/model/sys_option2_item_return_model_pos.bod");
        let d = game_file(&from, "").ok_or_else(|| format!("{from} not in the game files"))?;
        let b = game_file(&back, "").ok_or_else(|| format!("{back} not in the game files"))?;
        let (Some(y0), Some(y1)) = (layout_y(&d), layout_y(&b)) else { return Err(format!("{from}: no position")) };
        let moved = layout_moved(&renamed(&d, &layout_pairs("474cb551")), y0, (y0 + y1) / 2.0);
        out.push((format!("game/{side}/ui/gamemenu/model/sys_option2_item_pcsetting_model_pos.bod"), moved));
    }
    Ok(out)
}

/// The float4 records (x, y, 0, 1) of a menu layout file, as offsets.
fn layout_points(d: &[u8]) -> Vec<usize> {
    (0..d.len().saturating_sub(16))
        .step_by(4)
        .filter(|&i| {
            let f = |k: usize| f32::from_le_bytes([d[i + k], d[i + k + 1], d[i + k + 2], d[i + k + 3]]);
            f(12) == 1.0 && f(8) == 0.0 && f(0).abs() < 2.0 && f(4).abs() < 2.0 && (f(0) != 0.0 || f(4) != 0.0)
        })
        .collect()
}

/// A layout's place (the x and y of its first position, normalized).
pub fn layout_xy(d: &[u8]) -> Option<(f32, f32)> {
    let i = *layout_points(d).first()?;
    let f = |k: usize| f32::from_le_bytes([d[i + k], d[i + k + 1], d[i + k + 2], d[i + k + 3]]);
    Some((f(0), f(4)))
}

/// A layout's row (the y of its first position).
fn layout_y(d: &[u8]) -> Option<f32> {
    let i = *layout_points(d).first()?;
    Some(f32::from_le_bytes([d[i + 4], d[i + 5], d[i + 6], d[i + 7]]))
}

/// `d` with the positions in row `from` moved to row `to`.
fn layout_moved(d: &[u8], from: f32, to: f32) -> Vec<u8> {
    let mut out = d.to_vec();
    for i in layout_points(d) {
        let y = f32::from_le_bytes([d[i + 4], d[i + 5], d[i + 6], d[i + 7]]);
        if (y - from).abs() < 1e-4 {
            out[i + 4..i + 8].copy_from_slice(&to.to_le_bytes());
        }
    }
    out
}


// ---------------------------------------------------------------------------
// The archive.

/// Builds the archive and has the VFS serve it as /app0/trico_other.psarc.
pub fn init() {
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    files.push((format!("{ATLAS_NAME}.gnf"), gnf(ATLAS_W, ATLAS_H, &vec![0u8; (ATLAS_W * ATLAS_H * 4) as usize])));
    let (_, _, lang) = crate::hle::libs::app::game_language();
    match pc_item_files(lang) {
        Ok(f) => files.extend(f),
        Err(e) => crate::warn!("pc settings: the Options item's files: {e}; no item in the Options menu"),
    }
    // (The prompts' picture slots: pc_prompts.)
    match crate::pc_prompts::slot_files() {
        Some(f) => files.extend(f),
        None => crate::warn!("prompts: the game's prompt picture is not in the game files; no picture slots"),
    }
    // (The trophies' icons and pop-up frame: achievements.)
    files.extend(crate::achievements::files());
    let bytes = psarc(&files);
    crate::info!("pc settings: the port's archive trico_other.psarc, {} files, {} KiB", files.len(), bytes.len() >> 10);
    crate::hle::kernel::fs::add_virtual("/app0/trico_other.psarc", Arc::new(bytes), None);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn md5_known_values() {
        let hex = |d: [u8; 16]| d.iter().map(|b| format!("{b:02x}")).collect::<String>();
        assert_eq!(hex(md5(b"")), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(hex(md5(b"The quick brown fox jumps over the lazy dog")), "9e107d9d372bb6826bd81d3542a419d6");
        // (The game's patch archives hash upper-case names: trico_enup.psarc's first file.)
        assert_eq!(hex(md5(b"GAME/ASSETS/UI/GAMEMENU/MODEL/SYS_OPTION_ITEM_SCREEN_HDR_DISABLE.BOD")), "724b0b6e402595e85dcf0ec21dc0120b");
    }

    #[test]
    fn psarc_layout() {
        let big = vec![7u8; BLOCK + 10];
        let a = psarc(&[("a/b.gnf".into(), vec![1, 2, 3]), ("c/d.bin".into(), big.clone())]);
        assert_eq!(&a[0..4], b"PSAR");
        let be = |o: usize| u32::from_be_bytes([a[o], a[o + 1], a[o + 2], a[o + 3]]);
        let (toc_len, esz, count, flags) = (be(12) as usize, be(16) as usize, be(20) as usize, be(28));
        assert_eq!((esz, count, flags), (30, 3, 13));
        assert_eq!(toc_len, 32 + 3 * 30 + (1 + 1 + 2) * 2);
        // Entry 0: the manifest, zero hash.
        assert_eq!(&a[32..48], &[0u8; 16]);
        let size = |e: usize| u64::from_be_bytes([0, 0, 0, a[e + 20], a[e + 21], a[e + 22], a[e + 23], a[e + 24]]);
        let off = |e: usize| u64::from_be_bytes([0, 0, 0, a[e + 25], a[e + 26], a[e + 27], a[e + 28], a[e + 29]]);
        let man = &a[off(32) as usize..(off(32) + size(32)) as usize];
        assert_eq!(man, b"a/b.gnf\0c/d.bin");
        // The other entries sorted by hash; each finds its data.
        let (e1, e2) = (32 + 30, 32 + 60);
        assert!(a[e1..e1 + 16] < a[e2..e2 + 16]);
        for e in [e1, e2] {
            let h = &a[e..e + 16];
            let data = &a[off(e) as usize..(off(e) + size(e)) as usize];
            if h == md5(b"A/B.GNF") {
                assert_eq!(data, &[1, 2, 3]);
            } else {
                assert_eq!(h, md5(b"C/D.BIN"));
                assert_eq!(data, &big[..]);
            }
        }
    }

    #[test]
    fn gnf_header() {
        let g = gnf(64, 8, &[0u8; 64 * 8 * 4]);
        assert_eq!(&g[0..4], b"GNF ");
        assert_eq!(g.len(), 0x100 + 64 * 8 * 4);
        let w = |i: usize| u32::from_le_bytes([g[0x10 + i * 4], g[0x11 + i * 4], g[0x12 + i * 4], g[0x13 + i * 4]]);
        // (As the game's own RGBA8 UI textures: w1 0x00a00008, w3 0x94d00fac.)
        assert_eq!(w(1), 0x00a0_0008);
        assert_eq!(w(3), 0x94d0_0fac);
        assert_eq!(w(2) & 0x3fff, 63);
        assert_eq!((w(2) >> 14) & 0x3fff, 7);
        assert_eq!(((w(4) >> 13) & 0x3fff) + 1, 64);
        assert_eq!(w(7), 64 * 8 * 4);
    }

    #[test]
    fn tiled_texels_round_trip() {
        let (w, h) = (16u32, 16u32);
        let rgba: Vec<u8> = (0..w * h * 4).map(|i| (i * 7 % 251) as u8).collect();
        let t = tiled(w, h, &rgba);
        let mut back = vec![0u8; rgba.len()];
        crate::gpu::tiling::detile_slice(&t, &mut back, w, h, w, h, 32, crate::gpu::tiling::tile_mode(TILE));
        assert_eq!(back, rgba);
    }
}
