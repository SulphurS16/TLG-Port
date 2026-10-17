//! GPU resources backing guest memory: the image cache (render targets and
//! textures share images by guest address), the per-batch upload ring and
//! samplers.

use super::desc::{self, Ssharp, Tsharp};
use super::tiling;
use super::vk::{Ctx, ctx};
use ash::vk;
use std::collections::HashMap;

/// (768 MiB: 128 MiB per batch slot. SOTC writes back 4-16 MiB buffers
/// several times a frame; 42 MiB segments ended most batches after two or
/// three of them, and every new batch wrote the same ranges back again.)
pub const RING_SIZE: u64 = 768 << 20;
/// Device-local ring for buffers written by shaders (strided GPU writes to
/// host memory are very slow); copied back to the host ring per batch.
pub const DEV_RING_SIZE: u64 = 512 << 20;
/// Per-draw SGPR data in device-local host-visible memory (the BAR window
/// is ~256 MiB without resizable BAR).
pub const BAR_RING_SIZE: u64 = 48 << 20;
/// The compute queues' recording context while it is not used (`tune
/// cs_ctx` off at start; used, it gets the graphics queue's sizes): host
/// ring, device ring, BAR ring.
pub const CS_RING_SIZE: u64 = 12 << 20;
pub const CS_DEV_RING_SIZE: u64 = 12 << 20;
pub const CS_BAR_RING_SIZE: u64 = 6 << 20;

pub struct Ring {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub ptr: *mut u8,
    pub size: u64,
    pub used: u64,
    pub align: u64,
    pub seg_lo: u64,
    pub seg_hi: u64,
    /// Highest use of a segment (bytes) since the last `take_peak`.
    pub peak: u64,
}

unsafe impl Send for Ring {}

impl Ring {
    pub fn new(c: &Ctx, size: u64) -> Result<Ring, String> {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::UNIFORM_BUFFER;
        let (buffer, memory, ptr) = c.host_buffer(size, usage)?;
        let align = c.limits.min_storage_buffer_offset_alignment.max(16);
        Ok(Ring { buffer, memory, ptr, size, used: 0, align, seg_lo: 0, seg_hi: size, peak: 0 })
    }
    /// Device-local ring the CPU writes through the BAR window (write-combined:
    /// for data the GPU reads, written once by the CPU).
    pub fn new_bar(c: &Ctx, size: u64) -> Result<Ring, String> {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::UNIFORM_BUFFER;
        unsafe {
            let bi = vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = c.device.create_buffer(&bi, None).map_err(|e| format!("vkCreateBuffer: {e}"))?;
            let req = c.device.get_buffer_memory_requirements(buffer);
            let memory = c.alloc(req, vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT).inspect_err(|_| c.device.destroy_buffer(buffer, None))?;
            c.device.bind_buffer_memory(buffer, memory, 0).map_err(|e| format!("bind buffer: {e}"))?;
            let ptr = c.device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()).map_err(|e| format!("map: {e}"))? as *mut u8;
            let align = c.limits.min_storage_buffer_offset_alignment.max(16);
            Ok(Ring { buffer, memory, ptr, size, used: 0, align, seg_lo: 0, seg_hi: size, peak: 0 })
        }
    }
    /// Device-local ring (no CPU mapping: `write*` must not be used).
    pub fn new_device(c: &Ctx, size: u64) -> Result<Ring, String> {
        let mut usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDIRECT_BUFFER;
        if c.cond_render.is_some() {
            usage |= vk::BufferUsageFlags::CONDITIONAL_RENDERING_EXT;
        }
        let (buffer, memory) = c.device_buffer(size, usage)?;
        let align = c.limits.min_storage_buffer_offset_alignment.max(16);
        Ok(Ring { buffer, memory, ptr: std::ptr::null_mut(), size, used: 0, align, seg_lo: 0, seg_hi: size, peak: 0 })
    }
    /// Allocates `n` bytes in the current segment; returns the offset or None
    /// if the segment is full.
    pub fn alloc(&mut self, n: u64) -> Option<u64> {
        let off = self.used.next_multiple_of(self.align);
        if off + n.max(4) > self.seg_hi {
            return None;
        }
        self.used = off + n.max(4);
        self.peak = self.peak.max(self.used - self.seg_lo);
        Some(off)
    }
    /// The highest segment use since the last call, and the segment size.
    pub fn take_peak(&mut self) -> (u64, u64) {
        (std::mem::take(&mut self.peak), self.seg_hi - self.seg_lo)
    }
    /// Makes segment `i` of `count` current (its previous contents must no
    /// longer be in use by the GPU).
    pub fn set_segment(&mut self, i: usize, count: usize) {
        let seg = (self.size / count as u64) & !0xFFFF;
        self.seg_lo = seg * i as u64;
        self.seg_hi = self.seg_lo + seg;
        self.used = self.seg_lo;
    }
    pub fn remaining(&self) -> u64 {
        self.seg_hi.saturating_sub(self.used)
    }
    pub fn segment_size(&self) -> u64 {
        self.seg_hi - self.seg_lo
    }
    pub fn write(&mut self, off: u64, data: &[u8]) {
        unsafe { stream_copy(self.ptr.add(off as usize), data.as_ptr(), data.len()) };
    }
    /// Copies guest memory into the ring (unmapped bytes read as zero).
    pub fn write_guest(&mut self, off: u64, addr: u64, n: u64) {
        let dst = unsafe { std::slice::from_raw_parts_mut(self.ptr.add(off as usize), n as usize) };
        copy_from_guest_with(dst, addr, true);
    }
}

/// Copies `len` bytes to `dst` for the GPU alone (the upload rings): the
/// whole 64-byte lines with non-temporal stores, which go around the
/// caches (`tune stream_copies`) - the rings take more bytes per frame in
/// the later areas than the processor's last-level cache holds, and regular
/// stores fill it with lines the CPU does not read again, evicting the CPs'
/// working set; the ends and small copies as usual. An `sfence` orders the
/// streamed stores (weakly ordered) before what follows.
///
/// # Safety
/// `dst` and `src` valid for `len` bytes, not overlapping.
pub unsafe fn stream_copy(dst: *mut u8, src: *const u8, len: usize) {
    use core::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_sfence, _mm_stream_si128};
    let head = dst.align_offset(64);
    unsafe {
        if len < head + 64 || crate::tune::get(crate::tune::STREAM_COPIES) == 0 {
            std::ptr::copy_nonoverlapping(src, dst, len);
            return;
        }
        std::ptr::copy_nonoverlapping(src, dst, head);
        let (mut d, mut s) = (dst.add(head), src.add(head));
        let lines = (len - head) / 64;
        for _ in 0..lines {
            let a = _mm_loadu_si128(s as *const __m128i);
            let b = _mm_loadu_si128(s.add(16) as *const __m128i);
            let c = _mm_loadu_si128(s.add(32) as *const __m128i);
            let e = _mm_loadu_si128(s.add(48) as *const __m128i);
            _mm_stream_si128(d as *mut __m128i, a);
            _mm_stream_si128(d.add(16) as *mut __m128i, b);
            _mm_stream_si128(d.add(32) as *mut __m128i, c);
            _mm_stream_si128(d.add(48) as *mut __m128i, e);
            d = d.add(64);
            s = s.add(64);
        }
        std::ptr::copy_nonoverlapping(s, d, len - head - lines * 64);
        _mm_sfence();
    }
}

/// Copies from guest memory, zero-filling bytes that are not mapped.
pub fn copy_from_guest(dst: &mut [u8], addr: u64) {
    copy_from_guest_with(dst, addr, false);
}

/// `copy_from_guest`; `ring`: `dst` is upload-ring memory only the GPU
/// reads (`stream_copy`).
pub fn copy_from_guest_with(dst: &mut [u8], addr: u64, ring: bool) {
    let n = dst.len() as u64;
    if !crate::mem::is_guest(addr, n) {
        dst.fill(0);
        return;
    }
    let st = crate::mem::MEM.lock();
    let mut a = addr;
    while a < addr + n {
        let i = (a - addr) as usize;
        match st.find(a) {
            Some(v) if v.kind != crate::mem::VmaKind::Reserved => {
                let end = v.end().min(addr + n);
                let len = (end - a) as usize;
                if ring {
                    unsafe { stream_copy(dst[i..].as_mut_ptr(), a as *const u8, len) };
                } else {
                    unsafe { std::ptr::copy_nonoverlapping(a as *const u8, dst[i..].as_mut_ptr(), len) };
                }
                a = end;
            }
            _ => {
                // Unmapped: zero up to the next mapping or the end.
                let next = st.vmas.range(a..).next().map(|(_, v)| v.start).unwrap_or(addr + n).min(addr + n);
                let next = next.max(a + 1);
                dst[i..(next - addr) as usize].fill(0);
                a = next;
            }
        }
    }
}

/// Fills guest [addr, addr+n) with `byte` (one mapped region), like
/// `copy_to_guest` without a source buffer.
pub fn fill_guest(addr: u64, n: u64, byte: u8) -> bool {
    let st = crate::mem::MEM.lock();
    match st.find(addr) {
        Some(v) if v.kind != crate::mem::VmaKind::Reserved && addr + n <= v.end() => {
            drop(st);
            unsafe { std::ptr::write_bytes(addr as *mut u8, byte, n as usize) };
            true
        }
        _ => false,
    }
}

pub fn copy_to_guest(src: &[u8], addr: u64) -> bool {
    let n = src.len() as u64;
    let st = crate::mem::MEM.lock();
    match st.find(addr) {
        Some(v) if v.kind != crate::mem::VmaKind::Reserved && addr + n <= v.end() => {
            drop(st);
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), addr as *mut u8, src.len()) };
            true
        }
        _ => false,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageKey {
    pub base: u64,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub levels: u32,
    pub format: vk::Format,
    pub depth: bool,
    pub ty3d: bool,
}

/// Three words instead of eight fields: image maps are probed dozens of
/// times per draw. (Equal keys hash equally; the packing only mixes fields.)
impl std::hash::Hash for ImageKey {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        h.write_u64(self.base);
        h.write_u64(self.width as u64 | (self.height as u64) << 32);
        h.write_u64(self.layers as u64 ^ (self.levels as u64) << 24 ^ (self.depth as u64) << 30 ^ (self.ty3d as u64) << 31 ^ (self.format.as_raw() as u32 as u64) << 32);
    }
}

pub struct Image {
    pub key: ImageKey,
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub views: super::render::FastMap<(vk::Format, u16, bool, u32, u32), vk::ImageView>,
    /// Contents were produced by the GPU (do not re-upload from guest memory).
    pub gpu_written: bool,
    /// Hash of guest memory at last upload (CPU-sourced images).
    pub uploaded_hash: u64,
    /// Write-tracking stamp of the last validation against guest memory
    /// (None until first uploaded).
    pub validated: Option<u64>,
    pub upload_batch: u64,
    pub storage_ok: bool,
    /// Tile mode index of the guest surface (render targets), for write-back.
    pub tile_index: Option<u8>,
    /// CPU epoch of the last validation against the OS write watch.
    pub cpu_checked: u64,
    /// Pitch of the guest surface in elements when it exceeds the width
    /// (textures with a padded pitch; 0 = the image width).
    pub pitch: u32,
    /// GPU contents are newer than guest memory (rendered since last write-back).
    pub guest_stale: bool,
    /// Write-tracker stamp when the image last became stale: CPU writes to
    /// its guest memory after it mean the title reused the memory.
    pub stale_stamp: u64,
    /// Frame of the last check of a sampled, no longer rendered target for
    /// CPU writes to its memory (`Renderer::reused_target`).
    pub reuse_checked: u64,
    /// Guest-sourced textures not starting/ending on a page boundary: their
    /// bytes in the partially covered first and last page as last hashed
    /// (writes next to the texture dirty those pages too).
    pub edges: Option<Box<(Vec<u8>, Vec<u8>)>>,
    /// Renderer draw tick of the last write as a render target (0 = never).
    pub last_write: u64,
    /// Frame (flip count) of the last write as a render target.
    pub last_write_frame: u64,
    /// Compression metadata surface (HTILE of a depth target, CMASK of a
    /// fast-clear color target): guest (base, bytes).
    pub meta: Option<(u64, u64)>,
    /// GPU work wrote the metadata surface since it was last examined.
    pub meta_dirty: bool,
    /// The last write of the metadata surface was a fill of all of it with
    /// this dword (known without reading guest memory back).
    pub meta_fill: Option<u32>,
    /// In the renderer's list of images with metadata (`meta_images`).
    pub meta_listed: bool,
    /// GPU buffer writes (compute stores, GDS/DMA transfers) changed the
    /// guest surface after the image was last rendered: guest memory is
    /// newer and is reloaded into the image before its next use.
    pub guest_newer: bool,
    /// Device memory (bytes) and the frame of the last use (any view taken),
    /// for VRAM accounting and eviction.
    pub bytes: u64,
    pub last_used: u64,
}

pub fn is_depth(f: vk::Format) -> bool {
    matches!(f, vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::D32_SFLOAT_S8_UINT | vk::Format::D24_UNORM_S8_UINT)
}

pub fn storage_capable(c: &Ctx, f: vk::Format) -> bool {
    let p = unsafe { c.instance.get_physical_device_format_properties(c.pdev, f) };
    p.optimal_tiling_features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE)
}

fn attachment_capable(c: &Ctx, f: vk::Format) -> bool {
    let p = unsafe { c.instance.get_physical_device_format_properties(c.pdev, f) };
    p.optimal_tiling_features.intersects(vk::FormatFeatureFlags::COLOR_ATTACHMENT | vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
}

pub fn create_image(key: ImageKey) -> Result<Image, String> {
    let c = ctx();
    let depth = is_depth(key.format);
    let mut usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC;
    if attachment_capable(c, key.format) {
        usage |= if depth { vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT } else { vk::ImageUsageFlags::COLOR_ATTACHMENT };
    }
    let storage_ok = !depth && storage_capable(c, key.format);
    if storage_ok {
        usage |= vk::ImageUsageFlags::STORAGE;
    }
    let mut flags = vk::ImageCreateFlags::empty();
    if !depth {
        flags |= vk::ImageCreateFlags::MUTABLE_FORMAT;
    }
    if !key.ty3d && key.layers >= 6 && key.width == key.height {
        flags |= vk::ImageCreateFlags::CUBE_COMPATIBLE;
    }
    let ici = vk::ImageCreateInfo::default()
        .flags(flags)
        .image_type(if key.ty3d { vk::ImageType::TYPE_3D } else { vk::ImageType::TYPE_2D })
        .format(key.format)
        .extent(vk::Extent3D { width: key.width, height: key.height, depth: if key.ty3d { key.layers } else { 1 } })
        .mip_levels(key.levels)
        .array_layers(if key.ty3d { 1 } else { key.layers })
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    unsafe {
        let image = c.device.create_image(&ici, None).map_err(|e| format!("vkCreateImage {key:?}: {e}"))?;
        let req = c.device.get_image_memory_requirements(image);
        let memory = c.alloc(req, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        c.device.bind_image_memory(image, memory, 0).map_err(|e| e.to_string())?;
        let last_used = super::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
        Ok(Image { key, image, memory, views: Default::default(), gpu_written: false, uploaded_hash: 0, validated: None, upload_batch: u64::MAX, storage_ok, tile_index: None, cpu_checked: u64::MAX, pitch: 0, guest_stale: false, stale_stamp: 0, reuse_checked: 0, edges: None, last_write: 0, last_write_frame: 0, meta: None, meta_dirty: false, meta_fill: None, meta_listed: false, guest_newer: false, bytes: req.size, last_used })
    }
}

/// An evicted image kept for reuse by a later image with the same creation
/// parameters (`Renderer::image_pool`): its views stay valid.
#[derive(Default)]
pub struct PooledImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    views: super::render::FastMap<(vk::Format, u16, bool, u32, u32), vk::ImageView>,
    bytes: u64,
    storage_ok: bool,
}

impl PooledImage {
    pub fn of(img: Image) -> PooledImage {
        PooledImage { image: img.image, memory: img.memory, views: img.views, bytes: img.bytes, storage_ok: img.storage_ok }
    }

    /// (The GPU no longer uses it.)
    pub fn destroy(self) {
        let c = ctx();
        unsafe {
            for (_, v) in self.views {
                c.device.destroy_image_view(v, None);
            }
            c.device.destroy_image(self.image, None);
        }
        c.free(self.memory);
    }
}

/// A pooled image as a new image for `key` (same creation parameters): the
/// state of a newly created one; its contents are undefined until cleared.
pub fn reuse_image(key: ImageKey, p: PooledImage) -> Image {
    let last_used = super::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
    Image { key, image: p.image, memory: p.memory, views: p.views, gpu_written: false, uploaded_hash: 0, validated: None, upload_batch: u64::MAX, storage_ok: p.storage_ok, tile_index: None, cpu_checked: u64::MAX, pitch: 0, guest_stale: false, stale_stamp: 0, reuse_checked: 0, edges: None, last_write: 0, last_write_frame: 0, meta: None, meta_dirty: false, meta_fill: None, meta_listed: false, guest_newer: false, bytes: p.bytes, last_used }
}

pub fn aspect(f: vk::Format) -> vk::ImageAspectFlags {
    match f {
        vk::Format::D16_UNORM | vk::Format::D32_SFLOAT => vk::ImageAspectFlags::DEPTH,
        vk::Format::D32_SFLOAT_S8_UINT | vk::Format::D24_UNORM_S8_UINT => vk::ImageAspectFlags::DEPTH,
        _ => vk::ImageAspectFlags::COLOR,
    }
}

impl Image {
    /// View with a (compatible) format, T# dst_sel swizzle and dimensionality.
    /// `base_layer`/`layer_count` select the viewed slices (array views).
    pub fn view(&mut self, fmt: vk::Format, dst_sel: u16, view_type: vk::ImageViewType, storage: bool, base_layer: u32, layer_count: u32) -> Result<vk::ImageView, String> {
        self.view_levels(fmt, dst_sel, view_type, storage, base_layer, layer_count, 0, u32::MAX)
    }

    /// `view` of mip levels [base_level, base_level + level_count) (clamped
    /// to the image's levels; storage views see one level).
    #[allow(clippy::too_many_arguments)]
    pub fn view_levels(&mut self, fmt: vk::Format, dst_sel: u16, view_type: vk::ImageViewType, storage: bool, base_layer: u32, layer_count: u32, base_level: u32, level_count: u32) -> Result<vk::ImageView, String> {
        let base_level = base_level.min(self.key.levels.max(1) - 1);
        let level_count = if storage { 1 } else { level_count.clamp(1, self.key.levels.max(1) - base_level) };
        self.last_used = super::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
        let total = if self.key.ty3d { 1 } else { self.key.layers };
        let base_layer = base_layer.min(total - 1);
        let is_cube = matches!(view_type, vk::ImageViewType::CUBE | vk::ImageViewType::CUBE_ARRAY);
        if is_cube && total < 6 {
            return Err(format!("cube view of an image with {total} layers"));
        }
        // Cube views need whole cubes inside the image.
        let base_layer = if is_cube { (base_layer / 6 * 6).min(total / 6 * 6 - 6) } else { base_layer };
        let layer_count = match view_type {
            vk::ImageViewType::TYPE_2D | vk::ImageViewType::TYPE_1D | vk::ImageViewType::TYPE_3D => 1,
            vk::ImageViewType::CUBE => 6,
            vk::ImageViewType::CUBE_ARRAY => (layer_count / 6 * 6).clamp(6, (total - base_layer) / 6 * 6),
            _ => layer_count.clamp(1, total - base_layer),
        };
        let key = (fmt, (if storage { 0 } else { dst_sel }) ^ ((view_type.as_raw() as u16) << 12), storage, base_layer, layer_count | base_level << 16 | level_count << 24);
        if let Some(&v) = self.views.get(&key) {
            return Ok(v);
        }
        let c = ctx();
        let sel = |s: u16| match s {
            0 => vk::ComponentSwizzle::ZERO,
            1 => vk::ComponentSwizzle::ONE,
            4 => vk::ComponentSwizzle::R,
            5 => vk::ComponentSwizzle::G,
            6 => vk::ComponentSwizzle::B,
            7 => vk::ComponentSwizzle::A,
            _ => vk::ComponentSwizzle::IDENTITY,
        };
        let comps = if storage || dst_sel == 0 || is_depth(fmt) {
            vk::ComponentMapping::default()
        } else {
            vk::ComponentMapping { r: sel(dst_sel & 7), g: sel((dst_sel >> 3) & 7), b: sel((dst_sel >> 6) & 7), a: sel((dst_sel >> 9) & 7) }
        };
        let view_usage = if storage {
            vk::ImageUsageFlags::STORAGE
        } else if is_depth(fmt) {
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        } else {
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT
        };
        let mut usage_ci = vk::ImageViewUsageCreateInfo::default().usage(view_usage);
        let vci = vk::ImageViewCreateInfo::default()
            .image(self.image)
            .view_type(view_type)
            .format(fmt)
            .components(comps)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect(fmt),
                base_mip_level: base_level,
                level_count,
                base_array_layer: base_layer,
                layer_count,
            })
            .push_next(&mut usage_ci);
        let v = unsafe { c.device.create_image_view(&vci, None) }.map_err(|e| format!("vkCreateImageView {fmt:?} {view_type:?}: {e}"))?;
        self.views.insert(key, v);
        Ok(v)
    }
}

/// Bits per element for upload sizing.
pub fn element_bits(dfmt: u8) -> (u32, u32) {
    let (bytes, block) = desc::format_bytes(dfmt);
    (bytes * 8, block)
}

/// Mip levels our images hold for a T#: levels 0..=LAST_LEVEL of tiled
/// 2D, cube and array textures (TLG_MIPS=0: the base level only, as before);
/// 3D, linear and MSAA textures keep their base level.
pub fn texture_levels(t: &Tsharp) -> u32 {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("TLG_MIPS").map(|v| v != "0").unwrap_or(true)) {
        return 1;
    }
    let linear = tiling::tile_mode(t.tiling_index).array == tiling::ArrayMode::Linear;
    if linear || matches!(t.ty, desc::T_3D | desc::T_2D_MSAA | desc::T_2D_MSAA_ARRAY) {
        return 1;
    }
    let full = 32 - t.width.max(t.height).leading_zeros();
    (t.last_level + 1).clamp(1, full)
}

/// Guest layout of the levels `texture_levels` selects (`layers` slices).
pub fn texture_layout(t: &Tsharp, layers: u32) -> Vec<tiling::MipLevel> {
    let (bpp, block) = element_bits(t.dfmt);
    let levels = texture_levels(t);
    // (A lone base level keeps its descriptor height: POW2_PAD matters for
    // the offsets of further levels only.)
    tiling::mip_levels(t.width, t.height, t.pitch, block, bpp, t.tiling_index, t.pow2_pad && levels > 1, levels, layers, t.ty == desc::T_CUBE)
}

/// Linearizes a texture's levels (`texture_levels`, all layers each) from
/// guest memory: per level, per layer, tightly packed rows of elements.
pub fn read_texture(t: &Tsharp, layers: u32) -> Vec<u8> {
    let (bpp, _) = element_bits(t.dfmt);
    let eb = bpp as usize / 8;
    let lv = texture_layout(t, layers);
    let total: usize = lv.iter().map(|l| l.width as usize * l.height as usize * eb * layers as usize).sum();
    let mut out = vec![0u8; total];
    let mut pos = 0;
    let mut src = Vec::new();
    // Volumes in thick tile modes: groups of slices share micro tiles.
    let thick = tiling::thickness(t.tiling_index);
    if thick > 1 && t.ty == desc::T_3D {
        if let Some(l) = lv.first() {
            if t.tiling_index == 19 {
                let groups = layers.div_ceil(thick) as u64;
                src.resize((l.slice_bytes * thick as u64 * groups) as usize, 0);
                copy_from_guest(&mut src, t.base + l.offset);
                tiling::detile_thick_1d(&src, &mut out, l.width, l.height, layers, l.pitch, l.pheight, bpp, thick);
                return out;
            }
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                crate::warn!("volume texture {:#x} {}x{}x{} in tile mode {} (2D/3D thick) is detiled as thin slices: not implemented", t.base, t.width, t.height, layers, t.tiling_index);
            }
        }
    }
    for l in &lv {
        let per_layer = l.width as usize * l.height as usize * eb;
        src.resize(l.slice_bytes as usize, 0);
        for s in 0..layers as u64 {
            copy_from_guest(&mut src, t.base + l.offset + l.slice_bytes * s);
            tiling::detile_slice(&src, &mut out[pos..pos + per_layer], l.width, l.height, l.pitch, l.pheight, bpp, l.mode);
            pos += per_layer;
        }
    }
    out
}

/// Hash of guest memory [addr, addr+n) (at most 64 MiB), read in place
/// (unmapped bytes count as zeros) for texture change detection.
pub fn hash_guest(addr: u64, n: u64) -> u64 {
    let n = n.min(64 << 20);
    let end = addr.saturating_add(n);
    if end < addr || !crate::mem::is_guest(addr, end - addr) {
        return hash_slice(n, &[]);
    }
    // The lock keeps the mappings stable while they are read.
    let st = crate::mem::MEM.lock();
    let mut h = n;
    let mut a = addr;
    while a < end {
        match st.find(a) {
            Some(v) if v.kind != crate::mem::VmaKind::Reserved => {
                let e = v.end().min(end);
                h = hash_slice(h, unsafe { std::slice::from_raw_parts(a as *const u8, (e - a) as usize) });
                a = e;
            }
            _ => {
                let next = st.vmas.range(a..).next().map(|(_, v)| v.start).unwrap_or(end).min(end).max(a + 1);
                h = hash_slice(h ^ (next - a).rotate_left(32), &[]);
                a = next;
            }
        }
    }
    h
}

const P1: u64 = 0x9E37_79B1_85EB_CA87;
const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const P3: u64 = 0x1656_67B1_9E37_79F9;
const P4: u64 = 0x85EB_CA77_C2B2_AE63;
const P5: u64 = 0x27D4_EB2F_1656_67C5;

fn round64(acc: u64, x: u64) -> u64 {
    acc.wrapping_add(x.wrapping_mul(P2)).rotate_left(31).wrapping_mul(P1)
}

/// 64-bit hash of a byte slice (xxHash64's construction: four independent
/// multiply-rotate lanes over 32-byte stripes, then a tail and avalanche),
/// seeded so segments can be chained.
pub fn hash_slice(seed: u64, d: &[u8]) -> u64 {
    let mut chunks = d.chunks_exact(32);
    let mut h = if d.len() >= 32 {
        let mut v = [seed.wrapping_add(P1).wrapping_add(P2), seed.wrapping_add(P2), seed, seed.wrapping_sub(P1)];
        for c in &mut chunks {
            for (i, lane) in v.iter_mut().enumerate() {
                *lane = round64(*lane, u64::from_le_bytes(c[i * 8..i * 8 + 8].try_into().unwrap()));
            }
        }
        let mut h = v[0].rotate_left(1).wrapping_add(v[1].rotate_left(7)).wrapping_add(v[2].rotate_left(12)).wrapping_add(v[3].rotate_left(18));
        for lane in v {
            h = (h ^ round64(0, lane)).wrapping_mul(P1).wrapping_add(P4);
        }
        h
    } else {
        seed.wrapping_add(P5)
    };
    h = h.wrapping_add(d.len() as u64);
    let mut words = chunks.remainder().chunks_exact(8);
    for c in &mut words {
        h = (h ^ round64(0, u64::from_le_bytes(c.try_into().unwrap()))).rotate_left(27).wrapping_mul(P1).wrapping_add(P4);
    }
    for &b in words.remainder() {
        h = (h ^ (b as u64).wrapping_mul(P5)).rotate_left(11).wrapping_mul(P1);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(P2);
    h ^= h >> 29;
    h = h.wrapping_mul(P3);
    h ^ (h >> 32)
}

pub fn texture_bytes(t: &Tsharp, layers: u32) -> u64 {
    let (bpp, block) = element_bits(t.dfmt);
    let levels = texture_levels(t);
    // (Memoized: a binding needs it several times, and the same textures are
    // bound every frame.)
    let key = [t.width, t.height, t.pitch, t.dfmt as u32 | (t.tiling_index as u32) << 8 | (t.pow2_pad as u32) << 16 | (t.ty as u32) << 24, levels, layers];
    thread_local! {
        static MEMO: std::cell::RefCell<Vec<([u32; 6], u64)>> = std::cell::RefCell::new(vec![([u32::MAX; 6], 0); 64]);
    }
    let slot = (key.iter().fold(0u64, |h, &w| (h ^ w as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 58) as usize;
    if let Some(v) = MEMO.with(|m| m.borrow().get(slot).filter(|e| e.0 == key).map(|e| e.1)) {
        return v;
    }
    let v = texture_bytes_uncached(t, layers, bpp, block, levels);
    MEMO.with(|m| m.borrow_mut()[slot] = (key, v));
    v
}

fn texture_bytes_uncached(t: &Tsharp, layers: u32, bpp: u32, block: u32, levels: u32) -> u64 {
    let mut end = 0;
    tiling::for_each_mip_level(t.width, t.height, t.pitch, block, bpp, t.tiling_index, t.pow2_pad && levels > 1, levels, layers, t.ty == desc::T_CUBE, |l| {
        // (A lone base level: the slices actually used.)
        end = l.offset + l.slice_bytes * if levels > 1 { l.slices } else { layers } as u64;
    });
    end
}

pub struct Samplers {
    map: HashMap<[u32; 4], vk::Sampler>,
}

impl Samplers {
    pub fn new() -> Samplers {
        Samplers { map: HashMap::new() }
    }
    pub fn get(&mut self, s: &Ssharp) -> Result<vk::Sampler, String> {
        if let Some(&v) = self.map.get(&s.0) {
            return Ok(v);
        }
        let c = ctx();
        let filt = |f: u32| if f == 0 { vk::Filter::NEAREST } else { vk::Filter::LINEAR };
        let aniso = s.min_filter() >= 2 && s.max_aniso() > 0;
        let sci = vk::SamplerCreateInfo::default()
            .mag_filter(filt(s.mag_filter()))
            .min_filter(filt(s.min_filter() & 1))
            .mipmap_mode(if s.mip_filter() == 2 { vk::SamplerMipmapMode::LINEAR } else { vk::SamplerMipmapMode::NEAREST })
            .address_mode_u(desc::vk_address_mode(s.clamp(0)))
            .address_mode_v(desc::vk_address_mode(s.clamp(1)))
            .address_mode_w(desc::vk_address_mode(s.clamp(2)))
            .mip_lod_bias(s.lod_bias())
            .anisotropy_enable(aniso)
            .max_anisotropy((1u32 << s.max_aniso()).min(16) as f32)
            .compare_enable(s.compare_func() != 0)
            .compare_op(desc::vk_compare(s.compare_func()))
            .min_lod(s.min_lod())
            .max_lod(if s.mip_filter() == 0 { 0.25 } else { s.max_lod().max(s.min_lod()) })
            .border_color(match s.border_type() {
                0 => vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
                1 => vk::BorderColor::FLOAT_OPAQUE_BLACK,
                _ => vk::BorderColor::FLOAT_OPAQUE_WHITE,
            })
            .unnormalized_coordinates(false);
        let v = unsafe { c.device.create_sampler(&sci, None) }.map_err(|e| format!("vkCreateSampler: {e}"))?;
        self.map.insert(s.0, v);
        Ok(v)
    }
}
