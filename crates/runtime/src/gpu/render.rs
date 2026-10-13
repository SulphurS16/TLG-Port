//! Vulkan backend: executes draws and dispatches recorded by the command
//! processor using translated shaders.

use super::cp::{CommandProcessor, Hooks, Regs};
use super::desc::{self, Tsharp};
use super::tiling;
use super::resources::{self, Image, ImageKey, Ring, Samplers};
use super::shaders::{self, Prepared, ResolvedRes, StageRegs};
use super::vk::ctx;
use ash::vk;
use gcn::translate::{BindingKind, Stage, TessRole, TessSpec};
use parking_lot::Mutex;
use std::collections::HashMap;

/// GPU work just recorded (or about to be) writes memory and is not ordered
/// before later commands: the next consumer starts with a full barrier.
/// (`barrier_needed` alone is also set after dispatches, whose own buffer
/// accesses are tracked instead: see `tune dispatch_hazards`.)
macro_rules! unordered {
    ($s:expr) => {{
        $s.barrier_needed = true;
        $s.unordered_gen = BARRIERS.load(std::sync::atomic::Ordering::Relaxed);
        $s.unordered_line = line!();
    }};
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Attachment {
    key: ImageKey,
    view_format: vk::Format,
    /// Array layer rendered (CB_COLOR*_VIEW.SLICE_START).
    layer: u32,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct GfxKey {
    vs: u64,
    ps: Option<u64>,
    rect: bool,
    gs_params: u32,
    colors: smallvec::SmallVec<[vk::Format; 8]>,
    blend: smallvec::SmallVec<[u32; 8]>,
    masks: smallvec::SmallVec<[u32; 8]>,
    depth: vk::Format,
    stencil: bool,
    topology_class: u8,
    polygon: u8,
    depth_clamp: bool,
    /// Emulated tessellation: VGT_TF_PARAM and the factor clamp
    /// (VGT_HOS_MIN/MAX_TESS_LEVEL bits) of the generated control stage.
    tess: Option<(u32, u32, u32)>,
}

/// A tessellated draw whose LS and HS stages have been dispatched (see
/// `Renderer::tess_stages`).
struct TessDraw {
    patches: u32,
    tf_param: u32,
    min: f32,
    max: f32,
    /// The domain shader's specialization.
    ds: TessSpec,
}

struct Pipe {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
}

/// Pipeline cache persisted in `local-data/pipeline-cache.bin`: pipelines
/// the driver compiled in earlier runs load from it (the title's largest
/// compute shader alone takes ~38 s to compile; loading compiled ~50 s of
/// pipelines). The driver validates the header and ignores foreign data.
struct PipelineCacheFile {
    cache: vk::PipelineCache,
    path: std::path::PathBuf,
    dirty: bool,
    saved: std::time::Instant,
}

static PIPELINE_CACHE: parking_lot::Mutex<Option<PipelineCacheFile>> = parking_lot::Mutex::new(None);
static PIPELINE_CACHE_PATH: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();

/// Where the pipeline cache lives (set before `init`).
pub fn set_pipeline_cache_path(p: std::path::PathBuf) {
    PIPELINE_CACHE_PATH.set(p).ok();
}

/// The persistent pipeline cache (created on first use), or null without one.
fn pipeline_cache() -> vk::PipelineCache {
    let mut g = PIPELINE_CACHE.lock();
    if let Some(p) = g.as_ref() {
        return p.cache;
    }
    let Some(path) = PIPELINE_CACHE_PATH.get().cloned() else { return vk::PipelineCache::null() };
    let data = std::fs::read(&path).unwrap_or_default();
    let c = ctx();
    let create = |d: &[u8]| unsafe { c.device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default().initial_data(d), None) };
    let cache = match create(&data).or_else(|_| create(&[])) {
        Ok(cache) => cache,
        Err(e) => {
            crate::warn!("pipeline cache unavailable: {e}");
            return vk::PipelineCache::null();
        }
    };
    crate::info!("pipeline cache {} ({} KiB loaded)", path.display(), data.len() >> 10);
    *LAST_SAVED.lock() = Some((data.len(), resources::hash_slice(0, &data)));
    *g = Some(PipelineCacheFile { cache, path, dirty: false, saved: std::time::Instant::now() });
    cache
}

/// Length and hash of the pipeline cache data last loaded or written.
static LAST_SAVED: parking_lot::Mutex<Option<(usize, u64)>> = parking_lot::Mutex::new(None);
/// A background save is running.
static SAVING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Serializes the cache and writes it unless the data is what was last
/// loaded or written (pipelines served from the cache also mark it dirty).
fn write_pipeline_cache(cache: vk::PipelineCache, path: &std::path::Path) {
    let data = match unsafe { ctx().device.get_pipeline_cache_data(cache) } {
        Ok(d) => d,
        Err(e) => {
            crate::warn!("pipeline cache: {e}");
            return;
        }
    };
    let id = (data.len(), resources::hash_slice(0, &data));
    if *LAST_SAVED.lock() == Some(id) {
        return;
    }
    let tmp = path.with_extension("tmp");
    if std::fs::write(&tmp, &data).and_then(|_| std::fs::rename(&tmp, path)).is_ok() {
        *LAST_SAVED.lock() = Some(id);
        crate::info!("pipeline cache saved ({} KiB)", data.len() >> 10);
    }
}

/// Marks the pipeline cache as holding unsaved pipelines.
fn pipeline_cache_dirty() {
    if let Some(p) = PIPELINE_CACHE.lock().as_mut() {
        p.dirty = true;
    }
}

/// Writes the pipeline cache when pipelines were created and at least
/// `min_age` passed since the last write. Called at flips (GPU thread) and
/// at exit (zero: written on this thread). From flips the serialization and
/// the write run on a background thread: done at a flip they held up
/// presentation 60-100 ms each time (a 47 MiB cache), the worst frame of
/// the Dream in every test run.
pub fn save_pipeline_cache(min_age: std::time::Duration) {
    use std::sync::atomic::Ordering;
    let Some(mut g) = PIPELINE_CACHE.try_lock_for(std::time::Duration::from_millis(200)) else { return };
    let Some(p) = g.as_mut() else { return };
    if !p.dirty || p.saved.elapsed() < min_age {
        return;
    }
    if !min_age.is_zero() && SAVING.swap(true, Ordering::SeqCst) {
        return; // (still saving: the next flip tries again)
    }
    p.dirty = false;
    p.saved = std::time::Instant::now();
    let (cache, path) = (p.cache, p.path.clone());
    drop(g);
    if min_age.is_zero() {
        write_pipeline_cache(cache, &path);
        return;
    }
    let spawned = std::thread::Builder::new().name("pipeline-cache".into()).spawn(move || {
        write_pipeline_cache(cache, &path);
        SAVING.store(false, Ordering::SeqCst);
    });
    if spawned.is_err() {
        SAVING.store(false, Ordering::SeqCst);
    }
}

/// What a CPU-write epoch applies to (see `super::CPU_EPOCH`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum EpochKind {
    Graphics,
    Compute,
    Both,
}

/// Color attachments of a draw (at most 8, inline).
type Colors = smallvec::SmallVec<[Option<Attachment>; 8]>;

/// Dynamic graphics state last recorded in the current command buffer
/// (floats as bits): draws only record what changed.
#[derive(Clone, Copy, PartialEq, Default)]
struct DynState {
    viewport: [u32; 6],
    scissor: [i32; 4],
    cull: u32,
    front_cw: bool,
    topology: i32,
    restart: bool,
    depth_test: bool,
    depth_write: bool,
    depth_op: i32,
    bounds_test: bool,
    bounds: [u32; 2],
    stencil_test: bool,
    stencil_ops: [[i32; 4]; 2],
    stencil_ref: [u32; 2],
    stencil_compare: [u32; 2],
    stencil_write: [u32; 2],
    bias_enable: bool,
    bias: [u32; 3],
    blend_constants: [u32; 4],
}

struct Rendering {
    colors: Colors,
    depth: Option<Attachment>,
    extent: (u32, u32),
}

/// Buffer bindings at least this large are accessed in place (zero-copy) in
/// guest memory imported into Vulkan instead of being copied through the
/// rings (titles bind multi-GB ranges as scatter destinations).
const ZC_MIN_BYTES: u64 = 16 << 20;
/// Largest written buffer binding treated as a write of whatever surfaces it
/// overlaps (render-target reload, HTILE/CMASK checks); larger bindings are
/// sparse scatter destinations.
const SURFACE_WRITE_MAX: u64 = 64 << 20;

/// Number of batches that can be in flight (each with its own command
/// buffer, fence, descriptor pool and ring segment).
const SLOTS: usize = 6;

struct Slot {
    cmd: vk::CommandBuffer,
    /// The batch's prologue (`Renderer::prologue_cmd`), submitted before `cmd`.
    pro: vk::CommandBuffer,
    fence: vk::Fence,
    desc_pool: vk::DescriptorPool,
    /// Timestamp queries of the batch (TLG_GPU_PROFILE; null otherwise).
    queries: vk::QueryPool,
    /// Timestamps at the start and end of the batch (GPU execution time).
    exec: vk::QueryPool,
    /// Sequence number of the batch last submitted from this slot (0 = none).
    seq: u64,
}

/// Fast non-cryptographic hasher for the renderer's hot maps (keys are
/// addresses and image descriptions, not attacker-controlled).
#[derive(Default, Clone, Copy)]
pub struct FxHasher(u64);

impl FxHasher {
    #[inline]
    fn add(&mut self, w: u64) {
        self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

impl std::hash::Hasher for FxHasher {
    fn write(&mut self, bytes: &[u8]) {
        for c in bytes.chunks(8) {
            let mut b = [0u8; 8];
            b[..c.len()].copy_from_slice(c);
            self.add(u64::from_le_bytes(b));
        }
    }
    fn write_u8(&mut self, v: u8) {
        self.add(v as u64);
    }
    fn write_u16(&mut self, v: u16) {
        self.add(v as u64);
    }
    fn write_u32(&mut self, v: u32) {
        self.add(v as u64);
    }
    fn write_u64(&mut self, v: u64) {
        self.add(v);
    }
    fn write_usize(&mut self, v: usize) {
        self.add(v as u64);
    }
    fn write_i32(&mut self, v: i32) {
        self.add(v as u32 as u64);
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

pub type FastMap<K, V> = HashMap<K, V, std::hash::BuildHasherDefault<FxHasher>>;
pub type FastSet<K> = std::collections::HashSet<K, std::hash::BuildHasherDefault<FxHasher>>;

/// VRAM mirrors of guest memory for read-only vertex/index data: draws fetch
/// from device memory instead of over PCIe from imported guest memory.
/// Windows start at 256 MiB boundaries and span 320 MiB, so a binding of up
/// to `VCACHE_MAX` lies in the window of its base (same offsets as guest
/// memory, no extents to merge). Pages (4 KiB) are copied in on first use
/// and after writes: GPU and CP writes invalidate pages as they are recorded,
/// CPU writes are found through the OS write watch once per epoch and 64 KiB
/// block. Windows are allocated on demand within `VCACHE_BUDGET`.
struct VCache {
    windows: FastMap<u64, VWindow>,
    bytes: u64,
    /// Bytes copied in, bindings served (TLG_PERF).
    copied: u64,
    hits: u64,
}

struct VWindow {
    lo: u64,
    hi: u64,
    buf: vk::Buffer,
    mem: vk::DeviceMemory,
    /// Pages holding current guest data (bit per 4 KiB page).
    valid: Vec<u64>,
    /// Pages any binding has covered since the window was made (VRAM report).
    touched: Vec<u64>,
    /// Per 64 KiB block: epoch of the last CPU-write check, write-tracker
    /// stamp it established.
    checked: Vec<u64>,
    stamp: Vec<u64>,
    /// Frame of the last binding served (released under VRAM pressure).
    last_used: u64,
    /// Changes whenever pages of the window stop being current (unique
    /// across windows: `Renderer::vgen`), for `Renderer::vbind`.
    wgen: u64,
    /// Sparse windows (`Ctx::sparse`): `mem` is null, memory is bound block
    /// by block on first use, so a window costs what has been bound of it.
    sparse: Option<SparseMem>,
}

struct SparseMem {
    /// Block size (the buffer's memory alignment: 64 KiB on NVIDIA).
    block: u64,
    /// Bound blocks (bit per block) and their memory.
    resident: Vec<u64>,
    bound: Vec<(vk::DeviceMemory, u64)>,
}

impl VWindow {
    /// Device memory the window holds.
    fn bytes(&self) -> u64 {
        self.sparse.as_ref().map_or(self.hi - self.lo, |s| s.bound.len() as u64 * s.block)
    }

    /// The window's buffer and memory, to free once unused.
    fn retire(self) -> Retired {
        Retired::Window(self.buf, self.mem, self.sparse.map_or(Vec::new(), |s| s.bound))
    }
}

/// Mirror windows with memory bound block by block (`Ctx::sparse`;
/// TLG_SPARSE=0 turns sparse buffers off).
fn sparse_mirrors() -> bool {
    ctx().sparse
}

/// Device memory for sparse mirror windows: `SPARSE_CHUNK` allocations cut
/// into blocks, returned to the free list when a window is released; chunks
/// with no block in use are freed under VRAM pressure (`trim`).
struct BlockPool {
    block: u64,
    /// Memory type index (device-local, allowed for the window buffers).
    ty: u32,
    /// Chunks (by raw handle) and how many of their blocks are free.
    chunks: FastMap<u64, (vk::DeviceMemory, u64)>,
    free: Vec<(vk::DeviceMemory, u64)>,
}

const SPARSE_CHUNK: u64 = 16 << 20;

impl BlockPool {
    fn take(&mut self) -> Option<(vk::DeviceMemory, u64)> {
        if self.free.is_empty() {
            let req = vk::MemoryRequirements { size: SPARSE_CHUNK, alignment: self.block, memory_type_bits: 1 << self.ty };
            let mem = ctx().alloc(req, vk::MemoryPropertyFlags::DEVICE_LOCAL).ok()?;
            let n = SPARSE_CHUNK / self.block;
            self.chunks.insert(vk::Handle::as_raw(mem), (mem, n));
            self.free.extend((0..n).rev().map(|i| (mem, i * self.block)));
        }
        let b = self.free.pop()?;
        if let Some(c) = self.chunks.get_mut(&vk::Handle::as_raw(b.0)) {
            c.1 -= 1;
        }
        Some(b)
    }

    fn give(&mut self, blocks: impl IntoIterator<Item = (vk::DeviceMemory, u64)>) {
        for b in blocks {
            if let Some(c) = self.chunks.get_mut(&vk::Handle::as_raw(b.0)) {
                c.1 += 1;
            }
            self.free.push(b);
        }
    }

    /// Frees the chunks no window uses; returns the bytes freed.
    fn trim(&mut self) -> u64 {
        let n = SPARSE_CHUNK / self.block;
        let empty: FastSet<u64> = self.chunks.iter().filter(|(_, c)| c.1 == n).map(|(k, _)| *k).collect();
        if empty.is_empty() {
            return 0;
        }
        self.free.retain(|b| !empty.contains(&vk::Handle::as_raw(b.0)));
        for k in &empty {
            if let Some((m, _)) = self.chunks.remove(k) {
                ctx().free(m);
            }
        }
        empty.len() as u64 * SPARSE_CHUNK
    }
}

const VWIN_SHIFT: u32 = 28;
const VCACHE_MIN: u64 = 64 << 10;
const VCACHE_MAX: u64 = 64 << 20;
const VWIN_SIZE: u64 = (1 << VWIN_SHIFT) + VCACHE_MAX;
const VCACHE_BUDGET: u64 = 4 << 30;
/// Largest read-only copy `upload_guest_ro` puts in the BAR ring (bigger
/// ones would take a segment's room from many small ones).
const BAR_UPLOAD_MAX: u64 = 64 << 10;

/// VRAM pressure level from this process's device-local usage and budget
/// (`Ctx::vram`): 0 below 70 %, 1 below 85 %, 2 below 95 %, 3 above; 0
/// without budget information.
fn vram_pressure(usage: u64, budget: u64) -> u8 {
    if budget == 0 {
        return 0;
    }
    match usage.saturating_mul(100) / budget {
        0..70 => 0,
        70..85 => 1,
        85..95 => 2,
        _ => 3,
    }
}

/// Frames an evictable image may go unused before it is evicted, by
/// pressure level (from level 2 on, VRAM mirror windows too).
const EVICT_AGE: [u64; 4] = [5400, 600, 120, 30];

/// Largest CPU-sourced texture `tune evict_small` evicts (and the frames it
/// must have gone unused).
const SMALL_IMAGE_BYTES: u64 = 64 << 10;
const SMALL_IMAGE_AGE: u64 = 600;
const SMALL_PER_PASS: usize = 256;
/// Small images kept for reuse at most (`Renderer::image_pool`).
const IMAGE_POOL_MAX: usize = 512;
/// Image bounds (`Renderer::img_bases`) up to this are "near": a buffer
/// write is checked against them within this distance.
const IMG_NEAR_BYTES: u64 = 4 << 20;

/// Without VRAM pressure (levels 0 and 1), a housekeeping pass evicts at
/// most this many images / bytes (the oldest first) and destroys at most
/// `RETIRE_PER_PASS` retired objects: evicting 326 images (204 MiB) at once
/// held the GPU thread up 60-90 ms, a visible hitch in combat every few
/// seconds while an area's textures aged out (many came back within a
/// minute; the pressure-0 age was 1800 frames).
const EVICT_PER_PASS: (usize, u64) = (64, 128 << 20);
const RETIRE_PER_PASS: usize = 64;

/// `gpu_texture_levels` outcomes (TLG_PERF report): uploads, then declines
/// by reason (element size or 3D, pending write-back, layout, guest memory
/// binding, tile map, device ring, descriptor pool).
pub static GPU_MIP_STATS: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];
pub const GPU_MIP_STAT_NAMES: [&str; 8] = ["uploads", "element", "write-back", "layout", "binding", "tile map", "ring", "pool"];

fn mip_stat(i: usize) {
    GPU_MIP_STATS[i].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Images uploaded by `gpu_texture_levels` to verify (TLG_GPU_TEX_VERIFY).
static VERIFY_QUEUE: parking_lot::Mutex<Vec<(ImageKey, Tsharp, u32)>> = parking_lot::Mutex::new(Vec::new());

/// Shadow copies (`shadow_view`, `depth_shadow_view`) live at the guest
/// base with bit 62 or 63 flipped.
fn is_shadow(k: &ImageKey) -> bool {
    matches!(k.base >> 60, 0x4 | 0x8)
}

/// Images whose contents can be rebuilt when needed again: CPU-sourced
/// textures (re-uploaded from guest memory) and shadow copies (copied again
/// from their render targets). Images the GPU wrote are kept: their contents
/// may exist nowhere else.
fn evictable(k: &ImageKey, i: &Image) -> bool {
    (!i.gpu_written || is_shadow(k)) && !i.guest_stale
}

/// An indirect draw recorded with GPU-read arguments: their device copy
/// (buffer, offset), indexed (5 dwords) or not (4), the vertex stage's SGPR
/// data dwords that receive the base vertex and start instance, and the
/// index buffer's size in indices.
struct IndirectDraw {
    buf: vk::Buffer,
    off: u64,
    indexed: bool,
    vtx: Option<u32>,
    inst: Option<u32>,
    indices: u32,
}

/// Vulkan objects released by the caches, destroyed once the batches that
/// may use them have completed.
enum Retired {
    Image(vk::Image, vk::DeviceMemory, Vec<vk::ImageView>),
    /// A small evicted image for the pool once the GPU is done with it.
    Pooled(ImageKey, resources::PooledImage),
    Buffer(vk::Buffer, vk::DeviceMemory),
    /// A mirror window: its buffer, whole memory (null when sparse), the
    /// sparse blocks bound to it (back to the pool).
    Window(vk::Buffer, vk::DeviceMemory, Vec<(vk::DeviceMemory, u64)>),
    /// A persistent descriptor set (pool, set).
    Set(vk::DescriptorPool, vk::DescriptorSet),
}

/// `spans` minus the ranges `cut` (all half-open, in the same units).
fn subtract_ranges(spans: &[(u64, u64)], cut: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut out: Vec<(u64, u64)> = spans.to_vec();
    for &(c, d) in cut {
        let mut next = Vec::with_capacity(out.len() + 1);
        for (x, y) in out {
            if d <= x || y <= c {
                next.push((x, y));
                continue;
            }
            if x < c {
                next.push((x, c));
            }
            if d < y {
                next.push((d, y));
            }
        }
        out = next;
    }
    out
}

/// Bit helpers for `VWindow::valid`: set or clear bits [a, b).
fn bits_set(v: &mut [u64], a: usize, b: usize, on: bool) {
    let mut i = a;
    while i < b {
        let n = (64 - (i & 63)).min(b - i);
        let mask = if n == 64 { u64::MAX } else { ((1u64 << n) - 1) << (i & 63) };
        if on {
            v[i >> 6] |= mask;
        } else {
            v[i >> 6] &= !mask;
        }
        i += n;
    }
}

/// Runs [a, b) of clear bits within [lo, hi) (a word at a time).
fn bits_clear_runs(v: &[u64], lo: usize, hi: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = lo;
    while i < hi {
        // The next clear bit at or after i.
        let clear = !v[i >> 6] & (u64::MAX << (i & 63));
        if clear == 0 {
            i = (i | 63) + 1;
            continue;
        }
        let start = (i & !63) + clear.trailing_zeros() as usize;
        if start >= hi {
            break;
        }
        // The next set bit after it (or hi).
        let mut j = start;
        while j < hi {
            let set = v[j >> 6] & (u64::MAX << (j & 63));
            if set != 0 {
                j = (j & !63) + set.trailing_zeros() as usize;
                break;
            }
            j = (j | 63) + 1;
        }
        let end = j.min(hi);
        out.push((start, end));
        i = end;
    }
    out
}

/// Bytes of one element the engine's fill kernel stores for the value
/// `w` (four 32-bit floats) through a destination V# of data format `dfmt`,
/// number format `nfmt`: float32/float16/unorm8 formats; None when unknown,
/// inexact (16-bit floats), or dependent on whether stores apply the V#
/// destination swizzle (computed both ways).
fn fill_element(dfmt: u8, nfmt: u8, dst_sel: u16, w: &[u32; 4]) -> Option<Vec<u8>> {
    let f = w.map(f32::from_bits);
    let comp = |i: usize, swz: bool| -> f32 {
        if !swz {
            return f[i];
        }
        match (dst_sel >> (3 * i)) & 7 {
            0 => 0.0,
            1 => 1.0,
            s @ 4..=7 => f[s as usize - 4],
            _ => f[i],
        }
    };
    let half = |v: f32| -> Option<u16> {
        if v == 0.0 {
            return Some(if v.is_sign_negative() { 0x8000 } else { 0 });
        }
        let b = v.to_bits();
        let e = ((b >> 23) & 0xFF) as i32 - 127;
        ((-14..=15).contains(&e) && b & 0x1FFF == 0).then(|| (((b >> 16) & 0x8000) | (((e + 15) as u32) << 10) | ((b >> 13) & 0x3FF)) as u16)
    };
    let unorm8 = |v: f32| -> u8 { (v.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8 };
    // 32-bit integer components store the value's bits (constants 0 and 1
    // as integers).
    let bits = |i: usize, swz: bool| -> u32 {
        if !swz {
            return w[i];
        }
        match (dst_sel >> (3 * i)) & 7 {
            0 => 0,
            1 => 1,
            s @ 4..=7 => w[s as usize - 4],
            _ => w[i],
        }
    };
    let element = |swz: bool| -> Option<Vec<u8>> {
        let mut out = Vec::new();
        match (dfmt, nfmt) {
            (4 | 11 | 14, 4 | 5) => {
                for i in 0..[1, 2, 4][((dfmt == 11) as usize) + 2 * ((dfmt == 14) as usize)] {
                    out.extend_from_slice(&bits(i, swz).to_le_bytes());
                }
            }
            (4 | 11 | 14, 7) => {
                for i in 0..[1, 2, 4][((dfmt == 11) as usize) + 2 * ((dfmt == 14) as usize)] {
                    out.extend_from_slice(&comp(i, swz).to_bits().to_le_bytes());
                }
            }
            (2 | 5 | 12, 7) => {
                for i in 0..[1, 2, 4][((dfmt == 5) as usize) + 2 * ((dfmt == 12) as usize)] {
                    out.extend_from_slice(&half(comp(i, swz))?.to_le_bytes());
                }
            }
            (10, 0) => {
                for i in 0..4 {
                    out.push(unorm8(comp(i, swz)));
                }
            }
            _ => return None,
        }
        Some(out)
    };
    let (a, b) = (element(false)?, element(true)?);
    (a == b).then_some(a)
}

/// Clear color of an image of `format` whose texels hold the bytes `e`.
/// The clear value of an integer-format image whose texels are the element
/// `e` (any bit pattern clears exactly).
fn int_clear_for(format: vk::Format, e: &[u8]) -> Option<vk::ClearColorValue> {
    let bytes = match (format, e.len()) {
        (vk::Format::R32_UINT | vk::Format::R32_SINT, 4) | (vk::Format::R32G32_UINT, 8) | (vk::Format::R32G32B32A32_UINT, 16) => 4,
        (vk::Format::R16_UINT, 2) | (vk::Format::R16G16_UINT, 4) | (vk::Format::R16G16B16A16_UINT, 8) => 2,
        (vk::Format::R8_UINT, 1) | (vk::Format::R8G8_UINT, 2) | (vk::Format::R8G8B8A8_UINT | vk::Format::R8G8B8A8_SINT, 4) => 1,
        _ => return None,
    };
    let mut v = [0u32; 4];
    for (i, c) in e.chunks(bytes).enumerate().take(4) {
        v[i] = c.iter().rev().fold(0, |a, &b| a << 8 | b as u32);
    }
    Some(vk::ClearColorValue { uint32: v })
}

fn clear_color_for(format: vk::Format, e: &[u8]) -> Option<vk::ClearColorValue> {
    if let Some(c) = int_clear_for(format, e) {
        return Some(c);
    }
    let h = |i: usize| -> f32 {
        let v = u16::from_le_bytes([e[2 * i], e[2 * i + 1]]) as u32;
        let (s, ex, m) = (v >> 15, (v >> 10) & 0x1F, v & 0x3FF);
        if ex == 0 && m == 0 {
            return if s == 1 { -0.0 } else { 0.0 };
        }
        // (fill_element produces normal halves or zero only)
        f32::from_bits((s << 31) | ((ex + 127 - 15) << 23) | (m << 13))
    };
    let fl = |i: usize| f32::from_le_bytes(e[4 * i..4 * i + 4].try_into().unwrap());
    let u = |i: usize| e[i] as f32 / 255.0;
    let c = match (format, e.len()) {
        (vk::Format::R32_SFLOAT, 4) => [fl(0), 0.0, 0.0, 0.0],
        (vk::Format::R32G32_SFLOAT, 8) => [fl(0), fl(1), 0.0, 0.0],
        (vk::Format::R32G32B32A32_SFLOAT, 16) => [fl(0), fl(1), fl(2), fl(3)],
        (vk::Format::R16_SFLOAT, 2) => [h(0), 0.0, 0.0, 0.0],
        (vk::Format::R16G16_SFLOAT, 4) => [h(0), h(1), 0.0, 0.0],
        (vk::Format::R16G16B16A16_SFLOAT, 8) => [h(0), h(1), h(2), h(3)],
        (vk::Format::R8G8B8A8_UNORM, 4) => [u(0), u(1), u(2), u(3)],
        (vk::Format::B8G8R8A8_UNORM, 4) => [u(2), u(1), u(0), u(3)],
        _ => return None,
    };
    Some(vk::ClearColorValue { float32: c })
}

/// Disjoint, merged byte intervals [lo, hi), plus a small cache of recent
/// insertions (intervals only grow, so an exact repeat is covered).
#[derive(Default)]
pub struct Ranges(std::collections::BTreeMap<u64, u64>, [(u64, u64); 16]);

impl Ranges {
    pub const fn new() -> Ranges {
        Ranges(std::collections::BTreeMap::new(), [(0, 0); 16])
    }
    pub fn insert(&mut self, lo: u64, hi: u64) {
        let hi = hi.max(lo + 1);
        let slot = ((lo ^ hi) >> 4) as usize & 15;
        if self.1[slot] == (lo, hi) {
            return;
        }
        self.1[slot] = (lo, hi);
        let start = match self.0.range(..=lo).next_back() {
            Some((&s, &e)) if e >= lo => {
                if e >= hi {
                    return; // already covered (the common case: reads repeat)
                }
                s
            }
            _ => lo,
        };
        // Absorb the intervals starting in (start, hi]; extend the one at
        // `start` in place.
        let mut end = hi;
        while let Some((&s, &e)) = self.0.range(start + 1..=end).next() {
            self.0.remove(&s);
            end = end.max(e);
        }
        let e = self.0.entry(start).or_insert(end);
        *e = (*e).max(end);
    }
    pub fn overlaps(&self, lo: u64, hi: u64) -> bool {
        self.0.range(..hi).next_back().is_some_and(|(_, &e)| e > lo)
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// What CP threads check before they access guest memory directly, without
/// taking the renderer lock (`tune cp_sync_view`): the current batch's
/// pending GPU writes (write-backs at its completion, in-place stores) and
/// in-place reads, entered when the renderer records them and cleared once
/// the batch's completion is queued (where `wait_overlapping` and the CP's
/// completion checks see them); and the CP's direct writes, whose mirror and
/// upload-cache invalidations the renderer applies before its next command
/// (`drain_cp_writes`). A CP thread's own earlier commands are always
/// entered; another queue's command being recorded meanwhile is unordered
/// with the access, as it is on the hardware without a wait between them.
pub struct CpView {
    pending: Ranges,
    reads: Ranges,
    cp_writes: Vec<(u64, u64)>,
}

pub static CP_VIEW: parking_lot::Mutex<CpView> = parking_lot::Mutex::new(CpView { pending: Ranges::new(), reads: Ranges::new(), cp_writes: Vec::new() });
static CP_WRITES_QUEUED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn cp_view_on() -> bool {
    crate::tune::get(crate::tune::CP_SYNC_VIEW) != 0
}

/// Work performed (in submission order) when a batch completes.
struct Completion {
    seq: u64,
    fence: vk::Fence,
    /// The batch's start/end timestamps (null: none).
    exec: vk::QueryPool,
    ring_ptr: *mut u8,
    writebacks: Vec<(u64, u64, u64)>,
    /// Per write-back: the host-ring offset and dwords of its page bitmap
    /// when only marked pages were copied out (and are written back).
    wb_bits: Vec<Option<(u64, u32)>>,
    /// Where the write-backs' data is on the device: (guest base, buffer,
    /// offset, size), for later batches to forward from (see `upload_guest`).
    dev_writes: Vec<(u64, vk::Buffer, u64, u64)>,
    /// Write-backs of data the VRAM mirror holds (its pages stay current):
    /// the OS write watch forgets the write-back's own writes.
    mirror_writes: Vec<(u64, u64)>,
    /// Guest ranges written in place by zero-copy bindings (ordering only).
    zc_writes: Vec<(u64, u64)>,
    /// Guest ranges the batch reads in place (CP writes must wait for them).
    zc_reads: Ranges,
    /// Per write-back: whether it is of data the VRAM mirror holds (`lazywb`).
    wb_mirror: Vec<bool>,
    actions: Vec<super::cp::EopAction>,
    submitted: std::time::Instant,
    /// Timestamp query pool and the label of each written timestamp.
    profile: Option<(vk::QueryPool, Vec<String>)>,
    /// Bounds of `writebacks` and `actions` (set by `push_completion`): the
    /// scans for pending CPU-performed writes check it first (they run for
    /// every bound buffer and texture: ~3% of the GPU thread at 38 fps).
    cpu_span: Option<(u64, u64)>,
}

impl Completion {
    /// Whether this batch has CPU-performed writes (write-backs, end-of-pipe
    /// writes) overlapping [addr, addr+len) still to do.
    fn cpu_write_overlaps(&self, addr: u64, len: u64) -> bool {
        self.cpu_span.is_some_and(|(lo, hi)| lo < addr + len && addr < hi)
            && (self.writebacks.iter().any(|&(w, _, n)| w < addr + len && addr < w + n) || self.actions.iter().any(|a| a.overlaps(addr, len)))
    }
}

unsafe impl Send for Completion {}
unsafe impl Sync for Completion {}

/// Submitted batches not yet completed, oldest first. An entry stays queued
/// until its write-backs and actions have been performed, so scans for
/// pending guest-memory writes see it for its whole lifetime.
static COMPLETIONS: parking_lot::Mutex<std::collections::VecDeque<std::sync::Arc<Completion>>> = parking_lot::Mutex::new(std::collections::VecDeque::new());
static COMPLETIONS_CV: parking_lot::Condvar = parking_lot::Condvar::new();
static DONE: parking_lot::Mutex<u64> = parking_lot::Mutex::new(0);
static DONE_CV: parking_lot::Condvar = parking_lot::Condvar::new();
/// Diagnostics (TLG_PERF): host-observed GPU busy time and CP waits for
/// pending guest-memory writes, in nanoseconds.
pub static GPU_BUSY_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// GPU execution time of the batches (their start/end timestamps), ns.
/// (GPU_BUSY_NS also counts the time the completion thread was late to
/// notice a finished batch, e.g. while writing back the previous one.)
pub static GPU_EXEC_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static SYNC_WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// DMA_DATA memory copies the GPU did not take (`dma_on_gpu`), by reason
/// (`DMA_DECLINE_REASONS`).
pub static DMA_DECLINED: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];
pub const DMA_DECLINE_REASONS: [&str; 6] = ["off/size/alignment/overlap", "source has CPU-performed writes pending", "destination has CPU-performed writes pending", "(unused)", "destination not importable", "nothing pending"];
/// Last completed batch: fence wait and write-back durations (ns), bytes written back.
static LAST_FENCE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LAST_WB_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LAST_WB_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Bytes written back to guest memory by completions and the time spent
/// doing it (ns, write-watch resets included) (TLG_PERF).
pub static WB_BYTES_SUM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static WB_NS_SUM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Of that: write-watch resets (calls, ns).
pub static WB_RESETS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static WB_RESET_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn push_completion(mut c: Completion) {
    let spans = c.writebacks.iter().map(|&(w, _, n)| (w, w + n)).chain(c.actions.iter().filter_map(|a| a.span()).map(|(a, n)| (a, a + n)));
    c.cpu_span = spans.reduce(|x, y| (x.0.min(y.0), x.1.max(y.1)));
    COMPLETIONS.lock().push_back(std::sync::Arc::new(c));
    COMPLETIONS_CV.notify_one();
    // (Its end-of-pipe writes now meet waits in GPU order: `pending_label_value`.)
    labels_changed();
}

/// Generation of the memory that command-processor waits poll: bumped when a
/// batch is submitted (its end-of-pipe writes become pending values) or
/// completes (they and its write-backs land), and when a CP writes memory.
/// WAIT_REG_MEM and MEM_SEMAPHORE waits sleep until it changes or a poll
/// interval passes, on a per-thread event and high-resolution timer (each
/// `thread::sleep` of the old 100 us polls created and closed a timer: ~12%
/// of the graphics CP in the cave).
static LABEL_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Events of the threads blocked in `wait_labels`.
static LABEL_WAITERS: parking_lot::Mutex<Vec<usize>> = parking_lot::Mutex::new(Vec::new());

pub fn labels_changed() {
    LABEL_GEN.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    for &e in LABEL_WAITERS.lock().iter() {
        unsafe { windows_sys::Win32::System::Threading::SetEvent(e as _) };
    }
}

/// The current generation (read before testing a wait's condition).
pub fn label_gen() -> u64 {
    LABEL_GEN.load(std::sync::atomic::Ordering::SeqCst)
}

/// Waits until the generation moves past `seen`, at most `timeout` (memory
/// the title's CPU threads write is only seen by polling).
pub fn wait_labels(seen: u64, timeout: std::time::Duration) {
    use windows_sys::Win32::System::Threading::*;
    if crate::tune::get(crate::tune::CP_WAIT_NOTIFY) == 0 {
        std::thread::sleep(std::time::Duration::from_micros(100));
        return;
    }
    thread_local! {
        // (Auto-reset event, synchronization timer; kept for the thread's life.)
        static HANDLES: (usize, usize) = unsafe {
            let ev = CreateEventW(std::ptr::null(), 0, 0, std::ptr::null());
            let tm = CreateWaitableTimerExW(std::ptr::null(), std::ptr::null(), CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
            (ev as usize, tm as usize)
        };
        // When the armed timer fires (None: not armed, or it fired).
        static TIMER_DUE: std::cell::Cell<Option<std::time::Instant>> = const { std::cell::Cell::new(None) };
    }
    HANDLES.with(|&(ev, tm)| {
        if ev == 0 || tm == 0 {
            std::thread::sleep(timeout);
            return;
        }
        LABEL_WAITERS.lock().push(ev);
        if label_gen() == seen {
            // A timer still armed by an earlier wait that a label change
            // ended fires within the poll interval anyway: kept rather than
            // re-armed per wake-up (`tune wait_timer_keep`).
            let now = std::time::Instant::now();
            let keep = crate::tune::get(crate::tune::WAIT_TIMER_KEEP) != 0 && TIMER_DUE.get().is_some_and(|t| t > now);
            unsafe {
                if !keep {
                    // (Relative due time in 100 ns units.)
                    let due = -((timeout.as_nanos() / 100).max(1) as i64);
                    SetWaitableTimer(tm as _, &due, 0, None, std::ptr::null(), 0);
                    TIMER_DUE.set(Some(now + timeout));
                }
                let hs = [ev as _, tm as _];
                if WaitForMultipleObjects(2, hs.as_ptr(), 0, INFINITE) == windows_sys::Win32::Foundation::WAIT_OBJECT_0 + 1 {
                    TIMER_DUE.set(None);
                }
            }
        }
        let mut w = LABEL_WAITERS.lock();
        if let Some(i) = w.iter().position(|&x| x == ev) {
            w.swap_remove(i);
        }
        // (An event set after this returns makes the next wait return at
        // once: one extra test.)
    });
}

/// Waits for submitted batches, then performs their buffer write-backs and
/// end-of-pipe actions in order.
fn completion_thread() {
    crate::log::set_thread_name("GPU-complete");
    crate::thread::raise_priority();
    let mut last_done = std::time::Instant::now();
    loop {
        let c = {
            let mut q = COMPLETIONS.lock();
            while q.is_empty() {
                COMPLETIONS_CV.wait(&mut q);
            }
            q.front().unwrap().clone()
        };
        let tf = std::time::Instant::now();
        if c.fence != vk::Fence::null() {
            if let Err(e) = unsafe { ctx().device.wait_for_fences(&[c.fence], true, 10_000_000_000) } {
                crate::error!("render wait: {e} (GPU hang or device lost)");
                if e == vk::Result::ERROR_DEVICE_LOST {
                    super::vk::report_device_lost();
                }
            }
            let now = std::time::Instant::now();
            let start = c.submitted.max(last_done);
            GPU_BUSY_NS.fetch_add(now.saturating_duration_since(start).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            last_done = now;
            if c.exec != vk::QueryPool::null() {
                let mut ts = [0u64; 2];
                if unsafe { ctx().device.get_query_pool_results(c.exec, 0, &mut ts, vk::QueryResultFlags::TYPE_64) }.is_ok() {
                    let ns = (ts[1].saturating_sub(ts[0]) as f64 * ctx().limits.timestamp_period as f64) as u64;
                    GPU_EXEC_NS.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        if let Some((pool, labels)) = &c.profile {
            record_profile(*pool, labels);
        }
        let tw = std::time::Instant::now();
        LAST_FENCE_NS.store((tw - tf).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        let mut wb_bytes = 0;
        {
            // (Write-backs of mirror-resident data are forgotten by the write
            // watch together with their copy: see `mem::WRITE_WATCH_LOCK`.)
            let mut ww = (!c.mirror_writes.is_empty() || c.wb_bits.iter().any(|b| b.is_some())).then(|| crate::mem::WRITE_WATCH_LOCK.lock());
            let probe = super::wbprobe::enabled();
            // (`tune wb_lock_bump`: waiting write-watch queries go first
            // between entries; each entry's copy and reset stay together.)
            let bump = crate::tune::get(crate::tune::WB_LOCK_BUMP) != 0;
            for (k, &(addr, off, size)) in c.writebacks.iter().enumerate() {
                if bump && k > 0 {
                    if let Some(g) = ww.as_mut() {
                        parking_lot::MutexGuard::bump(g);
                    }
                }
                if probe {
                    super::wbprobe::before_write(addr, size);
                }
                super::lazywb::before_write(addr, size);
                if let Some((bits, words)) = c.wb_bits.get(k).copied().flatten() {
                    // Marked pages only (mirror-resident data: the write watch
                    // forgets these writes too), in runs of consecutive pages:
                    // each reset is a system call costing tens of microseconds.
                    let head = addr & 0xFFF;
                    let bm = unsafe { std::slice::from_raw_parts(c.ring_ptr.add(bits as usize) as *const u32, words as usize) };
                    let pages = (head + size).div_ceil(4096);
                    let marked = |p: u64| bm.get((p / 32) as usize).is_some_and(|w| w >> (p % 32) & 1 != 0);
                    let mut page = 0;
                    while page < pages {
                        if bm.get((page / 32) as usize).is_some_and(|&w| w == 0) {
                            page = (page / 32 + 1) * 32;
                            continue;
                        }
                        if !marked(page) {
                            page += 1;
                            continue;
                        }
                        let first = page;
                        while page < pages && marked(page) {
                            page += 1;
                        }
                        let start = (first * 4096).saturating_sub(head);
                        let end = (page * 4096 - head).min(size);
                        let src = unsafe { std::slice::from_raw_parts(c.ring_ptr.add((off + start) as usize), (end - start) as usize) };
                        resources::copy_to_guest(src, addr + start);
                        let tr = std::time::Instant::now();
                        crate::mem::reset_write_watch(addr + start, end - start);
                        WB_RESET_NS.fetch_add(tr.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                        WB_RESETS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        wb_bytes += end - start;
                    }
                    if probe {
                        super::wbprobe::after_write(addr, size);
                    }
                    super::lazywb::after_write(addr, size, true);
                    continue;
                }
                let src = unsafe { std::slice::from_raw_parts(c.ring_ptr.add(off as usize), size as usize) };
                resources::copy_to_guest(src, addr);
                wb_bytes += size;
                if probe {
                    super::wbprobe::after_write(addr, size);
                }
                let mirror = c.wb_mirror.get(k).copied().unwrap_or(false);
                super::lazywb::after_write(addr, size, mirror);
                if bump && mirror {
                    // (One of `mirror_writes`.)
                    crate::mem::reset_write_watch(addr, size);
                }
            }
            if !bump {
                for &(addr, size) in &c.mirror_writes {
                    crate::mem::reset_write_watch(addr, size);
                }
            }
        }
        LAST_WB_NS.store(tw.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        WB_NS_SUM.fetch_add(tw.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        LAST_WB_BYTES.store(wb_bytes, std::sync::atomic::Ordering::Relaxed);
        WB_BYTES_SUM.fetch_add(wb_bytes, std::sync::atomic::Ordering::Relaxed);
        for a in &c.actions {
            a.perform();
        }
        labels_changed();
        COMPLETIONS.lock().pop_front();
        *DONE.lock() = c.seq;
        DONE_CV.notify_all();
    }
}

/// GPU timestamp profiling (TLG_GPU_PROFILE=1): a timestamp is written after
/// each recorded operation; the GPU time between consecutive timestamps is
/// attributed to the later operation's label.
pub fn gpu_profile_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("TLG_GPU_PROFILE").is_some())
}

const PROFILE_QUERIES: u32 = 16384;

/// Accumulated GPU time per label: (count, nanoseconds).
static GPU_PROFILE: parking_lot::Mutex<Option<std::collections::HashMap<String, (u64, u64)>>> = parking_lot::Mutex::new(None);

fn record_profile(pool: vk::QueryPool, labels: &[String]) {
    let n = labels.len();
    let mut ts = vec![0u64; n];
    let ok = unsafe { ctx().device.get_query_pool_results(pool, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT) };
    if ok.is_err() {
        return;
    }
    let period = ctx().limits.timestamp_period as f64;
    let mut g = GPU_PROFILE.lock();
    let map = g.get_or_insert_with(std::collections::HashMap::new);
    for i in 1..n {
        let ns = (ts[i].saturating_sub(ts[i - 1]) as f64 * period) as u64;
        let e = map.entry(labels[i].clone()).or_insert((0, 0));
        e.0 += 1;
        e.1 += ns;
    }
}

/// Takes the accumulated GPU profile, most expensive first: (label, count, ns).
pub fn take_gpu_profile() -> Vec<(String, u64, u64)> {
    let mut v: Vec<(String, u64, u64)> = GPU_PROFILE.lock().take().unwrap_or_default().into_iter().map(|(k, (c, ns))| (k, c, ns)).collect();
    v.sort_by(|a, b| b.2.cmp(&a.2));
    v
}

/// Whether a submitted batch still has CPU-performed buffer write-backs (or
/// end-of-pipe writes) overlapping [addr, addr+len): GPU readers must wait
/// for those; GPU-performed writes are ordered on the queue.
fn cpu_writeback_in_flight(addr: u64, len: u64) -> bool {
    COMPLETIONS.lock().iter().any(|c| c.cpu_write_overlaps(addr, len))
}

/// Waits for the batches `cpu_writeback_in_flight` reports.
#[track_caller]
fn wait_cpu_writebacks(addr: u64, len: u64) {
    let seq = COMPLETIONS
        .lock()
        .iter()
        .filter(|c| c.cpu_write_overlaps(addr, len))
        .map(|c| c.seq)
        .max();
    if let Some(seq) = seq {
        let t0 = std::time::Instant::now();
        wait_done(seq);
        let ns = t0.elapsed().as_nanos() as u64;
        SYNC_WAIT_NS.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
        SYNC_WAIT_KIND_NS[1].fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
        let loc = std::panic::Location::caller().line();
        *FLUSH_SITES.lock().get_or_insert_with(HashMap::new).entry(100_000 + loc).or_default() += ns / 1000;
    }
}

/// Render target write-backs tiled on the GPU into the VRAM mirror when
/// guest memory is not imported (`gpu_remap`); TLG_MIRROR_TILING=0: on the
/// CPU after draining the GPU. (Under VRAM pressure 3 the title screen's 3D
/// background renders black with either path: a separate problem.)
fn mirror_tiling_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TLG_MIRROR_TILING").map(|v| v != "0").unwrap_or(true))
}

/// Whether a submitted batch's end-of-pipe writes overlap [addr, addr+len).
pub fn actions_in_flight(addr: u64, len: u64) -> bool {
    COMPLETIONS.lock().iter().any(|c| c.actions.iter().any(|a| a.overlaps(addr, len)))
}

/// The 32-bit value at `addr` once the in-flight end-of-pipe writes have
/// been performed, when one of them writes it with known data (None when
/// none does, or a timestamp is the last write).
pub fn pending_label_value(addr: u64) -> Option<u32> {
    let mut val = None;
    for c in COMPLETIONS.lock().iter() {
        for a in &c.actions {
            let n = match a.data_sel {
                1 => 4,
                2..=4 => 8,
                _ => continue,
            };
            if addr < a.addr || addr + 4 > a.addr + n {
                continue;
            }
            val = match a.data_sel {
                1 => Some(a.lo),
                2 => Some(if addr == a.addr { a.lo } else { a.hi }),
                _ => None,
            };
        }
    }
    val
}

/// Whether a submitted, not yet completed batch will write back GPU-written
/// buffer data overlapping guest memory [addr, addr+len).
fn writeback_in_flight(addr: u64, len: u64) -> bool {
    COMPLETIONS
        .lock()
        .iter()
        .any(|c| c.writebacks.iter().any(|&(w, _, n)| w < addr + len && addr < w + n) || c.zc_writes.iter().any(|&(w, n)| w < addr + len && addr < w + n))
}

/// Waits for in-flight batches whose completion writes guest memory in
/// [addr, addr+len) (end-of-pipe writes or buffer write-backs).
#[track_caller]
pub fn wait_overlapping(addr: u64, len: u64) {
    // Also classify what the newest overlapping batch writes (TLG_PERF
    // breakdown): end-of-pipe actions, device-ring write-backs, in-place
    // writes of huge (sparse scatter) bindings, other in-place writes.
    let mut kind = 0usize;
    let seq = COMPLETIONS
        .lock()
        .iter()
        .filter_map(|c| {
            let mut k = 0usize;
            if c.actions.iter().any(|a| a.overlaps(addr, len)) {
                k |= 1;
            }
            if c.writebacks.iter().any(|&(w, _, n)| w < addr + len && addr < w + n) {
                k |= 2;
            }
            for &(w, n) in &c.zc_writes {
                if w < addr + len && addr < w + n {
                    k |= if n > SURFACE_WRITE_MAX { 4 } else { 8 };
                }
            }
            (k != 0).then(|| {
                kind = k;
                c.seq
            })
        })
        .max();
    if let Some(seq) = seq {
        let t0 = std::time::Instant::now();
        wait_done(seq);
        let ns = t0.elapsed().as_nanos() as u64;
        SYNC_WAIT_NS.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
        let loc = std::panic::Location::caller().line();
        let mut g = FLUSH_SITES.lock();
        *g.get_or_insert_with(HashMap::new).entry(100_000 + loc).or_default() += ns / 1000;
        drop(g);
        for (i, bit) in [1usize, 2, 4, 8].iter().enumerate() {
            if kind & bit != 0 {
                SYNC_WAIT_KIND_NS[i].fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
}

/// Synchronizing flushes by source line of the caller (TLG_PERF).
pub static FLUSH_SITES: parking_lot::Mutex<Option<HashMap<u32, u64>>> = parking_lot::Mutex::new(None);

/// Sync wait time by what the awaited batch writes in the range (see
/// `wait_overlapping`): actions, write-backs, huge in-place, in-place.
pub static SYNC_WAIT_KIND_NS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// Blocks until the batch with sequence number `seq` (and all before it) completed.
pub fn wait_done(seq: u64) {
    let mut d = DONE.lock();
    while *d < seq {
        DONE_CV.wait(&mut d);
    }
}

pub struct Renderer {
    pool: vk::CommandPool,
    slots: Vec<Slot>,
    cur: usize,
    /// Sequence number of the last submitted batch.
    pub last_seq: u64,
    /// Current slot's command buffer and descriptor pool.
    cmd: vk::CommandBuffer,
    recording: bool,
    ring: Ring,
    desc_pool: vk::DescriptorPool,
    batch: u64,
    /// Texture validation epoch: CPU-sourced textures are re-hashed at most
    /// once per epoch. It advances per title submission and whenever guest
    /// memory may have been modified by the GPU side (write-backs, DMA).
    epoch: u64,
    images: FastMap<ImageKey, Image>,
    by_base: FastMap<u64, Vec<ImageKey>>,
    samplers: Samplers,
    modules: FastMap<u64, vk::ShaderModule>,
    gfx_pipes: FastMap<GfxKey, Pipe>,
    cs_pipes: FastMap<u64, Pipe>,
    rendering: Option<Rendering>,
    barrier_needed: bool,
    /// `BARRIERS` count when GPU work other than dispatches' tracked buffer
    /// accesses was last left unordered (`unordered!`; u64::MAX: never).
    unordered_gen: u64,
    /// Buffer ranges dispatches accessed since the full barrier that made
    /// `BARRIERS` `hz_gen` (buffer, offset, size, written), and whether one
    /// used resources not tracked that way (`tune dispatch_hazards`).
    hz: smallvec::SmallVec<[(vk::Buffer, u64, u64, bool); 32]>,
    hz_gen: u64,
    hz_unknown: bool,
    /// The accesses of the dispatch whose descriptors were written last.
    cur_access: smallvec::SmallVec<[(vk::Buffer, u64, u64, bool); 24]>,
    cur_unknown: bool,
    /// Page bitmaps of mirror write-backs (`mirror_write_back`): device-ring
    /// words zeroed at the start of the batch (`begin`): (batch, offset,
    /// words used, capacity in words).
    bits_arena: (u64, u64, u64, u64),
    /// Barriers before dispatches emitted, and skipped by the tracking.
    pub hz_stats: (u64, u64),
    /// Why those barriers were needed (TLG_PERF): indirect arguments,
    /// untracked resources, conflicting accesses; and by the render.rs line
    /// that left other GPU work unordered (`unordered!`).
    pub hz_reasons: [u64; 3],
    pub hz_other_lines: FastMap<u32, u64>,
    unordered_line: u32,
    /// See `DynState` (None at the start of a command buffer).
    dyn_state: Option<DynState>,
    /// Graphics pipeline bound in the current command buffer (consecutive
    /// draws mostly share one; the bind is skipped then).
    bound_gfx: vk::Pipeline,
    /// Device-local ring segment (same slot rotation as `ring`).
    dring: Ring,
    /// Device-local ring the CPU writes through the BAR window, for per-draw
    /// SGPR data (same slot rotation; None without a host-visible device
    /// heap).
    bar: Option<Ring>,
    /// Buffers written by shaders in the current batch, in program order.
    writebacks: Vec<PendingWrite>,
    /// Imported guest memory spans for zero-copy bindings: (start, end, buffer, memory).
    zc_spans: Vec<(u64, u64, vk::Buffer, vk::DeviceMemory)>,
    /// `mem::MAP_GEN` the spans were validated against.
    zc_gen: u64,
    /// Guest ranges written in place by zero-copy bindings in the current batch.
    zc_writes: Vec<(u64, u64)>,
    /// Guest ranges read in place in the current batch.
    zc_reads: Ranges,
    /// Device-ring copies of shader-written buffers to copy back to guest
    /// memory after the current command: (ring offset, guest buffer, offset, size).
    gpu_wb: Vec<(u64, vk::Buffer, u64, u64)>,
    /// Accumulated GPU-thread time per activity (see PROF_NAMES).
    pub prof: [std::time::Duration; 7],
    /// Time-stamp-counter ticks spent in each phase of `draw_inner`
    /// (`DRAW_PHASES`) and the draws timed (TLG_PERF: µs per draw).
    pub dprof: [u64; 8],
    pub dprof_draws: u64,
    /// Ticks in parts of the descriptor phase (`DESC_PARTS`).
    pub dsub: [u64; 6],
    /// Buffer bindings by path (`BUFFER_PATHS`): ticks and count.
    pub bsub: [u64; 9],
    pub bcnt: [u64; 9],
    /// Synchronizing flushes by reason (see FLUSH_REASONS).
    pub flushes: [u64; 6],
    /// Buffer bytes copied at record time by cause: shader-written (device
    /// ring initialisation), read-only misaligned, read-only with pending GPU
    /// writes, index data.
    pub upload_bytes: [u64; 4],
    /// Texture uploads performed and guest bytes hashed.
    pub uploads: u64,
    pub hashed_bytes: u64,
    /// Reported problems and how often they occurred.
    reported: FastMap<String, u64>,
    pub draws: u64,
    pub dispatches: u64,
    pub skipped: u64,
    /// Global data share (GDS_BYTES, device-local; zero-initialized).
    gds: vk::Buffer,
    /// GPU (de)tiling kernels: (set layout, pipeline layout, detile, tile).
    remap: Option<(vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline, vk::Pipeline)>,
    /// Indirect-dispatch argument validation (`indirect_args`): pipeline,
    /// diagnostics buffer (host-visible: rejections, last rejected x, y, z;
    /// mapped address) and the rejection count last logged.
    indirect_pipe: Option<(vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline)>,
    /// `kernels::depth_copy_cs` (created on first use).
    depth_copy_pipe: Option<(vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline)>,
    /// Selective write-back of mirror-resident writes (kernels::dirty_copy_cs):
    /// set layout, layout, pipeline and the pool of its per-source sets.
    dirty_pipe: Option<(vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline, vk::DescriptorPool)>,
    /// Descriptor set of the page copy per source buffer (VRAM mirror windows).
    dirty_sets: FastMap<vk::Buffer, vk::DescriptorSet>,
    /// Device-ring dwords where shaders mark pages of untracked writes: (batch, offset).
    dirty_sink: (u64, u64),
    /// Who asked for the next VRAM mirror binding (TLG_PERF copy-in report).
    vc_why: &'static str,
    /// Fast-clear metadata kernels (`meta_clear_on_gpu`): set layout,
    /// pipeline layout, probe and expand pipelines.
    meta_pipes: Option<(vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline, vk::Pipeline)>,
    /// Fast clears decided on the GPU (no drain to read the metadata).
    pub meta_gpu: u64,
    indirect_diag: Option<(vk::Buffer, vk::DeviceMemory, usize)>,
    indirect_rejected: u32,
    /// Element maps per surface layout (width, height, bpp, tile index,
    /// slice bytes): device buffer and element count.
    tile_maps: FastMap<(u32, u32, u32, u8, u64, u32), (vk::Buffer, u64)>,
    /// Images that were given a metadata surface (CMASK/HTILE): the writes
    /// that may change one (every buffer a shader writes) look only at
    /// these, not at every image (that scan was ~4% of the GPU thread).
    meta_images: Vec<ImageKey>,
    /// Render targets (de)tiled on the GPU: reloads, write-backs.
    pub gpu_remaps: [u64; 2],
    /// Textures read from guest memory and detiled on the GPU.
    pub gpu_tex_uploads: u64,
    /// See `new_epoch`.
    cpu_epoch: u64,
    /// Epoch validations for the command being recorded must reach
    /// (graphics or compute queue, see `super::CPU_EPOCH`).
    required_epoch: u64,
    /// All images by guest base (range queries) with an upper bound of each
    /// one's guest surface size, and the largest bound.
    img_bases: std::collections::BTreeMap<u64, Vec<(ImageKey, u64)>>,
    img_max_bytes: u64,
    /// The images of `img_bases` whose bound exceeds IMG_NEAR_BYTES (few):
    /// writes are checked against the others within that distance only.
    img_bases_large: std::collections::BTreeMap<u64, Vec<(ImageKey, u64)>>,
    /// Shader of the draw/dispatch being recorded (diagnostics); commands
    /// recorded in the current batch.
    cur_shader: u64,
    /// The shader being prepared (TLG_PERF: `GUARD_SITES`): pixel or
    /// compute shader, vertex shader.
    prep_shader: u64,
    prep_vs: u64,
    /// TLG_PERF: recent `note_pending_gpu_write` calls (base, len, shader,
    /// batch).
    pending_log: Vec<(u64, u64, u64, u64)>,
    batch_cmds: u64,
    /// Draws/dispatches whose descriptors were rewritten in a new batch
    /// because the ring filled up during preparation.
    pub ring_retries: u64,
    /// Descriptor writing is in its retry in a new batch (see
    /// `dring_for_detile`: no second retry).
    desc_retry: bool,
    /// Device-local scratch buffer for GPU-side image conversions (depth ->
    /// color copies), grown on demand; replaced buffers stay alive (they may
    /// still be referenced by submitted batches).
    scratch: Option<(vk::Buffer, u64)>,
    /// Page write tracking for CPU-sourced textures.
    pub writes: super::writes::WriteTracker,
    /// Labels of the timestamps written in the current batch (TLG_GPU_PROFILE).
    marks: Vec<String>,
    /// Images that may hold contents newer than guest memory (`guest_stale`;
    /// a superset, entries are dropped lazily) by guest base, for range
    /// lookups in `writeback_images`; largest surface among them; frame of
    /// the last aging pass.
    stale: std::collections::BTreeMap<u64, Vec<ImageKey>>,
    stale_max: u64,
    stale_aged: u64,
    /// Guest pages (4 KiB) with GPU writes that have not completed yet:
    /// page -> sequence number of the batch writing it.
    gpu_pending_pages: FastMap<u64, u64>,
    /// Batch with a pending GPU write too large to track by page.
    gpu_pending_all: u64,
    /// Ranges whose pages `gpu_pending_pages` already maps to batch
    /// `pending_noted.0` (a buffer written by many commands of a batch is
    /// entered once: 1,500 map inserts per binding of a 6 MiB buffer).
    pending_noted: (u64, FastSet<(u64, u64)>),
    /// Shader preparations that had to wait for GPU writes to the memory
    /// they read (descriptor tables, scalar loads), and scatter dispatches
    /// whose writes were narrowed to their records.
    pub prepare_syncs: u64,
    pub scatters: u64,
    /// Copy/fill kernels performed as image operations.
    pub elided: u64,
    /// Mirror copy-ins that left the window's own pending writes in place
    /// instead of draining the GPU for them.
    pub copyin_kept: u64,
    /// DMA_DATA copies done on the GPU through the mirror windows.
    pub dma_mirror: u64,
    /// The last `upload_guest` queued GPU copies into its ring range, and
    /// the copies not yet recorded (`flush_forwards`).
    upload_forwarded: bool,
    fwd_pending: Vec<(vk::Buffer, vk::BufferCopy)>,
    /// Read-only guest ranges copied into this batch's ring segments in CPU
    /// epoch `upload_reuse_epoch`, by (aligned) base: (bytes, ring buffer and
    /// offset, write-tracker stamp after the copy, CP queue). See
    /// `reused_upload`.
    upload_reuse: FastMap<u64, (u64, vk::Buffer, u64, u64, usize)>,
    upload_reuse_epoch: u64,
    /// The CP queue of the command being recorded.
    cmd_queue: usize,
    /// Bindings served by an earlier copy, bytes not copied again.
    pub upload_reused: (u64, u64),
    /// Read-only copies into the BAR ring (`upload_guest_ro`): count, bytes,
    /// and copies that went to the host ring for want of room.
    pub bar_uploads: (u64, u64, u64),
    /// Index data bound from the VRAM mirror (`tune index_mirror`): draws, bytes.
    pub index_mirrored: (u64, u64),
    retired_scratch: Vec<(vk::Buffer, vk::DeviceMemory)>,
    scratch_mem: Option<vk::DeviceMemory>,
    /// VRAM copies of read-only vertex/index data (`vcache_bind`).
    vcache: Option<VCache>,
    /// Sparse mirror windows: block memory, binds queued for the next
    /// submission and the semaphore it waits for them on.
    sparse_pool: Option<BlockPool>,
    sparse_binds: Vec<(vk::Buffer, Vec<vk::SparseMemoryBind>)>,
    bind_sem: vk::Semaphore,
    /// RT image write counters for texture shadow copies.
    write_gen: FastMap<ImageKey, u64>,
    shadows: FastMap<(ImageKey, u32, u32, vk::Format), (ImageKey, u64)>,
    /// Attachments of the draw being prepared: their images keep their keys
    /// until it is recorded (`grow_levels` leaves them alone).
    draw_targets: smallvec::SmallVec<[ImageKey; 9]>,
    /// Images replaced by ones with more mip levels (`grow_levels`).
    grown: u64,
    /// VRAM accounting (see `housekeeping`): device memory of all images and
    /// of the GPU (de)tiling element maps; images evicted (count, bytes);
    /// released objects with the batch sequence after which they are freed;
    /// frame of the next eviction pass; pressure level (`vram_pressure`) and
    /// when it was last reported.
    image_bytes: u64,
    tile_map_bytes: u64,
    pub evicted: (u64, u64),
    retired: Vec<(u64, Retired)>,
    next_trim: u64,
    pressure: u8,
    pressure_reported: Option<std::time::Instant>,
    /// Evicted keys with the frame of their eviction, and how often one was
    /// needed again (count, sum of frames in between): returning images show
    /// whether eviction ages fit the title's reuse.
    evicted_keys: FastMap<ImageKey, u64>,
    pub recreated: (u64, u64),
    /// Evicted small images kept for reuse (`tune image_pool`), by their
    /// creation parameters (the key with base 0), and how many there are.
    image_pool: FastMap<ImageKey, Vec<resources::PooledImage>>,
    image_pool_n: usize,
    /// The indirect draw being recorded with GPU-read arguments
    /// (`draw_indirect_gpu`), and the vertex stage's SGPR data of the draw
    /// being recorded (buffer, offset, bytes; `write_descriptors`).
    indirect_draw: Option<IndirectDraw>,
    vs_sgpr: Option<(vk::Buffer, u64, u64)>,
    /// VRAM-mirror bindings served (`tune vbind_cache`): (base, size) ->
    /// (window, its generation then, the oldest CPU-check epoch of the
    /// blocks, buffer, offset); and the generation counter of the windows.
    vbind: FastMap<(u64, u64), (u64, u64, u64, vk::Buffer, u64)>,
    vgen: u64,
    /// The window `note_mirror_write` writes (its generation is kept).
    vgen_keep: vk::Buffer,
    /// The current slot's prologue is being recorded; operations put there.
    pro_recording: bool,
    /// Shader stages the CP prepared before taking the lock for the next
    /// dispatch / draw (`tune prepare_unlocked`).
    pre_cs: Option<Result<shaders::Prepared, String>>,
    pre_draw: Option<(Result<shaders::Prepared, String>, Option<Result<shaders::Prepared, String>>)>,
    pub pro_ops: u64,
    /// The indirect draw kernel (set layout, layout, pipeline), its
    /// diagnostics buffer (buffer, memory, mapping) and the rejections seen.
    indirect_draw_pipe: Option<(vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline)>,
    indirect_draw_diag: Option<(vk::Buffer, vk::DeviceMemory, usize)>,
    indirect_draw_rejected: u32,
    /// Images created and taken from the pool (TLG_PERF).
    pub image_reuse: (u64, u64),
    /// How the last elided utility kernel was performed (frame traces).
    last_elide: &'static str,
    /// TLG_TRACE_DEPTH: last near fraction per depth target.
    depth_trace: FastMap<u64, f32>,
    /// TLG_VS_POS: host-visible buffer receiving position exports.
    debug_out: Option<(vk::Buffer, *mut u8)>,
    /// Occlusion counting (see `occlusion_dump`): query windows open (begin
    /// samples without their end yet), the query active in the command
    /// buffer being recorded, queries ended since the last dump, next query.
    occl_windows: u32,
    occl_active: Option<u32>,
    occl_ended: Vec<u32>,
    occl_next: u32,
    /// Emulated tessellation: device buffers holding the threadgroups' LDS
    /// and standing in for the factor ring (created on first use).
    tess_bufs: Option<(vk::Buffer, vk::Buffer)>,
    /// Write-backs skipped because the title reused the image's memory.
    reused_skips: u64,
    /// Sampled render targets whose memory the title reused for CPU data
    /// (`reused_target`): sampled from guest memory instead.
    reused_targets: u64,
    /// Texture validations settled by comparing shared edge pages.
    edge_skips: u64,
}

unsafe impl Send for Renderer {}

/// A guest buffer range written by a shader in the current batch: the shader
/// works on `dev_off` in the device ring; at batch end the data is copied to
/// `host_off` in the host ring, from where the completion thread writes it
/// back to guest memory.
#[derive(Clone, Copy)]
struct PendingWrite {
    base: u64,
    size: u64,
    host_off: u64,
    dev_off: u64,
    /// Where the shader wrote: the device ring, or a VRAM mirror window
    /// (`mirror`: the data stays there, current, for later commands).
    dev_buf: vk::Buffer,
    mirror: bool,
    /// Page bitmap of the range's writes (mirror writes): (device-ring
    /// offset, host-ring offset of its copy, dwords). Only marked pages are
    /// copied out and written back.
    bits: Option<(u64, u64, u32)>,
}

pub static RENDERER: Mutex<Option<Renderer>> = Mutex::new(None);

pub fn init() -> Result<(), String> {
    let c = ctx();
    unsafe {
        let pool = c
            .device
            .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)
            .map_err(|e| e.to_string())?;
        // (Each slot's batch and prologue command buffers.)
        let cmds = c.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(2 * SLOTS as u32)).map_err(|e| e.to_string())?;
        let sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 32768 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLED_IMAGE, descriptor_count: 32768 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 8192 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLER, descriptor_count: 32768 },
        ];
        let mut slots = Vec::with_capacity(SLOTS);
        for (i, &cmd) in cmds[..SLOTS].iter().enumerate() {
            slots.push(Slot {
                cmd,
                pro: cmds[SLOTS + i],
                fence: c.device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|e| e.to_string())?,
                desc_pool: c.device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(8192).pool_sizes(&sizes), None).map_err(|e| e.to_string())?,
                queries: if gpu_profile_on() {
                    c.device.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(PROFILE_QUERIES), None).map_err(|e| e.to_string())?
                } else {
                    vk::QueryPool::null()
                },
                exec: c.device.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2), None).unwrap_or(vk::QueryPool::null()),
                seq: 0,
            });
        }
        let mut ring = Ring::new(c, resources::RING_SIZE)?;
        ring.set_segment(0, SLOTS);
        let mut dring = Ring::new_device(c, resources::DEV_RING_SIZE)?;
        dring.set_segment(0, SLOTS);
        let bar = match Ring::new_bar(c, resources::BAR_RING_SIZE) {
            Ok(mut b) => {
                b.set_segment(0, SLOTS);
                crate::info!("BAR ring: {} MiB of device-local host-visible memory for per-draw SGPR data", resources::BAR_RING_SIZE >> 20);
                Some(b)
            }
            Err(e) => {
                crate::info!("BAR ring unavailable ({e}): per-draw SGPR data stays in host memory");
                None
            }
        };
        let (gds, _gds_mem) = c.device_buffer(
            gcn::translate::GDS_BYTES as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
        )?;
        {
            // Zero the GDS once.
            let cmd = slots[0].cmd;
            c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).map_err(|e| e.to_string())?;
            c.device.cmd_fill_buffer(cmd, gds, 0, vk::WHOLE_SIZE, 0);
            c.device.end_command_buffer(cmd).map_err(|e| e.to_string())?;
            c.submit(cmd, &[], &[], slots[0].fence)?;
            c.device.wait_for_fences(&[slots[0].fence], true, u64::MAX).map_err(|e| e.to_string())?;
            c.device.reset_fences(&[slots[0].fence]).map_err(|e| e.to_string())?;
            c.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).map_err(|e| e.to_string())?;
        }
        let occl_pool = c
            .device
            .create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::OCCLUSION).query_count(OCCL_QUERIES), None)
            .map_err(|e| format!("occlusion query pool: {e}"))?;
        OCCL_POOL.set(occl_pool).ok();
        crate::thread::spawn_host("GPU-complete", 1 << 20, Box::new(completion_thread)).map_err(|e| e.to_string())?;
        *RENDERER.lock() = Some(Renderer {
            pool,
            cmd: slots[0].cmd,
            desc_pool: slots[0].desc_pool,
            slots,
            cur: 0,
            last_seq: 0,
            recording: false,
            epoch: 0,
            ring,
            batch: 0,
            images: FastMap::default(),
            by_base: FastMap::default(),
            samplers: Samplers::new(),
            modules: FastMap::default(),
            gfx_pipes: FastMap::default(),
            cs_pipes: FastMap::default(),
            rendering: None,
            barrier_needed: false,
            unordered_gen: u64::MAX,
            hz: smallvec::SmallVec::new(),
            hz_gen: u64::MAX,
            hz_unknown: false,
            cur_access: smallvec::SmallVec::new(),
            cur_unknown: true,
            hz_stats: (0, 0),
            bits_arena: (u64::MAX, 0, 0, 0),
            hz_reasons: [0; 3],
            hz_other_lines: FastMap::default(),
            unordered_line: 0,
            dyn_state: None,
            bound_gfx: vk::Pipeline::null(),
            dring,
            bar,
            writebacks: Vec::new(),
            zc_spans: Vec::new(),
            zc_gen: 0,
            zc_writes: Vec::new(),
            zc_reads: Ranges::default(),
            gpu_wb: Vec::new(),
            prof: Default::default(),
            flushes: [0; 6],
            upload_bytes: [0; 4],
            uploads: 0,
            hashed_bytes: 0,
            reported: FastMap::default(),
            draws: 0,
            dispatches: 0,
            skipped: 0,
            gds,
            remap: None,
            indirect_pipe: None,
            depth_copy_pipe: None,
            dirty_pipe: None,
            dirty_sets: FastMap::default(),
            dirty_sink: (u64::MAX, 0),
            vc_why: "other",
            meta_pipes: None,
            meta_gpu: 0,
            indirect_diag: None,
            indirect_rejected: 0,
            tile_maps: FastMap::default(),
            meta_images: Vec::new(),
            gpu_remaps: [0; 2],
            gpu_tex_uploads: 0,
            dprof: [0; 8],
            dprof_draws: 0,
            dsub: [0; 6],
            bsub: [0; 9],
            bcnt: [0; 9],
            cpu_epoch: 0,
            required_epoch: 0,
            img_bases: std::collections::BTreeMap::new(),
            img_max_bytes: 0,
            img_bases_large: std::collections::BTreeMap::new(),
            cur_shader: 0,
            prep_shader: 0,
            prep_vs: 0,
            pending_log: Vec::new(),
            batch_cmds: 0,
            ring_retries: 0,
            desc_retry: false,
            scratch: None,
            writes: super::writes::WriteTracker::new(),
            marks: Vec::new(),
            stale: std::collections::BTreeMap::new(),
            stale_max: 0,
            stale_aged: 0,
            gpu_pending_pages: FastMap::default(),
            gpu_pending_all: 0,
            pending_noted: (0, FastSet::default()),
            prepare_syncs: 0,
            scatters: 0,
            elided: 0,
            copyin_kept: 0,
            dma_mirror: 0,
            upload_forwarded: false,
            fwd_pending: Vec::new(),
            upload_reuse: FastMap::default(),
            upload_reuse_epoch: 0,
            cmd_queue: 0,
            upload_reused: (0, 0),
            bar_uploads: (0, 0, 0),
            index_mirrored: (0, 0),
            retired_scratch: Vec::new(),
            scratch_mem: None,
            vcache: None,
            sparse_pool: None,
            sparse_binds: Vec::new(),
            bind_sem: if c.sparse { c.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).map_err(|e| format!("semaphore: {e}"))? } else { vk::Semaphore::null() },
            write_gen: FastMap::default(),
            shadows: FastMap::default(),
            draw_targets: smallvec::SmallVec::new(),
            grown: 0,
            image_bytes: 0,
            tile_map_bytes: 0,
            evicted: (0, 0),
            retired: Vec::new(),
            next_trim: 0,
            pressure: 0,
            pressure_reported: None,
            evicted_keys: FastMap::default(),
            recreated: (0, 0),
            image_pool: FastMap::default(),
            image_pool_n: 0,
            indirect_draw: None,
            vs_sgpr: None,
            vbind: FastMap::default(),
            vgen: 0,
            vgen_keep: vk::Buffer::null(),
            pro_recording: false,
            pre_cs: None,
            pre_draw: None,
            pro_ops: 0,
            indirect_draw_pipe: None,
            indirect_draw_diag: None,
            indirect_draw_rejected: 0,
            image_reuse: (0, 0),
            last_elide: "",
            depth_trace: FastMap::default(),
            debug_out: None,
            occl_windows: 0,
            occl_active: None,
            occl_ended: Vec::new(),
            occl_next: 0,
            tess_bufs: None,
            reused_skips: 0,
            reused_targets: 0,
            edge_skips: 0,
        });
    }
    crate::info!("renderer initialized (ring {} MiB)", resources::RING_SIZE >> 20);
    Ok(())
}

/// Guest address of the tessellation factor ring
/// (`sceGnmGetTheTessellationFactorRingBufferBaseAddress`): no guest memory
/// is behind it; buffers there bind the renderer's stand-in.
pub const TESS_FACTOR_RING: u64 = 0xFF_0000_0000;
const TESS_FACTOR_RING_BYTES: u64 = 1 << 20;
/// LDS of one tessellated draw's threadgroups.
const TESS_LDS_BYTES: u64 = 16 << 20;
/// Binding of the generated control stage's factor buffer.
const TESS_TF_BINDING: u32 = 96;

/// The bytes of guest [base, base+bytes) in its first and last page when
/// the range does not start/end on a page boundary (None otherwise).
fn page_edges(base: u64, bytes: u64) -> Option<Box<(Vec<u8>, Vec<u8>)>> {
    let end = base + bytes;
    if base & 0xFFF == 0 && end & 0xFFF == 0 {
        return None;
    }
    let head_end = ((base & !0xFFF) + 0x1000).min(end);
    let mut head = vec![0u8; if base & 0xFFF != 0 { (head_end - base) as usize } else { 0 }];
    resources::copy_from_guest(&mut head, base);
    let tail_start = ((end - 1) & !0xFFF).max(base);
    let mut tail = vec![0u8; if end & 0xFFF != 0 { (end - tail_start) as usize } else { 0 }];
    resources::copy_from_guest(&mut tail, tail_start);
    Some(Box::new((head, tail)))
}

/// Offset of a buffer in the tessellation factor ring.
fn tess_ring_offset(base: u64) -> Option<u64> {
    base.checked_sub(TESS_FACTOR_RING).filter(|&o| o < TESS_FACTOR_RING_BYTES)
}

/// TLG_VS_POS buffer size.
const DEBUG_OUT_BYTES: u64 = gcn::translate::DEBUG_OUT_VERTICES as u64 * gcn::translate::DEBUG_OUT_STRIDE as u64 * 4;

/// Occlusion queries (see `Renderer::occlusion_dump`).
const OCCL_QUERIES: u32 = 4096;
static OCCL_POOL: std::sync::OnceLock<vk::QueryPool> = std::sync::OnceLock::new();
/// Samples counted by all occlusion queries so far.
static OCCL_TOTAL: parking_lot::Mutex<u64> = parking_lot::Mutex::new(0);

/// Completion time of a PIXEL_PIPE_STAT_DUMP: adds the results of the
/// queries counted since the previous dump to the running total and writes
/// it as depth block 0's counter (the other 7 blocks count nothing), each
/// with the valid bit 63.
pub fn occlusion_write(addr: u64, queries: &[u32]) {
    let mut total = OCCL_TOTAL.lock();
    if let Some(&pool) = OCCL_POOL.get() {
        for &q in queries {
            let mut r = [0u64; 1];
            match unsafe { ctx().device.get_query_pool_results(pool, q, &mut r, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT) } {
                Ok(()) => *total += r[0],
                Err(e) => crate::warn!("occlusion query {q} result: {e}"),
            }
        }
    }
    let t = *total;
    drop(total);
    // One 64-bit sample per depth block, 16 bytes apart (the dump address
    // selects the begin or the end half of each entry).
    for db in 0..8u64 {
        let v = if db == 0 { t } else { 0 } | 1 << 63;
        unsafe { std::ptr::write_volatile((addr + db * 16) as *mut u64, v) };
    }
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
        crate::info!("occlusion dump to {addr:#x}: {t} samples so far ({} queries added)", queries.len());
    }
}

/// TLG_PERF diagnostics: buffer writes that made render targets older than
/// guest memory (writer and target), render-target reloads by target, and
/// submitted batches by closing call site (count, commands recorded).
pub static NEWER_SITES: parking_lot::Mutex<Option<HashMap<String, u64>>> = parking_lot::Mutex::new(None);
/// Shader preparations that waited for pending GPU writes to the memory they
/// read (reader, address, covering write) (TLG_PERF).
pub static GUARD_SITES: parking_lot::Mutex<Option<HashMap<String, u64>>> = parking_lot::Mutex::new(None);
/// Upload copies that needed forwards of pending GPU writes: (read range,
/// reader, written range, where the write is) -> count (TLG_PERF report).
pub static FORWARD_SITES: parking_lot::Mutex<Option<HashMap<String, u64>>> = parking_lot::Mutex::new(None);
/// Fast clears left to the CPU path, by reason (`meta_clear_on_gpu`), and
/// how often that path then waited for the GPU (TLG_PERF report).
pub static META_DECLINES: parking_lot::Mutex<Option<HashMap<&'static str, u64>>> = parking_lot::Mutex::new(None);

fn meta_decline(why: &'static str) -> Result<bool, String> {
    if perf_on() {
        *META_DECLINES.lock().get_or_insert_with(HashMap::new).entry(why).or_default() += 1;
    }
    Ok(false)
}
pub static RELOAD_SITES: parking_lot::Mutex<Option<HashMap<String, u64>>> = parking_lot::Mutex::new(None);
/// TLG_PERF: texture uploads by texture (count, microseconds reading/detiling).
pub static UPLOAD_SITES: parking_lot::Mutex<Option<HashMap<String, (u64, u64)>>> = parking_lot::Mutex::new(None);
pub static BATCH_SITES: parking_lot::Mutex<Option<HashMap<(&'static str, u32), (u64, u64)>>> = parking_lot::Mutex::new(None);

pub fn perf_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("TLG_PERF").is_some())
}

/// Images created (TLG_PERF): by (width, height, layers, levels, format) ->
/// (count, last guest base), reported and cleared every second.
pub static NEW_IMAGES: parking_lot::Mutex<Option<HashMap<(u32, u32, u32, u32, vk::Format), (u64, u64)>>> = parking_lot::Mutex::new(None);

/// Full barriers by source line (TLG_PERF).
pub static BARRIER_SITES: parking_lot::Mutex<Option<HashMap<u32, u64>>> = parking_lot::Mutex::new(None);

/// VRAM mirror copy-ins by the binding that needed them: (count, bytes)
/// per (base, bytes) (TLG_PERF report).
pub static VCACHE_FILL_SITES: parking_lot::Mutex<Option<HashMap<(u64, u64, &'static str), (u64, u64)>>> = parking_lot::Mutex::new(None);

/// Write-backs by (base, bytes, from the mirror) (TLG_PERF report).
pub static WRITEBACK_SITES: parking_lot::Mutex<Option<HashMap<(u64, u64, bool), u64>>> = parking_lot::Mutex::new(None);

/// Buffer bindings uploaded through the ring, by (base, bytes, written)
/// (TLG_PERF report per second).
pub static BUFFER_UPLOAD_SITES: parking_lot::Mutex<Option<HashMap<(u64, u64, bool), u64>>> = parking_lot::Mutex::new(None);

/// Device-ring bytes zeroed per batch for write-back page bitmaps (one bit
/// per 4 KiB page: 64 KiB cover 2 GiB of written ranges).
const BITS_ARENA_BYTES: u64 = 64 << 10;

/// Full barriers and render-pass begins recorded (TLG_PERF).
pub static BARRIERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PASS_BEGINS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A full barrier in a batch prologue: orders the prologue against earlier
/// batches and the batch after it, not commands of the batch against each
/// other, so `BARRIERS` (the hazard-tracking generation) stays.
fn prologue_barrier(cmd: vk::CommandBuffer) {
    PRO_BARRIERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
    unsafe { ctx().device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[mb], &[], &[]) };
}

/// Prologue barriers recorded (TLG_PERF).
pub static PRO_BARRIERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[track_caller]
fn global_barrier(cmd: vk::CommandBuffer) {
    BARRIERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ON.get_or_init(|| std::env::var_os("TLG_PERF").is_some()) {
        let line = std::panic::Location::caller().line();
        *BARRIER_SITES.lock().get_or_insert_with(HashMap::new).entry(line).or_default() += 1;
    }
    let c = ctx();
    let mb = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
    unsafe {
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[mb], &[], &[]);
    }
}

fn f32r(v: u32) -> f32 {
    f32::from_bits(v)
}

/// CB color FORMAT/NUMBER_TYPE/COMP_SWAP -> Vulkan format.
fn cb_format(fmt: u32, ntype: u32, swap: u32) -> Option<vk::Format> {
    let nfmt = match ntype {
        6 => 9,
        n => n as u8,
    };
    let f = desc::vk_format(fmt as u8, nfmt)?;
    Some(match (f, swap) {
        (vk::Format::R8G8B8A8_UNORM, 1) => vk::Format::B8G8R8A8_UNORM,
        (vk::Format::R8G8B8A8_SRGB, 1) => vk::Format::B8G8R8A8_SRGB,
        (vk::Format::R8G8B8A8_SNORM, 1) => vk::Format::B8G8R8A8_SNORM,
        _ => f,
    })
}

fn blend_factor(f: u32) -> vk::BlendFactor {
    use vk::BlendFactor as B;
    match f {
        0 => B::ZERO,
        1 => B::ONE,
        2 => B::SRC_COLOR,
        3 => B::ONE_MINUS_SRC_COLOR,
        4 => B::SRC_ALPHA,
        5 => B::ONE_MINUS_SRC_ALPHA,
        6 => B::DST_ALPHA,
        7 => B::ONE_MINUS_DST_ALPHA,
        8 => B::DST_COLOR,
        9 => B::ONE_MINUS_DST_COLOR,
        10 => B::SRC_ALPHA_SATURATE,
        13 => B::CONSTANT_COLOR,
        14 => B::ONE_MINUS_CONSTANT_COLOR,
        15 => B::SRC1_COLOR,
        16 => B::ONE_MINUS_SRC1_COLOR,
        17 => B::SRC1_ALPHA,
        18 => B::ONE_MINUS_SRC1_ALPHA,
        19 => B::CONSTANT_ALPHA,
        20 => B::ONE_MINUS_CONSTANT_ALPHA,
        _ => B::ONE,
    }
}

fn blend_op(f: u32) -> vk::BlendOp {
    match f {
        1 => vk::BlendOp::SUBTRACT,
        2 => vk::BlendOp::MIN,
        3 => vk::BlendOp::MAX,
        4 => vk::BlendOp::REVERSE_SUBTRACT,
        _ => vk::BlendOp::ADD,
    }
}

fn stencil_op(op: u32) -> vk::StencilOp {
    match op {
        1 => vk::StencilOp::ZERO,
        3 | 4 => vk::StencilOp::REPLACE,
        5 => vk::StencilOp::INCREMENT_AND_CLAMP,
        6 => vk::StencilOp::DECREMENT_AND_CLAMP,
        7 => vk::StencilOp::INVERT,
        8 => vk::StencilOp::INCREMENT_AND_WRAP,
        9 => vk::StencilOp::DECREMENT_AND_WRAP,
        _ => vk::StencilOp::KEEP,
    }
}

fn scissor_rect(tl: u32, br: u32) -> (i32, i32, i32, i32) {
    ((tl & 0x7FFF) as i32, ((tl >> 16) & 0x7FFF) as i32, (br & 0x7FFF) as i32, ((br >> 16) & 0x7FFF) as i32)
}

impl Renderer {
    fn report(&mut self, msg: String) {
        let n = self.reported.entry(msg.clone()).or_insert(0);
        *n += 1;
        if *n == 1 {
            crate::warn!("render: {msg}");
        }
    }

    fn begin(&mut self) {
        if self.recording {
            return;
        }
        let c = ctx();
        unsafe {
            c.device.begin_command_buffer(self.cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
        }
        global_barrier(self.cmd);
        // The page bitmaps of this batch's write-backs, zeroed once here
        // (no fill and barrier for each). After the batch-start barrier:
        // batches still running may read this segment (forwards and device
        // copies of in-flight writes keep their data until the slot's next
        // batch overwrites it after that barrier; a fill before it raced
        // with them and lost the device).
        self.bits_arena = match self.dring.alloc(BITS_ARENA_BYTES).filter(|_| crate::tune::get(crate::tune::BITS_ARENA) != 0) {
            Some(off) => {
                unsafe { c.device.cmd_fill_buffer(self.cmd, self.dring.buffer, off, BITS_ARENA_BYTES, 0) };
                global_barrier(self.cmd);
                (self.batch, off, 0, BITS_ARENA_BYTES / 4)
            }
            None => (u64::MAX, 0, 0, 0),
        };
        let exec = self.slots[self.cur].exec;
        if exec != vk::QueryPool::null() {
            unsafe {
                c.device.cmd_reset_query_pool(self.cmd, exec, 0, 2);
                c.device.cmd_write_timestamp(self.cmd, vk::PipelineStageFlags::TOP_OF_PIPE, exec, 0);
            }
        }
        self.recording = true;
        self.dyn_state = None;
        self.bound_gfx = vk::Pipeline::null();
        self.marks.clear();
        let q = self.slots[self.cur].queries;
        if q != vk::QueryPool::null() {
            unsafe {
                c.device.cmd_reset_query_pool(self.cmd, q, 0, PROFILE_QUERIES);
                c.device.cmd_write_timestamp(self.cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, q, 0);
            }
            self.marks.push("(batch start)".into());
        }
        // (Queries cannot span command buffers: an open window continues
        // with a new query.)
        if self.occl_windows > 0 {
            self.occl_begin_query();
        }
    }

    /// Starts a Vulkan occlusion query (outside rendering, in the command
    /// buffer being recorded).
    fn occl_begin_query(&mut self) {
        let Some(&pool) = OCCL_POOL.get() else { return };
        let q = self.occl_next % OCCL_QUERIES;
        self.occl_next = self.occl_next.wrapping_add(1);
        let flags = if ctx().occl_precise { vk::QueryControlFlags::PRECISE } else { vk::QueryControlFlags::empty() };
        unsafe {
            ctx().device.cmd_reset_query_pool(self.cmd, pool, q, 1);
            ctx().device.cmd_begin_query(self.cmd, pool, q, flags);
        }
        self.occl_active = Some(q);
    }

    fn occl_end_query(&mut self) {
        let (Some(&pool), Some(q)) = (OCCL_POOL.get(), self.occl_active.take()) else { return };
        unsafe { ctx().device.cmd_end_query(self.cmd, pool, q) };
        self.occl_ended.push(q);
    }

    /// PIXEL_PIPE_STAT_DUMP: ends the occlusion query counting since the
    /// previous dump, opens or closes a query window (`begin`: a begin
    /// sample) and keeps counting while windows are open. Returns the
    /// queries whose results the dump's write adds to the running total
    /// (see `occlusion_write`): differences between a query's end and begin
    /// samples are then the samples that passed in between.
    pub fn occlusion_dump(&mut self, begin: bool) -> Vec<u32> {
        self.begin();
        self.end_rendering();
        self.occl_end_query();
        self.occl_windows = if begin { self.occl_windows + 1 } else { self.occl_windows.saturating_sub(1) };
        if self.occl_windows > 0 {
            self.occl_begin_query();
        }
        std::mem::take(&mut self.occl_ended)
    }

    /// GPU profiling: timestamp after the operations recorded since the last
    /// mark, attributed to `label`.
    fn mark(&mut self, label: impl FnOnce() -> String) {
        let q = self.slots[self.cur].queries;
        let profile = q != vk::QueryPool::null() && self.recording && self.marks.len() < PROFILE_QUERIES as usize;
        let diag = self.recording && super::vk::checkpoints_enabled();
        if !profile && !diag {
            return;
        }
        let text = label();
        if diag {
            super::vk::checkpoint(self.cmd, text.clone());
        }
        if profile {
            unsafe { ctx().device.cmd_write_timestamp(self.cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, q, self.marks.len() as u32) };
            self.marks.push(text);
        }
    }

    fn end_rendering(&mut self) {
        if self.rendering.take().is_some() {
            unsafe { ctx().device.cmd_end_rendering(self.cmd) };
            unordered!(self);
        }
    }

    /// `flush` with accounting of why the synchronization happened (and,
    /// for TLG_PERF, from which source line).
    #[track_caller]
    pub fn flush_for(&mut self, why: usize) {
        if self.recording {
            self.flushes[why.min(5)] += 1;
            let loc = std::panic::Location::caller().line();
            *FLUSH_SITES.lock().get_or_insert_with(HashMap::new).entry(loc).or_default() += 1;
        }
        self.flush();
    }

    /// Keeps the GPU fed now that batches no longer end at every end-of-pipe
    /// drain: submits the batch once it holds `batch_max` commands, or
    /// `batch_idle` while the GPU has no submitted work left (after a
    /// synchronizing flush the GPU would otherwise idle until a whole batch
    /// is recorded). See `tune.rs`.
    fn maybe_submit(&mut self) {
        if !self.recording || self.batch_cmds == 0 {
            return;
        }
        let max = crate::tune::get(crate::tune::BATCH_MAX).max(1) as u64;
        let idle = crate::tune::get(crate::tune::BATCH_IDLE).max(1) as u64;
        if self.batch_cmds >= max || (self.batch_cmds >= idle && COMPLETIONS.lock().is_empty()) {
            self.submit_batch(Vec::new());
        }
    }

    /// The current batch's prologue command buffer (begun on first use,
    /// after a barrier behind earlier batches: ring segments they read are
    /// overwritten). It is submitted ahead of the batch with a barrier
    /// after it: for operations writing device-ring ranges nothing used yet
    /// from data fixed at record time (`tune prologue`).
    fn prologue_cmd(&mut self) -> vk::CommandBuffer {
        self.begin();
        let pro = self.slots[self.cur].pro;
        if !self.pro_recording {
            unsafe { ctx().device.begin_command_buffer(pro, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok() };
            prologue_barrier(pro);
            self.pro_recording = true;
        }
        self.pro_ops += 1;
        pro
    }

    /// Orders work recorded after this point behind all work recorded before
    /// it (the next command starts with a full barrier). Across batches the
    /// batch-start barrier does this already.
    pub fn order_barrier(&mut self) {
        if self.recording {
            self.end_rendering();
            unordered!(self);
        }
    }

    /// Starts a new texture validation epoch in which the CPU may have
    /// written guest memory (title submissions, direct CP memory accesses):
    /// textures, mirrored buffers and shader code are checked against the OS
    /// write watch / their hashes again. (Other epochs follow GPU writes,
    /// which are tracked as ranges.)
    pub fn new_epoch(&mut self, kind: EpochKind) {
        use std::sync::atomic::Ordering::Relaxed;
        self.epoch += 1;
        self.cpu_epoch += 1;
        self.writes.epoch = self.cpu_epoch;
        super::CPU_EPOCH.store(self.cpu_epoch, Relaxed);
        if kind != EpochKind::Compute {
            super::GFX_EPOCH.store(self.cpu_epoch, Relaxed);
        }
        if kind != EpochKind::Graphics {
            super::CS_EPOCH.store(self.cpu_epoch, Relaxed);
        }
    }

    /// Selects the CPU epoch validations must reach for the next command
    /// (draws and graphics-ring dispatches: graphics; compute queues: compute).
    fn set_required_epoch(&mut self, compute_queue: bool) {
        let e = if compute_queue { &super::CS_EPOCH } else { &super::GFX_EPOCH };
        self.required_epoch = e.load(std::sync::atomic::Ordering::Relaxed);
        self.writes.required = self.required_epoch;
        super::REQUIRED_EPOCH.with(|r| r.set(self.required_epoch));
    }

    /// Submits recorded work and waits until it, its buffer write-backs and
    /// all earlier end-of-pipe actions have completed.
    #[track_caller]
    pub fn flush(&mut self) {
        if !self.recording && *DONE.lock() >= self.last_seq {
            return;
        }
        if !self.writebacks.is_empty() {
            self.epoch += 1;
        }
        super::set_phase(super::phase::SUBMIT_WAIT, self.batch);
        let seq = self.submit_batch(Vec::new());
        let t0 = std::time::Instant::now();
        wait_done(seq);
        self.prof[0] += t0.elapsed();
    }

    /// Submits the recorded work (if any) without waiting; `actions` are
    /// performed after it and everything before it has completed. Returns the
    /// batch's sequence number.
    #[track_caller]
    pub fn submit_batch(&mut self, actions: Vec<super::cp::EopAction>) -> u64 {
        // (Forwards still queued belong to a command that was abandoned:
        // their ring ranges are used by nothing.)
        self.fwd_pending.clear();
        if self.recording && perf_on() {
            let loc = std::panic::Location::caller();
            let file = loc.file().rsplit(['/', '\\']).next().unwrap_or("");
            let mut g = BATCH_SITES.lock();
            let e = g.get_or_insert_with(HashMap::new).entry((file, loc.line())).or_default();
            e.0 += 1;
            e.1 += self.batch_cmds;
        }
        self.batch_cmds = 0;
        let mut pending = std::mem::take(&mut self.writebacks);
        // (In-flight forwards see the whole written ranges, lazy ones too.)
        let dev_writes: Vec<(u64, vk::Buffer, u64, u64)> = pending.iter().map(|p| (p.base, p.dev_buf, p.dev_off, p.size)).collect();
        // (Numbers of the skips, for `lazywb::submitted`.)
        let mut lazy = Vec::new();
        if self.recording && super::lazywb::enabled() {
            // Lazy ranges: only the partial pages at their ends are copied out.
            let mut kept = Vec::with_capacity(pending.len());
            for p in pending.drain(..) {
                let inner = super::lazywb::inner_pages(p.base, p.size).filter(|_| p.mirror);
                match inner.and_then(|(lo, hi)| super::lazywb::skip_writeback(lo, hi, p.dev_buf, p.dev_off + (lo - p.base)).map(|t| (lo, hi, t))) {
                    Some((lo, hi, t)) => {
                        lazy.push(t);
                        if lo > p.base {
                            kept.push(PendingWrite { base: p.base, size: lo - p.base, host_off: p.host_off, dev_off: p.dev_off, dev_buf: p.dev_buf, mirror: true, bits: None });
                        }
                        let end = p.base + p.size;
                        if end > hi {
                            kept.push(PendingWrite { base: hi, size: end - hi, host_off: p.host_off + (hi - p.base), dev_off: p.dev_off + (hi - p.base), dev_buf: p.dev_buf, mirror: true, bits: None });
                        }
                    }
                    None => kept.push(p),
                }
            }
            pending = kept;
        }
        if self.recording && !pending.is_empty() {
            // Shader-written data: device ring -> host ring, visible to the host
            // once the batch's fence signals. Ranges with a page bitmap: only
            // the marked pages, and the bitmap for the completion's write-back.
            self.end_rendering();
            global_barrier(self.cmd);
            if pending.iter().any(|p| p.bits.is_some()) && self.record_dirty_copies(&pending).is_err() {
                // (Without the kernel: whole ranges, as before.)
                for p in &mut pending {
                    p.bits = None;
                }
            }
            let mut srcs: smallvec::SmallVec<[vk::Buffer; 4]> = pending.iter().filter(|p| p.bits.is_none()).map(|p| p.dev_buf).collect();
            srcs.sort_unstable();
            srcs.dedup();
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            unsafe {
                let c = ctx();
                for src in srcs {
                    let regions: Vec<vk::BufferCopy> = pending.iter().filter(|p| p.dev_buf == src && p.bits.is_none()).map(|p| vk::BufferCopy { src_offset: p.dev_off, dst_offset: p.host_off, size: p.size }).collect();
                    c.device.cmd_copy_buffer(self.cmd, src, self.ring.buffer, &regions);
                }
                c.device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[mb], &[], &[]);
            }
            self.mark(|| "buffer write-back copies".into());
        }
        if perf_on() && !pending.is_empty() {
            let mut g = WRITEBACK_SITES.lock();
            let m = g.get_or_insert_with(HashMap::new);
            for p in &pending {
                *m.entry((p.base, p.size, p.mirror)).or_default() += 1;
            }
        }
        let writebacks: Vec<(u64, u64, u64)> = pending.iter().map(|p| (p.base, p.host_off, p.size)).collect();
        let wb_bits: Vec<Option<(u64, u32)>> = pending.iter().map(|p| p.bits.map(|(_, host, words)| (host, words))).collect();
        let mirror_writes: Vec<(u64, u64)> = pending.iter().filter(|p| p.mirror && p.bits.is_none()).map(|p| (p.base, p.size)).collect();
        let wb_mirror: Vec<bool> = pending.iter().map(|p| p.mirror).collect();
        let zc_writes = std::mem::take(&mut self.zc_writes);
        if self.recording && !zc_writes.is_empty() {
            // Shader writes to imported guest memory: visible to the host after the fence.
            self.end_rendering();
            let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
            unsafe { ctx().device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[mb], &[], &[]) };
        }
        if !self.recording {
            if actions.is_empty() && writebacks.is_empty() {
                return self.last_seq;
            }
            // Ordering-only completion (nothing new to wait for on the GPU).
            self.last_seq += 1;
            push_completion(Completion { seq: self.last_seq, fence: vk::Fence::null(), exec: vk::QueryPool::null(), ring_ptr: self.ring.ptr, writebacks, wb_bits, dev_writes, mirror_writes, zc_writes, zc_reads: Ranges::default(), wb_mirror, actions, submitted: std::time::Instant::now(), profile: None, cpu_span: None });
            // (In COMPLETIONS now: the view may forget them.)
            CP_VIEW.lock().pending = Ranges::new();
            return self.last_seq;
        }
        self.end_rendering();
        self.occl_end_query();
        let c = ctx();
        // Memory for sparse mirror blocks this batch uses: bound first.
        let mut waits = smallvec::SmallVec::<[(vk::Semaphore, vk::PipelineStageFlags); 1]>::new();
        if !self.sparse_binds.is_empty() {
            let binds = std::mem::take(&mut self.sparse_binds);
            match c.bind_sparse(&binds, self.bind_sem) {
                Ok(()) => waits.push((self.bind_sem, vk::PipelineStageFlags::ALL_COMMANDS)),
                Err(e) => crate::error!("render: {e}"),
            }
        }
        let slot = &mut self.slots[self.cur];
        unsafe {
            if slot.exec != vk::QueryPool::null() {
                c.device.cmd_write_timestamp(slot.cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, slot.exec, 1);
            }
            c.device.end_command_buffer(slot.cmd).ok();
            let r = if self.pro_recording {
                prologue_barrier(slot.pro);
                c.device.end_command_buffer(slot.pro).ok();
                c.submit_cmds(&[slot.pro, slot.cmd], &waits, &[], slot.fence)
            } else {
                c.submit(slot.cmd, &waits, &[], slot.fence)
            };
            self.pro_recording = false;
            if let Err(e) = r {
                crate::error!("render submit: {e}");
                if e.contains("lost") {
                    super::vk::report_device_lost();
                }
            }
        }
        if let Some(&t) = lazy.last() {
            super::lazywb::submitted(t);
        }
        self.last_seq += 1;
        slot.seq = self.last_seq;
        let profile = (slot.queries != vk::QueryPool::null() && self.marks.len() > 1).then(|| (slot.queries, std::mem::take(&mut self.marks)));
        let zc_reads = std::mem::take(&mut self.zc_reads);
        push_completion(Completion { seq: self.last_seq, fence: slot.fence, exec: slot.exec, ring_ptr: self.ring.ptr, writebacks, wb_bits, dev_writes, mirror_writes, zc_writes, zc_reads, wb_mirror, actions, submitted: std::time::Instant::now(), profile, cpu_span: None });
        // (In COMPLETIONS now: the view may forget them.)
        {
            let mut v = CP_VIEW.lock();
            v.pending = Ranges::new();
            v.reads = Ranges::new();
        }
        self.recording = false;
        self.batch += 1;
        // Move to the next slot once its previous batch has completed.
        self.cur = (self.cur + 1) % SLOTS;
        let next = &self.slots[self.cur];
        if next.seq != 0 {
            // (Back-pressure: the GPU has not finished the batch recorded
            // SLOTS batches ago.)
            let t0 = std::time::Instant::now();
            wait_done(next.seq);
            self.prof[6] += t0.elapsed();
            unsafe {
                c.device.reset_fences(&[next.fence]).ok();
                c.device.reset_command_buffer(next.cmd, vk::CommandBufferResetFlags::empty()).ok();
                c.device.reset_command_buffer(next.pro, vk::CommandBufferResetFlags::empty()).ok();
                c.device.reset_descriptor_pool(next.desc_pool, vk::DescriptorPoolResetFlags::empty()).ok();
            }
        }
        self.cmd = next.cmd;
        self.desc_pool = next.desc_pool;
        // (Copies in the previous segment: the CPU may overwrite them while
        // the GPU still runs the batch that would read them.)
        self.upload_reuse.clear();
        self.ring.set_segment(self.cur, SLOTS);
        self.dring.set_segment(self.cur, SLOTS);
        if let Some(b) = self.bar.as_mut() {
            b.set_segment(self.cur, SLOTS);
        }
        self.last_seq
    }

    fn module(&mut self, key: u64, spirv: &[u32]) -> Result<vk::ShaderModule, String> {
        if let Some(&m) = self.modules.get(&key) {
            return Ok(m);
        }
        let m = unsafe { ctx().device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spirv), None) }.map_err(|e| format!("vkCreateShaderModule: {e}"))?;
        self.modules.insert(key, m);
        Ok(m)
    }

    fn set_layout(&mut self, stages: &[&Prepared]) -> Result<vk::DescriptorSetLayout, String> {
        let mut bindings = Vec::new();
        for p in stages {
            let flags = match (p.stage, p.tess) {
                (Stage::Vertex, TessRole::Ds) => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
                (Stage::Vertex, _) => vk::ShaderStageFlags::VERTEX,
                (Stage::Fragment, _) => vk::ShaderStageFlags::FRAGMENT,
                (Stage::Compute, _) => vk::ShaderStageFlags::COMPUTE,
            };
            if p.tess == TessRole::Ds {
                // The generated control stage's factor buffer.
                bindings.push(vk::DescriptorSetLayoutBinding::default().binding(TESS_TF_BINDING).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::TESSELLATION_CONTROL));
            }
            for b in &p.translated.out.bindings {
                let ty = match b.kind {
                    BindingKind::SgprData | BindingKind::Buffer(_) | BindingKind::Gds | BindingKind::DebugOut | BindingKind::Lds | BindingKind::DirtyInfo | BindingKind::DirtyBits => vk::DescriptorType::STORAGE_BUFFER,
                    BindingKind::Image(_) => vk::DescriptorType::SAMPLED_IMAGE,
                    BindingKind::StorageImage(_) => vk::DescriptorType::STORAGE_IMAGE,
                    BindingKind::Sampler(_) => vk::DescriptorType::SAMPLER,
                };
                bindings.push(vk::DescriptorSetLayoutBinding::default().binding(b.binding).descriptor_type(ty).descriptor_count(1).stage_flags(flags));
            }
        }
        unsafe { ctx().device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None) }.map_err(|e| format!("descriptor set layout: {e}"))
    }

    fn rect_gs(&mut self, params: u32) -> Result<vk::ShaderModule, String> {
        let key = 0x5245_4354_0000_0000 | params as u64;
        if let Some(&m) = self.modules.get(&key) {
            return Ok(m);
        }
        let spv = gcn::translate::gs::rect_list_gs(params);
        self.module(key, &spv)
    }

    // -------------------------------------------------------------------
    // Images

    fn register(&mut self, img: Image) {
        let k = img.key;
        self.by_base.entry(k.base).or_default().push(k);
        // (Tiled surfaces pad to macro tiles -- at least 64 KiB -- and texture
        // pitches may exceed the width: twice the linear size bounds both.
        // Block formats by their blocks, other formats without a class at 16
        // bytes per texel: format_class's 0x1000+ codes made a 1024x1024 BC1
        // texture "8.8 GB", so `note_buffer_write` scanned nearly all images
        // for every buffer a shader wrote.)
        let linear = match bc_block_bits(k.format) {
            Some(bits) => k.width.div_ceil(4) as u64 * k.height.div_ceil(4) as u64 * (bits / 8) as u64,
            None => k.width as u64 * k.height as u64 * match format_class(k.format) { c @ 1..=16 => c.max(4), _ => 16 } as u64,
        };
        let bytes = (linear * k.layers.max(1) as u64 * 2).max(256 << 10);
        self.img_bases.entry(k.base).or_default().push((k, bytes));
        if bytes > IMG_NEAR_BYTES {
            self.img_bases_large.entry(k.base).or_default().push((k, bytes));
        }
        self.img_max_bytes = self.img_max_bytes.max(bytes);
        self.image_bytes += img.bytes;
        self.images.insert(k, img);
    }

    fn init_image(&mut self, key: ImageKey) -> Result<(), String> {
        if self.images.contains_key(&key) {
            return Ok(());
        }
        if let Some(at) = self.evicted_keys.remove(&key) {
            let gap = super::FRAMES.load(std::sync::atomic::Ordering::Relaxed).saturating_sub(at);
            self.recreated.0 += 1;
            self.recreated.1 += gap;
            if self.recreated.0 <= 20 || gap < 300 {
                crate::info!("VRAM: evicted image needed again after {gap} frames: {:#x} {}x{}x{} {:?}", key.base, key.width, key.height, key.layers, key.format);
            }
        }
        let pooled = if crate::tune::get(crate::tune::IMAGE_POOL) != 0 { self.image_pool.get_mut(&ImageKey { base: 0, ..key }).and_then(|v| v.pop()) } else { None };
        let img = match pooled {
            Some(p) => {
                // (Same creation parameters; cleared below like a new one.)
                self.image_pool_n -= 1;
                self.image_reuse.1 += 1;
                resources::reuse_image(key, p)
            }
            None => {
                self.image_reuse.0 += 1;
                resources::create_image(key)?
            }
        };
        if perf_on() {
            let mut g = NEW_IMAGES.lock();
            let e = g.get_or_insert_with(HashMap::new).entry((key.width, key.height, key.layers, key.levels, key.format)).or_default();
            e.0 += 1;
            e.1 = key.base;
        }
        self.begin();
        self.end_rendering();
        let c = ctx();
        let b = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(img.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: resources::aspect(key.format) | if key.format == vk::Format::D32_SFLOAT_S8_UINT { vk::ImageAspectFlags::STENCIL } else { vk::ImageAspectFlags::empty() },
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            });
        unsafe {
            c.device.cmd_pipeline_barrier(self.cmd, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &[b]);
            let range = vk::ImageSubresourceRange {
                aspect_mask: resources::aspect(key.format),
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            };
            if resources::is_depth(key.format) {
                let mut r = range;
                if key.format == vk::Format::D32_SFLOAT_S8_UINT {
                    r.aspect_mask |= vk::ImageAspectFlags::STENCIL;
                }
                c.device.cmd_clear_depth_stencil_image(self.cmd, img.image, vk::ImageLayout::GENERAL, &vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 }, &[r]);
            } else if !matches!(key.format.as_raw(), 131..=146) {
                c.device.cmd_clear_color_image(self.cmd, img.image, vk::ImageLayout::GENERAL, &vk::ClearColorValue::default(), &[range]);
            }
        }
        unordered!(self);
        self.register(img);
        Ok(())
    }

    /// Render target image for a color buffer (padded dimensions from pitch/slice).
    fn color_target(&mut self, regs: &Regs, n: u32) -> Result<Option<Attachment>, String> {
        let r = 0xA318 + n * 0xF;
        let info = regs.ctx(r + 4);
        let fmt = (info >> 2) & 0x1F;
        if fmt == 0 {
            return Ok(None);
        }
        let base = (regs.ctx(r) as u64) << 8;
        let pitch = ((regs.ctx(r + 1) & 0x7FF) + 1) * 8;
        let slice = ((regs.ctx(r + 2) & 0x3F_FFFF) + 1) * 64;
        let height = slice / pitch;
        // CB_COLOR*_VIEW: SLICE_START..=SLICE_MAX of an array. The title
        // renders arrays slice by slice (SLICE_START = SLICE_MAX = n); the
        // slices live in one image with at least SLICE_MAX + 1 layers (the
        // largest compatible one at this address, e.g. the array the
        // texture descriptors describe).
        let view = regs.ctx(r + 3);
        let layers = ((view >> 13) & 0x7FF) + 1;
        let slice_start = view & 0x7FF;
        let Some(format) = cb_format(fmt, (info >> 8) & 7, (info >> 11) & 3) else {
            self.report(format!("unsupported color format {fmt} type {} swap {}", (info >> 8) & 7, (info >> 11) & 3));
            return Ok(None);
        };
        // Storage class: use a canonical (non-sRGB) image format and an sRGB view where needed.
        let storage_fmt = canonical(format);
        let key = ImageKey { base, width: pitch, height, layers: layers.max(1), levels: 1, format: storage_fmt, depth: false, ty3d: false };
        let key = self.find_compatible(key).unwrap_or(key);
        self.init_image(key)?;
        if self.images[&key].guest_newer {
            self.refresh_from_guest(key, true)?;
        }
        self.sync_alias(key)?;
        // CB_COLOR*_ATTRIB.TILE_MODE_INDEX: guest surface layout for write-back.
        let img = self.images.get_mut(&key).unwrap();
        img.tile_index = Some((regs.ctx(r + 5) & 0x1F) as u8);
        // CB_COLOR*_INFO.FAST_CLEAR: CMASK (4 bits per 8x8 tile) tracks fast clears.
        let cmask = (regs.ctx(r + 7) as u64) << 8;
        img.meta = (info & (1 << 13) != 0 && cmask != 0).then(|| (cmask, (pitch.div_ceil(8) as u64 * height.div_ceil(8) as u64).div_ceil(2)));
        if img.meta.is_some() && !img.meta_listed {
            img.meta_listed = true;
            self.meta_images.push(key);
        }
        Ok(Some(Attachment { key, view_format: format, layer: slice_start.min(key.layers.max(1) - 1) }))
    }

    fn depth_target(&mut self, regs: &Regs) -> Result<Option<Attachment>, String> {
        let zinfo = regs.ctx(0xA010);
        let zfmt = zinfo & 3;
        let sinfo = regs.ctx(0xA011);
        let has_stencil = sinfo & 1 != 0;
        if zfmt == 0 && !has_stencil {
            return Ok(None);
        }
        let base = (regs.ctx(0xA014) as u64) << 8;
        let size = regs.ctx(0xA016);
        let pitch = ((size & 0x7FF) + 1) * 8;
        let height = (((size >> 11) & 0x7FF) + 1) * 8;
        let format = match (zfmt, has_stencil) {
            (1, false) => vk::Format::D16_UNORM,
            (_, true) => vk::Format::D32_SFLOAT_S8_UINT,
            _ => vk::Format::D32_SFLOAT,
        };
        let key = ImageKey { base, width: pitch, height, layers: 1, levels: 1, format, depth: true, ty3d: false };
        self.init_image(key)?;
        if self.images[&key].guest_newer {
            self.refresh_from_guest(key, true)?;
        }
        // DB_Z_INFO.TILE_MODE_INDEX
        let img = self.images.get_mut(&key).unwrap();
        img.tile_index = Some(((zinfo >> 20) & 7) as u8);
        // DB_Z_INFO.TILE_SURFACE_ENABLE: HTILE (one dword per 8x8 tile).
        let htile = (regs.ctx(0xA005) as u64) << 8;
        img.meta = (zinfo & (1 << 29) != 0 && htile != 0).then(|| (htile, pitch.div_ceil(8) as u64 * height.div_ceil(8) as u64 * 4));
        if img.meta.is_some() && !img.meta_listed {
            img.meta_listed = true;
            self.meta_images.push(key);
        }
        Ok(Some(Attachment { key, view_format: format, layer: 0 }))
    }

    /// An existing image at the same base with the same size and format
    /// class and at least as many layers (the one with the most: views of
    /// single slices and whole arrays share the array's image).
    fn find_compatible(&self, key: ImageKey) -> Option<ImageKey> {
        self.by_base
            .get(&key.base)?
            .iter()
            .copied()
            .filter(|k| k.width == key.width && k.height == key.height && k.layers >= key.layers && k.ty3d == key.ty3d && k.depth == key.depth && format_class(k.format) == format_class(key.format))
            .max_by_key(|k| k.layers)
    }

    /// A view of mip level `level` (`levels` = the descriptor's last level
    /// + 1) when the image at its address has fewer levels: chains written
    /// level by level (depth pyramids: level 0 by one kernel, then each
    /// level from the previous one) start as single-level images. Replaces
    /// the image with one of `levels` levels holding its old levels (views
    /// would otherwise clamp to the last existing level and every level
    /// would land in level 0). Not for this draw's attachments.
    fn grow_levels(&mut self, old: ImageKey, level: u32, levels: u32) -> Result<ImageKey, String> {
        if level < old.levels.max(1) || levels <= old.levels || old.depth || old.ty3d || self.draw_targets.contains(&old) {
            return Ok(old);
        }
        let new = ImageKey { levels, ..old };
        self.init_image(new)?;
        self.begin();
        self.end_rendering();
        if self.barrier_needed {
            global_barrier(self.cmd);
            self.barrier_needed = false;
        }
        let regions: Vec<vk::ImageCopy> = (0..old.levels.max(1))
            .map(|l| {
                let sub = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: l, base_array_layer: 0, layer_count: old.layers.max(1) };
                vk::ImageCopy { src_subresource: sub, src_offset: vk::Offset3D::default(), dst_subresource: sub, dst_offset: vk::Offset3D::default(), extent: vk::Extent3D { width: (old.width >> l).max(1), height: (old.height >> l).max(1), depth: 1 } }
            })
            .collect();
        unsafe { ctx().device.cmd_copy_image(self.cmd, self.images[&old].image, vk::ImageLayout::GENERAL, self.images[&new].image, vk::ImageLayout::GENERAL, &regions) };
        unordered!(self);
        self.mark(|| "level grow copy".into());
        // The new image takes over the old one's state and index entries.
        let o = self.images.remove(&old).unwrap();
        let n = self.images.get_mut(&new).unwrap();
        n.gpu_written = o.gpu_written;
        n.uploaded_hash = o.uploaded_hash;
        n.validated = o.validated;
        n.upload_batch = o.upload_batch;
        n.storage_ok = o.storage_ok;
        n.tile_index = o.tile_index;
        n.cpu_checked = o.cpu_checked;
        n.pitch = o.pitch;
        n.guest_stale = o.guest_stale;
        n.stale_stamp = o.stale_stamp;
        n.reuse_checked = o.reuse_checked;
        n.edges = o.edges;
        n.last_write = o.last_write;
        n.last_write_frame = o.last_write_frame;
        n.meta = o.meta;
        n.meta_dirty = o.meta_dirty;
        n.meta_fill = o.meta_fill;
        n.meta_listed = o.meta_listed;
        n.guest_newer = o.guest_newer;
        n.last_used = o.last_used;
        for k in self.meta_images.iter_mut().chain(self.stale.values_mut().flatten()) {
            if *k == old {
                *k = new;
            }
        }
        if let Some(v) = self.by_base.get_mut(&old.base) {
            v.retain(|x| *x != old);
        }
        if let Some(v) = self.img_bases.get_mut(&old.base) {
            v.retain(|x| x.0 != old);
        }
        if let Some(v) = self.img_bases_large.get_mut(&old.base) {
            v.retain(|x| x.0 != old);
        }
        if let Some(g) = self.write_gen.remove(&old) {
            self.write_gen.insert(new, g);
        }
        self.shadows.retain(|sk, (dk, _)| sk.0 != old && *dk != old);
        self.image_bytes -= o.bytes;
        // (Used by the batch being recorded: freed once it completes.)
        self.retired.push((self.last_seq + 1, Retired::Image(o.image, o.memory, o.views.values().copied().collect())));
        self.grown += 1;
        if self.grown <= 20 {
            crate::info!("image {:#x} {}x{}x{} {:?}: {} -> {} mip levels (view of level {level})", old.base, old.width, old.height, old.layers, old.format, old.levels, levels);
        }
        Ok(new)
    }

    /// Several images can alias one guest surface with different sizes: a
    /// render target with padded dimensions (1920x1088) and a texture or
    /// storage image of the visible size (1920x1080) written by compute
    /// shaders. Before `dst` is used, copies in the contents of the most
    /// recently written other image at its address when that one is newer.
    fn sync_alias(&mut self, dst: ImageKey) -> Result<(), String> {
        if dst.depth {
            return Ok(());
        }
        let Some(dlast) = self.images.get(&dst).map(|i| i.last_write) else { return Ok(()) };
        let src = self.by_base.get(&dst.base).and_then(|ks| {
            ks.iter()
                .copied()
                .filter(|k| *k != dst && !k.depth && k.ty3d == dst.ty3d && format_class(k.format) == format_class(dst.format) && same_surface(k.width, k.height, dst.width, dst.height))
                .filter(|k| self.images[k].gpu_written && self.images[k].last_write > dlast)
                .max_by_key(|k| self.images[k].last_write)
        });
        let Some(src) = src else { return Ok(()) };
        let (w, h, layers) = (src.width.min(dst.width), src.height.min(dst.height), src.layers.min(dst.layers).max(1));
        self.begin();
        self.end_rendering();
        if self.barrier_needed {
            global_barrier(self.cmd);
            self.barrier_needed = false;
        }
        let sub = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: layers };
        let region = vk::ImageCopy { src_subresource: sub, src_offset: vk::Offset3D::default(), dst_subresource: sub, dst_offset: vk::Offset3D::default(), extent: vk::Extent3D { width: w, height: h, depth: 1 } };
        unsafe { ctx().device.cmd_copy_image(self.cmd, self.images[&src].image, vk::ImageLayout::GENERAL, self.images[&dst].image, vk::ImageLayout::GENERAL, &[region]) };
        unordered!(self);
        self.mark(|| "alias copy".into());
        let last = self.images[&src].last_write;
        let d = self.images.get_mut(&dst).unwrap();
        d.last_write = last;
        d.gpu_written = true;
        *self.write_gen.entry(dst).or_default() += 1;
        if trace_frame() {
            crate::info!("FRAME alias {:#x}: {}x{} {:?} updated from newer {}x{} {:?}", dst.base, dst.width, dst.height, dst.format, src.width, src.height, src.format);
        }
        Ok(())
    }

    /// Adds `key` to the index of images that may be newer than guest memory.
    fn mark_stale(&mut self, key: ImageKey) {
        // Becoming stale: CPU writes to the image's memory from now on mean
        // the title has reused it (see `writeback_images`).
        if let Some(total) = self.images.get(&key).filter(|i| !i.guest_stale).and_then(|i| Self::image_guest_layout(&key, i)).map(|g| g.3) {
            let (_, stamp) = self.writes.cpu_written_since(key.base, total, u64::MAX);
            if let Some(i) = self.images.get_mut(&key) {
                i.stale_stamp = stamp;
            }
        }
        let ks = self.stale.entry(key.base).or_default();
        if !ks.contains(&key) {
            ks.push(key);
            if let Some(i) = self.images.get(&key) {
                if let Some(g) = Self::image_guest_layout(&key, i) {
                    self.stale_max = self.stale_max.max(g.3);
                }
            }
        }
    }

    /// Registers GPU work in the current batch that writes guest memory
    /// [base, base+len) (in-place stores, write-backs at completion).
    fn note_pending_gpu_write(&mut self, base: u64, len: u64) {
        self.vcache_invalidate(base, len);
        let seq = self.last_seq + 1;
        if perf_on() {
            if self.pending_log.len() >= 4096 {
                self.pending_log.drain(..2048);
            }
            self.pending_log.push((base, len, self.cur_shader, seq));
        }
        if len > SURFACE_WRITE_MAX {
            self.gpu_pending_all = seq;
            return;
        }
        // (Entries of this batch stay until it completes: a range noted for
        // it already is mapped.)
        if self.pending_noted.0 != seq {
            self.pending_noted.0 = seq;
            self.pending_noted.1.clear();
        }
        if !self.pending_noted.1.insert((base, len)) {
            return;
        }
        for p in (base >> 12)..=((base + len.max(1) - 1) >> 12) {
            self.gpu_pending_pages.insert(p, seq);
        }
    }

    /// GPU work of this batch writes guest [base, base+len) in the mirror
    /// window `buf` itself: copies elsewhere are stale (`note_pending_gpu_write`),
    /// `buf` holds the newest data, and bindings it served stay current (its
    /// generation is kept, `vbind`).
    fn note_mirror_write(&mut self, buf: vk::Buffer, base: u64, len: u64) {
        self.vgen_keep = buf;
        self.note_pending_gpu_write(base, len);
        self.vgen_keep = vk::Buffer::null();
        self.vcache_validate(buf, base, len);
    }

    /// Before record-time reads of guest memory by shader preparation
    /// (descriptor tables, scalar loads): lands pending GPU writes to the
    /// pages read. Returns whether it synchronized (the shader must then be
    /// prepared again).
    fn guard_reads(&mut self, reads: &[u64]) -> bool {
        if reads.is_empty() {
            return false;
        }
        let done = *DONE.lock();
        let mut need = if self.gpu_pending_all > done { self.gpu_pending_all } else { 0 };
        for &a in reads {
            if let Some(&s) = self.gpu_pending_pages.get(&(a >> 12)) {
                if s > done {
                    need = need.max(s);
                }
            }
        }
        if self.gpu_pending_pages.len() > 1 << 16 {
            self.gpu_pending_pages.retain(|_, s| *s > done);
        }
        if need == 0 {
            return false;
        }
        self.prepare_syncs += 1;
        if perf_on() {
            // What was read, by which command, and the pending write that
            // covers it (this batch's write-backs, else in flight).
            let mut g = GUARD_SITES.lock();
            let m = g.get_or_insert_with(HashMap::new);
            for &a in reads {
                if !self.gpu_pending_pages.get(&(a >> 12)).is_some_and(|&s| s > done) && self.gpu_pending_all <= done {
                    continue;
                }
                let w = self.writebacks.iter().rev().find(|w| w.base <= a && a < w.base + w.size).map(|w| (w.base, w.size, w.mirror, "this batch"));
                let w = w.or_else(|| COMPLETIONS.lock().iter().rev().find_map(|c| c.dev_writes.iter().find(|d| d.0 <= a && a < d.0 + d.3).map(|d| (d.0, d.3, true, "in flight"))));
                let who = self.pending_log.iter().rev().find(|e| e.0 <= a && a < e.0 + e.1.max(1)).map(|e| format!("; registered by {:#x} as {:#x}+{:#x}", e.2, e.0, e.1)).unwrap_or_default();
                let k = match w {
                    Some((b, n, mirror, when)) => format!("{:#x}/{:#x} reads {:#x} (+{:#x} into {b:#x}+{n:#x}, {}{when}{who})", self.prep_vs, self.prep_shader, a, a - b, if mirror { "mirror, " } else { "" }),
                    None => format!("{:#x}/{:#x} reads {:#x} (no covering write-back{who})", self.prep_vs, self.prep_shader, a),
                };
                *m.entry(k).or_default() += 1;
                break;
            }
        }
        if need > self.last_seq {
            self.flush_for(1);
        } else {
            let t0 = std::time::Instant::now();
            wait_done(need);
            SYNC_WAIT_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        }
        true
    }

    /// Byte ranges of the destination dwords the scatter kernel's records
    /// write in a dispatch of `groups` workgroups (merged).
    fn scatter_ranges(&self, cs: &Prepared, records: usize, dst: usize, groups: u64) -> Option<Vec<(u64, u64)>> {
        let (ResolvedRes::Buffer(rv), ResolvedRes::Buffer(dv)) = (&cs.resources[records], &cs.resources[dst]) else { return None };
        let stride = if rv.stride == 0 { 12 } else { rv.stride as u64 };
        let n = groups.min(rv.num_records as u64).min(1 << 20);
        let mut raw = vec![0u8; (n * stride) as usize];
        resources::copy_from_guest(&mut raw, rv.base);
        let limit = dv.size() / 4;
        let mut v: Vec<(u64, u64)> = (0..n as usize)
            .filter_map(|r| {
                let w = |k: usize| u32::from_le_bytes(raw[r * stride as usize + 4 * k..r * stride as usize + 4 * k + 4].try_into().unwrap()) as u64;
                let (d, cnt) = (w(0), w(2).min(63) + 1);
                let end = (d + cnt).min(limit);
                (d < end).then(|| (dv.base + d * 4, dv.base + end * 4))
            })
            .collect();
        v.sort_unstable();
        let mut out: Vec<(u64, u64)> = Vec::with_capacity(v.len());
        for (lo, hi) in v {
            match out.last_mut() {
                Some(last) if lo <= last.1 + 64 => last.1 = last.1.max(hi),
                _ => out.push((lo, hi)),
            }
        }
        Some(out.into_iter().map(|(lo, hi)| (lo, hi - lo)).collect())
    }

    /// Copies the device-ring working copies of buffers the command just
    /// wrote back to guest memory (GPU timeline).
    fn perform_gpu_writebacks(&mut self) {
        if self.gpu_wb.is_empty() {
            return;
        }
        self.end_rendering();
        global_barrier(self.cmd);
        let c = ctx();
        for (dev_off, zbuf, zoff, size) in std::mem::take(&mut self.gpu_wb) {
            unsafe { c.device.cmd_copy_buffer(self.cmd, self.dring.buffer, zbuf, &[vk::BufferCopy { src_offset: dev_off, dst_offset: zoff, size }]) };
        }
        unordered!(self);
        self.mark(|| "written buffer write-back copies (GPU)".into());
    }

    /// The image (not depth, not 3D) whose guest surface is exactly
    /// [base, base+bytes), most recently written first.
    fn surface_image(&self, base: u64, bytes: u64) -> Option<ImageKey> {
        self.by_base.get(&base)?.iter().copied().filter(|k| !k.depth && !k.ty3d && Self::image_guest_layout(k, &self.images[k]).is_some_and(|g| g.3 == bytes)).max_by_key(|k| self.images[k].last_write)
    }

    /// An array image (not depth, not 3D) one of whose layers' guest slices
    /// is exactly [base, base+bytes): (image, layer), most recently written
    /// first.
    fn surface_layer(&self, base: u64, bytes: u64) -> Option<(ImageKey, u32)> {
        if bytes == 0 {
            return None;
        }
        let first = if crate::tune::get(crate::tune::ELIDE_EXT) & 8 != 0 { 0 } else { 1 };
        (first..64u64)
            .filter_map(|l| {
                let b = base.checked_sub(l * bytes)?;
                let ks = self.by_base.get(&b)?;
                ks.iter()
                    .copied()
                    .filter(|k| !k.depth && !k.ty3d && k.layers as u64 > l && Self::image_guest_layout(k, &self.images[k]).is_some_and(|g| g.3 == bytes * k.layers as u64))
                    .max_by_key(|k| self.images[k].last_write)
                    .map(|k| (k, l as u32))
            })
            .next()
    }

    /// Marks `key` as rendered now (its guest memory is stale).
    fn note_image_written(&mut self, key: ImageKey) {
        let tick = self.draws + self.dispatches + 1;
        let frame = super::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
        if let Some(img) = self.images.get_mut(&key) {
            img.gpu_written = true;
            img.guest_stale = true;
            img.guest_newer = false;
            img.last_write = tick;
            img.last_write_frame = frame;
            self.mark_stale(key);
        }
        *self.write_gen.entry(key).or_default() += 1;
    }

    /// Performs an engine copy/fill kernel without running it. Copies of a
    /// whole render-target surface from its current image become image
    /// copies, zero/float fills of a target (or one array layer) become
    /// clears; guest memory is written back only if something reads it.
    /// Other copies/fills whose bytes are known are done in guest memory in
    /// place on the GPU (no device-ring round trip) with the kernel's effect
    /// on images: those there become older than guest memory. (Clearing an
    /// image for those is not equivalent: a surface can have aliases of
    /// other sizes kept coherent by recency.) Returns false to run the kernel.
    fn elide_utility(&mut self, cs: &Prepared, u: shaders::Utility, groups: u64) -> Result<bool, String> {
        if let shaders::Utility::Pattern { src, dst, params, modulo } = u {
            // SOTC's pattern copy as the fill or copy it amounts to.
            let (ResolvedRes::Buffer(sv), ResolvedRes::Buffer(dv), ResolvedRes::Buffer(pv)) = (&cs.resources[src], &cs.resources[dst], &cs.resources[params]) else { return Ok(false) };
            // (Same element format both ways: the elements are copied as they are.)
            if sv.stride != dv.stride || sv.dfmt != dv.dfmt || sv.nfmt != dv.nfmt || !matches!(dv.stride, 1 | 2 | 4) || resources::element_bits(dv.dfmt).0 != dv.stride * 8 {
                return Ok(false);
            }
            self.guard_reads(&[pv.base, pv.base + 4]);
            let mut raw = [0u8; 8];
            resources::copy_from_guest(&mut raw, pv.base);
            let (count, m) = (u32::from_le_bytes(raw[..4].try_into().unwrap()) as u64, u32::from_le_bytes(raw[4..].try_into().unwrap()) as u64);
            let n = (groups * 64).min(count).min(dv.num_records as u64);
            let period = if modulo { m } else if (m + 1).is_power_of_two() { m + 1 } else { 0 };
            if n == 0 || period == 0 {
                return Ok(false);
            }
            let bytes = n * dv.stride as u64;
            if period == 1 {
                self.guard_reads(&[sv.base]);
                let mut elem = vec![0u8; dv.stride as usize];
                resources::copy_from_guest(&mut elem, sv.base);
                let float = elem.iter().all(|&b| b == 0) || (dv.nfmt == 7 && matches!(dv.dfmt, 2 | 4 | 5 | 11 | 12 | 14));
                let r = self.elide_fill(dv, bytes, elem.clone(), float, "pattern fill");
                if perf_on() {
                    static SEEN: Mutex<Option<std::collections::HashSet<(u64, u64)>>> = Mutex::new(None);
                    if SEEN.lock().get_or_insert_with(Default::default).insert((dv.base, bytes)) {
                        let img = self.surface_image(dv.base, bytes).map(|k| format!("{}x{} {:?}", k.width, k.height, k.format));
                        crate::info!("pattern fill of {:#x}+{bytes:#x} (dfmt {} nfmt {}) with {elem:02x?}: {:?} as {} (surface image {img:?})", dv.base, dv.dfmt, dv.nfmt, r, self.last_elide);
                    }
                }
                return r;
            }
            if period >= n && n <= sv.num_records as u64 {
                return self.elide_copy(sv, dv, bytes);
            }
            // A period of one pixel over a whole render target (TLG: two
            // dwords over a 1024x576 RGBA16F target): a clear to that pixel.
            let px = period * dv.stride as u64;
            if !modulo && px <= 16 && n % period == 0 && period <= sv.num_records as u64 {
                if let Some(k) = self.surface_image(dv.base, bytes).filter(|k| format_class(k.format) as u64 == px) {
                    self.guard_reads(&[sv.base, sv.base + px - 4]);
                    let mut elem = vec![0u8; px as usize];
                    resources::copy_from_guest(&mut elem, sv.base);
                    // (Half floats: normal or zero ones convert exactly.)
                    let halves_ok = !matches!(k.format, vk::Format::R16_SFLOAT | vk::Format::R16G16_SFLOAT | vk::Format::R16G16B16A16_SFLOAT)
                        || elem.chunks(2).all(|h| {
                            let v = u16::from_le_bytes([h[0], h[1]]);
                            let ex = (v >> 10) & 0x1F;
                            ex != 31 && (ex != 0 || v & 0x3FF == 0)
                        });
                    if halves_ok {
                        let pdv = desc::Vsharp { stride: px as u32, ..*dv };
                        let r = self.elide_fill(&pdv, bytes, elem.clone(), true, "pattern pixel fill");
                        if perf_on() {
                            static SEEN: Mutex<Option<std::collections::HashSet<(u64, u64)>>> = Mutex::new(None);
                            if SEEN.lock().get_or_insert_with(Default::default).insert((dv.base, bytes)) {
                                crate::info!("pattern pixel fill of {:#x}+{bytes:#x} ({}x{} {:?}) with {elem:02x?}: {:?} as {}", dv.base, k.width, k.height, k.format, r, self.last_elide);
                            }
                        }
                        return r;
                    }
                }
            }
            return Ok(false);
        }
        let ext = crate::tune::get(crate::tune::ELIDE_EXT);
        match u {
            shaders::Utility::Pattern { .. } => Ok(false),
            shaders::Utility::Copy { src, dst } => {
                let (ResolvedRes::Buffer(sv), ResolvedRes::Buffer(dv)) = (&cs.resources[src], &cs.resources[dst]) else { return Ok(false) };
                if sv.stride != 4 || dv.stride != 4 || sv.dfmt != 4 || dv.dfmt != 4 {
                    return Ok(false);
                }
                let bytes = (groups * 64).min(dv.num_records as u64).min(sv.num_records as u64) * 4;
                let _ = ext;
                self.elide_copy(sv, dv, bytes)
            }
            shaders::Utility::Fill { value, dst, per_group } => {
                let (ResolvedRes::Buffer(cv), ResolvedRes::Buffer(dv)) = (&cs.resources[value], &cs.resources[dst]) else { return Ok(false) };
                let elems = (groups * per_group).min(dv.num_records as u64);
                let bytes = elems * dv.stride as u64;
                // The fill value (constants written by the CPU or earlier GPU work).
                self.guard_reads(&[cv.base, cv.base + 12]);
                let mut raw = [0u8; 16];
                resources::copy_from_guest(&mut raw, cv.base);
                let w: [u32; 4] = std::array::from_fn(|i| u32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().unwrap()));
                // What the kernel stores per element (format conversion of the
                // value); zero values store zeros in any format.
                let elem = if w == [0; 4] { Some(vec![0u8; dv.stride as usize]) } else { fill_element(dv.dfmt, dv.nfmt, dv.dst_sel, &w) };
                let Some(elem) = elem.filter(|e| e.len() as u32 == dv.stride) else { return Ok(false) };
                let float = w == [0; 4] || (dv.nfmt == 7 && matches!(dv.dfmt, 2 | 4 | 5 | 11 | 12 | 14));
                self.elide_fill(dv, bytes, elem, float, "fill")
            }
        }
    }

    /// The GPU destination of an elided fill/copy of guest [base, base+bytes)
    /// done "in place": the imported guest memory, or without the import the
    /// VRAM mirror (registered like a shader's mirror-resident write).
    fn inplace_dest(&mut self, base: u64, bytes: u64) -> Result<Option<(vk::Buffer, u64)>, String> {
        if ctx().ext_host.is_some() {
            let Some((zbuf, zoff, range)) = self.zc_binding(base, bytes) else { return Ok(None) };
            return Ok((range >= bytes).then_some((zbuf, zoff)));
        }
        if bytes == 0 || bytes > VCACHE_MAX || base % 4 != 0 {
            return Ok(None);
        }
        // (Pages the fill/copy covers entirely need no copy-in first.)
        self.vc_why = "fill/copy target";
        let Some((vb, voff)) = self.vcache_bind_ex(base, bytes, true, true) else { return Ok(None) };
        // (Every byte of the range is written.)
        self.mirror_write_back_full(base, bytes, vb, voff)?;
        self.writes.note_gpu_write(base, bytes);
        self.note_mirror_write(vb, base, bytes);
        if bytes <= SURFACE_WRITE_MAX {
            self.note_meta_write(base, bytes);
            self.note_buffer_write(base, bytes);
        }
        self.epoch += 1;
        Ok(Some((vb, voff)))
    }

    /// Registers the write-back of mirror-resident GPU writes to guest
    /// [base, base+size) at the end of this batch: once per range and batch
    /// (the copy-out then holds the last write). Returns the range's page
    /// bitmap (device-ring offset, dwords, the range's offset in its first
    /// page): writers mark the pages they write, and only those are copied
    /// out and written back.
    fn mirror_write_back(&mut self, base: u64, size: u64, buf: vk::Buffer, off: u64) -> Result<(u64, u32, u32), String> {
        let head = (base & 0xFFF) as u32;
        // (Only when no later write-back overlaps it: write-backs land in order.)
        if let Some(w) = self.writebacks.iter().rev().find(|w| w.base < base + size && base < w.base + w.size).filter(|w| w.mirror && w.base == base && w.size == size && w.dev_buf == buf) {
            if let Some((bits, _, words)) = w.bits {
                return Ok((bits, words, head));
            }
        }
        let pages = (head as u64 + size).div_ceil(4096);
        let words = pages.div_ceil(32) as u32;
        let host_off = self.ring.alloc(size).ok_or_else(|| format!("ring full: {size:#x}-byte write-back of {base:#x} (segment {:#x})", self.ring.segment_size()))?;
        let bits_host = self.ring.alloc(words as u64 * 4).ok_or("ring full: page bitmap")?;
        self.begin();
        let (arena_batch, arena_off, used, cap) = self.bits_arena;
        let bits = if arena_batch == self.batch && used + words as u64 <= cap && crate::tune::get(crate::tune::BITS_ARENA) != 0 {
            // (Pre-zeroed words of this batch: see `begin`.)
            self.bits_arena.2 += words as u64;
            arena_off + 4 * used
        } else {
            let bits = self.dring.alloc(words as u64 * 4).ok_or("device ring full: page bitmap")?;
            if crate::tune::get(crate::tune::PROLOGUE) != 0 {
                let pro = self.prologue_cmd();
                unsafe { ctx().device.cmd_fill_buffer(pro, self.dring.buffer, bits, words as u64 * 4, 0) };
            } else {
                self.end_rendering();
                unsafe { ctx().device.cmd_fill_buffer(self.cmd, self.dring.buffer, bits, words as u64 * 4, 0) };
                global_barrier(self.cmd);
            }
            bits
        };
        self.push_writeback(PendingWrite { base, size, host_off, dev_off: off, dev_buf: buf, mirror: true, bits: Some((bits, bits_host, words)) });
        Ok((bits, words, head))
    }

    /// `mirror_write_back` for GPU work that writes every byte of [base,
    /// base+size) in the mirror window `buf` (render-target tiling, copies,
    /// expanded fast-clear state): written back whole by a bulk copy at the
    /// end of the batch, without a page bitmap (the bitmap kernel copied
    /// render targets page by page: most of the 0.8 ms of write-back work per
    /// batch in Trico's cave).
    fn mirror_write_back_full(&mut self, base: u64, size: u64, buf: vk::Buffer, off: u64) -> Result<(), String> {
        // (Already registered and not overtaken by a later overlapping
        // write-back: lands with that one.)
        if self.writebacks.iter().rev().find(|w| w.base < base + size && base < w.base + w.size).is_some_and(|w| w.mirror && w.bits.is_none() && w.base == base && w.size == size && w.dev_buf == buf && w.dev_off == off) {
            return Ok(());
        }
        let host_off = self.ring.alloc(size).ok_or_else(|| format!("ring full: {size:#x}-byte write-back of {base:#x} (segment {:#x})", self.ring.segment_size()))?;
        self.push_writeback(PendingWrite { base, size, host_off, dev_off: off, dev_buf: buf, mirror: true, bits: None });
        Ok(())
    }

    /// Device-ring dwords where shaders mark the pages of written buffers
    /// whose writes are not tracked (a sink, allocated once per batch).
    fn dirty_sink(&mut self) -> Result<u64, String> {
        if self.dirty_sink.0 != self.batch {
            let off = self.dring.alloc(64).ok_or("device ring full: page bitmap sink")?;
            self.dirty_sink = (self.batch, off);
        }
        Ok(self.dirty_sink.1)
    }

    /// The source of an elided copy read "in place" (see `inplace_dest`).
    fn inplace_src(&mut self, base: u64, bytes: u64) -> Option<(vk::Buffer, u64)> {
        if ctx().ext_host.is_some() {
            let (b, o, r) = self.zc_binding(base, bytes)?;
            return (r >= bytes).then_some((b, o));
        }
        if bytes == 0 || bytes > VCACHE_MAX {
            return None;
        }
        self.vc_why = "copy source";
        self.vcache_bind_ex(base, bytes, false, true)
    }

    fn elide_copy(&mut self, sv: &desc::Vsharp, dv: &desc::Vsharp, bytes: u64) -> Result<bool, String> {
        let ext = crate::tune::get(crate::tune::ELIDE_EXT);
        let skey = self.by_base.get(&sv.base).and_then(|ks| {
            ks.iter().copied().find(|k| {
                let img = &self.images[k];
                img.gpu_written && !img.guest_newer && !k.ty3d && Self::image_guest_layout(k, img).is_some_and(|g| g.3 == bytes)
            })
        });
        // Contents written to the source memory by other means since
        // the image was rendered would be lost.
        if let Some(skey) = skey.filter(|_| !self.pending_write_overlaps(sv.base, bytes) && !writeback_in_flight(sv.base, bytes)) {
            let dkey = ImageKey { base: dv.base, ..skey };
            self.init_image(dkey)?;
            let tile = self.images[&skey].tile_index;
            self.images.get_mut(&dkey).unwrap().tile_index = tile;
            self.begin();
            self.end_rendering();
            if self.barrier_needed {
                global_barrier(self.cmd);
                self.barrier_needed = false;
            }
            let aspect = if skey.depth { vk::ImageAspectFlags::DEPTH } else { vk::ImageAspectFlags::COLOR };
            let sub = vk::ImageSubresourceLayers { aspect_mask: aspect, mip_level: 0, base_array_layer: 0, layer_count: skey.layers.max(1) };
            let region = vk::ImageCopy { src_subresource: sub, src_offset: vk::Offset3D::default(), dst_subresource: sub, dst_offset: vk::Offset3D::default(), extent: vk::Extent3D { width: skey.width, height: skey.height, depth: 1 } };
            unsafe { ctx().device.cmd_copy_image(self.cmd, self.images[&skey].image, vk::ImageLayout::GENERAL, self.images[&dkey].image, vk::ImageLayout::GENERAL, &[region]) };
            unordered!(self);
            self.note_image_written(dkey);
            self.mark(|| "surface copy (image)".into());
            self.last_elide = "surface copy (image)";
            return Ok(true);
        }
        // In guest memory: the source must hold what the kernel would
        // read (render targets written back to it first, as for the
        // kernel's buffer read).
        if ext & 4 == 0 || bytes == 0 || bytes > SURFACE_WRITE_MAX || sv.base % 4 != 0 || dv.base % 4 != 0 || (sv.base < dv.base + bytes && dv.base < sv.base + bytes) {
            return Ok(false);
        }
        // (Write-backs from the VRAM mirror leave it current.)
        let mirrored = ctx().ext_host.is_none();
        if self.writebacks.iter().any(|w| !(mirrored && w.mirror) && ((w.base < sv.base + bytes && sv.base < w.base + w.size) || (w.base < dv.base + bytes && dv.base < w.base + w.size))) {
            return Ok(false);
        }
        self.writeback_images(sv.base, bytes);
        if self.writebacks.iter().any(|w| !(mirrored && w.mirror) && w.base < sv.base + bytes && sv.base < w.base + w.size) {
            return Ok(false);
        }
        if cpu_writeback_in_flight(sv.base, bytes) {
            self.flushes[1] += 1;
            wait_cpu_writebacks(sv.base, bytes);
        }
        let Some((sb, so)) = self.inplace_src(sv.base, bytes) else { return Ok(false) };
        let Some((db, doff)) = self.inplace_dest(dv.base, bytes)? else { return Ok(false) };
        self.begin();
        self.end_rendering();
        global_barrier(self.cmd);
        unsafe { ctx().device.cmd_copy_buffer(self.cmd, sb, db, &[vk::BufferCopy { src_offset: so, dst_offset: doff, size: bytes }]) };
        unordered!(self);
        if !mirrored {
            self.zc_read(sv.base, sv.base + bytes);
            self.note_inplace_gpu_write(dv.base, bytes);
        }
        self.mark(|| "buffer copy (in place)".into());
        self.last_elide = if skey.is_some() { "buffer copy (in place; source image skipped: pending writes or write-back)" } else { "buffer copy (in place)" };
        Ok(true)
    }

    /// An engine fill of guest [dv.base, +bytes) with `elem` per element
    /// (`float`: a value a render target clear can store) done without its
    /// kernel: a clear of the render target there, or in place.
    fn elide_fill(&mut self, dv: &desc::Vsharp, bytes: u64, elem: Vec<u8>, float: bool, what: &str) -> Result<bool, String> {
        let ext = crate::tune::get(crate::tune::ELIDE_EXT);
        // Zero/float fills of a render target (or one array layer): clear.
        let target = match self.surface_image(dv.base, bytes) {
            Some(k) => Some((k, (0, k.layers.max(1)))),
            None => self.surface_layer(dv.base, bytes).map(|(k, l)| (k, (l, 1))),
        };
        // (Integer formats clear to any pattern exactly.)
        let clear = target.filter(|(k, _)| float || int_clear_for(k.format, &elem).is_some()).filter(|(k, _)| format_class(k.format) == dv.stride).and_then(|(k, layers)| {
            let c = if elem.iter().all(|&b| b == 0) { Some(vk::ClearColorValue::default()) } else { clear_color_for(k.format, &elem) };
            c.map(|c| (k, layers, c))
        });
        if let Some((key, layers, color)) = clear {
            if layers.1 < key.layers.max(1) {
                // The other layers must be current in the image.
                if self.images[&key].guest_newer {
                    self.refresh_from_guest(key, true)?;
                }
                self.sync_alias(key)?;
            }
            self.begin();
            self.end_rendering();
            if self.barrier_needed {
                global_barrier(self.cmd);
                self.barrier_needed = false;
            }
            let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: key.levels.max(1), base_array_layer: layers.0, layer_count: layers.1 };
            unsafe { ctx().device.cmd_clear_color_image(self.cmd, self.images[&key].image, vk::ImageLayout::GENERAL, &color, &[range]) };
            unordered!(self);
            self.note_image_written(key);
            self.mark(|| "surface fill (clear)".into());
            self.last_elide = "surface fill (clear)";
            return Ok(true);
        }
        // In guest memory, when the element repeats a dword (fills of a
        // render target not cleared above: `tune elide_ext` bit 1; of other
        // memory: bit 2). Narrower elements repeat within the dword.
        let allowed = if target.is_some() { ext & 1 != 0 } else { ext & 2 != 0 };
        let word: Vec<u8> = if !elem.is_empty() && elem.len() < 4 && 4 % elem.len() == 0 { elem.iter().copied().cycle().take(4).collect() } else { elem.clone() };
        if !allowed || word.len() % 4 != 0 || word.chunks(4).any(|c| c != &word[..4]) || bytes == 0 || bytes % 4 != 0 || dv.base % 4 != 0 {
            return Ok(false);
        }
        let declined = |why: &str| {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
                crate::info!("{what} of {:#x}+{bytes:#x} with {:02x?} not elided: {why}", dv.base, &elem);
            }
        };
        // (Write-backs from the VRAM mirror leave it current: the fill
        // lands after them in the mirror and is written back last.)
        let mirrored = ctx().ext_host.is_none();
        if self.writebacks.iter().any(|wb| !(mirrored && wb.mirror) && wb.base < dv.base + bytes && dv.base < wb.base + wb.size) {
            declined("pending write-back");
            return Ok(false);
        }
        let Some((zbuf, zoff)) = self.inplace_dest(dv.base, bytes)? else {
            declined("no in-place binding");
            return Ok(false);
        };
        let pattern = u32::from_le_bytes(word[..4].try_into().unwrap());
        self.begin();
        self.end_rendering();
        global_barrier(self.cmd);
        unsafe { ctx().device.cmd_fill_buffer(self.cmd, zbuf, zoff, bytes, pattern) };
        unordered!(self);
        if !mirrored {
            self.note_inplace_gpu_write(dv.base, bytes);
        }
        self.note_meta_fill(dv.base, bytes, pattern);
        self.mark(|| "buffer fill (in place)".into());
        self.last_elide = "buffer fill (in place)";
        Ok(true)
    }

    /// DMA_DATA memory -> memory whose source has pending GPU writes (a
    /// device-ring write-back of this batch, or in-place writes): the copy is
    /// recorded in queue order into guest memory instead of the CP draining
    /// the GPU to copy on the CPU (the engine copies GDS counters it stored to
    /// memory this way, ~150 times a second in Old Yharnam). False when the CP
    /// must copy: nothing pending, a write-back only partly covering the
    /// source, or CPU-performed writes (write-backs, end-of-pipe writes) that
    /// would land on either range after the copy.
    fn dma_on_gpu(&mut self, src: u64, dst: u64, bytes: u64) -> bool {
        let declined = |why: usize| {
            DMA_DECLINED[why].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            false
        };
        if crate::tune::get(crate::tune::DMA_GPU) == 0 || bytes == 0 || bytes % 4 != 0 || src % 4 != 0 || dst % 4 != 0 || bytes > SURFACE_WRITE_MAX || (src < dst + bytes && dst < src + bytes) {
            return declined(0);
        }
        if ctx().ext_host.is_none() {
            return self.dma_via_mirror(src, dst, bytes);
        }
        let Some((sbuf, soff)) = self.gpu_source(src, bytes) else {
            // (Nothing pending: the CP copies now without waiting.)
            let pending = self.writebacks.iter().any(|p| p.base < src + bytes && src < p.base + p.size) || cpu_writeback_in_flight(src, bytes) || self.zc_overlaps(src, bytes) || writeback_in_flight(src, bytes);
            return declined(if pending { 1 } else { 5 });
        };
        // (In-place writes of the destination by submitted batches precede
        // the copy on the queue; declining for them made the CP drain the
        // GPU for ~1,000 four-byte counter copies per report in the Dream.)
        if self.writebacks.iter().any(|w| w.base < dst + bytes && dst < w.base + w.size) || cpu_writeback_in_flight(dst, bytes) {
            return declined(2);
        }
        let Some((zbuf, zoff, range)) = self.zc_binding(dst, bytes) else { return declined(4) };
        if range < bytes {
            return declined(4);
        }
        self.begin();
        self.end_rendering();
        global_barrier(self.cmd);
        unsafe { ctx().device.cmd_copy_buffer(self.cmd, sbuf, zbuf, &[vk::BufferCopy { src_offset: soff, dst_offset: zoff, size: bytes }]) };
        if sbuf != self.dring.buffer {
            self.zc_read(src, src + bytes);
        }
        unordered!(self);
        self.note_inplace_gpu_write(dst, bytes);
        self.mark(|| "DMA_DATA copy (GPU)".into());
        true
    }

    /// `dma_on_gpu` without the import (the default): the source is read
    /// through its VRAM mirror, which binding brings up to date on the GPU
    /// (this batch's and in-flight writes forwarded, see `vcache_fill`), and
    /// the copy lands in the destination's mirror, written back to guest
    /// memory at the end of the batch like a shader's mirror-resident write
    /// -- in completion order, after earlier batches' CPU-performed writes.
    /// False (the CP copies) when nothing is pending on the source or a
    /// range has no mirror window.
    fn dma_via_mirror(&mut self, src: u64, dst: u64, bytes: u64) -> bool {
        let declined = |why: usize| {
            DMA_DECLINED[why].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            false
        };
        let pending = self.writebacks.iter().any(|p| p.base < src + bytes && src < p.base + p.size) || cpu_writeback_in_flight(src, bytes) || writeback_in_flight(src, bytes);
        if !pending {
            return declined(5);
        }
        self.vc_why = "DMA source";
        let Some((sbuf, soff)) = self.vcache_bind_ex(src, bytes, false, true) else { return declined(1) };
        self.vc_why = "DMA destination";
        let Some((dbuf, doff)) = self.vcache_bind_ex(dst, bytes, true, true) else { return declined(4) };
        // (Write-back first: when the ring cannot take it, the CP copies.)
        if self.mirror_write_back_full(dst, bytes, dbuf, doff).is_err() {
            return declined(4);
        }
        self.begin();
        self.end_rendering();
        global_barrier(self.cmd);
        unsafe { ctx().device.cmd_copy_buffer(self.cmd, sbuf, dbuf, &[vk::BufferCopy { src_offset: soff, dst_offset: doff, size: bytes }]) };
        unordered!(self);
        self.writes.note_gpu_write(dst, bytes);
        self.note_mirror_write(dbuf, dst, bytes);
        if bytes <= SURFACE_WRITE_MAX {
            self.note_meta_write(dst, bytes);
            self.note_buffer_write(dst, bytes);
        }
        self.epoch += 1;
        self.dma_mirror += 1;
        self.mark(|| "DMA_DATA copy (GPU, mirror)".into());
        true
    }

    /// Where a GPU copy recorded now reads the current contents of guest
    /// [src, src+bytes) that GPU writes not yet in guest memory produce: the
    /// newest device-ring write-back of this batch when it holds the whole
    /// range, else guest memory in place when the pending writes are in-place
    /// ones (this batch's or in flight: they land in queue order before the
    /// copy). None when nothing is pending (the CPU can read the range now),
    /// or when CPU-performed writes (write-backs, end-of-pipe writes) in
    /// flight would land after the copy.
    fn gpu_source(&mut self, src: u64, bytes: u64) -> Option<(vk::Buffer, u64)> {
        match self.writebacks.iter().rev().find(|p| p.base < src + bytes && src < p.base + p.size) {
            Some(p) if p.base <= src && src + bytes <= p.base + p.size => Some((p.dev_buf, p.dev_off + (src - p.base))),
            Some(_) => None,
            None => {
                if cpu_writeback_in_flight(src, bytes) || !(self.zc_overlaps(src, bytes) || writeback_in_flight(src, bytes)) {
                    return None;
                }
                self.zc_binding(src, bytes).filter(|z| z.2 >= bytes).map(|z| (z.0, z.1))
            }
        }
    }

    /// DMA_DATA memory -> GDS whose source has pending GPU writes (see
    /// `gpu_source`): copied on the GPU in queue order instead of the CP
    /// draining the GPU (the engine reloads GDS counters from memory it
    /// copied them to on the GPU).
    fn gds_load_on_gpu(&mut self, offset: u32, src: u64, bytes: u32) -> bool {
        let (o, n) = ((offset & !3) as u64, (bytes & !3) as u64);
        if crate::tune::get(crate::tune::DMA_GPU) == 0 || n == 0 || src % 4 != 0 || o + n > gcn::translate::GDS_BYTES as u64 {
            return false;
        }
        let Some((sbuf, soff)) = self.gpu_source(src, n) else { return false };
        self.begin();
        self.end_rendering();
        global_barrier(self.cmd);
        unsafe { ctx().device.cmd_copy_buffer(self.cmd, sbuf, self.gds, &[vk::BufferCopy { src_offset: soff, dst_offset: o, size: n }]) };
        if sbuf != self.dring.buffer {
            self.zc_read(src, src + n);
        }
        unordered!(self);
        self.mark(|| "GDS load (GPU source)".into());
        true
    }

    /// Registers a GPU write to guest memory [base, base+len) recorded in this
    /// batch outside a shader binding (copies, fills): host visibility at
    /// batch end, write tracking, readers' synchronization, images and
    /// metadata it overwrites.
    fn note_inplace_gpu_write(&mut self, base: u64, len: u64) {
        self.push_zc_write(base, len);
        self.writes.note_gpu_write(base, len);
        self.note_pending_gpu_write(base, len);
        if len <= SURFACE_WRITE_MAX {
            self.note_meta_write(base, len);
            self.note_buffer_write(base, len);
        }
        self.epoch += 1;
    }

    /// Records a shader write to a storage image (for `sync_alias`).
    fn note_storage_write(&mut self, key: ImageKey) {
        let tick = self.draws + self.dispatches + 1;
        if let Some(img) = self.images.get_mut(&key) {
            img.gpu_written = true;
            img.last_write = tick;
        }
        *self.write_gen.entry(key).or_default() += 1;
    }

    /// Stand-in for a T# that cannot be honored (invalid descriptor, address
    /// outside guest memory, or a format we do not implement): a zeroed 1x1
    /// image of the expected view type and numeric class, so the rest of the
    /// draw still runs. GCN returns zeros when sampling an invalid
    /// descriptor; titles also bind such descriptors for unused slots.
    fn dummy_texture(&mut self, t: &Tsharp, storage: bool, why: &str) -> Result<vk::ImageView, String> {
        self.report(format!("texture stand-in (zero image): {why}: dfmt {} nfmt {} type {}", t.dfmt, t.nfmt, t.ty));
        let format = match desc::num_class(t.nfmt) {
            gcn::translate::NumClass::Uint => vk::Format::R8G8B8A8_UINT,
            gcn::translate::NumClass::Sint => vk::Format::R8G8B8A8_SINT,
            _ => vk::Format::R8G8B8A8_UNORM,
        };
        let (view_type, layers, ty3d) = match t.ty {
            desc::T_3D => (vk::ImageViewType::TYPE_3D, 1, true),
            desc::T_CUBE if t.last_array > t.base_array => (vk::ImageViewType::CUBE_ARRAY, 6, false),
            desc::T_CUBE => (vk::ImageViewType::CUBE, 6, false),
            desc::T_1D_ARRAY | desc::T_2D_ARRAY | desc::T_2D_MSAA_ARRAY => (vk::ImageViewType::TYPE_2D_ARRAY, 1, false),
            _ => (vk::ImageViewType::TYPE_2D, 1, false),
        };
        const DUMMY_BASE: u64 = 0xD000_0000_0000_0000;
        let key = ImageKey {
            base: DUMMY_BASE + ((format.as_raw() as u64) << 8) + view_type.as_raw() as u64,
            width: 1,
            height: 1,
            layers,
            levels: 1,
            format,
            depth: false,
            ty3d,
        };
        self.init_image(key)?;
        let img = self.images.get_mut(&key).unwrap();
        img.gpu_written = true; // never sourced from guest memory
        const IDENTITY_SEL: u16 = 4 | (5 << 3) | (6 << 6) | (7 << 9);
        img.view(format, IDENTITY_SEL, view_type, storage, 0, layers)
    }

    /// Image view for sampling/storing a T#.
    /// `load_only`: the shader only loads texels by integer coordinates
    /// (`Prepared::load_only`).
    fn texture(&mut self, t: &Tsharp, storage: bool, load_only: bool) -> Result<vk::ImageView, String> {
        let Some(vfmt) = desc::vk_format(t.dfmt, t.nfmt) else {
            return self.dummy_texture(t, storage, "unsupported format");
        };
        if !crate::mem::is_guest(t.base, 1) {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 12 {
                crate::info!("texture outside guest memory: base {:#x} {}x{} dfmt {} nfmt {} type {}", t.base, t.width, t.height, t.dfmt, t.nfmt, t.ty);
            }
            return self.dummy_texture(t, storage, "address outside guest memory");
        }
        let layers = t.layers().max(1);
        let total_layers = t.total_layers().max(1);
        let view_base = t.view_base_layer();
        let ty3d = t.ty == desc::T_3D;
        let levels = resources::texture_levels(t);
        // Viewed levels: BASE_LEVEL..=LAST_LEVEL.
        let (view_level, view_levels) = (t.base_level, t.last_level.saturating_sub(t.base_level) + 1);
        let view_type = match t.ty {
            desc::T_1D => vk::ImageViewType::TYPE_1D,
            desc::T_3D => vk::ImageViewType::TYPE_3D,
            desc::T_CUBE => {
                if layers > 6 { vk::ImageViewType::CUBE_ARRAY } else { vk::ImageViewType::CUBE }
            }
            desc::T_1D_ARRAY => vk::ImageViewType::TYPE_1D_ARRAY,
            desc::T_2D_ARRAY | desc::T_2D_MSAA_ARRAY => vk::ImageViewType::TYPE_2D_ARRAY,
            _ => vk::ImageViewType::TYPE_2D,
        };
        // 1D textures are stored as 2D images with height 1; use 2D-array views for 1D arrays.
        let view_type = match view_type {
            vk::ImageViewType::TYPE_1D => vk::ImageViewType::TYPE_2D,
            vk::ImageViewType::TYPE_1D_ARRAY => vk::ImageViewType::TYPE_2D_ARRAY,
            v => v,
        };
        let store_fmt = canonical(vfmt);
        let depth_layers = if ty3d { t.depth } else { total_layers };
        // Render targets here whose memory the title reused (see
        // `reused_target`).
        if !storage {
            let old: smallvec::SmallVec<[ImageKey; 4]> = self.by_base.get(&t.base).map(|ks| ks.iter().copied().filter(|k| self.images[k].gpu_written).collect()).unwrap_or_default();
            for k in old {
                self.reused_target(k);
            }
        }
        // Render targets here whose guest memory was since written by buffer
        // stores: reload them before sampling (directly or via shadows).
        let newer: Vec<ImageKey> = self.by_base.get(&t.base).map(|ks| ks.iter().copied().filter(|k| self.images[k].guest_newer).collect()).unwrap_or_default();
        for k in newer {
            self.refresh_from_guest(k, false)?;
        }
        let key = ImageKey { base: t.base, width: t.width, height: t.height, layers: depth_layers, levels, format: store_fmt, depth: false, ty3d };
        // GPU-written image with the same size at this address?
        if let Some(k) = self.find_compatible(key) {
            if self.images[&k].gpu_written || storage {
                let k = self.grow_levels(k, view_level, levels)?;
                self.sync_alias(k)?;
                if storage {
                    self.note_storage_write(k);
                }
                let img = self.images.get_mut(&k).unwrap();
                let v = img.view_levels(if storage { canonical(vfmt) } else { vfmt }, t.dst_sel, view_type, storage, view_base, layers, view_level, view_levels)?;
                return Ok(v);
            }
        }
        // A depth render target at this address (sampled after a decompress):
        // copy its depth values into a color image of the sampled format.
        if matches!(vfmt, vk::Format::R32_SFLOAT | vk::Format::R16_UNORM) && !storage {
            if let Some(src) = self.by_base.get(&t.base).and_then(|ks| ks.iter().copied().find(|k| k.depth && self.images[k].gpu_written && (k.format == vk::Format::D16_UNORM) == (vfmt == vk::Format::R16_UNORM))) {
                return self.depth_shadow_view(src, t, vfmt, view_type, load_only);
            }
        }
        // A render target at this address with padded dimensions: copy into a
        // shadow image (from the most recently rendered one if several exist).
        if let Some(src) = self.by_base.get(&t.base).and_then(|ks| {
            ks.iter()
                .copied()
                .filter(|k| self.images[k].gpu_written && k.width >= t.width && k.height >= t.height && same_surface(k.width, k.height, t.width, t.height) && format_class(k.format) == format_class(store_fmt) && !k.depth)
                .max_by_key(|k| self.images[k].last_write)
        }) {
            if trace_frame() {
                crate::info!("FRAME texture {:#x} {}x{} {:?}: shadow of {}x{} {:?} (last write {})", t.base, t.width, t.height, vfmt, src.width, src.height, src.format, self.images[&src].last_write);
            }
            return self.shadow_view(src, t, vfmt, store_fmt, view_type);
        }
        if trace_frame() {
            let others: Vec<String> = self.by_base.get(&t.base).map(|ks| ks.iter().map(|k| format!("{}x{} {:?} gw {}", k.width, k.height, k.format, self.images[k].gpu_written)).collect()).unwrap_or_default();
            crate::info!("FRAME texture {:#x} {}x{} {:?}: guest memory (images here: [{}])", t.base, t.width, t.height, vfmt, others.join(", "));
        }
        self.init_image(key)?;
        let batch = self.epoch;
        let mut needs_upload = {
            let img = &self.images[&key];
            !img.gpu_written && img.upload_batch != batch
        };
        // Within one CPU epoch only GPU writes (tracked as ranges) can have
        // changed the texture: no write-watch query needed.
        if needs_upload {
            let bytes = resources::texture_bytes(t, depth_layers.max(1));
            let img = &self.images[&key];
            if img.cpu_checked != u64::MAX && img.cpu_checked >= self.required_epoch && img.validated.is_some_and(|v| !self.writes.gpu_written_since(t.base, bytes, v)) {
                self.images.get_mut(&key).unwrap().upload_batch = batch;
                needs_upload = false;
            }
        }
        let gpu_done = needs_upload && self.gpu_texture_validate(key, t, batch)?;
        if needs_upload && !gpu_done {
            let bytes = resources::texture_bytes(t, depth_layers.max(1));
            self.sync_guest(t.base, bytes);
            let t0 = std::time::Instant::now();
            // Hash only when the OS saw pages of the texture written (or the
            // GPU wrote there) since the last validation.
            let since = self.images[&key].validated;
            let (written, stamp) = self.writes.changed_since(t.base, bytes, since.unwrap_or(0));
            let mut changed = false;
            let mut h = 0;
            // TLG_WRITE_WATCH_VERIFY=1: hash anyway and report changes the
            // write tracking did not see (they would leave stale textures).
            static VERIFY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let verify = *VERIFY.get_or_init(|| std::env::var_os("TLG_WRITE_WATCH_VERIFY").is_some());
            // Writes only in the partially covered first/last page (data next
            // to the texture) that left the texture's bytes there unchanged:
            // the texture is unchanged without hashing all of it.
            let edge_only = written && !verify && since.is_some_and(|s| self.edges_unchanged(key, t.base, bytes, s));
            if edge_only {
                self.edge_skips += 1;
            } else if written || since.is_none() || verify {
                h = resources::hash_guest(t.base, bytes);
                self.hashed_bytes += bytes;
                changed = since.is_none() || self.images[&key].uploaded_hash != h;
                if changed && !written && since.is_some() {
                    self.report(format!("write tracking missed a change of texture {:#x} ({}x{}, {} bytes)", t.base, t.width, t.height, bytes));
                }
                let edges = page_edges(t.base, bytes);
                self.images.get_mut(&key).unwrap().edges = edges;
            }
            self.prof[3] += t0.elapsed();
            self.images.get_mut(&key).unwrap().upload_batch = batch;
            if changed {
                self.upload_texture(key, t, depth_layers.max(1))?;
                self.images.get_mut(&key).unwrap().uploaded_hash = h;
            }
            let img = self.images.get_mut(&key).unwrap();
            img.validated = Some(stamp);
            img.cpu_checked = self.cpu_epoch;
        }
        if storage {
            self.sync_alias(key)?;
            self.note_storage_write(key);
        }
        let img = self.images.get_mut(&key).unwrap();
        img.view_levels(if storage { canonical(vfmt) } else { vfmt }, t.dst_sel, view_type, storage, view_base, layers, view_level, view_levels)
    }

    /// A GPU-written image about to be sampled at a texture's address that
    /// has not been rendered for a few frames: did the CPU write its memory
    /// since it was last rendered (once per frame)? The title then freed the
    /// surface and reused the memory, e.g. for a texture of the same size
    /// loaded from disk: the image stops standing for that memory (no longer
    /// GPU-written), so the texture path validates and uploads guest memory
    /// with the texture's own layout. Only whole pages inside the surface
    /// count (data sharing its first or last page may change).
    fn reused_target(&mut self, k: ImageKey) -> bool {
        let frame = super::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
        let Some(img) = self.images.get(&k) else { return false };
        if !img.gpu_written || img.stale_stamp == 0 || img.last_write_frame + 2 >= frame || img.reuse_checked == frame || k.ty3d {
            return false;
        }
        let Some((_, _, _, total)) = Self::image_guest_layout(&k, img) else { return false };
        let since = img.stale_stamp;
        self.images.get_mut(&k).unwrap().reuse_checked = frame;
        let (lo, hi) = ((k.base + 0xFFF) & !0xFFF, (k.base + total) & !0xFFF);
        if lo >= hi || !self.writes.cpu_written_since(lo, hi - lo, since).0 {
            return false;
        }
        let img = self.images.get_mut(&k).unwrap();
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 32 {
            crate::info!(
                "render target {:#x} {}x{} {:?} (last rendered {} frames ago) sampled after the CPU wrote its memory: memory reused, sampled from guest memory",
                k.base,
                k.width,
                k.height,
                k.format,
                frame - img.last_write_frame
            );
        }
        img.gpu_written = false;
        img.guest_stale = false;
        img.guest_newer = false;
        img.validated = None;
        img.upload_batch = u64::MAX;
        img.cpu_checked = u64::MAX;
        self.reused_targets += 1;
        true
    }

    /// Whether the only pages of [base, base+bytes) written after stamp `since`
    /// are the range's partially covered first/last page and the range's
    /// bytes there equal those saved at the last hash (`page_edges`).
    fn edges_unchanged(&self, key: ImageKey, base: u64, bytes: u64, since: u64) -> bool {
        let Some(saved) = self.images.get(&key).and_then(|i| i.edges.as_deref()) else { return false };
        let end = base + bytes;
        let (head_page, tail_page) = (base & !0xFFF, (end - 1) & !0xFFF);
        let head_partial = base & 0xFFF != 0;
        let tail_partial = end & 0xFFF != 0;
        for (a, b) in self.writes.dirty_runs(base, end, since) {
            // (A run may span many pages: all of it must lie in one edge page.)
            let (first, last) = (a & !0xFFF, (b - 1) & !0xFFF);
            let ok = first == last && ((head_partial && first == head_page) || (tail_partial && first == tail_page));
            if !ok {
                return false;
            }
        }
        let cur = page_edges(base, bytes).unwrap_or_default();
        *cur == *saved
    }

    /// Texture validation and upload on the GPU for surfaces the remap kernel
    /// handles (32/64/128-bit elements, pitch = width, not 3D): when the write
    /// tracking reports the memory written, the surface is detiled from guest
    /// memory in place and in GPU order -- no hash, no CPU read, no wait for
    /// GPU work that wrote it. Returns false to use the CPU path.
    fn gpu_texture_validate(&mut self, key: ImageKey, t: &Tsharp, batch: u64) -> Result<bool, String> {
        let (bpp, block) = resources::element_bits(t.dfmt);
        if key.levels > 1 || block != 1 {
            return self.gpu_texture_levels(key, t, batch);
        }
        if block != 1 || !Self::gpu_remap_eligible(bpp) || key.ty3d || format_class(key.format) * 8 != bpp || crate::tune::get(crate::tune::GPU_TEXTURES) == 0 {
            if bpp >= 32 && block == 1 && t.width * t.height >= (1 << 20) {
                self.report(format!("texture {:#x} {}x{} pitch {} bpp {bpp} {:?}: GPU detiling not applicable", t.base, t.width, t.height, t.pitch, key.format));
            }
            return Ok(false);
        }
        let layers = key.layers.max(1);
        let bytes = resources::texture_bytes(t, layers);
        let img = self.images.get_mut(&key).unwrap();
        img.tile_index = Some(t.tiling_index);
        img.pitch = t.pitch.max(t.width);
        if Self::image_guest_layout(&key, &self.images[&key]).map(|g| g.3) != Some(bytes) {
            self.report(format!("texture {:#x} {}x{}: GPU detiling declined (surface size)", t.base, t.width, t.height));
            return Ok(false);
        }
        let since = self.images[&key].validated;
        let (written, stamp) = self.writes.changed_since(t.base, bytes, since.unwrap_or(0));
        if !written && since.is_some() {
            let cpu_epoch = self.cpu_epoch;
            let img = self.images.get_mut(&key).unwrap();
            img.upload_batch = batch;
            img.validated = Some(stamp);
            img.cpu_checked = cpu_epoch;
            return Ok(true);
        }
        // Device-ring write-backs land in guest memory on the CPU after their
        // batch; in-place and round-trip writes are ordered on the GPU.
        // (Mirror-resident writes too in principle, but the mirror copy-in
        // counts the pages they write as changed and reloads them from guest
        // memory before the write-back: detiling the title screen's tiled
        // render targets from the mirror gave a black background.)
        if self.writebacks.iter().any(|w| w.base < t.base + bytes && t.base < w.base + w.size) {
            self.report(format!("texture {:#x} {}x{}: GPU detiling declined (pending write-back)", t.base, t.width, t.height));
            return Ok(false);
        }
        if cpu_writeback_in_flight(t.base, bytes) {
            self.flushes[1] += 1;
            let t0 = std::time::Instant::now();
            wait_cpu_writebacks(t.base, bytes);
            self.prof[0] += t0.elapsed();
        }
        let lin_bytes = key.width as u64 * key.height as u64 * (bpp / 8) as u64 * layers as u64;
        let Some(lin_off) = self.dring_for_detile(lin_bytes)? else {
            self.report(format!("texture {:#x} {}x{}: GPU detiling declined (device ring)", t.base, t.width, t.height));
            return Ok(false);
        };
        if !self.gpu_remap(&key, false, lin_off)? {
            self.report(format!("texture {:#x} {}x{}: GPU detiling declined (remap)", t.base, t.width, t.height));
            return Ok(false);
        }
        if ctx().ext_host.is_some() {
            self.zc_read(t.base, t.base + bytes);
        }
        let region = vk::BufferImageCopy {
            buffer_offset: lin_off,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: layers },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D { width: key.width, height: key.height, depth: 1 },
        };
        unsafe { ctx().device.cmd_copy_buffer_to_image(self.cmd, self.dring.buffer, self.images[&key].image, vk::ImageLayout::GENERAL, &[region]) };
        unordered!(self);
        self.uploads += 1;
        self.gpu_tex_uploads += 1;
        self.mark(|| "texture detiling (GPU)".into());
        let cpu_epoch = self.cpu_epoch;
        let img = self.images.get_mut(&key).unwrap();
        img.upload_batch = batch;
        img.validated = Some(stamp);
        img.cpu_checked = cpu_epoch;
        img.edges = None; // (saved by CPU-path hashes only)
        Ok(true)
    }

    /// VRAM mirror of read-only guest [base, base+size) for a draw/dispatch
    /// binding (`VCache`): (buffer, offset), or None to read in place (small
    /// or huge ranges, budget exhausted, `tune vcache 0`). Records the copy-in
    /// of pages that are not current; the command's own barrier orders it
    /// before the read.
    fn vcache_bind(&mut self, base: u64, size: u64) -> Option<(vk::Buffer, u64)> {
        self.vcache_bind_ex(base, size, false, false)
    }

    /// `vcache_bind`; `overwrite`: the caller overwrites the whole range, so
    /// pages entirely inside it need no copy-in.
    /// (`small`: ranges below VCACHE_MIN too, in windows that exist.)
    fn vcache_bind_ex(&mut self, base: u64, size: u64, overwrite: bool, small: bool) -> Option<(vk::Buffer, u64)> {
        self.drain_cp_writes();
        // (Bindings up to a window: callers limit written ones to VCACHE_MAX.)
        if size == 0 || size > VWIN_SIZE || (size < VCACHE_MIN && !small) || crate::tune::get(crate::tune::VCACHE) == 0 {
            return None;
        }
        // Served before, and nothing of it stopped being current since: no
        // page invalidated in its window (CPU writes found, GPU writes
        // elsewhere, CP writes), its blocks' CPU-write checks still count.
        if !overwrite && crate::tune::get(crate::tune::VBIND_CACHE) != 0 {
            if let Some(&(k, wgen, checked, buf, off)) = self.vbind.get(&(base, size)) {
                let required = self.required_epoch;
                if let Some(vc) = self.vcache.as_mut() {
                    if let Some(w) = vc.windows.get_mut(&k).filter(|w| w.wgen == wgen && checked != u64::MAX && checked >= required) {
                        w.last_used = super::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
                        vc.hits += 1;
                        return Some((buf, off));
                    }
                }
            }
        }
        // Windows are keyed by their start: a 256 MiB slot can hold windows
        // of separate mappings, and a window is replaced when its mapping
        // has grown past it.
        // (The window of the base's own slot first: writers and later readers
        // of a range then meet in the same window.)
        let found = self.vcache.as_ref().and_then(|vc| {
            let fits = |w: &&VWindow| w.lo <= base && base + size <= w.hi;
            vc.windows.values().filter(fits).find(|w| w.lo >> VWIN_SHIFT == base >> VWIN_SHIFT).or_else(|| vc.windows.values().find(fits)).map(|w| w.lo)
        });
        let k = match found {
            Some(k) => k,
            // (Smaller ranges only in windows that exist: elided fills and
            // copies next to mirrored data.)
            None if size < VCACHE_MIN => return None,
            None => {
                let (start, end) = crate::mem::MEM.lock().committed_span(base)?;
                let slot = base >> VWIN_SHIFT << VWIN_SHIFT;
                let (lo, hi) = (slot.max(start), (slot + VWIN_SIZE).min(end));
                // (TLG_VCACHE_REFUSE=0: new windows even under VRAM pressure.
                // Sparse windows cost only what gets bound: always allowed.
                // Without one, data bound from there is uploaded per batch.)
                static REFUSE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                let refuse = *REFUSE.get_or_init(|| std::env::var("TLG_VCACHE_REFUSE").map(|v| v != "0").unwrap_or(true));
                if base + size > hi || (refuse && self.pressure >= 2 && !sparse_mirrors()) {
                    return None;
                }
                // Overlapped windows (made before the mapping grew) are
                // released once the batches using them have completed.
                let seq = self.last_seq + 1;
                let mut gone = Vec::new();
                if let Some(vc) = self.vcache.as_mut() {
                    // (Same slot only: windows of neighbouring slots overlap
                    // by design, see `VWIN_SIZE`.)
                    let stale: Vec<u64> = vc.windows.iter().filter(|(_, w)| w.lo >> VWIN_SHIFT == slot >> VWIN_SHIFT && w.lo < hi && lo < w.hi).map(|(k, _)| *k).collect();
                    for sk in stale {
                        let w = vc.windows.remove(&sk).unwrap();
                        // (Lazy data there to guest memory first: the new window copies in.)
                        super::lazywb::release_buffer(w.buf);
                        vc.bytes -= w.bytes();
                        gone.push(w.buf);
                        self.retired.push((seq, w.retire()));
                    }
                }
                for b in gone {
                    self.retire_dirty_set(b, seq);
                }
                let cost = if sparse_mirrors() { 0 } else { hi - lo };
                if self.vcache.as_ref().map_or(0, |vc| vc.bytes) + cost > VCACHE_BUDGET {
                    return None;
                }
                self.vcache_new_window(lo, hi)?;
                lo
            }
        };
        let (epoch, required) = (self.cpu_epoch, self.required_epoch);
        let vc = self.vcache.as_mut().unwrap();
        let w = vc.windows.get_mut(&k)?;
        if base < w.lo || base + size > w.hi {
            return None;
        }
        w.last_used = super::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
        self.vcache_fill(k, base, size, epoch, required, overwrite)
    }

    /// Allocates the VRAM mirror window [lo, hi) (all pages not yet current).
    fn vcache_new_window(&mut self, lo: u64, hi: u64) -> Option<()> {
        {
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::INDEX_BUFFER;
            let made = if sparse_mirrors() {
                // (Rounded to 1 MiB, a multiple of any block size: the last
                // block is bound whole.)
                ctx().sparse_buffer((hi - lo).next_multiple_of(1 << 20), usage).and_then(|(buf, req)| {
                    let ty = ctx().memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL);
                    match (ty, req.alignment) {
                        (Some(ty), block) if block.is_power_of_two() && block <= SPARSE_CHUNK && (1 << 20) % block == 0 => {
                            let pool = self.sparse_pool.get_or_insert_with(|| BlockPool { block, ty, chunks: FastMap::default(), free: Vec::new() });
                            if pool.block != block || pool.ty != ty {
                                unsafe { ctx().device.destroy_buffer(buf, None) };
                                return Err(format!("sparse block {block:#x} / memory type {ty} differ from the pool's"));
                            }
                            let n = (hi - lo).div_ceil(block) as usize;
                            Ok((buf, vk::DeviceMemory::null(), Some(SparseMem { block, resident: vec![0; n.div_ceil(64)], bound: Vec::new() })))
                        }
                        _ => {
                            unsafe { ctx().device.destroy_buffer(buf, None) };
                            Err(format!("sparse buffer: unusable requirements {req:?}"))
                        }
                    }
                })
            } else {
                ctx().device_buffer(hi - lo, usage).map(|(b, m)| (b, m, None))
            };
            let (buf, mem, sparse) = match made {
                Ok(b) => b,
                Err(e) => {
                    self.report(format!("VRAM mirror of {lo:#x}-{hi:#x} unavailable: {e}"));
                    return None;
                }
            };
            crate::info!("VRAM mirror window {lo:#x}-{hi:#x} ({} MiB{}) for read-only vertex/index data", (hi - lo) >> 20, if sparse.is_some() { ", sparse" } else { "" });
            let pages = ((hi - lo) >> 12) as usize;
            let blocks = ((hi - lo + 0xFFFF) >> 16) as usize;
            let cost = if sparse.is_some() { 0 } else { hi - lo };
            self.vgen += 1;
            let w = VWindow { lo, hi, buf, mem, valid: vec![0; pages.div_ceil(64)], touched: vec![0; pages.div_ceil(64)], checked: vec![u64::MAX; blocks], stamp: vec![0; blocks], last_used: 0, wgen: self.vgen, sparse };
            let vc = self.vcache.get_or_insert_with(|| VCache { windows: FastMap::default(), bytes: 0, copied: 0, hits: 0 });
            vc.bytes += cost;
            vc.windows.insert(lo, w);
        }
        Some(())
    }

    /// Binds memory to the blocks of sparse window `k` that [base, base+size)
    /// covers and that lack it; the binds go out before the batch is
    /// submitted (`submit_batch`). False when block memory is unavailable.
    fn vcache_resident(&mut self, k: u64, base: u64, size: u64) -> bool {
        let Some(vc) = self.vcache.as_mut() else { return false };
        let VCache { windows, bytes, .. } = vc;
        let Some(w) = windows.get_mut(&k) else { return false };
        let Some(sp) = w.sparse.as_mut() else { return true };
        let (b0, b1) = ((base - w.lo) / sp.block, (base + size - 1 - w.lo) / sp.block);
        let mut binds = Vec::new();
        let mut ok = true;
        for b in b0..=b1 {
            let (word, bit) = ((b / 64) as usize, b % 64);
            if sp.resident[word] >> bit & 1 != 0 {
                continue;
            }
            let block = (*bytes + sp.block <= VCACHE_BUDGET).then(|| self.sparse_pool.as_mut().and_then(BlockPool::take)).flatten();
            let Some((memory, memory_offset)) = block else {
                ok = false;
                break;
            };
            binds.push(vk::SparseMemoryBind { resource_offset: b * sp.block, size: sp.block, memory, memory_offset, flags: vk::SparseMemoryBindFlags::empty() });
            sp.bound.push((memory, memory_offset));
            sp.resident[word] |= 1 << bit;
            *bytes += sp.block;
        }
        if !binds.is_empty() {
            let buf = w.buf;
            match self.sparse_binds.iter_mut().find(|(b, _)| *b == buf) {
                Some((_, v)) => v.extend(binds),
                None => self.sparse_binds.push((buf, binds)),
            }
        }
        if !ok {
            self.report(format!("VRAM mirror: no memory for blocks of {base:#x}+{size:#x}"));
        }
        ok
    }

    /// Copies the pages of [base, base+size) in window `k` that are not
    /// current into it; returns the binding.
    fn vcache_fill(&mut self, k: u64, base: u64, size: u64, epoch: u64, required: u64, overwrite: bool) -> Option<(vk::Buffer, u64)> {
        if !self.vcache_resident(k, base, size) {
            return None;
        }
        let vc = self.vcache.as_mut().unwrap();
        let w = vc.windows.get_mut(&k)?;
        // CPU writes: checked once per epoch and 64 KiB block (runs of
        // unchecked blocks in one write-watch query).
        let (b0, b1) = (((base - w.lo) >> 16) as usize, ((base + size - 1 - w.lo) >> 16) as usize);
        let mut i = b0;
        let valid = |c: u64| c != u64::MAX && c >= required;
        while i <= b1 {
            if valid(w.checked[i]) {
                i += 1;
                continue;
            }
            let mut j = i;
            let mut since = u64::MAX;
            while j <= b1 && !valid(w.checked[j]) {
                since = since.min(w.stamp[j]);
                j += 1;
            }
            let (rlo, rhi) = (w.lo + ((i as u64) << 16), (w.lo + ((j as u64) << 16)).min(w.hi));
            // (The GPU's own writes to the window keep it current; GPU writes
            // elsewhere invalidate it explicitly: `note_pending_gpu_write`.)
            let with_gpu = crate::tune::get(crate::tune::MIRROR_GPU_STAMPS) != 0;
            let (written, stamp) = self.writes.cpu_changed_since(rlo, rhi - rlo, since, with_gpu);
            if written {
                let mut runs = if with_gpu { self.writes.dirty_runs(rlo, rhi, since) } else { self.writes.cpu_dirty_runs(rlo, rhi, since) };
                if runs.is_empty() {
                    // (Written but no page stamps: no write watching.)
                    runs.push((rlo, rhi));
                }
                for (a, b) in runs {
                    bits_set(&mut w.valid, ((a - w.lo) >> 12) as usize, ((b - w.lo + 0xFFF) >> 12) as usize, false);
                }
                self.vgen += 1;
                w.wgen = self.vgen;
            }
            for x in i..j {
                w.checked[x] = epoch;
                w.stamp[x] = stamp;
            }
            i = j;
        }
        if overwrite {
            // Whole pages the caller overwrites: current once it has.
            let (f0, f1) = (((base - w.lo + 0xFFF) >> 12) as usize, ((base + size - w.lo) >> 12) as usize);
            if f0 < f1 {
                bits_set(&mut w.valid, f0, f1, true);
            }
        }
        let (p0, p1) = (((base - w.lo) >> 12) as usize, ((base + size - w.lo + 0xFFF) >> 12) as usize);
        bits_set(&mut w.touched, p0, p1, true);
        let runs = bits_clear_runs(&w.valid, p0, p1);
        let (wlo, whi, vbuf) = (w.lo, w.hi, w.buf);
        if !runs.is_empty() {
            let (lo, hi) = (wlo + ((runs[0].0 as u64) << 12), (wlo + ((runs[runs.len() - 1].1 as u64) << 12)).min(whi));
            // Shader writes not in guest memory yet (this batch's write-backs,
            // submitted batches' not yet landed): copied over the pages on the
            // GPU after them, from where the writes are (as `upload_guest`
            // forwards), instead of waiting for them to land. Not from this
            // window (the copy-in replaces its pages), from in-place or
            // CPU-performed writes (end-of-pipe actions), or from batches whose
            // device-ring segment may be reused: those are waited for.
            let abs: Vec<(u64, u64)> = runs.iter().map(|&(a, b)| (wlo + ((a as u64) << 12), (wlo + ((b as u64) << 12)).min(whi))).collect();
            // (The window's own pending writes are left out of the copy-in:
            // the window holds their data already.)
            let fwd = self.mirror_forwards_keep(&abs, vbuf);
            if fwd.is_none() {
                if let Some(wb) = self.writebacks.iter().find(|wb| wb.base < hi && lo < wb.base + wb.size) {
                    if perf_on() {
                        let k = format!("mirror copy-in {lo:#x}-{hi:#x} ({}) behind this batch's write-back {:#x}+{:#x}{}", self.vc_why, wb.base, wb.size, if wb.mirror { " (mirror)" } else { "" });
                        *GUARD_SITES.lock().get_or_insert_with(HashMap::new).entry(k).or_default() += 1;
                    }
                    self.flush_for(1);
                } else if cpu_writeback_in_flight(lo, hi - lo) {
                    self.flushes[1] += 1;
                    wait_cpu_writebacks(lo, hi - lo);
                }
            }
            let keep: Vec<(u64, u64)> = fwd.as_ref().map_or(Vec::new(), |f| f.1.iter().map(|&(a, b)| (a.saturating_sub(wlo), b.saturating_sub(wlo))).collect());
            if !keep.is_empty() {
                self.copyin_kept += 1;
            }
            let spans = subtract_ranges(&runs.iter().map(|&(a, b)| ((a as u64) << 12, ((b as u64) << 12).min(whi - wlo))).collect::<Vec<_>>(), &keep);
            let (src, regions) = if spans.is_empty() {
                (vbuf, Vec::new())
            } else if ctx().ext_host.is_some() {
                let (zbuf, zoff, _) = self.zc_binding(wlo, whi - wlo).filter(|z| z.2 >= whi - wlo)?;
                (zbuf, spans.iter().map(|&(off, end)| vk::BufferCopy { src_offset: zoff + off, dst_offset: off, size: end - off }).collect::<Vec<_>>())
            } else {
                // Guest memory is not imported: staged through the upload
                // ring (a buffer of its own when the ring cannot take it),
                // snapshotted now like other uploads.
                let total: u64 = spans.iter().map(|&(o, e)| e - o).sum();
                let (sbuf, sbase, sptr) = match (self.ring.remaining() >= total + 4096).then(|| self.ring.alloc(total)).flatten() {
                    Some(off) => (self.ring.buffer, off, None),
                    None => {
                        let (sb, sm, sp) = match ctx().host_buffer(total, vk::BufferUsageFlags::TRANSFER_SRC) {
                            Ok(s) => s,
                            Err(e) => {
                                self.report(format!("VRAM mirror staging of {total:#x} bytes unavailable: {e}"));
                                return None;
                            }
                        };
                        self.retired.push((self.last_seq + 1, Retired::Buffer(sb, sm)));
                        (sb, 0, Some(sp))
                    }
                };
                let mut regions = Vec::with_capacity(spans.len());
                let mut at = 0;
                for &(off, end) in &spans {
                    let n = end - off;
                    match sptr {
                        Some(p) => resources::copy_from_guest(unsafe { std::slice::from_raw_parts_mut(p.add(at as usize), n as usize) }, wlo + off),
                        None => self.ring.write_guest(sbase + at, wlo + off, n),
                    }
                    regions.push(vk::BufferCopy { src_offset: sbase + at, dst_offset: off, size: n });
                    at += n;
                }
                (sbuf, regions)
            };
            // (After a full barrier: earlier commands of this batch may have
            // written the source in place.)
            self.begin();
            self.end_rendering();
            global_barrier(self.cmd);
            if !regions.is_empty() {
                unsafe { ctx().device.cmd_copy_buffer(self.cmd, src, vbuf, &regions) };
            }
            // Forwards land after the copy-in and earlier forwards they
            // overlap (a barrier then), in program order.
            let mut since: Vec<(u64, u64)> = regions.iter().map(|r| (r.dst_offset, r.dst_offset + r.size)).collect();
            for &(pbase, pbuf, poff, psize) in fwd.iter().flat_map(|f| f.0.iter()) {
                let over: Vec<vk::BufferCopy> = spans
                    .iter()
                    .filter_map(|&(off, end)| {
                        let (a, b) = (pbase.max(wlo + off), (pbase + psize).min(wlo + end));
                        (a < b).then(|| vk::BufferCopy { src_offset: poff + (a - pbase), dst_offset: a - wlo, size: b - a })
                    })
                    .collect();
                if !over.is_empty() {
                    if over.iter().any(|r| since.iter().any(|&(x, y)| r.dst_offset < y && x < r.dst_offset + r.size)) {
                        global_barrier(self.cmd);
                        since.clear();
                    }
                    since.extend(over.iter().map(|r| (r.dst_offset, r.dst_offset + r.size)));
                    unsafe { ctx().device.cmd_copy_buffer(self.cmd, pbuf, vbuf, &over) };
                    self.flushes[5] += 1;
                }
            }
            unordered!(self);
            let vc = self.vcache.as_mut().unwrap();
            let w = vc.windows.get_mut(&k).unwrap();
            for &(a, b) in &runs {
                bits_set(&mut w.valid, a, b, true);
            }
            let copied = regions.iter().map(|r| r.size).sum::<u64>();
            vc.copied += copied;
            if perf_on() {
                let mut g = VCACHE_FILL_SITES.lock();
                let e = g.get_or_insert_with(HashMap::new).entry((base, size, self.vc_why)).or_default();
                e.0 += 1;
                e.1 += copied;
            }
            self.vc_why = "other";
            // The copy reads guest memory in place when it executes.
            if ctx().ext_host.is_some() {
                for r in &regions {
                    self.zc_read(wlo + r.dst_offset, wlo + r.dst_offset + r.size);
                }
            }
            self.mark(|| "vertex/index data to VRAM".into());
        }
        let vc = self.vcache.as_mut().unwrap();
        vc.hits += 1;
        if !overwrite {
            if let Some(w) = vc.windows.get(&k) {
                let (b0, b1) = (((base - w.lo) >> 16) as usize, ((base + size - 1 - w.lo) >> 16) as usize);
                let checked = w.checked[b0..=b1].iter().copied().min().unwrap_or(u64::MAX);
                if self.vbind.len() >= 1 << 16 {
                    self.vbind.clear();
                }
                self.vbind.insert((base, size), (k, w.wgen, checked, vbuf, base - wlo));
            }
        }
        Some((vbuf, base - wlo))
    }

    /// Shader writes to the guest ranges `spans` that a copy of guest memory
    /// taken now lacks, in the order they land: submitted batches' (device
    /// copies intact while their slot is not reused), then this batch's, as
    /// (guest base, buffer, offset, size). None when one of them cannot be
    /// forwarded -- from the buffer `exclude`, from a released mirror window,
    /// from a batch whose slot may be reused, in-place or CPU-performed
    /// writes (end-of-pipe actions) -- or with the import: the caller waits.
    fn mirror_forwards(&self, spans: &[(u64, u64)], exclude: vk::Buffer) -> Option<Vec<(u64, vk::Buffer, u64, u64)>> {
        self.mirror_forwards_keep(spans, exclude).filter(|(_, keep)| keep.is_empty()).map(|(out, _)| out)
    }

    /// `mirror_forwards` for a copy-in into the mirror window `exclude`:
    /// writes the window already holds (its own mirror-resident writes) are
    /// returned as guest ranges to leave out of the copy-in instead of
    /// failing, unless a forward overlaps one (their order is not known).
    fn mirror_forwards_keep(&self, spans: &[(u64, u64)], exclude: vk::Buffer) -> Option<(Vec<(u64, vk::Buffer, u64, u64)>, Vec<(u64, u64)>)> {
        if ctx().ext_host.is_some() || spans.is_empty() {
            return None;
        }
        let (lo, hi) = (spans[0].0, spans[spans.len() - 1].1);
        let overlaps = |w: u64, n: u64| w < hi && lo < w + n && spans.iter().any(|&(a, b)| w < b && a < w + n);
        let mut out = Vec::new();
        if cpu_writeback_in_flight(lo, hi - lo) || writeback_in_flight(lo, hi - lo) {
            let oldest = (self.last_seq + 2).saturating_sub(SLOTS as u64);
            for c in COMPLETIONS.lock().iter() {
                if c.zc_writes.iter().any(|&(w, n)| overlaps(w, n)) || c.actions.iter().any(|a| spans.iter().any(|&(x, y)| a.overlaps(x, y - x))) {
                    return None;
                }
                if c.writebacks.iter().any(|&(w, _, n)| overlaps(w, n)) && (c.seq < oldest || c.fence == vk::Fence::null()) {
                    return None;
                }
                out.extend(c.dev_writes.iter().filter(|&&(w, _, _, n)| overlaps(w, n)).copied());
            }
        }
        out.extend(self.writebacks.iter().filter(|p| overlaps(p.base, p.size)).map(|p| (p.base, p.dev_buf, p.dev_off, p.size)));
        let live = |b: vk::Buffer| b == self.dring.buffer || self.vcache.as_ref().is_some_and(|vc| vc.windows.values().any(|w| w.buf == b));
        let (keep, out): (Vec<_>, Vec<_>) = out.into_iter().partition(|e| exclude != vk::Buffer::null() && e.1 == exclude);
        if out.iter().any(|e| !live(e.1)) {
            return None;
        }
        let keep: Vec<(u64, u64)> = keep.iter().map(|e| (e.0, e.0 + e.3)).collect();
        if out.iter().any(|e| keep.iter().any(|&(a, b)| e.0 < b && a < e.0 + e.3)) {
            return None;
        }
        Some((out, keep))
    }

    /// VRAM buffer cache statistics: (MiB allocated, windows, MiB copied in,
    /// bindings served).
    pub fn vcache_stats(&self) -> Option<(u64, usize, u64, u64)> {
        self.vcache.as_ref().map(|v| (v.bytes >> 20, v.windows.len(), v.copied >> 20, v.hits))
    }

    /// GPU or CP writes to guest [lo, lo+len): mirrored pages there must be
    /// copied in again before their next use.
    /// The pages of [lo, lo+len) in the mirror window `buf` hold the newest
    /// data (the GPU wrote them there); other windows overlapping the range
    /// (neighbouring slots) were invalidated with the write.
    fn vcache_validate(&mut self, buf: vk::Buffer, lo: u64, len: u64) {
        let Some(vc) = self.vcache.as_mut() else { return };
        let hi = lo.saturating_add(len.max(1));
        for w in vc.windows.values_mut().filter(|w| w.buf == buf) {
            let (a, b) = (lo.max(w.lo), hi.min(w.hi));
            if a < b {
                bits_set(&mut w.valid, ((a - w.lo) >> 12) as usize, ((b - w.lo + 0xFFF) >> 12) as usize, true);
            }
        }
    }

    fn vcache_invalidate(&mut self, lo: u64, len: u64) {
        let Some(vc) = self.vcache.as_mut() else { return };
        let hi = lo.saturating_add(len.max(1));
        for w in vc.windows.values_mut() {
            let (a, b) = (lo.max(w.lo), hi.min(w.hi));
            if a < b {
                bits_set(&mut w.valid, ((a - w.lo) >> 12) as usize, ((b - w.lo + 0xFFF) >> 12) as usize, false);
                // (Not the window a GPU write lands in: `note_mirror_write`.)
                if w.buf != self.vgen_keep {
                    self.vgen += 1;
                    w.wgen = self.vgen;
                }
            }
        }
    }

    /// The device-local scratch buffer, at least `size` bytes.
    fn scratch_buffer(&mut self, size: u64) -> Result<vk::Buffer, String> {
        if let Some((b, n)) = self.scratch {
            if n >= size {
                return Ok(b);
            }
        }
        let n = size.next_multiple_of(64 << 20);
        let (b, m) = ctx().device_buffer(n, vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER)?;
        if let (Some((ob, _)), Some(om)) = (self.scratch.take(), self.scratch_mem.take()) {
            self.retired_scratch.push((ob, om));
        }
        self.scratch = Some((b, n));
        self.scratch_mem = Some(m);
        crate::debug!("render: scratch buffer {} MiB", n >> 20);
        Ok(b)
    }

    /// Texture view of a depth render target's depth values: copied (depth
    /// aspect -> ring -> color image, cropped to the T# size) whenever the
    /// depth target was written since the last copy.
    fn depth_shadow_view(&mut self, src: ImageKey, t: &Tsharp, vfmt: vk::Format, view_type: vk::ImageViewType, load_only: bool) -> Result<vk::ImageView, String> {
        // Same size (e.g. shadow maps): sample the depth aspect directly; a
        // depth view returns the depth in R as the R32/R16 texture would.
        // TLG_DEPTH_COPY=1 always copies.
        static COPY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        // Smaller T# read only by integer texel coordinates (e.g. the
        // 1920x1080 view of a 2048x1152 depth target): the same texels.
        let fits = (t.width == src.width && t.height == src.height) || (load_only && t.width <= src.width && t.height <= src.height && crate::tune::get(crate::tune::DEPTH_LOAD_VIEW) != 0);
        if fits && view_type == vk::ImageViewType::TYPE_2D && !*COPY.get_or_init(|| std::env::var_os("TLG_DEPTH_COPY").is_some()) {
            if t.width != src.width || t.height != src.height {
                DEPTH_VIEWS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            let img = self.images.get_mut(&src).unwrap();
            return img.view(src.format, 0, view_type, false, 0, 1);
        }
        let wgen = self.write_gen.get(&src).copied().unwrap_or(0);
        let skey = (src, t.width, t.height, vfmt);
        let dst_key = ImageKey { base: t.base ^ 0x4000_0000_0000_0000, width: t.width, height: t.height, layers: 1, levels: 1, format: vfmt, depth: false, ty3d: false };
        let fresh = self.shadows.get(&skey).is_some_and(|&(_, g)| g == wgen);
        if !fresh {
            self.init_image(dst_key)?;
            self.begin();
            self.end_rendering();
            let (w, h) = (t.width.min(src.width), t.height.min(src.height));
            if crate::tune::get(crate::tune::DEPTH_COPY_CS) != 0 && resources::storage_capable(ctx(), vfmt) && self.depth_copy_compute(src, dst_key, w, h)? {
                self.mark(|| "depth copy for sampling (compute)".into());
                DEPTH_VIEWS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                DEPTH_VIEWS[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                self.shadows.insert(skey, (dst_key, wgen));
                self.images.get_mut(&dst_key).unwrap().gpu_written = true;
                let img = self.images.get_mut(&dst_key).unwrap();
                return img.view(vfmt, t.dst_sel, view_type, false, 0, 1);
            }
            let bpp = if vfmt == vk::Format::R16_UNORM { 2 } else { 4 };
            // Through device memory: shadow maps (4096x4096, 64 MiB) do not
            // fit a host ring segment.
            let scratch = self.scratch_buffer(w as u64 * h as u64 * bpp)?;
            let off = 0;
            global_barrier(self.cmd);
            let c = ctx();
            let to_buf = vk::BufferImageCopy {
                buffer_offset: off,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::DEPTH, mip_level: 0, base_array_layer: 0, layer_count: 1 },
                image_offset: vk::Offset3D::default(),
                image_extent: vk::Extent3D { width: w, height: h, depth: 1 },
            };
            let to_img = vk::BufferImageCopy { image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 }, ..to_buf };
            unsafe {
                c.device.cmd_copy_image_to_buffer(self.cmd, self.images[&src].image, vk::ImageLayout::GENERAL, scratch, &[to_buf]);
                global_barrier(self.cmd);
                c.device.cmd_copy_buffer_to_image(self.cmd, scratch, self.images[&dst_key].image, vk::ImageLayout::GENERAL, &[to_img]);
            }
            unordered!(self);
            self.mark(|| "depth copy for sampling".into());
            DEPTH_VIEWS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.shadows.insert(skey, (dst_key, wgen));
            self.images.get_mut(&dst_key).unwrap().gpu_written = true;
        }
        let img = self.images.get_mut(&dst_key).unwrap();
        img.view(vfmt, t.dst_sel, view_type, false, 0, 1)
    }

    /// Peak segment use of the host and device rings since the last call.
    pub fn ring_peaks(&mut self) -> ((u64, u64), (u64, u64)) {
        (self.ring.take_peak(), self.dring.take_peak())
    }

    /// The depth-copy kernel's pipeline (created on first use).
    fn depth_copy_pipe(&mut self) -> Result<(vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline), String> {
        if let Some(p) = self.depth_copy_pipe {
            return Ok(p);
        }
        let c = ctx();
        let bindings = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let set_layout = unsafe { c.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None) }.map_err(|e| e.to_string())?;
        let pcr = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 8 }];
        let sls = [set_layout];
        let layout = unsafe { c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sls).push_constant_ranges(&pcr), None) }.map_err(|e| e.to_string())?;
        let m = self.module(0x4445_5054_4843_5900, &gcn::translate::kernels::depth_copy_cs())?;
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(m).name(c"main");
        let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
        let pipeline = unsafe { c.device.create_compute_pipelines(vk::PipelineCache::null(), &[ci], None) }.map_err(|(_, e)| format!("depth copy pipeline: {e}"))?[0];
        self.depth_copy_pipe = Some((set_layout, layout, pipeline));
        Ok((set_layout, layout, pipeline))
    }

    /// Copies [0, w) x [0, h) of depth image `src` into color image `dst` by
    /// `kernels::depth_copy_cs`. False when it could not (no descriptor set
    /// left in the batch's pool: allocating one here must not split the
    /// batch, a draw's setup is under way): the caller transfers instead.
    fn depth_copy_compute(&mut self, src: ImageKey, dst: ImageKey, w: u32, h: u32) -> Result<bool, String> {
        let (set_layout, layout, pipeline) = self.depth_copy_pipe()?;
        let c = ctx();
        let layouts = [set_layout];
        let ai = vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.desc_pool).set_layouts(&layouts);
        let Ok(sets) = (unsafe { c.device.allocate_descriptor_sets(&ai) }) else { return Ok(false) };
        let set = sets[0];
        let sv = self.images.get_mut(&src).unwrap().view(src.format, 0, vk::ImageViewType::TYPE_2D, false, 0, 1)?;
        let dv = self.images.get_mut(&dst).unwrap().view(dst.format, 0, vk::ImageViewType::TYPE_2D, true, 0, 1)?;
        let si = [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: sv, image_layout: vk::ImageLayout::GENERAL }];
        let di = [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: dv, image_layout: vk::ImageLayout::GENERAL }];
        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::SAMPLED_IMAGE).image_info(&si),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&di),
        ];
        let pc: Vec<u8> = [w, h].iter().flat_map(|v| v.to_le_bytes()).collect();
        unsafe {
            c.device.update_descriptor_sets(&writes, &[]);
            global_barrier(self.cmd);
            c.device.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            c.device.cmd_bind_descriptor_sets(self.cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
            c.device.cmd_push_constants(self.cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, &pc);
            c.device.cmd_dispatch(self.cmd, w.div_ceil(8), h.div_ceil(8), 1);
        }
        unordered!(self);
        Ok(true)
    }

    fn shadow_view(&mut self, src: ImageKey, t: &Tsharp, vfmt: vk::Format, store_fmt: vk::Format, view_type: vk::ImageViewType) -> Result<vk::ImageView, String> {
        // Array/cube textures may span several render targets laid out back
        // to back in guest memory (e.g. six cube shadow maps sampled as one
        // cube array): collect the consecutive ones that cover the layers.
        let want = if matches!(t.ty, desc::T_CUBE | desc::T_1D_ARRAY | desc::T_2D_ARRAY) { t.total_layers().max(src.layers) } else { src.layers };
        let mut parts: smallvec::SmallVec<[(ImageKey, u32); 4]> = smallvec::smallvec![(src, 0u32)];
        let mut covered = src.layers;
        let mut next = self.images.get(&src).and_then(|i| Self::image_guest_layout(&src, i)).map(|g| src.base + g.3);
        while covered < want {
            let Some(addr) = next else { break };
            let Some(k) = self.by_base.get(&addr).and_then(|ks| {
                ks.iter().copied().find(|k| self.images[k].gpu_written && !k.depth && k.width == src.width && k.height == src.height && format_class(k.format) == format_class(src.format))
            }) else {
                break;
            };
            parts.push((k, covered));
            covered += k.layers;
            next = Self::image_guest_layout(&k, &self.images[&k]).map(|g| k.base + g.3);
        }
        let layers = covered.min(want).max(1);
        let wgen: u64 = parts.iter().map(|(k, _)| self.write_gen.get(k).copied().unwrap_or(0)).sum();
        let skey = (src, t.width, t.height, store_fmt);
        let dst_key = ImageKey { base: t.base ^ 0x8000_0000_0000_0000, width: t.width, height: t.height, layers, levels: 1, format: store_fmt, depth: false, ty3d: false };
        let fresh = self.shadows.get(&skey).is_some_and(|&(k, g)| g == wgen && k == dst_key);
        if !fresh {
            self.init_image(dst_key)?;
            self.begin();
            self.end_rendering();
            if self.barrier_needed {
                global_barrier(self.cmd);
                self.barrier_needed = false;
            }
            for &(k, at) in &parts {
                let n = k.layers.min(layers.saturating_sub(at));
                if n == 0 {
                    continue;
                }
                let region = vk::ImageCopy {
                    src_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: n },
                    src_offset: vk::Offset3D::default(),
                    dst_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: at, layer_count: n },
                    dst_offset: vk::Offset3D::default(),
                    extent: vk::Extent3D { width: t.width, height: t.height, depth: 1 },
                };
                unsafe {
                    ctx().device.cmd_copy_image(self.cmd, self.images[&k].image, vk::ImageLayout::GENERAL, self.images[&dst_key].image, vk::ImageLayout::GENERAL, &[region]);
                }
            }
            unordered!(self);
            self.mark(|| "shadow copy for sampling".into());
            self.shadows.insert(skey, (dst_key, wgen));
            self.images.get_mut(&dst_key).unwrap().gpu_written = true;
        }
        let img = self.images.get_mut(&dst_key).unwrap();
        img.view(vfmt, t.dst_sel, view_type, false, t.view_base_layer(), t.layers().max(1))
    }

    fn upload_texture(&mut self, key: ImageKey, t: &Tsharp, layers: u32) -> Result<(), String> {
        self.uploads += 1;
        super::set_phase(super::phase::UPLOAD, t.base);
        let t0 = std::time::Instant::now();
        let data = resources::read_texture(t, layers);
        self.prof[4] += t0.elapsed();
        if perf_on() {
            let s = format!("{:#x} {}x{}x{} dfmt {} nfmt {} tiling {} ({} KiB)", t.base, t.width, t.height, layers, t.dfmt, t.nfmt, t.tiling_index, data.len() >> 10);
            let us = t0.elapsed().as_micros() as u64;
            let mut g = UPLOAD_SITES.lock();
            let e = g.get_or_insert_with(HashMap::new).entry(s).or_default();
            e.0 += 1;
            e.1 += us;
        }
        let Some(off) = self.ring.alloc(data.len() as u64) else {
            return Err(format!("upload ring full: {:#x}-byte texture {:#x} {}x{}x{} (segment {:#x})", data.len(), t.base, t.width, t.height, layers, self.ring.segment_size()));
        };
        self.ring.write(off, &data);
        self.begin();
        self.end_rendering();
        if self.barrier_needed {
            global_barrier(self.cmd);
            self.barrier_needed = false;
        }
        let (bpp, block) = resources::element_bits(t.dfmt);
        // One region per level (`read_texture` packs level after level, each
        // with all its layers).
        let mut regions: smallvec::SmallVec<[vk::BufferImageCopy; 16]> = smallvec::SmallVec::new();
        let mut pos = off;
        for (l, lv) in resources::texture_layout(t, layers).iter().enumerate().take(key.levels.max(1) as usize) {
            regions.push(vk::BufferImageCopy {
                buffer_offset: pos,
                buffer_row_length: lv.width * block,
                buffer_image_height: lv.height * block,
                image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: l as u32, base_array_layer: 0, layer_count: if key.ty3d { 1 } else { layers } },
                image_offset: vk::Offset3D::default(),
                image_extent: vk::Extent3D { width: (t.width >> l).max(1), height: (t.height >> l).max(1), depth: if key.ty3d { layers } else { 1 } },
            });
            pos += lv.width as u64 * lv.height as u64 * (bpp / 8) as u64 * layers as u64;
        }
        unsafe {
            ctx().device.cmd_copy_buffer_to_image(self.cmd, self.ring.buffer, self.images[&key].image, vk::ImageLayout::GENERAL, &regions);
        }
        // TLG_MIP_CHECK=<n>: level-to-level correlations of the first n chains
        // per tile mode (chains whose level 1 correlates poorly are dumped).
        static CHECK: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
        static CHECKED: [std::sync::atomic::AtomicU32; 32] = [const { std::sync::atomic::AtomicU32::new(0) }; 32];
        let limit = *CHECK.get_or_init(|| std::env::var("TLG_MIP_CHECK").ok().and_then(|v| v.parse().ok()).unwrap_or(0));
        if limit > 0 && key.levels > 1 {
            static KINDS: Mutex<Option<std::collections::HashSet<(u8, u8, u8, bool, u32)>>> = Mutex::new(None);
            if KINDS.lock().get_or_insert_with(Default::default).insert((t.ty, t.dfmt, t.tiling_index, t.pow2_pad, layers.min(7))) {
                // Padding slices (power-of-two slice counts) should hold no data.
                let lv = resources::texture_layout(t, layers);
                let pad: Vec<String> = lv
                    .iter()
                    .take(3)
                    .map(|l| {
                        let n = (l.slices - layers.min(l.slices)) as u64 * l.slice_bytes;
                        let mut b = vec![0u8; n.min(1 << 20) as usize];
                        resources::copy_from_guest(&mut b, t.base + l.offset + layers.min(l.slices) as u64 * l.slice_bytes);
                        format!("{}/{} zero", b.iter().filter(|&&x| x == 0).count(), b.len())
                    })
                    .collect();
                crate::info!("mip chain kind: type {} dfmt {} tiling {} pow2 {} layers {} (first {:#x} {}x{} levels {}..{}; padding slices of levels 0-2: {})", t.ty, t.dfmt, t.tiling_index, t.pow2_pad, layers, t.base, t.width, t.height, t.base_level, t.last_level, pad.join(", "));
            }
        }
        let checked = &CHECKED[t.tiling_index as usize & 31];
        if key.levels > 1 && checked.load(std::sync::atomic::Ordering::Relaxed) < limit {
            let c = super::texdump::mip_correlations(t, &data, layers);
            if c.iter().any(|v| v.is_some()) {
                checked.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if c.first().copied().flatten().is_some_and(|v| v < 0.5) {
                    super::texdump::dump_levels(t, &data, layers, 3);
                }
                let s: Vec<String> = c.iter().map(|v| v.map_or("-".into(), |v| format!("{v:.2}"))).collect();
                crate::info!(
                    "mip check {:#x} {}x{}x{} dfmt {} tiling {} pow2 {} type {} levels {}..{}: [{}]",
                    t.base,
                    t.width,
                    t.height,
                    layers,
                    t.dfmt,
                    t.tiling_index,
                    t.pow2_pad,
                    t.ty,
                    t.base_level,
                    t.last_level,
                    s.join(" ")
                );
            }
        }
        unordered!(self);
        self.mark(|| "texture upload".into());
        if crate::log::enabled(crate::log::Level::Debug) {
            crate::debug!("uploaded texture {:#x} {}x{}x{} dfmt {} nfmt {} tiling {}", t.base, t.width, t.height, layers, t.dfmt, t.nfmt, t.tiling_index);
        }
        super::texdump::maybe_dump(t, &data);
        Ok(())
    }

    /// Whether GPU writes recorded in the current (unsubmitted) batch overlap
    /// guest memory [base, base+size).
    /// A write-back this batch performs at completion (also entered in
    /// `CP_VIEW`).
    fn push_writeback(&mut self, w: PendingWrite) {
        CP_VIEW.lock().pending.insert(w.base, w.base + w.size.max(1));
        self.writebacks.push(w);
    }

    /// An in-place GPU write of this batch (also entered in `CP_VIEW`).
    fn push_zc_write(&mut self, base: u64, len: u64) {
        CP_VIEW.lock().pending.insert(base, base + len.max(1));
        self.zc_writes.push((base, len));
    }

    /// An in-place GPU read of this batch (also entered in `CP_VIEW`).
    fn zc_read(&mut self, lo: u64, hi: u64) {
        CP_VIEW.lock().reads.insert(lo, hi);
        self.zc_reads.insert(lo, hi);
    }

    /// Applies the mirror and upload-cache invalidations of direct CP writes
    /// queued by `sync_write` (before any command that may use those caches).
    fn drain_cp_writes(&mut self) {
        if !CP_WRITES_QUEUED.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        let writes = {
            let mut v = CP_VIEW.lock();
            CP_WRITES_QUEUED.store(false, std::sync::atomic::Ordering::Release);
            std::mem::take(&mut v.cp_writes)
        };
        for (a, n) in writes {
            self.vcache_invalidate(a, n);
            self.forget_uploads(a, n);
        }
    }

    fn pending_write_overlaps(&self, base: u64, size: u64) -> bool {
        self.writebacks.iter().any(|p| p.base < base + size && base < p.base + p.size) || self.zc_overlaps(base, size)
    }

    /// Whether zero-copy writes recorded in this batch overlap [base, base+size).
    fn zc_overlaps(&self, base: u64, size: u64) -> bool {
        self.zc_writes.iter().any(|&(w, n)| w < base + size && base < w + n)
    }

    /// Whether a buffer binding uses zero-copy: large ranges (sparse scatter
    /// destinations, big tables) are bound in place instead of being copied.
    fn zc_candidate(&self, base: u64, size: u64) -> bool {
        size >= ZC_MIN_BYTES && base % 16 == 0 && ctx().ext_host.is_some()
    }

    /// Whether buffer binding `i` of `p` is accessed in place in imported
    /// guest memory: large ranges always, and read-only ranges whose guest
    /// memory is current (no device-ring writes of this batch, no write-back
    /// in flight) -- the GPU then reads at execution time as the hardware
    /// does, instead of a CPU copy per draw. TLG_INPLACE_READS=0 disables the
    /// read-only case.
    fn in_place(&self, p: &Prepared, i: usize, base: u64, size: u64) -> bool {
        if self.zc_candidate(base, size) {
            return true;
        }
        // The scatter kernel's destination: its records name the few dwords
        // it stores (`sparse_writes`), which go straight to guest memory
        // (a device-ring round trip would copy the whole binding twice and
        // mark every render target in it as overwritten).
        if p.sparse_writes.is_some() && p.scatter.is_some_and(|(_, d)| d == i) && base % 16 == 0 && ctx().ext_host.is_some() {
            return true;
        }
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var("TLG_INPLACE_READS").map(|v| v != "0").unwrap_or(true)) {
            return false;
        }
        ctx().ext_host.is_some()
            && !buffer_written(p, i)
            && !self.writebacks.iter().any(|w| w.base < base + size && base < w.base + w.size)
            && !cpu_writeback_in_flight(base, size)
    }

    /// Whether a buffer written by a shader can make its device-ring round
    /// trip on the GPU timeline (guest range importable).
    fn gpu_roundtrip_ok(&mut self, base: u64, size: u64) -> bool {
        ctx().ext_host.is_some() && self.zc_binding(base, size).is_some_and(|z| z.2 >= size)
    }

    /// Imported buffer covering guest [base, base+size): (buffer, offset, range).
    fn zc_binding(&mut self, base: u64, size: u64) -> Option<(vk::Buffer, u64, u64)> {
        if ctx().ext_host.is_none() {
            return None; // zero-copy disabled (TLG_ZERO_COPY)
        }
        let max_range = (ctx().limits.max_storage_buffer_range as u64) & !3;
        let span = match self.zc_spans.iter().find(|s| s.0 <= base && base < s.1) {
            Some(s) => (s.0, s.1, s.2),
            None => {
                let (start, end) = crate::mem::MEM.lock().committed_span(base)?;
                let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER;
                match ctx().import_host(start, end - start, usage) {
                    Ok((buf, mem)) => {
                        crate::info!("zero-copy: imported guest memory {start:#x}-{end:#x} ({} MiB)", (end - start) >> 20);
                        self.zc_spans.push((start, end, buf, mem));
                        (start, end, buf)
                    }
                    Err(e) => {
                        self.report(format!("zero-copy import of {start:#x}-{end:#x} failed: {e}"));
                        return None;
                    }
                }
            }
        };
        let range = size.min(span.1 - base).min(max_range) & !3;
        (range >= 4).then_some((span.2, base - span.0, range))
    }

    /// Drops imported spans when guest mappings changed (GPU idle first).
    fn zc_revalidate(&mut self) {
        let g = crate::mem::MAP_GEN.load(std::sync::atomic::Ordering::SeqCst);
        if g == self.zc_gen {
            return;
        }
        self.zc_gen = g;
        // VRAM mirror pages of remapped ranges are stale.
        for (lo, hi) in crate::mem::take_map_changes() {
            self.vcache_invalidate(lo, hi.saturating_sub(lo));
        }
        if self.zc_spans.is_empty() {
            return;
        }
        self.flush();
        if let Some(vc) = self.vcache.take() {
            for (_, w) in vc.windows {
                // (The GPU is idle: released now.)
                super::lazywb::release_buffer(w.buf);
                self.retire_dirty_set(w.buf, 0);
                self.sparse_binds.retain(|(b, _)| *b != w.buf);
                unsafe { ctx().device.destroy_buffer(w.buf, None) };
                if w.mem != vk::DeviceMemory::null() {
                    ctx().free(w.mem);
                }
                if let (Some(p), Some(s)) = (self.sparse_pool.as_mut(), w.sparse) {
                    p.give(s.bound);
                }
            }
        }
        let c = ctx();
        for (start, end, buf, mem) in self.zc_spans.drain(..) {
            unsafe {
                c.device.destroy_buffer(buf, None);
                c.free(mem);
            }
            crate::debug!("zero-copy: released {start:#x}-{end:#x} (guest mappings changed)");
        }
    }

    /// Makes guest memory [base, base+size) reflect all GPU writes recorded so
    /// far (submits and waits as needed). Returns whether it had to wait.
    #[track_caller]
    pub fn sync_guest(&mut self, base: u64, size: u64) -> bool {
        if self.pending_write_overlaps(base, size) {
            self.flush_for(1);
            return true;
        }
        if writeback_in_flight(base, size) {
            self.flushes[1] += 1;
            let t0 = std::time::Instant::now();
            wait_overlapping(base, size);
            self.prof[0] += t0.elapsed();
            return true;
        }
        false
    }

    /// Copies guest memory [base, base+size) into a new ring range for GPU
    /// reads and returns its offset. GPU writes that have not reached guest
    /// memory yet are applied: writes of submitted batches are waited for,
    /// writes recorded in this batch are forwarded with GPU copies (in
    /// program order) on top of the guest data.
    fn upload_guest(&mut self, base: u64, size: u64) -> Result<u64, String> {
        // Shader writes of submitted batches that have not reached guest
        // memory yet: forwarded from their device-ring copies (intact until
        // their slot is reused, which the queue orders after this batch)
        // rather than waited for.
        let mut inflight: Vec<(u64, vk::Buffer, u64, u64)> = Vec::new();
        if writeback_in_flight(base, size) {
            let oldest = (self.last_seq + 2).saturating_sub(SLOTS as u64);
            let mut forward = true;
            for c in COMPLETIONS.lock().iter() {
                let overlaps = |w: u64, n: u64| w < base + size && base < w + n;
                if c.zc_writes.iter().any(|&(w, n)| overlaps(w, n)) || (c.writebacks.iter().any(|&(w, _, n)| overlaps(w, n)) && (c.seq < oldest || c.fence == vk::Fence::null())) {
                    forward = false;
                    break;
                }
                inflight.extend(c.dev_writes.iter().filter(|&&(w, _, _, n)| overlaps(w, n)).copied());
            }
            // (Mirror windows hold data until they are released: only live
            // ones can be forwarded from.)
            let live = |b: vk::Buffer| b == self.dring.buffer || self.vcache.as_ref().is_some_and(|vc| vc.windows.values().any(|w| w.buf == b));
            if !inflight.iter().all(|e| live(e.1)) {
                forward = false;
            }
            if !forward {
                inflight.clear();
                self.flushes[1] += 1;
                let t0 = std::time::Instant::now();
                wait_overlapping(base, size);
                self.prof[0] += t0.elapsed();
            }
        }
        let Some(off) = self.ring.alloc(size) else {
            return Err(format!("ring full: {size:#x}-byte copy of {base:#x} (segment {:#x})", self.ring.segment_size()));
        };
        let t0 = std::time::Instant::now();
        self.ring.write_guest(off, base, size);
        self.prof[2] += t0.elapsed();
        let regions: Vec<(vk::Buffer, vk::BufferCopy)> = inflight
            .iter()
            .copied()
            .chain(self.writebacks.iter().map(|p| (p.base, p.dev_buf, p.dev_off, p.size)))
            .filter_map(|(pbase, buf, dev_off, psize)| {
                let (lo, hi) = (pbase.max(base), (pbase + psize).min(base + size));
                (lo < hi).then(|| (buf, vk::BufferCopy { src_offset: dev_off + (lo - pbase), dst_offset: off + (lo - base), size: hi - lo }))
            })
            .collect();
        self.upload_forwarded = !regions.is_empty();
        if perf_on() && !regions.is_empty() {
            let mut g = FORWARD_SITES.lock();
            let m = g.get_or_insert_with(HashMap::new);
            let over = |w: u64, n: u64| w < base + size && base < w + n;
            let shader = self.cur_shader;
            for &(pbase, _, _, psize) in inflight.iter().filter(|e| over(e.0, e.3)) {
                *m.entry(format!("{base:#x}+{size:#x} read by {shader:#x}: {pbase:#x}+{psize:#x} written in a batch in flight")).or_default() += 1;
            }
            for p in self.writebacks.iter().filter(|p| over(p.base, p.size)) {
                *m.entry(format!("{base:#x}+{size:#x} read by {shader:#x}: {:#x}+{:#x} written in this batch ({})", p.base, p.size, if p.mirror { "mirror" } else { "device-ring copy" })).or_default() += 1;
            }
        }
        if !regions.is_empty() {
            // Recorded by `flush_forwards` before the copy is used: the
            // forwards of all of a draw's bindings under one barrier.
            self.flushes[5] += 1;
            self.fwd_pending.extend(regions);
        }
        Ok(off)
    }

    /// `upload_guest` for data shaders only read: copies of up to
    /// `BAR_UPLOAD_MAX` bytes that need no forwards (no GPU write to the range
    /// pending in this batch or in flight) go to the device-local BAR ring
    /// while its segment has room, so shaders read them from VRAM rather than
    /// over PCIe. Returns the ring buffer and offset.
    fn upload_guest_ro(&mut self, base: u64, size: u64) -> Result<(vk::Buffer, u64), String> {
        if size <= BAR_UPLOAD_MAX && crate::tune::get(crate::tune::BAR_UPLOADS) != 0 && self.bar.is_some() && !self.writebacks.iter().any(|p| p.base < base + size && base < p.base + p.size) && !writeback_in_flight(base, size) {
            let r = self.bar.as_mut().unwrap();
            match r.alloc(size) {
                Some(off) => {
                    let t0 = std::time::Instant::now();
                    r.write_guest(off, base, size);
                    let buf = r.buffer;
                    self.prof[2] += t0.elapsed();
                    self.upload_forwarded = false;
                    self.bar_uploads.0 += 1;
                    self.bar_uploads.1 += size;
                    return Ok((buf, off));
                }
                None => self.bar_uploads.2 += 1,
            }
        }
        Ok((self.ring.buffer, self.upload_guest(base, size)?))
    }

    /// The ring offset of this batch's copy of [base, base+size) when a
    /// read-only binding can use it: copied in the current CPU epoch (within
    /// one, guest memory changes only through GPU and CP writes), with no
    /// write registered in the range since (GPU writes, write-backs, harvested
    /// CPU writes; CP writes drop entries in `forget_uploads`), by the same
    /// CP queue (another queue's CP may write the range between its copy and
    /// this command without either ordering against the other). Copies that
    /// needed forwards are not kept (their GPU copies belong to one command).
    fn reused_upload(&mut self, base: u64, size: u64) -> Option<(vk::Buffer, u64)> {
        self.drain_cp_writes();
        if self.upload_reuse_epoch != self.cpu_epoch {
            self.upload_reuse.clear();
            self.upload_reuse_epoch = self.cpu_epoch;
            return None;
        }
        let &(bytes, buf, off, stamp, queue) = self.upload_reuse.get(&base)?;
        if queue != self.cmd_queue {
            return None;
        }
        if bytes < size || self.writes.gpu_written_since(base, bytes, stamp) {
            self.upload_reuse.remove(&base);
            return None;
        }
        self.upload_reused.0 += 1;
        self.upload_reused.1 += size;
        Some((buf, off))
    }

    /// Remembers a read-only binding's fresh ring copy for `reused_upload`.
    fn note_upload(&mut self, base: u64, size: u64, buf: vk::Buffer, off: u64) {
        if self.upload_forwarded || crate::tune::get(crate::tune::UPLOAD_REUSE) == 0 {
            return;
        }
        if self.upload_reuse_epoch != self.cpu_epoch {
            self.upload_reuse.clear();
            self.upload_reuse_epoch = self.cpu_epoch;
        }
        self.upload_reuse.insert(base, (size, buf, off, self.writes.stamp(), self.cmd_queue));
    }

    /// The CP wrote guest [addr, addr+len) directly: copies of it are stale.
    fn forget_uploads(&mut self, addr: u64, len: u64) {
        let end = addr + len.max(1);
        self.upload_reuse.retain(|&b, &mut (n, _, _, _, _)| b >= end || b + n <= addr);
    }

    /// Records the forwards `upload_guest` queued: shader writes before the
    /// first copy; earlier forwards before a copy only where it overlaps
    /// them (later writes land later). Their ring ranges are fresh, used only
    /// by the command that asked for them, which comes after this.
    fn flush_forwards(&mut self) {
        if self.fwd_pending.is_empty() {
            return;
        }
        let regions = std::mem::take(&mut self.fwd_pending);
        self.begin();
        self.end_rendering();
        let c = ctx();
        let mut since: smallvec::SmallVec<[(u64, u64); 8]> = smallvec::SmallVec::new();
        for (buf, r) in &regions {
            let (a, b) = (r.dst_offset, r.dst_offset + r.size);
            if since.is_empty() || since.iter().any(|&(x, y)| a < y && x < b) {
                global_barrier(self.cmd);
                since.clear();
            }
            since.push((a, b));
            unsafe { c.device.cmd_copy_buffer(self.cmd, *buf, self.ring.buffer, std::slice::from_ref(r)) };
        }
        unordered!(self);
        self.mark(|| "pending-write forwards".into());
    }

    /// Submits the batch first if the ring segments may not fit `host` and
    /// `dev` more bytes (a draw/dispatch cannot be split across batches).
    fn reserve(&mut self, host: u64, dev: u64) {
        let need_host = host + self.ring.segment_size() / 8;
        let need_dev = dev + self.dring.segment_size() / 8;
        if self.ring.remaining() < need_host || self.dring.remaining() < need_dev {
            self.flushes[2] += 1;
            self.submit_batch(Vec::new());
        }
    }

    /// Upper bound of ring bytes texture uploads of a draw/dispatch may need
    /// (images not produced by the GPU; reloads of stale render targets).
    fn texture_upload_estimate(&self, stages: &[&Prepared]) -> u64 {
        // (Not skippable when the ring has room: a large estimate submits
        // the batch first, which also leaves the next batch's device ring
        // free for the texture's GPU detiling -- without that, a 64 MiB
        // shadow map sampled from guest memory every frame fell back to the
        // CPU detiler, ~150 ms per frame.)
        let mut n = 0u64;
        for p in stages {
            for b in &p.translated.out.bindings {
                if let BindingKind::Image(i) | BindingKind::StorageImage(i) = b.kind {
                    if let ResolvedRes::Image(t) = &p.resources[i] {
                        // Images that are GPU-produced or already validated this epoch need no upload.
                        let ready = self.by_base.get(&t.base).is_some_and(|ks| {
                            ks.iter().any(|k| {
                                let img = &self.images[k];
                                (img.gpu_written && !img.guest_newer) || (!img.gpu_written && img.upload_batch == self.epoch)
                            })
                        });
                        if !ready {
                            let layers = if t.ty == desc::T_3D { t.depth } else { t.total_layers() }.max(1);
                            n += resources::texture_bytes(t, layers).min(64 << 20);
                        }
                    }
                }
            }
        }
        n
    }

    /// Ring bytes a draw/dispatch's buffer bindings need (host, device).
    fn binding_bytes(&self, stages: &[&Prepared]) -> (u64, u64) {
        let (mut host, mut dev) = (0u64, 0u64);
        for p in stages {
            for b in &p.translated.out.bindings {
                if let BindingKind::Buffer(i) = b.kind {
                    if let ResolvedRes::Buffer(v) = &p.resources[i] {
                        if tess_ring_offset(v.base).is_some() {
                            continue;
                        }
                        if self.in_place(p, i, v.base, v.size().max(4)) {
                            continue; // in place (or copied below if the import fails)
                        }
                        let size = v.size().clamp(4, 64 << 20).next_multiple_of(256);
                        host += size;
                        if buffer_written(p, i) {
                            dev += size;
                        }
                    }
                }
            }
        }
        (host, dev)
    }

    /// Guest surface of a render target: (bits per element, tile mode, padded
    /// height, total bytes), if its layout is known.
    fn image_guest_layout(key: &ImageKey, img: &Image) -> Option<(u32, tiling::TileMode, u32, u64)> {
        let idx = img.tile_index?;
        let bpp = match key.format {
            vk::Format::D32_SFLOAT | vk::Format::D32_SFLOAT_S8_UINT => 32,
            vk::Format::D16_UNORM => 16,
            f => match format_class(f) {
                c @ (1 | 2 | 4 | 8 | 16) => c * 8,
                _ => return None,
            },
        };
        let mode = tiling::tile_mode(idx);
        let (pitch, ph) = tiling::padded(img.pitch.max(key.width), key.height, bpp, mode);
        Some((bpp, mode, ph, tiling::slice_bytes(pitch, ph, bpp) * key.layers as u64))
    }

    /// Writes render targets whose GPU contents are newer than guest memory
    /// and overlap [base, base+size) back to guest memory in their tiled
    /// layout, so guest-memory readers (buffer loads, CPU-sourced textures)
    /// see them. Synchronous; must not run while a draw is being prepared.
    fn writeback_images(&mut self, base: u64, size: u64) {
        // Only targets rendered in the last few frames: older ones are dead
        // (their memory may have been reused by the title; writing their
        // contents back would corrupt it).
        // Only images in the stale index can be newer than guest memory.
        let frame = super::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
        if self.stale_aged != frame {
            self.stale_aged = frame;
            let mut stale = std::mem::take(&mut self.stale);
            for ks in stale.values_mut() {
                ks.retain(|k| match self.images.get_mut(k) {
                    Some(i) => {
                        if i.guest_stale && i.last_write_frame + 2 < frame {
                            i.guest_stale = false;
                        }
                        i.guest_stale
                    }
                    None => false,
                });
            }
            stale.retain(|_, ks| !ks.is_empty());
            self.stale = stale;
        }
        let mut keys: Vec<ImageKey> = Vec::new();
        for ks in self.stale.range(base.saturating_sub(self.stale_max)..base + size).map(|(_, ks)| ks) {
            for &k in ks {
                let Some(i) = self.images.get(&k) else { continue };
                if i.guest_stale && !i.guest_newer && !k.ty3d && Self::image_guest_layout(&k, i).is_some_and(|g| base < k.base + g.3) {
                    keys.push(k);
                }
            }
        }
        for key in keys {
            let Some((bpp, mode, ph, total)) = Self::image_guest_layout(&key, &self.images[&key]) else { continue };
            // The title wrote this memory since the image was rendered: it
            // freed the surface and reused the memory (e.g. heap blocks), so
            // the contents are dead. (Our own CPU-path write-backs complete
            // before the image can become stale again, whose baseline
            // harvest then consumes them.)
            {
                let since = self.images[&key].stale_stamp;
                if self.writes.cpu_written_since(key.base, total, since).0 {
                    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 32 {
                        crate::info!("write-back of {:#x} {}x{} {:?} skipped: the CPU wrote its memory since it was rendered (reused)", key.base, key.width, key.height, key.format);
                    }
                    self.reused_skips += 1;
                    // Guest memory is now the newer copy: reloaded into the
                    // image should the title use a surface here again.
                    if let Some(i) = self.images.get_mut(&key) {
                        i.guest_stale = false;
                        i.guest_newer = true;
                    }
                    continue;
                }
            }
            let lin_slice = key.width as u64 * key.height as u64 * (bpp / 8) as u64;
            // GPU path: copy the image to the device ring and tile it in place
            // into guest memory (ordered with later GPU readers; CPU readers
            // wait for the batch like any in-place write).
            let lin_bytes = lin_slice * key.layers.max(1) as u64;
            // Device ring: the linear copy of the image and the tiled surface.
            let need = lin_bytes + total + 8192;
            // (On the GPU: into the imported guest memory, or into the VRAM
            // mirror, see `gpu_remap`.)
            let mirrored = ctx().ext_host.is_none();
            let gpu_tiling = Self::gpu_remap_eligible(bpp) && (!mirrored || (mirror_tiling_on() && total <= VCACHE_MAX));
            if gpu_tiling && self.dring.remaining() < need {
                self.flushes[2] += 1;
                self.submit_batch(Vec::new());
            }
            let gpu_off = (gpu_tiling && self.dring.remaining() >= need).then(|| self.dring.alloc(lin_bytes)).flatten();
            if let Some(lin_off) = gpu_off {
                self.begin();
                self.end_rendering();
                global_barrier(self.cmd);
                let region = vk::BufferImageCopy {
                    buffer_offset: lin_off,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: if key.depth { vk::ImageAspectFlags::DEPTH } else { vk::ImageAspectFlags::COLOR },
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: key.layers,
                    },
                    image_offset: vk::Offset3D::default(),
                    image_extent: vk::Extent3D { width: key.width, height: key.height, depth: 1 },
                };
                unsafe { ctx().device.cmd_copy_image_to_buffer(self.cmd, self.images[&key].image, vk::ImageLayout::GENERAL, self.dring.buffer, &[region]) };
                match self.gpu_remap(&key, true, lin_off) {
                    Ok(true) => {
                        // (Mirror tiling registered its write-back itself.)
                        if !mirrored {
                            self.push_zc_write(key.base, total);
                            self.writes.note_gpu_write(key.base, total);
                            self.note_pending_gpu_write(key.base, total);
                        }
                        self.images.get_mut(&key).unwrap().guest_stale = false;
                        self.epoch += 1;
                        self.gpu_remaps[1] += 1;
                        continue;
                    }
                    Ok(false) => {}
                    Err(e) => self.report(format!("GPU render target tiling failed: {e}")),
                }
            }
            if self.ring.remaining() < lin_slice * key.layers as u64 + 4096 {
                self.submit_batch(Vec::new());
            }
            let Some(off) = self.ring.alloc(lin_slice * key.layers as u64) else {
                self.report(format!("render target write-back {:#x} too large for the ring", key.base));
                continue;
            };
            self.begin();
            self.end_rendering();
            global_barrier(self.cmd);
            let region = vk::BufferImageCopy {
                buffer_offset: off,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: if key.depth { vk::ImageAspectFlags::DEPTH } else { vk::ImageAspectFlags::COLOR },
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: key.layers,
                },
                image_offset: vk::Offset3D::default(),
                image_extent: vk::Extent3D { width: key.width, height: key.height, depth: 1 },
            };
            unsafe { ctx().device.cmd_copy_image_to_buffer(self.cmd, self.images[&key].image, vk::ImageLayout::GENERAL, self.ring.buffer, &[region]) };
            self.flush_for(1);
            let t0 = std::time::Instant::now();
            let lin = unsafe { std::slice::from_raw_parts(self.ring.ptr.add(off as usize), (lin_slice * key.layers as u64) as usize) };
            let slice = total / key.layers as u64;
            let mut tiled = vec![0u8; total as usize];
            resources::copy_from_guest(&mut tiled, key.base);
            for l in 0..key.layers as u64 {
                let src = &lin[(l * lin_slice) as usize..((l + 1) * lin_slice) as usize];
                let dst = &mut tiled[(l * slice) as usize..((l + 1) * slice) as usize];
                tiling::tile_slice(src, dst, key.width, key.height, key.width, ph, bpp, mode);
            }
            resources::copy_to_guest(&tiled, key.base);
            self.writes.note_gpu_write(key.base, total);
            // (A new baseline: these CPU writes are ours, not the title's.)
            let (_, stamp) = self.writes.cpu_written_since(key.base, total, u64::MAX);
            if let Some(i) = self.images.get_mut(&key) {
                i.stale_stamp = stamp;
            }
            self.prof[1] += t0.elapsed();
            self.images.get_mut(&key).unwrap().guest_stale = false;
            self.epoch += 1;
            if crate::log::enabled(crate::log::Level::Debug) {
                crate::debug!("render target {:#x} {}x{} {:?} written back to guest memory", key.base, key.width, key.height, key.format);
            }
        }
    }

    /// Before a draw/dispatch allocates descriptors: makes guest memory read
    /// by its buffer bindings current with render targets, and lands pending
    /// buffer writes that its CPU-sourced textures may read (both may submit
    /// and wait, which is not allowed once preparation has started).
    fn prepare_guest_reads(&mut self, stages: &[&Prepared]) {
        self.zc_revalidate();
        // Buffers first: their render target write-backs may write guest
        // memory that CPU-sourced textures read.
        for pass in 0..2 {
            for p in stages {
                for b in &p.translated.out.bindings {
                    match b.kind {
                        BindingKind::Buffer(i) if pass == 0 => {
                            let ResolvedRes::Buffer(v) = &p.resources[i] else { continue };
                            if tess_ring_offset(v.base).is_some() {
                                continue;
                            }
                            let full = v.size().max(4);
                            if self.in_place(p, i, v.base, full) {
                                // Accessed in place at execution time: device-ring
                                // writes must have reached guest memory.
                                let range = full.min(ctx().limits.max_storage_buffer_range as u64);
                                if self.writebacks.iter().any(|w| w.base < v.base + range && v.base < w.base + w.size) {
                                    self.flush_for(1);
                                } else if cpu_writeback_in_flight(v.base, range) {
                                    wait_cpu_writebacks(v.base, range);
                                }
                                if p.read_buffers.contains(&i) && range <= SURFACE_WRITE_MAX {
                                    self.writeback_images(v.base, range);
                                }
                            } else {
                                let size = full.min(64 << 20);
                                if p.read_buffers.contains(&i) && v.size() <= SURFACE_WRITE_MAX {
                                    self.writeback_images(v.base, size);
                                }
                                // Snapshot at record time: in-place writes of this
                                // batch (render target write-backs included) must
                                // land first. (Written buffers copied on the GPU
                                // timeline are ordered after them.)
                                let gpu_copy = buffer_written(p, i) && self.gpu_roundtrip_ok(v.base & !15, size + (v.base & 15));
                                if !gpu_copy && self.zc_overlaps(v.base, size) {
                                    self.flush_for(1);
                                }
                                // VRAM mirror copy-ins (see the binding) read
                                // guest memory at record time: device-ring
                                // write-backs of this batch must land first,
                                // unless the copy-in can forward them.
                                // (Mirror-resident writes keep the mirror current.)
                                if ctx().ext_host.is_none()
                                    && size >= VCACHE_MIN
                                    && self.writebacks.iter().any(|w| !w.mirror && w.base < v.base + size && v.base < w.base + w.size)
                                    && self.mirror_forwards(&[(v.base, v.base + size)], vk::Buffer::null()).is_none()
                                {
                                    self.flush_for(1);
                                }
                            }
                        }
                        BindingKind::Image(i) | BindingKind::StorageImage(i) if pass == 1 => {
                            if let ResolvedRes::Image(t) = &p.resources[i] {
                                let layers = if t.ty == desc::T_3D { t.depth } else { t.total_layers() }.max(1);
                                let bytes = resources::texture_bytes(t, layers);
                                // Only guest-memory reads need this batch's
                                // writes landed first (`sync_guest` checks them
                                // again); a GPU-written image here is read on the
                                // GPU, reloads included (ordered by barriers).
                                let gpu_image = self.by_base.get(&t.base).is_some_and(|ks| ks.iter().any(|k| self.images[k].gpu_written));
                                if !gpu_image && self.pending_write_overlaps(t.base, bytes) {
                                    // Validate it now, while a flush is still allowed:
                                    // GPU-detiled textures follow this batch's in-place
                                    // writes in GPU order (no wait); the CPU path
                                    // flushes here (`sync_guest`) instead of later.
                                    let storage = matches!(b.kind, BindingKind::StorageImage(_));
                                    if crate::tune::get(crate::tune::EARLY_TEX) == 0 {
                                        self.flush_for(1);
                                    } else if let Err(e) = self.texture(t, storage, p.load_only.get(i).copied().unwrap_or(false)) {
                                        if e.contains("ring full") {
                                            // (Validated again in a fresh batch.)
                                            self.flushes[2] += 1;
                                            self.submit_batch(Vec::new());
                                            self.begin();
                                        } else {
                                            self.report(format!("early texture validation failed: {e}"));
                                            self.flush_for(1);
                                        }
                                    }
                                }
                                // Render targets here made stale by buffer stores.
                                let newer: Vec<ImageKey> = self.by_base.get(&t.base).map(|ks| ks.iter().copied().filter(|k| self.images[k].guest_newer).collect()).unwrap_or_default();
                                for k in newer {
                                    if let Err(e) = self.refresh_from_guest(k, true) {
                                        self.report(format!("render target reload failed: {e}"));
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------
    // Descriptors

    fn write_descriptors(&mut self, set: vk::DescriptorSet, prepared: &[&Prepared]) -> Result<(), String> {
        let c = ctx();
        enum Info {
            Buf(vk::DescriptorBufferInfo),
            Img(vk::DescriptorImageInfo),
        }
        let mut infos: smallvec::SmallVec<[(u32, vk::DescriptorType, Info); 24]> = smallvec::SmallVec::new();
        // (TLG_PERF timing: each binding's time is added to its kind when the
        // next one starts.)
        let mut tb = unsafe { core::arch::x86_64::_rdtsc() };
        let mut part = 4usize;
        let mut bpath = 0usize;
        for p in prepared {
            // Per resource: page bitmap of its writes (`BindingKind::DirtyInfo`);
            // writes not tracked mark the sink.
            let mut dinfo: Vec<u32> = Vec::new();
            if p.translated.out.bindings.iter().any(|b| b.kind == BindingKind::DirtyInfo) {
                let sink = (self.dirty_sink()? / 4) as u32;
                dinfo = (0..p.resources.len()).flat_map(|_| [sink, 0, 0]).collect();
            }
            for b in &p.translated.out.bindings {
                let t = unsafe { core::arch::x86_64::_rdtsc() };
                self.dsub[part] += t.wrapping_sub(tb);
                if part == 2 {
                    self.bsub[bpath] += t.wrapping_sub(tb);
                    self.bcnt[bpath] += 1;
                }
                tb = t;
                part = match b.kind {
                    BindingKind::SgprData => 1,
                    BindingKind::Buffer(_) => 2,
                    BindingKind::Image(_) | BindingKind::StorageImage(_) => 3,
                    _ => 4,
                };
                match b.kind {
                    BindingKind::SgprData => {
                        // (Little-endian host: the dwords' bytes as they are.)
                        let bytes = unsafe { std::slice::from_raw_parts(p.sgpr_data.as_ptr() as *const u8, p.sgpr_data.len() * 4) };
                        // (Device-local through the BAR window when there is
                        // room in this batch's segment, else host memory.)
                        let bar = self.bar.as_mut().filter(|_| crate::tune::get(crate::tune::BAR_SGPR) != 0).and_then(|r| r.alloc(bytes.len() as u64).map(|off| (r, off)));
                        let (buffer, off) = match bar {
                            Some((r, off)) => {
                                r.write(off, bytes);
                                (r.buffer, off)
                            }
                            None => {
                                let off = self.ring.alloc(bytes.len() as u64).ok_or("ring full: sgpr data")?;
                                self.ring.write(off, bytes);
                                (self.ring.buffer, off)
                            }
                        };
                        if p.stage == Stage::Vertex {
                            self.vs_sgpr = Some((buffer, off, bytes.len() as u64));
                        }
                        infos.push((b.binding, vk::DescriptorType::STORAGE_BUFFER, Info::Buf(vk::DescriptorBufferInfo { buffer, offset: off, range: bytes.len() as u64 })));
                    }
                    BindingKind::Buffer(i) => {
                        let ResolvedRes::Buffer(v) = &p.resources[i] else { return Err("binding/resource mismatch".into()) };
                        // (This binding is the resource's: `buffer_written` reduced.)
                        let written = p.written_buffers.contains(&i);
                        bpath = 0;
                        if let Some(off) = tess_ring_offset(v.base) {
                            // The tessellation factor ring (HS stores).
                            if off % 256 != 0 {
                                return Err(format!("tessellation factor buffer at ring offset {off:#x} misaligned"));
                            }
                            let (_, tf) = self.tess_buffers()?;
                            let range = v.size().max(4).min(TESS_FACTOR_RING_BYTES - off).next_multiple_of(4);
                            infos.push((b.binding, vk::DescriptorType::STORAGE_BUFFER, Info::Buf(vk::DescriptorBufferInfo { buffer: tf, offset: off, range })));
                            continue;
                        }
                        // Read-only buffers are bound at the 16-byte-aligned
                        // address; shaders add the bias (base & 15).
                        let bias = if written { 0 } else { v.base & 15 };
                        let base = v.base - bias;
                        if self.in_place(p, i, v.base, v.size().max(4)) {
                            if let Some((buf, off, range)) = self.zc_binding(base, v.size().max(4) + bias) {
                                bpath = 1;
                                if written {
                                    bpath = 2;
                                    // The scatter kernel's records name the
                                    // few dwords it writes in its 2 GB binding.
                                    let ranges = match (&p.sparse_writes, p.scatter) {
                                        (Some(rs), Some((_, d))) if d == i => rs.clone(),
                                        _ => vec![(base, range)],
                                    };
                                    for (b, n) in ranges {
                                        self.push_zc_write(b, n);
                                        self.writes.note_gpu_write(b, n);
                                        self.note_pending_gpu_write(b, n);
                                        // Other huge bindings are sparse writers
                                        // too: not a surface write.
                                        if n <= SURFACE_WRITE_MAX {
                                            self.note_meta_write(b, n);
                                            self.note_buffer_write(b, n);
                                        }
                                    }
                                }
                                if !written && range <= VCACHE_MAX {
                                    if let Some((vb, voff)) = self.vcache_bind(base, range) {
                                        bpath = 3;
                                        infos.push((b.binding, vk::DescriptorType::STORAGE_BUFFER, Info::Buf(vk::DescriptorBufferInfo { buffer: vb, offset: voff, range })));
                                        continue;
                                    }
                                    self.zc_read(base, base + range);
                                }
                                infos.push((b.binding, vk::DescriptorType::STORAGE_BUFFER, Info::Buf(vk::DescriptorBufferInfo { buffer: buf, offset: off, range })));
                                continue;
                            }
                        }
                        let extent = guest_extent(base, v.size() + bias);
                        let read_min = (crate::tune::get(crate::tune::VCACHE_READ_MIN_KIB).max(0) as u64) << 10;
                        // (Small ranges with GPU writes not in guest memory yet:
                        // from the mirror, which holds or forwards them.)
                        let gpu_pending = || self.writebacks.iter().any(|w| w.base < base + extent && base < w.base + w.size) || writeback_in_flight(base, extent);
                        if !written && ctx().ext_host.is_none() && (extent >= read_min || (crate::tune::get(crate::tune::VCACHE_GPU_READS) != 0 && gpu_pending())) {
                            // Large read-only data: from the VRAM mirror
                            // (copied in once, then per CPU-written page).
                            // Unbounded V#s (pools of hundreds of MiB) are
                            // bound to the end of their base's window: the
                            // 64 MiB the upload path would take did not fit a
                            // ring segment, and their dispatches were skipped.
                            let room = (base >> VWIN_SHIFT << VWIN_SHIFT) + VWIN_SIZE - base;
                            let n = extent.min(room);
                            self.vc_why = "read binding";
                            if let Some((vb, voff)) = self.vcache_bind_ex(base, n, false, n < VCACHE_MIN) {
                                bpath = 3;
                                infos.push((b.binding, vk::DescriptorType::STORAGE_BUFFER, Info::Buf(vk::DescriptorBufferInfo { buffer: vb, offset: voff, range: n.next_multiple_of(4) })));
                                continue;
                            }
                        }
                        let write_min = ((crate::tune::get(crate::tune::VCACHE_WRITE_MIN_KIB).max(0) as u64) << 10).max(4);
                        if written && ctx().ext_host.is_none() && (write_min..=VCACHE_MAX).contains(&extent) && base % 16 == 0 && p.scatter.is_none_or(|(_, d)| d != i) {
                            // Large written data: the shader works on the VRAM
                            // mirror itself, which stays current for later
                            // commands; guest memory gets the range when the
                            // batch completes (copied out at its end).
                            self.vc_why = "written binding";
                            if let Some((vb, voff)) = self.vcache_bind_ex(base, extent, false, extent < VCACHE_MIN) {
                                bpath = 4;
                                let (bits, words, head) = self.mirror_write_back(base, extent, vb, voff)?;
                                if let Some(d) = dinfo.get_mut(3 * i..3 * i + 3) {
                                    d.copy_from_slice(&[(bits / 4) as u32, head, words - 1]);
                                }
                                if extent <= SURFACE_WRITE_MAX {
                                    self.note_meta_write(base, extent);
                                    self.note_buffer_write(base, extent);
                                }
                                self.writes.note_gpu_write(base, extent);
                                self.note_mirror_write(vb, base, extent);
                                self.end_rendering();
                                // (A dispatch's own write: tracked as its access.)
                                if p.stage == Stage::Compute {
                                    self.barrier_needed = true;
                                } else {
                                    unordered!(self);
                                }
                                infos.push((b.binding, vk::DescriptorType::STORAGE_BUFFER, Info::Buf(vk::DescriptorBufferInfo { buffer: vb, offset: voff, range: extent.next_multiple_of(4) })));
                                continue;
                            }
                        }
                        let size = extent.clamp(4, 64 << 20);
                        if !written && crate::tune::get(crate::tune::UPLOAD_REUSE) != 0 {
                            if let Some((buf, off)) = self.reused_upload(base, size) {
                                bpath = 5;
                                infos.push((b.binding, vk::DescriptorType::STORAGE_BUFFER, Info::Buf(vk::DescriptorBufferInfo { buffer: buf, offset: off, range: size.next_multiple_of(4) })));
                                continue;
                            }
                        }
                        let cause = if written { 0 } else if bias != 0 { 1 } else { 2 };
                        self.upload_bytes[cause] += size;
                        if perf_on() {
                            let mut g = BUFFER_UPLOAD_SITES.lock();
                            let m = g.get_or_insert_with(HashMap::new);
                            if cause == 0 && size > 1 << 20 && !m.contains_key(&(base, size, true)) {
                                crate::info!("written buffer binding {base:#x}+{size:#x}: V# base {:#x} stride {} num_records {:#x} ({:#x} bytes) by {:?} shader {:#x}", v.base, v.stride, v.num_records, v.size(), p.stage, p.addr);
                            }
                            *m.entry((base, size, cause == 0)).or_default() += 1;
                        }
                        if written && v.size() <= SURFACE_WRITE_MAX {
                            self.note_meta_write(base, size);
                            self.note_buffer_write(base, size);
                        }
                        if written {
                            if let Some((zbuf, zoff, _)) = self.zc_binding(base, size).filter(|z| z.2 >= size) {
                                // Guest memory -> device ring (the shader's
                                // working copy) now and back right after the
                                // command, both as bulk copies on the GPU
                                // timeline (shader stores into host memory crawl).
                                bpath = 7;
                                let dev_off = self.dring.alloc(size).ok_or("device ring full")?;
                                self.end_rendering();
                                global_barrier(self.cmd);
                                unsafe { ctx().device.cmd_copy_buffer(self.cmd, zbuf, self.dring.buffer, &[vk::BufferCopy { src_offset: zoff, dst_offset: dev_off, size }]) };
                                unordered!(self);
                                self.gpu_wb.push((dev_off, zbuf, zoff, size));
                                self.push_zc_write(base, size);
                                self.writes.note_gpu_write(base, size);
                                self.note_pending_gpu_write(base, size);
                                self.mark(|| "written buffer setup copy".into());
                                infos.push((b.binding, vk::DescriptorType::STORAGE_BUFFER, Info::Buf(vk::DescriptorBufferInfo { buffer: self.dring.buffer, offset: dev_off, range: size.next_multiple_of(4) })));
                                continue;
                            }
                        }
                        if !written {
                            bpath = 6;
                            let (buf, off) = self.upload_guest_ro(base, size)?;
                            self.note_upload(base, size, buf, off);
                            infos.push((b.binding, vk::DescriptorType::STORAGE_BUFFER, Info::Buf(vk::DescriptorBufferInfo { buffer: buf, offset: off, range: size.next_multiple_of(4) })));
                            continue;
                        }
                        // Written: the shader works on a device-local copy;
                        // written back to guest memory when the batch
                        // completes. (A barrier only after GPU forwards into
                        // the ring copy: host writes are visible at
                        // submission, the device range is fresh.)
                        bpath = 8;
                        let off = self.upload_guest(base, size)?;
                        let forwarded = self.upload_forwarded;
                        self.flush_forwards();
                        let dev_off = self.dring.alloc(size).ok_or("device ring full")?;
                        let region = vk::BufferCopy { src_offset: off, dst_offset: dev_off, size };
                        if !forwarded && crate::tune::get(crate::tune::PROLOGUE) != 0 {
                            // (The guest data as copied now, into a range
                            // nothing used yet: ahead of the batch.)
                            let pro = self.prologue_cmd();
                            unsafe { ctx().device.cmd_copy_buffer(pro, self.ring.buffer, self.dring.buffer, &[region]) };
                        } else {
                            self.end_rendering();
                            if forwarded {
                                global_barrier(self.cmd);
                            }
                            unsafe { ctx().device.cmd_copy_buffer(self.cmd, self.ring.buffer, self.dring.buffer, &[region]) };
                            unordered!(self);
                        }
                        self.push_writeback(PendingWrite { base, size, host_off: off, dev_off, dev_buf: self.dring.buffer, mirror: false, bits: None });
                        self.writes.note_gpu_write(base, size);
                        self.note_pending_gpu_write(base, size);
                        self.mark(|| "written buffer setup copy".into());
                        infos.push((b.binding, vk::DescriptorType::STORAGE_BUFFER, Info::Buf(vk::DescriptorBufferInfo { buffer: self.dring.buffer, offset: dev_off, range: size.next_multiple_of(4) })));
                    }
                    BindingKind::Image(i) | BindingKind::StorageImage(i) => {
                        let ResolvedRes::Image(t) = &p.resources[i] else { return Err("binding/resource mismatch".into()) };
                        let storage = matches!(b.kind, BindingKind::StorageImage(_));
                        let view = self.texture(t, storage, p.load_only.get(i).copied().unwrap_or(false))?;
                        infos.push((
                            b.binding,
                            if storage { vk::DescriptorType::STORAGE_IMAGE } else { vk::DescriptorType::SAMPLED_IMAGE },
                            Info::Img(vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: view, image_layout: vk::ImageLayout::GENERAL }),
                        ));
                    }
                    BindingKind::Gds => {
                        infos.push((b.binding, vk::DescriptorType::STORAGE_BUFFER, Info::Buf(vk::DescriptorBufferInfo { buffer: self.gds, offset: 0, range: gcn::translate::GDS_BYTES as u64 })));
                    }
                    BindingKind::DirtyInfo => {
                        // (After every buffer binding of this stage: they
                        // filled `dinfo`.)
                        let bytes = unsafe { std::slice::from_raw_parts(dinfo.as_ptr() as *const u8, dinfo.len() * 4) };
                        let off = self.ring.alloc(bytes.len().max(4) as u64).ok_or("ring full: page bitmap info")?;
                        self.ring.write(off, bytes);
                        infos.push((b.binding, vk::DescriptorType::STORAGE_BUFFER, Info::Buf(vk::DescriptorBufferInfo { buffer: self.ring.buffer, offset: off, range: bytes.len().max(4) as u64 })));
                    }
                    BindingKind::DirtyBits => {
                        infos.push((b.binding, vk::DescriptorType::STORAGE_BUFFER, Info::Buf(vk::DescriptorBufferInfo { buffer: self.dring.buffer, offset: 0, range: vk::WHOLE_SIZE })));
                    }
                    BindingKind::DebugOut => {
                        let (buf, _) = self.debug_out_buffer()?;
                        infos.push((b.binding, vk::DescriptorType::STORAGE_BUFFER, Info::Buf(vk::DescriptorBufferInfo { buffer: buf, offset: 0, range: DEBUG_OUT_BYTES })));
                    }
                    BindingKind::Lds => {
                        let (lds, _) = self.tess_buffers()?;
                        infos.push((b.binding, vk::DescriptorType::STORAGE_BUFFER, Info::Buf(vk::DescriptorBufferInfo { buffer: lds, offset: 0, range: TESS_LDS_BYTES })));
                    }
                    BindingKind::Sampler(i) => {
                        let ResolvedRes::Sampler(s) = &p.resources[i] else { return Err("binding/resource mismatch".into()) };
                        let smp = self.samplers.get(s)?;
                        infos.push((b.binding, vk::DescriptorType::SAMPLER, Info::Img(vk::DescriptorImageInfo { sampler: smp, image_view: vk::ImageView::null(), image_layout: vk::ImageLayout::UNDEFINED })));
                    }
                }
            }
        }
        let t = unsafe { core::arch::x86_64::_rdtsc() };
        self.dsub[part] += t.wrapping_sub(tb);
        if part == 2 {
            self.bsub[bpath] += t.wrapping_sub(tb);
            self.bcnt[bpath] += 1;
        }
        tb = t;
        if prepared.iter().any(|p| p.tess == TessRole::Ds) {
            let (_, tf) = self.tess_buffers()?;
            infos.push((TESS_TF_BINDING, vk::DescriptorType::STORAGE_BUFFER, Info::Buf(vk::DescriptorBufferInfo { buffer: tf, offset: 0, range: TESS_FACTOR_RING_BYTES })));
        }
        // A dispatch's buffer accesses (`tune dispatch_hazards`): bound
        // ranges with the shader's write flag. Images are tracked only as
        // sampled reads (their writers order themselves: render passes,
        // copies, uploads); storage images, GDS and the rest are unknown.
        if let [p] = prepared {
            if p.stage == Stage::Compute {
                self.cur_access.clear();
                self.cur_unknown = false;
                for (binding, _, info) in &infos {
                    let kind = p.translated.out.bindings.iter().find(|b| b.binding == *binding).map(|b| b.kind);
                    match (kind, info) {
                        (Some(BindingKind::Buffer(i)), Info::Buf(bi)) => {
                            let n = if bi.range == vk::WHOLE_SIZE { u64::MAX - bi.offset } else { bi.range };
                            self.cur_access.push((bi.buffer, bi.offset, n, buffer_written(p, i)));
                        }
                        (Some(BindingKind::Image(_) | BindingKind::Sampler(_) | BindingKind::SgprData | BindingKind::DirtyInfo | BindingKind::DirtyBits), _) => {}
                        _ => self.cur_unknown = true,
                    }
                }
            }
        }
        let writes: smallvec::SmallVec<[vk::WriteDescriptorSet; 24]> = infos
            .iter()
            .map(|(binding, ty, info)| {
                let w = vk::WriteDescriptorSet::default().dst_set(set).dst_binding(*binding).descriptor_type(*ty);
                match info {
                    Info::Buf(b) => w.buffer_info(std::slice::from_ref(b)),
                    Info::Img(i) => w.image_info(std::slice::from_ref(i)),
                }
            })
            .collect();
        let t = unsafe { core::arch::x86_64::_rdtsc() };
        self.dsub[4] += t.wrapping_sub(tb);
        unsafe { c.device.update_descriptor_sets(&writes, &[]) };
        self.dsub[5] += unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(t);
        Ok(())
    }

    /// Allocates and writes a draw/dispatch's descriptor set. The ring
    /// reservation is an estimate made before guest-read preparation (which
    /// can reload render targets or turn in-place bindings into copies); when
    /// the ring fills up anyway, the batch is submitted and the descriptors
    /// are written again in the next one instead of dropping the command.
    /// Returns the set and the pending-write count before it was written.
    fn descriptors_with_retry(&mut self, set_layout: vk::DescriptorSetLayout, stages: &[&Prepared]) -> Result<(vk::DescriptorSet, usize), String> {
        let t0 = unsafe { core::arch::x86_64::_rdtsc() };
        let set = self.alloc_set(set_layout)?;
        self.dsub[0] += unsafe { core::arch::x86_64::_rdtsc() }.wrapping_sub(t0);
        let wb_before = self.writebacks.len() + self.zc_writes.len();
        match self.write_descriptors(set, stages) {
            Ok(()) => Ok((set, wb_before)),
            Err(e) if e.contains("ring full") => {
                self.flushes[2] += 1;
                self.ring_retries += 1;
                self.gpu_wb.clear();
                // (The failed attempt's ring copies are abandoned.)
                self.fwd_pending.clear();
                self.submit_batch(Vec::new());
                self.begin();
                let set = self.alloc_set(set_layout)?;
                let wb_before = self.writebacks.len() + self.zc_writes.len();
                self.desc_retry = true;
                let r = self.write_descriptors(set, stages);
                self.desc_retry = false;
                r.map_err(|e2| format!("{e2} (retried in a new batch after: {e}; ring {:#x} of {:#x} left)", self.ring.remaining(), self.ring.segment_size()))?;
                Ok((set, wb_before))
            }
            Err(e) => Err(e),
        }
    }

    fn alloc_set(&mut self, layout: vk::DescriptorSetLayout) -> Result<vk::DescriptorSet, String> {
        let layouts = [layout];
        let ai = vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.desc_pool).set_layouts(&layouts);
        // (One set into a local: ash's wrapper returns a Vec, a heap
        // allocation per draw.)
        let alloc = |ai: &vk::DescriptorSetAllocateInfo| {
            let d = &ctx().device;
            let mut set = vk::DescriptorSet::null();
            match unsafe { (d.fp_v1_0().allocate_descriptor_sets)(d.handle(), ai, &mut set) } {
                vk::Result::SUCCESS => Ok(set),
                e => Err(e),
            }
        };
        match alloc(&ai) {
            Ok(set) => Ok(set),
            Err(_) => {
                self.flushes[3] += 1;
                self.submit_batch(Vec::new());
                self.begin();
                // (The next batch allocates from its own pool.)
                let ai = vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.desc_pool).set_layouts(&layouts);
                alloc(&ai).map_err(|e| format!("descriptor alloc: {e}"))
            }
        }
    }

    // -------------------------------------------------------------------
    // Draw

    fn draw_inner(&mut self, cp: &CommandProcessor, kind: &'static str, count: u32, index_addr: u64) -> Result<(), String> {
        let pre = self.pre_draw.take();
        self.draw_targets.clear();
        self.cmd_queue = cp.queue;
        self.drain_cp_writes();
        let mut tl = unsafe { core::arch::x86_64::_rdtsc() };
        macro_rules! phase {
            ($i:expr) => {{
                let t = unsafe { core::arch::x86_64::_rdtsc() };
                self.dprof[$i] += t.wrapping_sub(tl);
                tl = t;
            }};
        }
        super::set_phase(super::phase::DRAW, ((cp.regs.sh(0x2C48) as u64) | ((cp.regs.sh(0x2C49) as u64) << 32)) << 8);
        if trace_frame() {
            self.flush(); // frame trace: time each command on its own
        }
        let regs = &cp.regs;
        if self.ring.remaining() < self.ring.segment_size() / 4 {
            self.flushes[2] += 1;
            self.submit_batch(Vec::new());
        }
        self.begin();
        // (Write-backs queued by a command that failed were never needed.)
        self.gpu_wb.clear();
        let prim = regs.uconfig[(0xC242 - 0xC000) as usize];
        // Patches: the LS and HS stages run first (as compute), the draw
        // itself with the domain shader (see `tess_stages`).
        let tess = if prim == 9 && crate::tune::get(crate::tune::TESS) != 0 {
            match self.tess_stages(cp, kind, count, index_addr) {
                Ok(t) => Some(t),
                Err(e) => {
                    if trace_frame() {
                        crate::info!("FRAME tessellated draw ({kind}, n={count}) skipped: {e}");
                    }
                    return Err(e);
                }
            }
        } else {
            None
        };
        if tess.as_ref().is_some_and(|t| t.patches == 0) {
            return Ok(());
        }
        if !matches!(prim, 1..=6 | 0x11) && tess.is_none() {
            if trace_frame() {
                let pgm = |lo: u32| ((regs.sh(lo) as u64) | ((regs.sh(lo + 1) as u64) << 32)) << 8;
                let cbs: Vec<String> = (0..8u32).filter(|n| (regs.ctx(0xA318 + n * 0xF + 4) >> 2) & 0x1F != 0).map(|n| format!("{:#x}", (regs.ctx(0xA318 + n * 0xF) as u64) << 8)).collect();
                crate::info!(
                    "FRAME skipped draw ({kind}, prim {prim:#x}, n={count}): ls {:#x} hs {:#x} es {:#x} gs {:#x} vs {:#x} ps {:#x}; VGT_SHADER_STAGES_EN {:#x} LS_HS_CONFIG {:#x} TF_PARAM {:#x}; rt [{}] z {:#x}",
                    pgm(0x2D48),
                    pgm(0x2D08),
                    pgm(0x2CC8),
                    pgm(0x2C88),
                    pgm(0x2C48),
                    pgm(0x2C08),
                    regs.ctx(0xA2D5),
                    regs.ctx(0xA2D6),
                    regs.ctx(0xA2DB),
                    cbs.join(", "),
                    (regs.ctx(0xA014) as u64) << 8
                );
                crate::info!(
                    "FRAME   tess state: VGT_HOS {:08x?} VGT 0xA2D0.. {:08x?} LS rsrc {:#x}/{:#x} HS rsrc {:#x}/{:#x} VS rsrc {:#x}/{:#x} index type {} instances {}",
                    (0xA286..=0xA28A).map(|r| regs.ctx(r)).collect::<Vec<_>>(),
                    (0xA2D0..=0xA2DF).map(|r| regs.ctx(r)).collect::<Vec<_>>(),
                    regs.sh(0x2D4A),
                    regs.sh(0x2D4B),
                    regs.sh(0x2D0A),
                    regs.sh(0x2D0B),
                    regs.sh(0x2C4A),
                    regs.sh(0x2C4B),
                    cp.index_type,
                    cp.num_instances
                );
                // Each stage's program into the run's shader dump (user data
                // at SPI_SHADER_USER_DATA_*_0, 4 registers after the address).
                for (stage, lo) in [("ls", 0x2D48), ("hs", 0x2D08), ("es", 0x2CC8), ("gs", 0x2C88), ("vs", 0x2C48), ("ps", 0x2C08)] {
                    let a = pgm(lo);
                    if a != 0 {
                        let user: Vec<u32> = (0..16).map(|k| regs.sh(lo + 4 + k)).collect();
                        super::shader_dump::dump(stage, a, &user);
                    }
                }
            }
            return Err(format!("primitive type {prim:#x} unsupported"));
        }
        self.set_required_epoch(false);
        phase!(0);
        let tp = std::time::Instant::now();
        let ps_addr = ((regs.sh(0x2C08) as u64) | ((regs.sh(0x2C09) as u64) << 32)) << 8;
        if ps_addr != 0 && SKIP_PS.lock().contains(&ps_addr) {
            self.skipped += 1;
            return Ok(());
        }
        let ds = tess.as_ref().map(|t| t.ds).unwrap_or_default();
        // (Prepared before the lock: not for tessellated draws, whose
        // vertex stage is the domain shader.)
        let (mut vs, mut ps) = match pre.filter(|_| tess.is_none()) {
            Some((v, p)) => (v?, p.transpose()?),
            None => (shaders::prepare_tess(&StageRegs { stage: Stage::Vertex, regs }, ds)?, if ps_addr != 0 { Some(shaders::prepare(&StageRegs { stage: Stage::Fragment, regs })?) } else { None }),
        };
        // Descriptors and scalar loads were read from guest memory just now:
        // pending GPU writes there must land first.
        self.prep_shader = ps.as_ref().map(|p| p.addr).unwrap_or(vs.addr);
        self.prep_vs = vs.addr;
        for _ in 0..3 {
            let reads: smallvec::SmallVec<[u64; 64]> = vs.mem_reads.iter().chain(ps.iter().flat_map(|p| p.mem_reads.iter())).copied().collect();
            if !self.guard_reads(&reads) {
                break;
            }
            self.begin();
            vs = shaders::prepare_tess(&StageRegs { stage: Stage::Vertex, regs }, ds)?;
            ps = if ps_addr != 0 { Some(shaders::prepare(&StageRegs { stage: Stage::Fragment, regs })?) } else { None };
        }
        self.prof[5] += tp.elapsed();
        phase!(1);
        self.cur_shader = ps.as_ref().map(|p| p.addr).unwrap_or(vs.addr);
        let rect = prim == 0x11;
        if vs.translated.out.pos_ignored != 0 {
            self.report(format!("VS {:#x}: position exports {:#b} ignored (PA_CL_VS_OUT_CNTL {:#x})", vs.addr, vs.translated.out.pos_ignored, regs.ctx(0xA207)));
        }
        if regs.ctx(0xA204) & 0x3F != 0 {
            // PA_CL_CLIP_CNTL.UCP_ENA_0..5: planes from PA_CL_UCP_*.
            self.report(format!("user clip planes {:#x} not implemented", regs.ctx(0xA204) & 0x3F));
        }
        // Attachments.
        let target_mask = regs.ctx(0xA08E);
        let color_mode = (regs.ctx(0xA202) >> 4) & 7;
        let mut colors = Colors::new();
        let mut blends = smallvec::SmallVec::<[u32; 8]>::new();
        let mut masks = smallvec::SmallVec::<[u32; 8]>::new();
        if color_mode != 0 {
            for n in 0..8u32 {
                let mask = (target_mask >> (4 * n)) & 0xF;
                let written = ps.as_ref().is_some_and(|p| p.translated.out.mrt_mask & (1 << n) != 0);
                let att = if mask != 0 && written { self.color_target(regs, n)? } else { None };
                if att.is_some() {
                    colors.resize(n as usize, None);
                    colors.push(att);
                    while blends.len() < n as usize {
                        blends.push(0);
                        masks.push(0);
                    }
                    blends.push(regs.ctx(0xA1E0 + n));
                    masks.push(mask);
                }
            }
        }
        let dctl = regs.ctx(0xA200);
        let depth_used = dctl & 0x7 != 0 || regs.ctx(0xA000) & 3 != 0;
        let depth = if depth_used { self.depth_target(regs)? } else { None };
        self.draw_targets = colors.iter().flatten().map(|a| a.key).chain(depth.map(|d| d.key)).collect();
        self.apply_meta_clears(regs, &colors, depth);
        if trace_frame() {
            let res: Vec<String> = std::iter::once(&vs).chain(ps.iter()).flat_map(|p| stage_res_summary(p)).collect();
            crate::info!(
                "FRAME draw {} n={} prim {:#x} vs {:#x} ps {:#x} rt {:?} depth {:?} ds [dctl {:#x} sctl {:#x} ref {:#x}/{:#x} rc {:#x} su {:#x} bias {}/{}/{}/{} dbfmt {:#x}] blend [{:#x} mask {:#x} cc {:#x}] vp [vte {:#x} x {}/{} y {}/{} z {}/{} clip {:#x} sc {:#x}/{:#x}] res [{}]",
                kind,
                count,
                prim,
                vs.addr,
                ps.as_ref().map(|p| p.addr).unwrap_or(0),
                colors.iter().map(|c| c.map(|a| format!("{:#x}:{}x{}:{:?}{}", a.key.base, a.key.width, a.key.height, a.view_format, if a.key.layers > 1 { format!(":layer {}/{}", a.layer, a.key.layers) } else { String::new() }))).collect::<Vec<_>>(),
                depth.map(|d| format!("{:#x} {}x{} {:?}", d.key.base, d.key.width, d.key.height, d.key.format)),
                regs.ctx(0xA200),
                regs.ctx(0xA10B),
                regs.ctx(0xA10C),
                regs.ctx(0xA10D),
                regs.ctx(0xA000),
                regs.ctx(0xA205),
                f32r(regs.ctx(0xA2DF)),
                f32r(regs.ctx(0xA2E0)),
                f32r(regs.ctx(0xA2E1)),
                f32r(regs.ctx(0xA2E3)),
                regs.ctx(0xA2DE),
                regs.ctx(0xA1E0),
                regs.ctx(0xA08E),
                regs.ctx(0xA202),
                regs.ctx(0xA206),
                f32r(regs.ctx(0xA10F)),
                f32r(regs.ctx(0xA110)),
                f32r(regs.ctx(0xA111)),
                f32r(regs.ctx(0xA112)),
                f32r(regs.ctx(0xA113)),
                f32r(regs.ctx(0xA114)),
                regs.ctx(0xA204),
                regs.ctx(0xA094),
                regs.ctx(0xA095),
                res.join(", ")
            );
        }
        if colors.iter().all(|c| c.is_none()) && depth.is_none() {
            if trace_frame() {
                let cb = |n: u32, o: u32| regs.ctx(0xA318 + n * 0xF + o);
                crate::info!(
                    "FRAME skipped draw: CB mode {} target mask {:#x} DB_RENDER_CONTROL {:#x} DB_RENDER_OVERRIDE {:#x} | CB0 base {:#x} info {:#x} attrib {:#x} cmask {:#x} fmask {:#x} clear {:#x}/{:#x} | CB1 base {:#x} info {:#x} | DB z {:#x} zinfo {:#x} htile {:#x}",
                    color_mode,
                    target_mask,
                    regs.ctx(0xA000),
                    regs.ctx(0xA003),
                    (cb(0, 0) as u64) << 8,
                    cb(0, 4),
                    cb(0, 5),
                    (cb(0, 7) as u64) << 8,
                    (cb(0, 9) as u64) << 8,
                    cb(0, 11),
                    cb(0, 12),
                    (cb(1, 0) as u64) << 8,
                    cb(1, 4),
                    (regs.ctx(0xA012) as u64) << 8,
                    regs.ctx(0xA010),
                    (regs.ctx(0xA005) as u64) << 8
                );
            }
            self.skipped += 1;
            return Ok(());
        }
        // Extent: smallest attachment.
        let mut extent = (u32::MAX, u32::MAX);
        for a in colors.iter().flatten().chain(depth.iter()) {
            extent = (extent.0.min(a.key.width), extent.1.min(a.key.height));
        }
        // Pipeline.
        let vs_params = vs.translated.out.params;
        let key = GfxKey {
            vs: vs.translated.key_hash ^ vs.addr,
            ps: ps.as_ref().map(|p| p.translated.key_hash ^ p.addr),
            rect,
            gs_params: if rect { vs_params } else { 0 },
            colors: colors.iter().map(|c| c.map(|a| a.view_format).unwrap_or(vk::Format::UNDEFINED)).collect(),
            blend: blends.clone(),
            masks: masks.clone(),
            depth: depth.map(|d| d.key.format).unwrap_or(vk::Format::UNDEFINED),
            stencil: depth.is_some_and(|d| d.key.format == vk::Format::D32_SFLOAT_S8_UINT),
            topology_class: match prim {
                _ if tess.is_some() => 3,
                1 => 0,
                2 | 3 => 1,
                _ => 2,
            },
            polygon: ((regs.ctx(0xA205) >> 3) & 3) as u8,
            depth_clamp: regs.ctx(0xA204) & (3 << 26) != 0,
            tess: tess.as_ref().map(|t| (t.tf_param, t.min.to_bits(), t.max.to_bits())),
        };
        phase!(2);
        if !self.gfx_pipes.contains_key(&key) {
            let t0 = std::time::Instant::now();
            let p = self.create_gfx_pipeline(&key, &vs, ps.as_ref())?;
            let ms = t0.elapsed().as_millis();
            if ms >= 100 {
                crate::info!("graphics pipeline for vs {:#x} ps {:#x} took {ms} ms to create", vs.addr, ps.as_ref().map(|p| p.addr).unwrap_or(0));
            }
            self.gfx_pipes.insert(key.clone(), p);
        }
        let (pipeline, layout, set_layout) = {
            let p = &self.gfx_pipes[&key];
            (p.pipeline, p.layout, p.set_layout)
        };
        phase!(3);
        // Descriptors (may record uploads: must happen before rendering starts).
        let mut stages = smallvec::SmallVec::<[&shaders::Prepared; 2]>::new();
        stages.push(&vs);
        if let Some(p) = &ps {
            stages.push(p);
        }
        let (host, dev) = self.binding_bytes(&stages);
        let tex = self.texture_upload_estimate(&stages);
        self.reserve(host + tex + count as u64 * 4, dev);
        self.prepare_guest_reads(&stages);
        self.begin();
        // Uploads end any active rendering; record them first.
        phase!(4);
        self.vs_sgpr = None;
        let (mut set, mut wb_before) = self.descriptors_with_retry(set_layout, &stages)?;
        // GPU-read indirect arguments: the kernel's descriptor set from this
        // batch's pool, as the draw's (a full pool starts a new batch: the
        // draw's descriptors are written again in it).
        let mut fixup_set = None;
        if self.indirect_draw.is_some() {
            let (fl, _, _) = self.indirect_draw_pipe()?;
            let batch = self.batch;
            let mut fs = self.alloc_set(fl)?;
            if self.batch != batch {
                (set, wb_before) = self.descriptors_with_retry(set_layout, &stages)?;
                fs = self.alloc_set(fl)?;
            }
            fixup_set = Some(fs);
        }
        phase!(5);
        // Index data.
        let mut indexed = None;
        // (Tessellated draws fetched their control points in the LS stage.)
        let index_kind = tess.is_none() && (kind == "index2" || kind == "index_offset2");
        if index_kind && count > cp.draw_index_valid {
            // Past the index buffer the VGT fetches index 0: the indices
            // inside it, then zeros (a copy in the ring).
            let isz: u64 = if cp.index_type == 1 { 4 } else { 2 };
            let ty = if isz == 4 { vk::IndexType::UINT32 } else { vk::IndexType::UINT16 };
            let valid = cp.draw_index_valid as u64 * isz;
            self.report("indexed draw reads past its index buffer (index 0 fetched beyond it)".into());
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
                crate::info!("indexed draw ({kind}) of {count} indices at {index_addr:#x}: {} inside the index buffer, the rest fetched as index 0 (vs {:#x})", cp.draw_index_valid, vs.addr);
            }
            let total = count as u64 * isz;
            let off = self.upload_guest(index_addr, total)?;
            self.ring.write(off + valid, &vec![0u8; (total - valid) as usize]);
            indexed = Some((self.ring.buffer, off, ty));
            self.upload_bytes[3] += count as u64 * isz;
        } else if index_kind {
            let isz = if cp.index_type == 1 { 4 } else { 2 };
            let n = count as u64 * isz;
            let ty = if isz == 4 { vk::IndexType::UINT32 } else { vk::IndexType::UINT16 };
            // Indices in place when guest memory is current (as for read-only buffers).
            let current = !self.writebacks.iter().any(|w| w.base < index_addr + n && index_addr < w.base + w.size) && !cpu_writeback_in_flight(index_addr, n);
            let zc = if current && index_addr % isz == 0 && ctx().ext_host.is_some() { self.zc_binding(index_addr & !15, n + (index_addr & 15)) } else { None };
            if zc.is_some() {
                self.zc_read(index_addr, index_addr + n);
            }
            indexed = Some(match zc {
                Some((buf, off, _)) => match self.vcache_bind(index_addr & !15, n + (index_addr & 15)) {
                    Some((vb, voff)) => (vb, voff + (index_addr & 15), ty),
                    None => (buf, off + (index_addr & 15), ty),
                },
                None => {
                    let mirrored = if current && index_addr % isz == 0 && crate::tune::get(crate::tune::INDEX_MIRROR) != 0 {
                        self.vc_why = "index data";
                        self.vcache_bind_ex(index_addr & !15, n + (index_addr & 15), false, true)
                    } else {
                        None
                    };
                    match mirrored {
                        Some((vb, voff)) => {
                            self.index_mirrored.0 += 1;
                            self.index_mirrored.1 += n;
                            (vb, voff + (index_addr & 15), ty)
                        }
                        None => {
                            self.upload_bytes[3] += n;
                            (self.ring.buffer, self.upload_guest(index_addr, n)?, ty)
                        }
                    }
                }
            });
        }
        // The ring copies' forwards (descriptors, index data), then the pass.
        self.flush_forwards();
        // GPU-read indirect arguments: the draw command and the vertex
        // stage's offsets prepared on the GPU, outside the pass.
        let indirect_slot = match (self.indirect_draw.take(), fixup_set) {
            (Some(d), Some(fs)) => Some(self.indirect_draw_fixup(&d, fs, regs.ctx(0xA102))?),
            _ => None,
        };
        // Begin rendering if attachments changed.
        let same = self.rendering.as_ref().is_some_and(|r| r.colors[..] == colors[..] && r.depth == depth);
        if !same {
            self.end_rendering();
            if self.barrier_needed && crate::tune::get(crate::tune::PASS_BARRIER) != 0 {
                global_barrier(self.cmd);
                self.barrier_needed = false;
            }
            self.begin_rendering(&colors, depth, extent)?;
        }
        let tick = self.draws + self.dispatches + 1;
        for a in colors.iter().flatten() {
            self.mark_stale(a.key);
            if let Some(img) = self.images.get_mut(&a.key) {
                img.gpu_written = true;
                img.guest_stale = true;
                img.last_write = tick;
                img.last_write_frame = super::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
            }
            *self.write_gen.entry(a.key).or_default() += 1;
        }
        if let Some(d) = depth {
            self.mark_stale(d.key);
            if let Some(img) = self.images.get_mut(&d.key) {
                img.gpu_written = true;
                img.guest_stale = true;
                img.last_write = tick;
                img.last_write_frame = super::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
            }
            *self.write_gen.entry(d.key).or_default() += 1;
        }
        phase!(6);
        let c = ctx();
        let cmd = self.cmd;
        unsafe {
            if pipeline != self.bound_gfx {
                c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                self.bound_gfx = pipeline;
            }
            c.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, layout, 0, &[set], &[]);
            self.set_dynamic_state(regs, extent, prim, depth.is_some());
            // Depth/stencil clears requested through DB_RENDER_CONTROL.
            let rc = regs.ctx(0xA000);
            if rc & 3 != 0 && depth.is_some() {
                let mut aspect = vk::ImageAspectFlags::empty();
                if rc & 1 != 0 {
                    aspect |= vk::ImageAspectFlags::DEPTH;
                }
                if rc & 2 != 0 && key.stencil {
                    aspect |= vk::ImageAspectFlags::STENCIL;
                }
                if !aspect.is_empty() {
                    let ca = vk::ClearAttachment {
                        aspect_mask: aspect,
                        color_attachment: 0,
                        clear_value: vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: f32r(regs.ctx(0xA00B)), stencil: regs.ctx(0xA00A) & 0xFF } },
                    };
                    let cr = vk::ClearRect { rect: vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: extent.0, height: extent.1 } }, base_array_layer: 0, layer_count: 1 };
                    c.device.cmd_clear_attachments(cmd, &[ca], &[cr]);
                }
                if colors.iter().all(|c| c.is_none()) {
                    self.draws += 1;
                    self.batch_cmds += 1;
                    return Ok(());
                }
            }
            let instances = cp.num_instances.max(1);
            match indexed {
                // One vertex per patch (PrimitiveId selects its factors and LDS).
                _ if tess.is_some() => c.device.cmd_draw(cmd, tess.as_ref().map_or(0, |t| t.patches), 1, 0, 0),
                Some((buf, off, ty)) => {
                    c.device.cmd_bind_index_buffer(cmd, buf, off, ty);
                    match indirect_slot {
                        Some(slot) => c.device.cmd_draw_indexed_indirect(cmd, self.dring.buffer, slot, 1, 20),
                        None => {
                            let base_vertex = regs.ctx(0xA102) as i32;
                            c.device.cmd_draw_indexed(cmd, count, instances, 0, base_vertex, 0);
                        }
                    }
                }
                None => match indirect_slot {
                    Some(slot) => c.device.cmd_draw_indirect(cmd, self.dring.buffer, slot, 1, 16),
                    None => c.device.cmd_draw(cmd, count, instances, 0, 0),
                },
            }
        }
        phase!(7);
        self.dprof_draws += 1;
        if trace_frame() {
            // TLG_TRACE_ATTR=<vs addr>:<resource>:<byte>: histogram of that byte
            // of every record of the vertex shader's buffer resource, and of
            // the records the draw's indices reference (traced frames).
            static SPEC: std::sync::OnceLock<Option<(u64, usize, u64)>> = std::sync::OnceLock::new();
            let spec = *SPEC.get_or_init(|| {
                let v = std::env::var("TLG_TRACE_ATTR").ok()?;
                let mut it = v.split(':');
                Some((u64::from_str_radix(it.next()?.trim_start_matches("0x"), 16).ok()?, it.next()?.parse().ok()?, it.next()?.parse().ok()?))
            });
            if let Some((a, ri, byte)) = spec.filter(|s| s.0 == vs.addr) {
                if let Some(ResolvedRes::Buffer(v)) = vs.resources.get(ri) {
                    let stride = v.stride.max(1) as u64;
                    let n = v.num_records as u64;
                    let mut all = std::collections::BTreeMap::new();
                    let mut buf = vec![0u8; (n * stride) as usize];
                    resources::copy_from_guest(&mut buf, v.base);
                    for r in 0..n {
                        *all.entry(buf[(r * stride + byte) as usize]).or_insert(0u64) += 1;
                    }
                    let mut used = std::collections::BTreeMap::new();
                    if kind == "index2" || kind == "index_offset2" {
                        let isz: u64 = if cp.index_type == 1 { 4 } else { 2 };
                        let mut ib = vec![0u8; (count as u64 * isz) as usize];
                        resources::copy_from_guest(&mut ib, index_addr);
                        for c in ib.chunks_exact(isz as usize) {
                            let i = if isz == 4 { u32::from_le_bytes(c.try_into().unwrap()) as u64 } else { u16::from_le_bytes(c.try_into().unwrap()) as u64 };
                            let i = i + regs.ctx(0xA102) as u64;
                            let val = if i < n { buf[(i * stride + byte) as usize] as i32 } else { -1 };
                            *used.entry(val).or_insert(0u64) += 1;
                        }
                    }
                    crate::info!("FRAME attr histogram vs {a:#x} res {ri} byte {byte} (n={count}): all records {all:?}; indexed {used:?}");
                }
            }
        }
        if self.occl_windows > 0 {
            // (Diagnostics: what the title's occlusion queries measure.)
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 24 {
                crate::info!(
                    "occlusion-counted draw: {kind} n={count} prim {prim:#x} vs {:#x} ps {:#x} colors {} depth {:?} dctl {:#x} mask {:#x}",
                    vs.addr,
                    ps.as_ref().map(|p| p.addr).unwrap_or(0),
                    colors.iter().flatten().count(),
                    depth.map(|d| d.key.base),
                    regs.ctx(0xA200),
                    regs.ctx(0xA08E)
                );
            }
        }
        let label_addr = ps.as_ref().map(|p| p.addr).unwrap_or(vs.addr);
        self.mark(|| format!("draw ps {label_addr:#x}"));
        self.perform_gpu_writebacks();
        self.draws += 1;
        self.batch_cmds += 1;
        if self.writebacks.len() + self.zc_writes.len() > wb_before {
            // Buffer writes: order later consumers; CPU-sourced textures may
            // alias the written memory.
            unordered!(self);
            self.epoch += 1;
        }
        if trace_frame() {
            if vs.translated.out.bindings.iter().any(|b| b.kind == BindingKind::DebugOut) {
                // (Before timing: the draw executes in trace_timing's flush.)
                self.trace_timing("draw");
                self.check_vs_positions(cp, &vs, kind, count, index_addr);
            } else {
                self.trace_timing("draw");
            }
            if let Some(p) = &ps {
                self.dump_after(p.addr, &colors, depth);
            }
            // TLG_TRACE_DEPTH=<threshold, default 0.985>: how much of the depth
            // target is nearer than the threshold after each draw of the
            // traced frame (logged on change; 0.99999 counts all geometry in
            // front of the cleared far plane).
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            if let Some(d) = depth.filter(|_| *ON.get_or_init(|| std::env::var_os("TLG_TRACE_DEPTH").is_some())) {
                self.end_rendering();
                if let Some(near) = self.depth_near_fraction(d.key) {
                    let prev = self.depth_trace.insert(d.key.base, near).unwrap_or(-1.0);
                    if (near - prev).abs() > 0.01 {
                        crate::info!("FRAME depth {:#x} near fraction {prev:.3} -> {near:.3} after draw vs {:#x} ps {:#x} n={count}", d.key.base, vs.addr, ps.as_ref().map(|p| p.addr).unwrap_or(0));
                    }
                }
            }
        }
        Ok(())
    }

    /// Write-backs skipped because the title had reused the image's memory.
    pub fn reused_skips(&self) -> u64 {
        self.reused_skips
    }
    pub fn reused_targets(&self) -> u64 {
        self.reused_targets
    }

    /// Texture validations settled by comparing shared edge pages.
    pub fn edge_skips(&self) -> u64 {
        self.edge_skips
    }

    /// TLG_VS_POS buffer (created on first use, filled with NaNs).
    fn debug_out_buffer(&mut self) -> Result<(vk::Buffer, *mut u8), String> {
        if let Some(b) = self.debug_out {
            return Ok(b);
        }
        let size = DEBUG_OUT_BYTES;
        let (buf, _mem, ptr) = ctx().host_buffer(size, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        unsafe { std::ptr::write_bytes(ptr, 0xFF, size as usize) };
        self.debug_out = Some((buf, ptr));
        Ok((buf, ptr))
    }

    /// After a traced draw of the TLG_VS_POS shader: the clip-space positions
    /// of the vertices its indices reference; logs those far outside the
    /// view (|x|,|y| > 4w), behind or at the eye (w < 1e-4) or not finite.
    fn check_vs_positions(&mut self, cp: &CommandProcessor, vs: &Prepared, kind: &str, count: u32, index_addr: u64) {
        let Some((_, ptr)) = self.debug_out else { return };
        const S: usize = gcn::translate::DEBUG_OUT_STRIDE as usize;
        let all = unsafe { std::slice::from_raw_parts(ptr as *const f32, gcn::translate::DEBUG_OUT_VERTICES as usize * S) };
        let pos = |i: u64| -> [f32; 4] {
            let i = i.min(gcn::translate::DEBUG_OUT_VERTICES as u64 - 1) as usize;
            [all[i * S], all[i * S + 1], all[i * S + 2], all[i * S + 3]]
        };
        let ncap = shaders::debug_vs_capture().len();
        let caps = |i: u64| -> String {
            if ncap == 0 {
                return String::new();
            }
            let i = i.min(gcn::translate::DEBUG_OUT_VERTICES as u64 - 1) as usize;
            let v: Vec<String> = (0..ncap).map(|k| format!("{:.4e}", all[i * S + 4 + k])).collect();
            format!(" captures [{}]", v.join(" "))
        };
        let base_vertex = cp.regs.ctx(0xA102) as u64;
        let mut idx = Vec::new();
        if kind == "index2" || kind == "index_offset2" {
            let isz: u64 = if cp.index_type == 1 { 4 } else { 2 };
            let mut ib = vec![0u8; (count as u64 * isz) as usize];
            resources::copy_from_guest(&mut ib, index_addr);
            idx.extend(ib.chunks_exact(isz as usize).map(|c| if isz == 4 { u32::from_le_bytes(c.try_into().unwrap()) as u64 } else { u16::from_le_bytes(c.try_into().unwrap()) as u64 } + base_vertex));
        } else {
            idx.extend(0..count as u64);
        }
        let (mut far, mut behind, mut bad, mut unwritten) = (0u32, 0u32, 0u32, 0u32);
        let mut examples = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for &i in &idx {
            if !seen.insert(i) {
                continue;
            }
            let p = pos(i);
            let raw = p.map(f32::to_bits);
            let kind = if raw.iter().all(|&r| r == u32::MAX) {
                unwritten += 1;
                continue;
            } else if p.iter().any(|v| !v.is_finite()) {
                bad += 1;
                "not finite"
            } else if p[3] < 1e-4 {
                behind += 1;
                if examples.len() < 2 {
                    examples.push(format!("v{i} behind clip ({:.4e}, {:.4e}, {:.4e}, {:.4e})", p[0], p[1], p[2], p[3]));
                }
                continue;
            } else if p[0].abs() > 4.0 * p[3] || p[1].abs() > 4.0 * p[3] {
                far += 1;
                "far outside"
            } else {
                continue;
            };
            if examples.len() < 6 {
                // The vertex's position attribute (resource 0).
                let attr = match vs.resources.first() {
                    Some(ResolvedRes::Buffer(v)) => {
                        let mut b = [0u8; 12];
                        resources::copy_from_guest(&mut b, v.base + i * v.stride.max(1) as u64);
                        let f = |k: usize| f32::from_le_bytes(b[k * 4..k * 4 + 4].try_into().unwrap());
                        format!("({:.3}, {:.3}, {:.3})", f(0), f(1), f(2))
                    }
                    _ => String::new(),
                };
                examples.push(format!("v{i} {kind} clip ({:.4e}, {:.4e}, {:.4e}, {:.4e}) attr0 {attr}{}", p[0], p[1], p[2], p[3], caps(i)));
            }
        }
        crate::info!(
            "FRAME vs {:#x} positions: {} vertices; {far} far outside the view, {behind} behind the eye, {bad} not finite, {unwritten} not written; {}",
            vs.addr,
            seen.len(),
            examples.join("; ")
        );
        // Reset for the next draw.
        unsafe { std::ptr::write_bytes(ptr, 0xFF, DEBUG_OUT_BYTES as usize) };
    }

    /// Debug: fraction of a depth image's texels nearer than TLG_TRACE_DEPTH.
    fn depth_near_fraction(&mut self, k: ImageKey) -> Option<f32> {
        let texel: u64 = match k.format {
            vk::Format::D32_SFLOAT | vk::Format::D32_SFLOAT_S8_UINT => 4,
            _ => return None,
        };
        let size = k.width as u64 * k.height as u64 * texel;
        self.flush();
        self.begin();
        global_barrier(self.cmd);
        let off = self.ring.alloc(size)?;
        let region = vk::BufferImageCopy {
            buffer_offset: off,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::DEPTH, mip_level: 0, base_array_layer: 0, layer_count: 1 },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D { width: k.width, height: k.height, depth: 1 },
        };
        unsafe { ctx().device.cmd_copy_image_to_buffer(self.cmd, self.images[&k].image, vk::ImageLayout::GENERAL, self.ring.buffer, &[region]) };
        self.flush();
        let raw = unsafe { std::slice::from_raw_parts(self.ring.ptr.add(off as usize) as *const f32, (size / 4) as usize) };
        static T: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
        let t = *T.get_or_init(|| std::env::var("TLG_TRACE_DEPTH").ok().and_then(|v| v.parse().ok()).filter(|v: &f32| *v > 0.0 && *v <= 1.0).unwrap_or(0.985));
        let near = raw.iter().filter(|&&d| d < t).count();
        Some(near as f32 / raw.len().max(1) as f32)
    }

    /// Frame trace: submits the command just recorded (with its uploads) on
    /// its own and logs how long the GPU took.
    fn trace_timing(&mut self, what: &str) {
        let t0 = std::time::Instant::now();
        self.flush();
        let ld = |a: &std::sync::atomic::AtomicU64| a.load(std::sync::atomic::Ordering::Relaxed);
        crate::info!(
            "FRAME timing {what}: {:.3} ms (fence {:.3} ms, write-back {} KiB {:.3} ms)",
            t0.elapsed().as_secs_f64() * 1e3,
            ld(&LAST_FENCE_NS) as f64 / 1e6,
            ld(&LAST_WB_BYTES) >> 10,
            ld(&LAST_WB_NS) as f64 / 1e6
        );
    }

    fn begin_rendering(&mut self, colors: &[Option<Attachment>], depth: Option<Attachment>, extent: (u32, u32)) -> Result<(), String> {
        PASS_BEGINS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut color_infos = Vec::new();
        for a in colors {
            match a {
                Some(a) => {
                    let img = self.images.get_mut(&a.key).ok_or("attachment image missing")?;
                    let view = img.view(a.view_format, 0, vk::ImageViewType::TYPE_2D, false, a.layer, 1)?;
                    color_infos.push(
                        vk::RenderingAttachmentInfo::default()
                            .image_view(view)
                            .image_layout(vk::ImageLayout::GENERAL)
                            .load_op(vk::AttachmentLoadOp::LOAD)
                            .store_op(vk::AttachmentStoreOp::STORE),
                    );
                }
                None => color_infos.push(vk::RenderingAttachmentInfo::default().image_view(vk::ImageView::null())),
            }
        }
        let depth_info = match depth {
            Some(d) => {
                let img = self.images.get_mut(&d.key).ok_or("depth image missing")?;
                let view = img.view(d.view_format, 0, vk::ImageViewType::TYPE_2D, false, 0, 1)?;
                Some(vk::RenderingAttachmentInfo::default().image_view(view).image_layout(vk::ImageLayout::GENERAL).load_op(vk::AttachmentLoadOp::LOAD).store_op(vk::AttachmentStoreOp::STORE))
            }
            None => None,
        };
        let stencil = depth.is_some_and(|d| d.key.format == vk::Format::D32_SFLOAT_S8_UINT);
        let mut ri = vk::RenderingInfo::default()
            .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: extent.0, height: extent.1 } })
            .layer_count(1)
            .color_attachments(&color_infos);
        if let Some(d) = &depth_info {
            ri = ri.depth_attachment(d);
            if stencil {
                ri = ri.stencil_attachment(d);
            }
        }
        unsafe { ctx().device.cmd_begin_rendering(self.cmd, &ri) };
        self.rendering = Some(Rendering { colors: colors.iter().copied().collect(), depth, extent });
        Ok(())
    }

    unsafe fn set_dynamic_state(&mut self, regs: &Regs, extent: (u32, u32), prim: u32, has_depth: bool) {
        let vte = regs.ctx(0xA206);
        let (xs, xo, ys, yo, zs, zo) = (f32r(regs.ctx(0xA10F)), f32r(regs.ctx(0xA110)), f32r(regs.ctx(0xA111)), f32r(regs.ctx(0xA112)), f32r(regs.ctx(0xA113)), f32r(regs.ctx(0xA114)));
        let vp = if vte & 0x3F == 0x3F && xs != 0.0 {
            vk::Viewport { x: xo - xs, y: yo - ys, width: 2.0 * xs, height: 2.0 * ys, min_depth: zo.clamp(0.0, 1.0), max_depth: (zo + zs).clamp(0.0, 1.0) }
        } else {
            if vte & 0x3F != 0x3F {
                self.report(format!("PA_CL_VTE_CNTL {vte:#x} (viewport transform disabled) approximated"));
            }
            vk::Viewport { x: 0.0, y: 0.0, width: extent.0 as f32, height: extent.1 as f32, min_depth: 0.0, max_depth: 1.0 }
        };
        let mut s = (0i32, 0i32, extent.0 as i32, extent.1 as i32);
        // Screen, window and generic scissors always apply; the viewport
        // scissor only with PA_SC_MODE_CNTL_0.VPORT_SCISSOR_ENABLE.
        let vport_scissor = regs.ctx(0xA292) & 2 != 0;
        for (tl, br) in [(0xA00C, 0xA00D), (0xA081, 0xA082), (0xA090, 0xA091), (0xA094, 0xA095)] {
            if tl == 0xA094 && !vport_scissor {
                continue;
            }
            let (x0, y0, x1, y1) = scissor_rect(regs.ctx(tl), regs.ctx(br));
            if x1 > x0 && y1 > y0 {
                s = (s.0.max(x0), s.1.max(y0), s.2.min(x1), s.3.min(y1));
            }
        }
        let mode = regs.ctx(0xA205);
        let mut cull = vk::CullModeFlags::NONE;
        if mode & 1 != 0 {
            cull |= vk::CullModeFlags::FRONT;
        }
        if mode & 2 != 0 {
            cull |= vk::CullModeFlags::BACK;
        }
        if prim == 0x11 {
            cull = vk::CullModeFlags::NONE;
        }
        let topo = match prim {
            1 => vk::PrimitiveTopology::POINT_LIST,
            2 => vk::PrimitiveTopology::LINE_LIST,
            3 => vk::PrimitiveTopology::LINE_STRIP,
            5 => vk::PrimitiveTopology::TRIANGLE_FAN,
            6 => vk::PrimitiveTopology::TRIANGLE_STRIP,
            9 => vk::PrimitiveTopology::PATCH_LIST,
            _ => vk::PrimitiveTopology::TRIANGLE_LIST,
        };
        let dctl = regs.ctx(0xA200);
        // TLG_NO_DEPTH_TEST=1 (diagnostic): depth testing off for all draws.
        static NO_DEPTH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let no_depth = *NO_DEPTH.get_or_init(|| std::env::var_os("TLG_NO_DEPTH_TEST").is_some());
        // DB_DEPTH_CONTROL.DEPTH_BOUNDS_ENABLE with DB_DEPTH_BOUNDS_MIN/MAX
        // (light volumes restrict lighting to their depth range).
        let bounds_test = has_depth && dctl & 8 != 0 && !no_depth;
        let (blo, bhi) = if bounds_test {
            let lo = f32r(regs.ctx(0xA008)).clamp(0.0, 1.0);
            (lo, f32r(regs.ctx(0xA009)).clamp(0.0, 1.0).max(lo))
        } else {
            (0.0, 1.0)
        };
        let sctl = regs.ctx(0xA10B);
        let refm = regs.ctx(0xA10C);
        let refm_bf = regs.ctx(0xA10D);
        let back_en = dctl & (1 << 7) != 0;
        let ffunc = desc::vk_compare((dctl >> 8) & 7);
        let bfunc = if back_en { desc::vk_compare((dctl >> 20) & 7) } else { ffunc };
        let bs = if back_en { sctl >> 12 } else { sctl };
        let rb = if back_en { refm_bf } else { refm };
        // PA_SU_SC_MODE_CNTL.POLY_OFFSET_FRONT/BACK_ENABLE; registers
        // PA_SU_POLY_OFFSET_CLAMP (0xA2DF), FRONT_SCALE (0xA2E0, 16x the
        // slope factor), FRONT_OFFSET (0xA2E1), BACK_* (0xA2E2/3).
        let bias_enable = mode & (3 << 11) != 0;
        let bias = if bias_enable {
            let (sr, or) = if mode & (1 << 11) != 0 { (0xA2E0, 0xA2E1) } else { (0xA2E2, 0xA2E3) };
            [f32r(regs.ctx(or)), f32r(regs.ctx(0xA2DF)), f32r(regs.ctx(sr)) / 16.0]
        } else {
            [0.0; 3]
        };
        let ops = |x: u32, f: vk::CompareOp| [stencil_op(x & 0xF).as_raw(), stencil_op((x >> 4) & 0xF).as_raw(), stencil_op((x >> 8) & 0xF).as_raw(), f.as_raw()];
        let n = DynState {
            viewport: [vp.x, vp.y, vp.width, vp.height, vp.min_depth, vp.max_depth].map(f32::to_bits),
            scissor: [s.0.max(0), s.1.max(0), (s.2 - s.0).max(0), (s.3 - s.1).max(0)],
            cull: cull.as_raw(),
            front_cw: mode & 4 != 0,
            topology: topo.as_raw(),
            restart: regs.ctx(0xA2A5) & 1 != 0 && matches!(prim, 3 | 5 | 6),
            depth_test: has_depth && dctl & 2 != 0 && !no_depth,
            depth_write: has_depth && dctl & 4 != 0,
            depth_op: desc::vk_compare((dctl >> 4) & 7).as_raw(),
            bounds_test,
            bounds: [blo.to_bits(), bhi.to_bits()],
            stencil_test: has_depth && dctl & 1 != 0,
            stencil_ops: [ops(sctl, ffunc), ops(bs, bfunc)],
            stencil_ref: [refm & 0xFF, rb & 0xFF],
            stencil_compare: [(refm >> 8) & 0xFF, (rb >> 8) & 0xFF],
            stencil_write: [(refm >> 16) & 0xFF, (rb >> 16) & 0xFF],
            bias_enable,
            bias: bias.map(f32::to_bits),
            blend_constants: [regs.ctx(0xA105), regs.ctx(0xA106), regs.ctx(0xA107), regs.ctx(0xA108)],
        };
        let o = self.dyn_state;
        if o == Some(n) {
            return;
        }
        let c = ctx();
        let cmd = self.cmd;
        unsafe {
            if o.is_none() {
                // Fixed for every draw.
                c.device.cmd_set_rasterizer_discard_enable(cmd, false);
                c.device.cmd_set_line_width(cmd, 1.0);
            }
            if o.is_none_or(|o| o.viewport != n.viewport) {
                c.device.cmd_set_viewport(cmd, 0, &[vp]);
            }
            if o.is_none_or(|o| o.scissor != n.scissor) {
                let sc = vk::Rect2D { offset: vk::Offset2D { x: n.scissor[0], y: n.scissor[1] }, extent: vk::Extent2D { width: n.scissor[2] as u32, height: n.scissor[3] as u32 } };
                c.device.cmd_set_scissor(cmd, 0, &[sc]);
            }
            if o.is_none_or(|o| o.cull != n.cull) {
                c.device.cmd_set_cull_mode(cmd, cull);
            }
            if o.is_none_or(|o| o.front_cw != n.front_cw) {
                c.device.cmd_set_front_face(cmd, if n.front_cw { vk::FrontFace::CLOCKWISE } else { vk::FrontFace::COUNTER_CLOCKWISE });
            }
            if o.is_none_or(|o| o.topology != n.topology) {
                c.device.cmd_set_primitive_topology(cmd, topo);
            }
            if o.is_none_or(|o| o.restart != n.restart) {
                c.device.cmd_set_primitive_restart_enable(cmd, n.restart);
            }
            if o.is_none_or(|o| o.depth_test != n.depth_test) {
                c.device.cmd_set_depth_test_enable(cmd, n.depth_test);
            }
            if o.is_none_or(|o| o.depth_write != n.depth_write) {
                c.device.cmd_set_depth_write_enable(cmd, n.depth_write);
            }
            if o.is_none_or(|o| o.depth_op != n.depth_op) {
                c.device.cmd_set_depth_compare_op(cmd, vk::CompareOp::from_raw(n.depth_op));
            }
            if o.is_none_or(|o| o.bounds_test != n.bounds_test) {
                c.device.cmd_set_depth_bounds_test_enable(cmd, n.bounds_test);
            }
            if n.bounds_test && o.is_none_or(|o| o.bounds != n.bounds || !o.bounds_test) {
                c.device.cmd_set_depth_bounds(cmd, blo, bhi);
            }
            if o.is_none_or(|o| o.stencil_test != n.stencil_test) {
                c.device.cmd_set_stencil_test_enable(cmd, n.stencil_test);
            }
            for (i, face) in [vk::StencilFaceFlags::FRONT, vk::StencilFaceFlags::BACK].into_iter().enumerate() {
                if o.is_none_or(|o| o.stencil_ops[i] != n.stencil_ops[i]) {
                    let p = n.stencil_ops[i];
                    c.device.cmd_set_stencil_op(cmd, face, vk::StencilOp::from_raw(p[0]), vk::StencilOp::from_raw(p[1]), vk::StencilOp::from_raw(p[2]), vk::CompareOp::from_raw(p[3]));
                }
                if o.is_none_or(|o| o.stencil_ref[i] != n.stencil_ref[i]) {
                    c.device.cmd_set_stencil_reference(cmd, face, n.stencil_ref[i]);
                }
                if o.is_none_or(|o| o.stencil_compare[i] != n.stencil_compare[i]) {
                    c.device.cmd_set_stencil_compare_mask(cmd, face, n.stencil_compare[i]);
                }
                if o.is_none_or(|o| o.stencil_write[i] != n.stencil_write[i]) {
                    c.device.cmd_set_stencil_write_mask(cmd, face, n.stencil_write[i]);
                }
            }
            if o.is_none_or(|o| o.bias_enable != n.bias_enable) {
                c.device.cmd_set_depth_bias_enable(cmd, n.bias_enable);
            }
            if n.bias_enable && o.is_none_or(|o| o.bias != n.bias || !o.bias_enable) {
                c.device.cmd_set_depth_bias(cmd, bias[0], bias[1], bias[2]);
            }
            if o.is_none_or(|o| o.blend_constants != n.blend_constants) {
                c.device.cmd_set_blend_constants(cmd, &n.blend_constants.map(f32::from_bits));
            }
        }
        self.dyn_state = Some(n);
    }

    fn create_gfx_pipeline(&mut self, key: &GfxKey, vs: &Prepared, ps: Option<&Prepared>) -> Result<Pipe, String> {
        super::set_phase(super::phase::PIPELINE, vs.addr);
        let c = ctx();
        let vsm = self.module(key.vs, &vs.translated.out.spirv)?;
        let psm = match ps {
            Some(p) => Some(self.module(key.ps.unwrap(), &p.translated.out.spirv)?),
            None => None,
        };
        let gsm = if key.rect { Some(self.rect_gs(key.gs_params)?) } else { None };
        let mut stages_p = vec![vs];
        if let Some(p) = ps {
            stages_p.push(p);
        }
        let set_layout = self.set_layout(&stages_p)?;
        let layouts = [set_layout];
        let layout = unsafe { c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts), None) }.map_err(|e| e.to_string())?;
        let main = c"main";
        let mut stages = Vec::new();
        if let Some((tf_param, min, max)) = key.tess {
            // Emulated tessellation: nothing per vertex, the HS's factors to
            // the tessellator, the domain shader as the evaluation stage.
            let null = self.tess_null_vs()?;
            let tcs = self.tess_control(tf_param, f32::from_bits(min), f32::from_bits(max))?;
            stages.push(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(null).name(main));
            stages.push(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::TESSELLATION_CONTROL).module(tcs).name(main));
            stages.push(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::TESSELLATION_EVALUATION).module(vsm).name(main));
        } else {
            stages.push(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vsm).name(main));
        }
        if let Some(g) = gsm {
            stages.push(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::GEOMETRY).module(g).name(main));
        }
        if let Some(p) = psm {
            stages.push(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(p).name(main));
        }
        let vi = vk::PipelineVertexInputStateCreateInfo::default();
        let topo = match key.topology_class {
            0 => vk::PrimitiveTopology::POINT_LIST,
            1 => vk::PrimitiveTopology::LINE_LIST,
            3 => vk::PrimitiveTopology::PATCH_LIST,
            _ => vk::PrimitiveTopology::TRIANGLE_LIST,
        };
        let ia = vk::PipelineInputAssemblyStateCreateInfo::default().topology(topo);
        // (Ignored without tessellation stages.)
        let tess_state = vk::PipelineTessellationStateCreateInfo::default().patch_control_points(1);
        let vps = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let rs = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(key.depth_clamp)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0);
        let ms = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let ds = vk::PipelineDepthStencilStateCreateInfo::default();
        let atts: Vec<vk::PipelineColorBlendAttachmentState> = key
            .colors
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let b = key.blend.get(i).copied().unwrap_or(0);
                let m = key.masks.get(i).copied().unwrap_or(0);
                let enable = b & (1 << 30) != 0 && *f != vk::Format::UNDEFINED;
                let sep = b & (1 << 29) != 0;
                let (cs, cd, cf) = (b & 0x1F, (b >> 8) & 0x1F, (b >> 5) & 7);
                let (as_, ad, af) = if sep { ((b >> 16) & 0x1F, (b >> 24) & 0x1F, (b >> 21) & 7) } else { (cs, cd, cf) };
                vk::PipelineColorBlendAttachmentState::default()
                    .blend_enable(enable)
                    .src_color_blend_factor(blend_factor(cs))
                    .dst_color_blend_factor(blend_factor(cd))
                    .color_blend_op(blend_op(cf))
                    .src_alpha_blend_factor(blend_factor(as_))
                    .dst_alpha_blend_factor(blend_factor(ad))
                    .alpha_blend_op(blend_op(af))
                    .color_write_mask(vk::ColorComponentFlags::from_raw(m))
            })
            .collect();
        let cb = vk::PipelineColorBlendStateCreateInfo::default().attachments(&atts);
        let dyns = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::LINE_WIDTH,
            vk::DynamicState::DEPTH_BIAS,
            vk::DynamicState::BLEND_CONSTANTS,
            vk::DynamicState::STENCIL_COMPARE_MASK,
            vk::DynamicState::STENCIL_WRITE_MASK,
            vk::DynamicState::STENCIL_REFERENCE,
            vk::DynamicState::CULL_MODE,
            vk::DynamicState::FRONT_FACE,
            vk::DynamicState::PRIMITIVE_TOPOLOGY,
            vk::DynamicState::DEPTH_TEST_ENABLE,
            vk::DynamicState::DEPTH_WRITE_ENABLE,
            vk::DynamicState::DEPTH_COMPARE_OP,
            vk::DynamicState::DEPTH_BOUNDS_TEST_ENABLE,
            vk::DynamicState::DEPTH_BOUNDS,
            vk::DynamicState::STENCIL_TEST_ENABLE,
            vk::DynamicState::STENCIL_OP,
            vk::DynamicState::DEPTH_BIAS_ENABLE,
            vk::DynamicState::PRIMITIVE_RESTART_ENABLE,
            vk::DynamicState::RASTERIZER_DISCARD_ENABLE,
        ];
        let dy = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyns);
        let mut rinfo = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&key.colors).depth_attachment_format(key.depth);
        if key.stencil {
            rinfo = rinfo.stencil_attachment_format(key.depth);
        }
        let gpci = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vi)
            .input_assembly_state(&ia)
            .tessellation_state(&tess_state)
            .viewport_state(&vps)
            .rasterization_state(&rs)
            .multisample_state(&ms)
            .depth_stencil_state(&ds)
            .color_blend_state(&cb)
            .dynamic_state(&dy)
            .layout(layout)
            .push_next(&mut rinfo);
        let pipeline = unsafe { c.device.create_graphics_pipelines(pipeline_cache(), &[gpci], None) }.map_err(|(_, e)| format!("vkCreateGraphicsPipelines: {e}"))?[0];
        pipeline_cache_dirty();
        crate::debug!("created graphics pipeline vs {:#x} ps {:?} colors {:?} depth {:?} rect {}", vs.addr, ps.map(|p| p.addr), key.colors, key.depth, key.rect);
        Ok(Pipe { pipeline, layout, set_layout })
    }

    // -------------------------------------------------------------------
    // Tessellation

    /// Device buffers for the threadgroups' LDS and the factor ring.
    fn tess_buffers(&mut self) -> Result<(vk::Buffer, vk::Buffer), String> {
        if let Some(b) = self.tess_bufs {
            return Ok(b);
        }
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let (lds, _) = ctx().device_buffer(TESS_LDS_BYTES, usage)?;
        let (tf, _) = ctx().device_buffer(TESS_FACTOR_RING_BYTES, usage)?;
        self.tess_bufs = Some((lds, tf));
        Ok((lds, tf))
    }

    fn tess_null_vs(&mut self) -> Result<vk::ShaderModule, String> {
        let key = 0x4E55_4C4C_5653_0000;
        if let Some(&m) = self.modules.get(&key) {
            return Ok(m);
        }
        self.module(key, &gcn::translate::tess::null_vertex_shader())
    }

    fn tess_control(&mut self, tf_param: u32, min: f32, max: f32) -> Result<vk::ShaderModule, String> {
        let key = 0x5443_5300_0000_0000 ^ ((tf_param as u64 & 0xFF) << 48) ^ ((min.to_bits() as u64) << 24) ^ max.to_bits() as u64;
        if let Some(&m) = self.modules.get(&key) {
            return Ok(m);
        }
        self.module(key, &gcn::translate::tess::control_shader(tf_param, min, max, TESS_TF_BINDING))
    }

    /// A patch draw: checks the tessellation configuration (VGT_SHADER_STAGES_EN
    /// LS -> HS -> VS as domain shader, VGT_LS_HS_CONFIG, VGT_TF_PARAM, the LDS
    /// size in SPI_SHADER_PGM_RSRC2_LS) and records the LS dispatch (one
    /// invocation per control point, its vertex id from the index buffer) and
    /// the HS dispatch (a workgroup per threadgroup of patches), which leave
    /// the LDS and factors in the renderer's buffers for the draw.
    fn tess_stages(&mut self, cp: &CommandProcessor, kind: &str, count: u32, index_addr: u64) -> Result<TessDraw, String> {
        let regs = &cp.regs;
        let en = regs.ctx(0xA2D5);
        if en & 3 != 1 || en & 4 == 0 || en & 0x38 != 0 || (en >> 6) & 3 != 1 {
            return Err(format!("tessellation with VGT_SHADER_STAGES_EN {en:#x} unsupported"));
        }
        let cfg = regs.ctx(0xA2D6);
        let (np, icp, ocp) = (cfg & 0xFF, (cfg >> 8) & 0x3F, (cfg >> 14) & 0x3F);
        let tf_param = regs.ctx(0xA2DB);
        let lds_dwords = ((regs.sh(0x2D4B) >> 7) & 0x1FF) * 128;
        if np == 0 || icp == 0 || ocp == 0 || lds_dwords == 0 {
            return Err(format!("tessellation config VGT_LS_HS_CONFIG {cfg:#x} LDS {lds_dwords} dwords unsupported"));
        }
        if cp.num_instances > 1 {
            return Err("instanced tessellation unsupported".into());
        }
        let patches = count / icp;
        let base = TessSpec { role: TessRole::None, patches: np, in_cp: icp, out_cp: ocp, lds_dwords, tf_param, rsrc2: 0 };
        let draw = TessDraw { patches, tf_param, min: f32r(regs.ctx(0xA287)), max: f32r(regs.ctx(0xA286)), ds: TessSpec { role: TessRole::Ds, ..base } };
        if patches == 0 {
            return Ok(draw);
        }
        let groups = patches.div_ceil(np);
        if groups as u64 * lds_dwords as u64 * 4 > TESS_LDS_BYTES || patches as u64 * gcn::translate::tf_dwords(tf_param) as u64 * 4 > TESS_FACTOR_RING_BYTES {
            return Err(format!("tessellated draw of {patches} patches exceeds the LDS/factor buffers"));
        }
        // Control points' vertex ids (past the index buffer the VGT fetches index 0).
        let n = patches * icp;
        let ids: Vec<u32> = if kind == "index2" || kind == "index_offset2" {
            let isz: usize = if cp.index_type == 1 { 4 } else { 2 };
            let mut ib = vec![0u8; n as usize * isz];
            let valid = (cp.draw_index_valid.min(n) as usize) * isz;
            self.guard_reads(&[index_addr, index_addr + valid.max(1) as u64 - 1]);
            resources::copy_from_guest(&mut ib[..valid], index_addr);
            let bv = regs.ctx(0xA102);
            ib.chunks_exact(isz).map(|c| if isz == 4 { u32::from_le_bytes(c.try_into().unwrap()) } else { u16::from_le_bytes(c.try_into().unwrap()) as u32 }.wrapping_add(bv)).collect()
        } else {
            (0..n).collect()
        };
        let prep = |role| shaders::prepare_tess(&StageRegs { stage: Stage::Compute, regs }, TessSpec { role, ..base });
        let (mut ls, mut hs) = (prep(TessRole::Ls)?, prep(TessRole::Hs)?);
        self.prep_shader = hs.addr;
        self.prep_vs = ls.addr;
        for _ in 0..3 {
            let reads: smallvec::SmallVec<[u64; 64]> = ls.mem_reads.iter().chain(hs.mem_reads.iter()).copied().collect();
            if !self.guard_reads(&reads) {
                break;
            }
            self.begin();
            (ls, hs) = (prep(TessRole::Ls)?, prep(TessRole::Hs)?);
        }
        ls.sgpr_data.push(n);
        ls.sgpr_data.extend_from_slice(&ids);
        hs.sgpr_data.push(patches);
        self.tess_dispatch(&ls, n.div_ceil(64))?;
        self.tess_dispatch(&hs, groups)?;
        Ok(draw)
    }

    /// Records one tessellation compute stage (LS or HS) over `groups`
    /// workgroups, ordered after all earlier work (the LDS and factor
    /// buffers are reused by every draw).
    fn tess_dispatch(&mut self, p: &Prepared, groups: u32) -> Result<(), String> {
        let key = p.translated.key_hash ^ p.addr;
        if !self.cs_pipes.contains_key(&key) {
            let c = ctx();
            let m = self.module(key, &p.translated.out.spirv)?;
            let set_layout = self.set_layout(&[p])?;
            let layouts = [set_layout];
            let layout = unsafe { c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts), None) }.map_err(|e| e.to_string())?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(m).name(c"main");
            let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
            let pipeline = unsafe { c.device.create_compute_pipelines(pipeline_cache(), &[ci], None) }.map_err(|(_, e)| format!("vkCreateComputePipelines: {e}"))?[0];
            pipeline_cache_dirty();
            self.cs_pipes.insert(key, Pipe { pipeline, layout, set_layout });
        }
        let (pipeline, layout, set_layout) = {
            let q = &self.cs_pipes[&key];
            (q.pipeline, q.layout, q.set_layout)
        };
        let (host, dev) = self.binding_bytes(&[p]);
        let tex = self.texture_upload_estimate(&[p]);
        self.reserve(host + tex, dev);
        self.prepare_guest_reads(&[p]);
        self.begin();
        let (set, _) = self.descriptors_with_retry(set_layout, &[p])?;
        self.end_rendering();
        global_barrier(self.cmd);
        let c = ctx();
        unsafe {
            c.device.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            c.device.cmd_bind_descriptor_sets(self.cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
            c.device.cmd_dispatch(self.cmd, groups, 1, 1);
        }
        self.mark(|| format!("{} {:#x}", if p.tess == TessRole::Ls { "ls" } else { "hs" }, p.addr));
        unordered!(self);
        self.perform_gpu_writebacks();
        self.batch_cmds += 1;
        Ok(())
    }

    // -------------------------------------------------------------------
    // Dispatch

    /// Records a dispatch of `x` x `y` x `z` groups, or with `indirect`, of the
    /// counts at that guest address read by the GPU (see `indirect_args`).
    /// Ok(false): the indirect form is not possible for this dispatch (the
    /// caller then waits for its arguments and dispatches directly).
    fn dispatch_inner(&mut self, cp: &CommandProcessor, x: u32, y: u32, z: u32, indirect: Option<u64>) -> Result<bool, String> {
        let pre = self.pre_cs.take();
        self.draw_targets.clear();
        self.cmd_queue = cp.queue;
        self.drain_cp_writes();
        let cs_addr = ((cp.regs.sh(0x2E0C) as u64) | ((cp.regs.sh(0x2E0D) as u64) << 32)) << 8;
        super::set_phase(super::phase::DISPATCH, cs_addr);
        if SKIP_CS.lock().contains(&cs_addr) {
            self.skipped += 1;
            return Ok(true);
        }
        if trace_frame() {
            self.flush();
        }
        if self.ring.remaining() < self.ring.segment_size() / 4 {
            self.flushes[2] += 1;
            self.submit_batch(Vec::new());
        }
        self.begin();
        self.gpu_wb.clear();
        self.set_required_epoch(cp.queue != 0);
        let tp = std::time::Instant::now();
        let mut cs = match pre {
            Some(p) => p?,
            None => shaders::prepare(&StageRegs { stage: Stage::Compute, regs: &cp.regs })?,
        };
        self.prep_shader = cs.addr;
        self.prep_vs = 0;
        for _ in 0..3 {
            if !self.guard_reads(&cs.mem_reads) {
                break;
            }
            self.begin();
            cs = shaders::prepare(&StageRegs { stage: Stage::Compute, regs: &cp.regs })?;
        }
        self.cur_shader = cs.addr;
        // GPU-read indirect arguments: not for kernels whose handling needs
        // the group counts here (scatter ranges, utility operations) or
        // dispatches from a start group, nor while CPU-performed writes of
        // the arguments are pending (device-ring write-backs, end-of-pipe
        // writes): those wait for their arguments as before.
        if let Some(a) = indirect {
            let start = (cp.regs.sh(0x2E04), cp.regs.sh(0x2E05), cp.regs.sh(0x2E06));
            // (Without the import the arguments come from the device copy of
            // the pending write: `indirect_args`.)
            let pending = ctx().ext_host.is_some() && (self.writebacks.iter().any(|w| w.base < a + 12 && a < w.base + w.size) || cpu_writeback_in_flight(a, 12));
            if cs.scatter.is_some() || cs.utility.is_some() || start != (0, 0, 0) || a % 4 != 0 || pending {
                return Ok(false);
            }
        }
        if let Some((records, dst)) = cs.scatter {
            cs.sparse_writes = self.scatter_ranges(&cs, records, dst, x as u64 * y as u64 * z as u64);
            self.scatters += cs.sparse_writes.is_some() as u64;
        }
        self.prof[5] += tp.elapsed();
        // Engine copy/fill kernels over whole render-target surfaces run as
        // image operations (`tune elide_utility 0` runs them as shaders).
        // (Kernels dispatched from a start group (COMPUTE_START_X/Y/Z) cover
        // only groups [start, x): the elided operations assume group 0 on.)
        let start = (cp.regs.sh(0x2E04), cp.regs.sh(0x2E05), cp.regs.sh(0x2E06));
        if cs.utility.is_some() && start != (0, 0, 0) {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
                crate::info!("utility kernel cs {:#x} dispatched {x}x{y}x{z} from group {start:?}: run as a shader", cs.addr);
            }
        }
        if let Some(u) = cs.utility.filter(|_| start == (0, 0, 0) && crate::tune::get(crate::tune::ELIDE_UTILITY) != 0) {
            if self.elide_utility(&cs, u, x as u64 * y as u64 * z as u64)? {
                self.elided += 1;
                self.dispatches += 1;
                if trace_frame() {
                    let dst = match u {
                        shaders::Utility::Copy { dst, .. } | shaders::Utility::Fill { dst, .. } | shaders::Utility::Pattern { dst, .. } => dst,
                    };
                    let target = match cs.resources.get(dst) {
                        Some(ResolvedRes::Buffer(v)) => format!("{:#x}+{:#x}", v.base, v.num_records as u64 * v.stride.max(1) as u64),
                        _ => "?".into(),
                    };
                    crate::info!("FRAME {} dispatch {x}x{y}x{z} cs {:#x} performed as {}: {u:?} on {target}", cp.name, cs.addr, self.last_elide);
                }
                return Ok(true);
            }
        }
        if trace_frame() {
            let res: Vec<String> = stage_res_summary(&cs);
            let groups = match indirect {
                Some(a) => format!("indirect (arguments at {a:#x}, read on the GPU)"),
                None => format!("{x}x{y}x{z}"),
            };
            let loads = &cs.sgpr_data[(gcn::translate::SGPR_DATA_USER as usize).min(cs.sgpr_data.len())..];
            crate::info!("FRAME {} dispatch {groups} cs {:#x} res [{}] written {:?} sgpr data after user [{:x?}]", cp.name, cs.addr, res.join(", "), cs.written_buffers, &loads[..loads.len().min(48)]);
        }
        let key = cs.translated.key_hash ^ cs.addr;
        if !self.cs_pipes.contains_key(&key) {
            let c = ctx();
            let m = self.module(key, &cs.translated.out.spirv)?;
            let set_layout = self.set_layout(&[&cs])?;
            let layouts = [set_layout];
            let layout = unsafe { c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts), None) }.map_err(|e| e.to_string())?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(m).name(c"main");
            let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout).flags(vk::PipelineCreateFlags::DISPATCH_BASE);
            let t0 = std::time::Instant::now();
            let pipeline = unsafe { c.device.create_compute_pipelines(pipeline_cache(), &[ci], None) }.map_err(|(_, e)| format!("vkCreateComputePipelines: {e}"))?[0];
            pipeline_cache_dirty();
            let ms = t0.elapsed().as_millis();
            if ms >= 100 {
                crate::info!("compute pipeline for cs {:#x} ({} SPIR-V words) took {ms} ms to create", cs.addr, cs.translated.out.spirv.len());
            }
            self.cs_pipes.insert(key, Pipe { pipeline, layout, set_layout });
        }
        let (pipeline, layout, set_layout) = {
            let p = &self.cs_pipes[&key];
            (p.pipeline, p.layout, p.set_layout)
        };
        let (host, dev) = self.binding_bytes(&[&cs]);
        let tex = self.texture_upload_estimate(&[&cs]);
        self.reserve(host + tex, dev);
        self.prepare_guest_reads(&[&cs]);
        self.begin();
        // (Before the descriptors: nothing of the dispatch is recorded yet
        // when this declines.)
        let args_slot = match indirect {
            Some(a) => match self.indirect_args(a)? {
                Some(slot) => Some(slot),
                None => return Ok(false),
            },
            None => None,
        };
        self.cur_unknown = true;
        let (set, wb_before) = self.descriptors_with_retry(set_layout, &[&cs])?;
        self.flush_forwards();
        self.end_rendering();
        // Ordering: behind unordered writes of other GPU work, and behind
        // dispatches since the last barrier whose buffer accesses conflict
        // with this one's (a write on either side of an overlap).
        let hazards = crate::tune::get(crate::tune::DISPATCH_HAZARDS) != 0;
        let bgen = BARRIERS.load(std::sync::atomic::Ordering::Relaxed);
        if self.hz_gen != bgen {
            self.hz.clear();
            self.hz_unknown = false;
            self.hz_gen = bgen;
        }
        let conflict = || {
            self.hz_unknown
                || self.cur_unknown
                || self.cur_access.iter().any(|&(b, o, n, w)| self.hz.iter().any(|&(hb, ho, hn, hw)| hb == b && ho < o.saturating_add(n) && o < ho.saturating_add(hn) && (w || hw)))
        };
        // (The validated arguments must be visible to the indirect read.)
        let need = args_slot.is_some() || (self.barrier_needed && (!hazards || self.unordered_gen == bgen || conflict()));
        if need && perf_on() {
            if args_slot.is_some() {
                self.hz_reasons[0] += 1;
            } else if !hazards || self.unordered_gen == bgen {
                *self.hz_other_lines.entry(if hazards { self.unordered_line } else { 0 }).or_default() += 1;
            } else if self.hz_unknown || self.cur_unknown {
                self.hz_reasons[1] += 1;
            } else {
                self.hz_reasons[2] += 1;
            }
        }
        if need {
            global_barrier(self.cmd);
            self.hz.clear();
            self.hz_unknown = false;
            self.hz_gen = BARRIERS.load(std::sync::atomic::Ordering::Relaxed);
            self.hz_stats.0 += 1;
        } else if self.barrier_needed {
            self.hz_stats.1 += 1;
        }
        if hazards {
            let acc = std::mem::take(&mut self.cur_access);
            self.hz.extend(acc.iter().copied());
            self.cur_access = acc;
            self.hz_unknown |= self.cur_unknown;
        }
        let c = ctx();
        let regs = &cp.regs;
        let (sx, sy, sz) = (regs.sh(0x2E04), regs.sh(0x2E05), regs.sh(0x2E06));
        unsafe {
            c.device.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            c.device.cmd_bind_descriptor_sets(self.cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
            if let Some(slot) = args_slot {
                c.device.cmd_dispatch_indirect(self.cmd, self.dring.buffer, slot);
            } else if sx | sy | sz != 0 {
                c.device.cmd_dispatch_base(self.cmd, sx, sy, sz, x.saturating_sub(sx), y.saturating_sub(sy), z.saturating_sub(sz));
            } else {
                c.device.cmd_dispatch(self.cmd, x, y, z);
            }
        }
        self.mark(|| format!("cs {:#x}", cs.addr));
        self.barrier_needed = true; // (its accesses: `hz`)
        self.perform_gpu_writebacks();
        self.dispatches += 1;
        self.batch_cmds += 1;
        if self.writebacks.len() + self.zc_writes.len() > wb_before {
            self.epoch += 1;
        }
        if trace_frame() {
            self.trace_timing("dispatch");
        }
        Ok(true)
    }

    /// GPU buffer writes to [base, base+size) overlapping the guest surface
    /// of a rendered image: guest memory is now newer than the image.
    /// (Callers skip bindings larger than SURFACE_WRITE_MAX.)
    fn note_buffer_write(&mut self, base: u64, size: u64) {
        // (Images starting up to IMG_NEAR_BYTES below the write, and the
        // large ones further below; one found twice is marked twice.)
        let near = self.img_bases.range(base.saturating_sub(self.img_max_bytes.min(IMG_NEAR_BYTES))..base + size);
        let far = self.img_bases_large.range(base.saturating_sub(self.img_max_bytes)..base + size);
        for (k, _) in near.chain(far).flat_map(|(_, ks)| ks.iter()).filter(|(k, bound)| k.base + bound > base) {
            let Some(img) = self.images.get_mut(k) else { continue };
            if !img.gpu_written || k.ty3d {
                continue;
            }
            let Some((_, _, _, total)) = Self::image_guest_layout(k, img) else { continue };
            if k.base < base + size && base < k.base + total {
                if perf_on() && !img.guest_newer {
                    let s = format!("{:#x} wrote {base:#x}+{size:#x} over {:#x} {}x{}x{} {:?}", self.cur_shader, k.base, k.width, k.height, k.layers, k.format);
                    *NEWER_SITES.lock().get_or_insert_with(HashMap::new).entry(s).or_default() += 1;
                }
                if trace_frame() && !img.guest_newer {
                    crate::info!("FRAME newer {:#x} {}x{}x{} {:?} (guest {:#x}+{total:#x}): {:#x} wrote {base:#x}+{size:#x}", k.base, k.width, k.height, k.layers, k.format, k.base, self.cur_shader);
                }
                img.guest_newer = true;
                img.guest_stale = false;
            }
        }
    }

    /// GPU upload of mip chains and block-compressed textures (32/64/128-bit
    /// elements: BC blocks are 64/128-bit elements on a quarter grid; not
    /// 3D): per level and layer, the remap kernel detiles the surface in
    /// place from guest memory into the device ring (each level's own tile
    /// map: small levels switch to 1D tiling), then one buffer-to-image
    /// copy per level. As in `gpu_texture_validate`, write tracking decides
    /// (no hash, no CPU read). Detiled on the CPU, streamed textures cost
    /// the GPU thread 3-10 ms in a frame. Returns false for the CPU path
    /// (`TLG_GPU_MIPS=0` always). `TLG_GPU_TEX_VERIFY=<n>`: the first n uploads
    /// are read back at the next flip and compared byte by byte with the CPU
    /// detiler's result (`verify_gpu_textures`; slow).
    fn gpu_texture_levels(&mut self, key: ImageKey, t: &Tsharp, batch: u64) -> Result<bool, String> {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var("TLG_GPU_MIPS").map(|v| v != "0").unwrap_or(true)) {
            return Ok(false);
        }
        let (bpp, block) = resources::element_bits(t.dfmt);
        // The image's element must be the guest's: texel size, or for block
        // formats the block size (format_class has no class for them).
        let element_ok = if block == 1 { format_class(key.format) * 8 == bpp } else { bc_block_bits(key.format) == Some(bpp) };
        if !Self::gpu_remap_eligible(bpp) || key.ty3d || !element_ok || crate::tune::get(crate::tune::GPU_TEXTURES) == 0 {
            mip_stat(1);
            return Ok(false);
        }
        let layers = key.layers.max(1);
        let bytes = resources::texture_bytes(t, layers);
        let since = self.images[&key].validated;
        let (written, stamp) = self.writes.changed_since(t.base, bytes, since.unwrap_or(0));
        if !written && since.is_some() {
            let cpu_epoch = self.cpu_epoch;
            let img = self.images.get_mut(&key).unwrap();
            img.upload_batch = batch;
            img.validated = Some(stamp);
            img.cpu_checked = cpu_epoch;
            return Ok(true);
        }
        if self.writebacks.iter().any(|w| w.base < t.base + bytes && t.base < w.base + w.size) {
            mip_stat(2);
            return Ok(false);
        }
        let levels: Vec<tiling::MipLevel> = resources::texture_layout(t, layers).into_iter().take(key.levels.max(1) as usize).collect();
        let eb = (bpp / 8) as u64;
        let lin_sizes: Vec<u64> = levels.iter().map(|l| l.width as u64 * l.height as u64 * eb * layers as u64).collect();
        let lin_total: u64 = lin_sizes.iter().sum();
        if levels.iter().any(|l| l.slice_bytes % 4 != 0 || l.offset + l.slice_bytes * layers as u64 > bytes) {
            mip_stat(3);
            return Ok(false);
        }
        // Tile maps first (their uploads are recorded in this batch too).
        let mut maps = Vec::with_capacity(levels.len());
        for l in &levels {
            let mode_key = 0x80 | ((l.mode.array as u8) << 4) | l.mode.micro as u8;
            let Some(m) = self.tile_map_raw(l.width, l.height, l.pitch, bpp, l.mode, mode_key, l.slice_bytes)? else {
                mip_stat(5);
                return Ok(false);
            };
            maps.push(m);
        }
        if cpu_writeback_in_flight(t.base, bytes) {
            self.flushes[1] += 1;
            let t0 = std::time::Instant::now();
            wait_cpu_writebacks(t.base, bytes);
            self.prof[0] += t0.elapsed();
        }
        let Some(lin_off) = self.dring_for_detile(lin_total)? else {
            mip_stat(6);
            return Ok(false);
        };
        let Some((zbuf, zoff)) = self.guest_surface_source(t.base, bytes)? else {
            mip_stat(4);
            return Ok(false);
        };
        let (set_layout, layout, detile, _) = self.remap_pipes()?;
        self.begin();
        self.end_rendering();
        global_barrier(self.cmd);
        let c = ctx();
        let mut lin_at = 0u64;
        let cmd_batch = self.batch;
        for (i, l) in levels.iter().enumerate() {
            let (map, n) = maps[i];
            let set = self.alloc_set(set_layout)?;
            if self.batch != cmd_batch {
                // The descriptor pool ran out and the batch was submitted:
                // `lin_off` belongs to the previous ring segment.
                mip_stat(7);
                return Ok(false);
            }
            let infos = [
                vk::DescriptorBufferInfo { buffer: zbuf, offset: zoff, range: bytes.next_multiple_of(4) },
                vk::DescriptorBufferInfo { buffer: map, offset: 0, range: n * 4 },
                vk::DescriptorBufferInfo { buffer: self.dring.buffer, offset: lin_off, range: lin_total },
            ];
            let writes: Vec<vk::WriteDescriptorSet> = (0..3)
                .map(|k| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(k as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&infos[k])))
                .collect();
            let per_layer = lin_sizes[i] / layers as u64;
            unsafe {
                c.device.update_descriptor_sets(&writes, &[]);
                c.device.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, detile);
                c.device.cmd_bind_descriptor_sets(self.cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
                for s in 0..layers as u64 {
                    let pc = [n as u32, bpp / 32, ((l.offset + s * l.slice_bytes) / 4) as u32, ((lin_at + s * per_layer) / 4) as u32];
                    let pcb: Vec<u8> = pc.iter().flat_map(|v| v.to_le_bytes()).collect();
                    c.device.cmd_push_constants(self.cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, &pcb);
                    c.device.cmd_dispatch(self.cmd, (n as u32).div_ceil(64), 1, 1);
                }
            }
            lin_at += lin_sizes[i];
        }
        global_barrier(self.cmd);
        if ctx().ext_host.is_some() {
            self.zc_read(t.base, t.base + bytes);
        }
        static VERIFY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        static QUEUED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let verify = *VERIFY.get_or_init(|| std::env::var("TLG_GPU_TEX_VERIFY").ok().and_then(|v| v.parse().ok()).unwrap_or(0));
        if verify > 0 && QUEUED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < verify {
            VERIFY_QUEUE.lock().push((key, *t, layers));
        }
        let mut regions: smallvec::SmallVec<[vk::BufferImageCopy; 16]> = smallvec::SmallVec::new();
        let mut pos = lin_off;
        for (i, l) in levels.iter().enumerate() {
            regions.push(vk::BufferImageCopy {
                buffer_offset: pos,
                buffer_row_length: l.width * block,
                buffer_image_height: l.height * block,
                image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: i as u32, base_array_layer: 0, layer_count: layers },
                image_offset: vk::Offset3D::default(),
                image_extent: vk::Extent3D { width: (t.width >> i).max(1), height: (t.height >> i).max(1), depth: 1 },
            });
            pos += lin_sizes[i];
        }
        unsafe { c.device.cmd_copy_buffer_to_image(self.cmd, self.dring.buffer, self.images[&key].image, vk::ImageLayout::GENERAL, &regions) };
        unordered!(self);
        self.uploads += 1;
        self.gpu_tex_uploads += 1;
        self.mark(|| "texture detiling, mip chain (GPU)".into());
        mip_stat(0);
        let cpu_epoch = self.cpu_epoch;
        let img = self.images.get_mut(&key).unwrap();
        img.upload_batch = batch;
        img.validated = Some(stamp);
        img.cpu_checked = cpu_epoch;
        Ok(true)
    }

    /// `TLG_GPU_TEX_VERIFY`: reads back the images `gpu_texture_levels`
    /// uploaded (all levels and layers) and compares them with the CPU
    /// detiler's result for the same guest memory. Called at flips (no draw
    /// is being prepared: it submits and waits).
    pub fn verify_gpu_textures(&mut self) {
        let pending: Vec<(ImageKey, Tsharp, u32)> = std::mem::take(&mut *VERIFY_QUEUE.lock());
        for (key, t, layers) in pending {
            let Some(image) = self.images.get(&key).map(|i| i.image) else { continue };
            let (bpp, block) = resources::element_bits(t.dfmt);
            let eb = (bpp / 8) as u64;
            let levels: Vec<tiling::MipLevel> = resources::texture_layout(&t, layers).into_iter().take(key.levels.max(1) as usize).collect();
            let total: u64 = levels.iter().map(|l| l.width as u64 * l.height as u64 * eb * layers as u64).sum();
            self.begin();
            self.end_rendering();
            global_barrier(self.cmd);
            let Some(host) = (self.ring.remaining() >= total + 4096).then(|| self.ring.alloc(total)).flatten() else { continue };
            let mut regions: smallvec::SmallVec<[vk::BufferImageCopy; 16]> = smallvec::SmallVec::new();
            let mut pos = host;
            for (i, l) in levels.iter().enumerate() {
                regions.push(vk::BufferImageCopy {
                    buffer_offset: pos,
                    buffer_row_length: l.width * block,
                    buffer_image_height: l.height * block,
                    image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: i as u32, base_array_layer: 0, layer_count: layers },
                    image_offset: vk::Offset3D::default(),
                    image_extent: vk::Extent3D { width: (t.width >> i).max(1), height: (t.height >> i).max(1), depth: 1 },
                });
                pos += l.width as u64 * l.height as u64 * eb * layers as u64;
            }
            unsafe { ctx().device.cmd_copy_image_to_buffer(self.cmd, image, vk::ImageLayout::GENERAL, self.ring.buffer, &regions) };
            let ring_ptr = self.ring.ptr;
            self.flush();
            let gpu = unsafe { std::slice::from_raw_parts(ring_ptr.add(host as usize), total as usize) };
            let cpu = resources::read_texture(&t, layers);
            let n = (total as usize).min(cpu.len());
            let bad = gpu[..n].iter().zip(&cpu[..n]).filter(|(a, b)| a != b).count();
            let first = gpu[..n].iter().zip(&cpu[..n]).position(|(a, b)| a != b);
            crate::info!(
                "GPU mip detiling check {:#x} {}x{}x{} levels {} dfmt {} tiling {}: {} of {} bytes differ{}",
                t.base,
                t.width,
                t.height,
                layers,
                levels.len(),
                t.dfmt,
                t.tiling_index,
                bad,
                n,
                first.map(|p| format!(" (first at byte {p})")).unwrap_or_default()
            );
        }
    }

    /// Whether render targets with `bpp`-bit elements can be (de)tiled on the
    /// GPU (whole-dword elements; the guest surface is accessed in place).
    /// TLG_GPU_TILING=0 keeps the CPU paths.
    fn gpu_remap_eligible(bpp: u32) -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        matches!(bpp, 32 | 64 | 128) && *ON.get_or_init(|| std::env::var("TLG_GPU_TILING").map(|v| v != "0").unwrap_or(true))
    }

    /// Where GPU detiling reads guest surface [base, base+bytes): the
    /// imported guest memory in place, or without the import a device-ring
    /// copy staged now (pending GPU writes forwarded, see `upload_guest`).
    /// None: not possible (the caller detiles on the CPU).
    fn guest_surface_source(&mut self, base: u64, bytes: u64) -> Result<Option<(vk::Buffer, u64)>, String> {
        if ctx().ext_host.is_some() {
            let Some((zbuf, zoff, zrange)) = self.zc_binding(base, bytes) else { return Ok(None) };
            if zrange < bytes || zoff % ctx().limits.min_storage_buffer_offset_alignment.max(4) != 0 {
                return Ok(None);
            }
            return Ok(Some((zbuf, zoff)));
        }
        let size = bytes.next_multiple_of(4);
        // The VRAM mirror: current pages in place (GPU-written surfaces need
        // no copy at all), others copied in.
        if (VCACHE_MIN..=VCACHE_MAX).contains(&size) && base % 16 == 0 {
            self.vc_why = "surface detile";
            if let Some(m) = self.vcache_bind(base, size) {
                return Ok(Some(m));
            }
        }
        if self.ring.remaining() < size + 4096 || self.dring.remaining() < size + 4096 {
            return Ok(None);
        }
        let batch = self.batch;
        self.begin();
        let up = self.upload_guest(base, size)?;
        self.flush_forwards();
        let Some(dev) = self.dring.alloc(size) else { return Ok(None) };
        if self.batch != batch {
            return Ok(None);
        }
        self.end_rendering();
        global_barrier(self.cmd);
        unsafe { ctx().device.cmd_copy_buffer(self.cmd, self.ring.buffer, self.dring.buffer, &[vk::BufferCopy { src_offset: up, dst_offset: dev, size }]) };
        unordered!(self);
        Ok(Some((self.dring.buffer, dev)))
    }

    /// Device-ring space for GPU detiling `bytes` of a texture. When the
    /// current segment lacks it but an empty one has it, a "ring full" error
    /// makes descriptor writing retry the command once in a new batch (or the
    /// early validation submit first) instead of declining to the CPU
    /// detiler (~150 ms for a 64 MiB shadow map). None: decline.
    fn dring_for_detile(&mut self, bytes: u64) -> Result<Option<u64>, String> {
        if self.dring.remaining() >= bytes + 4096 {
            return Ok(self.dring.alloc(bytes));
        }
        if !self.desc_retry && bytes + 4096 <= self.dring.segment_size() && self.dring.remaining() < self.dring.segment_size() {
            return Err(format!("ring full: {bytes:#x} bytes of the device ring for GPU detiling"));
        }
        Ok(None)
    }

    /// The GPU (de)tiling pipelines (created on first use).
    fn remap_pipes(&mut self) -> Result<(vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline, vk::Pipeline), String> {
        if let Some(r) = self.remap {
            return Ok(r);
        }
        let c = ctx();
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..3)
            .map(|i| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
            .collect();
        let set_layout = unsafe { c.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None) }.map_err(|e| e.to_string())?;
        let pcr = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 16 }];
        let sls = [set_layout];
        let layout = unsafe { c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sls).push_constant_ranges(&pcr), None) }.map_err(|e| e.to_string())?;
        let mut pipes = [vk::Pipeline::null(); 2];
        for (k, tile) in [false, true].into_iter().enumerate() {
            let m = self.module(0x5245_4D41_5000_0000 | tile as u64, &gcn::translate::kernels::remap_cs(tile))?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(m).name(c"main");
            let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
            pipes[k] = unsafe { c.device.create_compute_pipelines(vk::PipelineCache::null(), &[ci], None) }.map_err(|(_, e)| format!("remap pipeline: {e}"))?[0];
        }
        self.remap = Some((set_layout, layout, pipes[0], pipes[1]));
        Ok(self.remap.unwrap())
    }

    /// The selective write-back pipeline (`kernels::dirty_copy_cs`) and the
    /// pool of its descriptor sets (created on first use).
    fn dirty_pipe(&mut self) -> Result<(vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline, vk::DescriptorPool), String> {
        if let Some(p) = self.dirty_pipe {
            return Ok(p);
        }
        let c = ctx();
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..3)
            .map(|i| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
            .collect();
        let set_layout = unsafe { c.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None) }.map_err(|e| e.to_string())?;
        let pcr = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 24 }];
        let sls = [set_layout];
        let layout = unsafe { c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sls).push_constant_ranges(&pcr), None) }.map_err(|e| e.to_string())?;
        let m = self.module(0x4449_5254_5900_0000, &gcn::translate::kernels::dirty_copy_cs())?;
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(m).name(c"main");
        let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
        let pipeline = unsafe { c.device.create_compute_pipelines(vk::PipelineCache::null(), &[ci], None) }.map_err(|(_, e)| format!("dirty copy pipeline: {e}"))?[0];
        let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 3 * 256 }];
        let pool = unsafe { c.device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET).max_sets(256).pool_sizes(&sizes), None) }.map_err(|e| e.to_string())?;
        self.dirty_pipe = Some((set_layout, layout, pipeline, pool));
        Ok(self.dirty_pipe.unwrap())
    }

    /// The page copy's descriptor set for source buffer `src` (whole
    /// buffers: `src`, the host ring, the device ring).
    fn dirty_set(&mut self, src: vk::Buffer) -> Result<vk::DescriptorSet, String> {
        // (Kept only for live mirror windows: a retired window's handle can
        // be reused by a new buffer once it is destroyed.)
        let live = self.vcache.as_ref().is_some_and(|vc| vc.windows.values().any(|w| w.buf == src));
        if let Some(&s) = self.dirty_sets.get(&src).filter(|_| live) {
            return Ok(s);
        }
        let (set_layout, _, _, pool) = self.dirty_pipe()?;
        let c = ctx();
        let sls = [set_layout];
        let set = unsafe { c.device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&sls)) }.map_err(|e| format!("dirty copy set: {e}"))?[0];
        let infos = [src, self.ring.buffer, self.dring.buffer].map(|buffer| vk::DescriptorBufferInfo { buffer, offset: 0, range: vk::WHOLE_SIZE });
        let writes: Vec<vk::WriteDescriptorSet> = (0..3)
            .map(|i| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&infos[i])))
            .collect();
        unsafe { c.device.update_descriptor_sets(&writes, &[]) };
        if live {
            self.dirty_sets.insert(src, set);
        } else {
            self.retired.push((self.last_seq + 1, Retired::Set(pool, set)));
        }
        Ok(set)
    }

    /// Releases the page copy's set of a buffer being retired (after batch
    /// `seq`; 0: now, the GPU is idle).
    fn retire_dirty_set(&mut self, buf: vk::Buffer, seq: u64) {
        let (Some(set), Some((_, _, _, pool))) = (self.dirty_sets.remove(&buf), self.dirty_pipe) else { return };
        if seq == 0 {
            unsafe { ctx().device.free_descriptor_sets(pool, &[set]).ok() };
        } else {
            self.retired.push((seq, Retired::Set(pool, set)));
        }
    }

    /// Records the selective copy-out of the write-backs that have a page
    /// bitmap: their marked pages (device -> host ring) and the bitmaps.
    fn record_dirty_copies(&mut self, pending: &[PendingWrite]) -> Result<(), String> {
        let (_, layout, pipeline, _) = self.dirty_pipe()?;
        let c = ctx();
        let mut bitmaps = Vec::new();
        unsafe { c.device.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, pipeline) };
        for p in pending {
            let Some((bits, bits_host, words)) = p.bits else { continue };
            let set = self.dirty_set(p.dev_buf)?;
            let head = p.base & 0xFFF;
            let pages = (head + p.size).div_ceil(4096) as u32;
            let pc = [pages, head as u32, p.size as u32, (p.dev_off / 4) as u32, (p.host_off / 4) as u32, (bits / 4) as u32];
            let bytes: Vec<u8> = pc.iter().flat_map(|v| v.to_le_bytes()).collect();
            unsafe {
                c.device.cmd_bind_descriptor_sets(self.cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
                c.device.cmd_push_constants(self.cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, &bytes);
                c.device.cmd_dispatch(self.cmd, pages, 1, 1);
            }
            bitmaps.push(vk::BufferCopy { src_offset: bits, dst_offset: bits_host, size: words as u64 * 4 });
        }
        if !bitmaps.is_empty() {
            unsafe { c.device.cmd_copy_buffer(self.cmd, self.dring.buffer, self.ring.buffer, &bitmaps) };
        }
        self.bound_gfx = vk::Pipeline::null();
        Ok(())
    }

    /// The indirect-argument validation pipeline (`kernels::indirect_args_cs`)
    /// and its diagnostics buffer (created on first use).
    fn indirect_pipe(&mut self) -> Result<(vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline), String> {
        if let Some(p) = self.indirect_pipe {
            return Ok(p);
        }
        let c = ctx();
        let (dbuf, dmem, dptr) = c.host_buffer(16, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        unsafe { std::ptr::write_bytes(dptr, 0, 16) };
        self.indirect_diag = Some((dbuf, dmem, dptr as usize));
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..3)
            .map(|i| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
            .collect();
        let set_layout = unsafe { c.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None) }.map_err(|e| e.to_string())?;
        let pcr = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 20 }];
        let sls = [set_layout];
        let layout = unsafe { c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sls).push_constant_ranges(&pcr), None) }.map_err(|e| e.to_string())?;
        let m = self.module(0x494E_4449_5200_0000, &gcn::translate::kernels::indirect_args_cs())?;
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(m).name(c"main");
        let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
        let pipeline = unsafe { c.device.create_compute_pipelines(vk::PipelineCache::null(), &[ci], None) }.map_err(|(_, e)| format!("indirect argument pipeline: {e}"))?[0];
        self.indirect_pipe = Some((set_layout, layout, pipeline));
        Ok((set_layout, layout, pipeline))
    }

    /// DISPATCH_INDIRECT without waiting for the work that wrote its
    /// arguments: a one-invocation kernel reads them from guest memory in
    /// place (in queue order, after a barrier: the writers' in-place stores
    /// precede it) and writes them validated (`kernels::indirect_args_cs`:
    /// implausible counts dispatch nothing, counted for the log) to a
    /// device-ring slot for vkCmdDispatchIndirect. Returns the slot, or None
    /// when not possible here (memory not importable, descriptor pool or
    /// device ring exhausted).
    fn indirect_args(&mut self, addr: u64) -> Result<Option<u64>, String> {
        let c = ctx();
        // Rejections since the last look (the kernel counts them).
        if let Some((_, _, p)) = self.indirect_diag {
            let d = |i: usize| unsafe { std::ptr::read_volatile((p as *const u32).add(i)) };
            let n = d(0);
            if n != self.indirect_rejected {
                if n <= 16 || n.is_power_of_two() {
                    crate::error!("DISPATCH_INDIRECT: {n} dispatches with implausible arguments skipped on the GPU so far (last {}x{}x{})", d(1), d(2), d(3));
                }
                self.indirect_rejected = n;
            }
        }
        let align = c.limits.min_storage_buffer_offset_alignment.max(16);
        let (sbuf, soff, span, first) = if ctx().ext_host.is_some() {
            let base = addr & !(align - 1);
            let span = addr + 12 - base;
            let Some((sbuf, soff, srange)) = self.zc_binding(base, span) else { return Ok(None) };
            if srange < span || soff % align != 0 {
                return Ok(None);
            }
            (sbuf, soff, span, ((addr - base) / 4) as u32)
        } else {
            // Without the import: the device copy of the GPU write that
            // produced the arguments (none pending: the CP reads them itself
            // without waiting).
            let Some((buf, off)) = self.device_copy_of(addr, 12) else { return Ok(None) };
            let at = off & !(align - 1);
            (buf, at, off + 12 - at, ((off - at) / 4) as u32)
        };
        let (set_layout, layout, pipeline) = self.indirect_pipe()?;
        let Some((diag, _, _)) = self.indirect_diag else { return Ok(None) };
        let batch = self.batch;
        let set = self.alloc_set(set_layout)?;
        if self.batch != batch {
            return Ok(None);
        }
        let Some(slot) = self.dring.alloc(16) else { return Ok(None) };
        let infos = [
            vk::DescriptorBufferInfo { buffer: sbuf, offset: soff, range: span },
            vk::DescriptorBufferInfo { buffer: self.dring.buffer, offset: slot, range: 16 },
            vk::DescriptorBufferInfo { buffer: diag, offset: 0, range: 16 },
        ];
        let writes: Vec<vk::WriteDescriptorSet> = (0..3)
            .map(|i| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&infos[i])))
            .collect();
        let lim = c.limits.max_compute_work_group_count;
        let pc = [first, lim[0], lim[1], lim[2], 1 << 22];
        let bytes: Vec<u8> = pc.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.begin();
        self.end_rendering();
        global_barrier(self.cmd);
        unsafe {
            c.device.update_descriptor_sets(&writes, &[]);
            c.device.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            c.device.cmd_bind_descriptor_sets(self.cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
            c.device.cmd_push_constants(self.cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, &bytes);
            c.device.cmd_dispatch(self.cmd, 1, 1, 1);
        }
        unordered!(self);
        // (Read in place: CP writes of the arguments wait for this batch.)
        if ctx().ext_host.is_some() {
            self.zc_read(addr, addr + 12);
        }
        self.mark(|| "indirect dispatch arguments".into());
        Ok(Some(slot))
    }

    fn indirect_draw_pipe(&mut self) -> Result<(vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline), String> {
        if let Some(p) = self.indirect_draw_pipe {
            return Ok(p);
        }
        let c = ctx();
        let (dbuf, dmem, dptr) = c.host_buffer(16, vk::BufferUsageFlags::STORAGE_BUFFER)?;
        unsafe { std::ptr::write_bytes(dptr, 0, 16) };
        self.indirect_draw_diag = Some((dbuf, dmem, dptr as usize));
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..4)
            .map(|i| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
            .collect();
        let set_layout = unsafe { c.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None) }.map_err(|e| e.to_string())?;
        let pcr = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 32 }];
        let sls = [set_layout];
        let layout = unsafe { c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sls).push_constant_ranges(&pcr), None) }.map_err(|e| e.to_string())?;
        let m = self.module(0x494E_4452_4157_0000, &gcn::translate::kernels::indirect_draw_cs())?;
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(m).name(c"main");
        let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
        let pipeline = unsafe { c.device.create_compute_pipelines(vk::PipelineCache::null(), &[ci], None) }.map_err(|(_, e)| format!("indirect draw pipeline: {e}"))?[0];
        self.indirect_draw_pipe = Some((set_layout, layout, pipeline));
        Ok((set_layout, layout, pipeline))
    }

    /// An indirect draw whose arguments at guest `addr` earlier GPU work is
    /// still writing (this batch's write-backs, batches in flight): recorded
    /// with the arguments read on the GPU from that work's device copy
    /// (`kernels::indirect_draw_cs` turns them into the Vulkan command and
    /// the vertex stage's offsets, see `indirect_draw_fixup`) instead of the
    /// CP waiting for them to land in guest memory. False (nothing recorded)
    /// when the arguments are not pending on the GPU (the CP reads them
    /// without waiting) or the draw needs the CP's path: tessellation, rect
    /// lists, offsets in registers outside the vertex stage's user data, no
    /// device copy covering the arguments.
    fn draw_indirect_gpu(&mut self, cp: &CommandProcessor, addr: u64, indexed: bool, vtx_loc: u32, inst_loc: u32) -> Result<bool, String> {
        let len = if indexed { 20 } else { 16 };
        if !self.pending_write_overlaps(addr, len) && !writeback_in_flight(addr, len) {
            return Ok(false);
        }
        let prim = cp.regs.uconfig[(0xC242 - 0xC000) as usize];
        if !matches!(prim, 1..=6) {
            return Ok(false);
        }
        // SPI_SHADER_USER_DATA_VS_0..15 (SH offsets 0x4C..0x5B): the vertex
        // stage's user SGPRs, dwords 0..15 of its SGPR data.
        let user = |loc: u32| match loc {
            0 => Some(None),
            0x4C..=0x5B => Some(Some(loc - 0x4C)),
            _ => None,
        };
        let (Some(vtx), Some(inst)) = (user(vtx_loc), user(inst_loc)) else { return Ok(false) };
        let isz = if cp.index_type == 1 { 4 } else { 2 };
        // (The index binding covers the whole index buffer: small ones only.)
        if indexed && (cp.index_size == 0 || cp.index_size as u64 * isz > 1 << 20 || cp.index_base % isz != 0) {
            return Ok(false);
        }
        let Some((buf, off)) = self.device_copy_of(addr, len) else { return Ok(false) };
        if off % 4 != 0 {
            return Ok(false);
        }
        self.indirect_draw = Some(IndirectDraw { buf, off, indexed, vtx, inst, indices: cp.index_size });
        // (The index binding covers the whole index buffer.)
        let r = self.draw_inner(cp, if indexed { "index2" } else { "auto" }, if indexed { cp.index_size } else { 0 }, cp.index_base);
        self.indirect_draw = None;
        r.map(|_| true)
    }

    /// Records the indirect draw kernel for `d` (after the work that wrote
    /// the arguments, before the pass): the Vulkan draw command goes to a
    /// device-ring slot (returned), the base vertex / start instance into the
    /// vertex stage's SGPR data. `vofs`: VGT_INDX_OFFSET (the vertex offset
    /// of indexed draws).
    fn indirect_draw_fixup(&mut self, d: &IndirectDraw, set: vk::DescriptorSet, vofs: u32) -> Result<u64, String> {
        let c = ctx();
        // Rejections since the last look (the kernel counts them).
        if let Some((_, _, p)) = self.indirect_draw_diag {
            let r = |i: usize| unsafe { std::ptr::read_volatile((p as *const u32).add(i)) };
            let n = r(0);
            if n != self.indirect_draw_rejected {
                if n <= 16 || n.is_power_of_two() {
                    crate::error!("indirect draws: {n} with implausible arguments or past their index buffer drawn as nothing on the GPU so far (last count {} instances {} start {})", r(1), r(2), r(3));
                }
                self.indirect_draw_rejected = n;
            }
        }
        let (_, layout, pipeline) = self.indirect_draw_pipe()?;
        let Some((diag, _, _)) = self.indirect_draw_diag else { return Err("indirect draw diagnostics missing".into()) };
        let Some((sbuf, soff, slen)) = self.vs_sgpr else { return Err("indirect draw without vertex stage SGPR data".into()) };
        let align = c.limits.min_storage_buffer_offset_alignment.max(16);
        let len = if d.indexed { 20 } else { 16 };
        let at = d.off & !(align - 1);
        // (The user dwords are the SGPR data's first 16.)
        let field = |k: Option<u32>| k.filter(|&k| (k as u64 + 1) * 4 <= slen).unwrap_or(u32::MAX);
        let Some(slot) = self.dring.alloc(20) else { return Err("device ring full: indirect draw command".into()) };
        let infos = [
            vk::DescriptorBufferInfo { buffer: d.buf, offset: at, range: d.off + len - at },
            vk::DescriptorBufferInfo { buffer: self.dring.buffer, offset: slot, range: 20 },
            vk::DescriptorBufferInfo { buffer: sbuf, offset: soff, range: slen },
            vk::DescriptorBufferInfo { buffer: diag, offset: 0, range: 16 },
        ];
        let writes: Vec<vk::WriteDescriptorSet> = (0..4)
            .map(|i| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&infos[i])))
            .collect();
        let pc = [((d.off - at) / 4) as u32, d.indexed as u32, field(d.vtx), field(d.inst), d.indices, 1 << 26, 1 << 20, vofs];
        let bytes: Vec<u8> = pc.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.end_rendering();
        // (After the writers of the arguments; before the draw's reads.)
        global_barrier(self.cmd);
        unsafe {
            c.device.update_descriptor_sets(&writes, &[]);
            c.device.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            c.device.cmd_bind_descriptor_sets(self.cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
            c.device.cmd_push_constants(self.cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, &bytes);
            c.device.cmd_dispatch(self.cmd, 1, 1, 1);
        }
        global_barrier(self.cmd);
        self.mark(|| "indirect draw arguments".into());
        Ok(slot)
    }

    /// Where the newest GPU-written copy of guest [addr, addr+len) is on the
    /// device while it has not reached guest memory: a write-back of this
    /// batch or of a batch in flight (device ring segments stay intact until
    /// their slot is reused, mirror windows while they live) that covers it
    /// entirely. None when no such write is pending (or one covers it only
    /// partly).
    fn device_copy_of(&self, addr: u64, len: u64) -> Option<(vk::Buffer, u64)> {
        let overlaps = |b: u64, n: u64| b < addr + len && addr < b + n;
        let covers = |b: u64, n: u64| b <= addr && addr + len <= b + n;
        if let Some(p) = self.writebacks.iter().rev().find(|p| overlaps(p.base, p.size)) {
            return covers(p.base, p.size).then(|| (p.dev_buf, p.dev_off + (addr - p.base)));
        }
        let oldest = (self.last_seq + 2).saturating_sub(SLOTS as u64);
        let live = |b: vk::Buffer| b == self.dring.buffer || self.vcache.as_ref().is_some_and(|vc| vc.windows.values().any(|w| w.buf == b));
        for c in COMPLETIONS.lock().iter().rev() {
            // (CPU-performed end-of-pipe writes of it in flight: no device copy.)
            if c.actions.iter().any(|a| a.overlaps(addr, len)) {
                return None;
            }
            if let Some(&(b, buf, off, n)) = c.dev_writes.iter().rev().find(|&&(b, _, _, n)| overlaps(b, n)) {
                return (c.seq >= oldest && c.fence != vk::Fence::null() && covers(b, n) && live(buf)).then(|| (buf, off + (addr - b)));
            }
        }
        None
    }

    /// Element map of a render target's guest layout (built on first use).
    fn tile_map(&mut self, key: &ImageKey, bpp: u32, mode: tiling::TileMode, idx: u8, slice: u64) -> Result<Option<(vk::Buffer, u64)>, String> {
        // Padded pitch of the guest surface (textures may have one wider
        // than the image).
        let pitch = tiling::padded(self.images.get(key).map_or(0, |i| i.pitch).max(key.width), key.height, bpp, mode).0;
        self.tile_map_raw(key.width, key.height, pitch, bpp, mode, idx, slice)
    }

    /// The element map of a tiled surface of `width` x `height` elements
    /// (padded `pitch`) and `slice` bytes as a device buffer, cached by
    /// layout (`mode_key`: the tile index, or 0x80 + the tile mode for mip
    /// levels, whose mode can differ from the index's): (buffer, elements).
    #[allow(clippy::too_many_arguments)]
    fn tile_map_raw(&mut self, width: u32, height: u32, pitch: u32, bpp: u32, mode: tiling::TileMode, mode_key: u8, slice: u64) -> Result<Option<(vk::Buffer, u64)>, String> {
        let mk = (width, height, bpp, mode_key, slice, pitch);
        if let Some(&m) = self.tile_maps.get(&mk) {
            return Ok(Some(m));
        }
        let n = slice / (bpp / 8) as u64;
        let map = tiling::linear_index_map(width, height, pitch, bpp, mode, n as usize, gcn::translate::kernels::NO_ELEMENT);
        let bytes: Vec<u8> = map.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (buf, mem) = ctx().device_buffer(bytes.len() as u64, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)?;
        let ring_off = (self.ring.remaining() >= bytes.len() as u64 + 4096).then(|| self.ring.alloc(bytes.len() as u64)).flatten();
        if let Some(off) = ring_off {
            self.ring.write(off, &bytes);
        }
        self.begin();
        self.end_rendering();
        global_barrier(self.cmd);
        let (src, src_off) = match ring_off {
            Some(off) => (self.ring.buffer, off),
            None => {
                // Larger than the upload ring's segment (a 4096x4096 32-bit
                // surface's map is 64 MiB): a staging buffer of its own,
                // freed once this batch has completed. (Declining here sent
                // such a surface through the CPU detiler on every update:
                // ~85 ms each, ~8 s of GPU thread time while loading a save.)
                let staged = ctx().host_buffer(bytes.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC);
                let Ok((sb, sm, sp)) = staged else {
                    unsafe {
                        ctx().device.destroy_buffer(buf, None);
                        ctx().free(mem);
                    }
                    return Ok(None);
                };
                unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), sp, bytes.len()) };
                self.retired.push((self.last_seq + 1, Retired::Buffer(sb, sm)));
                (sb, 0)
            }
        };
        unsafe { ctx().device.cmd_copy_buffer(self.cmd, src, buf, &[vk::BufferCopy { src_offset: src_off, dst_offset: 0, size: bytes.len() as u64 }]) };
        global_barrier(self.cmd);
        self.tile_maps.insert(mk, (buf, n));
        self.tile_map_bytes += bytes.len() as u64;
        Ok(Some((buf, n)))
    }

    /// Records a (de)tiling pass between the render target's guest surface
    /// (in place through the imported guest memory) and `lin_off` in the
    /// device ring. Returns false when it cannot be done on the GPU (the
    /// caller then uses the CPU path).
    fn gpu_remap(&mut self, key: &ImageKey, tile: bool, lin_off: u64) -> Result<bool, String> {
        let Some(img) = self.images.get(key) else { return Ok(false) };
        let Some((bpp, mode, _ph, total)) = Self::image_guest_layout(key, img) else { return Ok(false) };
        let idx = img.tile_index.unwrap_or(0);
        if !Self::gpu_remap_eligible(bpp) {
            return Ok(false);
        }
        // Tiling back: into the imported guest memory, or without the import
        // into the VRAM mirror of the surface (registered as a mirror-resident
        // write once recorded; TLG_MIRROR_TILING=0: the CPU tiles instead).
        let mirrored = tile && ctx().ext_host.is_none();
        if mirrored && !mirror_tiling_on() {
            return Ok(false);
        }
        let layers = key.layers.max(1) as u64;
        let slice = total / layers;
        let lin_slice = key.width as u64 * key.height as u64 * (bpp / 8) as u64;
        let (set_layout, layout, detile, tilep) = self.remap_pipes()?;
        let Some((map, n)) = self.tile_map(key, bpp, mode, idx, slice)? else { return Ok(false) };
        let dest = if mirrored {
            if total > VCACHE_MAX || key.base % 16 != 0 {
                return Ok(false);
            }
            // (Pages not current are copied in first: tiling leaves the
            // surface padding as it is.)
            self.vc_why = "render target tiling";
            self.vcache_bind_ex(key.base, total, false, true)
        } else {
            self.guest_surface_source(key.base, total)?
        };
        let Some((zbuf, zoff)) = dest else { return Ok(false) };
        self.begin();
        self.end_rendering();
        global_barrier(self.cmd);
        let batch = self.batch;
        let set = self.alloc_set(set_layout)?;
        if self.batch != batch {
            // The descriptor pool ran out and the batch was submitted:
            // `lin_off` belongs to the previous ring segment.
            return Ok(false);
        }
        // Tiling writes the surface in device memory first, then moves it to
        // guest memory with one bulk copy: shader stores straight into host
        // memory crawl over PCIe (16 ms for an 8 MiB target).
        let tiled_off = if tile {
            let Some(o) = self.dring.alloc(total) else { return Ok(false) };
            // Surface padding (tiled elements outside the image) keeps its
            // guest contents.
            if n * (bpp / 8) as u64 > lin_slice {
                unsafe { ctx().device.cmd_copy_buffer(self.cmd, zbuf, self.dring.buffer, &[vk::BufferCopy { src_offset: zoff, dst_offset: o, size: total }]) };
                global_barrier(self.cmd);
            }
            Some(o)
        } else {
            None
        };
        let infos = [
            match tiled_off {
                Some(o) => vk::DescriptorBufferInfo { buffer: self.dring.buffer, offset: o, range: total },
                None => vk::DescriptorBufferInfo { buffer: zbuf, offset: zoff, range: total },
            },
            vk::DescriptorBufferInfo { buffer: map, offset: 0, range: n * 4 },
            vk::DescriptorBufferInfo { buffer: self.dring.buffer, offset: lin_off, range: lin_slice * layers },
        ];
        let writes: Vec<vk::WriteDescriptorSet> = (0..3)
            .map(|i| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&infos[i])))
            .collect();
        let c = ctx();
        unsafe {
            c.device.update_descriptor_sets(&writes, &[]);
            c.device.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, if tile { tilep } else { detile });
            c.device.cmd_bind_descriptor_sets(self.cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
            for l in 0..layers {
                let pc = [n as u32, bpp / 32, (l * slice / 4) as u32, (l * lin_slice / 4) as u32];
                let bytes: Vec<u8> = pc.iter().flat_map(|v| v.to_le_bytes()).collect();
                c.device.cmd_push_constants(self.cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, &bytes);
                c.device.cmd_dispatch(self.cmd, (n as u32).div_ceil(64), 1, 1);
            }
        }
        global_barrier(self.cmd);
        if let Some(o) = tiled_off {
            unsafe { c.device.cmd_copy_buffer(self.cmd, self.dring.buffer, zbuf, &[vk::BufferCopy { src_offset: o, dst_offset: zoff, size: total }]) };
            global_barrier(self.cmd);
        }
        if mirrored {
            // The mirror now holds the tiled surface: written back to guest
            // memory at the end of the batch like a shader's mirror-resident
            // write (every page), current for later readers of the mirror.
            // (Images over the range are not marked older than guest memory:
            // the one written back stays the current copy.)
            self.mirror_write_back_full(key.base, total, zbuf, zoff)?;
            self.writes.note_gpu_write(key.base, total);
            self.note_mirror_write(zbuf, key.base, total);
        }
        unordered!(self);
        self.mark(|| if tile { "render target tiling (GPU)".into() } else { "render target detiling (GPU)".into() });
        Ok(true)
    }

    /// Reloads a render target from its (tiled) guest surface after buffer
    /// writes made guest memory newer. Must run before descriptor
    /// allocation of the current draw/dispatch (it may wait for write-backs).
    fn refresh_from_guest(&mut self, key: ImageKey, may_submit: bool) -> Result<(), String> {
        if perf_on() {
            let s = format!("{:#x} {}x{}x{} {:?} (at {:#x})", key.base, key.width, key.height, key.layers, key.format, self.cur_shader);
            *RELOAD_SITES.lock().get_or_insert_with(HashMap::new).entry(s).or_default() += 1;
        }
        let Some(img) = self.images.get_mut(&key) else { return Ok(()) };
        img.guest_newer = false;
        let Some((bpp, mode, ph, total)) = Self::image_guest_layout(&key, img) else { return Ok(()) };
        if trace_frame() {
            crate::info!("FRAME reload {:#x} {}x{}x{} {:?} (guest {:#x}+{total:#x}) for {:#x}", key.base, key.width, key.height, key.layers, key.format, key.base, self.cur_shader);
        }
        // (Depth+stencil targets: the depth aspect is reloaded; stencil lives
        // in a separate guest surface.)
        let lin_bytes = key.width as u64 * key.height as u64 * (bpp / 8) as u64 * key.layers.max(1) as u64;
        // GPU path: detile in place from guest memory into the device ring.
        // Device-ring writes of this batch and in-flight write-backs must
        // have reached guest memory (as in `sync_guest`); in-place writes of
        // this batch are ordered on the GPU.
        if Self::gpu_remap_eligible(bpp) {
            // (Without the import the surface is staged with pending GPU
            // writes forwarded: nothing to wait for.)
            let staged = ctx().ext_host.is_none();
            if !staged {
                if self.writebacks.iter().any(|w| w.base < key.base + total && key.base < w.base + w.size) {
                    self.flush_for(1);
                } else if cpu_writeback_in_flight(key.base, total) {
                    self.flushes[1] += 1;
                    let t0 = std::time::Instant::now();
                    wait_cpu_writebacks(key.base, total);
                    self.prof[0] += t0.elapsed();
                }
            }
            let (dev_need, host_need) = if staged { (lin_bytes + total + 8192, total + 4096) } else { (lin_bytes + 4096, 0) };
            if may_submit && (self.dring.remaining() < dev_need || self.ring.remaining() < host_need) {
                self.flushes[2] += 1;
                self.submit_batch(Vec::new());
            }
            if let Some(lin_off) = (self.dring.remaining() >= dev_need).then(|| self.dring.alloc(lin_bytes)).flatten() {
                if self.gpu_remap(&key, false, lin_off)? {
                    if !staged {
                        self.zc_read(key.base, key.base + total);
                    }
                    let region = vk::BufferImageCopy {
                        buffer_offset: lin_off,
                        buffer_row_length: 0,
                        buffer_image_height: 0,
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: if key.depth { vk::ImageAspectFlags::DEPTH } else { vk::ImageAspectFlags::COLOR },
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: key.layers,
                        },
                        image_offset: vk::Offset3D::default(),
                        image_extent: vk::Extent3D { width: key.width, height: key.height, depth: 1 },
                    };
                    unsafe { ctx().device.cmd_copy_buffer_to_image(self.cmd, self.dring.buffer, self.images[&key].image, vk::ImageLayout::GENERAL, &[region]) };
                    unordered!(self);
                    *self.write_gen.entry(key).or_default() += 1;
                    self.images.get_mut(&key).unwrap().last_write = self.draws + self.dispatches + 1;
                    self.gpu_remaps[0] += 1;
                    return Ok(());
                }
            }
        }
        self.sync_guest(key.base, total);
        let t0 = std::time::Instant::now();
        let mut tiled = vec![0u8; total as usize];
        resources::copy_from_guest(&mut tiled, key.base);
        let lin_slice = key.width as usize * key.height as usize * (bpp / 8) as usize;
        let slice = total as usize / key.layers.max(1) as usize;
        let mut lin = vec![0u8; lin_slice * key.layers as usize];
        for l in 0..key.layers as usize {
            tiling::detile_slice(&tiled[l * slice..(l + 1) * slice], &mut lin[l * lin_slice..(l + 1) * lin_slice], key.width, key.height, key.width, ph, bpp, mode);
        }
        self.prof[4] += t0.elapsed();
        if may_submit && self.ring.remaining() < lin.len() as u64 + self.ring.segment_size() / 8 {
            self.flushes[2] += 1;
            self.submit_batch(Vec::new());
        }
        let off = self.ring.alloc(lin.len() as u64).ok_or("ring full (render target reload)")?;
        self.ring.write(off, &lin);
        self.begin();
        self.end_rendering();
        global_barrier(self.cmd);
        let region = vk::BufferImageCopy {
            buffer_offset: off,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: if key.depth { vk::ImageAspectFlags::DEPTH } else { vk::ImageAspectFlags::COLOR },
                mip_level: 0,
                base_array_layer: 0,
                layer_count: key.layers,
            },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D { width: key.width, height: key.height, depth: 1 },
        };
        unsafe { ctx().device.cmd_copy_buffer_to_image(self.cmd, self.ring.buffer, self.images[&key].image, vk::ImageLayout::GENERAL, &[region]) };
        unordered!(self);
        // Shadow copies of the old contents are out of date.
        *self.write_gen.entry(key).or_default() += 1;
        self.images.get_mut(&key).unwrap().last_write = self.draws + self.dispatches + 1;
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 12 {
            crate::info!("render target {:#x} {}x{} {:?} reloaded from guest memory (written by buffer stores)", key.base, key.width, key.height, key.format);
        }
        Ok(())
    }

    /// GPU buffer writes to [base, base+size) that touch a render target's
    /// HTILE/CMASK surface: examined when the target is next bound.
    fn note_meta_write(&mut self, base: u64, size: u64) {
        for k in &self.meta_images {
            let Some(img) = self.images.get_mut(k) else { continue };
            if let Some((m, n)) = img.meta {
                if m < base + size && base < m + n {
                    img.meta_dirty = true;
                    img.meta_fill = None;
                }
            }
        }
    }

    /// A fill of guest memory with a dword pattern was recorded: metadata
    /// surfaces it covers entirely hold that pattern (see `apply_meta_clears`).
    fn note_meta_fill(&mut self, base: u64, size: u64, pattern: u32) {
        for k in &self.meta_images {
            let Some(img) = self.images.get_mut(k) else { continue };
            if let Some((m, n)) = img.meta {
                if base <= m && m + n <= base + size {
                    img.meta_fill = Some(pattern);
                }
            }
        }
    }

    /// Fast clears: titles clear depth by writing HTILE (ZMask 0 = tile holds
    /// the clear value DB_DEPTH_CLEAR) and color targets by writing CMASK
    /// (0 = tile holds CB_COLOR*_CLEAR_WORD*), typically with a compute fill.
    /// When the metadata of a target about to be rendered was written, the
    /// cleared state is applied to the image (fully cleared surfaces only;
    /// HTILE/CMASK are not otherwise emulated).
    fn apply_meta_clears(&mut self, regs: &Regs, colors: &[Option<Attachment>], depth: Option<Attachment>) {
        let mut todo: Vec<(ImageKey, Option<u32>)> = Vec::new();
        for (n, a) in colors.iter().enumerate() {
            if let Some(a) = a {
                if self.images.get(&a.key).is_some_and(|i| i.meta_dirty) {
                    todo.push((a.key, Some(n as u32)));
                }
            }
        }
        if let Some(d) = depth {
            if self.images.get(&d.key).is_some_and(|i| i.meta_dirty) {
                todo.push((d.key, None));
            }
        }
        for (key, slot) in todo {
            let Some(((mbase, msize), fill)) = self.images.get_mut(&key).and_then(|i| {
                i.meta_dirty = false;
                i.meta.map(|m| (m, i.meta_fill))
            }) else {
                continue;
            };
            // Written by GPU work not complete yet: decided on the GPU.
            if fill.is_none() {
                match self.meta_clear_on_gpu(regs, colors, key, slot, mbase, msize) {
                    Ok(true) => continue,
                    Ok(false) => {}
                    Err(e) => self.report(format!("fast clear decided on the GPU failed (metadata read on the CPU instead): {e}")),
                }
            }
            // The metadata must have landed in guest memory (waits if needed)
            // unless the GPU filled all of it with a known dword.
            if fill.is_none() && self.sync_guest(mbase, msize) && perf_on() {
                *META_DECLINES.lock().get_or_insert_with(HashMap::new).entry("(CPU path waited for the GPU)").or_default() += 1;
            }
            let probe = |off: u64| -> Option<u32> {
                if let Some(p) = fill {
                    return Some(p);
                }
                let a = mbase + (off & !3);
                crate::mem::is_guest(a, 4).then(|| unsafe { std::ptr::read_volatile(a as *const u32) })
            };
            let offs: Vec<u64> = (0..16).map(|k| k * 4).chain((0..16).map(|k| ((msize / 2) & !3) + k * 4)).chain((0..16).map(|k| msize.saturating_sub(64) + k * 4)).collect();
            let words: Vec<u32> = offs.iter().filter_map(|&o| probe(o)).collect();
            if words.is_empty() {
                continue;
            }
            static SEEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 12 {
                crate::info!("metadata of {:#x} ({}) written: {:#x}+{:#x} sample {:08x?}", key.base, if slot.is_some() { "CMASK" } else { "HTILE" }, mbase, msize, &words[..4.min(words.len())]);
            }
            let c = ctx();
            let img = self.images[&key].image;
            match slot {
                None => {
                    // HTILE: ZMask [3:0] == 0 -> depth cleared; with stencil
                    // tiles enabled, SMem [9:8] == 0 -> stencil cleared.
                    let depth_clear = words.iter().all(|w| w & 0xF == 0);
                    let stencil_tiles = regs.ctx(0xA011) & (1 << 29) == 0 && key.format == vk::Format::D32_SFLOAT_S8_UINT;
                    let stencil_clear = stencil_tiles && words.iter().all(|w| (w >> 8) & 3 == 0);
                    let mut aspect = vk::ImageAspectFlags::empty();
                    if depth_clear {
                        aspect |= vk::ImageAspectFlags::DEPTH;
                    }
                    if stencil_clear {
                        aspect |= vk::ImageAspectFlags::STENCIL;
                    }
                    if aspect.is_empty() {
                        continue;
                    }
                    let value = vk::ClearDepthStencilValue { depth: f32r(regs.ctx(0xA00B)), stencil: regs.ctx(0xA00A) & 0xFF };
                    self.begin();
                    self.end_rendering();
                    global_barrier(self.cmd);
                    let range = vk::ImageSubresourceRange { aspect_mask: aspect, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: key.layers };
                    unsafe { c.device.cmd_clear_depth_stencil_image(self.cmd, img, vk::ImageLayout::GENERAL, &value, &[range]) };
                    unordered!(self);
                    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                        crate::info!("HTILE clear of depth {:#x} ({:?}, depth {} stencil {})", key.base, aspect, value.depth, value.stencil);
                    }
                }
                Some(n) => {
                    // CMASK: all tiles 0 -> fast cleared to the clear words.
                    if !words.iter().all(|&w| w == 0) {
                        continue;
                    }
                    let r = 0xA318 + n * 0xF;
                    let (w0, w1) = (regs.ctx(r + 11), regs.ctx(r + 12));
                    let format = colors[n as usize].map(|a| a.view_format).unwrap_or(key.format);
                    let Some(value) = clear_color(format, w0, w1) else {
                        self.report(format!("fast clear of {format:?} with clear words {w0:#x}/{w1:#x} not supported"));
                        continue;
                    };
                    self.begin();
                    self.end_rendering();
                    global_barrier(self.cmd);
                    let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: key.layers };
                    unsafe { c.device.cmd_clear_color_image(self.cmd, img, vk::ImageLayout::GENERAL, &value, &[range]) };
                    unordered!(self);
                    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                        crate::info!("CMASK fast clear of {:#x} {:?} to {w0:#x}/{w1:#x}", key.base, format);
                    }
                }
            }
            self.mark_stale(key);
            if let Some(i) = self.images.get_mut(&key) {
                i.gpu_written = true;
                i.guest_stale = true;
            }
            // The hardware marks tiles expanded as it renders them (and an
            // eliminate pass expands the rest); we applied the whole clear, so
            // record the expanded state: a later unrelated write near the
            // metadata must not make the stale "cleared" state reapply.
            if crate::mem::is_guest(mbase, msize) {
                let expanded = vec![0xFFu8; msize as usize];
                resources::copy_to_guest(&expanded, mbase);
                self.writes.note_gpu_write(mbase, msize);
            }
        }
    }

    /// The fast-clear metadata kernels (`kernels::meta_probe_cs`,
    /// `meta_expand_cs`; created on first use).
    fn meta_pipes(&mut self) -> Result<(vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline, vk::Pipeline), String> {
        if let Some(p) = self.meta_pipes {
            return Ok(p);
        }
        let c = ctx();
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..2)
            .map(|i| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
            .collect();
        let set_layout = unsafe { c.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None) }.map_err(|e| e.to_string())?;
        let pcr = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 16 }];
        let sls = [set_layout];
        let layout = unsafe { c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sls).push_constant_ranges(&pcr), None) }.map_err(|e| e.to_string())?;
        let mut pipes = [vk::Pipeline::null(); 2];
        for (k, spirv) in [gcn::translate::kernels::meta_probe_cs(), gcn::translate::kernels::meta_expand_cs()].iter().enumerate() {
            let m = self.module(0x4D45_5441_0000_0000 | k as u64, spirv)?;
            let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(m).name(c"main");
            let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
            pipes[k] = unsafe { c.device.create_compute_pipelines(vk::PipelineCache::null(), &[ci], None) }.map_err(|(_, e)| format!("metadata kernel pipeline: {e}"))?[0];
        }
        self.meta_pipes = Some((set_layout, layout, pipes[0], pipes[1]));
        Ok((set_layout, layout, pipes[0], pipes[1]))
    }

    /// `apply_meta_clears` for metadata that GPU work recorded in this batch
    /// or in flight writes in place (ordered on the queue): instead of
    /// draining the GPU to read it on the CPU (about once a frame: the title
    /// fast-clears depth by writing HTILE), `kernels::meta_probe_cs` samples
    /// the same dwords on the GPU into predicates that gate the clears
    /// (VK_EXT_conditional_rendering) and `meta_expand_cs` writes the
    /// expanded state when one applied. False when not applicable: nothing
    /// pending (the CPU reads it now), CPU-performed writes pending (they
    /// land after the GPU would read), layered or tiny targets, a clear
    /// color the CPU path reports, no conditional rendering.
    fn meta_clear_on_gpu(&mut self, regs: &Regs, colors: &[Option<Attachment>], key: ImageKey, slot: Option<u32>, mbase: u64, msize: u64) -> Result<bool, String> {
        let c = ctx();
        let Some(cond) = c.cond_render.as_ref() else { return meta_decline("no conditional rendering") };
        if crate::tune::get(crate::tune::META_GPU) == 0 || key.layers > 1 || msize < 128 || msize % 4 != 0 || mbase % 4 != 0 || msize / 4 > u32::MAX as u64 {
            return meta_decline("layered, tiny or unaligned metadata");
        }
        // Without the import (the default) the probe reads the metadata in
        // its VRAM mirror, which binding brings up to date on the GPU (this
        // batch's and in-flight shader writes forwarded), so writes recorded
        // in this batch need no drain either.
        let mirror = c.ext_host.is_none();
        let in_batch = self.writebacks.iter().any(|w| w.base < mbase + msize && mbase < w.base + w.size);
        if !(self.zc_overlaps(mbase, msize) || writeback_in_flight(mbase, msize) || (mirror && in_batch)) {
            return meta_decline("nothing pending");
        }
        // (In-flight write-backs: the mirror binding forwards them, or waits
        // where it cannot, e.g. for end-of-pipe writes.)
        if !mirror && (in_batch || cpu_writeback_in_flight(mbase, msize)) {
            return meta_decline(if in_batch { "in-batch writes (zero-copy)" } else { "CPU-performed writes pending (zero-copy)" });
        }
        // What a clear writes (from the registers now, as the CPU path).
        enum Clear {
            Depth { value: vk::ClearDepthStencilValue, stencil_tiles: bool },
            Color { format: vk::Format, layer: u32, value: vk::ClearColorValue },
        }
        let clear = match slot {
            None => Clear::Depth {
                value: vk::ClearDepthStencilValue { depth: f32r(regs.ctx(0xA00B)), stencil: regs.ctx(0xA00A) & 0xFF },
                stencil_tiles: regs.ctx(0xA011) & (1 << 29) == 0 && key.format == vk::Format::D32_SFLOAT_S8_UINT,
            },
            Some(n) => {
                let r = 0xA318 + n * 0xF;
                let att = colors.get(n as usize).copied().flatten();
                let format = att.map(|a| a.view_format).unwrap_or(key.format);
                let Some(value) = clear_color(format, regs.ctx(r + 11), regs.ctx(r + 12)) else { return meta_decline("clear color not supported") };
                Clear::Color { format, layer: att.map_or(0, |a| a.layer), value }
            }
        };
        let align = c.limits.min_storage_buffer_offset_alignment.max(16);
        let base = mbase & !(align - 1);
        let span = mbase + msize - base;
        let (zbuf, zoff) = if mirror {
            self.vc_why = "fast-clear metadata";
            match self.vcache_bind_ex(base, span, false, true) {
                Some((b, o)) if o % align == 0 => (b, o),
                _ => return meta_decline("no mirror binding"),
            }
        } else {
            let Some((zbuf, zoff, zrange)) = self.zc_binding(base, span) else { return meta_decline("no zero-copy binding") };
            if zrange < span || zoff % align != 0 {
                return meta_decline("zero-copy binding unaligned");
            }
            (zbuf, zoff)
        };
        let (set_layout, layout, probe, expand) = self.meta_pipes()?;
        let batch = self.batch;
        let set = self.alloc_set(set_layout)?;
        if self.batch != batch {
            return meta_decline("batch closed");
        }
        let Some(pred) = self.dring.alloc(16) else { return meta_decline("device ring full") };
        let infos = [vk::DescriptorBufferInfo { buffer: zbuf, offset: zoff, range: span }, vk::DescriptorBufferInfo { buffer: self.dring.buffer, offset: pred, range: 16 }];
        let writes: Vec<vk::WriteDescriptorSet> = (0..2)
            .map(|i| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(std::slice::from_ref(&infos[i])))
            .collect();
        let (first, n) = (((mbase - base) / 4) as u32, (msize / 4) as u32);
        let pc = match clear {
            Clear::Depth { stencil_tiles, .. } => [first, n, 0, stencil_tiles as u32],
            Clear::Color { .. } => [first, n, 1, 0],
        };
        let bytes: Vec<u8> = pc.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.begin();
        self.end_rendering();
        global_barrier(self.cmd);
        unsafe {
            c.device.update_descriptor_sets(&writes, &[]);
            c.device.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, probe);
            c.device.cmd_bind_descriptor_sets(self.cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
            c.device.cmd_push_constants(self.cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, &bytes);
            c.device.cmd_dispatch(self.cmd, 1, 1, 1);
        }
        // (The predicates: visible to the expand kernel and to conditional
        // rendering; the probe's reads precede the expanded state's writes.)
        global_barrier(self.cmd);
        unsafe {
            // (Same layout: the set and push constants stay bound.)
            c.device.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, expand);
            c.device.cmd_dispatch(self.cmd, n.div_ceil(64), 1, 1);
        }
        // The clears, each gated by its predicate.
        let extent = (key.width, key.height);
        let rect = vk::ClearRect { rect: vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: extent.0, height: extent.1 } }, base_array_layer: 0, layer_count: 1 };
        let mut clears: smallvec::SmallVec<[(u64, vk::ClearAttachment); 2]> = smallvec::SmallVec::new();
        match clear {
            Clear::Depth { value, stencil_tiles } => {
                self.begin_rendering(&[], Some(Attachment { key, view_format: key.format, layer: 0 }), extent)?;
                clears.push((0, vk::ClearAttachment { aspect_mask: vk::ImageAspectFlags::DEPTH, color_attachment: 0, clear_value: vk::ClearValue { depth_stencil: value } }));
                if stencil_tiles {
                    clears.push((4, vk::ClearAttachment { aspect_mask: vk::ImageAspectFlags::STENCIL, color_attachment: 0, clear_value: vk::ClearValue { depth_stencil: value } }));
                }
            }
            Clear::Color { format, layer, value } => {
                self.begin_rendering(&[Some(Attachment { key, view_format: format, layer })], None, extent)?;
                clears.push((8, vk::ClearAttachment { aspect_mask: vk::ImageAspectFlags::COLOR, color_attachment: 0, clear_value: vk::ClearValue { color: value } }));
            }
        }
        for (off, ca) in clears {
            let info = vk::ConditionalRenderingBeginInfoEXT::default().buffer(self.dring.buffer).offset(pred + off);
            unsafe {
                (cond.fp().cmd_begin_conditional_rendering_ext)(self.cmd, &info);
                c.device.cmd_clear_attachments(self.cmd, &[ca], &[rect]);
                (cond.fp().cmd_end_conditional_rendering_ext)(self.cmd);
            }
        }
        self.end_rendering();
        unordered!(self);
        if mirror {
            // The expanded state (when a clear applied) is in the mirror:
            // written back to guest memory at the end of the batch (every
            // page), current for later readers of the mirror.
            self.mirror_write_back_full(base, span, zbuf, zoff)?;
            self.writes.note_gpu_write(base, span);
            self.note_mirror_write(zbuf, base, span);
        } else {
            // The expanded state lands in place (when a clear applied); the
            // probe read the metadata in place.
            self.push_zc_write(mbase, msize);
            self.zc_read(mbase, mbase + msize);
            self.writes.note_gpu_write(mbase, msize);
            self.note_pending_gpu_write(mbase, msize);
        }
        // (As the CPU path's clear does; the draw that follows renders to it.)
        self.mark_stale(key);
        if let Some(i) = self.images.get_mut(&key) {
            i.gpu_written = true;
            i.guest_stale = true;
        }
        self.meta_gpu += 1;
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
            crate::info!("metadata of {:#x} ({}) {:#x}+{:#x} written by GPU work in flight: fast clear decided on the GPU", key.base, if slot.is_some() { "CMASK" } else { "HTILE" }, mbase, msize);
        }
        self.mark(|| "fast clear (GPU predicate)".into());
        Ok(true)
    }

    /// Records a GDS transfer after all work recorded so far.
    fn gds_op(&mut self, op: super::cp::GdsOp) -> Result<(), String> {
        use super::cp::GdsOp;
        let limit = gcn::translate::GDS_BYTES;
        let clamp = |offset: u32, bytes: u32| -> Option<(u64, u64)> {
            let (o, n) = (offset & !3, bytes & !3);
            (n > 0 && o < limit).then(|| (o as u64, n.min(limit - o) as u64))
        };
        if crate::log::enabled(crate::log::Level::Debug) {
            crate::debug!("gds: {op:?}");
        }
        match op {
            GdsOp::Fill { offset, bytes, value } => {
                let Some((o, n)) = clamp(offset, bytes) else { return Ok(()) };
                self.begin();
                self.end_rendering();
                global_barrier(self.cmd);
                unsafe { ctx().device.cmd_fill_buffer(self.cmd, self.gds, o, n, value) };
            }
            GdsOp::Load { offset, src, bytes } => {
                let Some((o, n)) = clamp(offset, bytes) else { return Ok(()) };
                self.reserve(n.next_multiple_of(256), 0);
                self.begin();
                let off = self.upload_guest(src, n)?;
                self.flush_forwards();
                self.end_rendering();
                global_barrier(self.cmd);
                let region = vk::BufferCopy { src_offset: off, dst_offset: o, size: n };
                unsafe { ctx().device.cmd_copy_buffer(self.cmd, self.ring.buffer, self.gds, &[region]) };
            }
            GdsOp::Store { offset, dst, bytes } => {
                let Some((o, n)) = clamp(offset, bytes) else { return Ok(()) };
                self.reserve(n.next_multiple_of(256), n.next_multiple_of(256));
                self.begin();
                // Into guest memory in place, in queue order (unless CPU-performed
                // writes of the range are pending: they would land after it;
                // `tune dma_gpu 0`: always as below).
                let cpu_pending = self.writebacks.iter().any(|w| w.base < dst + n && dst < w.base + w.size) || cpu_writeback_in_flight(dst, n);
                let in_place = crate::tune::get(crate::tune::DMA_GPU) != 0 && !cpu_pending;
                if let Some((zbuf, zoff, _)) = self.zc_binding(dst, n).filter(|z| z.2 >= n && in_place) {
                    self.end_rendering();
                    global_barrier(self.cmd);
                    unsafe { ctx().device.cmd_copy_buffer(self.cmd, self.gds, zbuf, &[vk::BufferCopy { src_offset: o, dst_offset: zoff, size: n }]) };
                    self.note_inplace_gpu_write(dst, n);
                    unordered!(self);
                    return Ok(());
                }
                // Otherwise like a shader write: device ring now, host ring at
                // batch end, guest memory on completion.
                let host_off = self.ring.alloc(n).ok_or("ring full")?;
                let dev_off = self.dring.alloc(n).ok_or("device ring full")?;
                self.end_rendering();
                global_barrier(self.cmd);
                let region = vk::BufferCopy { src_offset: o, dst_offset: dev_off, size: n };
                unsafe { ctx().device.cmd_copy_buffer(self.cmd, self.gds, self.dring.buffer, &[region]) };
                self.push_writeback(PendingWrite { base: dst, size: n, host_off, dev_off, dev_buf: self.dring.buffer, mirror: false, bits: None });
                self.writes.note_gpu_write(dst, n);
                self.note_pending_gpu_write(dst, n);
                self.note_buffer_write(dst, n);
                self.epoch += 1;
            }
        }
        unordered!(self);
        Ok(())
    }

    /// The most frequent reported problems (message, count).
    pub fn top_reports(&self, n: usize) -> Vec<(String, u64)> {
        let mut v: Vec<(String, u64)> = self.reported.iter().map(|(m, c)| (m.clone(), *c)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v.truncate(n);
        v
    }

    pub fn image_count(&self) -> usize {
        self.images.len()
    }
    pub fn pipeline_count(&self) -> usize {
        self.gfx_pipes.len() + self.cs_pipes.len()
    }

    /// Between batches (at flips): frees released objects whose batches have
    /// completed, and every 30 frames measures VRAM pressure and evicts
    /// images unused for `EVICT_AGE` frames -- textures from areas left
    /// behind would otherwise accumulate until the OS pages video memory --
    /// and, under pressure, idle VRAM mirror windows.
    pub fn housekeeping(&mut self) {
        let done = *DONE.lock();
        if self.retired.iter().any(|(s, _)| *s <= done) {
            let c = ctx();
            let mut budget = if self.pressure >= 2 { usize::MAX } else { RETIRE_PER_PASS };
            let mut blocks = Vec::new();
            let mut pooled: Vec<(ImageKey, resources::PooledImage)> = Vec::new();
            self.retired.retain_mut(|(s, r)| {
                if *s > done || budget == 0 {
                    return true;
                }
                budget -= 1;
                unsafe {
                    match r {
                        Retired::Image(i, m, views) => {
                            for &v in views.iter() {
                                c.device.destroy_image_view(v, None);
                            }
                            c.device.destroy_image(*i, None);
                            c.free(*m);
                        }
                        Retired::Buffer(b, m) => {
                            c.device.destroy_buffer(*b, None);
                            c.free(*m);
                        }
                        Retired::Window(b, m, bound) => {
                            c.device.destroy_buffer(*b, None);
                            if *m != vk::DeviceMemory::null() {
                                c.free(*m);
                            }
                            blocks.append(bound);
                        }
                        Retired::Set(pool, set) => {
                            c.device.free_descriptor_sets(*pool, &[*set]).ok();
                        }
                        Retired::Pooled(k, p) => {
                            pooled.push((*k, std::mem::take(p)));
                        }
                    }
                }
                false
            });
            if let Some(p) = self.sparse_pool.as_mut() {
                p.give(blocks);
            }
            for (k, p) in pooled {
                if self.image_pool_n < IMAGE_POOL_MAX && self.pressure < 2 {
                    self.image_pool.entry(k).or_default().push(p);
                    self.image_pool_n += 1;
                } else {
                    p.destroy();
                }
            }
        }
        let frame = super::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
        if frame < self.next_trim || self.recording || self.rendering.is_some() || crate::tune::get(crate::tune::EVICT) == 0 {
            return;
        }
        self.next_trim = frame + 30;
        let (usage, budget) = ctx().vram().unwrap_or((0, 0));
        self.pressure = vram_pressure(usage, budget);
        if self.pressure >= 2 && self.image_pool_n > 0 {
            // (Pooled images are idle: the GPU finished with them.)
            for (_, v) in self.image_pool.drain() {
                v.into_iter().for_each(resources::PooledImage::destroy);
            }
            self.image_pool_n = 0;
        }
        let age = EVICT_AGE[self.pressure as usize];
        // (Not recording: every command using these has been submitted.)
        let seq = self.last_seq;
        // Image eviction is opt-in (TLG_EVICT_IMAGES=1): under VRAM pressure 3
        // on the development machine (budget 2.6 GB while another process used
        // the GPU), evicting CPU-sourced textures left the title screen's 3D
        // background black or garbage with ~600 ms GPU stalls; keeping the
        // images (mirror windows still released) rendered correctly. Root
        // cause not found yet (shadow copies excluded: still black).
        // TLG_EVICT_WINDOWS=0 keeps idle VRAM mirror windows; TLG_EVICT_SHADOWS=0
        // keeps shadow copies when images are evicted.
        static IMAGES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static WINDOWS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let evict_images = *IMAGES.get_or_init(|| std::env::var("TLG_EVICT_IMAGES").is_ok_and(|v| v == "1"));
        let release_windows = *WINDOWS.get_or_init(|| std::env::var("TLG_EVICT_WINDOWS").map(|v| v != "0").unwrap_or(true));
        static SHADOWS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let evict_shadows = *SHADOWS.get_or_init(|| std::env::var("TLG_EVICT_SHADOWS").map(|v| v != "0").unwrap_or(true));
        let small = crate::tune::get(crate::tune::EVICT_SMALL) != 0;
        let mut candidates: Vec<(u64, u64, ImageKey)> = self
            .images
            .iter()
            .filter(|(k, i)| {
                let tiny = small && !i.gpu_written && !is_shadow(k) && i.bytes <= SMALL_IMAGE_BYTES && i.last_used + SMALL_IMAGE_AGE.min(age) <= frame;
                evictable(k, i) && (tiny || (evict_images && (evict_shadows || !is_shadow(k)) && i.last_used + age <= frame))
            })
            .map(|(k, i)| (i.last_used, i.bytes, *k))
            .collect();
        if self.pressure < 2 {
            candidates.sort_unstable_by_key(|c| c.0);
            // (Small textures under a cap of their own: cheap to destroy, and
            // they arrive faster than the general cap would take them.)
            let (mut n, mut bytes, mut small_n) = (0usize, 0u64, 0usize);
            candidates.retain(|c| {
                if c.1 <= SMALL_IMAGE_BYTES {
                    small_n += 1;
                    return small_n <= SMALL_PER_PASS;
                }
                let keep = n < EVICT_PER_PASS.0 && bytes < EVICT_PER_PASS.1;
                n += 1;
                bytes += c.1;
                keep
            });
        }
        // (Passes that evict only small textures are not logged: the VRAM
        // report counts them.)
        let any_big = candidates.iter().any(|c| c.1 > SMALL_IMAGE_BYTES);
        let victims: Vec<ImageKey> = candidates.into_iter().map(|c| c.2).collect();
        let mut freed = 0;
        for k in &victims {
            let Some(img) = self.images.remove(k) else { continue };
            if img.meta_listed {
                self.meta_images.retain(|x| x != k);
            }
            if let Some(v) = self.by_base.get_mut(&k.base) {
                v.retain(|x| x != k);
                if v.is_empty() {
                    self.by_base.remove(&k.base);
                }
            }
            if let Some(v) = self.img_bases.get_mut(&k.base) {
                v.retain(|x| x.0 != *k);
                if v.is_empty() {
                    self.img_bases.remove(&k.base);
                }
            }
            if let Some(v) = self.img_bases_large.get_mut(&k.base) {
                v.retain(|x| x.0 != *k);
                if v.is_empty() {
                    self.img_bases_large.remove(&k.base);
                }
            }
            self.write_gen.remove(k);
            if self.evicted_keys.len() >= 1 << 16 {
                self.evicted_keys.clear();
            }
            self.evicted_keys.insert(*k, frame);
            freed += img.bytes;
            self.image_bytes -= img.bytes;
            if img.bytes <= SMALL_IMAGE_BYTES && crate::tune::get(crate::tune::IMAGE_POOL) != 0 {
                self.retired.push((seq, Retired::Pooled(ImageKey { base: 0, ..*k }, resources::PooledImage::of(img))));
            } else {
                self.retired.push((seq, Retired::Image(img.image, img.memory, img.views.values().copied().collect())));
            }
        }
        if !victims.is_empty() {
            let gone: std::collections::HashSet<ImageKey> = victims.iter().copied().collect();
            self.shadows.retain(|sk, (dk, _)| !gone.contains(&sk.0) && !gone.contains(dk));
            self.evicted.0 += victims.len() as u64;
            self.evicted.1 += freed;
            if any_big {
                crate::info!("VRAM: evicted {} images unused for {age}+ frames ({} MiB; pressure {})", victims.len(), freed >> 20, self.pressure);
            }
        }
        let mut released = 0;
        if self.pressure >= 2 && release_windows {
            let mut gone = Vec::new();
            if let Some(vc) = self.vcache.as_mut() {
                let idle: Vec<u64> = vc.windows.iter().filter(|(_, w)| w.last_used + age <= frame).map(|(k, _)| *k).collect();
                for k in idle {
                    let w = vc.windows.remove(&k).unwrap();
                    super::lazywb::release_buffer(w.buf);
                    vc.bytes -= w.bytes();
                    released += w.bytes();
                    gone.push(w.buf);
                    self.retired.push((seq, w.retire()));
                }
            }
            for b in gone {
                self.retire_dirty_set(b, seq);
            }
            // (Blocks of windows released earlier are back by now.)
            if let Some(p) = self.sparse_pool.as_mut() {
                let freed = p.trim();
                if freed > 0 {
                    crate::info!("VRAM: freed {} MiB of unused mirror block memory (pressure {})", freed >> 20, self.pressure);
                }
            }
        }
        if self.pressure >= 2 && self.pressure_reported.is_none_or(|t| t.elapsed().as_secs() >= 30) {
            self.pressure_reported = Some(std::time::Instant::now());
            crate::warn!(
                "VRAM pressure {}: device-local usage {} MiB of a {} MiB budget (other processes count against it); images {} MiB, mirrors {} MiB; evicting what is unused for {age} frames{}",
                self.pressure,
                usage >> 20,
                budget >> 20,
                self.image_bytes >> 20,
                self.vcache.as_ref().map_or(0, |v| v.bytes) >> 20,
                if released > 0 { format!(", released {} MiB of mirrors", released >> 20) } else { String::new() }
            );
        }
    }

    /// One line of VRAM accounting for the periodic statistics.
    pub fn vram_report(&self) -> String {
        let (mut tex, mut gpu, mut shadow) = (0u64, 0u64, 0u64);
        for (k, i) in &self.images {
            if is_shadow(k) {
                shadow += i.bytes;
            } else if i.gpu_written {
                gpu += i.bytes;
            } else {
                tex += i.bytes;
            }
        }
        let (tracked, sites) = ctx().local_allocations();
        let sites: Vec<String> = sites.iter().take(8).map(|(at, b, n)| format!("{at} {} MiB in {n}", b >> 20)).collect();
        let budget = ctx().vram().map(|(u, b)| format!("; device-local usage {} MiB of a {} MiB budget; allocated by us {} MiB ({})", u >> 20, b >> 20, tracked >> 20, sites.join(", "))).unwrap_or_default();
        // Per mirror window: MiB ever bound / currently valid / allocated.
        let mut wins: Vec<(u64, u64, u64, u64)> = self.vcache.as_ref().map_or(Vec::new(), |vc| {
            let ones = |v: &[u64]| v.iter().map(|x| x.count_ones() as u64).sum::<u64>() << 12;
            vc.windows.values().map(|w| (w.lo, ones(&w.touched), ones(&w.valid), w.bytes())).collect()
        });
        wins.sort_unstable();
        let touched: u64 = wins.iter().map(|w| w.1).sum();
        let wins: Vec<String> = wins.iter().map(|w| format!("{:#x} {}/{}/{}", w.0, w.1 >> 20, w.2 >> 20, w.3 >> 20)).collect();
        format!(
            "VRAM: {} images {} MiB (textures {} MiB, GPU-written {} MiB, shadow copies {} MiB), mirrors {} MiB ({} MiB ever bound; per window MiB bound/valid/allocated: {}), element maps {} MiB; evicted {} images ({} MiB) so far, {} needed again (after {} frames on average){budget}",
            self.images.len(),
            self.image_bytes >> 20,
            tex >> 20,
            gpu >> 20,
            shadow >> 20,
            self.vcache.as_ref().map_or(0, |v| v.bytes) >> 20,
            touched >> 20,
            wins.join(", "),
            self.tile_map_bytes >> 20,
            self.evicted.0,
            self.evicted.1 >> 20,
            self.recreated.0,
            self.recreated.1.checked_div(self.recreated.0).unwrap_or(0)
        )
    }

    /// Image to present for a display buffer (after a flush).
    pub fn display_image(&mut self, base: u64, w: u32, h: u32) -> Option<(vk::Image, u32, u32)> {
        let k = self.display_key(base, w, h)?;
        Some((self.images[&k].image, w, h))
    }

    fn display_key(&self, base: u64, w: u32, h: u32) -> Option<ImageKey> {
        let ks = self.by_base.get(&base)?;
        ks.iter().copied().find(|k| !k.depth && k.width >= w && k.height >= h && self.images[k].gpu_written)
    }

    /// Reads back a display image as RGBA8 (for screenshots).
    pub fn capture(&mut self, base: u64, w: u32, h: u32) -> Option<Vec<u8>> {
        let k = self.display_key(base, w, h)?;
        if format_class(k.format) != 4 {
            return None;
        }
        self.begin();
        self.end_rendering();
        global_barrier(self.cmd);
        let off = self.ring.alloc((w * h * 4) as u64)?;
        let region = vk::BufferImageCopy {
            buffer_offset: off,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D { width: w, height: h, depth: 1 },
        };
        unsafe { ctx().device.cmd_copy_image_to_buffer(self.cmd, self.images[&k].image, vk::ImageLayout::GENERAL, self.ring.buffer, &[region]) };
        self.flush();
        let mut px = unsafe { std::slice::from_raw_parts(self.ring.ptr.add(off as usize), (w * h * 4) as usize) }.to_vec();
        if matches!(k.format, vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB) {
            for p in px.chunks_mut(4) {
                p.swap(0, 2);
            }
        }
        if matches!(k.format, vk::Format::A2B10G10R10_UNORM_PACK32 | vk::Format::A2R10G10B10_UNORM_PACK32) {
            // 10-bit channels (sRGB-encoded for A2R10G10B10 sRGB display
            // buffers): their top 8 bits.
            let bgr = k.format == vk::Format::A2R10G10B10_UNORM_PACK32;
            for p in px.chunks_mut(4) {
                let v = u32::from_le_bytes([p[0], p[1], p[2], p[3]]);
                let (lo, mid, hi) = ((v >> 2) & 0xFF, (v >> 12) & 0xFF, (v >> 22) & 0xFF);
                let (r, g, b) = if bgr { (hi, mid, lo) } else { (lo, mid, hi) };
                p.copy_from_slice(&[r as u8, g as u8, b as u8, 255]);
            }
        }
        for p in px.chunks_mut(4) {
            p[3] = 255;
        }
        Some(px)
    }
}

impl Renderer {
    /// Debug aid (TLG_DUMP_IMAGES=<flip>): reads back every GPU-written color
    /// image (first layer, base level), logs channel statistics and writes
    /// a viewable PNG next to the run log.
    pub fn dump_images(&mut self, flip: u64) {
        let Some(dir) = super::shader_dump::DIR.get().and_then(|d| d.parent()).map(|d| d.join(format!("images{flip:05}"))) else { return };
        let _ = std::fs::create_dir_all(&dir);
        // TLG_DUMP_BASES=<hex,...>: only images at those guest bases (shadow
        // copies of them included).
        static BASES: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
        let bases = BASES.get_or_init(|| {
            std::env::var("TLG_DUMP_BASES").map(|v| v.split(',').filter_map(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok()).collect()).unwrap_or_default()
        });
        let mut keys: Vec<ImageKey> = self
            .images
            .iter()
            .filter(|(k, i)| i.gpu_written && (bases.is_empty() || bases.contains(&(k.base & 0x0FFF_FFFF_FFFF_FFFF))))
            .map(|(k, _)| *k)
            .collect();
        keys.sort_by_key(|k| (k.base, k.width, k.height));
        for k in keys {
            // 3D images: a few depth slices (`layers` holds the depth).
            let slices: Vec<u32> = if k.ty3d { [0, 1, 4, 16, 32, k.layers - 1].into_iter().filter(|&z| z < k.layers).collect() } else { (0..k.layers.min(6)).collect() };
            for layer in slices {
                self.dump_image(&dir, k, layer, "");
            }
        }
    }

    /// Debug: writes one layer of an image as PNG (and logs value ranges).
    fn dump_image(&mut self, dir: &std::path::Path, k: ImageKey, layer: u32, prefix: &str) {
        // Depth targets: the depth aspect as floats (D16 as 16-bit unorm).
        let (texel, aspect, view_fmt) = match k.format {
            vk::Format::D32_SFLOAT | vk::Format::D32_SFLOAT_S8_UINT => (4, vk::ImageAspectFlags::DEPTH, vk::Format::R32_SFLOAT),
            vk::Format::D16_UNORM => (2, vk::ImageAspectFlags::DEPTH, vk::Format::R16_UNORM),
            f => match format_class(f) {
                c @ (1 | 2 | 4 | 8 | 16) => (c, vk::ImageAspectFlags::COLOR, f),
                _ => return,
            },
        };
        let size = (k.width * k.height * texel) as u64;
        self.flush();
        self.begin();
        global_barrier(self.cmd);
        let Some(off) = self.ring.alloc(size) else { return };
        let region = vk::BufferImageCopy {
            buffer_offset: off,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers { aspect_mask: aspect, mip_level: 0, base_array_layer: if k.ty3d { 0 } else { layer }, layer_count: 1 },
            image_offset: vk::Offset3D { x: 0, y: 0, z: if k.ty3d { layer as i32 } else { 0 } },
            image_extent: vk::Extent3D { width: k.width, height: k.height, depth: 1 },
        };
        unsafe { ctx().device.cmd_copy_image_to_buffer(self.cmd, self.images[&k].image, vk::ImageLayout::GENERAL, self.ring.buffer, &[region]) };
        self.flush();
        let raw = unsafe { std::slice::from_raw_parts(self.ring.ptr.add(off as usize), size as usize) }.to_vec();
        let (rgba, stats) = to_viewable(view_fmt, &raw, k.width, k.height);
        crate::info!("dump {prefix}{:#x} {}x{} layer {} {:?}: {}", k.base, k.width, k.height, layer, k.format, stats);
        let name = format!("{prefix}img_{:x}_{}x{}_l{}_{:?}", k.base, k.width, k.height, layer, k.format);
        if let Some(px) = rgba {
            let _ = std::fs::write(dir.join(format!("{name}.png")), super::texdump::png(k.width, k.height, &px));
        }
        // TLG_DUMP_RAW=1 (or control `dumpraw 1`): the texels too (tightly packed rows).
        if std::env::var_os("TLG_DUMP_RAW").is_some() || DUMP_RAW.load(std::sync::atomic::Ordering::Relaxed) {
            let _ = std::fs::write(dir.join(format!("{name}.bin")), &raw);
        }
    }

    /// Frame trace with `TLG_DUMP_AFTER=<ps addrs>`: dumps the color targets of
    /// draws using one of the listed pixel shaders right after the draw.
    fn dump_after(&mut self, ps: u64, colors: &[Option<Attachment>], depth: Option<Attachment>) {
        let n = {
            let mut d = DUMP_AFTER.lock();
            let d = d.get_or_insert_with(|| {
                let ps = std::env::var("TLG_DUMP_AFTER").map(|v| parse_hex_list(&v)).unwrap_or_default();
                // At most TLG_DUMP_AFTER_MAX (default 6) dumps.
                (ps, std::env::var("TLG_DUMP_AFTER_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(6), 0)
            });
            if !d.0.contains(&ps) || d.2 >= d.1 {
                return;
            }
            d.2 += 1;
            d.2 - 1
        };
        let Some(dir) = super::shader_dump::DIR.get().and_then(|d| d.parent()).map(|d| d.join("after")) else { return };
        let _ = std::fs::create_dir_all(&dir);
        self.end_rendering();
        for a in colors.iter().flatten().chain(depth.iter()) {
            self.dump_image(&dir, a.key, 0, &format!("{n:03}_ps{ps:x}_"));
        }
    }
}

/// Clear value for a color image from CB_COLOR*_CLEAR_WORD0/1, which hold
/// the clear color in the surface's own memory layout.
fn clear_color(f: vk::Format, w0: u32, w1: u32) -> Option<vk::ClearColorValue> {
    use vk::Format as F;
    let b = |i: u32| ((w0 >> (8 * i)) & 0xFF) as f32 / 255.0;
    let lin = |c: f32| if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) };
    // Unsigned small floats (5-bit exponent, no sign).
    let uf = |v: u32, mbits: u32| -> f32 {
        let m = v & ((1 << mbits) - 1);
        let e = (v >> mbits) & 0x1F;
        match e {
            0 => m as f32 / (1 << mbits) as f32 * 2f32.powi(-14),
            31 => if m == 0 { f32::INFINITY } else { f32::NAN },
            _ => (1.0 + m as f32 / (1 << mbits) as f32) * 2f32.powi(e as i32 - 15),
        }
    };
    let h = |v: u32| half(v as u16);
    let float = |v: [f32; 4]| Some(vk::ClearColorValue { float32: v });
    match f {
        F::R8G8B8A8_UNORM => float([b(0), b(1), b(2), b(3)]),
        F::R8G8B8A8_SRGB => float([lin(b(0)), lin(b(1)), lin(b(2)), b(3)]),
        F::B8G8R8A8_UNORM => float([b(2), b(1), b(0), b(3)]),
        F::B8G8R8A8_SRGB => float([lin(b(2)), lin(b(1)), lin(b(0)), b(3)]),
        F::R8_UNORM => float([b(0), 0.0, 0.0, 1.0]),
        F::R8G8_UNORM => float([b(0), b(1), 0.0, 1.0]),
        F::B10G11R11_UFLOAT_PACK32 => float([uf(w0 & 0x7FF, 6), uf((w0 >> 11) & 0x7FF, 6), uf(w0 >> 22, 5), 1.0]),
        F::A2B10G10R10_UNORM_PACK32 => float([(w0 & 0x3FF) as f32 / 1023.0, ((w0 >> 10) & 0x3FF) as f32 / 1023.0, ((w0 >> 20) & 0x3FF) as f32 / 1023.0, (w0 >> 30) as f32 / 3.0]),
        F::R16_SFLOAT => float([h(w0), 0.0, 0.0, 1.0]),
        F::R16G16_SFLOAT => float([h(w0), h(w0 >> 16), 0.0, 1.0]),
        F::R16G16B16A16_SFLOAT => float([h(w0), h(w0 >> 16), h(w1), h(w1 >> 16)]),
        F::R16_UNORM => float([(w0 & 0xFFFF) as f32 / 65535.0, 0.0, 0.0, 1.0]),
        F::R16G16_UNORM => float([(w0 & 0xFFFF) as f32 / 65535.0, (w0 >> 16) as f32 / 65535.0, 0.0, 1.0]),
        F::R16G16B16A16_UNORM => float([(w0 & 0xFFFF) as f32 / 65535.0, (w0 >> 16) as f32 / 65535.0, (w1 & 0xFFFF) as f32 / 65535.0, (w1 >> 16) as f32 / 65535.0]),
        F::R32_SFLOAT => float([f32::from_bits(w0), 0.0, 0.0, 1.0]),
        F::R32G32_SFLOAT => float([f32::from_bits(w0), f32::from_bits(w1), 0.0, 1.0]),
        F::R32_UINT => Some(vk::ClearColorValue { uint32: [w0, 0, 0, 1] }),
        F::R32G32_UINT => Some(vk::ClearColorValue { uint32: [w0, w1, 0, 1] }),
        _ if w0 == 0 && w1 == 0 => Some(vk::ClearColorValue { uint32: [0; 4] }),
        _ => None,
    }
}

fn half(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1F) as i32;
    let m = (h & 0x3FF) as f32;
    match e {
        0 => s * m * 2f32.powi(-24),
        31 => if m == 0.0 { s * f32::INFINITY } else { f32::NAN },
        _ => s * (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

/// Converts raw image data to RGBA8 for viewing; returns per-channel stats.
fn to_viewable(f: vk::Format, raw: &[u8], w: u32, h: u32) -> (Option<Vec<u8>>, String) {
    use vk::Format as F;
    let n = (w * h) as usize;
    let mut chans: Vec<[f32; 4]> = Vec::with_capacity(n);
    match f {
        F::R8G8B8A8_UNORM | F::R8G8B8A8_SRGB => chans.extend(raw.chunks_exact(4).map(|p| [p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0, p[3] as f32 / 255.0])),
        F::B8G8R8A8_UNORM | F::B8G8R8A8_SRGB => chans.extend(raw.chunks_exact(4).map(|p| [p[2] as f32 / 255.0, p[1] as f32 / 255.0, p[0] as f32 / 255.0, p[3] as f32 / 255.0])),
        F::R32_SFLOAT => chans.extend(raw.chunks_exact(4).map(|p| {
            let v = f32::from_le_bytes([p[0], p[1], p[2], p[3]]);
            [v, v, v, 1.0]
        })),
        F::R16G16B16A16_SFLOAT => chans.extend(raw.chunks_exact(8).map(|p| {
            let g = |i: usize| half(u16::from_le_bytes([p[i], p[i + 1]]));
            [g(0), g(2), g(4), g(6)]
        })),
        F::R16G16_SFLOAT => chans.extend(raw.chunks_exact(4).map(|p| {
            let g = |i: usize| half(u16::from_le_bytes([p[i], p[i + 1]]));
            [g(0), g(2), 0.0, 1.0]
        })),
        F::R16_SFLOAT => chans.extend(raw.chunks_exact(2).map(|p| {
            let v = half(u16::from_le_bytes([p[0], p[1]]));
            [v, v, v, 1.0]
        })),
        F::R8_UNORM => chans.extend(raw.iter().map(|&v| [v as f32 / 255.0, v as f32 / 255.0, v as f32 / 255.0, 1.0])),
        F::R8G8_UNORM => chans.extend(raw.chunks_exact(2).map(|p| [p[0] as f32 / 255.0, p[1] as f32 / 255.0, 0.0, 1.0])),
        F::A2B10G10R10_UNORM_PACK32 => chans.extend(raw.chunks_exact(4).map(|p| {
            let v = u32::from_le_bytes([p[0], p[1], p[2], p[3]]);
            [(v & 0x3FF) as f32 / 1023.0, ((v >> 10) & 0x3FF) as f32 / 1023.0, ((v >> 20) & 0x3FF) as f32 / 1023.0, (v >> 30) as f32 / 3.0]
        })),
        F::B10G11R11_UFLOAT_PACK32 => chans.extend(raw.chunks_exact(4).map(|p| {
            let v = u32::from_le_bytes([p[0], p[1], p[2], p[3]]);
            let f11 = |x: u32| { let e = ((x >> 6) & 0x1F) as i32; let m = (x & 0x3F) as f32; if e == 0 { m * 2f32.powi(-20) } else { (1.0 + m / 64.0) * 2f32.powi(e - 15) } };
            let f10 = |x: u32| { let e = ((x >> 5) & 0x1F) as i32; let m = (x & 0x1F) as f32; if e == 0 { m * 2f32.powi(-19) } else { (1.0 + m / 32.0) * 2f32.powi(e - 15) } };
            [f11(v & 0x7FF), f11((v >> 11) & 0x7FF), f10(v >> 22), 1.0]
        })),
        _ => return (None, format!("format not viewable ({} bytes)", raw.len())),
    }
    // Statistics over finite values; NaN/inf counted separately.
    let mut mn = [f32::INFINITY; 4];
    let mut mx = [f32::NEG_INFINITY; 4];
    let mut sum = [0f64; 4];
    let mut fin = [0usize; 4];
    let mut bad = [0usize; 4];
    for c in &chans {
        for i in 0..4 {
            if c[i].is_finite() {
                mn[i] = mn[i].min(c[i]);
                mx[i] = mx[i].max(c[i]);
                sum[i] += c[i] as f64;
                fin[i] += 1;
            } else {
                bad[i] += 1;
            }
        }
    }
    let mean = |i: usize| sum[i] / fin[i].max(1) as f64;
    let mut stats = format!(
        "min [{:.4} {:.4} {:.4} {:.4}] max [{:.4} {:.4} {:.4} {:.4}] mean [{:.4} {:.4} {:.4} {:.4}]",
        mn[0], mn[1], mn[2], mn[3], mx[0], mx[1], mx[2], mx[3], mean(0), mean(1), mean(2), mean(3)
    );
    if bad.iter().any(|&b| b > 0) {
        stats += &format!(" non-finite [{} {} {} {}] of {n}", bad[0], bad[1], bad[2], bad[3]);
    }
    // PNG: dark or HDR images scaled so the 99th percentile of their finite
    // RGB values maps to white; non-finite pixels magenta.
    let mut vals: Vec<f32> = chans.iter().flat_map(|c| [c[0], c[1], c[2]]).filter(|v| v.is_finite() && *v > 0.0).collect();
    let p99 = if vals.is_empty() {
        0.0
    } else {
        let k = (vals.len() - 1) * 99 / 100;
        *vals.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1
    };
    let scale = if p99 > 0.0 && !(0.25..=1.0).contains(&p99) { 1.0 / p99 } else { 1.0 };
    if scale != 1.0 {
        stats += &format!(" (png scaled x{scale:.3})");
    }
    let px: Vec<u8> = chans
        .iter()
        .flat_map(|c| if c[..3].iter().all(|v| v.is_finite()) { [c[0], c[1], c[2]].map(|v| ((v * scale).clamp(0.0, 1.0) * 255.0) as u8) } else { [255, 0, 255] }.into_iter().chain([255]))
        .collect();
    (Some(px), stats)
}

fn buffer_written(p: &Prepared, i: usize) -> bool {
    p.translated.out.bindings.iter().any(|b| b.kind == BindingKind::Buffer(i)) && p.written_buffers.contains(&i)
}

/// Bytes of a `size`-byte binding at guest `base` that lie in its mapping:
/// unbounded descriptors (num_records -1, common in SOTC) end where the
/// mapping does. (Small bindings are taken as they are.)
fn guest_extent(base: u64, size: u64) -> u64 {
    if size <= 1 << 20 {
        return size;
    }
    match crate::mem::MEM.lock().committed_span(base) {
        Some((_, end)) => size.min(end - base),
        None => size,
    }
}

/// Whether two images at one guest address are the same surface up to the
/// padding render targets get (1920x1088 for 1920x1080, pitch alignment):
/// more than half of each other in each dimension. Titles also place
/// different surfaces in the same transient memory at different times (a
/// 960x540 target where a 1920x1080 one was): their contents are unrelated
/// (copying one into the other painted a half-resolution quadrant into the
/// top left of the screen).
fn same_surface(w0: u32, h0: u32, w1: u32, h1: u32) -> bool {
    w0 * 2 > w1 && w1 * 2 > w0 && h0 * 2 > h1 && h1 * 2 > h0
}

/// Storage format for an image: sRGB stored as UNORM (views reinterpret).
fn canonical(f: vk::Format) -> vk::Format {
    match f {
        vk::Format::R8G8B8A8_SRGB => vk::Format::R8G8B8A8_UNORM,
        vk::Format::B8G8R8A8_SRGB => vk::Format::B8G8R8A8_UNORM,
        vk::Format::R8_SRGB => vk::Format::R8_UNORM,
        f => f,
    }
}

/// Bits per block of a block-compressed format.
fn bc_block_bits(f: vk::Format) -> Option<u32> {
    use vk::Format as F;
    match f {
        F::BC1_RGB_UNORM_BLOCK | F::BC1_RGB_SRGB_BLOCK | F::BC1_RGBA_UNORM_BLOCK | F::BC1_RGBA_SRGB_BLOCK | F::BC4_UNORM_BLOCK | F::BC4_SNORM_BLOCK => Some(64),
        F::BC2_UNORM_BLOCK | F::BC2_SRGB_BLOCK | F::BC3_UNORM_BLOCK | F::BC3_SRGB_BLOCK | F::BC5_UNORM_BLOCK | F::BC5_SNORM_BLOCK | F::BC6H_UFLOAT_BLOCK | F::BC6H_SFLOAT_BLOCK | F::BC7_UNORM_BLOCK | F::BC7_SRGB_BLOCK => Some(128),
        _ => None,
    }
}

/// Coarse format compatibility class (bytes per texel / block kind).
fn format_class(f: vk::Format) -> u32 {
    use vk::Format as F;
    match f {
        F::R8G8B8A8_UNORM | F::R8G8B8A8_SRGB | F::B8G8R8A8_UNORM | F::B8G8R8A8_SRGB | F::R8G8B8A8_SNORM | F::R8G8B8A8_UINT | F::R8G8B8A8_SINT | F::B8G8R8A8_SNORM | F::A2B10G10R10_UNORM_PACK32
        | F::R32_SFLOAT | F::R32_UINT | F::R32_SINT | F::R16G16_SFLOAT | F::R16G16_UNORM | F::R16G16_UINT | F::B10G11R11_UFLOAT_PACK32 => 4,
        F::R16G16B16A16_SFLOAT | F::R16G16B16A16_UNORM | F::R16G16B16A16_UINT | F::R32G32_SFLOAT | F::R32G32_UINT => 8,
        F::R32G32B32A32_SFLOAT | F::R32G32B32A32_UINT => 16,
        F::R8_UNORM | F::R8_UINT | F::R8_SRGB | F::R8_SNORM => 1,
        F::R16_UNORM | F::R16_SFLOAT | F::R16_UINT | F::R8G8_UNORM | F::R8G8_UINT => 2,
        other => 0x1000 + other.as_raw() as u32,
    }
}

/// Hooks implementation that drives the Vulkan renderer.
pub struct VkHooks {
    pub fallback: super::NullRenderer,
}

/// Per command-processor queue (0: graphics, else the compute queue's
/// virtual index): nanoseconds spent waiting for the renderer lock and
/// holding it, and in WAIT_REG_MEM (TLG_PERF report).
pub static QUEUE_LOCK_WAIT_NS: [std::sync::atomic::AtomicU64; 64] = [const { std::sync::atomic::AtomicU64::new(0) }; 64];
pub static QUEUE_LOCK_HOLD_NS: [std::sync::atomic::AtomicU64; 64] = [const { std::sync::atomic::AtomicU64::new(0) }; 64];
pub static QUEUE_WAIT_REG_NS: [std::sync::atomic::AtomicU64; 64] = [const { std::sync::atomic::AtomicU64::new(0) }; 64];

/// The renderer lock taken for queue `q`, timed (see `QUEUE_LOCK_WAIT_NS`).
struct TimedRenderer {
    g: std::mem::ManuallyDrop<parking_lot::MutexGuard<'static, Option<Renderer>>>,
    q: usize,
    since: std::time::Instant,
}

impl TimedRenderer {
    fn lock(q: usize) -> TimedRenderer {
        let t0 = std::time::Instant::now();
        let g = RENDERER.lock();
        let since = std::time::Instant::now();
        QUEUE_LOCK_WAIT_NS[q & 63].fetch_add((since - t0).as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        TimedRenderer { g: std::mem::ManuallyDrop::new(g), q, since }
    }
}

impl Drop for TimedRenderer {
    fn drop(&mut self) {
        QUEUE_LOCK_HOLD_NS[self.q & 63].fetch_add(self.since.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: `g` is not used after this.
        let g = unsafe { std::mem::ManuallyDrop::take(&mut self.g) };
        // (`tune lock_fair`: the graphics CP lets a waiting compute CP in.)
        if self.q == 0 && crate::tune::get(crate::tune::LOCK_FAIR) != 0 {
            parking_lot::MutexGuard::unlock_fair(g);
        } else {
            drop(g);
        }
    }
}

impl std::ops::Deref for TimedRenderer {
    type Target = Option<Renderer>;
    fn deref(&self) -> &Option<Renderer> {
        &self.g
    }
}

impl std::ops::DerefMut for TimedRenderer {
    fn deref_mut(&mut self) -> &mut Option<Renderer> {
        &mut self.g
    }
}

impl Hooks for VkHooks {
    fn draw(&mut self, cp: &CommandProcessor, kind: &'static str, count: u32, index_addr: u64) {
        // Shader preparation reads registers and guest memory only: before
        // the lock (its reads are checked under it). Not for patches.
        let prim = cp.regs.uconfig[(0xC242 - 0xC000) as usize];
        let pre = (crate::tune::get(crate::tune::PREPARE_UNLOCKED) != 0 && prim != 9).then(|| {
            let regs = &cp.regs;
            let ps_addr = ((regs.sh(0x2C08) as u64) | ((regs.sh(0x2C09) as u64) << 32)) << 8;
            let vs = shaders::prepare_tess(&StageRegs { stage: Stage::Vertex, regs }, Default::default());
            let ps = (ps_addr != 0).then(|| shaders::prepare(&StageRegs { stage: Stage::Fragment, regs }));
            (vs, ps)
        });
        let mut g = TimedRenderer::lock(cp.queue);
        let Some(r) = g.as_mut() else { return self.fallback.draw(cp, kind, count, index_addr) };
        r.pre_draw = pre;
        if let Err(e) = r.draw_inner(cp, kind, count, index_addr) {
            r.skipped += 1;
            r.report(format!("draw skipped: {e}"));
        }
        r.maybe_submit();
    }
    fn dispatch(&mut self, cp: &CommandProcessor, x: u32, y: u32, z: u32) {
        let pre = (crate::tune::get(crate::tune::PREPARE_UNLOCKED) != 0).then(|| shaders::prepare(&StageRegs { stage: Stage::Compute, regs: &cp.regs }));
        let mut g = TimedRenderer::lock(cp.queue);
        let Some(r) = g.as_mut() else { return self.fallback.dispatch(cp, x, y, z) };
        r.pre_cs = pre;
        if let Err(e) = r.dispatch_inner(cp, x, y, z, None) {
            r.skipped += 1;
            r.report(format!("dispatch skipped: {e}"));
        }
        r.maybe_submit();
    }
    fn dispatch_indirect(&mut self, cp: &CommandProcessor, addr: u64) -> bool {
        if crate::tune::get(crate::tune::GPU_INDIRECT) == 0 {
            return false;
        }
        let mut g = TimedRenderer::lock(cp.queue);
        let Some(r) = g.as_mut() else { return false };
        let done = match r.dispatch_inner(cp, 0, 0, 0, Some(addr)) {
            Ok(done) => done,
            Err(e) => {
                r.report(format!("indirect dispatch with GPU-read arguments failed (dispatched after waiting for them instead): {e}"));
                false
            }
        };
        if done {
            r.maybe_submit();
        }
        done
    }
    fn draw_indirect_gpu(&mut self, cp: &CommandProcessor, addr: u64, indexed: bool, vtx_loc: u32, inst_loc: u32) -> bool {
        let mut g = TimedRenderer::lock(cp.queue);
        let Some(r) = g.as_mut() else { return false };
        let done = match r.draw_indirect_gpu(cp, addr, indexed, vtx_loc, inst_loc) {
            Ok(done) => done,
            Err(e) => {
                // (Recorded up to the failure, as other draws that fail.)
                r.skipped += 1;
                r.report(format!("indirect draw with GPU-read arguments skipped: {e}"));
                true
            }
        };
        if done {
            r.maybe_submit();
        }
        done
    }
    fn gds(&mut self, cp: &CommandProcessor, op: super::cp::GdsOp) {
        let mut g = RENDERER.lock();
        let Some(r) = g.as_mut() else { return self.fallback.gds(cp, op) };
        if let Err(e) = r.gds_op(op) {
            r.report(format!("GDS transfer skipped: {e}"));
        }
    }
    fn flush(&mut self, _cp: &CommandProcessor) {
        if let Some(r) = RENDERER.lock().as_mut() {
            r.flush_for(0);
            // The caller is about to access guest memory directly (DMA etc.).
            r.new_epoch(EpochKind::Both);
        }
    }
    fn sync_range(&mut self, cp: &CommandProcessor, addr: u64, len: u64) {
        // (Without overlap in the view, nothing of this batch is in the way.)
        if !cp_view_on() || CP_VIEW.lock().pending.overlaps(addr, addr + len.max(1)) {
            if let Some(r) = TimedRenderer::lock(cp.queue).as_mut() {
                if r.pending_write_overlaps(addr, len) {
                    r.flushes[1] += 1;
                    r.submit_batch(Vec::new());
                }
            }
        }
        wait_overlapping(addr, len);
    }
    fn dma_on_gpu(&mut self, _cp: &CommandProcessor, src: u64, dst: u64, bytes: u64) -> bool {
        RENDERER.lock().as_mut().is_some_and(|r| r.dma_on_gpu(src, dst, bytes))
    }
    fn gds_load_on_gpu(&mut self, _cp: &CommandProcessor, offset: u32, src: u64, bytes: u32) -> bool {
        RENDERER.lock().as_mut().is_some_and(|r| r.gds_load_on_gpu(offset, src, bytes))
    }
    fn sync_write(&mut self, _cp: &CommandProcessor, addr: u64, len: u64) {
        let end = addr + len.max(1);
        let read = if cp_view_on() {
            let mut v = CP_VIEW.lock();
            v.cp_writes.push((addr, len));
            CP_WRITES_QUEUED.store(true, std::sync::atomic::Ordering::Release);
            v.reads.overlaps(addr, end)
        } else {
            if let Some(r) = RENDERER.lock().as_mut() {
                r.vcache_invalidate(addr, len);
                r.forget_uploads(addr, len);
            }
            true
        };
        if read {
            if let Some(r) = RENDERER.lock().as_mut() {
                if r.zc_reads.overlaps(addr, end) {
                    r.flushes[1] += 1;
                    r.submit_batch(Vec::new());
                }
            }
        }
        let seq = COMPLETIONS.lock().iter().filter(|c| c.zc_reads.overlaps(addr, end)).map(|c| c.seq).max();
        if let Some(seq) = seq {
            let t0 = std::time::Instant::now();
            wait_done(seq);
            let ns = t0.elapsed().as_nanos() as u64;
            SYNC_WAIT_NS.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
            SYNC_WAIT_KIND_NS[4].fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
        }
    }
    fn occlusion_dump(&mut self, _cp: &CommandProcessor, begin: bool) -> Option<Vec<u32>> {
        RENDERER.lock().as_mut().map(|r| r.occlusion_dump(begin))
    }
    fn order_barrier(&mut self, _cp: &CommandProcessor) {
        if let Some(r) = RENDERER.lock().as_mut() {
            r.order_barrier();
        }
    }
    fn label_only(&mut self, _cp: &CommandProcessor, addr: u64, len: u64) -> bool {
        let end = addr + len.max(1);
        let g = RENDERER.lock();
        let Some(r) = g.as_ref() else { return false };
        if r.pending_write_overlaps(addr, len) || r.zc_reads.overlaps(addr, end) {
            return false;
        }
        drop(g);
        !COMPLETIONS.lock().iter().any(|c| c.writebacks.iter().any(|&(w, _, n)| w < end && addr < w + n) || c.zc_writes.iter().any(|&(w, n)| w < end && addr < w + n) || c.zc_reads.overlaps(addr, end))
    }
    fn commit(&mut self, cp: &CommandProcessor, actions: Vec<super::cp::EopAction>) {
        let mut g = TimedRenderer::lock(cp.queue);
        match g.as_mut() {
            Some(r) => {
                r.submit_batch(actions);
            }
            None => {
                drop(g);
                self.fallback.commit(cp, actions);
            }
        }
    }
}

/// Flip to trace requested through the control channel (`trace` command).
pub static TRACE_REQUEST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);
/// Debugging (control commands `dumpafter`, `skipps`): pixel shaders whose
/// draws in traced frames get their targets dumped afterwards (addresses,
/// maximum number of dumps, dumps so far), and pixel shaders whose draws
/// are skipped.
pub static DUMP_AFTER: Mutex<Option<(Vec<u64>, u32, u32)>> = Mutex::new(None);
pub static SKIP_PS: Mutex<Vec<u64>> = Mutex::new(Vec::new());
/// Debug: dispatches of these compute shaders are skipped (control `skipcs`).
pub static SKIP_CS: Mutex<Vec<u64>> = Mutex::new(Vec::new());
/// Depth targets sampled through a smaller T#: cropped copies made, direct
/// views for texel loads (`depth_shadow_view`).
pub static DEPTH_VIEWS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
/// Debug: image dumps also write the raw texels (control `dumpraw 1`).
pub static DUMP_RAW: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// "0x1a2b,3c4d" -> addresses.
pub fn parse_hex_list(v: &str) -> Vec<u64> {
    v.split(',').filter_map(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok()).collect()
}

/// True while the flip selected by TLG_FRAME_TRACE (or a `trace` request) is being built.
pub(super) fn trace_frame() -> bool {
    static WANT: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    // TLG_TRACE_FROM=<seconds>: trace (and run synchronously) every command
    // processed after that much process time, e.g. to pin down a GPU hang.
    static FROM: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    if let Some(from) = *FROM.get_or_init(|| std::env::var("TLG_TRACE_FROM").ok().and_then(|v| v.parse::<f64>().ok())) {
        if crate::log::elapsed_ms() >= from * 1000.0 {
            return true;
        }
    }
    let frame = super::FRAMES.load(std::sync::atomic::Ordering::Relaxed);
    if TRACE_REQUEST.load(std::sync::atomic::Ordering::Relaxed) == frame {
        return true;
    }
    match *WANT.get_or_init(|| std::env::var("TLG_FRAME_TRACE").ok().and_then(|v| v.parse().ok())) {
        Some(w) => frame == w,
        None => false,
    }
}

/// Small buffers bound in the traced frame with their contents' hash when
/// the command was recorded (see `traced_rehash`).
static TRACED_READS: parking_lot::Mutex<Vec<(u64, u64, u64)>> = parking_lot::Mutex::new(Vec::new());

/// End of a traced frame: buffers whose contents changed after a command of
/// the frame used them -- by later GPU work of the frame, or by the title's
/// CPU while we processed the frame (data the command never saw).
pub fn traced_rehash() {
    let reads = std::mem::take(&mut *TRACED_READS.lock());
    let mut seen = std::collections::HashSet::new();
    let mut changed = Vec::new();
    for (base, size, h) in reads.into_iter().rev() {
        // (The last use of each buffer: earlier uses may precede legitimate
        // writes within the frame.)
        if !seen.insert((base, size)) {
            continue;
        }
        if resources::hash_guest(base, size) != h {
            changed.push(format!("{base:#x}+{size:#x}"));
        }
    }
    crate::info!("FRAME buffers changed after their last use in the frame: {} [{}]", changed.len(), changed.iter().take(40).cloned().collect::<Vec<_>>().join(" "));
}

/// Frame trace: a stage's resources with the pcs of the instructions using them.
fn stage_res_summary(p: &Prepared) -> Vec<String> {
    let tag = match p.stage {
        Stage::Vertex => "vs",
        Stage::Fragment => "ps",
        Stage::Compute => "cs",
    };
    let ud: Vec<String> = p.sgpr_data.iter().take(16).map(|w| format!("{w:x}")).collect();
    // TLG_TRACE_FULL=<shader addrs>: whole buffers (up to 256 dwords) of
    // these shaders' resources in traces.
    static FULL: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    let full = FULL.get_or_init(|| std::env::var("TLG_TRACE_FULL").map(|v| v.split(',').filter_map(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok()).collect()).unwrap_or_default());
    TRACE_FULL.with(|f| f.set(full.contains(&p.addr)));
    let reads: Vec<String> = p.mem_reads.iter().map(|a| format!("{a:x}")).collect();
    let out = std::iter::once(format!("{tag}ud [{}] reads [{}]", ud.join(" "), reads.join(" ")))
        .chain(p.resources.iter().enumerate().map(|(i, r)| {
            let pcs: Vec<String> = p.res_pcs.get(i).map(|v| v.iter().map(|pc| format!("{pc:x}")).collect()).unwrap_or_default();
            format!("{tag}{i}@{} {}", pcs.join("/"), res_summary(r))
        }))
        .collect();
    TRACE_FULL.with(|f| f.set(false));
    out
}

thread_local! {
    static TRACE_FULL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn res_summary(r: &ResolvedRes) -> String {
    match r {
        ResolvedRes::Buffer(v) => {
            // TLG_TRACE_DWORDS: how many leading dwords of each buffer to show (default 4).
            static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
            let n = (*N.get_or_init(|| std::env::var("TLG_TRACE_DWORDS").ok().and_then(|v| v.parse().ok()).unwrap_or(4))).min(256);
            let n = if TRACE_FULL.with(|f| f.get()) { (v.size() as usize / 4).clamp(4, 256) } else { n };
            let mut head = vec![0u8; n * 4];
            resources::copy_from_guest(&mut head, v.base);
            let w: Vec<String> = head.chunks(4).map(|c| format!("{:08x}", u32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect();
            // TLG_TRACE_BUFDUMP=1: save buffers up to 1 MiB to <run>/trace_bufs/
            // (contents when the command is recorded; traced commands run one
            // at a time, so earlier GPU writes have landed).
            static DUMP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let mut file = String::new();
            if *DUMP.get_or_init(|| std::env::var_os("TLG_TRACE_BUFDUMP").is_some()) && v.size() <= 1 << 20 {
                static SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if let Some(dir) = super::shader_dump::DIR.get().and_then(|d| d.parent()).map(|d| d.join("trace_bufs")) {
                    let _ = std::fs::create_dir_all(&dir);
                    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let mut data = vec![0u8; v.size().max(4) as usize];
                    resources::copy_from_guest(&mut data, v.base);
                    let name = format!("{seq:06}_{:x}.bin", v.base);
                    let _ = std::fs::write(dir.join(&name), &data);
                    file = format!(" file {name}");
                }
            }
            // Hash of small buffers' whole contents (comparing frames); kept
            // to re-check at the end of the traced frame (`traced_rehash`).
            let hash = if v.size() <= 64 << 10 {
                let h = resources::hash_guest(v.base, v.size());
                TRACED_READS.lock().push((v.base, v.size(), h));
                format!(" h{:08x}", h as u32)
            } else {
                String::new()
            };
            format!("V#{:#x}+{:#x}(s{} f{}/{}{}{hash} [{}])", v.base, v.size(), v.stride, v.dfmt, v.nfmt, file, w.join(" "))
        }
        ResolvedRes::Image(t) => {
            let sel: String = (0..4).map(|c| ['0', '1', '?', '?', 'r', 'g', 'b', 'a'][((t.dst_sel >> (c * 3)) & 7) as usize]).collect();
            format!("T#{:#x} {}x{}x{} f{}/{} t{} ty{} arr{}..{} d{} lv{}..{} sel {}", t.base, t.width, t.height, t.layers(), t.dfmt, t.nfmt, t.tiling_index, t.ty, t.base_array, t.last_array, t.depth, t.base_level, t.last_level, sel)
        }
        ResolvedRes::Sampler(s) => format!("S#[{:08x} {:08x} {:08x} {:08x}] cmp {}", s.0[0], s.0[1], s.0[2], s.0[3], s.compare_func()),
    }
}

/// `draw_inner` phases timed in `Renderer::dprof`.
pub const DRAW_PHASES: [&str; 8] = ["setup", "shaders", "targets", "pipeline", "guest reads", "descriptors", "pass", "record"];
/// Parts of the descriptor phase timed in `Renderer::dsub` (all commands).
pub const DESC_PARTS: [&str; 6] = ["alloc", "sgpr data", "buffers", "images", "samplers+other", "update"];
/// Buffer binding paths timed in `Renderer::bsub` (`write_descriptors`).
pub const BUFFER_PATHS: [&str; 9] = ["other", "in place read", "in place written", "mirror read", "mirror written", "reused copy", "read-only copy", "written round trip", "written copy"];

pub const PROF_NAMES: [&str; 7] = ["submit+wait", "writeback", "buffer upload", "texture hash", "texture read", "shader prep", "slot wait"];

pub const FLUSH_REASONS: [&str; 6] = ["eop/dma", "writeback sync", "ring full", "descriptors", "flip", "fwd copies"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_merge_and_contain() {
        let list = |r: &Ranges| r.0.iter().map(|(&a, &b)| (a, b)).collect::<Vec<_>>();
        let mut r = Ranges::default();
        r.insert(10, 20);
        r.insert(30, 40);
        r.insert(12, 18);
        assert_eq!(list(&r), [(10, 20), (30, 40)]);
        r.insert(20, 30);
        assert_eq!(list(&r), [(10, 40)]);
        r.insert(60, 70);
        r.insert(55, 65);
        r.insert(5, 50);
        assert_eq!(list(&r), [(5, 50), (55, 70)]);
        r.insert(45, 58);
        assert_eq!(list(&r), [(5, 70)]);
        assert!(r.overlaps(69, 80) && !r.overlaps(70, 80) && !r.overlaps(0, 5));
    }

    #[test]
    fn fill_elements_and_clear_colors() {
        let f = |v: [f32; 4]| v.map(f32::to_bits);
        const XYZW: u16 = 4 | 5 << 3 | 6 << 6 | 7 << 9;
        const ZYXW: u16 = 6 | 5 << 3 | 4 << 6 | 7 << 9;
        // Values seen in the clinic: unorm8 fills (0.5 rounds up like GCN), float16 1.0.
        assert_eq!(fill_element(10, 0, ZYXW, &f([0.5, 0.5, 0.5, 1.0])), Some(vec![0x80, 0x80, 0x80, 0xFF]));
        assert_eq!(fill_element(10, 0, XYZW, &[0x3efefeff, 0x3efefeff, 0x3f800000, 0]), Some(vec![0x7F, 0x7F, 0xFF, 0x00]));
        assert_eq!(fill_element(5, 7, XYZW, &f([1.0; 4])), Some(vec![0x00, 0x3C, 0x00, 0x3C]));
        // HTILE clear fill (32_32_32_32 UINT, "depth cleared" words).
        assert_eq!(fill_element(14, 4, XYZW, &[0xffff_fff0; 4]), Some([0xffff_fff0u32; 4].iter().flat_map(|v| v.to_le_bytes()).collect()));
        assert_eq!(fill_element(12, 7, XYZW, &f([0.5, -2.0, 0.0, 1.0])), Some(vec![0x00, 0x38, 0x00, 0xC0, 0x00, 0x00, 0x00, 0x3C]));
        assert_eq!(fill_element(4, 7, XYZW, &f([3.25, 0.0, 0.0, 0.0])), Some(3.25f32.to_le_bytes().to_vec()));
        // Depends on whether stores swizzle; inexact in float16; unknown formats.
        assert_eq!(fill_element(10, 0, ZYXW, &f([1.0, 0.0, 0.0, 1.0])), None);
        assert_eq!(fill_element(5, 7, XYZW, &f([0.1, 0.1, 0.0, 0.0])), None);
        assert_eq!(fill_element(10, 4, XYZW, &f([1.0; 4])), None);
        let c = |fmt, e: &[u8]| clear_color_for(fmt, e).map(|c| unsafe { c.float32 });
        assert_eq!(c(vk::Format::B8G8R8A8_UNORM, &[0x00, 0x33, 0xFF, 0x66]), Some([1.0, 0.2, 0.0, 0.4]));
        assert_eq!(c(vk::Format::R8G8B8A8_UNORM, &[0x00, 0x33, 0xFF, 0x66]), Some([0.0, 0.2, 1.0, 0.4]));
        assert_eq!(c(vk::Format::R16G16_SFLOAT, &[0x00, 0x3C, 0x00, 0xB8]), Some([1.0, -0.5, 0.0, 0.0]));
        assert_eq!(c(vk::Format::R32_SFLOAT, &3.25f32.to_le_bytes()), Some([3.25, 0.0, 0.0, 0.0]));
        assert_eq!(c(vk::Format::R8G8B8A8_UNORM, &[0; 8]), None);
    }

    #[test]
    fn vram_pressure_levels_and_eviction_classes() {
        assert_eq!(vram_pressure(0, 0), 0);
        assert_eq!(vram_pressure(50, 100), 0);
        assert_eq!(vram_pressure(70, 100), 1);
        assert_eq!(vram_pressure(90, 100), 2);
        assert_eq!(vram_pressure(99, 100), 3);
        assert_eq!(vram_pressure(150, 100), 3);
        let k = |base| ImageKey { base, width: 4, height: 4, layers: 1, levels: 1, format: vk::Format::R8G8B8A8_UNORM, depth: false, ty3d: false };
        assert!(is_shadow(&k(0x10_0000_0000 ^ 0x4000_0000_0000_0000)));
        assert!(is_shadow(&k(0x10_0000_0000 ^ 0x8000_0000_0000_0000)));
        assert!(!is_shadow(&k(0x10_0000_0000)));
        // (Zero-image stand-ins live at 0xD000...: GPU-written, kept.)
        assert!(!is_shadow(&k(0xD000_0000_0000_0000)));
    }

    #[test]
    fn vcache_page_bits() {
        let mut v = vec![0u64; 4];
        assert_eq!(bits_clear_runs(&v, 0, 256), vec![(0, 256)]);
        bits_set(&mut v, 3, 130, true);
        assert_eq!(v[0], !0b111);
        assert_eq!(v[1], u64::MAX);
        assert_eq!(v[2], 0b11);
        assert_eq!(bits_clear_runs(&v, 0, 256), vec![(0, 3), (130, 256)]);
        assert_eq!(bits_clear_runs(&v, 10, 120), vec![]);
        bits_set(&mut v, 64, 65, false);
        bits_set(&mut v, 100, 110, false);
        assert_eq!(bits_clear_runs(&v, 0, 140), vec![(0, 3), (64, 65), (100, 110), (130, 140)]);
        bits_set(&mut v, 0, 256, true);
        assert!(v.iter().all(|&w| w == u64::MAX));
        assert!(bits_clear_runs(&v, 0, 256).is_empty());
        // Against a bit-by-bit reference on pseudo-random patterns.
        let reference = |v: &[u64], lo: usize, hi: usize| {
            let mut out: Vec<(usize, usize)> = Vec::new();
            for i in lo..hi {
                if v[i >> 6] & (1 << (i & 63)) == 0 {
                    match out.last_mut() {
                        Some(r) if r.1 == i => r.1 = i + 1,
                        _ => out.push((i, i + 1)),
                    }
                }
            }
            out
        };
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..2000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let w: Vec<u64> = (0..4).map(|k| x.rotate_left(k * 17) & x.rotate_right(k * 5 + 3)).collect();
            let lo = (x % 256) as usize;
            let hi = lo + ((x >> 8) % (257 - lo as u64)) as usize;
            assert_eq!(bits_clear_runs(&w, lo, hi), reference(&w, lo, hi), "{w:x?} {lo}..{hi}");
        }
    }
}

#[cfg(test)]
mod range_tests {
    use super::subtract_ranges;

    #[test]
    fn subtract() {
        assert_eq!(subtract_ranges(&[(0, 100)], &[]), vec![(0, 100)]);
        assert_eq!(subtract_ranges(&[(0, 100)], &[(20, 30)]), vec![(0, 20), (30, 100)]);
        assert_eq!(subtract_ranges(&[(0, 100)], &[(0, 100)]), vec![]);
        assert_eq!(subtract_ranges(&[(0, 100)], &[(50, 200)]), vec![(0, 50)]);
        assert_eq!(subtract_ranges(&[(10, 20), (30, 40)], &[(15, 35)]), vec![(10, 15), (35, 40)]);
        assert_eq!(subtract_ranges(&[(0, 4096)], &[(2816, 4096), (0, 100)]), vec![(100, 2816)]);
        assert_eq!(subtract_ranges(&[(0, 10)], &[(10, 20)]), vec![(0, 10)]);
    }
}
