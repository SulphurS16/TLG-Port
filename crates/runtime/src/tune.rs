//! Run-time tunables for A/B measurements within one session: initial values
//! come from `BB_*` environment variables, `tune <name> <value>` on the
//! control channel changes them while the title runs.

use std::sync::atomic::{AtomicI64, Ordering};

/// Memory waits met by the CP's own queued end-of-pipe writes become barriers
/// in the current batch (`SOTC_INBATCH_WAITS`, 1).
pub const INBATCH_WAITS: usize = 0;
/// Commands after which a batch is submitted (`SOTC_BATCH_MAX`, 128). Old
/// Yharnam, in-session A/B: 32/16 27.0 fps, 64/32 28.3; once CP copies of
/// GPU-written data no longer drained the GPU, 64/32 29.35, 128/64 29.9 (the
/// title's cap), 128/32 29.65. Deeper batches let the GPU thread run ahead
/// through GPU-heavy passes (6 batch slots in flight).
pub const BATCH_MAX: usize = 1;
/// Commands after which a batch is submitted while the GPU has no submitted
/// work left (`SOTC_BATCH_IDLE`, 64; 4 submitted ~2,400 small batches/s, each
/// vkQueueSubmit costing tens of microseconds).
pub const BATCH_IDLE: usize = 2;

/// Draws fetch read-only vertex/index data from VRAM copies (`SOTC_VCACHE`, 1).
pub const VCACHE: usize = 3;
/// Textures read from guest memory are detiled on the GPU where possible
/// (`SOTC_GPU_TEXTURES`, 1).
pub const GPU_TEXTURES: usize = 4;

/// Engine fill/copy kernels done in guest memory in place on the GPU
/// instead of running them (bit mask, `SOTC_ELIDE_EXT`, 15): 1 = fills of a
/// render target that are not clears (unorm8), 2 = fills of other memory,
/// 4 = copies that are not whole-surface image copies; 8 = array layer 0
/// fills as layer clears.
pub const ELIDE_EXT: usize = 5;

/// Images unused for a while are evicted from VRAM (`SOTC_EVICT`, 1; 0 keeps
/// everything, as before eviction existed).
pub const EVICT: usize = 6;

/// Vertex shaders' extra position exports become clip/cull distances per
/// PA_CL_VS_OUT_CNTL (`SOTC_CLIP_DIST`, 1; 0 drops them as before).
pub const CLIP_DIST: usize = 7;

/// Occlusion query results (PIXEL_PIPE_STAT_DUMP) from Vulkan occlusion
/// queries (`SOTC_OCCL`, 1; 0 leaves the result memory unwritten as before).
pub const OCCL: usize = 8;

/// FP32 denormals flushed as the shaders' FLOAT_MODE asks (`SOTC_FTZ`, 1;
/// 0 leaves the driver's default).
pub const FTZ: usize = 9;

/// Tessellated draws emulated (`SOTC_TESS`, 1; 0 skips them as before).
pub const TESS: usize = 10;

/// Guest-memory textures overlapping this batch's GPU writes are validated
/// before descriptors are allocated instead of flushing (`SOTC_EARLY_TEX`, 1).
pub const EARLY_TEX: usize = 11;

/// CP memory-to-memory DMA of data this batch's shaders wrote is copied on
/// the GPU in queue order instead of draining the GPU (`SOTC_DMA_GPU`, 1).
pub const DMA_GPU: usize = 12;

/// Per-draw SGPR data (user data, scalar-load results) is written to a
/// device-local ring through the BAR window instead of host memory the GPU
/// reads over PCIe (`SOTC_BAR_SGPR`, 0: in Old Yharnam GPU busy -15 ms/s but
/// 29.1 vs 29.3 fps, in-session A/B).
pub const BAR_SGPR: usize = 13;

/// When more title submissions are already queued, a submission's
/// end-of-pipe actions stay pending in the command processor and its batch
/// continues into the next submission instead of being submitted at its end
/// (`SOTC_COALESCE`, 0; flips always submit, memory waits commit as before).
/// About 300 of ~690 batches per second came from these commits in combat.
pub const COALESCE: usize = 14;

/// Write-watch queries for texture and VRAM-mirror validation cover whole
/// 64 KiB blocks, each queried at most once per CPU epoch; later checks in
/// a block read the page stamps (`SOTC_WW_BLOCKS`, 1; 0 queries each range).
pub const WW_BLOCKS: usize = 15;

/// DISPATCH_INDIRECT reads its GPU-written arguments on the GPU (validated
/// by a kernel) instead of the CP waiting for the work that wrote them
/// (`SOTC_GPU_INDIRECT`, 1).
pub const GPU_INDIRECT: usize = 16;

/// Fast-clear metadata (HTILE/CMASK) written by GPU work not yet complete is
/// sampled on the GPU and gates the clears through conditional rendering,
/// instead of draining the GPU to read it (`SOTC_META_GPU`, 1).
pub const META_GPU: usize = 17;

/// Diagnostic, unsafe: 0 skips the full barrier at render-pass changes
/// (dependencies through sampled render targets break) to measure what those
/// barriers cost on the GPU (`SOTC_PASS_BARRIER`, 1).
pub const PASS_BARRIER: usize = 18;

/// Diagnostic: VRAM mirror pages count registered GPU writes (which the
/// mirror itself holds) as changes and copy them in again from guest memory
/// once written back (`SOTC_MIRROR_GPU_STAMPS`, 0; 1 = the behaviour before
/// the tracker separated CPU writes).
pub const MIRROR_GPU_STAMPS: usize = 19;

/// Shaders whose scalar loads use addresses computed at run time (vector
/// pointer arithmetic + v_readfirstlane) run, reading through a window at
/// the pointer (`SOTC_DYN_LOADS`, 1; 0 skips them as before).
pub const DYN_LOADS: usize = 20;

const TABLE: [(&str, &str, i64); 21] = [
    ("inbatch", "SOTC_INBATCH_WAITS", 1),
    ("batch_max", "SOTC_BATCH_MAX", 128),
    ("batch_idle", "SOTC_BATCH_IDLE", 64),
    ("vcache", "SOTC_VCACHE", 1),
    ("gpu_textures", "SOTC_GPU_TEXTURES", 1),
    ("elide_ext", "SOTC_ELIDE_EXT", 15),
    ("evict", "SOTC_EVICT", 1),
    ("clipdist", "SOTC_CLIP_DIST", 1),
    ("occl", "SOTC_OCCL", 1),
    ("ftz", "SOTC_FTZ", 1),
    ("tess", "SOTC_TESS", 1),
    ("early_tex", "SOTC_EARLY_TEX", 1),
    ("dma_gpu", "SOTC_DMA_GPU", 1),
    ("bar_sgpr", "SOTC_BAR_SGPR", 0),
    ("coalesce", "SOTC_COALESCE", 0),
    ("ww_blocks", "SOTC_WW_BLOCKS", 1),
    ("gpu_indirect", "SOTC_GPU_INDIRECT", 1),
    ("meta_gpu", "SOTC_META_GPU", 1),
    ("pass_barrier", "SOTC_PASS_BARRIER", 1),
    ("mirror_gpu_stamps", "SOTC_MIRROR_GPU_STAMPS", 0),
    ("dyn_loads", "SOTC_DYN_LOADS", 1),
];
const UNSET: i64 = i64::MIN;
static VALUES: [AtomicI64; TABLE.len()] = [const { AtomicI64::new(UNSET) }; TABLE.len()];

pub fn get(i: usize) -> i64 {
    let v = VALUES[i].load(Ordering::Relaxed);
    if v != UNSET {
        return v;
    }
    let (_, env, default) = TABLE[i];
    let v = std::env::var(env).ok().and_then(|s| s.parse().ok()).unwrap_or(default);
    VALUES[i].store(v, Ordering::Relaxed);
    v
}

pub fn set(name: &str, value: &str) -> Result<(), String> {
    let i = TABLE.iter().position(|t| t.0 == name).ok_or_else(|| format!("unknown tunable '{name}' (known: {})", TABLE.map(|t| t.0).join(", ")))?;
    let v: i64 = value.parse().map_err(|_| format!("tune {name}: '{value}' is not an integer"))?;
    VALUES[i].store(v, Ordering::Relaxed);
    Ok(())
}
