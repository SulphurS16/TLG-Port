//! Run-time tunables for A/B measurements within one session: initial values
//! come from `BB_*` environment variables, `tune <name> <value>` on the
//! control channel changes them while the title runs.

use std::sync::atomic::{AtomicI64, Ordering};

/// Memory waits met by the CP's own queued end-of-pipe writes become barriers
/// in the current batch (`TLG_INBATCH_WAITS`, 1).
pub const INBATCH_WAITS: usize = 0;
/// Commands after which a batch is submitted (`TLG_BATCH_MAX`, 128). Old
/// Yharnam, in-session A/B: 32/16 27.0 fps, 64/32 28.3; once CP copies of
/// GPU-written data no longer drained the GPU, 64/32 29.35, 128/64 29.9 (the
/// title's cap), 128/32 29.65. Deeper batches let the GPU thread run ahead
/// through GPU-heavy passes (6 batch slots in flight).
pub const BATCH_MAX: usize = 1;
/// Commands after which a batch is submitted while the GPU has no submitted
/// work left (`TLG_BATCH_IDLE`, 64; 4 submitted ~2,400 small batches/s, each
/// vkQueueSubmit costing tens of microseconds).
pub const BATCH_IDLE: usize = 2;

/// Draws fetch read-only vertex/index data from VRAM copies (`TLG_VCACHE`, 1).
pub const VCACHE: usize = 3;
/// Textures read from guest memory are detiled on the GPU where possible
/// (`TLG_GPU_TEXTURES`, 1).
pub const GPU_TEXTURES: usize = 4;

/// Engine fill/copy kernels done in guest memory in place on the GPU
/// instead of running them (bit mask, `TLG_ELIDE_EXT`, 11): 1 = fills of a
/// render target that are not clears (unorm8), 2 = fills of other memory,
/// 4 = copies that are not whole-surface image copies (off: in Trico's cave
/// they wait for in-flight write-backs, CP sync waits 59 vs 11 ms per frame);
/// 8 = array layer 0 fills as layer clears.
pub const ELIDE_EXT: usize = 5;

/// Images unused for a while are evicted from VRAM (`TLG_EVICT`, 1; 0 keeps
/// everything, as before eviction existed).
pub const EVICT: usize = 6;

/// Vertex shaders' extra position exports become clip/cull distances per
/// PA_CL_VS_OUT_CNTL (`TLG_CLIP_DIST`, 1; 0 drops them as before).
pub const CLIP_DIST: usize = 7;

/// Occlusion query results (PIXEL_PIPE_STAT_DUMP) from Vulkan occlusion
/// queries (`TLG_OCCL`, 1; 0 leaves the result memory unwritten as before).
pub const OCCL: usize = 8;

/// FP32 denormals flushed as the shaders' FLOAT_MODE asks (`TLG_FTZ`, 1;
/// 0 leaves the driver's default).
pub const FTZ: usize = 9;

/// Tessellated draws emulated (`TLG_TESS`, 1; 0 skips them as before).
pub const TESS: usize = 10;

/// Guest-memory textures overlapping this batch's GPU writes are validated
/// before descriptors are allocated instead of flushing (`TLG_EARLY_TEX`, 1).
pub const EARLY_TEX: usize = 11;

/// CP memory-to-memory DMA of data this batch's shaders wrote is copied on
/// the GPU in queue order instead of draining the GPU (`TLG_DMA_GPU`, 1).
pub const DMA_GPU: usize = 12;

/// Per-draw SGPR data (user data, scalar-load results) is written to a
/// device-local ring through the BAR window instead of host memory the GPU
/// reads over PCIe (`TLG_BAR_SGPR`, 0: in Old Yharnam GPU busy -15 ms/s but
/// 29.1 vs 29.3 fps, in-session A/B).
pub const BAR_SGPR: usize = 13;

/// When more title submissions are already queued, a submission's
/// end-of-pipe actions stay pending in the command processor and its batch
/// continues into the next submission instead of being submitted at its end
/// (`TLG_COALESCE`, 0; flips always submit, memory waits commit as before).
/// About 300 of ~690 batches per second came from these commits in combat.
pub const COALESCE: usize = 14;

/// Write-watch queries for texture and VRAM-mirror validation cover whole
/// 64 KiB blocks, each queried at most once per CPU epoch; later checks in
/// a block read the page stamps (`TLG_WW_BLOCKS`, 1; 0 queries each range).
pub const WW_BLOCKS: usize = 15;

/// DISPATCH_INDIRECT reads its GPU-written arguments on the GPU (validated
/// by a kernel) instead of the CP waiting for the work that wrote them
/// (`TLG_GPU_INDIRECT`, 1).
pub const GPU_INDIRECT: usize = 16;

/// Fast-clear metadata (HTILE/CMASK) written by GPU work not yet complete is
/// sampled on the GPU and gates the clears through conditional rendering,
/// instead of draining the GPU to read it (`TLG_META_GPU`, 1).
pub const META_GPU: usize = 17;

/// Diagnostic, unsafe: 0 skips the full barrier at render-pass changes
/// (dependencies through sampled render targets break) to measure what those
/// barriers cost on the GPU (`TLG_PASS_BARRIER`, 1).
pub const PASS_BARRIER: usize = 18;

/// Diagnostic: VRAM mirror pages count registered GPU writes (which the
/// mirror itself holds) as changes and copy them in again from guest memory
/// once written back (`TLG_MIRROR_GPU_STAMPS`, 0; 1 = the behaviour before
/// the tracker separated CPU writes).
pub const MIRROR_GPU_STAMPS: usize = 19;

/// Shaders whose scalar loads use addresses computed at run time (vector
/// pointer arithmetic + v_readfirstlane) run, reading through a window at
/// the pointer (`TLG_DYN_LOADS`, 1; 0 skips them as before).
pub const DYN_LOADS: usize = 20;

/// Read-only buffer bindings of at least this many KiB come from the VRAM
/// mirror (copied in when the CPU changes them) rather than a copy into the
/// host-memory upload ring per binding (`TLG_VCACHE_READ_MIN_KIB`, 64).
/// Below 64 only in mirror windows that exist.
pub const VCACHE_READ_MIN_KIB: usize = 21;

/// Depth targets read through a smaller T# only by integer texel coordinates
/// are bound directly instead of copied cropped (`TLG_DEPTH_LOAD_VIEW`, 1).
pub const DEPTH_LOAD_VIEW: usize = 22;

/// The engine's copy/fill kernels run as buffer or image operations
/// (`TLG_ELIDE_UTILITY`, 1; 0 runs them as shaders).
pub const ELIDE_UTILITY: usize = 23;

/// Written buffer bindings of at least this many KiB work in the VRAM mirror
/// (else a device-ring copy set up per binding and written back)
/// (`TLG_VCACHE_WRITE_MIN_KIB`, 64). Below 64 only in mirror windows that exist.
pub const VCACHE_WRITE_MIN_KIB: usize = 24;

/// Read-only bindings smaller than `vcache_read_min_kib` whose range has GPU
/// writes not yet in guest memory bind the VRAM mirror (which holds or gets
/// them) instead of an upload with forwards (`TLG_VCACHE_GPU_READS`, 0).
pub const VCACHE_GPU_READS: usize = 25;

/// Lazy write-back of mirror-resident GPU writes (gpu/lazywb.rs)
/// (`TLG_LAZY_WB`, 1; 0 writes everything back after each batch).
pub const LAZY_WB: usize = 26;

/// CP waits (WAIT_REG_MEM, MEM_SEMAPHORE) sleep until submissions,
/// completions or CP writes change memory (`TLG_CP_WAIT_NOTIFY`, 1; 0 polls
/// with 100 us sleeps).
pub const CP_WAIT_NOTIFY: usize = 27;

/// Depth targets sampled through a smaller T# are copied to their color
/// shadow by one compute pass reading the depth view (`TLG_DEPTH_COPY_CS`, 1;
/// 0: depth -> buffer -> color transfers).
pub const DEPTH_COPY_CS: usize = 28;

/// A read-only binding of a guest range this batch already copied into the
/// upload ring in the current CPU epoch, untouched by registered writes
/// since, reuses that copy (`TLG_UPLOAD_REUSE`, 1; 0: a copy per binding).
pub const UPLOAD_REUSE: usize = 29;

/// Read-only bindings of up to `BAR_UPLOAD_MAX` bytes that need no GPU
/// forwards are copied into the device-local BAR ring (when its segment has
/// room) instead of host memory the GPU reads over PCIe (`TLG_BAR_UPLOADS`,
/// 0: host ring. Cave A/B with upload reuse, run 20261013-030209: GPU busy
/// 129 vs 110 ms per frame with it, same fps; no BAR fallbacks).
pub const BAR_UPLOADS: usize = 30;

/// Index data whose guest memory is current is fetched from the VRAM mirror
/// (copied in once, then per CPU-written page) instead of a host-ring copy
/// per draw (`TLG_INDEX_MIRROR`, 1). Before the write protection of cold
/// blocks (`CPU_WATCH`) the mirror's write-watch checks ate the gain (run
/// 20261013-032442: 0.88 vs 1.00 fps); with it (run 20261013-120421, CPU
/// 31%) 4.70 vs 4.46 fps, GPU-thread CPU per command -5.5%, upload time
/// per command -49%.
pub const INDEX_MIRROR: usize = 31;

/// CP threads check a separately locked summary of the current batch's
/// pending GPU writes and in-place reads before direct memory accesses, and
/// take the renderer lock only on overlap; their writes' mirror/upload-cache
/// invalidations are applied by the renderer before its next command
/// (`TLG_CP_SYNC_VIEW`, 1; 0: the renderer lock for every access).
pub const CP_SYNC_VIEW: usize = 32;

/// CPU-sourced textures of at most `SMALL_IMAGE_BYTES` unused for 600
/// frames are evicted even with image eviction off (`TLG_EVICT_SMALL`, 1):
/// the cave's tiny R32F textures sit at a new address in the title's
/// per-frame memory every frame (~12 new images per second, never used again).
pub const EVICT_SMALL: usize = 33;

/// Dispatches are ordered against earlier dispatches by the buffer ranges
/// they access (`TLG_DISPATCH_HAZARDS`, 1): a full barrier only where one
/// writes what another since the last barrier read or wrote, or where other
/// GPU work (copies, fills, render passes, storage images, GDS, indirect
/// arguments) is unordered; 0: a full barrier before every dispatch after
/// any write.
pub const DISPATCH_HAZARDS: usize = 34;

/// Page bitmaps of mirror write-backs come from a device-ring area zeroed
/// once per batch (`TLG_BITS_ARENA`, 1) instead of a fill and a full
/// barrier per registration (0).
pub const BITS_ARENA: usize = 35;

/// Cold 64 KiB blocks (unwritten in consecutive write-watch queries) are made
/// read-only instead of being queried every CPU epoch; a CPU write faults and
/// makes them writable and queried again (`TLG_CPU_WATCH`, 1; cpuwatch.rs).
pub const CPU_WATCH: usize = 36;

/// Evicted small images (<= 64 KiB) are kept for reuse by later images with
/// the same creation parameters (`TLG_IMAGE_POOL`, 1): ~3 small textures per
/// frame at fresh addresses each cost a vkAllocateMemory and later a
/// vkFreeMemory (2.5% of the graphics CP in the cave).
pub const IMAGE_POOL: usize = 37;

/// Memory waits keep their poll timer armed across wake-ups by label changes
/// (`TLG_WAIT_TIMER_KEEP`, 1) instead of re-arming it per wake-up (0;
/// SetWaitableTimer was 3.2% of the graphics CP in the cave).
pub const WAIT_TIMER_KEEP: usize = 38;

/// The completion thread lets write-watch queries in between a batch's
/// write-back entries (`TLG_WB_LOCK_BUMP`, 1; each mirror-resident full
/// write-back forgets its pages right after its copy) instead of holding the
/// write-watch lock for all of them (0; the graphics CP's queries waited 1.8%).
pub const WB_LOCK_BUMP: usize = 39;

/// The graphics CP hands the renderer lock to a waiting compute CP after each
/// command (`TLG_LOCK_FAIR`, 0) instead of usually taking it again at once.
pub const LOCK_FAIR: usize = 40;

/// A memory wait met by another queue's queued (not yet committed)
/// end-of-pipe or label write is met in GPU order by a full barrier
/// (`TLG_CROSS_WAITS`, 0): all queues record into one batch, so work recorded
/// after the barrier runs after the work before that write. 0: such a wait
/// lasts until the writing queue commits (submits) its writes. Measured
/// worse in the cave (run 20261013-133752: -7% fps at full clock, -12% at
/// 31%): the graphics CP's label waits shrink, but the CPs then interleave
/// on the renderer lock (its waits 142 -> ~190 ms/s) and each command costs
/// more.
pub const CROSS_WAITS: usize = 41;

/// Indirect draws whose arguments earlier GPU work is still writing read
/// them on the GPU (`TLG_INDIRECT_DRAW_GPU`, 1; `kernels::indirect_draw_cs`)
/// instead of the CP waiting for that work to complete and land (~3 ms per
/// frame in the cave, with the GPU idle meanwhile).
pub const INDIRECT_DRAW_GPU: usize = 42;

/// VRAM-mirror bindings validated before are served again without walking
/// their blocks and pages while their window has had no pages invalidated
/// and their CPU-write checks still count (`TLG_VBIND_CACHE`, 1).
pub const VBIND_CACHE: usize = 43;

/// GPU work that only fills device-ring ranges nothing used yet from data
/// fixed at record time - written buffers' setup copies (guest data from the
/// upload ring), page-bitmap clears of mirror write-backs - goes into a
/// prologue command buffer submitted ahead of the batch, under one barrier
/// (`TLG_PROLOGUE`, 1), instead of each with a full barrier in the batch (0;
/// ~350 setup-copy and ~135 bitmap barriers per frame in the cave).
pub const PROLOGUE: usize = 44;

/// Shader preparation of a draw/dispatch (decoding its descriptors from
/// registers and guest memory) runs before the CP takes the renderer lock
/// (`TLG_PREPARE_UNLOCKED`, 1); under the lock its guest reads are checked
/// against pending GPU writes as before (prepared again when one is
/// pending). 0: prepared under the lock.
pub const PREPARE_UNLOCKED: usize = 45;

/// A read-only binding whose range one pending GPU write (this batch's or a
/// batch's in flight) covers entirely binds that write's device copy
/// (`TLG_DEVCOPY_READS`, 0) instead of an upload-ring copy of guest memory
/// with the write forwarded on top (a CPU copy, a GPU copy, two barriers).
/// Cave A/B (run 20261013-145735) on top of `chain_written`: GPU -1.6%,
/// barriers -5%, graphics-CP CPU +1-6%: off on this CPU-bound machine.
pub const DEVCOPY_READS: usize = 46;

/// A dispatch writing exactly the range an earlier command of the batch
/// wrote into a device-ring copy works on that copy in place
/// (`TLG_CHAIN_WRITTEN`, 1; its write-back lands the newest data, the
/// dispatch hazards order the two) instead of a new copy set up from guest
/// memory with the earlier write forwarded on top and a write-back of its
/// own (TLG's cs 0x1130f7bc00 updates one 28 KiB buffer ~325 times a frame).
/// Cave A/B (run 20261013-145735): full barriers -36%, GPU -11% per frame,
/// fps unchanged where the graphics CP is the limit.
pub const CHAIN_WRITTEN: usize = 47;

/// Measurement only (`TLG_SKIP_CMDS`, 0): 1 records no draws and no
/// dispatches, 2 no draws, 3 no dispatches - what the title and the rest of
/// the CP cost without that work (frames then show nothing valid).
pub const SKIP_CMDS: usize = 48;

const TABLE: [(&str, &str, i64); 49] = [
    ("inbatch", "TLG_INBATCH_WAITS", 1),
    ("batch_max", "TLG_BATCH_MAX", 128),
    ("batch_idle", "TLG_BATCH_IDLE", 64),
    ("vcache", "TLG_VCACHE", 1),
    ("gpu_textures", "TLG_GPU_TEXTURES", 1),
    ("elide_ext", "TLG_ELIDE_EXT", 11),
    ("evict", "TLG_EVICT", 1),
    ("clipdist", "TLG_CLIP_DIST", 1),
    ("occl", "TLG_OCCL", 1),
    ("ftz", "TLG_FTZ", 1),
    ("tess", "TLG_TESS", 1),
    ("early_tex", "TLG_EARLY_TEX", 1),
    ("dma_gpu", "TLG_DMA_GPU", 1),
    ("bar_sgpr", "TLG_BAR_SGPR", 0),
    ("coalesce", "TLG_COALESCE", 0),
    ("ww_blocks", "TLG_WW_BLOCKS", 1),
    ("gpu_indirect", "TLG_GPU_INDIRECT", 1),
    ("meta_gpu", "TLG_META_GPU", 1),
    ("pass_barrier", "TLG_PASS_BARRIER", 1),
    ("mirror_gpu_stamps", "TLG_MIRROR_GPU_STAMPS", 0),
    ("dyn_loads", "TLG_DYN_LOADS", 1),
    ("vcache_read_min_kib", "TLG_VCACHE_READ_MIN_KIB", 64),
    ("depth_load_view", "TLG_DEPTH_LOAD_VIEW", 1),
    ("elide_utility", "TLG_ELIDE_UTILITY", 1),
    ("vcache_write_min_kib", "TLG_VCACHE_WRITE_MIN_KIB", 64),
    ("vcache_gpu_reads", "TLG_VCACHE_GPU_READS", 0),
    ("lazy_wb", "TLG_LAZY_WB", 1),
    ("cp_wait_notify", "TLG_CP_WAIT_NOTIFY", 1),
    ("depth_copy_cs", "TLG_DEPTH_COPY_CS", 1),
    ("upload_reuse", "TLG_UPLOAD_REUSE", 1),
    ("bar_uploads", "TLG_BAR_UPLOADS", 0),
    ("index_mirror", "TLG_INDEX_MIRROR", 1),
    ("cp_sync_view", "TLG_CP_SYNC_VIEW", 1),
    ("evict_small", "TLG_EVICT_SMALL", 1),
    ("dispatch_hazards", "TLG_DISPATCH_HAZARDS", 1),
    ("bits_arena", "TLG_BITS_ARENA", 0),
    ("cpu_watch", "TLG_CPU_WATCH", 1),
    ("image_pool", "TLG_IMAGE_POOL", 1),
    ("wait_timer_keep", "TLG_WAIT_TIMER_KEEP", 1),
    ("wb_lock_bump", "TLG_WB_LOCK_BUMP", 1),
    ("lock_fair", "TLG_LOCK_FAIR", 0),
    ("cross_waits", "TLG_CROSS_WAITS", 0),
    ("indirect_draw_gpu", "TLG_INDIRECT_DRAW_GPU", 1),
    ("vbind_cache", "TLG_VBIND_CACHE", 1),
    ("prologue", "TLG_PROLOGUE", 1),
    ("prepare_unlocked", "TLG_PREPARE_UNLOCKED", 1),
    ("devcopy_reads", "TLG_DEVCOPY_READS", 0),
    ("chain_written", "TLG_CHAIN_WRITTEN", 1),
    ("skip_cmds", "TLG_SKIP_CMDS", 0),
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
