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
/// frames are evicted even with image eviction off (`TLG_EVICT_SMALL`, 2;
/// 1: only those): the cave's tiny R32F textures sit at a new address in the
/// title's per-frame memory every frame (~30 new images per second, nearly
/// never used again). 2: also, of those with the same creation parameters
/// idle for 30 frames, all but the 16 most recently used at once, into the
/// image pool - with 1 the pool stays empty until the first ones age out and
/// each such texture costs a new image (~1 ms at 31% clock, 3 per frame for
/// the first 600 frames after a load; run 20261016-070734: pooled ~85 frames
/// after a load, ~920 live images instead of ~2,600, re-creations at the same
/// rate).
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
/// write-watch lock for all of them (0; the graphics CP's queries waited 1.8%);
/// 2: also between the page runs of an entry and between 256 KiB pieces of
/// whole-range mirror write-backs (each piece forgotten right after its copy).
pub const WB_LOCK_BUMP: usize = 39;

/// The graphics CP hands the renderer lock to a waiting compute CP after each
/// command (`TLG_LOCK_FAIR` 1) instead of usually taking it again at once
/// (0); 2: the compute CPs hand it to a waiting thread (the graphics CP)
/// instead; 3: as 2, and compute CPs let a waiting graphics CP go first.
pub const LOCK_FAIR: usize = 40;

/// A memory wait met by another queue's queued (not yet committed)
/// end-of-pipe or label write is met in GPU order by a full barrier
/// (`TLG_CROSS_WAITS`, 0): all queues record into one batch, so work recorded
/// after the barrier runs after the work before that write. 0: such a wait
/// lasts until the writing queue commits (submits) its writes. Measured
/// worse in the cave (run 20261013-133752: -7% fps at full clock, -12% at
/// 31%): the graphics CP's label waits shrink, but the CPs then interleave
/// on the renderer lock (its waits 142 -> ~190 ms/s) and each command costs
/// more. 2: only for waits of the graphics queue and the device compute queue
/// 13 (their one-dispatch hand-offs at frame start), not of the per-frame
/// compute queues 17-19.
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
/// dispatches, 2 no draws, 3 no dispatches, 6 no dispatches of the compute
/// queues 17-19 - what the title and the rest of the CP cost without that
/// work (frames then show nothing valid).
pub const SKIP_CMDS: usize = 48;

/// Batches are recorded and submitted by the recorder thread
/// (`TLG_RECORDER`, 1; `gpu::rec`) instead of by the CP threads under the
/// renderer lock (0). Takes effect at the next batch.
pub const RECORDER: usize = 49;

/// Recorder operations the CPs collect before handing them over
/// (`TLG_REC_BATCH`, 64; submits hand over at once).
pub const REC_BATCH: usize = 50;

/// After shader preparation and before the renderer lock, the CP prefetches
/// the guest bytes of the stages' small read-only buffers (`TLG_PREFETCH`, 0):
/// the title's threads wrote them on other cores just before, and the upload
/// copies (~1 us each, ~75,000 per second in the cave) waited for those
/// cache lines under the lock. Measured: copies 1.18 -> 0.78 us, CP time per
/// frame unchanged (issuing the prefetches costs as much).
pub const PREFETCH: usize = 51;

/// Guest-memory copies into the upload ring (read-only and written buffer
/// bindings, mirror copy-in staging) run on the `GPU-copy` thread
/// (`TLG_ASYNC_COPIES`, 1; `gpu::copier`); a batch's submit waits for its
/// copies (0: the CP copies under the renderer lock).
pub const ASYNC_COPIES: usize = 52;

/// The mirror binding cache (`vbind_cache`) keys validity on the generation of
/// the 4 MiB parts of a window a binding touches (`TLG_VBIND_PARTS`, 1)
/// instead of the whole window's (0): a GPU write or CPU change elsewhere in a
/// 256 MiB window no longer drops every binding served from it.
pub const VBIND_PARTS: usize = 53;

/// The `GPU-prep` thread prepares the shader stages of graphics-queue draws
/// and dispatches ahead of the graphics CP, which uses a result once it
/// validates (`TLG_PREP_AHEAD`, 1; `gpu::prepahead`).
pub const PREP_AHEAD: usize = 54;

/// The graphics CP runs each submission as decoded by the `GPU-prep` thread
/// (register writes and packets in execution order) instead of parsing its
/// PM4 itself (`TLG_DECODED_CP`, 1; `gpu::prepahead::Decoded`). (TLG_POOL_WATCH
/// then counts no graphics indirect buffers.)
pub const DECODED_CP: usize = 55;

/// The `GPU-prep` thread runs at the graphics CP's raised priority
/// (`TLG_PREP_PRIORITY`, 0; 0: normal).
pub const PREP_PRIORITY: usize = 56;

/// Write-backs whose host writes the write watch then forgets run without
/// holding `mem::WRITE_WATCH_LOCK` (`TLG_WW_FILTER`, 1): write-watch queries
/// leave out their pages meanwhile (`mem::forget_begin`, in either mode).
/// The mirror's CPU-write checks no longer wait for write-back copies: 10.0
/// -> 5.7 us (cave, ~9,100 per second).
pub const WW_FILTER: usize = 57;

/// VRAM mirror copy-ins of pages no command of the batch used yet go into
/// the batch's prologue instead of the batch, between full barriers
/// (`TLG_HOIST_COPYINS`, 1): ~75% of copy-ins in the cave, full barriers
/// -15%, GPU time per frame -2.7%.
pub const HOIST_COPYINS: usize = 58;

/// Small guest-to-ring copies are made by the CP itself while the copy thread
/// sleeps instead of waking it (`TLG_COPY_INLINE`, 0; `gpu::copier`).
pub const COPY_INLINE: usize = 59;

/// Dispatch hazards follow the title's synchronization: a dispatch is
/// ordered behind the conflicting buffer accesses of earlier GPU work only
/// when the title synchronized after them (label waits, cache acquires,
/// events; `cp::TITLE_SYNC`) - PS4 command processors overlap dispatches
/// otherwise (`TLG_TITLE_SYNC`, 1: any queue's synchronization point; 2: the
/// dispatching queue's own; 0: every conflict). Barriers for our own copies
/// stay. Cave: full barriers -26%, GPU time per frame -21%, +6-9% fps (the
/// compute queues' single-workgroup jobs on shared multi-MiB buffers, which
/// the title issues back to back, no longer run one at a time).
pub const TITLE_SYNC: usize = 60;

/// Descriptor sets of translated shaders are pushed into the command buffer
/// (VK_KHR_push_descriptor) by the recorder thread instead of allocated from
/// the batch's pool and written on the CP threads, for layouts within the
/// device's push limit (`TLG_PUSH_DESC`, 1). Pipelines are kept per mode: a
/// change applies to pipelines looked up after it (the first use of a mode
/// creates its pipelines; the pipeline cache keeps both). The descriptors go
/// in a buffer published with the recorder's operations. Cave, 31% clock,
/// one boot: -4.6% per frame against template-written sets (rounds after the
/// switch; -1.4% before the buffer, with a heap allocation per push).
pub const PUSH_DESC: usize = 61;

/// Measurement only: busy-waits this many ns per draw / per dispatch of the
/// graphics queue, and per dispatch of the compute queues, under the
/// renderer lock (`TLG_DELAY_GFX_NS`, `TLG_DELAY_CS_NS`, 0): how much the
/// frame time follows each queue's per-command cost.
pub const DELAY_GFX_NS: usize = 62;
pub const DELAY_CS_NS: usize = 63;

/// Diagnostics: logs the next this many write-watch validation queries
/// (epochs, range, pages found; `ww:` lines), then 0 again (`TLG_WW_LOG`, 0).
pub const WW_LOG: usize = 64;

/// Write-watch queries in bulk: when a queue class's required epoch changes
/// (graphics: each submission; compute: each doorbell), the blocks the
/// previous round of its validations asked for and that are due are queried
/// at once, in runs merged across gaps of up to this many 64 KiB blocks (0:
/// each validation queries its own blocks; `TLG_WW_BULK`, 16). Every query
/// that resets written pages costs a TLB flush on all cores (cave, 31%
/// clock: ~37 us against ~9 us for one that finds none); the graphics CP
/// made ~240 per frame, the same ones each frame. Cave: calls 296 -> 104 per
/// frame, query time -45%, frame -2.5 to -3.4% (64 blocks: more pages, less gain).
pub const WW_BULK: usize = 65;

/// Microseconds the copy thread (`gpu::copier`) keeps polling for the next
/// copy before it sleeps (`TLG_COPY_SPIN_US`, 1000; was 50): each copy queued
/// while it sleeps wakes it (a kernel call on the CP; ~4% of the graphics CP
/// at 31% clock, where draws come more than 50 us apart). Cave, 31% clock:
/// wakes per frame 890 -> ~4, frame -4 to -6.5% (200 us: -1%). It polls while a CP
/// records and sleeps through the label waits.
pub const COPY_SPIN_US: usize = 66;

/// A draw reuses the previous draw's render targets when the registers that
/// select them are unchanged, no other draw or dispatch began in between,
/// the previous draw stored nothing but its targets, and the images still
/// exist with no newer guest data (`TLG_TARGET_CACHE`, 1): consecutive draws
/// of a pass skip the target lookups (~1.2 us per draw at full clock).
pub const TARGET_CACHE: usize = 67;

/// DMA_DATA from memory into a run of registers of one register file is
/// copied at once instead of word by word (`TLG_DMA_REG_BULK`, 1). Cave:
/// within noise (the reads miss the cache either way).
pub const DMA_REG_BULK: usize = 68;

/// Descriptor sets are written with an update template per set layout (one
/// array of descriptors in the layout's binding order) instead of a write
/// structure per binding (`TLG_DESC_TEMPLATES`, 1).
pub const DESC_TEMPLATES: usize = 69;

/// A sampled texture whose descriptor took the guest-memory path before
/// skips the search for render targets and shadows at its address while no
/// image there became GPU-written since (`TLG_TEX_CACHE`, 1). Cave, 31%
/// clock: ~85% of lookups, frame -2.5% (3 rounds).
pub const TEX_CACHE: usize = 70;

/// With TLG_PERF: the per-command statistics (lock, phase and binding
/// timers) are collected (`TLG_PERF_HOT`, 1; 0 leaves only the per-second
/// summaries' other sources). Without TLG_PERF they are never collected.
pub const PERF_HOT: usize = 71;

/// DMA_DATA of several words into one register (no destination increment:
/// the title's L2 prefetches of shader code) makes only the last write
/// (`TLG_DMA_REG_LAST`, 1; 0 writes every word).
pub const DMA_REG_LAST: usize = 72;

/// The recorder thread polls this many microseconds for more operations
/// before it sleeps (`TLG_REC_SPIN_US`, 1000): a CP publishes every few
/// commands, and waking the sleeping recorder costs the CP a system call.
pub const REC_SPIN_US: usize = 73;

/// Mirror bindings stay valid in the binding cache while no part of this
/// many KiB they touch was invalidated (`TLG_VBIND_PART_KIB`, 64; parts of
/// 64 KiB and up, powers of two). Cave, 31% clock: frame -1.6% against 4096
/// (part-invalidated misses -58% on the graphics queue).
pub const VBIND_PART_KIB: usize = 74;

/// Compute queues (bit n: queue n) that let a graphics CP waiting for the
/// renderer lock go first (`TLG_YIELD_QUEUES`, 0): for queues whose work is
/// not what the graphics CP waits for (`lock_fair 3` makes every compute
/// queue yield).
pub const YIELD_QUEUES: usize = 75;

/// Semaphore signals hand the count to waiting threads in queue order and
/// wake only those (`TLG_SEMA_WAKE`, 1); 0 wakes every waiter to race for it
/// (the title's seven job workers wait on one semaphore).
pub const SEMA_WAKE: usize = 76;

/// Pause instructions a contended guest mutex spins before parking
/// (`TLG_MUTEX_SPINS`, 4000; ~140 cycles each on this CPU).
pub const MUTEX_SPINS: usize = 77;

/// The title's render-kick job waits for ps4render in a loop without a pause
/// (eboot 0xd0aff0, patched): poll interval in microseconds after a short
/// spin (`TLG_KICK_WAIT`, 100); 0 busy-waits as the title does.
pub const KICK_WAIT: usize = 78;

/// Pause instructions between clock reads in the helper threads' spin
/// loops (`TLG_SPIN_PAUSES`, 1): a performance-counter read and its 128-bit
/// division cost more than a pause, also to the core's other hyperthread.
pub const SPIN_PAUSES: usize = 79;

/// The HR-timer thread's last stretch before a deadline (`TLG_HRT_SPIN`, 0):
/// 1 sleeps to ~200 us before it and spins the rest; 0 only sleeps (a
/// high-resolution sleep, later by up to a few hundred microseconds). The
/// title's job workers arm ~4,000-6,000 timers of 100 us per second while
/// idle; without the spins -0.27 core, fps equal (LOG: the CPU budget).
pub const HRT_SPIN: usize = 80;

/// The graphics CP thread runs on a physical core alone (`TLG_GFX_CORE`, -2:
/// chosen, see `thread::dedicate_core`; N: core N; -1: Windows schedules
/// it): on one of the core's logical processors while the process's other
/// threads avoid both (CPU sets). It lost ~10% to busy hyperthread
/// siblings: graphics CP -10 to -12% per frame (LOG: the CPU budget).
pub const GFX_CORE: usize = 81;

/// With `gfx_core`: 1 only prefers that core for the graphics CP thread
/// (ideal processor; it may run elsewhere) while the other threads still
/// avoid it (`TLG_GFX_CORE_SOFT`, 0).
pub const GFX_CORE_SOFT: usize = 82;

/// Experiment (`TLG_CS_CORE`, -1): with a dedicated graphics core, the
/// compute CPs (`GPU-acb*`) and the completion thread share another
/// physical core of their own (N: core N; -2: the core after the graphics
/// CP's), the rest avoid both.
pub const CS_CORE: usize = 83;

/// A smaller T# over a padded color render target that the shader only reads
/// with integer texel coordinates samples the target itself instead of a
/// shadow copy (`TLG_SHADOW_LOAD_VIEW`, 0): the same texels.
pub const SHADOW_LOAD_VIEW: usize = 84;

/// Measurement only (`TLG_ZERO_GROUPS`, 0): dispatches of the CP queues in
/// this bit mask (bit 0 graphics, 13, 17-19 compute) are recorded with zero
/// workgroups - their GPU work bounded while the CP work stays the same
/// (results then missing: frames and the title's GPU-fed state are wrong);
/// bit 30: only single-workgroup dispatches.
pub const ZERO_GROUPS: usize = 85;

/// Consecutive unwritten write-watch queries after which a 64 KiB block is
/// write-protected instead of queried (`TLG_ARM_AFTER`, 8; `tune cpu_watch`):
/// lower moves the cost of noticing CPU writes from the CPs' queries to
/// write faults on the title's threads.
pub const ARM_AFTER: usize = 86;

/// A command's ring-space estimate for its buffer bindings comes from its
/// preparation (computed on the prepare thread) instead of a walk over the
/// bindings per command (`TLG_RING_BYTES_AHEAD`, 1; not with zero-copy).
pub const RING_BYTES_AHEAD: usize = 87;

/// A mirror binding cached in an earlier CPU epoch whose window parts are
/// unchanged has its blocks' CPU-write checks redone in place and is served
/// again when they find nothing (`TLG_VBIND_RECHECK`, 1), instead of going
/// through the whole fill path.
pub const VBIND_RECHECK: usize = 88;

/// Dispatch hazard tracking keeps each buffer access since the last barrier
/// once (`TLG_HZ_DEDUPE`, 1): the compute queues bind the same written
/// ranges dispatch after dispatch, and every dispatch was tested against
/// all their copies.
pub const HZ_DEDUPE: usize = 89;

/// Flips no closer than the title's flip rate allows (`TLG_FRAME_CAP`; -1 =
/// settings.txt `frame_cap`, 1 on, 0 off: development measurements above
/// the game's own rate; 60: the 60 fps limit, the game's time step snapped
/// to 1/60 s - `videoout::frame_cap`).
pub const FRAME_CAP: usize = 90;

/// A draw's attachments are marked GPU-written (stale index, generations,
/// last write) once per rendering scope instead of at every draw
/// (`TLG_ATTACH_ONCE`, 1): within one scope nothing else changes them
/// (commands that would end the scope; the state-only paths reset it).
pub const ATTACH_ONCE: usize = 91;

/// Measurement only (`TLG_EARLY_EOP`, 0): a batch's end-of-pipe actions
/// with a graphics interrupt (the frame-done event the title waits for
/// before its next submission) are performed once it is submitted and every
/// compute queue is idle, instead of when it completes on the GPU. GPU
/// results still in flight are not protected from the title meanwhile.
pub const EARLY_EOP: usize = 92;

/// The per-frame compute queues (17-19) record into their own context
/// (`TLG_CS_CTX`, 0; set at start for its full-size rings): its own command
/// buffers and rings, submitted with the graphics queue's batch after its
/// work (merged submissions); one context records at a time - starting one
/// submits the other's open batch; queue 13 joins it while it records.
/// Neutral under the one renderer lock (run 20261016-000756: 10.0 vs 10.2
/// fps); the base for recording the compute queues without that lock.
pub const CS_CTX: usize = 93;

/// The compute queues record with a renderer instance of their own
/// (`TLG_CS_INST`, 0; set at start): its own batches and submissions, the
/// memory model and image state shared with the graphics queue's. For now
/// under the graphics renderer's lock (`gpu::render::RENDERER_CS`).
pub const CS_INST: usize = 94;

/// Labels released early (`TLG_EARLY_RELEASE`, 0): a queue about to wait for
/// a value another queue's queued (not yet committed) writes would produce
/// asks that queue to commit them at its next packet instead of at its own
/// next commit point (`gpu::cp::COMMIT_REQ`).
pub const EARLY_RELEASE: usize = 95;

/// With `cs_inst`: the compute queues record with their own renderer
/// (`TLG_CS_ROUTE`, 1) or with the graphics queue's (0; for A/B windows).
pub const CS_ROUTE: usize = 96;

/// Read-only ring copies reused by later batches of the same CPU epoch and
/// CP queue, up to this many batches back (`TLG_UPLOAD_XBATCH`, at most 4;
/// 0: within their batch only, as before). Takes effect only when set at
/// start: the upload ring then has 4 segments more than there are batch
/// slots, so a copy's segment is written again only after every batch that
/// may reuse it has completed (smaller segments: more batches end early).
pub const UPLOAD_XBATCH: usize = 97;

/// Write-back overlap checks answered from indexes when nothing can overlap
/// (`TLG_SCAN_INDEX`, 1): the in-flight batches' write-backs counted per
/// 2 MiB granule (no lock), the open batch's in a bloom filter of 64 KiB
/// blocks; 0: every check scans the lists.
pub const SCAN_INDEX: usize = 98;

/// Sampled textures that took the guest-memory path in the current
/// validation epoch return the same view again while no image at their base
/// and none of the image set changed (`TLG_TEX_MEMO`, 1; with `tex_cache`);
/// the guest-read checks skip such textures. 2: for the CPU epoch instead,
/// while no GPU write registered since overlaps the texture (the validation
/// epoch moves after every command that writes buffers).
pub const TEX_MEMO: usize = 99;

/// The `descriptors by shader` report with `TLG_PERF=1` too (`TLG_SHADER_STATS`,
/// 0: only with `TLG_PERF=2`, whose per-copy hashing inflates the copy paths).
pub const SHADER_STATS: usize = 100;

/// Texture memo entries are checked against their base's generation only
/// (`TLG_MEMO_BASE`, 1; 0: also the whole image set's, which every image
/// creation and eviction anywhere bumps): a lookup only consults the images
/// at its base, and every change there bumps the base's. (Neutral with
/// `tex_memo 1`, whose entries last one validation epoch: run
/// 20261016-070734, memo hits 24,300 vs 24,400 per second.)
pub const MEMO_BASE: usize = 101;

/// How shaders mark the pages of their buffer stores (`TLG_PAGE_MARKS`, 3: a
/// word per page set by a plain store, which the write-back's page copy
/// turns into the bitmap - no atomic and no branch of its own: shaders with
/// buffer stores compile ~1.2x as long as without marks; 1: bitmap bits kept
/// per invocation and flushed when a store moves on to another bitmap word
/// and at the end, ~2x; 0: one subgroup-coalesced mark per store, ~3.5x; 2: a
/// plain atomic per lane and store, ~1.5x). Part of the translation key:
/// switching translates again.
pub const PAGE_MARKS: usize = 102;
/// Forwards of GPU writes that only batches in flight made (their device
/// copies, not in guest memory yet) are recorded in the batch prologue,
/// which runs behind those batches, instead of in the batch behind a full
/// barrier (`TLG_HOIST_FWD`, 1): for read-only and written buffer uploads and
/// mirror copy-ins; forwards of the batch's own writes stay in the batch.
pub const HOIST_FWD: usize = 103;
/// Guest threads of SceKernel priority at or above (numerically at most)
/// this run above normal on the host, as our GPU threads (`TLG_GUEST_PRIO`,
/// 256; 0: all normal): the title's sound mixer (`sndx_out_thread`) and
/// render submission ask for 256, its job workers for 552, as the PS4
/// schedules them. Applied when a thread starts and to every thread on
/// `tune guest_prio`; an in-run A/B at the 31% clamp measured no difference
/// in underruns or fps (LOG: Playtest of 0.3.1).
pub const GUEST_PRIO: usize = 104;
/// Measurement only: the wet output of NGS2 reverb racks scaled by this many
/// dB (`TLG_REV_DB`, 0), for comparing reverb levels with a PS4 recording.
pub const REV_DB: usize = 105;
/// Measurement only: with N > 0 the reverb output gain cycles through 0,
/// +12, +18 and +24 dB (on top of `rev_db`), then 0 dB with voice ports 4-7
/// taking port & 3's volume and matrix (as before they had their own), in
/// windows of N seconds of rendered audio, each window's start logged with
/// its frame in the audio dump (`TLG_REV_AB`, 0).
pub const REV_AB: usize = 106;
/// Measurement only: NGS2 voice routing as before 0.3.3 (`TLG_NGS2_COMPAT`,
/// 0; bit mask): 1 = a killed sampler voice keeps its patches, 2 = voice
/// ports 4-7 take port & 3's volume and matrix.
pub const NGS2_COMPAT: usize = 107;
/// Pipelines are created by this many worker threads while the CP goes on
/// (`TLG_PIPE_ASYNC`, -1: the logical processors less 4, 2-8; 0: by the CP
/// when it first needs them; `gpu::pcompile`). Cold boot to the cave and 60 s
/// there, no pipeline or driver cache, ~95% clock: 190 s with 41.7 s in
/// hitches (worst frame 4.2 s) on the CP, 171 s with 18.6 s (1.8 s) with 8
/// workers. Workers start when a pipeline is first queued.
pub const PIPE_ASYNC: usize = 108;
/// The prepare thread starts the compiles of the pipelines a submission's
/// draws and dispatches need that the CP has not made yet, before the CP
/// gets there (`TLG_PIPE_AHEAD`, 1; with `pipe_async` and `prep_ahead`): a
/// submission's new pipelines compile side by side instead of one after the
/// other as the CP reaches them.
pub const PIPE_AHEAD: usize = 109;
/// The NGS2 mixer computes with flush-to-zero and denormals-are-zero set
/// whatever the calling thread's MXCSR (`TLG_NGS2_FTZ`, 1; Windows' initial
/// 0x1F80 has neither, `ps4_fpu` sets both): the reverbs' and filters'
/// decaying tails reach subnormal values, which the CPU handles ~100 times
/// slower (values below 1e-38: no audible difference). The thread's MXCSR is
/// restored when the call returns.
pub const NGS2_FTZ: usize = 110;
/// Guest threads start with the PS4's floating-point state: MXCSR 0x9FC0
/// (exceptions masked, flush-to-zero, denormals-are-zero) and x87 control
/// word 0x037F, the default environment of the title's own libc
/// (`__fe_dfl_env` in libc.prx), where Windows starts threads with 0x1F80
/// (denormals kept, and ~100 times slower) and 0x027F (`TLG_PS4_FPU`, 1).
pub const PS4_FPU: usize = 111;
/// The NGS2 mixer skips what silence passes through unchanged
/// (`TLG_NGS2_IDLE`, 1): the mixing of silent voice outputs, a submixer's
/// filters on silent input while they have no state left (silence out, bit
/// for bit), and reverbs and delays on silent input once their state has
/// died away under -180 dB (it is then cleared).
pub const NGS2_IDLE: usize = 112;
/// The pipeline manifest (`gpu::manifest`, `TLG_PIPE_MANIFEST`, 1): the
/// session's translations and pipelines are recorded beside the pipeline
/// cache, and earlier sessions' are compiled again at the start, behind the
/// session's own compiles - after an update that changes the translator,
/// places the player has been do not stall on first sight again. Read at
/// start.
pub const PIPE_MANIFEST: usize = 113;
/// Draws whose graphics pipeline is still compiling (`TLG_NEW_SHADERS`, -1:
/// settings.txt `new_shaders`; 0 wait for it, 1 leave the draw out of the
/// frame - not draws that write buffers or are counted by an occlusion
/// query, whose results the game reads; `render::skip_compiling`).
pub const NEW_SHADERS: usize = 114;
/// The NGS2 mixer's lanes in vector form (`TLG_NGS2_VEC`, 1): the matrix
/// and direct mixes over all eight lanes with a mask, a voice's filter
/// bands in one pass (0: the lane loops and band-by-band passes before;
/// the same output bit for bit).
pub const NGS2_VEC: usize = 115;
/// The NGS2 mixer's helper thread (`TLG_NGS2_THREADS`, 1): with 8 or more
/// sampler voices playing, a second thread renders every other one
/// (decoding, envelope, filters, peak) while the rendering thread does the
/// rest; the routing stays in the voices' order (the same mix bit for bit).
pub const NGS2_THREADS: usize = 116;
/// A draw finds its known graphics pipeline with one lookup of its key
/// (`TLG_PIPE_LOOKUP1`, 1; 0: the checks for a new pipeline first, four
/// hashes of the key per draw).
pub const PIPE_LOOKUP1: usize = 117;
/// Shader preparation's vector pool: each thread keeps a stash and trades
/// with the shared pool a batch at a time (`TLG_POOL_LOCAL`, 1; 0: the
/// shared pool's lock for every vector).
pub const POOL_LOCAL: usize = 118;
/// Dry preparation logs the words it reads (contiguous reads merged) and
/// the graphics CP compares guest memory with them (`TLG_DRY_WORDS`, 1; 0:
/// a hash of each read, hashed again to compare).
pub const DRY_WORDS: usize = 119;
/// Vectors of each kind the shared pool of shader preparation keeps
/// (`TLG_POOL_CAP`, 8192; more are freed).
pub const POOL_CAP: usize = 120;

const TABLE: [(&str, &str, i64); 121] = [
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
    ("evict_small", "TLG_EVICT_SMALL", 2),
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
    ("recorder", "TLG_RECORDER", 1),
    ("rec_batch", "TLG_REC_BATCH", 64),
    ("prefetch", "TLG_PREFETCH", 0),
    ("async_copies", "TLG_ASYNC_COPIES", 1),
    ("vbind_parts", "TLG_VBIND_PARTS", 1),
    ("prep_ahead", "TLG_PREP_AHEAD", 1),
    ("decoded_cp", "TLG_DECODED_CP", 1),
    ("prep_priority", "TLG_PREP_PRIORITY", 0),
    ("ww_filter", "TLG_WW_FILTER", 1),
    ("hoist_copyins", "TLG_HOIST_COPYINS", 1),
    ("copy_inline", "TLG_COPY_INLINE", 0),
    ("title_sync", "TLG_TITLE_SYNC", 1),
    ("push_desc", "TLG_PUSH_DESC", 1),
    ("delay_gfx_ns", "TLG_DELAY_GFX_NS", 0),
    ("delay_cs_ns", "TLG_DELAY_CS_NS", 0),
    ("ww_log", "TLG_WW_LOG", 0),
    ("ww_bulk", "TLG_WW_BULK", 16),
    ("copy_spin_us", "TLG_COPY_SPIN_US", 1000),
    ("target_cache", "TLG_TARGET_CACHE", 1),
    ("dma_reg_bulk", "TLG_DMA_REG_BULK", 1),
    ("desc_templates", "TLG_DESC_TEMPLATES", 1),
    ("tex_cache", "TLG_TEX_CACHE", 1),
    ("perf_hot", "TLG_PERF_HOT", 1),
    ("dma_reg_last", "TLG_DMA_REG_LAST", 1),
    ("rec_spin_us", "TLG_REC_SPIN_US", 1000),
    ("vbind_part_kib", "TLG_VBIND_PART_KIB", 64),
    ("yield_queues", "TLG_YIELD_QUEUES", 0),
    ("sema_wake", "TLG_SEMA_WAKE", 1),
    ("mutex_spins", "TLG_MUTEX_SPINS", 4000),
    ("kick_wait", "TLG_KICK_WAIT", 100),
    ("spin_pauses", "TLG_SPIN_PAUSES", 1),
    ("hrt_spin", "TLG_HRT_SPIN", 0),
    ("gfx_core", "TLG_GFX_CORE", -2),
    ("gfx_core_soft", "TLG_GFX_CORE_SOFT", 0),
    ("cs_core", "TLG_CS_CORE", -1),
    ("shadow_load_view", "TLG_SHADOW_LOAD_VIEW", 0),
    ("zero_groups", "TLG_ZERO_GROUPS", 0),
    ("arm_after", "TLG_ARM_AFTER", 8),
    ("ring_bytes_ahead", "TLG_RING_BYTES_AHEAD", 1),
    ("vbind_recheck", "TLG_VBIND_RECHECK", 1),
    ("hz_dedupe", "TLG_HZ_DEDUPE", 1),
    ("frame_cap", "TLG_FRAME_CAP", -1),
    ("attach_once", "TLG_ATTACH_ONCE", 1),
    ("early_eop", "TLG_EARLY_EOP", 0),
    ("cs_ctx", "TLG_CS_CTX", 0),
    ("cs_inst", "TLG_CS_INST", 0),
    ("early_release", "TLG_EARLY_RELEASE", 0),
    ("cs_route", "TLG_CS_ROUTE", 1),
    ("upload_xbatch", "TLG_UPLOAD_XBATCH", 0),
    ("scan_index", "TLG_SCAN_INDEX", 1),
    ("tex_memo", "TLG_TEX_MEMO", 1),
    ("shader_stats", "TLG_SHADER_STATS", 0),
    ("memo_base", "TLG_MEMO_BASE", 1),
    ("page_marks", "TLG_PAGE_MARKS", 3),
    ("hoist_fwd", "TLG_HOIST_FWD", 1),
    ("guest_prio", "TLG_GUEST_PRIO", 256),
    ("rev_db", "TLG_REV_DB", 0),
    ("rev_ab", "TLG_REV_AB", 0),
    ("ngs2_compat", "TLG_NGS2_COMPAT", 0),
    ("pipe_async", "TLG_PIPE_ASYNC", -1),
    ("pipe_ahead", "TLG_PIPE_AHEAD", 1),
    ("ngs2_ftz", "TLG_NGS2_FTZ", 1),
    ("ps4_fpu", "TLG_PS4_FPU", 1),
    ("ngs2_idle", "TLG_NGS2_IDLE", 1),
    ("pipe_manifest", "TLG_PIPE_MANIFEST", 1),
    ("new_shaders", "TLG_NEW_SHADERS", -1),
    ("ngs2_vec", "TLG_NGS2_VEC", 1),
    ("ngs2_threads", "TLG_NGS2_THREADS", 1),
    ("pipe_lookup1", "TLG_PIPE_LOOKUP1", 1),
    ("pool_local", "TLG_POOL_LOCAL", 1),
    ("dry_words", "TLG_DRY_WORDS", 1),
    ("pool_cap", "TLG_POOL_CAP", 8192),
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
