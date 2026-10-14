//! Deferred command recording (`tune recorder`): the CP threads, under the
//! renderer lock, queue the driver work of their batches - command-buffer
//! recording, ending, resets, sparse binds and submits - and one recorder
//! thread makes those calls in queue order while the CPs go on with the next
//! commands. The driver calls were ~15% of the graphics CP's time (profile
//! 2026-10-14), all of it under the renderer lock the compute CPs wait for.
//!
//! `REC` has the `ash::Device` methods the renderer records with, with the
//! same signatures: a call site only changes its receiver. Each batch is
//! recorded either deferred or directly (`set_mode` at its start); work that
//! relies on submission order outside the queue (presents) waits for it
//! first (`drain`), and lazy write-back read-backs learn of submits from the
//! recorder (`lazywb::submitted`). Objects recorded into a batch must stay
//! alive until it completes, as before; nothing waits on a batch before its
//! submit was queued (fences may be waited on before the recorder submits).
//!
//! Operations are queued only by renderer methods, under the renderer lock
//! (`LOCAL` relies on it): they collect in `LOCAL` and are published to the
//! recorder in groups (`tune rec_batch`) and with every submit.

use ash::vk;
use parking_lot::{Condvar, Mutex};
use smallvec::SmallVec;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use super::vk::ctx;

type Thunk = Box<dyn FnOnce(&ash::Device) + Send>;

/// One queued operation; the frequent ones without a heap allocation.
enum Op {
    Barrier(vk::CommandBuffer, vk::PipelineStageFlags, vk::PipelineStageFlags, vk::MemoryBarrier<'static>),
    BindPipeline(vk::CommandBuffer, vk::PipelineBindPoint, vk::Pipeline),
    BindSet(vk::CommandBuffer, vk::PipelineBindPoint, vk::PipelineLayout, u32, vk::DescriptorSet),
    PushConstants(vk::CommandBuffer, vk::PipelineLayout, vk::ShaderStageFlags, u32, SmallVec<[u8; 32]>),
    Dispatch(vk::CommandBuffer, u32, u32, u32),
    Draw(vk::CommandBuffer, u32, u32, u32, u32),
    DrawIndexed(vk::CommandBuffer, u32, u32, u32, i32, u32),
    BindIndexBuffer(vk::CommandBuffer, vk::Buffer, u64, vk::IndexType),
    CopyBuffer(vk::CommandBuffer, vk::Buffer, vk::Buffer, SmallVec<[vk::BufferCopy; 2]>),
    FillBuffer(vk::CommandBuffer, vk::Buffer, u64, u64, u32),
    Timestamp(vk::CommandBuffer, vk::PipelineStageFlags, vk::QueryPool, u32),
    EndRendering(vk::CommandBuffer),
    Viewport(vk::CommandBuffer, vk::Viewport),
    Scissor(vk::CommandBuffer, vk::Rect2D),
    Dyn(vk::CommandBuffer, DynOp),
    Begin(vk::CommandBuffer),
    End(vk::CommandBuffer),
    Reset(vk::CommandBuffer),
    BindSparse(Vec<(vk::Buffer, Vec<vk::SparseMemoryBind>)>, vk::Semaphore),
    Submit { cmds: SmallVec<[vk::CommandBuffer; 2]>, waits: SmallVec<[(vk::Semaphore, vk::PipelineStageFlags); 1]>, fence: vk::Fence, lazy: Option<u64>, seq: u64 },
    Fn(Thunk),
}

/// Dynamic-state settings (one per setter).
#[derive(Clone, Copy)]
enum DynOp {
    CullMode(vk::CullModeFlags),
    FrontFace(vk::FrontFace),
    Topology(vk::PrimitiveTopology),
    RestartEnable(bool),
    RasterizerDiscard(bool),
    LineWidth(f32),
    DepthTest(bool),
    DepthWrite(bool),
    DepthCompare(vk::CompareOp),
    DepthBoundsTest(bool),
    DepthBounds(f32, f32),
    DepthBiasEnable(bool),
    DepthBias(f32, f32, f32),
    StencilTest(bool),
    StencilOp(vk::StencilFaceFlags, vk::StencilOp, vk::StencilOp, vk::StencilOp, vk::CompareOp),
    StencilRef(vk::StencilFaceFlags, u32),
    StencilCompareMask(vk::StencilFaceFlags, u32),
    StencilWriteMask(vk::StencilFaceFlags, u32),
    BlendConstants([f32; 4]),
}

struct Queue {
    ops: Vec<Op>,
}

/// Operations queued but not yet published to the recorder. Only renderer
/// methods queue operations, with the renderer lock held: that lock
/// serializes all access.
struct Local(std::cell::UnsafeCell<Vec<Op>>);

// SAFETY: accessed only under the renderer lock (see `push`).
unsafe impl Sync for Local {}

static LOCAL: Local = Local(std::cell::UnsafeCell::new(Vec::new()));

static QUEUE: Mutex<Queue> = Mutex::new(Queue { ops: Vec::new() });
static CV: Condvar = Condvar::new();
static SLEEPING: AtomicBool = AtomicBool::new(false);
/// Operations queued and performed so far.
static PUSHED: AtomicU64 = AtomicU64::new(0);
static DONE: AtomicU64 = AtomicU64::new(0);
static DRAIN_LOCK: Mutex<()> = Mutex::new(());
static DRAIN_CV: Condvar = Condvar::new();
static DRAINERS: AtomicU32 = AtomicU32::new(0);
/// The current batch is recorded through the queue.
static DEFERRED: AtomicBool = AtomicBool::new(false);
static STARTED: std::sync::Once = std::sync::Once::new();
/// When the recorder submitted batch `seq` (`SUBMITS[seq % 64]`: seq and
/// nanoseconds since `EPOCH`), for the GPU-busy accounting.
static SUBMITS: [(AtomicU64, AtomicU64); 64] = [const { (AtomicU64::new(0), AtomicU64::new(0)) }; 64];
static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
/// Operations performed, and how many arrived while the recorder slept
/// (TLG_PERF).
pub static OPS: AtomicU64 = AtomicU64::new(0);
pub static WAKES: AtomicU64 = AtomicU64::new(0);
/// Recorder thread time in driver calls (ns).
pub static BUSY_NS: AtomicU64 = AtomicU64::new(0);

/// Queued operations at which a sleeping recorder is woken (submits and
/// `drain` always wake it).
const WAKE_AT: usize = 32;

fn deferred() -> bool {
    DEFERRED.load(Ordering::Relaxed)
}

/// At the start of a batch (renderer lock held, nothing recorded): records
/// it through the queue or directly. Going direct, queued work of earlier
/// batches is performed first.
pub fn set_mode(defer: bool) {
    if defer == deferred() {
        return;
    }
    if defer {
        STARTED.call_once(|| {
            EPOCH.get_or_init(std::time::Instant::now);
            crate::thread::spawn_host("GPU-record", 8 << 20, Box::new(run)).expect("recorder thread");
        });
    } else {
        // SAFETY: the renderer lock is held (a batch is starting).
        publish(unsafe { &mut *LOCAL.0.get() }, true);
        drain();
    }
    DEFERRED.store(defer, Ordering::Relaxed);
}

/// Queues `op` (renderer lock held: see `Local`); `now` publishes the
/// operations collected so far at once and wakes the recorder (submits).
fn push(op: Op, now: bool) {
    // SAFETY: only renderer methods queue operations, under the renderer
    // lock, which serializes every access to `LOCAL`.
    let l = unsafe { &mut *LOCAL.0.get() };
    l.push(op);
    if now || l.len() >= crate::tune::get(crate::tune::REC_BATCH).max(1) as usize {
        publish(l, now);
    }
}

fn publish(l: &mut Vec<Op>, wake: bool) {
    if l.is_empty() {
        return;
    }
    let n = l.len() as u64;
    let queued = {
        let mut q = QUEUE.lock();
        q.ops.append(l);
        PUSHED.fetch_add(n, Ordering::Release);
        q.ops.len()
    };
    if (wake || queued >= WAKE_AT) && SLEEPING.load(Ordering::SeqCst) {
        CV.notify_one();
    }
}

fn thunk(f: impl FnOnce(&ash::Device) + Send + 'static) {
    push(Op::Fn(Box::new(f)), false);
}

/// Waits until everything published so far has been performed (all
/// submits queued so far: they are published at once).
pub fn drain() {
    let target = PUSHED.load(Ordering::Acquire);
    if DONE.load(Ordering::Acquire) >= target {
        return;
    }
    DRAINERS.fetch_add(1, Ordering::SeqCst);
    {
        // (Wake the recorder if it waits for more work.)
        let _q = QUEUE.lock();
        CV.notify_one();
    }
    let mut g = DRAIN_LOCK.lock();
    while DONE.load(Ordering::Acquire) < target {
        DRAIN_CV.wait_for(&mut g, std::time::Duration::from_millis(1));
    }
    drop(g);
    DRAINERS.fetch_sub(1, Ordering::SeqCst);
}

/// When the recorder submitted batch `seq`, if it did (and still knows).
pub fn submit_time(seq: u64) -> Option<std::time::Instant> {
    let (s, ns) = &SUBMITS[(seq % 64) as usize];
    if s.load(Ordering::Acquire) != seq {
        return None;
    }
    Some(*EPOCH.get()? + std::time::Duration::from_nanos(ns.load(Ordering::Relaxed)))
}

fn run() {
    crate::log::set_thread_name("GPU-record");
    crate::thread::raise_priority();
    let mut spare: Vec<Op> = Vec::with_capacity(1024);
    // (A failed sparse bind never signals its semaphore: the submit waiting
    // for it must not.)
    let mut unsignaled: Option<vk::Semaphore> = None;
    loop {
        let mut ops = {
            let mut q = QUEUE.lock();
            if q.ops.is_empty() {
                drop(q);
                // Briefly busy: CPs recording queue more within microseconds.
                let t0 = std::time::Instant::now();
                while PUSHED.load(Ordering::Acquire) == DONE.load(Ordering::Acquire) && t0.elapsed() < std::time::Duration::from_micros(50) {
                    std::hint::spin_loop();
                }
                q = QUEUE.lock();
                while q.ops.is_empty() {
                    SLEEPING.store(true, Ordering::SeqCst);
                    CV.wait(&mut q);
                    SLEEPING.store(false, Ordering::SeqCst);
                    WAKES.fetch_add(1, Ordering::Relaxed);
                }
            }
            std::mem::replace(&mut q.ops, std::mem::take(&mut spare))
        };
        let n = ops.len() as u64;
        let t0 = std::time::Instant::now();
        let d = &ctx().device;
        for op in ops.drain(..) {
            unsafe { perform(d, op, &mut unsignaled) };
        }
        BUSY_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        OPS.fetch_add(n, Ordering::Relaxed);
        spare = ops;
        DONE.fetch_add(n, Ordering::Release);
        if DRAINERS.load(Ordering::SeqCst) != 0 {
            let _g = DRAIN_LOCK.lock();
            DRAIN_CV.notify_all();
        }
    }
}

unsafe fn perform(d: &ash::Device, op: Op, unsignaled: &mut Option<vk::Semaphore>) {
    unsafe {
        match op {
            Op::Barrier(cb, src, dst, mb) => d.cmd_pipeline_barrier(cb, src, dst, vk::DependencyFlags::empty(), &[mb], &[], &[]),
            Op::BindPipeline(cb, bp, p) => d.cmd_bind_pipeline(cb, bp, p),
            Op::BindSet(cb, bp, layout, first, set) => d.cmd_bind_descriptor_sets(cb, bp, layout, first, &[set], &[]),
            Op::PushConstants(cb, layout, stages, off, data) => d.cmd_push_constants(cb, layout, stages, off, &data),
            Op::Dispatch(cb, x, y, z) => d.cmd_dispatch(cb, x, y, z),
            Op::Draw(cb, a, b, c, e) => d.cmd_draw(cb, a, b, c, e),
            Op::DrawIndexed(cb, a, b, c, v, e) => d.cmd_draw_indexed(cb, a, b, c, v, e),
            Op::BindIndexBuffer(cb, buf, off, ty) => d.cmd_bind_index_buffer(cb, buf, off, ty),
            Op::CopyBuffer(cb, src, dst, regions) => d.cmd_copy_buffer(cb, src, dst, &regions),
            Op::FillBuffer(cb, buf, off, size, data) => d.cmd_fill_buffer(cb, buf, off, size, data),
            Op::Timestamp(cb, stage, pool, q) => d.cmd_write_timestamp(cb, stage, pool, q),
            Op::EndRendering(cb) => d.cmd_end_rendering(cb),
            Op::Viewport(cb, v) => d.cmd_set_viewport(cb, 0, &[v]),
            Op::Scissor(cb, r) => d.cmd_set_scissor(cb, 0, &[r]),
            Op::Dyn(cb, o) => match o {
                DynOp::CullMode(m) => d.cmd_set_cull_mode(cb, m),
                DynOp::FrontFace(f) => d.cmd_set_front_face(cb, f),
                DynOp::Topology(t) => d.cmd_set_primitive_topology(cb, t),
                DynOp::RestartEnable(e) => d.cmd_set_primitive_restart_enable(cb, e),
                DynOp::RasterizerDiscard(e) => d.cmd_set_rasterizer_discard_enable(cb, e),
                DynOp::LineWidth(w) => d.cmd_set_line_width(cb, w),
                DynOp::DepthTest(e) => d.cmd_set_depth_test_enable(cb, e),
                DynOp::DepthWrite(e) => d.cmd_set_depth_write_enable(cb, e),
                DynOp::DepthCompare(op) => d.cmd_set_depth_compare_op(cb, op),
                DynOp::DepthBoundsTest(e) => d.cmd_set_depth_bounds_test_enable(cb, e),
                DynOp::DepthBounds(lo, hi) => d.cmd_set_depth_bounds(cb, lo, hi),
                DynOp::DepthBiasEnable(e) => d.cmd_set_depth_bias_enable(cb, e),
                DynOp::DepthBias(c, cl, sl) => d.cmd_set_depth_bias(cb, c, cl, sl),
                DynOp::StencilTest(e) => d.cmd_set_stencil_test_enable(cb, e),
                DynOp::StencilOp(f, a, b, c, op) => d.cmd_set_stencil_op(cb, f, a, b, c, op),
                DynOp::StencilRef(f, r) => d.cmd_set_stencil_reference(cb, f, r),
                DynOp::StencilCompareMask(f, m) => d.cmd_set_stencil_compare_mask(cb, f, m),
                DynOp::StencilWriteMask(f, m) => d.cmd_set_stencil_write_mask(cb, f, m),
                DynOp::BlendConstants(c) => d.cmd_set_blend_constants(cb, &c),
            },
            Op::Begin(cb) => {
                if let Err(e) = d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)) {
                    crate::error!("recorder: vkBeginCommandBuffer: {e}");
                }
            }
            Op::End(cb) => {
                if let Err(e) = d.end_command_buffer(cb) {
                    crate::error!("recorder: vkEndCommandBuffer: {e}");
                }
            }
            Op::Reset(cb) => {
                if let Err(e) = d.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty()) {
                    crate::error!("recorder: vkResetCommandBuffer: {e}");
                }
            }
            Op::BindSparse(binds, sem) => {
                if let Err(e) = ctx().bind_sparse(&binds, sem) {
                    crate::error!("render: {e}");
                    *unsignaled = Some(sem);
                }
            }
            Op::Submit { cmds, mut waits, fence, lazy, seq } => {
                if let Some(s) = unsignaled.take() {
                    waits.retain(|w| w.0 != s);
                }
                if let Err(e) = ctx().submit_cmds(&cmds, &waits, &[], fence) {
                    crate::error!("render submit: {e}");
                    if e.contains("lost") {
                        super::vk::report_device_lost();
                    }
                }
                let ns = EPOCH.get().map_or(0, |t| t.elapsed().as_nanos() as u64);
                let (s, t) = &SUBMITS[(seq % 64) as usize];
                t.store(ns, Ordering::Relaxed);
                s.store(seq, Ordering::Release);
                if let Some(t) = lazy {
                    super::lazywb::submitted(t);
                }
            }
            Op::Fn(f) => f(d),
        }
    }
}

/// `MemoryBarrier` without its (unused) extension chain, to keep.
fn mem_barrier(m: &vk::MemoryBarrier<'_>) -> vk::MemoryBarrier<'static> {
    debug_assert!(m.p_next.is_null());
    vk::MemoryBarrier::default().src_access_mask(m.src_access_mask).dst_access_mask(m.dst_access_mask)
}

fn buffer_barrier(b: &vk::BufferMemoryBarrier<'_>) -> vk::BufferMemoryBarrier<'static> {
    debug_assert!(b.p_next.is_null());
    vk::BufferMemoryBarrier::default()
        .src_access_mask(b.src_access_mask)
        .dst_access_mask(b.dst_access_mask)
        .src_queue_family_index(b.src_queue_family_index)
        .dst_queue_family_index(b.dst_queue_family_index)
        .buffer(b.buffer)
        .offset(b.offset)
        .size(b.size)
}

fn image_barrier(b: &vk::ImageMemoryBarrier<'_>) -> vk::ImageMemoryBarrier<'static> {
    debug_assert!(b.p_next.is_null());
    vk::ImageMemoryBarrier::default()
        .src_access_mask(b.src_access_mask)
        .dst_access_mask(b.dst_access_mask)
        .old_layout(b.old_layout)
        .new_layout(b.new_layout)
        .src_queue_family_index(b.src_queue_family_index)
        .dst_queue_family_index(b.dst_queue_family_index)
        .image(b.image)
        .subresource_range(b.subresource_range)
}

fn attachment(a: &vk::RenderingAttachmentInfo<'_>) -> vk::RenderingAttachmentInfo<'static> {
    debug_assert!(a.p_next.is_null());
    vk::RenderingAttachmentInfo::default()
        .image_view(a.image_view)
        .image_layout(a.image_layout)
        .resolve_mode(a.resolve_mode)
        .resolve_image_view(a.resolve_image_view)
        .resolve_image_layout(a.resolve_image_layout)
        .load_op(a.load_op)
        .store_op(a.store_op)
        .clear_value(a.clear_value)
}

/// The recording entry points (see the module comment).
pub struct Rec;

pub static REC: Rec = Rec;

#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
impl Rec {
    /// Arbitrary recording work for the current batch, in order (`f` gets
    /// the device).
    pub fn raw(&self, f: impl FnOnce(&ash::Device) + Send + 'static) {
        if deferred() {
            thunk(f);
        } else {
            f(&ctx().device);
        }
    }

    pub unsafe fn begin_command_buffer(&self, cb: vk::CommandBuffer, info: &vk::CommandBufferBeginInfo<'_>) -> ash::prelude::VkResult<()> {
        if deferred() {
            // (The renderer begins every batch command buffer this way.)
            debug_assert!(info.flags == vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT && info.p_inheritance_info.is_null());
            push(Op::Begin(cb), false);
            return Ok(());
        }
        unsafe { ctx().device.begin_command_buffer(cb, info) }
    }

    pub unsafe fn end_command_buffer(&self, cb: vk::CommandBuffer) -> ash::prelude::VkResult<()> {
        if deferred() {
            push(Op::End(cb), false);
            return Ok(());
        }
        unsafe { ctx().device.end_command_buffer(cb) }
    }

    pub unsafe fn reset_command_buffer(&self, cb: vk::CommandBuffer, flags: vk::CommandBufferResetFlags) -> ash::prelude::VkResult<()> {
        if deferred() {
            debug_assert!(flags.is_empty());
            push(Op::Reset(cb), false);
            return Ok(());
        }
        unsafe { ctx().device.reset_command_buffer(cb, flags) }
    }

    /// `Ctx::bind_sparse` in queue order (a failure is logged by the
    /// recorder, whose submit then skips the semaphore).
    pub fn bind_sparse(&self, binds: Vec<(vk::Buffer, Vec<vk::SparseMemoryBind>)>, signal: vk::Semaphore) -> Result<(), String> {
        if deferred() {
            push(Op::BindSparse(binds, signal), false);
            return Ok(());
        }
        ctx().bind_sparse(&binds, signal)
    }

    /// `Ctx::submit_cmds` of batch `seq` in queue order; `lazy`: the
    /// batch's last lazy write-back skip (`lazywb::submitted` once submitted).
    pub fn submit(&self, cmds: &[vk::CommandBuffer], waits: &[(vk::Semaphore, vk::PipelineStageFlags)], fence: vk::Fence, lazy: Option<u64>, seq: u64) -> Result<(), String> {
        if deferred() {
            push(Op::Submit { cmds: SmallVec::from_slice(cmds), waits: SmallVec::from_slice(waits), fence, lazy, seq }, true);
            return Ok(());
        }
        let r = ctx().submit_cmds(cmds, waits, &[], fence);
        if let Some(t) = lazy {
            super::lazywb::submitted(t);
        }
        r
    }

    pub unsafe fn cmd_pipeline_barrier(
        &self,
        cb: vk::CommandBuffer,
        src: vk::PipelineStageFlags,
        dst: vk::PipelineStageFlags,
        dep: vk::DependencyFlags,
        mem: &[vk::MemoryBarrier<'_>],
        buf: &[vk::BufferMemoryBarrier<'_>],
        img: &[vk::ImageMemoryBarrier<'_>],
    ) {
        if !deferred() {
            return unsafe { ctx().device.cmd_pipeline_barrier(cb, src, dst, dep, mem, buf, img) };
        }
        if dep.is_empty() && mem.len() == 1 && buf.is_empty() && img.is_empty() {
            return push(Op::Barrier(cb, src, dst, mem_barrier(&mem[0])), false);
        }
        let mem: SmallVec<[vk::MemoryBarrier<'static>; 1]> = mem.iter().map(mem_barrier).collect();
        let buf: SmallVec<[vk::BufferMemoryBarrier<'static>; 1]> = buf.iter().map(buffer_barrier).collect();
        let img: SmallVec<[vk::ImageMemoryBarrier<'static>; 2]> = img.iter().map(image_barrier).collect();
        thunk(move |d| unsafe { d.cmd_pipeline_barrier(cb, src, dst, dep, &mem, &buf, &img) });
    }

    pub unsafe fn cmd_bind_pipeline(&self, cb: vk::CommandBuffer, bp: vk::PipelineBindPoint, p: vk::Pipeline) {
        if deferred() {
            return push(Op::BindPipeline(cb, bp, p), false);
        }
        unsafe { ctx().device.cmd_bind_pipeline(cb, bp, p) }
    }

    pub unsafe fn cmd_bind_descriptor_sets(&self, cb: vk::CommandBuffer, bp: vk::PipelineBindPoint, layout: vk::PipelineLayout, first: u32, sets: &[vk::DescriptorSet], dynamic: &[u32]) {
        if !deferred() {
            return unsafe { ctx().device.cmd_bind_descriptor_sets(cb, bp, layout, first, sets, dynamic) };
        }
        if sets.len() == 1 && dynamic.is_empty() {
            return push(Op::BindSet(cb, bp, layout, first, sets[0]), false);
        }
        let (sets, dynamic) = (sets.to_vec(), dynamic.to_vec());
        thunk(move |d| unsafe { d.cmd_bind_descriptor_sets(cb, bp, layout, first, &sets, &dynamic) });
    }

    pub unsafe fn cmd_push_constants(&self, cb: vk::CommandBuffer, layout: vk::PipelineLayout, stages: vk::ShaderStageFlags, offset: u32, data: &[u8]) {
        if deferred() {
            return push(Op::PushConstants(cb, layout, stages, offset, SmallVec::from_slice(data)), false);
        }
        unsafe { ctx().device.cmd_push_constants(cb, layout, stages, offset, data) }
    }

    pub unsafe fn cmd_dispatch(&self, cb: vk::CommandBuffer, x: u32, y: u32, z: u32) {
        if deferred() {
            return push(Op::Dispatch(cb, x, y, z), false);
        }
        unsafe { ctx().device.cmd_dispatch(cb, x, y, z) }
    }

    pub unsafe fn cmd_dispatch_base(&self, cb: vk::CommandBuffer, bx: u32, by: u32, bz: u32, x: u32, y: u32, z: u32) {
        if deferred() {
            return thunk(move |d| unsafe { d.cmd_dispatch_base(cb, bx, by, bz, x, y, z) });
        }
        unsafe { ctx().device.cmd_dispatch_base(cb, bx, by, bz, x, y, z) }
    }

    pub unsafe fn cmd_dispatch_indirect(&self, cb: vk::CommandBuffer, buf: vk::Buffer, off: u64) {
        if deferred() {
            return thunk(move |d| unsafe { d.cmd_dispatch_indirect(cb, buf, off) });
        }
        unsafe { ctx().device.cmd_dispatch_indirect(cb, buf, off) }
    }

    pub unsafe fn cmd_draw(&self, cb: vk::CommandBuffer, vertices: u32, instances: u32, first_vertex: u32, first_instance: u32) {
        if deferred() {
            return push(Op::Draw(cb, vertices, instances, first_vertex, first_instance), false);
        }
        unsafe { ctx().device.cmd_draw(cb, vertices, instances, first_vertex, first_instance) }
    }

    pub unsafe fn cmd_draw_indexed(&self, cb: vk::CommandBuffer, indices: u32, instances: u32, first_index: u32, vertex_offset: i32, first_instance: u32) {
        if deferred() {
            return push(Op::DrawIndexed(cb, indices, instances, first_index, vertex_offset, first_instance), false);
        }
        unsafe { ctx().device.cmd_draw_indexed(cb, indices, instances, first_index, vertex_offset, first_instance) }
    }

    pub unsafe fn cmd_draw_indirect(&self, cb: vk::CommandBuffer, buf: vk::Buffer, off: u64, count: u32, stride: u32) {
        if deferred() {
            return thunk(move |d| unsafe { d.cmd_draw_indirect(cb, buf, off, count, stride) });
        }
        unsafe { ctx().device.cmd_draw_indirect(cb, buf, off, count, stride) }
    }

    pub unsafe fn cmd_draw_indexed_indirect(&self, cb: vk::CommandBuffer, buf: vk::Buffer, off: u64, count: u32, stride: u32) {
        if deferred() {
            return thunk(move |d| unsafe { d.cmd_draw_indexed_indirect(cb, buf, off, count, stride) });
        }
        unsafe { ctx().device.cmd_draw_indexed_indirect(cb, buf, off, count, stride) }
    }

    pub unsafe fn cmd_bind_index_buffer(&self, cb: vk::CommandBuffer, buf: vk::Buffer, off: u64, ty: vk::IndexType) {
        if deferred() {
            return push(Op::BindIndexBuffer(cb, buf, off, ty), false);
        }
        unsafe { ctx().device.cmd_bind_index_buffer(cb, buf, off, ty) }
    }

    pub unsafe fn cmd_copy_buffer(&self, cb: vk::CommandBuffer, src: vk::Buffer, dst: vk::Buffer, regions: &[vk::BufferCopy]) {
        if deferred() {
            return push(Op::CopyBuffer(cb, src, dst, SmallVec::from_slice(regions)), false);
        }
        unsafe { ctx().device.cmd_copy_buffer(cb, src, dst, regions) }
    }

    pub unsafe fn cmd_fill_buffer(&self, cb: vk::CommandBuffer, buf: vk::Buffer, off: u64, size: u64, data: u32) {
        if deferred() {
            return push(Op::FillBuffer(cb, buf, off, size, data), false);
        }
        unsafe { ctx().device.cmd_fill_buffer(cb, buf, off, size, data) }
    }

    pub unsafe fn cmd_write_timestamp(&self, cb: vk::CommandBuffer, stage: vk::PipelineStageFlags, pool: vk::QueryPool, query: u32) {
        if deferred() {
            return push(Op::Timestamp(cb, stage, pool, query), false);
        }
        unsafe { ctx().device.cmd_write_timestamp(cb, stage, pool, query) }
    }

    pub unsafe fn cmd_reset_query_pool(&self, cb: vk::CommandBuffer, pool: vk::QueryPool, first: u32, count: u32) {
        if deferred() {
            return thunk(move |d| unsafe { d.cmd_reset_query_pool(cb, pool, first, count) });
        }
        unsafe { ctx().device.cmd_reset_query_pool(cb, pool, first, count) }
    }

    pub unsafe fn cmd_begin_query(&self, cb: vk::CommandBuffer, pool: vk::QueryPool, query: u32, flags: vk::QueryControlFlags) {
        if deferred() {
            return thunk(move |d| unsafe { d.cmd_begin_query(cb, pool, query, flags) });
        }
        unsafe { ctx().device.cmd_begin_query(cb, pool, query, flags) }
    }

    pub unsafe fn cmd_end_query(&self, cb: vk::CommandBuffer, pool: vk::QueryPool, query: u32) {
        if deferred() {
            return thunk(move |d| unsafe { d.cmd_end_query(cb, pool, query) });
        }
        unsafe { ctx().device.cmd_end_query(cb, pool, query) }
    }

    pub unsafe fn cmd_begin_rendering(&self, cb: vk::CommandBuffer, info: &vk::RenderingInfo<'_>) {
        if !deferred() {
            return unsafe { ctx().device.cmd_begin_rendering(cb, info) };
        }
        debug_assert!(info.p_next.is_null());
        let colors: SmallVec<[vk::RenderingAttachmentInfo<'static>; 4]> = unsafe { std::slice::from_raw_parts(info.p_color_attachments, info.color_attachment_count as usize) }.iter().map(attachment).collect();
        let depth = unsafe { info.p_depth_attachment.as_ref() }.map(attachment);
        let stencil = unsafe { info.p_stencil_attachment.as_ref() }.map(attachment);
        let (flags, area, layers, mask) = (info.flags, info.render_area, info.layer_count, info.view_mask);
        thunk(move |d| {
            let mut ri = vk::RenderingInfo::default().flags(flags).render_area(area).layer_count(layers).view_mask(mask).color_attachments(&colors);
            if let Some(a) = &depth {
                ri = ri.depth_attachment(a);
            }
            if let Some(a) = &stencil {
                ri = ri.stencil_attachment(a);
            }
            unsafe { d.cmd_begin_rendering(cb, &ri) }
        });
    }

    pub unsafe fn cmd_end_rendering(&self, cb: vk::CommandBuffer) {
        if deferred() {
            return push(Op::EndRendering(cb), false);
        }
        unsafe { ctx().device.cmd_end_rendering(cb) }
    }

    pub unsafe fn cmd_clear_attachments(&self, cb: vk::CommandBuffer, attachments: &[vk::ClearAttachment], rects: &[vk::ClearRect]) {
        if deferred() {
            let (a, r) = (attachments.to_vec(), rects.to_vec());
            return thunk(move |d| unsafe { d.cmd_clear_attachments(cb, &a, &r) });
        }
        unsafe { ctx().device.cmd_clear_attachments(cb, attachments, rects) }
    }

    pub unsafe fn cmd_clear_color_image(&self, cb: vk::CommandBuffer, image: vk::Image, layout: vk::ImageLayout, value: &vk::ClearColorValue, ranges: &[vk::ImageSubresourceRange]) {
        if deferred() {
            let (v, r) = (*value, ranges.to_vec());
            return thunk(move |d| unsafe { d.cmd_clear_color_image(cb, image, layout, &v, &r) });
        }
        unsafe { ctx().device.cmd_clear_color_image(cb, image, layout, value, ranges) }
    }

    pub unsafe fn cmd_clear_depth_stencil_image(&self, cb: vk::CommandBuffer, image: vk::Image, layout: vk::ImageLayout, value: &vk::ClearDepthStencilValue, ranges: &[vk::ImageSubresourceRange]) {
        if deferred() {
            let (v, r) = (*value, ranges.to_vec());
            return thunk(move |d| unsafe { d.cmd_clear_depth_stencil_image(cb, image, layout, &v, &r) });
        }
        unsafe { ctx().device.cmd_clear_depth_stencil_image(cb, image, layout, value, ranges) }
    }

    pub unsafe fn cmd_copy_buffer_to_image(&self, cb: vk::CommandBuffer, src: vk::Buffer, dst: vk::Image, layout: vk::ImageLayout, regions: &[vk::BufferImageCopy]) {
        if deferred() {
            let r = regions.to_vec();
            return thunk(move |d| unsafe { d.cmd_copy_buffer_to_image(cb, src, dst, layout, &r) });
        }
        unsafe { ctx().device.cmd_copy_buffer_to_image(cb, src, dst, layout, regions) }
    }

    pub unsafe fn cmd_copy_image_to_buffer(&self, cb: vk::CommandBuffer, src: vk::Image, layout: vk::ImageLayout, dst: vk::Buffer, regions: &[vk::BufferImageCopy]) {
        if deferred() {
            let r = regions.to_vec();
            return thunk(move |d| unsafe { d.cmd_copy_image_to_buffer(cb, src, layout, dst, &r) });
        }
        unsafe { ctx().device.cmd_copy_image_to_buffer(cb, src, layout, dst, regions) }
    }

    pub unsafe fn cmd_copy_image(&self, cb: vk::CommandBuffer, src: vk::Image, src_layout: vk::ImageLayout, dst: vk::Image, dst_layout: vk::ImageLayout, regions: &[vk::ImageCopy]) {
        if deferred() {
            let r = regions.to_vec();
            return thunk(move |d| unsafe { d.cmd_copy_image(cb, src, src_layout, dst, dst_layout, &r) });
        }
        unsafe { ctx().device.cmd_copy_image(cb, src, src_layout, dst, dst_layout, regions) }
    }

    pub unsafe fn cmd_set_viewport(&self, cb: vk::CommandBuffer, first: u32, viewports: &[vk::Viewport]) {
        if deferred() {
            if let ([v], 0) = (viewports, first) {
                return push(Op::Viewport(cb, *v), false);
            }
            let v: SmallVec<[vk::Viewport; 1]> = SmallVec::from_slice(viewports);
            return thunk(move |d| unsafe { d.cmd_set_viewport(cb, first, &v) });
        }
        unsafe { ctx().device.cmd_set_viewport(cb, first, viewports) }
    }

    pub unsafe fn cmd_set_scissor(&self, cb: vk::CommandBuffer, first: u32, scissors: &[vk::Rect2D]) {
        if deferred() {
            if let ([r], 0) = (scissors, first) {
                return push(Op::Scissor(cb, *r), false);
            }
            let s: SmallVec<[vk::Rect2D; 1]> = SmallVec::from_slice(scissors);
            return thunk(move |d| unsafe { d.cmd_set_scissor(cb, first, &s) });
        }
        unsafe { ctx().device.cmd_set_scissor(cb, first, scissors) }
    }
}

/// Dynamic-state setters: queued as `DynOp`s.
macro_rules! dyn_setters {
    ($($name:ident($($arg:ident: $ty:ty),*) => $variant:ident;)*) => {
        #[allow(clippy::missing_safety_doc)]
        impl Rec {
            $(
                pub unsafe fn $name(&self, cb: vk::CommandBuffer, $($arg: $ty),*) {
                    if deferred() {
                        return push(Op::Dyn(cb, DynOp::$variant($($arg),*)), false);
                    }
                    unsafe { ctx().device.$name(cb, $($arg),*) }
                }
            )*
        }
    };
}

dyn_setters! {
    cmd_set_cull_mode(mode: vk::CullModeFlags) => CullMode;
    cmd_set_front_face(face: vk::FrontFace) => FrontFace;
    cmd_set_primitive_topology(topology: vk::PrimitiveTopology) => Topology;
    cmd_set_primitive_restart_enable(enable: bool) => RestartEnable;
    cmd_set_rasterizer_discard_enable(enable: bool) => RasterizerDiscard;
    cmd_set_line_width(width: f32) => LineWidth;
    cmd_set_depth_test_enable(enable: bool) => DepthTest;
    cmd_set_depth_write_enable(enable: bool) => DepthWrite;
    cmd_set_depth_compare_op(op: vk::CompareOp) => DepthCompare;
    cmd_set_depth_bounds_test_enable(enable: bool) => DepthBoundsTest;
    cmd_set_depth_bounds(min: f32, max: f32) => DepthBounds;
    cmd_set_depth_bias_enable(enable: bool) => DepthBiasEnable;
    cmd_set_depth_bias(constant: f32, clamp: f32, slope: f32) => DepthBias;
    cmd_set_stencil_test_enable(enable: bool) => StencilTest;
    cmd_set_stencil_op(face: vk::StencilFaceFlags, fail: vk::StencilOp, pass: vk::StencilOp, depth_fail: vk::StencilOp, compare: vk::CompareOp) => StencilOp;
    cmd_set_stencil_reference(face: vk::StencilFaceFlags, reference: u32) => StencilRef;
    cmd_set_stencil_compare_mask(face: vk::StencilFaceFlags, mask: u32) => StencilCompareMask;
    cmd_set_stencil_write_mask(face: vk::StencilFaceFlags, mask: u32) => StencilWriteMask;
}

#[allow(clippy::missing_safety_doc)]
impl Rec {
    pub unsafe fn cmd_set_blend_constants(&self, cb: vk::CommandBuffer, constants: &[f32; 4]) {
        if deferred() {
            return push(Op::Dyn(cb, DynOp::BlendConstants(*constants)), false);
        }
        unsafe { ctx().device.cmd_set_blend_constants(cb, constants) }
    }
}
