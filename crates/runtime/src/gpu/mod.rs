//! GPU emulation front end: submission queues, compute rings and events.

pub mod copier;
pub mod cp;
pub mod cpuwatch;
pub mod desc;
pub mod lazywb;
pub mod pm4;
pub mod rec;
pub mod render;
pub mod resources;
pub mod shaders;
pub mod texdump;
pub mod tiling;
pub mod shader_dump;
pub mod vk;
pub mod wbprobe;
pub mod writes;

use cp::{CommandProcessor, Engine, Hooks, Stream};
use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct FlipRequest {
    pub handle: i32,
    pub index: i32,
    pub mode: u32,
    pub arg: i64,
}

pub struct Submission {
    pub dcb: Vec<(u64, u32)>,
    pub ccb: Vec<(u64, u32)>,
    pub flip: Option<FlipRequest>,
    /// Command words fetched at submission (see `cp::capture`).
    pub captured: cp::Captured,
    /// When the title submitted it (frame pacing attribution).
    pub submitted: std::time::Instant,
}

fn pm4_check() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("TLG_PM4_CHECK").is_some())
}

fn read_dwords(addr: u64, n: u32) -> Vec<u32> {
    (0..n as u64).map(|i| unsafe { std::ptr::read_volatile((addr + 4 * i) as *const u32) }).collect()
}

struct GfxQueue {
    pending: Mutex<VecDeque<Submission>>,
    cv: Condvar,
    /// Signalled whenever a submission has been processed (back-pressure).
    done: Condvar,
}

pub struct ComputeRing {
    pub base: u64,
    pub size_dw: u32,
    pub read_ptr_addr: u64,
    pub read_dw: u32,
    pub pending_to: Option<u32>,
}

struct ComputeQueue {
    ring: Mutex<ComputeRing>,
    cv: Condvar,
}

static GFX: OnceLock<GfxQueue> = OnceLock::new();
static COMPUTE: Mutex<Vec<Option<&'static ComputeQueue>>> = Mutex::new(Vec::new());
static EOP_EVENTS: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());
pub static SUBMITS: AtomicU64 = AtomicU64::new(0);
pub static FRAMES: AtomicU64 = AtomicU64::new(0);
/// CPU time the GPU thread has used (user + kernel, 100 ns units; TLG_PERF):
/// its work without waits, a measure that other GPU load hardly moves.
pub static GFX_CPU_100NS: AtomicU64 = AtomicU64::new(0);

/// This thread's CPU time (user + kernel) in 100 ns units.
fn thread_cpu_100ns() -> u64 {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadTimes};
    let z = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut c, mut e, mut k, mut u) = (z, z, z, z);
    if unsafe { GetThreadTimes(GetCurrentThread(), &mut c, &mut e, &mut k, &mut u) } == 0 {
        return 0;
    }
    let ft = |f: FILETIME| ((f.dwHighDateTime as u64) << 32) | f.dwLowDateTime as u64;
    ft(k) + ft(u)
}
/// CPU-write epochs (see `render::Renderer::new_epoch`): a sequence number
/// bumped whenever the title may have written guest memory for the GPU, and
/// its value at the last graphics submission / compute queue doorbell. A
/// validation done at sequence V holds for graphics work while V >= the
/// graphics epoch (likewise for compute queues): within one, guest memory
/// changes only through GPU work and the CP (both tracked).
pub static CPU_EPOCH: AtomicU64 = AtomicU64::new(0);
pub static GFX_EPOCH: AtomicU64 = AtomicU64::new(0);
pub static CS_EPOCH: AtomicU64 = AtomicU64::new(0);
thread_local! {
    /// Epoch validations made by this thread must reach (set per command).
    pub static REQUIRED_EPOCH: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// GPU-thread activity for the stall watchdog: phase in the top byte, a
/// phase-specific argument (usually a guest address) in the low 56 bits.
pub static PHASE: AtomicU64 = AtomicU64::new(0);
pub static PROGRESS: AtomicU64 = AtomicU64::new(0);
/// Per-queue activity for compute rings (index = virtual queue id).
pub static QPHASE: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];
pub static QPROGRESS: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];

#[inline]
pub fn set_qphase(q: usize, p: u8, arg: u64) {
    if let (Some(ph), Some(pr)) = (QPHASE.get(q), QPROGRESS.get(q)) {
        ph.store(((p as u64) << 56) | (arg & ((1 << 56) - 1)), Ordering::Relaxed);
        pr.fetch_add(1, Ordering::Relaxed);
    }
}

pub mod phase {
    pub const IDLE: u8 = 0;
    pub const PACKET: u8 = 1;
    pub const WAIT_REG_MEM: u8 = 2;
    pub const DRAW: u8 = 3;
    pub const DISPATCH: u8 = 4;
    pub const SUBMIT_WAIT: u8 = 5;
    pub const PIPELINE: u8 = 6;
    pub const UPLOAD: u8 = 7;
    pub const FLIP_WAIT: u8 = 8;
    pub const NAMES: [&str; 9] = ["idle", "packet", "WAIT_REG_MEM", "draw", "dispatch", "submit+wait", "pipeline creation", "texture upload", "flip wait"];
}

#[inline]
pub fn set_phase(p: u8, arg: u64) {
    PHASE.store(((p as u64) << 56) | (arg & ((1 << 56) - 1)), Ordering::Relaxed);
    PROGRESS.fetch_add(1, Ordering::Relaxed);
}

/// Before the OS itself reads or writes guest memory [addr, addr+len) (file
/// I/O): GPU-written pages kept inaccessible there (lazy write-back, the
/// write-back probe) made current and accessible - the OS fails such
/// accesses instead of faulting.
pub fn os_access(addr: u64, len: u64) {
    lazywb::touch(addr, len);
    wbprobe::touch(addr, len);
    cpuwatch::touch(addr, len);
}

/// Logs what the graphics ring and every compute ring are doing (diagnosing hangs).
pub fn dump_state() {
    let fmt = |ph: u64| format!("{} {:#x}", phase::NAMES.get((ph >> 56) as usize).copied().unwrap_or("?"), ph & ((1 << 56) - 1));
    let mut s = format!(
        "GPU: gfx {} (progress {}), queued submissions {}, flips {}",
        fmt(PHASE.load(Ordering::Relaxed)),
        PROGRESS.load(Ordering::Relaxed),
        GFX.get().map(|q| q.pending.lock().len()).unwrap_or(0),
        FRAMES.load(Ordering::Relaxed)
    );
    for q in 1..64 {
        let ph = QPHASE[q].load(Ordering::Relaxed);
        let pr = QPROGRESS[q].load(Ordering::Relaxed);
        if pr != 0 {
            s.push_str(&format!("\n  compute queue {q}: {} (progress {pr})", fmt(ph)));
        }
    }
    crate::info!("{s}");
}

/// Logs when the GPU thread makes no progress for 500 ms while busy.
fn watchdog() {
    let mut last = PROGRESS.load(Ordering::Relaxed);
    let mut since = std::time::Instant::now();
    let mut reported = false;
    let mut tick = 0u64;
    let mut perf_prev: Option<PerfSnap> = None;
    let mut qlast = [0u64; 64];
    let mut qsince = [std::time::Instant::now(); 64];
    let mut qreported = [false; 64];
    loop {
        std::thread::sleep(std::time::Duration::from_millis(100));
        tick += 1;
        if tick % 2 == 0 {
            pool_watch(tick);
        }
        if tick % 10 == 0 {
            perf_log(&mut perf_prev);
        }
        if tick % 20 == 0 && std::env::var_os("TLG_PERF").is_some() {
            let v: Vec<String> = cp::OPCODES.iter().enumerate().filter_map(|(i, c)| {
                let n = c.load(Ordering::Relaxed);
                (n != 0).then(|| format!("{}{:#04x}:{}", if i >= 256 { "c" } else { "" }, i & 0xFF, n))
            }).collect();
            crate::info!("pm4 opcodes: {}", v.join(" "));
        }
        for q in 1..64 {
            let now = QPROGRESS[q].load(Ordering::Relaxed);
            let ph = QPHASE[q].load(Ordering::Relaxed);
            if now != qlast[q] || (ph >> 56) as u8 == phase::IDLE {
                if qreported[q] {
                    crate::warn!("compute queue {q} stall over after {} ms", qsince[q].elapsed().as_millis());
                }
                qlast[q] = now;
                qsince[q] = std::time::Instant::now();
                qreported[q] = false;
            } else if !qreported[q] && qsince[q].elapsed() > std::time::Duration::from_millis(500) {
                qreported[q] = true;
                crate::warn!(
                    "compute queue {q} stall: no progress for {} ms in {} ({:#x})",
                    qsince[q].elapsed().as_millis(),
                    phase::NAMES.get((ph >> 56) as usize).copied().unwrap_or("?"),
                    ph & ((1 << 56) - 1)
                );
            }
        }
        let now = PROGRESS.load(Ordering::Relaxed);
        let ph = PHASE.load(Ordering::Relaxed);
        if now != last || (ph >> 56) as u8 == phase::IDLE {
            if reported {
                crate::warn!("GPU stall over after {} ms", since.elapsed().as_millis());
            }
            last = now;
            since = std::time::Instant::now();
            reported = false;
            continue;
        }
        if !reported && since.elapsed() > std::time::Duration::from_millis(500) {
            reported = true;
            let p = (ph >> 56) as usize;
            crate::warn!(
                "GPU stall: no progress for {} ms in {} ({:#x}); queued submissions {}",
                since.elapsed().as_millis(),
                phase::NAMES.get(p).copied().unwrap_or("?"),
                ph & ((1 << 56) - 1),
                GFX.get().map(|q| q.pending.lock().len()).unwrap_or(0)
            );
        }
    }
}
/// Image dump requested through the control channel (taken at the next flip).
pub static DUMP_REQUEST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The renderer report (and the GPU profile since the last one) is printed
/// after the next submission (`report` control command).
pub static REPORT_REQUEST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Placeholder renderer: records work until the Vulkan backend handles it.
pub struct NullRenderer {
    pub logged: u64,
    /// Graphics queue: EOP interrupts signal the registered graphics events.
    pub gfx: bool,
}

impl Hooks for NullRenderer {
    fn draw(&mut self, cp: &CommandProcessor, kind: &'static str, count: u32, index_addr: u64) {
        let r = &cp.regs;
        let vs = ((r.sh(0x2C48) as u64) | ((r.sh(0x2C49) as u64) << 32)) << 8;
        let ps = ((r.sh(0x2C08) as u64) | ((r.sh(0x2C09) as u64) << 32)) << 8;
        shader_dump::dump("vs", vs, &r.sh[0x4C..0x5C]);
        if ps != 0 {
            shader_dump::dump("ps", ps, &r.sh[0x0C..0x1C]);
        }
        for st in [gcn::translate::Stage::Vertex, gcn::translate::Stage::Fragment] {
            if st == gcn::translate::Stage::Fragment && ps == 0 {
                continue;
            }
            match shaders::prepare(&shaders::StageRegs { stage: st, regs: &cp.regs }) {
                Ok(p) => {
                    if self.logged < 12 {
                        crate::debug!("{:?} {:#x} resources: {:?}", st, p.addr, p.resources);
                    }
                }
                Err(e) => report_once(&e),
            }
        }
        if self.logged < 8 {
            self.logged += 1;
            let r = &cp.regs;
            crate::debug!(
                "{}: draw {} count {} idx {:#x} vs {:#x} ps {:#x} cb0 {:#x} markers {:?}",
                cp.name,
                kind,
                count,
                index_addr,
                ((r.sh(0x2C48) as u64) | ((r.sh(0x2C49) as u64) << 32)) << 8,
                ((r.sh(0x2C08) as u64) | ((r.sh(0x2C09) as u64) << 32)) << 8,
                (r.ctx(0xA318) as u64) << 8,
                cp.markers
            );
        }
    }
    fn dispatch(&mut self, cp: &CommandProcessor, x: u32, y: u32, z: u32) {
        let cs = ((cp.regs.sh(0x2E0C) as u64) | ((cp.regs.sh(0x2E0D) as u64) << 32)) << 8;
        shader_dump::dump("cs", cs, &cp.regs.sh[0x240..0x250]);
        if let Err(e) = shaders::prepare(&shaders::StageRegs { stage: gcn::translate::Stage::Compute, regs: &cp.regs }) {
            report_once(&e);
        }
        if self.logged < 8 {
            self.logged += 1;
            crate::debug!(
                "{}: dispatch {}x{}x{} cs {:#x} markers {:?}",
                cp.name,
                x,
                y,
                z,
                ((cp.regs.sh(0x2E0C) as u64) | ((cp.regs.sh(0x2E0D) as u64) << 32)) << 8,
                cp.markers
            );
        }
    }
    fn flush(&mut self, _cp: &CommandProcessor) {}
    fn commit(&mut self, _cp: &CommandProcessor, actions: Vec<cp::EopAction>) {
        for a in &actions {
            a.perform();
        }
    }
    fn sync_range(&mut self, _cp: &CommandProcessor, _addr: u64, _len: u64) {}
}

/// Delivers a graphics end-of-pipe interrupt to registered event queues.
pub fn graphics_eop_interrupt() {
    let evs = EOP_EVENTS.lock().clone();
    for (eq, id) in evs {
        if let Some(q) = unsafe { crate::hle::kernel::equeue::from_handle(eq) } {
            q.trigger(id, crate::hle::kernel::equeue::EVFILT_GRAPHICS, None, None);
        }
    }
}

pub fn add_eop_event(eq: u64, id: u64) {
    EOP_EVENTS.lock().push((eq, id));
    if let Some(q) = unsafe { crate::hle::kernel::equeue::from_handle(eq) } {
        q.add(id, crate::hle::kernel::equeue::EVFILT_GRAPHICS, 0, 0);
    }
}

pub fn remove_eop_event(eq: u64, id: u64) {
    EOP_EVENTS.lock().retain(|&(e, i)| !(e == eq && i == id));
    if let Some(q) = unsafe { crate::hle::kernel::equeue::from_handle(eq) } {
        q.remove(id, crate::hle::kernel::equeue::EVFILT_GRAPHICS);
    }
}

fn gfx_queue() -> &'static GfxQueue {
    GFX.get_or_init(|| {
        let q = GfxQueue { pending: Mutex::new(VecDeque::new()), cv: Condvar::new(), done: Condvar::new() };
        crate::thread::spawn_host("GPU-gfx", 16 << 20, Box::new(gfx_thread)).expect("gpu thread");
        crate::thread::spawn_host("GPU-watchdog", 1 << 20, Box::new(watchdog)).expect("gpu watchdog");
        q
    })
}

/// Debug (TLG_PM4_CAPTURE=1): execute command buffers as fetched at submit
/// time instead of reading them live. Live reading is what the hardware does
/// and is required: titles patch submitted command buffers (e.g. calls to
/// worker command lists written into reserved slots after submission).
/// Command buffers (and the indirect buffers they call) are snapshotted at
/// submission and executed from the snapshot: the title rewrites its DCBs
/// once it believes the GPU is done with them, which can happen while our
/// command processor still lags behind. `TLG_PM4_CAPTURE=0` executes live.
fn pm4_capture() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("TLG_PM4_CAPTURE").map(|v| v != "0").unwrap_or(true))
}

pub fn submit(mut s: Submission) {
    if pm4_capture() || pm4_check() {
        let roots: Vec<(u64, u32)> = s.dcb.iter().chain(s.ccb.iter()).copied().filter(|&(a, n)| a != 0 && n != 0).collect();
        s.captured = cp::capture(&roots);
    }
    let q = gfx_queue();
    SUBMITS.fetch_add(1, Ordering::Relaxed);
    let mut g = q.pending.lock();
    g.push_back(s);
    q.cv.notify_one();
    // Back-pressure: like a full submission ring, keep the title from running
    // far ahead of the GPU. Titles recycle per-frame memory (constants,
    // dynamic buffers) on their own schedule; with our command processor
    // lagging many submissions behind, draws would read later frames' data.
    // TLG_GPU_QUEUE_LIMIT (default 6, 0 = unlimited); waits at most
    // TLG_GPU_QUEUE_WAIT_MS (default 250) ms.
    static LIMIT: OnceLock<(usize, u64)> = OnceLock::new();
    let (limit, wait_ms) = *LIMIT.get_or_init(|| {
        let n = std::env::var("TLG_GPU_QUEUE_LIMIT").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
        let ms = std::env::var("TLG_GPU_QUEUE_WAIT_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(250);
        (n, ms)
    });
    if limit > 0 {
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(wait_ms);
        while g.len() > limit {
            if q.done.wait_until(&mut g, deadline).timed_out() {
                break;
            }
        }
    }
}

fn gfx_thread() {
    crate::log::set_thread_name("GPU-gfx");
    crate::thread::raise_priority();
    let q = gfx_queue();
    let mut cp = CommandProcessor::new("gfx");
    cp.trace = crate::log::enabled(crate::log::Level::Trace);
    let mut hooks = render::VkHooks { fallback: NullRenderer { logged: 0, gfx: true } };
    let mut busy = std::time::Duration::ZERO;
    let mut processed: u64 = 0;
    let started = std::time::Instant::now();
    loop {
        set_phase(phase::IDLE, 0);
        let mut s = {
            let mut g = q.pending.lock();
            while g.is_empty() {
                q.cv.wait(&mut g);
            }
            g.pop_front().unwrap()
        };
        set_phase(phase::PACKET, s.dcb.first().map(|d| d.0).unwrap_or(0));
        if let Some(r) = render::RENDERER.lock().as_mut() {
            r.new_epoch(render::EpochKind::Graphics);
        }
        let t_start = std::time::Instant::now();
        let has_flip = s.flip.is_some();
        let captured = std::sync::Arc::new(std::mem::take(&mut s.captured));
        let exec_captured = if pm4_capture() { captured.clone() } else { std::sync::Arc::new(cp::Captured::new()) };
        let mut de = Stream::chained(exec_captured.clone());
        let mut ce = Stream::chained(exec_captured.clone());
        for (i, &(addr, dw)) in s.dcb.iter().enumerate() {
            if pm4_check() {
                if let Some(snap) = captured.get(&addr) {
                    let now = read_dwords(addr, dw);
                    let diffs: Vec<usize> = (0..now.len().min(snap.len())).filter(|&k| now[k] != snap[k]).collect();
                    if let Some(&first) = diffs.first() {
                        let show: Vec<String> = diffs.iter().take(6).map(|&k| format!("+{:#x}: {:08x}->{:08x}", k * 4, snap[k], now[k])).collect();
                        crate::warn!(
                            "gfx: DCB {addr:#x} ({dw} dw) was rewritten after submission: {} dwords differ, first at {:#x}; queue depth {}; {}",
                            diffs.len(),
                            addr + 4 * first as u64,
                            q.pending.lock().len(),
                            show.join(" ")
                        );
                    }
                }
            }
            note_exec(addr, false);
            de.chain(addr, dw);
            if let Some(&(a, n)) = s.ccb.get(i) {
                ce.chain(a, n);
            }
        }
        cp.run(&mut de, &mut ce, &mut hooks);
        // (Coalescing, `tune::COALESCE`: the next queued submission continues
        // this batch; its end-of-pipe actions stay pending in the CP, where
        // memory waits on them are resolved as before.)
        let coalesce = !has_flip && crate::tune::get(crate::tune::COALESCE) != 0 && !q.pending.lock().is_empty();
        if !coalesce {
            cp.commit(&mut hooks);
            // Close the batch at the end of each title submission so shader
            // results the title may poll for reach memory without further work.
            if let Some(r) = render::RENDERER.lock().as_mut() {
                r.submit_batch(Vec::new());
            }
        }
        if let Some(f) = s.flip {
            let ended = FRAMES.fetch_add(1, Ordering::Relaxed);
            if render::TRACE_REQUEST.load(Ordering::Relaxed) == ended {
                render::traced_rehash();
                crate::info!("FRAME end of traced frame {ended}");
                // TLG_TRACE_DUMP=1: the images of each traced frame (see
                // TLG_DUMP_BASES) as they are at its end.
                if std::env::var_os("TLG_TRACE_DUMP").is_some() {
                    if let Some(r) = render::RENDERER.lock().as_mut() {
                        r.submit_batch(Vec::new());
                        r.dump_images(ended);
                    }
                }
            }
            // New pipelines reach the on-disk cache within half a minute.
            render::save_pipeline_cache(std::time::Duration::from_secs(30));
            if let Some(r) = render::RENDERER.lock().as_mut() {
                // Submit (no wait): presentation is ordered after it on the queue.
                r.submit_batch(Vec::new());
                r.housekeeping();
                r.verify_gpu_textures();
                note_frame(FRAMES.load(Ordering::Relaxed), [r.draws, r.dispatches, r.skipped]);
                let frame = FRAMES.load(Ordering::Relaxed);
                if DUMP_REQUEST.swap(false, Ordering::Relaxed) || std::env::var("TLG_DUMP_IMAGES").ok().and_then(|v| v.parse::<u64>().ok()) == Some(frame) {
                    r.dump_images(frame);
                }
            }
            let timing = crate::pacing::FlipTiming { submitted: s.submitted, queue: t_start.saturating_duration_since(s.submitted), process: t_start.elapsed() };
            crate::hle::libs::videoout::gpu_flip(f.handle, f.index, f.mode, f.arg, timing);
        }
        busy += t_start.elapsed();
        GFX_CPU_100NS.store(thread_cpu_100ns(), Ordering::Relaxed);
        processed += 1;
        if crate::sampler::active() && t_start.elapsed() >= crate::sampler::spike_threshold() {
            crate::sampler::spike(t_start, std::time::Instant::now(), format!("submission {processed} ({} DCBs{})", s.dcb.len(), if has_flip { ", flip" } else { "" }));
        }
        q.done.notify_all();
        let n = processed;
        if n <= 3 || n % 600 == 0 || REPORT_REQUEST.swap(false, Ordering::Relaxed) {
            crate::info!("gfx: submission {} done; busy {:.1}s of {:.1}s; queued {}; stats {:?}", n, busy.as_secs_f64(), started.elapsed().as_secs_f64(), q.pending.lock().len(), cp.stats);
            if let Some(r) = render::RENDERER.lock().as_mut() {
                crate::info!("renderer: draws {} dispatches {} skipped {} images {} pipelines {}; write-backs skipped for reused memory {}; reused render targets sampled from guest memory {}; texture validations settled by edge pages {}; depth views: cropped copies {} ({} by compute), direct for texel loads {}", r.draws, r.dispatches, r.skipped, r.image_count(), r.pipeline_count(), r.reused_skips(), r.reused_targets(), r.edge_skips(), render::DEPTH_VIEWS[0].load(std::sync::atomic::Ordering::Relaxed), render::DEPTH_VIEWS[2].load(std::sync::atomic::Ordering::Relaxed), render::DEPTH_VIEWS[1].load(std::sync::atomic::Ordering::Relaxed));
                let ub = r.upload_bytes;
                crate::info!("buffer copies: written {} MiB, misaligned {} MiB, pending writes {} MiB, indices {} MiB; read-only copies reused {} times ({} MiB); read-only copies into the BAR ring {} ({} MiB; {} to the host ring for want of room); index data from the mirror {} draws ({} MiB)", ub[0] >> 20, ub[1] >> 20, ub[2] >> 20, ub[3] >> 20, r.upload_reused.0, r.upload_reused.1 >> 20, r.bar_uploads.0, r.bar_uploads.1 >> 20, r.bar_uploads.2, r.index_mirrored.0, r.index_mirrored.1 >> 20);
                crate::info!("render targets (de)tiled on the GPU: reloads {}, write-backs {}; textures detiled on the GPU {}; ring-full retries {}", r.gpu_remaps[0], r.gpu_remaps[1], r.gpu_tex_uploads, r.ring_retries);
                let ((hp, hs), (dp, ds)) = r.ring_peaks();
                crate::info!("rings since last report: host peak {} of {} MiB per segment, device peak {} of {} MiB", hp >> 20, hs >> 20, dp >> 20, ds >> 20);
                let mips: Vec<String> = render::GPU_MIP_STAT_NAMES.iter().zip(render::GPU_MIP_STATS.iter()).map(|(n, v)| format!("{n} {}", v.load(Ordering::Relaxed))).collect();
                crate::info!("GPU mip-chain/BC texture path: {}", mips.join(", "));
                crate::info!("texture validations: {} unwritten, {} hashed ({} MiB hashed in total)", r.writes.clean, r.writes.dirty, r.hashed_bytes >> 20);
                if let Some((mib, n, copied, hits)) = r.vcache_stats() {
                    crate::info!("VRAM mirrors: {mib} MiB in {n} windows; {copied} MiB copied in; {hits} bindings served");
                }
                crate::info!("{}", r.vram_report());
                crate::info!("EVENT_WRITE types: {}", cp::event_type_counts());
                let k: Vec<u64> = render::SYNC_WAIT_KIND_NS.iter().map(|a| a.load(Ordering::Relaxed) / 1_000_000).collect();
                crate::info!("CP sync waits (ms, total): for eop actions {}, write-backs {}, huge in-place writes {}, in-place writes {}; CP writes behind in-place reads {}; virtual waits {}, in-batch waits {}, waits met by other queues' queued writes {}; label writes queued behind end-of-pipe writes {}", k[0], k[1], k[2], k[3], k[4], cp.stats.virtual_waits, cp.stats.inbatch_waits, cp.stats.cross_waits, cp::DEFERRED_WRITES.load(Ordering::Relaxed));
                let mut polls: Vec<(cp::PollKey, (u64, u64))> = cp::POLLS.lock().take().unwrap_or_default().into_iter().collect();
                polls.sort_by(|a, b| b.1.1.cmp(&a.1.1));
                let polls: Vec<String> = polls.iter().take(12).map(|((q, f, r, b, a), (n, us))| format!("q{q} func {f} ref {r:#x} {b:#x}->{a:#x}: {n}x {} ms", us / 1000)).collect();
                crate::info!("WAIT_REG_MEM polls over 0.5 ms since last report (queue, test, value before -> when met): {}", polls.join("; "));
                let mut declines: Vec<(&str, u64)> = render::META_DECLINES.lock().take().unwrap_or_default().into_iter().collect();
                declines.sort_by(|a, b| b.1.cmp(&a.1));
                crate::info!(
                    "shader preparations synchronized with GPU writes: {}; scatter dispatches narrowed: {}; copy/fill kernels as image operations: {}; mirror copy-ins around the window's own writes: {}; DMA copies on the GPU through mirrors: {}; fast clears decided on the GPU: {} (left to the CPU since last report: {:?})",
                    r.prepare_syncs,
                    r.scatters,
                    r.elided,
                    r.copyin_kept,
                    r.dma_mirror,
                    r.meta_gpu,
                    declines
                );
                wbprobe::report();
                lazywb::report();
                cpuwatch::report();
                let mut sites: Vec<(u32, u64)> = render::FLUSH_SITES.lock().take().unwrap_or_default().into_iter().collect();
                sites.sort_by(|a, b| b.1.cmp(&a.1));
                crate::info!("synchronizing flushes by render.rs line (count; 100000+line: wait microseconds) since last report: {:?}", &sites[..sites.len().min(12)]);
                let mut cs: Vec<(u32, (u64, u64))> = cp::SYNC_SITES.lock().take().unwrap_or_default().into_iter().collect();
                cs.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
                let cs: Vec<String> = cs.iter().take(10).map(|(l, (n, us))| format!("cp.rs:{l} {n}x {}ms", us / 1000)).collect();
                crate::info!("CP memory accesses that waited for the GPU since last report: {}", cs.join(", "));
                let dd: Vec<String> = render::DMA_DECLINE_REASONS.iter().zip(render::DMA_DECLINED.iter()).map(|(n, v)| format!("{n} {}", v.load(Ordering::Relaxed))).collect();
                crate::info!("DMA_DATA copies left to the CP (total, by reason): {}", dd.join(", "));
                let mut b: Vec<(u32, u64)> = render::BARRIER_SITES.lock().take().unwrap_or_default().into_iter().collect();
                b.sort_by(|x, y| y.1.cmp(&x.1));
                crate::info!("full barriers by render.rs line since last report: {:?}", &b[..b.len().min(14)]);
                let (emitted, skipped) = std::mem::take(&mut r.hz_stats);
                let [args, unknown, conflict] = std::mem::take(&mut r.hz_reasons);
                let mut other: Vec<(u32, u64)> = std::mem::take(&mut r.hz_other_lines).into_iter().collect();
                other.sort_by(|x, y| y.1.cmp(&x.1));
                crate::info!("barriers before dispatches since last report: {emitted} emitted, {skipped} left out (no conflicting accesses; tune dispatch_hazards); needed for indirect arguments {args}, untracked resources {unknown}, conflicting accesses {conflict}, other unordered work by render.rs line {:?}", &other[..other.len().min(10)]);
                if render::perf_on() {
                    let mut bs: Vec<((&str, u32), (u64, u64))> = render::BATCH_SITES.lock().take().unwrap_or_default().into_iter().collect();
                    bs.sort_by(|x, y| y.1 .0.cmp(&x.1 .0));
                    let bs: Vec<String> = bs.iter().take(12).map(|((f, l), (n, c))| format!("{f}:{l} {n}x ({:.1} cmds)", *c as f64 / *n as f64)).collect();
                    crate::info!("batches by closing call site since last report: {}", bs.join(", "));
                    let mut up: Vec<(String, (u64, u64))> = render::UPLOAD_SITES.lock().take().unwrap_or_default().into_iter().collect();
                    up.sort_by(|x, y| y.1 .1.cmp(&x.1 .1));
                    crate::info!("texture uploads since last report: {} textures, {} uploads, {} ms reading", up.len(), up.iter().map(|x| x.1 .0).sum::<u64>(), up.iter().map(|x| x.1 .1).sum::<u64>() / 1000);
                    for (s, (n, us)) in up.iter().take(8) {
                        crate::info!("  {n:5}x {:7.1} ms  {s}", *us as f64 / 1000.0);
                    }
                    for (name, m) in [
                        ("render targets made older than guest memory by buffer writes (writer, range, target)", &render::NEWER_SITES),
                        ("render-target reloads (target, at shader)", &render::RELOAD_SITES),
                        ("shader preparations that waited for pending GPU writes (reader, address, write)", &render::GUARD_SITES),
                        ("upload copies with forwards of pending GPU writes (read range, reader, write)", &render::FORWARD_SITES),
                    ] {
                        let mut v: Vec<(String, u64)> = m.lock().take().unwrap_or_default().into_iter().collect();
                        v.sort_by(|x, y| y.1.cmp(&x.1));
                        crate::info!("{name} since last report: {} kinds, {} total", v.len(), v.iter().map(|x| x.1).sum::<u64>());
                        for (s, n) in v.iter().take(12) {
                            crate::info!("  {n:6}x {s}");
                        }
                    }
                }
                if render::gpu_profile_on() {
                    let prof = render::take_gpu_profile();
                    let total: u64 = prof.iter().map(|p| p.2).sum();
                    crate::info!("GPU profile since last report: {:.1} ms in {} timed operations", total as f64 / 1e6, prof.iter().map(|p| p.1).sum::<u64>());
                    for (label, n, ns) in prof.iter().take(20) {
                        crate::info!("  {:9.2} ms {:6}x {:8.1} us  {}", *ns as f64 / 1e6, n, *ns as f64 / 1e3 / *n as f64, label);
                    }
                }
                for (m, c) in r.top_reports(6) {
                    crate::info!("  {c:>8} x {}", m.chars().take(160).collect::<String>());
                }
                let parts: Vec<String> = render::PROF_NAMES.iter().zip(r.prof.iter()).map(|(n, d)| format!("{n} {:.2}s", d.as_secs_f64())).collect();
                crate::info!("renderer time: {}", parts.join(", "));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Compute rings

pub fn map_compute_queue(pipe: u32, queue: u32, base: u64, size_dw: u32, read_ptr: u64) -> i32 {
    if pipe > 6 || queue > 7 || base == 0 || size_dw == 0 || !size_dw.is_power_of_two() {
        return 0x8026_0001u32 as i32;
    }
    let vq = (pipe * 8 + queue + 1) as usize;
    let cq: &'static ComputeQueue = Box::leak(Box::new(ComputeQueue {
        ring: Mutex::new(ComputeRing { base, size_dw, read_ptr_addr: read_ptr, read_dw: 0, pending_to: None }),
        cv: Condvar::new(),
    }));
    {
        let mut c = COMPUTE.lock();
        if c.len() <= vq {
            c.resize(vq + 1, None);
        }
        c[vq] = Some(cq);
    }
    let name = format!("GPU-acb{pipe}.{queue}");
    let tname = name.clone();
    crate::thread::spawn_host(&name, 8 << 20, Box::new(move || compute_thread(cq, tname, vq))).expect("compute thread");
    crate::info!("compute queue pipe {pipe} queue {queue} ring {base:#x} ({size_dw} dw) rptr {read_ptr:#x} -> vqueue {vq}");
    vq as i32
}

pub fn ding_dong(vq: u32, next_dw: u32) {
    if std::env::var_os("TLG_COMPUTE_LOG").is_some() {
        crate::info!("DingDong vq {vq} -> {next_dw:#x}");
    }
    let c = COMPUTE.lock().get(vq as usize).copied().flatten();
    match c {
        Some(cq) => {
            cq.ring.lock().pending_to = Some(next_dw);
            cq.cv.notify_one();
        }
        None => crate::error!("sceGnmDingDong on unmapped vqueue {vq}"),
    }
}

fn compute_thread(cq: &'static ComputeQueue, name: String, vq: usize) {
    crate::log::set_thread_name(&name);
    crate::thread::raise_priority();
    let mut cp = CommandProcessor::new(&name);
    cp.queue = vq;
    let mut hooks = render::VkHooks { fallback: NullRenderer { logged: 0, gfx: false } };
    loop {
        let (base, size, from, to, rptr) = {
            set_qphase(vq, phase::IDLE, 0);
            let mut r = cq.ring.lock();
            while r.pending_to.is_none() {
                cq.cv.wait(&mut r);
            }
            let to = r.pending_to.take().unwrap() & (r.size_dw - 1);
            (r.base, r.size_dw, r.read_dw, to, r.read_ptr_addr)
        };
        // The title may have written guest memory before ringing: textures
        // and cached buffers this queue's work reads are validated again.
        if let Some(r) = render::RENDERER.lock().as_mut() {
            r.new_epoch(render::EpochKind::Compute);
        }
        // Execute [from, to) with wrap-around.
        let mut pos = from;
        while pos != to {
            let end = if to > pos { to } else { size };
            let mut s = Stream::new(base + pos as u64 * 4, end - pos);
            cp.run_single(&mut s, Engine::Compute, &mut hooks);
            pos = end % size;
        }
        set_qphase(vq, phase::SUBMIT_WAIT, to as u64);
        cp.commit(&mut hooks);
        if let Some(r) = render::RENDERER.lock().as_mut() {
            r.submit_batch(Vec::new());
        }
        if std::env::var_os("TLG_COMPUTE_LOG").is_some() {
            let mut words = Vec::new();
            let mut p = from;
            while p != to && words.len() < 64 {
                words.push(format!("{:08x}", unsafe { std::ptr::read_volatile((base + p as u64 * 4) as *const u32) }));
                p = (p + 1) % size;
            }
            crate::info!("compute vq {vq}: executed [{from:#x}, {to:#x}) packets {}: {}", cp.stats.packets, words.join(" "));
        }
        cq.ring.lock().read_dw = to;
        if rptr != 0 {
            unsafe { std::ptr::write_volatile(rptr as *mut u32, to) };
        }
    }
}

fn report_once(msg: &str) {
    static SEEN: Mutex<Option<std::collections::HashSet<String>>> = Mutex::new(None);
    let mut g = SEEN.lock();
    if g.get_or_insert_with(Default::default).insert(msg.to_string()) {
        crate::error!("shader: {msg}");
    }
}

/// Debug (TLG_POOL_WATCH=<hex address>): every 30 flips, logs the status words
/// of a title command-chunk pool (object layout: +8 chunk qwords, +0x10 chunk
/// count, +0x18 base; status = last qword of each chunk, 0 = free).
/// Debug: number of buffer executions (top-level or indirect) that started
/// inside the watched pool's memory, and on the compute queue.
pub static POOL_EXEC: AtomicU64 = AtomicU64::new(0);
pub static POOL_EXEC_COMPUTE: AtomicU64 = AtomicU64::new(0);
static POOL_RANGE: Mutex<(u64, u64)> = Mutex::new((0, 0));

pub fn note_exec(addr: u64, compute: bool) {
    let (lo, hi) = *POOL_RANGE.lock();
    if addr >= lo && addr < hi {
        POOL_EXEC.fetch_add(1, Ordering::Relaxed);
        if compute {
            POOL_EXEC_COMPUTE.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn pool_watch(tick: u64) {
    static ADDR: OnceLock<Option<u64>> = OnceLock::new();
    static LAST: AtomicU64 = AtomicU64::new(u64::MAX);
    let Some(pool) = *ADDR.get_or_init(|| std::env::var("TLG_POOL_WATCH").ok().and_then(|v| u64::from_str_radix(v.trim_start_matches("0x"), 16).ok())) else { return };
    let frame = FRAMES.load(Ordering::Relaxed);
    let queued = GFX.get().map(|q| q.pending.lock().len()).unwrap_or(0);
    let ph = PHASE.load(Ordering::Relaxed);
    let rd = |a: u64| -> Option<u64> {
        let mut b = [0u8; 8];
        resources::copy_from_guest(&mut b, a);
        Some(u64::from_le_bytes(b))
    };
    let (Some(qw), Some(n), Some(base)) = (rd(pool + 8), rd(pool + 0x10), rd(pool + 0x18)) else { return };
    if n == 0 || n > 4096 || qw == 0 {
        return;
    }
    *POOL_RANGE.lock() = (base, base + n * qw * 8);
    let mut counts = [0u32; 4];
    for i in 0..n {
        let st = rd(base + (i * qw + qw - 1) * 8).unwrap_or(9);
        counts[(st.min(3)) as usize] += 1;
    }
    // Log on change (or every 5 s).
    let sig = (counts[0] as u64) | ((counts[2] as u64) << 16) | ((queued as u64) << 32);
    if LAST.swap(sig, Ordering::Relaxed) != sig || tick % 50 == 0 {
        crate::info!(
            "pool {pool:#x} flip {frame}: free {} claimed {} submitted {} other {} | executed {} (compute {}) | gfx queue {queued}, phase {}, progress {}",
            counts[0], counts[1], counts[2], counts[3],
            POOL_EXEC.load(Ordering::Relaxed),
            POOL_EXEC_COMPUTE.load(Ordering::Relaxed),
            phase::NAMES.get((ph >> 56) as usize).copied().unwrap_or("?"),
            PROGRESS.load(Ordering::Relaxed)
        );
    }
}

/// Draws, dispatches and skipped draws recorded for each of the last flips
/// (logged with screenshots: a frame missing scenery with far fewer draws
/// was culled by the title, with the usual count by the renderer).
static FRAME_LOG: Mutex<(VecDeque<(u64, [u64; 3])>, [u64; 3])> = Mutex::new((VecDeque::new(), [0; 3]));

fn note_frame(flip: u64, now: [u64; 3]) {
    let mut g = FRAME_LOG.lock();
    let prev = g.1;
    g.1 = now;
    g.0.push_back((flip, [now[0] - prev[0], now[1] - prev[1], now[2] - prev[2]]));
    if g.0.len() > 8 {
        g.0.pop_front();
    }
}

/// The last frames as "flip:draws/dispatches/skipped", oldest first.
pub fn recent_frames() -> String {
    let v: Vec<String> = FRAME_LOG.lock().0.iter().map(|(f, c)| format!("{f}:{}/{}/{}", c[0], c[1], c[2])).collect();
    v.join(" ")
}

#[derive(Clone, Copy)]
struct PerfSnap {
    frames: u64,
    draws: u64,
    dispatches: u64,
    flushes: [u64; 6],
    uploads: u64,
    hashed: u64,
    prof: [std::time::Duration; 7],
    gpu_busy_ns: u64,
    gpu_exec_ns: u64,
    sync_wait_ns: u64,
    barriers: u64,
    passes: u64,
    harvests: u64,
    harvest_pages: u64,
    harvest_ns: u64,
    ww_cached: u64,
    cpu_epochs: u64,
    gfx_cpu: u64,
    batches: u64,
    dprof: [u64; 8],
    dprof_draws: u64,
    dsub: [u64; 6],
    bsub: [u64; 11],
    bcnt: [u64; 11],
    image_reuse: (u64, u64),
    pro_ops: u64,
    pro_barriers: u64,
    tsc: u64,
}

/// Debug (TLG_PERF=1): per-second GPU-thread activity.
fn perf_log(prev: &mut Option<PerfSnap>) {
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("TLG_PERF").is_some()) {
        return;
    }
    let Some(g) = render::RENDERER.try_lock_for(std::time::Duration::from_millis(200)) else { return };
    let Some(r) = g.as_ref() else { return };
    let now = PerfSnap {
        frames: FRAMES.load(Ordering::Relaxed),
        draws: r.draws,
        dispatches: r.dispatches,
        flushes: r.flushes,
        uploads: r.uploads,
        hashed: r.hashed_bytes,
        prof: r.prof,
        gpu_busy_ns: render::GPU_BUSY_NS.load(Ordering::Relaxed),
        gpu_exec_ns: render::GPU_EXEC_NS.load(Ordering::Relaxed),
        sync_wait_ns: render::SYNC_WAIT_NS.load(Ordering::Relaxed),
        barriers: render::BARRIERS.load(Ordering::Relaxed),
        passes: render::PASS_BEGINS.load(Ordering::Relaxed),
        harvests: crate::mem::HARVESTS.load(Ordering::Relaxed),
        harvest_pages: crate::mem::HARVEST_PAGES.load(Ordering::Relaxed),
        harvest_ns: crate::mem::HARVEST_NS.load(Ordering::Relaxed),
        ww_cached: r.writes.cached,
        cpu_epochs: CPU_EPOCH.load(Ordering::Relaxed),
        gfx_cpu: GFX_CPU_100NS.load(Ordering::Relaxed),
        batches: r.last_seq,
        dprof: r.dprof,
        dprof_draws: r.dprof_draws,
        dsub: r.dsub,
        bsub: r.bsub,
        bcnt: r.bcnt,
        image_reuse: r.image_reuse,
        pro_ops: r.pro_ops,
        pro_barriers: render::PRO_BARRIERS.load(Ordering::Relaxed),
        tsc: unsafe { core::arch::x86_64::_rdtsc() },
    };
    drop(g);
    if let Some(p) = prev {
        let fl: Vec<String> = render::FLUSH_REASONS.iter().zip(now.flushes.iter().zip(p.flushes.iter())).filter(|(_, (a, b))| a != b).map(|(n, (a, b))| format!("{n} {}", a - b)).collect();
        let pr: Vec<String> = render::PROF_NAMES.iter().zip(now.prof.iter().zip(p.prof.iter())).map(|(n, (a, b))| format!("{n} {}ms", (*a - *b).as_millis())).collect();
        let vram = vk::try_ctx().and_then(|c| c.vram()).map(|(u, b)| format!(" | vram {}/{} MiB", u >> 20, b >> 20)).unwrap_or_default();
        // Draw phases in µs per draw (time-stamp counter, rate from this
        // second's wall time).
        static LAST_T: Mutex<Option<std::time::Instant>> = Mutex::new(None);
        let now_t = std::time::Instant::now();
        let secs = LAST_T.lock().replace(now_t).map_or(1.0, |t| now_t.duration_since(t).as_secs_f64().max(1e-3));
        let hz = now.tsc.wrapping_sub(p.tsc) as f64 / secs;
        let nd = now.dprof_draws.saturating_sub(p.dprof_draws).max(1) as f64;
        let ph: Vec<String> = render::DRAW_PHASES.iter().enumerate().map(|(i, n)| format!("{n} {:.1}", now.dprof[i].wrapping_sub(p.dprof[i]) as f64 / hz * 1e6 / nd)).collect();
        let ds: Vec<String> = render::DESC_PARTS.iter().enumerate().map(|(i, n)| format!("{n} {:.1}", now.dsub[i].wrapping_sub(p.dsub[i]) as f64 / hz * 1e6 / nd)).collect();
        crate::info!("draw µs/draw: {}; descriptor parts (all commands, per draw): {}", ph.join(", "), ds.join(", "));
        // Buffer bindings by path: count/s and µs each (all commands).
        let bp: Vec<String> = render::BUFFER_PATHS
            .iter()
            .enumerate()
            .filter(|(i, _)| now.bcnt[*i] != p.bcnt[*i])
            .map(|(i, n)| {
                let c = now.bcnt[i] - p.bcnt[i];
                format!("{n} {c} x {:.1} us", now.bsub[i].wrapping_sub(p.bsub[i]) as f64 / hz * 1e6 / c as f64)
            })
            .collect();
        crate::info!("buffer bindings/s by path: {}; batch prologue operations {} (prologue barriers {})", bp.join(", "), now.pro_ops - p.pro_ops, now.pro_barriers - p.pro_barriers);
        // (Their steps: us each, count.)
        static LAST_BP: Mutex<[(u64, u64); 17]> = Mutex::new([(0, 0); 17]);
        let cur: [(u64, u64); 17] = std::array::from_fn(|i| (render::BPARTS[i].0.load(Ordering::Relaxed), render::BPARTS[i].1.load(Ordering::Relaxed)));
        let prev = std::mem::replace(&mut *LAST_BP.lock(), cur);
        let parts: Vec<String> = (0..17)
            .filter(|&i| cur[i].1 > prev[i].1)
            .map(|i| format!("{} {:.2} us x {}", render::BPART_NAMES[i], (cur[i].0 - prev[i].0) as f64 / hz * 1e6 / (cur[i].1 - prev[i].1) as f64, cur[i].1 - prev[i].1))
            .collect();
        if !parts.is_empty() {
            let x = [render::XBATCH[0].swap(0, Ordering::Relaxed), render::XBATCH[1].swap(0, Ordering::Relaxed)];
            let v = [render::VBIND_STATS[0].swap(0, Ordering::Relaxed), render::VBIND_STATS[1].swap(0, Ordering::Relaxed)];
            crate::info!("buffer binding steps/s: {}; read-only copies an earlier batch could have served: {} same CPU epoch, {} same required epoch; mirror binding cache {} of {}", parts.join("; "), x[0], x[1], v[0], v[1]);
        }
        crate::info!(
            "perf/s: flips {} draws {} dispatches {} passes {} barriers {} batches {} cpu epochs {} write-watch queries {} ({} pages, {} ms; {} checks answered by this epoch's) uploads {} hashed {} MB | gpu busy {}ms (executing {}ms), cp sync waits {}ms, GPU thread CPU {}ms | flushes [{}] | {}{}",
            now.frames - p.frames,
            now.draws - p.draws,
            now.dispatches - p.dispatches,
            now.passes - p.passes,
            now.barriers - p.barriers,
            now.batches - p.batches,
            now.cpu_epochs - p.cpu_epochs,
            now.harvests - p.harvests,
            now.harvest_pages - p.harvest_pages,
            (now.harvest_ns - p.harvest_ns) / 1_000_000,
            now.ww_cached - p.ww_cached,
            now.uploads - p.uploads,
            (now.hashed - p.hashed) >> 20,
            (now.gpu_busy_ns - p.gpu_busy_ns) / 1_000_000,
            (now.gpu_exec_ns - p.gpu_exec_ns) / 1_000_000,
            (now.sync_wait_ns - p.sync_wait_ns) / 1_000_000,
            now.gfx_cpu.saturating_sub(p.gfx_cpu) / 10_000,
            fl.join(", "),
            pr.join(", "),
            vram
        );
        // Images created in the second, the most frequent kinds first.
        let mut ni: Vec<((u32, u32, u32, u32, ash::vk::Format), (u64, u64))> = render::NEW_IMAGES.lock().take().map(|m| m.into_iter().collect()).unwrap_or_default();
        if !ni.is_empty() {
            ni.sort_by_key(|e| std::cmp::Reverse(e.1 .0));
            let n: u64 = ni.iter().map(|e| e.1 .0).sum();
            let top: Vec<String> = ni.iter().take(6).map(|((w, h, l, m, f), (c, b))| format!("{c} x {w}x{h}x{l} {m} levels {f:?} (last at {b:#x})")).collect();
            let pooled = now.image_reuse.1 - p.image_reuse.1;
            crate::info!("images created/s: {n} ({pooled} from the pool of small evicted images); {}", top.join(", "));
        }
        // The largest buffer uploads of the second (base, bytes, written).
        let mut ups: Vec<((u64, u64, bool), u64)> = render::BUFFER_UPLOAD_SITES.lock().take().map(|m| m.into_iter().collect()).unwrap_or_default();
        if !ups.is_empty() {
            ups.sort_by_key(|&((_, size, _), n)| std::cmp::Reverse(size * n));
            let total: u64 = ups.iter().map(|&((_, s, _), n)| s * n).sum();
            let top: Vec<String> = ups.iter().take(8).map(|&((b, s, w), n)| format!("{n} x {b:#x}+{s:#x}{}", if w { " (written)" } else { "" })).collect();
            // (Read-only ranges: how often each was copied, and the bytes.)
            let ro: Vec<&((u64, u64, bool), u64)> = ups.iter().filter(|u| !u.0 .2).collect();
            let (ro_n, ro_bytes) = (ro.iter().map(|u| u.1).sum::<u64>(), ro.iter().map(|u| u.0 .1 * u.1).sum::<u64>());
            let same = render::UPLOADS_UNCHANGED.swap(0, Ordering::Relaxed);
            let reusable = render::UPLOADS_REUSABLE.swap(0, Ordering::Relaxed);
            crate::info!("buffer uploads/s: {} MiB in {} bindings ({} ranges; read-only: {} ranges, {} copies, {} MiB, {same} copies equal to the range's previous one, {reusable} with no write found since); largest: {}", total >> 20, ups.iter().map(|u| u.1).sum::<u64>(), ups.len(), ro.len(), ro_n, ro_bytes >> 20, top.join(", "));
        }
        // Per command-processor queue: renderer lock wait / hold, WAIT_REG_MEM (ms).
        static LAST_Q: Mutex<Option<[[u64; 3]; 64]>> = Mutex::new(None);
        let now_q: [[u64; 3]; 64] = std::array::from_fn(|q| [render::QUEUE_LOCK_WAIT_NS[q].load(Ordering::Relaxed), render::QUEUE_LOCK_HOLD_NS[q].load(Ordering::Relaxed), render::QUEUE_WAIT_REG_NS[q].load(Ordering::Relaxed)]);
        if let Some(prev_q) = LAST_Q.lock().replace(now_q) {
            let qs: Vec<String> = (0..64)
                .filter_map(|q| {
                    let d: [u64; 3] = std::array::from_fn(|k| (now_q[q][k] - prev_q[q][k]) / 1_000_000);
                    (d.iter().any(|&v| v > 0)).then(|| format!("{} lock wait {} hold {} wait-reg {}", if q == 0 { "gfx".to_string() } else { format!("q{q}") }, d[0], d[1], d[2]))
                })
                .collect();
            // (Deferred recording: the recorder thread's operations, its time
            // in them, and how often it had to be woken.)
            static LAST_REC: Mutex<(u64, u64, u64)> = Mutex::new((0, 0, 0));
            let r = (rec::OPS.load(Ordering::Relaxed), rec::BUSY_NS.load(Ordering::Relaxed), rec::WAKES.load(Ordering::Relaxed));
            let pr = std::mem::replace(&mut *LAST_REC.lock(), r);
            let mut recorder = if r.0 > pr.0 { format!("; recorder {} ops, busy {} ms, woken {}x", r.0 - pr.0, (r.1 - pr.1) / 1_000_000, r.2 - pr.2) } else { String::new() };
            // (Asynchronous ring copies: done, and waits for them.)
            static LAST_CP: Mutex<(u64, u64, u64)> = Mutex::new((0, 0, 0));
            let k = (copier::COPIES.load(Ordering::Relaxed), copier::WAITS.load(Ordering::Relaxed), copier::WAIT_NS.load(Ordering::Relaxed));
            let pk = std::mem::replace(&mut *LAST_CP.lock(), k);
            if k.0 > pk.0 {
                recorder.push_str(&format!("; ring copies {} by the copy thread, {} waits for them ({} ms)", k.0 - pk.0, k.1 - pk.1, (k.2 - pk.2) / 1_000_000));
            }
            crate::info!("queues/s (ms): {}{recorder}", qs.join("; "));
        }
        let mut fills: Vec<((u64, u64, &'static str), (u64, u64))> = render::VCACHE_FILL_SITES.lock().take().map(|m| m.into_iter().collect()).unwrap_or_default();
        if !fills.is_empty() {
            fills.sort_by_key(|&(_, (_, bytes))| std::cmp::Reverse(bytes));
            let total: u64 = fills.iter().map(|&(_, (_, b))| b).sum();
            let top: Vec<String> = fills.iter().take(8).map(|&((b, s, why), (n, bytes))| format!("{n} x {b:#x}+{s:#x} {why} ({} KiB)", bytes >> 10)).collect();
            crate::info!("mirror copy-ins/s: {} MiB for {} bindings; largest: {}", total >> 20, fills.iter().map(|f| f.1.0).sum::<u64>(), top.join(", "));
        }
        let mut wbs: Vec<((u64, u64, bool), u64)> = render::WRITEBACK_SITES.lock().take().map(|m| m.into_iter().collect()).unwrap_or_default();
        if !wbs.is_empty() {
            wbs.sort_by_key(|&((_, size, _), n)| std::cmp::Reverse(size * n));
            let total: u64 = wbs.iter().map(|&((_, s, _), n)| s * n).sum();
            let top: Vec<String> = wbs.iter().take(8).map(|&((b, s, m), n)| format!("{n} x {b:#x}+{s:#x}{}", if m { " (mirror)" } else { "" })).collect();
            static LAST: AtomicU64 = AtomicU64::new(0);
            static LAST_NS: AtomicU64 = AtomicU64::new(0);
            let sum = render::WB_BYTES_SUM.load(Ordering::Relaxed);
            let done = sum - LAST.swap(sum, Ordering::Relaxed);
            let ns = render::WB_NS_SUM.load(Ordering::Relaxed);
            let took = ns - LAST_NS.swap(ns, Ordering::Relaxed);
            static LAST_R: AtomicU64 = AtomicU64::new(0);
            static LAST_RNS: AtomicU64 = AtomicU64::new(0);
            let (r, rns) = (render::WB_RESETS.load(Ordering::Relaxed), render::WB_RESET_NS.load(Ordering::Relaxed));
            let (resets, reset_ns) = (r - LAST_R.swap(r, Ordering::Relaxed), rns - LAST_RNS.swap(rns, Ordering::Relaxed));
            crate::info!("write-backs/s: {} MiB registered in {} ranges, {} MiB written back in {} ms (write-watch resets: {resets} in {} ms); largest: {}", total >> 20, wbs.iter().map(|u| u.1).sum::<u64>(), done >> 20, took / 1_000_000, reset_ns / 1_000_000, top.join(", "));
        }
    }
    *prev = Some(now);
}
