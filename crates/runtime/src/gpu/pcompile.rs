//! Pipelines created on worker threads (`tune pipe_async`: the number of
//! workers, 0 = on the CP as before). A CP that needs a pipeline nobody has
//! made yet makes its shader modules and layouts, queues the driver's compile
//! here and records a bind of the pipeline's slot; the recorder thread waits
//! for the slot when it reaches that bind (`rec::Rec::cmd_bind_pipeline_slot`),
//! while the CP goes on with the next commands - their new pipelines then
//! compile at the same time, and the CP's other work overlaps the compile.
//! (First sight of a shader cost the CP 90 ms per graphics and up to 350 ms
//! per compute pipeline at the 31% clock, one after another.)

use ash::vk;
use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A pipeline being made: its result once the worker has it.
pub struct PipeSlot {
    state: Mutex<Option<Result<vk::Pipeline, String>>>,
    cv: Condvar,
}

impl PipeSlot {
    fn new() -> Arc<PipeSlot> {
        Arc::new(PipeSlot { state: Mutex::new(None), cv: Condvar::new() })
    }

    fn finish(&self, r: Result<vk::Pipeline, String>) {
        *self.state.lock() = Some(r);
        self.cv.notify_all();
    }

    /// The result, if the compile has finished.
    pub fn get(&self) -> Option<Result<vk::Pipeline, String>> {
        self.state.lock().clone()
    }

    /// The result, waiting for the compile.
    pub fn wait(&self) -> Result<vk::Pipeline, String> {
        let mut g = self.state.lock();
        loop {
            if let Some(r) = g.as_ref() {
                return r.clone();
            }
            self.cv.wait(&mut g);
        }
    }
}

pub type Job = Box<dyn FnOnce() -> Result<vk::Pipeline, String> + Send>;

struct Pool {
    jobs: Mutex<VecDeque<(Job, Arc<PipeSlot>)>>,
    cv: Condvar,
}

static POOL: Pool = Pool { jobs: Mutex::new(VecDeque::new()), cv: Condvar::new() };
/// Workers started so far.
static STARTED: AtomicUsize = AtomicUsize::new(0);
/// Compiles queued or running.
static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
/// Compiles done here, and the time they took (TLG_PERF, `render`).
pub static DONE: AtomicUsize = AtomicUsize::new(0);
pub static BUSY_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Worker threads to use (0: compile on the CP; `tune pipe_async` -1: the
/// logical processors less 4 - the game's busiest threads -, 2 to 8).
pub fn workers() -> usize {
    match crate::tune::get(crate::tune::PIPE_ASYNC) {
        n if n < 0 => {
            // (Asked at every draw by the prepare thread, `render::ahead_draw`.)
            static AUTO: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
            *AUTO.get_or_init(|| std::thread::available_parallelism().map_or(2, |p| p.get().saturating_sub(4)).clamp(2, 8))
        }
        n => n.min(8) as usize,
    }
}

/// Queues `job` (the driver's pipeline creation, from modules and a layout
/// that stay alive) and returns its slot.
pub fn spawn(job: Job) -> Arc<PipeSlot> {
    let want = workers().max(1);
    while STARTED.load(Ordering::Acquire) < want {
        let n = STARTED.fetch_add(1, Ordering::AcqRel);
        if n >= want {
            STARTED.fetch_sub(1, Ordering::AcqRel);
            break;
        }
        if crate::thread::spawn_host(&format!("GPU-pipe{n}"), 4 << 20, Box::new(worker)).is_err() {
            STARTED.fetch_sub(1, Ordering::AcqRel);
            break;
        }
    }
    let slot = PipeSlot::new();
    IN_FLIGHT.fetch_add(1, Ordering::AcqRel);
    POOL.jobs.lock().push_back((job, slot.clone()));
    POOL.cv.notify_one();
    // (No worker could be started: the job runs here.)
    if STARTED.load(Ordering::Acquire) == 0 {
        run_one();
    }
    slot
}

fn run_one() -> bool {
    let Some((job, slot)) = POOL.jobs.lock().pop_front() else { return false };
    let t0 = std::time::Instant::now();
    let r = job();
    BUSY_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    DONE.fetch_add(1, Ordering::Relaxed);
    if let Err(e) = &r {
        crate::error!("pipeline compile: {e}");
    }
    slot.finish(r);
    IN_FLIGHT.fetch_sub(1, Ordering::AcqRel);
    true
}

fn worker() {
    crate::log::set_thread_name("GPU-pipe");
    loop {
        {
            let mut g = POOL.jobs.lock();
            while g.is_empty() {
                POOL.cv.wait(&mut g);
            }
        }
        run_one();
    }
}

/// Waits (at most `timeout`) until no compile is queued or running: before
/// the pipeline cache is written at exit, so it holds them.
pub fn idle(timeout: std::time::Duration) {
    let t0 = std::time::Instant::now();
    while IN_FLIGHT.load(Ordering::Acquire) != 0 && t0.elapsed() < timeout {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}
