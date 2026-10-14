//! Asynchronous guest-memory copies into the upload ring (`tune
//! async_copies`): the CPs reserve ring space and queue the copy; the
//! `GPU-copy` thread performs it, so the copies (~1 us each, mostly cache
//! misses on data the title's threads just wrote on other cores, ~75,000 per
//! second in Trico's cave) leave the CPs' critical path.
//!
//! Ordering: a batch is submitted only once its copies are done (`wait`
//! before the submit, see `rec::Rec::submit`); before the CP writes guest
//! memory itself (WRITE_DATA, DMA_DATA, COPY_DATA, DUMP_CONST_RAM: the
//! renderer's `sync_write`), queued copies of that memory are finished first
//! (`wait_overlapping`). Copies read guest memory up to that point later than
//! at record time; what changes it in between is either later CP work (waited
//! for, above), GPU write-backs landing (which a binding forwards on top of
//! its copy anyway) or the title's threads, which do not overwrite data GPU
//! work they submitted still reads.

use parking_lot::{Condvar, Mutex};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

struct Job {
    dst: *mut u8,
    src: u64,
    len: usize,
    seq: u64,
}

// SAFETY: `dst` points into the upload ring, which outlives the process's
// use of the queue; the CP that queued the job does not touch those bytes
// before the job is done.
unsafe impl Send for Job {}

static QUEUE: Mutex<VecDeque<Job>> = Mutex::new(VecDeque::new());
static CV: Condvar = Condvar::new();
static SLEEPING: AtomicBool = AtomicBool::new(false);
/// Sequence numbers of the last queued and the last finished copy.
static QUEUED: AtomicU64 = AtomicU64::new(0);
static DONE: AtomicU64 = AtomicU64::new(0);
/// The copy in progress: guest [lo, hi) and its sequence number (0: none).
static CUR: Mutex<(u64, u64, u64)> = Mutex::new((0, 0, 0));
static DONE_LOCK: Mutex<()> = Mutex::new(());
static DONE_CV: Condvar = Condvar::new();
static WAITERS: AtomicU64 = AtomicU64::new(0);
static STARTED: std::sync::Once = std::sync::Once::new();
/// Copies performed and waits for them (TLG_PERF).
pub static COPIES: AtomicU64 = AtomicU64::new(0);
pub static WAITS: AtomicU64 = AtomicU64::new(0);
pub static WAIT_NS: AtomicU64 = AtomicU64::new(0);

/// Queues a copy of guest [src, src+len) to `dst` (upload ring bytes nothing
/// reads before `wait` returns for the result); returns its sequence number.
pub fn copy(dst: *mut u8, src: u64, len: usize) -> u64 {
    STARTED.call_once(|| {
        crate::thread::spawn_host("GPU-copy", 4 << 20, Box::new(run)).expect("copy thread");
    });
    let seq = {
        let mut q = QUEUE.lock();
        let seq = QUEUED.load(Ordering::Relaxed) + 1;
        QUEUED.store(seq, Ordering::Release);
        q.push_back(Job { dst, src, len, seq });
        seq
    };
    if SLEEPING.load(Ordering::SeqCst) {
        CV.notify_one();
    }
    seq
}

/// Waits until copy `seq` and all before it are done.
pub fn wait(seq: u64) {
    if DONE.load(Ordering::Acquire) >= seq {
        return;
    }
    let t0 = std::time::Instant::now();
    WAITS.fetch_add(1, Ordering::Relaxed);
    WAITERS.fetch_add(1, Ordering::SeqCst);
    if SLEEPING.load(Ordering::SeqCst) {
        let _q = QUEUE.lock();
        CV.notify_one();
    }
    let mut g = DONE_LOCK.lock();
    while DONE.load(Ordering::Acquire) < seq {
        DONE_CV.wait_for(&mut g, std::time::Duration::from_micros(200));
    }
    drop(g);
    WAITERS.fetch_sub(1, Ordering::SeqCst);
    WAIT_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// Before guest [addr, addr+len) is written by the CP: waits for the queued
/// copies that read it.
pub fn wait_overlapping(addr: u64, len: u64) {
    if DONE.load(Ordering::Acquire) >= QUEUED.load(Ordering::Acquire) {
        return;
    }
    let hi = addr + len.max(1);
    let mut need = 0;
    {
        let q = QUEUE.lock();
        for j in q.iter().rev() {
            if j.src < hi && addr < j.src + j.len as u64 {
                need = j.seq;
                break;
            }
        }
    }
    if need == 0 {
        let c = *CUR.lock();
        if c.2 != 0 && c.0 < hi && addr < c.1 {
            need = c.2;
        }
    }
    if need != 0 {
        wait(need);
    }
}

fn run() {
    crate::log::set_thread_name("GPU-copy");
    crate::thread::raise_priority();
    loop {
        let job = {
            let mut q = QUEUE.lock();
            if q.is_empty() {
                drop(q);
                // (Copies arrive within microseconds of each other while a
                // CP records: stay awake briefly.)
                let t0 = std::time::Instant::now();
                while QUEUED.load(Ordering::Acquire) == DONE.load(Ordering::Acquire) && t0.elapsed() < std::time::Duration::from_micros(50) {
                    std::hint::spin_loop();
                }
                q = QUEUE.lock();
                while q.is_empty() {
                    SLEEPING.store(true, Ordering::SeqCst);
                    CV.wait(&mut q);
                    SLEEPING.store(false, Ordering::SeqCst);
                }
            }
            let j = q.pop_front().unwrap();
            *CUR.lock() = (j.src, j.src + j.len as u64, j.seq);
            j
        };
        let dst = unsafe { std::slice::from_raw_parts_mut(job.dst, job.len) };
        super::resources::copy_from_guest(dst, job.src);
        *CUR.lock() = (0, 0, 0);
        COPIES.fetch_add(1, Ordering::Relaxed);
        DONE.store(job.seq, Ordering::Release);
        if WAITERS.load(Ordering::SeqCst) != 0 {
            let _g = DONE_LOCK.lock();
            DONE_CV.notify_all();
        }
    }
}
