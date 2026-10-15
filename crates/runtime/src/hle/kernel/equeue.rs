//! Event queues (kqueue-like `SceKernelEqueue`).

use super::errno::*;
use super::sce;
use crate::hle::{Registry, cstr};
use crate::{reg, sysv};
use parking_lot::{Condvar, Mutex};
use std::time::{Duration, Instant};

const EQ_MAGIC: u64 = 0x4242_4551_5545_0001;

pub const EVFILT_TIMER: i16 = -7;
pub const EVFILT_USER: i16 = -11;
pub const EVFILT_DISPLAY: i16 = -13;
pub const EVFILT_GRAPHICS: i16 = -14;
pub const EVFILT_HRTIMER: i16 = -15;

pub const EV_ONESHOT: u16 = 0x10;
pub const EV_CLEAR: u16 = 0x20;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Kevent {
    pub ident: u64,
    pub filter: i16,
    pub flags: u16,
    pub fflags: u32,
    pub data: u64,
    pub udata: u64,
}

#[derive(Clone, Debug)]
struct Entry {
    ev: Kevent,
    triggered: bool,
    /// Timer generation (high-resolution timers): a schedule fires only the
    /// entry it was made for, not one deleted and added again since.
    timer_gen: u64,
}

pub struct Equeue {
    magic: u64,
    pub name: String,
    events: Mutex<Vec<Entry>>,
    cv: Condvar,
}

impl Equeue {
    pub fn add(&self, ident: u64, filter: i16, flags: u16, udata: u64) {
        self.add_timed(ident, filter, flags, udata, 0);
    }
    fn add_timed(&self, ident: u64, filter: i16, flags: u16, udata: u64, timer_gen: u64) {
        let mut e = self.events.lock();
        e.retain(|x| !(x.ev.ident == ident && x.ev.filter == filter));
        e.push(Entry { ev: Kevent { ident, filter, flags, fflags: 0, data: 0, udata }, triggered: false, timer_gen });
    }
    /// Fires a high-resolution timer scheduled for generation `timer_gen`.
    fn fire_timer(&self, ident: u64, timer_gen: u64) {
        let mut e = self.events.lock();
        let mut any = false;
        for x in e.iter_mut().filter(|x| x.ev.ident == ident && x.ev.filter == EVFILT_HRTIMER && x.timer_gen == timer_gen) {
            x.triggered = true;
            x.ev.data = 1; // expirations
            any = true;
        }
        drop(e);
        if any {
            self.cv.notify_all();
        }
    }
    pub fn remove(&self, ident: u64, filter: i16) -> bool {
        let mut e = self.events.lock();
        let n = e.len();
        e.retain(|x| !(x.ev.ident == ident && x.ev.filter == filter));
        n != e.len()
    }
    /// Triggers matching events; returns whether any matched.
    pub fn trigger(&self, ident: u64, filter: i16, data: Option<u64>, udata: Option<u64>) -> bool {
        let mut e = self.events.lock();
        let mut any = false;
        for x in e.iter_mut().filter(|x| x.ev.ident == ident && x.ev.filter == filter) {
            x.triggered = true;
            if let Some(d) = data {
                x.ev.data = d;
            }
            if let Some(u) = udata {
                x.ev.udata = u;
            }
            any = true;
        }
        drop(e);
        if any {
            self.cv.notify_all();
        }
        any
    }
    pub fn has(&self, ident: u64, filter: i16) -> bool {
        self.events.lock().iter().any(|x| x.ev.ident == ident && x.ev.filter == filter)
    }
    fn wait(&self, out: &mut [Kevent], deadline: Option<Instant>) -> usize {
        let mut e = self.events.lock();
        loop {
            let n = Self::collect(&mut e, out);
            if n > 0 {
                return n;
            }
            match deadline {
                Some(d) => {
                    if self.cv.wait_until(&mut e, d).timed_out() {
                        // One last check after the timeout.
                        return Self::collect(&mut e, out);
                    }
                }
                None => self.cv.wait(&mut e),
            }
        }
    }
    /// Moves triggered events into `out`; one-shot events are removed once delivered.
    fn collect(e: &mut Vec<Entry>, out: &mut [Kevent]) -> usize {
        let mut n = 0;
        e.retain_mut(|x| {
            if n == out.len() || !x.triggered {
                return true;
            }
            out[n] = x.ev;
            n += 1;
            x.triggered = false;
            x.ev.flags & EV_ONESHOT == 0
        });
        n
    }
}

/// Every queue created (diagnostics: `dump`).
static QUEUES: Mutex<Vec<&'static Equeue>> = Mutex::new(Vec::new());
/// Threads waiting in sceKernelWaitEqueue: (guest thread id, queue handle).
static WAITERS: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// Logs each event queue, its events (triggered or not) and who waits on it.
pub fn dump() {
    let waiters = WAITERS.lock().clone();
    let mut s = String::from("event queues:");
    for q in QUEUES.lock().iter() {
        let h = *q as *const Equeue as u64;
        let w: Vec<u64> = waiters.iter().filter(|(_, qh)| *qh == h).map(|(t, _)| *t).collect();
        s.push_str(&format!("\n  {h:#x} '{}' waiters {w:?}:", q.name));
        if let Some(e) = q.events.try_lock() {
            for x in e.iter() {
                s.push_str(&format!(" [id {:#x} filter {} {}]", x.ev.ident, x.ev.filter, if x.triggered { "TRIGGERED" } else { "idle" }));
            }
        }
    }
    crate::info!("{s}");
}

pub unsafe fn from_handle<'a>(h: u64) -> Option<&'a Equeue> {
    if h == 0 {
        return None;
    }
    let q = unsafe { &*(h as *const Equeue) };
    if q.magic == EQ_MAGIC { Some(q) } else { None }
}

// ---------------------------------------------------------------------------
// High-resolution timers (EVFILT_HRTIMER): one-shot events, fired by one
// host thread at their deadline (coarse condvar wait, then a high-resolution
// sleep and a short spin for the last ~1.5 ms).

struct Timer {
    at: Instant,
    queue: u64,
    ident: u64,
    timer_gen: u64,
}

static TIMERS: Mutex<Vec<Timer>> = Mutex::new(Vec::new());
static TIMER_CV: Condvar = Condvar::new();
static TIMER_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
/// HR timers added, their requested time (us) and the timer thread's final
/// spins (us) - the `cpu` report.
pub static HRT_STATS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

fn schedule_timer(queue: u64, ident: u64, timer_gen: u64, at: Instant) {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        std::thread::Builder::new().name("hrtimer".into()).spawn(timer_thread).expect("hrtimer thread");
    });
    TIMERS.lock().push(Timer { at, queue, ident, timer_gen });
    TIMER_CV.notify_one();
}

fn timer_thread() {
    crate::log::set_thread_name("hrtimer");
    crate::thread::raise_priority();
    let mut g = TIMERS.lock();
    loop {
        let now = Instant::now();
        if g.iter().any(|t| t.at <= now) {
            let (due, rest): (Vec<Timer>, Vec<Timer>) = std::mem::take(&mut *g).into_iter().partition(|t| t.at <= now);
            *g = rest;
            drop(g);
            for t in due {
                if let Some(q) = unsafe { from_handle(t.queue) } {
                    q.fire_timer(t.ident, t.timer_gen);
                }
            }
            g = TIMERS.lock();
            continue;
        }
        match g.iter().map(|t| t.at).min() {
            None => TIMER_CV.wait(&mut g),
            Some(at) => {
                let rem = at - now;
                if rem > Duration::from_micros(1500) {
                    TIMER_CV.wait_for(&mut g, rem - Duration::from_millis(1));
                } else if crate::tune::get(crate::tune::HRT_SPIN) == 0 {
                    drop(g);
                    std::thread::sleep(rem);
                    g = TIMERS.lock();
                } else {
                    drop(g);
                    if rem > Duration::from_micros(300) {
                        std::thread::sleep(rem - Duration::from_micros(200));
                    }
                    let t0 = Instant::now();
                    crate::thread::spin_until(at.saturating_duration_since(t0), || false);
                    HRT_STATS[2].fetch_add(t0.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
                    g = TIMERS.lock();
                }
            }
        }
    }
}

sysv! {
    fn sceKernelCreateEqueue(out: *mut u64, name: *const u8) -> i32 {
        if out.is_null() { return sce(EINVAL); }
        let q: &'static Equeue = Box::leak(Box::new(Equeue { magic: EQ_MAGIC, name: cstr(name).to_string(), events: Mutex::new(Vec::new()), cv: Condvar::new() }));
        QUEUES.lock().push(q);
        *out = q as *const _ as u64;
        crate::debug!("equeue '{}' created {:#x}", q.name, *out);
        0
    }
    fn sceKernelDeleteEqueue(h: u64) -> i32 { if from_handle(h).is_some() { 0 } else { sce(EBADF) } }
    fn sceKernelWaitEqueue(h: u64, ev: *mut Kevent, num: i32, out: *mut i32, timeout: *const u32) -> i32 {
        let Some(q) = from_handle(h) else { return sce(EBADF) };
        if ev.is_null() || num < 1 { return sce(EINVAL); }
        let deadline = if timeout.is_null() { None } else { Some(Instant::now() + Duration::from_micros(*timeout as u64)) };
        let buf = std::slice::from_raw_parts_mut(ev, num as usize);
        let me = crate::thread::current().map_or(0, |t| t.id);
        WAITERS.lock().push((me, h));
        let n = q.wait(buf, deadline);
        {
            let mut w = WAITERS.lock();
            if let Some(i) = w.iter().position(|&(t, qh)| t == me && qh == h) { w.swap_remove(i); }
        }
        if !out.is_null() { *out = n as i32; }
        if n == 0 { sce(ETIMEDOUT) } else { 0 }
    }
    fn sceKernelAddUserEvent(h: u64, id: i32) -> i32 {
        let Some(q) = from_handle(h) else { return sce(EBADF) };
        q.add(id as u64, EVFILT_USER, 0, 0);
        0
    }
    fn sceKernelAddUserEventEdge(h: u64, id: i32) -> i32 {
        let Some(q) = from_handle(h) else { return sce(EBADF) };
        q.add(id as u64, EVFILT_USER, EV_CLEAR, 0);
        0
    }
    fn sceKernelTriggerUserEvent(h: u64, id: i32, udata: u64) -> i32 {
        let Some(q) = from_handle(h) else { return sce(EBADF) };
        if q.trigger(id as u64, EVFILT_USER, None, Some(udata)) { 0 } else { sce(ENOENT) }
    }
    fn sceKernelDeleteUserEvent(h: u64, id: i32) -> i32 {
        let Some(q) = from_handle(h) else { return sce(EBADF) };
        if q.remove(id as u64, EVFILT_USER) { 0 } else { sce(ENOENT) }
    }
    /// One-shot timer firing `ts` (relative timespec) from now.
    fn sceKernelAddHRTimerEvent(h: u64, id: i32, ts: *const i64, udata: u64) -> i32 {
        let Some(q) = from_handle(h) else { return sce(EBADF) };
        if ts.is_null() { return sce(EINVAL); }
        let (sec, nsec) = (*ts, *ts.add(1));
        if sec < 0 || !(0..1_000_000_000).contains(&nsec) { return sce(EINVAL); }
        let timer_gen = TIMER_GEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        HRT_STATS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        HRT_STATS[1].fetch_add(sec as u64 * 1_000_000 + nsec as u64 / 1000, std::sync::atomic::Ordering::Relaxed);
        q.add_timed(id as u64, EVFILT_HRTIMER, EV_ONESHOT, udata, timer_gen);
        schedule_timer(h, id as u64, timer_gen, Instant::now() + Duration::new(sec as u64, nsec as u32));
        0
    }
    fn sceKernelDeleteHRTimerEvent(h: u64, id: i32) -> i32 {
        let Some(q) = from_handle(h) else { return sce(EBADF) };
        if q.remove(id as u64, EVFILT_HRTIMER) { 0 } else { sce(ENOENT) }
    }
    fn sceKernelGetEventId(ev: *const Kevent) -> u64 { (*ev).ident }
    fn sceKernelGetEventFilter(ev: *const Kevent) -> i32 { (*ev).filter as i32 }
    fn sceKernelGetEventData(ev: *const Kevent) -> u64 { (*ev).data }
    fn sceKernelGetEventUserData(ev: *const Kevent) -> u64 { (*ev).udata }
    fn sceKernelGetEventFflags(ev: *const Kevent) -> u32 { (*ev).fflags }
    fn sceKernelGetEventError(ev: *const Kevent) -> i32 { 0 }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libkernel", [
        sceKernelCreateEqueue, sceKernelDeleteEqueue, sceKernelWaitEqueue, sceKernelAddUserEvent,
        sceKernelAddUserEventEdge, sceKernelTriggerUserEvent, sceKernelDeleteUserEvent, sceKernelGetEventId,
        sceKernelGetEventFilter, sceKernelGetEventData, sceKernelGetEventUserData, sceKernelGetEventFflags,
        sceKernelGetEventError, sceKernelAddHRTimerEvent, sceKernelDeleteHRTimerEvent,
    ]);
}
