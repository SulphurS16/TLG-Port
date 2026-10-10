//! Mutexes, condition variables, rwlocks, semaphores and event flags.
//!
//! Guest objects are pointer-sized handles; the guest variable holds a pointer to
//! a host object. Zero (and small constants) are static initializers that are
//! created lazily on first use.

use super::errno::*;
use super::sce;
use crate::hle::{Registry, cstr};
use crate::{reg, sysv};
use parking_lot::{Condvar, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

const MUTEX_MAGIC: u64 = 0x4242_4D55_5445_5801;
const COND_MAGIC: u64 = 0x4242_434F_4E44_5601;
const RW_MAGIC: u64 = 0x4242_5257_4C4B_0001;
const SEMA_MAGIC: u64 = 0x4242_5345_4D41_0001;
const EF_MAGIC: u64 = 0x4242_4546_4C47_0001;
const MATTR_MAGIC: u64 = 0x4242_4D41_5454_0001;

pub const MUTEX_ERRORCHECK: i32 = 1;
pub const MUTEX_RECURSIVE: i32 = 2;
pub const MUTEX_NORMAL: i32 = 3;
pub const MUTEX_ADAPTIVE: i32 = 4;

static INIT_LOCK: Mutex<()> = Mutex::new(());

fn tid() -> u64 {
    crate::thread::current().map(|t| t.id).unwrap_or(u64::MAX)
}

/// Relative timeout in microseconds -> deadline.
fn deadline_us(us: u64) -> Instant {
    Instant::now() + Duration::from_micros(us)
}

/// Absolute CLOCK_REALTIME timespec -> deadline.
unsafe fn deadline_abs(ts: *const i64) -> Option<Instant> {
    if ts.is_null() {
        return None;
    }
    let (sec, nsec) = unsafe { (*ts, *ts.add(1)) };
    let target = Duration::new(sec.max(0) as u64, nsec.clamp(0, 999_999_999) as u32);
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
    Some(Instant::now() + target.saturating_sub(now))
}

// ---------------------------------------------------------------------------
// Mutex

/// The start of every guest-visible sync object (the guest variable points at
/// it): laid out like the FreeBSD/PS4 libkernel structure the title's libc was
/// compiled against, which it reads and writes directly -- e.g. TLG's libc.prx
/// sets `PMUTEX_FLAG_PRIVATE` with `or dword [mutex+0x20], 0x100` right after
/// scePthreadMutexInit (libc+0x5d81c). None of our own state lives here.
#[repr(C)]
pub struct GuestPrefix([u32; 32]);

impl GuestPrefix {
    const fn zero() -> Self {
        GuestPrefix([0; 32])
    }
    /// `struct pthread_mutex`: `m_lock` (struct umutex, 0x20 bytes), then
    /// `m_flags` (+0x20) whose low byte is the mutex type.
    const fn mutex(kind: i32) -> Self {
        let mut p = [0u32; 32];
        p[8] = (kind as u32) & 0xff;
        GuestPrefix(p)
    }
}

/// Guest mutex: an atomic owner word (lock-free when uncontended), a short
/// spin before parking (the engine's job workers take one queue lock in
/// tight loops: parking on every contended acquisition made them spend most
/// of their time in kernel sleeps and wake-ups), then a condition variable.
#[repr(C)]
pub struct GMutex {
    guest: GuestPrefix,
    magic: u64,
    kind: i32,
    name: String,
    /// Owner thread id (0: free).
    owner: AtomicU64,
    /// Recursion count (changed by the owner only).
    count: std::sync::atomic::AtomicU32,
    /// Threads parked (or about to park) on `cv`.
    waiters: std::sync::atomic::AtomicU32,
    park: Mutex<()>,
    cv: Condvar,
}

/// Spins (pause instructions) before a contended lock parks.
const MUTEX_SPINS: u32 = 4000;

#[repr(C)]
struct MutexAttr {
    magic: u64,
    kind: i32,
    protocol: i32,
}

impl GMutex {
    fn new(kind: i32, name: &str) -> &'static GMutex {
        Box::leak(Box::new(GMutex {
            guest: GuestPrefix::mutex(kind),
            magic: MUTEX_MAGIC,
            kind,
            name: name.to_string(),
            owner: AtomicU64::new(0),
            count: std::sync::atomic::AtomicU32::new(0),
            waiters: std::sync::atomic::AtomicU32::new(0),
            park: Mutex::new(()),
            cv: Condvar::new(),
        }))
    }

    fn try_take(&self, me: u64) -> bool {
        self.owner.compare_exchange(0, me, Ordering::SeqCst, Ordering::Relaxed).is_ok()
    }

    /// Acquires for `me` (not already the owner); false on timeout.
    fn acquire(&self, me: u64, deadline: Option<Instant>) -> bool {
        loop {
            for _ in 0..MUTEX_SPINS {
                if self.owner.load(Ordering::Relaxed) == 0 && self.try_take(me) {
                    return true;
                }
                std::hint::spin_loop();
            }
            // Park until the owner releases (its unlock sees the waiter
            // count, or this check sees the release: both are SeqCst).
            self.waiters.fetch_add(1, Ordering::SeqCst);
            let mut g = self.park.lock();
            let parked = Instant::now();
            let mut reported = false;
            while self.owner.load(Ordering::SeqCst) != 0 {
                match deadline {
                    Some(d) => {
                        if self.cv.wait_until(&mut g, d).timed_out() && self.owner.load(Ordering::SeqCst) != 0 {
                            drop(g);
                            self.waiters.fetch_sub(1, Ordering::SeqCst);
                            return false;
                        }
                    }
                    None => {
                        // Diagnostic: a lock held for seconds names its owner once.
                        if self.cv.wait_for(&mut g, Duration::from_secs(5)).timed_out() && !reported {
                            reported = true;
                            let owner = self.owner.load(Ordering::SeqCst);
                            let raw = unsafe { std::slice::from_raw_parts(self as *const Self as *const u64, std::mem::size_of::<Self>() / 8) };
                            crate::warn!(
                                "mutex '{}' ({:p}, kind {}) still held by thread {} (count {}) after {:.1} s; waiter thread {} at {}; owner at +{:#x}, object {:x?}",
                                self.name,
                                self,
                                self.kind,
                                if owner == u64::MAX { "host".to_string() } else { owner.to_string() },
                                self.count.load(Ordering::Relaxed),
                                parked.elapsed().as_secs_f64(),
                                me,
                                crate::loader::describe_addr(crate::tls::hle_return_addr()),
                                std::ptr::addr_of!(self.owner) as usize - self as *const Self as usize,
                                raw
                            );
                        }
                    }
                }
            }
            drop(g);
            self.waiters.fetch_sub(1, Ordering::SeqCst);
            if self.try_take(me) {
                return true;
            }
        }
    }

    /// Releases ownership and wakes a parked waiter.
    fn release(&self) {
        self.owner.store(0, Ordering::SeqCst);
        if self.waiters.load(Ordering::SeqCst) != 0 {
            let _g = self.park.lock();
            self.cv.notify_one();
        }
    }

    pub fn lock(&self, deadline: Option<Instant>, try_only: bool) -> i32 {
        let me = tid();
        if self.owner.load(Ordering::Relaxed) == me {
            return match self.kind {
                MUTEX_RECURSIVE => {
                    self.count.fetch_add(1, Ordering::Relaxed);
                    0
                }
                MUTEX_ERRORCHECK => EDEADLK,
                _ => {
                    if try_only {
                        EBUSY
                    } else {
                        crate::error!("self-deadlock on non-recursive mutex '{}'", self.name);
                        EDEADLK
                    }
                }
            };
        }
        if self.try_take(me) {
            self.count.store(1, Ordering::Relaxed);
            return 0;
        }
        if try_only {
            return EBUSY;
        }
        if !self.acquire(me, deadline) {
            return ETIMEDOUT;
        }
        self.count.store(1, Ordering::Relaxed);
        0
    }

    pub fn unlock(&self) -> i32 {
        let me = tid();
        if self.owner.load(Ordering::Relaxed) != me {
            return EPERM;
        }
        if self.count.fetch_sub(1, Ordering::Relaxed) > 1 {
            return 0;
        }
        self.release();
        0
    }

    /// Fully releases the mutex (for condvar waits); returns the recursion count.
    fn release_all(&self) -> Result<u32, i32> {
        let me = tid();
        if self.owner.load(Ordering::Relaxed) != me {
            return Err(EPERM);
        }
        let n = self.count.swap(0, Ordering::Relaxed);
        self.release();
        Ok(n)
    }

    fn reacquire(&self, count: u32) {
        let me = tid();
        if !self.try_take(me) {
            self.acquire(me, None);
        }
        self.count.store(count, Ordering::Relaxed);
    }
}

unsafe fn mutex_from(slot: *mut u64) -> Result<&'static GMutex, i32> {
    if slot.is_null() {
        return Err(EINVAL);
    }
    let a = unsafe { AtomicU64::from_ptr(slot) };
    let v = a.load(Ordering::Acquire);
    if v > 0x10 {
        let m = unsafe { &*(v as *const GMutex) };
        return if m.magic == MUTEX_MAGIC { Ok(m) } else { Err(EINVAL) };
    }
    let _g = INIT_LOCK.lock();
    let v = a.load(Ordering::Acquire);
    if v > 0x10 {
        return Ok(unsafe { &*(v as *const GMutex) });
    }
    let kind = match v {
        0 => MUTEX_ERRORCHECK,
        1 => MUTEX_ADAPTIVE,
        2 => MUTEX_RECURSIVE,
        3 => MUTEX_NORMAL,
        _ => MUTEX_ERRORCHECK,
    };
    let m = GMutex::new(kind, "static");
    a.store(m as *const _ as u64, Ordering::Release);
    Ok(m)
}

unsafe fn mutex_init(slot: *mut u64, attr: *mut u64, name: &str) -> i32 {
    if slot.is_null() {
        return EINVAL;
    }
    let mut kind = MUTEX_ERRORCHECK;
    if !attr.is_null() {
        let p = unsafe { *attr } as *const MutexAttr;
        if !p.is_null() && unsafe { (*p).magic } == MATTR_MAGIC {
            kind = unsafe { (*p).kind };
        }
    }
    let m = GMutex::new(kind, name);
    unsafe { *slot = m as *const _ as u64 };
    // Debug: TLG_WATCH_MUTEX=<slot address> watches that mutex's owner word (with --trace).
    if std::env::var("TLG_WATCH_MUTEX").ok().and_then(|v| u64::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok()) == Some(slot as u64) {
        crate::hle::set_watch_word(std::ptr::addr_of!(m.owner) as u64, name);
    }
    0
}

unsafe fn mattr_init(slot: *mut u64) -> i32 {
    if slot.is_null() {
        return EINVAL;
    }
    let a = Box::into_raw(Box::new(MutexAttr { magic: MATTR_MAGIC, kind: MUTEX_ERRORCHECK, protocol: 0 }));
    unsafe { *slot = a as u64 };
    0
}

unsafe fn mattr_ref<'a>(slot: *mut u64) -> Option<&'a mut MutexAttr> {
    if slot.is_null() {
        return None;
    }
    let p = unsafe { *slot } as *mut MutexAttr;
    if p.is_null() || unsafe { (*p).magic } != MATTR_MAGIC {
        return None;
    }
    Some(unsafe { &mut *p })
}

// ---------------------------------------------------------------------------
// Condition variable

#[repr(C)]
pub struct GCond {
    guest: GuestPrefix,
    magic: u64,
    seq: Mutex<u64>,
    cv: Condvar,
}

unsafe fn cond_from(slot: *mut u64) -> Result<&'static GCond, i32> {
    if slot.is_null() {
        return Err(EINVAL);
    }
    let a = unsafe { AtomicU64::from_ptr(slot) };
    let v = a.load(Ordering::Acquire);
    if v > 0x10 {
        let c = unsafe { &*(v as *const GCond) };
        return if c.magic == COND_MAGIC { Ok(c) } else { Err(EINVAL) };
    }
    let _g = INIT_LOCK.lock();
    let v = a.load(Ordering::Acquire);
    if v > 0x10 {
        return Ok(unsafe { &*(v as *const GCond) });
    }
    let c: &'static GCond = Box::leak(Box::new(GCond { guest: GuestPrefix::zero(), magic: COND_MAGIC, seq: Mutex::new(0), cv: Condvar::new() }));
    a.store(c as *const _ as u64, Ordering::Release);
    Ok(c)
}

impl GCond {
    fn wait(&self, m: &GMutex, deadline: Option<Instant>) -> i32 {
        let mut g = self.seq.lock();
        let start = *g;
        let count = match m.release_all() {
            Ok(n) => n,
            Err(e) => return e,
        };
        let mut timed_out = false;
        loop {
            match deadline {
                Some(d) => {
                    if self.cv.wait_until(&mut g, d).timed_out() {
                        timed_out = *g == start;
                        break;
                    }
                }
                None => self.cv.wait(&mut g),
            }
            if *g != start {
                break;
            }
        }
        drop(g);
        m.reacquire(count);
        if timed_out { ETIMEDOUT } else { 0 }
    }
    fn signal(&self) {
        let mut g = self.seq.lock();
        *g = g.wrapping_add(1);
        self.cv.notify_one();
    }
    fn broadcast(&self) {
        let mut g = self.seq.lock();
        *g = g.wrapping_add(1);
        self.cv.notify_all();
    }
}

// ---------------------------------------------------------------------------
// Read/write lock

#[repr(C)]
pub struct GRwLock {
    guest: GuestPrefix,
    magic: u64,
    state: Mutex<(u32, u64)>, // (readers, writer tid)
    cv: Condvar,
}

unsafe fn rw_from(slot: *mut u64) -> Result<&'static GRwLock, i32> {
    if slot.is_null() {
        return Err(EINVAL);
    }
    let a = unsafe { AtomicU64::from_ptr(slot) };
    let v = a.load(Ordering::Acquire);
    if v > 0x10 {
        let c = unsafe { &*(v as *const GRwLock) };
        return if c.magic == RW_MAGIC { Ok(c) } else { Err(EINVAL) };
    }
    let _g = INIT_LOCK.lock();
    let v = a.load(Ordering::Acquire);
    if v > 0x10 {
        return Ok(unsafe { &*(v as *const GRwLock) });
    }
    let c: &'static GRwLock = Box::leak(Box::new(GRwLock { guest: GuestPrefix::zero(), magic: RW_MAGIC, state: Mutex::new((0, 0)), cv: Condvar::new() }));
    a.store(c as *const _ as u64, Ordering::Release);
    Ok(c)
}

impl GRwLock {
    fn rdlock(&self, deadline: Option<Instant>, try_only: bool) -> i32 {
        let mut s = self.state.lock();
        while s.1 != 0 {
            if s.1 == tid() {
                return EDEADLK;
            }
            if try_only {
                return EBUSY;
            }
            match deadline {
                Some(d) => {
                    if self.cv.wait_until(&mut s, d).timed_out() && s.1 != 0 {
                        return ETIMEDOUT;
                    }
                }
                None => self.cv.wait(&mut s),
            }
        }
        s.0 += 1;
        0
    }
    fn wrlock(&self, deadline: Option<Instant>, try_only: bool) -> i32 {
        let me = tid();
        let mut s = self.state.lock();
        if s.1 == me {
            return EDEADLK;
        }
        while s.0 != 0 || s.1 != 0 {
            if try_only {
                return EBUSY;
            }
            match deadline {
                Some(d) => {
                    if self.cv.wait_until(&mut s, d).timed_out() && (s.0 != 0 || s.1 != 0) {
                        return ETIMEDOUT;
                    }
                }
                None => self.cv.wait(&mut s),
            }
        }
        s.1 = me;
        0
    }
    fn unlock(&self) -> i32 {
        let mut s = self.state.lock();
        if s.1 != 0 {
            if s.1 != tid() {
                return EPERM;
            }
            s.1 = 0;
        } else if s.0 > 0 {
            s.0 -= 1;
        } else {
            return EPERM;
        }
        drop(s);
        self.cv.notify_all();
        0
    }
}

// ---------------------------------------------------------------------------
// Semaphore

pub struct GSema {
    magic: u64,
    name: String,
    count: Mutex<i32>,
    max: i32,
    cv: Condvar,
}

unsafe fn sema_from<'a>(h: u64) -> Option<&'a GSema> {
    if h == 0 {
        return None;
    }
    let s = unsafe { &*(h as *const GSema) };
    if s.magic == SEMA_MAGIC { Some(s) } else { None }
}

// ---------------------------------------------------------------------------
// Event flag

/// Event flag with the kernel's release semantics: waiters are released (and
/// their clear mode applied) when a set satisfies them, in queue order, not
/// when their threads next run, so a set quickly followed by a clear still
/// wakes them.
pub struct GEventFlag {
    magic: u64,
    name: String,
    attr: u32,
    st: Mutex<EfState>,
    cv: Condvar,
}

struct EfState {
    bits: u64,
    waiters: Vec<EfWaiter>,
    next_id: u64,
}

struct EfWaiter {
    id: u64,
    thread: u64,
    pat: u64,
    mode: u32,
    /// Set on release: (0 or an error, the bits when released).
    outcome: Option<(i32, u64)>,
}

const EVF_ATTR_SINGLE: u32 = 0x10;
const EVF_WAITMODE_AND: u32 = 0x01;
const EVF_WAITMODE_OR: u32 = 0x02;
const EVF_WAITMODE_CLEAR_ALL: u32 = 0x10;
const EVF_WAITMODE_CLEAR_PAT: u32 = 0x20;

/// Every event flag created (diagnostics: `dump_event_flags`).
static FLAGS: Mutex<Vec<&'static GEventFlag>> = Mutex::new(Vec::new());

unsafe fn ef_from<'a>(h: u64) -> Option<&'a GEventFlag> {
    if h == 0 {
        return None;
    }
    let s = unsafe { &*(h as *const GEventFlag) };
    if s.magic == EF_MAGIC { Some(s) } else { None }
}

fn ef_satisfied(bits: u64, pat: u64, mode: u32) -> bool {
    if mode & EVF_WAITMODE_AND != 0 { bits & pat == pat } else { bits & pat != 0 }
}

fn ef_clear(bits: &mut u64, pat: u64, mode: u32) {
    if mode & EVF_WAITMODE_CLEAR_ALL != 0 {
        *bits = 0;
    } else if mode & EVF_WAITMODE_CLEAR_PAT != 0 {
        *bits &= !pat;
    }
}

impl GEventFlag {
    /// Releases the waiters the current bits satisfy, in queue order.
    fn release(&self, s: &mut EfState) {
        let mut any = false;
        for w in s.waiters.iter_mut().filter(|w| w.outcome.is_none()) {
            if ef_satisfied(s.bits, w.pat, w.mode) {
                w.outcome = Some((0, s.bits));
                ef_clear(&mut s.bits, w.pat, w.mode);
                any = true;
            }
        }
        if any {
            self.cv.notify_all();
        }
    }
    /// Releases every waiter with `err`; returns how many there were.
    fn release_all(&self, s: &mut EfState, err: i32) -> i32 {
        let mut n = 0;
        let bits = s.bits;
        for w in s.waiters.iter_mut().filter(|w| w.outcome.is_none()) {
            w.outcome = Some((err, bits));
            n += 1;
        }
        self.cv.notify_all();
        n
    }
}

/// Logs each event flag that has waiters: bits and what each waiter needs.
pub fn dump_event_flags() {
    let mut out = String::from("event flags with waiters:");
    for f in FLAGS.lock().iter() {
        let Some(s) = f.st.try_lock() else {
            out.push_str(&format!("\n  {:#x} '{}': (locked)", *f as *const GEventFlag as u64, f.name));
            continue;
        };
        if s.waiters.is_empty() {
            continue;
        }
        out.push_str(&format!("\n  {:#x} '{}' bits {:#x}:", *f as *const GEventFlag as u64, f.name, s.bits));
        for w in &s.waiters {
            out.push_str(&format!(" [thread {} {} {:#x}{}]", w.thread, if w.mode & EVF_WAITMODE_AND != 0 { "all" } else { "any" }, w.pat, if w.outcome.is_some() { " released" } else { "" }));
        }
    }
    crate::info!("{out}");
}

fn sce_ret(e: i32) -> i32 {
    if e == 0 { 0 } else { sce(e) }
}

sysv! {
    fn scePthreadMutexattrInit(attr: *mut u64) -> i32 { sce_ret(mattr_init(attr)) }
    fn scePthreadMutexattrDestroy(attr: *mut u64) -> i32 {
        if let Some(a) = mattr_ref(attr) { a.magic = 0; drop(Box::from_raw(a as *mut MutexAttr)); *attr = 0; 0 } else { sce(EINVAL) }
    }
    fn scePthreadMutexattrSettype(attr: *mut u64, kind: i32) -> i32 {
        match mattr_ref(attr) { Some(a) => { a.kind = kind; 0 } None => sce(EINVAL) }
    }
    fn scePthreadMutexattrGettype(attr: *mut u64, kind: *mut i32) -> i32 {
        match mattr_ref(attr) { Some(a) => { *kind = a.kind; 0 } None => sce(EINVAL) }
    }
    fn scePthreadMutexattrSetprotocol(attr: *mut u64, p: i32) -> i32 {
        match mattr_ref(attr) { Some(a) => { a.protocol = p; 0 } None => sce(EINVAL) }
    }
    fn scePthreadMutexattrSetpshared(attr: *mut u64, p: i32) -> i32 {
        match mattr_ref(attr) { Some(_) => 0, None => sce(EINVAL) }
    }
    fn scePthreadMutexInit(m: *mut u64, attr: *mut u64, name: *const u8) -> i32 {
        sce_ret(mutex_init(m, attr, if name.is_null() { "mutex" } else { cstr(name) }))
    }
    fn scePthreadMutexDestroy(m: *mut u64) -> i32 {
        if m.is_null() { return sce(EINVAL); }
        *m = 0;
        0
    }
    fn scePthreadMutexLock(m: *mut u64) -> i32 {
        match mutex_from(m) { Ok(m) => sce_ret(m.lock(None, false)), Err(e) => sce(e) }
    }
    fn scePthreadMutexTrylock(m: *mut u64) -> i32 {
        match mutex_from(m) { Ok(m) => sce_ret(m.lock(None, true)), Err(e) => sce(e) }
    }
    fn scePthreadMutexTimedlock(m: *mut u64, usec: u32) -> i32 {
        match mutex_from(m) { Ok(m) => sce_ret(m.lock(Some(deadline_us(usec as u64)), false)), Err(e) => sce(e) }
    }
    fn scePthreadMutexUnlock(m: *mut u64) -> i32 {
        match mutex_from(m) { Ok(m) => sce_ret(m.unlock()), Err(e) => sce(e) }
    }
    fn scePthreadCondattrInit(attr: *mut u64) -> i32 { if !attr.is_null() { *attr = 1; } 0 }
    fn scePthreadCondattrDestroy(attr: *mut u64) -> i32 { 0 }
    fn scePthreadCondInit(c: *mut u64, attr: *mut u64, name: *const u8) -> i32 {
        if c.is_null() { return sce(EINVAL); }
        *c = 0;
        match cond_from(c) { Ok(_) => 0, Err(e) => sce(e) }
    }
    fn scePthreadCondDestroy(c: *mut u64) -> i32 { if !c.is_null() { *c = 0; } 0 }
    fn scePthreadCondWait(c: *mut u64, m: *mut u64) -> i32 {
        match (cond_from(c), mutex_from(m)) { (Ok(c), Ok(m)) => sce_ret(c.wait(m, None)), _ => sce(EINVAL) }
    }
    fn scePthreadCondTimedwait(c: *mut u64, m: *mut u64, usec: u32) -> i32 {
        match (cond_from(c), mutex_from(m)) { (Ok(c), Ok(m)) => sce_ret(c.wait(m, Some(deadline_us(usec as u64)))), _ => sce(EINVAL) }
    }
    fn scePthreadCondSignal(c: *mut u64) -> i32 { match cond_from(c) { Ok(c) => { c.signal(); 0 } Err(e) => sce(e) } }
    fn scePthreadCondBroadcast(c: *mut u64) -> i32 { match cond_from(c) { Ok(c) => { c.broadcast(); 0 } Err(e) => sce(e) } }
    fn scePthreadRwlockattrInit(attr: *mut u64) -> i32 { if !attr.is_null() { *attr = 1; } 0 }
    fn scePthreadRwlockattrDestroy(attr: *mut u64) -> i32 { 0 }
    fn scePthreadRwlockInit(rw: *mut u64, attr: *mut u64, name: *const u8) -> i32 {
        if rw.is_null() { return sce(EINVAL); }
        *rw = 0;
        match rw_from(rw) { Ok(_) => 0, Err(e) => sce(e) }
    }
    fn scePthreadRwlockDestroy(rw: *mut u64) -> i32 { if !rw.is_null() { *rw = 0; } 0 }
    fn scePthreadRwlockRdlock(rw: *mut u64) -> i32 { match rw_from(rw) { Ok(l) => sce_ret(l.rdlock(None, false)), Err(e) => sce(e) } }
    fn scePthreadRwlockWrlock(rw: *mut u64) -> i32 { match rw_from(rw) { Ok(l) => sce_ret(l.wrlock(None, false)), Err(e) => sce(e) } }
    fn scePthreadRwlockTryrdlock(rw: *mut u64) -> i32 { match rw_from(rw) { Ok(l) => sce_ret(l.rdlock(None, true)), Err(e) => sce(e) } }
    fn scePthreadRwlockTrywrlock(rw: *mut u64) -> i32 { match rw_from(rw) { Ok(l) => sce_ret(l.wrlock(None, true)), Err(e) => sce(e) } }
    fn scePthreadRwlockTimedrdlock(rw: *mut u64, usec: u32) -> i32 {
        match rw_from(rw) { Ok(l) => sce_ret(l.rdlock(Some(deadline_us(usec as u64)), false)), Err(e) => sce(e) }
    }
    fn scePthreadRwlockTimedwrlock(rw: *mut u64, usec: u32) -> i32 {
        match rw_from(rw) { Ok(l) => sce_ret(l.wrlock(Some(deadline_us(usec as u64)), false)), Err(e) => sce(e) }
    }
    fn scePthreadRwlockUnlock(rw: *mut u64) -> i32 { match rw_from(rw) { Ok(l) => sce_ret(l.unlock()), Err(e) => sce(e) } }

    // POSIX
    fn pthread_mutexattr_init(attr: *mut u64) -> i32 { mattr_init(attr) }
    fn pthread_mutexattr_destroy(attr: *mut u64) -> i32 {
        if let Some(a) = mattr_ref(attr) { a.magic = 0; drop(Box::from_raw(a as *mut MutexAttr)); *attr = 0; 0 } else { EINVAL }
    }
    fn pthread_mutexattr_settype(attr: *mut u64, kind: i32) -> i32 {
        match mattr_ref(attr) { Some(a) => { a.kind = kind; 0 } None => EINVAL }
    }
    fn pthread_mutexattr_setprotocol(attr: *mut u64, p: i32) -> i32 {
        match mattr_ref(attr) { Some(a) => { a.protocol = p; 0 } None => EINVAL }
    }
    fn pthread_mutex_init(m: *mut u64, attr: *mut u64) -> i32 { mutex_init(m, attr, "pmutex") }
    fn pthread_mutex_destroy(m: *mut u64) -> i32 { if m.is_null() { EINVAL } else { *m = 0; 0 } }
    fn pthread_mutex_lock(m: *mut u64) -> i32 { match mutex_from(m) { Ok(m) => m.lock(None, false), Err(e) => e } }
    fn pthread_mutex_trylock(m: *mut u64) -> i32 { match mutex_from(m) { Ok(m) => m.lock(None, true), Err(e) => e } }
    fn pthread_mutex_timedlock(m: *mut u64, ts: *const i64) -> i32 {
        match mutex_from(m) { Ok(m) => m.lock(deadline_abs(ts), false), Err(e) => e }
    }
    fn pthread_mutex_unlock(m: *mut u64) -> i32 { match mutex_from(m) { Ok(m) => m.unlock(), Err(e) => e } }
    fn pthread_cond_init(c: *mut u64, attr: *mut u64) -> i32 {
        if c.is_null() { return EINVAL; }
        *c = 0;
        match cond_from(c) { Ok(_) => 0, Err(e) => e }
    }
    fn pthread_cond_destroy(c: *mut u64) -> i32 { if !c.is_null() { *c = 0; } 0 }
    fn pthread_cond_wait(c: *mut u64, m: *mut u64) -> i32 {
        match (cond_from(c), mutex_from(m)) { (Ok(c), Ok(m)) => c.wait(m, None), _ => EINVAL }
    }
    fn pthread_cond_timedwait(c: *mut u64, m: *mut u64, ts: *const i64) -> i32 {
        match (cond_from(c), mutex_from(m)) { (Ok(c), Ok(m)) => c.wait(m, deadline_abs(ts)), _ => EINVAL }
    }
    fn pthread_cond_signal(c: *mut u64) -> i32 { match cond_from(c) { Ok(c) => { c.signal(); 0 } Err(e) => e } }
    fn pthread_cond_broadcast(c: *mut u64) -> i32 { match cond_from(c) { Ok(c) => { c.broadcast(); 0 } Err(e) => e } }
    fn pthread_rwlock_init(rw: *mut u64, attr: *mut u64) -> i32 {
        if rw.is_null() { return EINVAL; }
        *rw = 0;
        match rw_from(rw) { Ok(_) => 0, Err(e) => e }
    }
    fn pthread_rwlock_destroy(rw: *mut u64) -> i32 { if !rw.is_null() { *rw = 0; } 0 }
    fn pthread_rwlock_rdlock(rw: *mut u64) -> i32 { match rw_from(rw) { Ok(l) => l.rdlock(None, false), Err(e) => e } }
    fn pthread_rwlock_wrlock(rw: *mut u64) -> i32 { match rw_from(rw) { Ok(l) => l.wrlock(None, false), Err(e) => e } }
    fn pthread_rwlock_unlock(rw: *mut u64) -> i32 { match rw_from(rw) { Ok(l) => l.unlock(), Err(e) => e } }

    // Semaphores
    fn sceKernelCreateSema(out: *mut u64, name: *const u8, attr: u32, init: i32, max: i32, opt: u64) -> i32 {
        if out.is_null() || init < 0 || max <= 0 || init > max { return sce(EINVAL); }
        let s: &'static GSema = Box::leak(Box::new(GSema { magic: SEMA_MAGIC, name: cstr(name).to_string(), count: Mutex::new(init), max, cv: Condvar::new() }));
        *out = s as *const _ as u64;
        0
    }
    fn sceKernelDeleteSema(h: u64) -> i32 { match sema_from(h) { Some(s) => { s.cv.notify_all(); 0 } None => sce(ESRCH) } }
    fn sceKernelWaitSema(h: u64, need: i32, timeout: *mut u32) -> i32 {
        let Some(s) = sema_from(h) else { return sce(ESRCH) };
        if need < 1 || need > s.max { return sce(EINVAL); }
        let start = Instant::now();
        let deadline = if timeout.is_null() { None } else { Some(deadline_us(*timeout as u64)) };
        let mut c = s.count.lock();
        while *c < need {
            match deadline {
                Some(d) => {
                    if s.cv.wait_until(&mut c, d).timed_out() && *c < need {
                        *timeout = 0;
                        return sce(ETIMEDOUT);
                    }
                }
                None => s.cv.wait(&mut c),
            }
        }
        *c -= need;
        if !timeout.is_null() {
            let el = start.elapsed().as_micros() as u32;
            *timeout = (*timeout).saturating_sub(el);
        }
        0
    }
    fn sceKernelPollSema(h: u64, need: i32) -> i32 {
        let Some(s) = sema_from(h) else { return sce(ESRCH) };
        let mut c = s.count.lock();
        if *c < need { return sce(EBUSY); }
        *c -= need;
        0
    }
    fn sceKernelSignalSema(h: u64, n: i32) -> i32 {
        let Some(s) = sema_from(h) else { return sce(ESRCH) };
        let mut c = s.count.lock();
        if *c + n > s.max { return sce(EINVAL); }
        *c += n;
        s.cv.notify_all();
        0
    }
    fn sceKernelCancelSema(h: u64, count: i32, waiters: *mut i32) -> i32 {
        let Some(s) = sema_from(h) else { return sce(ESRCH) };
        if !waiters.is_null() { *waiters = 0; }
        if count >= 0 { *s.count.lock() = count; }
        s.cv.notify_all();
        0
    }

    // Event flags
    fn sceKernelCreateEventFlag(out: *mut u64, name: *const u8, attr: u32, init: u64, opt: u64) -> i32 {
        if out.is_null() { return sce(EINVAL); }
        let st = Mutex::new(EfState { bits: init, waiters: Vec::new(), next_id: 0 });
        let e: &'static GEventFlag = Box::leak(Box::new(GEventFlag { magic: EF_MAGIC, name: cstr(name).to_string(), attr, st, cv: Condvar::new() }));
        FLAGS.lock().push(e);
        *out = e as *const _ as u64;
        0
    }
    fn sceKernelDeleteEventFlag(h: u64) -> i32 {
        let Some(e) = ef_from(h) else { return sce(ESRCH) };
        let mut s = e.st.lock();
        e.release_all(&mut s, EACCES);
        0
    }
    fn sceKernelSetEventFlag(h: u64, bits: u64) -> i32 {
        let Some(e) = ef_from(h) else { return sce(ESRCH) };
        let mut s = e.st.lock();
        s.bits |= bits;
        e.release(&mut s);
        0
    }
    fn sceKernelClearEventFlag(h: u64, bits: u64) -> i32 {
        let Some(e) = ef_from(h) else { return sce(ESRCH) };
        e.st.lock().bits &= bits;
        0
    }
    fn sceKernelPollEventFlag(h: u64, pat: u64, mode: u32, result: *mut u64) -> i32 {
        let Some(e) = ef_from(h) else { return sce(ESRCH) };
        if pat == 0 || mode & (EVF_WAITMODE_AND | EVF_WAITMODE_OR) == 0 { return sce(EINVAL); }
        let mut s = e.st.lock();
        if !result.is_null() { *result = s.bits; }
        if !ef_satisfied(s.bits, pat, mode) { return sce(EBUSY); }
        ef_clear(&mut s.bits, pat, mode);
        0
    }
    fn sceKernelWaitEventFlag(h: u64, pat: u64, mode: u32, result: *mut u64, timeout: *mut u32) -> i32 {
        let Some(e) = ef_from(h) else { return sce(ESRCH) };
        if pat == 0 || mode & (EVF_WAITMODE_AND | EVF_WAITMODE_OR) == 0 { return sce(EINVAL); }
        let start = Instant::now();
        let deadline = if timeout.is_null() { None } else { Some(deadline_us(*timeout as u64)) };
        let mut s = e.st.lock();
        if ef_satisfied(s.bits, pat, mode) {
            if !result.is_null() { *result = s.bits; }
            ef_clear(&mut s.bits, pat, mode);
            return 0;
        }
        if e.attr & EVF_ATTR_SINGLE != 0 && s.waiters.iter().any(|w| w.outcome.is_none()) { return sce(EPERM); }
        let id = s.next_id;
        s.next_id += 1;
        s.waiters.push(EfWaiter { id, thread: tid(), pat, mode, outcome: None });
        let (err, bits) = loop {
            let i = s.waiters.iter().position(|w| w.id == id).unwrap();
            if let Some(o) = s.waiters[i].outcome {
                s.waiters.remove(i);
                break o;
            }
            match deadline {
                Some(d) => {
                    if e.cv.wait_until(&mut s, d).timed_out() {
                        let i = s.waiters.iter().position(|w| w.id == id).unwrap();
                        if let Some(o) = s.waiters[i].outcome {
                            s.waiters.remove(i);
                            break o;
                        }
                        s.waiters.remove(i);
                        break (ETIMEDOUT, s.bits);
                    }
                }
                None => e.cv.wait(&mut s),
            }
        };
        if !result.is_null() { *result = bits; }
        if !timeout.is_null() {
            *timeout = if err == ETIMEDOUT { 0 } else { (*timeout).saturating_sub(start.elapsed().as_micros() as u32) };
        }
        if err == 0 { 0 } else { sce(err) }
    }
    fn sceKernelCancelEventFlag(h: u64, set: u64, waiters: *mut i32) -> i32 {
        let Some(e) = ef_from(h) else { return sce(ESRCH) };
        let mut s = e.st.lock();
        s.bits = set;
        let n = e.release_all(&mut s, ECANCELED);
        if !waiters.is_null() { *waiters = n; }
        0
    }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libkernel", [
        scePthreadMutexattrInit, scePthreadMutexattrDestroy, scePthreadMutexattrSettype, scePthreadMutexattrGettype,
        scePthreadMutexattrSetprotocol, scePthreadMutexattrSetpshared, scePthreadMutexInit, scePthreadMutexDestroy,
        scePthreadMutexLock, scePthreadMutexTrylock, scePthreadMutexTimedlock, scePthreadMutexUnlock,
        scePthreadCondattrInit, scePthreadCondattrDestroy, scePthreadCondInit, scePthreadCondDestroy,
        scePthreadCondWait, scePthreadCondTimedwait, scePthreadCondSignal, scePthreadCondBroadcast,
        scePthreadRwlockattrInit, scePthreadRwlockattrDestroy,
        scePthreadRwlockInit, scePthreadRwlockDestroy, scePthreadRwlockRdlock, scePthreadRwlockWrlock,
        scePthreadRwlockTryrdlock, scePthreadRwlockTrywrlock, scePthreadRwlockTimedrdlock,
        scePthreadRwlockTimedwrlock, scePthreadRwlockUnlock, sceKernelCreateSema, sceKernelDeleteSema,
        sceKernelWaitSema, sceKernelPollSema, sceKernelSignalSema, sceKernelCancelSema, sceKernelCreateEventFlag,
        sceKernelDeleteEventFlag, sceKernelSetEventFlag, sceKernelClearEventFlag, sceKernelPollEventFlag,
        sceKernelWaitEventFlag, sceKernelCancelEventFlag,
    ]);
    reg!(r, "libScePosix", [
        pthread_mutexattr_init, pthread_mutexattr_destroy, pthread_mutexattr_settype, pthread_mutexattr_setprotocol,
        pthread_mutex_init, pthread_mutex_destroy, pthread_mutex_lock, pthread_mutex_trylock, pthread_mutex_timedlock,
        pthread_mutex_unlock, pthread_cond_init, pthread_cond_destroy, pthread_cond_wait, pthread_cond_timedwait,
        pthread_cond_signal, pthread_cond_broadcast, pthread_rwlock_init, pthread_rwlock_destroy,
        pthread_rwlock_rdlock, pthread_rwlock_wrlock, pthread_rwlock_unlock,
    ]);
}
