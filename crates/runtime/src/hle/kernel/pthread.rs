//! Thread creation/lifetime, attributes, keys and once-control.

use super::errno::*;
use super::sce;
use crate::hle::{Registry, cstr};
use crate::thread::{self, GuestThread, PTHREAD_KEYS_MAX};
use crate::{reg, sysv};
use parking_lot::{Condvar, Mutex};
use std::sync::atomic::{AtomicI32, Ordering};

pub const PRIO_DEFAULT: i32 = 700;
pub const DEFAULT_STACK: u64 = 2 << 20;
const ATTR_MAGIC: u64 = 0x4242_4154_5452_0001;

#[repr(C)]
pub struct Attr {
    magic: u64,
    pub stack_size: u64,
    pub stack_addr: u64,
    pub guard_size: u64,
    pub detach: i32,
    pub policy: i32,
    pub priority: i32,
    pub inherit: i32,
    pub affinity: u64,
    pub scope: i32,
}

impl Default for Attr {
    fn default() -> Self {
        Attr {
            magic: ATTR_MAGIC,
            stack_size: DEFAULT_STACK,
            stack_addr: 0,
            guard_size: 0x1000,
            detach: 0,
            policy: 1,
            priority: PRIO_DEFAULT,
            inherit: 4, // PTHREAD_EXPLICIT_SCHED
            affinity: 0x7F,
            scope: 0,
        }
    }
}

unsafe fn attr_ref<'a>(slot: *mut u64) -> Option<&'a mut Attr> {
    if slot.is_null() {
        return None;
    }
    let p = unsafe { *slot } as *mut Attr;
    if p.is_null() || unsafe { (*p).magic } != ATTR_MAGIC {
        return None;
    }
    Some(unsafe { &mut *p })
}

fn attr_init(slot: *mut u64) -> i32 {
    if slot.is_null() {
        return EINVAL;
    }
    let a = Box::into_raw(Box::new(Attr::default()));
    unsafe { *slot = a as u64 };
    0
}

fn attr_destroy(slot: *mut u64) -> i32 {
    unsafe {
        match attr_ref(slot) {
            Some(a) => {
                a.magic = 0;
                drop(Box::from_raw(a as *mut Attr));
                *slot = 0;
                0
            }
            None => EINVAL,
        }
    }
}

fn create(out: *mut u64, attr: *mut u64, entry: u64, arg: u64, name: &str) -> i32 {
    if out.is_null() || entry == 0 {
        return EINVAL;
    }
    let def = Attr::default();
    let a = unsafe { attr_ref(attr) }.map(|a| a as &Attr).unwrap_or(&def);
    let stack = if a.stack_size == 0 { DEFAULT_STACK } else { a.stack_size };
    let t = thread::new_thread(name, entry, arg, stack, a.priority);
    t.affinity.store(a.affinity, Ordering::Relaxed);
    if a.detach != 0 {
        t.detached.store(true, Ordering::Relaxed);
    }
    if a.stack_addr != 0 {
        crate::debug!("thread '{name}': guest-supplied stack {:#x} ignored (host stack used)", a.stack_addr);
    }
    unsafe { *out = t.handle() };
    crate::info!(
        "create thread '{}' id={} entry={} arg={:#x} stack={:#x} prio={} affinity={:#x}",
        name,
        t.id,
        crate::loader::describe_addr(entry),
        arg,
        stack,
        a.priority,
        a.affinity
    );
    match thread::spawn(t) {
        Ok(()) => 0,
        Err(e) => {
            crate::error!("host thread creation failed: {e}");
            EAGAIN
        }
    }
}

fn cur() -> &'static GuestThread {
    thread::current().expect("HLE thread call from a non-guest thread")
}

fn target(h: u64) -> Option<&'static GuestThread> {
    if h == 0 { thread::current() } else { thread::from_handle(h) }
}

// ---------------------------------------------------------------------------
// Keys

static KEYS: Mutex<[Option<u64>; PTHREAD_KEYS_MAX]> = Mutex::new([None; PTHREAD_KEYS_MAX]);
static KEY_USED: Mutex<[bool; PTHREAD_KEYS_MAX]> = Mutex::new([false; PTHREAD_KEYS_MAX]);

pub fn key_destructor(k: usize) -> Option<u64> {
    KEYS.lock()[k].filter(|d| *d != 0)
}

fn key_create(out: *mut i32, dtor: u64) -> i32 {
    let mut used = KEY_USED.lock();
    for (i, u) in used.iter_mut().enumerate() {
        if !*u {
            *u = true;
            KEYS.lock()[i] = Some(dtor);
            unsafe { *out = i as i32 };
            return 0;
        }
    }
    EAGAIN
}

fn key_delete(k: i32) -> i32 {
    if k < 0 || k as usize >= PTHREAD_KEYS_MAX {
        return EINVAL;
    }
    KEY_USED.lock()[k as usize] = false;
    KEYS.lock()[k as usize] = None;
    0
}

fn getspecific(k: i32) -> u64 {
    if k < 0 || k as usize >= PTHREAD_KEYS_MAX {
        return 0;
    }
    cur().specific.lock()[k as usize]
}

fn setspecific(k: i32, v: u64) -> i32 {
    if k < 0 || k as usize >= PTHREAD_KEYS_MAX || !KEY_USED.lock()[k as usize] {
        return EINVAL;
    }
    cur().specific.lock()[k as usize] = v;
    0
}

// ---------------------------------------------------------------------------
// Once

static ONCE_LOCK: Mutex<()> = Mutex::new(());
static ONCE_CV: Condvar = Condvar::new();
const ONCE_NEVER: i32 = 0;
const ONCE_DONE: i32 = 1;
const ONCE_IN_PROGRESS: i32 = 2;

fn once(ctl: *mut i32, init: u64) -> i32 {
    if ctl.is_null() {
        return EINVAL;
    }
    let state = unsafe { AtomicI32::from_ptr(ctl) };
    if state.load(Ordering::Acquire) == ONCE_DONE {
        return 0;
    }
    let mut g = ONCE_LOCK.lock();
    loop {
        match state.load(Ordering::Acquire) {
            ONCE_DONE => return 0,
            ONCE_NEVER => {
                state.store(ONCE_IN_PROGRESS, Ordering::Release);
                drop(g);
                unsafe { crate::hle::call_guest(init, &[]) };
                let _g2 = ONCE_LOCK.lock();
                state.store(ONCE_DONE, Ordering::Release);
                ONCE_CV.notify_all();
                return 0;
            }
            _ => ONCE_CV.wait(&mut g),
        }
    }
}

fn sce_ret(e: i32) -> i32 {
    if e == 0 { 0 } else { sce(e) }
}

sysv! {
    fn scePthreadAttrInit(attr: *mut u64) -> i32 { sce_ret(attr_init(attr)) }
    fn scePthreadAttrDestroy(attr: *mut u64) -> i32 { sce_ret(attr_destroy(attr)) }
    fn scePthreadAttrSetstacksize(attr: *mut u64, size: u64) -> i32 {
        match attr_ref(attr) { Some(a) => { a.stack_size = size; 0 } None => sce(EINVAL) }
    }
    fn scePthreadAttrGetstacksize(attr: *mut u64, size: *mut u64) -> i32 {
        match attr_ref(attr) { Some(a) => { *size = a.stack_size; 0 } None => sce(EINVAL) }
    }
    fn scePthreadAttrSetstackaddr(attr: *mut u64, addr: u64) -> i32 {
        match attr_ref(attr) { Some(a) => { a.stack_addr = addr; 0 } None => sce(EINVAL) }
    }
    fn scePthreadAttrSetdetachstate(attr: *mut u64, state: i32) -> i32 {
        match attr_ref(attr) { Some(a) => { a.detach = state; 0 } None => sce(EINVAL) }
    }
    fn scePthreadAttrGetdetachstate(attr: *mut u64, state: *mut i32) -> i32 {
        match attr_ref(attr) { Some(a) => { *state = a.detach; 0 } None => sce(EINVAL) }
    }
    fn scePthreadAttrSetschedpolicy(attr: *mut u64, policy: i32) -> i32 {
        match attr_ref(attr) { Some(a) => { a.policy = policy; 0 } None => sce(EINVAL) }
    }
    fn scePthreadAttrGetschedpolicy(attr: *mut u64, policy: *mut i32) -> i32 {
        match attr_ref(attr) { Some(a) => { *policy = a.policy; 0 } None => sce(EINVAL) }
    }
    fn scePthreadAttrSetschedparam(attr: *mut u64, param: *const i32) -> i32 {
        match attr_ref(attr) { Some(a) if !param.is_null() => { a.priority = *param; 0 } _ => sce(EINVAL) }
    }
    fn scePthreadAttrGetschedparam(attr: *mut u64, param: *mut i32) -> i32 {
        match attr_ref(attr) { Some(a) if !param.is_null() => { *param = a.priority; 0 } _ => sce(EINVAL) }
    }
    fn scePthreadAttrSetinheritsched(attr: *mut u64, inherit: i32) -> i32 {
        match attr_ref(attr) { Some(a) => { a.inherit = inherit; 0 } None => sce(EINVAL) }
    }
    fn scePthreadAttrSetaffinity(attr: *mut u64, mask: u64) -> i32 {
        match attr_ref(attr) { Some(a) => { a.affinity = mask; 0 } None => sce(EINVAL) }
    }
    fn scePthreadAttrGetaffinity(attr: *mut u64, mask: *mut u64) -> i32 {
        match attr_ref(attr) { Some(a) => { *mask = a.affinity; 0 } None => sce(EINVAL) }
    }
    fn scePthreadAttrSetguardsize(attr: *mut u64, size: u64) -> i32 {
        match attr_ref(attr) { Some(a) => { a.guard_size = size; 0 } None => sce(EINVAL) }
    }
    fn scePthreadAttrGet(thread: u64, attr: *mut u64) -> i32 {
        let Some(t) = target(thread) else { return sce(ESRCH) };
        match attr_ref(attr) {
            Some(a) => {
                a.stack_size = t.stack_size;
                a.priority = t.priority.load(Ordering::Relaxed);
                a.affinity = t.affinity.load(Ordering::Relaxed);
                a.detach = t.detached.load(Ordering::Relaxed) as i32;
                0
            }
            None => sce(EINVAL),
        }
    }
    fn scePthreadCreate(out: *mut u64, attr: *mut u64, entry: u64, arg: u64, name: *const u8) -> i32 {
        let n = if name.is_null() { "pthread".to_string() } else { cstr(name).to_string() };
        sce_ret(create(out, attr, entry, arg, &n))
    }
    fn scePthreadSelf() -> u64 { cur().handle() }
    fn scePthreadEqual(a: u64, b: u64) -> i32 { (a == b) as i32 }
    fn scePthreadYield() { std::thread::yield_now(); }
    fn scePthreadExit(value: u64) { thread::exit_current(value) }
    fn scePthreadJoin(t: u64, value: *mut u64) -> i32 {
        let Some(t) = thread::from_handle(t) else { return sce(ESRCH) };
        let v = thread::join(t);
        if !value.is_null() { *value = v; }
        0
    }
    fn scePthreadDetach(t: u64) -> i32 {
        let Some(t) = thread::from_handle(t) else { return sce(ESRCH) };
        t.detached.store(true, Ordering::Relaxed);
        0
    }
    fn scePthreadRename(t: u64, name: *const u8) -> i32 {
        let Some(t) = target(t) else { return sce(ESRCH) };
        *t.name.lock() = cstr(name).to_string();
        0
    }
    fn scePthreadGetname(t: u64, name: *mut u8) -> i32 {
        let Some(t) = target(t) else { return sce(ESRCH) };
        crate::hle::write_cstr(name, 32, &t.name());
        0
    }
    fn scePthreadGetprio(t: u64, prio: *mut i32) -> i32 {
        let Some(t) = target(t) else { return sce(ESRCH) };
        *prio = t.priority.load(Ordering::Relaxed);
        0
    }
    fn scePthreadSetprio(t: u64, prio: i32) -> i32 {
        let Some(t) = target(t) else { return sce(ESRCH) };
        t.priority.store(prio, Ordering::Relaxed);
        0
    }
    fn scePthreadGetaffinity(t: u64, mask: *mut u64) -> i32 {
        let Some(t) = target(t) else { return sce(ESRCH) };
        *mask = t.affinity.load(Ordering::Relaxed);
        0
    }
    fn scePthreadSetaffinity(t: u64, mask: u64) -> i32 {
        let Some(t) = target(t) else { return sce(ESRCH) };
        t.affinity.store(mask, Ordering::Relaxed);
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 64 {
            crate::info!("thread '{}' affinity {mask:#x} (recorded, not applied to the host thread)", t.name());
        }
        0
    }
    fn scePthreadGetthreadid() -> i32 { cur().id as i32 }
    fn scePthreadOnce(ctl: *mut i32, init: u64) -> i32 { sce_ret(once(ctl, init)) }
    fn scePthreadKeyCreate(key: *mut i32, dtor: u64) -> i32 { sce_ret(key_create(key, dtor)) }
    fn scePthreadKeyDelete(key: i32) -> i32 { sce_ret(key_delete(key)) }
    fn scePthreadGetspecific(key: i32) -> u64 { getspecific(key) }
    fn scePthreadSetspecific(key: i32, v: u64) -> i32 { sce_ret(setspecific(key, v)) }

    // POSIX variants (return errno values directly).
    fn pthread_attr_init(attr: *mut u64) -> i32 { attr_init(attr) }
    fn pthread_attr_destroy(attr: *mut u64) -> i32 { attr_destroy(attr) }
    fn pthread_attr_setstacksize(attr: *mut u64, size: u64) -> i32 {
        match attr_ref(attr) { Some(a) => { a.stack_size = size; 0 } None => EINVAL }
    }
    fn pthread_attr_getstacksize(attr: *mut u64, size: *mut u64) -> i32 {
        match attr_ref(attr) { Some(a) => { *size = a.stack_size; 0 } None => EINVAL }
    }
    fn pthread_attr_setdetachstate(attr: *mut u64, state: i32) -> i32 {
        match attr_ref(attr) { Some(a) => { a.detach = state; 0 } None => EINVAL }
    }
    fn pthread_attr_setschedparam(attr: *mut u64, param: *const i32) -> i32 {
        match attr_ref(attr) { Some(a) if !param.is_null() => { a.priority = *param; 0 } _ => EINVAL }
    }
    fn pthread_attr_getschedparam(attr: *mut u64, param: *mut i32) -> i32 {
        match attr_ref(attr) { Some(a) if !param.is_null() => { *param = a.priority; 0 } _ => EINVAL }
    }
    fn pthread_attr_setschedpolicy(attr: *mut u64, policy: i32) -> i32 {
        match attr_ref(attr) { Some(a) => { a.policy = policy; 0 } None => EINVAL }
    }
    fn pthread_create(out: *mut u64, attr: *mut u64, entry: u64, arg: u64) -> i32 {
        create(out, attr, entry, arg, "pthread")
    }
    fn pthread_create_name_np(out: *mut u64, attr: *mut u64, entry: u64, arg: u64, name: *const u8) -> i32 {
        let n = if name.is_null() { "pthread".to_string() } else { cstr(name).to_string() };
        create(out, attr, entry, arg, &n)
    }
    fn pthread_self() -> u64 { cur().handle() }
    fn pthread_equal(a: u64, b: u64) -> i32 { (a == b) as i32 }
    fn pthread_exit(value: u64) { thread::exit_current(value) }
    fn pthread_join(t: u64, value: *mut u64) -> i32 {
        let Some(t) = thread::from_handle(t) else { return ESRCH };
        let v = thread::join(t);
        if !value.is_null() { *value = v; }
        0
    }
    fn pthread_detach(t: u64) -> i32 {
        let Some(t) = thread::from_handle(t) else { return ESRCH };
        t.detached.store(true, Ordering::Relaxed);
        0
    }
    fn pthread_once(ctl: *mut i32, init: u64) -> i32 { once(ctl, init) }
    fn pthread_key_create(key: *mut i32, dtor: u64) -> i32 { key_create(key, dtor) }
    fn pthread_key_delete(key: i32) -> i32 { key_delete(key) }
    fn pthread_getspecific(key: i32) -> u64 { getspecific(key) }
    fn pthread_setspecific(key: i32, v: u64) -> i32 { setspecific(key, v) }
    fn pthread_yield() { std::thread::yield_now(); }
    fn sched_yield() -> i32 { std::thread::yield_now(); 0 }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libkernel", [
        scePthreadAttrInit, scePthreadAttrDestroy, scePthreadAttrSetstacksize, scePthreadAttrGetstacksize,
        scePthreadAttrSetstackaddr, scePthreadAttrSetdetachstate, scePthreadAttrGetdetachstate,
        scePthreadAttrSetschedpolicy, scePthreadAttrGetschedpolicy, scePthreadAttrSetschedparam,
        scePthreadAttrGetschedparam, scePthreadAttrSetinheritsched, scePthreadAttrSetaffinity,
        scePthreadAttrGetaffinity, scePthreadAttrSetguardsize, scePthreadAttrGet, scePthreadCreate,
        scePthreadSelf, scePthreadEqual, scePthreadYield, scePthreadExit, scePthreadJoin, scePthreadDetach,
        scePthreadRename, scePthreadGetname, scePthreadGetprio, scePthreadSetprio, scePthreadGetaffinity,
        scePthreadSetaffinity, scePthreadGetthreadid, scePthreadOnce, scePthreadKeyCreate, scePthreadKeyDelete,
        scePthreadGetspecific, scePthreadSetspecific,
    ]);
    reg!(r, "libScePosix", [
        pthread_attr_init, pthread_attr_destroy, pthread_attr_setstacksize, pthread_attr_getstacksize,
        pthread_attr_setdetachstate, pthread_attr_setschedparam, pthread_attr_getschedparam,
        pthread_attr_setschedpolicy, pthread_create, pthread_self, pthread_equal, pthread_exit, pthread_join,
        pthread_detach, pthread_once, pthread_key_create, pthread_key_delete, pthread_getspecific,
        pthread_setspecific, pthread_yield, sched_yield, pthread_create_name_np,
    ]);
}
