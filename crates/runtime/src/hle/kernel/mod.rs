//! libkernel HLE (threads, synchronization, memory, files, time, events, misc).

pub mod equeue;
pub mod extra;
pub mod fs;
pub mod memory;
pub mod misc;
pub mod pthread;
pub mod sync;
pub mod time;

use crate::hle::Registry;

pub mod errno {
    pub const EPERM: i32 = 1;
    pub const ENOENT: i32 = 2;
    pub const ESRCH: i32 = 3;
    pub const EINTR: i32 = 4;
    pub const EIO: i32 = 5;
    pub const EBADF: i32 = 9;
    pub const EDEADLK: i32 = 11;
    pub const ENOMEM: i32 = 12;
    pub const EACCES: i32 = 13;
    pub const EFAULT: i32 = 14;
    pub const EBUSY: i32 = 16;
    pub const EEXIST: i32 = 17;
    pub const ENOTDIR: i32 = 20;
    pub const EISDIR: i32 = 21;
    pub const EINVAL: i32 = 22;
    pub const EMFILE: i32 = 24;
    pub const ENOTTY: i32 = 25;
    pub const ENOSPC: i32 = 28;
    pub const ESPIPE: i32 = 29;
    pub const EROFS: i32 = 30;
    pub const EAGAIN: i32 = 35;
    pub const ENOTSUP: i32 = 45;
    pub const ETIMEDOUT: i32 = 60;
    pub const ENAMETOOLONG: i32 = 63;
    pub const ENOTEMPTY: i32 = 66;
    pub const ENOSYS: i32 = 78;
    pub const ECANCELED: i32 = 85;
}

/// SCE kernel error code for an errno value.
pub const fn sce(e: i32) -> i32 {
    (0x8002_0000u32 | e as u32) as i32
}

pub fn set_errno(e: i32) {
    if let Some(t) = crate::thread::current() {
        t.errno.store(e, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Converts an SCE kernel error to POSIX style: sets errno, returns -1.
pub fn posix_ret(r: i32) -> i32 {
    if r < 0 {
        set_errno(r & 0xFFFF);
        -1
    } else {
        r
    }
}

pub fn register(r: &mut Registry) {
    pthread::register(r);
    sync::register(r);
    memory::register(r);
    fs::register(r);
    time::register(r);
    equeue::register(r);
    misc::register(r);
    extra::register(r);
}
