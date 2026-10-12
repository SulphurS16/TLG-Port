//! Guest file system: mount table, path resolution and file descriptors.
//!
//! `/app0` layers the update tree over the base tree and is read-only. Writable
//! mounts live under the project's local-data runtime directory. Paths are
//! normalized component by component and can never escape a mount root.

use super::errno::*;
use super::{posix_ret, sce};
use crate::hle::{Registry, cstr};
use crate::{reg, sysv};
use parking_lot::{Mutex, RwLock};
use std::fs::{File, OpenOptions};
use std::os::windows::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const O_WRONLY: i32 = 0x1;
pub const O_RDWR: i32 = 0x2;
pub const O_ACCMODE: i32 = 0x3;
pub const O_APPEND: i32 = 0x8;
pub const O_CREAT: i32 = 0x200;
pub const O_TRUNC: i32 = 0x400;
pub const O_EXCL: i32 = 0x800;
pub const O_DIRECTORY: i32 = 0x20000;

#[derive(Clone, Debug)]
pub struct Mount {
    pub prefix: String,
    pub layers: Vec<PathBuf>,
    pub writable: bool,
}

static MOUNTS: RwLock<Vec<Mount>> = RwLock::new(Vec::new());

pub fn mount(prefix: &str, layers: Vec<PathBuf>, writable: bool) {
    let mut m = MOUNTS.write();
    m.retain(|x| x.prefix != prefix);
    crate::info!("mount {prefix} -> {:?} writable={writable}", layers);
    m.push(Mount { prefix: prefix.to_string(), layers, writable });
    m.sort_by(|a, b| b.prefix.len().cmp(&a.prefix.len()));
}

pub fn unmount(prefix: &str) {
    MOUNTS.write().retain(|x| x.prefix != prefix);
}

fn normalize(path: &str) -> Option<Vec<String>> {
    let mut out: Vec<String> = Vec::new();
    for c in path.split('/') {
        match c {
            "" | "." => {}
            ".." => {
                out.pop()?;
            }
            c => {
                if c.chars().any(|ch| matches!(ch, ':' | '\\' | '*' | '?' | '"' | '<' | '>' | '|') || (ch as u32) < 0x20) {
                    return None;
                }
                out.push(c.to_string());
            }
        }
    }
    Some(out)
}

pub struct Resolved {
    pub candidates: Vec<PathBuf>,
    pub writable: bool,
}

pub fn resolve(guest: &str) -> Option<Resolved> {
    let abs = if guest.starts_with('/') { guest.to_string() } else { format!("/app0/{guest}") };
    let comps = normalize(&abs)?;
    let mounts = MOUNTS.read();
    for m in mounts.iter() {
        let pre = normalize(&m.prefix)?;
        if comps.len() >= pre.len() && comps[..pre.len()] == pre[..] {
            let rel: PathBuf = comps[pre.len()..].iter().collect();
            return Some(Resolved { candidates: m.layers.iter().map(|l| l.join(&rel)).collect(), writable: m.writable });
        }
    }
    None
}

/// Host path for reading (first existing layer).
pub fn host_path_existing(guest: &str) -> Option<PathBuf> {
    resolve(guest)?.candidates.into_iter().find(|p| p.exists())
}

// ---------------------------------------------------------------------------
// File descriptors

pub enum Kind {
    File { file: File, writable: bool },
    Dir { entries: Vec<(String, bool)> },
    Console(i32),
}

pub struct Fd {
    pub kind: Kind,
    pub pos: Mutex<u64>,
    pub path: String,
}

static FDS: Mutex<Vec<Option<Arc<Fd>>>> = Mutex::new(Vec::new());

fn alloc_fd(fd: Fd) -> i32 {
    let mut t = FDS.lock();
    if t.is_empty() {
        t.push(Some(Arc::new(Fd { kind: Kind::Console(0), pos: Mutex::new(0), path: "stdin".into() })));
        t.push(Some(Arc::new(Fd { kind: Kind::Console(1), pos: Mutex::new(0), path: "stdout".into() })));
        t.push(Some(Arc::new(Fd { kind: Kind::Console(2), pos: Mutex::new(0), path: "stderr".into() })));
    }
    let a = Arc::new(fd);
    for (i, s) in t.iter_mut().enumerate().skip(3) {
        if s.is_none() {
            *s = Some(a);
            return i as i32;
        }
    }
    t.push(Some(a));
    (t.len() - 1) as i32
}

fn get_fd(fd: i32) -> Option<Arc<Fd>> {
    if (0..=2).contains(&fd) {
        return Some(Arc::new(Fd { kind: Kind::Console(fd), pos: Mutex::new(0), path: "console".into() }));
    }
    FDS.lock().get(fd as usize).and_then(|s| s.clone())
}

fn list_dir(cands: &[PathBuf]) -> Vec<(String, bool)> {
    let mut seen = std::collections::BTreeMap::new();
    for c in cands {
        if let Ok(rd) = std::fs::read_dir(c) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
                seen.entry(name).or_insert(is_dir);
            }
        }
    }
    let mut v = vec![(".".to_string(), true), ("..".to_string(), true)];
    v.extend(seen);
    v
}

fn io_errno(e: &std::io::Error) -> i32 {
    use std::io::ErrorKind::*;
    match e.kind() {
        NotFound => ENOENT,
        PermissionDenied => EACCES,
        AlreadyExists => EEXIST,
        InvalidInput => EINVAL,
        DirectoryNotEmpty => ENOTEMPTY,
        NotADirectory => ENOTDIR,
        IsADirectory => EISDIR,
        _ => EIO,
    }
}

/// Opens a guest path. Returns fd or SCE error.
pub fn vfs_open(path: &str, flags: i32, _mode: u32) -> i32 {
    let Some(r) = resolve(path) else {
        crate::warn!("open '{path}': no mount");
        return sce(ENOENT);
    };
    let wants_write = flags & O_ACCMODE != 0 || flags & (O_CREAT | O_TRUNC | O_APPEND) != 0;
    let existing = r.candidates.iter().find(|p| p.exists()).cloned();
    if wants_write && !r.writable {
        crate::warn!("open '{path}' for writing on read-only mount");
        return sce(EROFS);
    }
    if let Some(p) = &existing {
        if p.is_dir() {
            if flags & O_ACCMODE != 0 {
                return sce(EISDIR);
            }
            let entries = list_dir(&r.candidates);
            crate::debug!("opendir '{path}' ({} entries)", entries.len());
            return alloc_fd(Fd { kind: Kind::Dir { entries }, pos: Mutex::new(0), path: path.to_string() });
        }
        if flags & O_CREAT != 0 && flags & O_EXCL != 0 {
            return sce(EEXIST);
        }
    } else if flags & O_DIRECTORY != 0 || flags & O_CREAT == 0 {
        crate::debug!("open '{path}': not found");
        return sce(ENOENT);
    }
    let host = existing.unwrap_or_else(|| r.candidates[0].clone());
    let mut oo = OpenOptions::new();
    oo.read(true);
    if wants_write {
        oo.write(true);
    }
    if flags & O_CREAT != 0 {
        oo.create(true);
    }
    if flags & O_TRUNC != 0 {
        oo.truncate(true);
    }
    match oo.open(&host) {
        Ok(f) => {
            let pos = if flags & O_APPEND != 0 { f.metadata().map(|m| m.len()).unwrap_or(0) } else { 0 };
            let fd = alloc_fd(Fd { kind: Kind::File { file: f, writable: wants_write }, pos: Mutex::new(pos), path: path.to_string() });
            crate::debug!("open '{path}' flags={flags:#x} -> fd {fd}");
            fd
        }
        Err(e) => {
            crate::warn!("open '{path}' -> {}: {e}", host.display());
            sce(io_errno(&e))
        }
    }
}

pub fn vfs_close(fd: i32) -> i32 {
    let mut t = FDS.lock();
    match t.get_mut(fd as usize) {
        Some(s @ Some(_)) if fd > 2 => {
            *s = None;
            0
        }
        _ => sce(EBADF),
    }
}

fn console_write(fd: i32, data: &[u8]) {
    let s = String::from_utf8_lossy(data);
    for line in s.trim_end_matches('\n').split('\n') {
        crate::info!("[guest fd{fd}] {line}");
    }
}

pub fn pread_fd(fd: i32, buf: &mut [u8], off: u64) -> Result<usize, i32> {
    let f = get_fd(fd).ok_or(EBADF)?;
    match &f.kind {
        Kind::File { file, .. } => {
            crate::gpu::os_access(buf.as_ptr() as u64, buf.len() as u64);
            let mut done = 0;
            while done < buf.len() {
                match file.seek_read(&mut buf[done..], off + done as u64) {
                    Ok(0) => break,
                    Ok(n) => done += n,
                    Err(e) => return Err(io_errno(&e)),
                }
            }
            Ok(done)
        }
        Kind::Dir { .. } => Err(EISDIR),
        Kind::Console(_) => Ok(0),
    }
}

pub fn vfs_read(fd: i32, buf: &mut [u8]) -> Result<usize, i32> {
    let f = get_fd(fd).ok_or(EBADF)?;
    match &f.kind {
        Kind::File { .. } => {
            let mut pos = f.pos.lock();
            let n = pread_fd(fd, buf, *pos)?;
            *pos += n as u64;
            Ok(n)
        }
        Kind::Dir { .. } => Err(EISDIR),
        Kind::Console(_) => Ok(0),
    }
}

pub fn pwrite_fd(fd: i32, data: &[u8], off: Option<u64>) -> Result<usize, i32> {
    let f = get_fd(fd).ok_or(EBADF)?;
    match &f.kind {
        Kind::File { file, writable } => {
            if !writable {
                return Err(EBADF);
            }
            let mut pos = f.pos.lock();
            let at = off.unwrap_or(*pos);
            crate::gpu::os_access(data.as_ptr() as u64, data.len() as u64);
            let mut done = 0;
            while done < data.len() {
                match file.seek_write(&data[done..], at + done as u64) {
                    Ok(n) => done += n,
                    Err(e) => return Err(io_errno(&e)),
                }
            }
            if off.is_none() {
                *pos = at + done as u64;
            }
            Ok(done)
        }
        Kind::Dir { .. } => Err(EISDIR),
        Kind::Console(c) => {
            console_write(*c, data);
            Ok(data.len())
        }
    }
}

pub fn vfs_lseek(fd: i32, off: i64, whence: i32) -> i64 {
    let Some(f) = get_fd(fd) else { return sce(EBADF) as i64 };
    let size = match &f.kind {
        Kind::File { file, .. } => file.metadata().map(|m| m.len()).unwrap_or(0),
        Kind::Dir { entries } => entries.len() as u64,
        Kind::Console(_) => return sce(ESPIPE) as i64,
    };
    let mut pos = f.pos.lock();
    let base = match whence {
        0 => 0i64,
        1 => *pos as i64,
        2 => size as i64,
        _ => return sce(EINVAL) as i64,
    };
    let n = base + off;
    if n < 0 {
        return sce(EINVAL) as i64;
    }
    *pos = n as u64;
    n
}

/// FreeBSD `struct stat` (120 bytes).
#[repr(C)]
#[derive(Default)]
pub struct Stat {
    pub dev: u32,
    pub ino: u32,
    pub mode: u16,
    pub nlink: u16,
    pub uid: u32,
    pub gid: u32,
    pub rdev: u32,
    pub atime: [i64; 2],
    pub mtime: [i64; 2],
    pub ctime: [i64; 2],
    pub size: i64,
    pub blocks: i64,
    pub blksize: u32,
    pub flags: u32,
    pub gen_: u32,
    pub lspare: i32,
    pub birthtime: [i64; 2],
}

const S_IFDIR: u16 = 0o040000;
const S_IFREG: u16 = 0o100000;

fn fill_stat(md: &std::fs::Metadata) -> Stat {
    let mut st = Stat::default();
    let t = md
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| [d.as_secs() as i64, d.subsec_nanos() as i64])
        .unwrap_or([0, 0]);
    st.atime = t;
    st.mtime = t;
    st.ctime = t;
    st.birthtime = t;
    if md.is_dir() {
        st.mode = S_IFDIR | 0o777;
        st.size = 0x10000;
        st.nlink = 2;
    } else {
        st.mode = S_IFREG | 0o777;
        st.size = md.len() as i64;
        st.nlink = 1;
    }
    st.blksize = 0x10000;
    st.blocks = (st.size + 511) / 512;
    st
}

pub fn stat_path(path: &str, out: *mut Stat) -> i32 {
    let Some(p) = host_path_existing(path) else {
        crate::debug!("stat '{path}': not found");
        return sce(ENOENT);
    };
    match std::fs::metadata(&p) {
        Ok(md) => {
            unsafe { std::ptr::write_unaligned(out, fill_stat(&md)) };
            0
        }
        Err(e) => sce(io_errno(&e)),
    }
}

pub fn fstat_fd(fd: i32, out: *mut Stat) -> i32 {
    let Some(f) = get_fd(fd) else { return sce(EBADF) };
    let st = match &f.kind {
        Kind::File { file, .. } => match file.metadata() {
            Ok(md) => fill_stat(&md),
            Err(e) => return sce(io_errno(&e)),
        },
        Kind::Dir { .. } => Stat { mode: S_IFDIR | 0o777, nlink: 2, blksize: 0x10000, size: 0x10000, ..Default::default() },
        Kind::Console(_) => Stat { mode: 0o020000 | 0o666, ..Default::default() },
    };
    unsafe { std::ptr::write_unaligned(out, st) };
    0
}

/// FreeBSD getdirentries: packs `struct dirent` records.
pub fn getdirentries(fd: i32, buf: *mut u8, nbytes: usize, basep: *mut i64) -> i64 {
    let Some(f) = get_fd(fd) else { return sce(EBADF) as i64 };
    let Kind::Dir { entries } = &f.kind else { return sce(ENOTDIR) as i64 };
    let mut pos = f.pos.lock();
    if !basep.is_null() {
        unsafe { *basep = *pos as i64 };
    }
    let mut off = 0usize;
    while (*pos as usize) < entries.len() {
        let (name, is_dir) = &entries[*pos as usize];
        let nb = name.as_bytes();
        let namlen = nb.len().min(255);
        let reclen = 8 + ((namlen + 1 + 3) & !3);
        if off + reclen > nbytes {
            break;
        }
        unsafe {
            let p = buf.add(off);
            std::ptr::write_bytes(p, 0, reclen);
            std::ptr::write_unaligned(p as *mut u32, (*pos as u32) + 1);
            std::ptr::write_unaligned(p.add(4) as *mut u16, reclen as u16);
            *p.add(6) = if *is_dir { 4 } else { 8 };
            *p.add(7) = namlen as u8;
            std::ptr::copy_nonoverlapping(nb.as_ptr(), p.add(8), namlen);
        }
        off += reclen;
        *pos += 1;
    }
    off as i64
}

fn writable_target(path: &str) -> Result<PathBuf, i32> {
    let r = resolve(path).ok_or(ENOENT)?;
    if !r.writable {
        return Err(EROFS);
    }
    Ok(r.candidates[0].clone())
}

pub fn vfs_mkdir(path: &str) -> i32 {
    let p = match writable_target(path) {
        Ok(p) => p,
        Err(e) => return sce(e),
    };
    if p.exists() {
        return sce(EEXIST);
    }
    match std::fs::create_dir(&p) {
        Ok(()) => 0,
        Err(e) => sce(io_errno(&e)),
    }
}

pub fn vfs_rmdir(path: &str) -> i32 {
    match writable_target(path) {
        Ok(p) => match std::fs::remove_dir(&p) {
            Ok(()) => 0,
            Err(e) => sce(io_errno(&e)),
        },
        Err(e) => sce(e),
    }
}

pub fn vfs_unlink(path: &str) -> i32 {
    match writable_target(path) {
        Ok(p) => match std::fs::remove_file(&p) {
            Ok(()) => 0,
            Err(e) => sce(io_errno(&e)),
        },
        Err(e) => sce(e),
    }
}

pub fn vfs_rename(from: &str, to: &str) -> i32 {
    match (writable_target(from), writable_target(to)) {
        (Ok(a), Ok(b)) => match std::fs::rename(&a, &b) {
            Ok(()) => 0,
            Err(e) => sce(io_errno(&e)),
        },
        (Err(e), _) | (_, Err(e)) => sce(e),
    }
}

pub fn ftruncate(fd: i32, len: u64) -> i32 {
    let Some(f) = get_fd(fd) else { return sce(EBADF) };
    match &f.kind {
        Kind::File { file, writable: true } => match file.set_len(len) {
            Ok(()) => 0,
            Err(e) => sce(io_errno(&e)),
        },
        _ => sce(EINVAL),
    }
}

pub fn truncate(path: &str, len: u64) -> i32 {
    match writable_target(path) {
        Ok(p) => match OpenOptions::new().write(true).open(&p).and_then(|f| f.set_len(len)) {
            Ok(()) => 0,
            Err(e) => sce(io_errno(&e)),
        },
        Err(e) => sce(e),
    }
}

fn ret_len(r: Result<usize, i32>) -> i64 {
    match r {
        Ok(n) => n as i64,
        Err(e) => sce(e) as i64,
    }
}

fn posix_len(r: Result<usize, i32>) -> i64 {
    match r {
        Ok(n) => n as i64,
        Err(e) => {
            crate::hle::kernel::set_errno(e);
            -1
        }
    }
}

sysv! {
    fn sceKernelOpen(path: *const u8, flags: i32, mode: u32) -> i32 { vfs_open(cstr(path), flags, mode) }
    fn sceKernelClose(fd: i32) -> i32 { vfs_close(fd) }
    fn sceKernelRead(fd: i32, buf: *mut u8, n: u64) -> i64 {
        ret_len(vfs_read(fd, std::slice::from_raw_parts_mut(buf, n as usize)))
    }
    fn sceKernelWrite(fd: i32, buf: *const u8, n: u64) -> i64 {
        ret_len(pwrite_fd(fd, std::slice::from_raw_parts(buf, n as usize), None))
    }
    fn sceKernelPread(fd: i32, buf: *mut u8, n: u64, off: i64) -> i64 {
        ret_len(pread_fd(fd, std::slice::from_raw_parts_mut(buf, n as usize), off as u64))
    }
    fn sceKernelPwrite(fd: i32, buf: *const u8, n: u64, off: i64) -> i64 {
        ret_len(pwrite_fd(fd, std::slice::from_raw_parts(buf, n as usize), Some(off as u64)))
    }
    fn sceKernelLseek(fd: i32, off: i64, whence: i32) -> i64 { vfs_lseek(fd, off, whence) }
    fn sceKernelStat(path: *const u8, st: *mut Stat) -> i32 { stat_path(cstr(path), st) }
    fn sceKernelFstat(fd: i32, st: *mut Stat) -> i32 { fstat_fd(fd, st) }
    fn sceKernelMkdir(path: *const u8, mode: u32) -> i32 { vfs_mkdir(cstr(path)) }
    fn sceKernelRmdir(path: *const u8) -> i32 { vfs_rmdir(cstr(path)) }
    fn sceKernelUnlink(path: *const u8) -> i32 { vfs_unlink(cstr(path)) }
    fn sceKernelRename(from: *const u8, to: *const u8) -> i32 { vfs_rename(cstr(from), cstr(to)) }
    fn sceKernelGetdirentries(fd: i32, buf: *mut u8, n: i32, basep: *mut i64) -> i32 { getdirentries(fd, buf, n as usize, basep) as i32 }
    fn sceKernelGetdents(fd: i32, buf: *mut u8, n: i32) -> i32 { getdirentries(fd, buf, n as usize, std::ptr::null_mut()) as i32 }
    fn sceKernelFtruncate(fd: i32, len: i64) -> i32 { ftruncate(fd, len as u64) }
    fn sceKernelTruncate(path: *const u8, len: i64) -> i32 { truncate(cstr(path), len as u64) }
    fn sceKernelFsync(fd: i32) -> i32 { if get_fd(fd).is_some() { 0 } else { sce(EBADF) } }
    fn sceKernelCheckReachability(path: *const u8) -> i32 {
        if host_path_existing(cstr(path)).is_some() { 0 } else { sce(ENOENT) }
    }

    // POSIX
    fn open(path: *const u8, flags: i32, mode: u32) -> i32 { posix_ret(vfs_open(cstr(path), flags, mode)) }
    fn close(fd: i32) -> i32 { posix_ret(vfs_close(fd)) }
    fn read(fd: i32, buf: *mut u8, n: u64) -> i64 { posix_len(vfs_read(fd, std::slice::from_raw_parts_mut(buf, n as usize))) }
    fn _read(fd: i32, buf: *mut u8, n: u64) -> i64 { posix_len(vfs_read(fd, std::slice::from_raw_parts_mut(buf, n as usize))) }
    fn write(fd: i32, buf: *const u8, n: u64) -> i64 { posix_len(pwrite_fd(fd, std::slice::from_raw_parts(buf, n as usize), None)) }
    fn _write(fd: i32, buf: *const u8, n: u64) -> i64 { posix_len(pwrite_fd(fd, std::slice::from_raw_parts(buf, n as usize), None)) }
    fn pread(fd: i32, buf: *mut u8, n: u64, off: i64) -> i64 {
        posix_len(pread_fd(fd, std::slice::from_raw_parts_mut(buf, n as usize), off as u64))
    }
    fn pwrite(fd: i32, buf: *const u8, n: u64, off: i64) -> i64 {
        posix_len(pwrite_fd(fd, std::slice::from_raw_parts(buf, n as usize), Some(off as u64)))
    }
    fn lseek(fd: i32, off: i64, whence: i32) -> i64 {
        let r = vfs_lseek(fd, off, whence);
        if r < 0 { crate::hle::kernel::set_errno((r as i32) & 0xFFFF); -1 } else { r }
    }
    fn stat(path: *const u8, st: *mut Stat) -> i32 { posix_ret(stat_path(cstr(path), st)) }
    fn fstat(fd: i32, st: *mut Stat) -> i32 { posix_ret(fstat_fd(fd, st)) }
    fn mkdir(path: *const u8, mode: u32) -> i32 { posix_ret(vfs_mkdir(cstr(path))) }
    fn rmdir(path: *const u8) -> i32 { posix_ret(vfs_rmdir(cstr(path))) }
    fn unlink(path: *const u8) -> i32 { posix_ret(vfs_unlink(cstr(path))) }
    fn rename(from: *const u8, to: *const u8) -> i32 { posix_ret(vfs_rename(cstr(from), cstr(to))) }
    fn getdents(fd: i32, buf: *mut u8, n: i32) -> i32 {
        let r = getdirentries(fd, buf, n as usize, std::ptr::null_mut());
        if r < 0 { crate::hle::kernel::set_errno((r as i32) & 0xFFFF); -1 } else { r as i32 }
    }
    fn fsync(fd: i32) -> i32 { 0 }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libkernel", [
        sceKernelOpen, sceKernelClose, sceKernelRead, sceKernelWrite, sceKernelPread, sceKernelPwrite, sceKernelLseek,
        sceKernelStat, sceKernelFstat, sceKernelMkdir, sceKernelRmdir, sceKernelUnlink, sceKernelRename,
        sceKernelGetdirentries, sceKernelGetdents, sceKernelFtruncate, sceKernelTruncate, sceKernelFsync,
        sceKernelCheckReachability, _read, _write,
    ]);
    reg!(r, "libScePosix", [open, close, read, write, pread, pwrite, lseek, stat, fstat, mkdir, rmdir, unlink, rename, getdents, fsync]);
}

pub fn ensure_dir(p: &Path) {
    let _ = std::fs::create_dir_all(p);
}
