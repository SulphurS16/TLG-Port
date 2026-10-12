//! libSceSaveData: save directories and save-data memory backed by host
//! directories under local-data/saves/<user>/<title>/.

use crate::hle::{Registry, cstr};
use crate::{fixed, reg, sysv};
use parking_lot::Mutex;
use std::path::PathBuf;
use std::sync::OnceLock;

const ERR_PARAMETER: i32 = 0x809F_0000u32 as i32;
const ERR_NOT_MOUNTED: i32 = 0x809F_0004u32 as i32;
const ERR_EXISTS: i32 = 0x809F_0007u32 as i32;
const ERR_NOT_FOUND: i32 = 0x809F_0008u32 as i32;
const ERR_INTERNAL: i32 = 0x809F_000Bu32 as i32;
const ERR_MOUNT_FULL: i32 = 0x809F_000Cu32 as i32;

const MOUNT_RDONLY: u32 = 1;
const MOUNT_CREATE: u32 = 4;
const MOUNT_CREATE2: u32 = 32;

static ROOT: OnceLock<PathBuf> = OnceLock::new();
static MOUNTS: Mutex<[Option<PathBuf>; 16]> = Mutex::new([const { None }; 16]);

pub fn init(root: PathBuf) {
    ROOT.set(root).ok();
}

/// Where locally recorded trophy unlocks of the running title live.
pub fn trophy_file() -> PathBuf {
    ROOT.get().expect("savedata root").join(format!("trophies-{}.txt", title_id()))
}

fn title_id() -> String {
    match super::app::param_sfo().get("TITLE_ID") {
        Some(super::app::SfoValue::Str(s)) => s.clone(),
        _ => "CUSA03627".into(),
    }
}

fn user_root(user: i32, title: *const u8) -> PathBuf {
    let t = if title.is_null() { title_id() } else { unsafe { cstr(title) }.to_string() };
    let t = if t.is_empty() { title_id() } else { t };
    ROOT.get().expect("savedata root").join(format!("{user}")).join(t)
}

fn valid_dir_name(s: &str) -> bool {
    !s.is_empty() && s.len() < 32 && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

unsafe fn mount_point_index(mp: *const u8) -> Option<usize> {
    let s = unsafe { cstr(mp) };
    s.strip_prefix("/savedata").and_then(|n| n.parse::<usize>().ok()).filter(|&n| n < 16)
}

fn mounted_dir(mp: *const u8) -> Option<PathBuf> {
    let i = unsafe { mount_point_index(mp) }?;
    MOUNTS.lock()[i].clone()
}

#[repr(C)]
struct SaveParam {
    title: [u8; 128],
    sub_title: [u8; 128],
    detail: [u8; 1024],
    user_param: u32,
    pad: i32,
    mtime: i64,
    reserved: [u8; 32],
}

/// A save directory's parameters: ours (`sce_sys/param.bin`, written by
/// sceSaveDataSetParam) or, for saves made elsewhere (PS4 / other
/// emulators), `sce_sys/param.sfo` (MAINTITLE, SUBTITLE, DETAIL,
/// SAVEDATA_LIST_PARAM; time from the file).
fn read_save_param(dir: &std::path::Path) -> SaveParam {
    let mut p: SaveParam = unsafe { std::mem::zeroed() };
    if let Ok(b) = std::fs::read(dir.join("sce_sys").join("param.bin")) {
        if b.len() == std::mem::size_of::<SaveParam>() {
            unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), &mut p as *mut _ as *mut u8, b.len()) };
            return p;
        }
    }
    let sfo_path = dir.join("sce_sys").join("param.sfo");
    let Some(m) = std::fs::read(&sfo_path).ok().and_then(|b| super::app::parse_sfo(&b)) else { return p };
    let text = |k: &str| match m.get(k) {
        Some(super::app::SfoValue::Str(s)) => s.clone(),
        _ => String::new(),
    };
    let put = |dst: &mut [u8], s: &str| {
        let mut n = s.len().min(dst.len() - 1);
        while !s.is_char_boundary(n) {
            n -= 1;
        }
        dst[..n].copy_from_slice(&s.as_bytes()[..n]);
    };
    put(&mut p.title, &text("MAINTITLE"));
    put(&mut p.sub_title, &text("SUBTITLE"));
    put(&mut p.detail, &text("DETAIL"));
    if let Some(super::app::SfoValue::Int(v)) = m.get("SAVEDATA_LIST_PARAM") {
        p.user_param = *v;
    }
    if let Ok(t) = std::fs::metadata(&sfo_path).and_then(|md| md.modified()) {
        p.mtime = t.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    }
    p
}

fn memory_file(user: i32) -> PathBuf {
    user_root(user, std::ptr::null()).join("sdmemory").join("memory.dat")
}

// ---------------------------------------------------------------------------
// Save-data memory, version 2 (SDK 4: Setup/Get/Set SaveDataMemory2, Sync).
// The image lives in RAM; every Set2 and Sync replaces `memory.dat` atomically
// (temporary file, flushed, renamed over the old one, which is kept as
// `memory.dat.bak`), so a crash never leaves a half-written save.

struct SdMemory {
    user: i32,
    data: Vec<u8>,
}

static SDMEM: Mutex<Option<SdMemory>> = Mutex::new(None);

fn sdmem_dir(user: i32) -> PathBuf {
    user_root(user, std::ptr::null()).join("sdmemory")
}

fn sdmem_load(user: i32) -> Option<Vec<u8>> {
    let f = memory_file(user);
    match std::fs::read(&f) {
        Ok(b) => Some(b),
        Err(_) => {
            let bak = f.with_extension("dat.bak");
            let b = std::fs::read(&bak).ok()?;
            crate::warn!("save-data memory: {} missing, using backup {}", f.display(), bak.display());
            Some(b)
        }
    }
}

fn sdmem_store(user: i32, data: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let f = memory_file(user);
    std::fs::create_dir_all(f.parent().unwrap())?;
    let tmp = f.with_extension("dat.tmp");
    {
        let mut h = std::fs::File::create(&tmp)?;
        h.write_all(data)?;
        h.sync_all()?;
    }
    if f.exists() {
        std::fs::copy(&f, f.with_extension("dat.bak"))?;
    }
    std::fs::rename(&tmp, &f)
}

/// SceSaveDataMemoryData {buf*, bufSize, offset, reserved[40]}.
unsafe fn sdmem_data(p: *const u8) -> (*mut u8, usize, u64) {
    unsafe { (*(p as *const u64) as *mut u8, *(p.add(8) as *const u64) as usize, *(p.add(16) as *const i64) as u64) }
}

/// SceSaveDataIcon {buf*, bufSize, dataSize}: stores the icon's data.
unsafe fn store_icon(dir: &std::path::Path, icon: *const u64) {
    unsafe {
        let (buf, size) = (*icon as *const u8, *icon.add(2) as usize);
        if !buf.is_null() && size > 0 {
            let _ = std::fs::create_dir_all(dir.join("sce_sys"));
            let _ = std::fs::write(dir.join("sce_sys").join("icon0.png"), std::slice::from_raw_parts(buf, size));
        }
    }
}

unsafe fn store_param(dir: &std::path::Path, param: *const SaveParam) {
    unsafe {
        let mut p: SaveParam = std::ptr::read_unaligned(param);
        p.mtime = (crate::hle::libs::rtc::now_tick() / 1_000_000) as i64 - 62_135_596_800;
        let bytes = std::slice::from_raw_parts(&p as *const _ as *const u8, std::mem::size_of::<SaveParam>());
        let _ = std::fs::create_dir_all(dir.join("sce_sys"));
        let _ = std::fs::write(dir.join("sce_sys").join("param.bin"), bytes);
    }
}

/// Backup copies (sceSaveDataUmountWithBackup) live beside the saves:
/// `<user>/<title>/.backup/<dir>`.
fn backup_dir(save: &std::path::Path) -> Option<PathBuf> {
    let name = save.file_name()?;
    Some(save.parent()?.join(".backup").join(name))
}

fn copy_tree(from: &std::path::Path, to: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for e in std::fs::read_dir(from)? {
        let e = e?;
        let dst = to.join(e.file_name());
        if e.file_type()?.is_dir() {
            copy_tree(&e.path(), &dst)?;
        } else {
            std::fs::copy(e.path(), &dst)?;
        }
    }
    Ok(())
}

/// Replaces `to` with a copy of `from`, staging it first so an interrupted copy
/// never leaves `to` half-written.
fn replace_tree(from: &std::path::Path, to: &std::path::Path) -> std::io::Result<()> {
    let stage = to.with_extension("staging");
    let old = to.with_extension("old");
    let _ = std::fs::remove_dir_all(&stage);
    let _ = std::fs::remove_dir_all(&old);
    copy_tree(from, &stage)?;
    if to.exists() {
        std::fs::rename(to, &old)?;
    }
    std::fs::rename(&stage, to)?;
    let _ = std::fs::remove_dir_all(&old);
    Ok(())
}

/// Shared by sceSaveDataMount / sceSaveDataMount2.
unsafe fn mount_dir(user: i32, title: *const u8, dir: *const u8, blocks: u64, mode: u32, result: *mut u8) -> i32 {
    if dir.is_null() || result.is_null() {
        return ERR_PARAMETER;
    }
    let name = unsafe { cstr(dir) }.to_string();
    if !valid_dir_name(&name) {
        return ERR_PARAMETER;
    }
    let path = user_root(user, title).join(&name);
    let exists = path.join("sce_sys").exists() || path.exists();
    let mut created = false;
    if !exists {
        if mode & (MOUNT_CREATE | MOUNT_CREATE2) == 0 {
            crate::info!("sceSaveDataMount '{name}' mode {mode:#x}: not found");
            return ERR_NOT_FOUND;
        }
        if std::fs::create_dir_all(path.join("sce_sys")).is_err() {
            return ERR_INTERNAL;
        }
        created = true;
    } else if mode & MOUNT_CREATE != 0 && mode & MOUNT_CREATE2 == 0 {
        return ERR_EXISTS;
    }
    let mut mounts = MOUNTS.lock();
    let Some(slot) = mounts.iter().position(|s| s.is_none()) else { return ERR_MOUNT_FULL };
    mounts[slot] = Some(path.clone());
    let mp = format!("/savedata{slot}");
    crate::hle::kernel::fs::mount(&mp, vec![path], mode & MOUNT_RDONLY == 0);
    unsafe {
        std::ptr::write_bytes(result, 0, 64);
        crate::hle::write_cstr(result, 16, &mp);
        *(result.add(16) as *mut u64) = blocks;
        *(result.add(28) as *mut u32) = if created { 1 } else { 0 };
    }
    crate::info!("sceSaveDataMount '{name}' mode {mode:#x} blocks {blocks} -> {mp}{}", if created { " (created)" } else { "" });
    0
}

sysv! {
    /// {user, -, title*, dir*, fingerprint*, blocks @32, mode @40}
    fn sceSaveDataMount(m: *const u8, result: *mut u8) -> i32 {
        if m.is_null() { return ERR_PARAMETER; }
        mount_dir(*(m as *const i32), *(m.add(8) as *const u64) as *const u8, *(m.add(16) as *const u64) as *const u8, *(m.add(32) as *const u64), *(m.add(40) as *const u32), result)
    }
    /// {user, -, dir* @8, blocks @16, mode @24} (the title's own saves)
    fn sceSaveDataMount2(m: *const u8, result: *mut u8) -> i32 {
        if m.is_null() { return ERR_PARAMETER; }
        mount_dir(*(m as *const i32), std::ptr::null(), *(m.add(8) as *const u64) as *const u8, *(m.add(16) as *const u64), *(m.add(24) as *const u32), result)
    }
    fn sceSaveDataUmount(mp: *const u8) -> i32 {
        let Some(i) = mount_point_index(mp) else { return ERR_PARAMETER };
        let mut mounts = MOUNTS.lock();
        if mounts[i].take().is_none() { return ERR_NOT_MOUNTED; }
        crate::hle::kernel::fs::unmount(cstr(mp));
        0
    }
    /// Unmounts and keeps a backup copy of the (now consistent) save directory.
    fn sceSaveDataUmountWithBackup(mp: *const u8) -> i32 {
        let Some(i) = mount_point_index(mp) else { return ERR_PARAMETER };
        let Some(dir) = MOUNTS.lock()[i].take() else { return ERR_NOT_MOUNTED };
        crate::hle::kernel::fs::unmount(cstr(mp));
        let Some(bak) = backup_dir(&dir) else { return ERR_INTERNAL };
        match replace_tree(&dir, &bak) {
            Ok(()) => { crate::info!("save backup written: {}", bak.display()); 0 }
            Err(e) => { crate::error!("save backup of {} failed: {e}", dir.display()); ERR_INTERNAL }
        }
    }
    /// {user, -, title*, dir*, param*, icon*}: whether a backup exists (and its parameters).
    fn sceSaveDataCheckBackupData(c: *const u8) -> i32 {
        if c.is_null() { return ERR_PARAMETER; }
        let dir = *(c.add(16) as *const u64) as *const u8;
        if dir.is_null() || !valid_dir_name(cstr(dir)) { return ERR_PARAMETER; }
        let save = user_root(*(c as *const i32), *(c.add(8) as *const u64) as *const u8).join(cstr(dir));
        let Some(bak) = backup_dir(&save).filter(|b| b.is_dir()) else { return ERR_NOT_FOUND };
        let param = *(c.add(24) as *const u64) as *mut SaveParam;
        if !param.is_null() { std::ptr::write_unaligned(param, read_save_param(&bak)); }
        let icon = *(c.add(32) as *const u64) as *mut u64;
        if !icon.is_null() {
            // {buf*, bufSize, dataSize}
            let (buf, cap) = (*icon as *mut u8, *icon.add(1) as usize);
            let data = std::fs::read(bak.join("sce_sys").join("icon0.png")).unwrap_or_default();
            let n = data.len().min(cap);
            if !buf.is_null() { std::ptr::copy_nonoverlapping(data.as_ptr(), buf, n); }
            *icon.add(2) = n as u64;
        }
        0
    }
    /// {user, -, title*, dir*, fingerprint*}: replaces the save with its backup.
    fn sceSaveDataRestoreBackupData(c: *const u8) -> i32 {
        if c.is_null() { return ERR_PARAMETER; }
        let dir = *(c.add(16) as *const u64) as *const u8;
        if dir.is_null() || !valid_dir_name(cstr(dir)) { return ERR_PARAMETER; }
        let save = user_root(*(c as *const i32), *(c.add(8) as *const u64) as *const u8).join(cstr(dir));
        if MOUNTS.lock().iter().flatten().any(|m| *m == save) { return 0x809F_0002u32 as i32; } // busy
        let Some(bak) = backup_dir(&save).filter(|b| b.is_dir()) else { return ERR_NOT_FOUND };
        match replace_tree(&bak, &save) {
            Ok(()) => { crate::info!("save restored from backup: {}", save.display()); 0 }
            Err(e) => { crate::error!("restoring {} failed: {e}", save.display()); ERR_INTERNAL }
        }
    }
    /// {blocks, freeBlocks}: a fixed-size quota with the used space subtracted.
    fn sceSaveDataGetMountInfo(mp: *const u8, info: *mut u64) -> i32 {
        let Some(dir) = mounted_dir(mp) else { return ERR_NOT_MOUNTED };
        if info.is_null() { return ERR_PARAMETER; }
        fn used(p: &std::path::Path) -> u64 {
            std::fs::read_dir(p).map(|rd| rd.flatten().map(|e| match e.file_type() {
                Ok(t) if t.is_dir() => used(&e.path()),
                _ => e.metadata().map(|m| m.len().div_ceil(32768)).unwrap_or(0),
            }).sum()).unwrap_or(0)
        }
        let blocks = 1 << 15; // 1 GiB of 32 KiB blocks
        *info = blocks;
        *info.add(1) = blocks.saturating_sub(used(&dir));
        0
    }
    fn sceSaveDataDirNameSearch(cond: *const u8, result: *mut u8) -> i32 {
        if cond.is_null() || result.is_null() { return ERR_PARAMETER; }
        let user = *(cond as *const i32);
        let title = *(cond.add(8) as *const u64) as *const u8;
        let pattern = *(cond.add(16) as *const u64) as *const u8;
        let pat = if pattern.is_null() { String::new() } else { cstr(pattern).to_string() };
        let root = user_root(user, title);
        let mut names: Vec<String> = std::fs::read_dir(&root)
            .map(|rd| rd.flatten().filter(|e| e.path().is_dir()).map(|e| e.file_name().to_string_lossy().into_owned()).collect())
            .unwrap_or_default();
        // (Not the save-data memory file, backups or staging copies.)
        names.retain(|n| n != "sdmemory" && valid_dir_name(n));
        if !pat.is_empty() {
            let prefix = pat.trim_end_matches('%').to_string();
            let wildcard = pat.ends_with('%');
            names.retain(|n| if wildcard { n.starts_with(&prefix) } else { *n == pat });
        }
        names.sort();
        let dir_names = *(result.add(8) as *const u64) as *mut u8;
        let cap = *(result.add(16) as *const u32) as usize;
        let params = *(result.add(24) as *const u64) as *mut SaveParam;
        let set = names.len().min(cap);
        for (i, n) in names.iter().take(set).enumerate() {
            if !dir_names.is_null() { crate::hle::write_cstr(dir_names.add(i * 32), 32, n); }
            if !params.is_null() {
                std::ptr::write_unaligned(params.add(i), read_save_param(&root.join(n)));
            }
        }
        *(result as *mut u32) = names.len() as u32;
        *(result.add(20) as *mut u32) = set as u32;
        crate::info!("sceSaveDataDirNameSearch '{pat}': {} hit(s)", names.len());
        0
    }
    fn sceSaveDataSetParam(mp: *const u8, kind: u32, buf: *const u8, size: u64) -> i32 {
        let Some(dir) = mounted_dir(mp) else { return ERR_NOT_MOUNTED };
        let file = dir.join("sce_sys").join("param.bin");
        let mut p = read_save_param(&dir);
        match kind {
            0 if size as usize >= std::mem::size_of::<SaveParam>() => std::ptr::copy_nonoverlapping(buf, &mut p as *mut _ as *mut u8, std::mem::size_of::<SaveParam>()),
            1 => crate::hle::write_cstr(p.title.as_mut_ptr(), 128, cstr(buf)),
            2 => crate::hle::write_cstr(p.sub_title.as_mut_ptr(), 128, cstr(buf)),
            3 => crate::hle::write_cstr(p.detail.as_mut_ptr(), 1024, cstr(buf)),
            4 => p.user_param = *(buf as *const u32),
            _ => return ERR_PARAMETER,
        }
        p.mtime = (crate::hle::libs::rtc::now_tick() / 1_000_000) as i64 - 62_135_596_800;
        let bytes = std::slice::from_raw_parts(&p as *const _ as *const u8, std::mem::size_of::<SaveParam>());
        let _ = std::fs::create_dir_all(dir.join("sce_sys"));
        if std::fs::write(&file, bytes).is_err() { return ERR_INTERNAL; }
        0
    }
    fn sceSaveDataSaveIcon(mp: *const u8, icon: *const u64) -> i32 {
        let Some(dir) = mounted_dir(mp) else { return ERR_NOT_MOUNTED };
        if icon.is_null() { return ERR_PARAMETER; }
        let buf = *icon as *const u8;
        let size = *icon.add(2) as usize;
        if buf.is_null() || size == 0 { return ERR_PARAMETER; }
        let _ = std::fs::create_dir_all(dir.join("sce_sys"));
        if std::fs::write(dir.join("sce_sys").join("icon0.png"), std::slice::from_raw_parts(buf, size)).is_err() { return ERR_INTERNAL; }
        0
    }
    fn sceSaveDataDelete(del: *const u8) -> i32 {
        if del.is_null() { return ERR_PARAMETER; }
        let user = *(del as *const i32);
        let title = *(del.add(8) as *const u64) as *const u8;
        let dir = *(del.add(16) as *const u64) as *const u8;
        if dir.is_null() { return ERR_PARAMETER; }
        let name = cstr(dir).to_string();
        if !valid_dir_name(&name) { return ERR_PARAMETER; }
        let path = user_root(user, title).join(&name);
        if !path.exists() { return ERR_NOT_FOUND; }
        crate::info!("sceSaveDataDelete '{name}'");
        if std::fs::remove_dir_all(&path).is_err() { return ERR_INTERNAL; }
        0
    }
    fn sceSaveDataSetupSaveDataMemory(user: i32, size: u64, param: *const u8) -> i32 {
        let f = memory_file(user);
        let _ = std::fs::create_dir_all(f.parent().unwrap());
        let cur = std::fs::metadata(&f).map(|m| m.len()).unwrap_or(0);
        if cur < size {
            let file = match std::fs::OpenOptions::new().create(true).write(true).truncate(false).open(&f) { Ok(x) => x, Err(_) => return ERR_INTERNAL };
            if file.set_len(size).is_err() { return ERR_INTERNAL; }
        }
        crate::info!("sceSaveDataSetupSaveDataMemory user {user} size {size:#x}");
        0
    }
    fn sceSaveDataGetSaveDataMemory(user: i32, buf: *mut u8, size: u64, offset: i64) -> i32 {
        use std::os::windows::fs::FileExt;
        let f = match std::fs::File::open(memory_file(user)) { Ok(f) => f, Err(_) => return ERR_NOT_FOUND };
        let out = std::slice::from_raw_parts_mut(buf, size as usize);
        out.fill(0);
        let _ = f.seek_read(out, offset as u64);
        0
    }
    fn sceSaveDataSetSaveDataMemory(user: i32, buf: *const u8, size: u64, offset: i64) -> i32 {
        use std::os::windows::fs::FileExt;
        let f = match std::fs::OpenOptions::new().write(true).open(memory_file(user)) { Ok(f) => f, Err(_) => return ERR_NOT_FOUND };
        crate::gpu::os_access(buf as u64, size);
        if f.seek_write(std::slice::from_raw_parts(buf, size as usize), offset as u64).is_err() { return ERR_INTERNAL; }
        0
    }
    /// SceSaveDataMemorySetup2 {option u32, user i32, memorySize, iconMemorySize,
    /// initParam*, initIcon*, reserved}; result {existedMemorySize, reserved} (optional).
    fn sceSaveDataSetupSaveDataMemory2(setup: *const u8, result: *mut u64) -> i32 {
        if setup.is_null() { return ERR_PARAMETER; }
        let option = *(setup as *const u32);
        let user = *(setup.add(4) as *const i32);
        let size = *(setup.add(8) as *const u64) as usize;
        let icon_size = *(setup.add(16) as *const u64);
        let init_param = *(setup.add(24) as *const u64) as *const SaveParam;
        let init_icon = *(setup.add(32) as *const u64) as *const u64;
        if size == 0 || size > 16 << 20 { return ERR_PARAMETER; }
        let existing = sdmem_load(user);
        let existed = existing.as_ref().map_or(0, |d| d.len());
        let mut data = existing.unwrap_or_default();
        let grow = data.len() < size;
        if grow { data.resize(size, 0); }
        let dir = sdmem_dir(user);
        if existed == 0 || grow {
            if let Err(e) = sdmem_store(user, &data) {
                crate::error!("save-data memory: writing {} failed: {e}", memory_file(user).display());
                return ERR_INTERNAL;
            }
        }
        if existed == 0 {
            if option & 1 != 0 && !init_param.is_null() { store_param(&dir, init_param); }
            if !init_icon.is_null() { store_icon(&dir, init_icon); }
        }
        crate::info!("sceSaveDataSetupSaveDataMemory2 user {user} option {option:#x} size {size:#x} icon {icon_size:#x} existed {existed:#x} ({})", memory_file(user).display());
        *SDMEM.lock() = Some(SdMemory { user, data });
        if !result.is_null() { *result = existed as u64; }
        0
    }
    /// SceSaveDataMemoryGet2 {user i32, pad, data*, param*, icon*, reserved}.
    fn sceSaveDataGetSaveDataMemory2(get: *const u8) -> i32 {
        if get.is_null() { return ERR_PARAMETER; }
        let user = *(get as *const i32);
        let data = *(get.add(8) as *const u64) as *const u8;
        let param = *(get.add(16) as *const u64) as *mut SaveParam;
        let icon = *(get.add(24) as *const u64) as *mut u64;
        let g = SDMEM.lock();
        let Some(m) = g.as_ref().filter(|m| m.user == user) else { return ERR_NOT_FOUND };
        if !data.is_null() {
            let (buf, n, off) = sdmem_data(data);
            if buf.is_null() || off as usize + n > m.data.len() { return ERR_PARAMETER; }
            std::ptr::copy_nonoverlapping(m.data.as_ptr().add(off as usize), buf, n);
        }
        let dir = sdmem_dir(user);
        if !param.is_null() { std::ptr::write_unaligned(param, read_save_param(&dir)); }
        if !icon.is_null() {
            let (buf, cap) = (*icon as *mut u8, *icon.add(1) as usize);
            let d = std::fs::read(dir.join("sce_sys").join("icon0.png")).unwrap_or_default();
            let n = d.len().min(cap);
            if !buf.is_null() { std::ptr::copy_nonoverlapping(d.as_ptr(), buf, n); }
            *icon.add(2) = n as u64;
        }
        0
    }
    /// SceSaveDataMemorySet2 {user i32, pad, data*, param*, icon*, reserved}.
    fn sceSaveDataSetSaveDataMemory2(set: *const u8) -> i32 {
        if set.is_null() { return ERR_PARAMETER; }
        let user = *(set as *const i32);
        let data = *(set.add(8) as *const u64) as *const u8;
        let param = *(set.add(16) as *const u64) as *const SaveParam;
        let icon = *(set.add(24) as *const u64) as *const u64;
        let mut g = SDMEM.lock();
        let Some(m) = g.as_mut().filter(|m| m.user == user) else { return ERR_NOT_FOUND };
        if !data.is_null() {
            let (buf, n, off) = sdmem_data(data);
            if buf.is_null() || off as usize + n > m.data.len() { return ERR_PARAMETER; }
            std::ptr::copy_nonoverlapping(buf, m.data.as_mut_ptr().add(off as usize), n);
            if let Err(e) = sdmem_store(user, &m.data) {
                crate::error!("save-data memory: writing {} failed: {e}", memory_file(user).display());
                return ERR_INTERNAL;
            }
            crate::info!("save-data memory: {n:#x} bytes at {off:#x} written");
        }
        let dir = sdmem_dir(user);
        if !param.is_null() { store_param(&dir, param); }
        if !icon.is_null() { store_icon(&dir, icon); }
        0
    }
    /// SceSaveDataMemorySync {user i32, reserved}: the image is already on disk
    /// after every Set2; this rewrites it (atomically) to be sure.
    fn sceSaveDataSyncSaveDataMemory(sync: *const u8) -> i32 {
        if sync.is_null() { return ERR_PARAMETER; }
        let user = *(sync as *const i32);
        let g = SDMEM.lock();
        let Some(m) = g.as_ref().filter(|m| m.user == user) else { return ERR_NOT_FOUND };
        match sdmem_store(user, &m.data) {
            Ok(()) => 0,
            Err(e) => { crate::error!("save-data memory sync failed: {e}"); ERR_INTERNAL }
        }
    }
}

fixed! {
    sceSaveDataInitialize => 0,
    sceSaveDataInitialize3 => 0,
    sceSaveDataTerminate => 0,
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceSaveData", [
        sceSaveDataMount, sceSaveDataUmount, sceSaveDataDirNameSearch, sceSaveDataSetParam, sceSaveDataSaveIcon,
        sceSaveDataDelete, sceSaveDataSetupSaveDataMemory, sceSaveDataGetSaveDataMemory, sceSaveDataSetSaveDataMemory,
        sceSaveDataInitialize, sceSaveDataInitialize3, sceSaveDataTerminate, sceSaveDataMount2,
        sceSaveDataUmountWithBackup, sceSaveDataCheckBackupData, sceSaveDataRestoreBackupData, sceSaveDataGetMountInfo,
        sceSaveDataSetupSaveDataMemory2, sceSaveDataGetSaveDataMemory2, sceSaveDataSetSaveDataMemory2,
        sceSaveDataSyncSaveDataMemory,
    ]);
}
