//! The game's save folder and its backups.
//!
//! The title keeps its progress in save-data memory, stored by the runtime
//! as `saves\<user>\<title id>\sdmemory\memory.dat` (CUSA03627 for the US
//! release, CUSA03745 for the EU one; + `memory.dat.bak`, the
//! previous write, and `sce_sys\param.bin`). Before it replaces the file the
//! runtime copies the previous one into `save-backups\<UTC stamp>\` (at
//! most one copy per few minutes, the newest `KEEP` kept); the launcher
//! lists those copies and restores one (backing up the current save first).

use std::path::{Path, PathBuf};
use std::time::SystemTime;

pub const TITLE: &str = "CUSA03627";
/// The titles whose saves are looked for (US, EU).
pub const TITLES: [&str; 2] = ["CUSA03627", "CUSA03745"];
/// Backups kept.
pub const KEEP: usize = 20;
/// The files of a save (relative to its sdmemory folder).
const FILES: &[&str] = &["memory.dat", "memory.dat.bak", "sce_sys/param.bin", "sce_sys/icon0.png"];

#[derive(Clone, Debug)]
pub struct SaveInfo {
    /// The sdmemory folder (a save) or the backup folder.
    pub dir: PathBuf,
    pub modified: Option<SystemTime>,
    /// The checkpoint's stage and scene, e.g. s02_TricosCave /
    /// s02_02_Care_Step2_001.
    pub stage: Option<String>,
    pub scene: Option<String>,
}

/// The sdmemory folders under `saves` (one per user id and title).
pub fn memory_dirs(saves: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(saves)
        .into_iter()
        .flatten()
        .flatten()
        .flat_map(|e| TITLES.map(|t| e.path().join(t).join("sdmemory")))
        .filter(|d| d.join("memory.dat").is_file() || d.join("memory.dat.bak").is_file())
        .collect();
    v.sort();
    v
}

/// The memory folder of the first user (`saves\1\...` unless only another
/// user has one); of its titles the one saved last (US and EU saves apart).
pub fn main_memory_dir(saves: &Path) -> PathBuf {
    let dirs = memory_dirs(saves);
    let user = |d: &Path| d.parent().and_then(Path::parent).map(Path::to_path_buf);
    let first = dirs.first().and_then(|d| user(d));
    let saved = |d: &Path| std::fs::metadata(d.join("memory.dat")).or_else(|_| std::fs::metadata(d.join("memory.dat.bak"))).and_then(|m| m.modified()).ok();
    dirs.iter().filter(|d| user(d) == first).max_by_key(|d| saved(d)).cloned().unwrap_or_else(|| saves.join("1").join(TITLE).join("sdmemory"))
}

fn lp_string(d: &[u8], at: usize) -> Option<(String, usize)> {
    let n = u32::from_le_bytes(d.get(at..at + 4)?.try_into().ok()?) as usize;
    if n == 0 || n > 256 {
        return None;
    }
    let s = d.get(at + 4..at + 4 + n)?;
    s.iter().all(|c| c.is_ascii_graphic()).then(|| (String::from_utf8_lossy(s).into_owned(), at + 4 + n))
}

/// The checkpoint (stage, scene) recorded in save-data memory: the
/// GameProgress block's symbol table holds "Stage", <stage>, "Scene",
/// <scene> as length-prefixed strings.
pub fn checkpoint(d: &[u8]) -> (Option<String>, Option<String>) {
    let find = |key: &[u8]| -> Option<String> {
        let mut pat = (key.len() as u32).to_le_bytes().to_vec();
        pat.extend_from_slice(key);
        let at = d.windows(pat.len()).position(|w| w == pat.as_slice())?;
        lp_string(d, at + pat.len()).map(|(s, _)| s)
    };
    (find(b"Stage"), find(b"Scene"))
}

fn info(dir: &Path, mem: &Path) -> SaveInfo {
    let data = std::fs::read(mem.join("memory.dat")).or_else(|_| std::fs::read(mem.join("memory.dat.bak"))).unwrap_or_default();
    let (stage, scene) = checkpoint(&data);
    let modified = std::fs::metadata(mem.join("memory.dat")).and_then(|m| m.modified()).ok();
    SaveInfo { dir: dir.to_path_buf(), modified, stage, scene }
}

/// The current save, if there is one.
pub fn current(saves: &Path) -> Option<SaveInfo> {
    let m = Some(main_memory_dir(saves)).filter(|d| d.join("memory.dat").is_file() || d.join("memory.dat.bak").is_file())?;
    Some(info(&m, &m))
}

/// Backups, newest first.
pub fn backups(backups: &Path) -> Vec<SaveInfo> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(backups)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("sdmemory").join("memory.dat").is_file())
        .collect();
    v.sort();
    v.reverse();
    v.iter().map(|d| info(d, &d.join("sdmemory"))).collect()
}

fn copy_files(from: &Path, to: &Path) -> std::io::Result<u32> {
    let mut n = 0;
    for f in FILES {
        let src = from.join(f);
        if src.is_file() {
            let dst = to.join(f);
            std::fs::create_dir_all(dst.parent().unwrap())?;
            std::fs::copy(&src, &dst)?;
            n += 1;
        }
    }
    Ok(n)
}

/// Copies the save in `mem` (an sdmemory folder) to `backups\<stamp>`; keeps
/// the newest KEEP backups.
pub fn backup(mem: &Path, backups_dir: &Path, stamp: &str) -> std::io::Result<PathBuf> {
    let mut dst = backups_dir.join(stamp);
    let mut k = 1;
    while dst.exists() {
        k += 1;
        dst = backups_dir.join(format!("{stamp}-{k}"));
    }
    let n = copy_files(mem, &dst.join("sdmemory"))?;
    if n == 0 {
        let _ = std::fs::remove_dir_all(&dst);
        return Err(std::io::Error::new(std::io::ErrorKind::NotFound, "no save files to back up"));
    }
    prune(backups_dir, KEEP);
    Ok(dst)
}

/// When the newest backup was made (by its folder's time).
pub fn newest_backup_time(backups_dir: &Path) -> Option<SystemTime> {
    std::fs::read_dir(backups_dir).ok()?.flatten().filter_map(|e| e.metadata().ok()?.modified().ok()).max()
}

pub fn prune(backups_dir: &Path, keep: usize) {
    let mut v: Vec<PathBuf> = std::fs::read_dir(backups_dir).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    v.sort();
    if v.len() > keep {
        let n = v.len() - keep;
        for d in v.into_iter().take(n) {
            let _ = std::fs::remove_dir_all(d);
        }
    }
}

/// Restores a backup over the current save (the current save is backed up
/// first, as `<stamp>-before-restore`).
pub fn restore(saves: &Path, backups_dir: &Path, backup_dir: &Path, stamp: &str) -> std::io::Result<()> {
    let mem = main_memory_dir(saves);
    if mem.join("memory.dat").is_file() {
        backup(&mem, backups_dir, &format!("{stamp}-before-restore"))?;
    }
    let src = backup_dir.join("sdmemory");
    if !src.join("memory.dat").is_file() {
        return Err(std::io::Error::new(std::io::ErrorKind::NotFound, "the backup has no memory.dat"));
    }
    // (The previous write's copy would otherwise outlive the restore.)
    let _ = std::fs::remove_file(mem.join("memory.dat.bak"));
    copy_files(&src, &mem)?;
    Ok(())
}

/// UTC time as `YYYYMMDD-HHMMSS`.
pub fn utc_stamp(t: SystemTime) -> String {
    let secs = t.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400);
    // Civil-from-days (Howard Hinnant).
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{:04}{:02}{:02}-{:02}{:02}{:02}", y, m, d, rem / 3600, (rem / 60) % 60, rem % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lp(s: &str) -> Vec<u8> {
        let mut v = (s.len() as u32).to_le_bytes().to_vec();
        v.extend_from_slice(s.as_bytes());
        v
    }

    #[test]
    fn checkpoint_from_symbol_table() {
        let mut d = vec![0u8; 64];
        for s in ["GameFlag", "Stage", "s02_TricosCave", "Scene", "s02_02_Care_Step2_001"] {
            d.extend(lp(s));
        }
        d.extend([0u8; 16]);
        assert_eq!(checkpoint(&d), (Some("s02_TricosCave".into()), Some("s02_02_Care_Step2_001".into())));
        assert_eq!(checkpoint(&[0u8; 32]), (None, None));
    }

    #[test]
    fn backup_restore_prune() {
        let tmp = std::env::temp_dir().join(format!("tlg-config-saves-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let saves = tmp.join("saves");
        let bk = tmp.join("save-backups");
        let mem = saves.join("1").join(TITLE).join("sdmemory");
        std::fs::create_dir_all(mem.join("sce_sys")).unwrap();
        std::fs::write(mem.join("memory.dat"), b"one").unwrap();
        std::fs::write(mem.join("sce_sys").join("param.bin"), b"p").unwrap();
        assert_eq!(main_memory_dir(&saves), mem);
        let b1 = backup(&mem, &bk, "20261015-120000").unwrap();
        std::fs::write(mem.join("memory.dat"), b"two").unwrap();
        let b2 = backup(&mem, &bk, "20261015-120000").unwrap();
        assert!(b2.ends_with("20261015-120000-2"));
        assert_eq!(backups(&bk).len(), 2);
        restore(&saves, &bk, &b1, "20261015-130000").unwrap();
        assert_eq!(std::fs::read(mem.join("memory.dat")).unwrap(), b"one");
        let list = backups(&bk);
        assert_eq!(list.len(), 3);
        assert_eq!(std::fs::read(list[0].dir.join("sdmemory").join("memory.dat")).unwrap(), b"two");
        prune(&bk, 1);
        assert_eq!(backups(&bk).len(), 1);
        let _ = std::fs::remove_dir_all(&tmp);
        assert_eq!(utc_stamp(std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_760_529_600)), "20251015-120000");
    }
}
