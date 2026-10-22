//! Updates: a new version unzipped over the old one or into another folder
//! carries on with its game folders, settings, key bindings, saves and caches.
//!
//! The installed layout keeps them in `%APPDATA%\TLG` and `%LOCALAPPDATA%\TLG`,
//! which every version reads: nothing to carry. A portable install keeps them
//! in its own folder (`user` beside `portable.txt`), so a new version unzipped
//! into another folder would start empty: at its first start (an empty
//! profile) the launcher looks for the most recently used earlier profile - a
//! portable install in a folder beside this one, or the installed profile
//! when this one is portable - and copies it (`carry_over`); a portable one's
//! `portable.txt` comes along. Game files are never copied: folders left at
//! the earlier profile's default location are named in the settings instead.
//!
//! Game folders inside another version's folder (`other_version`) would go
//! with that folder: the launcher says so and can move them into its own
//! (`move_into`, a rename on the same drive).

use crate::paths::{BASE_DIR, Layout, PORTABLE_MARKER, Paths, UPDATE_DIR};
use crate::settings::Settings;
use std::io;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// The programs whose folder is a version of the port.
const PROGRAMS: [&str; 2] = ["tlg-launcher.exe", "tlg.exe"];

/// What a profile holds: (file or folder, what it is, in the user folder
/// rather than the cache folder).
const ITEMS: [(&str, &str, bool); 6] = [
    ("settings.txt", "settings", true),
    ("input.txt", "key bindings", true),
    ("saves", "saves", true),
    ("save-backups", "save backups", true),
    ("pipeline-cache.bin", "shader cache", false),
    ("pipeline-manifest.bin", "shader cache", false),
];

/// An earlier version's profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Earlier {
    /// The portable install's folder, or the installed profile's.
    pub dir: PathBuf,
    pub user: PathBuf,
    pub cache: PathBuf,
    /// Where its game folders are when the settings do not name them.
    pub game_root: PathBuf,
    pub portable: bool,
}

/// Whether a profile holds nothing yet (no settings, bindings or saves).
pub fn is_empty(paths: &Paths) -> bool {
    !paths.settings_file().exists() && !paths.input_file().exists() && std::fs::read_dir(paths.saves()).map_or(true, |mut d| d.next().is_none())
}

/// Whether `dir` holds a version of the port.
pub fn is_port_folder(dir: &Path) -> bool {
    PROGRAMS.iter().any(|p| dir.join(p).is_file())
}

/// The path with links resolved, without the `\\?\` prefix of a drive path
/// (it is shown to players).
fn canon(p: &Path) -> PathBuf {
    let c = std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    match c.to_str().and_then(|s| s.strip_prefix(r"\\?\")) {
        Some(s) if s.as_bytes().get(1) == Some(&b':') => PathBuf::from(s),
        _ => c,
    }
}

/// When a profile was last used: its settings or saves.
fn last_used(user: &Path) -> SystemTime {
    [user.join("settings.txt"), user.join("saves")].iter().filter_map(|p| p.metadata().and_then(|m| m.modified()).ok()).max().unwrap_or(SystemTime::UNIX_EPOCH)
}

/// The most recently used earlier profile for this version's empty one:
/// portable installs in the folders beside this one's, and the installed
/// profile when this one is portable. None for development checkouts and
/// `TLG_HOME`.
pub fn find_earlier(paths: &Paths) -> Option<Earlier> {
    if paths.layout == Layout::Dev || std::env::var_os("TLG_HOME").is_some_and(|h| !h.is_empty()) {
        return None;
    }
    find_earlier_in(paths, &Paths::installed(&paths.exe_dir))
}

fn find_earlier_in(paths: &Paths, installed: &Paths) -> Option<Earlier> {
    let mut found = Vec::new();
    let here = canon(&paths.exe_dir);
    if let Some(parent) = paths.exe_dir.parent() {
        for e in std::fs::read_dir(parent).into_iter().flatten().flatten() {
            let (dir, user) = (e.path(), e.path().join("user"));
            if dir.join(PORTABLE_MARKER).is_file() && user.join("settings.txt").is_file() && is_port_folder(&dir) && canon(&dir) != here {
                found.push(Earlier { game_root: user.join("game"), cache: user.clone(), user, dir, portable: true });
            }
        }
    }
    if paths.layout == Layout::Portable && installed.settings_file().is_file() {
        found.push(Earlier { dir: installed.user.clone(), user: installed.user.clone(), cache: installed.cache.clone(), game_root: installed.game_root.clone(), portable: false });
    }
    found.into_iter().max_by_key(|e| last_used(&e.user))
}

fn copy_tree(src: &Path, dst: &Path) -> io::Result<()> {
    let t = std::fs::symlink_metadata(src)?.file_type();
    if t.is_dir() {
        std::fs::create_dir_all(dst)?;
        for e in std::fs::read_dir(src)? {
            let e = e?;
            copy_tree(&e.path(), &dst.join(e.file_name()))?;
        }
    } else if t.is_file() {
        std::fs::copy(src, dst)?;
    }
    Ok(())
}

/// Copies an earlier profile into this version's empty one (nothing there is
/// replaced), with a portable install's mode. Returns this version's paths
/// afterwards and what was copied.
pub fn carry_over(paths: &Paths, from: &Earlier) -> io::Result<(Paths, Vec<&'static str>)> {
    let mut paths = paths.clone();
    if from.portable && paths.layout == Layout::User {
        std::fs::write(paths.exe_dir.join(PORTABLE_MARKER), "")?;
        paths = Paths::for_exe(&paths.exe_dir.join("tlg-launcher.exe"));
    }
    let mut copied = Vec::new();
    for (name, what, user) in ITEMS {
        let (src, dst) = if user { (&from.user, &paths.user) } else { (&from.cache, &paths.cache) };
        let (s, d) = (src.join(name), dst.join(name));
        if s.exists() && !d.exists() {
            std::fs::create_dir_all(dst)?;
            copy_tree(&s, &d)?;
            if !copied.contains(&what) {
                copied.push(what);
            }
        }
    }
    // (Game folders at the earlier profile's default location stay there.)
    let (mut s, _) = Settings::load(&paths.settings_file());
    let (base, update) = (from.game_root.join(BASE_DIR), from.game_root.join(UPDATE_DIR));
    if s.game_dir.is_none() && base.join("eboot.bin").is_file() && !paths.default_base().join("eboot.bin").is_file() {
        s.game_dir = Some(base);
        s.update_dir = update.join("eboot.bin").is_file().then_some(update);
        s.save(&paths.settings_file())?;
        copied.push("game folders");
    }
    Ok((paths, copied))
}

/// The folder of another version of the port that holds `dir` (a game
/// folder): deleting that folder would take the game files with it.
pub fn other_version(dir: &Path, exe_dir: &Path) -> Option<PathBuf> {
    let (dir, here) = (canon(dir), canon(exe_dir));
    let port = dir.ancestors().skip(1).find(|a| is_port_folder(a))?;
    (port != here).then(|| port.to_path_buf())
}

/// Moves a game folder into `exe_dir` (a rename: on the same drive it takes
/// a moment, another drive is refused); returns its new path.
pub fn move_into(dir: &Path, exe_dir: &Path) -> io::Result<PathBuf> {
    let name = dir.file_name().ok_or_else(|| io::Error::other(format!("{}: no folder name", dir.display())))?;
    let to = exe_dir.join(name);
    if to.exists() {
        return Err(io::Error::new(io::ErrorKind::AlreadyExists, format!("{} exists already", to.display())));
    }
    std::fs::rename(dir, &to)?;
    Ok(to)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("tlg-config-carry-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn put(p: &Path, text: &str) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, text).unwrap();
    }

    fn portable(dir: &Path) -> Paths {
        Paths { layout: Layout::Portable, exe_dir: dir.to_path_buf(), user: dir.join("user"), cache: dir.join("user"), game_root: dir.join("user").join("game") }
    }

    fn installed(root: &Path, exe_dir: &Path) -> Paths {
        Paths { layout: Layout::User, exe_dir: exe_dir.to_path_buf(), user: root.join("Roaming"), cache: root.join("Local"), game_root: root.join("Local").join("game") }
    }

    /// A portable 0.6.0 beside a new version unzipped into its own folder.
    #[test]
    fn portable_beside_carries_over() {
        let t = tmp("portable");
        let old = t.join("TLG-Port-0.6.0");
        put(&old.join("tlg-launcher.exe"), "");
        put(&old.join(PORTABLE_MARKER), "");
        put(&old.join("user").join("settings.txt"), "frame_cap = 60\n");
        put(&old.join("user").join("input.txt"), "walk = Alt\n");
        put(&old.join("user").join("saves").join("1").join("CUSA03627").join("sdmemory").join("memory.dat"), "save");
        put(&old.join("user").join("pipeline-cache.bin"), "cache");
        put(&old.join("user").join("game").join(BASE_DIR).join("eboot.bin"), "");
        put(&old.join("user").join("game").join(UPDATE_DIR).join("eboot.bin"), "");
        // (Another port's folder without a profile, and a stray folder.)
        put(&t.join("other").join("tlg.exe"), "");
        std::fs::create_dir_all(t.join("downloads")).unwrap();
        let new = t.join("TLG-Port-0.6.1");
        put(&new.join("tlg-launcher.exe"), "");
        let empty = installed(&t.join("appdata"), &new);
        assert!(is_empty(&empty));
        let e = find_earlier_in(&empty, &empty).expect("the portable 0.6.0");
        assert_eq!((e.dir.clone(), e.portable), (old.clone(), true));
        let (p, what) = carry_over(&empty, &e).unwrap();
        if std::env::var_os("TLG_HOME").is_none() {
            assert_eq!(p.layout, Layout::Portable);
            assert!(new.join(PORTABLE_MARKER).is_file());
        }
        assert_eq!(what, ["settings", "key bindings", "saves", "shader cache", "game folders"]);
        let (s, _) = Settings::load(&p.settings_file());
        assert_eq!(s.frame_cap, crate::settings::FrameCap::Fps60);
        // (The game files stay in the old folder: named, not copied.)
        assert_eq!(s.game_dir, Some(old.join("user").join("game").join(BASE_DIR)));
        assert_eq!(s.update_dir, Some(old.join("user").join("game").join(UPDATE_DIR)));
        assert!(!p.game_root.join(BASE_DIR).exists());
        assert_eq!(std::fs::read_to_string(p.saves().join("1").join("CUSA03627").join("sdmemory").join("memory.dat")).unwrap(), "save");
        assert_eq!(std::fs::read_to_string(p.input_file()).unwrap(), "walk = Alt\n");
        assert!(!is_empty(&p));
        assert_eq!(other_version(s.game_dir.as_deref().unwrap(), &new), Some(canon(&old)));
        let _ = std::fs::remove_dir_all(&t);
    }

    /// Nothing to carry: the installed profile is every version's; a used
    /// profile is never replaced.
    #[test]
    fn installed_and_used_profiles_stay() {
        let t = tmp("installed");
        let new = t.join("TLG-Port-0.6.1");
        put(&new.join("tlg-launcher.exe"), "");
        let inst = installed(&t.join("appdata"), &new);
        put(&inst.settings_file(), "volume = 50\n");
        assert!(!is_empty(&inst));
        assert_eq!(find_earlier_in(&inst, &inst), None);
        // (A portable copy of this version: the installed profile is its
        // earlier one.)
        let p = portable(&new);
        let e = find_earlier_in(&p, &inst).expect("the installed profile");
        assert!(!e.portable);
        let (q, what) = carry_over(&p, &e).unwrap();
        assert_eq!(q.layout, Layout::Portable);
        assert_eq!(what, ["settings"]);
        assert_eq!(std::fs::read_to_string(q.settings_file()).unwrap(), "volume = 50\n");
        // (Carried once: the profile is in use now.)
        put(&inst.settings_file(), "volume = 10\n");
        let (_, again) = carry_over(&q, &e).unwrap();
        assert!(again.is_empty());
        assert_eq!(std::fs::read_to_string(q.settings_file()).unwrap(), "volume = 50\n");
        let _ = std::fs::remove_dir_all(&t);
    }

    /// Game folders in another version's folder move into this one's.
    #[test]
    fn game_folders_move_in() {
        let t = tmp("move");
        let (old, new) = (t.join("old"), t.join("new"));
        put(&old.join("tlg.exe"), "");
        put(&new.join("tlg-launcher.exe"), "");
        let game = old.join("games").join(BASE_DIR);
        put(&game.join("eboot.bin"), "x");
        assert_eq!(other_version(&game, &new), Some(canon(&old)));
        assert_eq!(other_version(&game, &old), None);
        assert_eq!(other_version(&t.join("elsewhere").join(BASE_DIR), &new), None);
        let to = move_into(&game, &new).unwrap();
        assert_eq!(to, new.join(BASE_DIR));
        assert_eq!(std::fs::read_to_string(to.join("eboot.bin")).unwrap(), "x");
        assert!(!game.exists());
        assert_eq!(other_version(&to, &new), None);
        // (Never over an existing folder.)
        put(&old.join(BASE_DIR).join("eboot.bin"), "y");
        assert_eq!(move_into(&old.join(BASE_DIR), &new).unwrap_err().kind(), io::ErrorKind::AlreadyExists);
        let _ = std::fs::remove_dir_all(&t);
    }
}
