//! Where a build keeps its files.
//!
//! - Development checkout (the executable under a directory holding `local\`
//!   or the workspace `Cargo.toml`; in a release build `local\`, `Cargo.toml`
//!   and `crates\` together): everything in `<root>\local` (git-ignored),
//!   runs in `local\runs`.
//! - Portable (`portable.txt` beside the executable, or `TLG_HOME=<dir>`):
//!   everything in `<exe dir>\user` (or `<dir>`).
//! - Installed release (no checkout around it, or `tlg-release.txt` beside
//!   the executable, as packages up to 0.5.0 had): settings, bindings and
//!   saves in `%APPDATA%\TLG`,
//!   caches and logs in `%LOCALAPPDATA%\TLG`; game folders not set in the
//!   settings are looked for in `%LOCALAPPDATA%\TLG\game`.

use std::path::{Path, PathBuf};

pub const RELEASE_MARKER: &str = "tlg-release.txt";
pub const PORTABLE_MARKER: &str = "portable.txt";
/// The base application's and the update's folder names under `game_root`.
pub const BASE_DIR: &str = "CUSA03627";
pub const UPDATE_DIR: &str = "CUSA03627-patch";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    Dev,
    Portable,
    User,
}

#[derive(Clone, Debug)]
pub struct Paths {
    pub layout: Layout,
    /// The executable's directory.
    pub exe_dir: PathBuf,
    /// Settings, bindings, saves and their backups.
    pub user: PathBuf,
    /// Pipeline cache, the guest's runtime directories, logs.
    pub cache: PathBuf,
    /// Where game folders not set in the settings are looked for.
    pub game_root: PathBuf,
}

/// A release build (`TLG_RELEASE_BUILD` set while tools/package.ps1 builds
/// it): a checkout around the executable counts only as a whole - `local\`,
/// the workspace `Cargo.toml` and `crates\` - so a player's folder under one
/// that happens to hold a `local` folder or a `Cargo.toml` stays an installed
/// release without a marker file in the package.
const RELEASE_BUILD: bool = option_env!("TLG_RELEASE_BUILD").is_some();

fn dev_root(dir: &Path, strict: bool) -> Option<PathBuf> {
    let mut p = Some(dir);
    while let Some(d) = p {
        let (local, cargo) = (d.join("local").is_dir(), d.join("Cargo.toml").is_file());
        if if strict { local && cargo && d.join("crates").is_dir() } else { local || cargo } {
            return Some(d.to_path_buf());
        }
        p = d.parent();
    }
    None
}

impl Paths {
    pub fn for_exe(exe: &Path) -> Paths {
        let exe_dir = exe.parent().map(Path::to_path_buf).unwrap_or_default();
        let portable = |dir: PathBuf| Paths { layout: Layout::Portable, exe_dir: exe_dir.clone(), user: dir.clone(), cache: dir.clone(), game_root: dir.join("game") };
        if let Some(home) = std::env::var_os("TLG_HOME").filter(|h| !h.is_empty()) {
            return portable(PathBuf::from(home));
        }
        if exe_dir.join(PORTABLE_MARKER).is_file() {
            return portable(exe_dir.join("user"));
        }
        if !exe_dir.join(RELEASE_MARKER).is_file() {
            if let Some(root) = dev_root(&exe_dir, RELEASE_BUILD) {
                let local = root.join("local");
                return Paths { layout: Layout::Dev, exe_dir, user: local.clone(), cache: local.clone(), game_root: local.join("game") };
            }
        }
        let env_dir = |name: &str, fallback: &Path| std::env::var_os(name).map(PathBuf::from).unwrap_or_else(|| fallback.to_path_buf());
        let user = env_dir("APPDATA", &exe_dir).join("TLG");
        let cache = env_dir("LOCALAPPDATA", &exe_dir).join("TLG");
        Paths { layout: Layout::User, exe_dir, user, game_root: cache.join("game"), cache }
    }

    pub fn current() -> Paths {
        Paths::for_exe(&std::env::current_exe().unwrap_or_default())
    }

    pub fn settings_file(&self) -> PathBuf {
        self.user.join("settings.txt")
    }

    pub fn input_file(&self) -> PathBuf {
        self.user.join("input.txt")
    }

    pub fn saves(&self) -> PathBuf {
        self.user.join("saves")
    }

    /// Copies of earlier saves (`saves::backup`), newest last by name.
    pub fn save_backups(&self) -> PathBuf {
        self.user.join("save-backups")
    }

    /// One directory per run (log, crash report, screenshots).
    pub fn logs(&self) -> PathBuf {
        match self.layout {
            Layout::Dev => self.cache.join("runs"),
            _ => self.cache.join("logs"),
        }
    }

    /// How many run directories to keep.
    pub fn logs_kept(&self) -> usize {
        match self.layout {
            Layout::Dev => 30,
            _ => 10,
        }
    }

    pub fn runtime(&self) -> PathBuf {
        self.cache.join("runtime")
    }

    pub fn pipeline_cache(&self) -> PathBuf {
        self.cache.join("pipeline-cache.bin")
    }

    pub fn default_base(&self) -> PathBuf {
        self.game_root.join(BASE_DIR)
    }

    pub fn default_update(&self) -> PathBuf {
        self.game_root.join(UPDATE_DIR)
    }

    /// Marks a running game (crash detection: a marker left behind means the
    /// previous run did not exit cleanly).
    pub fn running_marker(&self) -> PathBuf {
        self.cache.join("running.txt")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts() {
        let tmp = std::env::temp_dir().join(format!("tlg-config-paths-{}", std::process::id()));
        let dev = tmp.join("checkout");
        std::fs::create_dir_all(dev.join("local")).unwrap();
        std::fs::create_dir_all(dev.join("target").join("release")).unwrap();
        let exe = dev.join("target").join("release").join("tlg.exe");
        // (TLG_HOME would override every layout.)
        if std::env::var_os("TLG_HOME").is_none() {
            let p = Paths::for_exe(&exe);
            assert_eq!(p.layout, Layout::Dev);
            assert_eq!(p.user, dev.join("local"));
            assert_eq!(p.logs(), dev.join("local").join("runs"));
            assert_eq!(p.default_base(), dev.join("local").join("game").join(BASE_DIR));

            let rel = dev.join("dist").join("TLG");
            std::fs::create_dir_all(&rel).unwrap();
            std::fs::write(rel.join(RELEASE_MARKER), "").unwrap();
            let p = Paths::for_exe(&rel.join("tlg.exe"));
            assert_eq!(p.layout, Layout::User);
            assert!(p.user.ends_with("TLG") && p.cache.ends_with("TLG"));

            std::fs::write(rel.join(PORTABLE_MARKER), "").unwrap();
            let p = Paths::for_exe(&rel.join("tlg.exe"));
            assert_eq!(p.layout, Layout::Portable);
            assert_eq!(p.user, rel.join("user"));
            assert_eq!(p.logs(), rel.join("user").join("logs"));
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn strict_checkout() {
        let tmp = std::env::temp_dir().join(format!("tlg-config-strict-{}", std::process::id()));
        let game = tmp.join("Games").join("TLG-Port");
        std::fs::create_dir_all(&game).unwrap();
        // (A stray local folder above a player's folder.)
        std::fs::create_dir_all(tmp.join("local")).unwrap();
        assert_eq!(dev_root(&game, false), Some(tmp.clone()));
        assert_eq!(dev_root(&game, true), None);
        std::fs::write(tmp.join("Cargo.toml"), "").unwrap();
        std::fs::create_dir_all(tmp.join("crates")).unwrap();
        assert_eq!(dev_root(&game, true), Some(tmp.clone()));
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
