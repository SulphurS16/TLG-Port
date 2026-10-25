//! The build's paths and the player's settings (crate tlg-config), resolved
//! once at startup, and player-facing error messages.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
pub use tlg_config::paths::{Layout, Paths};
pub use tlg_config::settings::{DisplayMode, Filter, FrameCap, Hdr, NewShaders, Scaling, Settings, VSync};

static PATHS: OnceLock<Paths> = OnceLock::new();
static SETTINGS: OnceLock<Settings> = OnceLock::new();
static SAFE_MODE: AtomicBool = AtomicBool::new(false);
static DIALOGS: AtomicBool = AtomicBool::new(false);

/// Loads the settings for `paths`. Safe mode uses the defaults (keeping
/// only where the game files are) and leaves the files alone. Returns the
/// messages about lines that could not be used.
pub fn init(paths: Paths, safe_mode: bool, dialogs: bool) -> Vec<String> {
    SAFE_MODE.store(safe_mode, Ordering::Relaxed);
    DIALOGS.store(dialogs, Ordering::Relaxed);
    let file = paths.settings_file();
    let (mut s, errs) = Settings::load(&file);
    if safe_mode {
        s = Settings { game_dir: s.game_dir, update_dir: s.update_dir, ..Settings::default() };
    } else if !file.exists() {
        // (A commented file to edit; the launcher writes the same.)
        let _ = std::fs::create_dir_all(&paths.user);
        let _ = std::fs::write(&file, tlg_config::settings::TEMPLATE.replace('\n', "\r\n"));
    }
    PATHS.set(paths).ok();
    SETTINGS.set(s).ok();
    errs
}

pub fn paths() -> &'static Paths {
    PATHS.get().expect("config::init")
}

pub fn try_paths() -> Option<&'static Paths> {
    PATHS.get()
}

pub fn settings() -> &'static Settings {
    SETTINGS.get_or_init(Settings::default)
}

pub fn safe_mode() -> bool {
    SAFE_MODE.load(Ordering::Relaxed)
}

/// Player-facing message boxes (off in development runs, which are driven
/// by scripts).
pub fn dialogs() -> bool {
    DIALOGS.load(Ordering::Relaxed)
}

/// Rewrites one setting in settings.txt, e.g. the display mode after
/// Alt+Enter (not in safe mode).
pub fn persist(key: &str, value: &str) {
    if safe_mode() {
        return;
    }
    let Some(p) = try_paths() else { return };
    let file = p.settings_file();
    match tlg_config::kv::KvFile::load(&file) {
        Ok(mut f) => {
            f.set(key, value);
            if let Err(e) = f.save(&file) {
                crate::warn!("settings: writing {}: {e}", file.display());
            }
        }
        Err(e) => crate::warn!("settings: reading {}: {e}", file.display()),
    }
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// Logs an error for the player and, with dialogs on, shows it in a
/// message box (waits until it is closed).
pub fn user_error(title: &str, text: &str) {
    crate::error!("{title}: {text}");
    crate::log::try_flush();
    if dialogs() {
        let text = crate::log::redact(text).into_owned();
        let (t, m) = (wide(title), wide(&text));
        unsafe {
            use windows_sys::Win32::UI::WindowsAndMessaging::{MB_ICONERROR, MB_OK, MB_SETFOREGROUND, MessageBoxW};
            MessageBoxW(std::ptr::null_mut(), m.as_ptr(), t.as_ptr(), MB_OK | MB_ICONERROR | MB_SETFOREGROUND);
        }
    }
}
