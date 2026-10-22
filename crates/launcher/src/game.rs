//! Game files page: the game folder (01.00) and the update folder (01.03),
//! checked when chosen; the settings keep them once they pass. Folders inside
//! another version's folder (they would go with it) can move into this
//! version's (Move here). The game's language.

use crate::files;
use crate::ui::{self, Builder};
use crate::{App, TITLE};
use std::path::{Path, PathBuf};
use tlg_config::carry;
use tlg_config::paths::Layout;
use tlg_config::settings::Language;
use windows_sys::Win32::Foundation::HWND;
use windows_sys::Win32::UI::WindowsAndMessaging::{IDOK, MB_ICONQUESTION, MB_ICONWARNING, MB_OK, MB_OKCANCEL};

const ID_STATUS: i32 = 101;
const ID_GAME_DIR: i32 = 130;
const ID_UPDATE_DIR: i32 = 132;
const ID_CHECK: i32 = 140;
const ID_MOVE: i32 = 141;
const ID_LANGUAGE: i32 = 150;

#[derive(Default)]
pub struct Page {
    status: usize,
    game_dir: usize,
    update_dir: usize,
    language: usize,
    move_here: usize,
    /// Whether the folders in the settings passed the check.
    pub ok: bool,
    /// Another version's folder that holds the game folders.
    elsewhere: Option<PathBuf>,
}

fn h(v: usize) -> HWND {
    v as HWND
}

/// A labelled path row: label, edit, Browse button (id + 1).
unsafe fn path_row(app: &mut App, b: &Builder, y: i32, label: &str, id: i32, text: &str) -> usize {
    let r = (16, y + 3, 120, 20);
    app.put(0, unsafe { b.label(label, r, -1) }, r, false);
    let r = (140, y, 520, 24);
    let e = app.put(0, unsafe { b.edit(text, r, id) }, r, false);
    let r = (668, y - 1, 96, 26);
    app.put(0, unsafe { b.button("Browse...", r, id + 1) }, r, false);
    e as usize
}

pub fn build(app: &mut App) {
    let b = app.builder();
    // (Folders not set: the default location, where the game also looks.)
    let (g, u) = match &app.settings.game_dir {
        Some(g) => (Some(g.clone()), app.settings.update_dir.clone()),
        None => {
            let (dg, du) = (app.paths.default_base(), app.paths.default_update());
            (dg.join("eboot.bin").is_file().then_some(dg), du.join("eboot.bin").is_file().then_some(du))
        }
    };
    let shown = |p: &Option<PathBuf>| p.as_ref().map(|p| p.display().to_string()).unwrap_or_default();
    let (game, update) = (shown(&g), shown(&u));
    unsafe {
        app.game.game_dir = path_row(app, &b, 60, "Game folder:", ID_GAME_DIR, &game);
        app.game.update_dir = path_row(app, &b, 96, "Update folder:", ID_UPDATE_DIR, &update);
        let r = (668, 132, 96, 28);
        app.put(0, b.button("Check", r, ID_CHECK), r, false);
        // (Shown only when it applies: not one of the page's controls.)
        let r = (556, 132, 104, 28);
        app.game.move_here = app.put(usize::MAX, b.button("Move here", r, ID_MOVE), r, false) as usize;
        let r = (16, 172, 748, 250);
        app.game.status = app.put(0, b.label("", r, ID_STATUS), r, false) as usize;
        let r = (16, 443, 120, 20);
        app.put(0, b.label("Language:", r, -1), r, false);
        let r = (140, 440, 260, 200);
        app.game.language = app.put(0, b.combo(r, ID_LANGUAGE), r, false) as usize;
        let names: Vec<String> = Language::ALL.iter().map(|l| l.label().to_string()).collect();
        ui::combo_fill(h(app.game.language), &names, Language::ALL.iter().position(|&l| l == app.settings.language).unwrap_or(0));
    }
    match g {
        Some(g) => check(app, &g, u.as_deref(), false),
        None => {
            unsafe { ui::set_text(h(app.game.status), "Choose the game folder.") };
            app.update_play();
        }
    }
}

/// Checks the folders; once they pass (and `keep`), the settings take them.
fn check(app: &mut App, game: &Path, update: Option<&Path>, keep: bool) {
    let r = files::check_pair(game, update);
    app.game.ok = r.is_ok();
    let text = match &r {
        Ok(s) => s.clone(),
        Err(e) => e.clone(),
    };
    if keep && r.is_ok() {
        app.settings.game_dir = Some(game.to_path_buf());
        app.settings.update_dir = update.map(Path::to_path_buf);
        app.save_settings();
    }
    app.game.elsewhere = (app.paths.layout != Layout::Dev).then(|| [Some(game), update].into_iter().flatten().find_map(|d| carry::other_version(d, &app.paths.exe_dir))).flatten();
    let text = match &app.game.elsewhere {
        Some(o) => format!("{text}\n\nThese folders are inside the folder of another version of the port:\n{}\nIf you delete that folder, they go with it. Keep it, or click Move here to move them into this version's folder.", short(o)),
        None => text,
    };
    unsafe { ui::set_text(h(app.game.status), &text) };
    show_move(app);
    app.update_play();
}

/// A path for one line of the page (its middle left out when long).
fn short(p: &Path) -> String {
    let s = p.display().to_string();
    let c: Vec<char> = s.chars().collect();
    if c.len() <= 90 {
        return s;
    }
    format!("{}...{}", c[..30].iter().collect::<String>(), c[c.len() - 56..].iter().collect::<String>())
}

/// Shows Move here on this page when the game folders are in another
/// version's folder.
pub fn show_move(app: &App) {
    unsafe { ui::show(h(app.game.move_here), app.page == 0 && app.game.elsewhere.is_some()) };
}

/// Moves the game folders that are in another version's folder into this
/// version's folder (the settings follow).
fn move_here(app: &mut App) {
    let here = app.paths.exe_dir.clone();
    let dirs: Vec<PathBuf> = [app.settings.game_dir.clone(), app.settings.update_dir.clone()].into_iter().flatten().filter(|d| carry::other_version(d, &here).is_some()).collect();
    if dirs.is_empty() {
        return;
    }
    let list: Vec<String> = dirs.iter().map(|d| d.display().to_string()).collect();
    let text = format!("Move these folders into this version's folder ({})?\n\n{}\n\nOn the same drive this takes a moment.", here.display(), list.join("\n"));
    if unsafe { ui::message(app.hwnd, &text, TITLE, MB_ICONQUESTION | MB_OKCANCEL) } != IDOK {
        return;
    }
    for d in dirs {
        match carry::move_into(&d, &here) {
            Ok(to) => {
                // (The update folder may be inside the game folder.)
                let rebase = |p: &mut Option<PathBuf>| {
                    if let Some(rest) = p.as_ref().and_then(|p| p.strip_prefix(&d).ok()) {
                        *p = Some(if rest.as_os_str().is_empty() { to.clone() } else { to.join(rest) });
                    }
                };
                rebase(&mut app.settings.game_dir);
                rebase(&mut app.settings.update_dir);
                app.save_settings();
            }
            Err(e) => {
                let text = format!("Could not move {}: {e}\n\nKeep the other version's folder, or move the game files yourself and choose them again here.", d.display());
                unsafe { ui::message(app.hwnd, &text, TITLE, MB_ICONWARNING | MB_OK) };
                break;
            }
        }
    }
    let shown = |p: &Option<PathBuf>| p.as_ref().map(|p| p.display().to_string()).unwrap_or_default();
    unsafe {
        ui::set_text(h(app.game.game_dir), &shown(&app.settings.game_dir));
        ui::set_text(h(app.game.update_dir), &shown(&app.settings.update_dir));
    }
    check_fields(app);
}

/// Checks the folders in the fields.
fn check_fields(app: &mut App) {
    let get = |v: usize| unsafe { ui::get_text(h(v)) }.trim().trim_matches('"').to_string();
    let (game, update) = (get(app.game.game_dir), get(app.game.update_dir));
    if game.is_empty() {
        app.game.ok = false;
        unsafe { ui::set_text(h(app.game.status), "Choose the game folder.") };
        app.update_play();
        return;
    }
    let update = (!update.is_empty()).then(|| PathBuf::from(update));
    check(app, Path::new(&game), update.as_deref(), true);
}

pub fn command(app: &mut App, id: i32, code: u32) {
    match (id, code) {
        (ID_CHECK, 0) => check_fields(app),
        (ID_MOVE, 0) => move_here(app),
        // (CBN_SELCHANGE)
        (ID_LANGUAGE, 1) => {
            let i = unsafe { ui::combo_sel(h(app.game.language)) };
            app.settings.language = Language::ALL.get(i).copied().unwrap_or(Language::English);
            app.save_settings();
            app.set_status("Saved; applies the next time the game starts.");
        }
        (x, 0) if x == ID_GAME_DIR + 1 || x == ID_UPDATE_DIR + 1 => {
            let (edit, title) = if x == ID_GAME_DIR + 1 { (app.game.game_dir, "Game folder (version 01.00)") } else { (app.game.update_dir, "Update folder (01.03)") };
            if let Some(p) = ui::pick_folder(app.hwnd, title) {
                unsafe { ui::set_text(h(edit), &p.display().to_string()) };
                check_fields(app);
            }
        }
        _ => {}
    }
}
