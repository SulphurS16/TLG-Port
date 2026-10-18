//! Game files page: the game folder (01.00) and the update folder (01.03),
//! checked when chosen; the settings keep them once they pass.

use crate::files;
use crate::ui::{self, Builder};
use crate::App;
use std::path::{Path, PathBuf};
use windows_sys::Win32::Foundation::HWND;

const ID_STATUS: i32 = 101;
const ID_GAME_DIR: i32 = 130;
const ID_UPDATE_DIR: i32 = 132;
const ID_CHECK: i32 = 140;

#[derive(Default)]
pub struct Page {
    status: usize,
    game_dir: usize,
    update_dir: usize,
    /// Whether the folders in the settings passed the check.
    pub ok: bool,
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
        let r = (16, 172, 748, 300);
        app.game.status = app.put(0, b.label("", r, ID_STATUS), r, false) as usize;
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
    unsafe { ui::set_text(h(app.game.status), &text) };
    if keep && r.is_ok() {
        app.settings.game_dir = Some(game.to_path_buf());
        app.settings.update_dir = update.map(Path::to_path_buf);
        app.save_settings();
    }
    app.update_play();
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
