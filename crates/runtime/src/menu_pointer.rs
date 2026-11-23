//! A mouse pointer in the game's own menus (PC Settings `menu_pointer`).
//!
//! In menus the mouse's raw motion moves a pointer (1920x1080 units, one
//! count one unit: the desktop's speed without acceleration) instead of the
//! camera; the port's script draws it in the game's frame. Over the
//! Options lists and their pages (the game's list helper: `menu_patches`
//! records the active list and its cursor) the item under the pointer is
//! selected, a left click confirms it (Cross) or, over a row's value,
//! steps the value toward the side clicked (Left / Right). Over the title
//! screen's and the pause menu's own lists (`menu_patches::MAIN_LIST`) the
//! item under the pointer is selected by the list's own update and a left
//! click confirms the item under it. Elsewhere a left click confirms the
//! selection, the wheel moves it and the right button goes back, as
//! before.
//!
//! The items' places: their layouts (`UseSc2Layout` entries of the game's
//! menu part table, gamemenusetting.lua; normalized x and y as the layouts
//! hold them: screen y = 543.4 - 1001.3 * y, x = 960 + 1001.3 * x).

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

struct Pointer {
    x: f32,
    y: f32,
    /// When it last moved (process ms).
    moved: f64,
}

static POINTER: Mutex<Pointer> = Mutex::new(Pointer { x: 960.0, y: 540.0, moved: -1e9 });

/// Part key -> its layout file (from the game's part table).
static LAYOUTS: Mutex<Option<HashMap<String, String>>> = Mutex::new(None);
/// Part key -> its place (screen x, y) once read.
static PLACES: Mutex<Option<HashMap<u32, Option<(f32, f32)>>>> = Mutex::new(None);
/// The last item the pointer selected (the list helper's symbol).
static HOVER: AtomicU64 = AtomicU64::new(0);

pub fn enabled() -> bool {
    crate::pc_settings::get("menu_pointer") != "off"
}

/// The game's menu part table (gamemenusetting.lua's text, as loaded):
/// keys and layouts of its `UseSc2Layout` entries.
pub fn note_part_table(text: &[u8]) {
    let t = String::from_utf8_lossy(text);
    let mut map = HashMap::new();
    let mut rest: &str = &t;
    while let Some(i) = rest.find("\"UseSc2Layout\"") {
        rest = &rest[i + 14..];
        // (The next three strings: key, entity, layout.)
        let mut q = Vec::new();
        let mut r = rest;
        for _ in 0..3 {
            let Some(a) = r.find('"') else { break };
            let Some(b) = r[a + 1..].find('"') else { break };
            q.push(r[a + 1..a + 1 + b].to_string());
            r = &r[a + 2 + b..];
        }
        if q.len() == 3 {
            map.insert(q[0].clone(), q[2].clone());
        }
    }
    crate::info!("menu pointer: {} menu items' layouts", map.len());
    *LAYOUTS.lock() = Some(map);
}

/// An item's place on screen (its layout's first position), by symbol.
fn place(sym: u32) -> Option<(f32, f32)> {
    if let Some(p) = PLACES.lock().as_ref().and_then(|m| m.get(&sym)) {
        return *p;
    }
    let key = crate::title_patches::symbol_name(sym);
    let layout = LAYOUTS.lock().as_ref().and_then(|m| m.get(&key).cloned());
    let p = layout.and_then(|l| {
        let name = format!("{}.bod", l.to_lowercase());
        let (_, _, lang) = crate::hle::libs::app::game_language();
        let d = crate::pc_files::game_file(&name, "").or_else(|| crate::pc_files::game_file(&name, lang))?;
        let (x, y) = crate::pc_files::layout_xy(&d)?;
        Some((960.0 + 1001.3 * x, 543.4 - 1001.3 * y))
    });
    PLACES.lock().get_or_insert_with(HashMap::new).insert(sym, p);
    p
}

/// Where the pointer's selection goes: an Options list's cursor (written:
/// the list helper follows it) or a main list's (put in by its update).
#[derive(Clone, Copy)]
enum Sel {
    Cursor(u64),
    Main(u64),
}

/// The active list (its update ran in the last 200 ms; the later one when
/// both did): its items' places and where its selection goes.
fn active() -> Option<(Vec<(u32, Option<(f32, f32)>)>, Sel)> {
    let now = crate::log::elapsed_ms() as u64;
    let fresh = |at: u64| now.saturating_sub(at) <= 200;
    let list = crate::menu_patches::ACTIVE_LIST.lock().as_ref().filter(|a| fresh(a.at)).map(|a| (a.items.clone(), Sel::Cursor(a.cursor), a.at));
    let main = crate::menu_patches::MAIN_LIST.lock().as_ref().filter(|a| fresh(a.at)).map(|a| (a.items.clone(), Sel::Main(a.cursor), a.at));
    let (items, sel, _) = match (list, main) {
        (Some(l), Some(m)) => if m.2 > l.2 { m } else { l },
        (l, m) => l.or(m)?,
    };
    let places: Vec<(u32, Option<(f32, f32)>)> = items.iter().map(|&s| (s, place(s))).collect();
    if let Sel::Main(_) = sel {
        static LOGGED: Mutex<Vec<Vec<u32>>> = Mutex::new(Vec::new());
        let mut l = LOGGED.lock();
        if !l.contains(&items) {
            let p: Vec<String> = places.iter().map(|(s, p)| format!("{} {}", crate::title_patches::symbol_name(*s), p.map_or("-".into(), |(x, y)| format!("{x:.0},{y:.0}")))).collect();
            crate::info!("menu pointer: a menu's own list: {}", p.join(", "));
            l.push(items);
        }
        // (No item placed: the list is left to the buttons.)
        if places.iter().all(|(_, p)| p.is_none()) {
            return None;
        }
    }
    Some((places, sel))
}

/// The value at a selection's cursor field, if readable.
fn cursor_at(cursor: u64) -> Option<u32> {
    // SAFETY: the menu's own field, checked readable.
    (cursor != 0 && crate::debug_mem::host_or_guest_readable(cursor, 4)).then(|| unsafe { std::ptr::read_volatile(cursor as *const u32) })
}

/// The item under the point, if any: its index, and where on its row
/// (-1 left of its value, 0 its label, 1 right of its value).
fn hit(items: &[(u32, Option<(f32, f32)>)], x: f32, y: f32) -> Option<(usize, i32)> {
    let mut best: Option<(usize, f32)> = None;
    for (i, (_, p)) in items.iter().enumerate() {
        let Some((px, py)) = *p else { continue };
        // (Rows about 45-80 px apart; items side by side on one row - Back
        // and Reset - are told apart by their left edges.)
        let right = items.iter().filter_map(|(_, q)| *q).filter(|&(qx, qy)| (qy - py).abs() < 4.0 && qx > px + 1.0).map(|(qx, _)| qx).fold(f32::MAX, f32::min);
        // (The last on its row reaches to 1060, or 300 past its place: a
        // question's No right of the middle.)
        let x1 = if right < f32::MAX { right - 10.0 } else { 1060f32.max(px + 300.0) };
        if x < px - 50.0 || x > x1 {
            continue;
        }
        let d = (y - py).abs();
        if d < 32.0 && best.is_none_or(|b| d < b.1) {
            best = Some((i, d));
        }
    }
    let (i, _) = best?;
    // (A value row: its value column, 470-1010, left or right of its middle.)
    let side = if x >= 470.0 && x <= 1010.0 { if x < 740.0 { -1 } else { 1 } } else { 0 };
    Some((i, side))
}

/// Raw mouse motion while a menu shows: the pointer moves (true), unless it
/// is off. Over an Options list, the item under it is selected.
pub fn moved(dx: i64, dy: i64) -> bool {
    if !enabled() || crate::pc_menu::is_open() {
        return false;
    }
    let (x, y) = {
        let mut p = POINTER.lock();
        p.x = (p.x + dx as f32).clamp(0.0, 1919.0);
        p.y = (p.y + dy as f32).clamp(0.0, 1079.0);
        p.moved = crate::log::elapsed_ms();
        (p.x, p.y)
    };
    if let Some((items, sel)) = active() {
        if let Some((i, _)) = hit(&items, x, y) {
            match sel {
                Sel::Cursor(cursor) => {
                    if let Some(cur) = cursor_at(cursor) {
                        if cur as usize != i {
                            // SAFETY: the list helper's cursor (checked readable).
                            unsafe { std::ptr::write_volatile(cursor as *mut u32, i as u32) };
                        }
                        HOVER.store(items[i].0 as u64, Ordering::Relaxed);
                    }
                }
                Sel::Main(_) => {
                    // (Also when it is selected already: a choice still
                    // waiting for the list's update is replaced.)
                    crate::menu_patches::select_main(i, items[0].0);
                    HOVER.store(items[i].0 as u64, Ordering::Relaxed);
                }
            }
        }
    }
    true
}

/// A left click in a menu: the game's buttons for it. Over an Options list:
/// Cross on an item's label, Left/Right on its value, nothing off its
/// items; over the title's or the pause menu's list: Cross on the item
/// under the pointer (selected first), nothing off its items; other menus:
/// Cross (their selection).
pub fn click() -> u32 {
    use crate::input::{BTN_CROSS, BTN_LEFT, BTN_RIGHT};
    if !enabled() {
        return BTN_CROSS;
    }
    let (x, y) = {
        let p = POINTER.lock();
        (p.x, p.y)
    };
    match active() {
        Some((items, Sel::Main(_))) => match hit(&items, x, y) {
            Some((i, _)) => {
                // (The list puts the selection in before it reads Cross.)
                crate::menu_patches::select_main(i, items[0].0);
                BTN_CROSS
            }
            None => 0,
        },
        Some((items, Sel::Cursor(_))) => match hit(&items, x, y) {
            Some((_, -1)) => BTN_LEFT,
            Some((_, 1)) => BTN_RIGHT,
            Some(_) => BTN_CROSS,
            None => 0,
        },
        None => BTN_CROSS,
    }
}

/// For the script: the pointer's place and whether to show it (moved in
/// the last 3 s while a menu shows).
pub fn script() -> Vec<String> {
    let p = POINTER.lock();
    let age = crate::log::elapsed_ms() - p.moved;
    let shown = enabled() && age < 3000.0 && crate::title_patches::in_menu() == Some(true) && !crate::pc_menu::is_open();
    vec![format!("{:.1}", p.x), format!("{:.1}", p.y), if shown { format!("{:.2}", (3000.0 - age) / 1000.0) } else { String::new() }, if active().is_some() { "list".into() } else { String::new() }]
}

#[cfg(test)]
mod tests {
    use super::hit;

    /// Items as the layouts place them: (symbol, place).
    fn items(places: &[(f32, f32)]) -> Vec<(u32, Option<(f32, f32)>)> {
        places.iter().enumerate().map(|(i, &p)| (i as u32, Some(p))).collect()
    }

    #[test]
    fn rows_by_their_middle() {
        // The title's rows (New Game, Continue, Options).
        let it = items(&[(109.0, 476.0), (109.0, 543.0), (109.0, 606.0)]);
        assert_eq!(hit(&it, 200.0, 480.0).map(|h| h.0), Some(0));
        assert_eq!(hit(&it, 150.0, 540.0).map(|h| h.0), Some(1));
        assert_eq!(hit(&it, 300.0, 620.0).map(|h| h.0), Some(2));
        // Off the rows, left of them, past the reach.
        assert_eq!(hit(&it, 200.0, 700.0), None);
        assert_eq!(hit(&it, 40.0, 543.0), None);
        assert_eq!(hit(&it, 1100.0, 543.0), None);
    }

    #[test]
    fn side_by_side_and_far_right() {
        // A question's Yes and No on one row: each up to the next one's left edge.
        let it = items(&[(278.0, 576.0), (471.0, 576.0)]);
        assert_eq!(hit(&it, 300.0, 576.0).map(|h| h.0), Some(0));
        assert_eq!(hit(&it, 470.0, 580.0).map(|h| h.0), Some(1));
        assert_eq!(hit(&it, 1000.0, 576.0).map(|h| h.0), Some(1));
        // An item placed right of the middle reaches 300 px past its place.
        let far = items(&[(900.0, 500.0)]);
        assert_eq!(hit(&far, 1150.0, 500.0).map(|h| h.0), Some(0));
        assert_eq!(hit(&far, 1250.0, 500.0), None);
        // Unplaced items are never hit.
        assert_eq!(hit(&[(7, None)], 200.0, 500.0), None);
    }

    #[test]
    fn value_sides() {
        let it = items(&[(148.0, 469.0)]);
        assert_eq!(hit(&it, 300.0, 469.0), Some((0, 0)));
        assert_eq!(hit(&it, 600.0, 469.0), Some((0, -1)));
        assert_eq!(hit(&it, 900.0, 469.0), Some((0, 1)));
    }
}
