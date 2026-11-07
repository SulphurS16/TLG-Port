//! The port's own in-game script: the PC settings pages, drawn and run by
//! the game itself (Havok Script with the game's GUI_* drawing, the way its
//! control guides draw), appended to system/scripts/utils.lua at load
//! (`script_fixes`). `TLG_LUA_APPEND=<file>` uses that file's text instead
//! (development: a changed script without a rebuild).
//!
//! The script talks to the port through the game's `GaPrint` (routed to
//! the port, `title_patches`): a call whose first argument is `MARKER`
//! is a request (`call`), its answers pushed back as strings. While a PC
//! page is open the game's pad reads see a neutral pad (`take_pad`) and the
//! page gets the input instead; in menus the captured mouse moves a pointer
//! the script draws (raw counts, no acceleration), and its buttons and
//! wheel select, go back and scroll.

use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// The first argument of a script request (`GaPrint(MARKER, ...)`).
pub const MARKER: &str = "\u{1}TLGPC";

/// The script text appended to the game's utils.lua.
pub fn lua_module() -> String {
    if let Ok(p) = std::env::var("TLG_LUA_APPEND") {
        match std::fs::read_to_string(&p) {
            Ok(t) => return t,
            Err(e) => crate::warn!("pc menu: TLG_LUA_APPEND {p}: {e}; the built-in script instead"),
        }
    }
    include_str!("pc_menu.lua").to_string()
}

/// A PC page is open: the game's pad reads see a neutral pad.
static OPEN: AtomicBool = AtomicBool::new(false);
/// A request to open the PC pages (the hotkey, the Options item).
static OPEN_REQUEST: AtomicBool = AtomicBool::new(false);
/// The pages were opened from the Options list (which they then replace).
static FROM_OPTIONS: AtomicBool = AtomicBool::new(false);
/// Buttons held when a page closed: kept from the game until released (the
/// press that closed the page is not the game's).
static MASK: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

struct Nav {
    /// The pad state the page last saw, and buttons pressed since its
    /// previous `input` request (edges).
    held: u32,
    pressed: u32,
    /// Stick direction held (for repeats): -1, 0, 1 per axis.
    stick: (i8, i8),
    /// The pointer in 1920x1080 units of the game's picture.
    px: f32,
    py: f32,
    /// Mouse buttons down (1 left, 2 right) and pressed since the last request.
    mbuttons: u32,
    mpressed: u32,
    wheel: i32,
    /// The pointer moved since the last request.
    moved: bool,
}

static NAV: Mutex<Nav> = Mutex::new(Nav { held: 0, pressed: 0, stick: (0, 0), px: 960.0, py: 540.0, mbuttons: 0, mpressed: 0, wheel: 0, moved: false });

/// Script text for the port's game script to run (the control command `lua`).
static LUA_QUEUE: Mutex<Vec<String>> = Mutex::new(Vec::new());

pub fn queue_lua(code: &str) {
    LUA_QUEUE.lock().push(code.to_string());
}

/// `s` (the port's English text) in the game's language, where translated.
pub fn tr(s: &'static str) -> &'static str {
    s
}

/// The game's own camera options, Vertical and Horizontal Camera
/// "Reversed" (x, y), which mouse look follows as the right stick does.
pub fn camera_inverted() -> (bool, bool) {
    crate::menu_patches::camera_options()
}

pub fn is_open() -> bool {
    OPEN.load(Ordering::Relaxed)
}

/// Asks the script to open the PC pages (it does on its next frame).
pub fn request_open() {
    OPEN_REQUEST.store(true, Ordering::Relaxed);
}

/// The Options list's PC Settings item was chosen (`menu_patches`).
pub fn open_from_options() {
    FROM_OPTIONS.store(true, Ordering::Relaxed);
    OPEN.store(true, Ordering::Relaxed);
    request_open();
}

/// The game's pad read: while a page is open, the real state goes to the
/// page and the game gets a neutral pad.
pub fn take_pad(mut st: crate::input::PadState) -> crate::input::PadState {
    if !is_open() {
        let m = MASK.load(Ordering::Relaxed);
        if m != 0 {
            MASK.store(m & st.buttons, Ordering::Relaxed);
            st.buttons &= !m;
        }
        return st;
    }
    let mut n = NAV.lock();
    n.pressed |= st.buttons & !n.held;
    n.held = st.buttons;
    let axis = |v: u8| if v < 0x40 { -1 } else if v > 0xC0 { 1 } else { 0 };
    n.stick = (axis(st.lx), axis(st.ly));
    crate::input::PadState::default()
}

/// Mouse input while a menu has the pointer: raw counts move it; buttons
/// (bit 0 left, bit 1 right) and wheel notches. Returns whether it was taken.
pub fn mouse(dx: i64, dy: i64, buttons: u32, wheel: i32) -> bool {
    if !is_open() {
        return false;
    }
    let mut n = NAV.lock();
    // (One count, one 1080p pixel: the pointer moves like the desktop's at
    // its default speed without acceleration.)
    if dx != 0 || dy != 0 {
        n.px = (n.px + dx as f32).clamp(0.0, 1919.0);
        n.py = (n.py + dy as f32).clamp(0.0, 1079.0);
        n.moved = true;
    }
    n.mpressed |= buttons & !n.mbuttons;
    n.mbuttons = buttons;
    n.wheel += wheel;
    true
}

/// The mouse buttons pressed since the last call (bit 0 left, 1 right),
/// taken from the page's input (a key binding's choice).
pub fn take_mouse_presses() -> u32 {
    std::mem::take(&mut NAV.lock().mpressed)
}

/// A script request (`GaPrint(MARKER, cmd, ...)`): the answers.
pub fn call(args: &[String]) -> Vec<String> {
    let a = |i: usize| args.get(i).map(String::as_str).unwrap_or("");
    match a(0) {
        "hello" => vec!["1".into()],
        // now -> seconds since start (the pages animate in real time: the
        // game's own step is 0 while it is paused)
        "now" => vec![format!("{:.4}", crate::log::elapsed_ms() / 1000.0)],
        // cap -> "game" while the frame-rate limit is the game's 30 fps
        "cap" => vec![if crate::hle::libs::videoout::frame_cap() == crate::config::FrameCap::Game { "game".into() } else { "above".into() }],
        // (The atlas texture's name, for the script's Load.)
        "atlas" => vec![crate::pc_files::ATLAS_NAME.into(), crate::pc_files::ATLAS_W.to_string(), crate::pc_files::ATLAS_H.to_string()],
        // text <string> <px> [weight] -> x y w h generation (atlas texels)
        "text" => {
            let px = a(2).parse().unwrap_or(24u32).clamp(6, 200);
            let weight = a(3).parse().unwrap_or(400i32);
            match crate::pc_files::text(a(1), px, weight) {
                Some((r, g)) => vec![r.x.to_string(), r.y.to_string(), r.w.to_string(), r.h.to_string(), g.to_string()],
                None => vec![],
            }
        }
        // wrap <string> <px> <max width> -> its lines (at spaces, measured as drawn)
        "wrap" => {
            let px = a(2).parse().unwrap_or(24u32).clamp(6, 200);
            let max_w = a(3).parse().unwrap_or(800u32).max(16);
            crate::pc_files::wrap(a(1), px, 400, max_w)
        }
        // poll -> "open" once after a request
        "poll" => vec![if OPEN_REQUEST.swap(false, Ordering::Relaxed) { "open".into() } else { String::new() }],
        "opened" => {
            OPEN.store(true, Ordering::Relaxed);
            let mut n = NAV.lock();
            (n.held, n.pressed, n.mbuttons, n.mpressed, n.wheel) = (u32::MAX, 0, 0, 0, 0);
            vec![]
        }
        "closed" => {
            let held = NAV.lock().held;
            MASK.store(if held == u32::MAX { 0 } else { held }, Ordering::Relaxed);
            OPEN.store(false, Ordering::Relaxed);
            if FROM_OPTIONS.swap(false, Ordering::Relaxed) {
                crate::menu_patches::closed();
            }
            vec![]
        }
        // refused: the pages could not open now (the pause menu shows, or
        // the game allows no pause)
        "refused" => {
            crate::info!("pc menu: not now (the pause menu shows, or no pause allowed here)");
            vec![]
        }
        // origin -> "options" when opened from the Options list (the page
        // replaces it; "Back" returns there)
        "origin" => vec![if FROM_OPTIONS.load(Ordering::Relaxed) { "options".into() } else { String::new() }],
        // item -> x y (the Options list's PC Settings row: left edge, middle;
        // 1080p pixels) and its label, while the list is on screen
        "item" => match crate::menu_patches::item_label() {
            Some((x, y, _)) => vec![format!("{x:.1}"), format!("{y:.1}"), tr("PC Settings").to_string()],
            None => vec![],
        },
        // input -> pressed buttons, held buttons, stick x y, pointer x y, moved, mouse pressed, mouse held, wheel
        "input" => {
            let mut n = NAV.lock();
            let held = if n.held == u32::MAX { 0 } else { n.held };
            let out = vec![
                n.pressed.to_string(),
                held.to_string(),
                n.stick.0.to_string(),
                n.stick.1.to_string(),
                format!("{:.1}", n.px),
                format!("{:.1}", n.py),
                (n.moved as u8).to_string(),
                n.mpressed.to_string(),
                n.mbuttons.to_string(),
                n.wheel.to_string(),
            ];
            (n.pressed, n.mpressed, n.wheel, n.moved) = (0, 0, 0, false);
            out
        }
        "log" => {
            crate::info!("pc menu: {}", args[1..].join(" "));
            vec![]
        }
        // lua -> queued script text (development console)
        "lua" => std::mem::take(&mut *LUA_QUEUE.lock()),
        // menu_ptr -> x y shown-seconds kind (the menu pointer in the
        // game's own menus: menu_pointer)
        "menu_ptr" => crate::menu_pointer::script(),
        // guide_slots -> the prompts' picture slots' texture names (the
        // script loads them ahead of the guides)
        // trophy_toast -> name, line above, icon u0 v0 u1 v1 (nothing waiting: nothing)
        "trophy_toast" => crate::achievements::next_toast(),
        // trophy_tex -> the icon atlas and the pop-up frame, frame width and height
        "trophy_tex" => vec![crate::achievements::ICONS_NAME.into(), crate::achievements::FRAME_NAME.into(), crate::achievements::FRAME_W.to_string(), crate::achievements::FRAME_H.to_string()],
        "guide_slots" => (0..crate::pc_prompts::SLOTS).map(crate::pc_prompts::slot_name).collect(),
        // guide_tex <buttons> -> the picture slot for a guide being made
        // with these buttons ("Circle+StickBaseL"; empty: none)
        "guide_tex" => vec![crate::pc_prompts::guide_tex(a(1))],
        // page <id> -> title, rows (pc_settings::page_rows)
        "page" => crate::pc_settings::page_rows(a(1)),
        // set <key> <value> -> ok | start | failed | an error
        "set" => vec![crate::pc_settings::set(a(1), a(2))],
        // act <key> -> ok | reload | a message
        "act" => vec![crate::pc_settings::act(a(1))],
        // bind_start <bind:name>, bind_poll -> wait | done | cleared |
        // cancel | a note (Key Bindings: the next key or mouse button)
        "bind_start" => vec![crate::pc_settings::bind_start(a(1))],
        "bind_poll" => vec![crate::pc_settings::bind_poll()],
        // reset <page id>: the page's settings to their defaults
        "reset" => {
            crate::pc_settings::reset_page(a(1));
            vec![]
        }
        // engine_gen -> the overrides' generation; engine -> count, then
        // object, field, mode, value per override
        "engine_gen" => vec![crate::pc_settings::engine_gen().to_string()],
        "engine" => crate::pc_settings::engine_overrides(),
        other => {
            crate::warn!("pc menu: unknown request {other:?}");
            vec![]
        }
    }
}
