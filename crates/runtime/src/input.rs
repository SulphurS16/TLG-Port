//! Host input aggregation for the DualShock 4 pad: XInput controller, a
//! DualShock 4 / DualSense over HID, the output window's keyboard and mouse
//! (bindings in the user folder's `input.txt`, tlg_config::bindings, written
//! with the defaults on first use), and a scripted timeline for
//! reproducible runs.
//!
//! Default keyboard layout (window focused; tlg_config::bindings::DEFAULTS):
//!   W/A/S/D move (Left Shift: walk) · Space jump (Triangle) · E or the left
//!   button pick up / grab (Circle) · C or Enter crouch / confirm (Cross) ·
//!   F or the right button tackle / throw (Square) · Q R1 · R L1 · X R2 ·
//!   Z L2 · V R3 · B L3 · arrow keys d-pad · I/J/K/L camera · Escape pause /
//!   back · Tab Options · T touchpad.
//! Escape (`pause_back`) is Options while playing and Circle - back - in
//! menus (`title_patches::in_menu`: title screen and main menu by the
//! scripts' markers; pause menu, loading screens and most cutscenes by the
//! game camera standing still); the choice is made when the key goes down. In menus the
//! mouse buttons select (left: Cross) and go back (right: Circle) and the
//! wheel moves the selection (d-pad), instead of their bindings
//! (`menu_mouse`); a button keeps its meaning until released.
//! Mouse: a click into the window captures it (raw counts, no pointer
//! acceleration); F1 or leaving the window releases it. Mouse look turns the
//! game camera by the raw counts (`title_patches` camera hook: no dead zone,
//! response curve or smoothing; sensitivity per axis); while that camera is
//! not updating it drives the right stick instead.
//!
//! Scripted input (`TLG_INPUT`): `flip:buttons[:hold]` entries separated by
//! `;`, e.g. `TLG_INPUT=700:cross;760:down:3;800:cross`. `buttons` joins names
//! with `+` (cross, circle, square, triangle, up, down, left, right, l1, r1,
//! l2, r2, l3, r3, options, touchpad, ls_up/down/left/right,
//! rs_up/down/left/right). Each press lasts `hold` GPU flips (default 6).

use std::sync::OnceLock;
pub use tlg_config::bindings::{
    BTN_CIRCLE, BTN_CROSS, BTN_DOWN, BTN_L1, BTN_L2, BTN_L3, BTN_LEFT, BTN_OPTIONS, BTN_R1, BTN_R2, BTN_R3, BTN_RIGHT, BTN_SQUARE, BTN_TOUCHPAD, BTN_TRIANGLE, BTN_UP,
};
use tlg_config::bindings::{Bindings, Control, Input, Stick};
use windows_sys::Win32::UI::Input::XboxController::{XINPUT_STATE, XInputGetState};

/// Pad state in DS4 terms (sticks 0..255, centre 0x80; y grows downwards).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PadState {
    pub buttons: u32,
    pub lx: u8,
    pub ly: u8,
    pub rx: u8,
    pub ry: u8,
    pub l2: u8,
    pub r2: u8,
}

impl Default for PadState {
    fn default() -> Self {
        PadState { buttons: 0, lx: 0x80, ly: 0x80, rx: 0x80, ry: 0x80, l2: 0, r2: 0 }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Step {
    flip: u64,
    hold: u64,
    buttons: u32,
    sticks: Vec<Stick>,
}

fn parse_name(n: &str) -> Option<Result<u32, Stick>> {
    Some(match n {
        "cross" | "x" => Ok(BTN_CROSS),
        "circle" | "o" => Ok(BTN_CIRCLE),
        "square" => Ok(BTN_SQUARE),
        "triangle" => Ok(BTN_TRIANGLE),
        "up" => Ok(BTN_UP),
        "down" => Ok(BTN_DOWN),
        "left" => Ok(BTN_LEFT),
        "right" => Ok(BTN_RIGHT),
        "l1" => Ok(BTN_L1),
        "r1" => Ok(BTN_R1),
        "l2" => Ok(BTN_L2),
        "r2" => Ok(BTN_R2),
        "l3" => Ok(BTN_L3),
        "r3" => Ok(BTN_R3),
        "options" | "start" => Ok(BTN_OPTIONS),
        "touchpad" => Ok(BTN_TOUCHPAD),
        "ls_up" => Err(Stick::LUp),
        "ls_down" => Err(Stick::LDown),
        "ls_left" => Err(Stick::LLeft),
        "ls_right" => Err(Stick::LRight),
        "rs_up" => Err(Stick::RUp),
        "rs_down" => Err(Stick::RDown),
        "rs_left" => Err(Stick::RLeft),
        "rs_right" => Err(Stick::RRight),
        _ => return None,
    })
}

fn parse_script(s: &str) -> Result<Vec<Step>, String> {
    let mut steps = Vec::new();
    for entry in s.split(';').map(str::trim).filter(|e| !e.is_empty()) {
        let parts: Vec<&str> = entry.split(':').collect();
        if parts.len() < 2 || parts.len() > 3 {
            return Err(format!("bad entry '{entry}' (want flip:buttons[:hold])"));
        }
        let flip = parts[0].trim().parse::<u64>().map_err(|_| format!("bad flip in '{entry}'"))?;
        let hold = match parts.get(2) {
            Some(h) => h.trim().parse::<u64>().map_err(|_| format!("bad hold in '{entry}'"))?.max(1),
            None => 6,
        };
        let mut step = Step { flip, hold, buttons: 0, sticks: Vec::new() };
        for name in parts[1].split('+').map(|n| n.trim().to_ascii_lowercase()) {
            match parse_name(&name) {
                Some(Ok(b)) => step.buttons |= b,
                Some(Err(st)) => step.sticks.push(st),
                None => return Err(format!("unknown button '{name}' in '{entry}'")),
            }
        }
        steps.push(step);
    }
    Ok(steps)
}

fn script() -> &'static [Step] {
    static SCRIPT: OnceLock<Vec<Step>> = OnceLock::new();
    SCRIPT.get_or_init(|| match std::env::var("TLG_INPUT") {
        Ok(s) => match parse_script(&s) {
            Ok(v) => {
                crate::info!("input script: {} steps", v.len());
                v
            }
            Err(e) => {
                crate::error!("TLG_INPUT ignored: {e}");
                Vec::new()
            }
        },
        Err(_) => Vec::new(),
    })
}

fn apply_stick(st: &mut PadState, s: Stick) {
    match s {
        Stick::LUp => st.ly = 0,
        Stick::LDown => st.ly = 255,
        Stick::LLeft => st.lx = 0,
        Stick::LRight => st.lx = 255,
        Stick::RUp => st.ry = 0,
        Stick::RDown => st.ry = 255,
        Stick::RLeft => st.rx = 0,
        Stick::RRight => st.rx = 255,
    }
}

/// Presses scheduled at run time (control channel), relative to the flip at
/// which they were requested.
static LIVE: parking_lot::Mutex<Vec<Step>> = parking_lot::Mutex::new(Vec::new());

/// Presses held for a stretch of real time (process milliseconds) instead of
/// a number of flips: the same input at any frame rate.
static LIVE_MS: parking_lot::Mutex<Vec<(f64, f64, Step)>> = parking_lot::Mutex::new(Vec::new());

/// Schedules a press of `names` (joined with '+') held for `ms` milliseconds
/// of real time from now.
pub fn press_ms(names: &str, ms: u64) -> Result<(), String> {
    let mut steps = parse_script(&format!("0:{names}:1"))?;
    let now = crate::log::elapsed_ms();
    let mut live = LIVE_MS.lock();
    live.retain(|s| s.1 > now);
    for s in steps.drain(..) {
        live.push((now, now + ms as f64, s));
    }
    Ok(())
}

/// Schedules a press of `names` (joined with '+') for `hold` flips starting
/// at the next flip.
pub fn press_now(names: &str, hold: u64) -> Result<(), String> {
    let flip = crate::gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed) + 1;
    let mut steps = parse_script(&format!("{flip}:{names}:{}", hold.max(1)))?;
    let mut live = LIVE.lock();
    live.retain(|s| s.flip + s.hold > flip);
    live.append(&mut steps);
    Ok(())
}

fn scripted(st: &mut PadState, flip: u64) {
    for step in LIVE.lock().iter() {
        if flip >= step.flip && flip < step.flip + step.hold {
            st.buttons |= step.buttons;
            for &s in &step.sticks {
                apply_stick(st, s);
            }
        }
    }
    let now = crate::log::elapsed_ms();
    for (a, b, step) in LIVE_MS.lock().iter() {
        if now >= *a && now < *b {
            st.buttons |= step.buttons;
            for &s in &step.sticks {
                apply_stick(st, s);
            }
        }
    }
    static LOGGED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    for (i, step) in script().iter().enumerate() {
        if flip >= step.flip && flip < step.flip + step.hold {
            st.buttons |= step.buttons;
            for &s in &step.sticks {
                apply_stick(st, s);
            }
            if LOGGED.fetch_max(i + 1, std::sync::atomic::Ordering::Relaxed) <= i {
                crate::info!("input script: flip {flip} step {i} ({:#x} {:?}) for {} flips", step.buttons, step.sticks, step.hold);
            }
        }
    }
}

pub fn xinput_state() -> Option<XINPUT_STATE> {
    for i in 0..4 {
        let mut s: XINPUT_STATE = unsafe { std::mem::zeroed() };
        if unsafe { XInputGetState(i, &mut s) } == 0 {
            return Some(s);
        }
    }
    None
}

fn axis(v: i16, invert: bool) -> u8 {
    let f = (v as f32 + 32768.0) / 65535.0;
    let f = if invert { 1.0 - f } else { f };
    (f * 255.0).round().clamp(0.0, 255.0) as u8
}

fn xinput(st: &mut PadState) {
    let Some(s) = xinput_state() else { return };
    let g = s.Gamepad;
    let map = [
        (0x0001, BTN_UP),
        (0x0002, BTN_DOWN),
        (0x0004, BTN_LEFT),
        (0x0008, BTN_RIGHT),
        (0x0010, BTN_OPTIONS),  // Start
        (0x0020, BTN_TOUCHPAD), // Back
        (0x0040, BTN_L3),
        (0x0080, BTN_R3),
        (0x0100, BTN_L1),
        (0x0200, BTN_R1),
        (0x1000, BTN_CROSS),
        (0x2000, BTN_CIRCLE),
        (0x4000, BTN_SQUARE),
        (0x8000, BTN_TRIANGLE),
    ];
    for (x, p) in map {
        if g.wButtons & x != 0 {
            st.buttons |= p;
        }
    }
    st.l2 = st.l2.max(g.bLeftTrigger);
    st.r2 = st.r2.max(g.bRightTrigger);
    st.lx = axis(g.sThumbLX, false);
    st.ly = axis(g.sThumbLY, true);
    st.rx = axis(g.sThumbRX, false);
    st.ry = axis(g.sThumbRY, true);
}

/// The keyboard and mouse bindings: input.txt over the defaults (safe mode:
/// the defaults, the file left alone).
fn settings() -> &'static Bindings {
    static S: OnceLock<Bindings> = OnceLock::new();
    S.get_or_init(|| {
        let s = match crate::config::try_paths() {
            Some(p) if !crate::config::safe_mode() => {
                let path = p.input_file();
                if !path.exists() {
                    let _ = std::fs::create_dir_all(&p.user);
                    if let Err(e) = std::fs::write(&path, tlg_config::bindings::DEFAULTS.replace('\n', "\r\n")) {
                        crate::warn!("could not write {}: {e}", path.display());
                    }
                }
                let (b, errs) = Bindings::load(&path);
                for e in errs {
                    crate::warn!("{e} (that line ignored)");
                }
                b
            }
            _ => Bindings::default(),
        };
        let conflicts = s.conflicts();
        if !conflicts.is_empty() {
            let list: Vec<String> = conflicts.iter().map(|(i, cs)| format!("{} -> {} controls", tlg_config::bindings::input_name(*i), cs.len())).collect();
            crate::info!("input: inputs bound to several controls (all of them act): {}", list.join(", "));
        }
        crate::gpu::vk::window::RELEASE_KEY.store(s.release_key as u32, std::sync::atomic::Ordering::Relaxed);
        s
    })
}

/// Whether a click captures the mouse (mouse look, menu mouse or a mouse
/// button bound).
pub fn mouse_capture_enabled() -> bool {
    let s = settings();
    s.mouse_look || s.menu_mouse || s.map.iter().any(|(_, ins)| ins.iter().any(|i| !matches!(i, Input::Key(_))))
}

/// Wheel notches turn into presses of this length.
const WHEEL_PRESS_MS: f64 = 120.0;

struct MouseState {
    /// When the previous pad read sampled the mouse (process ms).
    last: f64,
    /// Right-stick deflection given then, reused by reads in quick succession.
    rs: (f32, f32),
    wheel_up_until: f64,
    wheel_down_until: f64,
    /// The wheel presses were made in a menu (d-pad up/down).
    wheel_menu: bool,
    /// Mouse buttons pressed in a menu (bits as `MOUSE_BUTTONS`), held as
    /// menu input until released.
    menu_buttons: u32,
    /// The buttons down at the previous read.
    prev_buttons: u32,
    /// The button `pause_back` gave when it went down (held until released).
    pause_latch: Option<u32>,
}

static MOUSE: parking_lot::Mutex<MouseState> =
    parking_lot::Mutex::new(MouseState { last: 0.0, rs: (0.0, 0.0), wheel_up_until: 0.0, wheel_down_until: 0.0, wheel_menu: false, menu_buttons: 0, prev_buttons: 0, pause_latch: None });

/// Raw counts per second that give a full right-stick deflection at
/// sensitivity 1 (mouse look where the game camera hook does not apply).
const FULL_RATE: f32 = 2500.0;

/// Camera turn per raw count at sensitivity 1 (degrees; 9,000 counts per
/// full turn, e.g. 29 cm at 800 counts per inch).
const DEG_PER_COUNT: f32 = 0.04;

/// Samples the mouse for one pad read: the right-stick deflection the motion
/// since the previous read amounts to, and the wheel presses active now.
fn mouse_sample(s: &Bindings, m: &mut MouseState, in_menu: bool) -> ((f32, f32), bool, bool) {
    use crate::gpu::vk::window;
    let now = crate::log::elapsed_ms();
    let captured = window::mouse_live();
    let dt = now - m.last;
    // (Reads within a few milliseconds of each other share one sample.)
    if dt >= 4.0 || !captured {
        let (dx, dy, wheel) = window::take_mouse();
        m.last = now;
        let sx = if s.invert_x { -1.0 } else { 1.0 } * s.sensitivity_x;
        let sy = if s.invert_y { -1.0 } else { 1.0 } * s.sensitivity_y;
        let rad = DEG_PER_COUNT.to_radians();
        if !captured || !s.mouse_look {
            m.rs = (0.0, 0.0);
        } else if crate::title_patches::add_camera_turn(-(dx as f32) * rad * sx, -(dy as f32) * rad * sy) {
            // Straight into the game camera (right: turn right; forward: look up).
            m.rs = (0.0, 0.0);
        } else {
            let secs = (dt.min(250.0) / 1000.0) as f32;
            let k = 1.0 / FULL_RATE / secs.max(1e-3);
            m.rs = ((dx as f32 * k * sx).clamp(-1.0, 1.0), (dy as f32 * k * sy).clamp(-1.0, 1.0));
        }
        if wheel != 0 {
            m.wheel_menu = in_menu && s.menu_mouse;
        }
        if wheel > 0 {
            m.wheel_up_until = now + WHEEL_PRESS_MS;
        } else if wheel < 0 {
            m.wheel_down_until = now + WHEEL_PRESS_MS;
        }
    }
    (m.rs, now < m.wheel_up_until, now < m.wheel_down_until)
}

fn keyboard_mouse(st: &mut PadState) {
    use crate::gpu::vk::window;
    let s = settings();
    // Menus: the title screen and main menu, the pause menu, loading
    // screens, most cutscenes. (Without the camera hook: always "playing".)
    let in_menu = crate::title_patches::in_menu() == Some(true);
    {
        // (Logged at each change: 1 play, 2 menu.)
        static CONTEXT: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
        let c = if in_menu { 2 } else { 1 };
        if CONTEXT.swap(c, std::sync::atomic::Ordering::Relaxed) != c {
            crate::info!("input: {} (keyboard/mouse menu mapping {})", if in_menu { "menu" } else { "play" }, if in_menu { "on" } else { "off" });
        }
    }
    let mut m = MOUSE.lock();
    let (rs, wheel_up, wheel_down) = mouse_sample(s, &mut m, in_menu);
    let buttons = window::MOUSE_BUTTONS.load(std::sync::atomic::Ordering::Relaxed);
    // Mouse buttons pressed in a menu stay menu input until released (and
    // buttons held from play stay play input).
    m.menu_buttons &= buttons;
    if s.menu_mouse && in_menu {
        m.menu_buttons |= buttons & !m.prev_buttons;
    }
    m.prev_buttons = buttons;
    let menu_buttons = m.menu_buttons;
    let wheel_menu = m.wheel_menu;
    if menu_buttons & 1 != 0 {
        st.buttons |= BTN_CROSS;
    }
    if menu_buttons & 2 != 0 {
        st.buttons |= BTN_CIRCLE;
    }
    if wheel_menu && wheel_up {
        st.buttons |= BTN_UP;
    }
    if wheel_menu && wheel_down {
        st.buttons |= BTN_DOWN;
    }
    let active = |i: &Input| match *i {
        Input::Key(k) => window::key_down(k),
        Input::Mouse(b) => buttons & !menu_buttons & (1 << b) != 0,
        Input::WheelUp => wheel_up && !wheel_menu,
        Input::WheelDown => wheel_down && !wheel_menu,
    };
    let mut dirs: Vec<Stick> = Vec::new();
    let mut walk = false;
    let mut pause = false;
    for (control, inputs) in &s.map {
        if !inputs.iter().any(active) {
            continue;
        }
        match *control {
            Control::Button(b) => st.buttons |= b,
            Control::Stick(d) => dirs.push(d),
            Control::Walk => walk = true,
            Control::PauseBack => pause = true,
        }
    }
    // Pause / back: decided when it goes down.
    if pause {
        let b = *m.pause_latch.get_or_insert(if in_menu { BTN_CIRCLE } else { BTN_OPTIONS });
        st.buttons |= b;
    } else {
        m.pause_latch = None;
    }
    drop(m);
    if st.buttons & BTN_L2 != 0 {
        st.l2 = 255;
    }
    if st.buttons & BTN_R2 != 0 {
        st.r2 = 255;
    }
    let full = if walk { 0.5 } else { 1.0 };
    let axis_keys = |neg: Stick, pos: Stick, cur: u8, scale: f32| -> u8 {
        match (dirs.contains(&neg), dirs.contains(&pos)) {
            (true, false) => (128.0 - 128.0 * scale).round().clamp(0.0, 255.0) as u8,
            (false, true) => (127.0 + 128.0 * scale).round().clamp(0.0, 255.0) as u8,
            _ => cur,
        }
    };
    st.lx = axis_keys(Stick::LLeft, Stick::LRight, st.lx, full);
    st.ly = axis_keys(Stick::LUp, Stick::LDown, st.ly, full);
    st.rx = axis_keys(Stick::RLeft, Stick::RRight, st.rx, 1.0);
    st.ry = axis_keys(Stick::RUp, Stick::RDown, st.ry, 1.0);
    // Mouse look adds to the right stick (keys or a controller may also move it).
    if rs != (0.0, 0.0) {
        let add = |cur: u8, d: f32| ((cur as f32 - 128.0) / 127.0 + d).clamp(-1.0, 1.0);
        let to_u8 = |v: f32| (128.0 + v * 127.0).round().clamp(0.0, 255.0) as u8;
        st.rx = to_u8(add(st.rx, rs.0));
        st.ry = to_u8(add(st.ry, rs.1));
    }
}

/// A DualShock 4 / DualSense read over HID (`padhid`): its sticks and
/// triggers replace the XInput ones (the same physical pad when a wrapper
/// also presents it as XInput), buttons add up.
fn hid(st: &mut PadState) {
    let Some(h) = crate::padhid::state() else { return };
    st.buttons |= h.buttons;
    (st.lx, st.ly, st.rx, st.ry) = (h.lx, h.ly, h.rx, h.ry);
    st.l2 = st.l2.max(h.l2);
    st.r2 = st.r2.max(h.r2);
}

/// Current combined pad state.
pub fn pad_state() -> PadState {
    let mut st = PadState::default();
    xinput(&mut st);
    hid(&mut st);
    keyboard_mouse(&mut st);
    scripted(&mut st, crate::gpu::FRAMES.load(std::sync::atomic::Ordering::Relaxed));
    if st.l2 > 30 {
        st.buttons |= BTN_L2;
    }
    if st.r2 > 30 {
        st.buttons |= BTN_R2;
    }
    if st.buttons & BTN_L2 != 0 {
        st.l2 = st.l2.max(255);
    }
    if st.buttons & BTN_R2 != 0 {
        st.r2 = st.r2.max(255);
    }
    st
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn script_parses_and_applies() {
        let s = parse_script("700:cross; 760:down+ls_left:3 ;800:Cross").unwrap();
        assert_eq!(s.len(), 3);
        assert_eq!(s[0], Step { flip: 700, hold: 6, buttons: BTN_CROSS, sticks: vec![] });
        assert_eq!(s[1].buttons, BTN_DOWN);
        assert_eq!(s[1].sticks, vec![Stick::LLeft]);
        assert_eq!(s[1].hold, 3);
        assert!(parse_script("700:jump").is_err());
        assert!(parse_script("x:cross").is_err());
    }
}
