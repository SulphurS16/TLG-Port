//! Host input aggregation for the DualShock 4 pad: XInput controller, the
//! output window's keyboard and mouse (bindings in `local\input.txt`, written
//! with the defaults on first use), and a scripted timeline for reproducible
//! runs.
//!
//! Default keyboard layout (window focused; see `DEFAULTS`):
//!   W/A/S/D left stick (Left Ctrl: half deflection) · I/J/K/L right stick ·
//!   arrow keys d-pad · Enter or E = Cross · Escape or Space = Circle ·
//!   R = Square · F = Triangle · U = L1 · O = R1 · Y = L2 · P = R2 · C = L3 ·
//!   V = R3 · Tab = Options · T = Touchpad.
//! Mouse: a click into the window captures it (raw counts, no pointer
//! acceleration); F1 or leaving the window releases it. Mouse look turns the
//! game camera by the raw counts (`title_patches` camera hook: no dead zone,
//! response curve or smoothing); while that camera is not updating it drives
//! the right stick instead.
//!
//! Scripted input (`TLG_INPUT`): `flip:buttons[:hold]` entries separated by
//! `;`, e.g. `TLG_INPUT=700:cross;760:down:3;800:cross`. `buttons` joins names
//! with `+` (cross, circle, square, triangle, up, down, left, right, l1, r1,
//! l2, r2, l3, r3, options, touchpad, ls_up/down/left/right,
//! rs_up/down/left/right). Each press lasts `hold` GPU flips (default 6).

use std::sync::OnceLock;
use windows_sys::Win32::UI::Input::XboxController::{XINPUT_STATE, XInputGetState};

pub const BTN_L3: u32 = 0x2;
pub const BTN_R3: u32 = 0x4;
pub const BTN_OPTIONS: u32 = 0x8;
pub const BTN_UP: u32 = 0x10;
pub const BTN_RIGHT: u32 = 0x20;
pub const BTN_DOWN: u32 = 0x40;
pub const BTN_LEFT: u32 = 0x80;
pub const BTN_L2: u32 = 0x100;
pub const BTN_R2: u32 = 0x200;
pub const BTN_L1: u32 = 0x400;
pub const BTN_R1: u32 = 0x800;
pub const BTN_TRIANGLE: u32 = 0x1000;
pub const BTN_CIRCLE: u32 = 0x2000;
pub const BTN_CROSS: u32 = 0x4000;
pub const BTN_SQUARE: u32 = 0x8000;
pub const BTN_TOUCHPAD: u32 = 0x100000;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stick {
    LUp,
    LDown,
    LLeft,
    LRight,
    RUp,
    RDown,
    RLeft,
    RRight,
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

/// A bindable host input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Input {
    Key(u8),
    /// Mouse button bit (0 left, 1 right, 2 middle, 3/4 side buttons).
    Mouse(u8),
    WheelUp,
    WheelDown,
}

/// What a binding drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Control {
    Button(u32),
    Stick(Stick),
    /// Left stick at half deflection while held.
    Walk,
}

/// Bindable controls by their `input.txt` name.
const CONTROLS: [(&str, Control); 25] = [
    ("cross", Control::Button(BTN_CROSS)),
    ("circle", Control::Button(BTN_CIRCLE)),
    ("square", Control::Button(BTN_SQUARE)),
    ("triangle", Control::Button(BTN_TRIANGLE)),
    ("up", Control::Button(BTN_UP)),
    ("down", Control::Button(BTN_DOWN)),
    ("left", Control::Button(BTN_LEFT)),
    ("right", Control::Button(BTN_RIGHT)),
    ("l1", Control::Button(BTN_L1)),
    ("r1", Control::Button(BTN_R1)),
    ("l2", Control::Button(BTN_L2)),
    ("r2", Control::Button(BTN_R2)),
    ("l3", Control::Button(BTN_L3)),
    ("r3", Control::Button(BTN_R3)),
    ("options", Control::Button(BTN_OPTIONS)),
    ("touchpad", Control::Button(BTN_TOUCHPAD)),
    ("ls_up", Control::Stick(Stick::LUp)),
    ("ls_down", Control::Stick(Stick::LDown)),
    ("ls_left", Control::Stick(Stick::LLeft)),
    ("ls_right", Control::Stick(Stick::LRight)),
    ("rs_up", Control::Stick(Stick::RUp)),
    ("rs_down", Control::Stick(Stick::RDown)),
    ("rs_left", Control::Stick(Stick::RLeft)),
    ("rs_right", Control::Stick(Stick::RRight)),
    ("walk", Control::Walk),
];

/// Default bindings (also the commented `input.txt` written on first use).
const DEFAULTS: &str = "\
# The Last Guardian (tlg): keyboard and mouse settings. Edit and restart tlg;
# delete this file to get the defaults back.
#
# Bindings: <control> = <input>[, <input>...], or 'none'. Controls are the
# DualShock 4's: cross circle square triangle up down left right (d-pad) l1 r1
# l2 r2 l3 r3 options touchpad, stick directions ls_up ls_down ls_left
# ls_right rs_up rs_down rs_left rs_right, and walk (left stick at half
# deflection while held).
# Inputs: A-Z, 0-9, F1-F12, Space, Enter, Escape, Tab, Backspace, Shift,
# LShift, RShift, Ctrl, LCtrl, RCtrl, Alt, LAlt, RAlt, Up, Down, Left, Right,
# Insert, Delete, Home, End, PageUp, PageDown, Numpad0-Numpad9, Minus, Equals,
# LBracket, RBracket, Semicolon, Quote, Comma, Period, Slash, Backslash,
# Grave, CapsLock; mouse: MouseLeft, MouseRight, MouseMiddle, Mouse4, Mouse5,
# WheelUp, WheelDown (mouse inputs count while the mouse is captured).

cross = Enter, E
circle = Escape, Space
square = R
triangle = F
up = Up
down = Down
left = Left
right = Right
l1 = U
r1 = O
l2 = Y
r2 = P
l3 = C
r3 = V
options = Tab
touchpad = T
ls_up = W
ls_down = S
ls_left = A
ls_right = D
rs_up = I
rs_down = K
rs_left = J
rs_right = L
walk = LCtrl

# Mouse: a click into the window captures the mouse, release_mouse (or
# leaving the window) frees it. Mouse look turns the game camera directly by
# the raw motion (no acceleration, smoothing or dead zone; sensitivity 1.0 is
# 0.04 degrees per count); where the game camera is not in charge (other
# camera modes) it moves the right stick instead.
mouse_look = on
mouse_sensitivity = 1.0
mouse_invert_x = off
mouse_invert_y = off
release_mouse = F1
";

fn key_code(name: &str) -> Option<u8> {
    let n = name.to_ascii_lowercase();
    let b = n.as_bytes();
    if b.len() == 1 && (b[0].is_ascii_lowercase() || b[0].is_ascii_digit()) {
        return Some(b[0].to_ascii_uppercase());
    }
    if let Some(f) = n.strip_prefix('f').and_then(|d| d.parse::<u8>().ok()).filter(|f| (1..=24).contains(f)) {
        return Some(0x6F + f);
    }
    if let Some(d) = n.strip_prefix("numpad").and_then(|d| d.parse::<u8>().ok()).filter(|d| *d <= 9) {
        return Some(0x60 + d);
    }
    Some(match n.as_str() {
        "space" => 0x20,
        "enter" | "return" => 0x0D,
        "escape" | "esc" => 0x1B,
        "tab" => 0x09,
        "backspace" => 0x08,
        "shift" => 0x10,
        "lshift" => 0xA0,
        "rshift" => 0xA1,
        "ctrl" | "control" => 0x11,
        "lctrl" => 0xA2,
        "rctrl" => 0xA3,
        "alt" => 0x12,
        "lalt" => 0xA4,
        "ralt" => 0xA5,
        "up" => 0x26,
        "down" => 0x28,
        "left" => 0x25,
        "right" => 0x27,
        "insert" => 0x2D,
        "delete" => 0x2E,
        "home" => 0x24,
        "end" => 0x23,
        "pageup" => 0x21,
        "pagedown" => 0x22,
        "minus" => 0xBD,
        "equals" => 0xBB,
        "lbracket" => 0xDB,
        "rbracket" => 0xDD,
        "semicolon" => 0xBA,
        "quote" => 0xDE,
        "comma" => 0xBC,
        "period" => 0xBE,
        "slash" => 0xBF,
        "backslash" => 0xDC,
        "grave" => 0xC0,
        "capslock" => 0x14,
        _ => return None,
    })
}

fn parse_input(name: &str) -> Option<Input> {
    Some(match name.to_ascii_lowercase().as_str() {
        "mouseleft" => Input::Mouse(0),
        "mouseright" => Input::Mouse(1),
        "mousemiddle" => Input::Mouse(2),
        "mouse4" => Input::Mouse(3),
        "mouse5" => Input::Mouse(4),
        "wheelup" => Input::WheelUp,
        "wheeldown" => Input::WheelDown,
        _ => Input::Key(key_code(name)?),
    })
}

#[derive(Clone, Debug)]
struct Settings {
    bindings: Vec<(Control, Vec<Input>)>,
    mouse_look: bool,
    sensitivity: f32,
    invert_x: bool,
    invert_y: bool,
    release_key: u8,
}

/// Parses `input.txt`: settings over the defaults (a binding line replaces
/// that control's default inputs). Errors name the offending line.
fn parse_settings(text: &str, base: Option<&Settings>) -> Result<Settings, String> {
    let mut s = base.cloned().unwrap_or(Settings { bindings: Vec::new(), mouse_look: true, sensitivity: 1.0, invert_x: false, invert_y: false, release_key: 0x70 });
    let flag = |v: &str, line: &str| -> Result<bool, String> {
        match v.to_ascii_lowercase().as_str() {
            "on" | "true" | "yes" | "1" => Ok(true),
            "off" | "false" | "no" | "0" => Ok(false),
            _ => Err(format!("'{line}': want on or off")),
        }
    };
    for raw in text.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else { return Err(format!("'{line}': want <name> = <value>")) };
        let (k, v) = (k.trim().to_ascii_lowercase(), v.trim());
        match k.as_str() {
            "mouse_look" => s.mouse_look = flag(v, line)?,
            "mouse_invert_x" => s.invert_x = flag(v, line)?,
            "mouse_invert_y" => s.invert_y = flag(v, line)?,
            "mouse_sensitivity" => s.sensitivity = v.parse::<f32>().ok().filter(|x| x.is_finite() && *x > 0.0).ok_or_else(|| format!("'{line}': want a positive number"))?,
            "release_mouse" => s.release_key = key_code(v).ok_or_else(|| format!("'{line}': unknown key '{v}'"))?,
            _ => {
                let Some(&(_, control)) = CONTROLS.iter().find(|(n, _)| *n == k) else { return Err(format!("'{line}': unknown control '{k}'")) };
                let mut inputs = Vec::new();
                if !v.eq_ignore_ascii_case("none") {
                    for name in v.split(',').map(str::trim).filter(|n| !n.is_empty()) {
                        inputs.push(parse_input(name).ok_or_else(|| format!("'{line}': unknown input '{name}'"))?);
                    }
                }
                s.bindings.retain(|(c, _)| *c != control);
                s.bindings.push((control, inputs));
            }
        }
    }
    Ok(s)
}

fn settings() -> &'static Settings {
    static S: OnceLock<Settings> = OnceLock::new();
    S.get_or_init(|| {
        let defaults = parse_settings(DEFAULTS, None).expect("default input settings");
        let path = crate::data_dir().map(|d| d.join("input.txt"));
        let s = match path.as_ref().map(|p| (p, std::fs::read_to_string(p))) {
            Some((_, Ok(text))) => match parse_settings(&text, Some(&defaults)) {
                Ok(s) => s,
                Err(e) => {
                    crate::error!("input.txt: {e}; using the default keyboard and mouse settings");
                    defaults
                }
            },
            Some((p, Err(_))) => {
                if let Err(e) = std::fs::write(p, DEFAULTS) {
                    crate::warn!("could not write {}: {e}", p.display());
                }
                defaults
            }
            None => defaults,
        };
        crate::gpu::vk::window::RELEASE_KEY.store(s.release_key as u32, std::sync::atomic::Ordering::Relaxed);
        s
    })
}

/// Whether a click captures the mouse (mouse look or a mouse button bound).
pub fn mouse_capture_enabled() -> bool {
    let s = settings();
    s.mouse_look || s.bindings.iter().any(|(_, ins)| ins.iter().any(|i| !matches!(i, Input::Key(_))))
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
}

static MOUSE: parking_lot::Mutex<MouseState> = parking_lot::Mutex::new(MouseState { last: 0.0, rs: (0.0, 0.0), wheel_up_until: 0.0, wheel_down_until: 0.0 });

/// Raw counts per second that give a full right-stick deflection at
/// sensitivity 1 (mouse look where the game camera hook does not apply).
const FULL_RATE: f32 = 2500.0;

/// Camera turn per raw count at sensitivity 1 (degrees; 9,000 counts per
/// full turn, e.g. 29 cm at 800 counts per inch).
const DEG_PER_COUNT: f32 = 0.04;

/// Samples the mouse for one pad read: the right-stick deflection the motion
/// since the previous read amounts to, and the wheel presses active now.
fn mouse_sample(s: &Settings) -> ((f32, f32), bool, bool) {
    use crate::gpu::vk::window;
    let now = crate::log::elapsed_ms();
    let mut m = MOUSE.lock();
    let captured = window::mouse_live();
    let dt = now - m.last;
    // (Reads within a few milliseconds of each other share one sample.)
    if dt >= 4.0 || !captured {
        let (dx, dy, wheel) = window::take_mouse();
        m.last = now;
        let sx = if s.invert_x { -1.0 } else { 1.0 };
        let sy = if s.invert_y { -1.0 } else { 1.0 };
        let rad = (s.sensitivity * DEG_PER_COUNT).to_radians();
        if !captured || !s.mouse_look {
            m.rs = (0.0, 0.0);
        } else if crate::title_patches::add_camera_turn(-(dx as f32) * rad * sx, -(dy as f32) * rad * sy) {
            // Straight into the game camera (right: turn right; forward: look up).
            m.rs = (0.0, 0.0);
        } else {
            let secs = (dt.min(250.0) / 1000.0) as f32;
            let k = s.sensitivity / FULL_RATE / secs.max(1e-3);
            m.rs = ((dx as f32 * k * sx).clamp(-1.0, 1.0), (dy as f32 * k * sy).clamp(-1.0, 1.0));
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
    let (rs, wheel_up, wheel_down) = mouse_sample(s);
    let buttons = window::MOUSE_BUTTONS.load(std::sync::atomic::Ordering::Relaxed);
    let active = |i: &Input| match *i {
        Input::Key(k) => window::key_down(k),
        Input::Mouse(b) => buttons & (1 << b) != 0,
        Input::WheelUp => wheel_up,
        Input::WheelDown => wheel_down,
    };
    let mut dirs: Vec<Stick> = Vec::new();
    let mut walk = false;
    for (control, inputs) in &s.bindings {
        if !inputs.iter().any(active) {
            continue;
        }
        match *control {
            Control::Button(b) => st.buttons |= b,
            Control::Stick(d) => dirs.push(d),
            Control::Walk => walk = true,
        }
    }
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

    #[test]
    fn default_settings_parse() {
        let s = parse_settings(DEFAULTS, None).unwrap();
        assert_eq!(s.bindings.len(), CONTROLS.len());
        let cross = &s.bindings.iter().find(|(c, _)| *c == Control::Button(BTN_CROSS)).unwrap().1;
        assert_eq!(cross, &vec![Input::Key(0x0D), Input::Key(b'E')]);
        assert!(s.mouse_look);
        assert_eq!(s.release_key, 0x70);
    }

    #[test]
    fn user_settings_override_defaults() {
        let d = parse_settings(DEFAULTS, None).unwrap();
        let s = parse_settings("triangle = Space, MouseRight\ncircle = none\nmouse_sensitivity = 2.5 # faster\nmouse_invert_y = on\n", Some(&d)).unwrap();
        let get = |c: Control| s.bindings.iter().find(|(x, _)| *x == c).unwrap().1.clone();
        assert_eq!(get(Control::Button(BTN_TRIANGLE)), vec![Input::Key(0x20), Input::Mouse(1)]);
        assert!(get(Control::Button(BTN_CIRCLE)).is_empty());
        assert_eq!(get(Control::Button(BTN_CROSS)), vec![Input::Key(0x0D), Input::Key(b'E')]);
        assert_eq!(s.sensitivity, 2.5);
        assert!(s.invert_y && !s.invert_x);
        assert!(parse_settings("jump = Space", Some(&d)).is_err());
        assert!(parse_settings("cross = Hyper", Some(&d)).is_err());
        assert!(parse_settings("mouse_sensitivity = -1", Some(&d)).is_err());
    }

    #[test]
    fn key_names() {
        assert_eq!(key_code("a"), Some(b'A'));
        assert_eq!(key_code("7"), Some(b'7'));
        assert_eq!(key_code("F1"), Some(0x70));
        assert_eq!(key_code("f12"), Some(0x7B));
        assert_eq!(key_code("Numpad5"), Some(0x65));
        assert_eq!(key_code("LCtrl"), Some(0xA2));
        assert_eq!(key_code("nope"), None);
    }
}
