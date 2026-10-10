//! Host input aggregation for the DualShock 4 pad: XInput controller, the
//! output window's keyboard, and a scripted timeline for reproducible runs.
//!
//! Keyboard layout (window focused):
//!   W/A/S/D left stick · I/J/K/L right stick · arrow keys d-pad
//!   Enter or E = Cross · Escape or Space = Circle · R = Square · F = Triangle
//!   U = L1 · O = R1 · Y = L2 · P = R2 · C = L3 · V = R3 · Tab = Options · T = Touchpad
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

fn keyboard(st: &mut PadState) {
    use crate::gpu::vk::window::key_down;
    let k = |c: u8| key_down(c);
    let map: [(&[u8], u32); 16] = [
        (&[0x0D, b'E'], BTN_CROSS),
        (&[0x1B, 0x20], BTN_CIRCLE),
        (&[b'R'], BTN_SQUARE),
        (&[b'F'], BTN_TRIANGLE),
        (&[0x26], BTN_UP),
        (&[0x28], BTN_DOWN),
        (&[0x25], BTN_LEFT),
        (&[0x27], BTN_RIGHT),
        (&[b'U'], BTN_L1),
        (&[b'O'], BTN_R1),
        (&[b'Y'], BTN_L2),
        (&[b'P'], BTN_R2),
        (&[b'C'], BTN_L3),
        (&[b'V'], BTN_R3),
        (&[0x09], BTN_OPTIONS),
        (&[b'T'], BTN_TOUCHPAD),
    ];
    for (keys, b) in map {
        if keys.iter().any(|&c| k(c)) {
            st.buttons |= b;
        }
    }
    if k(b'Y') {
        st.l2 = 255;
    }
    if k(b'P') {
        st.r2 = 255;
    }
    let stick = |neg: u8, pos: u8, cur: u8| -> u8 {
        match (k(neg), k(pos)) {
            (true, false) => 0,
            (false, true) => 255,
            _ => cur,
        }
    };
    st.lx = stick(b'A', b'D', st.lx);
    st.ly = stick(b'W', b'S', st.ly);
    st.rx = stick(b'J', b'L', st.rx);
    st.ry = stick(b'I', b'K', st.ry);
}

/// Current combined pad state.
pub fn pad_state() -> PadState {
    let mut st = PadState::default();
    xinput(&mut st);
    keyboard(&mut st);
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
