//! Keyboard and mouse bindings (`input.txt` in the user folder): every
//! DualShock 4 button and stick direction can be driven by keys and mouse
//! inputs (the launcher edits two per control; the file takes any number),
//! plus the mouse-look settings. Unknown names and bad values are reported
//! and leave that line's control at its default.

use crate::kv::{KvFile, split_line, strip_comment};
use std::path::Path;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stick {
    LUp,
    LDown,
    LLeft,
    LRight,
    RUp,
    RDown,
    RLeft,
    RRight,
}

/// A bindable host input.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Input {
    /// A Windows virtual-key code (left/right modifiers apart: 0xA0-0xA5).
    Key(u8),
    /// Mouse button bit (0 left, 1 right, 2 middle, 3/4 side buttons).
    Mouse(u8),
    WheelUp,
    WheelDown,
}

/// What a binding drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Control {
    Button(u32),
    Stick(Stick),
    /// Left stick at half deflection while held.
    Walk,
    /// Options while playing, Circle (back) in menus.
    PauseBack,
}

pub struct ControlInfo {
    /// The name in input.txt.
    pub name: &'static str,
    pub control: Control,
    /// For the launcher.
    pub label: &'static str,
    pub group: &'static str,
}

const fn ci(name: &'static str, control: Control, label: &'static str, group: &'static str) -> ControlInfo {
    ControlInfo { name, control, label, group }
}

/// Every bindable control, in the launcher's order. (Action names from the
/// game's Controls page; the others by their DualShock 4 names.)
pub const CONTROLS: &[ControlInfo] = &[
    ci("ls_up", Control::Stick(Stick::LUp), "Move forward", "Movement"),
    ci("ls_down", Control::Stick(Stick::LDown), "Move back", "Movement"),
    ci("ls_left", Control::Stick(Stick::LLeft), "Move left", "Movement"),
    ci("ls_right", Control::Stick(Stick::LRight), "Move right", "Movement"),
    ci("walk", Control::Walk, "Walk (half speed while held)", "Movement"),
    ci("triangle", Control::Button(BTN_TRIANGLE), "Jump (Triangle)", "Actions"),
    ci("circle", Control::Button(BTN_CIRCLE), "Pick up / grab (Circle)", "Actions"),
    ci("cross", Control::Button(BTN_CROSS), "Crouch / confirm (Cross)", "Actions"),
    ci("square", Control::Button(BTN_SQUARE), "Tackle / throw (Square)", "Actions"),
    ci("r1", Control::Button(BTN_R1), "R1", "Actions"),
    ci("l1", Control::Button(BTN_L1), "L1", "Actions"),
    ci("r2", Control::Button(BTN_R2), "R2", "Actions"),
    ci("l2", Control::Button(BTN_L2), "L2", "Actions"),
    ci("r3", Control::Button(BTN_R3), "R3 (right stick click)", "Actions"),
    ci("l3", Control::Button(BTN_L3), "L3 (left stick click)", "Actions"),
    ci("up", Control::Button(BTN_UP), "D-pad up", "D-pad"),
    ci("down", Control::Button(BTN_DOWN), "D-pad down", "D-pad"),
    ci("left", Control::Button(BTN_LEFT), "D-pad left", "D-pad"),
    ci("right", Control::Button(BTN_RIGHT), "D-pad right", "D-pad"),
    ci("rs_up", Control::Stick(Stick::RUp), "Camera up", "Camera"),
    ci("rs_down", Control::Stick(Stick::RDown), "Camera down", "Camera"),
    ci("rs_left", Control::Stick(Stick::RLeft), "Camera left", "Camera"),
    ci("rs_right", Control::Stick(Stick::RRight), "Camera right", "Camera"),
    ci("pause_back", Control::PauseBack, "Pause / back (Options in play, Circle in menus)", "Menus"),
    ci("options", Control::Button(BTN_OPTIONS), "Options (pause menu)", "Menus"),
    ci("touchpad", Control::Button(BTN_TOUCHPAD), "Touchpad", "Menus"),
];

/// The commented file written on first use (also the defaults).
pub const DEFAULTS: &str = "\
# The Last Guardian (tlg): keyboard and mouse settings. The launcher's
# Controls page edits this file; changes apply the next time the game starts.
# Delete the file to get the defaults back.
#
# Bindings: <control> = <input>[, <input>...], or none. Controls are the
# DualShock 4's: cross circle square triangle up down left right (d-pad) l1
# r1 l2 r2 l3 r3 options touchpad, stick directions ls_up ls_down ls_left
# ls_right rs_up rs_down rs_left rs_right, walk (left stick at half
# deflection while held) and pause_back (Options while playing, Circle - back -
# in menus).
# Inputs: A-Z, 0-9, F1-F24, Space, Enter, Escape, Tab, Backspace, Shift,
# LShift, RShift, Ctrl, LCtrl, RCtrl, Alt, LAlt, RAlt, Up, Down, Left, Right,
# Insert, Delete, Home, End, PageUp, PageDown, Numpad0-Numpad9, Minus, Equals,
# LBracket, RBracket, Semicolon, Quote, Comma, Period, Slash, Backslash,
# Grave, CapsLock, vkXX (any virtual-key code in hex); mouse: MouseLeft,
# MouseRight, MouseMiddle, Mouse4, Mouse5, WheelUp, WheelDown (mouse inputs
# count while the mouse is captured).

ls_up = W
ls_down = S
ls_left = A
ls_right = D
walk = LShift
triangle = Space
circle = E, MouseLeft
cross = C, Enter
square = F, MouseRight
r1 = Q, Mouse5
l1 = R, Mouse4
r2 = X
l2 = Z
r3 = V, MouseMiddle
l3 = B
up = Up
down = Down
left = Left
right = Right
rs_up = I
rs_down = K
rs_left = J
rs_right = L
pause_back = Escape
options = Tab
touchpad = T

# Mouse: a click into the window captures the mouse; release_mouse (or
# switching away from the window) frees it. Mouse look turns the game camera
# directly by the raw motion (no acceleration, smoothing or dead zone;
# sensitivity 1.0 is 0.04 degrees per count, separately for x and y); where
# the game camera is not in charge it moves the right stick instead.
mouse_look = on
mouse_sensitivity_x = 1.0
mouse_sensitivity_y = 1.0
mouse_invert_x = off
mouse_invert_y = off
release_mouse = F1
# In menus (the title screen and main menu, the pause menu, loading
# screens, most cutscenes) the left button selects (Cross), the right button
# goes back (Circle) and the wheel moves the selection (d-pad up/down),
# whatever the bindings above.
menu_mouse = on
";

/// Named keys: (name, virtual-key code). The first name of a code is the
/// one written back.
const NAMED_KEYS: &[(&str, u8)] = &[
    ("Space", 0x20),
    ("Enter", 0x0D),
    ("Return", 0x0D),
    ("Escape", 0x1B),
    ("Esc", 0x1B),
    ("Tab", 0x09),
    ("Backspace", 0x08),
    ("Shift", 0x10),
    ("LShift", 0xA0),
    ("RShift", 0xA1),
    ("Ctrl", 0x11),
    ("Control", 0x11),
    ("LCtrl", 0xA2),
    ("RCtrl", 0xA3),
    ("Alt", 0x12),
    ("LAlt", 0xA4),
    ("RAlt", 0xA5),
    ("Up", 0x26),
    ("Down", 0x28),
    ("Left", 0x25),
    ("Right", 0x27),
    ("Insert", 0x2D),
    ("Delete", 0x2E),
    ("Home", 0x24),
    ("End", 0x23),
    ("PageUp", 0x21),
    ("PageDown", 0x22),
    ("Minus", 0xBD),
    ("Equals", 0xBB),
    ("LBracket", 0xDB),
    ("RBracket", 0xDD),
    ("Semicolon", 0xBA),
    ("Quote", 0xDE),
    ("Comma", 0xBC),
    ("Period", 0xBE),
    ("Slash", 0xBF),
    ("Backslash", 0xDC),
    ("Grave", 0xC0),
    ("CapsLock", 0x14),
    ("NumpadMultiply", 0x6A),
    ("NumpadAdd", 0x6B),
    ("NumpadSubtract", 0x6D),
    ("NumpadDecimal", 0x6E),
    ("NumpadDivide", 0x6F),
    ("NumLock", 0x90),
    ("ScrollLock", 0x91),
    ("Pause", 0x13),
    ("PrintScreen", 0x2C),
    ("LWin", 0x5B),
    ("RWin", 0x5C),
    ("Menu", 0x5D),
    ("Oem102", 0xE2),
];

const MOUSE_NAMES: [&str; 5] = ["MouseLeft", "MouseRight", "MouseMiddle", "Mouse4", "Mouse5"];

/// The virtual-key code of a key name (case-insensitive).
pub fn key_code(name: &str) -> Option<u8> {
    let n = name.trim().to_ascii_lowercase();
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
    if let Some(h) = n.strip_prefix("vk").and_then(|h| u8::from_str_radix(h, 16).ok()).filter(|&v| v != 0) {
        return Some(h);
    }
    NAMED_KEYS.iter().find(|(k, _)| k.eq_ignore_ascii_case(&n)).map(|&(_, v)| v)
}

/// The input.txt name of a key.
pub fn key_name(vk: u8) -> String {
    match vk {
        b'A'..=b'Z' | b'0'..=b'9' => (vk as char).to_string(),
        0x70..=0x87 => format!("F{}", vk - 0x6F),
        0x60..=0x69 => format!("Numpad{}", vk - 0x60),
        _ => match NAMED_KEYS.iter().find(|(_, v)| *v == vk) {
            Some((k, _)) => k.to_string(),
            None => format!("vk{vk:02X}"),
        },
    }
}

pub fn parse_input(name: &str) -> Option<Input> {
    let n = name.trim();
    if let Some(i) = MOUSE_NAMES.iter().position(|m| m.eq_ignore_ascii_case(n)) {
        return Some(Input::Mouse(i as u8));
    }
    if n.eq_ignore_ascii_case("wheelup") {
        return Some(Input::WheelUp);
    }
    if n.eq_ignore_ascii_case("wheeldown") {
        return Some(Input::WheelDown);
    }
    key_code(n).map(Input::Key)
}

/// The input.txt name of an input.
pub fn input_name(i: Input) -> String {
    match i {
        Input::Key(k) => key_name(k),
        Input::Mouse(b) => MOUSE_NAMES.get(b as usize).copied().unwrap_or("MouseLeft").to_string(),
        Input::WheelUp => "WheelUp".into(),
        Input::WheelDown => "WheelDown".into(),
    }
}

/// How the launcher shows an input.
pub fn input_label(i: Input) -> String {
    let n = input_name(i);
    let fixed = [
        ("LShift", "Left Shift"),
        ("RShift", "Right Shift"),
        ("LCtrl", "Left Ctrl"),
        ("RCtrl", "Right Ctrl"),
        ("LAlt", "Left Alt"),
        ("RAlt", "Right Alt"),
        ("MouseLeft", "Left mouse button"),
        ("MouseRight", "Right mouse button"),
        ("MouseMiddle", "Middle mouse button"),
        ("Mouse4", "Mouse button 4"),
        ("Mouse5", "Mouse button 5"),
        ("WheelUp", "Wheel up"),
        ("WheelDown", "Wheel down"),
        ("PageUp", "Page Up"),
        ("PageDown", "Page Down"),
        ("CapsLock", "Caps Lock"),
        ("Up", "Up arrow"),
        ("Down", "Down arrow"),
        ("Left", "Left arrow"),
        ("Right", "Right arrow"),
        ("Grave", "` (grave)"),
    ];
    fixed.iter().find(|(k, _)| *k == n).map(|(_, v)| v.to_string()).unwrap_or(n)
}

pub fn control_info(c: Control) -> Option<&'static ControlInfo> {
    CONTROLS.iter().find(|i| i.control == c)
}

#[derive(Clone, Debug, PartialEq)]
pub struct Bindings {
    /// Every control of CONTROLS, in that order, with its inputs.
    pub map: Vec<(Control, Vec<Input>)>,
    pub mouse_look: bool,
    pub sensitivity_x: f32,
    pub sensitivity_y: f32,
    pub invert_x: bool,
    pub invert_y: bool,
    pub release_key: u8,
    pub menu_mouse: bool,
}

fn flag(v: &str) -> Option<bool> {
    match v.to_ascii_lowercase().as_str() {
        "on" | "true" | "yes" | "1" => Some(true),
        "off" | "false" | "no" | "0" => Some(false),
        _ => None,
    }
}

fn on_off(b: bool) -> &'static str {
    if b { "on" } else { "off" }
}

impl Default for Bindings {
    fn default() -> Self {
        Bindings::parse(DEFAULTS, None).0
    }
}

impl Bindings {
    /// The bindings of a file's text over `base` (the defaults if None): a
    /// binding line replaces that control's inputs. Returns a message per
    /// line that could not be used.
    pub fn parse(text: &str, base: Option<&Bindings>) -> (Bindings, Vec<String>) {
        let mut b = match base {
            Some(b) => b.clone(),
            None => Bindings {
                map: CONTROLS.iter().map(|c| (c.control, Vec::new())).collect(),
                mouse_look: true,
                sensitivity_x: 1.0,
                sensitivity_y: 1.0,
                invert_x: false,
                invert_y: false,
                release_key: 0x70,
                menu_mouse: true,
            },
        };
        let mut errs = Vec::new();
        for raw in text.lines() {
            let Some((k, v)) = split_line(raw) else { continue };
            let v = strip_comment(v);
            let line = raw.trim();
            let sens = |v: &str| v.parse::<f32>().ok().filter(|x| x.is_finite() && *x > 0.0 && *x <= 100.0);
            match k.as_str() {
                "mouse_look" | "mouse_invert_x" | "mouse_invert_y" | "menu_mouse" => match flag(v) {
                    Some(f) => match k.as_str() {
                        "mouse_look" => b.mouse_look = f,
                        "mouse_invert_x" => b.invert_x = f,
                        "mouse_invert_y" => b.invert_y = f,
                        _ => b.menu_mouse = f,
                    },
                    None => errs.push(format!("input.txt: '{line}': want on or off")),
                },
                // (The single setting of earlier versions sets both axes.)
                "mouse_sensitivity" | "mouse_sensitivity_x" | "mouse_sensitivity_y" => match sens(v) {
                    Some(x) => {
                        if k != "mouse_sensitivity_y" {
                            b.sensitivity_x = x;
                        }
                        if k != "mouse_sensitivity_x" {
                            b.sensitivity_y = x;
                        }
                    }
                    None => errs.push(format!("input.txt: '{line}': want a number above 0 (up to 100)")),
                },
                "release_mouse" => match key_code(v) {
                    Some(c) => b.release_key = c,
                    None => errs.push(format!("input.txt: '{line}': unknown key '{v}'")),
                },
                _ => {
                    let Some(info) = CONTROLS.iter().find(|c| c.name == k) else {
                        errs.push(format!("input.txt: '{line}': unknown control '{k}'"));
                        continue;
                    };
                    let mut inputs = Vec::new();
                    let mut ok = true;
                    if !v.eq_ignore_ascii_case("none") {
                        for name in v.split(',').map(str::trim).filter(|n| !n.is_empty()) {
                            match parse_input(name) {
                                Some(i) if !inputs.contains(&i) => inputs.push(i),
                                Some(_) => {}
                                None => {
                                    errs.push(format!("input.txt: '{line}': unknown input '{name}'"));
                                    ok = false;
                                }
                            }
                        }
                    }
                    if ok {
                        if let Some(e) = b.map.iter_mut().find(|(c, _)| *c == info.control) {
                            e.1 = inputs;
                        }
                    }
                }
            }
        }
        (b, errs)
    }

    pub fn load(path: &Path) -> (Bindings, Vec<String>) {
        let defaults = Bindings::default();
        match std::fs::read_to_string(path) {
            Ok(t) => Bindings::parse(&t, Some(&defaults)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (defaults, Vec::new()),
            Err(e) => (defaults, vec![format!("{}: {e}", path.display())]),
        }
    }

    pub fn inputs(&self, c: Control) -> &[Input] {
        self.map.iter().find(|(x, _)| *x == c).map(|(_, v)| v.as_slice()).unwrap_or(&[])
    }

    /// Sets input `slot` (0 or 1) of a control; None clears it. Slots past
    /// the end append.
    pub fn set_slot(&mut self, c: Control, slot: usize, input: Option<Input>) {
        let Some(e) = self.map.iter_mut().find(|(x, _)| *x == c) else { return };
        match input {
            Some(i) => {
                e.1.retain(|x| *x != i);
                if slot < e.1.len() {
                    e.1[slot] = i;
                } else {
                    e.1.push(i);
                }
            }
            None => {
                if slot < e.1.len() {
                    e.1.remove(slot);
                }
            }
        }
    }

    /// Removes an input from every control except `keep`.
    pub fn unbind_elsewhere(&mut self, i: Input, keep: Control) {
        for (c, v) in self.map.iter_mut() {
            if *c != keep {
                v.retain(|x| *x != i);
            }
        }
    }

    /// The controls an input is bound to.
    pub fn bound_to(&self, i: Input) -> Vec<Control> {
        self.map.iter().filter(|(_, v)| v.contains(&i)).map(|(c, _)| *c).collect()
    }

    /// Inputs bound to more than one control.
    pub fn conflicts(&self) -> Vec<(Input, Vec<Control>)> {
        let mut out: Vec<(Input, Vec<Control>)> = Vec::new();
        for (_, v) in &self.map {
            for &i in v {
                if out.iter().any(|(x, _)| *x == i) {
                    continue;
                }
                let cs = self.bound_to(i);
                if cs.len() > 1 {
                    out.push((i, cs));
                }
            }
        }
        out
    }

    /// Writes every binding and mouse setting into `f` (other lines kept).
    pub fn store(&self, f: &mut KvFile) {
        for info in CONTROLS {
            let v = self.inputs(info.control);
            let text = if v.is_empty() { "none".to_string() } else { v.iter().map(|&i| input_name(i)).collect::<Vec<_>>().join(", ") };
            f.set(info.name, &text);
        }
        f.set("mouse_look", on_off(self.mouse_look));
        f.remove("mouse_sensitivity");
        f.set("mouse_sensitivity_x", &format!("{}", self.sensitivity_x));
        f.set("mouse_sensitivity_y", &format!("{}", self.sensitivity_y));
        f.set("mouse_invert_x", on_off(self.invert_x));
        f.set("mouse_invert_y", on_off(self.invert_y));
        f.set("release_mouse", &key_name(self.release_key));
        f.set("menu_mouse", on_off(self.menu_mouse));
    }

    /// Saves into `path`, keeping the file's comments (the commented
    /// defaults for a new file).
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut f = KvFile::load(path)?;
        if f.entries().next().is_none() {
            f = KvFile::parse(DEFAULTS);
        }
        self.store(&mut f);
        f.save(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_parse_cleanly() {
        let (b, errs) = Bindings::parse(DEFAULTS, None);
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(b.map.len(), CONTROLS.len());
        assert!(b.map.iter().all(|(_, v)| !v.is_empty() && v.len() <= 2), "every control has one or two default inputs");
        assert_eq!(b.inputs(Control::Button(BTN_TRIANGLE)), &[Input::Key(0x20)]);
        assert_eq!(b.inputs(Control::Button(BTN_CIRCLE)), &[Input::Key(b'E'), Input::Mouse(0)]);
        assert!(b.mouse_look && b.menu_mouse);
        assert_eq!(b.release_key, 0x70);
        assert!(b.conflicts().is_empty(), "{:?}", b.conflicts());
    }

    #[test]
    fn user_lines_override_defaults() {
        let d = Bindings::default();
        let (b, errs) = Bindings::parse("triangle = Space, MouseRight\ncircle = none\nmouse_sensitivity = 2.5 # faster\nmouse_sensitivity_y = 0.5\nmouse_invert_y = on\njump = Space\ncross = Hyper\n", Some(&d));
        assert_eq!(errs.len(), 2, "{errs:?}");
        assert_eq!(b.inputs(Control::Button(BTN_TRIANGLE)), &[Input::Key(0x20), Input::Mouse(1)]);
        assert!(b.inputs(Control::Button(BTN_CIRCLE)).is_empty());
        // (A bad line keeps the control's previous inputs.)
        assert_eq!(b.inputs(Control::Button(BTN_CROSS)), d.inputs(Control::Button(BTN_CROSS)));
        assert_eq!((b.sensitivity_x, b.sensitivity_y), (2.5, 0.5));
        assert!(b.invert_y && !b.invert_x);
        let c = b.conflicts();
        assert_eq!(c, vec![(Input::Mouse(1), vec![Control::Button(BTN_TRIANGLE), Control::Button(BTN_SQUARE)])]);
    }

    #[test]
    fn slots_and_store_round_trip() {
        let mut b = Bindings::default();
        let jump = Control::Button(BTN_TRIANGLE);
        b.set_slot(jump, 1, Some(Input::WheelUp));
        assert_eq!(b.inputs(jump), &[Input::Key(0x20), Input::WheelUp]);
        b.set_slot(jump, 0, Some(Input::Key(0x5B)));
        assert_eq!(b.inputs(jump), &[Input::Key(0x5B), Input::WheelUp]);
        b.set_slot(Control::Button(BTN_SQUARE), 0, Some(Input::Key(0x5B)));
        assert_eq!(b.bound_to(Input::Key(0x5B)).len(), 2);
        b.unbind_elsewhere(Input::Key(0x5B), jump);
        assert_eq!(b.bound_to(Input::Key(0x5B)), vec![jump]);
        b.set_slot(jump, 0, None);
        assert_eq!(b.inputs(jump), &[Input::WheelUp]);
        b.sensitivity_x = 1.75;
        b.release_key = 0x7A;
        let mut f = KvFile::parse("# mine\nmouse_sensitivity = 3\ncircle = E\n");
        b.store(&mut f);
        assert!(f.text().starts_with("# mine\r\n"));
        assert_eq!(f.get("mouse_sensitivity"), None);
        let (back, errs) = Bindings::parse(&f.text(), None);
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(back, b);
    }

    #[test]
    fn key_names_round_trip() {
        assert_eq!(key_code("a"), Some(b'A'));
        assert_eq!(key_code("F12"), Some(0x7B));
        assert_eq!(key_code("numpad5"), Some(0x65));
        assert_eq!(key_code("LCtrl"), Some(0xA2));
        assert_eq!(key_code("esc"), Some(0x1B));
        assert_eq!(key_code("vk5b"), Some(0x5B));
        assert_eq!(key_code("nope"), None);
        for vk in 1..=255u8 {
            assert_eq!(key_code(&key_name(vk)), Some(vk), "vk {vk:#x} -> {}", key_name(vk));
        }
        for i in [Input::Mouse(0), Input::Mouse(4), Input::WheelUp, Input::WheelDown, Input::Key(0x20)] {
            assert_eq!(parse_input(&input_name(i)), Some(i));
        }
        assert_eq!(input_label(Input::Key(0xA0)), "Left Shift");
    }
}
