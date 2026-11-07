//! Button prompts for the device in use. The game's prompts (its guides:
//! a frame, a line of text and a picture of a DualShock 4 with the button
//! to press glowing) show the device the player uses - drawn here in the
//! game's line-art style (black shapes, thin white outlines, white
//! labels). PC Settings `button_prompts`: automatic follows the device
//! last used, or one style always.
//!
//! - Xbox pad: the pad's picture, the glow on its button.
//! - Keyboard and mouse: the prompt's own keys, large and named as bound
//!   in input.txt, without a glow (it would cover the key's name); the
//!   camera's prompt shows the mouse, glowing.
//!
//! The pictures: the game draws every guide's picture from one texture
//! (game/textures/ui/alert/guide_controller_base_bc7, 256x128 BC7, 1D thin
//! tiling). Each guide gets one of the port's picture slots instead - its
//! copies under other names in the port's archive (their first block a
//! transparent marker: `slot_files`) - by the button set it prompts (the
//! port's script hands it to Guide_CreateGuide: `guide_tex`). When the
//! texture cache meets a slot (`note_texture`, by its marker) or the
//! shared picture (by its file's texels), the style's texels are written
//! over it: the game's own BC7 blocks for the PlayStation style, the
//! others drawn and encoded here (BC7 modes 5 and 6).
//!
//! The glow: each guide keeps a copy of its buttons' places (offset and
//! effect shape, made from the game's table when the guides are made); the
//! port records those copies as the engine makes them (a detour in
//! Guide_SetButtonParam's code), each with the slot of the guide being
//! made, and writes the style's places over them (the game's own back for
//! the PlayStation style).

use parking_lot::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, Ordering};
use tlg_config::bindings::{BTN_CIRCLE, BTN_CROSS, BTN_DOWN, BTN_L1, BTN_L2, BTN_L3, BTN_LEFT, BTN_OPTIONS, BTN_R1, BTN_R2, BTN_R3, BTN_RIGHT, BTN_SQUARE, BTN_TOUCHPAD, BTN_TRIANGLE, BTN_UP, Control, Input, Stick};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Style {
    PlayStation = 0,
    Xbox = 1,
    Keyboard = 2,
}

/// The device last used (`Style` as u8).
static LAST: AtomicU8 = AtomicU8::new(Style::PlayStation as u8);
/// The pictures are to be written (a style change, a picture seen, a
/// keyboard picture ready).
static WRITE_DUE: AtomicBool = AtomicBool::new(false);
/// The glow places are to be written (new entries or another style).
static GLOW_DUE: AtomicBool = AtomicBool::new(false);
/// The style the pictures and places were last written for.
static SHOWN: AtomicU8 = AtomicU8::new(u8::MAX);
/// Bumped when the key bindings change (the keyboard pictures follow).
static BINDINGS_GEN: AtomicU32 = AtomicU32::new(1);

fn from_u8(v: u8) -> Style {
    match v {
        1 => Style::Xbox,
        2 => Style::Keyboard,
        _ => Style::PlayStation,
    }
}

/// Input came from this kind of device.
pub fn used(s: Style) {
    if LAST.swap(s as u8, Ordering::Relaxed) != s as u8 && crate::pc_settings::get("button_prompts") == "auto" {
        crate::debug!("prompts: device in use {s:?}");
        refresh();
    }
}

/// The prompt style in force.
pub fn style() -> Style {
    match crate::pc_settings::get("button_prompts").as_str() {
        "playstation" => Style::PlayStation,
        "xbox" => Style::Xbox,
        "keyboard" => Style::Keyboard,
        _ => from_u8(LAST.load(Ordering::Relaxed)),
    }
}

/// The `button_prompts` setting changed.
pub fn changed() {
    crate::info!("prompts: {:?}", style());
    refresh();
}

/// The key bindings changed: the keyboard pictures are drawn again.
pub fn bindings_changed() {
    BINDINGS_GEN.fetch_add(1, Ordering::Relaxed);
    refresh_all();
}

/// The style in force may have changed: the pictures and the glow places
/// follow (at the game's next pad read: `tick`).
fn refresh() {
    if SHOWN.load(Ordering::Relaxed) != style() as u8 {
        refresh_all();
    }
}

fn refresh_all() {
    WRITE_DUE.store(true, Ordering::Relaxed);
    GLOW_DUE.store(true, Ordering::Relaxed);
}

/// At the game's pad reads (between its frames): writes what is due.
pub fn tick() {
    let s = style();
    if WRITE_DUE.swap(false, Ordering::Relaxed) {
        write_pictures(s);
        SHOWN.store(s as u8, Ordering::Relaxed);
    }
    if GLOW_DUE.swap(false, Ordering::Relaxed) {
        place_glows(s);
    }
}

// ---------------------------------------------------------------------------
// The picture slots.

/// Picture slots (the game's prompts use 24 different button sets).
pub const SLOTS: usize = 32;

/// A slot's texture name (for the game's Load; its file: `<name>.gnf`).
pub fn slot_name(i: usize) -> String {
    format!("game/textures/ui/alert/tlgpc_guide_{i:02}")
}

/// A slot texture's first block: transparent (alpha 0 everywhere), its
/// colour endpoints naming the slot (BC7 mode 6).
fn marker(i: usize) -> [u8; 16] {
    let mut bits = 0u128;
    let mut pos = 0u32;
    let mut put = |v: u128, n: u32| {
        bits |= (v & ((1u128 << n) - 1)) << pos;
        pos += n;
    };
    put(1 << 6, 7);
    // (R0 the slot, G0 and B0 a signature; alpha endpoints 0.)
    for (c0, c1) in [(i as u128, 0), (0x2a, 0x15), (0x55, 0x6b), (0, 0)] {
        put(c0, 7);
        put(c1, 7);
    }
    put(0, 1);
    put(0, 1);
    bits.to_le_bytes()
}

/// The archive's slot textures: the game's picture file under each slot's
/// name, its first block the slot's marker. None without the game's file.
pub fn slot_files() -> Option<Vec<(String, Vec<u8>)>> {
    let f = crate::pc_files::game_file(PAD_FILE, "")?;
    if f.len() < 256 + PAD_BYTES {
        return None;
    }
    Some(
        (0..SLOTS)
            .map(|i| {
                let mut d = f.clone();
                d[256..272].copy_from_slice(&marker(i));
                (format!("{}.gnf", slot_name(i)), d)
            })
            .collect(),
    )
}

struct Slot {
    /// The button set (the game's button keys, in the guide's order).
    set: Vec<String>,
    /// The game's copy of its texels (0: not seen yet).
    addr: u64,
    /// The keyboard picture (BC7, tiled) and the bindings it was drawn for.
    keyboard: Option<(u32, Vec<u8>)>,
}

struct Slots {
    slots: Vec<Slot>,
    /// The slot of the guide being made (its entries follow).
    current: Option<usize>,
    /// The shared picture's texels in the game's memory (0: not seen).
    shared: u64,
}

static SLOT_STATE: Mutex<Slots> = Mutex::new(Slots { slots: Vec::new(), current: None, shared: 0 });

/// A guide is being made with these buttons ("Triangle+StickBaseL", from
/// the game's guide data; empty: unknown): its picture slot's texture
/// name, or empty (no slot: the shared picture).
pub fn guide_tex(buttons: &str) -> String {
    let set: Vec<String> = buttons.split('+').filter(|s| !s.is_empty()).map(str::to_string).collect();
    let mut st = SLOT_STATE.lock();
    if set.is_empty() {
        st.current = None;
        return String::new();
    }
    let key = |s: &[String]| {
        let mut k = s.to_vec();
        k.sort();
        k.dedup();
        k
    };
    let want = key(&set);
    let i = match st.slots.iter().position(|s| key(&s.set) == want) {
        Some(i) => i,
        None if st.slots.len() < SLOTS => {
            st.slots.push(Slot { set, addr: 0, keyboard: None });
            st.slots.len() - 1
        }
        None => {
            crate::warn!("prompts: no picture slot left for {buttons}: the shared picture");
            st.current = None;
            return String::new();
        }
    };
    st.current = Some(i);
    slot_name(i)
}

// ---------------------------------------------------------------------------
// The guides' copies of the glow places.

/// The engine's code that makes a guide's button entry (Guide_SetButtonParam:
/// eboot 0xbc0b50), at `mov edx, [r12+rbx*8+18h]` with the new entry in r14
/// (0x40 bytes: +0xC the button's symbol, +0x10/+0x14 its offset, +0x18 the
/// effect shape's uv, +0x30 a "Move1" animation - +0x1C/+0x20 from, +0x24/
/// +0x28 to - and +0x38 the button's name).
const ENTRY_SITE: usize = 0xbc0e36;
const ENTRY_BYTES: [u8; 5] = [0x41, 0x8B, 0x54, 0xDC, 0x18];

struct Entry {
    addr: u64,
    /// The picture slot of the guide it belongs to.
    slot: Option<usize>,
    /// The game's own values (offset, uv, the animation's from and to),
    /// read before the first change.
    own: Option<([f32; 2], [f32; 4], Option<(u64, [f32; 4])>)>,
}

static ENTRIES: Mutex<Vec<Entry>> = Mutex::new(Vec::new());

pub fn apply(p: &mut crate::patch::Patcher, name: &str) {
    use iced_x86::code_asm::*;
    if p.image.get(ENTRY_SITE..ENTRY_SITE + ENTRY_BYTES.len()) != Some(&ENTRY_BYTES[..]) {
        crate::warn!("prompts: {name}: unexpected code at +{ENTRY_SITE:#x}; the glow stays at the DualShock's buttons");
        return;
    }
    let base = p.base;
    let f = tlg_guide_entry as *const () as usize as u64;
    // (rsp is 16-aligned in that function's body: the flags and 9 registers
    // keep it so; the flags are live there.)
    let r = p.detour(ENTRY_SITE, &ENTRY_BYTES, |a, ins| {
        a.pushfq()?;
        for r in [rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11] {
            a.push(r)?;
        }
        a.mov(rdi, r14)?;
        a.mov(rax, f)?;
        a.call(rax)?;
        for r in [r11, r10, r9, r8, rdi, rsi, rdx, rcx, rax] {
            a.pop(r)?;
        }
        a.popfq()?;
        a.add_instruction(*ins)?;
        a.jmp(base + (ENTRY_SITE + ENTRY_BYTES.len()) as u64)
    });
    match r {
        Ok(_) => crate::info!("prompts: the guides' button entries recorded at {name}+{ENTRY_SITE:#x}"),
        Err(e) => crate::warn!("prompts: {name}: {e}"),
    }
}

fn guide_entry(entry: u64) {
    let slot = SLOT_STATE.lock().current;
    ENTRIES.lock().push(Entry { addr: entry, slot, own: None });
    GLOW_DUE.store(true, Ordering::Relaxed);
}

crate::sysv! {
    fn tlg_guide_entry(entry: u64) -> i32 {
        guide_entry(entry);
        0
    }
}

fn rd(a: u64) -> f32 {
    // SAFETY: callers check the entry (and animation) is readable.
    unsafe { std::ptr::read_volatile(a as *const f32) }
}

fn wr(a: u64, v: f32) {
    // SAFETY: as rd; the engine's own entry, written as it writes it.
    unsafe { std::ptr::write_volatile(a as *mut f32, v) }
}

fn entry_name(e: u64) -> Option<String> {
    // SAFETY: the entry's name pointer, checked readable before reading.
    let p = unsafe { std::ptr::read_volatile((e + 0x38) as *const u64) };
    if p == 0 || !crate::debug_mem::host_or_guest_readable(p, 32) {
        return None;
    }
    let mut s = String::new();
    for i in 0..32 {
        let c = unsafe { std::ptr::read_volatile((p + i) as *const u8) };
        if c == 0 {
            break;
        }
        s.push(c as char);
    }
    Some(s)
}

/// The engine draws a glow's middle this far right and down of its place
/// in the picture's texels (measured in the game with both shapes).
const GLOW_SHIFT: (f32, f32) = (16.0, 16.5);

const STICK_UV: [f32; 4] = [0.0, 0.625, 0.1875, 1.0];
/// No glow: an empty shape.
const NO_UV: [f32; 4] = [0.0, 0.0, 0.0, 0.0];

/// Where an entry's glow goes in a style: its offset and shape, or None for
/// the game's own.
fn glow(s: Style, name: &str, slot: Option<&Slot>) -> Option<([f32; 2], [f32; 4])> {
    let at = |x: f32, y: f32| [x - GLOW_SHIFT.0, y - GLOW_SHIFT.1];
    match s {
        Style::PlayStation => None,
        Style::Xbox => XBOX.iter().find(|p| p.0 == name).map(|&(k, x, y)| (at(x, y), uv_of(k))),
        Style::Keyboard => {
            // The camera's own prompt: the mouse glows; keys never do.
            let mouse = slot.and_then(|s| keyboard_scene(&s.set).mouse_glow);
            match mouse {
                Some((x, y)) if name == "StickBaseR" => Some((at(x, y), STICK_UV)),
                _ => Some((at(0.0, 0.0), NO_UV)),
            }
        }
    }
}

/// Writes the style's places into the recorded entries.
fn place_glows(s: Style) {
    let slots = SLOT_STATE.lock();
    let mut v = ENTRIES.lock();
    let mut n = 0;
    for e in v.iter_mut() {
        if !crate::debug_mem::host_or_guest_readable(e.addr, 0x40) {
            continue;
        }
        let Some(name) = entry_name(e.addr) else { continue };
        if e.own.is_none() {
            // SAFETY: the entry, checked readable.
            let anim = unsafe { std::ptr::read_volatile((e.addr + 0x30) as *const u64) };
            let anim = (anim != 0 && crate::debug_mem::host_or_guest_readable(anim, 0x2c)).then(|| (anim, [rd(anim + 0x1c), rd(anim + 0x20), rd(anim + 0x24), rd(anim + 0x28)]));
            e.own = Some(([rd(e.addr + 0x10), rd(e.addr + 0x14)], [rd(e.addr + 0x18), rd(e.addr + 0x1c), rd(e.addr + 0x20), rd(e.addr + 0x24)], anim));
        }
        let (own_off, own_uv, anim) = e.own.unwrap();
        // (Guides without a slot show the shared picture: there only the
        // PlayStation and Xbox places fit; the keyboard style hides them.)
        let (off, uv) = glow(s, &name, e.slot.and_then(|i| slots.slots.get(i))).unwrap_or((own_off, own_uv));
        wr(e.addr + 0x10, off[0]);
        wr(e.addr + 0x14, off[1]);
        for (i, u) in uv.iter().enumerate() {
            wr(e.addr + 0x18 + 4 * i as u64, *u);
        }
        if let Some((a, [fx, fy, tx, ty])) = anim {
            // (The animation moves from the place by its own distance.)
            let (dx, dy) = (off[0] - own_off[0], off[1] - own_off[1]);
            wr(a + 0x1c, fx + dx);
            wr(a + 0x20, fy + dy);
            wr(a + 0x24, tx + dx);
            wr(a + 0x28, ty + dy);
        }
        n += 1;
    }
    if n > 0 {
        crate::debug!("prompts: {n} guide buttons placed for the {s:?} pictures");
    }
}

/// The game's effect shapes per button (as its table: shoulder buttons and
/// sticks have their own outlines).
fn uv_of(k: &str) -> [f32; 4] {
    match k {
        "R1" => [0.825, 0.75, 0.95, 1.0],
        "R2" => [0.8125, 0.5, 0.9375, 0.75],
        "L1" => [0.65, 0.75, 0.775, 1.0],
        "L2" => [0.625, 0.5, 0.75, 0.75],
        "Left" => [0.75, 0.25, 0.875, 0.5],
        "Right" => [0.875, 0.25, 1.0, 0.5],
        "Up" => [0.875, 0.0, 1.0, 0.25],
        "Down" => [0.75, 0.0, 0.875, 0.25],
        "Square" => [0.375, 0.0, 0.5, 0.25],
        "Cross" => [0.25, 0.0, 0.375, 0.25],
        "Circle" => [0.125, 0.0, 0.25, 0.25],
        "Triangle" => [0.0, 0.0, 0.125, 0.25],
        "PS" => [0.625, 0.25, 0.75, 0.5],
        "Select" => [0.625, 0.0, 0.75, 0.25],
        "Start" => [0.5, 0.0, 0.625, 0.25],
        _ => STICK_UV,
    }
}

/// The Xbox picture's buttons (texels of the 256x128 picture).
const XBOX: &[(&str, f32, f32)] = &[
    ("Triangle", 184.0, 33.0),
    ("Circle", 197.0, 46.0),
    ("Cross", 184.0, 59.0),
    ("Square", 171.0, 46.0),
    ("L1", 72.0, 16.0),
    ("R1", 184.0, 16.0),
    ("L2", 74.0, 7.0),
    ("R2", 182.0, 7.0),
    ("StickBaseL", 74.0, 46.0),
    ("StickBaseR", 156.0, 76.0),
    ("Push", 112.0, 41.0),
    ("Up", 100.0, 67.0),
    ("Down", 100.0, 85.0),
    ("Left", 91.0, 76.0),
    ("Right", 109.0, 76.0),
    ("Select", 112.0, 41.0),
    ("Start", 144.0, 41.0),
    ("PS", 128.0, 25.0),
];

// ---------------------------------------------------------------------------
// The keyboard pictures: a prompt's keys (and mouse) as bound.

fn bound_all(c: Control) -> Vec<Input> {
    let b = crate::input::bindings();
    b.map.iter().find(|(k, _)| *k == c).map(|(_, v)| v.clone()).unwrap_or_default()
}

/// The input a prompt names for a control: its first key, else its first
/// mouse button.
fn bound(c: Control) -> Option<Input> {
    let ins = bound_all(c);
    ins.iter().copied().find(|i| matches!(i, Input::Key(_))).or_else(|| ins.first().copied())
}

fn label(i: Option<Input>) -> String {
    let Some(i) = i else { return "-".into() };
    let n = tlg_config::bindings::input_name(i);
    let short = [
        ("LShift", "Shift"),
        ("RShift", "Shift"),
        ("LCtrl", "Ctrl"),
        ("RCtrl", "Ctrl"),
        ("LAlt", "Alt"),
        ("RAlt", "Alt"),
        ("Escape", "Esc"),
        ("Return", "Enter"),
        ("Up", "\u{2191}"),
        ("Down", "\u{2193}"),
        ("Left", "\u{2190}"),
        ("Right", "\u{2192}"),
        ("PageUp", "PgUp"),
        ("PageDown", "PgDn"),
        ("CapsLock", "Caps"),
        ("Backspace", "Bksp"),
        ("WheelUp", "Wheel"),
        ("WheelDown", "Wheel"),
    ];
    short.iter().find(|(k, _)| *k == n).map(|(_, v)| v.to_string()).unwrap_or(n)
}

/// One thing a keyboard picture shows.
#[derive(Clone, PartialEq)]
enum Elem {
    /// A key with its name.
    Key(String),
    /// Four keys as arrows: up, left, down, right.
    Cluster([String; 4]),
    /// The mouse moving (the camera).
    MouseMove,
    /// A mouse button (0 left, 1 right, 2 middle; else the side buttons).
    MouseButton(u8),
}

fn elem_for(c: Control) -> Elem {
    match bound(c) {
        Some(Input::Mouse(b)) => Elem::MouseButton(b),
        i => Elem::Key(label(i)),
    }
}

fn cluster(controls: [Control; 4]) -> Elem {
    Elem::Cluster(controls.map(|c| label(bound(c))))
}

/// The things a button set's picture shows, in the guide's order.
fn elems(set: &[String]) -> Vec<Elem> {
    let dpad = ["Up", "Down", "Left", "Right"].iter().all(|d| set.iter().any(|s| s == d));
    let mut out: Vec<Elem> = Vec::new();
    for b in set {
        let e = match b.as_str() {
            "StickBaseL" => cluster([Control::Stick(Stick::LUp), Control::Stick(Stick::LLeft), Control::Stick(Stick::LDown), Control::Stick(Stick::LRight)]),
            "StickBaseR" if crate::input::bindings().mouse_look => Elem::MouseMove,
            "StickBaseR" => cluster([Control::Stick(Stick::RUp), Control::Stick(Stick::RLeft), Control::Stick(Stick::RDown), Control::Stick(Stick::RRight)]),
            "Up" | "Down" | "Left" | "Right" if dpad => cluster([Control::Button(BTN_UP), Control::Button(BTN_LEFT), Control::Button(BTN_DOWN), Control::Button(BTN_RIGHT)]),
            "Up" => elem_for(Control::Button(BTN_UP)),
            "Down" => elem_for(Control::Button(BTN_DOWN)),
            "Left" => elem_for(Control::Button(BTN_LEFT)),
            "Right" => elem_for(Control::Button(BTN_RIGHT)),
            "Triangle" => elem_for(Control::Button(BTN_TRIANGLE)),
            "Circle" => elem_for(Control::Button(BTN_CIRCLE)),
            "Cross" => elem_for(Control::Button(BTN_CROSS)),
            "Square" => elem_for(Control::Button(BTN_SQUARE)),
            "L1" => elem_for(Control::Button(BTN_L1)),
            "R1" => elem_for(Control::Button(BTN_R1)),
            "L2" => elem_for(Control::Button(BTN_L2)),
            "R2" => elem_for(Control::Button(BTN_R2)),
            "Push" => elem_for(Control::Button(BTN_L3)),
            "PushR" => elem_for(Control::Button(BTN_R3)),
            "Start" => elem_for(Control::Button(BTN_OPTIONS)),
            "Select" => elem_for(Control::Button(BTN_TOUCHPAD)),
            _ => continue,
        };
        if !out.contains(&e) {
            out.push(e);
        }
    }
    out
}

/// A keyboard picture's layout: each thing's place and size (texels), the
/// scale, and where the mouse glows (the camera's prompt alone).
struct Scene {
    items: Vec<(Elem, f32, f32, f32)>,
    mouse_glow: Option<(f32, f32)>,
}

/// A thing's size at scale 1.
fn elem_size(e: &Elem) -> (f32, f32) {
    match e {
        Elem::Key(l) => {
            let n = l.chars().count() as f32;
            ((if n <= 1.0 { 52.0 } else { 24.0 + n * 14.0 }).max(52.0), 48.0)
        }
        Elem::Cluster(_) => (3.0 * 46.0 + 2.0 * 5.0, 2.0 * 44.0 + 5.0),
        Elem::MouseMove => (116.0, 116.0),
        Elem::MouseButton(_) => (56.0, 80.0),
    }
}

/// Places a set's things in rows within the picture (centred), scaled to fit.
fn keyboard_scene(set: &[String]) -> Scene {
    let es = elems(set);
    let gap = 12.0;
    let (aw, ah) = (240.0f32, 116.0f32);
    // One row, or two when that fits better.
    let row_w = |r: &[Elem]| r.iter().map(|e| elem_size(e).0).sum::<f32>() + gap * (r.len().saturating_sub(1)) as f32;
    let row_h = |r: &[Elem]| r.iter().map(|e| elem_size(e).1).fold(0.0f32, f32::max);
    let one = (aw / row_w(&es).max(1.0)).min(ah / row_h(&es).max(1.0)).min(1.0);
    let rows: Vec<Vec<Elem>> = if one < 0.6 && es.len() > 2 {
        let half = es.len().div_ceil(2);
        vec![es[..half].to_vec(), es[half..].to_vec()]
    } else {
        vec![es.clone()]
    };
    let total_h = rows.iter().map(|r| row_h(r)).sum::<f32>() + gap * (rows.len() - 1) as f32;
    let widest = rows.iter().map(|r| row_w(r)).fold(0.0f32, f32::max);
    let s = (aw / widest.max(1.0)).min(ah / total_h.max(1.0)).min(1.0);
    let mut items = Vec::new();
    let mut y = 64.0 - total_h * s / 2.0;
    for r in &rows {
        let h = row_h(r) * s;
        let mut x = 128.0 - row_w(r) * s / 2.0;
        for e in r {
            let (w, _) = elem_size(e);
            items.push((e.clone(), x + w * s / 2.0, y + h / 2.0, s));
            x += (w + gap) * s;
        }
        y += h + gap * s;
    }
    let mouse_glow = match &es[..] {
        [Elem::MouseMove] => items.first().map(|i| (i.1, i.2 + 5.0 * i.3)),
        _ => None,
    };
    Scene { items, mouse_glow }
}

fn draw_key(c: &mut Canvas, label: &str, x: f32, y: f32, w: f32, h: f32, s: f32) {
    c.part(rbox(x, y, w / 2.0, h / 2.0, 6.0 * s, 0.0));
    let n = label.chars().count();
    let px = ((if n <= 1 { 28.0 } else if n <= 3 { 21.0 } else { 17.0 }) * s).round().max(8.0) as u32;
    c.text(label, px, x, y);
}

/// A mouse: body, the buttons' split, the wheel; `filled` a button shown
/// pressed (white).
fn draw_mouse(c: &mut Canvas, x: f32, y: f32, s: f32, filled: Option<u8>) {
    let (hw, hh) = (19.0 * s, 29.0 * s);
    c.part(rbox(x, y, hw, hh, 17.0 * s, 0.0));
    let top = y - hh;
    let split = y - 6.0 * s;
    if let Some(b) = filled {
        // (The pressed button: its part of the body above the split.)
        let body = rbox(x, y, hw, hh, 17.0 * s, 0.0);
        let side = match b {
            0 => -1.0,
            1 => 1.0,
            _ => 0.0,
        };
        c.fill(
            move |px, py| {
                let inside = body(px, py);
                let above = py - split;
                let half = if side == 0.0 { (px - x).abs() - 3.0 * s } else { -(px - x) * side };
                inside.max(above).max(half)
            },
            LINE,
        );
    }
    c.stroke(rbox(x, (top + split) / 2.0, 0.1, (split - top) / 2.0, 0.0, 0.0), LINE_W, LINE);
    c.stroke(rbox(x, split, hw, 0.1, 0.0, 0.0), LINE_W, LINE);
    c.part(rbox(x, y - 16.0 * s, 2.6 * s, 5.0 * s, 2.5 * s, 0.0));
}

/// A small arrow head pointing along (dx, dy) at (x, y).
fn draw_arrow(c: &mut Canvas, x: f32, y: f32, dx: f32, dy: f32, s: f32) {
    let (l, w) = (8.0 * s, 6.0 * s);
    let (tx, ty) = (x + dx * l, y + dy * l);
    let (bx, by) = (x - dx * l * 0.2, y - dy * l * 0.2);
    let (nx, ny) = (-dy * w, dx * w);
    let tri = [(tx, ty), (bx + nx, by + ny), (bx - nx, by - ny)];
    c.fill(
        move |px, py| {
            // (Distance to the triangle by its edges.)
            let mut d = f32::MIN;
            for i in 0..3 {
                let (ax, ay) = tri[i];
                let (cx, cy) = tri[(i + 1) % 3];
                let (ex, ey) = (cx - ax, cy - ay);
                let len = (ex * ex + ey * ey).sqrt().max(1e-3);
                d = d.max(((px - ax) * ey - (py - ay) * ex) / len);
            }
            d
        },
        LINE,
    );
}

/// A button set's keyboard picture.
fn draw_keyboard_set(set: &[String]) -> Vec<u8> {
    let mut c = Canvas::new();
    for (e, x, y, s) in keyboard_scene(set).items {
        let (w, h) = elem_size(&e);
        let (w, h) = (w * s, h * s);
        match e {
            Elem::Key(l) => draw_key(&mut c, &l, x, y, w, h, s),
            Elem::Cluster(ls) => {
                let (kw, kh, g) = (46.0 * s, 44.0 * s, 5.0 * s);
                let top = y - h / 2.0 + kh / 2.0;
                let bottom = top + kh + g;
                draw_key(&mut c, &ls[0], x, top, kw, kh, s);
                draw_key(&mut c, &ls[1], x - kw - g, bottom, kw, kh, s);
                draw_key(&mut c, &ls[2], x, bottom, kw, kh, s);
                draw_key(&mut c, &ls[3], x + kw + g, bottom, kw, kh, s);
            }
            Elem::MouseMove => {
                draw_mouse(&mut c, x, y, s * 1.25, None);
                let r = 50.0 * s;
                for (dx, dy) in [(0.0, -1.0), (0.0, 1.0), (-1.0, 0.0), (1.0, 0.0)] {
                    draw_arrow(&mut c, x + dx * r, y + dy * r, dx, dy, s);
                }
            }
            Elem::MouseButton(b) => {
                draw_mouse(&mut c, x, y, s * 1.3, Some(b));
                if b > 2 {
                    c.text(&format!("{}", b + 1), (14.0 * s) as u32, x, y + 12.0 * s);
                }
            }
        }
    }
    c.rgba8()
}

/// The shared picture's keyboard version (guides without a slot): every
/// prompted control's key.
fn draw_keyboard_all() -> Vec<u8> {
    let all: Vec<String> = ["StickBaseL", "Triangle", "Circle", "Cross", "Square", "L1", "R1", "L2", "R2", "StickBaseR"].iter().map(|s| s.to_string()).collect();
    draw_keyboard_set(&all)
}

// ---------------------------------------------------------------------------
// Drawing: shapes as signed distances (texels, negative inside), filled and
// outlined with anti-aliased edges, premultiplied RGBA.

const W: usize = 256;
const H: usize = 128;

struct Canvas {
    px: Vec<[f32; 4]>,
}

const FILL: [f32; 4] = [0.0, 0.0, 0.0, 0.93];
const LINE: [f32; 4] = [0.9, 0.9, 0.9, 1.0];
const LINE_W: f32 = 1.2;

impl Canvas {
    fn new() -> Self {
        Canvas { px: vec![[0.0; 4]; W * H] }
    }

    fn paint(&mut self, cover: impl Fn(f32, f32) -> f32, c: [f32; 4]) {
        for y in 0..H {
            for x in 0..W {
                let a = cover(x as f32 + 0.5, y as f32 + 0.5).clamp(0.0, 1.0) * c[3];
                if a <= 0.0 {
                    continue;
                }
                let p = &mut self.px[y * W + x];
                for i in 0..3 {
                    p[i] = c[i] * a + p[i] * (1.0 - a);
                }
                p[3] = a + p[3] * (1.0 - a);
            }
        }
    }

    fn fill(&mut self, d: impl Fn(f32, f32) -> f32, c: [f32; 4]) {
        self.paint(|x, y| 0.5 - d(x, y), c);
    }

    fn stroke(&mut self, d: impl Fn(f32, f32) -> f32, w: f32, c: [f32; 4]) {
        self.paint(|x, y| w / 2.0 + 0.5 - d(x, y).abs(), c);
    }

    /// A shape filled black and outlined in the game's thin white line.
    fn part(&mut self, d: impl Fn(f32, f32) -> f32 + Copy) {
        self.fill(d, FILL);
        self.stroke(d, LINE_W, LINE);
    }

    /// Text centred on (cx, cy) in the line colour.
    fn text(&mut self, s: &str, px: u32, cx: f32, cy: f32) {
        let Some((w, h, cov)) = crate::pc_files::raster(s, px, 700) else { return };
        let (x0, y0) = ((cx - w as f32 / 2.0).round() as i32, (cy - h as f32 / 2.0).round() as i32);
        for ty in 0..h as i32 {
            for tx in 0..w as i32 {
                let (x, y) = (x0 + tx, y0 + ty);
                if x < 0 || y < 0 || x >= W as i32 || y >= H as i32 {
                    continue;
                }
                let a = cov[(ty as u32 * w + tx as u32) as usize] as f32 / 255.0;
                if a <= 0.0 {
                    continue;
                }
                let p = &mut self.px[y as usize * W + x as usize];
                for i in 0..3 {
                    p[i] = LINE[i] * a + p[i] * (1.0 - a);
                }
                p[3] = a + p[3] * (1.0 - a);
            }
        }
    }

    /// Straight-alpha RGBA8, row by row.
    fn rgba8(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(W * H * 4);
        for p in &self.px {
            let a = p[3].clamp(0.0, 1.0);
            let c = |v: f32| if a > 0.0 { (v / a * 255.0).round().clamp(0.0, 255.0) as u8 } else { 0 };
            out.extend_from_slice(&[c(p[0]), c(p[1]), c(p[2]), (a * 255.0).round() as u8]);
        }
        out
    }
}

fn circle(cx: f32, cy: f32, r: f32) -> impl Fn(f32, f32) -> f32 + Copy {
    move |x, y| ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() - r
}

/// A rounded box: centre, half sizes, corner radius, rotation (radians).
fn rbox(cx: f32, cy: f32, hw: f32, hh: f32, r: f32, rot: f32) -> impl Fn(f32, f32) -> f32 + Copy {
    let (s, c) = rot.sin_cos();
    move |x, y| {
        let (dx, dy) = (x - cx, y - cy);
        let (px, py) = ((dx * c + dy * s).abs(), (-dx * s + dy * c).abs());
        let (qx, qy) = (px - (hw - r), py - (hh - r));
        (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt() + qx.max(qy).min(0.0) - r
    }
}

fn capsule(ax: f32, ay: f32, bx: f32, by: f32, r: f32) -> impl Fn(f32, f32) -> f32 + Copy {
    move |x, y| {
        let (pax, pay, bax, bay) = (x - ax, y - ay, bx - ax, by - ay);
        let h = ((pax * bax + pay * bay) / (bax * bax + bay * bay)).clamp(0.0, 1.0);
        ((pax - bax * h).powi(2) + (pay - bay * h).powi(2)).sqrt() - r
    }
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    b * (1.0 - h) + a * h - k * h * (1.0 - h)
}

/// An Xbox-style pad: left stick high, d-pad low, A/B/X/Y, bumpers and
/// triggers, View, Menu and the guide button.
fn draw_xbox() -> Vec<u8> {
    let mut c = Canvas::new();
    for (x, rot) in [(74.0, -0.10f32), (182.0, 0.10)] {
        c.part(rbox(x, 8.0, 15.0, 6.0, 4.0, rot));
    }
    for (x, rot) in [(72.0, -0.14f32), (184.0, 0.14)] {
        c.part(rbox(x, 16.0, 27.0, 5.0, 4.5, rot));
    }
    let top = rbox(128.0, 51.0, 92.0, 39.0, 30.0, 0.0);
    let gl = capsule(64.0, 62.0, 42.0, 106.0, 21.0);
    let gr = capsule(192.0, 62.0, 214.0, 106.0, 21.0);
    let body = move |x: f32, y: f32| smin(smin(top(x, y), gl(x, y), 12.0), gr(x, y), 12.0);
    c.part(body);
    // Sticks: a ring with the cap inside.
    for (x, y) in [(74.0, 46.0), (156.0, 76.0)] {
        c.part(circle(x, y, 13.5));
        c.stroke(circle(x, y, 8.5), LINE_W, LINE);
    }
    // D-pad.
    let (dx, dy) = (100.0, 76.0);
    let h = rbox(dx, dy, 11.0, 4.0, 1.5, 0.0);
    let v = rbox(dx, dy, 4.0, 11.0, 1.5, 0.0);
    c.part(move |x, y| h(x, y).min(v(x, y)));
    // A, B, X and Y.
    for (l, x, y) in [("Y", 184.0, 33.0), ("X", 171.0, 46.0), ("B", 197.0, 46.0), ("A", 184.0, 59.0)] {
        c.part(circle(x, y, 6.5));
        c.text(l, 10, x, y);
    }
    // View and Menu, the guide button.
    for x in [112.0, 144.0] {
        c.part(circle(x, 41.0, 4.5));
    }
    c.stroke(rbox(110.5, 39.5, 1.6, 1.3, 0.2, 0.0), 0.8, LINE);
    c.stroke(rbox(113.0, 42.0, 1.6, 1.3, 0.2, 0.0), 0.8, LINE);
    for y in [39.5f32, 41.0, 42.5] {
        c.stroke(rbox(144.0, y, 2.0, 0.1, 0.0, 0.0), 0.7, LINE);
    }
    c.part(circle(128.0, 25.0, 7.0));
    c.rgba8()
}

// ---------------------------------------------------------------------------
// BC7 (mode 6: one subset, RGBA endpoints of 7 bits plus a p-bit, 4-bit
// indices) and the picture's tiling.

const W4: [u32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

fn interp(e0: [u32; 4], e1: [u32; 4], i: usize) -> [u32; 4] {
    let w = W4[i];
    let mut o = [0u32; 4];
    for c in 0..4 {
        o[c] = ((64 - w) * e0[c] + w * e1[c] + 32) >> 6;
    }
    o
}

/// The block's error with endpoints e0/e1 (8-bit, already quantized), and
/// each texel's best index.
fn block_fit(px: &[[u8; 4]; 16], e0: [u32; 4], e1: [u32; 4]) -> (u64, [u8; 16]) {
    let pal: Vec<[u32; 4]> = (0..16).map(|i| interp(e0, e1, i)).collect();
    let mut err = 0u64;
    let mut idx = [0u8; 16];
    for (t, p) in px.iter().enumerate() {
        let mut best = (u64::MAX, 0u8);
        for (i, q) in pal.iter().enumerate() {
            let e = texel_err(*p, *q);
            if e < best.0 {
                best = (e, i as u8);
            }
        }
        err += best.0;
        idx[t] = best.1;
    }
    (err, idx)
}

/// An endpoint quantized to 7 bits per channel with a shared p-bit: (the
/// 8-bit value it decodes to, the 7-bit values, the p-bit).
fn quantize(e: [u8; 4]) -> ([u32; 4], [u32; 4], u32) {
    let mut best = (u64::MAX, [0u32; 4], [0u32; 4], 0u32);
    for p in 0..2u32 {
        let mut q7 = [0u32; 4];
        let mut v = [0u32; 4];
        let mut err = 0u64;
        for c in 0..4 {
            let x = e[c] as i32;
            let mut k = ((x - p as i32) as f32 / 2.0).round() as i32;
            k = k.clamp(0, 127);
            q7[c] = k as u32;
            v[c] = (k as u32) << 1 | p;
            err += ((v[c] as i32 - x) * (v[c] as i32 - x)) as u64;
        }
        if err < best.0 {
            best = (err, v, q7, p);
        }
    }
    (best.1, best.2, best.3)
}

/// The error of decoded texels against the wanted ones (alpha weighs
/// double; colour by how much the texel shows).
fn texel_err(p: [u8; 4], q: [u32; 4]) -> u64 {
    let a = p[3] as i64;
    let mut e = 0i64;
    for c in 0..3 {
        let d = (p[c] as i64 - q[c] as i64) * a / 255;
        e += d * d;
    }
    let da = p[3] as i64 - q[3] as i64;
    (e + 2 * da * da) as u64
}

const W2: [u32; 4] = [0, 21, 43, 64];

fn lerp2(a: u32, b: u32, i: usize) -> u32 {
    ((64 - W2[i]) * a + W2[i] * b + 32) >> 6
}

/// Mode 5: RGB endpoints of 7 bits and alpha endpoints of 8 bits, each
/// with its own 2-bit indices (no rotation): colour and coverage apart,
/// as line art on a transparent ground needs. Returns the block and its
/// error.
fn encode_mode5(px: &[[u8; 4]; 16]) -> ([u8; 16], u64) {
    // Alpha: the block's extremes.
    let (amin, amax) = px.iter().fold((255u32, 0u32), |(lo, hi), p| (lo.min(p[3] as u32), hi.max(p[3] as u32)));
    let mut aidx = [0u8; 16];
    for (t, p) in px.iter().enumerate() {
        aidx[t] = (0..4).min_by_key(|&i| (lerp2(amin, amax, i) as i32 - p[3] as i32).abs()).unwrap() as u8;
    }
    // Colour: candidate endpoint pairs among the visible texels' colours.
    let q7 = |v: u8| ((v as u32 * 127 + 127) / 255).min(127);
    let x8 = |v: u32| v << 1 | v >> 6;
    let mut colours: Vec<[u8; 3]> = Vec::new();
    for p in px.iter().filter(|p| p[3] > 0) {
        let c = [p[0], p[1], p[2]];
        if !colours.contains(&c) {
            colours.push(c);
        }
    }
    if colours.is_empty() {
        colours.push([0, 0, 0]);
    }
    let mut cands: Vec<([u8; 3], [u8; 3])> = Vec::new();
    let (mut lo, mut hi) = ([255u8; 3], [0u8; 3]);
    for c in &colours {
        for k in 0..3 {
            lo[k] = lo[k].min(c[k]);
            hi[k] = hi[k].max(c[k]);
        }
    }
    cands.push((lo, hi));
    if colours.len() <= 12 {
        for i in 0..colours.len() {
            for j in i..colours.len() {
                cands.push((colours[i], colours[j]));
            }
        }
    }
    let mut best: Option<(u64, [u32; 3], [u32; 3], [u8; 16])> = None;
    for (a, b) in cands {
        let e0 = [q7(a[0]), q7(a[1]), q7(a[2])];
        let e1 = [q7(b[0]), q7(b[1]), q7(b[2])];
        let mut err = 0u64;
        let mut cidx = [0u8; 16];
        for (t, p) in px.iter().enumerate() {
            let al = lerp2(amin, amax, aidx[t] as usize);
            let (i, e) = (0..4)
                .map(|i| {
                    let q = [lerp2(x8(e0[0]), x8(e1[0]), i), lerp2(x8(e0[1]), x8(e1[1]), i), lerp2(x8(e0[2]), x8(e1[2]), i), al];
                    (i, texel_err(*p, q))
                })
                .min_by_key(|x| x.1)
                .unwrap();
            cidx[t] = i as u8;
            err += e;
        }
        if best.as_ref().is_none_or(|b| err < b.0) {
            best = Some((err, e0, e1, cidx));
        }
    }
    let (err, mut e0, mut e1, mut cidx) = best.unwrap();
    let (mut a0, mut a1) = (amin, amax);
    // (The first index of each set has its top bit implied 0.)
    if cidx[0] >= 2 {
        std::mem::swap(&mut e0, &mut e1);
        for i in cidx.iter_mut() {
            *i = 3 - *i;
        }
    }
    if aidx[0] >= 2 {
        std::mem::swap(&mut a0, &mut a1);
        for i in aidx.iter_mut() {
            *i = 3 - *i;
        }
    }
    let mut bits = 0u128;
    let mut pos = 0u32;
    let mut put = |v: u128, n: u32| {
        bits |= (v & ((1u128 << n) - 1)) << pos;
        pos += n;
    };
    put(1 << 5, 6);
    put(0, 2);
    for c in 0..3 {
        put(e0[c] as u128, 7);
        put(e1[c] as u128, 7);
    }
    put(a0 as u128, 8);
    put(a1 as u128, 8);
    for (i, &v) in cidx.iter().enumerate() {
        put(v as u128, if i == 0 { 1 } else { 2 });
    }
    for (i, &v) in aidx.iter().enumerate() {
        put(v as u128, if i == 0 { 1 } else { 2 });
    }
    (bits.to_le_bytes(), err)
}

/// The better of modes 5 and 6 for the block.
fn encode_block(px: &[[u8; 4]; 16]) -> [u8; 16] {
    let (b5, e5) = encode_mode5(px);
    let (b6, e6) = encode_mode6(px);
    if e5 <= e6 { b5 } else { b6 }
}

fn encode_mode6(px: &[[u8; 4]; 16]) -> ([u8; 16], u64) {
    // Candidates: every pair of the block's distinct colours (line art has
    // few), and the per-channel extremes.
    let mut colours: Vec<[u8; 4]> = Vec::new();
    for p in px {
        let q = if p[3] == 0 { [0, 0, 0, 0] } else { *p };
        if !colours.contains(&q) {
            colours.push(q);
        }
    }
    let (mut lo, mut hi) = ([255u8; 4], [0u8; 4]);
    for p in px {
        for c in 0..4 {
            lo[c] = lo[c].min(p[c]);
            hi[c] = hi[c].max(p[c]);
        }
    }
    let mut cands: Vec<([u8; 4], [u8; 4])> = vec![(lo, hi)];
    if colours.len() <= 12 {
        for i in 0..colours.len() {
            for j in 0..colours.len() {
                if i != j {
                    cands.push((colours[i], colours[j]));
                }
            }
        }
        if colours.len() == 1 {
            cands.push((colours[0], colours[0]));
        }
    }
    let mut best: Option<(u64, [u32; 4], [u32; 4], u32, u32, [u8; 16])> = None;
    for (a, b) in cands {
        let (v0, q0, p0) = quantize(a);
        let (v1, q1, p1) = quantize(b);
        let (err, idx) = block_fit(px, v0, v1);
        if best.as_ref().is_none_or(|b| err < b.0) {
            best = Some((err, q0, q1, p0, p1, idx));
        }
    }
    let (err, mut q0, mut q1, mut p0, mut p1, mut idx) = best.unwrap();
    // (The first index's top bit is implied 0: swap the endpoints if needed.)
    if idx[0] >= 8 {
        std::mem::swap(&mut q0, &mut q1);
        std::mem::swap(&mut p0, &mut p1);
        for i in idx.iter_mut() {
            *i = 15 - *i;
        }
    }
    let mut bits = 0u128;
    let mut pos = 0u32;
    let mut put = |v: u128, n: u32| {
        bits |= (v & ((1u128 << n) - 1)) << pos;
        pos += n;
    };
    put(1 << 6, 7);
    for c in 0..4 {
        put(q0[c] as u128, 7);
        put(q1[c] as u128, 7);
    }
    put(p0 as u128, 1);
    put(p1 as u128, 1);
    for (i, &v) in idx.iter().enumerate() {
        put(v as u128, if i == 0 { 3 } else { 4 });
    }
    (bits.to_le_bytes(), err)
}

/// RGBA8 (W x H, row by row) as BC7 blocks in the texture's tiled order.
fn bc7_tiled(rgba: &[u8]) -> Vec<u8> {
    let (bw, bh) = (W / 4, H / 4);
    let mut linear = vec![0u8; bw * bh * 16];
    for by in 0..bh {
        for bx in 0..bw {
            let mut px = [[0u8; 4]; 16];
            for t in 0..16 {
                let (x, y) = (bx * 4 + t % 4, by * 4 + t / 4);
                let o = (y * W + x) * 4;
                px[t] = [rgba[o], rgba[o + 1], rgba[o + 2], rgba[o + 3]];
            }
            let b = encode_block(&px);
            linear[(by * bw + bx) * 16..(by * bw + bx) * 16 + 16].copy_from_slice(&b);
        }
    }
    let mut out = vec![0u8; linear.len()];
    crate::gpu::tiling::tile_slice(&linear, &mut out, bw as u32, bh as u32, bw as u32, bh as u32, 128, crate::gpu::tiling::tile_mode(13));
    out
}

// ---------------------------------------------------------------------------
// The game's textures.

const PAD_FILE: &str = "game/textures/ui/alert/guide_controller_base_bc7.gnf";
const PAD_BYTES: usize = W * H;

struct Pictures {
    /// The game's own texels (its file's, after the 256-byte header).
    own: Vec<u8>,
    xbox: Vec<u8>,
    /// The shared picture's keyboard version.
    keyboard: Vec<u8>,
}

fn pictures() -> Option<&'static Pictures> {
    static P: OnceLock<Option<Pictures>> = OnceLock::new();
    P.get_or_init(|| {
        let f = crate::pc_files::game_file(PAD_FILE, "")?;
        if f.len() < 256 + PAD_BYTES {
            crate::warn!("prompts: {PAD_FILE}: {} bytes, not the expected picture", f.len());
            return None;
        }
        let t0 = std::time::Instant::now();
        let p = Pictures { own: f[256..256 + PAD_BYTES].to_vec(), xbox: bc7_tiled(&draw_xbox()), keyboard: bc7_tiled(&draw_keyboard_all()) };
        crate::info!("prompts: Xbox and keyboard pictures drawn and encoded in {:.0} ms", t0.elapsed().as_secs_f64() * 1000.0);
        Some(p)
    })
    .as_ref()
}

/// Draws and encodes the slots' keyboard pictures that are missing or drawn
/// for other bindings, away from the game's threads; the next pad read
/// writes them.
fn prepare_keyboard() {
    static BUSY: AtomicBool = AtomicBool::new(false);
    if BUSY.swap(true, Ordering::AcqRel) {
        return;
    }
    let _ = std::thread::Builder::new().name("prompt-pictures".into()).spawn(|| {
        loop {
            let bgen = BINDINGS_GEN.load(Ordering::Relaxed);
            let todo = {
                let st = SLOT_STATE.lock();
                st.slots.iter().position(|s| s.keyboard.as_ref().is_none_or(|k| k.0 != bgen)).map(|i| (i, st.slots[i].set.clone()))
            };
            let Some((i, set)) = todo else { break };
            let pic = bc7_tiled(&draw_keyboard_set(&set));
            if let Some(s) = SLOT_STATE.lock().slots.get_mut(i) {
                s.keyboard = Some((bgen, pic));
            }
            WRITE_DUE.store(true, Ordering::Relaxed);
        }
        BUSY.store(false, Ordering::Release);
    });
}

/// Writes `data` over a picture in the game's memory, the slot's marker
/// kept in its first block.
fn put(addr: u64, data: &[u8], marker: Option<[u8; 16]>) {
    if addr == 0 || !crate::mem::is_guest(addr, PAD_BYTES as u64) || data.len() != PAD_BYTES {
        return;
    }
    // SAFETY: the game's texture memory for a picture (its descriptor's
    // base, the whole texture checked as guest memory).
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), addr as *mut u8, data.len());
        if let Some(m) = marker {
            std::ptr::copy_nonoverlapping(m.as_ptr(), addr as *mut u8, 16);
        }
    }
}

/// Every picture seen gets the style's texels.
fn write_pictures(s: Style) {
    let Some(p) = pictures() else { return };
    let mut need_keyboard = false;
    let bgen = BINDINGS_GEN.load(Ordering::Relaxed);
    let st = SLOT_STATE.lock();
    put(st.shared, match s {
        Style::PlayStation => &p.own,
        Style::Xbox => &p.xbox,
        Style::Keyboard => &p.keyboard,
    }, None);
    for (i, slot) in st.slots.iter().enumerate() {
        let data = match s {
            Style::PlayStation => Some(&p.own),
            Style::Xbox => Some(&p.xbox),
            Style::Keyboard => match &slot.keyboard {
                Some((g, k)) if *g == bgen => Some(k),
                _ => {
                    need_keyboard = true;
                    None
                }
            },
        };
        if let Some(d) = data {
            put(slot.addr, d, Some(marker(i)));
        }
    }
    drop(st);
    if need_keyboard {
        prepare_keyboard();
    }
    crate::debug!("prompts: the pictures show the {s:?} style");
}

/// The texture cache made a 256x128 BC7 image at `base`: a slot's picture
/// (by its marker) or the shared one (by its file's texels) gets the style
/// in force.
pub fn note_texture(base: u64) {
    if !crate::mem::is_guest(base, PAD_BYTES as u64) {
        return;
    }
    let Some(p) = pictures() else { return };
    // SAFETY: guest memory, checked above.
    let head: [u8; 16] = unsafe { std::ptr::read_unaligned(base as *const [u8; 16]) };
    let mut st = SLOT_STATE.lock();
    if let Some(i) = (0..st.slots.len()).find(|&i| head == marker(i)) {
        if st.slots[i].addr != base {
            st.slots[i].addr = base;
            crate::debug!("prompts: picture slot {i} ({}) at {base:#x}", st.slots[i].set.join("+"));
            WRITE_DUE.store(true, Ordering::Relaxed);
        }
    } else if head[..] == p.own[..16] && st.shared != base {
        st.shared = base;
        crate::info!("prompts: the game's shared prompt picture is at {base:#x}");
        WRITE_DUE.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decodes a mode-5 or mode-6 block (what `encode_block` makes).
    fn decode6(b: [u8; 16]) -> [[u8; 4]; 16] {
        let bits = u128::from_le_bytes(b);
        if bits & 0x7f == 1 << 5 {
            return decode5(bits);
        }
        let mut pos = 0;
        let mut get = |n: u32| {
            let v = (bits >> pos) & ((1u128 << n) - 1);
            pos += n;
            v as u32
        };
        assert_eq!(get(7), 1 << 6);
        let mut e = [[0u32; 4]; 2];
        for c in 0..4 {
            e[0][c] = get(7);
            e[1][c] = get(7);
        }
        let (p0, p1) = (get(1), get(1));
        for c in 0..4 {
            e[0][c] = e[0][c] << 1 | p0;
            e[1][c] = e[1][c] << 1 | p1;
        }
        let mut out = [[0u8; 4]; 16];
        for (i, o) in out.iter_mut().enumerate() {
            let ix = get(if i == 0 { 3 } else { 4 }) as usize;
            let v = interp(e[0], e[1], ix);
            *o = [v[0] as u8, v[1] as u8, v[2] as u8, v[3] as u8];
        }
        out
    }

    fn decode5(bits: u128) -> [[u8; 4]; 16] {
        let mut pos = 0;
        let mut get = |n: u32| {
            let v = (bits >> pos) & ((1u128 << n) - 1);
            pos += n;
            v as u32
        };
        assert_eq!(get(6), 1 << 5);
        assert_eq!(get(2), 0);
        let mut e = [[0u32; 3]; 2];
        for c in 0..3 {
            e[0][c] = get(7);
            e[1][c] = get(7);
        }
        let (a0, a1) = (get(8), get(8));
        let ci: Vec<usize> = (0..16).map(|i| get(if i == 0 { 1 } else { 2 }) as usize).collect();
        let ai: Vec<usize> = (0..16).map(|i| get(if i == 0 { 1 } else { 2 }) as usize).collect();
        let x8 = |v: u32| v << 1 | v >> 6;
        let mut out = [[0u8; 4]; 16];
        for i in 0..16 {
            for c in 0..3 {
                out[i][c] = lerp2(x8(e[0][c]), x8(e[1][c]), ci[i]) as u8;
            }
            out[i][3] = lerp2(a0, a1, ai[i]) as u8;
        }
        out
    }

    #[test]
    fn bc7_mode6_round_trips_line_art() {
        // Transparent, black fill and a white line in one block.
        let mut px = [[0u8; 4]; 16];
        for (i, p) in px.iter_mut().enumerate() {
            *p = match i % 4 {
                0 => [0, 0, 0, 0],
                1 | 2 => [0, 0, 0, 237],
                _ => [230, 230, 230, 255],
            };
        }
        let d = decode6(encode_block(&px));
        for (a, b) in px.iter().zip(d.iter()) {
            // (Alpha within a few steps; colour where the texel shows.)
            assert!((a[3] as i32 - b[3] as i32).abs() <= 24, "{a:?} -> {b:?}");
        }
        let flat = [[200u8, 100, 50, 255]; 16];
        let d = decode6(encode_block(&flat));
        for p in d {
            assert!(p.iter().zip([200u8, 100, 50, 255]).all(|(x, y)| (*x as i32 - y as i32).abs() <= 1), "{p:?}");
        }
    }

    /// Development: the pictures as PNG files in TLG_DUMP_DIR (over a grey
    /// ground, the way they look in a prompt), scaled 3x.
    #[test]
    #[ignore]
    fn dump_pictures() {
        let Ok(dir) = std::env::var("TLG_DUMP_DIR") else { return };
        let set = |s: &str| s.split('+').map(str::to_string).collect::<Vec<_>>();
        let pics = [
            ("xbox", draw_xbox()),
            ("kb_move", draw_keyboard_set(&set("StickBaseL"))),
            ("kb_camera", draw_keyboard_set(&set("StickBaseR"))),
            ("kb_jump_move", draw_keyboard_set(&set("StickBaseL+Triangle"))),
            ("kb_grab", draw_keyboard_set(&set("Circle"))),
            ("kb_dpad", draw_keyboard_set(&set("Left+Right"))),
            ("kb_all", draw_keyboard_set(&set("StickBaseL+StickBaseR+Triangle+Circle+Square+Cross+R1+R2+L1+L2+Up+Down+Left+Right+Push"))),
        ];
        for (name, rgba) in pics {
            let s = 3;
            let mut out = vec![0u8; W * H * 4 * s * s];
            for y in 0..H * s {
                for x in 0..W * s {
                    let i = ((y / s) * W + x / s) * 4;
                    let a = rgba[i + 3] as u32;
                    let o = (y * W * s + x) * 4;
                    for c in 0..3 {
                        out[o + c] = ((rgba[i + c] as u32 * a + 70 * (255 - a)) / 255) as u8;
                    }
                    out[o + 3] = 255;
                }
            }
            std::fs::write(format!("{dir}/{name}.png"), crate::gpu::texdump::png((W * s) as u32, (H * s) as u32, &out)).unwrap();
        }
    }

    #[test]
    fn pictures_draw() {
        let x = draw_xbox();
        assert_eq!(x.len(), W * H * 4);
        // Something drawn, the corners empty.
        assert!(x.chunks(4).filter(|p| p[3] > 0).count() > 4000);
        assert_eq!(x[3], 0);
        assert_eq!(bc7_tiled(&x).len(), PAD_BYTES);
    }
}
