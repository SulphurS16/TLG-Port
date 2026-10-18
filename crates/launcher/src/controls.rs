//! Controls page: every keyboard/mouse binding (two inputs per control),
//! conflicts, reset; mouse look settings. A binding is changed in a small
//! window that takes the next key, mouse button or wheel turn.

use crate::ui::{self, wide};
use crate::{App, TITLE};
use std::cell::Cell;
use tlg_config::bindings::{self, Bindings, CONTROLS, Control, Input};
use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows_sys::Win32::Graphics::Gdi::{BeginPaint, DrawTextW, EndPaint, PAINTSTRUCT, SelectObject, SetBkMode, TRANSPARENT, DT_CENTER, DT_VCENTER, DT_WORDBREAK};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::Controls::{NM_DBLCLK, NMHDR, NMITEMACTIVATE};
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{EnableWindow, MAPVK_VSC_TO_VK_EX, MapVirtualKeyW, SetFocus};
use windows_sys::Win32::UI::WindowsAndMessaging::*;

const ID_LIST: i32 = 301;
const ID_PRIMARY: i32 = 302;
const ID_SECONDARY: i32 = 303;
const ID_CLEAR: i32 = 304;
const ID_DEFAULTS: i32 = 305;
const ID_LOOK: i32 = 310;
const ID_SENS_X: i32 = 311;
const ID_SENS_Y: i32 = 312;
const ID_INV_X: i32 = 313;
const ID_INV_Y: i32 = 314;
const ID_MENU_MOUSE: i32 = 315;
const ID_RELEASE_TEXT: i32 = 316;
const ID_RELEASE: i32 = 317;

#[derive(Default)]
pub struct Page {
    list: usize,
    look: usize,
    sens_x: usize,
    sens_y: usize,
    inv_x: usize,
    inv_y: usize,
    menu_mouse: usize,
    release_text: usize,
}

fn h(v: usize) -> HWND {
    v as HWND
}

pub fn build(app: &mut App) {
    let b = app.builder();
    unsafe {
        let r = (16, 52, 748, 304);
        let list = ui::list_view(&b, r, ID_LIST, &[("Action", 300), ("Primary", 150), ("Secondary", 150), ("Note", 126)]);
        app.controls.list = app.put(2, list, r, false) as usize;
        for (t, id, r) in [
            ("Change primary...", ID_PRIMARY, (16, 362, 140, 28)),
            ("Change secondary...", ID_SECONDARY, (164, 362, 150, 28)),
            ("Clear", ID_CLEAR, (322, 362, 90, 28)),
            ("Reset to defaults", ID_DEFAULTS, (624, 362, 140, 28)),
        ] {
            app.put(2, b.button(t, r, id), r, false);
        }
        let r = (16, 396, 748, 118);
        app.put(2, b.group("Mouse", r), r, false);
        let r = (32, 418, 160, 22);
        app.controls.look = app.put(2, b.check("Mouse look", r, ID_LOOK), r, false) as usize;
        let r = (200, 420, 110, 20);
        app.put(2, b.label("Sensitivity X:", r, -1), r, false);
        let r = (312, 416, 60, 24);
        app.controls.sens_x = app.put(2, b.edit("", r, ID_SENS_X), r, false) as usize;
        let r = (386, 420, 26, 20);
        app.put(2, b.label("Y:", r, -1), r, false);
        let r = (412, 416, 60, 24);
        app.controls.sens_y = app.put(2, b.edit("", r, ID_SENS_Y), r, false) as usize;
        let r = (32, 450, 120, 22);
        app.controls.inv_x = app.put(2, b.check("Invert X", r, ID_INV_X), r, false) as usize;
        let r = (160, 450, 120, 22);
        app.controls.inv_y = app.put(2, b.check("Invert Y", r, ID_INV_Y), r, false) as usize;
        let r = (290, 450, 460, 22);
        app.controls.menu_mouse = app.put(2, b.check("Mouse in menus", r, ID_MENU_MOUSE), r, false) as usize;
        let r = (32, 482, 160, 20);
        app.put(2, b.label("Release the mouse:", r, -1), r, false);
        let r = (196, 482, 100, 20);
        app.controls.release_text = app.put(2, b.label("", r, ID_RELEASE_TEXT), r, false) as usize;
        let r = (300, 478, 100, 26);
        app.put(2, b.button("Change...", r, ID_RELEASE), r, false);
    }
    load(app);
}

fn slot_text(b: &Bindings, c: Control, slot: usize) -> String {
    b.inputs(c).get(slot).map(|&i| bindings::input_label(i)).unwrap_or_default()
}

/// Fills the list and the mouse settings from the bindings.
pub fn load(app: &mut App) {
    let b = &app.bindings;
    let conflicts = b.conflicts();
    unsafe {
        let list = h(app.controls.list);
        let sel = ui::lv_selected(list);
        ui::lv_clear(list);
        for info in CONTROLS {
            let ins = b.inputs(info.control);
            let mut note = String::new();
            for i in ins {
                if let Some((_, cs)) = conflicts.iter().find(|(x, _)| x == i) {
                    let others: Vec<&str> = cs.iter().filter(|c| **c != info.control).filter_map(|c| bindings::control_info(*c)).map(|ci| ci.name).collect();
                    note = format!("{} also: {}", bindings::input_label(*i), others.join(", "));
                }
            }
            if ins.len() > 2 && note.is_empty() {
                note = format!("+{} more in input.txt", ins.len() - 2);
            }
            ui::lv_add(list, &[info.label.to_string(), slot_text(b, info.control, 0), slot_text(b, info.control, 1), note]);
        }
        if let Some(s) = sel {
            ui::lv_select(list, s);
        }
        ui::set_check(h(app.controls.look), b.mouse_look);
        ui::set_text(h(app.controls.sens_x), &format!("{}", b.sensitivity_x));
        ui::set_text(h(app.controls.sens_y), &format!("{}", b.sensitivity_y));
        ui::set_check(h(app.controls.inv_x), b.invert_x);
        ui::set_check(h(app.controls.inv_y), b.invert_y);
        ui::set_check(h(app.controls.menu_mouse), b.menu_mouse);
        ui::set_text(h(app.controls.release_text), &bindings::key_name(b.release_key));
    }
}

/// Changes input `slot` of the selected control.
fn change(app: &mut App, row: usize, slot: usize) {
    let Some(info) = CONTROLS.get(row) else { return };
    let prompt = format!("{}\n({} input)\n\nPress a key, click a mouse button here or turn the wheel.", info.label, if slot == 0 { "primary" } else { "secondary" });
    let Some(got) = capture(app.hwnd, &prompt, true) else { return };
    let Some(input) = got else {
        app.bindings.set_slot(info.control, slot, None);
        app.save_bindings();
        load(app);
        return;
    };
    let others: Vec<Control> = app.bindings.bound_to(input).into_iter().filter(|c| *c != info.control).collect();
    if !others.is_empty() {
        let names: Vec<&str> = others.iter().filter_map(|c| bindings::control_info(*c)).map(|c| c.label).collect();
        let text = format!(
            "{} is already used for: {}.\n\nYes: use it only for {} (removed from the others).\nNo: use it for both (pressing it does all of them).",
            bindings::input_label(input),
            names.join(", "),
            info.label
        );
        match unsafe { ui::message(app.hwnd, &text, TITLE, MB_ICONQUESTION | MB_YESNOCANCEL) } {
            IDYES => app.bindings.unbind_elsewhere(input, info.control),
            IDNO => {}
            _ => return,
        }
    }
    app.bindings.set_slot(info.control, slot, Some(input));
    app.save_bindings();
    load(app);
    app.set_status(&format!("{}: {} (saved; applies the next time the game starts).", info.label, bindings::input_label(input)));
}

fn parse_sens(s: &str) -> Option<f32> {
    s.trim().replace(',', ".").parse::<f32>().ok().filter(|v| v.is_finite() && *v > 0.0 && *v <= 100.0)
}

pub fn command(app: &mut App, id: i32, code: u32) {
    let row = unsafe { ui::lv_selected(h(app.controls.list)) };
    match (id, code) {
        (ID_PRIMARY, 0) | (ID_SECONDARY, 0) => match row {
            Some(r) => change(app, r, (id == ID_SECONDARY) as usize),
            None => app.set_status("Select an action in the list first."),
        },
        (ID_CLEAR, 0) => {
            if let Some(info) = row.and_then(|r| CONTROLS.get(r)) {
                app.bindings.set_slot(info.control, 1, None);
                app.bindings.set_slot(info.control, 0, None);
                app.save_bindings();
                load(app);
            }
        }
        (ID_DEFAULTS, 0) => {
            if unsafe { ui::message(app.hwnd, "Reset every keyboard and mouse binding and the mouse settings to the defaults?", TITLE, MB_ICONQUESTION | MB_OKCANCEL) } == IDOK {
                app.bindings = Bindings::default();
                app.save_bindings();
                load(app);
                app.set_status("Bindings reset to the defaults.");
            }
        }
        (ID_LOOK | ID_INV_X | ID_INV_Y | ID_MENU_MOUSE, 0) => {
            let c = &app.controls;
            unsafe {
                app.bindings.mouse_look = ui::checked(h(c.look));
                app.bindings.invert_x = ui::checked(h(c.inv_x));
                app.bindings.invert_y = ui::checked(h(c.inv_y));
                app.bindings.menu_mouse = ui::checked(h(c.menu_mouse));
            }
            app.save_bindings();
        }
        // EN_KILLFOCUS: a sensitivity was edited.
        (ID_SENS_X | ID_SENS_Y, 0x200) => {
            let e = if id == ID_SENS_X { app.controls.sens_x } else { app.controls.sens_y };
            match parse_sens(&unsafe { ui::get_text(h(e)) }) {
                Some(v) => {
                    if id == ID_SENS_X {
                        app.bindings.sensitivity_x = v;
                    } else {
                        app.bindings.sensitivity_y = v;
                    }
                    app.save_bindings();
                }
                None => app.set_status("Sensitivity: a number above 0 and up to 100 (1.0 is the default)."),
            }
            load(app);
        }
        (ID_RELEASE, 0) => {
            if let Some(Some(Input::Key(k))) = capture(app.hwnd, "The key that frees the captured mouse\n\nPress a key.", false) {
                app.bindings.release_key = k;
                app.save_bindings();
                load(app);
            }
        }
        _ => {}
    }
}

/// List notifications: a double click changes the clicked slot.
pub fn notify(app: &mut App, hdr: &NMHDR, lp: LPARAM) {
    if hdr.idFrom != ID_LIST as usize || hdr.code != NM_DBLCLK {
        return;
    }
    let a = unsafe { &*(lp as *const NMITEMACTIVATE) };
    if a.iItem >= 0 {
        let slot = if a.iSubItem == 2 { 1 } else { 0 };
        change(app, a.iItem as usize, slot);
    }
}

// --- The capture window -------------------------------------------------

thread_local! {
    /// The captured input: None while waiting; Some(None) = cleared;
    /// Some(Some(i)) = an input; the window closes with `DONE`.
    static RESULT: Cell<Option<Option<Input>>> = const { Cell::new(None) };
    static DONE: Cell<bool> = const { Cell::new(false) };
    static MOUSE_OK: Cell<bool> = const { Cell::new(true) };
    static PROMPT: std::cell::RefCell<Vec<u16>> = const { std::cell::RefCell::new(Vec::new()) };
}

const ID_CAP_CLEAR: i32 = 1;
const ID_CAP_CANCEL: i32 = 2;

fn finish(r: Option<Option<Input>>) {
    RESULT.with(|x| x.set(r));
    DONE.with(|d| d.set(true));
}

/// The left/right-specific key of a key message.
fn specific_vk(wp: WPARAM, lp: LPARAM) -> u8 {
    let vk = (wp & 0xFF) as u8;
    let ext = (lp >> 24) & 1 != 0;
    match vk {
        0x10 => {
            let sc = ((lp >> 16) & 0xFF) as u32;
            if unsafe { MapVirtualKeyW(sc, MAPVK_VSC_TO_VK_EX) } == 0xA1 { 0xA1 } else { 0xA0 }
        }
        0x11 => {
            if ext {
                0xA3
            } else {
                0xA2
            }
        }
        0x12 => {
            if ext {
                0xA5
            } else {
                0xA4
            }
        }
        v => v,
    }
}

unsafe extern "system" fn capture_proc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    let mouse = MOUSE_OK.with(|m| m.get());
    match msg {
        WM_KEYDOWN | WM_SYSKEYDOWN => {
            finish(Some(Some(Input::Key(specific_vk(wp, lp)))));
            0
        }
        WM_LBUTTONDOWN if mouse => {
            finish(Some(Some(Input::Mouse(0))));
            0
        }
        WM_RBUTTONDOWN if mouse => {
            finish(Some(Some(Input::Mouse(1))));
            0
        }
        WM_MBUTTONDOWN if mouse => {
            finish(Some(Some(Input::Mouse(2))));
            0
        }
        WM_XBUTTONDOWN if mouse => {
            let b = if (wp >> 16) & 0xFFFF == 1 { 3 } else { 4 };
            finish(Some(Some(Input::Mouse(b))));
            1
        }
        WM_MOUSEWHEEL if mouse => {
            let d = ((wp >> 16) & 0xFFFF) as i16;
            finish(Some(Some(if d > 0 { Input::WheelUp } else { Input::WheelDown })));
            0
        }
        WM_COMMAND => {
            match (wp & 0xFFFF) as i32 {
                ID_CAP_CLEAR => finish(Some(None)),
                _ => finish(None),
            }
            // (Back to the capture: buttons must not keep the focus.)
            unsafe { SetFocus(hwnd) };
            0
        }
        WM_CLOSE => {
            finish(None);
            0
        }
        WM_PAINT => {
            let mut ps: PAINTSTRUCT = unsafe { std::mem::zeroed() };
            let dc = unsafe { BeginPaint(hwnd, &mut ps) };
            let mut r = RECT { left: 0, top: 0, right: 0, bottom: 0 };
            unsafe {
                GetClientRect(hwnd, &mut r);
                r.bottom -= r.bottom / 4;
                SetBkMode(dc, TRANSPARENT as _);
                let font = SendMessageW(hwnd, WM_GETFONT, 0, 0);
                if font != 0 {
                    SelectObject(dc, font as _);
                }
                PROMPT.with(|p| {
                    let mut t = p.borrow().clone();
                    DrawTextW(dc, t.as_mut_ptr(), -1, &mut r, DT_CENTER | DT_VCENTER | DT_WORDBREAK);
                });
                EndPaint(hwnd, &ps);
            }
            0
        }
        // (Every key goes to the window: no dialog navigation.)
        WM_GETDLGCODE => DLGC_WANTALLKEYS as LRESULT,
        _ => unsafe { DefWindowProcW(hwnd, msg, wp, lp) },
    }
}

/// Shows the capture window (modal) and returns the input: None cancelled,
/// Some(None) cleared (only with `mouse`: the binding dialog).
fn capture(owner: HWND, prompt: &str, mouse: bool) -> Option<Option<Input>> {
    RESULT.with(|x| x.set(None));
    DONE.with(|d| d.set(false));
    MOUSE_OK.with(|m| m.set(mouse));
    PROMPT.with(|p| *p.borrow_mut() = wide(prompt));
    unsafe {
        let hinst = GetModuleHandleW(std::ptr::null());
        let class = wide("tlg_capture");
        let wc = WNDCLASSEXW {
            cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
            style: CS_HREDRAW | CS_VREDRAW,
            lpfnWndProc: Some(capture_proc),
            hInstance: hinst,
            hCursor: LoadCursorW(std::ptr::null_mut(), IDC_HAND),
            hbrBackground: (windows_sys::Win32::Graphics::Gdi::COLOR_WINDOW + 1) as usize as _,
            lpszClassName: class.as_ptr(),
            ..std::mem::zeroed()
        };
        RegisterClassExW(&wc);
        let dpi = windows_sys::Win32::UI::HiDpi::GetDpiForWindow(owner).max(96);
        let s = ui::Scale(dpi);
        let mut or = RECT { left: 0, top: 0, right: 0, bottom: 0 };
        GetWindowRect(owner, &mut or);
        let (w, hh) = (s.px(420), s.px(220));
        let x = or.left + ((or.right - or.left) - w) / 2;
        let y = or.top + ((or.bottom - or.top) - hh) / 2;
        let title = wide("Change binding");
        let hwnd = CreateWindowExW(WS_EX_DLGMODALFRAME, class.as_ptr(), title.as_ptr(), WS_POPUP | WS_CAPTION | WS_SYSMENU, x, y, w, hh, owner, std::ptr::null_mut(), hinst, std::ptr::null());
        if hwnd.is_null() {
            return None;
        }
        let font = ui::message_font(dpi, false, 110);
        SendMessageW(hwnd, WM_SETFONT, font as WPARAM, 0);
        let b = ui::Builder { parent: hwnd, scale: s, font };
        if mouse {
            b.button("Clear", (110, 150, 90, 28), ID_CAP_CLEAR);
        }
        b.button("Cancel", (210, 150, 90, 28), ID_CAP_CANCEL);
        EnableWindow(owner, 0);
        ShowWindow(hwnd, SW_SHOW);
        SetFocus(hwnd);
        let mut msg: MSG = std::mem::zeroed();
        while !DONE.with(|d| d.get()) && GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) > 0 {
            // Keys typed anywhere in the window are the capture.
            if matches!(msg.message, WM_KEYDOWN | WM_SYSKEYDOWN) && msg.hwnd != hwnd {
                msg.hwnd = hwnd;
            }
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        EnableWindow(owner, 1);
        DestroyWindow(hwnd);
        windows_sys::Win32::Graphics::Gdi::DeleteObject(font as _);
        SetForegroundWindow(owner);
    }
    RESULT.with(|x| x.take())
}
