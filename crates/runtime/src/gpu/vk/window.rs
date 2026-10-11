//! Output window (Win32) running its own message loop thread.
//!
//! Keyboard state comes from key messages (left/right modifiers kept apart).
//! The mouse is read as raw input (relative counts, no pointer acceleration)
//! while it is captured: a click into the window captures it (cursor hidden
//! and confined), losing focus or the release key (`input.txt`) frees it.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, AtomicU32, Ordering};
use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, RECT, WPARAM};
use windows_sys::Win32::Graphics::Gdi::ClientToScreen;
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{MAPVK_VSC_TO_VK_EX, MapVirtualKeyW};
use windows_sys::Win32::UI::Input::{GetRawInputData, HRAWINPUT, RAWINPUT, RAWINPUTDEVICE, RAWINPUTHEADER, RID_INPUT, RIM_TYPEMOUSE, RegisterRawInputDevices};
use windows_sys::Win32::UI::WindowsAndMessaging::*;

pub struct Window {
    pub hwnd: isize,
    pub hinstance: isize,
}

static WINDOW: OnceLock<Window> = OnceLock::new();
pub static CLOSED: AtomicBool = AtomicBool::new(false);
pub static RESIZED: AtomicBool = AtomicBool::new(false);
pub static CLIENT_W: AtomicU32 = AtomicU32::new(1280);
pub static CLIENT_H: AtomicU32 = AtomicU32::new(720);
/// Virtual-key states (true = down). Shift, Ctrl and Alt are recorded as
/// their left/right codes (VK_LSHIFT...); the generic codes are set too.
pub static KEYS: [AtomicBool; 256] = [const { AtomicBool::new(false) }; 256];

/// The mouse is captured (raw counts and buttons go to the game).
pub static CAPTURED: AtomicBool = AtomicBool::new(false);
/// Raw mouse counts accumulated while captured (taken by `take_mouse`).
static MOUSE_DX: AtomicI64 = AtomicI64::new(0);
static MOUSE_DY: AtomicI64 = AtomicI64::new(0);
/// Wheel notches (WHEEL_DELTA units / 120) accumulated while captured.
static WHEEL: AtomicI32 = AtomicI32::new(0);
/// Mouse buttons down while captured: bit 0 left, 1 right, 2 middle, 3/4 side buttons.
pub static MOUSE_BUTTONS: AtomicU32 = AtomicU32::new(0);
/// Virtual key that releases the captured mouse (set from `input.txt`).
pub static RELEASE_KEY: AtomicU32 = AtomicU32::new(0x70); // F1

pub fn get() -> Option<&'static Window> {
    WINDOW.get()
}

pub fn key_down(vk: u8) -> bool {
    KEYS[vk as usize].load(Ordering::Relaxed)
}

/// Raw mouse counts and wheel notches since the previous call.
pub fn take_mouse() -> (i64, i64, i32) {
    (MOUSE_DX.swap(0, Ordering::Relaxed), MOUSE_DY.swap(0, Ordering::Relaxed), WHEEL.swap(0, Ordering::Relaxed))
}

const VK_SHIFT: usize = 0x10;
const VK_CONTROL: usize = 0x11;
const VK_MENU: usize = 0x12;

/// The left/right-specific key of a key message (Shift by scan code, Ctrl
/// and Alt by the extended-key flag).
fn specific_vk(wp: WPARAM, lp: LPARAM) -> usize {
    let vk = wp & 0xFF;
    let extended = (lp >> 24) & 1 != 0;
    match vk {
        VK_SHIFT => {
            let sc = ((lp >> 16) & 0xFF) as u32;
            let v = unsafe { MapVirtualKeyW(sc, MAPVK_VSC_TO_VK_EX) } as usize;
            if v == 0xA1 { 0xA1 } else { 0xA0 }
        }
        VK_CONTROL => {
            if extended {
                0xA3
            } else {
                0xA2
            }
        }
        VK_MENU => {
            if extended {
                0xA5
            } else {
                0xA4
            }
        }
        v => v,
    }
}

fn set_key(wp: WPARAM, lp: LPARAM, down: bool) {
    let vk = specific_vk(wp, lp);
    KEYS[vk].store(down, Ordering::Relaxed);
    // The generic modifier is down while either side is.
    let generic = match vk {
        0xA0 | 0xA1 => Some((VK_SHIFT, 0xA0, 0xA1)),
        0xA2 | 0xA3 => Some((VK_CONTROL, 0xA2, 0xA3)),
        0xA4 | 0xA5 => Some((VK_MENU, 0xA4, 0xA5)),
        _ => None,
    };
    if let Some((g, l, r)) = generic {
        KEYS[g].store(KEYS[l].load(Ordering::Relaxed) || KEYS[r].load(Ordering::Relaxed), Ordering::Relaxed);
    }
}

fn clear_input() {
    for k in KEYS.iter() {
        k.store(false, Ordering::Relaxed);
    }
    MOUSE_BUTTONS.store(0, Ordering::Relaxed);
    MOUSE_DX.store(0, Ordering::Relaxed);
    MOUSE_DY.store(0, Ordering::Relaxed);
    WHEEL.store(0, Ordering::Relaxed);
}

/// Confines the (hidden) cursor to the client area.
unsafe fn clip(hwnd: HWND) {
    let mut r = RECT { left: 0, top: 0, right: 0, bottom: 0 };
    unsafe {
        GetClientRect(hwnd, &mut r);
        let mut tl = POINT { x: r.left, y: r.top };
        let mut br = POINT { x: r.right, y: r.bottom };
        ClientToScreen(hwnd, &mut tl);
        ClientToScreen(hwnd, &mut br);
        let screen = RECT { left: tl.x, top: tl.y, right: br.x, bottom: br.y };
        ClipCursor(&screen);
    }
}

unsafe fn capture(hwnd: HWND) {
    if CAPTURED.swap(true, Ordering::SeqCst) {
        return;
    }
    unsafe {
        clip(hwnd);
        ShowCursor(0);
    }
    MOUSE_DX.store(0, Ordering::Relaxed);
    MOUSE_DY.store(0, Ordering::Relaxed);
}

unsafe fn release() {
    if !CAPTURED.swap(false, Ordering::SeqCst) {
        return;
    }
    unsafe {
        ClipCursor(std::ptr::null());
        ShowCursor(1);
    }
    MOUSE_BUTTONS.store(0, Ordering::Relaxed);
}

unsafe fn raw_input(lp: LPARAM) {
    let mut ri: RAWINPUT = unsafe { std::mem::zeroed() };
    let mut size = std::mem::size_of::<RAWINPUT>() as u32;
    let n = unsafe { GetRawInputData(lp as HRAWINPUT, RID_INPUT, &mut ri as *mut RAWINPUT as *mut _, &mut size, std::mem::size_of::<RAWINPUTHEADER>() as u32) };
    if n == u32::MAX || ri.header.dwType != RIM_TYPEMOUSE || !CAPTURED.load(Ordering::Relaxed) {
        return;
    }
    let m = unsafe { ri.data.mouse };
    // (Absolute-position devices such as tablets and remote desktops are not
    // raw counts: MOUSE_MOVE_ABSOLUTE.)
    if m.usFlags & 1 == 0 {
        MOUSE_DX.fetch_add(m.lLastX as i64, Ordering::Relaxed);
        MOUSE_DY.fetch_add(m.lLastY as i64, Ordering::Relaxed);
    }
    let flags = unsafe { m.Anonymous.Anonymous.usButtonFlags } as u32;
    let data = unsafe { m.Anonymous.Anonymous.usButtonData } as i16;
    for b in 0..5 {
        let (down, up) = (1u32 << (2 * b), 2u32 << (2 * b));
        if flags & down != 0 {
            MOUSE_BUTTONS.fetch_or(1 << b, Ordering::Relaxed);
        }
        if flags & up != 0 {
            MOUSE_BUTTONS.fetch_and(!(1 << b), Ordering::Relaxed);
        }
    }
    if flags & RI_MOUSE_WHEEL as u32 != 0 {
        WHEEL.fetch_add(data as i32 / 120, Ordering::Relaxed);
    }
}

unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    match msg {
        WM_CLOSE => {
            CLOSED.store(true, Ordering::SeqCst);
            0
        }
        WM_SIZE => {
            let w = (lp & 0xFFFF) as u32;
            let h = ((lp >> 16) & 0xFFFF) as u32;
            if w > 0 && h > 0 {
                CLIENT_W.store(w, Ordering::Relaxed);
                CLIENT_H.store(h, Ordering::Relaxed);
                RESIZED.store(true, Ordering::SeqCst);
            }
            if CAPTURED.load(Ordering::Relaxed) {
                unsafe { clip(hwnd) };
            }
            0
        }
        WM_MOVE => {
            if CAPTURED.load(Ordering::Relaxed) {
                unsafe { clip(hwnd) };
            }
            0
        }
        WM_INPUT => {
            unsafe {
                raw_input(lp);
                DefWindowProcW(hwnd, msg, wp, lp)
            }
        }
        WM_LBUTTONDOWN | WM_RBUTTONDOWN | WM_MBUTTONDOWN => {
            // The first click captures the mouse; it is not a game input.
            if !CAPTURED.load(Ordering::Relaxed) && crate::input::mouse_capture_enabled() {
                unsafe { capture(hwnd) };
            }
            0
        }
        WM_KEYDOWN | WM_SYSKEYDOWN => {
            let vk = (wp & 0xFF) as u32;
            if vk == RELEASE_KEY.load(Ordering::Relaxed) && CAPTURED.load(Ordering::Relaxed) {
                unsafe { release() };
                return 0;
            }
            set_key(wp, lp, true);
            // Alt+F4 still closes the window.
            if msg == WM_SYSKEYDOWN && vk == 0x73 {
                return unsafe { DefWindowProcW(hwnd, msg, wp, lp) };
            }
            0
        }
        WM_KEYUP | WM_SYSKEYUP => {
            set_key(wp, lp, false);
            0
        }
        WM_ACTIVATE => {
            if wp & 0xFFFF == 0 {
                unsafe { release() };
                clear_input();
            }
            unsafe { DefWindowProcW(hwnd, msg, wp, lp) }
        }
        WM_KILLFOCUS => {
            unsafe { release() };
            clear_input();
            0
        }
        _ => unsafe { DefWindowProcW(hwnd, msg, wp, lp) },
    }
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// Creates the window on a dedicated thread and waits until it exists.
pub fn create(title: &str, width: u32, height: u32) -> Result<&'static Window, String> {
    if let Some(w) = WINDOW.get() {
        return Ok(w);
    }
    let (tx, rx) = std::sync::mpsc::channel::<Result<(isize, isize), String>>();
    let title = title.to_string();
    std::thread::Builder::new()
        .name("ui".into())
        .spawn(move || unsafe {
            crate::log::set_thread_name("ui");
            let hinst = GetModuleHandleW(std::ptr::null());
            let class = wide("tlg_window");
            let wc = WNDCLASSEXW {
                cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
                style: CS_HREDRAW | CS_VREDRAW,
                lpfnWndProc: Some(wndproc),
                cbClsExtra: 0,
                cbWndExtra: 0,
                hInstance: hinst,
                hIcon: std::ptr::null_mut(),
                hCursor: LoadCursorW(std::ptr::null_mut(), IDC_ARROW),
                hbrBackground: std::ptr::null_mut(),
                lpszMenuName: std::ptr::null(),
                lpszClassName: class.as_ptr(),
                hIconSm: std::ptr::null_mut(),
            };
            RegisterClassExW(&wc);
            let style = WS_OVERLAPPEDWINDOW;
            let mut rect = RECT { left: 0, top: 0, right: width as i32, bottom: height as i32 };
            AdjustWindowRect(&mut rect, style, 0);
            let t = wide(&title);
            let hwnd = CreateWindowExW(
                0,
                class.as_ptr(),
                t.as_ptr(),
                style,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                rect.right - rect.left,
                rect.bottom - rect.top,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                hinst,
                std::ptr::null(),
            );
            if hwnd.is_null() {
                let _ = tx.send(Err("CreateWindowExW failed".into()));
                return;
            }
            // Raw mouse input (generic desktop page, mouse usage) for this window.
            let dev = RAWINPUTDEVICE { usUsagePage: 1, usUsage: 2, dwFlags: 0, hwndTarget: hwnd };
            if RegisterRawInputDevices(&dev, 1, std::mem::size_of::<RAWINPUTDEVICE>() as u32) == 0 {
                crate::warn!("raw mouse input unavailable: mouse look disabled");
            }
            CLIENT_W.store(width, Ordering::Relaxed);
            CLIENT_H.store(height, Ordering::Relaxed);
            ShowWindow(hwnd, SW_SHOW);
            let _ = tx.send(Ok((hwnd as isize, hinst as isize)));
            let mut msg: MSG = std::mem::zeroed();
            while GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) > 0 {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        })
        .map_err(|e| format!("ui thread: {e}"))?;
    let (hwnd, hinstance) = rx.recv().map_err(|e| format!("ui thread died: {e}"))??;
    WINDOW.set(Window { hwnd, hinstance }).ok();
    Ok(WINDOW.get().unwrap())
}
