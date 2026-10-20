//! Output window (Win32) running its own message loop thread.
//!
//! Display modes (settings.txt, `config`): windowed (resizable), borderless
//! (the monitor's whole area, no mode change) and fullscreen (exclusive:
//! the monitor switches to the chosen resolution and refresh rate while the
//! window is active; the desktop mode comes back on Alt+Tab and at exit).
//! Alt+Enter or F11 switches between windowed and full screen (borderless,
//! or exclusive when that is the configured mode); the choice is kept in
//! settings.txt. The process is per-monitor DPI aware (main.rs): sizes are
//! physical pixels, and a window moved to a monitor of another scale takes
//! the size Windows suggests. Display changes (resolution, monitors added or
//! removed) refit full-screen windows to their monitor.
//!
//! Keyboard state comes from key messages (left/right modifiers kept apart).
//! The mouse is read as raw input (relative counts, no pointer acceleration)
//! while it is captured: a click into the window captures it (cursor hidden
//! and confined), losing focus or the release key (`input.txt`) frees it.

use crate::config::DisplayMode;
use parking_lot::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, AtomicU32, Ordering};
use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, RECT, WPARAM};
use windows_sys::Win32::Graphics::Gdi::*;
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::HiDpi::{AdjustWindowRectExForDpi, GetDpiForWindow};
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

/// The window's display state (the ui thread changes it).
struct Display {
    mode: DisplayMode,
    /// The full-screen mode Alt+Enter switches to.
    full: DisplayMode,
    /// The windowed placement to return to from full screen.
    windowed: Option<RECT>,
    /// Exclusive mode: (width, height, refresh Hz; 0 = current).
    exclusive: (u32, u32, u32),
}

static DISPLAY: Mutex<Display> = Mutex::new(Display { mode: DisplayMode::Windowed, full: DisplayMode::Borderless, windowed: None, exclusive: (0, 0, 0) });

/// Posted to the window: switch the display mode (WPARAM 0 toggle, 1
/// windowed, 2 borderless, 3 fullscreen).
const WM_APP_DISPLAY: u32 = WM_APP + 10;

/// Switches the display mode from another thread (control `display`):
/// None toggles as Alt+Enter does.
pub fn request_mode(m: Option<DisplayMode>) -> Result<(), String> {
    let w = WINDOW.get().ok_or("no window (headless run)")?;
    let code = match m {
        None => 0,
        Some(DisplayMode::Windowed) => 1,
        Some(DisplayMode::Borderless) => 2,
        Some(DisplayMode::Fullscreen) => 3,
    };
    unsafe { PostMessageW(w.hwnd as HWND, WM_APP_DISPLAY, code, 0) };
    Ok(())
}

/// The monitor whose display mode was switched (device name, NUL-terminated).
static SWITCHED: Mutex<Option<Vec<u16>>> = Mutex::new(None);

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

/// Until when (process ms) injected motion counts as captured mouse input.
static INJECTED_UNTIL: AtomicI64 = AtomicI64::new(0);

/// Debug (control `mouse <dx> <dy> [ms]`): raw counts as if the captured
/// mouse had moved, spread evenly over `ms` (5 ms steps).
pub fn inject_mouse(dx: i64, dy: i64, ms: u64) {
    let steps = (ms / 5).max(1) as i64;
    INJECTED_UNTIL.store(crate::log::elapsed_ms() as i64 + ms as i64 + 500, Ordering::Relaxed);
    let _ = std::thread::Builder::new().name("mouse-inject".into()).spawn(move || {
        let (mut sx, mut sy) = (0i64, 0i64);
        for i in 1..=steps {
            let (tx, ty) = (dx * i / steps, dy * i / steps);
            if !crate::input::mouse_motion(tx - sx, ty - sy) {
                MOUSE_DX.fetch_add(tx - sx, Ordering::Relaxed);
                MOUSE_DY.fetch_add(ty - sy, Ordering::Relaxed);
            }
            (sx, sy) = (tx, ty);
            if i < steps {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
    });
}

/// Mouse input is live: captured, or injected motion pending.
pub fn mouse_live() -> bool {
    CAPTURED.load(Ordering::Relaxed) || crate::log::elapsed_ms() as i64 <= INJECTED_UNTIL.load(Ordering::Relaxed)
}

const VK_SHIFT: usize = 0x10;
const VK_CONTROL: usize = 0x11;
const VK_MENU: usize = 0x12;
const VK_RETURN: u32 = 0x0D;
const VK_F4: u32 = 0x73;
const VK_F11: u32 = 0x7A;

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
    // (Into the game camera at once - `input::mouse_motion` - or kept for
    // the pad read.)
    if m.usFlags & 1 == 0 && !crate::input::mouse_motion(m.lLastX as i64, m.lLastY as i64) {
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

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// A monitor: its handle, whole area, work area (without the taskbar) and
/// device name (NUL-terminated).
struct Monitor {
    handle: HMONITOR,
    rect: RECT,
    work: RECT,
    device: Vec<u16>,
}

unsafe fn monitor_info(h: HMONITOR) -> Option<Monitor> {
    let mut mi: MONITORINFOEXW = unsafe { std::mem::zeroed() };
    mi.monitorInfo.cbSize = std::mem::size_of::<MONITORINFOEXW>() as u32;
    if unsafe { GetMonitorInfoW(h, &mut mi as *mut MONITORINFOEXW as *mut MONITORINFO) } == 0 {
        return None;
    }
    let n = mi.szDevice.iter().position(|&c| c == 0).unwrap_or(mi.szDevice.len());
    let mut device = mi.szDevice[..n].to_vec();
    device.push(0);
    Some(Monitor { handle: h, rect: mi.monitorInfo.rcMonitor, work: mi.monitorInfo.rcWork, device })
}

unsafe extern "system" fn enum_monitor(h: HMONITOR, _dc: HDC, _r: *mut RECT, data: LPARAM) -> windows_sys::core::BOOL {
    let v = unsafe { &mut *(data as *mut Vec<HMONITOR>) };
    v.push(h);
    1
}

/// The monitor named in settings (a device name such as \\.\DISPLAY2), or
/// the primary one.
unsafe fn find_monitor(name: &str) -> Option<Monitor> {
    let mut hs: Vec<HMONITOR> = Vec::new();
    unsafe { EnumDisplayMonitors(std::ptr::null_mut(), std::ptr::null(), Some(enum_monitor), &mut hs as *mut Vec<HMONITOR> as LPARAM) };
    let all: Vec<Monitor> = hs.into_iter().filter_map(|h| unsafe { monitor_info(h) }).collect();
    let dev = |m: &Monitor| String::from_utf16_lossy(&m.device[..m.device.len() - 1]);
    if !name.eq_ignore_ascii_case("primary") {
        if let Some(i) = all.iter().position(|m| dev(m).eq_ignore_ascii_case(name)) {
            return all.into_iter().nth(i);
        }
        crate::warn!("display: monitor '{name}' not found; using the primary monitor");
    }
    let primary = unsafe { MonitorFromWindow(std::ptr::null_mut(), MONITOR_DEFAULTTOPRIMARY) };
    unsafe { monitor_info(primary) }
}

unsafe fn window_monitor(hwnd: HWND) -> Option<Monitor> {
    unsafe { monitor_info(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)) }
}

const WINDOWED_STYLE: WINDOW_STYLE = WS_OVERLAPPEDWINDOW;
const FULL_STYLE: WINDOW_STYLE = WS_POPUP;

/// The window rect for a client size at the window's DPI.
unsafe fn frame_for_client(hwnd: HWND, w: i32, h: i32) -> (i32, i32) {
    let mut r = RECT { left: 0, top: 0, right: w, bottom: h };
    let dpi = unsafe { GetDpiForWindow(hwnd) }.max(96);
    unsafe { AdjustWindowRectExForDpi(&mut r, WINDOWED_STYLE, 0, 0, dpi) };
    (r.right - r.left, r.bottom - r.top)
}

/// The default windowed client size on a monitor: settings, else the
/// largest 16:9 area within 3/4 of its work area.
fn windowed_client(m: &Monitor) -> (i32, i32) {
    let work = ((m.work.right - m.work.left).max(0) as u32, (m.work.bottom - m.work.top).max(0) as u32);
    let (w, h) = tlg_config::settings::window_size(crate::config::settings(), work);
    (w as i32, h as i32)
}

/// The size the game renders at (fixed when its executable loads): the
/// setting `render_resolution` (development override TLG_RENDER_SIZE), or
/// 16:9 within the display - the exclusive mode, the monitor (borderless)
/// or the window's client area (windowed).
pub fn render_size() -> (u32, u32) {
    use tlg_config::settings::{render_size_for, render_size_ok};
    let dev = std::env::var("TLG_RENDER_SIZE").ok().and_then(|v| {
        let (w, h) = v.split_once('x')?;
        Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
    });
    if let Some(r) = dev.filter(|&r| render_size_ok(r)) {
        return r;
    }
    let s = crate::config::settings();
    if let Some(r) = s.render_resolution {
        return r;
    }
    let Some(m) = (unsafe { find_monitor(&s.monitor) }) else { return (1920, 1080) };
    let full = ((m.rect.right - m.rect.left).max(0) as u32, (m.rect.bottom - m.rect.top).max(0) as u32);
    let (w, h) = match s.display_mode {
        DisplayMode::Fullscreen => s.resolution.unwrap_or(full),
        DisplayMode::Borderless => full,
        DisplayMode::Windowed => {
            let (w, h) = windowed_client(&m);
            (w as u32, h as u32)
        }
    };
    render_size_for(w, h)
}

/// Switches the monitor to the exclusive mode (only what differs from its
/// current mode); false if Windows refused.
unsafe fn switch_display(m: &Monitor, mode: (u32, u32, u32)) -> bool {
    let mut dm: DEVMODEW = unsafe { std::mem::zeroed() };
    dm.dmSize = std::mem::size_of::<DEVMODEW>() as u16;
    if unsafe { EnumDisplaySettingsW(m.device.as_ptr(), ENUM_CURRENT_SETTINGS, &mut dm) } == 0 {
        return false;
    }
    let (w, h, hz) = mode;
    let want_w = if w > 0 { w } else { dm.dmPelsWidth };
    let want_h = if h > 0 { h } else { dm.dmPelsHeight };
    let want_hz = if hz > 0 { hz } else { dm.dmDisplayFrequency };
    if (want_w, want_h, want_hz) == (dm.dmPelsWidth, dm.dmPelsHeight, dm.dmDisplayFrequency) {
        return true;
    }
    dm.dmPelsWidth = want_w;
    dm.dmPelsHeight = want_h;
    dm.dmDisplayFrequency = want_hz;
    dm.dmFields = DM_PELSWIDTH | DM_PELSHEIGHT | DM_DISPLAYFREQUENCY;
    let r = unsafe { ChangeDisplaySettingsExW(m.device.as_ptr(), &dm, std::ptr::null_mut(), CDS_FULLSCREEN, std::ptr::null()) };
    if r == DISP_CHANGE_SUCCESSFUL {
        crate::info!("display: {} switched to {want_w}x{want_h} @ {want_hz} Hz", String::from_utf16_lossy(&m.device[..m.device.len() - 1]));
        *SWITCHED.lock() = Some(m.device.clone());
        true
    } else {
        crate::warn!("display: switching to {want_w}x{want_h} @ {want_hz} Hz failed ({r}); staying at the desktop mode");
        false
    }
}

/// Returns a monitor switched for exclusive fullscreen to its desktop mode
/// (any thread; also at exit and after a crash). True if one was switched.
pub fn restore_display() -> bool {
    let dev = SWITCHED.lock().take();
    let Some(dev) = dev else { return false };
    unsafe { ChangeDisplaySettingsExW(dev.as_ptr(), std::ptr::null(), std::ptr::null_mut(), 0, std::ptr::null()) };
    crate::info!("display: desktop mode restored");
    true
}

/// Puts the window into `mode` (ui thread). (No lock is held across calls
/// that send messages: the window procedure takes the same locks.)
unsafe fn apply_mode(hwnd: HWND, mode: DisplayMode) {
    let (was, exclusive) = {
        let d = DISPLAY.lock();
        (d.mode, d.exclusive)
    };
    unsafe {
        if was == DisplayMode::Windowed && mode != DisplayMode::Windowed {
            let mut r = RECT { left: 0, top: 0, right: 0, bottom: 0 };
            GetWindowRect(hwnd, &mut r);
            DISPLAY.lock().windowed = Some(r);
        }
        if was == DisplayMode::Fullscreen && mode != DisplayMode::Fullscreen {
            restore_display();
        }
        let windowed = {
            let mut d = DISPLAY.lock();
            d.mode = mode;
            d.windowed
        };
        match mode {
            DisplayMode::Windowed => {
                SetWindowLongPtrW(hwnd, GWL_STYLE, (WINDOWED_STYLE | WS_VISIBLE) as isize);
                let r = match windowed {
                    Some(r) => r,
                    None => {
                        let m = window_monitor(hwnd).unwrap_or_else(|| find_monitor("primary").expect("a monitor"));
                        let (cw, ch) = windowed_client(&m);
                        let (fw, fh) = frame_for_client(hwnd, cw, ch);
                        let x = m.work.left + ((m.work.right - m.work.left) - fw).max(0) / 2;
                        let y = m.work.top + ((m.work.bottom - m.work.top) - fh).max(0) / 2;
                        RECT { left: x, top: y, right: x + fw, bottom: y + fh }
                    }
                };
                SetWindowPos(hwnd, HWND_NOTOPMOST, r.left, r.top, r.right - r.left, r.bottom - r.top, SWP_FRAMECHANGED | SWP_SHOWWINDOW);
            }
            DisplayMode::Borderless | DisplayMode::Fullscreen => {
                let Some(mut m) = window_monitor(hwnd) else { return };
                if mode == DisplayMode::Fullscreen && switch_display(&m, exclusive) {
                    // (The monitor's area changes with its mode.)
                    m = monitor_info(m.handle).unwrap_or(m);
                }
                SetWindowLongPtrW(hwnd, GWL_STYLE, (FULL_STYLE | WS_VISIBLE) as isize);
                let r = m.rect;
                SetWindowPos(hwnd, HWND_TOP, r.left, r.top, r.right - r.left, r.bottom - r.top, SWP_FRAMECHANGED | SWP_SHOWWINDOW);
            }
        }
    }
    crate::info!("display: {mode:?}");
    RESIZED.store(true, Ordering::SeqCst);
}

fn mode_name(m: DisplayMode) -> &'static str {
    match m {
        DisplayMode::Windowed => "windowed",
        DisplayMode::Borderless => "borderless",
        DisplayMode::Fullscreen => "fullscreen",
    }
}

/// Alt+Enter / F11: windowed <-> the full-screen mode.
unsafe fn toggle_full(hwnd: HWND) {
    let (cur, full) = {
        let d = DISPLAY.lock();
        (d.mode, d.full)
    };
    let next = if cur == DisplayMode::Windowed { full } else { DisplayMode::Windowed };
    unsafe { apply_mode(hwnd, next) };
    crate::config::persist("display_mode", mode_name(next));
}

/// Keeps a full-screen window covering its monitor (after display changes).
unsafe fn refit(hwnd: HWND) {
    let mode = DISPLAY.lock().mode;
    if mode == DisplayMode::Windowed {
        return;
    }
    unsafe {
        if let Some(m) = window_monitor(hwnd) {
            let r = m.rect;
            SetWindowPos(hwnd, HWND_TOP, r.left, r.top, r.right - r.left, r.bottom - r.top, SWP_NOZORDER | SWP_NOACTIVATE);
        }
    }
    RESIZED.store(true, Ordering::SeqCst);
}

/// Records the windowed client size in settings.txt (after a resize by the user).
unsafe fn remember_size(hwnd: HWND) {
    if DISPLAY.lock().mode != DisplayMode::Windowed {
        return;
    }
    let mut r = RECT { left: 0, top: 0, right: 0, bottom: 0 };
    unsafe { GetClientRect(hwnd, &mut r) };
    let (w, h) = (r.right - r.left, r.bottom - r.top);
    if w >= 320 && h >= 180 {
        crate::config::persist("window_width", &w.to_string());
        crate::config::persist("window_height", &h.to_string());
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
        WM_EXITSIZEMOVE => {
            unsafe { remember_size(hwnd) };
            0
        }
        WM_GETMINMAXINFO => {
            let mi = unsafe { &mut *(lp as *mut MINMAXINFO) };
            mi.ptMinTrackSize = POINT { x: 320, y: 200 };
            0
        }
        WM_DPICHANGED => {
            // Windows' suggestion keeps the window's size on the new scale.
            if DISPLAY.lock().mode == DisplayMode::Windowed {
                let r = unsafe { &*(lp as *const RECT) };
                unsafe { SetWindowPos(hwnd, std::ptr::null_mut(), r.left, r.top, r.right - r.left, r.bottom - r.top, SWP_NOZORDER | SWP_NOACTIVATE) };
            } else {
                unsafe { refit(hwnd) };
            }
            0
        }
        WM_DISPLAYCHANGE => {
            unsafe { refit(hwnd) };
            unsafe { DefWindowProcW(hwnd, msg, wp, lp) }
        }
        WM_ACTIVATEAPP => {
            // Exclusive fullscreen: the desktop mode while switched away.
            let active = wp != 0;
            if DISPLAY.lock().mode == DisplayMode::Fullscreen {
                if active {
                    unsafe {
                        ShowWindow(hwnd, SW_RESTORE);
                        apply_mode(hwnd, DisplayMode::Fullscreen);
                    }
                } else if restore_display() {
                    unsafe { ShowWindow(hwnd, SW_MINIMIZE) };
                }
            }
            unsafe { DefWindowProcW(hwnd, msg, wp, lp) }
        }
        WM_INPUT => unsafe {
            raw_input(lp);
            DefWindowProcW(hwnd, msg, wp, lp)
        },
        WM_LBUTTONDOWN | WM_RBUTTONDOWN | WM_MBUTTONDOWN => {
            // The first click captures the mouse; it is not a game input.
            if !CAPTURED.load(Ordering::Relaxed) && crate::input::mouse_capture_enabled() {
                unsafe { capture(hwnd) };
            }
            0
        }
        WM_KEYDOWN | WM_SYSKEYDOWN => {
            let vk = (wp & 0xFF) as u32;
            let alt = (lp >> 29) & 1 != 0;
            let repeat = (lp >> 30) & 1 != 0;
            if vk == RELEASE_KEY.load(Ordering::Relaxed) && CAPTURED.load(Ordering::Relaxed) {
                unsafe { release() };
                return 0;
            }
            // Alt+Enter and F11 switch the display mode (not game input).
            if (vk == VK_RETURN && alt) || vk == VK_F11 {
                if !repeat {
                    unsafe { toggle_full(hwnd) };
                }
                return 0;
            }
            set_key(wp, lp, true);
            // Alt+F4 still closes the window.
            if msg == WM_SYSKEYDOWN && vk == VK_F4 {
                return unsafe { DefWindowProcW(hwnd, msg, wp, lp) };
            }
            0
        }
        WM_KEYUP | WM_SYSKEYUP => {
            set_key(wp, lp, false);
            0
        }
        // (No menu beep for Alt+key combinations.)
        WM_SYSCHAR => 0,
        WM_APP_DISPLAY => {
            match wp {
                0 => unsafe { toggle_full(hwnd) },
                1 => unsafe { apply_mode(hwnd, DisplayMode::Windowed) },
                2 => unsafe { apply_mode(hwnd, DisplayMode::Borderless) },
                _ => unsafe { apply_mode(hwnd, DisplayMode::Fullscreen) },
            }
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

/// Creates the window on a dedicated thread (in the display mode of the
/// settings) and waits until it exists.
pub fn create(title: &str) -> Result<&'static Window, String> {
    if let Some(w) = WINDOW.get() {
        return Ok(w);
    }
    let (tx, rx) = std::sync::mpsc::channel::<Result<(isize, isize), String>>();
    let title = title.to_string();
    std::thread::Builder::new()
        .name("ui".into())
        .spawn(move || unsafe {
            crate::log::set_thread_name("ui");
            // (Raw mouse input goes from here into the game camera: handled
            // at once even while the GPU threads, above normal, keep cores
            // busy. The thread waits for messages otherwise.)
            {
                use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_HIGHEST};
                SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
            }
            let s = crate::config::settings();
            {
                let mut d = DISPLAY.lock();
                d.full = if s.display_mode == DisplayMode::Fullscreen { DisplayMode::Fullscreen } else { DisplayMode::Borderless };
                d.exclusive = (s.resolution.map_or(0, |r| r.0), s.resolution.map_or(0, |r| r.1), s.refresh_rate);
            }
            let hinst = GetModuleHandleW(std::ptr::null());
            let class = wide("tlg_window");
            let wc = WNDCLASSEXW {
                cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
                style: CS_HREDRAW | CS_VREDRAW,
                lpfnWndProc: Some(wndproc),
                cbClsExtra: 0,
                cbWndExtra: 0,
                hInstance: hinst,
                // (Resource 1, when the executable has an icon.)
                hIcon: LoadIconW(hinst, 1 as _),
                hCursor: LoadCursorW(std::ptr::null_mut(), IDC_ARROW),
                hbrBackground: GetStockObject(BLACK_BRUSH) as HBRUSH,
                lpszMenuName: std::ptr::null(),
                lpszClassName: class.as_ptr(),
                hIconSm: std::ptr::null_mut(),
            };
            RegisterClassExW(&wc);
            let Some(m) = find_monitor(&s.monitor) else {
                let _ = tx.send(Err("no monitor found".into()));
                return;
            };
            let (cw, ch) = windowed_client(&m);
            let t = wide(&title);
            // Created hidden at the monitor's work area, sized for its DPI below.
            let hwnd = CreateWindowExW(
                0,
                class.as_ptr(),
                t.as_ptr(),
                WINDOWED_STYLE,
                m.work.left,
                m.work.top,
                cw,
                ch,
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
            let (fw, fh) = frame_for_client(hwnd, cw, ch);
            let x = m.work.left + ((m.work.right - m.work.left) - fw).max(0) / 2;
            let y = m.work.top + ((m.work.bottom - m.work.top) - fh).max(0) / 2;
            SetWindowPos(hwnd, std::ptr::null_mut(), x, y, fw, fh, SWP_NOZORDER | SWP_NOACTIVATE);
            DISPLAY.lock().windowed = Some(RECT { left: x, top: y, right: x + fw, bottom: y + fh });
            crate::info!(
                "display: monitor {} ({}x{}, DPI {}), windowed client {cw}x{ch}",
                String::from_utf16_lossy(&m.device[..m.device.len() - 1]),
                m.rect.right - m.rect.left,
                m.rect.bottom - m.rect.top,
                GetDpiForWindow(hwnd)
            );
            apply_mode(hwnd, s.display_mode);
            let mut r = RECT { left: 0, top: 0, right: 0, bottom: 0 };
            GetClientRect(hwnd, &mut r);
            CLIENT_W.store((r.right - r.left).max(1) as u32, Ordering::Relaxed);
            CLIENT_H.store((r.bottom - r.top).max(1) as u32, Ordering::Relaxed);
            ShowWindow(hwnd, SW_SHOW);
            SetForegroundWindow(hwnd);
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
