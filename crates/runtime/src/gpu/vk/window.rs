//! Output window (Win32) running its own message loop thread.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
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
/// Virtual-key states (true = down).
pub static KEYS: [AtomicBool; 256] = [const { AtomicBool::new(false) }; 256];

pub fn get() -> Option<&'static Window> {
    WINDOW.get()
}

pub fn key_down(vk: u8) -> bool {
    KEYS[vk as usize].load(Ordering::Relaxed)
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
            0
        }
        WM_KEYDOWN | WM_SYSKEYDOWN => {
            KEYS[(wp & 0xFF) as usize].store(true, Ordering::Relaxed);
            0
        }
        WM_KEYUP | WM_SYSKEYUP => {
            KEYS[(wp & 0xFF) as usize].store(false, Ordering::Relaxed);
            0
        }
        WM_KILLFOCUS => {
            for k in KEYS.iter() {
                k.store(false, Ordering::Relaxed);
            }
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
            let mut rect = windows_sys::Win32::Foundation::RECT { left: 0, top: 0, right: width as i32, bottom: height as i32 };
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
