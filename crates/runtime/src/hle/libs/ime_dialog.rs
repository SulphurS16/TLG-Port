//! System IME dialog (libSceImeDialog): text entry such as the character
//! name. The console's on-screen keyboard is replaced by a native Windows text
//! box; automated runs supply the text with `TLG_IME_TEXT` or the `ime <text>`
//! control command. The entered text is written to the title's UTF-16 buffer
//! when it next polls the dialog status.

use crate::hle::Registry;
use crate::{reg, sysv};
use parking_lot::Mutex;

const STATUS_NONE: i32 = 0;
const STATUS_RUNNING: i32 = 1;
const STATUS_FINISHED: i32 = 2;
const END_OK: i32 = 0;
const END_CANCELED: i32 = 1;
const END_ABORTED: i32 = 2;

struct Dialog {
    /// Guest char16 buffer receiving the text (SceImeDialogParam.inputTextBuffer).
    buf: u64,
    max_len: u32,
    status: i32,
    end: i32,
    text: Option<Vec<u16>>,
    delivered: bool,
    /// Native text box window (0 = none).
    hwnd: isize,
}

static DIALOG: Mutex<Option<Dialog>> = Mutex::new(None);
/// Text supplied through the control channel (`ime <text>`).
pub static CONTROL_TEXT: Mutex<Option<String>> = Mutex::new(None);

unsafe fn read_u16z(p: u64, max: usize) -> Vec<u16> {
    let mut v = Vec::new();
    if p == 0 {
        return v;
    }
    for i in 0..max {
        let c = unsafe { *((p + 2 * i as u64) as *const u16) };
        if c == 0 {
            break;
        }
        v.push(c);
    }
    v
}

fn finish(d: &mut Dialog, text: Option<Vec<u16>>, end: i32) {
    if let Some(mut t) = text {
        t.truncate(d.max_len as usize);
        d.text = Some(t);
    }
    d.end = end;
    d.status = STATUS_FINISHED;
}

sysv! {
    fn sceImeDialogInit(param: *const u8, extended: *const u8) -> i32 {
        if param.is_null() {
            return crate::hle::kernel::sce(22); // EINVAL
        }
        let max_len = *(param.add(0x24) as *const u32);
        let buf = *(param.add(0x28) as *const u64);
        let placeholder = *(param.add(0x40) as *const u64);
        let title = *(param.add(0x48) as *const u64);
        let s = |v: Vec<u16>| String::from_utf16_lossy(&v);
        let title_s = s(read_u16z(title, 256));
        let initial = read_u16z(buf, max_len.min(1024) as usize);
        crate::info!(
            "ime dialog: title '{}', placeholder '{}', initial '{}', max {} chars",
            title_s,
            s(read_u16z(placeholder, 256)),
            s(initial.clone()),
            max_len
        );
        let mut d = Dialog { buf, max_len, status: STATUS_RUNNING, end: END_CANCELED, text: None, delivered: false, hwnd: 0 };
        if let Ok(t) = std::env::var("TLG_IME_TEXT") {
            crate::info!("ime dialog: text supplied by TLG_IME_TEXT");
            finish(&mut d, Some(t.encode_utf16().collect()), END_OK);
        } else {
            let init = String::from_utf16_lossy(&initial);
            std::thread::Builder::new()
                .name("ime".into())
                .spawn(move || native::prompt(&title_s, &init, max_len))
                .ok();
        }
        *DIALOG.lock() = Some(d);
        0
    }
    fn sceImeDialogGetStatus() -> i32 {
        let mut g = DIALOG.lock();
        let Some(d) = g.as_mut() else { return STATUS_NONE };
        if d.status == STATUS_RUNNING {
            if let Some(t) = CONTROL_TEXT.lock().take() {
                crate::info!("ime dialog: text supplied by control command");
                native::close(d.hwnd);
                finish(d, Some(t.encode_utf16().collect()), END_OK);
            }
        }
        if d.status == STATUS_FINISHED && !d.delivered {
            d.delivered = true;
            if let (Some(t), true) = (&d.text, d.end == END_OK && d.buf != 0) {
                for (i, c) in t.iter().chain(std::iter::once(&0)).enumerate() {
                    *((d.buf + 2 * i as u64) as *mut u16) = *c;
                }
                crate::info!("ime dialog: entered '{}'", String::from_utf16_lossy(t));
            }
        }
        d.status
    }
    fn sceImeDialogGetResult(result: *mut i32) -> i32 {
        let g = DIALOG.lock();
        let end = g.as_ref().map(|d| d.end).unwrap_or(END_CANCELED);
        if !result.is_null() {
            std::ptr::write_bytes(result as *mut u8, 0, 16);
            *result = end;
        }
        0
    }
    fn sceImeDialogTerm() -> i32 {
        if let Some(d) = DIALOG.lock().take() {
            native::close(d.hwnd);
        }
        0
    }
    fn sceImeDialogAbort() -> i32 {
        if let Some(d) = DIALOG.lock().as_mut() {
            native::close(d.hwnd);
            finish(d, None, END_ABORTED);
        }
        0
    }
}

/// Called by the native text box when the user confirms or cancels.
fn native_done(text: Option<String>) {
    if let Some(d) = DIALOG.lock().as_mut() {
        if d.status == STATUS_RUNNING {
            match text {
                Some(t) => finish(d, Some(t.encode_utf16().collect()), END_OK),
                None => finish(d, None, END_CANCELED),
            }
        }
    }
}

fn set_hwnd(hwnd: isize) {
    if let Some(d) = DIALOG.lock().as_mut() {
        d.hwnd = hwnd;
    }
}

mod native {
    use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows_sys::Win32::UI::WindowsAndMessaging::*;

    const ID_EDIT: isize = 100;
    const ID_OK: isize = 1;
    const ID_CANCEL: isize = 2;
    const EM_LIMITTEXT: u32 = 0x00C5;
    const ES_AUTOHSCROLL: u32 = 0x0080;

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    pub fn close(hwnd: isize) {
        if hwnd != 0 {
            unsafe { PostMessageW(hwnd as HWND, WM_CLOSE, 0, 0) };
        }
    }

    unsafe fn text_of(hwnd: HWND) -> String {
        unsafe {
            let edit = GetDlgItem(hwnd, ID_EDIT as i32);
            let mut buf = [0u16; 512];
            let n = GetWindowTextW(edit, buf.as_mut_ptr(), buf.len() as i32).max(0) as usize;
            String::from_utf16_lossy(&buf[..n])
        }
    }

    unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
        unsafe {
            match msg {
                WM_COMMAND => {
                    let id = (wp & 0xFFFF) as isize;
                    if id == ID_OK {
                        super::native_done(Some(text_of(hwnd)));
                        DestroyWindow(hwnd);
                        return 0;
                    }
                    if id == ID_CANCEL {
                        super::native_done(None);
                        DestroyWindow(hwnd);
                        return 0;
                    }
                    DefWindowProcW(hwnd, msg, wp, lp)
                }
                WM_CLOSE => {
                    super::native_done(None);
                    DestroyWindow(hwnd);
                    0
                }
                WM_DESTROY => {
                    PostQuitMessage(0);
                    0
                }
                _ => DefWindowProcW(hwnd, msg, wp, lp),
            }
        }
    }

    /// Shows a small top-most text box and runs its message loop until it closes.
    pub fn prompt(title: &str, initial: &str, max_len: u32) {
        crate::log::set_thread_name("ime");
        unsafe {
            let hinst = GetModuleHandleW(std::ptr::null());
            let class = wide("tlg_ime");
            let wc = WNDCLASSEXW {
                cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
                style: 0,
                lpfnWndProc: Some(wndproc),
                cbClsExtra: 0,
                cbWndExtra: 0,
                hInstance: hinst,
                hIcon: std::ptr::null_mut(),
                hCursor: LoadCursorW(std::ptr::null_mut(), IDC_ARROW),
                hbrBackground: 16 as _, // COLOR_BTNFACE + 1
                lpszMenuName: std::ptr::null(),
                lpszClassName: class.as_ptr(),
                hIconSm: std::ptr::null_mut(),
            };
            RegisterClassExW(&wc);
            let caption = wide(if title.is_empty() { "Text entry" } else { title });
            let hwnd = CreateWindowExW(
                WS_EX_TOPMOST | WS_EX_DLGMODALFRAME,
                class.as_ptr(),
                caption.as_ptr(),
                WS_CAPTION | WS_SYSMENU | WS_VISIBLE,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                440,
                130,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                hinst,
                std::ptr::null(),
            );
            if hwnd.is_null() {
                crate::error!("ime dialog: could not create the text box window");
                super::native_done(None);
                return;
            }
            super::set_hwnd(hwnd as isize);
            let edit_class = wide("EDIT");
            let button = wide("BUTTON");
            let init = wide(initial);
            let edit = CreateWindowExW(
                WS_EX_CLIENTEDGE,
                edit_class.as_ptr(),
                init.as_ptr(),
                WS_CHILD | WS_VISIBLE | WS_TABSTOP | ES_AUTOHSCROLL,
                12,
                12,
                400,
                24,
                hwnd,
                ID_EDIT as _,
                hinst,
                std::ptr::null(),
            );
            SendMessageW(edit, EM_LIMITTEXT, max_len.max(1) as usize, 0);
            let ok = wide("OK");
            let cancel = wide("Cancel");
            CreateWindowExW(0, button.as_ptr(), ok.as_ptr(), WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_DEFPUSHBUTTON as u32, 236, 50, 84, 28, hwnd, ID_OK as _, hinst, std::ptr::null());
            CreateWindowExW(0, button.as_ptr(), cancel.as_ptr(), WS_CHILD | WS_VISIBLE | WS_TABSTOP, 328, 50, 84, 28, hwnd, ID_CANCEL as _, hinst, std::ptr::null());
            SetForegroundWindow(hwnd);
            windows_sys::Win32::UI::Input::KeyboardAndMouse::SetFocus(edit);
            let mut msg: MSG = std::mem::zeroed();
            while GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) > 0 {
                // Enter / Escape in the edit box act on the buttons.
                if msg.message == WM_KEYDOWN && (msg.wParam == 0x0D || msg.wParam == 0x1B) {
                    let id = if msg.wParam == 0x0D { ID_OK } else { ID_CANCEL };
                    SendMessageW(hwnd, WM_COMMAND, id as usize, 0);
                    continue;
                }
                if IsDialogMessageW(hwnd, &msg) == 0 {
                    TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
            }
        }
    }
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceImeDialog", [sceImeDialogInit, sceImeDialogGetStatus, sceImeDialogGetResult, sceImeDialogTerm, sceImeDialogAbort]);
}
