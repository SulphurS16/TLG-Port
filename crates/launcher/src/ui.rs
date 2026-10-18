//! Small Win32 helpers: controls laid out in DPI-independent units, text,
//! combo boxes, check boxes, list views, file and folder pickers.

use std::path::PathBuf;
use windows_sys::Win32::Foundation::{HWND, LPARAM, WPARAM};
use windows_sys::Win32::Graphics::Gdi::{CreateFontIndirectW, HFONT, LOGFONTW};
use windows_sys::Win32::UI::Controls::*;
use windows_sys::Win32::UI::HiDpi::SystemParametersInfoForDpi;
use windows_sys::Win32::UI::WindowsAndMessaging::*;

pub fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

pub fn from_wide(b: &[u16]) -> String {
    let n = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf16_lossy(&b[..n])
}

/// The layout scale: device pixels per 1/96 inch.
#[derive(Clone, Copy)]
pub struct Scale(pub u32);

impl Scale {
    pub fn px(self, v: i32) -> i32 {
        (v as i64 * self.0 as i64 / 96) as i32
    }
}

/// The message font at a DPI (normal, and bold when `bold`).
pub unsafe fn message_font(dpi: u32, bold: bool, size_pct: i32) -> HFONT {
    let mut ncm: NONCLIENTMETRICSW = unsafe { std::mem::zeroed() };
    ncm.cbSize = std::mem::size_of::<NONCLIENTMETRICSW>() as u32;
    unsafe { SystemParametersInfoForDpi(SPI_GETNONCLIENTMETRICS, ncm.cbSize, &mut ncm as *mut _ as *mut _, 0, dpi) };
    let mut lf: LOGFONTW = ncm.lfMessageFont;
    lf.lfHeight = lf.lfHeight * size_pct / 100;
    if bold {
        lf.lfWeight = 600;
    }
    unsafe { CreateFontIndirectW(&lf) }
}

pub struct Builder {
    pub parent: HWND,
    pub scale: Scale,
    pub font: HFONT,
}

impl Builder {
    /// A child control at (x, y, w, h) in 1/96-inch units.
    pub unsafe fn add(&self, class: &str, text: &str, style: u32, ex: u32, r: (i32, i32, i32, i32), id: i32) -> HWND {
        let s = self.scale;
        let c = wide(class);
        let t = wide(text);
        let h = unsafe {
            CreateWindowExW(
                ex,
                c.as_ptr(),
                t.as_ptr(),
                WS_CHILD | WS_VISIBLE | style,
                s.px(r.0),
                s.px(r.1),
                s.px(r.2),
                s.px(r.3),
                self.parent,
                id as isize as _,
                std::ptr::null_mut(),
                std::ptr::null(),
            )
        };
        unsafe { SendMessageW(h, WM_SETFONT, self.font as WPARAM, 1) };
        h
    }

    pub unsafe fn label(&self, text: &str, r: (i32, i32, i32, i32), id: i32) -> HWND {
        unsafe { self.add("STATIC", text, 0 /* SS_LEFT */ | 0x80 /* SS_NOPREFIX */, 0, r, id) }
    }

    pub unsafe fn button(&self, text: &str, r: (i32, i32, i32, i32), id: i32) -> HWND {
        unsafe { self.add("BUTTON", text, WS_TABSTOP | BS_PUSHBUTTON as u32, 0, r, id) }
    }

    pub unsafe fn check(&self, text: &str, r: (i32, i32, i32, i32), id: i32) -> HWND {
        unsafe { self.add("BUTTON", text, WS_TABSTOP | BS_AUTOCHECKBOX as u32, 0, r, id) }
    }

    pub unsafe fn group(&self, text: &str, r: (i32, i32, i32, i32)) -> HWND {
        unsafe { self.add("BUTTON", text, BS_GROUPBOX as u32, 0, r, -1) }
    }

    pub unsafe fn edit(&self, text: &str, r: (i32, i32, i32, i32), id: i32) -> HWND {
        unsafe { self.add("EDIT", text, WS_TABSTOP | ES_AUTOHSCROLL as u32, WS_EX_CLIENTEDGE, r, id) }
    }

    /// A drop-down list (the height is the closed box's; the list opens below).
    pub unsafe fn combo(&self, r: (i32, i32, i32, i32), id: i32) -> HWND {
        unsafe { self.add("COMBOBOX", "", WS_TABSTOP | WS_VSCROLL | CBS_DROPDOWNLIST as u32, 0, (r.0, r.1, r.2, 300), id) }
    }
}

pub unsafe fn set_text(h: HWND, s: &str) {
    let w = wide(s);
    unsafe { SetWindowTextW(h, w.as_ptr()) };
}

pub unsafe fn get_text(h: HWND) -> String {
    let n = unsafe { GetWindowTextLengthW(h) }.max(0) as usize;
    let mut b = vec![0u16; n + 1];
    unsafe { GetWindowTextW(h, b.as_mut_ptr(), b.len() as i32) };
    from_wide(&b)
}

pub unsafe fn combo_fill(h: HWND, items: &[String], sel: usize) {
    unsafe {
        SendMessageW(h, CB_RESETCONTENT, 0, 0);
        for i in items {
            let w = wide(i);
            SendMessageW(h, CB_ADDSTRING, 0, w.as_ptr() as LPARAM);
        }
        SendMessageW(h, CB_SETCURSEL, sel.min(items.len().saturating_sub(1)), 0);
    }
}

pub unsafe fn combo_sel(h: HWND) -> usize {
    (unsafe { SendMessageW(h, CB_GETCURSEL, 0, 0) }).max(0) as usize
}

pub unsafe fn set_check(h: HWND, on: bool) {
    unsafe { SendMessageW(h, BM_SETCHECK, on as WPARAM, 0) };
}

pub unsafe fn checked(h: HWND) -> bool {
    unsafe { SendMessageW(h, BM_GETCHECK, 0, 0) == 1 }
}

pub unsafe fn enable(h: HWND, on: bool) {
    unsafe { windows_sys::Win32::UI::Input::KeyboardAndMouse::EnableWindow(h, on as i32) };
}

pub unsafe fn show(h: HWND, on: bool) {
    unsafe { ShowWindow(h, if on { SW_SHOW } else { SW_HIDE }) };
}

pub unsafe fn message(parent: HWND, text: &str, title: &str, flags: MESSAGEBOX_STYLE) -> i32 {
    let (t, m) = (wide(title), wide(text));
    unsafe { MessageBoxW(parent, m.as_ptr(), t.as_ptr(), flags) }
}

/// A report-style list view with columns (title, width).
pub unsafe fn list_view(b: &Builder, r: (i32, i32, i32, i32), id: i32, cols: &[(&str, i32)]) -> HWND {
    let h = unsafe { b.add("SysListView32", "", WS_TABSTOP | LVS_REPORT | LVS_SINGLESEL | LVS_SHOWSELALWAYS, WS_EX_CLIENTEDGE, r, id) };
    unsafe {
        SendMessageW(h, LVM_SETEXTENDEDLISTVIEWSTYLE, 0, (LVS_EX_FULLROWSELECT | LVS_EX_GRIDLINES | LVS_EX_DOUBLEBUFFER) as LPARAM);
        for (i, (t, w)) in cols.iter().enumerate() {
            let mut tw = wide(t);
            let col = LVCOLUMNW { mask: LVCF_TEXT | LVCF_WIDTH | LVCF_SUBITEM, cx: b.scale.px(*w), pszText: tw.as_mut_ptr(), iSubItem: i as i32, ..Default::default() };
            SendMessageW(h, LVM_INSERTCOLUMNW, i, &col as *const _ as LPARAM);
        }
    }
    h
}

pub unsafe fn lv_clear(h: HWND) {
    unsafe { SendMessageW(h, LVM_DELETEALLITEMS, 0, 0) };
}

pub unsafe fn lv_add(h: HWND, cells: &[String]) {
    let n = unsafe { SendMessageW(h, LVM_GETITEMCOUNT, 0, 0) } as i32;
    for (c, text) in cells.iter().enumerate() {
        let mut w = wide(text);
        let mut it: LVITEMW = unsafe { std::mem::zeroed() };
        it.mask = LVIF_TEXT;
        it.iItem = n;
        it.iSubItem = c as i32;
        it.pszText = w.as_mut_ptr();
        unsafe {
            if c == 0 {
                SendMessageW(h, LVM_INSERTITEMW, 0, &it as *const _ as LPARAM);
            } else {
                SendMessageW(h, LVM_SETITEMTEXTW, n as WPARAM, &it as *const _ as LPARAM);
            }
        }
    }
}

pub unsafe fn lv_selected(h: HWND) -> Option<usize> {
    let r = unsafe { SendMessageW(h, LVM_GETNEXTITEM, usize::MAX, LVNI_SELECTED as LPARAM) };
    (r >= 0).then_some(r as usize)
}

pub unsafe fn lv_select(h: HWND, row: usize) {
    let mut it: LVITEMW = unsafe { std::mem::zeroed() };
    it.stateMask = LVIS_SELECTED | LVIS_FOCUSED;
    it.state = LVIS_SELECTED | LVIS_FOCUSED;
    unsafe {
        SendMessageW(h, LVM_SETITEMSTATE, row, &it as *const _ as LPARAM);
        SendMessageW(h, LVM_ENSUREVISIBLE, row, 0);
    }
}

/// An open-file dialog; `filter`: (description, pattern) pairs.
pub unsafe fn pick_file(parent: HWND, title: &str, filter: &[(&str, &str)], initial: &str) -> Option<PathBuf> {
    use windows_sys::Win32::UI::Controls::Dialogs::*;
    let mut f: Vec<u16> = Vec::new();
    for (d, p) in filter {
        f.extend(d.encode_utf16());
        f.push(0);
        f.extend(p.encode_utf16());
        f.push(0);
    }
    f.push(0);
    let mut buf = vec![0u16; 4096];
    let init: Vec<u16> = initial.encode_utf16().take(4000).collect();
    buf[..init.len()].copy_from_slice(&init);
    let t = wide(title);
    let mut ofn: OPENFILENAMEW = unsafe { std::mem::zeroed() };
    ofn.lStructSize = std::mem::size_of::<OPENFILENAMEW>() as u32;
    ofn.hwndOwner = parent;
    ofn.lpstrFilter = f.as_ptr();
    ofn.lpstrFile = buf.as_mut_ptr();
    ofn.nMaxFile = buf.len() as u32;
    ofn.lpstrTitle = t.as_ptr();
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_EXPLORER | OFN_NOCHANGEDIR | OFN_HIDEREADONLY;
    if unsafe { GetOpenFileNameW(&mut ofn) } != 0 { Some(PathBuf::from(from_wide(&buf))) } else { None }
}

/// A folder picker (the shell's file dialog in folder mode).
pub fn pick_folder(parent: HWND, title: &str) -> Option<PathBuf> {
    use windows::Win32::System::Com::{CLSCTX_INPROC_SERVER, CoCreateInstance, CoTaskMemFree};
    use windows::Win32::UI::Shell::{FOS_FORCEFILESYSTEM, FOS_PICKFOLDERS, FileOpenDialog, IFileOpenDialog, SIGDN_FILESYSPATH};
    unsafe {
        let d: IFileOpenDialog = CoCreateInstance(&FileOpenDialog, None, CLSCTX_INPROC_SERVER).ok()?;
        let opts = d.GetOptions().ok()?;
        d.SetOptions(opts | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM).ok()?;
        let t = wide(title);
        let _ = d.SetTitle(windows::core::PCWSTR(t.as_ptr()));
        d.Show(Some(windows::Win32::Foundation::HWND(parent as _))).ok()?;
        let item = d.GetResult().ok()?;
        let p = item.GetDisplayName(SIGDN_FILESYSPATH).ok()?;
        let s = p.to_string().ok();
        CoTaskMemFree(Some(p.0 as *const _));
        s.map(PathBuf::from)
    }
}

/// Opens a folder (or file) with Explorer.
pub fn open_path(p: &std::path::Path) {
    let _ = std::fs::create_dir_all(p);
    let (op, f) = (wide("open"), wide(&p.display().to_string()));
    unsafe { windows_sys::Win32::UI::Shell::ShellExecuteW(std::ptr::null_mut(), op.as_ptr(), f.as_ptr(), std::ptr::null(), std::ptr::null(), SW_SHOWNORMAL) };
}
