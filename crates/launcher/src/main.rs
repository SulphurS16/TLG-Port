//! tlg-launcher: sets up and starts the tlg runtime for players.
//!
//! - Game files: the player's game folder (01.00) and update folder (01.03),
//!   checked when chosen.
//! - Display: window mode, monitor, sizes and modes, vsync, the frame-rate
//!   limit, new shaders, GPU, volume.
//! - Controls: every keyboard/mouse binding (two per action), conflicts
//!   shown, reset; mouse settings.
//! - Play / Reset settings / Open logs.
//!
//! Settings are the same files the game reads (tlg_config: settings.txt,
//! input.txt in the user folder); changes are saved as they are made and
//! apply the next time the game starts.

#![windows_subsystem = "windows"]

mod controls;
mod display;
mod files;
mod game;
mod sys;
mod ui;

use std::cell::RefCell;
use std::path::PathBuf;
use tlg_config::bindings::Bindings;
use tlg_config::paths::Paths;
use tlg_config::settings::Settings;
use ui::{Builder, Scale, wide};
use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows_sys::Win32::Graphics::Gdi::{DeleteObject, HFONT};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::Controls::*;
use windows_sys::Win32::UI::HiDpi::{AdjustWindowRectExForDpi, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, GetDpiForWindow, SetProcessDpiAwarenessContext};
use windows_sys::Win32::UI::WindowsAndMessaging::*;

pub const TITLE: &str = "The Last Guardian PC port";
const CLIENT: (i32, i32) = (780, 600);

/// Posted when the game process exits (WPARAM: exit code).
pub const WM_GAME_EXIT: u32 = WM_APP + 2;
/// Retries a game exit that arrived while a modal dialog held the app.
const TIMER_RETRY: usize = 2;

const ID_TABS: i32 = 10;
const ID_PLAY: i32 = 901;
const ID_RESET: i32 = 903;
const ID_LOGS: i32 = 904;
const ID_STATUS_BAR: i32 = 906;

pub struct App {
    pub hwnd: HWND,
    pub scale: Scale,
    pub font: HFONT,
    pub bold: HFONT,
    /// Every control with its 1/96-inch rectangle and font (re-laid out
    /// when the DPI changes).
    pub layout: Vec<(HWND, (i32, i32, i32, i32), bool)>,
    pub paths: Paths,
    pub settings: Settings,
    pub bindings: Bindings,
    pub tabs: HWND,
    /// The controls of each page.
    pub pages: Vec<Vec<HWND>>,
    pub page: usize,
    pub status: HWND,
    pub play: HWND,
    pub game: game::Page,
    pub display: display::Page,
    pub controls: controls::Page,
    /// The game process while it runs.
    pub running: Option<u32>,
}

thread_local! {
    static APP: RefCell<Option<App>> = const { RefCell::new(None) };
    /// Game exits that could not be handled yet (a message box or dialog
    /// was open): their exit codes.
    static PENDING: RefCell<Vec<WPARAM>> = const { RefCell::new(Vec::new()) };
}

/// Handles a game exit, or keeps it for a retry.
fn event(hwnd: HWND, wp: WPARAM) {
    if with_app(|app| game_exited(app, wp as u32 as i32)).is_none() {
        PENDING.with(|p| p.borrow_mut().push(wp));
        unsafe { SetTimer(hwnd, TIMER_RETRY, 250, None) };
    }
}

/// Runs `f` on the app unless it is already borrowed (a nested message
/// during a modal loop): then None.
pub fn with_app<R>(f: impl FnOnce(&mut App) -> R) -> Option<R> {
    APP.with(|a| a.try_borrow_mut().ok().and_then(|mut g| g.as_mut().map(f)))
}

impl App {
    /// A control builder for the main window (records the control for re-layout).
    pub fn builder(&self) -> Builder {
        Builder { parent: self.hwnd, scale: self.scale, font: self.font }
    }

    /// Adds a control to the current page being built.
    pub fn put(&mut self, page: usize, h: HWND, r: (i32, i32, i32, i32), bold: bool) -> HWND {
        self.layout.push((h, r, bold));
        if page < self.pages.len() {
            self.pages[page].push(h);
        }
        if bold {
            unsafe { SendMessageW(h, WM_SETFONT, self.bold as WPARAM, 1) };
        }
        h
    }

    pub fn save_settings(&mut self) {
        if let Err(e) = self.settings.save(&self.paths.settings_file()) {
            self.set_status(&format!("Could not save {}: {e}", self.paths.settings_file().display()));
        }
    }

    pub fn save_bindings(&mut self) {
        if let Err(e) = self.bindings.save(&self.paths.input_file()) {
            self.set_status(&format!("Could not save {}: {e}", self.paths.input_file().display()));
        }
    }

    pub fn set_status(&self, s: &str) {
        unsafe { ui::set_text(self.status, s) };
    }

    fn show_page(&mut self, n: usize) {
        self.page = n;
        for (i, p) in self.pages.iter().enumerate() {
            for &h in p {
                unsafe { ui::show(h, i == n) };
            }
        }
    }

    pub fn update_play(&self) {
        unsafe { ui::enable(self.play, self.game.ok && self.running.is_none()) };
    }

    fn relayout(&mut self, dpi: u32) {
        self.scale = Scale(dpi);
        unsafe {
            DeleteObject(self.font as _);
            DeleteObject(self.bold as _);
            self.font = ui::message_font(dpi, false, 100);
            self.bold = ui::message_font(dpi, true, 110);
            for &(h, r, bold) in &self.layout {
                let s = self.scale;
                SetWindowPos(h, std::ptr::null_mut(), s.px(r.0), s.px(r.1), s.px(r.2), s.px(r.3), SWP_NOZORDER | SWP_NOACTIVATE);
                SendMessageW(h, WM_SETFONT, if bold { self.bold } else { self.font } as WPARAM, 1);
            }
        }
    }
}

/// Starts the game (the launcher minimized while it runs; it comes back
/// when the game exits and reports a crash).
fn play(app: &mut App) {
    if app.running.is_some() {
        return;
    }
    let exe = app.paths.exe_dir.join("tlg.exe");
    if !exe.is_file() {
        unsafe { ui::message(app.hwnd, &format!("tlg.exe was not found next to the launcher ({}).", exe.display()), TITLE, MB_ICONERROR) };
        return;
    }
    let mut cmd = std::process::Command::new(&exe);
    cmd.arg("--launcher").current_dir(&app.paths.exe_dir);
    match cmd.spawn() {
        Ok(mut child) => {
            app.running = Some(child.id());
            app.update_play();
            let hwnd = app.hwnd as usize;
            std::thread::spawn(move || {
                let code = child.wait().ok().and_then(|s| s.code()).unwrap_or(-1);
                unsafe { PostMessageW(hwnd as HWND, WM_GAME_EXIT, code as u32 as WPARAM, 0) };
            });
            app.set_status("");
            unsafe { ShowWindow(app.hwnd, SW_MINIMIZE) };
        }
        Err(e) => unsafe {
            ui::message(app.hwnd, &format!("Could not start {}: {e}", exe.display()), TITLE, MB_ICONERROR);
        },
    }
}

/// The newest run folder in the logs (a game session's log and crash report).
fn newest_log(paths: &Paths) -> Option<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(paths.logs()).ok()?.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    v.sort();
    v.pop()
}

fn game_exited(app: &mut App, code: i32) {
    app.running = None;
    app.update_play();
    unsafe {
        ShowWindow(app.hwnd, SW_RESTORE);
        SetForegroundWindow(app.hwnd);
    }
    if code == 0 {
        return;
    }
    let log = newest_log(&app.paths);
    let mut details = String::new();
    if let Some(d) = &log {
        if let Ok(t) = std::fs::read_to_string(d.join("crash.txt")) {
            details = t.lines().take(12).collect::<Vec<_>>().join("\n");
        } else if let Ok(t) = std::fs::read_to_string(d.join("tlg.log")) {
            let errs: Vec<&str> = t.lines().filter(|l| l.contains("] E: ")).collect();
            details = errs[errs.len().saturating_sub(6)..].join("\n");
        }
    }
    app.set_status(&format!("The game stopped with an error (exit code {code:#x})."));
    let text = format!(
        "The game stopped with an error (exit code {code:#x}).\n\n{}\n\nThe log is in:\n{}\n\nOpen the log folder?",
        if details.is_empty() { "No details were recorded.".to_string() } else { details },
        log.as_ref().map(|d| d.display().to_string()).unwrap_or_else(|| app.paths.logs().display().to_string())
    );
    if unsafe { ui::message(app.hwnd, &text, TITLE, MB_ICONERROR | MB_YESNO) } == IDYES {
        ui::open_path(&log.unwrap_or_else(|| app.paths.logs()));
    }
}

fn reset_settings(app: &mut App) {
    let text = "Reset the display settings and the key bindings to their defaults?\n\nThe current files are kept as settings.txt.bak and input.txt.bak.";
    if unsafe { ui::message(app.hwnd, text, TITLE, MB_ICONQUESTION | MB_OKCANCEL) } != IDOK {
        return;
    }
    let keep = (app.settings.game_dir.clone(), app.settings.update_dir.clone());
    for f in [app.paths.settings_file(), app.paths.input_file()] {
        if f.exists() {
            let mut b = f.clone().into_os_string();
            b.push(".bak");
            let _ = std::fs::rename(&f, PathBuf::from(b));
        }
    }
    app.settings = Settings { game_dir: keep.0, update_dir: keep.1, ..Settings::default() };
    app.bindings = Bindings::default();
    app.save_settings();
    app.save_bindings();
    display::load(app);
    controls::load(app);
    app.set_status("Settings reset.");
}

unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    let handled = match msg {
        WM_COMMAND => {
            let (id, code) = ((wp & 0xFFFF) as i32, ((wp >> 16) & 0xFFFF) as u32);
            with_app(|app| match id {
                ID_PLAY => play(app),
                ID_RESET => reset_settings(app),
                ID_LOGS => ui::open_path(&app.paths.logs()),
                100..=199 => game::command(app, id, code),
                200..=299 => display::command(app, id, code),
                300..=399 => controls::command(app, id, code),
                _ => {}
            })
            .is_some()
        }
        WM_NOTIFY => {
            let h = unsafe { &*(lp as *const NMHDR) };
            with_app(|app| {
                if h.idFrom == ID_TABS as usize && h.code == TCN_SELCHANGE {
                    let n = unsafe { SendMessageW(app.tabs, TCM_GETCURSEL, 0, 0) }.max(0) as usize;
                    app.show_page(n);
                } else if h.idFrom / 100 == 3 {
                    controls::notify(app, h, lp);
                }
            })
            .is_some()
        }
        WM_HSCROLL => with_app(|app| display::scroll(app, lp as HWND)).is_some(),
        WM_TIMER if wp == TIMER_RETRY => {
            unsafe { KillTimer(hwnd, TIMER_RETRY) };
            let pending: Vec<WPARAM> = PENDING.with(|p| std::mem::take(&mut *p.borrow_mut()));
            for w in pending {
                event(hwnd, w);
            }
            true
        }
        WM_GAME_EXIT => {
            event(hwnd, wp);
            true
        }
        WM_DPICHANGED => {
            let r = unsafe { &*(lp as *const RECT) };
            unsafe { SetWindowPos(hwnd, std::ptr::null_mut(), r.left, r.top, r.right - r.left, r.bottom - r.top, SWP_NOZORDER | SWP_NOACTIVATE) };
            let dpi = (wp & 0xFFFF) as u32;
            with_app(|app| app.relayout(dpi));
            true
        }
        WM_DESTROY => {
            unsafe { PostQuitMessage(0) };
            true
        }
        _ => false,
    };
    if handled && matches!(msg, WM_COMMAND | WM_NOTIFY | WM_HSCROLL | WM_TIMER | WM_GAME_EXIT | WM_DPICHANGED | WM_DESTROY) {
        return 0;
    }
    unsafe { DefWindowProcW(hwnd, msg, wp, lp) }
}

fn main() {
    unsafe {
        SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        let _ = windows::Win32::System::Com::CoInitializeEx(None, windows::Win32::System::Com::COINIT_APARTMENTTHREADED);
        let icc = INITCOMMONCONTROLSEX {
            dwSize: std::mem::size_of::<INITCOMMONCONTROLSEX>() as u32,
            dwICC: ICC_TAB_CLASSES | ICC_LISTVIEW_CLASSES | ICC_BAR_CLASSES | ICC_STANDARD_CLASSES,
        };
        InitCommonControlsEx(&icc);
        let hinst = GetModuleHandleW(std::ptr::null());
        let class = wide("tlg_launcher");
        let wc = WNDCLASSEXW {
            cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
            style: CS_HREDRAW | CS_VREDRAW,
            lpfnWndProc: Some(wndproc),
            cbClsExtra: 0,
            cbWndExtra: 0,
            hInstance: hinst,
            hIcon: LoadIconW(hinst, 1 as _),
            hCursor: LoadCursorW(std::ptr::null_mut(), IDC_ARROW),
            hbrBackground: (windows_sys::Win32::Graphics::Gdi::COLOR_BTNFACE + 1) as usize as _,
            lpszMenuName: std::ptr::null(),
            lpszClassName: class.as_ptr(),
            hIconSm: LoadIconW(hinst, 1 as _),
        };
        RegisterClassExW(&wc);
        let style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
        let title = wide(&format!("{TITLE} {}", tlg_config::VERSION));
        let hwnd = CreateWindowExW(0, class.as_ptr(), title.as_ptr(), style, CW_USEDEFAULT, CW_USEDEFAULT, 800, 640, std::ptr::null_mut(), std::ptr::null_mut(), hinst, std::ptr::null());
        if hwnd.is_null() {
            return;
        }
        let dpi = GetDpiForWindow(hwnd).max(96);
        let scale = Scale(dpi);
        let mut r = RECT { left: 0, top: 0, right: scale.px(CLIENT.0), bottom: scale.px(CLIENT.1) };
        AdjustWindowRectExForDpi(&mut r, style, 0, 0, dpi);
        SetWindowPos(hwnd, std::ptr::null_mut(), 0, 0, r.right - r.left, r.bottom - r.top, SWP_NOMOVE | SWP_NOZORDER);

        let paths = Paths::current();
        let (settings, _) = Settings::load(&paths.settings_file());
        let (bindings, _) = Bindings::load(&paths.input_file());
        let font = ui::message_font(dpi, false, 100);
        let bold = ui::message_font(dpi, true, 110);
        let mut app = App {
            hwnd,
            scale,
            font,
            bold,
            layout: Vec::new(),
            paths,
            settings,
            bindings,
            tabs: std::ptr::null_mut(),
            pages: vec![Vec::new(); 3],
            page: 0,
            status: std::ptr::null_mut(),
            play: std::ptr::null_mut(),
            game: game::Page::default(),
            display: display::Page::default(),
            controls: controls::Page::default(),
            running: None,
        };
        let b = app.builder();
        let rt = (12, 10, 756, 30);
        let tabs = b.add("SysTabControl32", "", WS_TABSTOP | WS_CLIPSIBLINGS, 0, rt, ID_TABS);
        app.tabs = app.put(usize::MAX, tabs, rt, false);
        for (i, t) in ["Game files", "Display", "Controls"].iter().enumerate() {
            let mut w = wide(t);
            let item = TCITEMW { mask: TCIF_TEXT, pszText: w.as_mut_ptr(), ..Default::default() };
            SendMessageW(tabs, TCM_INSERTITEMW, i, &item as *const _ as LPARAM);
        }
        let b = app.builder();
        let r = (16, 548, 120, 36);
        app.play = app.put(usize::MAX, b.add("BUTTON", "Play", WS_TABSTOP | BS_DEFPUSHBUTTON as u32, 0, r, ID_PLAY), r, true);
        let r = (16, 520, 748, 20);
        app.status = app.put(usize::MAX, b.label("", r, ID_STATUS_BAR), r, false);
        game::build(&mut app);
        display::build(&mut app);
        controls::build(&mut app);
        let b = app.builder();
        let r = (144, 548, 130, 36);
        app.put(usize::MAX, b.button("Reset settings...", r, ID_RESET), r, false);
        let r = (282, 548, 100, 36);
        app.put(usize::MAX, b.button("Open logs", r, ID_LOGS), r, false);
        // (--page N opens another page: checks of the layout.)
        let first = std::env::args().skip_while(|a| a != "--page").nth(1).and_then(|n| n.parse::<usize>().ok()).unwrap_or(0).min(2);
        SendMessageW(app.tabs, TCM_SETCURSEL, first, 0);
        app.show_page(first);
        app.update_play();
        startup_status(&mut app);
        APP.with(|a| *a.borrow_mut() = Some(app));
        ShowWindow(hwnd, SW_SHOW);
        let mut msg: MSG = std::mem::zeroed();
        while GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) > 0 {
            if IsDialogMessageW(hwnd, &msg) == 0 {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
    }
}

/// The opening status line: a game still running, or a previous session
/// that did not end normally.
fn startup_status(app: &mut App) {
    let marker = app.paths.running_marker();
    let Ok(t) = std::fs::read_to_string(&marker) else { return };
    let pid = t.lines().find_map(|l| l.strip_prefix("pid ")).and_then(|p| p.trim().parse::<u32>().ok());
    if let Some(pid) = pid.filter(|&p| sys::process_alive(p)) {
        app.running = Some(pid);
        app.update_play();
        app.set_status("The game is running.");
    } else {
        app.set_status("The previous session did not end normally; Open logs shows its log.");
    }
}
