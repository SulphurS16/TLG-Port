//! Display page: window mode, monitor, window size, exclusive mode and
//! refresh rate, render resolution, vsync, frame-rate limit, new shaders,
//! HDR output, GPU, volume. (Scaling and its filter: settings.txt.)

use crate::App;
use crate::sys::{self, Monitor};
use crate::ui;
use tlg_config::settings::{self, DisplayMode, FrameCap, Hdr, NewShaders, VSync};
use windows_sys::Win32::Foundation::{HWND, LPARAM};
use windows_sys::Win32::UI::Controls::TBM_SETPOS;
use windows_sys::Win32::UI::WindowsAndMessaging::SendMessageW;

const ID_MODE: i32 = 201;
const ID_MONITOR: i32 = 202;
const ID_WINDOW: i32 = 203;
const ID_RES: i32 = 204;
const ID_HZ: i32 = 205;
const ID_VSYNC: i32 = 206;
const ID_CAP: i32 = 209;
const ID_GPU: i32 = 212;
const ID_GPU_NOTE: i32 = 213;
const ID_VOLUME: i32 = 214;
const ID_VOLUME_TEXT: i32 = 215;
const ID_RENDER: i32 = 216;
const ID_SHADERS: i32 = 217;
const ID_HDR: i32 = 218;

/// HDR white levels offered (nits; 0 = Windows' SDR content brightness).
const HDR_WHITES: [u32; 7] = [0, 100, 150, 200, 250, 300, 400];

/// Render resolutions offered besides the display's (16:9).
const RENDER_SIZES: [(u32, u32); 6] = [(1280, 720), (1600, 900), (1920, 1080), (2560, 1440), (3200, 1800), (3840, 2160)];

const TBM_GETPOS: u32 = 0x400;
const TBM_SETRANGEMIN: u32 = 0x407;
const TBM_SETRANGEMAX: u32 = 0x408;

#[derive(Default)]
pub struct Page {
    mode: usize,
    monitor: usize,
    window: usize,
    res: usize,
    hz: usize,
    render: usize,
    vsync: usize,
    cap: usize,
    shaders: usize,
    hdr: usize,
    gpu: usize,
    gpu_note: usize,
    volume: usize,
    volume_text: usize,
    monitors: Vec<Monitor>,
    /// Window sizes offered (0, 0 = automatic).
    sizes: Vec<(u32, u32)>,
    /// Exclusive modes offered for the monitor (None = desktop).
    resolutions: Vec<Option<(u32, u32)>>,
    rates: Vec<u32>,
    /// Render resolutions offered (None = the display's).
    renders: Vec<Option<(u32, u32)>>,
    /// HDR choices offered: off, or on with a white level.
    hdrs: Vec<(Hdr, u32)>,
    gpus: Vec<String>,
}

fn h(v: usize) -> HWND {
    v as HWND
}

pub fn build(app: &mut App) {
    let b = app.builder();
    let rows: [(&str, i32); 11] = [
        ("Display mode:", ID_MODE),
        ("Monitor:", ID_MONITOR),
        ("Window size:", ID_WINDOW),
        ("Fullscreen resolution:", ID_RES),
        ("Refresh rate:", ID_HZ),
        ("Render resolution:", ID_RENDER),
        ("Vertical sync:", ID_VSYNC),
        ("Frame-rate limit:", ID_CAP),
        ("New shaders:", ID_SHADERS),
        ("HDR output:", ID_HDR),
        ("Graphics card:", ID_GPU),
    ];
    let mut hs = Vec::new();
    unsafe {
        for (i, (label, id)) in rows.iter().enumerate() {
            let y = 56 + i as i32 * 34;
            let r = (32, y + 4, 170, 20);
            app.put(1, b.label(label, r, -1), r, false);
            let r = (210, y, 330, 300);
            hs.push(app.put(1, b.combo(r, *id), r, false) as usize);
        }
        let r = (210, 56 + 10 * 34 + 30, 554, 34);
        app.display.gpu_note = app.put(1, b.label("", r, ID_GPU_NOTE), r, false) as usize;
        let y = 56 + 11 * 34 + 30;
        let r = (32, y + 6, 170, 20);
        app.put(1, b.label("Volume:", r, -1), r, false);
        let r = (204, y, 280, 30);
        app.display.volume = app.put(1, b.add("msctls_trackbar32", "", 0x0001 /* TBS_AUTOTICKS */, 0, r, ID_VOLUME), r, false) as usize;
        SendMessageW(h(app.display.volume), TBM_SETRANGEMIN, 0, 0);
        SendMessageW(h(app.display.volume), TBM_SETRANGEMAX, 1, 100);
        let r = (492, y + 6, 60, 20);
        app.display.volume_text = app.put(1, b.label("", r, ID_VOLUME_TEXT), r, false) as usize;
    }
    let p = &mut app.display;
    (p.mode, p.monitor, p.window, p.res, p.hz, p.render, p.vsync, p.cap, p.shaders, p.hdr, p.gpu) = (hs[0], hs[1], hs[2], hs[3], hs[4], hs[5], hs[6], hs[7], hs[8], hs[9], hs[10]);
    p.monitors = sys::monitors();
    match sys::gpus() {
        Ok(g) => {
            p.gpus = g.iter().map(|x| x.name.clone()).collect();
            // (Only a problem is shown: no card the game can use.)
            if g.iter().all(|x| x.problem.is_some()) {
                let notes: Vec<String> = g.iter().filter_map(|x| x.problem.as_ref().map(|e| format!("{}: {e}", x.name))).collect();
                unsafe { ui::set_text(h(p.gpu_note), &format!("No usable graphics card. {}", notes.join("; "))) };
            }
        }
        Err(e) => unsafe { ui::set_text(h(p.gpu_note), &e) },
    }
    load(app);
}

fn monitor_index(app: &App) -> usize {
    let m = &app.settings.monitor;
    app.display.monitors.iter().position(|x| x.device.eq_ignore_ascii_case(m)).map(|i| i + 1).unwrap_or(0)
}

fn current_monitor(app: &App) -> Option<&Monitor> {
    match monitor_index(app) {
        0 => app.display.monitors.iter().find(|m| m.primary).or(app.display.monitors.first()),
        i => app.display.monitors.get(i - 1),
    }
}

/// Fills every control from the settings.
pub fn load(app: &mut App) {
    let s = app.settings.clone();
    unsafe {
        let p = &app.display;
        ui::combo_fill(
            h(p.mode),
            &["Windowed".into(), "Borderless".into(), "Fullscreen".into()],
            match s.display_mode {
                DisplayMode::Windowed => 0,
                DisplayMode::Borderless => 1,
                DisplayMode::Fullscreen => 2,
            },
        );
        let mut mons = vec!["Primary monitor".to_string()];
        mons.extend(p.monitors.iter().map(|m| format!("{} - {}x{} @ {} Hz{}{}", m.device.trim_start_matches("\\\\.\\"), m.width, m.height, m.hz, if m.name.is_empty() { String::new() } else { format!(", {}", m.name) }, if m.primary { " (primary)" } else { "" })));
        ui::combo_fill(h(p.monitor), &mons, monitor_index(app));
        ui::combo_fill(h(p.vsync), &["On".into(), "Off".into(), "Fast".into()], match s.vsync {
            VSync::On => 0,
            VSync::Off => 1,
            VSync::Fast => 2,
        });
        let caps: Vec<String> = settings::OFFERED_FRAME_CAPS.iter().map(|&c| cap_name(c).into()).collect();
        ui::combo_fill(h(p.cap), &caps, settings::OFFERED_FRAME_CAPS.iter().position(|&c| c == s.frame_cap).unwrap_or(0));
        ui::combo_fill(h(p.shaders), &["Wait for them".into(), "Draw without them meanwhile".into()], (s.new_shaders == NewShaders::Skip) as usize);
        // HDR: used where Windows HDR is on for the game's monitor.
        let hdrs = hdr_choices(&s);
        let names: Vec<String> = hdrs
            .iter()
            .map(|&(m, w)| match (m, w) {
                (Hdr::Off, _) => "Off (SDR output)".to_string(),
                (Hdr::Game, _) => "The game's own HDR (experimental)".to_string(),
                (_, 0) => "Auto, as bright as Windows' SDR content".to_string(),
                (_, 200) => "Auto, white at 200 nits (default)".to_string(),
                (_, w) => format!("Auto, white at {w} nits"),
            })
            .collect();
        let hi = match s.hdr {
            Hdr::Off => 0,
            Hdr::Game => hdrs.len() - 1,
            Hdr::Auto => hdrs.iter().position(|&x| x == (Hdr::Auto, s.hdr_white)).unwrap_or(0),
        };
        ui::combo_fill(h(p.hdr), &names, hi);
        let mut gpus = vec!["Automatic".to_string()];
        gpus.extend(p.gpus.iter().cloned());
        let gi = if s.gpu.eq_ignore_ascii_case("auto") { 0 } else { p.gpus.iter().position(|g| g.to_ascii_lowercase().contains(&s.gpu.to_ascii_lowercase())).map(|i| i + 1).unwrap_or(0) };
        ui::combo_fill(h(p.gpu), &gpus, gi);
        SendMessageW(h(p.volume), TBM_SETPOS, 1, s.volume as LPARAM);
        ui::set_text(h(p.volume_text), &format!("{}%", s.volume));
    }
    app.display.hdrs = hdr_choices(&s);
    fill_monitor_lists(app);
}

fn cap_name(c: FrameCap) -> &'static str {
    match c {
        FrameCap::Game => "30 fps",
        FrameCap::Fps60 => "60 fps",
        FrameCap::Off => "Uncapped (Not recommended)",
    }
}

/// Window sizes, resolutions and refresh rates of the chosen monitor.
fn fill_monitor_lists(app: &mut App) {
    let s = app.settings.clone();
    let Some(m) = current_monitor(app).cloned() else { return };
    let mut sizes = vec![(0u32, 0u32)];
    for (w, hh) in [(1280, 720), (1366, 768), (1600, 900), (1920, 1080), (2560, 1440), (3200, 1800), (3840, 2160)] {
        if w <= m.work.0 && hh <= m.work.1 {
            sizes.push((w, hh));
        }
    }
    if s.window_width > 0 && !sizes.contains(&(s.window_width, s.window_height)) {
        sizes.push((s.window_width, s.window_height));
    }
    let mut res: Vec<Option<(u32, u32)>> = vec![None];
    for &(w, hh, _) in &m.modes {
        if !res.contains(&Some((w, hh))) {
            res.push(Some((w, hh)));
        }
    }
    let p = &mut app.display;
    p.sizes = sizes;
    p.resolutions = res;
    let size_names: Vec<String> = p.sizes.iter().map(|&(w, hh)| if w == 0 { "Automatic".into() } else { format!("{w} x {hh}") }).collect();
    let si = p.sizes.iter().position(|&x| x == (s.window_width, s.window_height)).unwrap_or(0);
    let res_names: Vec<String> = p.resolutions.iter().map(|r| match r {
        None => format!("Desktop ({}x{})", m.width, m.height),
        Some((w, hh)) => format!("{w} x {hh}"),
    }).collect();
    let ri = p.resolutions.iter().position(|&r| r == s.resolution).unwrap_or(0);
    unsafe {
        ui::combo_fill(h(p.window), &size_names, si);
        ui::combo_fill(h(p.res), &res_names, ri);
    }
    fill_rates(app);
}

fn fill_rates(app: &mut App) {
    let s = app.settings.clone();
    let Some(m) = current_monitor(app).cloned() else { return };
    let (w, hh) = s.resolution.unwrap_or((m.width, m.height));
    let mut rates: Vec<u32> = m.modes.iter().filter(|x| x.0 == w && x.1 == hh).map(|x| x.2).collect();
    rates.sort_unstable_by(|a, b| b.cmp(a));
    rates.dedup();
    let p = &mut app.display;
    p.rates = rates;
    let mut names = vec![format!("Desktop ({} Hz)", m.hz)];
    names.extend(p.rates.iter().map(|r| format!("{r} Hz")));
    let i = p.rates.iter().position(|&r| r == s.refresh_rate).map(|i| i + 1).unwrap_or(0);
    let excl = s.display_mode == DisplayMode::Fullscreen;
    unsafe {
        ui::combo_fill(h(p.hz), &names, i);
        ui::enable(h(p.res), excl);
        ui::enable(h(p.hz), excl);
        ui::enable(h(p.window), s.display_mode == DisplayMode::Windowed);
    }
    fill_render(app);
}

/// The render resolutions: the display's (as the game will work it out
/// from the mode, monitor and size chosen above) and the fixed sizes.
fn fill_render(app: &mut App) {
    let s = app.settings.clone();
    let Some(m) = current_monitor(app).cloned() else { return };
    let area = match s.display_mode {
        DisplayMode::Fullscreen => s.resolution.unwrap_or((m.width, m.height)),
        DisplayMode::Borderless => (m.width, m.height),
        DisplayMode::Windowed => settings::window_size(&s, m.work),
    };
    let (dw, dh) = settings::render_size_for(area.0, area.1);
    let mut renders: Vec<Option<(u32, u32)>> = vec![None];
    renders.extend(RENDER_SIZES.iter().map(|&r| Some(r)));
    if let Some(r) = s.render_resolution.filter(|r| !RENDER_SIZES.contains(r)) {
        renders.push(Some(r));
    }
    let names: Vec<String> = renders
        .iter()
        .map(|r| match *r {
            None => format!("Same as the display ({dw} x {dh})"),
            Some((w, hh)) => format!("{w} x {hh}"),
        })
        .collect();
    let i = renders.iter().position(|&r| r == s.render_resolution).unwrap_or(0);
    let p = &mut app.display;
    p.renders = renders;
    unsafe { ui::combo_fill(h(p.render), &names, i) };
}

/// The HDR choices in the order `load` lists them (the game's own HDR keeps
/// the white level set).
fn hdr_choices(s: &settings::Settings) -> Vec<(Hdr, u32)> {
    let mut v: Vec<(Hdr, u32)> = vec![(Hdr::Off, s.hdr_white)];
    v.extend(HDR_WHITES.iter().map(|&w| (Hdr::Auto, w)));
    if !HDR_WHITES.contains(&s.hdr_white) {
        v.push((Hdr::Auto, s.hdr_white));
    }
    v.push((Hdr::Game, s.hdr_white));
    v
}

pub fn command(app: &mut App, id: i32, code: u32) {
    // CBN_SELCHANGE
    if code != 1 {
        return;
    }
    let sel = |v: usize| unsafe { ui::combo_sel(h(v)) };
    let p = &app.display;
    match id {
        ID_MODE => {
            app.settings.display_mode = [DisplayMode::Windowed, DisplayMode::Borderless, DisplayMode::Fullscreen][sel(p.mode).min(2)];
            fill_rates(app);
        }
        ID_MONITOR => {
            let i = sel(p.monitor);
            app.settings.monitor = if i == 0 { "primary".into() } else { p.monitors.get(i - 1).map(|m| m.device.clone()).unwrap_or_else(|| "primary".into()) };
            app.settings.resolution = None;
            app.settings.refresh_rate = 0;
            fill_monitor_lists(app);
        }
        ID_WINDOW => {
            let (w, hh) = p.sizes.get(sel(p.window)).copied().unwrap_or((0, 0));
            (app.settings.window_width, app.settings.window_height) = (w, hh);
            fill_render(app);
        }
        ID_RENDER => app.settings.render_resolution = p.renders.get(sel(p.render)).copied().flatten(),
        ID_RES => {
            app.settings.resolution = p.resolutions.get(sel(p.res)).copied().flatten();
            app.settings.refresh_rate = 0;
            fill_rates(app);
        }
        ID_HZ => {
            let i = sel(p.hz);
            app.settings.refresh_rate = if i == 0 { 0 } else { p.rates.get(i - 1).copied().unwrap_or(0) };
        }
        ID_VSYNC => app.settings.vsync = [VSync::On, VSync::Off, VSync::Fast][sel(p.vsync).min(2)],
        ID_CAP => app.settings.frame_cap = settings::OFFERED_FRAME_CAPS.get(sel(p.cap)).copied().unwrap_or(FrameCap::Game),
        ID_SHADERS => app.settings.new_shaders = if sel(p.shaders) == 1 { NewShaders::Skip } else { NewShaders::Wait },
        ID_HDR => {
            let i = sel(p.hdr);
            if let Some(&(m, w)) = p.hdrs.get(i) {
                app.settings.hdr = m;
                if m == Hdr::Auto {
                    app.settings.hdr_white = w;
                }
                app.display.hdrs = hdr_choices(&app.settings);
            }
        }
        ID_GPU => {
            let i = sel(p.gpu);
            app.settings.gpu = if i == 0 { "auto".into() } else { p.gpus.get(i - 1).cloned().unwrap_or_else(|| "auto".into()) };
        }
        _ => return,
    }
    app.save_settings();
    app.set_status("Saved; applies the next time the game starts.");
}

/// The volume slider moved.
pub fn scroll(app: &mut App, from: HWND) {
    if from != h(app.display.volume) {
        return;
    }
    let v = unsafe { SendMessageW(from, TBM_GETPOS, 0, 0) }.clamp(0, 100) as u32;
    if v != app.settings.volume {
        app.settings.volume = v;
        unsafe { ui::set_text(h(app.display.volume_text), &format!("{v}%")) };
        app.save_settings();
    }
}
