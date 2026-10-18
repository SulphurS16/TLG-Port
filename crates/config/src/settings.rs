//! Display, frame-rate, audio and game-file settings (`settings.txt` in the
//! user folder). Settings change presentation only - never the game's
//! speed, physics or behaviour. Unknown names and bad values are reported
//! and leave that setting at its default.

use crate::kv::{KvFile, strip_comment};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplayMode {
    Windowed,
    /// The monitor's full area without a mode change.
    Borderless,
    /// Exclusive: the monitor switches to `resolution` / `refresh_rate`.
    Fullscreen,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VSync {
    /// FIFO: no tearing.
    On,
    /// Immediate: lowest latency, tears.
    Off,
    /// Mailbox: no tearing, the newest frame shown (when the driver offers it).
    Fast,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scaling {
    /// The frame's 16:9 shape kept (black bars where the window differs).
    Fit,
    Stretch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    Smooth,
    Sharp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameCap {
    /// The game's own limit (its flip rate: 30 fps in update 01.03).
    Game,
    /// Up to 60 fps: flips at up to 60 Hz and the game's time step snapped
    /// to 1/60 s near that rate instead of 1/30 s, so game speed stays real
    /// time (experimental: not yet played at 60 fps on a machine reaching it).
    Fps60,
    /// No limit (development measurements only: untested game timing).
    Off,
}

/// Frame caps the launcher offers, in its order.
pub const OFFERED_FRAME_CAPS: &[FrameCap] = &[FrameCap::Game, FrameCap::Fps60];

/// What a frame does with an effect whose shader is still being compiled
/// (the first time the game shows it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NewShaders {
    /// Waits for the compile: the game pauses for a moment.
    Wait,
    /// Draws the frame without that effect until its shader is ready: no
    /// pause, the effect may be missing for a moment. (Only effects that
    /// change nothing the game reads back: the rest are waited for.)
    Skip,
}

/// Render resolutions accepted (the tested range).
pub const RENDER_MIN: (u32, u32) = (640, 360);
pub const RENDER_MAX: (u32, u32) = (3840, 2160);

pub fn render_size_ok((w, h): (u32, u32)) -> bool {
    (RENDER_MIN.0..=RENDER_MAX.0).contains(&w) && (RENDER_MIN.1..=RENDER_MAX.1).contains(&h)
}

/// The render size for a display area (`render_resolution = display`): the
/// largest 16:9 size inside it in the accepted range, its width a multiple
/// of 16 and its height even (as in all the game's own display modes).
pub fn render_size_for(w: u32, h: u32) -> (u32, u32) {
    let w = (w as u64).min(h as u64 * 16 / 9) as u32;
    let w = (w & !15).clamp(RENDER_MIN.0, RENDER_MAX.0);
    (w, (w * 9 / 16) & !1)
}

/// The windowed client size on a monitor with work area `work`: the
/// configured size (within the work area), else the largest 16:9 size
/// within 3/4 of the work area (exactly 16:9, width a multiple of 32: the
/// display render size is then the window's).
pub fn window_size(s: &Settings, work: (u32, u32)) -> (u32, u32) {
    let (ww, wh) = work;
    if s.window_width >= 320 && s.window_height >= 180 {
        return (s.window_width.min(ww), s.window_height.min(wh));
    }
    let w = ((ww as u64 * 3 / 4).min(wh as u64 * 3 / 4 * 16 / 9) as u32).max(320) & !31;
    (w, w * 9 / 16)
}

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub display_mode: DisplayMode,
    /// "primary" or a monitor's device name (`\\.\DISPLAY2`).
    pub monitor: String,
    /// Windowed client size; 0 = automatic.
    pub window_width: u32,
    pub window_height: u32,
    /// Exclusive fullscreen mode; None = the monitor's current mode.
    pub resolution: Option<(u32, u32)>,
    /// Exclusive fullscreen refresh rate in Hz; 0 = the current one.
    pub refresh_rate: u32,
    /// The size the game renders at; None = the display's (16:9 inside the
    /// fullscreen mode, the monitor, or the window's client area).
    pub render_resolution: Option<(u32, u32)>,
    pub vsync: VSync,
    pub scaling: Scaling,
    pub filter: Filter,
    pub frame_cap: FrameCap,
    pub new_shaders: NewShaders,
    /// "auto" or part of a GPU's name (or its index in the device list).
    pub gpu: String,
    /// Master volume 0-100.
    pub volume: u32,
    pub game_dir: Option<PathBuf>,
    pub update_dir: Option<PathBuf>,
    /// The launcher closes when the game starts.
    pub close_launcher: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            display_mode: DisplayMode::Windowed,
            monitor: "primary".into(),
            window_width: 0,
            window_height: 0,
            resolution: None,
            refresh_rate: 0,
            render_resolution: None,
            vsync: VSync::On,
            scaling: Scaling::Fit,
            filter: Filter::Smooth,
            frame_cap: FrameCap::Game,
            new_shaders: NewShaders::Wait,
            gpu: "auto".into(),
            volume: 100,
            game_dir: None,
            update_dir: None,
            close_launcher: false,
        }
    }
}

/// The commented file written on first use.
pub const TEMPLATE: &str = "\
# The Last Guardian (tlg): display and game settings. The launcher edits
# this file; changes apply the next time the game starts. Delete the file to
# get the defaults back.

# windowed, borderless (the whole monitor, no mode change) or fullscreen
# (exclusive: the monitor switches to `resolution` and `refresh_rate`).
# Alt+Enter or F11 switches between windowed and full screen while playing.
display_mode = windowed
# primary, or a monitor's device name such as \\\\.\\DISPLAY2.
monitor = primary
# Window client size in pixels; 0 = automatic (16:9, about 3/4 of the screen).
window_width = 0
window_height = 0
# Exclusive fullscreen: WIDTHxHEIGHT and Hz, or desktop (the current mode).
resolution = desktop
refresh_rate = desktop
# The resolution the game renders at (set when it starts): display = the
# screen's (the fullscreen resolution, the monitor's for borderless, the
# window's size for windowed; 16:9 within it), or WIDTHxHEIGHT, e.g.
# 1280x720 (faster) or 3840x2160 (sharper, slower).
render_resolution = display
# on (no tearing), off (lowest latency, may tear) or fast (no tearing, no
# waiting; where the driver offers it).
vsync = on
# Where the window's size differs from the rendered frame's: fit keeps the
# frame's 16:9 shape (black bars where the window's shape differs), stretch
# fills the window.
scaling = fit
# smooth (bilinear) or sharp (nearest pixel) scaling.
filter = smooth
# game = the game's own frame-rate limit (30 fps with update 01.03); 60 = up
# to 60 fps (experimental, needs a fast PC: the game's time step follows the
# frame rate, so its speed stays the same; not yet played at 60 fps).
frame_cap = game
# The first time the game shows an effect, its shader is compiled: wait =
# the game pauses for a moment until it is ready; skip = the frame is drawn
# without that effect meanwhile (no pause, the effect may be missing for a
# moment; effects whose results the game reads are always waited for).
new_shaders = wait
# auto (a discrete GPU first) or part of the graphics card's name.
gpu = auto
# Master volume, 0-100.
volume = 100
# The game folders (the launcher sets these).
game_dir =
update_dir =
# The launcher closes when the game starts (on/off).
close_launcher = off
";

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

impl Settings {
    /// Settings from a file's text over the defaults, with a message per
    /// line that could not be used.
    pub fn parse(text: &str) -> (Settings, Vec<String>) {
        Self::from_kv(&KvFile::parse(text))
    }

    pub fn from_kv(f: &KvFile) -> (Settings, Vec<String>) {
        let mut s = Settings::default();
        let mut errs = Vec::new();
        for (_, k, raw) in f.entries() {
            // (Paths are taken as written: they may contain '#'.)
            let v = if k.ends_with("_dir") || k == "monitor" { raw } else { strip_comment(raw) };
            let lv = v.to_ascii_lowercase();
            let bad = |what: &str| format!("settings.txt: '{k} = {v}': want {what}");
            let num = |v: &str| v.parse::<u32>().ok();
            match k.as_str() {
                "display_mode" => match lv.as_str() {
                    "windowed" | "window" => s.display_mode = DisplayMode::Windowed,
                    "borderless" => s.display_mode = DisplayMode::Borderless,
                    "fullscreen" | "exclusive" => s.display_mode = DisplayMode::Fullscreen,
                    _ => errs.push(bad("windowed, borderless or fullscreen")),
                },
                "monitor" => s.monitor = if v.is_empty() { "primary".into() } else { v.to_string() },
                "window_width" => match num(v) {
                    Some(n) => s.window_width = n,
                    None => errs.push(bad("a number of pixels")),
                },
                "window_height" => match num(v) {
                    Some(n) => s.window_height = n,
                    None => errs.push(bad("a number of pixels")),
                },
                "resolution" => {
                    if lv == "desktop" || lv.is_empty() {
                        s.resolution = None;
                    } else {
                        match lv.split_once('x').and_then(|(w, h)| Some((num(w.trim())?, num(h.trim())?))).filter(|&(w, h)| w >= 320 && h >= 200) {
                            Some(r) => s.resolution = Some(r),
                            None => errs.push(bad("WIDTHxHEIGHT or desktop")),
                        }
                    }
                }
                "refresh_rate" => {
                    if lv == "desktop" || lv.is_empty() {
                        s.refresh_rate = 0;
                    } else {
                        match num(lv.trim_end_matches("hz").trim()) {
                            Some(n) => s.refresh_rate = n,
                            None => errs.push(bad("a rate in Hz or desktop")),
                        }
                    }
                }
                "render_resolution" => {
                    if lv == "display" || lv == "auto" || lv.is_empty() {
                        s.render_resolution = None;
                    } else {
                        match lv.split_once('x').and_then(|(w, h)| Some((num(w.trim())?, num(h.trim())?))).filter(|&r| render_size_ok(r)) {
                            Some(r) => s.render_resolution = Some(r),
                            None => errs.push(bad(&format!(
                                "WIDTHxHEIGHT ({}x{} to {}x{}) or display",
                                RENDER_MIN.0, RENDER_MIN.1, RENDER_MAX.0, RENDER_MAX.1
                            ))),
                        }
                    }
                }
                "vsync" => match lv.as_str() {
                    "on" | "fifo" => s.vsync = VSync::On,
                    "off" | "immediate" => s.vsync = VSync::Off,
                    "fast" | "mailbox" => s.vsync = VSync::Fast,
                    _ => errs.push(bad("on, off or fast")),
                },
                "scaling" => match lv.as_str() {
                    "fit" => s.scaling = Scaling::Fit,
                    "stretch" => s.scaling = Scaling::Stretch,
                    _ => errs.push(bad("fit or stretch")),
                },
                "filter" => match lv.as_str() {
                    "smooth" | "linear" => s.filter = Filter::Smooth,
                    "sharp" | "nearest" => s.filter = Filter::Sharp,
                    _ => errs.push(bad("smooth or sharp")),
                },
                "frame_cap" => match lv.as_str() {
                    "game" | "30" => s.frame_cap = FrameCap::Game,
                    "60" => s.frame_cap = FrameCap::Fps60,
                    "off" | "uncapped" | "0" => s.frame_cap = FrameCap::Off,
                    _ => errs.push(bad("game or 60 (other limits are not tested yet)")),
                },
                "new_shaders" => match lv.as_str() {
                    "wait" => s.new_shaders = NewShaders::Wait,
                    "skip" => s.new_shaders = NewShaders::Skip,
                    _ => errs.push(bad("wait or skip")),
                },
                "gpu" => s.gpu = if v.is_empty() { "auto".into() } else { v.to_string() },
                "volume" => match num(v).filter(|&n| n <= 100) {
                    Some(n) => s.volume = n,
                    None => errs.push(bad("0-100")),
                },
                "game_dir" => s.game_dir = (!v.is_empty()).then(|| PathBuf::from(v)),
                "update_dir" => s.update_dir = (!v.is_empty()).then(|| PathBuf::from(v)),
                "close_launcher" => match flag(v) {
                    Some(b) => s.close_launcher = b,
                    None => errs.push(bad("on or off")),
                },
                _ => errs.push(format!("settings.txt: unknown setting '{k}'")),
            }
        }
        (s, errs)
    }

    /// Loads `path` (the defaults if it does not exist yet).
    pub fn load(path: &Path) -> (Settings, Vec<String>) {
        match KvFile::load(path) {
            Ok(f) => Self::from_kv(&f),
            Err(e) => (Settings::default(), vec![format!("{}: {e}", path.display())]),
        }
    }

    /// Writes every setting into `f` (other lines kept).
    pub fn store(&self, f: &mut KvFile) {
        let mode = match self.display_mode {
            DisplayMode::Windowed => "windowed",
            DisplayMode::Borderless => "borderless",
            DisplayMode::Fullscreen => "fullscreen",
        };
        f.set("display_mode", mode);
        f.set("monitor", &self.monitor);
        f.set("window_width", &self.window_width.to_string());
        f.set("window_height", &self.window_height.to_string());
        f.set("resolution", &self.resolution.map_or("desktop".to_string(), |(w, h)| format!("{w}x{h}")));
        f.set("refresh_rate", &if self.refresh_rate == 0 { "desktop".to_string() } else { self.refresh_rate.to_string() });
        f.set("render_resolution", &self.render_resolution.map_or("display".to_string(), |(w, h)| format!("{w}x{h}")));
        f.set("vsync", match self.vsync {
            VSync::On => "on",
            VSync::Off => "off",
            VSync::Fast => "fast",
        });
        f.set("scaling", match self.scaling {
            Scaling::Fit => "fit",
            Scaling::Stretch => "stretch",
        });
        f.set("filter", match self.filter {
            Filter::Smooth => "smooth",
            Filter::Sharp => "sharp",
        });
        f.set("frame_cap", match self.frame_cap {
            FrameCap::Game => "game",
            FrameCap::Fps60 => "60",
            FrameCap::Off => "off",
        });
        f.set("new_shaders", match self.new_shaders {
            NewShaders::Wait => "wait",
            NewShaders::Skip => "skip",
        });
        f.set("gpu", &self.gpu);
        f.set("volume", &self.volume.to_string());
        f.set("game_dir", &self.game_dir.as_ref().map(|p| p.display().to_string()).unwrap_or_default());
        f.set("update_dir", &self.update_dir.as_ref().map(|p| p.display().to_string()).unwrap_or_default());
        f.set("close_launcher", on_off(self.close_launcher));
    }

    /// Saves into `path`, keeping the file's comments (the commented
    /// template for a new file).
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut f = KvFile::load(path)?;
        if f.entries().next().is_none() {
            f = KvFile::parse(TEMPLATE);
        }
        self.store(&mut f);
        f.save(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_is_the_defaults() {
        let (s, errs) = Settings::parse(TEMPLATE);
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(s, Settings::default());
        assert_eq!(s.monitor, "primary");
    }

    #[test]
    fn values_and_errors() {
        let (s, errs) = Settings::parse(
            "display_mode = Borderless\nresolution = 2560x1440\nrefresh_rate = 144hz\nvsync = fast # tear-free\nvolume = 70\ngame_dir = D:\\Games #1\\TLG\nframe_cap = 45\nbogus = 1\nwindow_width = wide\n",
        );
        assert_eq!(s.display_mode, DisplayMode::Borderless);
        assert_eq!(s.resolution, Some((2560, 1440)));
        assert_eq!(s.refresh_rate, 144);
        assert_eq!(s.vsync, VSync::Fast);
        assert_eq!(s.volume, 70);
        assert_eq!(s.game_dir, Some(PathBuf::from("D:\\Games #1\\TLG")));
        assert_eq!(s.frame_cap, FrameCap::Game);
        assert_eq!(errs.len(), 3, "{errs:?}");
        let (s, errs) = Settings::parse("frame_cap = 60 # experimental\n");
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(s.frame_cap, FrameCap::Fps60);
    }

    #[test]
    fn render_resolution() {
        let (s, errs) = Settings::parse("render_resolution = 2560x1440\n");
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(s.render_resolution, Some((2560, 1440)));
        let (s, errs) = Settings::parse("render_resolution = 8000x5000\n");
        assert_eq!((s.render_resolution, errs.len()), (None, 1));
        assert_eq!(render_size_for(1920, 1080), (1920, 1080));
        assert_eq!(render_size_for(1440, 810), (1440, 810));
        assert_eq!(render_size_for(1920, 1200), (1920, 1080));
        assert_eq!(render_size_for(3440, 1440), (2560, 1440));
        assert_eq!(render_size_for(1366, 768), (1360, 764));
        assert_eq!(render_size_for(5120, 2880), (3840, 2160));
        assert_eq!(render_size_for(300, 200), (640, 360));
        // Automatic windows render 1:1 (work areas of 2560x1440, 1920x1080
        // and 1366x768 screens).
        let s = Settings::default();
        for work in [(2560, 1392), (1920, 1032), (1366, 728), (3840, 2112)] {
            let (w, h) = window_size(&s, work);
            assert_eq!(render_size_for(w, h), (w, h), "{work:?}");
        }
        assert_eq!(window_size(&s, (2560, 1392)), (1856, 1044));
    }

    #[test]
    fn store_round_trips() {
        let s = Settings {
            display_mode: DisplayMode::Fullscreen,
            monitor: r"\\.\DISPLAY2".into(),
            resolution: Some((1920, 1080)),
            render_resolution: Some((1280, 720)),
            refresh_rate: 60,
            vsync: VSync::Off,
            filter: Filter::Sharp,
            frame_cap: FrameCap::Fps60,
            new_shaders: NewShaders::Skip,
            game_dir: Some(PathBuf::from(r"E:\TLG\CUSA03627")),
            close_launcher: true,
            ..Settings::default()
        };
        let mut f = KvFile::parse(TEMPLATE);
        s.store(&mut f);
        let (back, errs) = Settings::from_kv(&f);
        assert!(errs.is_empty(), "{errs:?}");
        assert_eq!(back, s);
        assert!(f.text().starts_with("# The Last Guardian (tlg)"));
    }
}
