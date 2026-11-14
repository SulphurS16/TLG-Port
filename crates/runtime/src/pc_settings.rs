//! The PC Settings pages' settings: what each page holds (`PAGES`), the
//! values in force (settings.txt and input.txt, changed by the pages while
//! the game runs and written back at once) and what applies them.
//!
//! Where the game itself holds a setting, its own object does the work:
//! the engine's post-processing (`_default_env_pp`), render (`_rcfg`) and
//! camera objects, changed by the port's game script (`engine`: overrides
//! it keeps in force each frame - the game's scripts change some of these
//! fields as they go, and an override remembers the game's own value to
//! put back when it is lifted). The others apply here: the presenter, the
//! frame-rate limit, the sound output, the button prompts.
//!
//! A setting that cannot change while the game runs (the render size and
//! shape: the game allocates its render targets once) is marked as taking
//! effect at the next start, and the page says so.

use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, Ordering};

pub enum Kind {
    /// One of these (value, label).
    Choice(&'static [(&'static str, &'static str)]),
    /// A control's keys (input.txt): chosen by pressing them.
    Bind,
    /// A whole number: min, max, step, the unit after it.
    Slider(i32, i32, i32, &'static str),
    /// Opens the page with this id.
    Page(&'static str),
    /// Runs `act(key)`.
    Action,
}

pub struct Item {
    pub key: &'static str,
    pub label: &'static str,
    pub help: &'static str,
    pub kind: Kind,
}

pub struct Page {
    pub id: &'static str,
    pub title: &'static str,
    pub items: &'static [Item],
}

const ON_OFF: &[(&str, &str)] = &[("on", "On"), ("off", "Off")];

const fn item(key: &'static str, label: &'static str, help: &'static str, kind: Kind) -> Item {
    Item { key, label, help, kind }
}

pub static PAGES: &[Page] = &[
    Page {
        id: "main",
        title: "PC Settings",
        items: &[
            item("display", "Display", "Window, resolution and frame rate.", Kind::Page("display")),
            item("graphics", "Graphics", "Anti-aliasing, upscaling, shadows and lighting.", Kind::Page("graphics")),
            item("effects", "Camera Effects", "Motion blur, depth of field, bloom and lens effects.", Kind::Page("effects")),
            item("sound", "Sound", "Speakers, headphones and sound device.", Kind::Page("sound")),
            item("controls", "Controls", "Button prompts, mouse and keys.", Kind::Page("controls")),
            item("interface", "Interface", "Menu size, mouse pointer and intro logos.", Kind::Page("interface")),
            item("saves", "Saved Games", "Where saved games are kept.", Kind::Page("saves")),
            item("achievements", "Achievements", "Trophies and which you have.", Kind::Page("achievements")),
            item("extras", "Extras", "Cut content still in the game files.", Kind::Page("extras")),
        ],
    },
    Page {
        id: "display",
        title: "Display",
        items: &[
            item("display_mode", "Display Mode", "Windowed, borderless or exclusive full screen.", Kind::Choice(&[("windowed", "Windowed"), ("borderless", "Borderless"), ("fullscreen", "Full Screen")])),
            item("aspect_ratio", "Aspect Ratio", "Automatic matches the screen. Wider shows more to the sides.", Kind::Choice(&[("auto", "Automatic"), ("16:9", "16:9"), ("16:10", "16:10"), ("21:9", "21:9"), ("32:9", "32:9")])),
            item("render_resolution", "Render Resolution", "The game's render height. Display uses the screen's.", Kind::Choice(&[("display", "Display"), ("1280x720", "720p"), ("1600x900", "900p"), ("1920x1080", "1080p"), ("2560x1440", "1440p"), ("3840x2160", "2160p")])),
            item("frame_cap", "Frame Rate Limit", "30 is the game's own. Game speed stays the same.", Kind::Choice(&[("game", "30"), ("60", "60"), ("off", "None")])),
            item("vsync", "Vertical Sync", "On: no tearing. Fast: no tearing, less lag. Off: least lag.", Kind::Choice(&[("on", "On"), ("fast", "Fast"), ("off", "Off")])),
            item("hdr", "HDR", "HDR when Windows HDR is on.", Kind::Choice(&[("auto", "Automatic"), ("off", "Off")])),
            item("filter", "Scaling Filter", "Smooth or sharp scaling to the window.", Kind::Choice(&[("smooth", "Smooth"), ("sharp", "Sharp")])),
        ],
    },
    Page {
        id: "graphics",
        title: "Graphics",
        items: &[
            item("upscaler", "Anti-Aliasing", "SMAA is the game's own. DLSS needs your nvngx_dlss.dll (see the readme).", Kind::Choice(&[("off", "SMAA (the game's)")])),
            item("upscaling", "Upscaling", "DLSS, XeSS or FSR render lower and scale up. Faster on weaker graphics.", Kind::Choice(&[("off", "Off"), ("quality", "Quality (67%)"), ("balanced", "Balanced (58%)"), ("performance", "Performance (50%)"), ("ultra_performance", "Ultra Performance (33%)")])),
            item("sharpening", "Sharpening", "100 is the game's own.", Kind::Slider(0, 200, 10, "")),
            item("shadows", "Shadows", "PS4 is the game's own. High is sharper and uses more video memory.", Kind::Choice(&[("low", "Low"), ("ps4", "PS4"), ("high", "High")])),
            item("texture_filtering", "Texture Filtering", "Keeps textures sharp at an angle. 4x is the PS4's.", Kind::Choice(&[("2x", "2x"), ("4x", "4x"), ("8x", "8x"), ("16x", "16x")])),
            item("ambient_occlusion", "Ambient Occlusion", "Soft shadowing where surfaces meet. Half is the PS4's.", Kind::Choice(&[("off", "Off"), ("half", "Half"), ("full", "Full")])),
            item("reflections", "Reflections", "Reflections on water and wet surfaces.", Kind::Choice(ON_OFF)),
            item("volumetric_fog", "Volumetric Fog", "Fog that light shines through.", Kind::Choice(ON_OFF)),
            item("light_shafts", "Light Shafts", "Rays of light through openings and foliage.", Kind::Choice(ON_OFF)),
        ],
    },
    Page {
        id: "effects",
        title: "Camera Effects",
        items: &[
            item("motion_blur", "Motion Blur", "Blur when the camera turns quickly.", Kind::Choice(ON_OFF)),
            item("depth_of_field", "Depth of Field", "Blur of what is out of focus.", Kind::Choice(ON_OFF)),
            item("bloom", "Bloom", "Glow around bright light.", Kind::Choice(ON_OFF)),
            item("chromatic_aberration", "Chromatic Aberration", "Colour fringes in light shafts.", Kind::Choice(ON_OFF)),
            item("vignette", "Vignette", "Darker corners of the picture.", Kind::Choice(ON_OFF)),
        ],
    },
    Page {
        id: "sound",
        title: "Sound",
        items: &[
            item("speakers", "Speakers", "Automatic follows Windows. Headphones uses Windows Sonic or Dolby Atmos.", Kind::Choice(&[("auto", "Automatic"), ("stereo", "Stereo"), ("5.1", "5.1"), ("7.1", "7.1"), ("headphones", "Headphones")])),
            item("audio_device", "Sound Device", "The device the game plays on.", Kind::Choice(&[])),
            item("volume", "Master Volume", "Overall volume. The game's Audio page sets the rest.", Kind::Slider(0, 100, 5, "")),
        ],
    },
    Page {
        id: "controls",
        title: "Controls",
        items: &[
            item("button_prompts", "Button Prompts", "Automatic shows the device you last used.", Kind::Choice(&[("auto", "Automatic"), ("playstation", "PlayStation"), ("xbox", "Xbox"), ("keyboard", "Keyboard")])),
            item("mouse_sensitivity", "Mouse Sensitivity", "How fast the mouse turns the camera.", Kind::Slider(10, 300, 10, "")),
            item("keys", "Key Bindings", "Keys and mouse buttons for each control.", Kind::Page("keys")),
        ],
    },
    // (Its rows: `keys_rows`, one per control of input.txt.)
    Page { id: "keys", title: "Key Bindings", items: &[] },
    Page {
        id: "interface",
        title: "Interface",
        items: &[
            item("hud_size", "Menu and HUD Size", "Size of menus, prompts and subtitles.", Kind::Slider(60, 110, 5, "%")),
            item("menu_pointer", "Mouse Pointer", "Point and click in the menus.", Kind::Choice(ON_OFF)),
            item("skip_intro", "Skip Intro Logos", "Start at the title screen.", Kind::Choice(&[("off", "Off"), ("on", "On")])),
        ],
    },
    Page {
        id: "saves",
        title: "Saved Games",
        items: &[
            item("save_folder", "Save Folder", "Where saved games are kept. Changing it copies them there.", Kind::Choice(&[])),
            item("choose_folder", "Choose a Folder...", "Pick another folder for saved games.", Kind::Action),
            item("open_folder", "Open the Folder", "Shows the save folder in Explorer.", Kind::Action),
        ],
    },
    // (What each one switches on: script_fixes' EXTRAS.)
    Page {
        id: "extras",
        title: "Extras",
        items: &[
            item("trico_drool", "Trico's Drool", "Trico drools over the smell pots. Cut before release.", Kind::Choice(OFF_ON)),
            item("cut_hints", "Cut Narration", "Narrator hints cut shortly before release. They don't count for the hints trophy.", Kind::Choice(OFF_ON)),
            item("e3_slow_motion", "E3 Slow Motion", "The jump at the collapsing bridge slows down as in the E3 demo.", Kind::Choice(OFF_ON)),
            item("trico_play", "Trico Plays", "Trico now and then sniffs and paws the boy. Cut before release.", Kind::Choice(OFF_ON)),
        ],
    },
];

const OFF_ON: &[(&str, &str)] = &[("off", "Off"), ("on", "On")];

/// The Extras page's settings and the globals that switch them on in the
/// game's scripts (script_fixes' EXTRAS).
const EXTRAS: &[(&str, &str)] = &[("trico_drool", "TLG_X_DROOL"), ("cut_hints", "TLG_X_HINTS"), ("e3_slow_motion", "TLG_X_E3"), ("trico_play", "TLG_X_PLAY")];

/// For the script: per Extras setting its global and whether it is on.
pub fn extras() -> Vec<String> {
    EXTRAS.iter().flat_map(|&(k, g)| [g.to_string(), if get(k) == "on" { "1".to_string() } else { String::new() }]).collect()
}

/// Settings that take effect when the game next starts.
const AT_START: &[&str] = &["aspect_ratio", "render_resolution", "skip_intro", "upscaling", "shadows"];

pub fn page(id: &str) -> Option<&'static Page> {
    PAGES.iter().find(|p| p.id == id)
}

fn find(key: &str) -> Option<&'static Item> {
    PAGES.iter().flat_map(|p| p.items.iter()).find(|i| i.key == key)
}

/// Settings.txt keys the pages show besides `tlg_config::settings::PC`.
const SETTINGS_KEYS: &[&str] = &["display_mode", "render_resolution", "frame_cap", "vsync", "hdr", "filter", "volume"];

struct State {
    values: BTreeMap<&'static str, String>,
    /// The values the game started with (for `AT_START` settings).
    started: BTreeMap<&'static str, String>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);
/// Changes with every engine override change (the script refetches).
static ENGINE_GEN: AtomicU32 = AtomicU32::new(1);

fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
    let mut g = STATE.lock();
    let st = g.get_or_insert_with(|| {
        let s = crate::config::settings();
        let mut f = tlg_config::kv::KvFile::parse("");
        s.store(&mut f);
        let mut values = BTreeMap::new();
        for &(k, _, _) in tlg_config::settings::PC {
            values.insert(k, s.pc[k].clone());
        }
        for &k in SETTINGS_KEYS {
            values.insert(k, f.get(k).unwrap_or_default().to_string());
        }
        // (The pages' 16:9 render sizes; another size from settings.txt stays as it is.)
        let b = crate::input::bindings();
        values.insert("mouse_sensitivity", format!("{}", (b.sensitivity_x * 100.0).round() as i32));
        State { started: values.clone(), values }
    });
    f(st)
}

pub fn get(key: &str) -> String {
    with(|s| s.values.get(key).cloned().unwrap_or_default())
}

/// The choices of a setting whose list depends on the PC.
fn dynamic_choices(key: &str) -> Option<Vec<(String, String)>> {
    match key {
        "audio_device" => {
            let mut v = vec![("default".to_string(), "Windows Default".to_string())];
            v.extend(crate::audio::devices().into_iter().map(|n| (n.clone(), n)));
            let cur = get(key);
            if !v.iter().any(|(k, _)| *k == cur) {
                v.push((cur.clone(), cur));
            }
            Some(v)
        }
        "upscaler" => {
            let mut v = crate::gpu::upscale::choices();
            let cur = get(key);
            if !v.iter().any(|(k, _)| *k == cur) {
                v.push((cur.clone(), format!("{} (not available)", cur.to_uppercase())));
            }
            Some(v)
        }
        "save_folder" => {
            let cur = get(key);
            let mut v = vec![("default".to_string(), "Default".to_string())];
            if let Some(d) = documents_saves() {
                v.push((d.display().to_string(), "Documents".to_string()));
            }
            if !v.iter().any(|(k, _)| *k == cur) {
                v.push((cur.clone(), short_path(&cur)));
            }
            Some(v)
        }
        _ => None,
    }
}

fn short_path(p: &str) -> String {
    if p.chars().count() <= 40 {
        return p.to_string();
    }
    let tail: String = p.chars().rev().take(37).collect::<Vec<_>>().into_iter().rev().collect();
    format!("...{tail}")
}

/// `Documents\The Last Guardian (tlg)\Saves`.
fn documents_saves() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("USERPROFILE")?;
    Some(std::path::PathBuf::from(home).join("Documents").join("The Last Guardian (tlg)").join("Saves"))
}

/// A short name for an input on the pages.
fn input_short(i: tlg_config::bindings::Input) -> String {
    use tlg_config::bindings::Input;
    match i {
        Input::Mouse(0) => "Left Mouse".into(),
        Input::Mouse(1) => "Right Mouse".into(),
        Input::Mouse(2) => "Middle Mouse".into(),
        Input::Mouse(b) => format!("Mouse {}", b + 1),
        Input::WheelUp => "Wheel Up".into(),
        Input::WheelDown => "Wheel Down".into(),
        Input::Key(_) => tlg_config::bindings::input_label(i),
    }
}

/// The Key Bindings page's rows: per control its label and keys.
fn keys_rows() -> Vec<String> {
    let b = crate::input::bindings();
    let mut out = vec![crate::pc_menu::tr("Key Bindings").to_string(), tlg_config::bindings::CONTROLS.len().to_string()];
    for info in tlg_config::bindings::CONTROLS {
        let v: Vec<String> = b.inputs(info.control).iter().map(|&i| input_short(i)).collect();
        let value = if v.is_empty() { "-".to_string() } else { v.join(", ") };
        out.extend(["bind".to_string(), format!("bind:{}", info.name), info.label.to_string(), value, String::new(), crate::pc_menu::tr("Select, then press a key or mouse button. Backspace clears.").to_string(), String::new()]);
    }
    out
}

/// A key or mouse button being chosen for a control: (its input.txt name,
/// the keys held when the choice began).
static CAPTURE: Mutex<Option<(&'static str, [bool; 256])>> = Mutex::new(None);

/// Starts choosing a control's key (`bind:<name>`).
pub fn bind_start(key: &str) -> String {
    let Some(info) = key.strip_prefix("bind:").and_then(|n| tlg_config::bindings::CONTROLS.iter().find(|c| c.name == n)) else { return "unknown".into() };
    let held = std::array::from_fn(|k| crate::gpu::vk::window::key_down(k as u8));
    crate::pc_menu::take_mouse_presses();
    *CAPTURE.lock() = Some((info.name, held));
    "ok".into()
}

/// While choosing: "wait", "done" (bound), "cleared", "cancel", or a note.
pub fn bind_poll() -> String {
    use tlg_config::bindings::Input;
    let mut cap = CAPTURE.lock();
    let Some((name, held)) = *cap else { return "cancel".into() };
    let down = |k: usize| crate::gpu::vk::window::key_down(k as u8);
    // (A key newly down; Shift, Ctrl and Alt by their left/right codes.)
    let key = (8..256usize).find(|&k| !matches!(k, 0x10 | 0x11 | 0x12) && down(k) && !held[k]);
    let mouse = crate::pc_menu::take_mouse_presses();
    let input = match key {
        Some(0x1B) => {
            *cap = None;
            return "cancel".into();
        }
        Some(0x08) => None,
        Some(k) if k == 0x79 || k == 0x7A => {
            *cap = None;
            return crate::pc_menu::tr("F10 and F11 are taken (PC Settings, full screen).").to_string();
        }
        Some(k) => Some(Input::Key(k as u8)),
        None if mouse & 1 != 0 => Some(Input::Mouse(0)),
        None if mouse & 2 != 0 => Some(Input::Mouse(1)),
        None => return "wait".into(),
    };
    *cap = None;
    drop(cap);
    let Some(info) = tlg_config::bindings::CONTROLS.iter().find(|c| c.name == name) else { return "cancel".into() };
    let mut b = crate::input::bindings().clone();
    let note = match input {
        Some(i) => {
            let others: Vec<&str> = b.bound_to(i).iter().filter(|c| **c != info.control).filter_map(|c| tlg_config::bindings::control_info(*c)).map(|c| c.label).collect();
            b.unbind_elsewhere(i, info.control);
            b.set_slot(info.control, 0, Some(i));
            crate::info!("pc settings: {} = {}", info.name, tlg_config::bindings::input_name(i));
            if others.is_empty() { "done".to_string() } else { format!("{} {}", crate::pc_menu::tr("Taken from:"), others.join(", ")) }
        }
        None => {
            while !b.inputs(info.control).is_empty() {
                b.set_slot(info.control, 0, None);
            }
            crate::info!("pc settings: {} cleared", info.name);
            "cleared".to_string()
        }
    };
    apply_bindings(b);
    note
}

/// New bindings: in force at once, written to input.txt, the prompts' key
/// pictures drawn again.
fn apply_bindings(b: tlg_config::bindings::Bindings) {
    if !crate::config::safe_mode() {
        if let Some(p) = crate::config::try_paths() {
            if let Err(e) = b.save(&p.input_file()) {
                crate::warn!("pc settings: writing {}: {e}", p.input_file().display());
            }
        }
    }
    crate::input::set_bindings(b);
    crate::pc_prompts::bindings_changed();
}

/// The Key Bindings page's Reset: every control's keys as at first (the
/// mouse settings kept).
fn reset_keys() {
    let cur = crate::input::bindings();
    let (mut b, _) = tlg_config::bindings::Bindings::parse(tlg_config::bindings::DEFAULTS, None);
    (b.mouse_look, b.sensitivity_x, b.sensitivity_y, b.invert_x, b.invert_y, b.release_key, b.menu_mouse) = (cur.mouse_look, cur.sensitivity_x, cur.sensitivity_y, cur.invert_x, cur.invert_y, cur.release_key, cur.menu_mouse);
    apply_bindings(b);
}

/// One page for the script: title, row count, then per row kind, key,
/// label, value, options (choices: value and label pairs; sliders: min,
/// max, step, unit; tab-separated), help and flags (`s`: takes effect at
/// the next start, `p`: a change waits for the next start).
pub fn page_rows(id: &str) -> Vec<String> {
    if id == "keys" {
        return keys_rows();
    }
    if id == "achievements" {
        return crate::achievements::page_rows();
    }
    let Some(p) = page(id) else { return vec![] };
    let mut out = vec![crate::pc_menu::tr(p.title).to_string(), p.items.len().to_string()];
    for it in p.items {
        let (kind, opts) = match &it.kind {
            Kind::Choice(list) => {
                let list: Vec<(String, String)> = dynamic_choices(it.key).unwrap_or_else(|| list.iter().map(|&(v, l)| (v.to_string(), crate::pc_menu::tr(l).to_string())).collect());
                ("choice", list.iter().flat_map(|(v, l)| [v.as_str(), l.as_str()]).collect::<Vec<_>>().join("\t"))
            }
            Kind::Slider(a, b, s, u) => ("slider", format!("{a}\t{b}\t{s}\t{u}")),
            Kind::Page(to) => ("page", to.to_string()),
            Kind::Action => ("action", String::new()),
            Kind::Bind => ("bind", String::new()),
        };
        let mut flags = String::new();
        if AT_START.contains(&it.key) {
            flags.push('s');
            if with(|s| s.values.get(it.key) != s.started.get(it.key)) {
                flags.push('p');
            }
        }
        out.extend([kind.to_string(), it.key.to_string(), crate::pc_menu::tr(it.label).to_string(), get(it.key), opts, crate::pc_menu::tr(it.help).to_string(), flags]);
    }
    out
}

/// Sets a setting from the pages: checks the value, keeps it, writes it to
/// its file and applies it. Returns "ok", "start" (takes effect at the next
/// start) or an error text.
pub fn set(key: &str, value: &str) -> String {
    let Some(it) = find(key) else { return format!("unknown setting {key}") };
    let ok = match &it.kind {
        Kind::Choice(list) => match dynamic_choices(key) {
            Some(d) => d.iter().any(|(v, _)| v == value) || (key == "save_folder" && std::path::Path::new(value).is_absolute()),
            None => list.iter().any(|(v, _)| *v == value),
        },
        Kind::Slider(a, b, _, _) => value.parse::<i32>().is_ok_and(|v| (*a..=*b).contains(&v)),
        _ => false,
    };
    if !ok {
        return format!("{key}: {value} not accepted");
    }
    if get(key) == value {
        return "ok".into();
    }
    let old = get(key);
    with(|s| s.values.insert(it.key, value.to_string()));
    crate::info!("pc settings: {key} = {value} (was {old})");
    let r = apply(key, value, &old);
    if r.as_deref() == Some("revert") {
        with(|s| s.values.insert(it.key, old));
        return "failed".into();
    }
    persist(key, value);
    if AT_START.contains(&key) { "start".into() } else { r.unwrap_or_else(|| "ok".into()) }
}

fn persist(key: &str, value: &str) {
    match key {
        "mouse_sensitivity" => {
            let v = value.parse::<f32>().unwrap_or(100.0) / 100.0;
            crate::input::persist_binding("mouse_sensitivity_x", &format!("{v}"));
            crate::input::persist_binding("mouse_sensitivity_y", &format!("{v}"));
        }
        _ => crate::config::persist(key, value),
    }
}

/// Applies a changed setting. None: done; Some("revert"): it could not be
/// applied (the old value stays); Some(text): done, with a note.
fn apply(key: &str, value: &str, old: &str) -> Option<String> {
    use crate::config::DisplayMode;
    match key {
        "display_mode" => {
            let m = match value {
                "borderless" => DisplayMode::Borderless,
                "fullscreen" => DisplayMode::Fullscreen,
                _ => DisplayMode::Windowed,
            };
            if let Err(e) = crate::gpu::vk::window::request_mode(Some(m)) {
                crate::warn!("pc settings: display mode: {e}");
                return Some("revert".into());
            }
        }
        "frame_cap" => {
            let n = match value {
                "60" => "60",
                "off" => "0",
                _ => "1",
            };
            let _ = crate::tune::set("frame_cap", n);
            crate::title_patches::set_frame_steps(value != "game");
        }
        "vsync" => crate::gpu::vk::present::set_vsync(value),
        "hdr" => crate::gpu::vk::present::set_hdr(value),
        "filter" => crate::gpu::vk::present::set_filter(value),
        "volume" => crate::audio::set_master_volume(value.parse().unwrap_or(100)),
        "speakers" | "audio_device" => crate::audio::reopen(),
        "mouse_sensitivity" => crate::input::set_mouse_sensitivity(value.parse::<f32>().unwrap_or(100.0) / 100.0),
        "button_prompts" => crate::pc_prompts::changed(),
        "hud_size" => crate::gpu::set_ui_scale(value.parse().unwrap_or(100)),
        "upscaler" => crate::gpu::upscale::set_choice(value),
        "save_folder" => {
            if let Err(e) = crate::hle::libs::savedata::move_saves(old, value) {
                crate::warn!("pc settings: save folder: {e}");
                return Some("revert".into());
            }
        }
        _ => {}
    }
    if engine_keys().contains(&key) || EXTRAS.iter().any(|&(k, _)| k == key) {
        ENGINE_GEN.fetch_add(1, Ordering::Relaxed);
    }
    None
}

/// Runs a page action. Returns what the page shows next: "ok", "reload"
/// (the page's values changed) or a message.
pub fn act(key: &str) -> String {
    match key {
        "choose_folder" => match crate::gpu::vk::window::pick_folder("Choose a folder for saved games") {
            Some(dir) => {
                let r = set("save_folder", &dir.display().to_string());
                if r == "ok" { "reload".into() } else { r }
            }
            None => "ok".into(),
        },
        "trophy_key" => crate::achievements::paste_key(),
        "open_folder" => {
            let dir = crate::hle::libs::savedata::save_root();
            let _ = std::fs::create_dir_all(&dir);
            let _ = std::process::Command::new("explorer.exe").arg(&dir).spawn();
            "ok".into()
        }
        _ => format!("unknown action {key}"),
    }
}

/// Restores a page's settings to their defaults.
pub fn reset_page(id: &str) {
    if id == "keys" {
        reset_keys();
        return;
    }
    let Some(p) = page(id) else { return };
    let defaults = crate::config::Settings::default();
    let mut f = tlg_config::kv::KvFile::parse("");
    defaults.store(&mut f);
    for it in p.items {
        let d = match it.key {
            "mouse_sensitivity" => "100".to_string(),
            k => match f.get(k) {
                Some(v) => v.to_string(),
                None => continue,
            },
        };
        if matches!(it.kind, Kind::Choice(_) | Kind::Slider(..)) {
            let _ = set(it.key, &d);
        }
    }
}

fn engine_keys() -> &'static [&'static str] {
    &["sharpening", "shadows", "texture_filtering", "ambient_occlusion", "reflections", "volumetric_fog", "light_shafts", "motion_blur", "depth_of_field", "bloom", "chromatic_aberration", "vignette"]
}

pub fn engine_gen() -> u32 {
    ENGINE_GEN.load(Ordering::Relaxed)
}

/// The engine overrides in force, for the script: per override its object,
/// field, mode (`set`: this value; `mul`: the game's value times this;
/// `flag`: the named flag of a flags field on or off, as `name=on`) and
/// value.
pub fn engine_overrides() -> Vec<String> {
    let v = |k: &str| get(k);
    let mut out: Vec<[String; 4]> = Vec::new();
    let mut ov = |obj: &str, field: &str, mode: &str, val: String| out.push([obj.into(), field.into(), mode.into(), val]);
    const PP: &str = "_default_env_pp";
    const RC: &str = "_rcfg";
    if let Ok(p) = v("sharpening").parse::<i32>() {
        if p != 100 {
            ov(PP, "sharpenScale", "mul", format!("{}", p as f32 / 100.0));
        }
    }
    // (The shadow map's size as decided at start with the graphics heap:
    // title_patches::SHADOW_MAP. Resized in play, the game found no block
    // that large and crashed - GitHub issue #12.)
    match crate::title_patches::SHADOW_MAP.load(Ordering::Relaxed) {
        0 | 2048 => {}
        n => ov(RC, "shadowMapSize", "set", n.to_string()),
    }
    let aniso = match v("texture_filtering").as_str() {
        "2x" => Some("2"),
        "8x" => Some("8"),
        "16x" => Some("16"),
        _ => None,
    };
    if let Some(a) = aniso {
        ov(RC, "normalMaxAniso", "set", a.into());
        ov(RC, "materialMaxAniso", "set", a.into());
    }
    match v("ambient_occlusion").as_str() {
        "off" => ov(PP, "features", "flag", "SSAO=off".into()),
        "full" => ov(PP, "ssaoResType", "set", "Full".into()),
        _ => {}
    }
    for (key, flag) in [("reflections", "SSR"), ("volumetric_fog", "FogVolume"), ("light_shafts", "LightVolume"), ("motion_blur", "CameraBlur"), ("bloom", "Bloom")] {
        if v(key) == "off" {
            ov(PP, "features", "flag", format!("{flag}=off"));
        }
    }
    if v("depth_of_field") == "off" {
        ov(PP, "cameraMaxDOFScale", "set", "0".into());
    }
    if v("chromatic_aberration") == "off" {
        ov(PP, "lightVolumeChromaticAberration", "set", "0 0 0 0".into());
        ov(PP, "chromaticAberrationAmount", "set", "0".into());
    }
    if v("vignette") == "off" {
        ov(PP, "vignetteParams", "set", "0 0 0 0".into());
    }
    let mut flat = vec![out.len().to_string()];
    for o in out {
        flat.extend(o);
    }
    flat
}

/// At start: the settings applied outside the game's objects.
pub fn init() {
    let _ = with(|_| ());
    crate::gpu::set_ui_scale(get("hud_size").parse().unwrap_or(100));
}
