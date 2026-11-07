//! First start: settings that suit the PC (settings.txt `auto_settings =
//! pending`, as a new file has it; an older file without the line, or with
//! `done`, is left as it is).
//!
//! Before the window opens: the graphics card (DXGI: its vendor and
//! dedicated memory), the processor's threads and the monitor's mode give a
//! tier, and the tier the display mode, render resolution, frame-rate
//! limit, texture filtering, shadows and ambient occlusion. Every choice is
//! a setting the PC Settings pages show and change.

use tlg_config::kv::KvFile;

struct Adapter {
    name: String,
    vendor: u32,
    vram: u64,
}

/// The adapter with the most dedicated memory (the discrete card where
/// there is one).
fn best_adapter() -> Option<Adapter> {
    use windows::Win32::Graphics::Dxgi::*;
    unsafe {
        let f: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
        let mut best: Option<Adapter> = None;
        let mut i = 0;
        while let Ok(a) = f.EnumAdapters1(i) {
            i += 1;
            let Ok(d) = a.GetDesc1() else { continue };
            // (Software adapters: not a card.)
            if d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                continue;
            }
            let name = String::from_utf16_lossy(&d.Description[..d.Description.iter().position(|&c| c == 0).unwrap_or(d.Description.len())]);
            let ad = Adapter { name, vendor: d.VendorId, vram: d.DedicatedVideoMemory as u64 };
            if best.as_ref().is_none_or(|b| ad.vram > b.vram) {
                best = Some(ad);
            }
        }
        best
    }
}

/// The primary monitor's mode: width, height, refresh rate.
fn monitor_mode() -> (u32, u32, u32) {
    use windows_sys::Win32::Graphics::Gdi::{DEVMODEW, ENUM_CURRENT_SETTINGS, EnumDisplaySettingsW};
    unsafe {
        let mut dm: DEVMODEW = std::mem::zeroed();
        dm.dmSize = std::mem::size_of::<DEVMODEW>() as u16;
        if EnumDisplaySettingsW(std::ptr::null(), ENUM_CURRENT_SETTINGS, &mut dm) != 0 {
            return (dm.dmPelsWidth, dm.dmPelsHeight, dm.dmDisplayFrequency);
        }
    }
    (1920, 1080, 60)
}

/// Picks the settings when due (before the settings are read).
pub fn first_start(paths: &tlg_config::paths::Paths, safe_mode: bool) {
    if safe_mode {
        return;
    }
    let file = paths.settings_file();
    let mut f = if file.exists() {
        match KvFile::load(&file) {
            Ok(f) => f,
            Err(_) => return,
        }
    } else {
        KvFile::parse(tlg_config::settings::TEMPLATE)
    };
    if f.get("auto_settings").map(str::trim) != Some("pending") {
        return;
    }
    let gpu = best_adapter();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let (mw, mh, hz) = monitor_mode();
    let (vram, vendor, name) = gpu.as_ref().map_or((0, 0, "unknown".to_string()), |g| (g.vram, g.vendor, g.name.clone()));
    let gib = vram as f64 / (1u64 << 30) as f64;
    // (Integrated graphics report little dedicated memory.)
    let tier = if gib >= 7.5 && threads >= 8 {
        2
    } else if gib >= 3.5 {
        1
    } else {
        0
    };
    let mut set = |k: &str, v: &str| f.set(k, v);
    set("display_mode", "borderless");
    // (The fixed sizes name a height: the width follows the screen's shape.)
    let render = match tier {
        2 => "display".to_string(),
        1 if mh > 1440 => "2560x1440".into(),
        1 => "display".into(),
        _ if mh > 900 => "1600x900".into(),
        _ => "display".into(),
    };
    set("render_resolution", &render);
    set("frame_cap", if tier == 2 && threads >= 12 && hz >= 59 { "60" } else { "game" });
    set("texture_filtering", if tier >= 1 { "16x" } else { "4x" });
    set("shadows", ["low", "ps4", "high"][tier]);
    set("ambient_occlusion", if tier == 2 { "full" } else { "half" });
    set("auto_settings", "done");
    let vendor_name = match vendor {
        0x10de => "NVIDIA",
        0x1002 | 0x1022 => "AMD",
        0x8086 => "Intel",
        _ => "other",
    };
    crate::info!(
        "first start: {name} ({vendor_name}, {gib:.1} GiB), {threads} threads, monitor {mw}x{mh} {hz} Hz: tier {tier} - borderless, render {render}, frame cap {}, filtering {}, shadows {}",
        f.get("frame_cap").unwrap_or(""),
        f.get("texture_filtering").unwrap_or(""),
        f.get("shadows").unwrap_or("")
    );
    let _ = std::fs::create_dir_all(&paths.user);
    if let Err(e) = f.save(&file) {
        crate::warn!("first start: writing {}: {e}", file.display());
    }
}
