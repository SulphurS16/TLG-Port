//! Other programs' implicit Vulkan layers (overlays, recording and capture
//! hooks): listed in the log at start, named when code of one faults, and
//! kept out with settings.txt `overlays = off`. A capture hook loaded as a
//! layer (a copy of OBS's graphics-hook64.dll) crashed one player's game as
//! it created its Vulkan device.

use std::sync::OnceLock;

/// An implicit layer registered for all Vulkan programs on this PC.
pub struct Layer {
    pub name: String,
    /// Its library's path, as the manifest gives it (relative ones resolved).
    pub library: String,
}

static LAYERS: OnceLock<Vec<Layer>> = OnceLock::new();

/// Before the Vulkan loader starts: lists the implicit layers registered in
/// HKLM and HKCU (SOFTWARE\Khronos\Vulkan\ImplicitLayers, enabled ones) and,
/// with `overlays` off, keeps every implicit layer but the graphics drivers'
/// own out (the loader's VK_LOADER_LAYERS_DISABLE / _ALLOW).
pub fn prepare(overlays: bool) {
    let layers = LAYERS.get_or_init(scan);
    if layers.is_empty() {
        crate::info!("Vulkan implicit layers: none registered for all programs");
    } else {
        let list: Vec<String> = layers.iter().map(|l| format!("{} ({})", l.name, l.library)).collect();
        crate::info!("Vulkan implicit layers: {}", list.join(", "));
    }
    if !overlays {
        // SAFETY: Windows serializes environment access; nothing else
        // reads these variables before the loader does.
        unsafe {
            std::env::set_var("VK_LOADER_LAYERS_DISABLE", "~implicit~");
            std::env::set_var("VK_LOADER_LAYERS_ALLOW", "VK_LAYER_NV_*,VK_LAYER_AMD_*,VK_LAYER_INTEL_*");
        }
        crate::info!("overlays = off: other programs' Vulkan layers are not loaded (the graphics drivers' own are)");
    }
}

/// The implicit layer whose library is the module `file` (a file name, as
/// a fault's module), if one is.
pub fn layer_of(file: &str) -> Option<&'static Layer> {
    LAYERS.get()?.iter().find(|l| l.library.rsplit(['\\', '/']).next().is_some_and(|f| f.eq_ignore_ascii_case(file)))
}

fn scan() -> Vec<Layer> {
    use windows_sys::Win32::System::Registry::{HKEY, HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE, KEY_READ, REG_DWORD, RegCloseKey, RegEnumValueW, RegOpenKeyExW};
    let mut out = Vec::new();
    let sub: Vec<u16> = "SOFTWARE\\Khronos\\Vulkan\\ImplicitLayers\0".encode_utf16().collect();
    for root in [HKEY_LOCAL_MACHINE, HKEY_CURRENT_USER] {
        let mut hk: HKEY = std::ptr::null_mut();
        // SAFETY: plain registry reads into local buffers.
        if unsafe { RegOpenKeyExW(root, sub.as_ptr(), 0, KEY_READ, &mut hk) } != 0 {
            continue;
        }
        for i in 0..256u32 {
            let mut name = [0u16; 1024];
            let mut name_len = name.len() as u32;
            let mut ty = 0u32;
            let mut data = 0u32;
            let mut data_len = 4u32;
            let r = unsafe { RegEnumValueW(hk, i, name.as_mut_ptr(), &mut name_len, std::ptr::null(), &mut ty, &mut data as *mut u32 as *mut u8, &mut data_len) };
            // (ERROR_NO_MORE_ITEMS ends the list; a value that is not a
            // DWORD - ERROR_MORE_DATA - is skipped.)
            if r == 259 {
                break;
            }
            if r != 0 {
                continue;
            }
            // (A value of 0 enables the manifest.)
            if ty != REG_DWORD || data != 0 {
                continue;
            }
            let path = String::from_utf16_lossy(&name[..name_len as usize]);
            out.extend(read_manifest(&path));
        }
        unsafe { RegCloseKey(hk) };
    }
    out
}

/// The layers a manifest (JSON) names: each "name" with its "library_path".
fn read_manifest(path: &str) -> Vec<Layer> {
    let Ok(text) = std::fs::read_to_string(path) else { return Vec::new() };
    let dir = std::path::Path::new(path).parent().map(|d| d.to_path_buf()).unwrap_or_default();
    let names: Vec<String> = json_strings(&text, "name").into_iter().filter(|n| n.starts_with("VK_LAYER")).collect();
    let libs = json_strings(&text, "library_path");
    names
        .into_iter()
        .zip(libs)
        .map(|(name, lib)| {
            let p = std::path::Path::new(&lib);
            let library = if p.is_absolute() { lib } else { dir.join(p).display().to_string() };
            Layer { name, library }
        })
        .collect()
}

/// The string values of every `"key": "..."` in JSON text, in order.
fn json_strings(text: &str, key: &str) -> Vec<String> {
    let pat = format!("\"{key}\"");
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(at) = rest.find(&pat) {
        rest = &rest[at + pat.len()..];
        let t = rest.trim_start();
        let Some(t) = t.strip_prefix(':') else { continue };
        let t = t.trim_start();
        let Some(t) = t.strip_prefix('"') else { continue };
        let mut s = String::new();
        let mut chars = t.chars();
        while let Some(c) = chars.next() {
            match c {
                '"' => break,
                '\\' => match chars.next() {
                    Some('n') => s.push('\n'),
                    Some('t') => s.push('\t'),
                    Some(o) => s.push(o),
                    None => break,
                },
                c => s.push(c),
            }
        }
        out.push(s);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_strings() {
        let j = r#"{ "file_format_version": "1.1.0", "layer": { "name": "VK_LAYER_OBS_hook", "type": "GLOBAL",
            "library_path": ".\\graphics-hook64.dll", "api_version": "1.3.0",
            "disable_environment": { "DISABLE_VULKAN_OBS_CAPTURE": "1" } } }"#;
        assert_eq!(json_strings(j, "name"), ["VK_LAYER_OBS_hook"]);
        assert_eq!(json_strings(j, "library_path"), [".\\graphics-hook64.dll"]);
    }
}
