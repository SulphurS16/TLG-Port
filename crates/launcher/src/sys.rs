//! System facts for the launcher: monitors and their display modes, the
//! Vulkan GPUs and whether they can run tlg, processes.

use crate::ui::{from_wide, wide};
use windows_sys::Win32::Foundation::{LPARAM, RECT};
use windows_sys::Win32::Graphics::Gdi::*;

// (The game's Vulkan requirements, the same file.)
#[path = "../../runtime/src/gpu/vk/requirements.rs"]
mod requirements;

#[derive(Clone, Debug)]
pub struct Monitor {
    /// The device name (\\.\DISPLAY1): what settings.txt stores.
    pub device: String,
    /// The monitor's model name where Windows has one.
    pub name: String,
    pub primary: bool,
    /// Current mode.
    pub width: u32,
    pub height: u32,
    pub hz: u32,
    /// Work area size (without the taskbar).
    pub work: (u32, u32),
    /// Display modes (width, height, Hz) at 32 bits per pixel, largest first.
    pub modes: Vec<(u32, u32, u32)>,
}

unsafe extern "system" fn enum_cb(h: HMONITOR, _dc: HDC, _r: *mut RECT, data: LPARAM) -> windows_sys::core::BOOL {
    unsafe { (*(data as *mut Vec<HMONITOR>)).push(h) };
    1
}

pub fn monitors() -> Vec<Monitor> {
    let mut hs: Vec<HMONITOR> = Vec::new();
    unsafe { EnumDisplayMonitors(std::ptr::null_mut(), std::ptr::null(), Some(enum_cb), &mut hs as *mut _ as LPARAM) };
    let mut out = Vec::new();
    for h in hs {
        let mut mi: MONITORINFOEXW = unsafe { std::mem::zeroed() };
        mi.monitorInfo.cbSize = std::mem::size_of::<MONITORINFOEXW>() as u32;
        if unsafe { GetMonitorInfoW(h, &mut mi as *mut _ as *mut MONITORINFO) } == 0 {
            continue;
        }
        let device = from_wide(&mi.szDevice);
        let dev_w = wide(&device);
        let mut dm: DEVMODEW = unsafe { std::mem::zeroed() };
        dm.dmSize = std::mem::size_of::<DEVMODEW>() as u16;
        unsafe { EnumDisplaySettingsW(dev_w.as_ptr(), ENUM_CURRENT_SETTINGS, &mut dm) };
        let (width, height, hz) = (dm.dmPelsWidth, dm.dmPelsHeight, dm.dmDisplayFrequency);
        let mut modes = Vec::new();
        let mut i = 0;
        loop {
            let mut m: DEVMODEW = unsafe { std::mem::zeroed() };
            m.dmSize = std::mem::size_of::<DEVMODEW>() as u16;
            if unsafe { EnumDisplaySettingsW(dev_w.as_ptr(), i, &mut m) } == 0 {
                break;
            }
            if m.dmBitsPerPel == 32 && m.dmPelsWidth >= 800 && m.dmPelsHeight >= 600 {
                let k = (m.dmPelsWidth, m.dmPelsHeight, m.dmDisplayFrequency);
                if !modes.contains(&k) {
                    modes.push(k);
                }
            }
            i += 1;
        }
        modes.sort_by(|a, b| (b.0 * b.1, b.2).cmp(&(a.0 * a.1, a.2)));
        // The monitor's model ("Generic PnP Monitor" etc.).
        let mut dd: DISPLAY_DEVICEW = unsafe { std::mem::zeroed() };
        dd.cb = std::mem::size_of::<DISPLAY_DEVICEW>() as u32;
        let name = if unsafe { EnumDisplayDevicesW(dev_w.as_ptr(), 0, &mut dd, 0) } != 0 { from_wide(&dd.DeviceString) } else { String::new() };
        let w = mi.monitorInfo.rcWork;
        out.push(Monitor {
            device,
            name,
            primary: mi.monitorInfo.dwFlags & 1 != 0,
            width,
            height,
            hz,
            work: ((w.right - w.left).max(0) as u32, (w.bottom - w.top).max(0) as u32),
            modes,
        });
    }
    out.sort_by_key(|m| !m.primary);
    out
}

#[derive(Clone, Debug)]
pub struct Gpu {
    pub name: String,
    pub discrete: bool,
    /// Why tlg cannot use it (None: usable; the same check as the game's).
    pub problem: Option<String>,
}

/// The Vulkan devices (an error if Vulkan itself is missing).
pub fn gpus() -> Result<Vec<Gpu>, String> {
    use ash::vk;
    unsafe {
        let entry = ash::Entry::load().map_err(|e| format!("Vulkan is not available ({e}); install or update the graphics driver"))?;
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let instance = entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None).map_err(|e| format!("Vulkan: {e}"))?;
        let mut out = Vec::new();
        for pd in instance.enumerate_physical_devices().unwrap_or_default() {
            let p = instance.get_physical_device_properties(pd);
            let name = std::ffi::CStr::from_ptr(p.device_name.as_ptr()).to_string_lossy().into_owned();
            // (The game's own requirement list.)
            let missing = requirements::missing(&instance, pd, &p);
            let problem = if p.api_version < vk::API_VERSION_1_3 {
                Some(format!("Vulkan {}.{} only (needs 1.3: a newer driver may help)", vk::api_version_major(p.api_version), vk::api_version_minor(p.api_version)))
            } else if !missing.is_empty() {
                Some(format!("lacks {} (a newer driver may add them)", missing.join(", ")))
            } else {
                None
            };
            out.push(Gpu { name, discrete: p.device_type == vk::PhysicalDeviceType::DISCRETE_GPU, problem });
        }
        instance.destroy_instance(None);
        Ok(out)
    }
}

pub fn process_alive(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::{CloseHandle, STILL_ACTIVE};
    use windows_sys::Win32::System::Threading::{GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};
    unsafe {
        let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if h.is_null() {
            return false;
        }
        let mut code = 0u32;
        let ok = GetExitCodeProcess(h, &mut code) != 0 && code == STILL_ACTIVE as u32;
        CloseHandle(h);
        ok
    }
}
