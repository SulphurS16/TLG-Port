//! DualShock 4 and DualSense controllers read directly over HID (USB and
//! Bluetooth), so they work without an XInput wrapper: buttons, sticks and
//! analog triggers in the DS4 layout the title expects. A thread looks for a
//! Sony controller every two seconds while none is open and reads its input
//! reports; `state()` returns the latest one (None when no controller
//! reported recently). TLG_HID=0 disables it.
//!
//! Report layouts (public controller documentation):
//! - DS4 USB report 0x01 and Bluetooth "reduced" 0x01, data at byte 1;
//!   Bluetooth full report 0x11, data at byte 3: LX LY RX RY, buttons
//!   (hat in the low nibble, square cross circle triangle), L1 R1 L2 R2
//!   share options L3 R3, PS and touchpad click, L2 and R2 analog.
//! - DualSense USB report 0x01, data at byte 1, and Bluetooth 0x31, data at
//!   byte 2: LX LY RX RY, L2 R2 analog, counter, then the buttons as above.
//!   Its Bluetooth "simple" report 0x01 uses the DS4 order.

use std::sync::atomic::{AtomicBool, Ordering};

/// What the pad reports, in `input::PadState` terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct HidState {
    pub buttons: u32,
    pub lx: u8,
    pub ly: u8,
    pub rx: u8,
    pub ry: u8,
    pub l2: u8,
    pub r2: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    Ds4,
    DualSense,
}

const SONY: u16 = 0x054C;

/// Sony controllers by product id.
pub fn model(vid: u16, pid: u16) -> Option<Model> {
    if vid != SONY {
        return None;
    }
    match pid {
        0x05C4 | 0x09CC | 0x0BA0 => Some(Model::Ds4),
        0x0CE6 | 0x0DF2 => Some(Model::DualSense),
        _ => None,
    }
}

/// The d-pad hat value (0 = up, clockwise, 8 = released) as DS4 buttons.
fn hat(v: u8) -> u32 {
    use crate::input::{BTN_DOWN, BTN_LEFT, BTN_RIGHT, BTN_UP};
    match v & 0xF {
        0 => BTN_UP,
        1 => BTN_UP | BTN_RIGHT,
        2 => BTN_RIGHT,
        3 => BTN_DOWN | BTN_RIGHT,
        4 => BTN_DOWN,
        5 => BTN_DOWN | BTN_LEFT,
        6 => BTN_LEFT,
        7 => BTN_UP | BTN_LEFT,
        _ => 0,
    }
}

/// The three button bytes shared by both controllers (face buttons and hat,
/// shoulder/stick/menu buttons, PS/touchpad).
fn buttons(b0: u8, b1: u8, b2: u8) -> u32 {
    use crate::input::*;
    let mut v = hat(b0);
    let bit = |byte: u8, n: u8, btn: u32| if byte & (1 << n) != 0 { btn } else { 0 };
    v |= bit(b0, 4, BTN_SQUARE) | bit(b0, 5, BTN_CROSS) | bit(b0, 6, BTN_CIRCLE) | bit(b0, 7, BTN_TRIANGLE);
    v |= bit(b1, 0, BTN_L1) | bit(b1, 1, BTN_R1) | bit(b1, 2, BTN_L2) | bit(b1, 3, BTN_R2);
    // (Share/Create and PS belong to the system on a PS4: not passed on.)
    v |= bit(b1, 5, BTN_OPTIONS) | bit(b1, 6, BTN_L3) | bit(b1, 7, BTN_R3);
    v |= bit(b2, 1, BTN_TOUCHPAD);
    v
}

/// DS4 order from `d` (LX LY RX RY B0 B1 B2 L2 R2).
fn ds4_layout(d: &[u8]) -> Option<HidState> {
    let d = d.get(..9)?;
    Some(HidState { lx: d[0], ly: d[1], rx: d[2], ry: d[3], buttons: buttons(d[4], d[5], d[6]), l2: d[7], r2: d[8] })
}

/// DualSense order from `d` (LX LY RX RY L2 R2 counter B0 B1 B2).
fn dualsense_layout(d: &[u8]) -> Option<HidState> {
    let d = d.get(..10)?;
    Some(HidState { lx: d[0], ly: d[1], rx: d[2], ry: d[3], l2: d[4], r2: d[5], buttons: buttons(d[7], d[8], d[9]) })
}

/// Parses one input report (`report[0]` = report id). `usb`: the device's
/// input reports are the 64-byte USB ones (Bluetooth: 78 bytes).
pub fn parse(model: Model, usb: bool, report: &[u8]) -> Option<HidState> {
    let id = *report.first()?;
    match (model, id) {
        (Model::Ds4, 0x01) => ds4_layout(report.get(1..)?),
        (Model::Ds4, 0x11) => ds4_layout(report.get(3..)?),
        (Model::DualSense, 0x01) if usb => dualsense_layout(report.get(1..)?),
        (Model::DualSense, 0x01) => ds4_layout(report.get(1..)?),
        (Model::DualSense, 0x31) => dualsense_layout(report.get(2..)?),
        _ => None,
    }
}

static LATEST: parking_lot::Mutex<Option<(HidState, std::time::Instant)>> = parking_lot::Mutex::new(None);
static STARTED: AtomicBool = AtomicBool::new(false);

/// The connected controller's latest state (reported within the last 0.5 s).
pub fn state() -> Option<HidState> {
    if !STARTED.swap(true, Ordering::Relaxed) && std::env::var("TLG_HID").map(|v| v != "0").unwrap_or(true) {
        std::thread::Builder::new().name("hid-pad".into()).spawn(reader).ok();
    }
    LATEST.lock().filter(|(_, t)| t.elapsed() < std::time::Duration::from_millis(500)).map(|(s, _)| s)
}

fn reader() {
    crate::log::set_thread_name("hid-pad");
    loop {
        match unsafe { win::open_first() } {
            Some(dev) => {
                crate::info!("controller: {:?} over {} (HID {:04x}:{:04x})", dev.model, if dev.usb { "USB" } else { "Bluetooth" }, SONY, dev.pid);
                unsafe { win::read_loop(&dev) };
                crate::info!("controller: {:?} disconnected", dev.model);
                *LATEST.lock() = None;
            }
            None => std::thread::sleep(std::time::Duration::from_secs(2)),
        }
    }
}

mod win {
    use super::{HidState, LATEST, Model};
    use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
        DIGCF_DEVICEINTERFACE, DIGCF_PRESENT, SP_DEVICE_INTERFACE_DATA, SP_DEVICE_INTERFACE_DETAIL_DATA_W, SetupDiDestroyDeviceInfoList, SetupDiEnumDeviceInterfaces, SetupDiGetClassDevsW,
        SetupDiGetDeviceInterfaceDetailW,
    };
    use windows_sys::Win32::Devices::HumanInterfaceDevice::{HIDD_ATTRIBUTES, HIDP_CAPS, HidD_FreePreparsedData, HidD_GetAttributes, HidD_GetHidGuid, HidD_GetPreparsedData, HidP_GetCaps, PHIDP_PREPARSED_DATA};
    use windows_sys::Win32::Foundation::{CloseHandle, GENERIC_READ, GENERIC_WRITE, HANDLE, INVALID_HANDLE_VALUE, WAIT_OBJECT_0};
    use windows_sys::Win32::Storage::FileSystem::{CreateFileW, FILE_FLAG_OVERLAPPED, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING, ReadFile};
    use windows_sys::Win32::System::IO::{CancelIo, GetOverlappedResult, OVERLAPPED};
    use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

    pub struct Device {
        pub handle: HANDLE,
        pub model: Model,
        pub pid: u16,
        pub usb: bool,
        pub report_len: usize,
    }

    impl Drop for Device {
        fn drop(&mut self) {
            unsafe { CloseHandle(self.handle) };
        }
    }

    /// Opens the first present Sony controller.
    pub unsafe fn open_first() -> Option<Device> {
        unsafe {
            let mut guid = std::mem::zeroed();
            HidD_GetHidGuid(&mut guid);
            let set = SetupDiGetClassDevsW(&guid, std::ptr::null(), std::ptr::null_mut(), DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
            if set as isize == -1 {
                return None;
            }
            let mut found = None;
            for i in 0..256 {
                let mut ifd: SP_DEVICE_INTERFACE_DATA = std::mem::zeroed();
                ifd.cbSize = std::mem::size_of::<SP_DEVICE_INTERFACE_DATA>() as u32;
                if SetupDiEnumDeviceInterfaces(set, std::ptr::null(), &guid, i, &mut ifd) == 0 {
                    break;
                }
                let mut need = 0u32;
                SetupDiGetDeviceInterfaceDetailW(set, &ifd, std::ptr::null_mut(), 0, &mut need, std::ptr::null_mut());
                if need == 0 {
                    continue;
                }
                let mut buf = vec![0u64; (need as usize).div_ceil(8)];
                let detail = buf.as_mut_ptr() as *mut SP_DEVICE_INTERFACE_DETAIL_DATA_W;
                (*detail).cbSize = std::mem::size_of::<SP_DEVICE_INTERFACE_DETAIL_DATA_W>() as u32;
                if SetupDiGetDeviceInterfaceDetailW(set, &ifd, detail, need, std::ptr::null_mut(), std::ptr::null_mut()) == 0 {
                    continue;
                }
                let path = std::ptr::addr_of!((*detail).DevicePath) as *const u16;
                // (Query access first: keyboards and mice refuse read access.)
                let probe = CreateFileW(path, 0, FILE_SHARE_READ | FILE_SHARE_WRITE, std::ptr::null(), OPEN_EXISTING, 0, std::ptr::null_mut());
                if probe == INVALID_HANDLE_VALUE {
                    continue;
                }
                let mut attr: HIDD_ATTRIBUTES = std::mem::zeroed();
                attr.Size = std::mem::size_of::<HIDD_ATTRIBUTES>() as u32;
                let ok = HidD_GetAttributes(probe, &mut attr);
                CloseHandle(probe);
                let Some(model) = ok.then(|| super::model(attr.VendorID, attr.ProductID)).flatten() else { continue };
                let h = CreateFileW(path, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, std::ptr::null(), OPEN_EXISTING, FILE_FLAG_OVERLAPPED, std::ptr::null_mut());
                if h == INVALID_HANDLE_VALUE {
                    continue;
                }
                let mut pp: PHIDP_PREPARSED_DATA = 0;
                let mut len = 64usize;
                if HidD_GetPreparsedData(h, &mut pp) {
                    let mut caps: HIDP_CAPS = std::mem::zeroed();
                    if HidP_GetCaps(pp, &mut caps) >= 0 {
                        len = (caps.InputReportByteLength as usize).max(16);
                    }
                    HidD_FreePreparsedData(pp);
                }
                // USB reports are 64 bytes, Bluetooth ones 78 (or more).
                found = Some(Device { handle: h, model, pid: attr.ProductID, usb: len <= 64, report_len: len });
                break;
            }
            SetupDiDestroyDeviceInfoList(set);
            found
        }
    }

    /// Reads reports until the device goes away.
    pub unsafe fn read_loop(dev: &Device) {
        unsafe {
            let event = CreateEventW(std::ptr::null(), 1, 0, std::ptr::null());
            let mut buf = vec![0u8; dev.report_len.max(80)];
            let mut timeouts = 0;
            loop {
                let mut ov: OVERLAPPED = std::mem::zeroed();
                ov.hEvent = event;
                let mut n = 0u32;
                let ok = ReadFile(dev.handle, buf.as_mut_ptr(), dev.report_len as u32, &mut n, &mut ov) != 0;
                if !ok && windows_sys::Win32::Foundation::GetLastError() != windows_sys::Win32::Foundation::ERROR_IO_PENDING {
                    break;
                }
                if !ok {
                    if WaitForSingleObject(event, 1000) != WAIT_OBJECT_0 {
                        CancelIo(dev.handle);
                        GetOverlappedResult(dev.handle, &ov, &mut n, 1);
                        // (A controller with nothing to report sends nothing over
                        // Bluetooth for a while: only give up after a minute.)
                        timeouts += 1;
                        if timeouts > 60 {
                            break;
                        }
                        continue;
                    }
                    if GetOverlappedResult(dev.handle, &ov, &mut n, 0) == 0 {
                        break;
                    }
                }
                timeouts = 0;
                if let Some(s) = super::parse(dev.model, dev.usb, &buf[..n as usize]) {
                    let s: HidState = s;
                    *LATEST.lock() = Some((s, std::time::Instant::now()));
                }
            }
            CloseHandle(event);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::*;

    #[test]
    fn ds4_usb_report() {
        // Report 0x01: sticks, hat 2 (right) + cross, L1 + options, touchpad, L2/R2.
        let mut r = [0u8; 64];
        r[0] = 0x01;
        r[1..5].copy_from_slice(&[0x10, 0x80, 0xF0, 0x7F]);
        r[5] = 0x02 | 0x20;
        r[6] = 0x01 | 0x20;
        r[7] = 0x02;
        r[8] = 200;
        r[9] = 7;
        let s = parse(Model::Ds4, true, &r).unwrap();
        assert_eq!((s.lx, s.ly, s.rx, s.ry, s.l2, s.r2), (0x10, 0x80, 0xF0, 0x7F, 200, 7));
        assert_eq!(s.buttons, BTN_RIGHT | BTN_CROSS | BTN_L1 | BTN_OPTIONS | BTN_TOUCHPAD);
    }

    #[test]
    fn ds4_bluetooth_full_report() {
        let mut r = [0u8; 78];
        r[0] = 0x11;
        r[3..7].copy_from_slice(&[1, 2, 3, 4]);
        r[7] = 0x08 | 0x80; // released hat, triangle
        r[8] = 0x80; // R3
        r[10] = 9;
        r[11] = 10;
        let s = parse(Model::Ds4, false, &r).unwrap();
        assert_eq!((s.lx, s.ly, s.rx, s.ry, s.l2, s.r2), (1, 2, 3, 4, 9, 10));
        assert_eq!(s.buttons, BTN_TRIANGLE | BTN_R3);
    }

    #[test]
    fn dualsense_reports() {
        let mut usb = [0u8; 64];
        usb[0] = 0x01;
        usb[1..7].copy_from_slice(&[5, 6, 7, 8, 100, 101]);
        usb[8] = 0x07 | 0x10; // up-left, square
        usb[9] = 0x02 | 0x40; // R1, L3
        let s = parse(Model::DualSense, true, &usb).unwrap();
        assert_eq!((s.lx, s.ly, s.rx, s.ry, s.l2, s.r2), (5, 6, 7, 8, 100, 101));
        assert_eq!(s.buttons, BTN_UP | BTN_LEFT | BTN_SQUARE | BTN_R1 | BTN_L3);
        // Bluetooth 0x31: one byte further.
        let mut bt = [0u8; 78];
        bt[0] = 0x31;
        bt[2..11].copy_from_slice(&usb[1..10]);
        assert_eq!(parse(Model::DualSense, false, &bt), Some(s));
        // Bluetooth simple report 0x01: DS4 order.
        let mut simple = [0u8; 78];
        simple[0] = 0x01;
        simple[1..10].copy_from_slice(&[5, 6, 7, 8, 0x40, 0x00, 0x00, 1, 2]);
        let t = parse(Model::DualSense, false, &simple).unwrap();
        assert_eq!((t.l2, t.r2, t.buttons), (1, 2, BTN_CIRCLE | BTN_UP));
    }

    #[test]
    fn models_and_unknown_reports() {
        assert_eq!(model(0x054C, 0x09CC), Some(Model::Ds4));
        assert_eq!(model(0x054C, 0x0CE6), Some(Model::DualSense));
        assert_eq!(model(0x045E, 0x028E), None);
        assert_eq!(parse(Model::Ds4, true, &[0x05, 0, 0]), None);
        assert_eq!(parse(Model::Ds4, true, &[0x01, 0, 0]), None);
    }
}
