//! System services Shadow of the Colossus imports beyond the imported Bloodborne set:
//! disc map, internal module loading, dialogs, keyboard, touch gestures, PNG
//! encoding, offline sockets/HTTP helpers, display safe area, capture services.
//!
//! System dialogs have no on-screen UI yet: each one is answered at once with
//! OK / YES (the default a player would usually pick), its content logged. That is
//! an explicit stand-in, reported in the log, not an emulation of the system UI.

use crate::hle::{Registry, cstr};
use crate::{fixed, reg, sysv};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

const DIALOG_NONE: u32 = 0;
const DIALOG_INITIALIZED: u32 = 1;
const DIALOG_FINISHED: u32 = 3;
const ERR_NOT_INITIALIZED: i32 = 0x80B8_0003u32 as i32;
const ERR_NOT_FINISHED: i32 = 0x80B8_0005u32 as i32;
const ERR_ARG_NULL: i32 = 0x80B8_000Du32 as i32;

static ERROR_DIALOG: AtomicU32 = AtomicU32::new(DIALOG_NONE);

const DISCMAP_NO_BITMAP_INFO: i32 = 0x8110_0004u32 as i32;

/// PNG encoder instances (scePngEncCreate handles are guest memory blocks we were given).
static PNG_HANDLES: Mutex<Vec<u64>> = Mutex::new(Vec::new());

fn uri_unreserved(c: u8) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, b'-' | b'.' | b'_' | b'~')
}

sysv! {
    // ---- Sysmodule (internal loads by title-shipped modules): provided by HLE.
    fn sceSysmoduleLoadModuleInternalWithArg(id: u32, argc: i32, argv: u64, flags: u64, res: *mut i32) -> i32 {
        crate::info!("sceSysmoduleLoadModuleInternalWithArg({id:#x}) -> provided by HLE");
        if !res.is_null() { *res = 0; }
        0
    }
    fn sceSysmoduleLoadModuleInternal(id: u32) -> i32 {
        crate::info!("sceSysmoduleLoadModuleInternal({id:#x}) -> provided by HLE");
        0
    }

    // ---- DiscMap: a digital installation has no disc layout bitmap (all data on the drive).
    fn sceDiscMapIsRequestOnHDD(path: *const u8, offset: i64, size: i64, ret: *mut i32) -> i32 { DISCMAP_NO_BITMAP_INFO }
    /// fJgP+wqifno (name unknown): (path, offset, size, *flags, *r1, *r2)
    fn sceDiscMap_fJgP(path: *const u8, offset: i64, size: i64, flags: *mut i32, r1: *mut i32, r2: *mut i32) -> i32 {
        if !flags.is_null() { *flags = 0; }
        if !r1.is_null() { *r1 = 0; }
        if !r2.is_null() { *r2 = 0; }
        0
    }

    // ---- Common dialogs
    fn sceCommonDialogIsUsed() -> i32 { 0 }
    fn sceMsgDialogClose() -> i32 { 0 }
    /// {mode u32, result u32, buttonId u32, reserved[32]}
    fn sceMsgDialogGetResult(result: *mut u32) -> i32 {
        if result.is_null() { return ERR_ARG_NULL; }
        std::ptr::write_bytes(result as *mut u8, 0, 44);
        *result.add(1) = 0; // OK
        *result.add(2) = 1; // OK / YES
        crate::info!("sceMsgDialogGetResult -> OK/YES (auto-answered: no dialog UI)");
        0
    }
    fn sceSaveDataDialogClose() -> i32 { 0 }
    fn sceSaveDataDialogGetStatus() -> u32 { DIALOG_FINISHED }
    fn sceSaveDataDialogIsReadyToDisplay() -> i32 { 1 }
    fn sceSaveDataDialogProgressBarSetValue(target: u32, rate: u32) -> i32 { 0 }
    /// {mode u32, result u32, buttonId u32, -, dirName*, param*, userData*, reserved}
    fn sceSaveDataDialogGetResult(result: *mut u8) -> i32 {
        if result.is_null() { return ERR_ARG_NULL; }
        *(result.add(4) as *mut u32) = 0; // OK
        *(result.add(8) as *mut u32) = 1; // OK / YES
        crate::info!("sceSaveDataDialogGetResult -> OK/YES (auto-answered: no dialog UI)");
        0
    }
    fn sceErrorDialogInitialize() -> i32 {
        ERROR_DIALOG.store(DIALOG_INITIALIZED, Ordering::Relaxed);
        0
    }
    fn sceErrorDialogTerminate() -> i32 {
        ERROR_DIALOG.store(DIALOG_NONE, Ordering::Relaxed);
        0
    }
    /// {size, userId, errorCode, ...}
    fn sceErrorDialogOpen(param: *const i32) -> i32 {
        if ERROR_DIALOG.load(Ordering::Relaxed) == DIALOG_NONE { return ERR_NOT_INITIALIZED; }
        if param.is_null() { return ERR_ARG_NULL; }
        crate::warn!("system error dialog requested by the title: error code {:#010x} (auto-dismissed: no dialog UI)", *param.add(2) as u32);
        ERROR_DIALOG.store(DIALOG_FINISHED, Ordering::Relaxed);
        0
    }
    fn sceErrorDialogUpdateStatus() -> u32 { ERROR_DIALOG.load(Ordering::Relaxed) }
    fn sceErrorDialogClose() -> i32 {
        if ERROR_DIALOG.load(Ordering::Relaxed) == DIALOG_NONE { return ERR_NOT_INITIALIZED; }
        ERROR_DIALOG.store(DIALOG_FINISHED, Ordering::Relaxed);
        0
    }

    // ---- Ime: a software keyboard for text entry. None is shown (no text entry
    // is part of normal play); the open is refused as with no keyboard available.
    fn sceImeKeyboardOpen(user: i32, param: *const u8) -> i32 {
        crate::info!("sceImeKeyboardOpen(user {user}) refused: no keyboard device emulated");
        0x8080_0001u32 as i32
    }
    fn sceImeKeyboardClose(user: i32) -> i32 { 0x8080_0003u32 as i32 }
    fn sceImeUpdate(handler: u64) -> i32 { 0 }

    // ---- SystemGesture: touch-pad gestures. Touch input is not emulated: the
    // recognizers report no events.
    fn sceSystemGestureOpen(input_type: i32, param: *const u8) -> i32 { 1 }
    fn sceSystemGestureClose(h: i32) -> i32 { 0 }
    fn sceSystemGestureCreateTouchRecognizer(h: i32, rec: *mut u8, kind: i32, rect: *const u8, param: *const u8) -> i32 { 0 }
    fn sceSystemGestureInitializePrimitiveTouchRecognizer(h: i32, param: *const u8) -> i32 { 0 }
    fn sceSystemGestureFinalizePrimitiveTouchRecognizer(h: i32) -> i32 { 0 }
    fn sceSystemGestureUpdatePrimitiveTouchRecognizer(h: i32, data: *const u8) -> i32 { 0 }
    fn sceSystemGestureUpdateTouchRecognizer(h: i32, rec: *mut u8) -> i32 { 0 }
    fn sceSystemGestureGetTouchEventsCount(h: i32, rec: *const u8) -> i32 { 0 }
    fn sceSystemGestureGetTouchEventByIndex(h: i32, rec: *const u8, index: u32, ev: *mut u8) -> i32 { 0x8086_0004u32 as i32 }

    // ---- PngEnc: real PNG encoding of RGBA8 / BGRA8 images.
    /// {this_size, attribute, max_image_width, max_filter_number}
    fn scePngEncQueryMemorySize(param: *const u32) -> i32 {
        if param.is_null() { return 0x8069_0002u32 as i32; }
        0x1000
    }
    fn scePngEncCreate(param: *const u32, mem: u64, size: u32, handle: *mut u64) -> i32 {
        if param.is_null() || mem == 0 || handle.is_null() { return 0x8069_0002u32 as i32; }
        PNG_HANDLES.lock().push(mem);
        *handle = mem;
        0
    }
    fn scePngEncDelete(handle: u64) -> i32 {
        PNG_HANDLES.lock().retain(|&h| h != handle);
        0
    }
    /// {image*, png*, image_size, png_size, width, height, pitch (pixels), pixel_format, color_space, bit_depth, ...}
    fn scePngEncEncode(handle: u64, p: *const u8, out: *mut u32) -> i32 {
        if p.is_null() || !PNG_HANDLES.lock().contains(&handle) { return 0x8069_0002u32 as i32; }
        let src = *(p as *const u64) as *const u8;
        let dst = *(p.add(8) as *const u64) as *mut u8;
        let dst_size = *(p.add(20) as *const u32) as usize;
        let (w, h, pitch) = (*(p.add(24) as *const u32), *(p.add(28) as *const u32), *(p.add(32) as *const u32));
        let fmt = *(p.add(36) as *const u32);
        if src.is_null() || dst.is_null() || w == 0 || h == 0 { return 0x8069_0002u32 as i32; }
        let mut px = vec![0u8; (w * h * 4) as usize];
        for y in 0..h as usize {
            let row = std::slice::from_raw_parts(src.add(y * pitch.max(w) as usize * 4), w as usize * 4);
            px[y * w as usize * 4..(y + 1) * w as usize * 4].copy_from_slice(row);
        }
        if fmt == 1 {
            for c in px.chunks_mut(4) { c.swap(0, 2); } // BGRA -> RGBA
        }
        let png = crate::gpu::texdump::png(w, h, &px);
        if png.len() > dst_size { return 0x8069_0006u32 as i32; } // output buffer too small
        std::ptr::copy_nonoverlapping(png.as_ptr(), dst, png.len());
        if !out.is_null() { *out = png.len() as u32; *out.add(1) = h; }
        png.len() as i32
    }

    // ---- Offline networking. No network is emulated: sockets cannot be created.
    fn socket(domain: i32, kind: i32, proto: i32) -> i32 {
        static N: AtomicU32 = AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::info!("socket({domain}, {kind}, {proto}) -> ENETDOWN (offline: no network emulated)");
        }
        crate::hle::kernel::set_errno(50); // ENETDOWN
        -1
    }
    fn bind(s: i32, addr: u64, len: u32) -> i32 { crate::hle::kernel::set_errno(9); -1 }
    fn listen(s: i32, backlog: i32) -> i32 { crate::hle::kernel::set_errno(9); -1 }
    fn accept(s: i32, addr: u64, len: u64) -> i32 { crate::hle::kernel::set_errno(9); -1 }
    fn connect(s: i32, addr: u64, len: u32) -> i32 { crate::hle::kernel::set_errno(9); -1 }
    fn send(s: i32, buf: u64, len: u64, flags: i32) -> i64 { crate::hle::kernel::set_errno(9); -1 }
    fn recv(s: i32, buf: u64, len: u64, flags: i32) -> i64 { crate::hle::kernel::set_errno(9); -1 }
    fn setsockopt(s: i32, level: i32, name: i32, val: u64, len: u32) -> i32 { crate::hle::kernel::set_errno(9); -1 }
    fn shutdown(s: i32, how: i32) -> i32 { crate::hle::kernel::set_errno(9); -1 }
    fn select(n: i32, r: *mut u8, w: *mut u8, e: *mut u8, timeout: *const i64) -> i32 {
        if n > 0 { crate::hle::kernel::set_errno(9); return -1; }
        if !timeout.is_null() {
            let us = (*timeout).max(0) as u64 * 1_000_000 + (*timeout.add(1)).max(0) as u64;
            crate::hle::kernel::time::sleep_us(us);
        }
        0
    }
    fn sceNetGetSockInfo(s: i32, info: *mut u8, n: i32, flags: i32) -> i32 { 0 }
    fn sceHttpGetAllResponseHeaders(req: i32, header: *mut u64, size: *mut u64) -> i32 { 0x8043_1013u32 as i32 } // before send
    /// Percent-encodes `src` into `out` (RFC 3986 unreserved characters kept).
    fn sceHttpUriEscape(out: *mut u8, require: *mut u64, prepare: u64, src: *const u8) -> i32 {
        if src.is_null() { return 0x8043_1002u32 as i32; }
        let s = std::ffi::CStr::from_ptr(src as *const std::ffi::c_char).to_bytes();
        let mut e = Vec::with_capacity(s.len() * 3 + 1);
        for &c in s {
            if uri_unreserved(c) { e.push(c) } else { e.extend_from_slice(format!("%{c:02X}").as_bytes()) }
        }
        e.push(0);
        if !require.is_null() { *require = e.len() as u64; }
        if out.is_null() { return 0; }
        if (prepare as usize) < e.len() { return 0x8043_1022u32 as i32; } // out of memory
        std::ptr::copy_nonoverlapping(e.as_ptr(), out, e.len());
        0
    }

    // ---- System service
    /// {f32 ratio, reserved[128]}: the whole picture is safe on a PC display.
    fn sceSystemServiceGetDisplaySafeAreaInfo(info: *mut f32) -> i32 {
        if info.is_null() { return 0x8000_0001u32 as i32; }
        *info = 1.0;
        0
    }
    fn sceSystemServiceLoadExec(path: *const u8, argv: *const u64) -> i32 {
        crate::error!("sceSystemServiceLoadExec('{}'): launching another executable is not supported", cstr(path));
        crate::fatal_exit(3, "sceSystemServiceLoadExec");
    }

    // ---- Trophies
    fn sceNpTrophyDestroyContext(ctx: i32) -> i32 { 0 }
    fn sceNpTrophyDestroyHandle(h: i32) -> i32 { 0 }
}

fixed! {
    // Console capture and sharing services, not provided on a PC.
    sceScreenShotEnable => 0,
    sceScreenShotDisable => 0,
    sceVideoRecordingSetInfo => 0,
    sceGameLiveStreamingEnableLiveStreaming => 0,
    // (type, thread stack size, priority, affinity): the share menu service is not provided.
    sceShareUtilityInitializeEx2 => 0,
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceSysmodule", [sceSysmoduleLoadModuleInternalWithArg, sceSysmoduleLoadModuleInternal]);
    reg!(r, "libSceDiscMap", [sceDiscMapIsRequestOnHDD]);
    r.func_nid("libSceDiscMap", "fJgP+wqifno", "sceDiscMap_fJgP+wqifno", sceDiscMap_fJgP as *const () as usize as u64);
    reg!(r, "libSceCommonDialog", [sceCommonDialogIsUsed]);
    reg!(r, "libSceMsgDialog", [sceMsgDialogClose, sceMsgDialogGetResult]);
    reg!(r, "libSceSaveDataDialog", [
        sceSaveDataDialogClose, sceSaveDataDialogGetStatus, sceSaveDataDialogIsReadyToDisplay,
        sceSaveDataDialogProgressBarSetValue, sceSaveDataDialogGetResult,
    ]);
    reg!(r, "libSceErrorDialog", [
        sceErrorDialogInitialize, sceErrorDialogTerminate, sceErrorDialogOpen, sceErrorDialogUpdateStatus,
        sceErrorDialogClose,
    ]);
    reg!(r, "libSceIme", [sceImeKeyboardOpen, sceImeKeyboardClose, sceImeUpdate]);
    reg!(r, "libSceSystemGesture", [
        sceSystemGestureOpen, sceSystemGestureClose, sceSystemGestureCreateTouchRecognizer,
        sceSystemGestureInitializePrimitiveTouchRecognizer, sceSystemGestureFinalizePrimitiveTouchRecognizer,
        sceSystemGestureUpdatePrimitiveTouchRecognizer, sceSystemGestureUpdateTouchRecognizer,
        sceSystemGestureGetTouchEventsCount, sceSystemGestureGetTouchEventByIndex,
    ]);
    reg!(r, "libScePngEnc", [scePngEncQueryMemorySize, scePngEncCreate, scePngEncDelete, scePngEncEncode]);
    reg!(r, "libScePosix", [socket, bind, listen, accept, connect, send, recv, setsockopt, shutdown, select]);
    reg!(r, "libSceNet", [sceNetGetSockInfo]);
    reg!(r, "libSceHttp", [sceHttpGetAllResponseHeaders, sceHttpUriEscape]);
    reg!(r, "libSceSystemService", [sceSystemServiceGetDisplaySafeAreaInfo, sceSystemServiceLoadExec]);
    reg!(r, "libSceNpTrophy", [sceNpTrophyDestroyContext, sceNpTrophyDestroyHandle]);
    reg!(r, "libSceScreenShot", [sceScreenShotEnable, sceScreenShotDisable]);
    reg!(r, "libSceVideoRecording", [sceVideoRecordingSetInfo]);
    reg!(r, "libSceGameLiveStreaming", [sceGameLiveStreamingEnableLiveStreaming]);
    reg!(r, "libSceShareUtility", [sceShareUtilityInitializeEx2]);
}
