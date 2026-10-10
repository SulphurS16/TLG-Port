//! Application services: AppContent (+ param.sfo), UserService, SystemService, PlayGo.

use crate::hle::{Registry, cstr, write_cstr};
use crate::{fixed, reg, sysv};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, Clone)]
pub enum SfoValue {
    Int(u32),
    Str(String),
}

/// Parses an SFO (param.sfo) file: keys with integer or string values
/// (binary values as lossy strings). None for a malformed file.
pub fn parse_sfo(b: &[u8]) -> Option<HashMap<String, SfoValue>> {
    let rd32 = |o: usize| b.get(o..o + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap()));
    let rd16 = |o: usize| b.get(o..o + 2).map(|s| u16::from_le_bytes(s.try_into().unwrap()));
    if rd32(0)? != 0x4653_5000 {
        return None;
    }
    let (keys, data, n) = (rd32(8)? as usize, rd32(12)? as usize, rd32(16)? as usize);
    let mut m = HashMap::new();
    for i in 0..n.min(1024) {
        let e = 20 + i * 16;
        let ko = keys + rd16(e)? as usize;
        let fmt = rd16(e + 2)?;
        let len = rd32(e + 4)? as usize;
        let off = data + rd32(e + 12)? as usize;
        let kb = b.get(ko..)?;
        let key = String::from_utf8_lossy(&kb[..kb.iter().position(|&c| c == 0).unwrap_or(kb.len())]).into_owned();
        let v = if fmt == 0x0404 {
            SfoValue::Int(rd32(off)?)
        } else {
            let s = b.get(off..off + len)?;
            let end = s.iter().position(|&c| c == 0).unwrap_or(s.len());
            SfoValue::Str(String::from_utf8_lossy(&s[..end]).into_owned())
        };
        m.insert(key, v);
    }
    Some(m)
}

/// Parses `/app0/sce_sys/param.sfo` once.
pub fn param_sfo() -> &'static HashMap<String, SfoValue> {
    static SFO: OnceLock<HashMap<String, SfoValue>> = OnceLock::new();
    SFO.get_or_init(|| {
        let Some(p) = crate::hle::kernel::fs::host_path_existing("/app0/sce_sys/param.sfo") else {
            crate::warn!("param.sfo not found");
            return HashMap::new();
        };
        let Ok(b) = std::fs::read(&p) else { return HashMap::new() };
        let Some(m) = parse_sfo(&b) else {
            crate::warn!("param.sfo has a bad header");
            return HashMap::new();
        };
        crate::info!("param.sfo: TITLE_ID={:?} APP_VER={:?} CATEGORY={:?}", m.get("TITLE_ID"), m.get("APP_VER"), m.get("CATEGORY"));
        m
    })
}

fn sfo_int(key: &str) -> Option<u32> {
    match param_sfo().get(key) {
        Some(SfoValue::Int(v)) => Some(*v),
        _ => None,
    }
}

pub const USER_ID: i32 = 1;
static USER_EVENT_SENT: AtomicBool = AtomicBool::new(false);
static PLAYGO_CHUNKS: Mutex<Option<u32>> = Mutex::new(None);

fn playgo_chunk_count() -> u32 {
    let mut g = PLAYGO_CHUNKS.lock();
    if let Some(n) = *g {
        return n;
    }
    // playgo-chunk.dat: "plgo" header; chunk count at 0x0A (u16).
    let n = crate::hle::kernel::fs::host_path_existing("/app0/sce_sys/playgo-chunk.dat")
        .and_then(|p| std::fs::read(p).ok())
        .filter(|b| b.len() > 0x10 && &b[0..4] == b"plgo")
        .map(|b| u16::from_le_bytes([b[0x0A], b[0x0B]]) as u32)
        .filter(|&n| n > 0)
        .unwrap_or(1);
    crate::info!("PlayGo: {n} chunk(s), all reported installed");
    *g = Some(n);
    n
}

sysv! {
    // ---- AppContent
    fn sceAppContentInitialize(init: *const u8, boot: *mut u8) -> i32 {
        if !boot.is_null() { std::ptr::write_bytes(boot, 0, 40); }
        param_sfo();
        0
    }
    fn sceAppContentAppParamGetInt(id: i32, value: *mut i32) -> i32 {
        if value.is_null() { return 0x80D9_0002u32 as i32; }
        *value = match id {
            0 => 3, // SKU_FLAG_FULL
            1..=4 => sfo_int(&format!("USER_DEFINED_PARAM_{id}")).unwrap_or(0) as i32,
            _ => return 0x80D9_0002u32 as i32,
        };
        crate::debug!("sceAppContentAppParamGetInt({id}) = {}", *value);
        0
    }
    fn sceAppContentGetAddcontInfoList(label: u32, list: *mut u8, num: u32, hit: *mut u32) -> i32 {
        if !hit.is_null() { *hit = 0; }
        crate::info!("sceAppContentGetAddcontInfoList: no additional content installed");
        0
    }

    // ---- UserService
    fn sceUserServiceInitialize(params: *const u8) -> i32 { 0 }
    fn sceUserServiceTerminate() -> i32 { 0 }
    fn sceUserServiceGetInitialUser(user: *mut i32) -> i32 {
        if user.is_null() { return 0x8096_0003u32 as i32; }
        *user = USER_ID;
        0
    }
    fn sceUserServiceGetLoginUserIdList(list: *mut i32) -> i32 {
        if list.is_null() { return 0x8096_0003u32 as i32; }
        *list = USER_ID;
        for i in 1..4 { *list.add(i) = -1; }
        0
    }
    fn sceUserServiceGetUserName(user: i32, name: *mut u8, size: u64) -> i32 {
        write_cstr(name, size as usize, "Hunter");
        0
    }
    fn sceUserServiceGetEvent(ev: *mut i32) -> i32 {
        if !USER_EVENT_SENT.swap(true, Ordering::SeqCst) {
            *ev = 0; // LOGIN
            *ev.add(1) = USER_ID;
            return 0;
        }
        0x8096_0007u32 as i32 // NO_EVENT
    }

    // ---- SystemService
    fn sceSystemServiceGetStatus(status: *mut u8) -> i32 {
        // SceSystemServiceStatus: int32 eventNum + 5 bools (+ reserved, left untouched).
        if !status.is_null() { std::ptr::write_bytes(status, 0, 9); }
        0
    }
    fn sceSystemServiceReceiveEvent(ev: *mut u8) -> i32 { 0x80A1_0003u32 as i32 }
    fn sceSystemServiceParamGetInt(id: i32, value: *mut i32) -> i32 {
        if value.is_null() { return 0x80A1_0002u32 as i32; }
        *value = match id {
            1 => 1,    // LANG: English (United States)
            2 => 0,    // DATE_FORMAT: YYYYMMDD
            3 => 1,    // TIME_FORMAT: 24 hour
            4 => 0,    // TIME_ZONE (minutes)
            5 => 0,    // SUMMERTIME
            7 => 0,    // GAME_PARENTAL_LEVEL: off
            1000 => 1, // ENTER_BUTTON_ASSIGN: cross
            _ => {
                crate::warn!("sceSystemServiceParamGetInt: unknown id {id}");
                return 0x80A1_0002u32 as i32;
            }
        };
        0
    }
    fn sceSystemServiceParamGetString(id: i32, buf: *mut u8, size: u64) -> i32 {
        write_cstr(buf, size as usize, "PS4");
        0
    }
    fn sceSystemServiceHideSplashScreen() -> i32 {
        crate::info!("sceSystemServiceHideSplashScreen");
        0
    }

    // ---- PlayGo
    fn scePlayGoOpen(handle: *mut i32, param: *const u8) -> i32 {
        if !handle.is_null() { *handle = 1; }
        playgo_chunk_count();
        0
    }
    fn scePlayGoGetLocus(handle: i32, ids: *const u16, n: u32, loci: *mut i8) -> i32 {
        for i in 0..n as usize { *loci.add(i) = 3; } // LOCAL_FAST
        0
    }
    fn scePlayGoGetChunkId(handle: i32, out: *mut u16, n: u32, entries: *mut u32) -> i32 {
        let count = playgo_chunk_count();
        if out.is_null() {
            if !entries.is_null() { *entries = count; }
            return 0;
        }
        let k = count.min(n);
        for i in 0..k { *out.add(i as usize) = i as u16; }
        if !entries.is_null() { *entries = k; }
        0
    }
}

fixed! {
    scePlayGoInitialize => 0,
    scePlayGoSetInstallSpeed => 0,
    sceSystemServiceLaunchWebBrowser => 0x80A1_0002u32,
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceAppContent", [sceAppContentInitialize, sceAppContentAppParamGetInt, sceAppContentGetAddcontInfoList]);
    reg!(r, "libSceUserService", [
        sceUserServiceInitialize, sceUserServiceTerminate, sceUserServiceGetInitialUser,
        sceUserServiceGetLoginUserIdList, sceUserServiceGetUserName, sceUserServiceGetEvent,
    ]);
    reg!(r, "libSceSystemService", [
        sceSystemServiceGetStatus, sceSystemServiceReceiveEvent, sceSystemServiceParamGetInt,
        sceSystemServiceParamGetString, sceSystemServiceHideSplashScreen, sceSystemServiceLaunchWebBrowser,
    ]);
    reg!(r, "libScePlayGo", [scePlayGoInitialize, scePlayGoOpen, scePlayGoGetLocus, scePlayGoGetChunkId, scePlayGoSetInstallSpeed]);
    let _ = cstr; // shared helper import
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds an SFO: header, 16-byte entries, key table, data table.
    fn sfo(entries: &[(&str, u16, Vec<u8>)]) -> Vec<u8> {
        let keys_off = 20 + 16 * entries.len();
        let mut keys = Vec::new();
        let mut data = Vec::new();
        let mut ents = Vec::new();
        for (k, fmt, v) in entries {
            ents.extend_from_slice(&(keys.len() as u16).to_le_bytes());
            ents.extend_from_slice(&fmt.to_le_bytes());
            ents.extend_from_slice(&(v.len() as u32).to_le_bytes());
            ents.extend_from_slice(&(v.len() as u32).to_le_bytes());
            ents.extend_from_slice(&(data.len() as u32).to_le_bytes());
            keys.extend_from_slice(k.as_bytes());
            keys.push(0);
            data.extend_from_slice(v);
        }
        while keys.len() % 4 != 0 {
            keys.push(0);
        }
        let mut b = Vec::new();
        b.extend_from_slice(&0x4653_5000u32.to_le_bytes());
        b.extend_from_slice(&0x0101u32.to_le_bytes());
        b.extend_from_slice(&(keys_off as u32).to_le_bytes());
        b.extend_from_slice(&((keys_off + keys.len()) as u32).to_le_bytes());
        b.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        b.extend(ents);
        b.extend(keys);
        b.extend(data);
        b
    }

    #[test]
    fn sfo_values_and_malformed_files() {
        let b = sfo(&[
            ("MAINTITLE", 0x0204, b"Bloodborne\0".to_vec()),
            ("SAVEDATA_LIST_PARAM", 0x0404, 7u32.to_le_bytes().to_vec()),
        ]);
        let m = parse_sfo(&b).unwrap();
        assert!(matches!(m.get("MAINTITLE"), Some(SfoValue::Str(s)) if s == "Bloodborne"));
        assert!(matches!(m.get("SAVEDATA_LIST_PARAM"), Some(SfoValue::Int(7))));
        // Truncated files and bad magic are rejected instead of panicking.
        for n in [0, 8, 30, b.len() - 1] {
            assert!(parse_sfo(&b[..n]).is_none(), "truncated at {n}");
        }
        let mut bad = b.clone();
        bad[0] = 1;
        assert!(parse_sfo(&bad).is_none());
    }
}
