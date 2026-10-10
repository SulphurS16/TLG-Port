//! Network, PSN (Np*) and system dialogs, emulating an offline console:
//! the user is signed out of PSN, the network is disconnected, and dialogs
//! complete immediately (their text is logged).

use crate::hle::{Registry, cstr};
use crate::{fixed, reg, sysv};
use std::sync::atomic::{AtomicI32, Ordering};

pub const NP_ERROR_SIGNED_OUT: u32 = 0x8055_0006;
const NET_ERROR_ENETDOWN: u32 = 0x8041_0132;
const NET_ERROR_EBADF: u32 = 0x8041_0109;
const NETCTL_ERROR_NOT_CONNECTED: u32 = 0x8041_2108;
const HTTP_ERROR_NETWORK: u32 = 0x8043_1063;
const DIALOG_FINISHED: i32 = 3;

static NEXT_ID: AtomicI32 = AtomicI32::new(1);
fn next_id() -> i32 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

static NET_ERRNO: AtomicI32 = AtomicI32::new(0);

// ---- Locally recorded trophies (one unlocked id per line; no PSN sync).
static TROPHY_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

fn trophy_unlocked() -> Vec<i32> {
    let _g = TROPHY_LOCK.lock();
    std::fs::read_to_string(super::savedata::trophy_file())
        .map(|t| t.lines().filter_map(|l| l.trim().parse().ok()).collect())
        .unwrap_or_default()
}

/// Records an unlock; false when it was already recorded.
fn trophy_unlock(id: i32) -> bool {
    let mut ids = trophy_unlocked();
    if ids.contains(&id) {
        return false;
    }
    ids.push(id);
    ids.sort();
    let _g = TROPHY_LOCK.lock();
    let text: String = ids.iter().map(|i| format!("{i}\n")).collect();
    let f = super::savedata::trophy_file();
    let tmp = f.with_extension("tmp");
    if std::fs::write(&tmp, text).and_then(|_| std::fs::rename(&tmp, &f)).is_err() {
        crate::warn!("trophy {id}: could not record the unlock in {}", f.display());
    }
    true
}

/// Trophies in the title's set: the TROPnnn.PNG icon entries of the plain
/// trophy00.trp entry table (magic dca24d00, count @0x10, 0x40-byte entries @0x60).
fn trophy_count() -> u32 {
    static N: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        let Some(p) = crate::hle::kernel::fs::host_path_existing("/app0/sce_sys/trophy/trophy00.trp") else { return 0 };
        let Ok(b) = std::fs::read(&p) else { return 0 };
        if b.len() < 0x60 || b[0..4] != [0xdc, 0xa2, 0x4d, 0x00] { return 0; }
        let n = u32::from_be_bytes(b[0x10..0x14].try_into().unwrap()) as usize;
        let count = (0..n)
            .filter_map(|i| b.get(0x60 + i * 0x40..0x60 + i * 0x40 + 32))
            .map(|name| String::from_utf8_lossy(name.split(|&c| c == 0).next().unwrap_or(&[])).to_string())
            .filter(|s| s.starts_with("TROP") && s.ends_with(".PNG") && s[4..s.len() - 4].chars().all(|c| c.is_ascii_digit()))
            .count() as u32;
        crate::info!("trophy set: {count} trophies ({})", p.display());
        count
    })
}

sysv! {
    // ---- libSceNpManager
    fn sceNpGetState(user: i32, state: *mut i32) -> i32 {
        if !state.is_null() { *state = 1; } // SIGNED_OUT
        0
    }
    fn sceNpGetGamePresenceStatus(user: i32, status: *mut i32) -> i32 {
        if !status.is_null() { *status = 0; }
        0
    }
    fn sceNpCreateAsyncRequest(param: *const u8) -> i32 { next_id() }
    fn sceNpPollAsync(req: i32, result: *mut i32) -> i32 {
        if !result.is_null() { *result = NP_ERROR_SIGNED_OUT as i32; }
        0 // finished
    }
    fn sceNpCheckPlus(req: i32, param: *const u8, result: *mut u8) -> i32 {
        if !result.is_null() { *result = 0; } // authorized = false
        0
    }
    fn sceNpRegisterStateCallback(cb: u64, arg: u64) -> i32 { 1 }
    fn sceNpRegisterGamePresenceCallback(cb: u64, arg: u64) -> i32 { 1 }
    fn sceNpRegisterPlusEventCallback(cb: u64, arg: u64) -> i32 { 0 }

    // ---- libSceNpCommon
    fn sceNpCmpNpId(a: *const u8, b: *const u8) -> i32 {
        if a.is_null() || b.is_null() { return 0x8055_0003u32 as i32; }
        let x = std::slice::from_raw_parts(a, 16);
        let y = std::slice::from_raw_parts(b, 16);
        if x == y { 0 } else { 0x8055_0009u32 as i32 }
    }
    fn sceNpCmpOnlineId(a: *const u8, b: *const u8) -> i32 {
        if a.is_null() || b.is_null() { return 0x8055_0003u32 as i32; }
        let x = std::slice::from_raw_parts(a, 16);
        let y = std::slice::from_raw_parts(b, 16);
        if x == y { 0 } else { 0x8055_0009u32 as i32 }
    }

    // ---- libSceNpTrophy
    fn sceNpTrophyCreateContext(ctx: *mut i32, user: i32, label: u32, opts: u64) -> i32 {
        if !ctx.is_null() { *ctx = 1; }
        0
    }
    fn sceNpTrophyCreateHandle(h: *mut i32) -> i32 {
        if !h.is_null() { *h = next_id(); }
        0
    }
    fn sceNpTrophyUnlockTrophy(ctx: i32, h: i32, trophy: i32, platinum: *mut i32) -> i32 {
        let first = trophy_unlock(trophy);
        crate::info!("trophy {trophy} unlocked{} (recorded locally only)", if first { "" } else { " again" });
        if !platinum.is_null() { *platinum = -1; }
        0
    }
    /// SceNpTrophyFlagArray (128 bits) of the locally recorded unlocks; `count`
    /// = trophies in the title's set (its trophy00.trp icon entries).
    fn sceNpTrophyGetTrophyUnlockState(ctx: i32, h: i32, flags: *mut u32, count: *mut u32) -> i32 {
        if !flags.is_null() {
            std::ptr::write_bytes(flags, 0, 4);
            for id in trophy_unlocked() {
                if (0..128).contains(&id) { *flags.add(id as usize / 32) |= 1 << (id % 32); }
            }
        }
        if !count.is_null() { *count = trophy_count(); }
        0
    }
    fn sceNpTrophyGetGameInfo(ctx: i32, h: i32, details: *mut u8, data: *mut u8) -> i32 {
        if !details.is_null() { let size = *(details as *const u64); std::ptr::write_bytes(details.add(8), 0, (size as usize).saturating_sub(8).min(0x1000)); }
        if !data.is_null() { let size = *(data as *const u64); std::ptr::write_bytes(data.add(8), 0, (size as usize).saturating_sub(8).min(0x1000)); }
        0
    }
    fn sceNpTrophyGetTrophyInfo(ctx: i32, h: i32, trophy: i32, details: *mut u8, data: *mut u8) -> i32 {
        if !details.is_null() { let size = *(details as *const u64); std::ptr::write_bytes(details.add(8), 0, (size as usize).saturating_sub(8).min(0x1000)); }
        if !data.is_null() { let size = *(data as *const u64); std::ptr::write_bytes(data.add(8), 0, (size as usize).saturating_sub(8).min(0x1000)); }
        0
    }

    // ---- libSceNpWebApi / Auth
    fn sceNpWebApiInitialize(http_ctx: i32, pool: u64) -> i32 { next_id() }
    fn sceNpAuthCreateAsyncRequest(param: *const u8) -> i32 { next_id() }
    fn sceNpAuthPollAsync(req: i32, result: *mut i32) -> i32 {
        if !result.is_null() { *result = NP_ERROR_SIGNED_OUT as i32; }
        0
    }

    // ---- libSceNet
    fn sceNetErrnoLoc() -> *mut i32 { NET_ERRNO.as_ptr() }
    fn sceNetHtons(v: u16) -> u16 { v.swap_bytes() }
    fn sceNetNtohs(v: u16) -> u16 { v.swap_bytes() }
    fn sceNetHtonl(v: u32) -> u32 { v.swap_bytes() }
    fn sceNetNtohl(v: u32) -> u32 { v.swap_bytes() }
    fn sceNetInetPton(af: i32, src: *const u8, dst: *mut u8) -> i32 {
        if af != 2 { return 0; }
        match cstr(src).parse::<std::net::Ipv4Addr>() {
            Ok(a) => { std::ptr::copy_nonoverlapping(a.octets().as_ptr(), dst, 4); 1 }
            Err(_) => 0,
        }
    }
    fn sceNetInetNtop(af: i32, src: *const u8, dst: *mut u8, size: u32) -> *mut u8 {
        if af != 2 || src.is_null() { return std::ptr::null_mut(); }
        let o = std::slice::from_raw_parts(src, 4);
        crate::hle::write_cstr(dst, size as usize, &format!("{}.{}.{}.{}", o[0], o[1], o[2], o[3]));
        dst
    }
    fn sceNetEpollWait(eid: i32, events: *mut u8, max: i32, timeout_us: i32) -> i32 {
        if timeout_us > 0 { std::thread::sleep(std::time::Duration::from_micros(timeout_us.min(100_000) as u64)); }
        0
    }
    fn sceNetCtlGetState(state: *mut i32) -> i32 {
        if !state.is_null() { *state = 0; } // DISCONNECTED
        0
    }
    fn sceNetCtlRegisterCallback(f: u64, arg: u64, cid: *mut i32) -> i32 {
        if !cid.is_null() { *cid = next_id(); }
        0
    }

    // ---- libSceHttp
    fn sceHttpCreateEpoll(ctx: i32, eh: *mut u64) -> i32 {
        if !eh.is_null() { *eh = next_id() as u64; }
        0
    }
    fn sceHttpWaitRequest(eh: u64, ev: *mut u8, max: i32, timeout_us: i32) -> i32 {
        if timeout_us > 0 { std::thread::sleep(std::time::Duration::from_micros(timeout_us.min(100_000) as u64)); }
        0
    }
    fn sceHttpCreateConnectionWithURL(tmpl: i32, url: *const u8, keep: i32) -> i32 {
        crate::info!("sceHttpCreateConnectionWithURL({}) (network is offline)", cstr(url));
        next_id()
    }
    fn sceHttpCreateRequestWithURL(conn: i32, method: i32, url: *const u8, len: u64) -> i32 {
        crate::info!("sceHttpCreateRequestWithURL(method {method}, {}) (network is offline)", cstr(url));
        next_id()
    }

    // ---- Dialogs
    fn sceMsgDialogOpen(param: *const u8) -> i32 {
        if !param.is_null() {
            let mode = *(param.add(0x38) as *const i32);
            let user = *(param.add(0x40) as *const u64);
            let sys = *(param.add(0x50) as *const u64);
            if mode == 1 && user != 0 {
                let msg = *((user + 8) as *const u64) as *const u8;
                crate::info!("MsgDialog (user message): {}", cstr(msg));
            } else if mode == 3 && sys != 0 {
                crate::info!("MsgDialog (system message type {})", *(sys as *const i32));
            } else {
                crate::info!("MsgDialog mode {mode}");
            }
        }
        0
    }
    fn sceSaveDataDialogOpen(param: *const u8) -> i32 {
        if !param.is_null() {
            crate::info!("SaveDataDialog open (mode {}, type {})", *(param.add(0x30) as *const i32), *(param.add(0x34) as *const i32));
        }
        0
    }
    fn sceNpProfileDialogGetResult(result: *mut u8) -> i32 {
        if !result.is_null() { std::ptr::write_bytes(result, 0, 4); }
        0
    }
}

fixed! {
    // NpManager
    sceNpSetContentRestriction => 0,
    sceNpSetNpTitleId => 0,
    sceNpNotifyPlusFeature => 0,
    sceNpCheckCallback => 0,
    sceNpAbortRequest => 0,
    sceNpDeleteRequest => 0,
    sceNpUnregisterStateCallback => 0,
    sceNpUnregisterPlusEventCallback => 0,
    sceNpGetOnlineId => NP_ERROR_SIGNED_OUT,
    sceNpGetNpId => NP_ERROR_SIGNED_OUT,
    sceNpCheckNpAvailability => NP_ERROR_SIGNED_OUT,
    sceNpGetParentalControlInfo => NP_ERROR_SIGNED_OUT,
    // NpTrophy
    sceNpTrophyRegisterContext => 0,
    // NpScore
    sceNpScoreCreateNpTitleCtx => NP_ERROR_SIGNED_OUT,
    sceNpScoreDeleteNpTitleCtx => 0,
    sceNpScoreCreateRequest => NP_ERROR_SIGNED_OUT,
    sceNpScoreDeleteRequest => 0,
    sceNpScoreAbortRequest => 0,
    sceNpScoreGetRankingByRange => NP_ERROR_SIGNED_OUT,
    sceNpScoreSetPlayerCharacterId => 0,
    sceNpScoreGetGameData => NP_ERROR_SIGNED_OUT,
    sceNpScoreRecordScore => NP_ERROR_SIGNED_OUT,
    sceNpScoreRecordGameData => NP_ERROR_SIGNED_OUT,
    sceNpScoreCensorComment => NP_ERROR_SIGNED_OUT,
    sceNpScoreSanitizeComment => NP_ERROR_SIGNED_OUT,
    // NpMatching2
    sceNpMatching2Initialize => 0,
    sceNpMatching2Terminate => 0,
    sceNpMatching2RegisterContextCallback => 0,
    sceNpMatching2RegisterRoomEventCallback => 0,
    sceNpMatching2RegisterSignalingCallback => 0,
    sceNpMatching2RegisterLobbyEventCallback => 0,
    sceNpMatching2CreateContext => NP_ERROR_SIGNED_OUT,
    sceNpMatching2ContextStart => NP_ERROR_SIGNED_OUT,
    sceNpMatching2ContextStop => 0,
    sceNpMatching2DestroyContext => 0,
    sceNpMatching2GetServerId => NP_ERROR_SIGNED_OUT,
    sceNpMatching2SetDefaultRequestOptParam => 0,
    sceNpMatching2SearchRoom => NP_ERROR_SIGNED_OUT,
    sceNpMatching2GetWorldInfoList => NP_ERROR_SIGNED_OUT,
    sceNpMatching2SignalingGetPingInfo => NP_ERROR_SIGNED_OUT,
    sceNpMatching2CreateJoinRoom => NP_ERROR_SIGNED_OUT,
    sceNpMatching2JoinRoom => NP_ERROR_SIGNED_OUT,
    sceNpMatching2LeaveRoom => NP_ERROR_SIGNED_OUT,
    sceNpMatching2KickoutRoomMember => NP_ERROR_SIGNED_OUT,
    sceNpMatching2GrantRoomOwner => NP_ERROR_SIGNED_OUT,
    sceNpMatching2SetRoomDataInternal => NP_ERROR_SIGNED_OUT,
    sceNpMatching2SetRoomDataExternal => NP_ERROR_SIGNED_OUT,
    sceNpMatching2SetRoomMemberDataInternal => NP_ERROR_SIGNED_OUT,
    sceNpMatching2SignalingGetConnectionStatus => NP_ERROR_SIGNED_OUT,
    // NpSignaling
    sceNpSignalingInitialize => 0,
    sceNpSignalingTerminate => 0,
    sceNpSignalingCreateContext => NP_ERROR_SIGNED_OUT,
    sceNpSignalingDeleteContext => 0,
    sceNpSignalingActivateConnection => NP_ERROR_SIGNED_OUT,
    sceNpSignalingDeactivateConnection => 0,
    sceNpSignalingGetConnectionStatus => NP_ERROR_SIGNED_OUT,
    // NpUtility
    sceNpLookupCreateTitleCtx => NP_ERROR_SIGNED_OUT,
    sceNpLookupDeleteTitleCtx => 0,
    sceNpLookupAbortRequest => 0,
    sceNpLookupDeleteRequest => 0,
    sceNpLookupNpId => NP_ERROR_SIGNED_OUT,
    // NpWebApi
    sceNpWebApiTerminate => 0,
    sceNpWebApiCreateContext => NP_ERROR_SIGNED_OUT,
    sceNpWebApiDeleteContext => 0,
    sceNpWebApiCreatePushEventFilter => NP_ERROR_SIGNED_OUT,
    sceNpWebApiDeletePushEventFilter => 0,
    sceNpWebApiRegisterPushEventCallback => NP_ERROR_SIGNED_OUT,
    sceNpWebApiUnregisterPushEventCallback => 0,
    sceNpWebApiCreateRequest => NP_ERROR_SIGNED_OUT,
    sceNpWebApiDeleteRequest => 0,
    sceNpWebApiAbortRequest => 0,
    sceNpWebApiSendRequest => NP_ERROR_SIGNED_OUT,
    sceNpWebApiReadData => NP_ERROR_SIGNED_OUT,
    sceNpWebApiGetHttpStatusCode => NP_ERROR_SIGNED_OUT,
    sceNpWebApiUtilityParseNpId => NP_ERROR_SIGNED_OUT,
    sceNpWebApiGetHttpResponseHeaderValueLength => NP_ERROR_SIGNED_OUT,
    sceNpWebApiGetHttpResponseHeaderValue => NP_ERROR_SIGNED_OUT,
    // NpAuth
    sceNpAuthGetAuthorizationCode => NP_ERROR_SIGNED_OUT,
    sceNpAuthDeleteRequest => 0,
    // Net
    sceNetInit => 0,
    sceNetTerm => 0,
    sceNetPoolCreate => 1,
    sceNetPoolDestroy => 0,
    sceNetResolverCreate => 1,
    sceNetResolverDestroy => 0,
    sceNetResolverStartNtoa => NET_ERROR_ENETDOWN,
    sceNetResolverStartAton => NET_ERROR_ENETDOWN,
    sceNetEpollCreate => 1,
    sceNetEpollControl => 0,
    sceNetEpollDestroy => 0,
    sceNetEpollAbort => 0,
    sceNetSocket => NET_ERROR_ENETDOWN,
    sceNetSocketClose => NET_ERROR_EBADF,
    sceNetSocketAbort => NET_ERROR_EBADF,
    sceNetRecv => NET_ERROR_EBADF,
    sceNetRecvfrom => NET_ERROR_EBADF,
    sceNetSend => NET_ERROR_EBADF,
    sceNetSendto => NET_ERROR_EBADF,
    sceNetConnect => NET_ERROR_EBADF,
    sceNetBind => NET_ERROR_EBADF,
    sceNetListen => NET_ERROR_EBADF,
    sceNetAccept => NET_ERROR_EBADF,
    sceNetSetsockopt => NET_ERROR_EBADF,
    sceNetGetsockopt => NET_ERROR_EBADF,
    sceNetGetsockname => NET_ERROR_EBADF,
    sceNetShutdown => NET_ERROR_EBADF,
    // NetCtl
    sceNetCtlInit => 0,
    sceNetCtlTerm => 0,
    sceNetCtlGetInfo => NETCTL_ERROR_NOT_CONNECTED,
    sceNetCtlGetNatInfo => NETCTL_ERROR_NOT_CONNECTED,
    sceNetCtlCheckCallback => 0,
    sceNetCtlUnregisterCallback => 0,
    // Http / Ssl
    sceHttpInit => 1,
    sceHttpTerm => 0,
    sceHttpCreateTemplate => 1,
    sceHttpDeleteTemplate => 0,
    sceHttpSetNonblock => 0,
    sceHttpDestroyEpoll => 0,
    sceHttpsEnableOption => 0,
    sceHttpsDisableOption => 0,
    sceHttpAddRequestHeader => 0,
    sceHttpSetRequestContentLength => 0,
    sceHttpSetConnectTimeOut => 0,
    sceHttpSetEpoll => 0,
    sceHttpUnsetEpoll => 0,
    sceHttpDeleteRequest => 0,
    sceHttpDeleteConnection => 0,
    sceHttpSendRequest => HTTP_ERROR_NETWORK,
    sceHttpGetStatusCode => HTTP_ERROR_NETWORK,
    sceHttpGetResponseContentLength => HTTP_ERROR_NETWORK,
    sceHttpReadData => HTTP_ERROR_NETWORK,
    sceSslInit => 1,
    sceSslTerm => 0,
    // Dialogs
    sceCommonDialogInitialize => 0,
    sceMsgDialogInitialize => 0,
    sceMsgDialogUpdateStatus => DIALOG_FINISHED,
    sceMsgDialogTerminate => 0,
    sceSaveDataDialogInitialize => 0,
    sceSaveDataDialogUpdateStatus => DIALOG_FINISHED,
    sceSaveDataDialogTerminate => 0,
    sceNpProfileDialogInitialize => 0,
    sceNpProfileDialogOpen => 0,
    sceNpProfileDialogUpdateStatus => DIALOG_FINISHED,
    sceNpProfileDialogTerminate => 0,
    sceNpCommerceDialogInitialize => 0,
    sceNpCommerceDialogOpen => 0,
    sceNpCommerceDialogUpdateStatus => DIALOG_FINISHED,
    sceNpCommerceDialogTerminate => 0,
}

pub fn register(r: &mut Registry) {
    reg!(r, "libSceNpManager", [
        sceNpGetState, sceNpGetGamePresenceStatus, sceNpCreateAsyncRequest, sceNpPollAsync, sceNpCheckPlus,
        sceNpRegisterStateCallback, sceNpRegisterGamePresenceCallback, sceNpRegisterPlusEventCallback,
        sceNpSetContentRestriction, sceNpSetNpTitleId, sceNpNotifyPlusFeature, sceNpCheckCallback, sceNpAbortRequest,
        sceNpDeleteRequest, sceNpUnregisterStateCallback, sceNpUnregisterPlusEventCallback, sceNpGetOnlineId,
        sceNpGetNpId, sceNpCheckNpAvailability, sceNpGetParentalControlInfo,
    ]);
    reg!(r, "libSceNpCommon", [sceNpCmpNpId, sceNpCmpOnlineId]);
    reg!(r, "libSceNpTrophy", [
        sceNpTrophyCreateContext, sceNpTrophyCreateHandle, sceNpTrophyUnlockTrophy, sceNpTrophyGetGameInfo,
        sceNpTrophyGetTrophyInfo, sceNpTrophyRegisterContext, sceNpTrophyGetTrophyUnlockState,
    ]);
    reg!(r, "libSceNpScore", [
        sceNpScoreCreateNpTitleCtx, sceNpScoreDeleteNpTitleCtx, sceNpScoreCreateRequest, sceNpScoreDeleteRequest,
        sceNpScoreAbortRequest, sceNpScoreGetRankingByRange, sceNpScoreSetPlayerCharacterId, sceNpScoreGetGameData,
        sceNpScoreRecordScore, sceNpScoreRecordGameData, sceNpScoreCensorComment, sceNpScoreSanitizeComment,
    ]);
    reg!(r, "libSceNpMatching2", [
        sceNpMatching2Initialize, sceNpMatching2Terminate, sceNpMatching2RegisterContextCallback,
        sceNpMatching2RegisterRoomEventCallback, sceNpMatching2RegisterSignalingCallback,
        sceNpMatching2RegisterLobbyEventCallback, sceNpMatching2CreateContext, sceNpMatching2ContextStart,
        sceNpMatching2ContextStop, sceNpMatching2DestroyContext, sceNpMatching2GetServerId,
        sceNpMatching2SetDefaultRequestOptParam, sceNpMatching2SearchRoom, sceNpMatching2GetWorldInfoList,
        sceNpMatching2SignalingGetPingInfo, sceNpMatching2CreateJoinRoom, sceNpMatching2JoinRoom,
        sceNpMatching2LeaveRoom, sceNpMatching2KickoutRoomMember, sceNpMatching2GrantRoomOwner,
        sceNpMatching2SetRoomDataInternal, sceNpMatching2SetRoomDataExternal, sceNpMatching2SetRoomMemberDataInternal,
        sceNpMatching2SignalingGetConnectionStatus,
    ]);
    reg!(r, "libSceNpSignaling", [
        sceNpSignalingInitialize, sceNpSignalingTerminate, sceNpSignalingCreateContext, sceNpSignalingDeleteContext,
        sceNpSignalingActivateConnection, sceNpSignalingDeactivateConnection, sceNpSignalingGetConnectionStatus,
    ]);
    reg!(r, "libSceNpUtility", [
        sceNpLookupCreateTitleCtx, sceNpLookupDeleteTitleCtx, sceNpLookupAbortRequest, sceNpLookupDeleteRequest,
        sceNpLookupNpId,
    ]);
    reg!(r, "libSceNpWebApi", [
        sceNpWebApiInitialize, sceNpWebApiTerminate, sceNpWebApiCreateContext, sceNpWebApiDeleteContext,
        sceNpWebApiCreatePushEventFilter, sceNpWebApiDeletePushEventFilter, sceNpWebApiRegisterPushEventCallback,
        sceNpWebApiUnregisterPushEventCallback, sceNpWebApiCreateRequest, sceNpWebApiDeleteRequest,
        sceNpWebApiAbortRequest, sceNpWebApiSendRequest, sceNpWebApiReadData, sceNpWebApiGetHttpStatusCode,
        sceNpWebApiUtilityParseNpId, sceNpWebApiGetHttpResponseHeaderValueLength, sceNpWebApiGetHttpResponseHeaderValue,
    ]);
    reg!(r, "libSceNpAuth", [sceNpAuthCreateAsyncRequest, sceNpAuthPollAsync, sceNpAuthGetAuthorizationCode, sceNpAuthDeleteRequest]);
    reg!(r, "libSceNet", [
        sceNetErrnoLoc, sceNetHtons, sceNetNtohs, sceNetHtonl, sceNetNtohl, sceNetInetPton, sceNetInetNtop,
        sceNetEpollWait, sceNetInit, sceNetTerm, sceNetPoolCreate, sceNetPoolDestroy, sceNetResolverCreate,
        sceNetResolverDestroy, sceNetResolverStartNtoa, sceNetResolverStartAton, sceNetEpollCreate,
        sceNetEpollControl, sceNetEpollDestroy, sceNetEpollAbort, sceNetSocket, sceNetSocketClose,
        sceNetSocketAbort, sceNetRecv, sceNetRecvfrom, sceNetSend, sceNetSendto, sceNetConnect, sceNetBind,
        sceNetListen, sceNetAccept, sceNetSetsockopt, sceNetGetsockopt, sceNetGetsockname, sceNetShutdown,
    ]);
    reg!(r, "libSceNetCtl", [
        sceNetCtlGetState, sceNetCtlRegisterCallback, sceNetCtlInit, sceNetCtlTerm, sceNetCtlGetInfo,
        sceNetCtlGetNatInfo, sceNetCtlCheckCallback, sceNetCtlUnregisterCallback,
    ]);
    reg!(r, "libSceHttp", [
        sceHttpCreateEpoll, sceHttpWaitRequest, sceHttpCreateConnectionWithURL, sceHttpCreateRequestWithURL,
        sceHttpInit, sceHttpTerm, sceHttpCreateTemplate, sceHttpDeleteTemplate, sceHttpSetNonblock,
        sceHttpDestroyEpoll, sceHttpsEnableOption, sceHttpsDisableOption, sceHttpAddRequestHeader,
        sceHttpSetRequestContentLength, sceHttpSetConnectTimeOut, sceHttpSetEpoll, sceHttpUnsetEpoll,
        sceHttpDeleteRequest, sceHttpDeleteConnection, sceHttpSendRequest, sceHttpGetStatusCode,
        sceHttpGetResponseContentLength, sceHttpReadData,
    ]);
    reg!(r, "libSceSsl", [sceSslInit, sceSslTerm]);
    reg!(r, "libSceCommonDialog", [sceCommonDialogInitialize]);
    reg!(r, "libSceMsgDialog", [sceMsgDialogInitialize, sceMsgDialogOpen, sceMsgDialogUpdateStatus, sceMsgDialogTerminate]);
    reg!(r, "libSceSaveDataDialog", [sceSaveDataDialogInitialize, sceSaveDataDialogOpen, sceSaveDataDialogUpdateStatus, sceSaveDataDialogTerminate]);
    reg!(r, "libSceNpProfileDialog", [
        sceNpProfileDialogInitialize, sceNpProfileDialogOpen, sceNpProfileDialogUpdateStatus,
        sceNpProfileDialogGetResult, sceNpProfileDialogTerminate,
    ]);
    reg!(r, "libSceNpCommerce", [
        sceNpCommerceDialogInitialize, sceNpCommerceDialogOpen, sceNpCommerceDialogUpdateStatus,
        sceNpCommerceDialogTerminate,
    ]);
}
