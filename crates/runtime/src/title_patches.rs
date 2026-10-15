//! Title code patches for The Last Guardian (CUSA03627, update 01.03, main
//! module `launch_tricom`).
//!
//! Raw mouse look (TLG_MOUSE_CAMERA=0 turns it off): the game camera
//! (`GameCamera`, object `cam`) turns from the right stick in three stages:
//! the input stage (eboot 0x67b0c6) maps the deflection past a dead zone
//! through x*|x| and the camera-speed options to targets at [input+0x170]
//! (pitch) and [input+0x174] (yaw); a smoothing stage (0x6a5f4f) eases
//! [cam+0x2490]/[cam+0x2494] toward them (it zeroes them at 0x6a5980 when
//! no input is present or camera input is locked: speed 0, lock flags and
//! timers); the integrator (0x706225) turns them into this frame's angle
//! changes in radians (degrees/s parameter * pi/180 * dt), pitch limited
//! against [cam+0x2240], yaw [cam+0x2244]. Mouse motion is added there
//! instead, as radians: no dead zone, response curve, smoothing or
//! acceleration, the game's pitch limits and input locks still applying.
//! The integrator also stores the turn as angular velocities ([cam+0x2530]
//! pitch, [cam+0x2534] yaw, delta/dt) which the idle path keeps applying,
//! halved each frame, after input stops (momentum: +30% overshoot measured
//! for a mouse flick); on frames with mouse motion they are zeroed.
//! Hooks: 0x6a58b1 lets pending mouse motion into the lock checks, 0x6a5fae
//! (checks passed) and 0x6a5980 (failed or no input: pending motion is
//! dropped) count camera updates, 0x70617e sends a frame with pending
//! motion to the integrator, 0x70625f and 0x706272 add it to the pitch and
//! yaw changes, 0x706943 drops the momentum. Stick right gives a negative
//! yaw change, stick up a positive pitch change (looks up). Applied only
//! when all seven sites hold the expected bytes.
//!
//! Render resolution (`render_resolution`, TLG_RENDER_SIZE): the game
//! renders at the size of its display mode, the first entry of a table the
//! graphics init copies (render and output size, checkerboard flag, name:
//! "1920x1080", then "1760x990" and "1600x900"; the PS4 Pro table for 4K
//! displays has "1680x1890(cb)" first and "3840x2160(native)" among them).
//! That entry is set to the chosen size for both; its render targets, the
//! viewports and constants derived from it and the display buffers follow
//! (checked at 1280x720 and 3840x2160: picture and interface whole). Above
//! 1080p the graphics heap takes the PS4 Pro's size (+512 MB, the direct
//! memory with it). Applied only when the bytes are the expected ones.
//!
//! Script prints: every `log(fmt, ...)` and `GaPrint(...)` in the Lua (Havok
//! Script) game scripts ends in the C function `GaPrint`, which the retail
//! build compiled out (`xor eax, eax; ret` at eboot 0xd51780). The patch
//! routes it to `tlg_lua_print`, which converts its arguments with the
//! engine's own `tolstring` (eboot 0xe75cd0: lua_State, TValue*, size_t*),
//! notes the title screen's markers (`in_menu`) and, with TLG_LUA_LOG=1,
//! logs them ("lua: ..."). It returns 0 as the retail function does (about
//! one call a second). Applied only when the code bytes are the expected
//! ones.
//!
//! Render-kick wait: the job that hands a frame to the ps4render thread
//! (eboot 0xd0afa0) then waits for that thread's state ([obj+0x31ef8]) to
//! reach 2 in a loop with no pause instruction (0xd0afe4..0xd0affe), on one
//! of the job workers, for as long as ps4render is busy - while the title
//! waits for our GPU work, most of a frame. The loop is detoured to
//! `tlg_kick_wait`, which waits for the same condition with a short spin and
//! then sleeps between polls (`tune kick_wait`). Applied only when the code
//! bytes are the expected ones.

use crate::patch::Patcher;
use crate::sysv;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// The title's own measured frame time, where a patch exposes it.
pub fn guest_frame_ms() -> Option<f32> {
    None
}

/// Frame-rate cap requested for the title (none: the game paces itself).
pub fn target_fps() -> Option<u32> {
    None
}

/// `GaPrint`: the retail no-op and its padding.
const GAPRINT: usize = 0xd51780;
const GAPRINT_BYTES: [u8; 16] = [0x31, 0xC0, 0xC3, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90];
/// The engine's tolstring and its prologue (push rbp; mov rbp, rsp; push r15; push r14).
const TOLSTRING: u64 = 0xe75cd0;
const TOLSTRING_BYTES: [u8; 8] = [0x55, 0x48, 0x89, 0xE5, 0x41, 0x57, 0x41, 0x56];

/// Module base of the patched executable (for calling its tolstring).
static BASE: AtomicU64 = AtomicU64::new(0);

pub fn apply(p: &mut Patcher, name: &str) {
    patch_gaprint(p, name);
    let (w, h) = crate::gpu::vk::window::render_size();
    match patch_render_size(p, w, h) {
        Ok(heap) => {
            let more = if heap { "; graphics heap +512 MB (the PS4 Pro's)" } else { "" };
            crate::info!("render resolution: {w}x{h} (the game's display mode, {name}+{MODE_TABLE:#x}){more}");
        }
        Err(e) => crate::warn!("render resolution: {name}: {e}; the game renders 1920x1080"),
    }
    match patch_kick_wait(p) {
        Ok(()) => crate::info!("render-kick wait: busy loop at {name}+{KICK_SITE:#x} detoured to a polling wait"),
        Err(e) => crate::warn!("render-kick wait: {name}: {e}; not patched"),
    }
    if !std::env::var("TLG_MOUSE_CAMERA").is_ok_and(|v| v == "0") {
        match patch_mouse_camera(p) {
            Ok(slots) => {
                CAM_SLOTS.store(slots, Ordering::Relaxed);
                crate::info!("mouse camera: raw mouse look hooked into the game camera ({name}, slots {slots:#x})");
            }
            Err(e) => crate::warn!("mouse camera: {name}: {e}; mouse look drives the right stick instead"),
        }
    }
}

/// The game's display-mode table for 1080p displays (the graphics init at
/// eboot 0xc4fa30 copies it to 0x59b1820): 24-byte entries of render size
/// and output size (each width | height << 16), a checkerboard flag and a
/// name pointer; the first entry ("1920x1080") is the mode the game runs
/// in, its output size the display buffers' size.
const MODE_TABLE: usize = 0x16ec3a0;
const MODE_1080P: [u8; 12] = [0x80, 0x07, 0x38, 0x04, 0x80, 0x07, 0x38, 0x04, 0, 0, 0, 0];
/// The graphics heap's size choices (eboot 0xdd3a46..): 0xb0000000, or
/// 0xd0000000 when sceKernelIsNeoMode (PS4 Pro) - allocated, mapped and
/// recorded by these `cmove reg, r15` (the base size where not Neo; two on
/// the error paths).
const HEAP_SITES: [(usize, [u8; 4]); 5] = [
    (0xdd3aad, [0x49, 0x0F, 0x44, 0xD7]), // cmove rdx, r15 (allocation)
    (0xdd3b21, [0x49, 0x0F, 0x44, 0xF7]), // cmove rsi, r15 (mapping)
    (0xdd3b47, [0x4D, 0x0F, 0x44, 0xEF]), // cmove r13, r15 (recorded size)
    (0xdd3bf7, [0x4D, 0x0F, 0x44, 0xEF]),
    (0xdd3c22, [0x49, 0x0F, 0x44, 0xCF]),
];
const HEAP_EXTRA: u64 = 0x2000_0000;

/// Sets the game's display mode to render and show `w` x `h`; above 1080p
/// also gives it the PS4 Pro's graphics heap (its higher-resolution modes'
/// memory). Returns whether the heap grew.
fn patch_render_size(p: &mut Patcher, w: u32, h: u32) -> Result<bool, String> {
    if !(64..=8192).contains(&w) || !(64..=8192).contains(&h) {
        return Err(format!("{w}x{h} is out of range"));
    }
    if p.image.get(MODE_TABLE..MODE_TABLE + 12) != Some(&MODE_1080P[..]) {
        return Err("not the expected executable (display-mode table differs)".into());
    }
    let heap = w as u64 * h as u64 > 1920 * 1080;
    if heap && HEAP_SITES.iter().any(|(off, b)| p.image.get(*off..*off + 4) != Some(&b[..])) {
        return Err("not the expected executable (graphics heap code differs)".into());
    }
    let v = (w | h << 16).to_le_bytes();
    p.image[MODE_TABLE..MODE_TABLE + 4].copy_from_slice(&v);
    p.image[MODE_TABLE + 4..MODE_TABLE + 8].copy_from_slice(&v);
    if heap {
        for (off, _) in HEAP_SITES {
            p.image[off..off + 4].copy_from_slice(&[0x0F, 0x1F, 0x40, 0x00]); // nop: the Neo size stays
        }
        crate::mem::add_dmem(HEAP_EXTRA);
    }
    Ok(heap)
}

/// Mouse-look data next to the module: pending pitch and yaw change
/// (radians, f32) at +0 and +4, camera update count at +8, the pitch and
/// yaw change applied in the current integrator pass at +12 and +16, the
/// camera object at +24 (0 = no hook).
static CAM_SLOTS: AtomicU64 = AtomicU64::new(0);

/// (camera update count last seen, when it changed: process ms and flip).
static CAM_SEEN: parking_lot::Mutex<(u32, f64, u64)> = parking_lot::Mutex::new((0, 0.0, 0));

/// Hook sites: module offset, the instruction's bytes.
const CAM_GATE: (usize, &[u8]) = (0x6a58b1, &[0xC5, 0xFA, 0x10, 0x80, 0x74, 0x01, 0x00, 0x00]); // vmovss xmm0,[rax+174h]
const CAM_PASSED: (usize, &[u8]) = (0x6a5fae, &[0xC4, 0xC1, 0x7A, 0x11, 0x8F, 0x94, 0x24, 0x00, 0x00]); // vmovss [r15+2494h],xmm1
const CAM_IDLE: (usize, &[u8]) = (0x6a5980, &[0x49, 0xC7, 0x87, 0x90, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]); // mov qword [r15+2490h],0
const CAM_ROUTE: (usize, &[u8]) = (0x70617e, &[0xC4, 0xC1, 0x7A, 0x10, 0x87, 0x90, 0x24, 0x00, 0x00]); // vmovss xmm0,[r15+2490h]
const CAM_PITCH: (usize, &[u8]) = (0x70625f, &[0xC5, 0xFA, 0x59, 0x5A, 0x0C]); // vmulss xmm3,xmm0,[rdx+0Ch]
const CAM_YAW: (usize, &[u8]) = (0x706272, &[0xC4, 0xC1, 0x72, 0x59, 0xAF, 0x94, 0x24, 0x00, 0x00]); // vmulss xmm5,xmm1,[r15+2494h]
const CAM_MOMENTUM: (usize, &[u8]) = (0x706943, &[0xC4, 0xC1, 0x7A, 0x11, 0x97, 0x34, 0x25, 0x00, 0x00]); // vmovss [r15+2534h],xmm2
/// Where those continue: lock checks, after the store, after the zeroing,
/// the integrator (with eax = [cam+0x2444]), after the original instructions.
const CAM_GATE_CHECKS: u64 = 0x6a58cf;
const CAM_INTEGRATE: u64 = 0x7061fb;

/// Installs the raw mouse-look hooks (module docs); returns the data slots.
fn patch_mouse_camera(p: &mut Patcher) -> Result<u64, String> {
    use iced_x86::code_asm::*;
    let sites = [CAM_GATE, CAM_PASSED, CAM_IDLE, CAM_ROUTE, CAM_PITCH, CAM_YAW, CAM_MOMENTUM];
    if let Some((off, _)) = sites.iter().find(|(off, b)| p.image.get(*off..*off + b.len()) != Some(*b)) {
        return Err(format!("unexpected code at +{off:#x} (not update 01.03?)"));
    }
    let slots = p.alloc_data(&[0u8; 32]).ok_or("no data space")?;
    let base = p.base;
    let next = |site: (usize, &[u8])| base + (site.0 + site.1.len()) as u64;
    // Stubs keep every register and the flags except where noted (the yaw
    // site sits between a compare and the branch using it).
    p.detour(CAM_GATE.0, CAM_GATE.1, |a, ins| {
        a.add_instruction(*ins)?;
        a.lea(rsp, ptr(rsp - 128))?;
        a.push(rcx)?;
        a.mov(rcx, slots)?;
        a.cmp(qword_ptr(rcx), 0)?; // (flags are dead here)
        a.pop(rcx)?;
        a.lea(rsp, ptr(rsp + 128))?;
        a.jne(base + CAM_GATE_CHECKS)?;
        a.jmp(next(CAM_GATE))
    })?;
    // Camera update counter (+8), and on the idle/locked path the pending
    // motion dropped.
    let count = |a: &mut CodeAssembler| -> Result<(), IcedError> {
        a.mov(eax, dword_ptr(rcx + 8))?;
        a.lea(eax, ptr(rax + 1))?;
        a.mov(dword_ptr(rcx + 8), eax)?;
        // (The camera object, for diagnostics: `camera_object`.)
        a.mov(qword_ptr(rcx + 24), r15)
    };
    p.detour(CAM_PASSED.0, CAM_PASSED.1, |a, ins| {
        a.add_instruction(*ins)?;
        a.lea(rsp, ptr(rsp - 128))?;
        a.push(rax)?;
        a.push(rcx)?;
        a.mov(rcx, slots)?;
        count(a)?;
        a.pop(rcx)?;
        a.pop(rax)?;
        a.lea(rsp, ptr(rsp + 128))?;
        a.jmp(next(CAM_PASSED))
    })?;
    p.detour(CAM_IDLE.0, CAM_IDLE.1, |a, ins| {
        a.add_instruction(*ins)?;
        a.lea(rsp, ptr(rsp - 128))?;
        a.push(rax)?;
        a.push(rcx)?;
        a.mov(rcx, slots)?;
        a.mov(eax, 0)?;
        a.xchg(dword_ptr(rcx), eax)?;
        a.mov(eax, 0)?;
        a.xchg(dword_ptr(rcx + 4), eax)?;
        count(a)?;
        a.pop(rcx)?;
        a.pop(rax)?;
        a.lea(rsp, ptr(rsp + 128))?;
        a.jmp(next(CAM_IDLE))
    })?;
    p.detour(CAM_ROUTE.0, CAM_ROUTE.1, |a, ins| {
        let mut idle = a.create_label();
        a.add_instruction(*ins)?;
        a.lea(rsp, ptr(rsp - 128))?;
        a.push(rcx)?;
        a.mov(rcx, slots)?;
        a.cmp(qword_ptr(rcx), 0)?; // (flags are dead here)
        a.pop(rcx)?;
        a.lea(rsp, ptr(rsp + 128))?;
        a.je(idle)?;
        a.mov(eax, dword_ptr(r15 + 0x2444))?; // (what the integrator path tests)
        a.jmp(base + CAM_INTEGRATE)?;
        a.set_label(&mut idle)?;
        a.jmp(next(CAM_ROUTE))
    })?;
    // Pending pitch (+0) / yaw (+4) taken atomically, added to xmm3 / xmm5
    // and noted as applied (+12 / +16).
    let add = |a: &mut CodeAssembler, ins: &iced_x86::Instruction, slot: i32, x: AsmRegisterXmm, back: u64| -> Result<(), IcedError> {
        a.add_instruction(*ins)?;
        a.lea(rsp, ptr(rsp - 128))?;
        a.push(rax)?;
        a.push(rcx)?;
        a.mov(rcx, slots)?;
        a.mov(eax, 0)?;
        a.xchg(dword_ptr(rcx + slot), eax)?;
        a.mov(dword_ptr(rcx + slot + 12), eax)?;
        a.push(rax)?;
        a.vaddss(x, x, dword_ptr(rsp))?;
        a.pop(rax)?;
        a.pop(rcx)?;
        a.pop(rax)?;
        a.lea(rsp, ptr(rsp + 128))?;
        a.jmp(back)
    };
    p.detour(CAM_PITCH.0, CAM_PITCH.1, |a, ins| add(a, ins, 0, xmm3, next(CAM_PITCH)))?;
    p.detour(CAM_YAW.0, CAM_YAW.1, |a, ins| add(a, ins, 4, xmm5, next(CAM_YAW)))?;
    // After the velocities are stored: none when mouse motion was applied.
    p.detour(CAM_MOMENTUM.0, CAM_MOMENTUM.1, |a, ins| {
        let mut done = a.create_label();
        a.add_instruction(*ins)?;
        a.lea(rsp, ptr(rsp - 128))?;
        a.push(rcx)?;
        a.mov(rcx, slots)?;
        a.cmp(qword_ptr(rcx + 12), 0)?; // (flags are dead here)
        a.je(done)?;
        a.mov(qword_ptr(rcx + 12), 0)?;
        a.mov(dword_ptr(r15 + 0x2530), 0)?;
        a.mov(dword_ptr(r15 + 0x2534), 0)?;
        a.set_label(&mut done)?;
        a.pop(rcx)?;
        a.lea(rsp, ptr(rsp + 128))?;
        a.jmp(next(CAM_MOMENTUM))
    })?;
    Ok(slots)
}

/// Adds mouse motion to the game camera's pending turn (radians; positive
/// yaw turns left, positive pitch looks up). Returns false when the game
/// camera has not updated for over 1.5 s and 3 frames (another camera mode,
/// a menu, or no hook): the motion is then not kept, so it cannot land all
/// at once later.
pub fn add_camera_turn(yaw: f32, pitch: f32) -> bool {
    use std::sync::atomic::AtomicU32;
    let s = CAM_SLOTS.load(Ordering::Relaxed);
    if s == 0 {
        return false;
    }
    // SAFETY: the slots are 32 bytes of the module's data page, mapped for
    // the process lifetime; the stubs access them with atomic exchanges.
    let slot = |o: u64| unsafe { &*((s + o) as *const AtomicU32) };
    let now = crate::log::elapsed_ms();
    let flip = crate::gpu::FRAMES.load(Ordering::Relaxed);
    let updates = slot(8).load(Ordering::Relaxed);
    let mut seen = CAM_SEEN.lock();
    if updates != seen.0 || seen.1 == 0.0 {
        *seen = (updates, now, flip);
    }
    if now - seen.1 > 1500.0 && flip >= seen.2 + 3 {
        slot(0).store(0, Ordering::Relaxed);
        slot(4).store(0, Ordering::Relaxed);
        return false;
    }
    for (o, v) in [(0, pitch), (4, yaw)] {
        if v == 0.0 {
            continue;
        }
        let a = slot(o);
        let mut cur = a.load(Ordering::Relaxed);
        loop {
            let new = (f32::from_bits(cur) + v).clamp(-1.0, 1.0).to_bits();
            match a.compare_exchange_weak(cur, new, Ordering::AcqRel, Ordering::Relaxed) {
                Ok(_) => break,
                Err(c) => cur = c,
            }
        }
    }
    true
}

/// The title screen and its menus show: from boot (logos, title) and the
/// scripts' "Change Clut : titleScreen" until a stage starts from the menu
/// ("RestartStageFromMenu", "LoadCheckPoint", "=====RESTART=====") or a
/// load (`in_menu`).
static TITLE: AtomicBool = AtomicBool::new(true);
/// The title screen's marker was seen since boot (before it, loads do not
/// end the title context: boot loads before the logos).
static TITLE_SHOWN: AtomicBool = AtomicBool::new(false);

/// Notes the title screen's script markers in a print's text.
fn note_script(text: &str) {
    if text.starts_with("Change Clut : titleScreen") {
        TITLE_SHOWN.store(true, Ordering::Relaxed);
        if !TITLE.swap(true, Ordering::Relaxed) {
            crate::info!("input: the title screen shows");
        }
    } else if text.starts_with("RestartStageFromMenu") || text.starts_with("LoadCheckPoint") || text.contains("=====RESTART=====") {
        if TITLE.swap(false, Ordering::Relaxed) {
            crate::info!("input: a stage starts from the title menu");
        }
    }
}

/// Keyboard/mouse context: a menu shows - the title screen and main menu
/// (script markers; a load of over 3 s - the camera still - also ends
/// them, e.g. New Game), or the game camera stands still (no update for
/// half a second and 3 frames: the pause menu, loading screens, most
/// cutscenes). None without the camera hook.
pub fn in_menu() -> Option<bool> {
    use std::sync::atomic::AtomicU32;
    let s = CAM_SLOTS.load(Ordering::Relaxed);
    if s == 0 {
        return None;
    }
    // SAFETY: as in add_camera_turn.
    let updates = unsafe { &*((s + 8) as *const AtomicU32) }.load(Ordering::Relaxed);
    let now = crate::log::elapsed_ms();
    let flip = crate::gpu::FRAMES.load(Ordering::Relaxed);
    let still_ms = {
        let mut seen = CAM_SEEN.lock();
        if updates != seen.0 || seen.1 == 0.0 {
            *seen = (updates, now, flip);
        }
        if flip >= seen.2 + 3 { now - seen.1 } else { 0.0 }
    };
    if still_ms > 3000.0 && TITLE_SHOWN.load(Ordering::Relaxed) && TITLE.swap(false, Ordering::Relaxed) {
        crate::info!("input: the title screen ended (a load)");
    }
    Some(TITLE.load(Ordering::Relaxed) || still_ms > 500.0)
}

/// The camera hook is installed (update 01.03, TLG_MOUSE_CAMERA not 0).
pub fn camera_hooked() -> bool {
    CAM_SLOTS.load(Ordering::Relaxed) != 0
}

/// The game camera object (the hooks' r15) once it has updated; 0 before.
pub fn camera_object() -> u64 {
    let s = CAM_SLOTS.load(Ordering::Relaxed);
    if s == 0 {
        return 0;
    }
    // SAFETY: as in add_camera_turn.
    unsafe { &*((s + 24) as *const AtomicU64) }.load(Ordering::Relaxed)
}

/// Flips left to log the camera for (control `camlog <flips>`).
static CAMLOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub fn camlog(flips: u32) {
    CAMLOG.store(flips, Ordering::Relaxed);
}

/// At each displayed flip while `camlog` runs: the camera's pitch and yaw
/// (+0x2240/+0x2244, radians), eye (+0x2370) and pivot (+0x23c0), and the
/// mouse turn applied in the last integrator pass.
pub fn camlog_tick(flip: u64) {
    if CAMLOG.load(Ordering::Relaxed) == 0 {
        return;
    }
    CAMLOG.fetch_sub(1, Ordering::Relaxed);
    let cam = camera_object();
    if cam == 0 || !crate::mem::is_guest(cam + 0x2240, 0x190) {
        crate::info!("camlog flip {flip}: no camera object");
        return;
    }
    let f = |o: u64| unsafe { std::ptr::read_volatile((cam + o) as *const f32) };
    let s = CAM_SLOTS.load(Ordering::Relaxed);
    let applied = if s != 0 { unsafe { (std::ptr::read_volatile((s + 12) as *const f32), std::ptr::read_volatile((s + 16) as *const f32)) } } else { (0.0, 0.0) };
    crate::info!(
        "camlog flip {flip} t {:.1}: pitch {:.5} yaw {:.5} eye ({:.3} {:.3} {:.3}) pivot ({:.3} {:.3} {:.3}) applied ({:.5} {:.5})",
        crate::log::elapsed_ms(),
        f(0x2240),
        f(0x2244),
        f(0x2370),
        f(0x2374),
        f(0x2378),
        f(0x23c0),
        f(0x23c4),
        f(0x23c8),
        applied.0,
        applied.1
    );
}

fn patch_gaprint(p: &mut Patcher, name: &str) {
    let t = TOLSTRING as usize;
    if p.image.get(GAPRINT..GAPRINT + 16) != Some(&GAPRINT_BYTES[..]) || p.image.get(t..t + 8) != Some(&TOLSTRING_BYTES[..]) {
        crate::warn!("script prints: {name} is not the expected executable (GaPrint/tolstring bytes differ): not patched");
        return;
    }
    // mov rax, imm64; jmp rax (12 bytes; the rest stays padding).
    let target = tlg_lua_print as *const () as usize as u64;
    let mut code = vec![0x48, 0xB8];
    code.extend_from_slice(&target.to_le_bytes());
    code.extend_from_slice(&[0xFF, 0xE0]);
    p.image[GAPRINT..GAPRINT + code.len()].copy_from_slice(&code);
    BASE.store(p.base, Ordering::Relaxed);
    crate::info!("script prints: GaPrint at {name}+{GAPRINT:#x} routed to tlg (logged with TLG_LUA_LOG=1)");
}

/// The render-kick job's spin-counter reset, its first test of the
/// ps4render state and its busy loop (module docs), and where it continues.
const KICK_SITE: usize = 0xd0afdd;
const KICK_BYTES: [u8; 33] = [
    0xC7, 0x45, 0xE4, 0x00, 0x00, 0x00, 0x00, // mov dword [rbp-1Ch], 0
    0x8B, 0x83, 0xF8, 0x1E, 0x03, 0x00, // mov eax, [rbx+31EF8h]
    0x83, 0xF8, 0x01, // cmp eax, 1
    0x7F, 0x0F, // jg done
    0x90, // nop
    0xFF, 0x45, 0xE4, // spin: inc dword [rbp-1Ch]
    0x8B, 0x83, 0xF8, 0x1E, 0x03, 0x00, // mov eax, [rbx+31EF8h]
    0x83, 0xF8, 0x02, // cmp eax, 2
    0x7C, 0xF2, // jl spin
];
const KICK_DONE: u64 = 0xd0affe;
const KICK_STATE: i32 = 0x31ef8;

/// Render-kick waits that had to wait, and their time (ns).
pub static KICK_STATS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

fn patch_kick_wait(p: &mut Patcher) -> Result<(), String> {
    use iced_x86::code_asm::*;
    if p.image.get(KICK_SITE..KICK_SITE + KICK_BYTES.len()) != Some(&KICK_BYTES[..]) {
        return Err(format!("unexpected code at +{KICK_SITE:#x} (not update 01.03?)"));
    }
    let base = p.base;
    let target = tlg_kick_wait as *const () as usize as u64;
    // (rsp is 16-byte aligned here; past the loop only rbx, r14 and rbp
    // are live, and the flags are set again.)
    p.detour(KICK_SITE, &KICK_BYTES[..7], |a, ins| {
        a.add_instruction(*ins)?;
        a.lea(rdi, ptr(rbx + KICK_STATE))?;
        a.mov(rax, target)?;
        a.call(rax)?;
        a.jmp(base + KICK_DONE)
    })?;
    Ok(())
}

/// Waits until the dword at `state` is at least 2 (the loop it replaces).
fn kick_wait(state: u64) {
    // SAFETY: the title's object field the original loop reads.
    let a = unsafe { &*(state as *const std::sync::atomic::AtomicI32) };
    if a.load(Ordering::Acquire) >= 2 {
        return;
    }
    let t0 = std::time::Instant::now();
    let poll = crate::tune::get(crate::tune::KICK_WAIT);
    let mut n = 0u32;
    while a.load(Ordering::Acquire) < 2 {
        if poll <= 0 {
            continue; // (busy, as the title)
        }
        n += 1;
        if n <= 64 {
            std::hint::spin_loop();
        } else {
            std::thread::sleep(std::time::Duration::from_micros(poll as u64));
        }
    }
    KICK_STATS[0].fetch_add(1, Ordering::Relaxed);
    KICK_STATS[1].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// A NUL-terminated guest string (at most `max` bytes).
unsafe fn cstr(p: u64, max: usize) -> String {
    let mut v = Vec::new();
    for i in 0..max as u64 {
        if !crate::mem::is_guest(p + i, 1) {
            break;
        }
        let b = unsafe { *((p + i) as *const u8) };
        if b == 0 {
            break;
        }
        v.push(b);
    }
    String::from_utf8_lossy(&v).into_owned()
}

/// Logs the string forms of the call's arguments.
unsafe fn lua_print(l: u64) {
    let base_mod = BASE.load(Ordering::Relaxed);
    if base_mod == 0 || !crate::mem::is_guest(l, 0x60) {
        return;
    }
    // lua_State: stack top at +0x48, the call's first argument at +0x50; 16-byte TValues.
    let (top, base) = unsafe { (*((l + 0x48) as *const u64), *((l + 0x50) as *const u64)) };
    if top < base || top - base > 32 * 16 || !crate::mem::is_guest(base, (top - base).max(1)) {
        return;
    }
    let mut parts = Vec::new();
    for i in 0..(top - base) / 16 {
        let s = unsafe { crate::hle::call_guest(base_mod + TOLSTRING, &[l, base + i * 16, 0]) };
        parts.push(if s != 0 { unsafe { cstr(s, 2048) } } else { "<?>".to_string() });
    }
    let text = parts.join(" ");
    note_script(&text);
    static LOG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *LOG.get_or_init(|| std::env::var("TLG_LUA_LOG").is_ok_and(|v| v == "1")) {
        crate::info!("lua: {text}");
    }
}

sysv! {
    fn tlg_lua_print(l: u64) -> i32 {
        lua_print(l);
        0
    }
    fn tlg_kick_wait(state: u64) -> i32 {
        kick_wait(state);
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced_x86::{Decoder, DecoderOptions, FlowControl};

    /// Decodes the trampoline a site's `jmp rel32` leads to, up to and
    /// including its final unconditional jump.
    fn stub(image: &[u8], tramp: &[u8], base: u64, tbase: u64, off: usize) -> Vec<iced_x86::Instruction> {
        assert_eq!(image[off], 0xE9, "site +{off:#x} is not detoured");
        let rel = i32::from_le_bytes(image[off + 1..off + 5].try_into().unwrap()) as i64;
        let t = (base as i64 + off as i64 + 5 + rel) as u64;
        let mut d = Decoder::with_ip(64, &tramp[(t - tbase) as usize..], t, DecoderOptions::NONE);
        let mut v = Vec::new();
        loop {
            let i = d.decode();
            assert!(!i.is_invalid());
            v.push(i);
            if i.flow_control() == FlowControl::UnconditionalBranch {
                return v;
            }
        }
    }

    #[test]
    fn kick_wait_stub() {
        let base = 0x6000_0000u64;
        let mut image = vec![0x90u8; 0xd1_0000];
        image[KICK_SITE..KICK_SITE + KICK_BYTES.len()].copy_from_slice(&KICK_BYTES);
        let mut tramp = vec![0u8; 0x1_0000];
        let tbase = base + 0xe0_0000;
        {
            let mut p = Patcher { image: &mut image, base, tls_slot_disp: 0, tramp_base: tbase, tramp: &mut tramp, tramp_used: 0, data_used: 0 };
            patch_kick_wait(&mut p).unwrap();
        }
        let v = stub(&image, &tramp, base, tbase, KICK_SITE);
        // The counter reset, rdi = the state field, the call, then past the loop.
        assert_eq!(&tramp[(v[0].ip() - tbase) as usize..][..7], &KICK_BYTES[..7]);
        assert!(v.iter().any(|i| i.mnemonic() == iced_x86::Mnemonic::Lea && i.op0_register() == iced_x86::Register::RDI && i.memory_base() == iced_x86::Register::RBX && i.memory_displacement64() == KICK_STATE as u64));
        assert!(v.iter().any(|i| i.mnemonic() == iced_x86::Mnemonic::Call));
        assert_eq!(v.last().unwrap().near_branch_target(), base + KICK_DONE);
        // Other bytes: not patched.
        let mut image2 = vec![0x90u8; 0xd1_0000];
        image2[KICK_SITE..KICK_SITE + KICK_BYTES.len()].copy_from_slice(&KICK_BYTES);
        image2[KICK_SITE + 30] = 0x03;
        let mut tramp2 = vec![0u8; 0x1_0000];
        let mut p = Patcher { image: &mut image2, base, tls_slot_disp: 0, tramp_base: tbase, tramp: &mut tramp2, tramp_used: 0, data_used: 0 };
        assert!(patch_kick_wait(&mut p).is_err());
        assert_eq!(p.image[KICK_SITE], 0xC7);
    }

    #[test]
    fn mouse_camera_stubs() {
        let base = 0x6000_0000u64;
        let sites = [CAM_GATE, CAM_PASSED, CAM_IDLE, CAM_ROUTE, CAM_PITCH, CAM_YAW, CAM_MOMENTUM];
        let mut image = vec![0x90u8; 0x71_0000];
        for (off, b) in sites {
            image[off..off + b.len()].copy_from_slice(b);
        }
        let mut tramp = vec![0u8; 0x1_0000];
        let tbase = base + 0x80_0000;
        let slots = {
            let mut p = Patcher { image: &mut image, base, tls_slot_disp: 0, tramp_base: tbase, tramp: &mut tramp, tramp_used: 0, data_used: 0 };
            patch_mouse_camera(&mut p).unwrap()
        };
        assert!(slots >= tbase && slots < tbase + 0x1_0000);
        let next = |s: (usize, &[u8])| base + (s.0 + s.1.len()) as u64;
        for site in sites {
            let v = stub(&image, &tramp, base, tbase, site.0);
            // The original instruction first, its bytes unchanged.
            let first = &tramp[(v[0].ip() - tbase) as usize..][..v[0].len()];
            assert_eq!(first, site.1, "site +{:#x}", site.0);
            // Stack balanced: what is pushed is popped, the red zone skipped.
            let pushes = v.iter().filter(|i| i.mnemonic() == iced_x86::Mnemonic::Push).count();
            let pops = v.iter().filter(|i| i.mnemonic() == iced_x86::Mnemonic::Pop).count();
            assert_eq!(pushes, pops, "site +{:#x}", site.0);
            // Back after the replaced instruction (the route stub's first exit
            // is the integrator; its fall-back jump follows).
            let last = v.last().unwrap();
            let want = if site.0 == CAM_ROUTE.0 { base + CAM_INTEGRATE } else { next(site) };
            assert_eq!(last.near_branch_target(), want, "site +{:#x}", site.0);
            if site.0 == CAM_ROUTE.0 {
                let after = Decoder::with_ip(64, &tramp[(last.next_ip() - tbase) as usize..], last.next_ip(), DecoderOptions::NONE).decode();
                assert_eq!(after.near_branch_target(), next(CAM_ROUTE));
                let cond = v.iter().find(|i| i.flow_control() == FlowControl::ConditionalBranch).unwrap();
                assert_eq!(cond.near_branch_target(), last.next_ip(), "route: the idle exit");
            }
        }
        // The yaw site lies between a compare and the branch on it: its stub
        // must leave the flags alone (the pitch stub too, for good measure).
        for site in [CAM_YAW, CAM_PITCH, CAM_PASSED, CAM_IDLE] {
            for i in stub(&image, &tramp, base, tbase, site.0) {
                assert_eq!(i.rflags_modified(), 0, "site +{:#x}: {i} modifies flags", site.0);
            }
        }
        // Pending motion takes the gate stub to the lock checks.
        let gate = stub(&image, &tramp, base, tbase, CAM_GATE.0);
        assert!(gate.iter().any(|i| i.flow_control() == FlowControl::ConditionalBranch && i.near_branch_target() == base + CAM_GATE_CHECKS));
        // The pitch / yaw stubs note what they applied (+12 / +16), which the
        // momentum stub tests and clears along with the two velocities.
        let stores = |site: (usize, &[u8])| -> Vec<u64> {
            stub(&image, &tramp, base, tbase, site.0).iter().filter(|i| i.mnemonic() == iced_x86::Mnemonic::Mov && i.op0_kind() == iced_x86::OpKind::Memory).map(|i| i.memory_displacement64()).collect()
        };
        assert_eq!(stores(CAM_PITCH), vec![12]);
        assert_eq!(stores(CAM_YAW), vec![16]);
        assert_eq!(stores(CAM_MOMENTUM), vec![12, 0x2530, 0x2534]);
        // Wrong bytes at any site: nothing patched.
        let mut image2 = vec![0x90u8; 0x71_0000];
        for (off, b) in &sites[..6] {
            image2[*off..*off + b.len()].copy_from_slice(b);
        }
        let mut tramp2 = vec![0u8; 0x1_0000];
        let mut p = Patcher { image: &mut image2, base, tls_slot_disp: 0, tramp_base: tbase, tramp: &mut tramp2, tramp_used: 0, data_used: 0 };
        assert!(patch_mouse_camera(&mut p).is_err());
        assert_eq!(p.tramp_used, 0);
        assert_eq!(p.image[CAM_GATE.0], CAM_GATE.1[0]);
    }
}
