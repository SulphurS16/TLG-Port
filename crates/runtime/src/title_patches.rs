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
//! Diagnostics (opt-in): TLG_LUA_LOG=1 restores the game scripts' logging.
//! Every `log(fmt, ...)` and `GaPrint(...)` in the Lua (Havok Script) game
//! scripts ends in the C function `GaPrint`, which the retail build compiled
//! out (`xor eax, eax; ret` at eboot 0xd51780). The patch routes it to
//! `tlg_lua_print`, which logs its arguments ("lua: ...") converted by the
//! engine's own `tolstring` (eboot 0xe75cd0: lua_State, TValue*, size_t*).
//! Applied only when the code bytes are the expected ones.
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
use std::sync::atomic::{AtomicU64, Ordering};

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
    if std::env::var("TLG_LUA_LOG").is_ok_and(|v| v == "1") {
        patch_gaprint(p, name);
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

/// Mouse-look data next to the module: pending pitch and yaw change
/// (radians, f32) at +0 and +4, camera update count at +8, the pitch and
/// yaw change applied in the current integrator pass at +12 and +16 (0 =
/// no hook).
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
        a.mov(dword_ptr(rcx + 8), eax)
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

fn patch_gaprint(p: &mut Patcher, name: &str) {
    let t = TOLSTRING as usize;
    if p.image.get(GAPRINT..GAPRINT + 16) != Some(&GAPRINT_BYTES[..]) || p.image.get(t..t + 8) != Some(&TOLSTRING_BYTES[..]) {
        crate::warn!("TLG_LUA_LOG: {name} is not the expected executable (GaPrint/tolstring bytes differ): not patched");
        return;
    }
    // mov rax, imm64; jmp rax (12 bytes; the rest stays padding).
    let target = tlg_lua_print as *const () as usize as u64;
    let mut code = vec![0x48, 0xB8];
    code.extend_from_slice(&target.to_le_bytes());
    code.extend_from_slice(&[0xFF, 0xE0]);
    p.image[GAPRINT..GAPRINT + code.len()].copy_from_slice(&code);
    BASE.store(p.base, Ordering::Relaxed);
    crate::info!("TLG_LUA_LOG: script logging restored (GaPrint at {name}+{GAPRINT:#x})");
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
    crate::info!("lua: {}", parts.join(" "));
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
