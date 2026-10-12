//! Title code patches for The Last Guardian (CUSA03627, update 01.03, main
//! module `launch_tricom`).
//!
//! No behaviour patches: the title's code runs unmodified apart from the
//! generic load-time patches in `patch.rs` (TLS, syscalls, SSE4a).
//!
//! Diagnostics (opt-in): TLG_LUA_LOG=1 restores the game scripts' logging.
//! Every `log(fmt, ...)` and `GaPrint(...)` in the Lua (Havok Script) game
//! scripts ends in the C function `GaPrint`, which the retail build compiled
//! out (`xor eax, eax; ret` at eboot 0xd51780). The patch routes it to
//! `tlg_lua_print`, which logs its arguments ("lua: ...") converted by the
//! engine's own `tolstring` (eboot 0xe75cd0: lua_State, TValue*, size_t*).
//! Applied only when the code bytes are the expected ones.

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
}
