//! Title code patches for Shadow of the Colossus (CUSA08809, base application 01.00,
//! main module `Shadow_Shipping`).
//!
//! None so far: the title's code runs unmodified apart from the generic load-time
//! patches in `patch.rs` (TLS, syscalls, SSE4a). The Bloodborne frame-rate patches this
//! module held in the imported revision are not applicable and were removed; frame-rate
//! options for this title do not exist yet (`target_fps` is always `None`).

use crate::patch::Patcher;

/// The title's own measured frame time, where a patch exposes it.
pub fn guest_frame_ms() -> Option<f32> {
    None
}

/// Frame-rate cap requested for the title (none: the game paces itself).
pub fn target_fps() -> Option<u32> {
    None
}

pub fn apply(_p: &mut Patcher, _name: &str) {}
