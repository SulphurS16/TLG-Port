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
//! (checked at 1280x720 and 3840x2160: picture and interface whole). The
//! graphics heap takes the PS4 Pro's size (+512 MB, the direct memory with
//! it) at any size: the base size ran out on the way into the ending's last
//! cutscene (GitHub issue #7). Applied only when the bytes are the expected
//! ones.
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
//!
//! Trico's foot pins: a per-frame job (eboot 0xe15db0) finds the entity
//! named "Condor" (Trico) and reads four bones of its skeleton component
//! (`pin_FL_pin`, ...) through the component lookup at 0xd73040, which
//! returns null when the entity has no such component - and this caller
//! never checks. While a stage script sets Trico up (`SetupAI`, about 54 ms
//! in: the title screen's Trico, 33-35 s into a boot) the entity is there
//! without it: ACCESS_VIOLATION at +0xe19214, a read of [null+0x478] (runs
//! 20261011-101844, ..., 20261025-201344). The compare there is detoured: a
//! null component takes the game's own path for a missing entity (+0xe19320:
//! the bone's slot gets the defaults for that frame) and is counted
//! (`foot_pin_misses`, logged at shutdown). Applied only when the code bytes
//! are the expected ones (`TLG_FOOT_PINS=0` keeps the original).
//!
//! Above the game's 30 fps (the 60 fps limit or none): the game's three
//! calls of `hksL_loadbuffer` (eboot 0xe7e430) go through
//! `tlg_lua_loadbuffer`, which hands the compiler `script_fixes`' copy of a
//! known script (frame-counted timers made time-based), and two fixed 1/30 s
//! steps per frame - the environment shaders' phases (0xe0f07b) and GPU
//! particle emission (0xbcb5c9) - take the frame's step instead
//! (`patch_frame_steps`). Applied only when the code bytes are the expected
//! ones; at 30 fps neither is (TLG_SCRIPT_LOG=1 routes the loads only to log
//! them).

use crate::patch::Patcher;
use crate::sysv;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

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

/// Havok Script's `hksL_loadbuffer(lua_State*, const HksCompilerSettings*,
/// const char* text, size_t size, const char* chunk name)`, which compiles
/// every script the game loads, its prologue, and its three calls: the
/// game's script-file runner (eboot 0xd4fde0, after its file read),
/// loadstring-style loads and the package loader.
const LOADBUFFER: usize = 0xe7e430;
const LOADBUFFER_BYTES: [u8; 20] = [0x55, 0x48, 0x89, 0xE5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53, 0x48, 0x81, 0xEC, 0x58, 0x01, 0x00, 0x00];
const LOADBUFFER_CALLS: [usize; 3] = [0xd4fe8e, 0xe7e37f, 0xec24bc];

pub fn apply(p: &mut Patcher, name: &str) {
    patch_gaprint(p, name);
    // (Script loads pass through tlg: the port's own script joins the game's
    // utils.lua, and the scripts' frame-counted timers get their fixes for
    // frame rates above 30, idle while the limit is the game's 30 fps - the
    // limit can change while the game runs.)
    set_frame_steps(crate::hle::libs::videoout::frame_cap() != crate::config::FrameCap::Game);
    match patch_script_loads(p) {
        Ok(()) => crate::info!("scripts: loads at {name}+{LOADBUFFER:#x} pass through tlg (the port's script; frame-counted timers time-based above 30 fps)"),
        Err(e) => crate::warn!("scripts: {name}: {e}; no PC settings in the game, and the scripts' frame-counted timers stay frame-counted"),
    }
    // (Debug: TLG_FRAME_STEPS=0 keeps them at 1/30 s per frame.)
    if !std::env::var("TLG_FRAME_STEPS").is_ok_and(|v| v == "0") {
        match patch_frame_steps(p) {
            Ok(()) => crate::info!("frame steps: above 30 fps the environment shaders' phases ({name}+{:#x}) and particle emission (+{:#x}) advance by the frame's time, not 1/30 s per frame", ENV_PHASE.0, PARTICLE_EMIT.0),
            Err(e) => crate::warn!("frame steps: {name}: {e}; they advance 1/30 s per frame"),
        }
    }
    let (w, h) = crate::gpu::vk::window::render_size();
    // (A picture wider or taller than 16:9: the game's interface keeps a
    // 16:9 area in its middle.)
    crate::gpu::UI_NOT_16_9.store(w * 9 != h * 16, Ordering::Relaxed);
    crate::gpu::RENDER_SIZE.store((w as u64) << 32 | h as u64, Ordering::Relaxed);
    match patch_render_size(p, w, h) {
        Ok(heap) => {
            let more = heap.map_or(String::new(), |s| format!("; graphics heap {} MB (+{} MB; the PS4 Pro's {} MB)", s >> 20, (s - HEAP_BASE) >> 20, HEAP_PRO >> 20));
            crate::info!("render resolution: {w}x{h} (the game's display mode, {name}+{MODE_TABLE:#x}){more}");
        }
        Err(e) => crate::warn!("render resolution: {name}: {e}; the game renders 1920x1080"),
    }
    // (At 16:9 too: the hook also notes the render camera for the
    // upscalers, `render_camera`.)
    if !std::env::var("TLG_VIEW_SHAPE").is_ok_and(|v| v == "0") {
        match patch_view_shape(p, w, h) {
            Ok((fx, fy)) if w * 9 != h * 16 => crate::info!("view shape: {w}x{h} shows the game's 16:9 views whole (projection at {name}+{:#x}: horizontal x{fx:.4}, vertical x{fy:.4})", PROJ_SITE.0),
            Ok(_) => crate::info!("view shape: 16:9 (the projection hook at {name}+{:#x} notes the render camera only)", PROJ_SITE.0),
            Err(e) if w * 9 != h * 16 => crate::warn!("view shape: {name}: {e}; the 16:9 views are stretched to {w}x{h}"),
            Err(e) => crate::warn!("view shape: {name}: {e}; no render camera for the upscalers"),
        }
    }
    match patch_kick_wait(p) {
        Ok(()) => crate::info!("render-kick wait: busy loop at {name}+{KICK_SITE:#x} detoured to a polling wait"),
        Err(e) => crate::warn!("render-kick wait: {name}: {e}; not patched"),
    }
    if !std::env::var("TLG_FOOT_PINS").is_ok_and(|v| v == "0") {
        match patch_foot_pins(p) {
            Ok(()) => crate::info!("foot pins: Trico's bone lookup at {name}+{PINS_SITE:#x} checked for a missing skeleton"),
            Err(e) => crate::warn!("foot pins: {name}: {e}; not patched"),
        }
    }
    let sixty = crate::hle::libs::videoout::frame_cap() == crate::config::FrameCap::Fps60;
    match find_step(p) {
        Ok(a) => {
            STEP_ADDR.store(a, Ordering::Relaxed);
            if sixty {
                p.image[STEP_CONSTS..STEP_CONSTS + 12].copy_from_slice(&step_bytes(true));
                STEP_60.store(true, Ordering::Relaxed);
                crate::info!("frame rate: up to 60 fps; the game's time step snaps to 1/60 s near it instead of 1/30 s ({name}+{STEP_CONSTS:#x})");
            }
        }
        Err(e) if sixty => crate::warn!("time step: {name}: {e}; 60 fps unavailable (the game's own limit applies)"),
        Err(e) => crate::info!("time step: {name}: {e} (60 fps would be unavailable)"),
    }
    crate::menu_patches::apply(p, name);
    crate::pc_prompts::apply(p, name);
    if crate::config::settings().pc.get("skip_intro").is_some_and(|v| v == "on") {
        match patch_logos(p) {
            Ok(()) => crate::info!("intro: the SIE and Japan Studio logos skipped (PC Settings; {name}+{LOGO_SITE:#x})"),
            Err(e) => crate::warn!("intro: {name}: {e}; the logos show"),
        }
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
/// The Neo size: `lea r13, [r15 + 0x20000000]` (eboot 0xdd3a8c), the base
/// size in r15 plus this displacement.
const HEAP_NEO_LEA: usize = 0xdd3a8c;
const HEAP_NEO_LEA_BYTES: [u8; 7] = [0x4D, 0x8D, 0xAF, 0x00, 0x00, 0x00, 0x20];
const HEAP_BASE: u64 = 0xb000_0000;
const HEAP_PRO: u64 = 0xd000_0000;
/// The largest heap given (4,032 MB, what 3840x2160 needs; the Pro's size
/// already exceeds a signed 32-bit size, so the game's arithmetic takes
/// sizes below 4 GiB).
const HEAP_MAX: u64 = 0xfc00_0000;
/// Graphics-heap bytes the game's render targets take per pixel of its
/// render size: at boot 466 MB at 1920x1080 and 1,138 MB at 3840x2160
/// (runs 20261015-162327, 20261106-114511), ~108 bytes per pixel.
const TARGET_BYTES_PER_PIXEL: u64 = 108;

/// The heap for a display mode of `pixels`: the PS4 Pro's size plus the
/// render targets' growth above 1080p, so the game keeps the headroom a
/// 1080p Pro has. At 3840x2160 the Pro's own size left less than the base
/// PS4 had at 1080p: 91% of it in use after loading the Canyon's start
/// from a save (run 20261106-114931); played out of the cave, with the
/// cave still loaded, a texture's creation failed and the game crashed on
/// its own error path (GitHub issue #10: launch_tricom+0x20654b in
/// JsCreateTextureFromFile, the texture pointer null + 0x70).
fn heap_size(pixels: u64) -> u64 {
    let more = pixels.saturating_sub(1920 * 1080) * TARGET_BYTES_PER_PIXEL;
    (HEAP_PRO + more.div_ceil(64 << 20) * (64 << 20)).min(HEAP_MAX)
}

/// The Shadows setting's shadow map size (the game's `_rcfg.shadowMapSize`:
/// one 32-bit depth map, 2048 on the PS4). 0.8.0's Ultra (8192, a 256 MB
/// map) is gone: the game's allocator (Gomalloc's Garlic heap, ~44 KB
/// chunks) seldom keeps a free run that long - resized in play it failed
/// with 82 MB the largest block, set at start a 64 MB load of the cave
/// failed after it (largest block 58 MB with 40% of the heap in use), and
/// either way the game crashed at launch_tricom+0x1476321 after its 10 s
/// of retries (GitHub issue #12; #7's crash). A setting of `ultra` is High.
pub fn shadow_map_size(setting: &str) -> u32 {
    match setting {
        "low" => 1024,
        "high" | "ultra" => 4096,
        _ => 2048,
    }
}

/// The graphics heap a shadow map of `size` needs beyond the PS4's (in
/// 64 MB steps): +64 MB for 4096.
fn shadow_heap(size: u32) -> u64 {
    let (s, base) = (size as u64, 2048u64);
    if s <= base { 0 } else { ((s * s - base * base) * 4).div_ceil(64 << 20) * (64 << 20) }
}

/// The shadow map size for this run, decided at start with the graphics
/// heap (`patch_render_size`; 0: the game's own). A changed setting waits
/// for the next start: the map is resized at the title, while the heap is
/// still nearly empty - resized in play the game may find no block that
/// large any more (GitHub issue #12).
pub static SHADOW_MAP: AtomicU32 = AtomicU32::new(0);

/// The graphics heap with room for a shadow map of `size` beside a heap of
/// `heap` bytes, up to the largest heap (at 3840x2160 the heap is at its
/// largest already: High's map then takes 48 MB of the game's headroom).
fn heap_with_shadows(heap: u64, size: u32) -> u64 {
    (heap + shadow_heap(size)).min(HEAP_MAX.max(heap))
}

/// The cameras' projection (eboot 0xde04fd: the perspective path of the
/// camera's matrix update, every view's): m00 = 1 / tan(fov / 2) - the
/// field of view ([cam+0x400], degrees) is horizontal - and m11 = m00 *
/// aspect ([cam+0x414], 16:9 for the game's views). Here m00 is in xmm10
/// and m11 in xmm8, both stored next (m00 here and again at 0xde0604, m11
/// at 0xde059d and, plus the lens shift, at 0xde0650); the view-projection
/// matrices and the culling planes follow from them.
const PROJ_SITE: (usize, &[u8]) = (0xde058d, &[0xC5, 0x7A, 0x11, 0x93, 0x40, 0x02, 0x00, 0x00]); // vmovss [rbx+240h],xmm10
/// m11 = m00 * aspect, just before (the aspect from [rsp+160h]).
const PROJ_M11: (usize, &[u8]) = (0xde0584, &[0xC5, 0x2A, 0x59, 0x84, 0x24, 0x60, 0x01, 0x00, 0x00]); // vmulss xmm8,xmm10,[rsp+160h]
/// The game's 16:9 views: aspect 1.7777778 (f32 0x3FE38E39); any between
/// 1.76 and 1.80 counts (positive floats order as their bits).
const ASPECT_LOW: u32 = 0x3FE1_47AE;
const ASPECT_HIGH: u32 = 0x3FE6_6666;

/// The scales of the projection's horizontal and vertical terms for a
/// `w` x `h` picture: the game's 16:9 view fits inside it - a wider picture
/// shows more to the sides (the height seen stays), a taller one more above
/// and below (the width seen stays). (1, 1) at 16:9.
pub fn view_scales(w: u32, h: u32) -> (f32, f32) {
    let (a, base) = (w as f64 / h.max(1) as f64, 16.0 / 9.0);
    if w * 9 == h * 16 {
        (1.0, 1.0)
    } else if a > base {
        ((base / a) as f32, 1.0)
    } else {
        (1.0, (a / base) as f32)
    }
}

/// The data slot of the projection hook (`patch_view_shape`): the scales,
/// then the render camera of the last 16:9 view built (0 before).
static RENDER_CAM: AtomicU64 = AtomicU64::new(0);

/// The game's render camera of its last 16:9 view, as the upscalers want
/// it: near and far plane and the vertical field of view (radians). Err:
/// why not (before the first view, without the hook, unexpected values).
pub fn render_camera() -> Result<(f32, f32, f32), String> {
    let slot = RENDER_CAM.load(Ordering::Relaxed);
    if slot == 0 {
        return Err("no projection hook".into());
    }
    // SAFETY: the hook's data slot (16 bytes, the pointer at +8) and the
    // camera object in guest memory (checked).
    let cam = unsafe { std::ptr::read_volatile((slot + 8) as *const u64) };
    if cam == 0 || !crate::mem::is_guest(cam + 0x400, 0x20) {
        return Err(format!("no render camera yet ({cam:#x})"));
    }
    let f = |o: u64| unsafe { std::ptr::read_volatile((cam + o) as *const f32) };
    // (+0x400 the horizontal field of view in degrees - 70 in the cave -,
    // +0x414 the aspect, +0x418 and +0x41c the near and far planes - 0.1
    // and 2000 there: the projection builder's inputs.)
    let (hfov, aspect, near, far) = (f(0x400), f(0x414), f(0x418), f(0x41c));
    if !(hfov > 1.0 && hfov < 179.0 && aspect > 0.1 && near > 0.0 && far > near) {
        return Err(format!("unexpected values at {cam:#x}: field of view {hfov}, aspect {aspect}, near {near}, far {far}"));
    }
    Ok((near, far, 2.0 * ((hfov.to_radians() * 0.5).tan() / aspect).atan()))
}

/// Pictures that are not 16:9 (PC Settings `aspect_ratio`): each 16:9
/// view's projection (camera aspect 16:9: the game's own views; others,
/// such as square shadow views, stay) scaled by `view_scales`. The game's
/// fields of view, aspect and focal lengths stay as they are, so its
/// scripts, depth of field and level of detail see the PS4's values. At
/// 16:9 the scales are 1; either way the hook keeps the render camera of
/// the last 16:9 view (`render_camera`).
fn patch_view_shape(p: &mut Patcher, w: u32, h: u32) -> Result<(f32, f32), String> {
    use iced_x86::code_asm::*;
    let (fx, fy) = view_scales(w, h);
    if p.image.get(PROJ_M11.0..PROJ_M11.0 + PROJ_M11.1.len()) != Some(PROJ_M11.1) {
        return Err(format!("unexpected code at +{:#x} (not update 01.03?)", PROJ_M11.0));
    }
    let data: Vec<u8> = [&fx.to_le_bytes()[..], &fy.to_le_bytes()[..], &0u64.to_le_bytes()[..]].concat();
    let slot = p.alloc_data(&data).ok_or("no data space")?;
    let next = p.base + (PROJ_SITE.0 + PROJ_SITE.1.len()) as u64;
    // (Flags are dead here: the next use follows a compare at 0xde06a1.)
    p.detour(PROJ_SITE.0, PROJ_SITE.1, |a, ins| {
        let mut skip = a.create_label();
        a.lea(rsp, ptr(rsp - 128))?;
        a.push(rax)?;
        a.mov(eax, dword_ptr(rbx + 0x414))?;
        a.sub(eax, ASPECT_LOW as i32)?;
        a.cmp(eax, (ASPECT_HIGH - ASPECT_LOW) as i32)?;
        a.ja(skip)?;
        a.mov(rax, slot)?;
        a.mov(qword_ptr(rax + 8), rbx)?;
        a.vmulss(xmm10, xmm10, dword_ptr(rax))?;
        a.vmulss(xmm8, xmm8, dword_ptr(rax + 4))?;
        a.set_label(&mut skip)?;
        a.pop(rax)?;
        a.lea(rsp, ptr(rsp + 128))?;
        a.add_instruction(*ins)?;
        a.jmp(next)
    })?;
    RENDER_CAM.store(slot, Ordering::Relaxed);
    Ok((fx, fy))
}

/// Sets the game's display mode to render and show `w` x `h`, and gives it
/// at least the PS4 Pro's graphics heap, as a PS4 Pro does at any output:
/// the base size ran out where the game, played into the ending's last
/// cutscene, loads it over the area before (Gomalloc "Garlic: Out of
/// Memory", "vram allocation failed! wait for 10 secs...", then a crash in
/// its loader at launch_tricom+0x1476321; GitHub issue #7: all 2816 MB in
/// use); above 1080p it grows with the render targets (`heap_size`).
/// Returns the heap's size in bytes when it grew (`TLG_PRO_HEAP=0` keeps
/// the base size).
fn patch_render_size(p: &mut Patcher, w: u32, h: u32) -> Result<Option<u64>, String> {
    if !(64..=8192).contains(&w) || !(64..=8192).contains(&h) {
        return Err(format!("{w}x{h} is out of range"));
    }
    if p.image.get(MODE_TABLE..MODE_TABLE + 12) != Some(&MODE_1080P[..]) {
        return Err("not the expected executable (display-mode table differs)".into());
    }
    let heap = !std::env::var("TLG_PRO_HEAP").is_ok_and(|v| v == "0");
    if heap && (HEAP_SITES.iter().any(|(off, b)| p.image.get(*off..*off + 4) != Some(&b[..])) || p.image.get(HEAP_NEO_LEA..HEAP_NEO_LEA + 7) != Some(&HEAP_NEO_LEA_BYTES[..])) {
        return Err("not the expected executable (graphics heap code differs)".into());
    }
    let v = (w | h << 16).to_le_bytes();
    // The game's inner render size: upscaling (`upscale::render_scale`) or
    // TLG_INNER_SIZE=WxH; the game then scales its finished picture up to the
    // output in one draw (the upscaler's place: render.rs `upscale_hook`).
    let scale = crate::gpu::upscale::render_scale();
    let inner = std::env::var("TLG_INNER_SIZE")
        .ok()
        .and_then(|s| {
            let (a, b) = s.split_once('x')?;
            Some((a.trim().parse::<u32>().ok()?, b.trim().parse::<u32>().ok()?))
        })
        .or_else(|| (scale < 1.0).then(|| (((w as f32 * scale / 2.0).round() as u32 * 2).max(64), ((h as f32 * scale / 2.0).round() as u32 * 2).max(64))));
    if let Some((iw, ih)) = inner {
        crate::gpu::SCENE_SIZE.store((iw as u64) << 32 | ih as u64, Ordering::Relaxed);
        crate::info!("upscaling: the game renders {iw}x{ih} and scales it to {w}x{h} (render scale {:.2}, texture LOD bias {:.2})", iw as f32 / w as f32, crate::gpu::upscale::mip_bias());
    }
    let r = inner.map_or(v, |(iw, ih)| (iw | ih << 16).to_le_bytes());
    p.image[MODE_TABLE..MODE_TABLE + 4].copy_from_slice(&r);
    p.image[MODE_TABLE + 4..MODE_TABLE + 8].copy_from_slice(&v);
    let wanted = shadow_map_size(crate::config::settings().pc.get("shadows").map_or("ps4", String::as_str));
    if !heap {
        // (The base heap: no room for larger shadow maps.)
        SHADOW_MAP.store(wanted.min(2048), Ordering::Relaxed);
        return Ok(None);
    }
    for (off, _) in HEAP_SITES {
        p.image[off..off + 4].copy_from_slice(&[0x0F, 0x1F, 0x40, 0x00]); // nop: the Neo size stays
    }
    let px = |b: [u8; 4]| u32::from_le_bytes(b) as u64 & 0xffff;
    let pixels = (px(r) * (u32::from_le_bytes(r) as u64 >> 16)).max(px(v) * (u32::from_le_bytes(v) as u64 >> 16));
    let size = heap_size(pixels);
    // (The Shadows setting's map, with the heap it needs on top.)
    SHADOW_MAP.store(wanted, Ordering::Relaxed);
    let with = heap_with_shadows(size, wanted);
    if wanted != 2048 {
        crate::info!("shadows: {wanted}x{wanted} shadow map (the PS4's 2048x2048){}", if with > size { format!(", graphics heap +{} MB for it", (with - size) >> 20) } else { String::new() });
    }
    let size = with;
    p.image[HEAP_NEO_LEA + 3..HEAP_NEO_LEA + 7].copy_from_slice(&((size - HEAP_BASE) as u32).to_le_bytes());
    crate::mem::add_dmem(size - HEAP_BASE);
    Ok(Some(size))
}

/// The title's frame time step (eboot 0xd16480, called by the frame update
/// at 0xdd48f7): the measured real time, smoothed (0.8-1.25x through an
/// error accumulator), clamped to the minimum frame rate (1/20 s) and scaled
/// by the game speed; while the Havok world exists a step between 0.031667
/// and 0.035 s is snapped to exactly 1/30 s, then filtered and rounded down
/// to 1/360 s (floor(t * 360 + 0.1) / 360). The snap keeps the step exact at
/// the game's 30 fps limit; at the 60 fps limit it moves to 1/60 s (0.015833
/// to 0.0175 s) - without it a 1/60 s step a little short rounds down to
/// 1/72 s. Between the two rates the step is real time either way. These
/// are the snap's three constants (upper bound, step, lower bound) at
/// 0x14f17f8, read only by the instructions in `STEP_SITES`.
const STEP_CONSTS: usize = 0x14f17f8;
const SNAP_30: [u32; 3] = [0x3d0f5c29, 0x3d088889, 0x3d01b4e8]; // 0.035, 1/30, 0.031666666
const SNAP_60: [u32; 3] = [0x3c8f5c29, 0x3c888889, 0x3c81b4e8]; // 0.0175, 1/60, 0.015833333
const STEP_SITES: [(usize, &[u8]); 3] = [
    (0xd165e2, &[0xC5, 0xFA, 0xC2, 0x15, 0x0D, 0xB2, 0x7D, 0x00, 0x01]), // vcmpltss xmm2,xmm0,[14F17F8h]
    (0xd165eb, &[0xC5, 0xFA, 0x10, 0x1D, 0x09, 0xB2, 0x7D, 0x00]),       // vmovss xmm3,[14F17FCh]
    (0xd165ff, &[0xC5, 0xFA, 0x10, 0x1D, 0xF9, 0xB1, 0x7D, 0x00]),       // vmovss xmm3,[14F1800h]
];

/// The loaded snap constants (0: the executable's step code is not the
/// expected one), and whether they snap to 1/60 s.
static STEP_ADDR: AtomicU64 = AtomicU64::new(0);
static STEP_60: AtomicBool = AtomicBool::new(false);

fn step_bytes(sixty: bool) -> [u8; 12] {
    let mut b = [0u8; 12];
    for (i, v) in (if sixty { SNAP_60 } else { SNAP_30 }).iter().enumerate() {
        b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    b
}

/// The snap constants' address when the step code and its constants are
/// the expected ones.
fn find_step(p: &Patcher) -> Result<u64, String> {
    if let Some((off, _)) = STEP_SITES.iter().find(|(off, b)| p.image.get(*off..*off + b.len()) != Some(*b)) {
        return Err(format!("unexpected code at +{off:#x} (not update 01.03?)"));
    }
    if p.image.get(STEP_CONSTS..STEP_CONSTS + 12) != Some(&step_bytes(false)[..]) {
        return Err("unexpected time-step constants".into());
    }
    Ok(p.base + STEP_CONSTS as u64)
}

/// Snaps the game's time step to 1/60 s (`sixty`) or 1/30 s near that rate
/// from the next frame on; false where that is not possible (the
/// executable's step code is not the expected one: it stays at 1/30 s).
pub fn set_step_60(sixty: bool) -> bool {
    if STEP_60.load(Ordering::Relaxed) == sixty {
        return true;
    }
    let a = STEP_ADDR.load(Ordering::Relaxed);
    if a == 0 {
        return false;
    }
    if let Err(e) = crate::debug_mem::write_guest(a, &step_bytes(sixty)) {
        crate::warn!("time step: {e}");
        return false;
    }
    STEP_60.store(sixty, Ordering::Relaxed);
    crate::info!("time step: snapped to 1/{} s near that frame rate from now on", if sixty { 60 } else { 30 });
    true
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

/// Debug (control `mouselat <n>`): n turns of 40 counts injected at random
/// moments, each timed until the game camera's yaw ([cam+0x2244]) moves
/// (polled every ~0.2 ms); logs the median, 90th percentile and worst time.
pub fn mouse_latency_probe(n: u32) {
    let _ = std::thread::Builder::new().name("mouse-latency".into()).spawn(move || {
        let cam = camera_object();
        if cam == 0 || !crate::mem::is_guest(cam + 0x2244, 4) {
            crate::info!("mouselat: no camera object");
            return;
        }
        let yaw = || unsafe { std::ptr::read_volatile((cam + 0x2244) as *const f32) };
        let mut times = Vec::new();
        let mut seed = crate::log::elapsed_ms().to_bits() | 1;
        for i in 0..n {
            // (A random phase against the game's frames: 40-140 ms apart.)
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            std::thread::sleep(std::time::Duration::from_millis(40 + seed % 100));
            let before = yaw();
            let t0 = std::time::Instant::now();
            crate::gpu::vk::window::inject_mouse(if i % 2 == 0 { 40 } else { -40 }, 0, 0);
            while yaw() == before && t0.elapsed().as_millis() < 500 {
                std::hint::spin_loop();
                std::thread::yield_now();
            }
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(f64::total_cmp);
        let pct = |p: f64| times[((times.len() as f64 * p) as usize).min(times.len() - 1)];
        crate::info!(
            "mouselat: {} turns, motion to camera: median {:.1} ms, p90 {:.1} ms, worst {:.1} ms, best {:.1} ms (mouse_direct {})",
            times.len(),
            pct(0.5),
            pct(0.9),
            times[times.len() - 1],
            times[0],
            crate::tune::get(crate::tune::MOUSE_DIRECT)
        );
    });
}

/// Flips left to log the camera for (control `camlog <flips>`).
static CAMLOG: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub fn camlog(flips: u32) {
    CAMLOG.store(flips, Ordering::Relaxed);
}

/// At each displayed flip while `camlog` runs: the camera's pitch and yaw
/// (+0x2240/+0x2244, radians), eye (+0x2370) and pivot (+0x23c0), its follow
/// target (+0x22b0: the boy) and the mouse turn applied in the last
/// integrator pass.
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
        "camlog flip {flip} t {:.1}: pitch {:.5} yaw {:.5} eye ({:.3} {:.3} {:.3}) pivot ({:.3} {:.3} {:.3}) boy ({:.3} {:.3} {:.3}) applied ({:.5} {:.5})",
        crate::log::elapsed_ms(),
        f(0x2240),
        f(0x2244),
        f(0x2370),
        f(0x2374),
        f(0x2378),
        f(0x23c0),
        f(0x23c4),
        f(0x23c8),
        f(0x22b0),
        f(0x22b4),
        f(0x22b8),
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

/// Routes the game's three calls of `hksL_loadbuffer` through
/// `tlg_lua_loadbuffer` (a trampoline near the module: `mov rax, imm64; jmp
/// rax`), which may hand the compiler a fixed copy of the script's text.
fn patch_script_loads(p: &mut Patcher) -> Result<(), String> {
    if p.image.get(LOADBUFFER..LOADBUFFER + LOADBUFFER_BYTES.len()) != Some(&LOADBUFFER_BYTES[..]) {
        return Err(format!("unexpected code at +{LOADBUFFER:#x} (not update 01.03?)"));
    }
    for &site in &LOADBUFFER_CALLS {
        let ok = p.image.get(site..site + 5).is_some_and(|b| b[0] == 0xE8 && (site as i64 + 5 + i32::from_le_bytes([b[1], b[2], b[3], b[4]]) as i64) as usize == LOADBUFFER);
        if !ok {
            return Err(format!("+{site:#x} is not a call of +{LOADBUFFER:#x}"));
        }
    }
    let target = tlg_lua_loadbuffer as *const () as usize as u64;
    let mut code = vec![0x48, 0xB8];
    code.extend_from_slice(&target.to_le_bytes());
    code.extend_from_slice(&[0xFF, 0xE0]);
    let t = p.emit_trampoline(&code).ok_or("trampoline space exhausted")?;
    for &site in &LOADBUFFER_CALLS {
        let rel = t as i64 - (p.base as i64 + site as i64 + 5);
        let rel = i32::try_from(rel).map_err(|_| "trampoline out of rel32 range".to_string())?;
        p.image[site + 1..site + 5].copy_from_slice(&rel.to_le_bytes());
    }
    BASE.store(p.base, Ordering::Relaxed);
    Ok(())
}

/// A script load: the text as the game read it, or a fixed copy
/// (`script_fixes`), compiled by the game's `hksL_loadbuffer`.
unsafe fn lua_loadbuffer(l: u64, opts: u64, text: u64, size: u64, chunk: u64) -> i32 {
    let orig = BASE.load(Ordering::Relaxed) + LOADBUFFER as u64;
    let fixed = (text != 0 && size != 0 && size <= 64 << 20).then(|| {
        let name = if chunk != 0 { unsafe { cstr(chunk, 512) } } else { String::new() };
        let bytes = unsafe { std::slice::from_raw_parts(text as *const u8, size as usize) };
        crate::script_fixes::load(l, &name, bytes)
    });
    match fixed.flatten() {
        // (Ours lives until the compiler is done with it.)
        Some(v) => {
            let r = unsafe { crate::hle::call_guest(orig, &[l, opts, v.as_ptr() as u64, v.len() as u64, chunk]) as i32 };
            if r != 0 {
                let name = if chunk != 0 { unsafe { cstr(chunk, 512) } } else { String::new() };
                crate::error!("scripts: {name}: the fixed text did not compile (status {r})");
            }
            r
        }
        None => unsafe { crate::hle::call_guest(orig, &[l, opts, text, size, chunk]) as i32 },
    }
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

/// The foot-pin job's compare of the looked-up skeleton component (module
/// docs), the branch on its flags, and the path for a missing entity.
const PINS_SITE: usize = 0xe19214;
const PINS_BYTES: [u8; 7] = [0x83, 0xB8, 0x78, 0x04, 0x00, 0x00, 0x00]; // cmp dword [rax+478h], 0
const PINS_NEXT: u64 = 0xe1921b;
const PINS_MISSING: u64 = 0xe19320;
/// The data slot counting lookups that found no skeleton (0: not patched).
static PINS_COUNT: AtomicU64 = AtomicU64::new(0);

fn patch_foot_pins(p: &mut Patcher) -> Result<(), String> {
    use iced_x86::code_asm::*;
    let count = p.alloc_data(&[0u8; 8]).ok_or("no data space")?;
    let base = p.base;
    p.detour(PINS_SITE, &PINS_BYTES, |a, ins| {
        let mut found = a.create_label();
        a.test(rax, rax)?;
        a.jne(found)?;
        // (rax and the flags are dead on the missing-entity path.)
        a.mov(rax, count)?;
        a.lock().inc(qword_ptr(rax))?;
        a.jmp(base + PINS_MISSING)?;
        a.set_label(&mut found)?;
        a.add_instruction(*ins)?;
        a.jmp(base + PINS_NEXT)
    })?;
    PINS_COUNT.store(count, Ordering::Relaxed);
    Ok(())
}

/// The game's symbol table: 32-byte entries, the name's pointer first.
const SYMBOLS: u64 = 0x1716230;

/// A symbol's name (the game's symbol table).
pub fn symbol_name(id: u32) -> String {
    let b = BASE.load(Ordering::Relaxed);
    if b == 0 || id == 0 || id > 0x10_0000 {
        return format!("#{id}");
    }
    let e = b + SYMBOLS + id as u64 * 32;
    if !crate::mem::is_guest(e, 8) {
        return format!("#{id}");
    }
    // SAFETY: the symbol table's entry (guest memory checked).
    let s = unsafe { *(e as *const u64) };
    if s == 0 { format!("#{id}") } else { unsafe { cstr(s, 128) } }
}

/// Foot-pin lookups that found no skeleton (`patch_foot_pins`), if patched.
pub fn foot_pin_misses() -> Option<u64> {
    let slot = PINS_COUNT.load(Ordering::Relaxed);
    // SAFETY: the patch's data slot (writable, never freed).
    (slot != 0).then(|| unsafe { std::ptr::read_volatile(slot as *const u64) })
}

/// The app object's pointer (module-relative) and its frame step: the
/// game's time step for the frame, seconds (1/30 at its own limit).
const APP_PTR: u64 = 0x59a1b40;
const APP_STEP: i32 = 0x19c;
/// Effects that advance a fixed 1/30 s per frame (LOG: The scripts'
/// frame-counted timers): the environment shaders' twelve phases
/// (`_default_env_cs`: phase += rate x speed / 30) and the GPU particle
/// systems' emission (count = rate / 30 + carry). Above 30 fps they take the
/// frame's step instead; both constants are 1/30.
const ENV_PHASE: (usize, &[u8], usize) = (0xe0f07b, &[0xC5, 0xF2, 0x59, 0x0D, 0xC9, 0x2A, 0x6E, 0x00], 0x14f1b4c); // vmulss xmm1,xmm1,[1/30]
const PARTICLE_EMIT: (usize, &[u8], usize) = (0xbcb5c9, &[0xC5, 0xFA, 0x10, 0x05, 0xCB, 0x5A, 0x92, 0x00], 0x14f109c); // vmovss xmm0,[1/30]

/// Whether the frame-step detours take the frame's step (1) or the game's
/// 1/30 s (0: the frame-rate limit is the game's own).
static FRAME_STEPS: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// The frame-rate limit changed: above 30 fps the environment phases and
/// particle emission follow the frame's time, at 30 the game's 1/30 s.
pub fn set_frame_steps(on: bool) {
    FRAME_STEPS.store(on as u8, Ordering::Relaxed);
}

fn patch_frame_steps(p: &mut Patcher) -> Result<(), String> {
    use iced_x86::code_asm::*;
    let third = (1.0f32 / 30.0).to_le_bytes();
    for (off, _, k) in [ENV_PHASE, PARTICLE_EMIT] {
        if p.image.get(k..k + 4) != Some(&third[..]) {
            return Err(format!("+{off:#x}: its constant at +{k:#x} is not 1/30 (not update 01.03?)"));
        }
    }
    let base = p.base;
    // `step` (the step, or the constant before the app object exists) into
    // the instruction's operand; every register and the flags kept.
    let with_step = |a: &mut CodeAssembler, (off, bytes, k): (usize, &[u8], usize), op: &dyn Fn(&mut CodeAssembler, AsmMemoryOperand) -> Result<(), IcedError>| -> Result<(), IcedError> {
        let mut none = a.create_label();
        let mut done = a.create_label();
        a.lea(rsp, ptr(rsp - 128))?;
        a.pushfq()?;
        a.push(rax)?;
        a.mov(rax, &FRAME_STEPS as *const _ as u64)?;
        a.cmp(byte_ptr(rax), 0)?;
        a.je(none)?;
        a.mov(rax, base + APP_PTR)?;
        a.mov(rax, qword_ptr(rax))?;
        a.test(rax, rax)?;
        a.je(none)?;
        op(a, dword_ptr(rax + APP_STEP))?;
        a.jmp(done)?;
        a.set_label(&mut none)?;
        a.mov(rax, base + k as u64)?;
        op(a, dword_ptr(rax))?;
        a.set_label(&mut done)?;
        a.pop(rax)?;
        a.popfq()?;
        a.lea(rsp, ptr(rsp + 128))?;
        a.jmp(base + (off + bytes.len()) as u64)
    };
    p.detour(ENV_PHASE.0, ENV_PHASE.1, |a, _| with_step(a, ENV_PHASE, &|a, m| a.vmulss(xmm1, xmm1, m)))?;
    p.detour(PARTICLE_EMIT.0, PARTICLE_EMIT.1, |a, _| with_step(a, PARTICLE_EMIT, &|a, m| a.vmovss(xmm0, m)))?;
    Ok(())
}

/// The boot sequence's logo loop (eboot 0x231f21, `lea r13, [rsp+5F8h]`,
/// right after its table of logos is built on the stack: entries of 0x1C
/// bytes from rsp+0x6F0, each starting with its "shown" flag - the SIE
/// logo's at rsp+0x728, Japan Studio's at rsp+0x744).
const LOGO_SITE: usize = 0x231f21;
const LOGO_BYTES: [u8; 8] = [0x4C, 0x8D, 0xAC, 0x24, 0xF8, 0x05, 0x00, 0x00];

/// PC Settings `skip_intro`: the boot sequence shows neither logo (the
/// title screen and the game's own boot steps stay).
fn patch_logos(p: &mut Patcher) -> Result<(), String> {
    use iced_x86::code_asm::*;
    let base = p.base;
    p.detour(LOGO_SITE, &LOGO_BYTES, |a, ins| {
        // (Flags are set again after the site; rsp as the code has it.)
        a.mov(dword_ptr(rsp + 0x728), 0)?;
        a.mov(dword_ptr(rsp + 0x744), 0)?;
        a.add_instruction(*ins)?;
        a.jmp(base + (LOGO_SITE + LOGO_BYTES.len()) as u64)
    })?;
    Ok(())
}

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

/// Havok Script's `lua_pushlstring(lua_State*, const char*, size_t)` (as
/// the game's `GameMenu_GetResult` binding uses it) and its prologue.
const PUSHLSTRING: u64 = 0xd505e0;
const PUSHLSTRING_BYTES: [u8; 8] = [0x55, 0x48, 0x89, 0xE5, 0x41, 0x56, 0x53, 0x48];

/// Logs the string forms of the call's arguments; a request of the port's
/// script (`pc_menu::MARKER` first) is answered instead: the answers joined
/// by U+001F into one string, pushed once (pc_menu.lua's `call` splits it).
/// One push only: a C function may use just the few stack slots Lua
/// guarantees it, and pushing each answer (a page's rows are 180 values)
/// ran past the script stack into the heap - the Achievements page with
/// the trophies' names crashed the game in a table lookup.
unsafe fn lua_print(l: u64) -> i32 {
    let base_mod = BASE.load(Ordering::Relaxed);
    if base_mod == 0 || !crate::mem::is_guest(l, 0x60) {
        return 0;
    }
    // lua_State: stack top at +0x48, the call's first argument at +0x50; 16-byte TValues.
    let (top, base) = unsafe { (*((l + 0x48) as *const u64), *((l + 0x50) as *const u64)) };
    if top < base || top - base > 32 * 16 || !crate::mem::is_guest(base, (top - base).max(1)) {
        return 0;
    }
    let mut parts = Vec::new();
    for i in 0..(top - base) / 16 {
        let s = unsafe { crate::hle::call_guest(base_mod + TOLSTRING, &[l, base + i * 16, 0]) };
        parts.push(if s != 0 { unsafe { cstr(s, 2048) } } else { "<?>".to_string() });
    }
    if parts.first().is_some_and(|p| p == crate::pc_menu::MARKER) {
        static PUSH_OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        // SAFETY: the module's code (read-only image).
        let ok = *PUSH_OK.get_or_init(|| unsafe { std::slice::from_raw_parts((base_mod + PUSHLSTRING) as *const u8, 8) } == PUSHLSTRING_BYTES);
        if !ok {
            return 0;
        }
        let answers = crate::pc_menu::call(&parts[1..]);
        if answers.is_empty() {
            return 0;
        }
        let joined = join_answers(&answers);
        unsafe { crate::hle::call_guest(base_mod + PUSHLSTRING, &[l, joined.as_ptr() as u64, joined.len() as u64]) };
        return 1;
    }
    let text = parts.join(" ");
    note_script(&text);
    static LOG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *LOG.get_or_init(|| std::env::var("TLG_LUA_LOG").is_ok_and(|v| v == "1")) {
        crate::info!("lua: {text}");
    }
    0
}

/// The answers as one string, joined by U+001F (a separator inside an
/// answer becomes a space).
fn join_answers(answers: &[String]) -> String {
    answers.iter().map(|a| a.replace('\u{1f}', " ")).collect::<Vec<_>>().join("\u{1f}")
}

sysv! {
    fn tlg_lua_print(l: u64) -> i32 {
        lua_print(l)
    }
    fn tlg_lua_loadbuffer(l: u64, opts: u64, text: u64, size: u64, chunk: u64) -> i32 {
        lua_loadbuffer(l, opts, text, size, chunk)
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

    #[test]
    fn answers_join() {
        let a = ["x".to_string(), String::new(), "a\u{1f}b".to_string()];
        assert_eq!(join_answers(&a), "x\u{1f}\u{1f}a b");
        assert_eq!(join_answers(&["".to_string()]), "");
    }

    #[test]
    fn shadow_maps_and_the_heap() {
        assert_eq!((shadow_map_size("low"), shadow_map_size("ps4"), shadow_map_size("high"), shadow_map_size("ultra")), (1024, 2048, 4096, 4096));
        assert_eq!(shadow_heap(1024), 0);
        assert_eq!(shadow_heap(2048), 0);
        assert_eq!(shadow_heap(4096), 64 << 20);
        // 1080p and 1440p: +64 MB for High's map.
        assert_eq!(heap_with_shadows(heap_size(1920 * 1080), 4096), HEAP_PRO + (64 << 20));
        assert_eq!(heap_with_shadows(heap_size(2560 * 1440), 4096), heap_size(2560 * 1440) + (64 << 20));
        // 3840x2160: the heap is at its largest already.
        assert_eq!(heap_with_shadows(heap_size(3840 * 2160), 4096), HEAP_MAX);
        assert_eq!(heap_with_shadows(heap_size(1920 * 1080), 2048), HEAP_PRO);
    }

    #[test]
    fn heap_grows_with_the_render_size() {
        assert_eq!(heap_size(1280 * 720), HEAP_PRO);
        assert_eq!(heap_size(1920 * 1080), HEAP_PRO);
        // 2560x1440: +175 MB of targets, in 64 MB steps.
        assert_eq!(heap_size(2560 * 1440), HEAP_PRO + (192 << 20));
        // 3840x2160: +641 MiB -> +704 MiB, the cap.
        assert_eq!(heap_size(3840 * 2160), HEAP_PRO + (704 << 20));
        assert_eq!(heap_size(3840 * 2160), HEAP_MAX);
        assert_eq!(heap_size(7680 * 4320), HEAP_MAX);
        // The Neo displacement stays a positive 32-bit value.
        assert!(HEAP_MAX - HEAP_BASE < 0x8000_0000);
    }

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
    fn frame_step_stubs() {
        let base = 0x6000_0000u64;
        let mut image = vec![0x90u8; 0x150_0000];
        for (off, b, k) in [ENV_PHASE, PARTICLE_EMIT] {
            image[off..off + b.len()].copy_from_slice(b);
            image[k..k + 4].copy_from_slice(&(1.0f32 / 30.0).to_le_bytes());
        }
        let mut tramp = vec![0u8; 0x1_0000];
        let tbase = base + 0x160_0000;
        {
            let mut p = Patcher { image: &mut image, base, tls_slot_disp: 0, tramp_base: tbase, tramp: &mut tramp, tramp_used: 0, data_used: 0 };
            patch_frame_steps(&mut p).unwrap();
        }
        for (off, b, _) in [ENV_PHASE, PARTICLE_EMIT] {
            assert_eq!(image[off], 0xE9);
            let rel = i32::from_le_bytes(image[off + 1..off + 5].try_into().unwrap()) as i64;
            let t = (base as i64 + off as i64 + 5 + rel) as u64;
            let mut d = Decoder::with_ip(64, &tramp[(t - tbase) as usize..], t, DecoderOptions::NONE);
            let back = base + (off + b.len()) as u64;
            let mut v = Vec::new();
            loop {
                let i = d.decode();
                assert!(!i.is_invalid() && v.len() < 40, "site +{off:#x}");
                v.push(i);
                if i.flow_control() == FlowControl::UnconditionalBranch && i.near_branch_target() == back {
                    break;
                }
            }
            use iced_x86::Mnemonic as M;
            let n = |m: M| v.iter().filter(|i| i.mnemonic() == m).count();
            assert_eq!(n(M::Push) + n(M::Pushfq), n(M::Pop) + n(M::Popfq), "site +{off:#x}");
            assert!(v.iter().any(|i| matches!(i.mnemonic(), M::Vmulss | M::Vmovss) && i.memory_base() == iced_x86::Register::RAX && i.memory_displacement64() == APP_STEP as u64), "site +{off:#x}");
        }
        // Another constant: not patched.
        let mut image2 = vec![0x90u8; 0x150_0000];
        image2[ENV_PHASE.0..ENV_PHASE.0 + ENV_PHASE.1.len()].copy_from_slice(ENV_PHASE.1);
        let mut tramp2 = vec![0u8; 0x1_0000];
        let mut p = Patcher { image: &mut image2, base, tls_slot_disp: 0, tramp_base: tbase, tramp: &mut tramp2, tramp_used: 0, data_used: 0 };
        assert!(patch_frame_steps(&mut p).is_err());
        assert_eq!(p.image[ENV_PHASE.0], 0xC5);
    }

    #[test]
    fn view_shape_stub() {
        assert_eq!(view_scales(1920, 1080), (1.0, 1.0));
        assert_eq!(view_scales(1360, 764).1, 1.0);
        let (fx, fy) = view_scales(2560, 1080);
        assert!((fx - 0.75).abs() < 1e-6 && fy == 1.0, "{fx} {fy}");
        let (fx, fy) = view_scales(1920, 1200);
        assert!(fx == 1.0 && (fy - 0.9).abs() < 1e-6, "{fx} {fy}");
        let base = 0x6000_0000u64;
        let mut image = vec![0x90u8; 0xe0_0000];
        for (off, b) in [PROJ_SITE, PROJ_M11] {
            image[off..off + b.len()].copy_from_slice(b);
        }
        let mut tramp = vec![0u8; 0x1_0000];
        let tbase = base + 0x100_0000;
        {
            let mut p = Patcher { image: &mut image, base, tls_slot_disp: 0, tramp_base: tbase, tramp: &mut tramp, tramp_used: 0, data_used: 0 };
            assert_eq!(patch_view_shape(&mut p, 2560, 1080).unwrap(), view_scales(2560, 1080));
        }
        let v = stub(&image, &tramp, base, tbase, PROJ_SITE.0);
        // The original store last, then back after it; the stack balanced.
        let last = v.last().unwrap();
        assert_eq!(last.near_branch_target(), base + (PROJ_SITE.0 + PROJ_SITE.1.len()) as u64);
        let store = &v[v.len() - 2];
        assert_eq!(&tramp[(store.ip() - tbase) as usize..][..store.len()], PROJ_SITE.1);
        let count = |m| v.iter().filter(|i| i.mnemonic() == m).count();
        assert_eq!(count(iced_x86::Mnemonic::Push), count(iced_x86::Mnemonic::Pop));
        // Only views of about the game's 16:9 aspect, both terms scaled.
        let load = v.iter().find(|i| i.mnemonic() == iced_x86::Mnemonic::Mov && i.op1_kind() == iced_x86::OpKind::Memory).unwrap();
        assert_eq!((load.memory_base(), load.memory_displacement64()), (iced_x86::Register::RBX, 0x414));
        assert!(ASPECT_LOW < 0x3FE3_8E39 && 0x3FE3_8E39 < ASPECT_HIGH);
        assert_eq!((f32::from_bits(ASPECT_LOW), f32::from_bits(ASPECT_HIGH)), (1.76, 1.8));
        assert_eq!(count(iced_x86::Mnemonic::Vmulss), 2);
        // A different executable: refused.
        image[PROJ_M11.0] = 0x90;
        let mut p = Patcher { image: &mut image, base, tls_slot_disp: 0, tramp_base: tbase, tramp: &mut tramp, tramp_used: 0, data_used: 0 };
        assert!(patch_view_shape(&mut p, 2560, 1080).is_err());
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

    #[test]
    fn foot_pins_stub() {
        use iced_x86::{Mnemonic as M, OpKind, Register as R};
        let base = 0x6000_0000u64;
        let mut image = vec![0x90u8; 0xe2_0000];
        image[PINS_SITE..PINS_SITE + PINS_BYTES.len()].copy_from_slice(&PINS_BYTES);
        let mut tramp = vec![0u8; 0x1_0000];
        let tbase = base + 0xf0_0000;
        {
            let mut p = Patcher { image: &mut image, base, tls_slot_disp: 0, tramp_base: tbase, tramp: &mut tramp, tramp_used: 0, data_used: 0 };
            patch_foot_pins(&mut p).unwrap();
        }
        let count = PINS_COUNT.swap(0, Ordering::Relaxed);
        assert!(count >= tbase && count < tbase + 0x1_0000);
        // The rest of the replaced compare is never executed.
        assert_eq!(&image[PINS_SITE + 5..PINS_SITE + 7], &[0xCC, 0xCC]);
        // A null component: counted, then the game's path for a missing entity.
        let v = stub(&image, &tramp, base, tbase, PINS_SITE);
        assert_eq!((v[0].mnemonic(), v[0].op0_register(), v[0].op1_register()), (M::Test, R::RAX, R::RAX));
        assert_eq!(v[1].mnemonic(), M::Jne);
        assert!(v.iter().any(|i| i.mnemonic() == M::Mov && i.op0_register() == R::RAX && matches!(i.op1_kind(), OpKind::Immediate64 | OpKind::Immediate32to64) && i.immediate(1) == count));
        assert!(v.iter().any(|i| i.mnemonic() == M::Inc && i.has_lock_prefix() && i.memory_base() == R::RAX));
        assert_eq!(v.last().unwrap().near_branch_target(), base + PINS_MISSING);
        // Otherwise the original compare, then the branch on its flags.
        let found = v[1].near_branch_target();
        let mut d = Decoder::with_ip(64, &tramp[(found - tbase) as usize..], found, DecoderOptions::NONE);
        let cmp = d.decode();
        assert_eq!(&tramp[(found - tbase) as usize..][..cmp.len()], &PINS_BYTES);
        let j = d.decode();
        assert_eq!((j.flow_control(), j.near_branch_target()), (FlowControl::UnconditionalBranch, base + PINS_NEXT));
        // Other bytes: not patched.
        let mut image2 = vec![0x90u8; 0xe2_0000];
        image2[PINS_SITE..PINS_SITE + 7].copy_from_slice(&[0x83, 0xB8, 0x70, 0x04, 0x00, 0x00, 0x00]);
        let mut tramp2 = vec![0u8; 0x1_0000];
        let mut p = Patcher { image: &mut image2, base, tls_slot_disp: 0, tramp_base: tbase, tramp: &mut tramp2, tramp_used: 0, data_used: 0 };
        assert!(patch_foot_pins(&mut p).is_err());
        assert_eq!(p.image[PINS_SITE], 0x83);
        PINS_COUNT.store(0, Ordering::Relaxed);
    }
}
