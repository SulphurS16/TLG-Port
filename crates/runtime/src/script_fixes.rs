//! The game's scripts above its own 30 fps: some Lua (Havok Script) scripts
//! count script updates - one per frame - where they mean time (a counter
//! that adds 1 per update and compares with a frame count, an angle that
//! turns a fixed amount per update, a 30-sample position history meant as
//! one second...). At the 60 fps limit, or without a limit, those would run
//! up to twice as fast. `title_patches` routes every script load
//! (`hksL_loadbuffer`) through `load`, which hands the compiler a copy of a
//! known script (by the hash of its text) with those spots weighted by the
//! frame's time (`FIXES`); a
//! script whose text is not the expected one is left as it is (logged). At
//! the game's own 30 fps no script is changed (the loads are not even routed
//! then). The game's step and most of its scripts already use real time;
//! the list of spots came from a survey of all 620 scripts (LOG: The
//! scripts' frame-counted timers).
//!
//! `TLG_SCRIPT_LOG=1` logs every load (chunk name, size, hash, Lua state) at
//! any frame rate; `TLG_SCRIPT_PROBE=1` prints the weight as the AI and
//! mission scripts see it (with `TLG_LUA_LOG=1`).

use std::sync::atomic::{AtomicU64, Ordering};

/// Scripts loaded, and those given a fixed copy.
pub static LOADS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

pub fn log_loads() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TLG_SCRIPT_LOG").is_ok_and(|v| v == "1"))
}

/// One change to a script's original text: its bytes [at, at + old.len()),
/// which must be `old` (mostly empty: an insertion), become `new`. Only our
/// text and the few characters a change replaces are kept here, not the
/// game's code.
pub struct Edit {
    pub at: usize,
    pub old: &'static str,
    pub new: &'static str,
}

/// The edits of one script: its file's path (for the log), and the FNV-1a
/// hash of its text as the game loads it, which picks the fix (the game's
/// loads of these scripts carry no chunk name; another text - a changed
/// script - gets no edits).
pub struct ScriptFix {
    pub chunk: &'static str,
    pub hash: u64,
    pub edits: &'static [Edit],
}

/// The known scripts' frame-counted spots (see the module docs). Each
/// update weighs `TLG_W` 30 Hz frames: 1 at the game's 30 fps and below,
/// 0.5 at 60 (set every frame by the global `Tick` in system/scripts/utils.lua,
/// from the unscaled frame time). Counters add the weight instead of 1,
/// `== N` tests become crossing tests, sample histories fill at 30 Hz,
/// per-update steps, decays and displacement thresholds scale by it.
/// (Every offset and anchor checked against the scripts of 01.00 with update
/// 01.03: `fixes_match_the_scripts`.)
static FIXES: &[ScriptFix] = &[
    ScriptFix {
        chunk: "system/scripts/utils.lua",
        hash: 0x3fb910d48c91fd3c,
        edits: &[
            // TLG frame weight: TLG_W = 1 at load (one update = one 30 Hz frame)
            Edit { at: 5702, old: "", new: " TLG_W = 1 TLG_W_N = 0" },
            // TLG frame weight: each frame, before the user ticks, TLG_W = the frame's real time x 30 = dt / game speed x 30
            // (a slowed speed s as the step has it above 30 fps: floor(12 s + 0.1) / 12, title_patches' slow-motion rounding)
            // (the scripts' own real time, missionscript.lua UpdateTimer_Sub; >= 0.97 -> 1, as are 30 fps and slower frames;
            // edt runs ~10% above the frame time); logs once at tick 300
            // (TLG_W30 - the port's script sets it while the frame-rate limit is the game's 30 - keeps it at 1:
            // every update one 30 Hz frame, as the game has it.)
            Edit { at: 5788, old: "", new: r#" local tlg_s = (MS and MS.GetGameSpeed) and MS.GetGameSpeed() or 1 if tlg_s and tlg_s > 0 and tlg_s < 1 then local tlg_f = math.floor(12 * tlg_s + 0.1) if tlg_f > 0 then tlg_s = tlg_f / 12 end end if TLG_W30 then TLG_W = 1 elseif dt and dt > 0 and tlg_s and tlg_s > 0 then local tlg_t = dt / tlg_s * 30 if tlg_t >= 0.97 then tlg_t = 1 end TLG_W = tlg_t end if TLG_W_N and TLG_W_N < 300 then TLG_W_N = TLG_W_N + 1 if TLG_W_N == 300 and GaPrint then GaPrint("TLG_W " .. tostring(TLG_W) .. " dt " .. tostring(dt) .. " edt " .. tostring(edt)) end end"# },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/ai2/condor/condorcommonsurface.lua",
        hash: 0x62ca19b21e4bae26,
        edits: &[
            // Sneeze: 30 Hz gate for the 30-sample boy position history (1 s window, per-sample displacement as at 30 fps)
            Edit { at: 5398, old: "", new: "work.TLG_SnzAcc = (work.TLG_SnzAcc or 0) + (TLG_W or 1) local tlg_s = work.TLG_SnzAcc > 0.999 " },
            Edit { at: 5400, old: "", new: " tlg_s then work.TLG_SnzAcc = work.TLG_SnzAcc - 1 end if tlg_s and" },
            // Sneeze: sample the boy position only on gated updates
            Edit { at: 5545, old: "", new: "if tlg_s then " },
            Edit { at: 5590, old: "", new: " end" },
            // Sneeze: trim the contact history only on gated updates
            Edit { at: 5911, old: "", new: " tlg_s and" },
            // Sneeze: sample the head contact only on gated updates
            Edit { at: 6291, old: "", new: "if tlg_s then " },
            Edit { at: 6348, old: "", new: " end" },
            // Petting: RelaxToSleepTimer counts 30 Hz frames (lies down after 100)
            Edit { at: 21638, old: "", new: "(TLG_W or " },
            Edit { at: 21639, old: "", new: ")" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/ai2/condor/condorcommonactionbarrel.lua",
        hash: 0x39442000e487bb21,
        edits: &[
            // Forced sit-eat: 10-frame grace for the barrel to become the interest
            Edit { at: 36934, old: "", new: "(TLG_W or " },
            Edit { at: 36935, old: "", new: ")" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/ai2/condor/condorcommonactioneatable.lua",
        hash: 0xe07ca5b41726bb16,
        edits: &[
            // Face-down walk gaze: per-update smoothing 0.95/0.99 -> rate^W
            Edit { at: 39945, old: "", new: " ^ (TLG_W or 1" },
            Edit { at: 39946, old: "", new: ")" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/ai2/condor/stage/s08_largetower.lua",
        hash: 0x560e3b9935359624,
        edits: &[
            // Look-around for Arico: 40/110/180-frame gaze cycle (2nd match is a commented copy)
            Edit { at: 17112, old: "", new: "(TLG_W or " },
            Edit { at: 17113, old: "", new: ")" },
            Edit { at: 22355, old: "", new: "(TLG_W or " },
            Edit { at: 22356, old: "", new: ")" },
            // Mirror rotation re-command: count 30 Hz frames
            Edit { at: 29456, old: "", new: "(TLG_W or " },
            Edit { at: 29457, old: "", new: ")" },
            // Mirror rotation re-command: every 50 frames via an accumulator (same phase as % 50 at 30 fps)
            Edit { at: 29463, old: "(", new: "" },
            Edit { at: 29471, old: "%", new: "<" },
            Edit { at: 29475, old: ") ~= 0", new: "" },
            Edit { at: 29497, old: "", new: " dt_cnt = dt_cnt - 50" },
            // Stranded-boy failsafe: mirror timer random 1-3 per frame -> scaled by W
            Edit { at: 62148, old: "", new: " * (TLG_W or 1)" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/ai2/condor/stage/s09_fallingbridge2.lua",
        hash: 0x70dbdf5c3f3e11e2,
        edits: &[
            // Helmetless soldiers gaze cycle 300/600/900 frames
            Edit { at: 41777, old: "", new: "(TLG_W or " },
            Edit { at: 41778, old: "", new: ")" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/ai2/condor/stage/s10_pit.lua",
        hash: 0xefadeb6af2780f1a,
        edits: &[
            // Smell-pot cart: 30 Hz gate for the 6-sample pot position history (lead x30 and 0.02 threshold as at 30 fps)
            Edit { at: 6966, old: "", new: "work.TLG_PotAcc = (work.TLG_PotAcc or 0) + (TLG_W or 1) local tlg_s = work.TLG_PotAcc > 0.999 " },
            Edit { at: 6968, old: "", new: " tlg_s then work.TLG_PotAcc = work.TLG_PotAcc - 1 end if tlg_s and" },
            // Smell-pot cart: sample the pot position only on gated updates
            Edit { at: 7124, old: "", new: "if tlg_s then " },
            Edit { at: 7167, old: "", new: " end" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/ai2/condor/stage/s12_thenest.lua",
        hash: 0x7351106e193a1daa,
        edits: &[
            // Cage struggle: 30 Hz gate for the 300-sample stick wiggle queue (10 s window)
            Edit { at: 11022, old: "", new: "pwork.TLG_GachaAcc = (pwork.TLG_GachaAcc or 0) + (TLG_W or 1) local tlg_s = pwork.TLG_GachaAcc > 0.999 " },
            Edit { at: 11024, old: "", new: " tlg_s then pwork.TLG_GachaAcc = pwork.TLG_GachaAcc - 1 end if tlg_s and" },
            // Cage struggle: sample the stick only on gated updates
            Edit { at: 11152, old: "", new: "if tlg_s then " },
            Edit { at: 11207, old: "", new: " end" },
            // Sniff targets orbit 1 degree per 30 Hz frame (two copies)
            Edit { at: 58316, old: "", new: "(TLG_W or " },
            Edit { at: 58317, old: "", new: ")" },
            Edit { at: 104477, old: "", new: "(TLG_W or " },
            Edit { at: 104478, old: "", new: ")" },
            // Rolling cage: 30 Hz gate for the 60-sample cage velocity history (2 s window)
            Edit { at: 106562, old: "", new: "work.TLG_CageAcc = (work.TLG_CageAcc or 0) + (TLG_W or 1) local tlg_s = work.TLG_CageAcc > 0.999 " },
            Edit { at: 106564, old: "", new: " tlg_s then work.TLG_CageAcc = work.TLG_CageAcc - 1 end if tlg_s and" },
            // Rolling cage: sample the cage velocity only on gated updates
            Edit { at: 106722, old: "", new: "if tlg_s then " },
            Edit { at: 106767, old: "", new: " end" },
            // Cheek priority time: per-update decay /1.5 -> /1.5^W
            Edit { at: 125156, old: "", new: "(" },
            Edit { at: 125159, old: "", new: " ^ (TLG_W or 1))" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/ai2/condor/stage/s15_secondflight.lua",
        hash: 0x84affb6adac283f9,
        edits: &[
            // Long jump look ahead/behind cycle 120/190 frames
            Edit { at: 17987, old: "", new: "(TLG_W or " },
            Edit { at: 17988, old: "", new: ")" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/ai2/condor/stage/s16_soldiersden.lua",
        hash: 0x76bc1704a61fd34b,
        edits: &[
            // Shutter fret gaze circle: angle step per 30 Hz frame
            Edit { at: 65220, old: "", new: "*(TLG_W or 1)" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/ai2/condor/stage/s17_sentineltower.lua",
        hash: 0x60eaa63df13ca821,
        edits: &[
            // Rodeo release: boy displacement threshold per update scaled (same speed, 4.5 m/s)
            Edit { at: 25591, old: "", new: "* (TLG_W or 1) " },
            // Rodeo release: countdown 0.033 per 30 Hz frame
            Edit { at: 25693, old: "", new: "*(TLG_W or 1" },
            Edit { at: 25694, old: "", new: ")" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/ai2/interestacts/intactcondor.lua",
        hash: 0xabd928e8eb1f2014,
        edits: &[
            // FB_Chain_see: continuation lasts 4 frames on goal
            Edit { at: 356989, old: "", new: "(TLG_W or " },
            Edit { at: 356990, old: "", new: ")" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/ai2/actions/condor/actionmove.lua",
        hash: 0xa43befed005b14ef,
        edits: &[
            // Action freemove: look-ahead = 6 frames of per-update movement -> divide by W
            Edit { at: 2585, old: "", new: "/ (TLG_W or 1" },
            Edit { at: 2586, old: "", new: " )" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/ai2/actions/condor/freemove.lua",
        hash: 0x52f0662a18095c3b,
        edits: &[
            // Freemove: per-update displacement threshold scaled (same speed)
            Edit { at: 14321, old: "", new: "* (TLG_W or 1) " },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/ai2/approach/aprcommon.lua",
        hash: 0x4019bd676cedf6ff,
        edits: &[
            // Approach: interest position history capped at 32 frames-equivalent (32/W updates) so the 1 s chase window (ceil(1/dt) samples) fits
            Edit { at: 1431, old: "if", new: "while" },
            Edit { at: 1481, old: "then", new: "/ (TLG_W or 1) do" },
            // Approach: interest moved threshold per update scaled (same speed)
            Edit { at: 1822, old: "", new: "* (TLG_W or 1) " },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/common/tatoostate.lua",
        hash: 0x1d163a67ca78dd69,
        edits: &[
            // Curse tattoo glow fade-in step per 30 Hz frame
            Edit { at: 1181, old: "", new: " * (TLG_W or 1)" },
            // Curse tattoo glow fade-out step per 30 Hz frame
            Edit { at: 1507, old: "", new: " * (TLG_W or 1)" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/stages/s19_flashback/stage_common.lua",
        hash: 0xdb2e373c76d1f7f8,
        edits: &[
            // Flashback horn glow pulse step per 30 Hz frame
            Edit { at: 82995, old: "", new: " * (TLG_W or 1)" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/common/soundsystem.lua",
        hash: 0xe28a7f68605a4b12,
        edits: &[
            // Pulse sound stop after Circle release: count 30 Hz frames
            Edit { at: 35775, old: "", new: "(TLG_W or " },
            Edit { at: 35777, old: "", new: ")" },
            // Pulse sound stop: == 20 -> crossing test (fires once)
            Edit { at: 35787, old: "", new: " val < 20 and" },
            Edit { at: 35825, old: "=", new: ">" },
            // Pulse sound stop: first count is one update
            Edit { at: 36182, old: "", new: "(TLG_W or " },
            Edit { at: 36184, old: "", new: ")" },
            // Swim ripple/splash particle interval counts 30 Hz frames
            Edit { at: 40583, old: "", new: "(TLG_W or " },
            Edit { at: 40585, old: "", new: ")" },
            // Tail sound emitter volume re-check every 30 frames
            Edit { at: 45286, old: "", new: "(TLG_W or " },
            Edit { at: 45287, old: "", new: ")" },
            // Sound ticks skip tiny steps: threshold scaled so they still run above 100 fps (same slow-motion cut-off as at 30 fps)
            Edit { at: 38592, old: "", new: "* (TLG_W or 1) " },
            Edit { at: 40712, old: "", new: "* (TLG_W or 1) " },
            Edit { at: 41589, old: "", new: "* (TLG_W or 1) " },
            Edit { at: 42371, old: "", new: "* (TLG_W or 1) " },
            Edit { at: 44375, old: "", new: "* (TLG_W or 1) " },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/common/sound/soundsystem_state.lua",
        hash: 0x3ffdf839759eecc1,
        edits: &[
            // State-group loop sound end: 3-frame debounce counts 30 Hz frames
            Edit { at: 6861, old: "", new: "(TLG_W or " },
            Edit { at: 6862, old: "", new: ")" },
            // State-group loop sound end: count == 0 -> the update the count reaches 0 (fires once)
            Edit { at: 7008, old: "=", new: "<" },
            Edit { at: 7013, old: "", new: "and SoundSystem_State.state.sound.group_emitter_list[k][ii].count > -(TLG_W or 1) " },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/stages/common/extension/statemachine/sm_callbacks_sound.lua",
        hash: 0x666e5244a7e8f8fa,
        edits: &[
            // Elevator rise sound: still-threshold per update scaled (same speed)
            Edit { at: 12359, old: "", new: "* (TLG_W or 1) " },
            // Elevator rise sound: stops after 7 still frames
            Edit { at: 12444, old: "", new: "(TLG_W or " },
            Edit { at: 12445, old: "", new: ")" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/stages/common/extension_bgm/bgm_s02_tc.lua",
        hash: 0x4d64c3f36d3725c2,
        edits: &[
            // Chain stretch sound: 20-frame cooldown; clamped so the == 0 test is reached
            Edit { at: 4778, old: "", new: "(TLG_W or " },
            Edit { at: 4779, old: "", new: ") if val < 0 then val = 0 end" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/stages/common/extension_bgm/bgm_s17_st.lua",
        hash: 0x83040fe5acb2d2c4,
        edits: &[
            // Dark dome sounds: 5-frame retry cooldowns (already clamped at 0)
            Edit { at: 14618, old: "", new: "(TLG_W or " },
            Edit { at: 14619, old: "", new: ")" },
            Edit { at: 15476, old: "", new: "(TLG_W or " },
            Edit { at: 15477, old: "", new: ")" },
            Edit { at: 16289, old: "", new: "(TLG_W or " },
            Edit { at: 16290, old: "", new: ")" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/stages/common/extension/msext_darkdome.lua",
        hash: 0x7c57628ff3b2f0cf,
        edits: &[
            // Dark dome regrowth per 30 Hz frame (3 of 4 matches are commented-out copies)
            Edit { at: 4721, old: "", new: "*(TLG_W or 1)" },
            Edit { at: 10656, old: "", new: "*(TLG_W or 1)" },
            Edit { at: 11285, old: "", new: "*(TLG_W or 1)" },
            Edit { at: 11428, old: "", new: "*(TLG_W or 1)" },
            // Dark dome shrink under the pulse per 30 Hz frame (3 of 4 matches are commented-out copies)
            Edit { at: 4852, old: "", new: "*(TLG_W or 1)" },
            Edit { at: 10827, old: "", new: "*(TLG_W or 1)" },
            Edit { at: 11075, old: "", new: "*(TLG_W or 1)" },
            Edit { at: 11579, old: "", new: "*(TLG_W or 1)" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/stages/common/extension/statemachine/sm_callbacks_unique.lua",
        hash: 0x575f58cbfedd2062,
        edits: &[
            // Yajirobe mirror twist limits move changeAnglePerFrame per 30 Hz frame
            Edit { at: 179774, old: "", new: " * (TLG_W or 1)" },
            // Rail balancer controllers (s07, s08): frame counter counts 30 Hz frames
            Edit { at: 181421, old: "", new: "(TLG_W or " },
            Edit { at: 181422, old: "", new: ")" },
            Edit { at: 188273, old: "", new: "(TLG_W or " },
            Edit { at: 188274, old: "", new: ")" },
            // Rail balancer (s07): fire every N frames via an accumulator (same phase as % N == 0 at 30 fps)
            Edit { at: 181481, old: "", new: " local tlg_fire = false if (tonumber(frameIntervalForVelocity) or 0) > 0 and state.frameCounter >= frameIntervalForVelocity then state.frameCounter = state.frameCounter - frameIntervalForVelocity tlg_fire = true end" },
            // Rail balancer (s08): fire every N frames via an accumulator
            Edit { at: 188333, old: "", new: " local tlg_fire = false if (tonumber(frameIntervalForVelocity) or 0) > 0 and state.frameCounter >= frameIntervalForVelocity then state.frameCounter = state.frameCounter - frameIntervalForVelocity tlg_fire = true end" },
            // Rail balancers: the velocity kicks use the accumulator
            Edit { at: 182831, old: "state.frameCounter % frameIntervalForVelocity  == 0", new: "tlg_fire" },
            Edit { at: 183231, old: "state.frameCounter % frameIntervalForVelocity  == 0", new: "tlg_fire" },
            Edit { at: 184873, old: "state.frameCounter % frameIntervalForVelocity  == 0", new: "tlg_fire" },
            Edit { at: 185273, old: "state.frameCounter % frameIntervalForVelocity  == 0", new: "tlg_fire" },
            Edit { at: 186919, old: "state.frameCounter % frameIntervalForVelocity  == 0", new: "tlg_fire" },
            Edit { at: 187319, old: "state.frameCounter % frameIntervalForVelocity  == 0", new: "tlg_fire" },
            Edit { at: 189660, old: "state.frameCounter % frameIntervalForVelocity  == 0", new: "tlg_fire" },
            Edit { at: 190063, old: "state.frameCounter % frameIntervalForVelocity  == 0", new: "tlg_fire" },
            Edit { at: 191684, old: "state.frameCounter % frameIntervalForVelocity  == 0", new: "tlg_fire" },
            Edit { at: 192087, old: "state.frameCounter % frameIntervalForVelocity  == 0", new: "tlg_fire" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/stages/s06_fallingbridge/missionscript.lua",
        hash: 0xe8085537552a5989,
        edits: &[
            // Wheel interest after 3 s of motion: 1/30 per 30 Hz frame
            Edit { at: 33895, old: "", new: " * (TLG_W or 1)" },
            // Bridge plank shake: 15-frame window
            Edit { at: 141500, old: "", new: "(TLG_W or " },
            Edit { at: 141501, old: "", new: ")" },
            // Bridge plank shake: 6-frame delay
            Edit { at: 142852, old: "", new: "(TLG_W or " },
            Edit { at: 142853, old: "", new: ")" },
            // Helmet generators: 1 s check cadence
            Edit { at: 172013, old: "", new: " * (TLG_W or 1)" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/stages/s09_fallingbridge2/missionscript.lua",
        hash: 0x60980f9554fe454a,
        edits: &[
            // Tail reach: escape from the branch after 0.5 s
            Edit { at: 24833, old: "", new: " * (TLG_W or 1)" },
            // Hanging boy camera volume re-enabled after 10 s
            Edit { at: 27134, old: "", new: " * (TLG_W or 1)" },
            // Branch jump enabled after 5 s riding Trico
            Edit { at: 30670, old: "", new: " * (TLG_W or 1)" },
            // Guidance to the pit after 1 s riding Trico
            Edit { at: 61122, old: "", new: " * (TLG_W or 1)" },
            // The shields hint's 180 s timer adds 1/30 s per update (its hint is one of the Extras page's cut lines)
            Edit { at: 106556, old: "", new: " * (TLG_W or 1)" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/stages/s11_columncorridor/missionscript.lua",
        hash: 0x6e734c7628175a51,
        edits: &[
            // Passage hint interest after 3000 frames, reset at 4500
            Edit { at: 41542, old: "", new: "(TLG_W or " },
            Edit { at: 41543, old: "", new: ")" },
            // Jumping cart helmet impulse window 90 frames
            Edit { at: 116489, old: "", new: "(TLG_W or " },
            Edit { at: 116490, old: "", new: ")" },
            // See-saw trophy: 15 frames riding
            Edit { at: 127519, old: "", new: "(TLG_W or " },
            Edit { at: 127520, old: "", new: ")" },
            // See-saw trophy: 90-frame grace after being thrown
            Edit { at: 128055, old: "", new: "(TLG_W or " },
            Edit { at: 128056, old: "", new: ")" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/stages/s13_firstflight/missionscript.lua",
        hash: 0x39724e7346a90ad8,
        edits: &[
            // Tunnel L instruction hint 2500/3600 frames
            Edit { at: 90673, old: "", new: "(TLG_W or " },
            Edit { at: 90674, old: "", new: ")" },
            // Tunnel R instruction hint 2500/3600 frames
            Edit { at: 100851, old: "", new: "(TLG_W or " },
            Edit { at: 100852, old: "", new: ")" },
            // Stained-glass load wait 60 frames
            Edit { at: 198079, old: "", new: "(TLG_W or " },
            Edit { at: 198080, old: "", new: ")" },
            // Auto-jump interest after 5400 frames
            Edit { at: 260709, old: "", new: "(TLG_W or " },
            Edit { at: 260710, old: "", new: ")" },
            // Tunnel subduction animation back to default after 10 frames
            Edit { at: 285838, old: "", new: "(TLG_W or " },
            Edit { at: 285839, old: "", new: ")" },
            // Bridge shake propagation delays 6/12/16/18 frames
            Edit { at: 311784, old: "", new: "(TLG_W or " },
            Edit { at: 311785, old: "", new: ")" },
            // Bridge shake re-trigger delay 15 frames
            Edit { at: 312000, old: "", new: "(TLG_W or " },
            Edit { at: 312001, old: "", new: ")" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/stages/s16_soldiersden/missionscript.lua",
        hash: 0x0fd0b9ad3904689c,
        edits: &[
            // Big elevator switch: 30 frames of charge/irradiation (two copies)
            Edit { at: 127892, old: "", new: "(TLG_W or " },
            Edit { at: 127893, old: "", new: ")" },
            Edit { at: 128496, old: "", new: "(TLG_W or " },
            Edit { at: 128497, old: "", new: ")" },
            // Big elevator switch: == 1 (first update) -> first-update test (fires once)
            Edit { at: 129948, old: "=", new: "> 0 and state.sub_sd_ev01_pulseTimer <" },
            Edit { at: 129951, old: "", new: "(TLG_W or " },
            Edit { at: 129952, old: "", new: ")" },
            // Small elevator anti-softlock start after 60 frames
            Edit { at: 136729, old: "", new: "(TLG_W or " },
            Edit { at: 136730, old: "", new: ")" },
            // Small elevator pulse start after 30 frames
            Edit { at: 138051, old: "", new: "(TLG_W or " },
            Edit { at: 138052, old: "", new: ")" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/stages/s16_soldiersden/stage_common.lua",
        hash: 0xc734b6e954b421bf,
        edits: &[
            // Trigger switches: per-switch previous count, for crossing tests
            Edit { at: 473, old: "", new: "local tlg_swprev = {} " },
            // Trigger switches: == 30 -> first update at or past 30 (fires once; 2nd match is the unused old switch)
            Edit { at: 5066, old: "", new: "local tlg_p = tlg_swprev[SV_pulse_timer] or -1 tlg_swprev[SV_pulse_timer] = pulsetime " },
            Edit { at: 5079, old: "=", new: ">" },
            Edit { at: 5092, old: "", new: "and tlg_p < need_time " },
            Edit { at: 7033, old: "", new: "local tlg_p = tlg_swprev[SV_pulse_timer] or -1 tlg_swprev[SV_pulse_timer] = pulsetime " },
            Edit { at: 7046, old: "=", new: ">" },
            Edit { at: 7059, old: "", new: "and tlg_p < need_time " },
            // Trigger switches: == 20 emblem sound -> crossing test (fires once)
            Edit { at: 7372, old: "=", new: ">" },
            Edit { at: 7378, old: "", new: "and tlg_p < 20 " },
            // Trigger switches: 30 frames of charge/irradiation
            Edit { at: 5692, old: "", new: "(TLG_W or " },
            Edit { at: 5693, old: "", new: ")" },
            Edit { at: 7654, old: "", new: "(TLG_W or " },
            Edit { at: 7655, old: "", new: ")" },
            // Gate trigger switch: end_time (90) frames; the gate-open reader (>= 30 in missionscript) keeps frame units
            Edit { at: 9399, old: "", new: "(TLG_W or " },
            Edit { at: 9400, old: "", new: ")" },
            Edit { at: 9852, old: "", new: "(TLG_W or " },
            Edit { at: 9853, old: "", new: ")" },
            // Gate trigger switch: == 1 (first update) -> first-update test
            Edit { at: 11297, old: "=", new: "> 0 and pulseTimer <" },
            Edit { at: 11300, old: "", new: "(TLG_W or " },
            Edit { at: 11301, old: "", new: ")" },
        ],
    },
];

/// The PC Settings' Extras page: content the game's scripts still hold but
/// switch off, back while a global the port's script sets is true
/// (`TLG_X_DROOL`, `TLG_X_HINTS`, `TLG_X_E3`, `TLG_X_PLAY`: pc_settings'
/// `extras`). Every edit is in the scripts at any setting (so a change
/// applies at once): with its global unset a script does what the game's
/// own does. The cut lines are the hint table's "cut at genDESIGN's
/// request" (2016-09-20, common/hintlist.lua) whose triggers still run
/// (their voice and subtitles ship); left out are those whose trigger was
/// removed with them. They stay out of the hint table (the hints trophy
/// counts its entries).
static EXTRAS: &[ScriptFix] = &[
    ScriptFix {
        chunk: "game/scripts/ai2/condor/condorcommonslaver.lua",
        hash: 0x2473fa1d37827bc4,
        edits: &[
            // Drool: the step that shows the drool mesh, skipped since 2016 ("remove the drool for now"), runs while TLG_X_DROOL is set (the mesh hidden again when it is cleared)
            Edit { at: 3182, old: "do return end", new: r#"if not (TLG_X_DROOL and param.sknMeshSlaver) then if param.TLG_Drool then param.TLG_Drool = nil if param.sknMeshSlaver then param.sknMeshSlaver.renderableFlags = "|Hidden" end end return end param.TLG_Drool = true"# },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/ai2/interestactprobability/condor/intactprobabilitycondor_player.lua",
        hash: 0x9be3c971e6e3c4f9,
        edits: &[
            // Trico plays (sniffs, then paws the boy): sealed ("for now"), valid while TLG_X_PLAY is set and the boy is clear of ledges (the cheek rubs' check)
            Edit { at: 1225, old: "false", new: "TLG_X_PLAY == true and ActHierarchy_CheekActCheck(props)" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/stages/s06_fallingbridge/missionscript.lua",
        hash: 0xe8085537552a5989,
        edits: &[
            // E3 slow motion: the jump to Trico at the falling bridge slows to 0.15 in 0.25 s (the commented "original E3" call) while TLG_X_E3 is set
            Edit { at: 66391, old: "", new: "if TLG_X_E3 then MS.SetGameSpeed(0.15, 0.25) else " },
            Edit { at: 66426, old: "", new: " end" },
            // Cut hint S06_FB_HNT_V_0040 (wheel door)
            Edit { at: 225038, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 225087, old: "", new: " end" },
            // Cut hint S06_FB_HNT_V_0080 (stained glass)
            Edit { at: 228349, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 228398, old: "", new: " end" },
            // Cut hint S06_FB_HNT_V_0092 (the jump)
            Edit { at: 230809, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 230858, old: "", new: " end" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/stages/s03_canyon/missionscript.lua",
        hash: 0x70e0ece551af112c,
        edits: &[
            // Cut hint S03_CA_HNT_V_0021 (the lake)
            Edit { at: 32832, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 32879, old: "", new: " end" },
            // Cut hint S03_CA_HNT_V_0110 (the chain room)
            Edit { at: 66712, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 66760, old: "", new: " end" },
            // Cut hint S03_CA_HNT_V_0120 (over the wall)
            Edit { at: 80154, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 80201, old: "", new: " end" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/stages/s07_alone/missionscript.lua",
        hash: 0x841b7b6675bfdb24,
        edits: &[
            // Cut hint S07_AL_HNT_V_0050 (stained glass, once on entering)
            Edit { at: 28800, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 28847, old: "", new: " end" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/stages/s08_largetower/missionscript.lua",
        hash: 0x46402f2c71566e54,
        edits: &[
            // Cut hint S08_LT_HNT_V_0031 (along the tail)
            Edit { at: 83301, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 83330, old: "", new: " end" },
            Edit { at: 83461, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 83509, old: "", new: " end" },
            // Cut hint S08_LT_HNT_V_0040 (a wall to climb)
            Edit { at: 84668, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 84698, old: "", new: " end" },
            Edit { at: 84777, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 84828, old: "", new: " end" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/stages/s09_fallingbridge2/missionscript.lua",
        hash: 0x60980f9554fe454a,
        edits: &[
            // Cut hint S09_FB2_HNT_V_0060 (the soldiers' shields)
            Edit { at: 106799, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 106848, old: "", new: " end" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/stages/s10_pit/missionscript.lua",
        hash: 0x249d50b65643093d,
        edits: &[
            // Cut hint S10_PI_HNT_V_0030 (the collapsed corridor)
            Edit { at: 107397, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 107426, old: "", new: " end" },
            Edit { at: 107539, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 107590, old: "", new: " end" },
            // Cut line S10_PI_MRY_V_0040 (no way on without Trico)
            Edit { at: 108784, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 108813, old: "", new: " end" },
            Edit { at: 108926, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 108977, old: "", new: " end" },
            // Cut line S10_PI_MRY_V_0060 (the smell pots' scent)
            Edit { at: 111441, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 111470, old: "", new: " end" },
            Edit { at: 111583, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 111634, old: "", new: " end" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/stages/s11_columncorridor/missionscript.lua",
        hash: 0x6e734c7628175a51,
        edits: &[
            // Cut hint S11_CC_HNT_V_0020 (the cart)
            Edit { at: 170186, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 170230, old: "", new: " end" },
            // Cut hint S11_CC_HNT_V_0051 (diving)
            Edit { at: 172422, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 172466, old: "", new: " end" },
            // Cut hint S11_CC_HNT_V_0060 (the wave)
            Edit { at: 174631, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 174675, old: "", new: " end" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/stages/s13_firstflight/missionscript.lua",
        hash: 0x39724e7346a90ad8,
        edits: &[
            // Cut hint S13_FF_HNT_V_0001 (along the cliff)
            Edit { at: 314521, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 314565, old: "", new: " end" },
            // Cut hint S13_FF_HNT_V_0033 (back to Trico)
            Edit { at: 321034, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 321078, old: "", new: " end" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/stages/s14_condorclash/missionscript.lua",
        hash: 0xfe23675803a3adef,
        edits: &[
            // Cut hint S14_CH_HNT_V_0140 (drop the cart)
            Edit { at: 175680, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 175727, old: "", new: " end" },
            // Cut hint S14_CH_HNT_V_0120 (upwards again)
            Edit { at: 186732, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 186779, old: "", new: " end" },
            // Cut hint S14_CH_HNT_V_0130 (the shield and the soldier)
            Edit { at: 187554, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 187601, old: "", new: " end" },
        ],
    },
    ScriptFix {
        chunk: "game/scripts/stages/s15_secondflight/missionscript.lua",
        hash: 0x663d8a3745ee2d44,
        edits: &[
            // Cut hint S15_SF_HNT_V_0030 (over the roofs)
            Edit { at: 8449, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 8492, old: "", new: " end" },
            // Cut hint S15_SF_HNT_V_0040 (Trico's beam)
            Edit { at: 11140, old: "--", new: "if TLG_X_HINTS then " },
            Edit { at: 11183, old: "", new: " end" },
        ],
    },
];

/// TLG_SCRIPT_PROBE=1 (a test): the weight as the AI and mission scripts see
/// it, printed once each (logged with TLG_LUA_LOG=1).
static PROBES: &[ScriptFix] = &[
    ScriptFix {
        chunk: "system/scripts/utils.lua",
        hash: 0x3fb910d48c91fd3c,
        edits: &[Edit { at: 5788, old: "", new: r#" TLG_PROBE_T = (TLG_PROBE_T or 0) + 1 if TLG_PROBE_T % 600 == 0 and GaPrint then GaPrint("TLG probe tick " .. tostring(TLG_PROBE_T) .. " W " .. tostring(TLG_W) .. " dt " .. tostring(dt) .. " edt " .. tostring(edt)) end"# }],
    },
    ScriptFix {
        chunk: "game/scripts/ai2/ai2.lua",
        hash: 0x526fe4848040a017,
        edits: &[Edit { at: 11380, old: "", new: r#" TLG_PROBE_A = (TLG_PROBE_A or 0) + 1 if TLG_PROBE_A == 900 and GaPrint then GaPrint("TLG_W in ai2 " .. tostring(TLG_W) .. " dt " .. tostring(_p.dt)) end"# }],
    },
    ScriptFix {
        chunk: "game/scripts/common/missionscript.lua",
        hash: 0x93db314309dfc9f5,
        edits: &[Edit { at: 62695, old: "", new: r#" TLG_PROBE_M = (TLG_PROBE_M or 0) + 1 if TLG_PROBE_M == 900 and GaPrint then GaPrint("TLG_W in missions " .. tostring(TLG_W) .. " dt " .. tostring(dt) .. " edt " .. tostring(edt)) end"# }],
    },
];

fn fnv(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100_0000_01b3))
}

/// The text the compiler gets for the script `chunk` (`text` as the game
/// read it, loaded into the Lua state `l`): None to keep the game's.
pub fn load(l: u64, chunk: &str, text: &[u8]) -> Option<Vec<u8>> {
    LOADS[0].fetch_add(1, Ordering::Relaxed);
    let hash = fnv(text);
    if log_loads() {
        crate::info!("script load: {chunk} ({} bytes, fnv {hash:016x}, state {l:#x})", text.len());
    }
    // (The menus' part table: where their items are, for the menu pointer.)
    if hash == ALWAYS[0].hash {
        crate::menu_pointer::note_part_table(text);
    }
    // (The port's own additions, at any frame rate.)
    if let Some(f) = ALWAYS.iter().find(|f| f.hash == hash) {
        return match apply(&[f], text) {
            Ok(out) => {
                crate::info!("scripts: {}: the port's {} additions ({} -> {} bytes)", f.chunk, f.edits.len(), text.len(), out.len());
                Some(out)
            }
            Err(e) => {
                crate::warn!("scripts: {}: {e}; left as the game has it", f.chunk);
                None
            }
        };
    }
    let fixed = frame_fixes(hash, text);
    // The global tick's script also carries the port's own script
    // (`crate::pc_menu`), at any frame rate.
    if hash != UTILS_HASH {
        return fixed;
    }
    let mut out = fixed.unwrap_or_else(|| text.to_vec());
    out.extend_from_slice(b"\n");
    out.extend_from_slice(crate::pc_menu::lua_module().as_bytes());
    crate::info!("scripts: system/scripts/utils.lua: the port's script appended ({} -> {} bytes)", text.len(), out.len());
    Some(out)
}

/// system/scripts/utils.lua (the global `Tick`), as 01.03 has it.
const UTILS_HASH: u64 = 0x3fb910d48c91fd3c;

/// The port's additions to the game's scripts (at any frame rate; none of
/// these scripts has frame-rate fixes): the Options lists' "PC Settings"
/// items as parts of the menus' part table (`menu_patches`; the object the
/// game's unused "Exit Game" label, `pc_files`), with places of their own:
/// below Copyright on the title screen, above Back in the pause menu (in
/// Treasure's row while the list has no Treasure).
static ALWAYS: &[ScriptFix] = &[ScriptFix {
    chunk: "game/scripts/common/gamemenusetting.lua",
    hash: 0xe8671be506b746c3,
    // (On the line of the title screen's Back item, line numbers unchanged.)
    edits: &[Edit {
        at: 5888,
        old: "",
        new: " {\"option\", \"UseSc2Layout\", \"OptionMenu_ITEM_PC2\", \"game/objects/ui/gamemenu/model/SYS_EXITGAME\", \"game/objects/ui/gamemenu/model/SYS_OPTION2_ITEM_PCSETTING_model_pos\" }, {\"option\", \"UseSc2Layout\", \"OptionMenu_ITEM_PC\", \"game/objects/ui/gamemenu/model/SYS_EXITGAME\", \"game/objects/ui/gamemenu/model/SYS_OPTION_ITEM_PCSETTING_model_pos\" }, {\"option\", \"UseSc2Layout\", \"OptionMenu_ITEM_PC3\", \"game/objects/ui/gamemenu/model/SYS_EXITGAME\", \"game/objects/ui/gamemenu/model/SYS_OPTION_ITEM_PCSETTNG3_model_pos\" },",
    }],
}];

/// The frame-counted timers' fixes and the Extras page's edits of the
/// script with this hash. They are in every script at any frame-rate limit
/// and setting (both can change while the game runs): while the limit is
/// the game's 30 fps, TLG_W stays 1, and with the Extras off their globals
/// are unset, so the scripts do what the game's own code does.
fn frame_fixes(hash: u64, text: &[u8]) -> Option<Vec<u8>> {
    // (Debug: TLG_SCRIPT_FIXES=0 keeps the game's scripts above 30 fps too.)
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let off = *OFF.get_or_init(|| std::env::var("TLG_SCRIPT_FIXES").is_ok_and(|v| v == "0"));
    static PROBE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let probe = *PROBE.get_or_init(|| std::env::var("TLG_SCRIPT_PROBE").is_ok_and(|v| v == "1"));
    let frame: Vec<&ScriptFix> = if off { vec![] } else { FIXES.iter().chain(if probe { PROBES } else { &[] }).filter(|f| f.hash == hash).collect() };
    let extras: Vec<&ScriptFix> = EXTRAS.iter().filter(|f| f.hash == hash).collect();
    let fixes: Vec<&ScriptFix> = frame.iter().chain(&extras).copied().collect();
    if fixes.is_empty() {
        return None;
    }
    match apply(&fixes, text) {
        Ok(out) => {
            LOADS[1].fetch_add(1, Ordering::Relaxed);
            let count = |v: &[&ScriptFix]| v.iter().map(|f| f.edits.len()).sum::<usize>();
            let (n, x) = (count(&frame), count(&extras));
            let what = match (n, x) {
                (_, 0) => format!("{n} changes for frame-counted timers"),
                (0, _) => format!("{x} changes for the Extras page"),
                _ => format!("{n} changes for frame-counted timers, {x} for the Extras page"),
            };
            crate::info!("scripts: {}: {what} ({} -> {} bytes)", fixes[0].chunk, text.len(), out.len());
            Some(out)
        }
        Err(e) => {
            crate::warn!("scripts: {}: {e}; left as the game has it", fixes[0].chunk);
            None
        }
    }
}

/// The edits of `fixes` (one script's) on its original `text`, in one pass
/// by offset (equal offsets in table order), all or none.
fn apply(fixes: &[&ScriptFix], text: &[u8]) -> Result<Vec<u8>, String> {
    let mut edits: Vec<&Edit> = fixes.iter().flat_map(|f| f.edits.iter()).collect();
    edits.sort_by_key(|e| e.at);
    let mut out = Vec::with_capacity(text.len() + edits.iter().map(|e| e.new.len()).sum::<usize>());
    let mut last = 0;
    for e in edits {
        let end = e.at + e.old.len();
        if e.at < last || end > text.len() || &text[e.at..end] != e.old.as_bytes() {
            return Err(format!("the text at {} is not the expected {:?}", e.at, e.old));
        }
        out.extend_from_slice(&text[last..e.at]);
        out.extend_from_slice(e.new.as_bytes());
        last = end;
    }
    out.extend_from_slice(&text[last..]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edits_apply_in_one_pass() {
        let text = b"t = t + 1\nif t == 100 then end\n";
        static A: ScriptFix = ScriptFix { chunk: "a.lua", hash: 0, edits: &[Edit { at: 8, old: "", new: "(W or " }, Edit { at: 9, old: "", new: ")" }, Edit { at: 15, old: "==", new: ">=" }] };
        static B: ScriptFix = ScriptFix { chunk: "a.lua", hash: 0, edits: &[Edit { at: 0, old: "", new: "-- probe\n" }] };
        let v = apply(&[&A, &B], text).unwrap();
        assert_eq!(std::str::from_utf8(&v).unwrap(), "-- probe\nt = t + (W or 1)\nif t >= 100 then end\n");
        // Another text there, or overlapping edits: nothing changes.
        static BAD: ScriptFix = ScriptFix { chunk: "a.lua", hash: 0, edits: &[Edit { at: 15, old: "~=", new: ">=" }] };
        assert!(apply(&[&BAD], text).is_err());
        static OVER: ScriptFix = ScriptFix { chunk: "a.lua", hash: 0, edits: &[Edit { at: 4, old: "t + 1", new: "x" }, Edit { at: 6, old: "+", new: "-" }] };
        assert!(apply(&[&OVER], text).is_err());
    }

    /// Every edit applies to the game's scripts as extracted on this machine
    /// (`local/research/scripts-all`: 01.00 with the update's overrides);
    /// skipped where they are not.
    #[test]
    fn fixes_match_the_scripts() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../local/research/scripts-all");
        if !root.is_dir() {
            return;
        }
        let lines = |b: &[u8]| b.iter().filter(|&&c| c == b'\n').count();
        for fix in ALWAYS {
            let text = std::fs::read(root.join(fix.chunk)).unwrap_or_else(|e| panic!("{}: {e}", fix.chunk));
            assert_eq!(fnv(&text), fix.hash, "{}: the hash", fix.chunk);
            let out = apply(&[fix], &text).unwrap_or_else(|e| panic!("{}: {e}", fix.chunk));
            assert_eq!(lines(&text), lines(&out), "{}: line count changed", fix.chunk);
        }
        for fix in FIXES.iter().chain(PROBES).chain(EXTRAS) {
            let text = std::fs::read(root.join(fix.chunk)).unwrap_or_else(|e| panic!("{}: {e}", fix.chunk));
            assert_eq!(fnv(&text), fix.hash, "{}: the hash", fix.chunk);
            // Alone, and with everything else for the same text (a fix, its
            // probe and the Extras page's edits).
            let all: Vec<&ScriptFix> = FIXES.iter().chain(PROBES).chain(EXTRAS).filter(|f| f.hash == fix.hash).collect();
            for set in [vec![fix], all] {
                let out = apply(&set, &text).unwrap_or_else(|e| panic!("{}: {e}", fix.chunk));
                assert_eq!(lines(&text), lines(&out), "{}: line count changed", fix.chunk);
            }
        }
    }

    /// Writes local/research/script-check.lua: every changed script (the
    /// frame fixes and the Extras page's edits, as the game gets them) in a
    /// `loadstring`, for the game's own compiler: the control command
    /// `luafile local/research/script-check.lua` (it answers in the log,
    /// `lua> ... => true <name>=ok ...`). Run with `--ignored`.
    #[test]
    #[ignore]
    fn write_compile_check() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../local/research");
        let mut hashes: Vec<(u64, &str)> = FIXES.iter().chain(EXTRAS).map(|f| (f.hash, f.chunk)).collect();
        hashes.sort();
        hashes.dedup();
        let mut lua = String::from("local r = {}\n");
        for (hash, chunk) in hashes {
            let text = std::fs::read(root.join("scripts-all").join(chunk)).unwrap();
            let set: Vec<&ScriptFix> = FIXES.iter().chain(EXTRAS).filter(|f| f.hash == hash).collect();
            let out = String::from_utf8(apply(&set, &text).unwrap()).unwrap();
            let eq = (1..).map(|n| "=".repeat(n)).find(|q| !out.contains(&format!("]{q}]"))).unwrap();
            let name = chunk.rsplit('/').take(2).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("/");
            lua += &format!("do local f, e = loadstring([{eq}[\n{out}]{eq}], \"{name}\") r[#r + 1] = \"{name}=\" .. (f and \"ok\" or tostring(e)) end\n");
        }
        lua += "return table.concat(r, \" \")\n";
        std::fs::write(root.join("script-check.lua"), lua).unwrap();
    }
}
