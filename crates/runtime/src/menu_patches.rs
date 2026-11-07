//! The "PC Settings" item of the game's own Options menu (title screen and
//! pause menu), run by the game's menu code:
//!
//! - Its lists: the Options update (eboot 0x923110) picks the title
//!   screen's list (static array 0x5887460, 5 item symbols: Camera, Screen,
//!   Audio, Copyright, Back) or the pause menu's (0x5887480, 5: Camera,
//!   Screen, Audio, Guide, Back; 0x58874a0, 6, with Treasure) into
//!   [rbp-8C0h] / [rbp-8C4h]; at 0x92756c, after either, the list goes
//!   through `options_list`, which hands back a copy with the PC item before
//!   Back (its part keys "OptionMenu_ITEM_PC2" / "OptionMenu_ITEM_PC",
//!   entries of the game's part table added by `script_fixes`). The list
//!   helper (0x93e610) moves the cursor over the copy as over any list.
//! - Its choice: the list helper calls the list's choice callback with the
//!   chosen item's symbol (0x93e8e8..0x93e91d); at 0x93e8ef the PC items'
//!   symbols branch to `options_choice` instead (the callback, and its
//!   symbol reference, skipped), which hides the list the way the game's
//!   pages do (its title, items and cursor: "reqHide") and opens the PC
//!   settings (`pc_menu`).
//! - Part requests (0x93dc50: menu manager, part key symbol, request
//!   symbol): while the PC settings are open, the list's per-frame
//!   "reqShow" of its cursor is dropped (`menu_request`); after they close,
//!   the next request first shows the list again (its title "reqShow", its
//!   items "reqSelect", as the game's pages do on their way back).
//!   TLG_MENU_TRACE=1 also logs every request with its callers.
//!
//! Applied only when every site holds the expected bytes.

use crate::patch::Patcher;
use crate::sysv;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// `mov [rbp-1060h], r14`, right after the list and its count are set.
const LIST_SITE: usize = 0x92756c;
const LIST_BYTES: [u8; 7] = [0x4C, 0x89, 0xB5, 0xA0, 0xEF, 0xFF, 0xFF];
const LIST_VAR: i32 = -0x8c0;
const COUNT_VAR: i32 = -0x8c4;
/// The game's three lists (module-relative).
const TITLE_LIST: u64 = 0x5887460;
const PAUSE_LIST: u64 = 0x5887480;
const PAUSE_LIST_T: u64 = 0x58874a0;
/// `lea rbx, [symbols]` after the chosen item's symbol is in eax.
/// The list helper's entry (after its prologue): `mov r15, [stack guard]`,
/// with the items in rsi - which list is active (`list_helper`).
const HELPER_SITE: usize = 0x93e621;
const HELPER_BYTES: [u8; 7] = [0x4C, 0x8B, 0x3D, 0x70, 0x20, 0xDC, 0x00];
/// The title and pause menus' cursor placement (eboot 0x922d30: context,
/// the selected item's symbol, the cursor parts' symbols), at `mov rax,
/// [stack guard]` after its prologue.
const PLACE_SITE: usize = 0x922d41;
const PLACE_BYTES: [u8; 7] = [0x48, 0x8B, 0x05, 0x50, 0xD9, 0xDD, 0x00];
/// The game's option values: [[main+OPTIONS_PTR]+0x100]: Vertical Camera
/// at +0x40, Horizontal Camera at +0x44 (1.0 Standard, -1.0 Reversed).
const OPTIONS_PTR: u64 = 0x5832568;
const CHOICE_SITE: usize = 0x93e8ef;
const CHOICE_BYTES: [u8; 7] = [0x48, 0x8D, 0x1D, 0x3A, 0x79, 0xDD, 0x00];
/// Where the list helper goes on after the callback (its reference released).
const CHOICE_DONE: u64 = 0x93e94b;
/// The part request function, a load in it after the request moved to
/// r12d, and its epilogue (a dropped request returns there).
const SEND_REQUEST: u64 = 0x93dc50;
const REQ_SITE: usize = 0x93dc61;
const REQ_BYTES: [u8; 6] = [0x8B, 0x97, 0x50, 0x01, 0x00, 0x00]; // mov edx, [rdi+150h]
const REQ_EPILOGUE: u64 = 0x93ddbb;
const REQ_EPILOGUE_BYTES: [u8; 15] = [0x48, 0x83, 0xC4, 0x08, 0x5B, 0x41, 0x5C, 0x41, 0x5D, 0x41, 0x5E, 0x41, 0x5F, 0x5D, 0xC3];
/// The request names' symbols (set by the game's static initializer).
const REQ_SHOW: u64 = 0x588742c;
const REQ_HIDE: u64 = 0x588743c;
const REQ_SELECT: u64 = 0x5887440;
/// The game's symbol functions: a name's hash, and its symbol
/// (name, hash, length, flags).
const SYM_HASH: u64 = 0xe45080;
const SYM_HASH_BYTES: [u8; 8] = [0x55, 0x48, 0x89, 0xE5, 0x41, 0x56, 0x53, 0x49];
const SYM_INTERN: u64 = 0xe45240;
const SYM_INTERN_BYTES: [u8; 8] = [0x55, 0x48, 0x89, 0xE5, 0x41, 0x57, 0x41, 0x56];

/// The PC items' part keys (title screen, pause menu).
pub const KEY_TITLE: &str = "OptionMenu_ITEM_PC2";
pub const KEY_PAUSE: &str = "OptionMenu_ITEM_PC";
/// The pause menu's list without Treasure: the item in Treasure's row.
pub const KEY_PAUSE3: &str = "OptionMenu_ITEM_PC3";

static BASE: AtomicU64 = AtomicU64::new(0);
/// Data slots of the patch: the PC items' symbols (u32 title, u32 pause),
/// the request filter's switch (u8) and its answer (u8).
static SLOTS: AtomicU64 = AtomicU64::new(0);

struct State {
    /// Symbols, once interned: PC title, PC pause, title, title 2, cursor,
    /// PC pause without Treasure.
    syms: Option<[u32; 6]>,
    /// The menu manager the requests go to (the game has one).
    seen_mgr: u64,
    /// The previous list pick had the PC item (its part existed).
    extending: bool,
    /// The list copies handed out (kept alive: the game reads them).
    lists: Vec<Box<[u32]>>,
    /// The menu manager, the list shown, whether it is the title screen's.
    mgr: u64,
    items: Vec<u32>,
    title: bool,
    /// The PC item's row in the layouts (`item_label`).
    row: f32,
    /// The PC settings were opened from the list (it is hidden).
    from_list: bool,
    /// The PC settings closed: show the list again at the next request.
    show_pending: bool,
}

static STATE: Mutex<State> = Mutex::new(State { syms: None, seen_mgr: 0, extending: false, lists: Vec::new(), mgr: 0, items: Vec::new(), title: false, row: 0.0, from_list: false, show_pending: false });

/// Whether the menu manager `mgr` has a part with the key symbol `key`
/// (its parts: [mgr+0x148], [mgr+0x150] of them, 48-byte entries, the
/// part at +8 and its key at +0x28 there, as the request function finds them).
fn has_part(mgr: u64, key: u32) -> bool {
    if mgr == 0 || key == 0 {
        return false;
    }
    // SAFETY: the game's menu manager and its part table.
    unsafe {
        let (parts, n) = (*((mgr + 0x148) as *const u64), *((mgr + 0x150) as *const u32));
        (0..n as u64).any(|i| {
            let part = *((parts + i * 48 + 8) as *const u64);
            part != 0 && *((part + 0x28) as *const u32) == key
        })
    }
}

std::thread_local! {
    /// The port's own requests (not filtered again).
    static OWN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn trace_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TLG_MENU_TRACE").is_ok_and(|v| v == "1"))
}

fn slot_u8(i: u64) -> *mut u8 {
    (SLOTS.load(Ordering::Relaxed) + 8 + i) as *mut u8
}

/// The request filter's switch: on (the port follows the list's cursor for
/// the PC item's label; requests come a few a frame, in menus only).
fn set_filter(_on: bool) {
    // SAFETY: the patch's data slot (writable, never freed).
    unsafe { std::ptr::write_volatile(slot_u8(0), 1) };
}

pub fn apply(p: &mut Patcher, name: &str) {
    if std::env::var("TLG_OPTIONS_ITEM").is_ok_and(|v| v == "0") {
        return;
    }
    match patch(p) {
        Ok(()) => crate::info!("pc settings: an item in the Options lists ({name}+{LIST_SITE:#x}, choice +{CHOICE_SITE:#x}, requests +{REQ_SITE:#x})"),
        Err(e) => crate::warn!("pc settings: {name}: {e}; no PC Settings item in the Options menu (F10 opens them)"),
    }
}

fn patch(p: &mut Patcher) -> Result<(), String> {
    use iced_x86::code_asm::*;
    let checks: [(usize, &[u8]); 7] = [
        (REQ_EPILOGUE as usize, &REQ_EPILOGUE_BYTES),
        (SYM_HASH as usize, &SYM_HASH_BYTES),
        (SYM_INTERN as usize, &SYM_INTERN_BYTES),
        (LIST_SITE, &LIST_BYTES),
        (CHOICE_SITE, &CHOICE_BYTES),
        (HELPER_SITE, &HELPER_BYTES),
        (PLACE_SITE, &PLACE_BYTES),
    ];
    for (off, b) in checks {
        if p.image.get(off..off + b.len()) != Some(b) {
            return Err(format!("unexpected code at +{off:#x} (not update 01.03?)"));
        }
    }
    let base = p.base;
    BASE.store(base, Ordering::Relaxed);
    // (Two symbols, the filter's switch and answer, the third symbol.)
    let slots = p.alloc_data(&[0u8; 16]).ok_or("no data space")?;
    SLOTS.store(slots, Ordering::Relaxed);
    let list_fn = tlg_options_list as *const () as usize as u64;
    let choice_fn = tlg_options_choice as *const () as usize as u64;
    let req_fn = tlg_menu_request as *const () as usize as u64;
    let helper_fn = tlg_list_helper as *const () as usize as u64;
    let place_fn = tlg_cursor_place as *const () as usize as u64;
    // Caller-saved registers around a call into the port (the stack is
    // 16-aligned at each site; 9 pushes and 8 bytes more keep it so).
    let save = |a: &mut CodeAssembler| -> Result<(), IcedError> {
        for r in [rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11] {
            a.push(r)?;
        }
        a.sub(rsp, 8)
    };
    let restore = |a: &mut CodeAssembler| -> Result<(), IcedError> {
        a.add(rsp, 8)?;
        for r in [r11, r10, r9, r8, rdi, rsi, rdx, rcx, rax] {
            a.pop(r)?;
        }
        Ok(())
    };
    p.detour(LIST_SITE, &LIST_BYTES, |a, ins| {
        save(a)?;
        a.lea(rdi, ptr(rbp + LIST_VAR))?;
        a.lea(rsi, ptr(rbp + COUNT_VAR))?;
        // (r12: the Options update's object, the menu manager - its part
        // table at +0x148 - loaded from [rbp-0CC0h] at 0x9274b1.)
        a.mov(rdx, r12)?;
        a.mov(rax, list_fn)?;
        a.call(rax)?;
        restore(a)?;
        a.add_instruction(*ins)?;
        a.jmp(base + (LIST_SITE + LIST_BYTES.len()) as u64)
    })?;
    p.detour(HELPER_SITE, &HELPER_BYTES, |a, ins| {
        // (Flags are set again before they are read; the stack is 16-aligned
        // here. The helper's own first four arguments - context, items,
        // count, the cursor's address - are still in place.)
        save(a)?;
        a.mov(rax, helper_fn)?;
        a.call(rax)?;
        restore(a)?;
        a.add_instruction(*ins)?;
        a.jmp(base + (HELPER_SITE + HELPER_BYTES.len()) as u64)
    })?;
    p.detour(PLACE_SITE, &PLACE_BYTES, |a, ins| {
        // (Its arguments still in place; the stack is 16-aligned here.)
        save(a)?;
        a.mov(rax, place_fn)?;
        a.call(rax)?;
        restore(a)?;
        a.add_instruction(*ins)?;
        a.jmp(base + (PLACE_SITE + PLACE_BYTES.len()) as u64)
    })?;
    p.detour(CHOICE_SITE, &CHOICE_BYTES, |a, ins| {
        let mut ours = a.create_label();
        // (r11 holds the helper's item count: kept.)
        a.push(r11)?;
        a.mov(r11, slots)?;
        a.cmp(eax, dword_ptr(r11))?;
        a.je(ours)?;
        a.cmp(eax, dword_ptr(r11 + 4))?;
        a.je(ours)?;
        a.cmp(eax, dword_ptr(r11 + 12))?;
        a.je(ours)?;
        a.pop(r11)?;
        a.add_instruction(*ins)?;
        a.jmp(base + (CHOICE_SITE + CHOICE_BYTES.len()) as u64)?;
        a.set_label(&mut ours)?;
        a.pop(r11)?;
        // (What the skipped code stores for the helper's rest: the cursor
        // pointer at [rbp-58h].)
        a.mov(qword_ptr(rbp - 0x58), rcx)?;
        save(a)?;
        a.mov(rdi, r12)?;
        a.mov(esi, eax)?;
        a.mov(rax, choice_fn)?;
        a.call(rax)?;
        restore(a)?;
        a.jmp(base + CHOICE_DONE)
    })?;
    p.detour(REQ_SITE, &REQ_BYTES, |a, ins| {
        let mut pass = a.create_label();
        a.push(rax)?;
        a.mov(rax, slots + 8)?;
        a.cmp(byte_ptr(rax), 0)?;
        a.pop(rax)?;
        a.je(pass)?;
        save(a)?;
        a.mov(edx, r12d)?;
        a.mov(rcx, rbp)?;
        a.mov(rax, req_fn)?;
        a.call(rax)?;
        restore(a)?;
        a.push(rax)?;
        a.mov(rax, slots + 9)?;
        a.cmp(byte_ptr(rax), 0)?;
        a.pop(rax)?;
        a.jne(base + REQ_EPILOGUE)?;
        a.set_label(&mut pass)?;
        a.add_instruction(*ins)?;
        a.jmp(base + (REQ_SITE + REQ_BYTES.len()) as u64)
    })?;
    set_filter(false);
    Ok(())
}

/// A symbol of the game's table for `name` (interned if new).
fn intern(name: &str) -> u32 {
    let b = BASE.load(Ordering::Relaxed);
    let c = std::ffi::CString::new(name).unwrap_or_default();
    unsafe {
        let h = crate::hle::call_guest(b + SYM_HASH, &[c.as_ptr() as u64]);
        crate::hle::call_guest(b + SYM_INTERN, &[c.as_ptr() as u64, h & 0xffff_ffff, name.len() as u64, 0]) as u32
    }
}

fn syms(st: &mut State) -> [u32; 6] {
    if let Some(s) = st.syms {
        return s;
    }
    let s = [intern(KEY_TITLE), intern(KEY_PAUSE), intern("OptionMenu_TITLE"), intern("OptionMenu_TITLE2"), intern("OptionMenu_ITEM_CURSOR"), intern(KEY_PAUSE3)];
    // SAFETY: the patch's data slots (writable, never freed).
    unsafe {
        let slots = SLOTS.load(Ordering::Relaxed) as *mut u32;
        std::ptr::write_volatile(slots, s[0]);
        std::ptr::write_volatile(slots.add(1), s[1]);
        std::ptr::write_volatile(slots.add(3), s[5]);
    }
    crate::info!("pc settings: Options item symbols {} / {} / {}", s[0], s[1], s[5]);
    st.syms = Some(s);
    s
}

fn global_u32(off: u64) -> u32 {
    // SAFETY: a static of the game's image.
    unsafe { std::ptr::read_volatile((BASE.load(Ordering::Relaxed) + off) as *const u32) }
}

fn send(mgr: u64, part: u32, req: u32) {
    if mgr == 0 || part == 0 || req == 0 {
        return;
    }
    OWN.with(|o| o.set(true));
    unsafe { crate::hle::call_guest(BASE.load(Ordering::Relaxed) + SEND_REQUEST, &[mgr, part as u64, req as u64]) };
    OWN.with(|o| o.set(false));
}

/// The Options update's list (pointer and count, in place): a copy with the
/// PC item before the last one (Back).
fn options_list(list_var: u64, count_var: u64, mgr: u64) {
    let b = BASE.load(Ordering::Relaxed);
    // SAFETY: the Options update's locals (its frame).
    let (list, count) = unsafe { (*(list_var as *const u64), *(count_var as *const u32)) };
    let title = list == b + TITLE_LIST;
    if !(title || list == b + PAUSE_LIST || list == b + PAUSE_LIST_T) || !(2..=16).contains(&count) {
        return;
    }
    let mut st = STATE.lock();
    let s = syms(&mut st);
    // (The pause menu's list without Treasure: the item in Treasure's row.)
    let (pc, row) = if title {
        (s[0], -0.1164f32)
    } else if list == b + PAUSE_LIST_T {
        (s[1], -0.141)
    } else {
        (s[5], -0.0941)
    };
    // (Only with its part made by the menu manager: the list code takes
    // every item's part as there. The manager makes the Options parts
    // right after the list's first pick: from the next frame on the list
    // has the item, which then gets the "reqSelect" the others got.)
    if !has_part(mgr, pc) {
        st.extending = false;
        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        // SAFETY: the game's menu manager (its part count).
        let made = mgr != 0 && unsafe { *((mgr + 0x150) as *const u32) } != 0;
        if made && !LOGGED.swap(true, Ordering::Relaxed) {
            crate::warn!("pc settings: the menus have no part {} (manager {mgr:#x}); the Options list stays as it is", if title { KEY_TITLE } else { KEY_PAUSE });
            if trace_on() {
                // SAFETY: the game's menu manager and its part table.
                let keys: Vec<String> = unsafe {
                    let (parts, n) = (*((mgr + 0x148) as *const u64), *((mgr + 0x150) as *const u32));
                    (0..n as u64)
                        .filter_map(|i| {
                            let part = *((parts + i * 48 + 8) as *const u64);
                            (part != 0).then(|| crate::title_patches::symbol_name(*((part + 0x28) as *const u32)))
                        })
                        .collect()
                };
                crate::info!("pc settings: the menu manager's {} parts: {}", keys.len(), keys.join(" "));
            }
        }
        return;
    }
    st.seen_mgr = mgr;
    // SAFETY: the game's static array of `count` symbols.
    let items = unsafe { std::slice::from_raw_parts(list as *const u32, count as usize) };
    let mut v: Vec<u32> = items[..count as usize - 1].to_vec();
    v.push(pc);
    v.push(items[count as usize - 1]);
    // (The copy matching this one, if made before: the game's arrays do not change.)
    let copy = match st.lists.iter().position(|l| **l == v[..]) {
        Some(i) => &st.lists[i],
        None => {
            st.lists.push(v.clone().into_boxed_slice());
            st.lists.last().unwrap()
        }
    };
    let p = copy.as_ptr() as u64;
    st.items = v;
    st.title = title;
    st.row = row;
    let first = !std::mem::replace(&mut st.extending, true);
    drop(st);
    // SAFETY: as above.
    unsafe {
        *(list_var as *mut u64) = p;
        *(count_var as *mut u32) = count + 1;
    }
    if first {
        send(mgr, pc, global_u32(REQ_SHOW));
        send(mgr, pc, global_u32(REQ_SELECT));
    }
}

/// The PC item was chosen in the list helper (`ctx` its first argument).
fn options_choice(ctx: u64, sym: u32) {
    // SAFETY: the list helper's context: [ctx+0x18] points at the menu manager pointer.
    let mgr = unsafe { *(*((ctx + 0x18) as *const u64) as *const u64) };
    let (items, title, s) = {
        let mut st = STATE.lock();
        let s = syms(&mut st);
        st.mgr = mgr;
        st.from_list = true;
        st.show_pending = false;
        (st.items.clone(), st.title, s)
    };
    crate::info!("pc settings: chosen in the {} Options list (symbol {sym})", if title { "title screen's" } else { "pause menu's" });
    let hide = global_u32(REQ_HIDE);
    send(mgr, if title { s[3] } else { s[2] }, hide);
    for &i in &items {
        send(mgr, i, hide);
    }
    send(mgr, s[4], hide);
    set_filter(true);
    crate::pc_menu::open_from_options();
}

/// When the list last showed its cursor (process ms; the list state shows
/// it every frame while the list is on screen).
static CURSOR_SEEN: AtomicU64 = AtomicU64::new(0);
/// When the list helper last ran over a list with the PC item (process ms:
/// every frame while that list is the active one, not on its pages).
static LIST_ACTIVE: AtomicU64 = AtomicU64::new(0);

/// The title or pause menu placed its cursor at `item` (with the cursor
/// parts `a`, `b`).
fn cursor_place(_ctx: u64, item: u32, a: u32, b: u32) {
    static LAST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if LAST.swap(item, Ordering::Relaxed) != item && trace_on() {
        crate::info!("menu cursor: {} ({} {})", crate::title_patches::symbol_name(item), crate::title_patches::symbol_name(a), crate::title_patches::symbol_name(b));
    }
}

/// The list the game's list helper last ran over (each frame while a list
/// menu is up): its items (symbols), the cursor's address, when (ms).
pub struct ActiveList {
    pub items: Vec<u32>,
    pub cursor: u64,
    pub at: u64,
}

pub static ACTIVE_LIST: Mutex<Option<ActiveList>> = Mutex::new(None);

/// The list helper starts over `count` items (symbols at `items`) with the
/// cursor at `cursor`: notes the active list (and whether it is one with
/// the PC item).
fn list_helper(_ctx: u64, items: u64, count: u32, cursor: u64) {
    let now = crate::log::elapsed_ms() as u64;
    // (Never waits: the locks are the port's, taken briefly elsewhere.)
    if let Some(st) = STATE.try_lock() {
        if st.lists.iter().any(|l| l.as_ptr() as u64 == items) {
            LIST_ACTIVE.store(now, Ordering::Relaxed);
        }
    }
    if !(1..=32).contains(&count) || items == 0 {
        return;
    }
    // SAFETY: the helper's arguments: `count` symbols at `items`.
    let syms = unsafe { std::slice::from_raw_parts(items as *const u32, count as usize) };
    if let Some(mut a) = ACTIVE_LIST.try_lock() {
        let same = a.as_ref().is_some_and(|l| l.items == syms && l.cursor == cursor);
        if !same && trace_on() {
            let names: Vec<String> = syms.iter().map(|&s| crate::title_patches::symbol_name(s)).collect();
            crate::info!("menu list: {} (cursor at {cursor:#x})", names.join(" "));
        }
        *a = Some(ActiveList { items: syms.to_vec(), cursor, at: now });
    }
}

/// The PC item's label while its list is on screen: the place of the row's
/// left edge and middle, in 1080p pixels (the layouts' rows map to screen
/// rows as y = 543.4 - 1001.3 * row: Camera 0.0743 at 469, Back -0.1634 at
/// 707 on the title screen), and whether it is the title screen's list.
pub fn item_label() -> Option<(f32, f32, bool)> {
    let now = crate::log::elapsed_ms() as u64;
    if now.saturating_sub(CURSOR_SEEN.load(Ordering::Relaxed)) > 150 || now.saturating_sub(LIST_ACTIVE.load(Ordering::Relaxed)) > 150 {
        return None;
    }
    let st = STATE.lock();
    if !st.extending || st.from_list {
        return None;
    }
    // (Rows: the title Copyright row and Back's half way, -0.1164; the
    // pause menu's free row, -0.141, or Treasure's, -0.0941.)
    Some((148.0, 543.4 - 1001.3 * st.row, st.title))
}

/// The PC settings closed after `options_choice`: the list comes back at
/// the game's next part request.
pub fn closed() {
    let mut st = STATE.lock();
    if st.from_list {
        st.from_list = false;
        st.show_pending = true;
    }
}

/// A part request while the filter is on: whether to drop it (answer slot).
fn menu_request(mgr: u64, part: u32, req: u32, rbp: u64) {
    let mut drop = false;
    if trace_on() {
        trace(mgr, part, req, rbp);
    }
    if !OWN.with(|o| o.get()) {
        let (show, from_list, cursor, items, title_sym, first) = {
            let mut st = STATE.lock();
            let first = st.seen_mgr == 0;
            st.seen_mgr = mgr;
            let s = syms(&mut st);
            let show = std::mem::take(&mut st.show_pending);
            (show, st.from_list, s[4], st.items.clone(), if st.title { s[3] } else { s[2] }, first)
        };
        if show {
            send(mgr, title_sym, global_u32(REQ_SHOW));
            for &i in &items {
                send(mgr, i, global_u32(REQ_SELECT));
            }
            set_filter(false);
        } else if from_list && part == cursor && req == global_u32(REQ_SHOW) {
            drop = true;
        } else if part == cursor && req == global_u32(REQ_SHOW) {
            CURSOR_SEEN.store(crate::log::elapsed_ms() as u64, Ordering::Relaxed);
        }
        let _ = first;
    }
    // SAFETY: the patch's data slot.
    unsafe { std::ptr::write_volatile(slot_u8(1), drop as u8) };
}

fn trace(mgr: u64, part: u32, req: u32, rbp: u64) {
    let b = BASE.load(Ordering::Relaxed);
    let mut callers = Vec::new();
    let mut fp = rbp;
    for _ in 0..3 {
        // (Guest threads run on host stacks: frames checked for order and alignment.)
        if fp == 0 || fp % 8 != 0 {
            break;
        }
        // SAFETY: a frame of this thread's stack (the chain only moves up).
        let (next, r) = unsafe { (*(fp as *const u64), *((fp + 8) as *const u64)) };
        callers.push(format!("+{:#x}", r.wrapping_sub(b)));
        if next <= fp || next - fp > 1 << 20 {
            break;
        }
        fp = next;
    }
    crate::info!("menu request: {} {} (mgr {mgr:#x}, from {})", crate::title_patches::symbol_name(part), crate::title_patches::symbol_name(req), callers.join(" < "));
}

sysv! {
    fn tlg_options_list(list_var: u64, count_var: u64, mgr: u64) -> i32 {
        options_list(list_var, count_var, mgr);
        0
    }
    fn tlg_options_choice(ctx: u64, sym: u32) -> i32 {
        options_choice(ctx, sym);
        0
    }
    fn tlg_menu_request(mgr: u64, part: u32, req: u32, rbp: u64) -> i32 {
        menu_request(mgr, part, req, rbp);
        0
    }
    fn tlg_cursor_place(ctx: u64, item: u32, a: u32, b: u32) -> i32 {
        cursor_place(ctx, item, a, b);
        0
    }
    fn tlg_list_helper(ctx: u64, items: u64, count: u32, cursor: u64) -> i32 {
        list_helper(ctx, items, count, cursor);
        0
    }
}

/// The game's camera options, Horizontal and Vertical Camera "Reversed":
/// (x, y) (its option values: `OPTIONS_PTR`; before they exist, Standard).
pub fn camera_options() -> (bool, bool) {
    // (The option object, once found: made at start, never freed.)
    static OPTS: AtomicU64 = AtomicU64::new(0);
    let mut q = OPTS.load(Ordering::Relaxed);
    if q == 0 {
        let b = BASE.load(Ordering::Relaxed);
        if b == 0 {
            return (false, false);
        }
        // SAFETY: a pointer of the game's image, then the object it points
        // at, each checked readable.
        unsafe {
            let p = std::ptr::read_volatile((b + OPTIONS_PTR) as *const u64);
            if p == 0 || !crate::debug_mem::host_or_guest_readable(p + 0x100, 8) {
                return (false, false);
            }
            q = std::ptr::read_volatile((p + 0x100) as *const u64);
        }
        if q == 0 || !crate::debug_mem::host_or_guest_readable(q + 0x40, 8) {
            return (false, false);
        }
        OPTS.store(q, Ordering::Relaxed);
    }
    // SAFETY: as above.
    let (v, h) = unsafe { (std::ptr::read_volatile((q + 0x40) as *const f32), std::ptr::read_volatile((q + 0x44) as *const f32)) };
    (h < 0.0, v < 0.0)
}
