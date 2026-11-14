//! Achievements: the title's trophies (libSceNpTrophy) kept by the port -
//! unlocks recorded locally beside the saves (one line per trophy: its id
//! and the Unix time it unlocked), a pop-up in the game's style when one
//! unlocks (pc_menu.lua draws it with the game's GUI), and the list on the
//! PC Settings page Achievements.
//!
//! The set comes from the player's own trophy00.trp (sce_sys/trophy): its
//! TROPnnn.PNG icons are plain PNG and go into a texture of the port's
//! archive at start (made from the player's files on their PC, never
//! shipped). Names and descriptions are in the set's encrypted
//! TROP.ESFM/TROPCONF.ESFM: where decrypted copies (TROP.SFM, TROPCONF.SFM:
//! the set's XML) sit beside trophy00.trp they are shown, otherwise the
//! trophies are numbered. The platinum is taken to be trophy 0 (as in PS4
//! sets) and unlocks with all the others, as the console does.

use parking_lot::Mutex;
use std::collections::VecDeque;
use tlg_config::trophy;

/// The icon atlas and the pop-up's frame in the port's archive.
pub const ICONS_NAME: &str = "game/textures/ui/tlgpc/trophy_icons";
pub const FRAME_NAME: &str = "game/textures/ui/tlgpc/trophy_frame";
/// Icon size in the atlas, icons per row.
const ICON: u32 = 128;
const COLS: u32 = 6;
/// The pop-up frame's size (1080p units, drawn 1:1).
pub const FRAME_W: u32 = 544;
pub const FRAME_H: u32 = 120;

#[derive(Clone, Debug, Default)]
pub struct Trophy {
    pub id: i32,
    pub name: Option<String>,
    pub detail: Option<String>,
    /// P, G, S or B (TROPCONF), when known.
    pub grade: Option<char>,
    pub hidden: bool,
}

struct Set {
    /// trophy00.trp.
    path: std::path::PathBuf,
    /// Its trophies (names filled in again when the player gives a key).
    trophies: parking_lot::RwLock<Vec<Trophy>>,
    /// trophy00.trp's entries by name.
    entries: Vec<(String, Vec<u8>)>,
}

static SET: std::sync::OnceLock<Option<Set>> = std::sync::OnceLock::new();
/// Unlocks waiting for their pop-up.
static TOASTS: Mutex<VecDeque<i32>> = Mutex::new(VecDeque::new());
static RECORD_LOCK: Mutex<()> = Mutex::new(());

/// The set: trophies by their TROPnnn.PNG icons, with names where the
/// decrypted XML is there.
fn set() -> Option<&'static Set> {
    SET.get_or_init(|| {
        let p = crate::hle::kernel::fs::host_path_existing("/app0/sce_sys/trophy/trophy00.trp")?;
        let b = std::fs::read(&p).ok()?;
        let entries = trophy::trp_entries(&b);
        let mut ids: Vec<i32> = entries.iter().filter_map(|(n, _)| n.strip_prefix("TROP")?.strip_suffix(".PNG")?.parse().ok()).collect();
        ids.sort();
        let trophies: Vec<Trophy> = ids.iter().map(|&id| Trophy { id, ..Default::default() }).collect();
        let s = Set { path: p, trophies: parking_lot::RwLock::new(trophies), entries };
        let (_, how) = s.load_texts(key_file());
        crate::info!("achievements: {} trophies in {} ({how})", s.trophies.read().len(), s.path.display());
        Some(s)
    })
    .as_ref()
}

impl Set {
    /// Fills the trophies' names, descriptions and grades from the set's
    /// XML: decrypted with the trophy key `tk` (in memory), or decrypted
    /// copies beside the file, or plain in it. Returns how many have names
    /// and a note on how.
    fn load_texts(&self, tk: Option<[u8; 16]>) -> (usize, String) {
        let (p, entries) = (&self.path, &self.entries);
        let decrypt = tk.as_ref().and_then(|k| esfm_decrypter(entries, k));
        let xml = |name: &str| -> Option<String> {
            let side = p.with_file_name(name);
            let esfm = name.strip_suffix(".SFM").map(|b| format!("{b}.ESFM"));
            std::fs::read(&side)
                .ok()
                .or_else(|| entries.iter().find(|(n, _)| n == name).map(|(_, d)| d.clone()))
                .or_else(|| {
                    let (key, _) = decrypt.as_ref()?;
                    let e = entries.iter().find(|(n, _)| Some(n) == esfm.as_ref())?;
                    trophy::esfm_decrypt(key, &e.1)
                })
                .map(|d| String::from_utf8_lossy(&d).to_string())
        };
        // (The game's language: TROP_nn by the PS4's language code; the
        // set's default otherwise.)
        let (_, lang_code, _) = crate::hle::libs::app::game_language();
        let trop = Some(format!("TROP_{lang_code:02}.SFM")).filter(|n| entries.iter().any(|(e, _)| e.as_str() == n.replace(".SFM", ".ESFM")) || p.with_file_name(n).exists()).unwrap_or_else(|| "TROP.SFM".into());
        let mut trophies = self.trophies.write();
        let mut named = 0;
        if let Some(x) = xml(&trop) {
            for (id, _, body) in xml_trophies(&x) {
                if let Some(t) = trophies.iter_mut().find(|t| t.id == id) {
                    t.name = xml_text(&body, "name");
                    t.detail = xml_text(&body, "detail");
                    named += t.name.is_some() as usize;
                }
            }
        }
        if let Some(x) = xml("TROPCONF.SFM") {
            for (id, attrs, _) in xml_trophies(&x) {
                if let Some(t) = trophies.iter_mut().find(|t| t.id == id) {
                    t.grade = xml_attr(&attrs, "ttype").and_then(|g| g.chars().next());
                    t.hidden = xml_attr(&attrs, "hidden").is_some_and(|h| h == "yes");
                }
            }
        }
        let how = match (&decrypt, named) {
            (_, 0) if tk.is_none() => format!("numbered: no trophy key ({})", key_path().display()),
            (None, 0) => "numbered: the trophy key opens none of the set's files".to_string(),
            (_, 0) => "numbered: no names in the set's files".to_string(),
            (Some((_, id)), n) => format!("{n} named from {trop} ({id})"),
            (None, n) => format!("{n} named from {trop}"),
        };
        (named, how)
    }
}

// ---------------------------------------------------------------------------
// The set's encrypted XML (TROP*.ESFM, TROPCONF.ESFM) and the player's
// trophy key that opens it: tlg_config::trophy (shared with the launcher's
// Trophy key field). The key is the player's own (never shipped): 32 hex
// digits in trophy-key.txt in the user folder, from the launcher, the
// Achievements page's Trophy Names button or by hand.

pub fn key_path() -> std::path::PathBuf {
    trophy::key_path(&crate::config::paths().user)
}

fn key_file() -> Option<[u8; 16]> {
    trophy::read_key(&crate::config::paths().user)
}

/// The key that opens the set's files with `tk`, and its NP communication
/// ID (sce_sys/npbind.dat's when it fits, else found by a search).
fn esfm_decrypter(entries: &[(String, Vec<u8>)], tk: &[u8; 16]) -> Option<([u8; 16], String)> {
    let bound = crate::hle::kernel::fs::host_path_existing("/app0/sce_sys/npbind.dat").and_then(|p| std::fs::read(p).ok()).and_then(|d| trophy::comm_id_in(&d));
    let t0 = std::time::Instant::now();
    let r = trophy::find_key(entries, tk, bound.as_deref());
    match &r {
        Some((_, id)) if bound.as_deref() == Some(id.as_str()) => {}
        Some((_, id)) => crate::info!("achievements: the set's NP communication ID is {id} (found in {:.0} ms)", t0.elapsed().as_secs_f64() * 1000.0),
        None => crate::warn!("achievements: the trophy key opens none of the set's files"),
    }
    r
}

/// The player pasted a trophy key on the Achievements page (Trophy Names):
/// the clipboard's text checked against the set, kept in trophy-key.txt
/// and the names shown at once. Returns "reload" or what went wrong.
pub fn paste_key() -> String {
    let Some(text) = clipboard_text() else {
        return crate::pc_menu::tr("The clipboard is empty. Copy the trophy key first.").to_string();
    };
    match set_key(&text) {
        Ok(n) => {
            crate::info!("achievements: trophy key pasted on the Achievements page ({n} named)");
            "reload".into()
        }
        Err(e) => e,
    }
}

/// Checks a trophy key (text holding its 32 hex digits) against the set;
/// when it opens it, keeps it and names the trophies. Returns how many are
/// named, or what went wrong.
pub fn set_key(text: &str) -> Result<usize, String> {
    let Some(key) = trophy::parse_key(text) else {
        return Err(crate::pc_menu::tr("No trophy key in the clipboard (32 hex digits).").to_string());
    };
    let Some(s) = set() else { return Err(crate::pc_menu::tr("No trophy file in the game files.").to_string()) };
    // (A key that opens nothing leaves the names as they were.)
    let (n, how) = s.load_texts(Some(key));
    if n == 0 {
        return Err(crate::pc_menu::tr("That key does not fit this game's trophies.").to_string());
    }
    crate::info!("achievements: {how}");
    if let Err(e) = trophy::write_key(&crate::config::paths().user, &key) {
        return Err(format!("{} {}: {e}", crate::pc_menu::tr("Could not save the key in"), key_path().display()));
    }
    Ok(n)
}

/// The clipboard's text (CF_UNICODETEXT), if any.
fn clipboard_text() -> Option<String> {
    use windows_sys::Win32::System::DataExchange::{CloseClipboard, GetClipboardData, IsClipboardFormatAvailable, OpenClipboard};
    use windows_sys::Win32::System::Memory::{GlobalLock, GlobalSize, GlobalUnlock};
    const CF_UNICODETEXT: u32 = 13;
    // SAFETY: the clipboard is opened, read within its global block's size
    // while locked, and closed again.
    unsafe {
        if IsClipboardFormatAvailable(CF_UNICODETEXT) == 0 || OpenClipboard(std::ptr::null_mut()) == 0 {
            return None;
        }
        let mut out = None;
        let h = GetClipboardData(CF_UNICODETEXT);
        if !h.is_null() {
            let p = GlobalLock(h) as *const u16;
            if !p.is_null() {
                let max = GlobalSize(h) / 2;
                let s = std::slice::from_raw_parts(p, max);
                let n = s.iter().position(|&c| c == 0).unwrap_or(max).min(4096);
                out = Some(String::from_utf16_lossy(&s[..n]));
                GlobalUnlock(h);
            }
        }
        CloseClipboard();
        out
    }
}

/// `<trophy id="N" ...>body</trophy>` elements: id, attribute text, body.
fn xml_trophies(x: &str) -> Vec<(i32, String, String)> {
    let mut out = Vec::new();
    let mut rest = x;
    while let Some(i) = rest.find("<trophy ") {
        let after = &rest[i + 8..];
        let Some(gt) = after.find('>') else { break };
        let attrs = after[..gt].to_string();
        let body_start = &after[gt + 1..];
        let end = body_start.find("</trophy>").unwrap_or(body_start.len());
        if let Some(id) = xml_attr(&attrs, "id").and_then(|v| v.trim().parse().ok()) {
            out.push((id, attrs, body_start[..end].to_string()));
        }
        rest = &body_start[end..];
    }
    out
}

fn xml_attr(attrs: &str, name: &str) -> Option<String> {
    let k = format!("{name}=\"");
    let i = attrs.find(&k)? + k.len();
    let j = attrs[i..].find('"')?;
    Some(attrs[i..i + j].to_string())
}

fn xml_text(body: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let i = body.find(&open)? + open.len();
    let j = body[i..].find(&format!("</{tag}>"))?;
    let t = body[i..i + j].replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&apos;", "'");
    Some(t.trim().to_string()).filter(|s| !s.is_empty())
}

pub fn count() -> u32 {
    set().map_or(0, |s| s.trophies.read().len() as u32)
}

/// The recorded unlocks: id and Unix time (0: not recorded, older files).
pub fn unlocked() -> Vec<(i32, u64)> {
    let _g = RECORD_LOCK.lock();
    std::fs::read_to_string(crate::hle::libs::savedata::trophy_file())
        .map(|t| {
            t.lines()
                .filter_map(|l| {
                    let mut w = l.split_whitespace();
                    let id = w.next()?.parse().ok()?;
                    Some((id, w.next().and_then(|v| v.parse().ok()).unwrap_or(0)))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn record(ids: &[(i32, u64)]) -> bool {
    let _g = RECORD_LOCK.lock();
    let text: String = ids.iter().map(|(i, t)| format!("{i} {t}\n")).collect();
    let f = crate::hle::libs::savedata::trophy_file();
    let tmp = f.with_extension("tmp");
    std::fs::write(&tmp, text).and_then(|_| std::fs::rename(&tmp, &f)).is_ok()
}

/// The title unlocked a trophy: recorded (with the time), its pop-up queued;
/// the platinum (0) follows when every other trophy is in. Returns whether
/// it is new and the platinum's id when this unlock completed the set.
pub fn unlock(id: i32) -> (bool, Option<i32>) {
    let mut ids = unlocked();
    if ids.iter().any(|&(i, _)| i == id) {
        return (false, None);
    }
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    ids.push((id, now));
    let mut platinum = None;
    let n = count() as i32;
    if n > 1 && id != 0 && !ids.iter().any(|&(i, _)| i == 0) && (1..n).all(|t| ids.iter().any(|&(i, _)| i == t)) {
        ids.push((0, now));
        platinum = Some(0);
    }
    ids.sort();
    if !record(&ids) {
        crate::warn!("achievements: could not record trophy {id} in {}", crate::hle::libs::savedata::trophy_file().display());
    }
    let mut q = TOASTS.lock();
    q.push_back(id);
    if let Some(p) = platinum {
        q.push_back(p);
    }
    (true, platinum)
}

fn name_of(t: &Trophy) -> String {
    match &t.name {
        Some(n) => n.clone(),
        None => format!("{} {}", crate::pc_menu::tr("Trophy"), t.id + 1),
    }
}

fn grade_of(t: &Trophy) -> &'static str {
    match t.grade {
        Some('P') => "Platinum",
        Some('G') => "Gold",
        Some('S') => "Silver",
        Some('B') => "Bronze",
        _ => "",
    }
}

/// A trophy's icon in the atlas: u0 v0 u1 v1.
fn icon_uv(id: i32) -> [f32; 4] {
    let n = count().max(1);
    let rows = n.div_ceil(COLS);
    let (aw, ah) = ((COLS * ICON) as f32, (rows * ICON) as f32);
    let Some(i) = set().and_then(|s| s.trophies.read().iter().position(|t| t.id == id)) else { return [0.0; 4] };
    let (c, r) = (i as u32 % COLS, i as u32 / COLS);
    [(c * ICON) as f32 / aw, (r * ICON) as f32 / ah, ((c + 1) * ICON) as f32 / aw, ((r + 1) * ICON) as f32 / ah]
}

/// The next pop-up: name, line above it, icon u0 v0 u1 v1; none waiting:
/// empty.
pub fn next_toast() -> Vec<String> {
    let Some(id) = TOASTS.lock().pop_front() else { return Vec::new() };
    let Some(t) = set().and_then(|s| s.trophies.read().iter().find(|t| t.id == id).cloned()) else { return Vec::new() };
    let t = &t;
    let grade = grade_of(t);
    let above = if grade.is_empty() { crate::pc_menu::tr("Trophy unlocked").to_string() } else { format!("{} - {}", crate::pc_menu::tr("Trophy unlocked"), crate::pc_menu::tr(grade)) };
    let uv = icon_uv(id);
    let mut out = vec![name_of(t), above];
    out.extend(uv.iter().map(|v| format!("{v:.5}")));
    out
}

/// The Achievements page's rows (pc_settings::page_rows' layout): first the
/// Trophy Names button (kind `action`: pastes a trophy key; its value says
/// whether the names are there), then the trophies (kind `trophy`, value:
/// when it unlocked or "Locked", options: the icon's uv, flags `u` when
/// unlocked).
pub fn page_rows() -> Vec<String> {
    let Some(s) = set() else { return vec![crate::pc_menu::tr("Achievements").to_string(), "0".into()] };
    let got = unlocked();
    let list = s.trophies.read();
    let done = list.iter().filter(|t| got.iter().any(|&(i, _)| i == t.id)).count();
    let title = format!("{}   {done} / {}", crate::pc_menu::tr("Achievements"), list.len());
    let mut out = vec![title, (list.len() + 1).to_string()];
    let named = list.iter().any(|t| t.name.is_some());
    let (value, help) = if named {
        ("Shown", "Names from your PS4 trophy key. Copy another key and choose this to change it.")
    } else {
        ("Paste Key", "Copy your PS4 trophy key (32 hex digits), then choose this to show trophy names.")
    };
    out.extend(["action".to_string(), "trophy_key".into(), crate::pc_menu::tr("Trophy Names").into(), crate::pc_menu::tr(value).into(), String::new(), crate::pc_menu::tr(help).into(), String::new()]);
    for t in list.iter() {
        let when = got.iter().find(|&&(i, _)| i == t.id).map(|&(_, at)| at);
        let value = match when {
            Some(0) => crate::pc_menu::tr("Unlocked").to_string(),
            Some(at) => date(at),
            None => crate::pc_menu::tr("Locked").to_string(),
        };
        let hidden = t.hidden && when.is_none();
        let label = if hidden { crate::pc_menu::tr("Hidden trophy").to_string() } else { name_of(t) };
        let mut help = if hidden { String::new() } else { t.detail.clone().unwrap_or_default() };
        if t.name.is_none() {
            help = crate::pc_menu::tr("Names need the PS4 trophy key (see Trophy Names above).").to_string();
        }
        let grade = grade_of(t);
        if !grade.is_empty() {
            help = if help.is_empty() { crate::pc_menu::tr(grade).to_string() } else { format!("{}: {help}", crate::pc_menu::tr(grade)) };
        }
        let uv = icon_uv(t.id).iter().map(|v| format!("{v:.5}")).collect::<Vec<_>>().join("\t");
        out.extend(["trophy".to_string(), format!("trophy:{}", t.id), label, value, uv, help, if when.is_some() { "u".into() } else { String::new() }]);
    }
    out
}

/// A Unix time as the local date (YYYY-MM-DD).
fn date(at: u64) -> String {
    use windows_sys::Win32::Foundation::{FILETIME, SYSTEMTIME};
    use windows_sys::Win32::System::Time::{FileTimeToSystemTime, SystemTimeToTzSpecificLocalTime};
    let ft_val = (at + 11_644_473_600) * 10_000_000;
    let ft = FILETIME { dwLowDateTime: ft_val as u32, dwHighDateTime: (ft_val >> 32) as u32 };
    let mut utc: SYSTEMTIME = unsafe { std::mem::zeroed() };
    let mut local: SYSTEMTIME = unsafe { std::mem::zeroed() };
    unsafe {
        if FileTimeToSystemTime(&ft, &mut utc) == 0 || SystemTimeToTzSpecificLocalTime(std::ptr::null(), &utc, &mut local) == 0 {
            return String::new();
        }
    }
    format!("{:04}-{:02}-{:02}", local.wYear, local.wMonth, local.wDay)
}

// ---------------------------------------------------------------------------
// Textures for the port's archive.

/// A PNG's pixels as RGBA8 (8-bit RGB or RGBA, not interlaced: the trophy
/// icons' kind).
pub fn png_rgba(d: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    if d.get(..8)? != b"\x89PNG\r\n\x1a\n" {
        return None;
    }
    let (mut w, mut h, mut ct) = (0u32, 0u32, 0u8);
    let mut idat = Vec::new();
    let mut pos = 8;
    while pos + 8 <= d.len() {
        let n = u32::from_be_bytes(d[pos..pos + 4].try_into().ok()?) as usize;
        let kind = &d[pos + 4..pos + 8];
        let body = d.get(pos + 8..pos + 8 + n)?;
        match kind {
            b"IHDR" => {
                w = u32::from_be_bytes(body[0..4].try_into().ok()?);
                h = u32::from_be_bytes(body[4..8].try_into().ok()?);
                ct = body[9];
                if body[8] != 8 || body[12] != 0 || !(ct == 2 || ct == 6) {
                    return None;
                }
            }
            b"IDAT" => idat.extend_from_slice(body),
            b"IEND" => break,
            _ => {}
        }
        pos += 12 + n;
    }
    if w == 0 || h == 0 || w > 4096 || h > 4096 {
        return None;
    }
    let raw = miniz_oxide::inflate::decompress_to_vec_zlib(&idat).ok()?;
    let bpp = if ct == 6 { 4 } else { 3 };
    let stride = w as usize * bpp;
    if raw.len() < h as usize * (stride + 1) {
        return None;
    }
    let mut prev = vec![0u8; stride];
    let mut out = Vec::with_capacity(w as usize * h as usize * 4);
    for y in 0..h as usize {
        let f = raw[y * (stride + 1)];
        let mut line = raw[y * (stride + 1) + 1..(y + 1) * (stride + 1)].to_vec();
        for x in 0..stride {
            let a = if x >= bpp { line[x - bpp] as i32 } else { 0 };
            let b = prev[x] as i32;
            let c = if x >= bpp { prev[x - bpp] as i32 } else { 0 };
            let p = match f {
                0 => 0,
                1 => a,
                2 => b,
                3 => (a + b) / 2,
                4 => {
                    let p = a + b - c;
                    let (pa, pb, pc) = ((p - a).abs(), (p - b).abs(), (p - c).abs());
                    if pa <= pb && pa <= pc {
                        a
                    } else if pb <= pc {
                        b
                    } else {
                        c
                    }
                }
                _ => return None,
            };
            line[x] = (line[x] as i32 + p) as u8;
        }
        for px in line.chunks_exact(bpp) {
            out.extend_from_slice(&[px[0], px[1], px[2], if bpp == 4 { px[3] } else { 255 }]);
        }
        prev = line;
    }
    Some((w, h, out))
}

/// `rgba` (`w` x `h`) scaled to `size` x `size` (area average of 4x4
/// bilinear samples per texel).
fn scaled(w: u32, h: u32, rgba: &[u8], size: u32) -> Vec<u8> {
    let at = |x: f32, y: f32, c: usize| -> f32 {
        let (x, y) = (x.clamp(0.0, w as f32 - 1.0), y.clamp(0.0, h as f32 - 1.0));
        let (x0, y0) = (x.floor() as u32, y.floor() as u32);
        let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let p = |xx: u32, yy: u32| rgba[((yy * w + xx) * 4) as usize + c] as f32;
        (p(x0, y0) * (1.0 - fx) + p(x1, y0) * fx) * (1.0 - fy) + (p(x0, y1) * (1.0 - fx) + p(x1, y1) * fx) * fy
    };
    let (sx, sy) = (w as f32 / size as f32, h as f32 / size as f32);
    let mut out = vec![0u8; (size * size * 4) as usize];
    for oy in 0..size {
        for ox in 0..size {
            let mut acc = [0f32; 4];
            for j in 0..4 {
                for i in 0..4 {
                    let x = (ox as f32 + (i as f32 + 0.5) / 4.0) * sx - 0.5;
                    let y = (oy as f32 + (j as f32 + 0.5) / 4.0) * sy - 0.5;
                    for (c, v) in acc.iter_mut().enumerate() {
                        *v += at(x, y, c);
                    }
                }
            }
            let o = ((oy * size + ox) * 4) as usize;
            for c in 0..4 {
                out[o + c] = (acc[c] / 16.0).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// The icon atlas (RGBA8, `COLS` icons of `ICON` per row) from the set's
/// TROPnnn.PNG.
fn icon_atlas(s: &Set) -> Option<Vec<u8>> {
    let list = s.trophies.read();
    let n = list.len() as u32;
    if n == 0 {
        return None;
    }
    let rows = n.div_ceil(COLS);
    let (aw, ah) = (COLS * ICON, rows * ICON);
    let mut atlas = vec![0u8; (aw * ah * 4) as usize];
    for (i, t) in list.iter().enumerate() {
        let name = format!("TROP{:03}.PNG", t.id);
        let Some((w, h, px)) = s.entries.iter().find(|(n, _)| *n == name).and_then(|(_, d)| png_rgba(d)) else {
            crate::warn!("achievements: {name} is not a PNG this reader takes");
            continue;
        };
        let icon = scaled(w, h, &px, ICON);
        let (c, r) = (i as u32 % COLS, i as u32 / COLS);
        for y in 0..ICON {
            let dst = (((r * ICON + y) * aw + c * ICON) * 4) as usize;
            let src = (y * ICON * 4) as usize;
            atlas[dst..dst + (ICON * 4) as usize].copy_from_slice(&icon[src..src + (ICON * 4) as usize]);
        }
    }
    Some(crate::pc_files::gnf(aw, ah, &atlas))
}

/// The pop-up's frame: the game's guide boxes' look - a dark translucent
/// rounded box with a thin light edge.
fn frame() -> Vec<u8> {
    let (w, h) = (FRAME_W as f32, FRAME_H as f32);
    let r = 10.0f32;
    let mut px = vec![0u8; (FRAME_W * FRAME_H * 4) as usize];
    for y in 0..FRAME_H {
        for x in 0..FRAME_W {
            // Signed distance to the rounded box (inside negative).
            let (cx, cy) = (x as f32 + 0.5 - w / 2.0, y as f32 + 0.5 - h / 2.0);
            let (qx, qy) = (cx.abs() - (w / 2.0 - r - 1.0), cy.abs() - (h / 2.0 - r - 1.0));
            let d = (qx.max(0.0).hypot(qy.max(0.0)) + qx.max(qy).min(0.0)) - r;
            let inside = (0.5 - d).clamp(0.0, 1.0);
            let edge = (1.0 - (d + 0.9).abs() / 0.9).clamp(0.0, 1.0);
            let fill_a = 0.62 * inside;
            let a = edge * 0.85 + fill_a * (1.0 - edge * 0.85);
            let lum = if a > 0.0 { (edge * 0.85 * 0.82) / a } else { 0.0 };
            let o = ((y * FRAME_W + x) * 4) as usize;
            let v = (lum * 255.0).round() as u8;
            px[o..o + 4].copy_from_slice(&[v, v, v, (a * 255.0).round() as u8]);
        }
    }
    crate::pc_files::gnf(FRAME_W, FRAME_H, &px)
}

/// The archive's files: the icon atlas and the pop-up frame (none without
/// a trophy set).
pub fn files() -> Vec<(String, Vec<u8>)> {
    let Some(s) = set() else { return Vec::new() };
    let mut out = Vec::new();
    if let Some(a) = icon_atlas(s) {
        out.push((format!("{ICONS_NAME}.gnf"), a));
        out.push((format!("{FRAME_NAME}.gnf"), frame()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A PNG (RGB, one IDAT, the given filter per row) for the reader.
    fn png(w: u32, h: u32, rows: &[(u8, Vec<u8>)]) -> Vec<u8> {
        let mut raw = Vec::new();
        for (f, r) in rows {
            raw.push(*f);
            raw.extend_from_slice(r);
        }
        let z = miniz_oxide::deflate::compress_to_vec_zlib(&raw, 6);
        let chunk = |kind: &[u8], body: &[u8]| -> Vec<u8> {
            let mut c = (body.len() as u32).to_be_bytes().to_vec();
            c.extend_from_slice(kind);
            c.extend_from_slice(body);
            c.extend_from_slice(&[0, 0, 0, 0]); // (CRC unchecked)
            c
        };
        let mut ihdr = w.to_be_bytes().to_vec();
        ihdr.extend_from_slice(&h.to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
        let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
        out.extend(chunk(b"IHDR", &ihdr));
        out.extend(chunk(b"IDAT", &z));
        out.extend(chunk(b"IEND", &[]));
        out
    }

    #[test]
    fn png_filters() {
        // Row 0 plain; row 1 "up" (adds row 0); row 2 "sub" (adds the left pixel).
        let d = png(2, 3, &[(0, vec![10, 20, 30, 40, 50, 60]), (2, vec![1, 1, 1, 2, 2, 2]), (1, vec![5, 5, 5, 1, 2, 3])]);
        let (w, h, px) = png_rgba(&d).unwrap();
        assert_eq!((w, h), (2, 3));
        assert_eq!(&px[0..8], &[10, 20, 30, 255, 40, 50, 60, 255]);
        assert_eq!(&px[8..16], &[11, 21, 31, 255, 42, 52, 62, 255]);
        assert_eq!(&px[16..24], &[5, 5, 5, 255, 6, 7, 8, 255]);
    }

    #[test]
    fn xml_fields() {
        let x = r#"<trophyconf><trophy id="000" hidden="no" ttype="P" pid="-1"><name>All &amp; more</name><detail>Every trophy.</detail></trophy><trophy id="001" hidden="yes" ttype="B"><name>Two</name></trophy></trophyconf>"#;
        let t = xml_trophies(x);
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].0, 0);
        assert_eq!(xml_text(&t[0].2, "name").as_deref(), Some("All & more"));
        assert_eq!(xml_attr(&t[1].1, "hidden").as_deref(), Some("yes"));
        assert_eq!(xml_attr(&t[1].1, "ttype").as_deref(), Some("B"));
        assert_eq!(xml_text(&t[1].2, "detail"), None);
    }

    #[test]
    fn scaled_keeps_flat_colour() {
        let px: Vec<u8> = std::iter::repeat_n([200u8, 100, 50, 255], 240 * 240).flatten().collect();
        let s = scaled(240, 240, &px, 128);
        assert_eq!(s.len(), 128 * 128 * 4);
        assert!(s.chunks_exact(4).all(|p| p == [200, 100, 50, 255]));
    }
}
