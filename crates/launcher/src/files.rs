//! The game folders: what a folder holds (by its param.sfo) and whether the
//! files the game needs are there - its executable modules decrypted (SELF or
//! ELF), its archives at their sizes.

use std::path::Path;

/// The supported release: The Last Guardian, US (CUSA03627), version 01.00
/// with update 01.03.
pub const TITLE_ID: &str = "CUSA03627";
pub const BASE_VERSION: &str = "01.00";
pub const UPDATE_VERSION: &str = "01.03";

const MODULES: [&str; 3] = ["eboot.bin", "sce_module/libc.prx", "sce_module/libSceFios2.prx"];

/// The base application's archives and their sizes.
const BASE_ARCHIVES: [(&str, u64); 8] = [
    ("trico_001.psarc", 6_216_761_600),
    ("trico_002.psarc", 5_743_801_247),
    ("trico_003.psarc", 7_238_353_034),
    ("trico_boot.psarc", 2_903_783_755),
    ("trico_enu.psarc", 296_510_720),
    ("trico_esl.psarc", 286_613_248),
    ("trico_frc.psarc", 288_351_232),
    ("trico_ptb.psarc", 288_843_008),
];

/// The update's archives and their sizes.
const UPDATE_ARCHIVES: [(&str, u64); 8] = [
    ("trico_001p.psarc", 270_074_112),
    ("trico_002p.psarc", 301_924_608),
    ("trico_003p.psarc", 528_810_240),
    ("trico_bootp.psarc", 547_745_304),
    ("trico_enup.psarc", 485_213),
    ("trico_eslp.psarc", 275_172),
    ("trico_frcp.psarc", 275_172),
    ("trico_ptbp.psarc", 275_172),
];

/// What a folder holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Contents {
    /// Version 01.00 alone.
    Base,
    /// Update 01.03 alone (needs the base folder too).
    Update,
    /// Update 01.03 merged over 01.00 in one folder.
    Merged,
    /// Another title, region or version (its description).
    Other(String),
    /// No game files.
    Empty,
}

#[derive(Clone, Debug)]
pub struct Report {
    pub contents: Contents,
    /// Problems found (empty: usable for its contents).
    pub problems: Vec<String>,
}

impl Report {
    pub fn ok(&self) -> bool {
        self.problems.is_empty() && matches!(self.contents, Contents::Base | Contents::Update | Contents::Merged)
    }
}

/// The string values of a param.sfo.
fn sfo_strings(b: &[u8]) -> Option<Vec<(String, String)>> {
    let u16_at = |o: usize| b.get(o..o + 2).map(|x| u16::from_le_bytes([x[0], x[1]]) as usize);
    let u32_at = |o: usize| b.get(o..o + 4).map(|x| u32::from_le_bytes([x[0], x[1], x[2], x[3]]) as usize);
    if b.get(0..4)? != b"\0PSF" {
        return None;
    }
    let (keys, data, n) = (u32_at(8)?, u32_at(12)?, u32_at(16)?);
    let mut out = Vec::new();
    for i in 0..n.min(256) {
        let e = 20 + i * 16;
        let (key_off, fmt, len, data_off) = (u16_at(e)?, u16_at(e + 2)?, u32_at(e + 4)?, u32_at(e + 12)?);
        let k = b.get(keys + key_off..)?;
        let key = String::from_utf8_lossy(&k[..k.iter().position(|&c| c == 0)?]).into_owned();
        if fmt == 0x0204 || fmt == 0x0004 {
            let v = b.get(data + data_off..data + data_off + len)?;
            let v = &v[..v.iter().position(|&c| c == 0).unwrap_or(v.len())];
            out.push((key, String::from_utf8_lossy(v).into_owned()));
        }
    }
    Some(out)
}

/// Whether a module is decrypted: a SELF container or a plain ELF.
fn executable(p: &Path) -> std::io::Result<bool> {
    use std::io::Read;
    let mut m = [0u8; 4];
    std::fs::File::open(p)?.read_exact(&mut m)?;
    Ok(m == [0x4F, 0x15, 0x3D, 0x1D] || m == [0x7F, b'E', b'L', b'F'])
}

/// Checks a folder: its title and version, then every file the game needs.
pub fn check(dir: &Path) -> Report {
    let mut r = Report { contents: Contents::Empty, problems: Vec::new() };
    let sfo = match std::fs::read(dir.join("sce_sys").join("param.sfo")) {
        Ok(b) => b,
        Err(_) if !dir.join("eboot.bin").exists() => return r,
        Err(_) => {
            r.contents = Contents::Other("no sce_sys\\param.sfo".into());
            r.problems.push("sce_sys\\param.sfo is missing".into());
            return r;
        }
    };
    let Some(values) = sfo_strings(&sfo) else {
        r.contents = Contents::Other("unreadable param.sfo".into());
        r.problems.push("sce_sys\\param.sfo is not readable".into());
        return r;
    };
    let get = |k: &str| values.iter().find(|(key, _)| key == k).map(|(_, v)| v.as_str()).unwrap_or("");
    let (title, version, category) = (get("TITLE_ID"), get("APP_VER"), get("CATEGORY"));
    r.contents = match (title, version, category) {
        (TITLE_ID, BASE_VERSION, "gd") => Contents::Base,
        (TITLE_ID, UPDATE_VERSION, "gp") if dir.join(BASE_ARCHIVES[0].0).exists() => Contents::Merged,
        (TITLE_ID, UPDATE_VERSION, "gp") => Contents::Update,
        (TITLE_ID, v, _) => Contents::Other(format!("version {v}")),
        (t, v, _) => Contents::Other(format!("{t} {v}")),
    };
    let archives: Vec<(&str, u64)> = match r.contents {
        Contents::Base => BASE_ARCHIVES.to_vec(),
        Contents::Update => UPDATE_ARCHIVES.to_vec(),
        Contents::Merged => BASE_ARCHIVES.iter().chain(UPDATE_ARCHIVES.iter()).copied().collect(),
        _ => {
            r.problems.push(format!("not The Last Guardian {BASE_VERSION} or update {UPDATE_VERSION} ({TITLE_ID})"));
            return r;
        }
    };
    for m in MODULES {
        let p = dir.join(m);
        match executable(&p) {
            Ok(true) => {}
            Ok(false) => r.problems.push(format!("{m} is not a decrypted executable")),
            Err(_) => r.problems.push(format!("{m} is missing")),
        }
    }
    for (name, size) in archives {
        match std::fs::metadata(dir.join(name)) {
            Ok(m) if m.len() == size => {}
            Ok(m) => r.problems.push(format!("{name}: {} bytes, expected {size}", m.len())),
            Err(_) => r.problems.push(format!("{name} is missing")),
        }
    }
    r
}

/// Checks the game folder and the update folder (None: 01.03 merged into
/// the game folder). Ok: what to show; Err: the problems.
pub fn check_pair(game: &Path, update: Option<&Path>) -> Result<String, String> {
    let g = check(game);
    let mut problems: Vec<String> = g.problems.iter().map(|p| format!("Game folder: {p}")).collect();
    match (&g.contents, update) {
        (Contents::Empty, _) => problems.insert(0, "Game folder: no game files".into()),
        (Contents::Merged, None) | (Contents::Base, Some(_)) => {}
        (Contents::Base, None) => problems.push(format!("Update {UPDATE_VERSION}: choose its folder")),
        (Contents::Update, _) => problems.push(format!("Game folder: holds update {UPDATE_VERSION}; choose version {BASE_VERSION} here")),
        (Contents::Merged, Some(_)) => problems.push("Update folder: leave it empty (the game folder already holds the update)".into()),
        (Contents::Other(_), _) => {}
    }
    if let (Contents::Base, Some(u)) = (&g.contents, update) {
        let ur = check(u);
        problems.extend(ur.problems.iter().map(|p| format!("Update folder: {p}")));
        match ur.contents {
            Contents::Update => {}
            Contents::Empty => problems.push("Update folder: no game files".into()),
            Contents::Base | Contents::Merged => problems.push(format!("Update folder: holds the game; choose update {UPDATE_VERSION} here")),
            Contents::Other(_) => {}
        }
    }
    if problems.is_empty() {
        Ok(format!("The Last Guardian {BASE_VERSION} with update {UPDATE_VERSION}"))
    } else {
        Err(problems.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A param.sfo with the given string values.
    fn sfo(values: &[(&str, &str)]) -> Vec<u8> {
        let mut keys = Vec::new();
        let mut data = Vec::new();
        let mut entries = Vec::new();
        for (k, v) in values {
            let (ko, d_o) = (keys.len(), data.len());
            keys.extend_from_slice(k.as_bytes());
            keys.push(0);
            let mut val = v.as_bytes().to_vec();
            val.push(0);
            let len = val.len();
            val.resize(len.div_ceil(4) * 4, 0);
            data.extend_from_slice(&val);
            entries.extend_from_slice(&(ko as u16).to_le_bytes());
            entries.extend_from_slice(&0x0204u16.to_le_bytes());
            entries.extend_from_slice(&(len as u32).to_le_bytes());
            entries.extend_from_slice(&(val.len() as u32).to_le_bytes());
            entries.extend_from_slice(&(d_o as u32).to_le_bytes());
        }
        let key_start = 20 + entries.len();
        let data_start = (key_start + keys.len()).div_ceil(4) * 4;
        let mut b = b"\0PSF".to_vec();
        b.extend_from_slice(&0x101u32.to_le_bytes());
        b.extend_from_slice(&(key_start as u32).to_le_bytes());
        b.extend_from_slice(&(data_start as u32).to_le_bytes());
        b.extend_from_slice(&(values.len() as u32).to_le_bytes());
        b.extend_from_slice(&entries);
        b.extend_from_slice(&keys);
        b.resize(data_start, 0);
        b.extend_from_slice(&data);
        b
    }

    #[test]
    fn folders_by_their_param_sfo() {
        let tmp = std::env::temp_dir().join(format!("tlg-launcher-files-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("sce_sys")).unwrap();
        assert_eq!(check(&tmp).contents, Contents::Empty);
        std::fs::write(tmp.join("sce_sys").join("param.sfo"), sfo(&[("APP_VER", "01.03"), ("CATEGORY", "gp"), ("TITLE_ID", "CUSA03627")])).unwrap();
        let r = check(&tmp);
        assert_eq!(r.contents, Contents::Update);
        // Every module and archive reported missing, nothing else.
        assert_eq!(r.problems.len(), MODULES.len() + UPDATE_ARCHIVES.len(), "{:?}", r.problems);
        std::fs::write(tmp.join("eboot.bin"), [0x7F, b'E', b'L', b'F', 2, 1]).unwrap();
        let r = check(&tmp);
        assert!(!r.problems.iter().any(|p| p.starts_with("eboot.bin")), "{:?}", r.problems);
        std::fs::write(tmp.join("eboot.bin"), b"PKG encrypted").unwrap();
        assert!(check(&tmp).problems.iter().any(|p| p == "eboot.bin is not a decrypted executable"));
        std::fs::write(tmp.join("sce_sys").join("param.sfo"), sfo(&[("APP_VER", "01.00"), ("CATEGORY", "gd"), ("TITLE_ID", "CUSA00000")])).unwrap();
        assert!(matches!(check(&tmp).contents, Contents::Other(_)));
        assert!(check_pair(&tmp, None).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
