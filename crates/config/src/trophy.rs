//! The trophy set's encrypted names and the player's PS4 trophy key that
//! opens them, shared by the runtime (the Achievements page and its Trophy
//! Names button) and the launcher (its Trophy key field).
//!
//! trophy00.trp (sce_sys/trophy) holds the set's TROPnnn.PNG icons, plain,
//! and its XML - names, descriptions, grades - encrypted as TROP*.ESFM and
//! TROPCONF.ESFM: a 16-byte IV, then AES-128-CBC data. Their key is the NP
//! communication ID (NPWRnnnnn_00, zero padded to 16 bytes) encrypted with
//! the PS4 trophy key (AES-128-CBC, IV zero), as psdevwiki's Trophy00.trp
//! page describes. The trophy key is the player's own and never shipped: 32
//! hex digits in trophy-key.txt in the user folder.

use aes::cipher::{BlockDecrypt, BlockEncrypt, KeyInit, generic_array::GenericArray};
use std::path::{Path, PathBuf};

/// The file the player's trophy key is kept in.
pub fn key_path(user: &Path) -> PathBuf {
    user.join("trophy-key.txt")
}

/// The key in the user folder's file, when there is one.
pub fn read_key(user: &Path) -> Option<[u8; 16]> {
    parse_key(&std::fs::read_to_string(key_path(user)).ok()?)
}

/// Keeps the key in the user folder's file (32 hex digits).
pub fn write_key(user: &Path, key: &[u8; 16]) -> std::io::Result<()> {
    std::fs::create_dir_all(user)?;
    std::fs::write(key_path(user), format!("{}\r\n", key_text(key)))
}

/// The key as 32 upper-case hex digits.
pub fn key_text(key: &[u8; 16]) -> String {
    key.iter().map(|b| format!("{b:02X}")).collect()
}

/// 16 bytes from text holding 32 hex digits - in one word, or in groups
/// split by spaces, dashes or colons ("Trophy key: 21F41A6B AD8A..."
/// works) - or from a C# byte array (`0x21, 0xF4, ...`).
pub fn parse_key(t: &str) -> Option<[u8; 16]> {
    if t.contains("0x") || t.contains("0X") {
        let bytes: Vec<u8> = t.split(|c: char| !c.is_ascii_alphanumeric()).filter_map(|w| w.strip_prefix("0x").or_else(|| w.strip_prefix("0X"))).filter_map(|h| u8::from_str_radix(h, 16).ok()).collect();
        return bytes.try_into().ok();
    }
    let hex = |s: &str| !s.is_empty() && s.bytes().all(|c| c.is_ascii_hexdigit());
    let words: Vec<&str> = t.split(|c: char| c.is_whitespace() || matches!(c, '-' | ':' | ',' | ';' | '"' | '\'')).filter(|w| !w.is_empty()).collect();
    // (One word of 32 digits; else a run of hex words that makes 32.)
    let found = words.iter().find(|w| w.len() == 32 && hex(w)).map(|w| w.to_string()).or_else(|| {
        let mut runs: Vec<String> = vec![String::new()];
        for w in &words {
            if hex(w) {
                runs.last_mut().unwrap().push_str(w);
            } else {
                runs.push(String::new());
            }
        }
        runs.into_iter().find(|r| r.len() == 32)
    })?;
    let bytes: Vec<u8> = found.as_bytes().chunks_exact(2).filter_map(|p| u8::from_str_radix(std::str::from_utf8(p).ok()?, 16).ok()).collect();
    bytes.try_into().ok()
}

/// trophy00.trp's entry table (magic dca24d00, count @0x10, 0x40-byte
/// entries @0x60: name[32], offset u64, size u64, flags).
pub fn trp_entries(b: &[u8]) -> Vec<(String, Vec<u8>)> {
    if b.len() < 0x60 || b[0..4] != [0xdc, 0xa2, 0x4d, 0x00] {
        return Vec::new();
    }
    let n = u32::from_be_bytes(b[0x10..0x14].try_into().unwrap()) as usize;
    let esz = (u32::from_be_bytes(b[0x14..0x18].try_into().unwrap()) as usize).max(0x40);
    let mut out = Vec::new();
    for i in 0..n.min(1024) {
        let Some(e) = b.get(0x60 + i * esz..0x60 + i * esz + 0x34) else { break };
        let name = String::from_utf8_lossy(e[..32].split(|&c| c == 0).next().unwrap_or(&[])).to_string();
        let off = u64::from_be_bytes(e[32..40].try_into().unwrap()) as usize;
        let size = u64::from_be_bytes(e[40..48].try_into().unwrap()) as usize;
        if let Some(d) = b.get(off..off.saturating_add(size)) {
            out.push((name, d.to_vec()));
        }
    }
    out
}

/// The ESFM key for an NP communication ID.
pub fn esfm_key(trophy_key: &[u8; 16], comm_id: &str) -> [u8; 16] {
    let mut block = [0u8; 16];
    let id = comm_id.as_bytes();
    block[..id.len().min(16)].copy_from_slice(&id[..id.len().min(16)]);
    // (CBC with a zero IV over one block: the block encrypted.)
    let mut b = GenericArray::clone_from_slice(&block);
    aes::Aes128::new(GenericArray::from_slice(trophy_key)).encrypt_block(&mut b);
    b.into()
}

/// The XML's start, as the set's files begin (the signature comment) or
/// plain XML would.
pub fn looks_like_xml(p: &[u8]) -> bool {
    p.starts_with(b"<!--Sce-Np-Troph") || p.starts_with(b"<?xml") || p.starts_with(b"<trophyconf")
}

/// An ESFM's first block decrypted.
pub fn esfm_first(key: &[u8; 16], esfm: &[u8]) -> Option<[u8; 16]> {
    let (iv, c) = (esfm.get(..16)?, esfm.get(16..32)?);
    let mut b = GenericArray::clone_from_slice(c);
    aes::Aes128::new(GenericArray::from_slice(key)).decrypt_block(&mut b);
    let mut out = [0u8; 16];
    for i in 0..16 {
        out[i] = b[i] ^ iv[i];
    }
    Some(out)
}

/// An ESFM decrypted (the XML, its padding cut after the last '>'); None
/// when the key does not fit.
pub fn esfm_decrypt(key: &[u8; 16], esfm: &[u8]) -> Option<Vec<u8>> {
    if esfm.len() < 32 || (esfm.len() - 16) % 16 != 0 {
        return None;
    }
    let aes = aes::Aes128::new(GenericArray::from_slice(key));
    let mut prev: [u8; 16] = esfm[..16].try_into().ok()?;
    let mut out = Vec::with_capacity(esfm.len() - 16);
    for c in esfm[16..].chunks_exact(16) {
        let mut b = GenericArray::clone_from_slice(c);
        aes.decrypt_block(&mut b);
        for i in 0..16 {
            b[i] ^= prev[i];
        }
        prev = c.try_into().ok()?;
        out.extend_from_slice(&b);
    }
    if !looks_like_xml(&out) {
        return None;
    }
    let end = out.iter().rposition(|&c| c == b'>').map_or(out.len(), |i| i + 1);
    out.truncate(end);
    Some(out)
}

/// The first NPWRnnnnn_nn in a file's bytes (npbind.dat).
pub fn comm_id_in(d: &[u8]) -> Option<String> {
    d.windows(12).find(|w| w.starts_with(b"NPWR") && w[4..9].iter().all(u8::is_ascii_digit) && w[9] == b'_' && w[10..12].iter().all(u8::is_ascii_digit)).map(|w| String::from_utf8_lossy(w).to_string())
}

/// The ESFM key that `trophy_key` gives for the set's files, and the set's
/// NP communication ID: the bound one (npbind.dat's) when it opens
/// TROPCONF.ESFM, else the NPWRnnnnn_00 that does (a search of 100,000
/// single-block decryptions). None: the trophy key opens none of them.
pub fn find_key(entries: &[(String, Vec<u8>)], trophy_key: &[u8; 16], bound: Option<&str>) -> Option<([u8; 16], String)> {
    let conf = &entries.iter().find(|(n, _)| n == "TROPCONF.ESFM").or_else(|| entries.iter().find(|(n, _)| n.ends_with(".ESFM")))?.1;
    let fits = |id: &str| -> Option<[u8; 16]> {
        let k = esfm_key(trophy_key, id);
        esfm_first(&k, conf).filter(|b| looks_like_xml(b)).map(|_| k)
    };
    if let Some((k, id)) = bound.and_then(|id| fits(id).map(|k| (k, id))) {
        return Some((k, id.to_string()));
    }
    (0..100_000u32).map(|n| format!("NPWR{n:05}_00")).find_map(|id| fits(&id).map(|k| (k, id)))
}

/// Checks a trophy key against the trophy set in the game folders (the
/// first that has one; the update folder first): the set's NP communication
/// ID when the key opens it, or why not.
pub fn check_key(dirs: &[&Path], trophy_key: &[u8; 16]) -> Result<String, String> {
    for d in dirs {
        let trp = d.join("sce_sys").join("trophy").join("trophy00.trp");
        let Ok(b) = std::fs::read(&trp) else { continue };
        let entries = trp_entries(&b);
        if entries.is_empty() {
            return Err(format!("{} is not a trophy file", trp.display()));
        }
        let bound = dirs.iter().find_map(|d| std::fs::read(d.join("sce_sys").join("npbind.dat")).ok()).and_then(|n| comm_id_in(&n));
        return match find_key(&entries, trophy_key, bound.as_deref()) {
            Some((_, id)) => Ok(id),
            None => Err("This key does not open the game's trophies.".into()),
        };
    }
    Err("No trophy file in the game folders (sce_sys\\trophy\\trophy00.trp).".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ESFM files made with a made-up trophy key (not the console's): the
    /// documented scheme round trip, and the ID search.
    #[test]
    fn esfm_round_trip() {
        let tk = [7u8; 16];
        let id = "NPWR01234_00";
        let key = esfm_key(&tk, id);
        let xml = b"<!--Sce-Np-Trophy-Signature: 00--><trophyconf version=\"1.1\"><trophy id=\"000\" ttype=\"P\"><name>A</name></trophy></trophyconf>";
        let mut plain = xml.to_vec();
        plain.resize(plain.len().div_ceil(16) * 16, 0);
        let iv = [3u8; 16];
        let aes = aes::Aes128::new(GenericArray::from_slice(&key));
        let mut esfm = iv.to_vec();
        let mut prev = iv;
        for c in plain.chunks_exact(16) {
            let mut b = [0u8; 16];
            for i in 0..16 {
                b[i] = c[i] ^ prev[i];
            }
            let mut g = GenericArray::clone_from_slice(&b);
            aes.encrypt_block(&mut g);
            prev = g.into();
            esfm.extend_from_slice(&prev);
        }
        assert_eq!(esfm_decrypt(&key, &esfm).as_deref(), Some(&xml[..]));
        assert_eq!(esfm_decrypt(&esfm_key(&tk, "NPWR01235_00"), &esfm), None);
        assert!(looks_like_xml(&esfm_first(&key, &esfm).unwrap()));
        assert_eq!(comm_id_in(b"\x00\x01junkNPWR01234_00\x00more"), Some(id.to_string()));
        let entries = vec![("TROPCONF.ESFM".to_string(), esfm.clone())];
        assert_eq!(find_key(&entries, &tk, None).map(|(_, i)| i).as_deref(), Some(id));
        assert_eq!(find_key(&entries, &tk, Some("NPWR01234_00")).map(|(k, _)| k), Some(key));
        assert_eq!(find_key(&entries, &[8u8; 16], Some(id)), None);
    }

    #[test]
    fn keys_parse() {
        let tk = [7u8; 16];
        assert_eq!(parse_key("07070707070707070707070707070707\n"), Some(tk));
        assert_eq!(parse_key(" 0707 0707-0707 0707 0707 0707 0707 0707 "), Some(tk));
        assert_eq!(parse_key("Trophy key: 07070707 07070707 07070707 07070707\r\n"), Some(tk));
        assert_eq!(parse_key("my key is \"07070707070707070707070707070707\", see"), Some(tk));
        assert_eq!(parse_key("new byte[16] { 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07 };"), Some(tk));
        assert_eq!(parse_key("0707"), None);
        assert_eq!(parse_key("070707070707070707070707070707070707"), None);
        assert_eq!(key_text(&tk), "07070707070707070707070707070707");
        assert_eq!(parse_key(&key_text(&tk)), Some(tk));
    }
}
