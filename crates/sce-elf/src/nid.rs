//! NID ("name ID") hashing and the 64-character ID encoding used in SCE symbol names.
//!
//! A NID is the first 8 bytes of SHA-1(name || salt), read as a little-endian u64,
//! written big-endian and base64-encoded with the alphabet below (11 characters,
//! no padding). Symbol names in SCE modules look like `NID#L#M` where `L` and `M`
//! are library/module ids in the same alphabet (variable length).

pub const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";

const SALT: [u8; 16] = [
    0x51, 0x8D, 0x64, 0xA6, 0x35, 0xDE, 0xD8, 0xC1, 0xE6, 0xB0, 0x39, 0xB1, 0xC3, 0xE5, 0x52, 0x30,
];

/// Computes the 11-character NID string for a symbol name.
pub fn nid_for_name(name: &str) -> String {
    let mut msg = Vec::with_capacity(name.len() + SALT.len());
    msg.extend_from_slice(name.as_bytes());
    msg.extend_from_slice(&SALT);
    let digest = sha1(&msg);
    let value = u64::from_le_bytes(digest[0..8].try_into().unwrap());
    encode_nid(value)
}

/// Encodes a 64-bit NID value as 11 characters.
pub fn encode_nid(value: u64) -> String {
    let bytes = value.to_be_bytes();
    // 64 bits -> ten 6-bit groups + one 4-bit group padded with two zero bits.
    let mut bits: u128 = u128::from_be_bytes({
        let mut b = [0u8; 16];
        b[..8].copy_from_slice(&bytes);
        b
    });
    let mut out = String::with_capacity(11);
    for _ in 0..11 {
        let idx = (bits >> 122) as usize & 0x3F;
        out.push(ALPHABET[idx] as char);
        bits <<= 6;
    }
    out
}

/// Decodes a variable-length id (library/module id in `NID#L#M`).
pub fn decode_id(s: &str) -> Option<u64> {
    if s.is_empty() || s.len() > 10 {
        return None;
    }
    let mut v: u64 = 0;
    for c in s.bytes() {
        let d = ALPHABET.iter().position(|&a| a == c)? as u64;
        v = (v << 6) | d;
    }
    Some(v)
}

/// Splits an encoded SCE symbol name into (nid, library id, module id).
pub fn split_symbol(name: &str) -> Option<(&str, u64, u64)> {
    let mut it = name.split('#');
    let nid = it.next()?;
    let lib = decode_id(it.next()?)?;
    let module = decode_id(it.next()?)?;
    if it.next().is_some() {
        return None;
    }
    Some((nid, lib, module))
}

/// Minimal SHA-1 (FIPS 180-4), sufficient for NID computation.
pub fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
        for (i, wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | (!b & d), 0x5A827999),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
                _ => (b ^ c ^ d, 0xCA62C1D6),
            };
            let t = a.rotate_left(5).wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(*wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = t;
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
    }
    let mut out = [0u8; 20];
    for i in 0..5 {
        out[i * 4..i * 4 + 4].copy_from_slice(&h[i].to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha1_vectors() {
        let hex = |d: [u8; 20]| d.iter().map(|b| format!("{b:02x}")).collect::<String>();
        assert_eq!(hex(sha1(b"abc")), "a9993e364706816aba3e25717850c26c9cd0d89d");
        assert_eq!(hex(sha1(b"")), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        let long = vec![b'a'; 1000];
        assert_eq!(hex(sha1(&long)), "291e9a6c66994949b57ba5e650361e98fc36b1ba");
    }

    #[test]
    fn id_decoding() {
        assert_eq!(decode_id("A"), Some(0));
        assert_eq!(decode_id("B"), Some(1));
        assert_eq!(decode_id("BA"), Some(64));
        assert_eq!(split_symbol("Ou3iL1abvng#C#D"), Some(("Ou3iL1abvng", 2, 3)));
        assert_eq!(split_symbol("plain"), None);
    }

    #[test]
    fn nid_roundtrip_shape() {
        let n = nid_for_name("sceKernelUsleep");
        assert_eq!(n.len(), 11);
    }
}
