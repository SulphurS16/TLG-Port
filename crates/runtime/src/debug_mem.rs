//! Debug access to guest memory from the control channel (`peek`, `poke`):
//! reading engine state and changing it while the title runs.
//!
//! Addresses: hex numbers (`0x10a0` or `10a0`), `main` (the main module's
//! base) plus hex terms (`main+55404f8`), and `[expr]` for the 64-bit value
//! stored at expr (`[main+55404f8]+8`).


fn parse_term(t: &str) -> Result<u64, String> {
    let t = t.trim();
    if t == "main" {
        return crate::loader::main_module().map(|m| m.base).ok_or_else(|| "no main module".to_string());
    }
    u64::from_str_radix(t.trim_start_matches("0x"), 16).map_err(|_| format!("bad address term '{t}'"))
}

/// Evaluates an address expression (see the module docs).
pub fn parse_addr(s: &str) -> Result<u64, String> {
    let s = s.trim();
    let (mut v, rest) = if let Some(inner) = s.strip_prefix('[') {
        let end = inner.rfind(']').ok_or("unbalanced [")?;
        let p = parse_addr(&inner[..end])?;
        if !readable(p, 8) {
            return Err(format!("{p:#x} is not mapped"));
        }
        (unsafe { std::ptr::read_unaligned(p as *const u64) }, &inner[end + 1..])
    } else {
        let mut it = s.splitn(2, '+');
        let first = it.next().unwrap_or("");
        (parse_term(first)?, it.next().unwrap_or(""))
    };
    for t in rest.split('+').filter(|t| !t.trim().is_empty()) {
        v = v.wrapping_add(parse_term(t)?);
    }
    Ok(v)
}

fn readable(addr: u64, len: u64) -> bool {
    if !crate::mem::is_guest(addr, len) {
        return false;
    }
    let st = crate::mem::MEM.lock();
    st.find(addr).is_some_and(|v| v.kind != crate::mem::VmaKind::Reserved && addr + len <= v.end())
}

pub fn peek(expr: &str, n: u64) -> Result<(), String> {
    let a = parse_addr(expr)?;
    if !readable(a, n.max(4)) {
        return Err(format!("peek {a:#x}+{n:#x}: not mapped"));
    }
    let mut out = format!("peek {expr} = {a:#x} ({}):", crate::loader::describe_addr(a));
    for o in (0..n).step_by(16) {
        out.push_str(&format!("\n  +{o:04x}"));
        for k in (o..(o + 16).min(n)).step_by(4) {
            let w = unsafe { std::ptr::read_unaligned((a + k) as *const u32) };
            out.push_str(&format!(" {w:08x}"));
        }
        out.push_str("  |");
        for k in (o..(o + 16).min(n)).step_by(4) {
            let w = unsafe { std::ptr::read_unaligned((a + k) as *const u32) };
            let f = f32::from_bits(w);
            if f.is_finite() && (f == 0.0 || (1e-6..1e7).contains(&f.abs())) {
                out.push_str(&format!(" {f}"));
            } else {
                out.push_str(" -");
            }
        }
    }
    crate::info!("{out}");
    Ok(())
}

pub fn poke(expr: &str, words: &[String]) -> Result<(), String> {
    let a = parse_addr(expr)?;
    let vals: Vec<u32> = words.iter().map(|w| u32::from_str_radix(w.trim_start_matches("0x"), 16).map_err(|_| format!("bad word '{w}'"))).collect::<Result<_, _>>()?;
    let len = vals.len() as u64 * 4;
    if vals.is_empty() || !readable(a, len) {
        return Err(format!("poke {a:#x}: nothing to write or not mapped"));
    }
    write_guest(a, &vals.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
    crate::info!("poke {expr} = {a:#x} ({}): {:x?}", crate::loader::describe_addr(a), vals);
    Ok(())
}

/// Writes guest memory, code pages included (made writable for the write;
/// the instruction cache is flushed afterwards).
pub fn write_guest(a: u64, bytes: &[u8]) -> Result<(), String> {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let len = bytes.len();
    let mut old = 0u32;
    unsafe {
        if VirtualProtect(a as _, len, PAGE_EXECUTE_READWRITE, &mut old) == 0 {
            return Err(format!("VirtualProtect({a:#x}) failed"));
        }
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), a as *mut u8, len);
        let mut tmp = 0u32;
        VirtualProtect(a as _, len, old, &mut tmp);
        FlushInstructionCache(GetCurrentProcess(), a as _, len);
    }
    Ok(())
}
