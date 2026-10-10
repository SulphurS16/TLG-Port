//! Module loading, relocation, import resolution and code patching.

use crate::mem::{self, MEM, PAGE, VmaKind};
use crate::patch::{PatchStats, Patcher};
use crate::tls::{self, TlsModule};
use parking_lot::{Mutex, RwLock};
use sce_elf::{DynamicInfo, Module, PF_R, PF_W, PF_X, Phdr, reloc};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

const TRAMPOLINE_SPACE: u64 = 0x10000;

pub struct LoadedModule {
    pub name: String,
    pub file_name: String,
    pub handle: i32,
    pub base: u64,
    pub image_size: u64,
    pub size: u64,
    pub entry: u64,
    pub init: Option<u64>,
    pub fini: Option<u64>,
    pub is_main: bool,
    pub dynamic: DynamicInfo,
    pub phdrs: Vec<Phdr>,
    pub tls: Option<TlsModule>,
    pub eh_frame_hdr: Option<(u64, u64)>,
    pub eh_frame: Option<(u64, u64)>,
    pub proc_param: Option<u64>,
    pub exports: HashMap<(String, String), u64>,
    pub code: (u64, u64),
    pub fingerprint: [u8; 20],
}

impl LoadedModule {
    pub fn contains(&self, addr: u64) -> bool {
        addr >= self.base && addr < self.base + self.size
    }
}

pub static MODULES: RwLock<Vec<Arc<LoadedModule>>> = RwLock::new(Vec::new());

pub fn module_at(addr: u64) -> Option<Arc<LoadedModule>> {
    MODULES.read().iter().find(|m| m.contains(addr)).cloned()
}

pub fn main_module() -> Option<Arc<LoadedModule>> {
    MODULES.read().iter().find(|m| m.is_main).cloned()
}

/// Human-readable location of an address (guest module offset, stub, or host).
pub fn describe_addr(addr: u64) -> String {
    if let Some(m) = module_at(addr) {
        return format!("{}+{:#x}", m.name, addr - m.base);
    }
    if let Some(s) = STUBS.lock().describe(addr) {
        return s;
    }
    let host = crate::exception::host_image();
    if addr >= host.0 && addr < host.1 {
        return format!("tlg+{:#x}", addr - host.0);
    }
    format!("{addr:#x}")
}

fn prot_from_flags(flags: u32) -> u32 {
    let mut p = 0;
    if flags & PF_R != 0 {
        p |= mem::PROT_CPU_READ;
    }
    if flags & PF_W != 0 {
        p |= mem::PROT_CPU_WRITE;
    }
    if flags & PF_X != 0 {
        p |= mem::PROT_CPU_EXEC | mem::PROT_CPU_READ;
    }
    p
}

/// Loads (maps and copies) a module without relocating it.
pub fn load(path: &Path, is_main: bool) -> Result<Arc<LoadedModule>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let m = Module::parse(bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    let dynamic = m.dynamic().map_err(|e| format!("{}: {e}", path.display()))?;
    let image_size = m.image_span(PAGE);
    let size = image_size + TRAMPOLINE_SPACE;
    let base = {
        let mut st = MEM.lock();
        let base = if is_main {
            mem::MODULE_BASE
        } else {
            st.find_free(mem::MODULE_BASE, size, 0x10_0000, mem::MODULE_END).ok_or("no module space")?
        };
        st.map(base, size, VmaKind::Module, mem::PROT_CPU_READ | mem::PROT_CPU_WRITE, 0, 0, &path.display().to_string())
            .map_err(|e| format!("mapping module: {e:?}"))?;
        base
    };
    let mut code = (0, 0);
    let mut tls_mod = None;
    let mut eh_frame_hdr = None;
    let mut proc_param = None;
    for (i, ph) in m.phdrs.iter().enumerate() {
        match ph.p_type {
            sce_elf::PT_LOAD | sce_elf::PT_SCE_RELRO => {
                if let Some(data) = m.segment_data(i) {
                    unsafe {
                        std::ptr::copy_nonoverlapping(data.as_ptr(), (base + ph.p_vaddr) as *mut u8, data.len());
                    }
                }
                if ph.p_flags & PF_X != 0 {
                    code = (base + ph.p_vaddr, base + ph.p_vaddr + ph.p_memsz);
                }
            }
            sce_elf::PT_TLS => {
                tls_mod = Some(tls::register_module(base + ph.p_vaddr, ph.p_filesz, ph.p_memsz, ph.p_align, ph.p_vaddr));
            }
            sce_elf::PT_GNU_EH_FRAME => eh_frame_hdr = Some((base + ph.p_vaddr, ph.p_memsz)),
            sce_elf::PT_SCE_PROCPARAM => proc_param = Some(base + ph.p_vaddr),
            _ => {}
        }
    }
    let name = dynamic
        .module_info
        .as_ref()
        .map(|mi| mi.name.clone())
        .unwrap_or_else(|| path.file_stem().unwrap().to_string_lossy().into_owned());
    let mut exports = HashMap::new();
    for s in &dynamic.symbols {
        if !s.is_defined() || s.bind() == sce_elf::STB_LOCAL {
            continue;
        }
        if let Some((nid, lib, _)) = sce_elf::nid::split_symbol(&s.name) {
            if let Some(l) = dynamic.export_lib_by_id(lib) {
                exports.insert((l.name.clone(), nid.to_string()), base + s.value);
            }
        }
    }
    let eh_frame = eh_frame_hdr.and_then(|(hdr, _)| eh_frame_extent(hdr));
    let mut fingerprint = [0u8; 20];
    if let Some(fp) = &dynamic.fingerprint {
        fingerprint.copy_from_slice(&fp[..20]);
    }
    let lm = Arc::new(LoadedModule {
        handle: MODULES.read().len() as i32 + 1,
        file_name: path.file_name().unwrap().to_string_lossy().into_owned(),
        name,
        base,
        image_size,
        size,
        entry: base + m.ehdr.e_entry,
        init: dynamic.init.map(|v| base + v),
        fini: dynamic.fini.map(|v| base + v),
        is_main,
        phdrs: m.phdrs.clone(),
        tls: tls_mod,
        eh_frame_hdr,
        eh_frame,
        proc_param,
        exports,
        code,
        fingerprint,
        dynamic,
    });
    crate::info!(
        "loaded {} ({}) at {:#x} size {:#x} entry {:#x} exports {} tls {:?}",
        lm.name,
        lm.file_name,
        base,
        size,
        lm.entry,
        lm.exports.len(),
        lm.tls.map(|t| (t.index, t.offset, t.mem_size))
    );
    MODULES.write().push(lm.clone());
    Ok(lm)
}

/// Finds the address and size of `.eh_frame` from the header's pointer.
fn eh_frame_extent(hdr: u64) -> Option<(u64, u64)> {
    unsafe {
        let enc = *((hdr + 1) as *const u8);
        let ptr = match enc {
            0x1B => (hdr + 4).wrapping_add(*((hdr + 4) as *const i32) as i64 as u64),
            0x03 => *((hdr + 4) as *const u32) as u64,
            _ => return None,
        };
        // Walk records until the zero terminator.
        let mut p = ptr;
        for _ in 0..10_000_000u32 {
            let len = *(p as *const u32) as u64;
            if len == 0 {
                return Some((ptr, p + 4 - ptr));
            }
            if len == 0xFFFF_FFFF {
                return Some((ptr, 0));
            }
            p += 4 + len;
        }
        Some((ptr, 0))
    }
}

// ---------------------------------------------------------------------------
// Unresolved import stubs

pub struct Stubs {
    code_base: u64,
    code_cap: u64,
    code_used: u64,
    data_base: u64,
    data_used: u64,
    by_key: HashMap<(String, String, bool), u64>,
    labels: Vec<(u64, u64, String)>,
}

pub static STUBS: std::sync::LazyLock<Mutex<Stubs>> = std::sync::LazyLock::new(|| {
    Mutex::new(Stubs { code_base: 0, code_cap: 0, code_used: 0, data_base: 0, data_used: 0, by_key: HashMap::new(), labels: Vec::new() })
});

const STUB_CODE_SPACE: u64 = 0x10_0000;
const STUB_DATA_SPACE: u64 = 0x10_0000;

impl Stubs {
    fn ensure(&mut self) {
        if self.code_base == 0 {
            let mut st = MEM.lock();
            self.code_base = st
                .alloc_internal(STUB_CODE_SPACE, mem::PROT_CPU_READ | mem::PROT_CPU_WRITE | mem::PROT_CPU_EXEC, "import-stubs")
                .expect("stub space");
            self.code_cap = STUB_CODE_SPACE;
            // Data stubs: inaccessible so any use faults and is reported.
            self.data_base = st.alloc_internal(STUB_DATA_SPACE, 0, "unresolved-data").expect("stub data space");
        }
    }

    fn get(&mut self, nid: &str, lib: &str, importer: &str, is_data: bool) -> u64 {
        self.ensure();
        let key = (lib.to_string(), nid.to_string(), is_data);
        if let Some(&a) = self.by_key.get(&key) {
            return a;
        }
        let label = format!("unresolved {} [{}]", crate::hle::describe_nid(nid), lib);
        let addr = if is_data {
            let a = self.data_base + self.data_used;
            self.data_used += 0x40;
            assert!(self.data_used <= STUB_DATA_SPACE);
            self.labels.push((a, 0x40, label));
            a
        } else {
            let mut list = crate::hle::UNRESOLVED.write();
            let index = list.len() as u64;
            list.push(crate::hle::Unresolved {
                nid: nid.to_string(),
                lib: lib.to_string(),
                importer: importer.to_string(),
                is_data,
            });
            let a = self.code_base + self.code_used;
            self.code_used += 32;
            assert!(self.code_used <= self.code_cap);
            let code = crate::hle::unresolved_stub_code(index);
            unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), a as *mut u8, code.len()) };
            self.labels.push((a, 32, label));
            a
        };
        self.by_key.insert(key, addr);
        addr
    }

    pub fn describe(&self, addr: u64) -> Option<String> {
        self.labels.iter().find(|(a, n, _)| addr >= *a && addr < *a + *n).map(|(_, _, l)| l.clone())
    }
}

// ---------------------------------------------------------------------------
// Linking

#[derive(Default, Debug, Clone)]
pub struct LinkStats {
    pub lle: usize,
    pub hle: usize,
    pub unresolved: usize,
    pub weak_null: usize,
}

fn resolve_import(m: &LoadedModule, sym: &sce_elf::Symbol, stats: &mut LinkStats) -> u64 {
    let Some((nid, lib_id, _mod_id)) = sce_elf::nid::split_symbol(&sym.name) else {
        crate::warn!("{}: non-NID import symbol '{}'", m.name, sym.name);
        stats.unresolved += 1;
        return STUBS.lock().get(&sym.name, "?", &m.name, sym.sym_type() == sce_elf::STT_OBJECT);
    };
    let lib = m.dynamic.import_lib_by_id(lib_id).map(|l| l.name.as_str()).unwrap_or("?");
    // 1. LLE exports
    {
        let mods = MODULES.read();
        for other in mods.iter() {
            if let Some(&a) = other.exports.get(&(lib.to_string(), nid.to_string())) {
                stats.lle += 1;
                return a;
            }
        }
    }
    // 2. HLE
    if let Some(e) = crate::hle::registry().lookup(lib, nid) {
        stats.hle += 1;
        return e.addr;
    }
    // 3. libSceLibcInternal is served by the title's libc.prx (same NIDs).
    if lib == "libSceLibcInternal" {
        let mods = MODULES.read();
        for other in mods.iter() {
            if let Some(&a) = other.exports.get(&("libc".to_string(), nid.to_string())) {
                stats.lle += 1;
                return a;
            }
        }
    }
    if sym.bind() == sce_elf::STB_WEAK {
        stats.weak_null += 1;
        crate::debug!("{}: weak import {} [{}] unresolved -> 0", m.name, crate::hle::describe_nid(nid), lib);
        return 0;
    }
    stats.unresolved += 1;
    let is_data = sym.sym_type() == sce_elf::STT_OBJECT;
    crate::debug!("{}: unresolved {} [{}] data={}", m.name, crate::hle::describe_nid(nid), lib, is_data);
    STUBS.lock().get(nid, lib, &m.name, is_data)
}

/// Applies all relocations of a loaded module.
pub fn relocate(m: &LoadedModule) -> Result<LinkStats, String> {
    let mut stats = LinkStats::default();
    let mut cache: HashMap<usize, u64> = HashMap::new();
    let own_tls = m.tls.map(|t| t.index).unwrap_or(0);
    for r in m.dynamic.relas.iter().chain(m.dynamic.plt_relas.iter()) {
        let target = m.base + r.offset;
        if r.offset >= m.image_size {
            return Err(format!("{}: relocation offset {:#x} outside image", m.name, r.offset));
        }
        let symval = |cache: &mut HashMap<usize, u64>, stats: &mut LinkStats| -> Result<u64, String> {
            let idx = r.sym();
            if let Some(v) = cache.get(&idx) {
                return Ok(*v);
            }
            let sym = m.dynamic.symbols.get(idx).ok_or_else(|| format!("{}: bad symbol index {idx}", m.name))?;
            let v = if sym.is_defined() { m.base + sym.value } else { resolve_import(m, sym, stats) };
            cache.insert(idx, v);
            Ok(v)
        };
        let value: u64 = match r.r_type() {
            reloc::NONE => continue,
            reloc::RELATIVE => m.base.wrapping_add(r.addend as u64),
            reloc::R64 => symval(&mut cache, &mut stats)?.wrapping_add(r.addend as u64),
            reloc::GLOB_DAT | reloc::JUMP_SLOT => symval(&mut cache, &mut stats)?,
            reloc::DTPMOD64 => {
                if r.sym() != 0 {
                    let sym = &m.dynamic.symbols[r.sym()];
                    let local = sym.bind() == sce_elf::STB_LOCAL || sym.name.is_empty();
                    if !sym.is_defined() && !local {
                        return Err(format!("{}: imported TLS module symbol {} unsupported", m.name, sym.name));
                    }
                }
                own_tls
            }
            reloc::DTPOFF64 => {
                let sv = if r.sym() != 0 { m.dynamic.symbols[r.sym()].value } else { 0 };
                sv.wrapping_add(r.addend as u64)
            }
            reloc::TPOFF64 => {
                let t = m.tls.ok_or_else(|| format!("{}: TPOFF64 without TLS", m.name))?;
                if t.offset == 0 {
                    return Err(format!("{}: TPOFF64 for dynamically allocated TLS", m.name));
                }
                let sv = if r.sym() != 0 { m.dynamic.symbols[r.sym()].value } else { 0 };
                sv.wrapping_add(r.addend as u64).wrapping_sub(t.offset)
            }
            t => return Err(format!("{}: unsupported relocation type {t}", m.name)),
        };
        unsafe { std::ptr::write_unaligned(target as *mut u64, value) };
    }
    Ok(stats)
}

/// Patches TLS/syscall sites, then applies final protections.
pub fn finalize(m: &LoadedModule) -> Result<PatchStats, String> {
    let mut stats = PatchStats::default();
    let image = unsafe { std::slice::from_raw_parts_mut(m.base as *mut u8, m.size as usize) };
    let (img, tramp) = image.split_at_mut(m.image_size as usize);
    let ranges = match m.eh_frame_hdr {
        Some((hdr, _)) => {
            let view = crate::ehframe::Image { base: m.base, bytes: unsafe { std::slice::from_raw_parts(m.base as *const u8, m.image_size as usize) } };
            crate::ehframe::function_ranges(&view, hdr - m.base).map_err(|e| format!("{}: eh_frame: {e}", m.name))?
        }
        None => Vec::new(),
    };
    let mut p = Patcher {
        image: img,
        base: m.base,
        tls_slot_disp: crate::patch::teb_tls_slot_offset(tls::slot()),
        tramp_base: m.base + m.image_size,
        tramp,
        tramp_used: 0,
        data_used: 0,
    };
    p.patch_functions(&ranges, &mut stats);
    if m.code.1 > m.code.0 {
        p.patch_raw_fs_zero(m.code.1 - m.base, &mut stats);
    }
    if m.is_main {
        crate::title_patches::apply(&mut p, &m.name);
    }
    let (tramp_used, data_used) = (p.tramp_used, p.data_used);
    // Protections.
    let mut st = MEM.lock();
    for ph in &m.phdrs {
        let prot = match ph.p_type {
            sce_elf::PT_LOAD => prot_from_flags(ph.p_flags),
            sce_elf::PT_SCE_RELRO => mem::PROT_CPU_READ,
            _ => continue,
        };
        let start = m.base + mem::align_down(ph.p_vaddr, PAGE);
        let end = m.base + mem::align_up(ph.p_vaddr + ph.p_memsz, PAGE);
        st.protect(start, end, prot).map_err(|e| format!("{}: protect: {e:?}", m.name))?;
    }
    let data_page = m.base + m.size - crate::patch::DATA_SPACE as u64;
    if tramp_used > 0 {
        st.protect(m.base + m.image_size, data_page, mem::PROT_CPU_READ | mem::PROT_CPU_EXEC)
            .map_err(|e| format!("{}: protect trampolines: {e:?}", m.name))?;
    }
    if data_used > 0 {
        st.protect(data_page, m.base + m.size, mem::PROT_CPU_READ | mem::PROT_CPU_WRITE)
            .map_err(|e| format!("{}: protect patch data: {e:?}", m.name))?;
    }
    drop(st);
    unsafe {
        windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache(
            windows_sys::Win32::System::Threading::GetCurrentProcess(),
            m.base as *const _,
            m.size as usize,
        );
    }
    Ok(stats)
}

pub fn report_patch(m: &LoadedModule, s: &PatchStats) {
    crate::info!(
        "{}: patched {} functions: fs:[0] {} (+{} raw), fs:[disp] {}, unknown fs {}, syscall {}",
        m.name,
        s.functions,
        s.fs_zero,
        s.fs_raw_extra,
        s.fs_disp,
        s.fs_unknown.len(),
        s.syscalls.len()
    );
    for (rel, text, range) in s.fs_unknown.iter().take(8) {
        crate::info!("{}+{:#x}: unhandled FS-prefixed decode '{}' in function {:#x}..{:#x} (left unpatched)", m.name, rel, text, range.0, range.1);
    }
    if std::env::var_os("TLG_DUMP_FS").is_some() {
        for (rel, _, range) in s.fs_unknown.iter().take(4) {
            crate::patch::dump_around(m.base, *range, *rel);
        }
    }
    for rel in s.syscalls.iter().take(20) {
        crate::info!("{}+{:#x}: syscall replaced by ud2", m.name, rel);
    }
    if s.sse4a_patched + s.sse4a_trapped.len() > 0 {
        crate::info!(
            "{}: SSE4a (not implemented by this CPU): {} sites emulated by trampolines, {} by trap emulation",
            m.name,
            s.sse4a_patched,
            s.sse4a_trapped.len()
        );
        for (rel, text) in &s.sse4a_trapped {
            crate::info!("{}+{:#x}: '{}' left for trap emulation", m.name, rel, text);
        }
    }
}
