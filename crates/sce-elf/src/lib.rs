//! PS4 module container parsing: SELF (fake/decrypted) and raw ELF64, program
//! headers, SCE dynamic-link metadata, symbols and relocations.
//!
//! Only unencrypted, uncompressed SELF payloads are supported; anything else is
//! reported as an error rather than guessed.

pub mod nid;

use std::fmt;

pub const SELF_MAGIC: u32 = 0x1D3D154F;
pub const ELF_MAGIC: [u8; 4] = [0x7F, b'E', b'L', b'F'];

pub const ET_SCE_EXEC: u16 = 0xFE00;
pub const ET_SCE_REPLAY_EXEC: u16 = 0xFE01;
pub const ET_SCE_RELEXEC: u16 = 0xFE04;
pub const ET_SCE_STUBLIB: u16 = 0xFE0C;
pub const ET_SCE_DYNEXEC: u16 = 0xFE10;
pub const ET_SCE_DYNAMIC: u16 = 0xFE18;

pub const PT_LOAD: u32 = 1;
pub const PT_DYNAMIC: u32 = 2;
pub const PT_INTERP: u32 = 3;
pub const PT_TLS: u32 = 7;
pub const PT_GNU_EH_FRAME: u32 = 0x6474_E550;
pub const PT_SCE_RELA: u32 = 0x6000_0000;
pub const PT_SCE_DYNLIBDATA: u32 = 0x6100_0000;
pub const PT_SCE_PROCPARAM: u32 = 0x6100_0001;
pub const PT_SCE_MODULE_PARAM: u32 = 0x6100_0002;
pub const PT_SCE_RELRO: u32 = 0x6100_0010;
pub const PT_SCE_COMMENT: u32 = 0x6FFF_FF00;
pub const PT_SCE_LIBVERSION: u32 = 0x6FFF_FF01;

pub const PF_X: u32 = 1;
pub const PF_W: u32 = 2;
pub const PF_R: u32 = 4;

pub mod dt {
    pub const NULL: u64 = 0;
    pub const NEEDED: u64 = 1;
    pub const PLTRELSZ: u64 = 2;
    pub const PLTGOT: u64 = 3;
    pub const HASH: u64 = 4;
    pub const STRTAB: u64 = 5;
    pub const SYMTAB: u64 = 6;
    pub const RELA: u64 = 7;
    pub const RELASZ: u64 = 8;
    pub const RELAENT: u64 = 9;
    pub const STRSZ: u64 = 10;
    pub const SYMENT: u64 = 11;
    pub const INIT: u64 = 12;
    pub const FINI: u64 = 13;
    pub const SONAME: u64 = 14;
    pub const SYMBOLIC: u64 = 16;
    pub const DEBUG: u64 = 21;
    pub const TEXTREL: u64 = 22;
    pub const JMPREL: u64 = 23;
    pub const BIND_NOW: u64 = 24;
    pub const INIT_ARRAY: u64 = 25;
    pub const FINI_ARRAY: u64 = 26;
    pub const INIT_ARRAYSZ: u64 = 27;
    pub const FINI_ARRAYSZ: u64 = 28;
    pub const FLAGS: u64 = 30;
    pub const PREINIT_ARRAY: u64 = 32;
    pub const PREINIT_ARRAYSZ: u64 = 33;
    pub const SCE_IDTABENTSZ: u64 = 0x6100_0005;
    pub const SCE_FINGERPRINT: u64 = 0x6100_0007;
    pub const SCE_ORIGINAL_FILENAME: u64 = 0x6100_0009;
    pub const SCE_MODULE_INFO: u64 = 0x6100_000D;
    pub const SCE_NEEDED_MODULE: u64 = 0x6100_000F;
    pub const SCE_MODULE_ATTR: u64 = 0x6100_0011;
    pub const SCE_EXPORT_LIB: u64 = 0x6100_0013;
    pub const SCE_IMPORT_LIB: u64 = 0x6100_0015;
    pub const SCE_EXPORT_LIB_ATTR: u64 = 0x6100_0017;
    pub const SCE_IMPORT_LIB_ATTR: u64 = 0x6100_0019;
    pub const SCE_STUB_MODULE_NAME: u64 = 0x6100_001D;
    pub const SCE_STUB_MODULE_VERSION: u64 = 0x6100_001F;
    pub const SCE_STUB_LIBRARY_NAME: u64 = 0x6100_0021;
    pub const SCE_STUB_LIBRARY_VERSION: u64 = 0x6100_0023;
    pub const SCE_HASH: u64 = 0x6100_0025;
    pub const SCE_PLTGOT: u64 = 0x6100_0027;
    pub const SCE_JMPREL: u64 = 0x6100_0029;
    pub const SCE_PLTREL: u64 = 0x6100_002B;
    pub const SCE_PLTRELSZ: u64 = 0x6100_002D;
    pub const SCE_RELA: u64 = 0x6100_002F;
    pub const SCE_RELASZ: u64 = 0x6100_0031;
    pub const SCE_RELAENT: u64 = 0x6100_0033;
    pub const SCE_STRTAB: u64 = 0x6100_0035;
    pub const SCE_STRSZ: u64 = 0x6100_0037;
    pub const SCE_SYMTAB: u64 = 0x6100_0039;
    pub const SCE_SYMENT: u64 = 0x6100_003B;
    pub const SCE_HASHSZ: u64 = 0x6100_003D;
    pub const SCE_SYMTABSZ: u64 = 0x6100_003F;
}

pub mod reloc {
    pub const NONE: u32 = 0;
    pub const R64: u32 = 1;
    pub const PC32: u32 = 2;
    pub const GLOB_DAT: u32 = 6;
    pub const JUMP_SLOT: u32 = 7;
    pub const RELATIVE: u32 = 8;
    pub const DTPMOD64: u32 = 16;
    pub const DTPOFF64: u32 = 17;
    pub const TPOFF64: u32 = 18;
}

pub const STB_LOCAL: u8 = 0;
pub const STB_GLOBAL: u8 = 1;
pub const STB_WEAK: u8 = 2;
pub const STT_NOTYPE: u8 = 0;
pub const STT_OBJECT: u8 = 1;
pub const STT_FUNC: u8 = 2;
pub const STT_SECTION: u8 = 3;
pub const STT_TLS: u8 = 6;
pub const STT_SCE: u8 = 11;

#[derive(Debug)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

macro_rules! bail {
    ($($t:tt)*) => { return Err(Error(format!($($t)*))) };
}

fn rd<const N: usize>(b: &[u8], off: usize) -> Result<[u8; N]> {
    match b.get(off..off.checked_add(N).unwrap_or(usize::MAX)) {
        Some(s) => Ok(s.try_into().unwrap()),
        None => bail!("read of {N} bytes at {off:#x} out of bounds (len {:#x})", b.len()),
    }
}
fn u16_at(b: &[u8], off: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(rd::<2>(b, off)?))
}
fn u32_at(b: &[u8], off: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(rd::<4>(b, off)?))
}
fn u64_at(b: &[u8], off: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(rd::<8>(b, off)?))
}

#[derive(Clone, Copy, Debug)]
pub struct Ehdr {
    pub e_type: u16,
    pub e_machine: u16,
    pub e_entry: u64,
    pub e_phoff: u64,
    pub e_phentsize: u16,
    pub e_phnum: u16,
    pub osabi: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct Phdr {
    pub p_type: u32,
    pub p_flags: u32,
    pub p_offset: u64,
    pub p_vaddr: u64,
    pub p_paddr: u64,
    pub p_filesz: u64,
    pub p_memsz: u64,
    pub p_align: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct SelfEntry {
    pub props: u64,
    pub offset: u64,
    pub filesz: u64,
    pub memsz: u64,
}

impl SelfEntry {
    pub fn is_blocked(&self) -> bool {
        self.props & (1 << 11) != 0
    }
    pub fn is_encrypted(&self) -> bool {
        self.props & (1 << 1) != 0
    }
    pub fn is_compressed(&self) -> bool {
        self.props & (1 << 3) != 0
    }
    pub fn segment_id(&self) -> usize {
        ((self.props >> 20) & 0xFFF) as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Container {
    SelfFile,
    RawElf,
}

/// A parsed module file. Holds the whole file and knows where each program
/// header's bytes live inside it.
pub struct Module {
    pub container: Container,
    pub ehdr: Ehdr,
    pub phdrs: Vec<Phdr>,
    pub self_entries: Vec<SelfEntry>,
    /// File offset of each program header's data (None if not present in the file).
    sources: Vec<Option<u64>>,
    bytes: Vec<u8>,
}

impl Module {
    pub fn parse(bytes: Vec<u8>) -> Result<Module> {
        let magic = u32_at(&bytes, 0)?;
        let (container, elf_off, self_entries) = if magic == SELF_MAGIC {
            let num_entries = u16_at(&bytes, 0x18)? as usize;
            let mut entries = Vec::with_capacity(num_entries);
            for i in 0..num_entries {
                let o = 0x20 + i * 0x20;
                entries.push(SelfEntry {
                    props: u64_at(&bytes, o)?,
                    offset: u64_at(&bytes, o + 8)?,
                    filesz: u64_at(&bytes, o + 16)?,
                    memsz: u64_at(&bytes, o + 24)?,
                });
            }
            (Container::SelfFile, 0x20 + num_entries * 0x20, entries)
        } else if rd::<4>(&bytes, 0)? == ELF_MAGIC {
            (Container::RawElf, 0, Vec::new())
        } else {
            bail!("not a SELF or ELF file (magic {magic:#010x})");
        };

        let e = &bytes[elf_off..];
        if rd::<4>(e, 0)? != ELF_MAGIC {
            bail!("embedded ELF magic missing at {elf_off:#x}");
        }
        if e[4] != 2 || e[5] != 1 {
            bail!("only ELF64 little-endian is supported");
        }
        let ehdr = Ehdr {
            osabi: e[7],
            e_type: u16_at(e, 0x10)?,
            e_machine: u16_at(e, 0x12)?,
            e_entry: u64_at(e, 0x18)?,
            e_phoff: u64_at(e, 0x20)?,
            e_phentsize: u16_at(e, 0x36)?,
            e_phnum: u16_at(e, 0x38)?,
        };
        if ehdr.e_machine != 62 {
            bail!("unexpected machine {}", ehdr.e_machine);
        }
        if ehdr.e_phentsize != 56 {
            bail!("unexpected phentsize {}", ehdr.e_phentsize);
        }
        let mut phdrs = Vec::with_capacity(ehdr.e_phnum as usize);
        for i in 0..ehdr.e_phnum as usize {
            let o = ehdr.e_phoff as usize + i * 56;
            phdrs.push(Phdr {
                p_type: u32_at(e, o)?,
                p_flags: u32_at(e, o + 4)?,
                p_offset: u64_at(e, o + 8)?,
                p_vaddr: u64_at(e, o + 16)?,
                p_paddr: u64_at(e, o + 24)?,
                p_filesz: u64_at(e, o + 32)?,
                p_memsz: u64_at(e, o + 40)?,
                p_align: u64_at(e, o + 48)?,
            });
        }

        // Locate every program header's bytes.
        let mut sources = vec![None; phdrs.len()];
        match container {
            Container::RawElf => {
                for (i, p) in phdrs.iter().enumerate() {
                    if p.p_filesz == 0 {
                        continue;
                    }
                    let end = p.p_offset.checked_add(p.p_filesz);
                    if end.is_some_and(|end| end <= bytes.len() as u64) {
                        sources[i] = Some(p.p_offset);
                    }
                }
            }
            Container::SelfFile => {
                // Direct entries first.
                for ent in &self_entries {
                    if !ent.is_blocked() {
                        continue;
                    }
                    let id = ent.segment_id();
                    let Some(p) = phdrs.get(id) else {
                        bail!("SELF entry references program header {id} which does not exist");
                    };
                    if ent.is_encrypted() || ent.is_compressed() {
                        bail!("SELF segment {id} is encrypted/compressed; only decrypted dumps are supported");
                    }
                    if ent.offset.checked_add(p.p_filesz).is_none_or(|end| end > bytes.len() as u64) {
                        bail!("SELF segment {id} data exceeds the file");
                    }
                    sources[id] = Some(ent.offset);
                }
                // Nested program headers resolve through the smallest direct container.
                for i in 0..phdrs.len() {
                    if sources[i].is_some() || phdrs[i].p_filesz == 0 {
                        continue;
                    }
                    let t = phdrs[i];
                    let mut best: Option<(u64, u64)> = None; // (size, file offset)
                    for (j, c) in phdrs.iter().enumerate() {
                        let Some(src) = sources[j] else { continue };
                        if self_entries.iter().all(|en| !en.is_blocked() || en.segment_id() != j) {
                            continue;
                        }
                        if t.p_offset >= c.p_offset && t.p_offset + t.p_filesz <= c.p_offset + c.p_filesz {
                            let off = src + (t.p_offset - c.p_offset);
                            if best.is_none_or(|(s, _)| c.p_filesz < s) {
                                best = Some((c.p_filesz, off));
                            }
                        }
                    }
                    sources[i] = best.map(|(_, off)| off);
                }
            }
        }
        Ok(Module { container, ehdr, phdrs, self_entries, sources, bytes })
    }

    pub fn file_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The file-backed bytes of program header `i` (length p_filesz), if present.
    pub fn segment_data(&self, i: usize) -> Option<&[u8]> {
        let p = self.phdrs.get(i)?;
        let off = self.sources.get(i).copied().flatten()? as usize;
        self.bytes.get(off..off + p.p_filesz as usize)
    }

    pub fn find_phdr(&self, p_type: u32) -> Option<usize> {
        self.phdrs.iter().position(|p| p.p_type == p_type)
    }

    pub fn is_executable(&self) -> bool {
        matches!(self.ehdr.e_type, ET_SCE_EXEC | ET_SCE_DYNEXEC | ET_SCE_REPLAY_EXEC)
    }

    /// Size of the virtual image (max end of loadable segments), page aligned.
    pub fn image_span(&self, page: u64) -> u64 {
        let mut end = 0u64;
        for p in &self.phdrs {
            if p.p_type == PT_LOAD || p.p_type == PT_SCE_RELRO {
                end = end.max(align_up(p.p_vaddr + p.p_memsz, page));
            }
        }
        end
    }

    pub fn dynamic(&self) -> Result<DynamicInfo> {
        DynamicInfo::parse(self)
    }
}

pub fn align_up(v: u64, a: u64) -> u64 {
    if a <= 1 { v } else { (v + a - 1) & !(a - 1) }
}

#[derive(Clone, Debug)]
pub struct ModuleRef {
    pub name: String,
    pub id: u16,
    pub version_major: u8,
    pub version_minor: u8,
}

#[derive(Clone, Debug)]
pub struct LibraryRef {
    pub name: String,
    pub id: u16,
    pub version: u16,
    pub attr: u16,
}

#[derive(Clone, Debug)]
pub struct Symbol {
    pub name: String,
    pub info: u8,
    pub other: u8,
    pub shndx: u16,
    pub value: u64,
    pub size: u64,
}

impl Symbol {
    pub fn bind(&self) -> u8 {
        self.info >> 4
    }
    pub fn sym_type(&self) -> u8 {
        self.info & 0xF
    }
    pub fn is_defined(&self) -> bool {
        self.shndx != 0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Rela {
    pub offset: u64,
    pub info: u64,
    pub addend: i64,
}

impl Rela {
    pub fn r_type(&self) -> u32 {
        self.info as u32
    }
    pub fn sym(&self) -> usize {
        (self.info >> 32) as usize
    }
}

/// Decoded dynamic-link information of one module.
#[derive(Clone, Debug, Default)]
pub struct DynamicInfo {
    pub module_info: Option<ModuleRef>,
    pub needed: Vec<String>,
    pub needed_modules: Vec<ModuleRef>,
    pub import_libs: Vec<LibraryRef>,
    pub export_libs: Vec<LibraryRef>,
    pub symbols: Vec<Symbol>,
    pub relas: Vec<Rela>,
    pub plt_relas: Vec<Rela>,
    pub init: Option<u64>,
    pub fini: Option<u64>,
    pub init_array: Option<(u64, u64)>,
    pub fini_array: Option<(u64, u64)>,
    pub preinit_array: Option<(u64, u64)>,
    pub pltgot: Option<u64>,
    pub flags: u64,
    pub soname: Option<String>,
    pub fingerprint: Option<Vec<u8>>,
    pub original_filename: Option<String>,
}

impl DynamicInfo {
    fn parse(m: &Module) -> Result<DynamicInfo> {
        let dyn_idx = m.find_phdr(PT_DYNAMIC).ok_or_else(|| Error("no PT_DYNAMIC".into()))?;
        let data_idx = m.find_phdr(PT_SCE_DYNLIBDATA).ok_or_else(|| Error("no PT_SCE_DYNLIBDATA".into()))?;
        let dynb = m.segment_data(dyn_idx).ok_or_else(|| Error("PT_DYNAMIC bytes not in file".into()))?;
        let data = m.segment_data(data_idx).ok_or_else(|| Error("PT_SCE_DYNLIBDATA bytes not in file".into()))?;

        let mut tags: Vec<(u64, u64)> = Vec::new();
        for i in 0..dynb.len() / 16 {
            let tag = u64_at(dynb, i * 16)?;
            let val = u64_at(dynb, i * 16 + 8)?;
            if tag == dt::NULL {
                break;
            }
            tags.push((tag, val));
        }
        let get = |t: u64| tags.iter().find(|(k, _)| *k == t).map(|(_, v)| *v);
        let strtab = get(dt::SCE_STRTAB).ok_or_else(|| Error("no DT_SCE_STRTAB".into()))? as usize;
        let strsz = get(dt::SCE_STRSZ).unwrap_or(0) as usize;
        let strs = data
            .get(strtab..strtab + strsz)
            .ok_or_else(|| Error("string table out of bounds".into()))?;
        let cstr = |off: u64| -> Result<String> {
            let off = off as usize;
            let s = strs.get(off..).ok_or_else(|| Error(format!("string offset {off:#x} out of bounds")))?;
            let end = s.iter().position(|&c| c == 0).unwrap_or(s.len());
            Ok(String::from_utf8_lossy(&s[..end]).into_owned())
        };

        let mut info = DynamicInfo::default();
        let mut import_attr: Vec<(u16, u16)> = Vec::new();
        let mut export_attr: Vec<(u16, u16)> = Vec::new();
        for &(tag, val) in &tags {
            let name_off = val & 0xFFFF_FFFF;
            let id = (val >> 48) as u16;
            match tag {
                dt::NEEDED => info.needed.push(cstr(val)?),
                dt::SONAME => info.soname = Some(cstr(val)?),
                dt::SCE_MODULE_INFO => {
                    info.module_info = Some(ModuleRef {
                        name: cstr(name_off)?,
                        id,
                        version_major: (val >> 40) as u8,
                        version_minor: (val >> 32) as u8,
                    })
                }
                dt::SCE_NEEDED_MODULE => info.needed_modules.push(ModuleRef {
                    name: cstr(name_off)?,
                    id,
                    version_major: (val >> 40) as u8,
                    version_minor: (val >> 32) as u8,
                }),
                dt::SCE_IMPORT_LIB | dt::SCE_EXPORT_LIB => {
                    let lib = LibraryRef { name: cstr(name_off)?, id, version: (val >> 32) as u16, attr: 0 };
                    if tag == dt::SCE_IMPORT_LIB {
                        info.import_libs.push(lib)
                    } else {
                        info.export_libs.push(lib)
                    }
                }
                dt::SCE_IMPORT_LIB_ATTR => import_attr.push((id, val as u16)),
                dt::SCE_EXPORT_LIB_ATTR => export_attr.push((id, val as u16)),
                dt::INIT => info.init = Some(val),
                dt::FINI => info.fini = Some(val),
                dt::FLAGS => info.flags = val,
                dt::SCE_PLTGOT => info.pltgot = Some(val),
                dt::SCE_FINGERPRINT => {
                    info.fingerprint = data.get(val as usize..val as usize + 20).map(|s| s.to_vec())
                }
                dt::SCE_ORIGINAL_FILENAME => info.original_filename = Some(cstr(val)?),
                _ => {}
            }
        }
        for (id, attr) in import_attr {
            if let Some(l) = info.import_libs.iter_mut().find(|l| l.id == id) {
                l.attr = attr;
            }
        }
        for (id, attr) in export_attr {
            if let Some(l) = info.export_libs.iter_mut().find(|l| l.id == id) {
                l.attr = attr;
            }
        }
        let pair = |a: u64, s: u64| -> Option<(u64, u64)> { Some((get(a)?, get(s).unwrap_or(0))) };
        info.init_array = pair(dt::INIT_ARRAY, dt::INIT_ARRAYSZ);
        info.fini_array = pair(dt::FINI_ARRAY, dt::FINI_ARRAYSZ);
        info.preinit_array = pair(dt::PREINIT_ARRAY, dt::PREINIT_ARRAYSZ);

        // Symbols
        if let Some(symtab) = get(dt::SCE_SYMTAB) {
            let symsz = get(dt::SCE_SYMTABSZ).unwrap_or(0) as usize;
            let syment = get(dt::SCE_SYMENT).unwrap_or(24) as usize;
            if syment != 24 {
                bail!("unexpected symbol entry size {syment}");
            }
            let st = data
                .get(symtab as usize..symtab as usize + symsz)
                .ok_or_else(|| Error("symbol table out of bounds".into()))?;
            for i in 0..symsz / 24 {
                let o = i * 24;
                info.symbols.push(Symbol {
                    name: cstr(u32_at(st, o)? as u64)?,
                    info: st[o + 4],
                    other: st[o + 5],
                    shndx: u16_at(st, o + 6)?,
                    value: u64_at(st, o + 8)?,
                    size: u64_at(st, o + 16)?,
                });
            }
        }
        let read_relas = |off: Option<u64>, sz: Option<u64>| -> Result<Vec<Rela>> {
            let (Some(off), Some(sz)) = (off, sz) else { return Ok(Vec::new()) };
            let r = data
                .get(off as usize..(off + sz) as usize)
                .ok_or_else(|| Error("relocation table out of bounds".into()))?;
            let mut v = Vec::with_capacity(r.len() / 24);
            for i in 0..r.len() / 24 {
                let o = i * 24;
                v.push(Rela { offset: u64_at(r, o)?, info: u64_at(r, o + 8)?, addend: u64_at(r, o + 16)? as i64 });
            }
            Ok(v)
        };
        info.relas = read_relas(get(dt::SCE_RELA), get(dt::SCE_RELASZ))?;
        info.plt_relas = read_relas(get(dt::SCE_JMPREL), get(dt::SCE_PLTRELSZ))?;
        Ok(info)
    }

    pub fn import_lib_by_id(&self, id: u64) -> Option<&LibraryRef> {
        self.import_libs.iter().find(|l| l.id as u64 == id)
    }
    pub fn export_lib_by_id(&self, id: u64) -> Option<&LibraryRef> {
        self.export_libs.iter().find(|l| l.id as u64 == id)
    }
    pub fn needed_module_by_id(&self, id: u64) -> Option<&ModuleRef> {
        self.needed_modules.iter().find(|m| m.id as u64 == id)
    }
}
