//! Temporal upscalers in place of the game's SMAA: NVIDIA DLSS, Intel XeSS
//! and AMD FSR. The game's G-buffer already holds per-pixel motion
//! vectors (R16G16_SFLOAT, screen fraction per frame, current minus
//! previous - its characters included), the port jitters the scene
//! (`gpu::JITTER`, a Halton sequence) and the renderer hands SMAA's input -
//! the HDR scene colour - with the depth and the motion vectors to the
//! upscaler, which writes SMAA's output (render.rs `upscale_hook`): the
//! native anti-aliasing modes, the render size in and out. Upscaling (PC
//! Settings `upscaling`, at the next start: `render_scale`): the game
//! renders its scene and post-processing below the output size (its own
//! inner render size, `title_patches`) and scales the finished picture up in
//! one draw before its interface; the upscaler replaces that draw - the
//! picture (tone-mapped, LDR) with the depth and motion vectors in, the
//! output-size picture out - while SMAA stays at the render size.
//!
//! The choice is live (PC Settings `upscaler`, `set_choice`): every
//! upscaler whose library is installed is prepared when the Vulkan device is
//! made (its extensions and features), its context made at its first use,
//! and switching resets the history. The libraries are loaded from the
//! `upscalers` folder beside tlg.exe (else from beside it; TLG_XESS_DLL,
//! TLG_SL_DIR and TLG_FFX_DIR name development copies), each after a check
//! of its Authenticode signature and its publisher (WinVerifyTrust: NVIDIA,
//! Intel or AMD; TLG_UPSCALER_UNSIGNED=1 skips it for unsigned development
//! builds). The port's GPL-3.0 section 7 permission covers linking them.
//!
//! XeSS: Intel's libxess.dll (Intel Simplified Software License). Its Vulkan
//! path needs the mutable-descriptor-type extension and feature, enabled
//! when the device is made.
//!
//! DLSS: NVIDIA Streamline 2.14 (sl.interposer.dll with its plugins
//! sl.common.dll and sl.dlss.dll, MIT, shipped) and DLSS itself,
//! nvngx_dlss.dll (NVIDIA's RTX SDK license: not shipped - the player's own
//! copy, from a game with DLSS or NVIDIA's DLSS SDK; without it Streamline
//! is not started), in Streamline's "manual hooking" mode: slInit before the device, the DLSS
//! plugin's Vulkan extensions and 1.2 features added to the instance and
//! device, slSetVulkanInfo once it exists (on a thread of its own: NGX's
//! start takes ~1.7 s, the game boots meanwhile and DLSS is offered once it
//! is done), the present through Streamline's vkQueuePresentKHR (its
//! per-frame bookkeeping; its other hooks serve frame generation only) while
//! everything else keeps the system's Vulkan loader, and each frame the four
//! images tagged on the slEvaluateFeature call itself.
//!
//! FSR: AMD's FSR API (amd_fidelityfx_loader_vk.dll, which loads
//! amd_fidelityfx_upscaler_vk.dll beside it: FSR 4 on the AMD GPUs it
//! supports, else FSR 3.1; MIT). Its Vulkan backend uses whatever optional
//! features the device offers (its capability query reads the device's
//! extension list), so those are enabled where present.
//!
//! Jitter: all three take the image's shift in pixels - our viewport offset
//! as it is (measured with a constant jitter: LOG, Upscalers).
//!
//! Debug: TLG_UPSCALER=off|xess|dlss|fsr chooses at start and prepares only
//! that library; TLG_FFX_DIR names a development copy of FSR's; TLG_UPSCALE_MV=<x>,<y> multiplies the motion vectors' scale,
//! TLG_UPSCALE_JITTER=<x>,<y> the jitter passed to the upscaler, `tune
//! up_jitter <0-3>` flips its signs.

use ash::vk;
use parking_lot::Mutex;
use std::ffi::{CStr, CString, c_char, c_void};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, Ordering::Relaxed};
use windows_sys::Win32::Foundation::HMODULE;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Off = 0,
    Xess = 1,
    Dlss = 2,
    Fsr = 3,
}

impl Kind {
    fn parse(v: &str) -> Kind {
        match v.trim() {
            "xess" => Kind::Xess,
            "dlss" => Kind::Dlss,
            "fsr" => Kind::Fsr,
            _ => Kind::Off,
        }
    }

    fn from_u8(v: u8) -> Kind {
        match v {
            1 => Kind::Xess,
            2 => Kind::Dlss,
            3 => Kind::Fsr,
            _ => Kind::Off,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Kind::Off => "SMAA",
            Kind::Xess => "XeSS",
            Kind::Dlss => "DLSS",
            Kind::Fsr => "FSR",
        }
    }
}

/// The upscaler chosen now (`Kind` as u8; u8::MAX until read from the
/// settings).
static CHOICE: AtomicU8 = AtomicU8::new(u8::MAX);
/// The choice changed: the next frame starts a new history.
static SWITCHED: AtomicBool = AtomicBool::new(false);

/// The upscaler chosen now: PC Settings `upscaler` (TLG_UPSCALER at start).
pub fn choice() -> Kind {
    let v = CHOICE.load(Relaxed);
    if v != u8::MAX {
        return Kind::from_u8(v);
    }
    let k = Kind::parse(&std::env::var("TLG_UPSCALER").ok().or_else(|| crate::config::settings().pc.get("upscaler").cloned()).unwrap_or_default());
    CHOICE.store(k as u8, Relaxed);
    k
}

/// A new choice from the settings (applied from the next frame).
pub fn set_choice(v: &str) {
    let k = Kind::parse(v);
    if CHOICE.swap(k as u8, Relaxed) != k as u8 {
        SWITCHED.store(true, Relaxed);
        crate::info!("upscaler: {} chosen{}", k.name(), if k != Kind::Off && !ready(k) { " (not available here: SMAA stays)" } else { "" });
    }
}

/// Whether this session prepares `k`'s library: TLG_UPSCALER names the one
/// (development runs), else each that is installed.
fn prepares(k: Kind) -> bool {
    match std::env::var("TLG_UPSCALER") {
        Ok(v) => Kind::parse(&v) == k,
        Err(_) => true,
    }
}

static XESS_READY: AtomicBool = AtomicBool::new(false);
static DLSS_READY: AtomicBool = AtomicBool::new(false);
static FSR_READY: AtomicBool = AtomicBool::new(false);
static XESS_FAILED: AtomicBool = AtomicBool::new(false);
static DLSS_FAILED: AtomicBool = AtomicBool::new(false);
static FSR_FAILED: AtomicBool = AtomicBool::new(false);

fn failed_flag(k: Kind) -> &'static AtomicBool {
    match k {
        Kind::Dlss => &DLSS_FAILED,
        Kind::Fsr => &FSR_FAILED,
        _ => &XESS_FAILED,
    }
}

/// Whether `k` is ready to run in this session.
fn ready(k: Kind) -> bool {
    match k {
        Kind::Off => false,
        Kind::Xess => XESS_READY.load(Relaxed) && !XESS_FAILED.load(Relaxed),
        Kind::Dlss => DLSS_READY.load(Relaxed) && !DLSS_FAILED.load(Relaxed),
        Kind::Fsr => FSR_READY.load(Relaxed) && !FSR_FAILED.load(Relaxed),
    }
}

/// Whether an upscaler replaces SMAA now (chosen and ready).
pub fn active() -> bool {
    ready(choice())
}

/// The choices the settings offer: SMAA and the upscalers ready here.
pub fn choices() -> Vec<(String, String)> {
    let mut v = vec![("off".to_string(), "SMAA (the game's)".to_string())];
    if ready(Kind::Dlss) {
        v.push(("dlss".into(), "DLSS (DLAA)".into()));
    }
    if ready(Kind::Xess) {
        v.push(("xess".into(), "XeSS Native AA".into()));
    }
    if ready(Kind::Fsr) {
        v.push(("fsr".into(), "FSR Native AA".into()));
    }
    v
}

/// The render scale of upscaling (PC Settings `upscaling`, at start;
/// TLG_UPSCALING overrides): 1 = off.
pub fn render_scale() -> f32 {
    static S: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        let v = std::env::var("TLG_UPSCALING").ok().or_else(|| crate::config::settings().pc.get("upscaling").cloned()).unwrap_or_default();
        match v.trim() {
            "quality" => 2.0 / 3.0,
            "balanced" => 0.58,
            "performance" => 0.5,
            "ultra_performance" => 1.0 / 3.0,
            _ => 1.0,
        }
    })
}

/// The textures' level-of-detail bias while the game renders below the
/// output (the upscalers' guidance: log2 of the render scale).
pub fn mip_bias() -> f32 {
    let s = render_scale();
    if s < 1.0 { s.log2() } else { 0.0 }
}

/// `k` cannot run in this session: logged once, SMAA stays for it.
fn fail(k: Kind, msg: impl std::fmt::Display) {
    if !failed_flag(k).swap(true, Relaxed) {
        crate::warn!("upscaler: {} unavailable: {msg}", k.name());
    }
}

// ---------------------------------------------------------------------------
// Loading.

trait WideNull {
    fn encode_wide_null(&self) -> Vec<u16>;
}

impl WideNull for std::ffi::OsStr {
    fn encode_wide_null(&self) -> Vec<u16> {
        use std::os::windows::ffi::OsStrExt;
        self.encode_wide().chain(std::iter::once(0)).collect()
    }
}

/// The upscalers' folder: `upscalers` beside tlg.exe, else its own folder.
fn lib_dir() -> PathBuf {
    let dir = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)).unwrap_or_default();
    let sub = dir.join("upscalers");
    if sub.is_dir() { sub } else { dir }
}

/// Checks that `path` carries a valid Authenticode signature whose signer's
/// name starts with `publisher` (no revocation check, nothing fetched from
/// the network). TLG_UPSCALER_UNSIGNED=1 skips it (development builds).
fn check_signature(path: &Path, publisher: &str) -> Result<(), String> {
    use windows_sys::Win32::Security::Cryptography::{CERT_NAME_SIMPLE_DISPLAY_TYPE, CertGetNameStringW};
    use windows_sys::Win32::Security::WinTrust::*;
    if std::env::var("TLG_UPSCALER_UNSIGNED").is_ok_and(|v| v == "1") {
        return Ok(());
    }
    let wide = path.as_os_str().encode_wide_null();
    // SAFETY: plain structures for WinVerifyTrust, alive until its state is
    // closed; the signer's certificate is read before that.
    unsafe {
        let mut file: WINTRUST_FILE_INFO = std::mem::zeroed();
        file.cbStruct = std::mem::size_of::<WINTRUST_FILE_INFO>() as u32;
        file.pcwszFilePath = wide.as_ptr();
        let mut data: WINTRUST_DATA = std::mem::zeroed();
        data.cbStruct = std::mem::size_of::<WINTRUST_DATA>() as u32;
        data.dwUIChoice = WTD_UI_NONE;
        data.fdwRevocationChecks = WTD_REVOKE_NONE;
        data.dwUnionChoice = WTD_CHOICE_FILE;
        data.Anonymous.pFile = &mut file;
        data.dwStateAction = WTD_STATEACTION_VERIFY;
        data.dwProvFlags = WTD_CACHE_ONLY_URL_RETRIEVAL;
        let mut action = WINTRUST_ACTION_GENERIC_VERIFY_V2;
        let r = WinVerifyTrust(std::ptr::null_mut(), &mut action, &mut data as *mut WINTRUST_DATA as *mut c_void);
        // (The signer: the first certificate of the first signer's chain.)
        let mut signer = String::new();
        if r == 0 {
            let prov = WTHelperProvDataFromStateData(data.hWVTStateData);
            let sgnr = if prov.is_null() { std::ptr::null_mut() } else { WTHelperGetProvSignerFromChain(prov, 0, 0, 0) };
            if !sgnr.is_null() && (*sgnr).csCertChain > 0 && !(*sgnr).pasCertChain.is_null() {
                let cert = (*(*sgnr).pasCertChain).pCert;
                if !cert.is_null() {
                    let mut buf = [0u16; 256];
                    let n = CertGetNameStringW(cert, CERT_NAME_SIMPLE_DISPLAY_TYPE, 0, std::ptr::null(), buf.as_mut_ptr(), buf.len() as u32);
                    if n > 1 {
                        signer = String::from_utf16_lossy(&buf[..n as usize - 1]);
                    }
                }
            }
        }
        data.dwStateAction = WTD_STATEACTION_CLOSE;
        WinVerifyTrust(std::ptr::null_mut(), &mut action, &mut data as *mut WINTRUST_DATA as *mut c_void);
        if r != 0 {
            return Err(format!("{} has no valid signature (WinVerifyTrust {:#x})", path.display(), r as u32));
        }
        if !signer.starts_with(publisher) {
            return Err(format!("{} is signed by {signer:?}, not {publisher}", path.display()));
        }
        Ok(())
    }
}

/// Why a library was not loaded.
enum LoadError {
    /// Not installed (quietly skipped).
    Missing,
    Failed(String),
}

impl From<String> for LoadError {
    fn from(s: String) -> Self {
        LoadError::Failed(s)
    }
}

/// Loads a library signed by `publisher` (its dependencies from its own
/// folder).
fn load_lib(path: &Path, publisher: &str) -> Result<HMODULE, LoadError> {
    use windows_sys::Win32::System::LibraryLoader::{LOAD_WITH_ALTERED_SEARCH_PATH, LoadLibraryExW};
    if !path.is_file() {
        return Err(LoadError::Missing);
    }
    check_signature(path, publisher)?;
    let wide = path.as_os_str().encode_wide_null();
    let h = unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), LOAD_WITH_ALTERED_SEARCH_PATH) };
    if h.is_null() {
        return Err(LoadError::Failed(format!("loading {} failed", path.display())));
    }
    Ok(h)
}

/// A library's export.
fn export(h: HMODULE, path: &Path, name: &CStr) -> Result<*const c_void, String> {
    use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
    match unsafe { GetProcAddress(h, name.as_ptr() as *const u8) } {
        Some(p) => Ok(p as *const c_void),
        None => Err(format!("{} has no {}", path.display(), name.to_string_lossy())),
    }
}

/// C strings out of a library's list (count, array of pointers).
unsafe fn names(count: u32, list: *const *const c_char) -> Vec<CString> {
    if list.is_null() {
        return Vec::new();
    }
    (0..count as usize).filter_map(|i| unsafe { list.add(i).read().as_ref() }.map(|p| unsafe { CStr::from_ptr(p) }.to_owned())).collect()
}

// ---------------------------------------------------------------------------
// What the instance and device need.

/// Instance extensions the prepared upscalers need (before
/// vkCreateInstance), among those the loader has (`avail`); an upscaler
/// whose extension is missing is dropped.
pub fn instance_extensions(avail: &[vk::ExtensionProperties]) -> Vec<CString> {
    let has = |e: &CStr| avail.iter().any(|x| unsafe { CStr::from_ptr(x.extension_name.as_ptr()) } == e);
    let mut out: Vec<CString> = Vec::new();
    for (k, list) in [(Kind::Xess, xess_instance_extensions()), (Kind::Dlss, sl().map(|s| s.reqs.instance_ext.clone()).unwrap_or_default())] {
        if let Some(e) = list.iter().find(|e| !has(e)) {
            fail(k, format!("the Vulkan loader lacks the instance extension {}", e.to_string_lossy()));
            continue;
        }
        for e in list {
            if !out.contains(&e) {
                out.push(e);
            }
        }
    }
    out
}

/// What the device must enable for the prepared upscalers.
#[derive(Default)]
pub struct DeviceNeeds {
    pub extensions: Vec<CString>,
    /// VK_EXT_mutable_descriptor_type's feature.
    pub mutable_descriptor: bool,
    /// Vulkan 1.0-1.3 features by name (`apply_features`).
    pub features: Vec<String>,
}

static XESS_DEVICE_OK: AtomicBool = AtomicBool::new(false);
static DLSS_DEVICE_OK: AtomicBool = AtomicBool::new(false);
static FSR_DEVICE_OK: AtomicBool = AtomicBool::new(false);

/// Device extensions and features of the upscalers that can run on this
/// device (before vkCreateDevice).
pub fn device_needs(instance: &ash::Instance, pdev: vk::PhysicalDevice) -> DeviceNeeds {
    let mut n = DeviceNeeds::default();
    for (k, needs, ok) in [
        (Kind::Xess, xess_device_needs(instance, pdev), &XESS_DEVICE_OK),
        (Kind::Dlss, dlss_device_needs(instance, pdev), &DLSS_DEVICE_OK),
        (Kind::Fsr, fsr_device_needs(instance, pdev), &FSR_DEVICE_OK),
    ] {
        let Some(d) = needs else { continue };
        if failed_flag(k).load(Relaxed) {
            continue;
        }
        for e in d.extensions {
            if !n.extensions.contains(&e) {
                n.extensions.push(e);
            }
        }
        for f in d.features {
            if !n.features.contains(&f) {
                n.features.push(f);
            }
        }
        n.mutable_descriptor |= d.mutable_descriptor;
        ok.store(true, Relaxed);
    }
    n
}

/// A feature the port knows, by its Vulkan name: its field in the 1.0 to
/// 1.3 feature structures.
fn feature_slot<'a>(
    f10: &'a mut vk::PhysicalDeviceFeatures,
    f11: &'a mut vk::PhysicalDeviceVulkan11Features<'_>,
    f12: &'a mut vk::PhysicalDeviceVulkan12Features<'_>,
    f13: &'a mut vk::PhysicalDeviceVulkan13Features<'_>,
    name: &str,
) -> Option<&'a mut vk::Bool32> {
    Some(match name {
        "shaderInt16" => &mut f10.shader_int16,
        "storageBuffer16BitAccess" => &mut f11.storage_buffer16_bit_access,
        "uniformAndStorageBuffer16BitAccess" => &mut f11.uniform_and_storage_buffer16_bit_access,
        "timelineSemaphore" => &mut f12.timeline_semaphore,
        "descriptorIndexing" => &mut f12.descriptor_indexing,
        "bufferDeviceAddress" => &mut f12.buffer_device_address,
        "shaderFloat16" => &mut f12.shader_float16,
        "shaderStorageBufferArrayNonUniformIndexing" => &mut f12.shader_storage_buffer_array_non_uniform_indexing,
        "subgroupSizeControl" => &mut f13.subgroup_size_control,
        "computeFullSubgroups" => &mut f13.compute_full_subgroups,
        _ => return None,
    })
}

/// Turns on the features named in `n` (those the port knows).
pub fn apply_features(
    f10: &mut vk::PhysicalDeviceFeatures,
    f11: &mut vk::PhysicalDeviceVulkan11Features<'_>,
    f12: &mut vk::PhysicalDeviceVulkan12Features<'_>,
    f13: &mut vk::PhysicalDeviceVulkan13Features<'_>,
    n: &DeviceNeeds,
) {
    for name in &n.features {
        if let Some(v) = feature_slot(f10, f11, f12, f13, name) {
            *v = vk::TRUE;
        }
    }
}

/// Whether the device has each feature in `names` (None: unknown to the
/// port).
fn device_features(instance: &ash::Instance, pdev: vk::PhysicalDevice, names: &[&str]) -> Vec<Option<bool>> {
    let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
    let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f11).push_next(&mut f12).push_next(&mut f13);
    unsafe { instance.get_physical_device_features2(pdev, &mut f2) };
    let mut f10 = f2.features;
    names.iter().map(|n| feature_slot(&mut f10, &mut f11, &mut f12, &mut f13, n).map(|v| *v != 0)).collect()
}

/// The features in `names`, each one the device has (Err: one it lacks or
/// the port does not know).
fn features_supported(instance: &ash::Instance, pdev: vk::PhysicalDevice, names: &[String]) -> Result<(), String> {
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    for (name, have) in names.iter().zip(device_features(instance, pdev, &refs)) {
        match have {
            None => return Err(format!("it asks for the Vulkan feature {name}, unknown to the port")),
            Some(false) => return Err(format!("the device lacks the Vulkan feature {name}")),
            Some(true) => {}
        }
    }
    Ok(())
}

/// Called right after vkCreateDevice: `queues` of family `qf` are the
/// port's.
pub fn device_created(instance: &ash::Instance, pdev: vk::PhysicalDevice, device: &ash::Device, qf: u32, queues: u32) {
    if XESS_DEVICE_OK.load(Relaxed) && xess().is_some() {
        XESS_READY.store(true, Relaxed);
        crate::info!("upscaler: XeSS ready");
    }
    if FSR_DEVICE_OK.load(Relaxed) && ffx().is_some() {
        FSR_READY.store(true, Relaxed);
        crate::info!("upscaler: FSR ready");
    }
    if DLSS_DEVICE_OK.load(Relaxed) {
        dlss_device_created(instance, pdev, device, qf, queues);
    }
}

/// Streamline's vkQueuePresentKHR once DLSS is ready (its per-frame
/// bookkeeping runs there); None: the usual present.
pub fn present_hook() -> Option<vk::PFN_vkQueuePresentKHR> {
    SL_PRESENT.get().copied()
}

static SL_PRESENT: std::sync::OnceLock<vk::PFN_vkQueuePresentKHR> = std::sync::OnceLock::new();

// ---------------------------------------------------------------------------
// The per-frame inputs.

/// One image the upscaler reads or writes.
#[derive(Clone, Copy)]
pub struct Tex {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub range: vk::ImageSubresourceRange,
    pub format: vk::Format,
    /// The image's own size (the render size may be a part of it).
    pub size: (u32, u32),
    pub usage: vk::ImageUsageFlags,
    /// Its layout while the upscaler runs.
    pub layout: vk::ImageLayout,
}

/// The inputs of one replacement: the scene colour, depth and motion vectors
/// sampled (at the render size), the output written (at the output size:
/// the same in the native anti-aliasing modes).
pub struct Inputs {
    pub color: Tex,
    pub depth: Tex,
    pub velocity: Tex,
    pub output: Tex,
    /// The render size (the inputs').
    pub in_width: u32,
    pub in_height: u32,
    /// The output size.
    pub width: u32,
    pub height: u32,
    /// The colour is tone-mapped (the upscaling path: the game's finished
    /// picture, sRGB-encoded) rather than the HDR scene.
    pub ldr: bool,
}

impl Inputs {
    /// Output size over render size (1: native anti-aliasing).
    fn ratio(&self) -> f32 {
        self.width as f32 / self.in_width.max(1) as f32
    }
}

/// Records the chosen upscaler into `cmd` in place of SMAA or the game's
/// scaling draw (layouts: as each `Tex` says - the caller's barriers).
/// Returns false when it cannot run (then the game's own draw runs).
pub fn record(cmd: vk::CommandBuffer, inp: &Inputs) -> bool {
    let reset = SWITCHED.swap(false, Relaxed);
    match choice() {
        Kind::Xess if ready(Kind::Xess) => xess_record(cmd, inp, reset),
        Kind::Dlss if ready(Kind::Dlss) => dlss_record(cmd, inp, reset),
        Kind::Fsr if ready(Kind::Fsr) => fsr_record(cmd, inp, reset),
        _ => false,
    }
}

fn env_pair(name: &str) -> (f32, f32) {
    std::env::var(name)
        .ok()
        .and_then(|v| {
            let (a, b) = v.split_once(',')?;
            Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
        })
        .unwrap_or((1.0, 1.0))
}

/// The motion vectors' scale multipliers (TLG_UPSCALE_MV).
fn mv_scale() -> (f32, f32) {
    static V: std::sync::OnceLock<(f32, f32)> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_pair("TLG_UPSCALE_MV"))
}

/// The jitter's scale for the upscaler: TLG_UPSCALE_JITTER with the signs
/// of the `up_jitter` tunable.
fn jitter_scale() -> (f32, f32) {
    static V: std::sync::OnceLock<(f32, f32)> = std::sync::OnceLock::new();
    let (x, y) = *V.get_or_init(|| env_pair("TLG_UPSCALE_JITTER"));
    match crate::tune::get(crate::tune::UP_JITTER) {
        1 => (-x, -y),
        2 => (x, -y),
        3 => (-x, y),
        _ => (x, y),
    }
}

/// The Halton(2, 3) point `i` (from 1), centred: [-0.5, 0.5).
fn halton(mut i: u32, base: u32) -> f32 {
    let (mut f, mut r) = (1.0f32, 0.0f32);
    while i > 0 {
        f /= base as f32;
        r += f * (i % base) as f32;
        i /= base;
    }
    r - 0.5
}

/// Sets the scene's jitter for the next frame (`gpu::JITTER`); the value of
/// the frame being finished is returned (pixels, the viewport's offset).
/// `ratio`: output over render size - 8 phases at the native size, 8 x
/// ratio squared when upscaling (the vendors' guidance).
fn advance_jitter(frame: u32, ratio: f32) -> (f32, f32) {
    let bits = super::JITTER.load(Relaxed);
    let cur = if bits == 0 { (0.0, 0.0) } else { (f32::from_bits((bits >> 32) as u32), f32::from_bits(bits as u32)) };
    // (`up_jitter_fixed`: a constant one.)
    let phases = ((8.0 * ratio * ratio).ceil() as u32).clamp(8, 128);
    let i = frame % phases + 1;
    let fixed = crate::tune::get(crate::tune::UP_JITTER_FIXED);
    let (x, y) = if fixed != 0 { (fixed as f32 / 1000.0, fixed as f32 / 1000.0) } else { (halton(i, 2), halton(i, 3)) };
    let next = ((x.to_bits() as u64) << 32 | y.to_bits() as u64).max(1);
    super::JITTER.store(next, Relaxed);
    cur
}

/// Logs the first few failures of a call made on the recorder.
fn warn_some(counter: &AtomicU32, what: &str, r: i32) {
    if counter.fetch_add(1, Relaxed) < 4 {
        crate::warn!("upscaler: {what} failed ({r})");
    }
}

// ---------------------------------------------------------------------------
// XeSS's C interface (xess.h, xess_vk.h; structures packed to 8, which is
// their natural layout).

/// xess_vk_image_view_info.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct XessView {
    image_view: vk::ImageView,
    image: vk::Image,
    range: vk::ImageSubresourceRange,
    format: vk::Format,
    width: u32,
    height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Xess2d {
    x: u32,
    y: u32,
}

/// xess_vk_execute_params_t.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct XessExecute {
    color: XessView,
    velocity: XessView,
    depth: XessView,
    exposure: XessView,
    responsive: XessView,
    output: XessView,
    jitter_x: f32,
    jitter_y: f32,
    exposure_scale: f32,
    reset_history: u32,
    input_width: u32,
    input_height: u32,
    color_base: Xess2d,
    velocity_base: Xess2d,
    depth_base: Xess2d,
    responsive_base: Xess2d,
    reserved0: Xess2d,
    output_base: Xess2d,
}

/// xess_vk_init_params_t.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct XessInit {
    output: Xess2d,
    quality: i32,
    flags: u32,
    creation_node_mask: u32,
    visible_node_mask: u32,
    temp_buffer_heap: vk::DeviceMemory,
    buffer_heap_offset: u64,
    temp_texture_heap: vk::DeviceMemory,
    texture_heap_offset: u64,
    pipeline_cache: vk::PipelineCache,
}

const XESS_QUALITY_AA: i32 = 106;

type Handle = *mut c_void;
type LogFn = unsafe extern "C" fn(*const c_char, i32);

struct XessLib {
    instance_exts: unsafe extern "C" fn(*mut u32, *mut *const *const c_char, *mut u32) -> i32,
    device_exts: unsafe extern "C" fn(vk::Instance, vk::PhysicalDevice, *mut u32, *mut *const *const c_char) -> i32,
    create: unsafe extern "C" fn(vk::Instance, vk::PhysicalDevice, vk::Device, *mut Handle) -> i32,
    init: unsafe extern "C" fn(Handle, *const XessInit) -> i32,
    execute: unsafe extern "C" fn(Handle, vk::CommandBuffer, *const XessExecute) -> i32,
    velocity_scale: unsafe extern "C" fn(Handle, f32, f32) -> i32,
    logging: unsafe extern "C" fn(Handle, i32, LogFn) -> i32,
}

// SAFETY: plain function pointers into a library that stays loaded.
unsafe impl Send for XessLib {}
unsafe impl Sync for XessLib {}

/// libxess.dll and its functions (None: not prepared, not installed or
/// unusable).
fn xess() -> Option<&'static XessLib> {
    static L: std::sync::OnceLock<Option<XessLib>> = std::sync::OnceLock::new();
    L.get_or_init(|| {
        if !prepares(Kind::Xess) {
            return None;
        }
        match load_xess() {
            Ok(l) => Some(l),
            Err(LoadError::Missing) => {
                crate::info!("upscaler: XeSS not installed");
                None
            }
            Err(LoadError::Failed(e)) => {
                fail(Kind::Xess, e);
                None
            }
        }
    })
    .as_ref()
}

fn load_xess() -> Result<XessLib, LoadError> {
    let path = match std::env::var_os("TLG_XESS_DLL") {
        Some(p) => PathBuf::from(p),
        None => lib_dir().join("libxess.dll"),
    };
    let h = load_lib(&path, "Intel Corporation")?;
    let f = |name: &CStr| export(h, &path, name);
    // SAFETY: the signatures of xess.h / xess_vk.h.
    unsafe {
        let lib = XessLib {
            instance_exts: std::mem::transmute(f(c"xessVKGetRequiredInstanceExtensions")?),
            device_exts: std::mem::transmute(f(c"xessVKGetRequiredDeviceExtensions")?),
            create: std::mem::transmute(f(c"xessVKCreateContext")?),
            init: std::mem::transmute(f(c"xessVKInit")?),
            execute: std::mem::transmute(f(c"xessVKExecute")?),
            velocity_scale: std::mem::transmute(f(c"xessSetVelocityScale")?),
            logging: std::mem::transmute(f(c"xessSetLoggingCallback")?),
        };
        crate::info!("upscaler: XeSS loaded from {}", path.display());
        Ok(lib)
    }
}

fn xess_instance_extensions() -> Vec<CString> {
    let Some(x) = xess() else { return Vec::new() };
    let (mut n, mut list, mut api) = (0u32, std::ptr::null(), 0u32);
    let r = unsafe { (x.instance_exts)(&mut n, &mut list, &mut api) };
    if r != 0 {
        fail(Kind::Xess, format!("its instance requirements: error {r}"));
        return Vec::new();
    }
    let v = unsafe { names(n, list) };
    crate::info!("upscaler: XeSS asks for Vulkan {}.{} and instance extensions {:?}", vk::api_version_major(api), vk::api_version_minor(api), v);
    v
}

fn xess_device_needs(instance: &ash::Instance, pdev: vk::PhysicalDevice) -> Option<DeviceNeeds> {
    let x = xess()?;
    if XESS_FAILED.load(Relaxed) {
        return None;
    }
    let (mut n, mut list) = (0u32, std::ptr::null());
    let r = unsafe { (x.device_exts)(instance.handle(), pdev, &mut n, &mut list) };
    if r != 0 {
        fail(Kind::Xess, format!("it cannot run on this device (error {r})"));
        return None;
    }
    let extensions = unsafe { names(n, list) };
    // (The feature its Vulkan path requires.)
    let mut mdt = vk::PhysicalDeviceMutableDescriptorTypeFeaturesEXT::default();
    let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut mdt);
    unsafe { instance.get_physical_device_features2(pdev, &mut f2) };
    if mdt.mutable_descriptor_type == 0 {
        fail(Kind::Xess, "the device lacks mutableDescriptorType");
        return None;
    }
    let avail = unsafe { instance.enumerate_device_extension_properties(pdev) }.unwrap_or_default();
    if let Some(e) = extensions.iter().find(|e| !avail.iter().any(|a| unsafe { CStr::from_ptr(a.extension_name.as_ptr()) } == e.as_c_str())) {
        fail(Kind::Xess, format!("the device lacks {}", e.to_string_lossy()));
        return None;
    }
    crate::info!("upscaler: XeSS asks for device extensions {extensions:?} and mutableDescriptorType");
    Some(DeviceNeeds { extensions, mutable_descriptor: true, features: Vec::new() })
}

unsafe extern "C" fn xess_log(msg: *const c_char, level: i32) {
    if msg.is_null() {
        return;
    }
    let m = unsafe { CStr::from_ptr(msg) }.to_string_lossy();
    if level >= 3 {
        crate::warn!("XeSS: {m}");
    } else {
        crate::info!("XeSS: {m}");
    }
}

/// The context, made at the first use for its sizes.
struct XessState {
    ctx: Handle,
    key: (u32, u32, u32, u32, bool),
    frame: u32,
}

/// XeSS's preset nearest to an output/render ratio (its fixed scalings:
/// 1.0, 1.3, 1.5, 1.7, 2.0, 2.3, 3.0).
fn xess_quality(ratio: f32) -> i32 {
    [(1.0, XESS_QUALITY_AA), (1.3, 105), (1.5, 104), (1.7, 103), (2.0, 102), (2.3, 101), (3.0, 100)]
        .into_iter()
        .min_by(|a, b| (a.0 - ratio).abs().total_cmp(&(b.0 - ratio).abs()))
        .map_or(XESS_QUALITY_AA, |q| q.1)
}

const XESS_INIT_FLAG_LDR_INPUT_COLOR: u32 = 1 << 6;

// SAFETY: the context handle is used under this mutex and on the recorder.
unsafe impl Send for XessState {}

static XESS_STATE: Mutex<Option<XessState>> = Mutex::new(None);

fn xess_view(t: &Tex, w: u32, h: u32) -> XessView {
    XessView { image_view: t.view, image: t.image, range: t.range, format: t.format, width: w, height: h }
}

fn xess_record(cmd: vk::CommandBuffer, inp: &Inputs, reset: bool) -> bool {
    let Some(x) = xess() else { return false };
    let key = (inp.in_width, inp.in_height, inp.width, inp.height, inp.ldr);
    let mut st = XESS_STATE.lock();
    if st.as_ref().is_some_and(|s| s.key != key) {
        // (Other sizes: a new context; the old one leaks - rare.)
        *st = None;
    }
    if st.is_none() {
        let c = super::vk::ctx();
        let mut h: Handle = std::ptr::null_mut();
        let r = unsafe { (x.create)(c.instance.handle(), c.pdev, c.device.handle(), &mut h) };
        if r != 0 || h.is_null() {
            fail(Kind::Xess, format!("xessVKCreateContext failed ({r})"));
            return false;
        }
        unsafe { (x.logging)(h, 2, xess_log) };
        let quality = xess_quality(inp.ratio());
        let flags = if inp.ldr { XESS_INIT_FLAG_LDR_INPUT_COLOR } else { 0 };
        let init = XessInit { output: Xess2d { x: inp.width, y: inp.height }, quality, flags, ..Default::default() };
        let r = unsafe { (x.init)(h, &init) };
        if r != 0 {
            fail(Kind::Xess, format!("xessVKInit failed ({r})"));
            return false;
        }
        // The game's motion vectors: screen fraction, current - previous;
        // XeSS's: pixels, current -> previous.
        let (mx, my) = mv_scale();
        unsafe { (x.velocity_scale)(h, -(inp.in_width as f32) * mx, -(inp.in_height as f32) * my) };
        if inp.ldr {
            crate::info!("upscaler: XeSS upscaling {}x{} to {}x{} (preset {quality}) in place of the game's scaling", inp.in_width, inp.in_height, inp.width, inp.height);
        } else {
            crate::info!("upscaler: XeSS native anti-aliasing at {}x{} in place of SMAA", inp.width, inp.height);
        }
        *st = Some(XessState { ctx: h, key, frame: 0 });
    }
    let s = st.as_mut().unwrap();
    let (jx, jy) = advance_jitter(s.frame, inp.ratio());
    let (sx, sy) = jitter_scale();
    let (w, h) = (inp.in_width, inp.in_height);
    // (XeSS's jitter: the image's shift, as rendered.)
    let p = XessExecute {
        color: xess_view(&inp.color, w, h),
        velocity: xess_view(&inp.velocity, w, h),
        depth: xess_view(&inp.depth, w, h),
        output: xess_view(&inp.output, inp.width, inp.height),
        jitter_x: jx * sx,
        jitter_y: jy * sy,
        exposure_scale: 1.0,
        reset_history: (s.frame == 0 || reset) as u32,
        input_width: w,
        input_height: h,
        ..Default::default()
    };
    s.frame += 1;
    let (h, exec) = (s.ctx as usize, x.execute);
    drop(st);
    unsafe {
        super::rec::REC.cmd_external(move |_| {
            static N: AtomicU32 = AtomicU32::new(0);
            let r = (exec)(h as Handle, cmd, &p);
            if r != 0 {
                warn_some(&N, "xessVKExecute", r);
            }
        })
    };
    true
}

// ---------------------------------------------------------------------------
// Streamline's C interface (sl_struct.h, sl_core_types.h, sl_consts.h,
// sl_dlss.h, sl_helpers_vk.h of SDK 2.14.16; MSVC x64 layout, which
// repr(C) reproduces - the sizes are checked in the tests).

/// sl::StructType (a GUID).
#[repr(C)]
#[derive(Clone, Copy)]
struct SlType(u32, u16, u16, [u8; 8]);

/// sl::BaseStructure.
#[repr(C)]
#[derive(Clone, Copy)]
struct SlBase {
    next: *const SlBase,
    ty: SlType,
    version: usize,
}

impl SlBase {
    const fn new(ty: SlType, version: usize) -> Self {
        Self { next: std::ptr::null(), ty, version }
    }
}

// SAFETY: `next` is always null here (no chained structures): the per-frame
// structures move to the recorder thread as plain data.
unsafe impl Send for SlBase {}

const SL_PREFERENCES: SlType = SlType(0x1ca10965, 0xbf8e, 0x432b, [0x8d, 0xa1, 0x67, 0x16, 0xd8, 0x79, 0xfb, 0x14]);
const SL_RESOURCE: SlType = SlType(0x3a9d70cf, 0x2418, 0x4b72, [0x83, 0x91, 0x13, 0xf8, 0x72, 0x1c, 0x72, 0x61]);
const SL_RESOURCE_TAG: SlType = SlType(0x4c6a5aad, 0xb445, 0x496c, [0x87, 0xff, 0x1a, 0xf3, 0x84, 0x5b, 0xe6, 0x53]);
const SL_VIEWPORT: SlType = SlType(0x171b6435, 0x9b3c, 0x4fc8, [0x99, 0x94, 0xfb, 0xe5, 0x25, 0x69, 0xaa, 0xa4]);
const SL_FEATURE_REQUIREMENTS: SlType = SlType(0x66714097, 0xac6d, 0x4bc6, [0x89, 0x15, 0x1e, 0x0f, 0x55, 0xa6, 0xb6, 0x1f]);
const SL_FEATURE_VERSION: SlType = SlType(0x6d5b51f0, 0x076b, 0x486d, [0x99, 0x95, 0x5a, 0x56, 0x10, 0x43, 0xf5, 0xc1]);
const SL_ADAPTER_INFO: SlType = SlType(0x0677315f, 0xa746, 0x4492, [0x9f, 0x42, 0xcb, 0x61, 0x42, 0xc9, 0xc3, 0xd4]);
const SL_VULKAN_INFO: SlType = SlType(0x0eed6fd5, 0x82cd, 0x43a9, [0xbd, 0xb5, 0x47, 0xa5, 0xba, 0x2f, 0x45, 0xd6]);
const SL_CONSTANTS: SlType = SlType(0xdcd35ad7, 0x4e4a, 0x4bad, [0xa9, 0x0c, 0xe0, 0xc4, 0x9e, 0xb2, 0x3a, 0xfe]);
const SL_DLSS_OPTIONS: SlType = SlType(0x6ac826e4, 0x4c61, 0x4101, [0xa9, 0x2d, 0x63, 0x8d, 0x42, 0x10, 0x57, 0xb8]);

/// sl::kSDKVersion of 2.14.16.
const SL_SDK_VERSION: u64 = (2u64 << 48) | (14u64 << 32) | (16u64 << 16) | 0xfedc;
const SL_FEATURE_DLSS: u32 = 0;
const SL_OK: i32 = 0;
// PreferenceFlags.
const SL_DISABLE_CL_STATE_TRACKING: u64 = 1 << 0;
const SL_USE_MANUAL_HOOKING: u64 = 1 << 2;
// BufferType.
const SL_DEPTH: u32 = 0;
const SL_MOTION_VECTORS: u32 = 1;
const SL_SCALING_INPUT_COLOR: u32 = 3;
const SL_SCALING_OUTPUT_COLOR: u32 = 4;
/// ResourceLifecycle::eValidUntilEvaluate.
const SL_VALID_UNTIL_EVALUATE: u32 = 2;
/// DLSSMode::eDLAA.
const SL_DLSS_DLAA: u32 = 6;

/// DLSS's mode for an output/render ratio (DLAA 1, quality 1.5, balanced
/// 1.72, performance 2, ultra performance 3).
fn dlss_mode(ratio: f32) -> u32 {
    match ratio {
        r if r < 1.2 => SL_DLSS_DLAA,
        r if r < 1.6 => 3,
        r if r < 1.9 => 2,
        r if r < 2.6 => 1,
        _ => 4,
    }
}
/// RenderAPI::eVulkan.
const SL_VULKAN: u32 = 2;
// sl::Boolean.
const SL_FALSE: u8 = 0;
const SL_TRUE: u8 = 1;

/// sl::Preferences.
#[repr(C)]
struct SlPreferences {
    base: SlBase,
    show_console: bool,
    log_level: u32,
    paths_to_plugins: *const *const u16,
    num_paths_to_plugins: u32,
    path_to_logs_and_data: *const u16,
    allocate_callback: *const c_void,
    release_callback: *const c_void,
    log_message_callback: Option<unsafe extern "C" fn(u32, *const c_char)>,
    flags: u64,
    features_to_load: *const u32,
    num_features_to_load: u32,
    application_id: u32,
    engine: u32,
    engine_version: *const c_char,
    project_id: *const c_char,
    render_api: u32,
}

/// sl::Resource (Vulkan: the image, its view, layout and description).
#[repr(C)]
#[derive(Clone, Copy)]
struct SlResource {
    base: SlBase,
    ty: u8,
    native: u64,
    memory: u64,
    view: u64,
    state: u32,
    width: u32,
    height: u32,
    native_format: u32,
    mip_levels: u32,
    array_layers: u32,
    gpu_virtual_address: u64,
    flags: u32,
    usage: u32,
    internal_flags: u16,
    reserved: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SlExtent {
    top: u32,
    left: u32,
    width: u32,
    height: u32,
}

/// sl::ResourceTag.
#[repr(C)]
#[derive(Clone, Copy)]
struct SlResourceTag {
    base: SlBase,
    resource: *const SlResource,
    ty: u32,
    lifecycle: u32,
    extent: SlExtent,
}

/// sl::ViewportHandle.
#[repr(C)]
#[derive(Clone, Copy)]
struct SlViewport {
    base: SlBase,
    value: u32,
}

/// sl::FeatureRequirements (version 2).
#[repr(C)]
struct SlFeatureRequirements {
    base: SlBase,
    flags: u32,
    max_cpu_threads: u32,
    max_viewports: u32,
    num_required_tags: u32,
    required_tags: *const u32,
    os_detected: [u32; 3],
    os_required: [u32; 3],
    driver_detected: [u32; 3],
    driver_required: [u32; 3],
    vk_compute_queues: u32,
    vk_graphics_queues: u32,
    vk_num_device_extensions: u32,
    vk_device_extensions: *const *const c_char,
    vk_num_instance_extensions: u32,
    vk_instance_extensions: *const *const c_char,
    vk_num_features12: u32,
    vk_features12: *const *const c_char,
    vk_num_features13: u32,
    vk_features13: *const *const c_char,
    vk_optical_flow_queues: u32,
}

/// sl::FeatureVersion.
#[repr(C)]
struct SlFeatureVersion {
    base: SlBase,
    sl: [u32; 3],
    ngx: [u32; 3],
}

/// sl::AdapterInfo.
#[repr(C)]
struct SlAdapterInfo {
    base: SlBase,
    luid: *const u8,
    luid_size: u32,
    vk_physical_device: vk::PhysicalDevice,
}

/// sl::VulkanInfo (version 3).
#[repr(C)]
struct SlVulkanInfo {
    base: SlBase,
    device: vk::Device,
    instance: vk::Instance,
    physical_device: vk::PhysicalDevice,
    compute_queue_index: u32,
    compute_queue_family: u32,
    graphics_queue_index: u32,
    graphics_queue_family: u32,
    optical_flow_queue_index: u32,
    optical_flow_queue_family: u32,
    use_native_optical_flow: bool,
    compute_queue_create_flags: u32,
    graphics_queue_create_flags: u32,
    optical_flow_queue_create_flags: u32,
}

/// A row-major 4x4 matrix (sl::float4x4).
type SlMatrix = [[f32; 4]; 4];

const IDENTITY: SlMatrix = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];

/// sl::Constants (version 2).
#[repr(C)]
#[derive(Clone, Copy)]
struct SlConstants {
    base: SlBase,
    camera_view_to_clip: SlMatrix,
    clip_to_camera_view: SlMatrix,
    clip_to_lens_clip: SlMatrix,
    clip_to_prev_clip: SlMatrix,
    prev_clip_to_clip: SlMatrix,
    jitter_offset: [f32; 2],
    mvec_scale: [f32; 2],
    camera_pinhole_offset: [f32; 2],
    camera_pos: [f32; 3],
    camera_up: [f32; 3],
    camera_right: [f32; 3],
    camera_fwd: [f32; 3],
    camera_near: f32,
    camera_far: f32,
    camera_fov: f32,
    camera_aspect_ratio: f32,
    motion_vectors_invalid_value: f32,
    depth_inverted: u8,
    camera_motion_included: u8,
    motion_vectors_3d: u8,
    reset: u8,
    orthographic_projection: u8,
    motion_vectors_dilated: u8,
    motion_vectors_jittered: u8,
    min_relative_linear_depth_object_separation: f32,
}

/// sl::DLSSOptions (version 3).
#[repr(C)]
#[derive(Clone, Copy)]
struct SlDlssOptions {
    base: SlBase,
    mode: u32,
    output_width: u32,
    output_height: u32,
    sharpness: f32,
    pre_exposure: f32,
    exposure_scale: f32,
    color_buffers_hdr: u8,
    indicator_invert_x: u8,
    indicator_invert_y: u8,
    dlaa_preset: u32,
    quality_preset: u32,
    balanced_preset: u32,
    performance_preset: u32,
    ultra_performance_preset: u32,
    ultra_quality_preset: u32,
    use_auto_exposure: u8,
    alpha_upscaling_enabled: u8,
}

/// The port's NGX project id (Streamline passes it with the engine type
/// "custom" when no NVIDIA application id is given).
const NGX_PROJECT_ID: &CStr = c"48421953-2aae-4cec-9b59-7e92288edb4a";

/// DLSS's Vulkan needs (slGetFeatureRequirements).
#[derive(Default)]
struct DlssReqs {
    instance_ext: Vec<CString>,
    device_ext: Vec<CString>,
    features12: Vec<String>,
    features13: Vec<String>,
    queues: u32,
}

struct SlLib {
    is_supported: unsafe extern "C" fn(u32, *const SlAdapterInfo) -> i32,
    set_vulkan_info: unsafe extern "C" fn(*const SlVulkanInfo) -> i32,
    frame_token: unsafe extern "C" fn(*mut *const c_void, *const u32) -> i32,
    set_constants: unsafe extern "C" fn(*const SlConstants, *const c_void, *const SlViewport) -> i32,
    evaluate: unsafe extern "C" fn(u32, *const c_void, *const *const SlBase, u32, vk::CommandBuffer) -> i32,
    feature_function: unsafe extern "C" fn(u32, *const c_char, *mut *const c_void) -> i32,
    version: unsafe extern "C" fn(u32, *mut SlFeatureVersion) -> i32,
    gipa: vk::PFN_vkGetInstanceProcAddr,
    reqs: DlssReqs,
}

// SAFETY: plain function pointers into a library that stays loaded.
unsafe impl Send for SlLib {}
unsafe impl Sync for SlLib {}

/// Streamline, initialised with DLSS (None: not prepared, not installed or
/// unusable).
fn sl() -> Option<&'static SlLib> {
    static L: std::sync::OnceLock<Option<SlLib>> = std::sync::OnceLock::new();
    L.get_or_init(|| {
        if !prepares(Kind::Dlss) {
            return None;
        }
        match load_sl() {
            Ok(l) => Some(l),
            Err(LoadError::Missing) => {
                crate::info!("upscaler: DLSS not installed (nvngx_dlss.dll - the player's own - and Streamline in the upscalers folder)");
                None
            }
            Err(LoadError::Failed(e)) => {
                fail(Kind::Dlss, e);
                None
            }
        }
    })
    .as_ref()
}

unsafe extern "C" fn sl_log(ty: u32, msg: *const c_char) {
    if msg.is_null() {
        return;
    }
    let m = unsafe { CStr::from_ptr(msg) }.to_string_lossy();
    let m = m.trim_end();
    // (Its Reflex low-latency library is not shipped - the port uses no
    // Reflex: that failure is expected.)
    let ty = if m.contains("NvLowLatencyVk.dll") { 0 } else { ty };
    match ty {
        2 => crate::warn!("Streamline error: {m}"),
        1 => crate::warn!("Streamline: {m}"),
        _ => crate::debug!("Streamline: {m}"),
    }
}

fn load_sl() -> Result<SlLib, LoadError> {
    let dir = std::env::var_os("TLG_SL_DIR").map(PathBuf::from).unwrap_or_else(lib_dir);
    // DLSS itself: the player's own nvngx_dlss.dll (from a game with DLSS or
    // NVIDIA's DLSS SDK; NVIDIA's license keeps it out of the package), signed
    // by NVIDIA. Without it Streamline is not started.
    let ngx = dir.join("nvngx_dlss.dll");
    if !ngx.is_file() {
        return Err(LoadError::Missing);
    }
    check_signature(&ngx, "NVIDIA Corporation")?;
    let path = dir.join("sl.interposer.dll");
    let h = load_lib(&path, "NVIDIA Corporation")?;
    let f = |name: &CStr| export(h, &path, name);
    // SAFETY: the signatures of sl_core_api.h / sl_helpers_vk.h.
    unsafe {
        let init: unsafe extern "C" fn(*const SlPreferences, u64) -> i32 = std::mem::transmute(f(c"slInit")?);
        let requirements: unsafe extern "C" fn(u32, *mut SlFeatureRequirements) -> i32 = std::mem::transmute(f(c"slGetFeatureRequirements")?);
        let mut lib = SlLib {
            is_supported: std::mem::transmute(f(c"slIsFeatureSupported")?),
            set_vulkan_info: std::mem::transmute(f(c"slSetVulkanInfo")?),
            frame_token: std::mem::transmute(f(c"slGetNewFrameToken")?),
            set_constants: std::mem::transmute(f(c"slSetConstants")?),
            evaluate: std::mem::transmute(f(c"slEvaluateFeature")?),
            feature_function: std::mem::transmute(f(c"slGetFeatureFunction")?),
            version: std::mem::transmute(f(c"slGetFeatureVersion")?),
            gipa: std::mem::transmute(f(c"vkGetInstanceProcAddr")?),
            reqs: DlssReqs::default(),
        };
        // slInit: DLSS only, the port restores its own command buffer state
        // (it does after any foreign recording), manual hooking, no
        // over-the-air downloads, logs into the port's log.
        let t0 = std::time::Instant::now();
        let dir_w = dir.as_os_str().encode_wide_null();
        let paths = [dir_w.as_ptr()];
        let features = [SL_FEATURE_DLSS];
        let prefs = SlPreferences {
            base: SlBase::new(SL_PREFERENCES, 1),
            show_console: false,
            log_level: 1,
            paths_to_plugins: paths.as_ptr(),
            num_paths_to_plugins: 1,
            path_to_logs_and_data: std::ptr::null(),
            allocate_callback: std::ptr::null(),
            release_callback: std::ptr::null(),
            log_message_callback: Some(sl_log),
            flags: SL_DISABLE_CL_STATE_TRACKING | SL_USE_MANUAL_HOOKING,
            features_to_load: features.as_ptr(),
            num_features_to_load: 1,
            application_id: 0,
            engine: 0,
            engine_version: c"1.0".as_ptr(),
            project_id: NGX_PROJECT_ID.as_ptr(),
            render_api: SL_VULKAN,
        };
        let r = init(&prefs, SL_SDK_VERSION);
        if r != SL_OK {
            return Err(format!("slInit failed ({r})").into());
        }
        let mut reqs: SlFeatureRequirements = std::mem::zeroed();
        reqs.base = SlBase::new(SL_FEATURE_REQUIREMENTS, 2);
        let r = requirements(SL_FEATURE_DLSS, &mut reqs);
        if r != SL_OK {
            return Err(format!("no DLSS here ({r}): it needs an NVIDIA RTX GPU and driver").into());
        }
        // (FeatureRequirementFlags::eVulkanSupported.)
        if reqs.flags & 4 == 0 {
            return Err("DLSS does not support Vulkan here".to_string().into());
        }
        let v = |x: [u32; 3]| format!("{}.{}.{}", x[0], x[1], x[2]);
        crate::info!(
            "upscaler: Streamline loaded from {} in {:.2} s (OS {} needs {}, driver {} needs {})",
            dir.display(),
            t0.elapsed().as_secs_f32(),
            v(reqs.os_detected),
            v(reqs.os_required),
            v(reqs.driver_detected),
            v(reqs.driver_required)
        );
        let strings = |n: u32, p: *const *const c_char| names(n, p).into_iter().map(|c| c.to_string_lossy().into_owned()).collect::<Vec<_>>();
        lib.reqs = DlssReqs {
            instance_ext: names(reqs.vk_num_instance_extensions, reqs.vk_instance_extensions),
            device_ext: names(reqs.vk_num_device_extensions, reqs.vk_device_extensions),
            features12: strings(reqs.vk_num_features12, reqs.vk_features12),
            features13: strings(reqs.vk_num_features13, reqs.vk_features13),
            queues: reqs.vk_compute_queues + reqs.vk_graphics_queues + reqs.vk_optical_flow_queues,
        };
        crate::info!(
            "upscaler: DLSS asks for instance extensions {:?}, device extensions {:?}, Vulkan 1.2 features {:?}, 1.3 features {:?}, {} extra queues",
            lib.reqs.instance_ext,
            lib.reqs.device_ext,
            lib.reqs.features12,
            lib.reqs.features13,
            lib.reqs.queues
        );
        Ok(lib)
    }
}

fn dlss_device_needs(instance: &ash::Instance, pdev: vk::PhysicalDevice) -> Option<DeviceNeeds> {
    let s = sl()?;
    if DLSS_FAILED.load(Relaxed) {
        return None;
    }
    let info = SlAdapterInfo { base: SlBase::new(SL_ADAPTER_INFO, 1), luid: std::ptr::null(), luid_size: 0, vk_physical_device: pdev };
    let r = unsafe { (s.is_supported)(SL_FEATURE_DLSS, &info) };
    if r != SL_OK {
        fail(Kind::Dlss, format!("not supported on this GPU or driver (Streamline result {r})"));
        return None;
    }
    if s.reqs.queues > 0 || !s.reqs.features13.is_empty() {
        fail(Kind::Dlss, format!("it asks for {} extra queues and 1.3 features {:?}, which the port does not provide", s.reqs.queues, s.reqs.features13));
        return None;
    }
    if let Err(e) = features_supported(instance, pdev, &s.reqs.features12) {
        fail(Kind::Dlss, e);
        return None;
    }
    // The device's extensions. (VK_EXT_buffer_device_address must not be
    // enabled with the 1.2 bufferDeviceAddress feature: the KHR extension
    // in its place, as Streamline's own device creation does.)
    let avail = unsafe { instance.enumerate_device_extension_properties(pdev) }.unwrap_or_default();
    let has = |n: &CStr| avail.iter().any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == n);
    let mut extensions = Vec::new();
    for e in &s.reqs.device_ext {
        let e = if e.as_c_str() == ash::ext::buffer_device_address::NAME { ash::khr::buffer_device_address::NAME.to_owned() } else { e.clone() };
        if !has(&e) {
            fail(Kind::Dlss, format!("the device lacks {}", e.to_string_lossy()));
            return None;
        }
        if !extensions.contains(&e) {
            extensions.push(e);
        }
    }
    Some(DeviceNeeds { extensions, mutable_descriptor: false, features: s.reqs.features12.clone() })
}

/// Hands the device to Streamline (slSetVulkanInfo, which starts NGX) on a
/// thread of its own; DLSS is ready - offered and its present hook used -
/// once that is done.
fn dlss_device_created(instance: &ash::Instance, pdev: vk::PhysicalDevice, device: &ash::Device, qf: u32, queues: u32) {
    let Some(s) = sl() else { return };
    // (No extra queues: Streamline's would start after the port's.)
    let info = SlVulkanInfo {
        base: SlBase::new(SL_VULKAN_INFO, 3),
        device: device.handle(),
        instance: instance.handle(),
        physical_device: pdev,
        compute_queue_index: queues,
        compute_queue_family: qf,
        graphics_queue_index: queues,
        graphics_queue_family: qf,
        optical_flow_queue_index: 0,
        optical_flow_queue_family: 0,
        use_native_optical_flow: false,
        compute_queue_create_flags: 0,
        graphics_queue_create_flags: 0,
        optical_flow_queue_create_flags: 0,
    };
    let spawned = std::thread::Builder::new().name("dlss-init".into()).spawn(move || {
        let t0 = std::time::Instant::now();
        let r = unsafe { (s.set_vulkan_info)(&info) };
        if r != SL_OK {
            fail(Kind::Dlss, format!("slSetVulkanInfo failed ({r})"));
            return;
        }
        // Streamline's vkQueuePresentKHR, through its vkGetDeviceProcAddr.
        // SAFETY: Streamline's loader functions, valid for this device.
        let present = unsafe {
            let gdpa: Option<vk::PFN_vkGetDeviceProcAddr> = std::mem::transmute((s.gipa)(info.instance, c"vkGetDeviceProcAddr".as_ptr()));
            gdpa.and_then(|g| g(info.device, c"vkQueuePresentKHR".as_ptr())).map(|p| std::mem::transmute::<unsafe extern "system" fn(), vk::PFN_vkQueuePresentKHR>(p))
        };
        let Some(present) = present else {
            fail(Kind::Dlss, "no vkQueuePresentKHR from Streamline");
            return;
        };
        let _ = SL_PRESENT.set(present);
        let mut ver: SlFeatureVersion = unsafe { std::mem::zeroed() };
        ver.base = SlBase::new(SL_FEATURE_VERSION, 1);
        let ngx = if unsafe { (s.version)(SL_FEATURE_DLSS, &mut ver) } == SL_OK { format!(", NGX DLSS {}.{}.{}", ver.ngx[0], ver.ngx[1], ver.ngx[2]) } else { String::new() };
        DLSS_READY.store(true, Relaxed);
        crate::info!("upscaler: DLSS ready in {:.2} s (Streamline {}.{}.{}{ngx})", t0.elapsed().as_secs_f32(), ver.sl[0], ver.sl[1], ver.sl[2]);
    });
    if let Err(e) = spawned {
        fail(Kind::Dlss, format!("no thread for its start ({e})"));
    }
}

/// slDLSSSetOptions (from the plugin, once the device is set).
type DlssSetOptions = unsafe extern "C" fn(*const SlViewport, *const SlDlssOptions) -> i32;

struct DlssState {
    set_options: DlssSetOptions,
    key: (u32, u32, u32, u32, bool),
    frame: u32,
}

static DLSS_STATE: Mutex<Option<DlssState>> = Mutex::new(None);

fn sl_resource(t: &Tex) -> SlResource {
    use ash::vk::Handle;
    SlResource {
        base: SlBase::new(SL_RESOURCE, 1),
        ty: 0,
        native: t.image.as_raw(),
        memory: 0,
        view: t.view.as_raw(),
        state: t.layout.as_raw() as u32,
        width: t.size.0,
        height: t.size.1,
        native_format: t.format.as_raw() as u32,
        mip_levels: 1,
        array_layers: 1,
        gpu_virtual_address: 0,
        flags: 0,
        usage: t.usage.as_raw(),
        internal_flags: 0,
        reserved: 0,
    }
}

fn dlss_record(cmd: vk::CommandBuffer, inp: &Inputs, reset: bool) -> bool {
    let Some(s) = sl() else { return false };
    let mut st = DLSS_STATE.lock();
    if st.is_none() {
        let mut p: *const c_void = std::ptr::null();
        let r = unsafe { (s.feature_function)(SL_FEATURE_DLSS, c"slDLSSSetOptions".as_ptr(), &mut p) };
        if r != SL_OK || p.is_null() {
            fail(Kind::Dlss, format!("no slDLSSSetOptions ({r})"));
            return false;
        }
        // SAFETY: sl_dlss.h's PFun_slDLSSSetOptions.
        let set_options: DlssSetOptions = unsafe { std::mem::transmute(p) };
        *st = Some(DlssState { set_options, key: (0, 0, 0, 0, false), frame: 0 });
    }
    let d = st.as_mut().unwrap();
    let key = (inp.in_width, inp.in_height, inp.width, inp.height, inp.ldr);
    let new_size = d.key != key;
    let mode = dlss_mode(inp.ratio());
    if new_size {
        if inp.ldr {
            crate::info!("upscaler: DLSS upscaling {}x{} to {}x{} (mode {mode}) in place of the game's scaling", inp.in_width, inp.in_height, inp.width, inp.height);
        } else {
            crate::info!("upscaler: DLSS native anti-aliasing (DLAA) at {}x{} in place of SMAA", inp.width, inp.height);
        }
        d.key = key;
    }
    let (jx, jy) = advance_jitter(d.frame, inp.ratio());
    let (sx, sy) = jitter_scale();
    // The game's motion vectors: screen fraction, current - previous;
    // Streamline's: scaled into [-1, 1] screen fractions, current ->
    // previous.
    let (mx, my) = mv_scale();
    let reset = reset || d.frame == 0 || new_size;
    d.frame += 1;
    let set_options = d.set_options;
    drop(st);
    let consts = SlConstants {
        base: SlBase::new(SL_CONSTANTS, 2),
        // (DLSS's upscaling reads the jitter and the motion vectors; the
        // camera's matrices and vectors are for frame generation: neutral
        // values.)
        camera_view_to_clip: IDENTITY,
        clip_to_camera_view: IDENTITY,
        clip_to_lens_clip: IDENTITY,
        clip_to_prev_clip: IDENTITY,
        prev_clip_to_clip: IDENTITY,
        // (DLSS's jitter: the image's shift, as rendered.)
        jitter_offset: [jx * sx, jy * sy],
        mvec_scale: [-mx, -my],
        camera_pinhole_offset: [0.0, 0.0],
        camera_pos: [0.0, 0.0, 0.0],
        camera_up: [0.0, 1.0, 0.0],
        camera_right: [1.0, 0.0, 0.0],
        camera_fwd: [0.0, 0.0, 1.0],
        camera_near: 0.1,
        camera_far: 10000.0,
        camera_fov: 1.0,
        camera_aspect_ratio: inp.in_width as f32 / inp.in_height.max(1) as f32,
        motion_vectors_invalid_value: 0.0,
        depth_inverted: SL_FALSE,
        camera_motion_included: SL_TRUE,
        motion_vectors_3d: SL_FALSE,
        reset: if reset { SL_TRUE } else { SL_FALSE },
        orthographic_projection: SL_FALSE,
        motion_vectors_dilated: SL_FALSE,
        motion_vectors_jittered: SL_FALSE,
        min_relative_linear_depth_object_separation: 40.0,
    };
    let opts = SlDlssOptions {
        base: SlBase::new(SL_DLSS_OPTIONS, 3),
        mode,
        output_width: inp.width,
        output_height: inp.height,
        sharpness: 0.0,
        pre_exposure: 1.0,
        exposure_scale: 1.0,
        color_buffers_hdr: if inp.ldr { SL_FALSE } else { SL_TRUE },
        indicator_invert_x: SL_FALSE,
        indicator_invert_y: SL_FALSE,
        dlaa_preset: 0,
        quality_preset: 0,
        balanced_preset: 0,
        performance_preset: 0,
        ultra_performance_preset: 0,
        ultra_quality_preset: 0,
        use_auto_exposure: if inp.ldr { SL_FALSE } else { SL_TRUE },
        alpha_upscaling_enabled: SL_FALSE,
    };
    let res = [sl_resource(&inp.depth), sl_resource(&inp.velocity), sl_resource(&inp.color), sl_resource(&inp.output)];
    let extent = SlExtent { top: 0, left: 0, width: inp.in_width, height: inp.in_height };
    let out_extent = SlExtent { top: 0, left: 0, width: inp.width, height: inp.height };
    let (frame_token, set_constants, evaluate) = (s.frame_token, s.set_constants, s.evaluate);
    unsafe {
        super::rec::REC.cmd_external(move |_| {
            static N_TOKEN: AtomicU32 = AtomicU32::new(0);
            static N_CONST: AtomicU32 = AtomicU32::new(0);
            static N_OPT: AtomicU32 = AtomicU32::new(0);
            static N_EVAL: AtomicU32 = AtomicU32::new(0);
            let viewport = SlViewport { base: SlBase::new(SL_VIEWPORT, 1), value: 0 };
            let mut token: *const c_void = std::ptr::null();
            let r = frame_token(&mut token, std::ptr::null());
            if r != SL_OK || token.is_null() {
                warn_some(&N_TOKEN, "slGetNewFrameToken", r);
                return;
            }
            let r = set_constants(&consts, token, &viewport);
            if r != SL_OK {
                warn_some(&N_CONST, "slSetConstants", r);
            }
            let r = set_options(&viewport, &opts);
            if r != SL_OK {
                warn_some(&N_OPT, "slDLSSSetOptions", r);
            }
            let tag = |i: usize, ty: u32| SlResourceTag { base: SlBase::new(SL_RESOURCE_TAG, 1), resource: &res[i], ty, lifecycle: SL_VALID_UNTIL_EVALUATE, extent: if i == 3 { out_extent } else { extent } };
            let tags = [tag(0, SL_DEPTH), tag(1, SL_MOTION_VECTORS), tag(2, SL_SCALING_INPUT_COLOR), tag(3, SL_SCALING_OUTPUT_COLOR)];
            let inputs: [*const SlBase; 5] = [&viewport.base, &tags[0].base, &tags[1].base, &tags[2].base, &tags[3].base];
            let r = evaluate(SL_FEATURE_DLSS, token, inputs.as_ptr(), inputs.len() as u32, cmd);
            if r != SL_OK {
                warn_some(&N_EVAL, "slEvaluateFeature (DLSS)", r);
            }
        })
    };
    true
}

// ---------------------------------------------------------------------------
// AMD's FSR API (ffx_api.h, ffx_api_types.h, vk/ffx_api_vk.h, ffx_upscale.h
// of FSR SDK 2.4.0; natural C layout - the sizes are checked in the tests).

/// ffxApiHeader.
#[repr(C)]
#[derive(Clone, Copy)]
struct FfxHeader {
    ty: u64,
    next: *mut FfxHeader,
}

// SAFETY: `next` is null except while a creation chain is passed.
unsafe impl Send for FfxHeader {}

impl FfxHeader {
    const fn new(ty: u64) -> Self {
        Self { ty, next: std::ptr::null_mut() }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FfxDims {
    width: u32,
    height: u32,
}

/// FfxApiResourceDescription (the unions as their first members).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FfxResDesc {
    ty: u32,
    format: u32,
    width: u32,
    height: u32,
    depth: u32,
    mip_count: u32,
    flags: u32,
    usage: u32,
}

/// FfxApiResource (Vulkan: the VkImage).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct FfxResource {
    resource: u64,
    desc: FfxResDesc,
    state: u32,
}

/// ffxCreateContextDescUpscale.
#[repr(C)]
struct FfxCreateUpscale {
    header: FfxHeader,
    flags: u32,
    max_render_size: FfxDims,
    max_upscale_size: FfxDims,
    message: Option<unsafe extern "C" fn(u32, *const u16)>,
}

/// ffxCreateBackendVKDesc (its VkDeviceContext inline).
#[repr(C)]
struct FfxBackendVk {
    header: FfxHeader,
    device: vk::Device,
    physical_device: vk::PhysicalDevice,
    get_device_proc_addr: vk::PFN_vkGetDeviceProcAddr,
}

/// ffxDispatchDescUpscale.
#[repr(C)]
#[derive(Clone, Copy)]
struct FfxDispatchUpscale {
    header: FfxHeader,
    command_list: vk::CommandBuffer,
    color: FfxResource,
    depth: FfxResource,
    motion_vectors: FfxResource,
    exposure: FfxResource,
    reactive: FfxResource,
    transparency_and_composition: FfxResource,
    output: FfxResource,
    jitter_offset: [f32; 2],
    motion_vector_scale: [f32; 2],
    render_size: FfxDims,
    upscale_size: FfxDims,
    enable_sharpening: bool,
    sharpness: f32,
    frame_time_delta: f32,
    pre_exposure: f32,
    reset: bool,
    camera_near: f32,
    camera_far: f32,
    camera_fov_angle_vertical: f32,
    view_space_to_meters_factor: f32,
    flags: u32,
}

/// ffxQueryGetProviderVersion.
#[repr(C)]
struct FfxProviderVersion {
    header: FfxHeader,
    version_id: u64,
    version_name: *const c_char,
}

const FFX_CREATE_UPSCALE: u64 = 0x0001_0000;
const FFX_DISPATCH_UPSCALE: u64 = 0x0001_0001;
const FFX_BACKEND_VK: u64 = 0x0100_0003;
const FFX_QUERY_PROVIDER_VERSION: u64 = 6;
const FFX_OK: u32 = 0;
// FfxApiCreateContextUpscaleFlags; FfxApiDispatchFsrUpscaleFlags.
const FFX_UPSCALE_HDR: u32 = 1 << 0;
const FFX_UPSCALE_AUTO_EXPOSURE: u32 = 1 << 5;
const FFX_UPSCALE_NON_LINEAR_COLORSPACE: u32 = 1 << 8;
const FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB: u32 = 1 << 1;
// FfxApiResourceType, -Usage, -State.
const FFX_TEXTURE2D: u32 = 2;
const FFX_USAGE_UAV: u32 = 1 << 1;
const FFX_USAGE_DEPTH: u32 = 1 << 2;
const FFX_USAGE_STENCIL: u32 = 1 << 5;
const FFX_STATE_UNORDERED_ACCESS: u32 = 1 << 1;
const FFX_STATE_COMPUTE_READ: u32 = 1 << 2;

type FfxContext = *mut c_void;

/// ffxConfigureDescGlobalDebug.
#[repr(C)]
struct FfxGlobalDebug {
    header: FfxHeader,
    effect_id: u64,
    message: Option<unsafe extern "C" fn(u32, *const u16)>,
    debug_level: u32,
}

const FFX_CONFIGURE_GLOBAL_DEBUG: u64 = 7;
const FFX_EFFECT_UPSCALE: u64 = 0x0001_0000;
const FFX_UPSCALE_DEBUG_CHECKING: u32 = 1 << 7;

struct FfxLib {
    configure: unsafe extern "C" fn(*mut FfxContext, *const FfxHeader) -> u32,
    create: unsafe extern "C" fn(*mut FfxContext, *mut FfxHeader, *const c_void) -> u32,
    query: unsafe extern "C" fn(*mut FfxContext, *mut FfxHeader) -> u32,
    dispatch: unsafe extern "C" fn(*mut FfxContext, *const FfxHeader) -> u32,
}

// SAFETY: plain function pointers into a library that stays loaded.
unsafe impl Send for FfxLib {}
unsafe impl Sync for FfxLib {}

/// FSR's loader library (None: not prepared, not installed or unusable).
fn ffx() -> Option<&'static FfxLib> {
    static L: std::sync::OnceLock<Option<FfxLib>> = std::sync::OnceLock::new();
    L.get_or_init(|| {
        if !prepares(Kind::Fsr) {
            return None;
        }
        match load_ffx() {
            Ok(l) => Some(l),
            Err(LoadError::Missing) => {
                crate::info!("upscaler: FSR not installed");
                None
            }
            Err(LoadError::Failed(e)) => {
                fail(Kind::Fsr, e);
                None
            }
        }
    })
    .as_ref()
}

fn load_ffx() -> Result<FfxLib, LoadError> {
    let dir = std::env::var_os("TLG_FFX_DIR").map(PathBuf::from).unwrap_or_else(lib_dir);
    // (The loader loads the effect library beside it: both are checked.)
    let effect = dir.join("amd_fidelityfx_upscaler_vk.dll");
    if !effect.is_file() {
        return Err(LoadError::Missing);
    }
    check_signature(&effect, "Advanced Micro Devices")?;
    let path = dir.join("amd_fidelityfx_loader_vk.dll");
    let h = load_lib(&path, "Advanced Micro Devices")?;
    let f = |name: &CStr| export(h, &path, name);
    // SAFETY: the signatures of ffx_api.h.
    unsafe {
        let lib = FfxLib {
            configure: std::mem::transmute(f(c"ffxConfigure")?),
            create: std::mem::transmute(f(c"ffxCreateContext")?),
            query: std::mem::transmute(f(c"ffxQuery")?),
            dispatch: std::mem::transmute(f(c"ffxDispatch")?),
        };
        crate::info!("upscaler: FSR loaded from {}", dir.display());
        Ok(lib)
    }
}

/// FSR's Vulkan backend enables nothing itself and uses what the device
/// offers: these features where the device has them.
fn fsr_device_needs(instance: &ash::Instance, pdev: vk::PhysicalDevice) -> Option<DeviceNeeds> {
    ffx()?;
    if FSR_FAILED.load(Relaxed) {
        return None;
    }
    let want = [
        "shaderFloat16",
        "shaderInt16",
        "storageBuffer16BitAccess",
        "uniformAndStorageBuffer16BitAccess",
        "shaderStorageBufferArrayNonUniformIndexing",
        "subgroupSizeControl",
        "computeFullSubgroups",
    ];
    let have = device_features(instance, pdev, &want);
    let features: Vec<String> = want.iter().zip(have).filter(|(_, h)| *h == Some(true)).map(|(n, _)| n.to_string()).collect();
    crate::info!("upscaler: FSR gets the device features {features:?}");
    Some(DeviceNeeds { extensions: Vec::new(), mutable_descriptor: false, features })
}

unsafe extern "C" fn ffx_log(ty: u32, msg: *const u16) {
    if msg.is_null() {
        return;
    }
    let n = (0..).take_while(|&i| unsafe { *msg.add(i) } != 0).count();
    let m = String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(msg, n) });
    match ty {
        0 => crate::warn!("FSR error: {}", m.trim_end()),
        1 => crate::warn!("FSR: {}", m.trim_end()),
        _ => crate::info!("FSR: {}", m.trim_end()),
    }
}

/// TLG_FSR_DEBUG=1: FSR's parameter checks and all its messages.
fn fsr_debug() -> bool {
    static D: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *D.get_or_init(|| std::env::var("TLG_FSR_DEBUG").is_ok_and(|v| v == "1"))
}

/// FfxApiSurfaceFormat of the formats the hook hands over (0: none).
fn ffx_format(f: vk::Format) -> u32 {
    match f {
        vk::Format::R16G16B16A16_SFLOAT => 4,
        vk::Format::R32G32_SFLOAT => 6,
        vk::Format::D24_UNORM_S8_UINT | vk::Format::X8_D24_UNORM_PACK32 => 8,
        vk::Format::R8G8B8A8_UNORM => 10,
        vk::Format::B8G8R8A8_UNORM => 14,
        vk::Format::B10G11R11_UFLOAT_PACK32 => 16,
        vk::Format::A2B10G10R10_UNORM_PACK32 => 17,
        vk::Format::R16G16_SFLOAT => 18,
        vk::Format::R16_UNORM | vk::Format::D16_UNORM | vk::Format::D16_UNORM_S8_UINT => 23,
        vk::Format::R32_SFLOAT | vk::Format::D32_SFLOAT | vk::Format::D32_SFLOAT_S8_UINT => 28,
        _ => 0,
    }
}

fn ffx_resource(t: &Tex, state: u32) -> FfxResource {
    use ash::vk::Handle;
    let depth = matches!(t.format, vk::Format::D16_UNORM | vk::Format::D16_UNORM_S8_UINT | vk::Format::D24_UNORM_S8_UINT | vk::Format::X8_D24_UNORM_PACK32 | vk::Format::D32_SFLOAT | vk::Format::D32_SFLOAT_S8_UINT);
    let stencil = matches!(t.format, vk::Format::D16_UNORM_S8_UINT | vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT);
    let mut usage = 0;
    if depth {
        usage |= FFX_USAGE_DEPTH;
    }
    if stencil {
        usage |= FFX_USAGE_STENCIL;
    }
    if t.usage.contains(vk::ImageUsageFlags::STORAGE) {
        usage |= FFX_USAGE_UAV;
    }
    FfxResource {
        resource: t.image.as_raw(),
        desc: FfxResDesc { ty: FFX_TEXTURE2D, format: ffx_format(t.format), width: t.size.0, height: t.size.1, depth: 1, mip_count: 1, flags: 0, usage },
        state,
    }
}

/// The context (made at the first use for its sizes) and the frame clock.
struct FsrState {
    ctx: usize,
    key: (u32, u32, u32, u32, bool),
    frame: u32,
    last: std::time::Instant,
}

static FSR_STATE: Mutex<Option<FsrState>> = Mutex::new(None);

fn fsr_record(cmd: vk::CommandBuffer, inp: &Inputs, reset: bool) -> bool {
    let Some(f) = ffx() else { return false };
    for t in [&inp.color, &inp.depth, &inp.velocity, &inp.output] {
        if ffx_format(t.format) == 0 {
            fail(Kind::Fsr, format!("no FSR format for {:?}", t.format));
            return false;
        }
    }
    let key = (inp.in_width, inp.in_height, inp.width, inp.height, inp.ldr);
    let mut st = FSR_STATE.lock();
    if st.as_ref().is_some_and(|s| s.key != key) {
        // (Other sizes: a new context; the old one leaks - rare.)
        *st = None;
    }
    if st.is_none() {
        let c = super::vk::ctx();
        let mut backend = FfxBackendVk {
            header: FfxHeader::new(FFX_BACKEND_VK),
            device: c.device.handle(),
            physical_device: c.pdev,
            get_device_proc_addr: c.instance.fp_v1_0().get_device_proc_addr,
        };
        // (HDR scene colour with auto exposure; the upscaling path's
        // finished picture is sRGB-encoded LDR.)
        let colour = if inp.ldr { FFX_UPSCALE_NON_LINEAR_COLORSPACE } else { FFX_UPSCALE_HDR | FFX_UPSCALE_AUTO_EXPOSURE };
        let mut desc = FfxCreateUpscale {
            header: FfxHeader { ty: FFX_CREATE_UPSCALE, next: &mut backend.header },
            flags: colour | if fsr_debug() { FFX_UPSCALE_DEBUG_CHECKING } else { 0 },
            max_render_size: FfxDims { width: inp.in_width, height: inp.in_height },
            max_upscale_size: FfxDims { width: inp.width, height: inp.height },
            message: Some(ffx_log),
        };
        let mut ctx: FfxContext = std::ptr::null_mut();
        let r = unsafe { (f.create)(&mut ctx, &mut desc.header, std::ptr::null()) };
        if r != FFX_OK || ctx.is_null() {
            fail(Kind::Fsr, format!("ffxCreateContext failed ({r})"));
            return false;
        }
        // Its messages to the port's log: errors and warnings (the
        // context's own through `message`), everything with
        // TLG_FSR_DEBUG=1 - the loader's and the upscaler effect's.
        for effect_id in [0, FFX_EFFECT_UPSCALE] {
            let dbg = FfxGlobalDebug { header: FfxHeader::new(FFX_CONFIGURE_GLOBAL_DEBUG), effect_id, message: Some(ffx_log), debug_level: if fsr_debug() { 0x0fff_ffff } else { 2 } };
            let r = unsafe { (f.configure)(&mut ctx, &dbg.header) };
            if r != FFX_OK && fsr_debug() {
                crate::info!("upscaler: FSR's message setup ({effect_id:#x}): {r}");
            }
        }
        let mut ver = FfxProviderVersion { header: FfxHeader::new(FFX_QUERY_PROVIDER_VERSION), version_id: 0, version_name: std::ptr::null() };
        let name = if unsafe { (f.query)(&mut ctx, &mut ver.header) } == FFX_OK && !ver.version_name.is_null() {
            unsafe { CStr::from_ptr(ver.version_name) }.to_string_lossy().into_owned()
        } else {
            "?".into()
        };
        if inp.ldr {
            crate::info!("upscaler: FSR upscaling {}x{} to {}x{} in place of the game's scaling (provider {name})", inp.in_width, inp.in_height, inp.width, inp.height);
        } else {
            crate::info!("upscaler: FSR native anti-aliasing at {}x{} in place of SMAA (provider {name})", inp.width, inp.height);
        }
        *st = Some(FsrState { ctx: ctx as usize, key, frame: 0, last: std::time::Instant::now() });
    }
    let s = st.as_mut().unwrap();
    let (jx, jy) = advance_jitter(s.frame, inp.ratio());
    let (sx, sy) = jitter_scale();
    let (mx, my) = mv_scale();
    let now = std::time::Instant::now();
    let dt_ms = if s.frame == 0 { 16.7 } else { (now - s.last).as_secs_f32() * 1000.0 };
    s.last = now;
    // (The camera's planes and field of view: FSR 3's depth tests.)
    let cam = crate::title_patches::render_camera();
    if s.frame == 0 {
        if let Err(e) = &cam {
            crate::warn!("upscaler: FSR without the game camera ({e}): stand-in planes 0.1 / 10000, 57 deg");
        }
    }
    let (near, far, fov) = cam.unwrap_or((0.1, 10000.0, 1.0));
    let d = FfxDispatchUpscale {
        header: FfxHeader::new(FFX_DISPATCH_UPSCALE),
        command_list: cmd,
        color: ffx_resource(&inp.color, FFX_STATE_COMPUTE_READ),
        depth: ffx_resource(&inp.depth, FFX_STATE_COMPUTE_READ),
        motion_vectors: ffx_resource(&inp.velocity, FFX_STATE_COMPUTE_READ),
        exposure: FfxResource::default(),
        reactive: FfxResource::default(),
        transparency_and_composition: FfxResource::default(),
        output: ffx_resource(&inp.output, FFX_STATE_UNORDERED_ACCESS),
        // (FSR's jitter: the image's shift, as rendered.)
        jitter_offset: [jx * sx, jy * sy],
        // The game's motion vectors: screen fraction, current - previous;
        // FSR's: pixels, current -> previous.
        motion_vector_scale: [-(inp.in_width as f32) * mx, -(inp.in_height as f32) * my],
        render_size: FfxDims { width: inp.in_width, height: inp.in_height },
        upscale_size: FfxDims { width: inp.width, height: inp.height },
        enable_sharpening: false,
        sharpness: 0.0,
        frame_time_delta: dt_ms.clamp(1.0, 100.0),
        pre_exposure: 1.0,
        reset: reset || s.frame == 0,
        camera_near: near,
        camera_far: far,
        camera_fov_angle_vertical: fov,
        view_space_to_meters_factor: 1.0,
        flags: if inp.ldr { FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB } else { 0 },
    };
    if s.frame == 0 {
        crate::info!("upscaler: FSR camera near {near:.3}, far {far:.1}, vertical field of view {:.1} deg", fov.to_degrees());
    }
    s.frame += 1;
    let (ctx, dispatch) = (s.ctx, f.dispatch);
    drop(st);
    unsafe {
        super::rec::REC.cmd_external(move |_| {
            static N: AtomicU32 = AtomicU32::new(0);
            let mut c = ctx as FfxContext;
            // (The whole description: a closure naming only `d.header`
            // would capture just those 16 bytes.)
            let desc: *const FfxHeader = (&d as *const FfxDispatchUpscale).cast();
            let r = dispatch(&mut c, desc);
            if r != FFX_OK {
                warn_some(&N, "ffxDispatch (FSR)", r as i32);
            }
        })
    };
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_and_halton() {
        assert_eq!(std::mem::size_of::<XessView>(), 48);
        assert_eq!(std::mem::size_of::<XessExecute>(), 360);
        assert_eq!(std::mem::size_of::<XessInit>(), 64);
        assert_eq!(halton(1, 2), 0.0);
        assert_eq!(halton(2, 2), -0.25);
        assert!((halton(1, 3) - (1.0 / 3.0 - 0.5)).abs() < 1e-6);
        for i in 1..=8 {
            let (x, y) = (halton(i, 2), halton(i, 3));
            assert!((-0.5..0.5).contains(&x) && (-0.5..0.5).contains(&y));
        }
    }

    /// Streamline's structures as its MSVC build lays them out (offsets
    /// from the SDK's headers, e.g. its own static_assert on VulkanInfo).
    #[test]
    fn streamline_layouts() {
        use std::mem::{offset_of, size_of};
        assert_eq!(size_of::<SlBase>(), 32);
        assert_eq!(size_of::<SlPreferences>(), 144);
        assert_eq!(offset_of!(SlPreferences, flags), 88);
        assert_eq!(offset_of!(SlPreferences, render_api), 136);
        assert_eq!(size_of::<SlResource>(), 112);
        assert_eq!(offset_of!(SlResource, native), 40);
        assert_eq!(offset_of!(SlResource, state), 64);
        assert_eq!(offset_of!(SlResource, gpu_virtual_address), 88);
        assert_eq!(size_of::<SlResourceTag>(), 64);
        assert_eq!(offset_of!(SlResourceTag, extent), 48);
        assert_eq!(size_of::<SlViewport>(), 40);
        assert_eq!(size_of::<SlFeatureRequirements>(), 184);
        assert_eq!(offset_of!(SlFeatureRequirements, vk_device_extensions), 120);
        assert_eq!(offset_of!(SlFeatureRequirements, vk_optical_flow_queues), 176);
        assert_eq!(size_of::<SlAdapterInfo>(), 56);
        assert_eq!(size_of::<SlVulkanInfo>(), 96);
        assert_eq!(offset_of!(SlVulkanInfo, use_native_optical_flow), 80);
        assert_eq!(size_of::<SlConstants>(), 456);
        assert_eq!(offset_of!(SlConstants, jitter_offset), 352);
        assert_eq!(offset_of!(SlConstants, depth_inverted), 444);
        assert_eq!(offset_of!(SlConstants, min_relative_linear_depth_object_separation), 452);
        assert_eq!(size_of::<SlDlssOptions>(), 88);
        assert_eq!(offset_of!(SlDlssOptions, dlaa_preset), 60);
        assert_eq!(offset_of!(SlDlssOptions, use_auto_exposure), 84);
        assert_eq!(size_of::<SlFeatureVersion>(), 56);
        assert_eq!(SL_SDK_VERSION, 0x0002_000e_0010_fedc);
    }

    /// FSR's structures (natural C layout, checked against the headers).
    #[test]
    fn ffx_layouts() {
        use std::mem::{offset_of, size_of};
        assert_eq!(size_of::<FfxHeader>(), 16);
        assert_eq!(size_of::<FfxResDesc>(), 32);
        assert_eq!(size_of::<FfxResource>(), 48);
        assert_eq!(size_of::<FfxCreateUpscale>(), 48);
        assert_eq!(offset_of!(FfxCreateUpscale, message), 40);
        assert_eq!(size_of::<FfxBackendVk>(), 40);
        assert_eq!(offset_of!(FfxDispatchUpscale, color), 24);
        assert_eq!(offset_of!(FfxDispatchUpscale, output), 312);
        assert_eq!(offset_of!(FfxDispatchUpscale, jitter_offset), 360);
        assert_eq!(offset_of!(FfxDispatchUpscale, enable_sharpening), 392);
        assert_eq!(offset_of!(FfxDispatchUpscale, sharpness), 396);
        assert_eq!(offset_of!(FfxDispatchUpscale, reset), 408);
        assert_eq!(offset_of!(FfxDispatchUpscale, camera_near), 412);
        assert_eq!(offset_of!(FfxDispatchUpscale, flags), 428);
        assert_eq!(size_of::<FfxDispatchUpscale>(), 432);
        assert_eq!(size_of::<FfxProviderVersion>(), 32);
        assert_eq!(size_of::<FfxGlobalDebug>(), 40);
        assert_eq!(ffx_format(vk::Format::B10G11R11_UFLOAT_PACK32), 16);
        assert_eq!(ffx_format(vk::Format::R16G16_SFLOAT), 18);
        assert_eq!(ffx_format(vk::Format::D32_SFLOAT), 28);
    }

    #[test]
    fn modes_by_ratio() {
        assert_eq!(xess_quality(1.0), XESS_QUALITY_AA);
        assert_eq!(xess_quality(1.5), 104);
        assert_eq!(xess_quality(1.0 / 0.58), 103);
        assert_eq!(xess_quality(2.0), 102);
        assert_eq!(xess_quality(3.0), 100);
        assert_eq!(dlss_mode(1.0), SL_DLSS_DLAA);
        assert_eq!(dlss_mode(1.5), 3);
        assert_eq!(dlss_mode(1.0 / 0.58), 2);
        assert_eq!(dlss_mode(2.0), 1);
        assert_eq!(dlss_mode(3.0), 4);
    }

    #[test]
    fn choices_parse() {
        assert_eq!(Kind::parse("dlss"), Kind::Dlss);
        assert_eq!(Kind::parse(" xess "), Kind::Xess);
        assert_eq!(Kind::parse("off"), Kind::Off);
        assert_eq!(Kind::parse("fsr"), Kind::Fsr);
        assert_eq!(Kind::parse("taa"), Kind::Off);
        for k in [Kind::Off, Kind::Xess, Kind::Dlss, Kind::Fsr] {
            assert_eq!(Kind::from_u8(k as u8), k);
        }
    }
}
