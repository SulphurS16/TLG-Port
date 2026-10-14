//! Vulkan device context shared by the renderer and the presenter.

pub mod present;
pub mod window;

use ash::vk;
use parking_lot::Mutex;
use std::ffi::{CStr, c_char};
use std::sync::OnceLock;

/// Live `Ctx::alloc` allocations: memory handle -> (bytes, device-local,
/// allocating source location).
type AllocMap = std::collections::HashMap<u64, (u64, bool, &'static std::panic::Location<'static>)>;
static ALLOCS: std::sync::LazyLock<Mutex<AllocMap>> = std::sync::LazyLock::new(|| Mutex::new(AllocMap::new()));

pub struct Ctx {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub pdev: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: Mutex<vk::Queue>,
    pub queue_family: u32,
    pub mem_props: vk::PhysicalDeviceMemoryProperties,
    pub limits: vk::PhysicalDeviceLimits,
    pub surface_i: ash::khr::surface::Instance,
    pub win32_i: ash::khr::win32_surface::Instance,
    pub swapchain_d: ash::khr::swapchain::Device,
    pub push_desc: ash::khr::push_descriptor::Device,
    /// VK_EXT_external_memory_host (zero-copy access to guest memory).
    pub ext_host: Option<ash::ext::external_memory_host::Device>,
    /// VK_EXT_memory_budget enabled (see `vram`).
    pub mem_budget: bool,
    /// VK_EXT_conditional_rendering (fast-clear metadata decided on the GPU).
    pub cond_render: Option<ash::ext::conditional_rendering::Device>,
    /// shaderClipDistance / shaderCullDistance enabled.
    pub clip_cull: (bool, bool),
    /// occlusionQueryPrecise enabled (exact sample counts).
    pub occl_precise: bool,
    /// shaderDenormFlushToZeroFloat32 (shaders may request FP32 flushing).
    pub denorm_flush32: bool,
    pub device_name: String,
    pub validation: bool,
    /// VK_EXT_device_fault: what the GPU was doing when the device was lost.
    pub device_fault: Option<ash::ext::device_fault::Device>,
    /// VK_NV_device_diagnostic_checkpoints (TLG_GPU_DIAG=1).
    pub checkpoints: Option<ash::nv::device_diagnostic_checkpoints::Device>,
    /// sparseBinding + sparseResidencyBuffer on the queue's family: buffers
    /// whose memory is bound block by block (`bind_sparse`; TLG_SPARSE=0 off).
    pub sparse: bool,
}

static CTX: OnceLock<Ctx> = OnceLock::new();

/// Descriptions of recent checkpoints (sequence number -> recorded command).
static DIAG_NOTES: Mutex<std::collections::VecDeque<(u64, String)>> = Mutex::new(std::collections::VecDeque::new());
static DIAG_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Records a checkpoint describing the next command (TLG_GPU_DIAG=1).
pub fn checkpoint(cmd: vk::CommandBuffer, text: String) {
    let Some(cp) = try_ctx().and_then(|c| c.checkpoints.as_ref()) else { return };
    let seq = DIAG_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    {
        let mut n = DIAG_NOTES.lock();
        n.push_back((seq, text));
        if n.len() > 1 << 16 {
            n.pop_front();
        }
    }
    // (In the batch's order: through the recorder.)
    crate::gpu::rec::REC.raw(move |_| unsafe { cp.cmd_set_checkpoint(cmd, seq as usize as *const std::ffi::c_void) });
}

pub fn checkpoints_enabled() -> bool {
    try_ctx().is_some_and(|c| c.checkpoints.is_some())
}

/// After a device loss: logs the fault report and the last checkpoints reached (once).
pub fn report_device_lost() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| unsafe {
        let Some(c) = try_ctx() else { return };
        if let Some(df) = &c.device_fault {
            let mut counts = vk::DeviceFaultCountsEXT::default();
            let r = (df.fp().get_device_fault_info_ext)(c.device.handle(), &mut counts, std::ptr::null_mut());
            if r == vk::Result::SUCCESS || r == vk::Result::INCOMPLETE {
                let mut addrs = vec![vk::DeviceFaultAddressInfoEXT::default(); counts.address_info_count as usize];
                let mut vendor = vec![vk::DeviceFaultVendorInfoEXT::default(); counts.vendor_info_count as usize];
                let mut info = vk::DeviceFaultInfoEXT::default();
                info.p_address_infos = addrs.as_mut_ptr();
                info.p_vendor_infos = vendor.as_mut_ptr();
                let r2 = (df.fp().get_device_fault_info_ext)(c.device.handle(), &mut counts, &mut info);
                let desc = CStr::from_ptr(info.description.as_ptr()).to_string_lossy().into_owned();
                let mut s = format!("device fault report ({r2:?}): '{desc}'");
                for a in &addrs {
                    s.push_str(&format!("\n  {:?} address {:#x} (precision {:#x})", a.address_type, a.reported_address, a.address_precision));
                }
                for v in &vendor {
                    s.push_str(&format!("\n  vendor: {} code {:#x} data {:#x}", CStr::from_ptr(v.description.as_ptr()).to_string_lossy(), v.vendor_fault_code, v.vendor_fault_data));
                }
                crate::error!("{s}");
            } else {
                crate::error!("device fault report unavailable: {r:?}");
            }
        }
        if let Some(cp) = &c.checkpoints {
            let q = *c.queue.lock();
            let n = cp.get_queue_checkpoint_data_len(q);
            let mut data = vec![vk::CheckpointDataNV::default(); n];
            cp.get_queue_checkpoint_data(q, &mut data);
            let notes = DIAG_NOTES.lock();
            let mut last = 0;
            for d in &data {
                let seq = d.p_checkpoint_marker as usize as u64;
                let text = notes.iter().find(|(s, _)| *s == seq).map(|(_, t)| t.as_str()).unwrap_or("?");
                crate::error!("last checkpoint at {:?}: #{seq} {text}", d.stage);
                last = last.max(seq);
            }
            // (A checkpoint follows its command: the hang is in a command recorded after it.)
            for (s, t) in notes.iter().filter(|(s, _)| *s > last).take(8) {
                crate::error!("  recorded after it: #{s} {t}");
            }
        }
    });
}

pub fn ctx() -> &'static Ctx {
    CTX.get().expect("Vulkan not initialized")
}

pub fn try_ctx() -> Option<&'static Ctx> {
    CTX.get()
}

unsafe extern "system" fn debug_cb(
    sev: vk::DebugUtilsMessageSeverityFlagsEXT,
    _ty: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let msg = unsafe { CStr::from_ptr((*data).p_message) }.to_string_lossy();
    if sev.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        crate::error!("vulkan: {msg}");
    } else {
        crate::warn!("vulkan: {msg}");
    }
    vk::FALSE
}

pub fn init() -> Result<(), String> {
    if CTX.get().is_some() {
        return Ok(());
    }
    unsafe {
        let entry = ash::Entry::load().map_err(|e| format!("loading Vulkan: {e}"))?;
        let app_name = c"tlg";
        let app = vk::ApplicationInfo::default()
            .application_name(app_name)
            .engine_name(app_name)
            .api_version(vk::API_VERSION_1_3);
        let layers_avail = entry.enumerate_instance_layer_properties().unwrap_or_default();
        let want_validation = std::env::var_os("TLG_VK_VALIDATION").is_some();
        let validation_name = c"VK_LAYER_KHRONOS_validation";
        let has_validation = layers_avail.iter().any(|l| CStr::from_ptr(l.layer_name.as_ptr()) == validation_name);
        let validation = want_validation && has_validation;
        if want_validation && !has_validation {
            crate::warn!("TLG_VK_VALIDATION set but the Khronos validation layer is not installed");
        }
        let layers: Vec<*const c_char> = if validation { vec![validation_name.as_ptr()] } else { vec![] };
        let mut exts = vec![ash::khr::surface::NAME.as_ptr(), ash::khr::win32_surface::NAME.as_ptr()];
        if validation {
            exts.push(ash::ext::debug_utils::NAME.as_ptr());
        }
        let ici = vk::InstanceCreateInfo::default().application_info(&app).enabled_layer_names(&layers).enabled_extension_names(&exts);
        let instance = entry.create_instance(&ici, None).map_err(|e| format!("vkCreateInstance: {e}"))?;
        if validation {
            let du = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let mci = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING)
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL)
                .pfn_user_callback(Some(debug_cb));
            let _ = du.create_debug_utils_messenger(&mci, None);
        }
        // Pick a discrete GPU with a graphics+compute queue.
        let pdevs = instance.enumerate_physical_devices().map_err(|e| format!("enumerate devices: {e}"))?;
        let mut chosen = None;
        for &pd in &pdevs {
            let props = instance.get_physical_device_properties(pd);
            let qfs = instance.get_physical_device_queue_family_properties(pd);
            if let Some(qf) = qfs.iter().position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)) {
                let discrete = props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
                if chosen.is_none() || discrete {
                    chosen = Some((pd, qf as u32, props));
                }
            }
        }
        let (pdev, qf, props) = chosen.ok_or("no Vulkan device with graphics+compute")?;
        let device_name = CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy().into_owned();
        let prio = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default().queue_family_index(qf).queue_priorities(&prio)];
        let host_import = instance
            .enumerate_device_extension_properties(pdev)
            .unwrap_or_default()
            .iter()
            .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == ash::ext::external_memory_host::NAME);
        let mut dexts = vec![ash::khr::swapchain::NAME.as_ptr(), ash::khr::push_descriptor::NAME.as_ptr()];
        if host_import {
            dexts.push(ash::ext::external_memory_host::NAME.as_ptr());
        }
        let mem_budget = instance
            .enumerate_device_extension_properties(pdev)
            .unwrap_or_default()
            .iter()
            .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == ash::ext::memory_budget::NAME);
        if mem_budget {
            dexts.push(ash::ext::memory_budget::NAME.as_ptr());
        }
        // Out-of-range buffer/image accesses must not fault: GCN bounds-checks
        // buffer accesses against the descriptor (loads return 0, stores are
        // dropped), and titles rely on it. robustness2 gives exactly that.
        let has_robust2 = instance
            .enumerate_device_extension_properties(pdev)
            .unwrap_or_default()
            .iter()
            .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == ash::ext::robustness2::NAME);
        let mut r2_supported = vk::PhysicalDeviceRobustness2FeaturesEXT::default();
        if has_robust2 {
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut r2_supported);
            instance.get_physical_device_features2(pdev, &mut f2);
        }
        let mut r2 = vk::PhysicalDeviceRobustness2FeaturesEXT::default()
            .robust_buffer_access2(r2_supported.robust_buffer_access2 != 0)
            .robust_image_access2(r2_supported.robust_image_access2 != 0)
            .null_descriptor(r2_supported.null_descriptor != 0);
        if has_robust2 {
            dexts.push(ash::ext::robustness2::NAME.as_ptr());
        }
        let has_cond = instance
            .enumerate_device_extension_properties(pdev)
            .unwrap_or_default()
            .iter()
            .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == ash::ext::conditional_rendering::NAME);
        let mut cond_supported = vk::PhysicalDeviceConditionalRenderingFeaturesEXT::default();
        if has_cond {
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut cond_supported);
            instance.get_physical_device_features2(pdev, &mut f2);
        }
        let cond_render = has_cond && cond_supported.conditional_rendering != 0;
        let mut cond_features = vk::PhysicalDeviceConditionalRenderingFeaturesEXT::default().conditional_rendering(true);
        if cond_render {
            dexts.push(ash::ext::conditional_rendering::NAME.as_ptr());
        }
        let have = instance.get_physical_device_features(pdev);
        let denorm_flush32 = {
            let mut fc = vk::PhysicalDeviceFloatControlsProperties::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut fc);
            instance.get_physical_device_properties2(pdev, &mut p2);
            crate::info!(
                "Vulkan float controls: FP32 denorm flush {} preserve {}, independence {:?}",
                fc.shader_denorm_flush_to_zero_float32,
                fc.shader_denorm_preserve_float32,
                fc.denorm_behavior_independence
            );
            fc.shader_denorm_flush_to_zero_float32 != 0
        };
        let clip_cull = (have.shader_clip_distance != 0, have.shader_cull_distance != 0);
        let occl_precise = have.occlusion_query_precise != 0;
        let sparse = have.sparse_binding != 0
            && have.sparse_residency_buffer != 0
            && instance.get_physical_device_queue_family_properties(pdev)[qf as usize].queue_flags.contains(vk::QueueFlags::SPARSE_BINDING)
            && !std::env::var("TLG_SPARSE").is_ok_and(|v| v == "0");
        let features = vk::PhysicalDeviceFeatures::default()
            .sparse_binding(sparse)
            .sparse_residency_buffer(sparse)
            .occlusion_query_precise(occl_precise)
            .shader_clip_distance(clip_cull.0)
            .shader_cull_distance(clip_cull.1)
            .robust_buffer_access(true)
            .depth_bounds(true)
            .geometry_shader(true)
            .tessellation_shader(true)
            .independent_blend(true)
            .dual_src_blend(true)
            .logic_op(true)
            .fill_mode_non_solid(true)
            .depth_clamp(true)
            .depth_bias_clamp(true)
            .sampler_anisotropy(true)
            .texture_compression_bc(true)
            .image_cube_array(true)
            .shader_int64(true)
            .shader_storage_image_write_without_format(true)
            .shader_storage_image_read_without_format(true)
            .vertex_pipeline_stores_and_atomics(true)
            .fragment_stores_and_atomics(true)
            .multi_draw_indirect(true)
            .sample_rate_shading(true)
            // Non-constant texel offsets (GCN sample/gather offsets come from registers).
            .shader_image_gather_extended(have.shader_image_gather_extended != 0);
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(true)
            .scalar_block_layout(true)
            .shader_float16(true)
            .storage_buffer8_bit_access(true)
            .uniform_and_storage_buffer8_bit_access(true)
            .shader_int8(true);
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true).synchronization2(true).maintenance4(true);
        // Device-loss diagnostics: VK_EXT_device_fault (always, no cost) and NV
        // checkpoints (TLG_GPU_DIAG=1: one per recorded command).
        let has_ext = |n: &CStr| instance.enumerate_device_extension_properties(pdev).unwrap_or_default().iter().any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == n);
        let mut fault_supported = vk::PhysicalDeviceFaultFeaturesEXT::default();
        if has_ext(ash::ext::device_fault::NAME) {
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut fault_supported);
            instance.get_physical_device_features2(pdev, &mut f2);
        }
        let device_fault = fault_supported.device_fault != 0;
        let mut fault_features = vk::PhysicalDeviceFaultFeaturesEXT::default().device_fault(true);
        if device_fault {
            dexts.push(ash::ext::device_fault::NAME.as_ptr());
        }
        let checkpoints = std::env::var("TLG_GPU_DIAG").is_ok_and(|v| v != "0") && has_ext(ash::nv::device_diagnostic_checkpoints::NAME);
        if checkpoints {
            dexts.push(ash::nv::device_diagnostic_checkpoints::NAME.as_ptr());
        }
        let mut dci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qci)
            .enabled_extension_names(&dexts)
            .enabled_features(&features)
            .push_next(&mut f12)
            .push_next(&mut f13);
        if has_robust2 {
            dci = dci.push_next(&mut r2);
        }
        if cond_render {
            dci = dci.push_next(&mut cond_features);
        }
        if device_fault {
            dci = dci.push_next(&mut fault_features);
        }
        let device = instance.create_device(pdev, &dci, None).map_err(|e| format!("vkCreateDevice: {e}"))?;
        crate::info!("Vulkan diagnostics: device fault reports {device_fault}, checkpoints {checkpoints} (TLG_GPU_DIAG)");
        crate::info!(
            "Vulkan robustness: robustBufferAccess, robustness2 {} (buffer {}, image {}, null descriptors {})",
            has_robust2,
            r2_supported.robust_buffer_access2 != 0,
            r2_supported.robust_image_access2 != 0,
            r2_supported.null_descriptor != 0
        );
        let queue = device.get_device_queue(qf, 0);
        let mem_props = instance.get_physical_device_memory_properties(pdev);
        let ctx = Ctx {
            surface_i: ash::khr::surface::Instance::new(&entry, &instance),
            win32_i: ash::khr::win32_surface::Instance::new(&entry, &instance),
            swapchain_d: ash::khr::swapchain::Device::new(&instance, &device),
            push_desc: ash::khr::push_descriptor::Device::new(&instance, &device),
            // Imported guest memory (zero-copy) is opt-in (TLG_ZERO_COPY=1): with it the
            // GPU device is lost ~20 s into Shadow of the Colossus' boot on this host (a copy
            // out of an imported range; cause not yet found). Without it every access goes
            // through staging copies.
            ext_host: (host_import && std::env::var("TLG_ZERO_COPY").is_ok_and(|v| v == "1"))
                .then(|| ash::ext::external_memory_host::Device::new(&instance, &device)),
            mem_budget,
            cond_render: cond_render.then(|| ash::ext::conditional_rendering::Device::new(&instance, &device)),
            device_fault: device_fault.then(|| ash::ext::device_fault::Device::new(&instance, &device)),
            checkpoints: checkpoints.then(|| ash::nv::device_diagnostic_checkpoints::Device::new(&instance, &device)),
            sparse,
            clip_cull,
            occl_precise,
            denorm_flush32,
            entry,
            instance,
            pdev,
            device,
            queue: Mutex::new(queue),
            queue_family: qf,
            mem_props,
            limits: props.limits,
            device_name,
            validation,
        };
        crate::info!("Vulkan device: {} (queue family {qf}, validation {}, sparse buffers {})", ctx.device_name, validation, ctx.sparse);
        for i in 0..mem_props.memory_heap_count as usize {
            let h = mem_props.memory_heaps[i];
            let types: Vec<String> = (0..mem_props.memory_type_count as usize).filter(|&t| mem_props.memory_types[t].heap_index as usize == i).map(|t| format!("{:?}", mem_props.memory_types[t].property_flags)).collect();
            crate::info!("Vulkan memory heap {i}: {} MiB {:?}; types [{}]", h.size >> 20, h.flags, types.join("; "));
        }
        {
            let exts = ctx.instance.enumerate_device_extension_properties(pdev).unwrap_or_default();
            let has = |n: &CStr| exts.iter().any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == n);
            let mut host = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut host);
            ctx.instance.get_physical_device_properties2(pdev, &mut p2);
            crate::info!(
                "Vulkan limits: maxStorageBufferRange {:#x}, minStorageBufferOffsetAlignment {}, external_memory_host {} (import alignment {:#x}), buffer_device_address {}",
                ctx.limits.max_storage_buffer_range,
                ctx.limits.min_storage_buffer_offset_alignment,
                has(ash::ext::external_memory_host::NAME),
                host.min_imported_host_pointer_alignment,
                has(ash::khr::buffer_device_address::NAME)
            );
        }
        CTX.set(ctx).ok();
        Ok(())
    }
}

impl Ctx {
    /// Device-local memory: (this process's usage, its budget) in bytes from
    /// VK_EXT_memory_budget (None without it). The budget accounts for what
    /// other processes use; beyond it the OS pages video memory.
    pub fn vram(&self) -> Option<(u64, u64)> {
        if !self.mem_budget {
            return None;
        }
        let mut b = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        {
            let mut p2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut b);
            unsafe { self.instance.get_physical_device_memory_properties2(self.pdev, &mut p2) };
        }
        let (mut used, mut budget) = (0, 0);
        for i in 0..self.mem_props.memory_heap_count as usize {
            if self.mem_props.memory_heaps[i].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                used += b.heap_usage[i];
                budget += b.heap_budget[i];
            }
        }
        Some((used, budget))
    }

    pub fn memory_type(&self, bits: u32, want: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.mem_props.memory_type_count)
            .find(|&i| bits & (1 << i) != 0 && self.mem_props.memory_types[i as usize].property_flags.contains(want))
    }

    /// Allocates memory (recorded with the caller's source location for the
    /// VRAM report; release it with `free`).
    #[track_caller]
    pub fn alloc(&self, req: vk::MemoryRequirements, want: vk::MemoryPropertyFlags) -> Result<vk::DeviceMemory, String> {
        let ty = self.memory_type(req.memory_type_bits, want).ok_or("no suitable memory type")?;
        let ai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty);
        let mem = unsafe { self.device.allocate_memory(&ai, None) }.map_err(|e| format!("vkAllocateMemory({}): {e}", req.size))?;
        let heap = self.mem_props.memory_types[ty as usize].heap_index as usize;
        let local = self.mem_props.memory_heaps[heap].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL);
        ALLOCS.lock().insert(vk::Handle::as_raw(mem), (req.size, local, std::panic::Location::caller()));
        Ok(mem)
    }

    /// Frees memory from `alloc` (or imported).
    pub fn free(&self, mem: vk::DeviceMemory) {
        ALLOCS.lock().remove(&vk::Handle::as_raw(mem));
        unsafe { self.device.free_memory(mem, None) };
    }

    /// Device-local memory this process holds through `alloc`, by allocating
    /// source location: (total bytes, [(location, bytes, allocations)]
    /// largest first).
    pub fn local_allocations(&self) -> (u64, Vec<(String, u64, usize)>) {
        let mut by: std::collections::HashMap<String, (u64, usize)> = std::collections::HashMap::new();
        let mut total = 0;
        for &(size, local, at) in ALLOCS.lock().values() {
            if local {
                total += size;
                let e = by.entry(format!("{}:{}", at.file().rsplit(['/', '\\']).next().unwrap_or("?"), at.line())).or_default();
                e.0 += size;
                e.1 += 1;
            }
        }
        let mut v: Vec<(String, u64, usize)> = by.into_iter().map(|(k, (b, n))| (k, b, n)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        (total, v)
    }

    /// Device-local buffer (not host-visible).
    #[track_caller]
    pub fn device_buffer(&self, size: u64, usage: vk::BufferUsageFlags) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
        unsafe {
            let bi = vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buf = self.device.create_buffer(&bi, None).map_err(|e| format!("vkCreateBuffer: {e}"))?;
            let req = self.device.get_buffer_memory_requirements(buf);
            let mem = self.alloc(req, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            self.device.bind_buffer_memory(buf, mem, 0).map_err(|e| format!("bind buffer: {e}"))?;
            Ok((buf, mem))
        }
    }

    /// Imports existing host memory [ptr, ptr+size) (page aligned) as a GPU
    /// buffer (VK_EXT_external_memory_host): the GPU reads and writes it in place.
    pub fn import_host(&self, ptr: u64, size: u64, usage: vk::BufferUsageFlags) -> Result<(vk::Buffer, vk::DeviceMemory), String> {
        let ext = self.ext_host.as_ref().ok_or("VK_EXT_external_memory_host not available")?;
        let ht = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
        unsafe {
            let mut props = vk::MemoryHostPointerPropertiesEXT::default();
            let r = (ext.fp().get_memory_host_pointer_properties_ext)(self.device.handle(), ht, ptr as *const std::ffi::c_void, &mut props);
            if r != vk::Result::SUCCESS {
                return Err(format!("vkGetMemoryHostPointerPropertiesEXT: {r}"));
            }
            let mut ebi = vk::ExternalMemoryBufferCreateInfo::default().handle_types(ht);
            let bi = vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE).push_next(&mut ebi);
            let buf = self.device.create_buffer(&bi, None).map_err(|e| format!("vkCreateBuffer: {e}"))?;
            let req = self.device.get_buffer_memory_requirements(buf);
            let bits = req.memory_type_bits & props.memory_type_bits;
            let hv = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let ty = self
                .memory_type(bits, hv | vk::MemoryPropertyFlags::HOST_CACHED)
                .or_else(|| self.memory_type(bits, hv))
                .or_else(|| self.memory_type(bits, vk::MemoryPropertyFlags::empty()));
            let Some(ty) = ty else {
                self.device.destroy_buffer(buf, None);
                return Err(format!("no memory type for imported host memory (bits {bits:#x})"));
            };
            let mut imp = vk::ImportMemoryHostPointerInfoEXT::default().handle_type(ht).host_pointer(ptr as *mut std::ffi::c_void);
            let ai = vk::MemoryAllocateInfo::default().allocation_size(size).memory_type_index(ty).push_next(&mut imp);
            let mem = match self.device.allocate_memory(&ai, None) {
                Ok(m) => m,
                Err(e) => {
                    self.device.destroy_buffer(buf, None);
                    return Err(format!("vkAllocateMemory(import {size:#x}): {e}"));
                }
            };
            if let Err(e) = self.device.bind_buffer_memory(buf, mem, 0) {
                self.device.destroy_buffer(buf, None);
                self.device.free_memory(mem, None);
                return Err(format!("bind imported buffer: {e}"));
            }
            Ok((buf, mem))
        }
    }

    /// Host-visible, coherent buffer (staging / per-draw data).
    #[track_caller]
    pub fn host_buffer(&self, size: u64, usage: vk::BufferUsageFlags) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8), String> {
        unsafe {
            let bi = vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buf = self.device.create_buffer(&bi, None).map_err(|e| format!("vkCreateBuffer: {e}"))?;
            let req = self.device.get_buffer_memory_requirements(buf);
            // Prefer cached host memory: results are read back by the CPU
            // (buffer write-back), which is very slow from write-combined memory.
            let hv = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let mem = self.alloc(req, hv | vk::MemoryPropertyFlags::HOST_CACHED).or_else(|_| self.alloc(req, hv))?;
            self.device.bind_buffer_memory(buf, mem, 0).map_err(|e| format!("bind buffer: {e}"))?;
            let ptr = self.device.map_memory(mem, 0, size, vk::MemoryMapFlags::empty()).map_err(|e| format!("map: {e}"))? as *mut u8;
            Ok((buf, mem, ptr))
        }
    }

    /// A buffer whose memory is bound later block by block (`bind_sparse`):
    /// the buffer and the block size (its memory alignment).
    pub fn sparse_buffer(&self, size: u64, usage: vk::BufferUsageFlags) -> Result<(vk::Buffer, vk::MemoryRequirements), String> {
        let bi = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .flags(vk::BufferCreateFlags::SPARSE_BINDING | vk::BufferCreateFlags::SPARSE_RESIDENCY);
        unsafe {
            let buf = self.device.create_buffer(&bi, None).map_err(|e| format!("vkCreateBuffer (sparse): {e}"))?;
            Ok((buf, self.device.get_buffer_memory_requirements(buf)))
        }
    }

    /// Binds memory blocks into sparse buffers on the queue; `signal` is
    /// signalled once bound (the next submission waits for it).
    pub fn bind_sparse(&self, binds: &[(vk::Buffer, Vec<vk::SparseMemoryBind>)], signal: vk::Semaphore) -> Result<(), String> {
        let infos: Vec<vk::SparseBufferMemoryBindInfo> = binds.iter().map(|(b, v)| vk::SparseBufferMemoryBindInfo::default().buffer(*b).binds(v)).collect();
        let sig = [signal];
        let bi = vk::BindSparseInfo::default().buffer_binds(&infos).signal_semaphores(&sig);
        let q = self.queue.lock();
        unsafe { self.device.queue_bind_sparse(*q, &[bi], vk::Fence::null()) }.map_err(|e| format!("vkQueueBindSparse: {e}"))
    }

    pub fn submit(&self, cmd: vk::CommandBuffer, wait: &[(vk::Semaphore, vk::PipelineStageFlags)], signal: &[vk::Semaphore], fence: vk::Fence) -> Result<(), String> {
        self.submit_cmds(&[cmd], wait, signal, fence)
    }

    /// `submit` of several command buffers, executed in order.
    pub fn submit_cmds(&self, cmds: &[vk::CommandBuffer], wait: &[(vk::Semaphore, vk::PipelineStageFlags)], signal: &[vk::Semaphore], fence: vk::Fence) -> Result<(), String> {
        let ws: Vec<vk::Semaphore> = wait.iter().map(|w| w.0).collect();
        let wst: Vec<vk::PipelineStageFlags> = wait.iter().map(|w| w.1).collect();
        let si = vk::SubmitInfo::default().command_buffers(cmds).wait_semaphores(&ws).wait_dst_stage_mask(&wst).signal_semaphores(signal);
        let q = self.queue.lock();
        unsafe { self.device.queue_submit(*q, &[si], fence) }.map_err(|e| format!("vkQueueSubmit: {e}"))
    }
}
