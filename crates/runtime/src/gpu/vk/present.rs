//! Swapchain presentation: copies a rendered image (or uploaded pixels) on
//! the game's queue and scales it into a swapchain image on the present
//! queue (`Ctx::present_queue`), so the game's GPU work never waits for the
//! window to hand out an image (a minimized or hidden window, a throttled
//! compositor). A frame whose slot is still being presented is not shown.

use super::{ctx, window};
use ash::vk;
use parking_lot::Mutex;

const FRAMES: usize = 3;

/// How long a present may wait for a swapchain image before the frame is
/// left out (ns).
const ACQUIRE_TIMEOUT_NS: u64 = 2_000_000;

/// One present: the frame copied into `image` on the game's queue, then
/// scaled into a swapchain image on the present queue.
struct Slot {
    copy: vk::CommandBuffer,
    blit: vk::CommandBuffer,
    /// Copy done (game queue) -> blit (present queue); unused with one queue.
    copied: vk::Semaphore,
    acquire: vk::Semaphore,
    /// The slot's last blit done (created signalled).
    fence: vk::Fence,
    /// The copy of the frame: image, memory, size, format.
    image: Option<(vk::Image, vk::DeviceMemory, u32, u32, vk::Format)>,
    /// Staging buffer of `Source::Pixels`.
    staging: Option<(vk::Buffer, vk::DeviceMemory, *mut u8, u64)>,
    /// HDR10 output: a view of `image` for the kernel, and its output (32-bit
    /// packed A2B10G10R10 at the swapchain's size: image, memory, view, size).
    view: Option<vk::ImageView>,
    out: Option<(vk::Image, vk::DeviceMemory, vk::ImageView, u32, u32)>,
}

/// HDR10 output (`hdr = auto` with Windows HDR on for the window's monitor):
/// the frame converted by `gcn::translate::kernels::hdr10_cs` (gamma 2.2
/// decode, paper white, BT.709 -> BT.2020, PQ) into a 32-bit image that is
/// copied into the ST 2084 swapchain image.
struct Hdr10 {
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    /// One descriptor set per slot.
    sets: Vec<vk::DescriptorSet>,
    /// Bilinear and nearest (`filter`).
    samplers: [vk::Sampler; 2],
}

struct Presenter {
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    format: vk::Format,
    extent: vk::Extent2D,
    pool: vk::CommandPool,
    slots: Vec<Slot>,
    done: Vec<vk::Semaphore>,
    frame: usize,
    /// Frames not shown because the window was not taking images.
    skipped: u64,
    /// Swapchains replaced (with their present semaphores) and the frame
    /// count then: destroyed once every slot's blit since has finished
    /// (`release_retired`). Waiting for the device instead
    /// (vkDeviceWaitIdle) on the presenting thread, while the renderer
    /// threads keep submitting to the game's queue, hung windowed boots now
    /// and then: no GPU work completed again (the threads' stacks showed this
    /// one in vkDeviceWaitIdle, the rest waiting for fences).
    retired: Vec<(usize, vk::SwapchainKHR, Vec<vk::Semaphore>)>,
    /// The kernel's objects (made for the first HDR10 swapchain) and, for
    /// the current swapchain, its paper-white scale (nits / 10000) and
    /// flags; None = SDR.
    hdr: Option<Hdr10>,
    hdr_out: Option<(f32, u32)>,
}

unsafe impl Send for Presenter {}

static PRESENTER: Mutex<Option<Presenter>> = Mutex::new(None);

/// The title's display buffers hold PQ-encoded BT.2020 (its own HDR mode,
/// `hdr = game`): set by libSceVideoOut from their pixel format. The HDR10
/// swapchain stays while it is set, whatever Windows' HDR state.
pub static GAME_PQ: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The current swapchain takes HDR10 (for libSceVideoOut's device
/// capability: the title's own HDR mode is offered only then).
pub fn hdr_active() -> bool {
    PRESENTER.lock().as_ref().is_some_and(|p| p.hdr_out.is_some())
}

pub enum Source {
    /// An image the GPU wrote; `layout` is its current layout (kept); `pq`:
    /// it holds PQ-encoded BT.2020 (the title's HDR mode).
    Image { image: vk::Image, layout: vk::ImageLayout, width: u32, height: u32, format: vk::Format, pq: bool },
    /// Linear BGRA8 pixels.
    Pixels { data: Vec<u8>, width: u32, height: u32 },
}

unsafe fn create_swapchain(p: &mut Presenter) -> Result<(), String> {
    let c = ctx();
    unsafe {
        let caps = c.surface_i.get_physical_device_surface_capabilities(c.pdev, p.surface).map_err(|e| format!("surface caps: {e}"))?;
        let formats = c.surface_i.get_physical_device_surface_formats(c.pdev, p.surface).map_err(|e| format!("surface formats: {e}"))?;
        // UNORM: display images hold sRGB-encoded values in UNORM storage
        // (8-bit sRGB targets are stored as UNORM, A2R10G10B10 sRGB has no
        // sRGB format), which a blit must pass through unchanged; an sRGB
        // swapchain would encode them a second time. HDR10 (ST 2084) where
        // wanted and offered.
        let white = hdr_wanted();
        let hdr_fmt = white.and_then(|_| {
            formats.iter().find(|f| f.color_space == vk::ColorSpaceKHR::HDR10_ST2084_EXT && matches!(f.format, vk::Format::A2B10G10R10_UNORM_PACK32 | vk::Format::A2R10G10B10_UNORM_PACK32)).copied()
        });
        if white.is_some() && hdr_fmt.is_none() {
            crate::warn!("HDR output: the window's surface offers no HDR10 (ST 2084) format; SDR output");
        }
        let fmt = hdr_fmt
            .or_else(|| formats.iter().find(|f| f.format == vk::Format::B8G8R8A8_UNORM && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR).copied())
            .or_else(|| formats.iter().find(|f| f.format == vk::Format::R8G8B8A8_UNORM && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR).copied())
            .or_else(|| formats.first().copied())
            .ok_or("no surface format")?;
        p.hdr_out = None;
        if let (Some(white), Some(f)) = (white, hdr_fmt) {
            if p.hdr.is_none() {
                p.hdr = Some(make_hdr10(p.slots.len())?);
            }
            // Flags: bit 0 the piecewise sRGB curve (TLG_HDR_SRGB=1; gamma 2.2
            // by default, as SDR monitors show the frame), bit 1 red in the
            // high bits (A2R10G10B10).
            let flags = (std::env::var_os("TLG_HDR_SRGB").is_some() as u32) | (((f.format == vk::Format::A2R10G10B10_UNORM_PACK32) as u32) << 1);
            p.hdr_out = Some((white / 10000.0, flags));
            let why = if crate::config::settings().hdr_white == 0 { "the monitor's SDR content brightness" } else { "setting hdr_white" };
            crate::info!("HDR output: HDR10 (ST 2084, BT.2020) {:?}, white {white} nits ({why})", f.format);
        }
        let mut extent = caps.current_extent;
        if extent.width == u32::MAX {
            extent = vk::Extent2D { width: window::CLIENT_W.load(std::sync::atomic::Ordering::Relaxed), height: window::CLIENT_H.load(std::sync::atomic::Ordering::Relaxed) };
        }
        if extent.width == 0 || extent.height == 0 {
            return Err("window minimized".into());
        }
        let count = (caps.min_image_count + 1).min(if caps.max_image_count == 0 { u32::MAX } else { caps.max_image_count });
        let old = p.swapchain;
        let sci = vk::SwapchainCreateInfoKHR::default()
            .surface(p.surface)
            .min_image_count(count)
            .image_format(fmt.format)
            .image_color_space(fmt.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode(p.surface))
            .clipped(true)
            .old_swapchain(old);
        // Windowed and borderless: no exclusive full screen. Left to them,
        // drivers take a window that covers its monitor exclusively, with a
        // display mode of their own choosing (NVIDIA's: GitHub issue #4).
        let mut no_fse = vk::SurfaceFullScreenExclusiveInfoEXT::default().full_screen_exclusive(vk::FullScreenExclusiveEXT::DISALLOWED);
        let disallow = c.fse && window::current_mode() != crate::config::DisplayMode::Fullscreen;
        let sci = if disallow { sci.push_next(&mut no_fse) } else { sci };
        let sc = c.swapchain_d.create_swapchain(&sci, None).map_err(|e| format!("create swapchain: {e}"))?;
        if old != vk::SwapchainKHR::null() {
            let done = std::mem::take(&mut p.done);
            p.retired.push((p.frame, old, done));
        }
        p.swapchain = sc;
        p.images = c.swapchain_d.get_swapchain_images(sc).map_err(|e| format!("swapchain images: {e}"))?;
        p.format = fmt.format;
        p.extent = extent;
        for _ in 0..p.images.len() {
            p.done.push(c.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).map_err(|e| e.to_string())?);
        }
        crate::info!("swapchain {}x{} {:?} ({} images, {:?}{})", extent.width, extent.height, fmt.format, p.images.len(), sci.present_mode, if disallow { ", exclusive full screen disallowed" } else { "" });
    }
    Ok(())
}

/// The swapchain's present mode: FIFO (vsync), or `TLG_PRESENT=mailbox`
/// (vsync without back-pressure: the newest frame is shown, for uncapped
/// rates above the display's) or `TLG_PRESENT=immediate` (no vsync, tears)
/// when the surface supports them; else settings.txt `vsync`.
/// The title's PQ frames on an SDR swapchain: the nits shown as white.
const PQ_SDR_WHITE: f32 = 100.0;

/// The gain on the title's own PQ output: TLG_HDR_GAIN, else 1 (as the
/// title renders it). (`white`: the SDR paper-white scale, nits / 10000.)
fn hdr_game_gain(_white: f32) -> f32 {
    static GAIN: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *GAIN.get_or_init(|| std::env::var("TLG_HDR_GAIN").ok().and_then(|v| v.parse().ok()).filter(|g: &f32| *g > 0.0 && *g <= 10.0).unwrap_or(1.0))
}

/// HDR10 output for the next swapchain: `hdr = auto` and Windows HDR on for
/// the window's monitor (development: TLG_HDR=force always, =off never).
/// Returns the paper white (nits): `hdr_white`, or for 0 the monitor's SDR
/// content brightness.
fn hdr_wanted() -> Option<f32> {
    let s = crate::config::settings();
    let force = std::env::var("TLG_HDR").ok();
    let disp = window::display_hdr();
    static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
        match &disp {
            Some(d) => crate::info!(
                "display {}: Windows HDR {} ({}), SDR content brightness {} nits; setting hdr {:?}, hdr_white {}",
                d.device,
                if d.enabled { "on" } else { "off" },
                if d.supported { "supported" } else { "not supported" },
                d.sdr_white,
                s.hdr,
                s.hdr_white
            ),
            None => crate::info!("display: Windows HDR state unknown; setting hdr {:?}", s.hdr),
        }
    }
    match force.as_deref() {
        Some("off") => return None,
        Some("force") => {}
        // (The title renders PQ: HDR10 regardless.)
        _ if GAME_PQ.load(std::sync::atomic::Ordering::Relaxed) => {}
        _ => {
            if s.hdr == crate::config::Hdr::Off || !disp.as_ref().is_some_and(|d| d.enabled) {
                return None;
            }
        }
    }
    Some(if s.hdr_white == 0 { disp.map_or(80.0, |d| d.sdr_white) } else { s.hdr_white as f32 })
}

/// The HDR10 kernel's pipeline, descriptor sets (one per slot) and samplers.
unsafe fn make_hdr10(slots: usize) -> Result<Hdr10, String> {
    let c = ctx();
    unsafe {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let set_layout = c.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).map_err(|e| e.to_string())?;
        let pcr = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 32 }];
        let sls = [set_layout];
        let layout = c.device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sls).push_constant_ranges(&pcr), None).map_err(|e| e.to_string())?;
        let code = gcn::translate::kernels::hdr10_cs();
        let module = c.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None).map_err(|e| e.to_string())?;
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
        let pci = [vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)];
        let pipeline = c.device.create_compute_pipelines(vk::PipelineCache::null(), &pci, None).map_err(|(_, e)| e.to_string())?[0];
        c.device.destroy_shader_module(module, None);
        let n = slots as u32;
        let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: n }, vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: n }];
        let pool = c.device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(n).pool_sizes(&sizes), None).map_err(|e| e.to_string())?;
        let layouts = vec![set_layout; slots];
        let sets = c.device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts)).map_err(|e| e.to_string())?;
        let sampler = |f: vk::Filter| {
            let sci = vk::SamplerCreateInfo::default()
                .mag_filter(f)
                .min_filter(f)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .max_lod(0.0);
            c.device.create_sampler(&sci, None).map_err(|e| e.to_string())
        };
        let samplers = [sampler(vk::Filter::LINEAR)?, sampler(vk::Filter::NEAREST)?];
        Ok(Hdr10 { layout, pipeline, sets, samplers })
    }
}

/// TLG_HDR_CHECK=<n>: the n-th HDR10 frame's output read back once and its
/// brightest pixels and mean logged in nits (PQ decoded), to check the
/// conversion without an HDR display.
unsafe fn hdr_check(c: &super::Ctx, cmd: vk::CommandBuffer, out: vk::Image, w: u32, h: u32, scale: f32, pq: bool) {
    static FRAMES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let Some(at) = std::env::var("TLG_HDR_CHECK").ok().and_then(|v| v.parse::<u64>().ok()) else { return };
    if FRAMES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1 != at.max(1) {
        return;
    }
    let size = (w * h * 4) as u64;
    let Ok((buf, _mem, ptr)) = c.host_buffer(size, vk::BufferUsageFlags::TRANSFER_DST) else { return };
    let layers = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
    let region = vk::BufferImageCopy { buffer_offset: 0, buffer_row_length: 0, buffer_image_height: 0, image_subresource: layers, image_offset: vk::Offset3D::default(), image_extent: vk::Extent3D { width: w, height: h, depth: 1 } };
    unsafe { c.device.cmd_copy_image_to_buffer(cmd, out, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, buf, &[region]) };
    let ptr = ptr as usize;
    std::thread::spawn(move || {
        // (After the blit has finished; leaked: once per run.)
        std::thread::sleep(std::time::Duration::from_secs(2));
        let px = unsafe { std::slice::from_raw_parts(ptr as *const u32, (w * h) as usize) };
        let nits = |code: u32| {
            let e = (code as f64 / 1023.0).powf(1.0 / 78.843_75);
            let y = ((e - 0.835_937_5).max(0.0) / (18.851_562_5 - 18.687_5 * e)).powf(1.0 / 0.159_301_757_812_5);
            y * 10000.0
        };
        let (mut sum, mut max) = (0.0f64, 0.0f64);
        for &v in px {
            let g = nits((v >> 10) & 0x3FF);
            sum += g;
            max = max.max(g);
        }
        let what = if pq { format!("the title's PQ frame, gain {scale}") } else { format!("white set to {:.0} nits", scale as f64 * 10000.0) };
        crate::info!("HDR check: output {w}x{h}, green channel mean {:.1} nits, max {:.1} nits ({what})", sum / px.len() as f64, max);
    });
}

fn present_mode(surface: vk::SurfaceKHR) -> vk::PresentModeKHR {
    use crate::config::VSync;
    let want = match std::env::var("TLG_PRESENT").as_deref() {
        Ok("mailbox") => vk::PresentModeKHR::MAILBOX,
        Ok("immediate") => vk::PresentModeKHR::IMMEDIATE,
        Ok("fifo") => vk::PresentModeKHR::FIFO,
        _ => match crate::config::settings().vsync {
            VSync::On => vk::PresentModeKHR::FIFO,
            VSync::Off => vk::PresentModeKHR::IMMEDIATE,
            VSync::Fast => vk::PresentModeKHR::MAILBOX,
        },
    };
    if want == vk::PresentModeKHR::FIFO {
        return want;
    }
    let c = ctx();
    let modes = unsafe { c.surface_i.get_physical_device_surface_present_modes(c.pdev, surface) }.unwrap_or_default();
    if modes.contains(&want) {
        want
    } else {
        crate::warn!("present mode {want:?} unsupported (supported: {modes:?}): FIFO");
        vk::PresentModeKHR::FIFO
    }
}

pub fn init(title: &str) -> Result<(), String> {
    super::init()?;
    let w = window::create(title)?;
    let c = ctx();
    unsafe {
        let sci = vk::Win32SurfaceCreateInfoKHR::default().hinstance(w.hinstance).hwnd(w.hwnd);
        let surface = c.win32_i.create_win32_surface(&sci, None).map_err(|e| format!("create surface: {e}"))?;
        let supported = c.surface_i.get_physical_device_surface_support(c.pdev, c.queue_family, surface).unwrap_or(false);
        if !supported {
            return Err("queue family cannot present".into());
        }
        let pool = c
            .device
            .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(c.queue_family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)
            .map_err(|e| e.to_string())?;
        let cmds = c
            .device
            .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(2 * FRAMES as u32))
            .map_err(|e| e.to_string())?;
        let sem = || c.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).map_err(|e| e.to_string());
        let mut slots = Vec::new();
        for k in 0..FRAMES {
            slots.push(Slot {
                copy: cmds[2 * k],
                blit: cmds[2 * k + 1],
                copied: sem()?,
                acquire: sem()?,
                fence: c.device.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None).map_err(|e| e.to_string())?,
                image: None,
                staging: None,
                view: None,
                out: None,
            });
        }
        let mut p = Presenter {
            surface,
            swapchain: vk::SwapchainKHR::null(),
            images: Vec::new(),
            format: vk::Format::UNDEFINED,
            extent: vk::Extent2D::default(),
            pool,
            slots,
            done: Vec::new(),
            frame: 0,
            skipped: 0,
            retired: Vec::new(),
            hdr: None,
            hdr_out: None,
        };
        create_swapchain(&mut p)?;
        *PRESENTER.lock() = Some(p);
    }
    Ok(())
}

fn barrier(c: &super::Ctx, cmd: vk::CommandBuffer, img: vk::Image, old: vk::ImageLayout, new: vk::ImageLayout) {
    let b = vk::ImageMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
        .old_layout(old)
        .new_layout(new)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(img)
        .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
    unsafe {
        c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &[b]);
    }
}

/// Where a `sw` x `sh` frame goes in a `dw` x `dh` image: all of it
/// (`scaling = stretch`), or the largest centred area of the frame's shape
/// (`fit`; the rest black).
fn dst_rect(sw: u32, sh: u32, dw: u32, dh: u32) -> (i32, i32, i32, i32) {
    if crate::config::settings().scaling == crate::config::Scaling::Stretch || sw == 0 || sh == 0 {
        return (0, 0, dw as i32, dh as i32);
    }
    fit_rect(sw, sh, dw, dh)
}

/// A frame up to 1% smaller than the target in each direction is shown
/// pixel for pixel, centred (e.g. 1360x764 rendered for a 1366x768
/// screen); otherwise it is scaled to fit.
fn fit_rect(sw: u32, sh: u32, dw: u32, dh: u32) -> (i32, i32, i32, i32) {
    if sw <= dw && sh <= dh && (dw - sw) * 100 <= dw && (dh - sh) * 100 <= dh {
        let (x, y) = ((dw - sw) / 2, (dh - sh) / 2);
        return (x as i32, y as i32, (x + sw) as i32, (y + sh) as i32);
    }
    let (sw, sh, dw, dh) = (sw as u64, sh as u64, dw as u64, dh as u64);
    // Width-limited if the target is relatively narrower than the frame.
    let (w, h) = if dw * sh <= dh * sw { (dw, (dw * sh + sw / 2) / sw) } else { ((dh * sw + sh / 2) / sh, dh) };
    let (x, y) = ((dw - w) / 2, (dh - h) / 2);
    (x as i32, y as i32, (x + w) as i32, (y + h) as i32)
}

#[cfg(test)]
mod tests {
    #[test]
    fn fit() {
        assert_eq!(super::fit_rect(1920, 1080, 1920, 1080), (0, 0, 1920, 1080));
        assert_eq!(super::fit_rect(1920, 1080, 2560, 1440), (0, 0, 2560, 1440));
        // 16:10: bars above and below; 21:9: at the sides.
        assert_eq!(super::fit_rect(1920, 1080, 1920, 1200), (0, 60, 1920, 1140));
        assert_eq!(super::fit_rect(1920, 1080, 3440, 1440), (440, 0, 3000, 1440));
        assert_eq!(super::fit_rect(1920, 1080, 1280, 1024), (0, 152, 1280, 872));
        // Nearly the target's size: pixel for pixel.
        assert_eq!(super::fit_rect(1360, 764, 1366, 768), (3, 2, 1363, 766));
        assert_eq!(super::fit_rect(1280, 720, 1366, 768), (0, 0, 1365, 768));
    }
}

/// Presents one frame. Returns false if it was not shown (a minimized
/// window, or the window still holding the images of earlier frames).
pub fn present(src: Source) -> bool {
    // (The frame's batches are submitted ahead of the copy: queue order.)
    crate::gpu::rec::drain();
    let c = ctx();
    let mut g = PRESENTER.lock();
    let Some(p) = g.as_mut() else { return false };
    unsafe {
        if window::RESIZED.swap(false, std::sync::atomic::Ordering::SeqCst) {
            if let Err(e) = create_swapchain(p) {
                crate::debug!("swapchain recreate: {e}");
                return false;
            }
        }
        release_retired(p);
        let k = p.frame % FRAMES;
        // The slot's previous frame not yet presented: the window is not
        // taking images (minimized, hidden, a throttled compositor). This
        // frame is left out; nothing waits for the window.
        if c.device.get_fence_status(p.slots[k].fence) != Ok(true) {
            p.skipped += 1;
            note_skipped(p.skipped);
            return false;
        }
        let (sw, sh, fmt, pq) = match &src {
            Source::Image { width, height, format, pq, .. } => (*width, *height, *format, *pq),
            Source::Pixels { width, height, .. } => (*width, *height, vk::Format::B8G8R8A8_UNORM, false),
        };
        // The kernel's mode: scale and flags (HDR10 output of SDR or PQ frames,
        // or PQ frames on an SDR swapchain); None = a blit.
        let kmode = match (p.hdr_out, pq) {
            (Some((white, flags)), false) => Some((white, flags)),
            (Some((white, flags)), true) => Some((hdr_game_gain(white), flags | 4)),
            (None, true) => Some((10000.0 / PQ_SDR_WHITE, 4 | 8 | (((p.format == vk::Format::B8G8R8A8_UNORM) as u32) << 1))),
            (None, false) => None,
        };
        let kmode = match kmode {
            Some(m) if p.hdr.is_none() => match make_hdr10(p.slots.len()) {
                Ok(h) => {
                    p.hdr = Some(h);
                    Some(m)
                }
                Err(e) => {
                    crate::error!("present: HDR kernel: {e}");
                    None
                }
            },
            m => m,
        };
        let out_size = kmode.map(|_| (p.extent.width, p.extent.height));
        if let Err(e) = prepare_slot(&mut p.slots[k], sw, sh, fmt, &src, out_size) {
            crate::error!("present: {e}");
            return false;
        }
        let (idx, _) = match c.swapchain_d.acquire_next_image(p.swapchain, ACQUIRE_TIMEOUT_NS, p.slots[k].acquire, vk::Fence::null()) {
            Ok(r) => r,
            Err(vk::Result::TIMEOUT | vk::Result::NOT_READY) => {
                p.skipped += 1;
                note_skipped(p.skipped);
                return false;
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                window::RESIZED.store(true, std::sync::atomic::Ordering::SeqCst);
                return false;
            }
            Err(e) => {
                crate::error!("acquire: {e}");
                return false;
            }
        };
        p.frame += 1;
        let two = c.present_queue.is_some();
        let s = &p.slots[k];
        let (copy_img, ..) = s.image.unwrap();
        // The copy, on the game's queue after its work.
        let cmd = s.copy;
        c.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).ok();
        c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
        barrier(c, cmd, copy_img, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        let layers = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
        match &src {
            Source::Image { image, layout, .. } => {
                // (Its layout kept: the game's writes before the copy, the
                // copy before its later writes.)
                barrier(c, cmd, *image, *layout, *layout);
                let region = vk::ImageCopy { src_subresource: layers, src_offset: vk::Offset3D::default(), dst_subresource: layers, dst_offset: vk::Offset3D::default(), extent: vk::Extent3D { width: sw, height: sh, depth: 1 } };
                c.device.cmd_copy_image(cmd, *image, *layout, copy_img, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);
                barrier(c, cmd, *image, *layout, *layout);
            }
            Source::Pixels { data, .. } => {
                let (sb, _, sp, _) = s.staging.unwrap();
                std::ptr::copy_nonoverlapping(data.as_ptr(), sp, (sw * sh * 4) as usize);
                let region = vk::BufferImageCopy { buffer_offset: 0, buffer_row_length: 0, buffer_image_height: 0, image_subresource: layers, image_offset: vk::Offset3D::default(), image_extent: vk::Extent3D { width: sw, height: sh, depth: 1 } };
                c.device.cmd_copy_buffer_to_image(cmd, sb, copy_img, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);
            }
        }
        barrier(c, cmd, copy_img, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        c.device.end_command_buffer(cmd).ok();
        let copied: &[vk::Semaphore] = if two { std::slice::from_ref(&s.copied) } else { &[] };
        if let Err(e) = c.submit(cmd, &[], copied, vk::Fence::null()) {
            crate::error!("{e}");
            return false;
        }
        // The blit into the swapchain image, on the present queue.
        let cmd = s.blit;
        c.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).ok();
        c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
        let dst = p.images[idx as usize];
        let (dx0, dy0, dx1, dy1) = dst_rect(sw, sh, p.extent.width, p.extent.height);
        let filter = match crate::config::settings().filter {
            crate::config::Filter::Smooth => vk::Filter::LINEAR,
            crate::config::Filter::Sharp => vk::Filter::NEAREST,
        };
        if let (Some((scale, flags)), Some(h)) = (kmode, p.hdr.as_ref()) {
            // HDR10: the kernel samples the copy into the slot's output (black
            // outside the frame's rectangle), copied into the swapchain image.
            let (out, _, out_view, ow, oh) = s.out.unwrap();
            barrier(c, cmd, copy_img, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            barrier(c, cmd, out, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL);
            let set = h.sets[k];
            let src_info = [vk::DescriptorImageInfo { sampler: h.samplers[(filter == vk::Filter::NEAREST) as usize], image_view: s.view.unwrap(), image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL }];
            let out_info = [vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: out_view, image_layout: vk::ImageLayout::GENERAL }];
            let writes = [
                vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&src_info),
                vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&out_info),
            ];
            c.device.update_descriptor_sets(&writes, &[]);
            c.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, h.pipeline);
            c.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, h.layout, 0, &[set], &[]);
            let pc: [u32; 8] = [(dx0 as f32).to_bits(), (dy0 as f32).to_bits(), (dx1 as f32).to_bits(), (dy1 as f32).to_bits(), ow, oh, scale.to_bits(), flags];
            let bytes: Vec<u8> = pc.iter().flat_map(|v| v.to_le_bytes()).collect();
            c.device.cmd_push_constants(cmd, h.layout, vk::ShaderStageFlags::COMPUTE, 0, &bytes);
            c.device.cmd_dispatch(cmd, ow.div_ceil(8), oh.div_ceil(8), 1);
            barrier(c, cmd, out, vk::ImageLayout::GENERAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            barrier(c, cmd, dst, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            let region = vk::ImageCopy { src_subresource: layers, src_offset: vk::Offset3D::default(), dst_subresource: layers, dst_offset: vk::Offset3D::default(), extent: vk::Extent3D { width: ow.min(p.extent.width), height: oh.min(p.extent.height), depth: 1 } };
            c.device.cmd_copy_image(cmd, out, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);
            if flags & 8 == 0 {
                hdr_check(c, cmd, out, ow, oh, scale, pq);
            }
        } else {
        barrier(c, cmd, dst, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        if (dx0, dy0, dx1, dy1) != (0, 0, p.extent.width as i32, p.extent.height as i32) {
            // Black bars around the frame.
            let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
            c.device.cmd_clear_color_image(cmd, dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] }, &[range]);
            let b = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            c.device.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[b], &[], &[]);
        }
        let blit = vk::ImageBlit {
            src_subresource: layers,
            src_offsets: [vk::Offset3D::default(), vk::Offset3D { x: sw as i32, y: sh as i32, z: 1 }],
            dst_subresource: layers,
            dst_offsets: [vk::Offset3D { x: dx0, y: dy0, z: 0 }, vk::Offset3D { x: dx1, y: dy1, z: 1 }],
        };
        c.device.cmd_blit_image(cmd, copy_img, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[blit], filter);
        }
        barrier(c, cmd, dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR);
        c.device.end_command_buffer(cmd).ok();
        let done = p.done[idx as usize];
        let queue = c.present_queue.as_ref().unwrap_or(&c.queue);
        let mut waits = vec![(s.acquire, vk::PipelineStageFlags::TRANSFER)];
        if two {
            // (The HDR10 kernel reads the copy.)
            waits.push((s.copied, vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COMPUTE_SHADER));
        }
        c.device.reset_fences(&[s.fence]).ok();
        if let Err(e) = c.submit_to(queue, &[cmd], &waits, &[done], s.fence) {
            crate::error!("{e}");
            return false;
        }
        let scs = [p.swapchain];
        let idxs = [idx];
        let waits = [done];
        let pi = vk::PresentInfoKHR::default().wait_semaphores(&waits).swapchains(&scs).image_indices(&idxs);
        let q = queue.lock();
        match c.swapchain_d.queue_present(*q, &pi) {
            Ok(false) => {}
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => window::RESIZED.store(true, std::sync::atomic::Ordering::SeqCst),
            Err(e) => crate::error!("present: {e}"),
        }
    }
    true
}

/// Destroys the replaced swapchains whose images no blit uses any more:
/// every slot has been used since the replacement (`FRAMES` presents) and
/// its blit has finished (the present queue runs in order, so the earlier
/// blits and presents into the old swapchain are done too). Never waits.
unsafe fn release_retired(p: &mut Presenter) {
    if p.retired.is_empty() {
        return;
    }
    let c = ctx();
    let idle = p.slots.iter().all(|s| unsafe { c.device.get_fence_status(s.fence) } == Ok(true));
    let frame = p.frame;
    p.retired.retain(|(at, sc, sems)| {
        if !idle || frame < at + FRAMES {
            return true;
        }
        unsafe {
            c.swapchain_d.destroy_swapchain(*sc, None);
            for &s in sems {
                c.device.destroy_semaphore(s, None);
            }
        }
        false
    });
}

/// Logs the first frames left out and then every 300th.
fn note_skipped(n: u64) {
    if n <= 3 || n % 300 == 0 {
        crate::info!("present: {n} frames not shown so far (the window was not taking images)");
    }
}

/// The slot's copy image (and staging buffer for pixels) for a `w` x `h`
/// frame of `fmt`, and with HDR10 output (`out`: the swapchain's size) the
/// kernel's view and output image; the slot is idle (its fence signalled).
unsafe fn prepare_slot(s: &mut Slot, w: u32, h: u32, fmt: vk::Format, src: &Source, out: Option<(u32, u32)>) -> Result<(), String> {
    let c = ctx();
    unsafe {
        if s.image.is_none_or(|i| (i.2, i.3, i.4) != (w, h, fmt)) {
            if let Some(v) = s.view.take() {
                c.device.destroy_image_view(v, None);
            }
            if let Some((img, mem, ..)) = s.image.take() {
                c.device.destroy_image(img, None);
                c.free(mem);
            }
            let ici = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(fmt)
                .extent(vk::Extent3D { width: w, height: h, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED);
            let img = c.device.create_image(&ici, None).map_err(|e| e.to_string())?;
            let mem = c.alloc(c.device.get_image_memory_requirements(img), vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            c.device.bind_image_memory(img, mem, 0).map_err(|e| e.to_string())?;
            s.image = Some((img, mem, w, h, fmt));
        }
        let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
        if out.is_some() && s.view.is_none() {
            let (img, ..) = s.image.unwrap();
            let vci = vk::ImageViewCreateInfo::default().image(img).view_type(vk::ImageViewType::TYPE_2D).format(fmt).subresource_range(range);
            s.view = Some(c.device.create_image_view(&vci, None).map_err(|e| e.to_string())?);
        }
        if let Some((ow, oh)) = out {
            if s.out.is_none_or(|o| (o.3, o.4) != (ow, oh)) {
                if let Some((img, mem, view, ..)) = s.out.take() {
                    c.device.destroy_image_view(view, None);
                    c.device.destroy_image(img, None);
                    c.free(mem);
                }
                let ici = vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R32_UINT)
                    .extent(vk::Extent3D { width: ow, height: oh, depth: 1 })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC);
                let img = c.device.create_image(&ici, None).map_err(|e| e.to_string())?;
                let mem = c.alloc(c.device.get_image_memory_requirements(img), vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
                c.device.bind_image_memory(img, mem, 0).map_err(|e| e.to_string())?;
                let vci = vk::ImageViewCreateInfo::default().image(img).view_type(vk::ImageViewType::TYPE_2D).format(vk::Format::R32_UINT).subresource_range(range);
                let view = c.device.create_image_view(&vci, None).map_err(|e| e.to_string())?;
                s.out = Some((img, mem, view, ow, oh));
            }
        }
        if let Source::Pixels { data, .. } = src {
            let size = (w * h * 4) as u64;
            if data.len() < size as usize {
                return Err(format!("{} bytes of pixels for {w}x{h}", data.len()));
            }
            if s.staging.as_ref().is_none_or(|st| st.3 < size) {
                let (b, m, ptr) = c.host_buffer(size, vk::BufferUsageFlags::TRANSFER_SRC)?;
                s.staging = Some((b, m, ptr, size));
            }
        }
    }
    Ok(())
}
