//! Swapchain presentation: blits a rendered image (or uploaded pixels) to the window.

use super::{ctx, window};
use ash::vk;
use parking_lot::Mutex;

const FRAMES: usize = 2;

struct Presenter {
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    format: vk::Format,
    extent: vk::Extent2D,
    pool: vk::CommandPool,
    cmds: Vec<vk::CommandBuffer>,
    fences: Vec<vk::Fence>,
    acquire: Vec<vk::Semaphore>,
    done: Vec<vk::Semaphore>,
    frame: usize,
    staging: Option<(vk::Buffer, vk::DeviceMemory, *mut u8, u64)>,
    upload: Option<(vk::Image, vk::DeviceMemory, u32, u32)>,
}

unsafe impl Send for Presenter {}

static PRESENTER: Mutex<Option<Presenter>> = Mutex::new(None);

pub enum Source {
    /// An image in TRANSFER_SRC_OPTIMAL-compatible state; `layout` is its current layout.
    Image { image: vk::Image, layout: vk::ImageLayout, width: u32, height: u32 },
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
        // swapchain would encode them a second time.
        let fmt = formats
            .iter()
            .find(|f| f.format == vk::Format::B8G8R8A8_UNORM)
            .or_else(|| formats.iter().find(|f| f.format == vk::Format::R8G8B8A8_UNORM))
            .or_else(|| formats.first())
            .copied()
            .ok_or("no surface format")?;
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
        let sc = c.swapchain_d.create_swapchain(&sci, None).map_err(|e| format!("create swapchain: {e}"))?;
        if old != vk::SwapchainKHR::null() {
            c.device.device_wait_idle().ok();
            c.swapchain_d.destroy_swapchain(old, None);
            for s in p.done.drain(..) {
                c.device.destroy_semaphore(s, None);
            }
        }
        p.swapchain = sc;
        p.images = c.swapchain_d.get_swapchain_images(sc).map_err(|e| format!("swapchain images: {e}"))?;
        p.format = fmt.format;
        p.extent = extent;
        for _ in 0..p.images.len() {
            p.done.push(c.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).map_err(|e| e.to_string())?);
        }
        crate::info!("swapchain {}x{} {:?} ({} images, {:?})", extent.width, extent.height, fmt.format, p.images.len(), sci.present_mode);
    }
    Ok(())
}

/// The swapchain's present mode: FIFO (vsync), or `TLG_PRESENT=mailbox`
/// (vsync without back-pressure: the newest frame is shown, for uncapped
/// rates above the display's) or `TLG_PRESENT=immediate` (no vsync, tears)
/// when the surface supports them.
fn present_mode(surface: vk::SurfaceKHR) -> vk::PresentModeKHR {
    let want = match std::env::var("TLG_PRESENT").as_deref() {
        Ok("mailbox") => vk::PresentModeKHR::MAILBOX,
        Ok("immediate") => vk::PresentModeKHR::IMMEDIATE,
        _ => return vk::PresentModeKHR::FIFO,
    };
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
    let w = window::create(title, 1280, 720)?;
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
            .allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(FRAMES as u32))
            .map_err(|e| e.to_string())?;
        let mut p = Presenter {
            surface,
            swapchain: vk::SwapchainKHR::null(),
            images: Vec::new(),
            format: vk::Format::UNDEFINED,
            extent: vk::Extent2D::default(),
            pool,
            cmds,
            fences: (0..FRAMES).map(|_| c.device.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None).unwrap()).collect(),
            acquire: (0..FRAMES).map(|_| c.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap()).collect(),
            done: Vec::new(),
            frame: 0,
            staging: None,
            upload: None,
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

/// Presents one frame. Returns false if presentation is not possible (minimized).
pub fn present(src: Source) -> bool {
    // (The frame's batches are submitted ahead of the blit: queue order.)
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
        let f = p.frame % FRAMES;
        p.frame += 1;
        c.device.wait_for_fences(&[p.fences[f]], true, u64::MAX).ok();
        let (idx, _) = match c.swapchain_d.acquire_next_image(p.swapchain, u64::MAX, p.acquire[f], vk::Fence::null()) {
            Ok(r) => r,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                window::RESIZED.store(true, std::sync::atomic::Ordering::SeqCst);
                return false;
            }
            Err(e) => {
                crate::error!("acquire: {e}");
                return false;
            }
        };
        c.device.reset_fences(&[p.fences[f]]).ok();
        let cmd = p.cmds[f];
        c.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).ok();
        c.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).ok();
        let dst = p.images[idx as usize];
        barrier(c, cmd, dst, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        let (src_img, src_layout, sw, sh) = match src {
            Source::Image { image, layout, width, height } => {
                if layout != vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
                    barrier(c, cmd, image, layout, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
                }
                (image, layout, width, height)
            }
            Source::Pixels { data, width, height } => match upload_pixels(p, cmd, &data, width, height) {
                Ok(img) => (img, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, width, height),
                Err(e) => {
                    crate::error!("present upload: {e}");
                    c.device.end_command_buffer(cmd).ok();
                    return false;
                }
            },
        };
        let blit = vk::ImageBlit {
            src_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
            src_offsets: [vk::Offset3D::default(), vk::Offset3D { x: sw as i32, y: sh as i32, z: 1 }],
            dst_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
            dst_offsets: [vk::Offset3D::default(), vk::Offset3D { x: p.extent.width as i32, y: p.extent.height as i32, z: 1 }],
        };
        c.device.cmd_blit_image(cmd, src_img, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[blit], vk::Filter::LINEAR);
        if src_layout != vk::ImageLayout::TRANSFER_SRC_OPTIMAL && src_img != vk::Image::null() {
            barrier(c, cmd, src_img, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, src_layout);
        }
        barrier(c, cmd, dst, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR);
        c.device.end_command_buffer(cmd).ok();
        let done = p.done[idx as usize];
        if let Err(e) = c.submit(cmd, &[(p.acquire[f], vk::PipelineStageFlags::TRANSFER)], &[done], p.fences[f]) {
            crate::error!("{e}");
            return false;
        }
        let scs = [p.swapchain];
        let idxs = [idx];
        let waits = [done];
        let pi = vk::PresentInfoKHR::default().wait_semaphores(&waits).swapchains(&scs).image_indices(&idxs);
        let q = c.queue.lock();
        match c.swapchain_d.queue_present(*q, &pi) {
            Ok(false) => {}
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => window::RESIZED.store(true, std::sync::atomic::Ordering::SeqCst),
            Err(e) => crate::error!("present: {e}"),
        }
    }
    true
}

unsafe fn upload_pixels(p: &mut Presenter, cmd: vk::CommandBuffer, data: &[u8], w: u32, h: u32) -> Result<vk::Image, String> {
    let c = ctx();
    let size = (w * h * 4) as u64;
    unsafe {
        if p.staging.as_ref().is_none_or(|s| s.3 < size) {
            let (b, m, ptr) = c.host_buffer(size, vk::BufferUsageFlags::TRANSFER_SRC)?;
            p.staging = Some((b, m, ptr, size));
        }
        if p.upload.as_ref().is_none_or(|u| u.2 != w || u.3 != h) {
            let ici = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::B8G8R8A8_UNORM)
                .extent(vk::Extent3D { width: w, height: h, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC);
            let img = c.device.create_image(&ici, None).map_err(|e| e.to_string())?;
            let mem = c.alloc(c.device.get_image_memory_requirements(img), vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            c.device.bind_image_memory(img, mem, 0).map_err(|e| e.to_string())?;
            p.upload = Some((img, mem, w, h));
        }
        let (sb, _, sp, _) = p.staging.unwrap();
        std::ptr::copy_nonoverlapping(data.as_ptr(), sp, size as usize);
        let img = p.upload.unwrap().0;
        barrier(c, cmd, img, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        let region = vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D { width: w, height: h, depth: 1 },
        };
        c.device.cmd_copy_buffer_to_image(cmd, sb, img, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);
        barrier(c, cmd, img, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        Ok(img)
    }
}
