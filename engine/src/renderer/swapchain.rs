use crate::rendering_context::{Image, ImageAttributes, RenderingContext, Surface};
use anyhow::Result;
use ash::vk;
use ash::vk::AcquireNextImageInfoKHR;
use gpu_allocator::vulkan::AllocationScheme;
use gpu_allocator::MemoryLocation;
use std::sync::Arc;
use winit::window::Window;

pub struct Swapchain {
    pub desired_image_count: u32,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub images: Vec<Image>,
    handle: vk::SwapchainKHR,
    surface: Surface,
    window: Arc<Window>,
    context: Arc<RenderingContext>,
    pub is_dirty: bool,
}

impl Swapchain {
    pub fn new(context: Arc<RenderingContext>, window: Arc<Window>) -> Result<Self> {
        let surface = unsafe { context.create_surface(window.as_ref())? };
        let format = vk::Format::B8G8R8A8_SRGB;
        let extent = if surface.capabilities.current_extent.width != u32::MAX {
            surface.capabilities.current_extent
        } else {
            let size = window.inner_size();
            vk::Extent2D {
                width: size.width,
                height: size.height,
            }
        };
        let desired_image_count = (surface.capabilities.min_image_count + 1).clamp(
            surface.capabilities.min_image_count,
            if surface.capabilities.max_image_count == 0 {
                u32::MAX
            } else {
                surface.capabilities.max_image_count
            },
        );

        Ok(Self {
            desired_image_count,
            format,
            extent,
            images: Default::default(),
            handle: Default::default(),
            surface,
            window,
            context,
            is_dirty: true,
        })
    }

    pub fn resize(&mut self) -> Result<()> {
        let size = self.window.inner_size();
        self.extent = vk::Extent2D {
            width: size.width,
            height: size.height,
        };

        if self.extent.width == 0 || self.extent.height == 0 {
            return Ok(());
        }

        self.is_dirty = false;

        unsafe {
            let new_swapchain = self.context.swapchain_extension.create_swapchain(
                &vk::SwapchainCreateInfoKHR::default()
                    .surface(self.surface.handle)
                    .min_image_count(self.desired_image_count)
                    .image_format(self.format)
                    .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                    .image_extent(self.extent)
                    .image_array_layers(1)
                    .image_usage(
                        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    )
                    .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                    .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                    .present_mode(vk::PresentModeKHR::MAILBOX)
                    .clipped(true)
                    .old_swapchain(self.handle),
                None,
            )?;
            self.images.drain(..).for_each(|image| {
                self.context.device.destroy_image_view(image.view, None);
            });
            self.context
                .swapchain_extension
                .destroy_swapchain(self.handle, None);

            self.handle = new_swapchain;
            self.images = self
                .context
                .swapchain_extension
                .get_swapchain_images(self.handle)?
                .into_iter()
                .map(|handle| {
                    Ok(Image::wrap(
                        self.context.clone(),
                        handle,
                        ImageAttributes {
                            format: self.format,
                            extent: self.extent.into(),
                            usage: vk::ImageUsageFlags::TRANSFER_DST
                                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                            location: MemoryLocation::Unknown,
                            linear: false,
                            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                            subresource_range: vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        },
                    )?)
                })
                .collect::<Result<Vec<_>>>()?;
        }
        Ok(())
    }

    pub fn acquire_next_image(&mut self, image_available_semaphore: vk::Semaphore) -> Result<u32> {
        let (image_index, is_suboptimal) = unsafe {
            self.context.swapchain_extension.acquire_next_image2(
                &AcquireNextImageInfoKHR::default()
                    .swapchain(self.handle)
                    .timeout(u64::MAX)
                    .semaphore(image_available_semaphore)
                    .fence(vk::Fence::null())
                    .device_mask(1),
            )?
        };
        if is_suboptimal {
            self.is_dirty = true;
        }
        Ok(image_index)
    }

    pub fn present(
        &mut self,
        image_index: u32,
        render_finished_semaphore: vk::Semaphore,
    ) -> Result<()> {
        let is_suboptimal = unsafe {
            match self.context.swapchain_extension.queue_present(
                self.context.queues[self.context.queue_families.present as usize],
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(&[render_finished_semaphore])
                    .swapchains(&[self.handle])
                    .image_indices(&[image_index]),
            ) {
                Ok(is_suboptimal) => is_suboptimal,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
                Err(error) => return Err(error.into()),
            }
        };
        if is_suboptimal {
            self.is_dirty = true;
        }
        Ok(())
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.images.drain(..).for_each(|image| {
                self.context.device.destroy_image_view(image.view, None);
            });
            self.context
                .swapchain_extension
                .destroy_swapchain(self.handle, None);
            self.context
                .surface_extension
                .destroy_surface(self.surface.handle, None);
        }
    }
}
