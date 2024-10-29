use crate::renderer::swapchain::Swapchain;
use crate::renderer::{Renderer, RendererAttributes};
use crate::rendering_context::{ImageLayoutState, RenderingContext};
use ash::vk;
use ash::vk::CommandBuffer;
use std::sync::Arc;
use winit::window::Window;

use crate::renderer::commands::Commands;
use anyhow::Result;
use tracing::trace;

struct Frame {
    command_buffer: CommandBuffer,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    in_flight_fence: vk::Fence,
}

#[derive(Clone)]
pub struct WindowRendererAttributes {
    pub format: vk::Format,
    pub depth_format: vk::Format,
    pub clear_color: vk::ClearColorValue,
    pub ssaa: f32,
    pub ssaa_filter: vk::Filter,
    pub in_flight_frames_count: usize,
}

pub struct WindowRenderer {
    frame_index: usize,
    frames: Vec<Frame>,
    command_pool: vk::CommandPool,
    swapchain: Swapchain,
    context: Arc<RenderingContext>,

    attributes: WindowRendererAttributes,

    pub renderer: Renderer,
    pub window: Arc<Window>,
}

fn scale_extent(extent: vk::Extent2D, scale: f32) -> vk::Extent2D {
    vk::Extent2D {
        width: (extent.width as f32 * scale) as u32,
        height: (extent.height as f32 * scale) as u32,
    }
}

impl WindowRenderer {
    pub fn new(
        context: Arc<RenderingContext>,
        window: Arc<Window>,
        attributes: WindowRendererAttributes,
    ) -> Result<Self> {
        let mut swapchain = Swapchain::new(context.clone(), window.clone())?;
        swapchain.resize()?;

        unsafe {
            let command_pool = context.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(context.queue_families.graphics)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;

            let command_buffers = context.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(attributes.in_flight_frames_count as u32),
            )?;

            let mut frames = Vec::with_capacity(command_buffers.len());

            for &command_buffer in command_buffers.iter() {
                let image_available_semaphore = context
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
                let render_finished_semaphore = context
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
                let in_flight_fence = context.device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )?;

                frames.push(Frame {
                    command_buffer,
                    image_available_semaphore,
                    render_finished_semaphore,
                    in_flight_fence,
                });
            }

            let command_buffer = frames[0].command_buffer;

            let commands = Commands::new(context.clone(), command_buffer)?;

            let renderer = Renderer::new(
                context.clone(),
                &commands,
                RendererAttributes {
                    extent: scale_extent(swapchain.extent, attributes.ssaa),
                    format: attributes.format,
                    depth_format: attributes.depth_format,
                    buffering: attributes.in_flight_frames_count,
                },
            )?;

            let fence = context
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)?;

            commands.submit(
                context.queues[context.queue_families.graphics as usize],
                Default::default(),
                Default::default(),
                fence,
            )?;

            context.device.wait_for_fences(&[fence], true, u64::MAX)?;

            context.device.destroy_fence(fence, None);

            Ok(Self {
                frame_index: 0,
                frames,
                command_pool,
                swapchain,
                context,
                renderer,
                window,
                attributes,
            })
        }
    }

    pub fn resize(&mut self) {
        self.swapchain.is_dirty = true;
    }

    pub fn render(&mut self) -> Result<()> {
        let frame = &self.frames[self.frame_index];

        unsafe {
            self.context
                .device
                .wait_for_fences(&[frame.in_flight_fence], true, u64::MAX)?;

            if self.swapchain.is_dirty {
                self.context.device.device_wait_idle()?;
                self.swapchain.resize()?;
                let swapchain_extent = self.swapchain.extent;
                if swapchain_extent.width == 0 || swapchain_extent.height == 0 {
                    return Ok(());
                }
                self.renderer
                    .resize(scale_extent(swapchain_extent, self.attributes.ssaa))?;
            }

            let swapchain_extent = self.swapchain.extent;

            if swapchain_extent.width == 0 || swapchain_extent.height == 0 {
                return Ok(());
            }

            let image_index = match self
                .swapchain
                .acquire_next_image(frame.image_available_semaphore)
            {
                Ok(image_index) => image_index,
                Err(_) => {
                    self.swapchain.is_dirty = true;
                    return Ok(());
                }
            };

            trace!(
                "Rendering frame {} to image {}",
                self.frame_index,
                image_index
            );

            let graphics_queue = self.context.queues[self.context.queue_families.graphics as usize];

            self.context.device.reset_fences(&[frame.in_flight_fence])?;

            let command_buffer = frame.command_buffer;

            let swapchain_image = &mut self.swapchain.images[image_index as usize];
            let commands = Commands::new(self.context.clone(), command_buffer)?;
            let render_target =
                self.renderer
                    .render(&commands, self.attributes.clear_color, self.frame_index)?;
            commands
                .blit_full_image(render_target, swapchain_image, self.attributes.ssaa_filter)
                .transition_image_layout(swapchain_image, ImageLayoutState::present())
                .submit(
                    graphics_queue,
                    (
                        frame.image_available_semaphore,
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    ),
                    (
                        frame.render_finished_semaphore,
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    ),
                    frame.in_flight_fence,
                )?;

            self.swapchain
                .present(image_index, frame.render_finished_semaphore)?;

            self.frame_index = (self.frame_index + 1) % self.attributes.in_flight_frames_count;
            Ok(())
        }
    }
}

impl Drop for WindowRenderer {
    fn drop(&mut self) {
        unsafe {
            self.context.device.device_wait_idle().unwrap();

            self.frames.drain(..).for_each(|frame| {
                self.context
                    .device
                    .destroy_semaphore(frame.image_available_semaphore, None);
                self.context
                    .device
                    .destroy_semaphore(frame.render_finished_semaphore, None);
                self.context
                    .device
                    .destroy_fence(frame.in_flight_fence, None);
                self.context
                    .device
                    .free_command_buffers(self.command_pool, &[frame.command_buffer]);
            });
            self.context
                .device
                .destroy_command_pool(self.command_pool, None);
        }
    }
}
