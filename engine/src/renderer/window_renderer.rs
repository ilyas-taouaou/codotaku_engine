use crate::renderer::swapchain::Swapchain;
use crate::renderer::Renderer;
use crate::rendering_context::{ImageLayoutState, RenderingContext};
use ash::vk;
use ash::vk::CommandBuffer;
use std::sync::Arc;
use winit::window::Window;

use crate::renderer::commands::Commands;
use anyhow::Result;

struct Frame {
    command_buffer: CommandBuffer,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    in_flight_fence: vk::Fence,
}

pub struct WindowRenderer {
    in_flight_frames_count: usize,
    frame_index: usize,
    frames: Vec<Frame>,
    command_pool: vk::CommandPool,
    swapchain: Swapchain,
    context: Arc<RenderingContext>,

    clear_color: vk::ClearColorValue,

    pub renderer: Renderer,
    pub window: Arc<Window>,
}

impl WindowRenderer {
    pub fn new(
        context: Arc<RenderingContext>,
        window: Arc<Window>,
        in_flight_frames_count: usize,
        format: vk::Format,
        clear_color: vk::ClearColorValue,
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
                    .command_buffer_count(in_flight_frames_count as u32),
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

            let renderer = Renderer::new(
                context.clone(),
                swapchain.extent,
                format,
                in_flight_frames_count,
            )?;

            Ok(Self {
                in_flight_frames_count,
                frame_index: 0,
                frames,
                command_pool,
                swapchain,
                context,
                clear_color,
                renderer,
                window,
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
                self.renderer.resize(swapchain_extent)?;
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

            let graphics_queue = self.context.queues[self.context.queue_families.graphics as usize];

            self.context.device.reset_fences(&[frame.in_flight_fence])?;

            let command_buffer = frame.command_buffer;

            let swapchain_image = &mut self.swapchain.images[image_index as usize];
            let commands = Commands::new(self.context.clone(), command_buffer)?;
            let render_target =
                self.renderer
                    .render(&commands, self.clear_color, self.frame_index)?;
            commands
                .transition_image_layout(swapchain_image, ImageLayoutState::transfer_destination())
                .transition_image_layout(render_target, ImageLayoutState::transfer_source())
                .blit_image(
                    render_target,
                    swapchain_image,
                    render_target.attributes.extent,
                    swapchain_extent.into(),
                )
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

            self.frame_index = (self.frame_index + 1) % self.in_flight_frames_count;
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
