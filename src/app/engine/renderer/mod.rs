mod swapchain;
pub mod window_renderer;

use crate::app::engine::rendering_context::{
    Image, ImageAttributes, ImageLayoutState, RenderingContext,
};
use anyhow::Result;
use ash::vk;
use ash::vk::{CommandBuffer, Extent3D};
use gpu_allocator::vulkan::{AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;
use std::sync::Arc;

pub struct Renderer {
    allocator: Allocator,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    context: Arc<RenderingContext>,
    render_targets: Vec<Image>,
}

const SHADERS_DIR: &str = "res/shaders/";

fn load_shader_module(context: &RenderingContext, path: &str) -> Result<vk::ShaderModule> {
    let code = std::fs::read(format!("{}{}", SHADERS_DIR, path))?;
    context.create_shader_module(&code)
}

fn create_render_target(
    context: &RenderingContext,
    allocator: &mut Allocator,
    resolution: (u32, u32),
    format: vk::Format,
) -> Result<Image> {
    context.create_image(
        allocator,
        "render target",
        ImageAttributes {
            extent: Extent3D::default()
                .width(resolution.0)
                .height(resolution.1)
                .depth(1),
            format,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        },
    )
}

impl Renderer {
    pub fn new(
        context: Arc<RenderingContext>,
        resolution: (u32, u32),
        format: vk::Format,
        buffering: usize,
    ) -> Result<Self> {
        let vertex_shader = load_shader_module(context.as_ref(), "vert.spv")?;
        let fragment_shader = load_shader_module(context.as_ref(), "frag.spv")?;

        let mut allocator = context.create_allocator(Default::default(), Default::default())?;

        let render_targets = (0..buffering)
            .map(|_| create_render_target(context.as_ref(), &mut allocator, resolution, format))
            .collect::<Result<Vec<_>>>()?;

        unsafe {
            let pipeline_layout = context
                .device
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)?;

            let pipeline = context.create_graphics_pipeline(
                vertex_shader,
                fragment_shader,
                vk::Extent2D {
                    width: resolution.0,
                    height: resolution.1,
                },
                format,
                pipeline_layout,
                Default::default(),
            )?;

            context.device.destroy_shader_module(vertex_shader, None);
            context.device.destroy_shader_module(fragment_shader, None);

            Ok(Self {
                allocator,
                pipeline,
                pipeline_layout,
                context,
                render_targets,
            })
        }
    }

    pub fn resize(&mut self, resolution: (u32, u32)) -> Result<()> {
        for render_target in self.render_targets.iter_mut() {
            self.context
                .destroy_image(&mut self.allocator, render_target)?;
            *render_target = create_render_target(
                self.context.as_ref(),
                &mut self.allocator,
                resolution,
                render_target.attributes.format,
            )?;
        }
        Ok(())
    }

    pub fn render(
        &mut self,
        command_buffer: CommandBuffer,
        clear_color: vk::ClearColorValue,
        render_target_index: usize,
    ) -> Result<()> {
        let render_target = &mut self.render_targets[render_target_index];

        let state = ImageLayoutState {
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        };

        unsafe {
            self.context.transition_image_layout(
                command_buffer,
                render_target.handle,
                ImageLayoutState {
                    layout: vk::ImageLayout::UNDEFINED,
                    access_mask: vk::AccessFlags2::NONE,
                    stage_mask: vk::PipelineStageFlags2::NONE,
                    queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                },
                state,
            );
            render_target.layout = state;

            self.context.begin_rendering(
                command_buffer,
                render_target.view,
                clear_color,
                vk::Rect2D::default().extent(
                    vk::Extent2D::default()
                        .width(render_target.attributes.extent.width)
                        .height(render_target.attributes.extent.height),
                ),
            );
            self.draw(command_buffer, render_target_index);
            self.context.device.cmd_end_rendering(command_buffer);

            Ok(())
        }
    }

    pub fn draw(&self, command_buffer: CommandBuffer, render_target_index: usize) {
        let render_target = &self.render_targets[render_target_index];

        unsafe {
            self.context.device.cmd_set_viewport(
                command_buffer,
                0,
                &[vk::Viewport::default()
                    .width(render_target.attributes.extent.width as f32)
                    .height(render_target.attributes.extent.height as f32)],
            );

            self.context.device.cmd_set_scissor(
                command_buffer,
                0,
                &[vk::Rect2D::default().extent(
                    vk::Extent2D::default()
                        .width(render_target.attributes.extent.width)
                        .height(render_target.attributes.extent.height),
                )],
            );

            self.context.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            self.context.device.cmd_draw(command_buffer, 3, 1, 0, 0);
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.context.device.device_wait_idle().unwrap();

            for render_target in self.render_targets.iter_mut() {
                self.context
                    .destroy_image(&mut self.allocator, render_target)
                    .unwrap();
            }

            self.context.device.destroy_pipeline(self.pipeline, None);
            self.context
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
