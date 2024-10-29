mod commands;
mod swapchain;
pub mod window_renderer;

use crate::renderer::commands::Commands;
use crate::rendering_context::{Image, ImageAttributes, ImageLayoutState, RenderingContext};
use anyhow::Result;
use ash::vk;
use ash::vk::Extent3D;
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
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            subresource_range: vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1),
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
        commands: &Commands,
        clear_color: vk::ClearColorValue,
        render_target_index: usize,
    ) -> Result<&mut Image> {
        let render_target = &mut self.render_targets[render_target_index];

        render_target.reset_layout();

        unsafe {
            commands
                .transition_image_layout(render_target, ImageLayoutState::color_attachment())
                .begin_rendering(
                    render_target.view,
                    clear_color,
                    vk::Rect2D::default().extent(
                        vk::Extent2D::default()
                            .width(render_target.attributes.extent.width)
                            .height(render_target.attributes.extent.height),
                    ),
                );
            self.draw(commands, render_target_index);
            commands.end_rendering();

            Ok(&mut self.render_targets[render_target_index])
        }
    }

    pub fn draw(&self, commands: &Commands, render_target_index: usize) {
        let render_target = &self.render_targets[render_target_index];

        unsafe {
            commands
                .set_viewport(
                    vk::Viewport::default()
                        .width(render_target.attributes.extent.width as f32)
                        .height(render_target.attributes.extent.height as f32),
                )
                .set_scissor(
                    vk::Rect2D::default().extent(
                        vk::Extent2D::default()
                            .width(render_target.attributes.extent.width)
                            .height(render_target.attributes.extent.height),
                    ),
                )
                .bind_pipeline(self.pipeline)
                .draw(0..3, 0..1);
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
