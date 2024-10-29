mod commands;
mod swapchain;
pub mod window_renderer;

use crate::buffer::{Buffer, BufferAttributes};
use crate::renderer::commands::Commands;
use crate::rendering_context::{Image, ImageLayoutState, RenderingContext};
use anyhow::Result;
use ash::vk;
use gpu_allocator::vulkan::{AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;
use std::sync::Arc;

pub struct Renderer {
    allocator: Allocator,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    context: Arc<RenderingContext>,
    render_targets: Vec<Image>,
    format: vk::Format,

    vertex_buffer: Buffer,
    index_buffer: Buffer,
    staging_buffer: Buffer,

    indices_count: u32,
}

const SHADERS_DIR: &str = "res/shaders/";

fn load_shader_module(context: &RenderingContext, path: &str) -> Result<vk::ShaderModule> {
    let code = std::fs::read(format!("{}{}", SHADERS_DIR, path))?;
    context.create_shader_module(&code)
}

use nalgebra as na;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: na::Vector3<f32>,
    color: na::Vector3<f32>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConstants {
    vertex_buffer_address: vk::DeviceAddress,
}

impl Renderer {
    pub fn new(
        context: Arc<RenderingContext>,
        resolution: vk::Extent2D,
        format: vk::Format,
        buffering: usize,
        commands: &Commands,
    ) -> Result<Self> {
        let vertex_shader = load_shader_module(context.as_ref(), "vert.spv")?;
        let fragment_shader = load_shader_module(context.as_ref(), "frag.spv")?;

        let mut allocator = context.create_allocator(Default::default(), Default::default())?;

        let render_targets = (0..buffering)
            .map(|_| {
                Image::new_render_target(
                    context.clone(),
                    &mut allocator,
                    "render_target",
                    resolution,
                    format,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        unsafe {
            let pipeline_layout = context.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&[
                    vk::PushConstantRange::default()
                        .stage_flags(vk::ShaderStageFlags::VERTEX)
                        .offset(0)
                        .size(size_of::<PushConstants>() as u32),
                ]),
                None,
            )?;

            let pipeline = context.create_graphics_pipeline(
                vertex_shader,
                fragment_shader,
                resolution,
                format,
                pipeline_layout,
                Default::default(),
            )?;

            context.device.destroy_shader_module(vertex_shader, None);
            context.device.destroy_shader_module(fragment_shader, None);

            // generate circle vertices and indices
            let radius = 1.0;
            let segments = 24;
            let mut vertices = Vec::with_capacity(segments + 1);
            vertices.push(Vertex {
                position: na::Vector3::new(0.0, 0.0, 0.0),
                color: na::Vector3::new(1.0, 1.0, 1.0),
            });
            for i in 0..segments {
                let angle = 2.0 * std::f32::consts::PI * (i as f32) / (segments as f32);
                vertices.push(Vertex {
                    position: na::Vector3::new(radius * angle.cos(), radius * angle.sin(), 0.0),
                    color: na::Vector3::new(1.0, 1.0, 1.0),
                });
            }
            let mut indices = Vec::with_capacity(segments * 3);
            for i in 0..segments {
                indices.push(0);
                indices.push(i as u32 + 1);
                indices.push(((i + 1) % segments) as u32 + 1);
            }

            let mut vertex_buffer = Buffer::new(
                &mut allocator,
                BufferAttributes {
                    name: "vertex_buffer".into(),
                    context: context.clone(),
                    size: (vertices.len() * size_of::<Vertex>()) as vk::DeviceSize,
                    usage: vk::BufferUsageFlags::VERTEX_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::TRANSFER_DST,
                    location: MemoryLocation::GpuOnly,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                },
            )?;

            let mut index_buffer = Buffer::new(
                &mut allocator,
                BufferAttributes {
                    name: "index_buffer".into(),
                    context: context.clone(),
                    size: (indices.len() * size_of::<u32>()) as vk::DeviceSize,
                    usage: vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                    location: MemoryLocation::GpuOnly,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                },
            )?;

            let mut staging_buffer = Buffer::new(
                &mut allocator,
                BufferAttributes {
                    name: "staging_buffer".into(),
                    context: context.clone(),
                    size: vertex_buffer.attributes.size + index_buffer.attributes.size,
                    usage: vk::BufferUsageFlags::TRANSFER_SRC,
                    location: MemoryLocation::CpuToGpu,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                },
            )?;
            staging_buffer.write(&vertices, 0)?;
            staging_buffer.write(&indices, vertex_buffer.attributes.size)?;

            commands.copy_buffer(&staging_buffer, &vertex_buffer, 0);
            commands.copy_buffer(
                &staging_buffer,
                &index_buffer,
                vertex_buffer.attributes.size,
            );

            let indices_count = indices.len() as u32;

            Ok(Self {
                allocator,
                pipeline,
                pipeline_layout,
                context,
                render_targets,
                format,
                vertex_buffer,
                index_buffer,
                staging_buffer,
                indices_count,
            })
        }
    }

    pub fn resize(&mut self, resolution: vk::Extent2D) -> Result<()> {
        for render_target in self.render_targets.iter_mut() {
            render_target.destroy(&mut self.allocator)?;
            *render_target = Image::new_render_target(
                self.context.clone(),
                &mut self.allocator,
                "render_target",
                resolution,
                self.format,
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

        commands
            .transition_image_layout(render_target, ImageLayoutState::color_attachment())
            .begin_rendering(
                render_target,
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

    pub fn draw(&self, commands: &Commands, render_target_index: usize) {
        let render_target = &self.render_targets[render_target_index];

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
            .bind_index_buffer(&self.index_buffer)
            .set_push_constants(
                self.pipeline_layout,
                PushConstants {
                    vertex_buffer_address: self.vertex_buffer.address,
                },
            )
            .draw_indexed(0..self.indices_count, 0..1);
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.context.device.device_wait_idle().unwrap();

            self.index_buffer.destroy(&mut self.allocator).unwrap();
            self.staging_buffer.destroy(&mut self.allocator).unwrap();
            self.vertex_buffer.destroy(&mut self.allocator).unwrap();

            for render_target in self.render_targets.iter_mut() {
                render_target.destroy(&mut self.allocator).unwrap();
            }

            self.context.device.destroy_pipeline(self.pipeline, None);
            self.context
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
