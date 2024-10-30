mod commands;
mod geometry;
mod staging_belt;
mod swapchain;
pub mod window_renderer;

use crate::renderer::commands::Commands;
use crate::renderer::geometry::GPUGeometry;
use crate::renderer::staging_belt::StagingBelt;
use crate::rendering_context::{Image, RenderingContext};
use anyhow::Result;
use ash::vk;
use geometry::Geometry;
use gpu_allocator::vulkan::{AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

struct Frame {
    render_target: Image,
    depth_buffer: Image,
}

pub struct Renderer {
    allocator: Allocator,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    context: Arc<RenderingContext>,
    frames: Vec<Frame>,
    staging_belt: StagingBelt,
    gpu_geometry: GPUGeometry,
    camera_buffer: Buffer,
    cameras: Vec<Camera>,
    pub start_time: Instant,
    attributes: RendererAttributes,
    instance_buffer: Buffer,
    instances: Vec<Instance>,
}

const SHADERS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/res/shaders/");

fn load_shader_module(
    context: &RenderingContext,
    path: impl AsRef<Path>,
) -> Result<vk::ShaderModule> {
    let code = std::fs::read(path)?;
    context.create_shader_module(&code)
}

use crate::buffer::{Buffer, BufferAttributes};
use nalgebra as na;

struct Camera {
    view: na::Isometry3<f32>,
    projection: na::Perspective3<f32>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GPUCamera {
    view: na::Matrix4<f32>,
    projection: na::Matrix4<f32>,
    position: na::Vector3<f32>,
}

struct Instance {
    transform: na::Affine3<f32>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GPUInstance {
    transform: na::Matrix4<f32>,
}

impl Instance {
    fn new(
        position: na::Vector3<f32>,
        rotation: na::UnitQuaternion<f32>,
        scale: na::Vector3<f32>,
    ) -> Self {
        Self {
            transform: na::Affine3::from_matrix_unchecked(
                na::Matrix4::new_translation(&position)
                    * na::Matrix4::from(rotation)
                    * na::Matrix4::new_nonuniform_scaling(&scale),
            ),
        }
    }

    fn to_gpu_instance(&self) -> GPUInstance {
        GPUInstance {
            transform: self.transform.to_homogeneous(),
        }
    }
}

impl Camera {
    fn new(
        eye: &na::Point3<f32>,
        target: &na::Point3<f32>,
        aspect_ratio: f32,
        fovy: f32,
        znear: f32,
        zfar: f32,
    ) -> Self {
        Self {
            view: na::Isometry3::look_at_rh(eye, target, &na::Vector3::y()),
            projection: na::Perspective3::new(aspect_ratio, fovy, znear, zfar),
        }
    }

    fn view_projection(&self) -> na::Matrix4<f32> {
        self.projection.to_homogeneous() * self.view.to_homogeneous()
    }

    fn to_gpu_camera(&self) -> GPUCamera {
        GPUCamera {
            view: self.view.to_homogeneous(),
            projection: self.projection.to_homogeneous(),
            position: self.view.translation.vector,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PushConstants {
    vertex_buffer_address: vk::DeviceAddress,
    instance_buffer_address: vk::DeviceAddress,
    camera_buffer_address: vk::DeviceAddress,
}

pub struct RendererAttributes {
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    pub depth_format: vk::Format,
    pub buffering: usize,
}

impl Renderer {
    pub fn new(
        context: Arc<RenderingContext>,
        commands: &Commands,
        attributes: RendererAttributes,
    ) -> Result<Self> {
        let vertex_shader =
            load_shader_module(context.as_ref(), SHADERS_DIR.to_owned() + "shader.vert.spv")?;
        let fragment_shader =
            load_shader_module(context.as_ref(), SHADERS_DIR.to_owned() + "shader.frag.spv")?;

        let mut allocator = context.create_allocator(Default::default(), Default::default())?;

        let render_targets = (0..attributes.buffering)
            .map(|_| {
                Image::new_render_target(
                    context.clone(),
                    &mut allocator,
                    "render_target",
                    attributes.extent,
                    attributes.format,
                    1.0,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let depth_buffers = (0..attributes.buffering)
            .map(|_| {
                Image::new_depth_buffer(
                    context.clone(),
                    &mut allocator,
                    "depth_buffer",
                    attributes.extent,
                    attributes.depth_format,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let frames = render_targets
            .into_iter()
            .zip(depth_buffers)
            .map(|(render_target, depth_image)| Frame {
                render_target,
                depth_buffer: depth_image,
            })
            .collect::<Vec<_>>();

        unsafe {
            let pipeline_layout = context.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&[
                    vk::PushConstantRange::default()
                        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                        .offset(0)
                        .size(size_of::<PushConstants>() as u32),
                ]),
                None,
            )?;

            let pipeline = context.create_graphics_pipeline(
                vertex_shader,
                fragment_shader,
                attributes.extent,
                attributes.format,
                attributes.depth_format,
                pipeline_layout,
                Default::default(),
            )?;

            context.device.destroy_shader_module(vertex_shader, None);
            context.device.destroy_shader_module(fragment_shader, None);

            let gpu_geometry = Geometry::load_obj("res/viking_room.obj")?
                .create_gpu_geometry(context.clone(), &mut allocator)?;

            // generate instances in a grid
            let instances = (-2..2)
                .flat_map(|x| {
                    (-2..2).map(move |y| {
                        Instance::new(
                            na::Vector3::new(x as f32 * 2.0, 0.0, y as f32 * 2.0),
                            // rotate 90 degrees around the y axis
                            na::UnitQuaternion::from_axis_angle(
                                &na::Unit::new_normalize(na::Vector3::x()),
                                std::f32::consts::FRAC_PI_2,
                            ),
                            na::Vector3::new(1.0, 1.0, 1.0),
                        )
                    })
                })
                .collect::<Vec<_>>();

            let gpu_instances = instances
                .iter()
                .map(Instance::to_gpu_instance)
                .collect::<Vec<_>>();

            let instance_buffer = Buffer::new(
                &mut allocator,
                BufferAttributes {
                    name: "instance_buffer".into(),
                    context: context.clone(),
                    size: (instances.len() * size_of::<Instance>()) as vk::DeviceSize,
                    usage: vk::BufferUsageFlags::VERTEX_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::TRANSFER_DST,
                    location: MemoryLocation::GpuOnly,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                    allocation_priority: 1.0,
                },
            )?;

            let mut staging_belt = StagingBelt::new(
                context.clone(),
                &mut allocator,
                gpu_geometry.geometry.size() as vk::DeviceSize + instance_buffer.attributes.size,
            )?;

            staging_belt
                .stage_geometry(&gpu_geometry, commands)?
                .write(&gpu_instances)?
                .copy_to(&instance_buffer, commands)
                .done();

            let cameras = vec![Camera::new(
                &na::Point3::new(0.0, 0.0, 2.0),
                &na::Point3::new(0.0, 0.0, 0.0),
                attributes.extent.width as f32 / attributes.extent.height as f32,
                std::f32::consts::FRAC_PI_2,
                0.1,
                1000.0,
            )];

            let gpu_cameras = cameras
                .iter()
                .map(Camera::to_gpu_camera)
                .collect::<Vec<_>>();

            let mut camera_buffer = Buffer::new(
                &mut allocator,
                BufferAttributes {
                    name: "camera_buffer".into(),
                    context: context.clone(),
                    size: (cameras.len() * size_of::<GPUCamera>()) as vk::DeviceSize,
                    usage: vk::BufferUsageFlags::UNIFORM_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    location: MemoryLocation::CpuToGpu,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                    allocation_priority: 1.0,
                },
            )?;
            camera_buffer.write(&gpu_cameras, 0)?;

            let start_time = Instant::now();

            Ok(Self {
                allocator,
                pipeline,
                pipeline_layout,
                context,
                staging_belt,
                gpu_geometry,
                camera_buffer,
                cameras,
                start_time,
                frames,
                attributes,
                instance_buffer,
                instances,
            })
        }
    }

    pub fn resize(&mut self, resolution: vk::Extent2D) -> Result<()> {
        for frame in self.frames.iter_mut() {
            frame.render_target.destroy(&mut self.allocator)?;
            frame.depth_buffer.destroy(&mut self.allocator)?;
            frame.render_target = Image::new_render_target(
                self.context.clone(),
                &mut self.allocator,
                "render_target",
                resolution,
                self.attributes.format,
                1.0,
            )?;
            frame.depth_buffer = Image::new_depth_buffer(
                self.context.clone(),
                &mut self.allocator,
                "depth_buffer",
                resolution,
                self.attributes.depth_format,
            )?;
        }

        self.attributes.extent = resolution;
        self.cameras[0].projection = na::Perspective3::new(
            resolution.width as f32 / resolution.height as f32,
            std::f32::consts::FRAC_PI_2,
            0.1,
            1000.0,
        );

        Ok(())
    }

    pub fn render(
        &mut self,
        commands: &Commands,
        clear_color: vk::ClearColorValue,
        render_target_index: usize,
    ) -> Result<&mut Image> {
        let frame = &mut self.frames[render_target_index];
        let render_target = &mut frame.render_target;
        let depth_buffer = &mut frame.depth_buffer;

        render_target.reset_layout();

        let camera = &mut self.cameras[0];
        let t = (Instant::now() - self.start_time).as_secs_f32();
        camera.view = na::Isometry3::look_at_rh(
            &na::Point3::new(t.cos(), -1.0, t.sin()),
            &na::Point3::new(0.0, 0.0, 0.0),
            &na::Vector3::y(),
        );

        let gpu_cameras = self
            .cameras
            .iter()
            .map(Camera::to_gpu_camera)
            .collect::<Vec<_>>();
        self.camera_buffer.write(&gpu_cameras, 0)?;

        commands.begin_rendering(
            render_target,
            depth_buffer,
            clear_color,
            vk::Rect2D::default().extent(self.attributes.extent),
        );
        self.draw(commands, render_target_index);
        commands.end_rendering();

        Ok(&mut self.frames[render_target_index].render_target)
    }

    pub fn draw(&self, commands: &Commands, render_target_index: usize) {
        let render_target = &self.frames[render_target_index].render_target;

        commands
            .set_viewport(
                vk::Viewport::default()
                    .width(render_target.attributes.extent.width as f32)
                    .height(render_target.attributes.extent.height as f32)
                    .max_depth(1.0),
            )
            .set_scissor(
                vk::Rect2D::default().extent(
                    vk::Extent2D::default()
                        .width(render_target.attributes.extent.width)
                        .height(render_target.attributes.extent.height),
                ),
            )
            .bind_pipeline(self.pipeline)
            .bind_index_buffer(&self.gpu_geometry.index_buffer)
            .set_push_constants(
                self.pipeline_layout,
                PushConstants {
                    vertex_buffer_address: self.gpu_geometry.vertex_buffer.address,
                    instance_buffer_address: self.instance_buffer.address,
                    camera_buffer_address: self.camera_buffer.address,
                },
            )
            .draw_indexed(
                0..self.gpu_geometry.geometry.indices.len() as u32,
                0..self.instances.len() as u32,
            );
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.context.device.device_wait_idle().unwrap();

            self.instance_buffer.destroy(&mut self.allocator).unwrap();
            self.camera_buffer.destroy(&mut self.allocator).unwrap();
            self.staging_belt.destroy(&mut self.allocator).unwrap();
            self.gpu_geometry.destroy(&mut self.allocator).unwrap();
            for mut frame in self.frames.drain(..) {
                frame.render_target.destroy(&mut self.allocator).unwrap();
                frame.depth_buffer.destroy(&mut self.allocator).unwrap();
            }

            self.context.device.destroy_pipeline(self.pipeline, None);
            self.context
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
