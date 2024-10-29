use crate::buffer::{Buffer, BufferAttributes};
use crate::rendering_context::RenderingContext;
use anyhow::Result;
use ash::vk;
use gpu_allocator::vulkan::{AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;
use nalgebra as na;
use std::sync::Arc;

type VertexIndex = u32;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: na::Vector3<f32>,
    color: na::Vector3<f32>,
}

pub struct Geometry {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<VertexIndex>,
}

pub struct GPUGeometry {
    pub geometry: Geometry,
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
}

impl GPUGeometry {
    pub fn destroy(&mut self, allocator: &mut Allocator) -> Result<()> {
        self.index_buffer.destroy(allocator)?;
        self.vertex_buffer.destroy(allocator)?;
        Ok(())
    }
}

pub struct Circle {
    pub radius: f32,
    pub segments: usize,
}

impl Geometry {
    pub fn new_circle(circle: Circle) -> Self {
        let Circle { radius, segments } = circle;
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

        Self { vertices, indices }
    }

    pub fn create_gpu_geometry(
        self,
        context: Arc<RenderingContext>,
        allocator: &mut Allocator,
    ) -> Result<GPUGeometry> {
        let mut vertex_buffer = Buffer::new(
            allocator,
            BufferAttributes {
                name: "vertex_buffer".into(),
                context: context.clone(),
                size: (self.vertices.len() * size_of::<Vertex>()) as vk::DeviceSize,
                usage: vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::TRANSFER_DST,
                location: MemoryLocation::GpuOnly,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            },
        )?;

        let mut index_buffer = Buffer::new(
            allocator,
            BufferAttributes {
                name: "index_buffer".into(),
                context: context.clone(),
                size: (self.indices.len() * size_of::<VertexIndex>()) as vk::DeviceSize,
                usage: vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                location: MemoryLocation::GpuOnly,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            },
        )?;

        Ok(GPUGeometry {
            geometry: self,
            vertex_buffer,
            index_buffer,
        })
    }

    pub fn size(&self) -> usize {
        self.vertices.len() * size_of::<Vertex>() + self.indices.len() * size_of::<VertexIndex>()
    }

    pub fn vertices_size(&self) -> usize {
        self.vertices.len() * size_of::<Vertex>()
    }
}
