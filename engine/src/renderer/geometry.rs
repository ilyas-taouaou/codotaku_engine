use crate::buffer::{Buffer, BufferAttributes};
use crate::rendering_context::RenderingContext;
use anyhow::Result;
use ash::vk;
use gpu_allocator::vulkan::{AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;
use nalgebra as na;
use std::fmt;
use std::path::Path;
use std::sync::Arc;
use tobj::GPU_LOAD_OPTIONS;

type VertexIndex = u32;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: na::Vector3<f32>,
    pub normal: na::Vector3<f32>,
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

impl Geometry {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<VertexIndex>) -> Self {
        Self { vertices, indices }
    }

    pub fn load_obj(path: impl AsRef<Path> + fmt::Debug) -> Result<Self> {
        let (models, _materials) = tobj::load_obj(path.as_ref(), &GPU_LOAD_OPTIONS)?;

        let mesh = models.into_iter().next().unwrap().mesh;

        Ok(Self {
            vertices: mesh
                .positions
                .chunks(3)
                .zip(mesh.normals.chunks(3))
                .map(|(position, normal)| Vertex {
                    position: na::Vector3::new(position[0], position[1], position[2]),
                    normal: na::Vector3::new(normal[0], normal[1], normal[2]),
                })
                .collect(),
            indices: mesh.indices,
        })
    }

    pub fn create_gpu_geometry(
        self,
        context: Arc<RenderingContext>,
        allocator: &mut Allocator,
    ) -> Result<GPUGeometry> {
        let vertex_buffer = Buffer::new(
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
                allocation_priority: 1.0,
            },
        )?;

        let index_buffer = Buffer::new(
            allocator,
            BufferAttributes {
                name: "index_buffer".into(),
                context: context.clone(),
                size: (self.indices.len() * size_of::<VertexIndex>()) as vk::DeviceSize,
                usage: vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                location: MemoryLocation::GpuOnly,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                allocation_priority: 1.0,
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
