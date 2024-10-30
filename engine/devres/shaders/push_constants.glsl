#extension GL_EXT_buffer_reference: require
#extension GL_EXT_scalar_block_layout: require

struct Vertex {
    vec3 position;
    vec3 normal;
};

struct Camera {
    mat4 view;
    mat4 projection;
};

struct Instance {
    mat4 model;
};

layout (buffer_reference, scalar) buffer VertexBuffer {
    Vertex vertices[];
};

layout (buffer_reference, scalar) buffer CameraBuffer {
    Camera cameras[];
};

layout (buffer_reference, scalar) buffer InstanceBuffer {
    Instance instances[];
};

layout (scalar, push_constant) uniform Registers
{
    VertexBuffer vertexBuffer;
    InstanceBuffer instanceBuffer;
    CameraBuffer cameraBuffer;
} pushConstants;