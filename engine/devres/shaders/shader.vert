#version 460
#extension GL_EXT_buffer_reference: require
#extension GL_EXT_scalar_block_layout: require

layout (location = 0) out vec3 fragColor;

struct Vertex {
    vec3 position;
    vec3 color;
};

struct Camera {
    mat4 view_projection;
};

layout (buffer_reference, scalar) buffer VertexBuffer {
    Vertex vertices[];
};

layout (buffer_reference, scalar) buffer CameraBuffer {
    Camera cameras[];
};

layout (buffer_reference, scalar) buffer InstanceBuffer {
    mat4 model[];
};

layout (scalar, push_constant) uniform Registers
{
    VertexBuffer vertexBuffer;
    InstanceBuffer instanceBuffer;
    CameraBuffer cameraBuffer;
} pushConstants;

void main() {
    vec3 position = pushConstants.vertexBuffer.vertices[gl_VertexIndex].position;
    mat4 vp = pushConstants.cameraBuffer.cameras[0].view_projection;
    mat4 mvp = vp * pushConstants.instanceBuffer.model[gl_InstanceIndex];
    gl_Position = mvp * vec4(position, 1.0);
    fragColor = pushConstants.vertexBuffer.vertices[gl_VertexIndex].color;
}