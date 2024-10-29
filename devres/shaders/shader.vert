#version 460
#extension GL_EXT_buffer_reference: require
#extension GL_EXT_scalar_block_layout: require

layout (location = 0) out vec3 fragColor;

struct Vertex {
    vec3 position;
    vec3 color;
};

layout (buffer_reference, scalar) buffer VertexBuffer {
    Vertex vertices[];
};

layout (scalar, push_constant) uniform Registers
{
    VertexBuffer vertexBuffer;
} pushConstants;

void main() {
    gl_Position = vec4(pushConstants.vertexBuffer.vertices[gl_VertexIndex].position, 1.0);
    fragColor = pushConstants.vertexBuffer.vertices[gl_VertexIndex].color;
}