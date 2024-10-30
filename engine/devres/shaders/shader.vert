#version 460
layout (location = 0) out vec3 fragColor;

#include "push_constants.glsl"

void main() {
    vec3 position = pushConstants.vertexBuffer.vertices[gl_VertexIndex].position;
    mat4 vp = pushConstants.cameraBuffer.cameras[0].view_projection;
    mat4 mvp = vp * pushConstants.instanceBuffer.model[gl_InstanceIndex];
    gl_Position = mvp * vec4(position, 1.0);
    fragColor = pushConstants.vertexBuffer.vertices[gl_VertexIndex].color;
}