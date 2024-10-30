#version 460
#include "push_constants.glsl"

layout (location = 0) out vec3 fragColor;

void main() {
    Vertex vertex = pushConstants.vertexBuffer.vertices[gl_VertexIndex];
    Instance instance = pushConstants.instanceBuffer.instances[gl_InstanceIndex];
    Camera camera = pushConstants.cameraBuffer.cameras[0];

    vec3 position = vertex.position;
    mat4 mvp = camera.projection * camera.view * instance.model;
    gl_Position = mvp * vec4(position, 1.0);
    fragColor = vertex.color;
}