#version 460
#include "push_constants.glsl"

layout (location = 0) out vec3 fragPosition;
layout (location = 1) out vec3 fragNormal;
layout (location = 2) out vec2 fragTexCoord;

void main() {
    Vertex vertex = pushConstants.vertexBuffer.vertices[gl_VertexIndex];
    Instance instance = pushConstants.instanceBuffer.instances[gl_InstanceIndex];
    Camera camera = pushConstants.cameraBuffer.cameras[0];

    mat4 mvp = camera.projection * camera.view * instance.model;
    gl_Position = mvp * vec4(vertex.position, 1.0);
    fragPosition = vec3(instance.model * vec4(vertex.position, 1.0));

    mat3 normalMatrix = transpose(inverse(mat3(instance.model)));
    fragNormal = normalize(normalMatrix * vertex.normal);

    fragTexCoord = vertex.texCoord;
}