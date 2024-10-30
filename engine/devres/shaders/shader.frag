#version 460
#include "push_constants.glsl"

layout (location = 0) in vec3 fragPosition;
layout (location = 1) in vec3 fragNormal;

layout (location = 0) out vec4 outColor;

// constant sun direction
const vec3 sunDir = normalize(vec3(0.5, 0.5, 0.5));

void main() {
    Camera camera = pushConstants.cameraBuffer.cameras[0];
    vec3 cameraPosition = vec3(inverse(camera.view)[3]);

    float ambient = 0.1;
    float diffuse = max(dot(fragNormal, sunDir), 0.0);
    float specularStrength = 0.5;

    vec3 viewDir = normalize(cameraPosition - fragPosition);
    vec3 reflectDir = reflect(-sunDir, fragNormal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);

    outColor = vec4(vec3(ambient + diffuse + spec * specularStrength), 1.0);
}