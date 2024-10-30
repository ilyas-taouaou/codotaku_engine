#version 460

layout (location = 0) in vec3 fragColor;

layout (location = 0) out vec4 outColor;

#include "push_constants.glsl"

void main() {
    outColor = vec4(fragColor, 1.0);
}