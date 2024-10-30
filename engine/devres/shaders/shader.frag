#version 460
#include "push_constants.glsl"

layout (location = 0) in vec3 fragColor;

layout (location = 0) out vec4 outColor;

void main() {
    outColor = vec4(fragColor, 1.0);
}