#version 460
#include "push_constants.glsl"

layout (location = 0) in vec3 fragPosition;
layout (location = 1) in vec3 fragNormal;
layout (location = 2) in vec2 fragTexCoord;

layout (location = 0) out vec4 outColor;

layout (set = 0, binding = 0) uniform sampler2D textures[];

const vec3 sunDirection = normalize(vec3(0.5, -1.0, 0.5));
const float specularStrength = 0.5;
const float ambient = 0.1;

void main() {
    Camera camera = pushConstants.cameraBuffer.cameras[0];
    vec3 cameraPosition = camera.position;

    // get texture color
    vec4 texColor = texture(textures[0], fragTexCoord);

    float diffuse = max(dot(fragNormal, sunDirection), 0.0);

    vec3 viewDirection = normalize(cameraPosition - fragPosition);
    vec3 reflectDirection = reflect(-sunDirection, fragNormal);
    float specular = pow(max(dot(viewDirection, reflectDirection), 0.0), 32);

    outColor = vec4(texColor.rgb * (diffuse + ambient) + specularStrength * specular, texColor.a);
}