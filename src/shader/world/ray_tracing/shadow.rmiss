#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "../util/ray_payloads.glsl"
#include "common/shared.hpp"

layout(set = 2, binding = 0) uniform WorldUniform {
    WorldUBO worldUBO;
};

layout(set = 2, binding = 2) uniform SkyUniform {
    SkyUBO skyUBO;
};

layout(set = 2, binding = 4) buffer AtmosphereLight {
    vec4 sunColor;
    vec4 moonColor;
} atmosphereLight;

layout(location = 1) rayPayloadInEXT ShadowRay shadowRay;

void main() {
    vec3 toSun = normalize(skyUBO.sunDirection);
    vec3 radiance;

    if (toSun.y > 0) {
        radiance = atmosphereLight.sunColor.xyz;
    } else {
        radiance = atmosphereLight.moonColor.xyz;
    }

    float factor = 1.0;
    float threshold = 0.3;
    if (abs(toSun.y) < threshold) { factor = sin(PI / (2 * threshold) * abs(toSun.y)); }
    radiance *= factor;

    shadowRay.radiance += radiance * shadowRay.throughput;
}
