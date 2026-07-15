#ifndef ATMOSPHERE_COMMON_GLSL
#define ATMOSPHERE_COMMON_GLSL

vec2 transmittanceUv(float r, float mu, SkyUBO ubo) {
    float u = clamp(mu * 0.5 + 0.5, 0.0, 1.0);
    float v = clamp((r - ubo.Rg) / (ubo.Rt - ubo.Rg), 0.0, 1.0);
    return vec2(u, v);
}

vec3 sampleTransmittance(sampler2D transLUT, float r, float mu, SkyUBO ubo) {
    vec2 uv = transmittanceUv(r, mu, ubo);
    vec2 invSize = 1.0 / vec2(textureSize(transLUT, 0));
    uv = clamp(uv, 0.5 * invSize, 1.0 - 0.5 * invSize);
    return texture(transLUT, uv).rgb;
}

#endif
