#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "../util/ray_payloads.glsl"

layout(location = 1) rayPayloadInEXT ShadowRay shadowRay;

void main() {
    shadowRay.hitT = gl_HitTEXT;
    shadowRay.instanceIndex = gl_InstanceCustomIndexEXT;
    shadowRay.geometryIndex = gl_GeometryIndexEXT;
    shadowRay.primitiveIndex = gl_PrimitiveID;
}
