#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_query : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_opacity_micromap : require

#include "../util/disney.glsl"
#include "../util/random.glsl"
#include "../util/ray_cone.glsl"
#include "../util/ray_payloads.glsl"
#include "../util/util.glsl"
#include "common/shared.hpp"

layout(set = 0, binding = 0) uniform sampler2D textures[];

#include "../util/clouds.glsl"

layout(set = 1, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(set = 1, binding = 1) readonly buffer BLASOffsets {
    uint offsets[];
}
blasOffsets;

layout(set = 1, binding = 2) readonly buffer VertexBufferAddr {
    uint64_t addrs[];
}
vertexBufferAddrs;

layout(set = 1, binding = 3) readonly buffer IndexBufferAddr {
    uint64_t addrs[];
}
indexBufferAddrs;

layout(set = 1, binding = 4) readonly buffer LastVertexBufferAddr {
    uint64_t addrs[];
}
lastVertexBufferAddrs;

layout(set = 1, binding = 5) readonly buffer LastIndexBufferAddr {
    uint64_t addrs[];
}
lastIndexBufferAddrs;

layout(set = 1, binding = 6) readonly buffer LastObjToWorldMat {
    mat4 mat[];
}
lastObjToWorldMats;

layout(set = 1, binding = 7) readonly buffer TextureMappingBuffer {
    TextureMapping mapping;
};

layout(set = 2, binding = 0) uniform WorldUniform {
    WorldUBO worldUbo;
};

layout(set = 2, binding = 1) uniform LastWorldUniform {
    WorldUBO lastWorldUbo;
};

layout(set = 2, binding = 2) uniform SkyUniform {
    SkyUBO skyUBO;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertexBuffer {
    PBRTriangle vertices[];
}
vertexBuffer;

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer IndexBuffer {
    uint indices[];
}
indexBuffer;

layout(push_constant) uniform PushConstant {
    int numRayBounces;
    int flags;
} pc;
#define SIMPLIFIED_INDIRECT ((pc.flags & 1) != 0)

layout(location = 0) rayPayloadInEXT PrimaryRay mainRay;
layout(location = 1) rayPayloadEXT ShadowRay shadowRay;
hitAttributeEXT vec2 attribs;

vec2 wavedx(vec2 position, vec2 direction, float frequency, float timeshift)
{
    float x = dot(direction, position) * frequency + timeshift;
    float wave = exp(sin(x) - 1.0);
    float dx = wave * cos(x);
    return vec2(wave, -dx);
}


float getwaves(vec2 position)
{
    float wavePhaseShift = length(position) * 0.1;

    float iter = 0.0;
    float frequency = 1.5;
    float weight = 0.25;
    float timeMultiplier = 1.0;
    float sumOfValues = 0.0;
    float sumOfWeights = 0.0;

    for(int i = 0; i < 20; i++)
    {
        vec2 p = vec2(sin(iter), cos(iter));

        vec2 res = wavedx(
            position,
            p,
            frequency,
            worldUbo.gameTime * 3000.0 * timeMultiplier + wavePhaseShift
        );

        position += p * res.y * weight * 0.38;

        sumOfValues += res.x * weight;
        sumOfWeights += weight;

        weight = mix(weight, 0.0, 0.12);
        frequency *= 1.17;
        timeMultiplier *= 1.05;

        iter += 1261152.3999531981663;
    }

    return sumOfValues / sumOfWeights;
}


vec3 calculateNormalWater(vec2 uv)
{
    float e = 0.01;

    float H  = getwaves(uv) * 0.15;
    float Hx = getwaves(uv - vec2(e,0)) * 0.15;
    float Hz = getwaves(uv + vec2(0,e)) * 0.15;

    vec3 p  = vec3(uv.x, H,  uv.y);
    vec3 px = vec3(uv.x - e, Hx, uv.y);
    vec3 pz = vec3(uv.x, Hz, uv.y + e);

    return normalize(cross(p - px, p - pz));
}

vec3 calculateNormal(vec3 p0, vec3 p1, vec3 p2, vec2 uv0, vec2 uv1, vec2 uv2, vec3 matNormal, vec3 viewDir) {
    vec3 edge1 = p1 - p0;
    vec3 edge2 = p2 - p0;
    vec3 geoNormalObj = normalize(cross(edge1, edge2));

    mat3 normalMatrix = transpose(mat3(gl_WorldToObject3x4EXT));
    vec3 geometricNormalWorld = normalize(normalMatrix * geoNormalObj);

    if (any(isnan(matNormal))) { return geometricNormalWorld; }

    // TBN
    vec2 deltaUV1 = uv1 - uv0;
    vec2 deltaUV2 = uv2 - uv0;
    float det = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;

    vec3 tangentObj;
    if (abs(det) < 1e-6) {
        tangentObj = (abs(geoNormalObj.x) > 0.99) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    } else {
        float f = 1.0 / det;
        tangentObj.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
        tangentObj.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
        tangentObj.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
    }

    // Gram-Schmidt
    vec3 TObj = normalize(tangentObj - geoNormalObj * dot(geoNormalObj, tangentObj));
    vec3 BObj = cross(geoNormalObj, TObj);

    vec3 T = normalize(normalMatrix * TObj);
    vec3 B = normalize(normalMatrix * BObj);
    vec3 N = geometricNormalWorld;

    // LabPBR / DirectX (Y-)
    vec3 correctedLocalNormal = matNormal;
    correctedLocalNormal.y = -correctedLocalNormal.y;

    vec3 finalNormal = normalize(T * correctedLocalNormal.x + B * correctedLocalNormal.y + N * correctedLocalNormal.z);

    // unseenable faces
    if (dot(viewDir, finalNormal) < 0.0)
        return geometricNormalWorld;
    else
        return finalNormal;
}

bool compareVec(vec3 a, vec3 b) {
    const float e = 0.001;
    return !any(greaterThan(abs(a - b), vec3(e)));
}

bool compareVec(vec3 a, vec3 b, float e) {
    return !any(greaterThan(abs(a - b), vec3(e)));
}

bool compareVec(vec4 a, vec4 b) {
    const float e = 0.001;
    return !any(greaterThan(abs(a - b), vec4(e)));
}

void main() {
    vec3 viewDir = -mainRay.direction;

    uint instanceID = gl_InstanceCustomIndexEXT;
    uint geometryID = gl_GeometryIndexEXT;

    uint blasOffset = blasOffsets.offsets[instanceID];

    IndexBuffer indexBuffer = IndexBuffer(indexBufferAddrs.addrs[blasOffset + geometryID]);
    uint indexBaseID = 3 * gl_PrimitiveID;
    uint i0 = indexBuffer.indices[indexBaseID];
    uint i1 = indexBuffer.indices[indexBaseID + 1];
    uint i2 = indexBuffer.indices[indexBaseID + 2];

    VertexBuffer vertexBuffer = VertexBuffer(vertexBufferAddrs.addrs[blasOffset + geometryID]);
    PBRTriangle v0 = vertexBuffer.vertices[i0];
    PBRTriangle v1 = vertexBuffer.vertices[i1];
    PBRTriangle v2 = vertexBuffer.vertices[i2];

    vec3 baryCoords = vec3(1.0 - (attribs.x + attribs.y), attribs.x, attribs.y);
    vec3 localPos = baryCoords.x * v0.pos + baryCoords.y * v1.pos + baryCoords.z * v2.pos;
    vec3 worldPos = vec4(localPos, 1.0) * gl_ObjectToWorld3x4EXT;

    uint useColorLayer = v0.useColorLayer;
    vec3 colorLayer;
    if (useColorLayer > 0) {
        colorLayer = (baryCoords.x * v0.colorLayer + baryCoords.y * v1.colorLayer + baryCoords.z * v2.colorLayer).rgb;
    } else {
        colorLayer = vec3(1.0);
    }

    uint useTexture = v0.useTexture;
    float albedoEmission =
        baryCoords.x * v0.albedoEmission + baryCoords.y * v1.albedoEmission + baryCoords.z * v2.albedoEmission;
    uint textureID = v0.textureID;
    int specularTextureID = mapping.entries[textureID].specular;
    int normalTextureID = mapping.entries[textureID].normal;
    int flagTextureID = mapping.entries[textureID].flag;
    vec4 albedoValue;
    vec4 specularValue;
    vec4 normalValue;
    ivec4 flagValue;
    vec2 textureUV;
    if (useTexture > 0) {
        textureUV = baryCoords.x * v0.textureUV + baryCoords.y * v1.textureUV + baryCoords.z * v2.textureUV;

        // ray cone
        float coneRadiusWorld = mainRay.coneWidth + gl_HitTEXT * mainRay.coneSpread;
        vec3 dposdu, dposdv;
        computedposduDv(v0.pos, v1.pos, v2.pos, v0.textureUV, v1.textureUV, v2.textureUV, dposdu, dposdv);
        // lod still has issues, temporally disable
        float lod = 0; // lodWithCone(textures[nonuniformEXT(textureID)], textureUV, coneRadiusWorld, dposdu, dposdv);

        albedoValue = textureLod(textures[nonuniformEXT(textureID)], textureUV, lod);
        if (specularTextureID >= 0) {
            specularValue = textureLod(textures[nonuniformEXT(specularTextureID)], textureUV, lod);
        } else {
            specularValue = vec4(0.0);
        }
        if (normalTextureID >= 0) {
            normalValue = textureLod(textures[nonuniformEXT(normalTextureID)], textureUV, lod);
        } else {
            normalValue = vec4(0.0);
        }
        if (flagTextureID >= 0) {
            vec4 floatFlagValue = textureLod(textures[nonuniformEXT(flagTextureID)], textureUV, ceil(lod));
            flagValue = ivec4(round(floatFlagValue * 255.0));
        } else {
            flagValue = ivec4(0);
        }
    } else {
        albedoValue = vec4(1.0);
        specularValue = vec4(0.0);
        normalValue = vec4(0.0);
        flagValue = ivec4(0);
    }

    vec3 glint = vec3(0.0);
    vec4 overlayColor = vec4(0.0);
    if (!SIMPLIFIED_INDIRECT || mainRay.index <= 1) {
        uint useGlint = v0.useGlint;
        uint glintTexture = v0.glintTexture;
        vec2 glintUV = baryCoords.x * v0.glintUV + baryCoords.y * v1.glintUV + baryCoords.z * v2.glintUV;
        glintUV = (worldUbo.textureMat * vec4(glintUV, 0.0, 1.0)).xy;
        glint = useGlint * texture(textures[nonuniformEXT(glintTexture)], glintUV).rgb;
        glint = glint * glint;

        uint useOverlay = v0.useOverlay;
        ivec2 overlayUV = v0.overlayUV;
        overlayColor = texelFetch(textures[nonuniformEXT(worldUbo.overlayTextureID)], overlayUV, 0);
    }

    vec3 tint;
    if (v0.useOverlay > 0) {
        tint = mix(overlayColor.rgb, albedoValue.rgb * colorLayer, overlayColor.a) + glint;
    } else {
        tint = albedoValue.rgb * colorLayer + glint;
    }

    albedoValue = vec4(tint, albedoValue.a);


    LabPBRMat mat = convertLabPBRMaterial(albedoValue, specularValue, normalValue);
    // the provided normal is unreliable! (such as grass, etc.)
    // calculate on the fly for now
    vec3 normal =
        calculateNormal(v0.pos, v1.pos, v2.pos, v0.textureUV, v1.textureUV, v2.textureUV, mat.normal, viewDir);

    bool isWater = false;
    if (flagValue.g == 255) {
        isWater = true;
        albedoValue = vec4(0.8, 0.85, 1.0, 0.5);
        mat.albedo = albedoValue.rgb;
        mat.f0 = vec3(0.1); 
        mat.roughness = 0.001;
        mat.metallic = 0.0;
        mat.transmission = 1.0;
        mat.ior = 1.33;
        mat.emission = 0.0;
        if (dot(normal, vec3(0.0, 1.0, 0.0)) > 0.5)
            normal = calculateNormalWater((worldPos.xz + vec2(worldUbo.cameraPos.xz)));
    }


    // --- 添加湿润效果：仅在下雨时，且当前点露天（无上方遮挡）时应用 ---
    if (skyUBO.rainGradient > 0.0) {
        vec3 upDir = vec3(0.0, 1.0, 0.0);                       // 世界向上方向（假设Y轴向上）
        vec3 rayOrigin = worldPos + normal * 0.001;              // 偏移避免自遮挡

        // 保存当前的 shadowRay，以便发射后恢复
        ShadowRay savedShadowRay = shadowRay;
        // 初始化 hitT 为无穷大（表示未命中）
        shadowRay.hitT = INF_DISTANCE;

        traceRayEXT(topLevelAS,
                    gl_RayFlagsTerminateOnFirstHitEXT,           // 找到第一个命中即停止
                    WORLD_MASK,                                   // 只检测世界方块
                    0,                                            // sbtRecordOffset
                    0,                                            // sbtRecordStride
                    2,                                            // missIndex（复用阴影的 miss 索引）
                    rayOrigin,
                    0.001,                                        // tMin
                    upDir,
                    1000.0,                                       // tMax（足够高）
                    1                                             // payload location（与阴影射线一致）
                    );

        // 如果 hitT 仍是无穷大，说明未命中任何物体 → 露天
        bool isExposed = (shadowRay.hitT >= INF_DISTANCE);

        // 恢复原来的 shadowRay
        shadowRay = savedShadowRay;

        if (isExposed) {
            if(!isWater){
                float wetness = skyUBO.rainGradient;
                float minRoughness = 0.005;
                float maxRoughness = mat.roughness;
                mat.roughness = mix(minRoughness, maxRoughness, 1.0 - wetness); // 越湿润越光滑
                mat.albedo.rgb *= (1.0 - 0.3 * wetness); // 雨水会略微降低反射率，增加暗沉感
                if (mat.metallic == 0.0) {
                    vec3 targetF0 = vec3(0.2);             // 比默认的 0.04 明显增强
                    float factor = wetness * 0.8;            // 控制插值强度，最大影响80%
                    mat.f0 = mix(mat.f0, targetF0, factor);
                }

                // 4. 减弱法线贴图影响：根据湿润强度向几何法线混合
                // 重新计算几何法线（未受法线贴图影响的真实表面法线）
                vec3 edge1 = v1.pos - v0.pos;
                vec3 edge2 = v2.pos - v0.pos;
                vec3 geoNormalObj = normalize(cross(edge1, edge2));
                mat3 normalMatrix = transpose(mat3(gl_WorldToObject3x4EXT));
                vec3 geometricNormalWorld = normalize(normalMatrix * geoNormalObj);
                // 混合：几何法线占主导的程度随 wetness 增加
                float normalStrength = 1.0 - wetness * 0.8; // 0.8 是最大减弱系数，可调
                normal = normalize(mix(geometricNormalWorld, normal, normalStrength));
            }


        }
}



    // add glowing radiance
    float factor = mainRay.index == 0 ? 1.0 : 16.0 * skyUBO.hdrRadianceScale;
    vec3 emissionRadiance = factor * tint * mat.emission * mainRay.throughput;
    emissionRadiance += tint * albedoEmission * mainRay.throughput;
    mainRay.radiance += emissionRadiance;

    mainRay.hitT = gl_HitTEXT;
    mainRay.coneWidth += gl_HitTEXT * mainRay.coneSpread;

    // shadow ray for direct lighting
    vec3 sunDir = normalize(skyUBO.sunDirection);
    vec3 lightDir = sunDir;
    // Softer sun sampling for hand = wider penumbras (500 vs 3000)
    float kappa = (mainRay.isHand > 0) ? 500.0 : 3000.0;
    if (sunDir.y < 0) { lightDir = normalize(skyUBO.moonDirection); }
    vec3 sampledLightDir = SampleVMF(mainRay.seed, lightDir, kappa);
    vec3 shadowRayOrigin = worldPos + (dot(sampledLightDir, normal) > 0.0 ? normal : -normal) * 0.001;

    if (worldUbo.skyType == 1) {
        float pdf; // not used
        vec3 lightBRDF = DisneyEval(mat, viewDir, normal, sampledLightDir, pdf);

        shadowRay.radiance = vec3(0.0);
        shadowRay.throughput = vec3(1.0);
        shadowRay.seed = mainRay.seed;
        shadowRay.hitT = INF_DISTANCE;
        shadowRay.insideBoat = mainRay.insideBoat;
        shadowRay.bounceIndex = mainRay.index;

        uint shadowMask = WORLD_MASK;
        if (mainRay.isHand == 0) {
            shadowMask |= PLAYER_MASK;  // world surfaces see player shadows; hand does not (prevents self-shadowing)
        }

        traceRayEXT(topLevelAS, gl_RayFlagsNoneEXT,
                    shadowMask,
                    0,                                     // sbtRecordOffset
                    0,                                     // sbtRecordStride
                    2,                                     // missIndex
                    shadowRayOrigin, 0.001, sampledLightDir, 1000, 1);

        // Add direct lighting contribution
        vec3 lightContribution = shadowRay.radiance;

        // Apply cloud shadowing (procedural volumetric slab).
        // This is evaluated at the shading point so it works for primary and reflected paths.
        float cloudT = cloudTransmittance(worldPos + sampledLightDir * 0.01, sampledLightDir, 0.0, 1000.0, worldUbo, skyUBO, 0);
        float shadowStrength = max(skyUBO.cloudLighting.x, 0.0);
        cloudT = pow(max(cloudT, 1e-6), shadowStrength);
        lightContribution *= cloudT;

        float progress = skyUBO.rainGradient;
        vec3 lightRadiance = lightContribution * mainRay.throughput * lightBRDF;
        vec3 finalLightRadiance = mix(lightRadiance, vec3(0.0), progress);

        // Hand shadow smoothing: apply ambient floor with smooth falloff
        // Prevents harsh black transitions on hand geometry in shadow
        if (mainRay.isHand > 0) {
            float shadowLum = dot(finalLightRadiance, vec3(0.2126, 0.7152, 0.0722));
            float ambientFloor = 0.08 * dot(mainRay.throughput, vec3(0.2126, 0.7152, 0.0722));
            float blend = smoothstep(0.0, ambientFloor * 2.0, shadowLum);
            vec3 handAmbient = ambientFloor * tint * mainRay.throughput;
            finalLightRadiance = mix(handAmbient, finalLightRadiance, blend);
        }

        mainRay.radiance += finalLightRadiance;
        mainRay.directLightRadiance = finalLightRadiance;
        mainRay.directLightHitT = shadowRay.hitT;

        if (isWater) {
            // energy is not conserved
            float waterLobe = GTR(dot(normal, normalize(viewDir + lightDir)), 0.0001 * 0.0001, 1.2) * 0.1; // tweak
            vec3 specular = lightContribution * mainRay.throughput * waterLobe;
            vec3 finalSpecular = mix(specular, vec3(0.0), progress);

            mainRay.radiance += finalSpecular;
            mainRay.directLightRadiance += finalSpecular;
        }

    }

    mainRay.instanceIndex = instanceID;
    mainRay.geometryIndex = geometryID;
    mainRay.primitiveIndex = gl_PrimitiveID;
    mainRay.baryCoords = baryCoords;
    mainRay.worldPos = worldPos;
    mainRay.normal = normal;
    mainRay.albedoValue = albedoValue;
    mainRay.specularValue = specularValue;
    mainRay.normalValue = normalValue;
    mainRay.flagValue = flagValue;

    // sample next direction using Disney BSDF
    vec3 sampleDir;
    float pdf;
    uint lobeType;
    vec3 bsdf;
    if(isWater){
        bsdf = DisneySampleSmooth(mat, viewDir, normal, sampleDir, pdf, mainRay.seed, lobeType);
    } 
    else {
        bsdf = DisneySample(mat, viewDir, normal, sampleDir, pdf, mainRay.seed, lobeType);
    }

    mainRay.throughput *= bsdf / max(pdf, 1e-4);
    mainRay.lobeType = lobeType;
    mainRay.noisy = 1;

    // early exit if sampling failed
    if (pdf <= 1e-6) {
        mainRay.stop = 1;
        return;
    }

    vec3 offsetDir = dot(sampleDir, normal) > 0.0 ? normal : -normal;
    mainRay.origin = worldPos + offsetDir * 0.001;

    mainRay.direction = sampleDir;
    mainRay.stop = 0;
}
