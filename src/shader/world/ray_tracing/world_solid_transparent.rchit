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

layout(set = 1, binding = 8) uniform accelerationStructureEXT waterTopLevelAS;

layout(set = 1, binding = 14) readonly buffer WaterChunkOriginBuffer {
    vec4 origins[];
} waterChunkOrigins;

layout(set = 1, binding = 15) readonly buffer WaterChunkSizeBuffer {
    ivec4 sizes[];
} waterChunkSizes;

layout(set = 1, binding = 16) readonly buffer WaterOccupancyOffsetBuffer {
    uint offsets[];
} waterOccupancyOffsets;

layout(set = 1, binding = 17) readonly buffer SolidOccupancyOffsetBuffer {
    uint offsets[];
} solidOccupancyOffsets;

layout(set = 1, binding = 18) readonly buffer WaterOccupancyDataBuffer {
    uint values[];
} waterOccupancyData;

layout(set = 1, binding = 19) readonly buffer SolidOccupancyDataBuffer {
    uint values[];
} solidOccupancyData;

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
#define USE_BILATERAL_RAIN 1
#define WATER_WAVE_HEIGHT 0.5f
#define WATER_PARALLAX 0

layout(location = 0) rayPayloadInEXT PrimaryRay mainRay;
layout(location = 1) rayPayloadEXT ShadowRay shadowRay;
hitAttributeEXT vec2 attribs;
float frameTime;

float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

float smoothNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f*f*(3.0-2.0*f);
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float AlmostIdentity(float x, float m, float n) {
    if (x > m) return x;
    float a = 2.0 * n - m;
    float b = 2.0 * m - 3.0 * n;
    float t = x / m;
    return (a * t + b) * t * t + n;
}

vec3 linearToSrgb(vec3 c) {
    vec3 lo = c * 12.92;
    vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, step(vec3(0.0031308), c));
}

vec3 srgbToLinear(vec3 c) {
    vec3 lo = c / 12.92;
    vec3 hi = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(lo, hi, step(vec3(0.04045), c));
}

vec3 GetRainAnimationTex(sampler2D tex, vec2 uv, float wet) {
    float frame = mod(floor(frameTime * 60.0), 60.0);
    // 修正 UV 计算，确保在 [0,1]
    float v = uv.y / 60.0 - frame / 60.0;
    v = v - floor(v);  // 等价于 fract，处理负值
    vec2 coord = vec2(fract(uv.x), v);

    // 采样法线贴图（纹理被错误加载为 sRGB 格式）
    vec3 texValue = textureLod(tex, coord, 0.0).rgb;
    
    // 修复：如果纹理以 sRGB 格式加载，GPU 已自动进行 sRGB->linear 转换
    // 但法线数据原本是线性值，需要反转这个转换
    texValue = linearToSrgb(texValue);  // 新增这一行

    // 法线贴图通常存储为 (0.5,0.5,1.0) 表示 (0,0,1)，需要转换到 [-1,1]
    vec3 n = texValue * 2.0 - 1.0;
    n.y *= -1.0;
    return n;
}

vec3 BilateralRainTex(sampler2D tex, vec2 uv, float wet) {
    vec3 n  = GetRainAnimationTex(tex, uv, wet);
    vec3 nR = GetRainAnimationTex(tex, uv + vec2(1.0, 0.0) / 128.0, wet);
    vec3 nU = GetRainAnimationTex(tex, uv + vec2(0.0, 1.0) / 128.0, wet);
    vec3 nUR= GetRainAnimationTex(tex, uv + vec2(1.0, 1.0) / 128.0, wet);
    vec2 fractCoord = fract(uv * 128.0);
    vec3 lerpX  = mix(n, nR, fractCoord.x);
    vec3 lerpX2 = mix(nU, nUR, fractCoord.x);
    vec3 lerpY  = mix(lerpX, lerpX2, fractCoord.y);
    return lerpY;
}

vec2 getRainDropOffset(vec3 worldPos, float wetness, vec3 worldNormal) {

    frameTime = worldUbo.gameTime * 3000.0; 

    // 构建基础坐标
    vec3 pos = worldPos * 0.5;
    vec2 uv = pos.xz;

    vec3 n1 = vec3(0.0);
    #ifdef RAIN_SPLASH_BILATERAL
        n1 = BilateralRainTex(textures[nonuniformEXT(worldUbo.rippleTextureID)], uv, wetness);
    #else
        n1 = GetRainAnimationTex(textures[nonuniformEXT(worldUbo.rippleTextureID)], uv, wetness);
    #endif

    pos.x -= frameTime * 1.5;
    float downfall = smoothNoise(pos.xz * 0.0025);
    downfall = clamp(downfall * 1.5 - 0.25, 0.0, 1.0);

    vec3 n = n1 * 0.4; // n1 已经是 [-1,1] 范围，缩放为 [-0.4,0.4]
    n.xy *= wetness; // 湿润度越高，偏移越大（wetness 已包含 rainGradient）

    float verticalFactor = clamp(worldNormal.y, 0.0, 1.0);
    n = mix(vec3(0.0, 0.0, 1.0), n, verticalFactor);

    return n.xy;
}


vec2 getWaterOffset(vec3 worldPos, float wetness, vec3 surfaceNormal) {
    // --- 可调参数 ---
    float amplitude = 0.2;                // 基础扰动幅度
    float frequencyScale = 50.0;           // 新增：波浪频率控制（默认1.0）
    float heightScale = 1.5;              // 新增：波浪高度控制（默认1.0）
    // ----------------

    // 使用与 GetWaves 一致的时间基准
    float frameTime = worldUbo.gameTime * 3000.0 * 30; // 与第151行的 frameTime 保持一致
    
    // 构建与 GetWaves 一致的坐标变换，应用频率缩放
    vec2 p = (worldPos.xz + vec2(worldUbo.cameraPos.xz)) / (5.0f / frequencyScale);  // 修改：除以频率缩放因子
    //p.xy -= worldPos.y / 10.0f;
    p.x = -p.x;
    
    // 应用时间动画（与 GetWaves 第1层一致的速度）
    p.x += (frameTime / 40.0f) * 0.9f; // speed = 0.9f, isIce = 0.0f
    p.y -= (frameTime / 40.0f) * 0.9f;
    
    // 保存原始 p 用于后续层计算
    vec2 p_layer = p;
    
    // 初始化权重和偏移
    float weightSum = 0.0f;
    vec2 offset = vec2(0.0f);
    
    // === 移植 GetWaves 的6层噪声计算，但输出向量而非标量 ===
    
    // 第1层：vec2(2.0f, 1.2f) + vec2(0.0f, p.x * 2.1f)
    {
        vec2 uv = p_layer * vec2(2.0f, 1.2f) + vec2(0.0f, p_layer.x * 2.1f);
        float noise = smoothNoise(uv);
        float weight = 1.0f * 0.5f; // 原始权重1.0 * 0.5（第1层特殊处理）
        // 方向：基于 p.x * 2.1f，可以理解为主要影响Y方向
        vec2 dir = normalize(vec2(0.0, 1.0));
        offset += dir * noise * weight;
        weightSum += weight;
        
        // 更新 p_layer 用于下一层（与 GetWaves 保持一致）
        p_layer /= 2.1f;
        p_layer.y -= (frameTime / 20.0f) * 0.9f;
        p_layer.x -= (frameTime / 30.0f) * 0.9f;
    }
    
    // 第2层：vec2(2.0f, 1.4f) + vec2(0.0f, -p.x * 2.1f)
    {
        vec2 uv = p_layer * vec2(2.0f, 1.4f) + vec2(0.0f, -p_layer.x * 2.1f);
        float noise = smoothNoise(uv);
        float weight = 2.1f;
        // 方向：反向Y方向
        vec2 dir = normalize(vec2(0.0, -1.0));
        offset += dir * noise * weight;
        weightSum += weight;
        
        p_layer /= 1.5f;
        p_layer.x += (frameTime / 20.0f) * 0.9f;
    }
    
    // 第3层：vec2(1.0f, 0.75f) + vec2(0.0f, p.x * 1.1f)
    {
        vec2 uv = p_layer * vec2(1.0f, 0.75f) + vec2(0.0f, p_layer.x * 1.1f);
        float noise = smoothNoise(uv);
        float weight = 17.25f;
        vec2 dir = normalize(vec2(0.0, 1.0));
        offset += dir * noise * weight;
        weightSum += weight;
        
        p_layer /= 1.5f;
        p_layer.x -= (frameTime / 55.0f) * 0.9f;
    }
    
    // 第4层：vec2(1.0f, 0.75f) + vec2(0.0f, -p.x * 1.7f)
    {
        vec2 uv = p_layer * vec2(1.0f, 0.75f) + vec2(0.0f, -p_layer.x * 1.7f);
        float noise = smoothNoise(uv);
        float weight = 15.25f;
        vec2 dir = normalize(vec2(0.0, -1.0));
        offset += dir * noise * weight;
        weightSum += weight;
        
        p_layer /= 1.9f;
        p_layer.x += (frameTime / 155.0f) * 0.9f;
    }
    
    // 第5层：有特殊处理 (abs * 2.0 - 1.0) 和 AlmostIdentity
    {
        vec2 uv = p_layer * vec2(1.0f, 0.8f) + vec2(0.0f, -p_layer.x * 1.7f);
        float noise = abs(smoothNoise(uv) * 2.0 - 1.0);
        noise = 1.0 - AlmostIdentity(noise, 0.2f, 0.1f);
        float weight = 29.25f;
        vec2 dir = normalize(vec2(0.0, -1.0));
        offset += dir * noise * weight;
        weightSum += weight;
        
        p_layer /= 2.0f;
        p_layer.x += (frameTime / 155.0f) * 0.9f;
    }
    
    // 第6层
    {
        vec2 uv = p_layer * vec2(1.0f, 0.8f) + vec2(0.0f, p_layer.x * 1.7f);
        float noise = abs(smoothNoise(uv) * 2.0 - 1.0);
        noise = 1.0 - AlmostIdentity(noise, 0.2f, 0.1f);
        float weight = 15.25f;
        vec2 dir = normalize(vec2(0.0, 1.0));
        offset += dir * noise * weight;
        weightSum += weight;
    }
    
    // 归一化（与 GetWaves 一致）
    if (weightSum > 0.0) {
        offset /= weightSum;
    }
    
    // 应用高度缩放
    offset *= heightScale;  // 新增：控制波浪高度
    
    // 幅度调整（保持与原 getWaterOffset 一致的湿润度处理）
    float amp = amplitude * wetness;
    
    // 根据表面陡峭程度减弱（与原 getWaterOffset 一致）
    float verticalFactor = dot(surfaceNormal, vec3(0.0, 1.0, 0.0));
    verticalFactor = clamp(verticalFactor * 2.0 - 0.5, 0.0, 1.0);
    amp *= verticalFactor;
    
    return offset * amp;
}

float GetWaves(vec3 position, float scale, float frameTime, float isIce) {
    float speed = 0.9f;
    speed = mix(speed, 0.0f, isIce);

    vec2 p = position.xz / 5.0f;
    p.xy -= position.y / 10.0f;
    p.x = -p.x;

    p.x += (frameTime / 40.0f) * speed;
    p.y -= (frameTime / 40.0f) * speed;

    float weight = 1.0f;
    float weights = weight;
    float allwaves = 0.0f;

    // 第1层：替换 textureSmooth 为 smoothNoise
    float wave = smoothNoise((p * vec2(2.0f, 1.2f)) + vec2(0.0f, p.x * 2.1f));
    p /= 2.1f;
    p.y -= (frameTime / 20.0f) * speed;
    p.x -= (frameTime / 30.0f) * speed;
    allwaves += wave * 0.5;

    // 第2层
    weight = 2.1f;
    weights += weight;
    wave = smoothNoise((p * vec2(2.0f, 1.4f)) + vec2(0.0f, -p.x * 2.1f));
    p /= 1.5f;
    p.x += (frameTime / 20.0f) * speed;
    wave *= weight;
    allwaves += wave;

    // 第3层
    weight = 17.25f;
    weights += weight;
    wave = smoothNoise((p * vec2(1.0f, 0.75f)) + vec2(0.0f, p.x * 1.1f));
    p /= 1.5f;
    p.x -= (frameTime / 55.0f) * speed;
    wave *= weight;
    allwaves += wave;

    // 第4层
    weight = 15.25f;
    weights += weight;
    wave = smoothNoise((p * vec2(1.0f, 0.75f)) + vec2(0.0f, -p.x * 1.7f));
    p /= 1.9f;
    p.x += (frameTime / 155.0f) * speed;
    wave *= weight;
    allwaves += wave;

    // 第5层（有特殊处理）
    weight = 29.25f;
    weights += weight;
    wave = abs(smoothNoise((p * vec2(1.0f, 0.8f)) + vec2(0.0f, -p.x * 1.7f)) * 2.0 - 1.0);
    p /= 2.0f;
    p.x += (frameTime / 155.0f) * speed;
    wave = 1.0 - AlmostIdentity(wave, 0.2f, 0.1f);
    wave *= weight;
    allwaves += wave;

    // 第6层
    weight = 15.25f;
    weights += weight;
    wave = abs(smoothNoise((p * vec2(1.0f, 0.8f)) + vec2(0.0f, p.x * 1.7f)) * 2.0 - 1.0);
    wave = 1.0 - AlmostIdentity(wave, 0.2f, 0.1f);
    wave *= weight;
    allwaves += wave;

    allwaves /= weights;
    return allwaves;
}

vec3 GetWaterParallaxCoord(vec3 position, vec3 viewVector, float frameTime) {
    vec3 parallaxCoord = position.xyz;
    
    #ifdef WATER_PARALLAX
    vec3 stepSize = vec3(0.6f * WATER_WAVE_HEIGHT, 0.6f * WATER_WAVE_HEIGHT, 1.0f) * 0.5;
    float waveHeight = GetWaves(position, 1.0f, frameTime, 0.0f);
    
    vec3 pCoord = vec3(0.0f, 0.0f, 1.0f);
    vec3 step = viewVector * stepSize;
    float distAngleWeight = 1.0f; 
    step *= distAngleWeight;
    
    float sampleHeight = waveHeight;
    
    for (int i = 0; sampleHeight < pCoord.z && i < 60; ++i) {
        pCoord.xy = mix(pCoord.xy, pCoord.xy + step.xy, 
                       clamp((pCoord.z - sampleHeight) / (stepSize.z * 0.2f * distAngleWeight / (-viewVector.z + 0.05f)), 0.0f, 1.0f));
        pCoord.z += step.z;
        sampleHeight = GetWaves(position + vec3(pCoord.x, 0.0f, pCoord.y), 1.0f, frameTime, 0.0f);
    }
    
    parallaxCoord = position.xyz + vec3(pCoord.x, 0.0f, pCoord.y);
    #endif
    
    return parallaxCoord;
}

vec3 GetWavesNormal(vec3 position, float scale, vec3 viewDir, float frameTime) {
    vec3 viewVector = normalize(viewDir);
    
    #ifdef WATER_PARALLAX
    position = GetWaterParallaxCoord(position, viewVector, frameTime);
    #endif
    
    const float sampleDistance = 4.0f;
    position -= vec3(0.005f, 0.0f, 0.005f) * sampleDistance;
    
    float wavesCenter = GetWaves(position, scale, frameTime, 0.0f);
    float wavesLeft = GetWaves(position + vec3(0.01f * sampleDistance, 0.0f, 0.0f), scale, frameTime, 0.0f);
    float wavesUp   = GetWaves(position + vec3(0.0f, 0.0f, 0.01f * sampleDistance), scale, frameTime, 0.0f);
    
    vec3 wavesNormal;
    wavesNormal.r = wavesCenter - wavesLeft;
    wavesNormal.g = wavesCenter - wavesUp;
    
    wavesNormal.r *= 20.0f * WATER_WAVE_HEIGHT / sampleDistance;
    wavesNormal.g *= 20.0f * WATER_WAVE_HEIGHT / sampleDistance;
    
    wavesNormal.b = 1.0;
    wavesNormal.rgb = normalize(wavesNormal.rgb);
    
    return wavesNormal.rgb;
}

vec3 calculateNormalWater(vec2 uv)
{
    vec3 waterPos = vec3(uv.x, 0.0, uv.y);
    vec3 viewDir = normalize(vec3(0.0, -1.0, 0.0)); 
    
    float currentTime = frameTime;
    vec3 waveNormal = GetWavesNormal(waterPos, 1.0f, viewDir, currentTime);
    return waveNormal;
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

float traceWorldWaterBoundary(vec3 origin, vec3 dir, float tMin, float tMax, out uvec3 hitID) {
    ShadowRay savedShadowRay = shadowRay;
    shadowRay.hitT = INF_DISTANCE;
    shadowRay.queryMode = 1u;
    shadowRay.instanceIndex = 0u;
    shadowRay.geometryIndex = 0u;
    shadowRay.primitiveIndex = 0u;
    traceRayEXT(waterTopLevelAS,
                gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
                0x01,
                0, 0, 2,
                origin,
                tMin,
                dir,
                tMax,
                1);

    float hitT = shadowRay.hitT;
    hitID = uvec3(shadowRay.instanceIndex, shadowRay.geometryIndex, shadowRay.primitiveIndex);
    shadowRay = savedShadowRay;
    return hitT;
}

float sampleWaterCaustic(vec3 receiverWorldPos, vec3 lightDir, out float waterPathLength) {
    uvec3 waterHitID;
    vec3 traceOrigin = receiverWorldPos + lightDir * 0.01;
    waterPathLength = traceWorldWaterBoundary(traceOrigin, lightDir, 0.001, 500.0, waterHitID);

    if (waterPathLength >= INF_DISTANCE) {
        waterPathLength = 0.0;
        return 1.0;
    }

    vec3 waterSurfacePos = traceOrigin + lightDir * max(waterPathLength, 0.0);
    vec3 surfaceNormal = vec3(0.0, 1.0, 0.0);
    const float sampleStep = 0.22;

    vec2 offsetCenter = getWaterOffset(waterSurfacePos, 1.0, surfaceNormal);
    vec2 offsetXp = getWaterOffset(waterSurfacePos + vec3(sampleStep, 0.0, 0.0), 1.0, surfaceNormal);
    vec2 offsetXm = getWaterOffset(waterSurfacePos - vec3(sampleStep, 0.0, 0.0), 1.0, surfaceNormal);
    vec2 offsetZp = getWaterOffset(waterSurfacePos + vec3(0.0, 0.0, sampleStep), 1.0, surfaceNormal);
    vec2 offsetZm = getWaterOffset(waterSurfacePos - vec3(0.0, 0.0, sampleStep), 1.0, surfaceNormal);

    vec2 gradX = (offsetXp - offsetXm) / (2.0 * sampleStep);
    vec2 gradZ = (offsetZp - offsetZm) / (2.0 * sampleStep);
    float signedConvergence = -(gradX.x + gradZ.y);

    vec3 waveNormal = normalize(vec3(offsetCenter.x * 10.0, 1.0, offsetCenter.y * 8.0));
    vec3 refractedLight = refract(-lightDir, waveNormal, 1.0 / 1.333);
    if (length(refractedLight) < 1e-4) {
        refractedLight = -lightDir;
    } else {
        refractedLight = normalize(refractedLight);
    }

    float directionalFocus = pow(max(dot(refractedLight, vec3(0.0, -1.0, 0.0)), 0.0), 2.5);
    float slopeStrength = clamp(length(offsetCenter) * 24.0, 0.0, 1.0);
    float depthFade = exp(-waterPathLength * 0.12);
    float lowSunBoost = mix(2.4, 1.0, smoothstep(0.08, 0.55, max(lightDir.y, 0.0)));
    float signedCaustic = signedConvergence * (0.35 + 0.65 * slopeStrength) * (0.25 + 0.75 * directionalFocus);
    signedCaustic *= depthFade * 8.0 * lowSunBoost;

    float positiveCaustic = clamp(signedCaustic, 0.0, 1.0);
    float negativeCaustic = clamp(-signedCaustic, 0.0, 0.35);
    positiveCaustic = pow(positiveCaustic, 1.8);
    signedCaustic = positiveCaustic - negativeCaustic;
    signedCaustic = clamp(signedCaustic, -0.35, 1.0);

    return 1.0 + signedCaustic;
}

float traceWorldOpaqueBlocker(vec3 origin, vec3 dir, float tMin, float tMax) {
    ShadowRay savedShadowRay = shadowRay;
    shadowRay.hitT = INF_DISTANCE;
    shadowRay.throughput = vec3(1.0);
    shadowRay.radiance = vec3(0.0);
    shadowRay.queryMode = 1u;

    traceRayEXT(topLevelAS,
                gl_RayFlagsTerminateOnFirstHitEXT,
                WORLD_MASK,
                0, 0, 2,
                origin,
                tMin,
                dir,
                tMax,
                1);

    float hitT = shadowRay.hitT;
    shadowRay = savedShadowRay;
    return hitT;
}

int findWaterChunk(vec3 worldPos) {
    for (int i = 0; i < waterChunkOrigins.origins.length(); ++i) {
        vec3 origin = waterChunkOrigins.origins[i].xyz;
        vec3 size = vec3(waterChunkSizes.sizes[i].xyz);
        vec3 rel = worldPos - origin;
        if (all(greaterThanEqual(rel, vec3(0.0))) && all(lessThan(rel, size))) {
            return i;
        }
    }
    return -1;
}

int occupancyIndex(ivec3 localPos, ivec3 size) {
    return localPos.y * size.z * size.x + localPos.z * size.x + localPos.x;
}

bool sampleWaterOccupancy(int chunkIndex, ivec3 localPos) {
    if (chunkIndex < 0) return false;
    ivec3 size = waterChunkSizes.sizes[chunkIndex].xyz;
    if (any(lessThan(localPos, ivec3(0))) || any(greaterThanEqual(localPos, size))) return false;
    uint base = waterOccupancyOffsets.offsets[chunkIndex];
    uint idx = uint(occupancyIndex(localPos, size));
    return waterOccupancyData.values[base + idx] != 0u;
}

bool sampleSolidOccupancy(int chunkIndex, ivec3 localPos) {
    if (chunkIndex < 0) return false;
    ivec3 size = waterChunkSizes.sizes[chunkIndex].xyz;
    if (any(lessThan(localPos, ivec3(0))) || any(greaterThanEqual(localPos, size))) return false;
    uint base = solidOccupancyOffsets.offsets[chunkIndex];
    uint idx = uint(occupancyIndex(localPos, size));
    return solidOccupancyData.values[base + idx] != 0u;
}

float traceWaterOccupancyPath(vec3 origin, vec3 dir, float tMin, float tMax) {
    vec3 rayOrigin = origin + dir * tMin;
    vec3 rayDir = normalize(dir);
    float t = 0.0;
    const float stepSize = 0.5;

    bool enteredWater = false;

    while (t < tMax) {
        vec3 p = rayOrigin + rayDir * t;
        int chunkIndex = findWaterChunk(p);
        if (chunkIndex < 0) {
            if (enteredWater) {
               return t;
            }
            t += stepSize;
            continue;
        }

        vec3 originCell = waterChunkOrigins.origins[chunkIndex].xyz;
        ivec3 localPos = ivec3(floor(p - originCell));

        if (sampleSolidOccupancy(chunkIndex, localPos)) {
            return t;
        }

        if (sampleWaterOccupancy(chunkIndex, localPos)) {
            enteredWater = true;
            t += stepSize;
            continue;
        }

        if (enteredWater) {
            return t;
        }

        t += stepSize;
    }

    return enteredWater ? tMax : 0.0;
}

vec3 encodeWaterThicknessDebugColor(float pathLength) {
    if (pathLength <= 0.0) {
        return vec3(1.0, 0.0, 0.0);
    }

    float thicknessBand = 1.0 - clamp(pathLength / 32.0, 0.0, 1.0);
    float stripe = 0.55 + 0.45 * step(0.5, fract(pathLength * 0.25));
    return vec3(0.0, 1.0, 0.35) * (0.35 + 0.65 * thicknessBand) * stripe;
}

bool isContinuousWaterPath(vec3 origin, vec3 dir, float tEnd) {
    const float stepSize = 0.2;
    float t = 0.0;
    vec3 rayDir = normalize(dir);

    while (t < tEnd) {
        vec3 p = origin + rayDir * t;
        int chunkIndex = findWaterChunk(p);
        if (chunkIndex < 0) {
            return false;
        }

        vec3 originCell = waterChunkOrigins.origins[chunkIndex].xyz;
        ivec3 localPos = ivec3(floor(p - originCell));
        if (!sampleWaterOccupancy(chunkIndex, localPos)) {
            return false;
        }

        t += stepSize;
    }

    return true;
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
    // albedoValue = vec4(pow(tint, vec3(2.2)), albedoValue.a);

    LabPBRMat mat = convertLabPBRMaterial(albedoValue, specularValue, normalValue);
    // the provided normal is unreliable! (such as grass, etc.)
    // calculate on the fly for now
    vec3 normal =
        calculateNormal(v0.pos, v1.pos, v2.pos, v0.textureUV, v1.textureUV, v2.textureUV, mat.normal, viewDir);

    bool isWater = false;
    if (flagValue.g == 255) {
        isWater = true;
        albedoValue = vec4(0.85, 0.90, 0.90, 1.0);
        mat.albedo = albedoValue.rgb;
        mat.f0 = vec3(0.06); 
        mat.roughness = 0.001;
        mat.metallic = 0.0;
        mat.transmission = albedoValue.a;
        mat.ior = 1.5;
        mat.emission = 0.0;

        if (dot(normal, vec3(0.0, 1.0, 0.0)) > 0.5){
            vec2 WaterOffset = getWaterOffset(worldPos, 1.0f, normal);
            normal = normalize(normal + vec3(WaterOffset.x, 0.0, WaterOffset.y));
            //normal = calculateNormalWater((worldPos.xz + vec2(worldUbo.cameraPos.xz)));
        }

        if (mainRay.index == 0) {
            vec3 viewDir = normalize(mainRay.direction);
            vec3 n = normalize(normal);

            vec3 refractDir = refract(viewDir, n, 1.0 / 1.33);
            if (length(refractDir) < 1e-4) {
                refractDir = viewDir;
            } else {
                refractDir = normalize(refractDir);
            }

            float waterPathLength = 16.0;

            PrimaryRay savedMainRay = mainRay;
            mainRay.index = 1u;
            mainRay.hitT = 0.0;
            mainRay.directLightRadiance = vec3(0.0);
            mainRay.directLightHitT = 0.0;
            mainRay.stop = 0u;
            mainRay.cont = 0u;

            uint rayMask = WORLD_MASK | PLAYER_MASK | FISHING_BOBBER_MASK | WEATHER_MASK | PARTICLE_MASK | CLOUD_MASK;
            uint rayFlags = gl_RayFlagsCullBackFacingTrianglesEXT | gl_RayFlagsForceOpacityMicromap2StateEXT;
            traceRayEXT(topLevelAS,
                        rayFlags,
                        rayMask,
                        1,
                        1,
                        0,
                        worldPos + refractDir * 0.01,
                        0.001,
                        refractDir,
                        500.0,
                        0);

            waterPathLength = (mainRay.hitT < INF_DISTANCE) ? max(mainRay.hitT, 0.0) : 0.0;
            mainRay = savedMainRay;

            vec3 absorptionCoeff = vec3(0.65, 0.18, 0.06);
            vec3 scatteringCoeff = vec3(0.010, 0.018, 0.025);
            vec3 sigmaT = absorptionCoeff + scatteringCoeff;

            vec3 transmittance = exp(-sigmaT * waterPathLength);
            vec3 waterFogColor = vec3(0.02, 0.10, 0.16);

            mat.albedo = albedoValue.rgb * transmittance
                    + waterFogColor * (vec3(1.0) - transmittance);

        }
        
    }

    #define ENABLE_WETNESS_SMOOTH_TRANSITION 0
    if (skyUBO.rainGradient > 0.0 && mainRay.index == 0) {
        vec3 upDir = vec3(0.0, 1.0, 0.0);                       
        vec3 rayOrigin = worldPos + normal * 0.001;             

        ShadowRay savedShadowRay = shadowRay;
        shadowRay.hitT = INF_DISTANCE;
        shadowRay.queryMode = 0;

        traceRayEXT(topLevelAS,
                    gl_RayFlagsTerminateOnFirstHitEXT,       
                    WORLD_MASK,                                  
                    0,                                            // sbtRecordOffset
                    0,                                            // sbtRecordStride
                    2,                                            // missIndex
                    rayOrigin,
                    0.001,                                        // tMin
                    upDir,
                    1000.0,                                       // tMax
                    1                                             // payload location
                    );


        bool isExposed;
        float exposure;
        float wetness;
        float hitDist = shadowRay.hitT;  
        shadowRay = savedShadowRay;      
        
        if(ENABLE_WETNESS_SMOOTH_TRANSITION == 1){
            float minDist = 0.5;                
            float maxDist = 1.5;               
            exposure = smoothstep(minDist, maxDist, hitDist);
            wetness = skyUBO.rainGradient * exposure;
        }
        else{
            isExposed = (hitDist >= INF_DISTANCE);
            wetness = isExposed ? skyUBO.rainGradient : 0.0;
        }
      
        vec3 edge1 = v1.pos - v0.pos;
        vec3 edge2 = v2.pos - v0.pos;
        vec3 geoNormalObj = normalize(cross(edge1, edge2));
        mat3 normalMatrix = transpose(mat3(gl_WorldToObject3x4EXT));
        vec3 geometricNormalWorld = normalize(normalMatrix * geoNormalObj);
        bool isTopFace = geometricNormalWorld.y > 0.9; 

        if(!isWater){

            if(!isTopFace){
                wetness *= 0.8; 
            }
            float minRoughness = 0.001;
            float maxRoughness = mat.roughness;
            mat.roughness = mix(minRoughness, maxRoughness, 1.0 - wetness); 
            mat.albedo.rgb *= (1.0 - 0.3 * wetness); 
            if (mat.metallic == 0.0) {
                vec3 targetF0 = vec3(0.2);             
                float factor = wetness * 0.8;           
                mat.f0 = mix(mat.f0, targetF0, factor);
            }

            
            
            float normalStrength = 1.0 - wetness * 0.95; 
            normal = normalize(mix(geometricNormalWorld, normal, normalStrength));


        }
        if (wetness > 0.0 && isTopFace) {
            vec2 rainOffset = getRainDropOffset(worldPos, wetness, normal);
            normal = normalize(normal + vec3(rainOffset.x, 0.0, rainOffset.y));
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
        shadowRay.hitWater = 0u;
        shadowRay.seed = mainRay.seed;
        shadowRay.hitT = INF_DISTANCE;
        shadowRay.insideBoat = mainRay.insideBoat;
        shadowRay.bounceIndex = mainRay.index;
        shadowRay.queryMode = 0;

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
        bool shadowRayHitWater = (shadowRay.hitWater != 0u);
        float waterShadowPathLength = 0.0;

        if(mainRay.index == 0 && shadowRayHitWater){   

            PrimaryRay savedMainRay = mainRay;
            mainRay.index = 1u;
            mainRay.hitT = 0.0;
            mainRay.directLightRadiance = vec3(0.0);
            mainRay.directLightHitT = 0.0;
            mainRay.stop = 0u;
            mainRay.cont = 0u;
            uint rayMask = WORLD_MASK | PLAYER_MASK | FISHING_BOBBER_MASK | WEATHER_MASK | PARTICLE_MASK | CLOUD_MASK;
            uint rayFlags = gl_RayFlagsCullBackFacingTrianglesEXT | gl_RayFlagsForceOpacityMicromap2StateEXT;
            traceRayEXT(topLevelAS,
                        rayFlags,
                        rayMask,
                        1,
                        1,
                        0,
                        shadowRayOrigin + sampledLightDir * 0.01,
                        0.001,
                        sampledLightDir,
                        500.0,
                        0);


            waterShadowPathLength = (mainRay.hitT < INF_DISTANCE) ? max(mainRay.hitT, 0.0) : 0.0;
            mainRay = savedMainRay;

            vec3 absorptionCoeff = vec3(0.65, 0.18, 0.06);
            vec3 scatteringCoeff = vec3(0.010, 0.018, 0.025);
            vec3 sigmaT = absorptionCoeff + scatteringCoeff;
            vec3 waterShadowTransmittance = exp(-sigmaT * waterShadowPathLength);
            //shadowRay.waterShadowTransmittance = waterShadowTransmittance;
            mainRay.waterShadowTransmittance = waterShadowTransmittance;
            lightContribution *= waterShadowTransmittance;
            //mainRay.throughput *= waterShadowTransmittance;
        }

        if (mainRay.index != 0u && shadowRayHitWater) {
            lightContribution *= mainRay.waterShadowTransmittance;
        }

        if (shadowRayHitWater) {
            float waterCausticPathLength = 0.0;
            float waterCaustic = sampleWaterCaustic(worldPos, sampledLightDir, waterCausticPathLength);
            lightContribution *= waterCaustic;
        }

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

