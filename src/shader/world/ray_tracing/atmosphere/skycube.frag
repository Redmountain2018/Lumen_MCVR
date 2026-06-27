#version 460
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"

layout(set = 1, binding = 0) uniform WorldUniform {
    WorldUBO worldUBO;
};

layout(set = 1, binding = 1) uniform SkyUniform {
    SkyUBO skyUBO;
};

layout(push_constant) uniform Push {
    int face; // 0..5
}
pc;

layout(location = 0) in vec2 texCoord;

layout(location = 0) out vec4 outColor;

bool intersectSphere(vec3 ro, vec3 rd, float R, out float tNear, out float tFar) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - R * R;
    float h = b * b - c;
    if (h < 0.0) return false;
    h = sqrt(h);
    tNear = -b - h;
    tFar = -b + h;
    return true;
}

float densityExpImproved(float h, float H) {
    // Variable scale height: increases with altitude
    float H_eff = H * (1.0 + h / 100000.0);
    float density = exp(-max(h, 0.0) / H_eff);
    // Extra attenuation in upper atmosphere to simulate transition to space
    float extra = exp(-max(h - 50000.0, 0.0) / 50000.0);
    return density * extra;
}

float ozoneAbsorption(float h) {
    // Ozone layer centered at 25 km, Gaussian profile
    float peakHeight = 25000.0;
    float width = 10000.0;
    return 0.1 * exp(-pow((h - peakHeight) / width, 2.0));
}

float rayleighPhase(float cosTheta) {
    return (3.0 / (16.0 * PI)) * (1.0 + cosTheta * cosTheta);
}

float miePhase(float cosTheta, float g) {
    float g2 = g * g;
    float denom = 4.0 * PI * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5);
    return (1.0 - g2) / denom;
}

float fogDensity(float height) {
    // exponential fog density
    float fogHeight = skyUBO.Hr; // reuse Rayleigh scale height as fog scale height
    return exp(-max(height, 0.0) / fogHeight);
}

float isotropicPhase() {
    return 1.0 / (4.0 * PI);
}

vec3 integrateFogScattering(vec3 ro, vec3 rd, vec3 sunDir, vec3 sunColor) {
    float tAtm0, tAtm1;
    if (!intersectSphere(ro, rd, skyUBO.Rt, tAtm0, tAtm1)) return vec3(0.0);
    tAtm0 = max(tAtm0, 0.0);

    // stop at ground
    float tG0, tG1;
    if (intersectSphere(ro, rd, skyUBO.Rg, tG0, tG1)) {
        float tHitG = (tG0 > 0.0) ? tG0 : tG1;
        if (tHitG > 0.0) tAtm1 = min(tAtm1, tHitG);
    }

    const int STEPS = 64;
    float dt = (tAtm1 - tAtm0) / float(STEPS);

    vec3 L = vec3(0.0);
    vec3 transmittance = vec3(1.0);

    for (int i = 0; i < STEPS; i++) {
        float t = tAtm0 + (float(i) + 0.5) * dt;
        vec3 x = ro + rd * t;
        float r = length(x);
        float h = r - skyUBO.Rg;

        // densities
        float dR = densityExpImproved(h, skyUBO.Hr);
        float dM = densityExpImproved(h, skyUBO.Hm);
        float ozone = ozoneAbsorption(h);

        // scattering coefficients
        vec3 sigmaS_R = skyUBO.betaR * dR;
        vec3 sigmaS_M = skyUBO.betaM * dM;
        vec3 sigmaAbs = vec3(ozone * 0.1); // absorption coefficient
        vec3 sigmaE = sigmaS_R + sigmaS_M + sigmaAbs; // extinction

        // phase functions
        float cosTheta = dot(normalize(sunDir), rd);
        float phaseR = rayleighPhase(cosTheta);
        float phaseM = miePhase(cosTheta, skyUBO.mieG);

        // sun‑light attenuation along the path from top of atmosphere to x
        float cosSunZenith = dot(normalize(x), sunDir);
        // clamp to avoid division by zero and negative values (sun below horizon)
        float secant = 1.0 / max(cosSunZenith, 0.001);
        // vertical optical depths (unitless)
        float verticalDepthR = skyUBO.Hr * dR;
        float verticalDepthM = skyUBO.Hm * dM;
        float verticalDepthOzone = ozone * 0.1 * 10000.0; // approximate scale height for ozone
        vec3 opticalDepthSun = (skyUBO.betaR * verticalDepthR +
                                skyUBO.betaM * verticalDepthM +
                                vec3(verticalDepthOzone)) * secant;
        vec3 transmittanceSun = exp(-opticalDepthSun);

        // sunlight contribution
        vec3 sunLight = sunColor * transmittanceSun;
        vec3 scattering = (sigmaS_R * phaseR + sigmaS_M * phaseM) * sunLight;

        L += transmittance * scattering * dt;
        transmittance *= exp(-sigmaE * dt);
    }

    return L;
}

vec3 cubemapDir(int face, vec2 uv01) {
    vec2 uv = uv01 * 2.0 - 1.0;
    if (face == 0) return normalize(vec3(1, -uv.y, -uv.x)); // +X
    if (face == 1) return normalize(vec3(-1, -uv.y, uv.x)); // -X
    if (face == 2) return normalize(vec3(uv.x, 1, uv.y));   // +Y
    if (face == 3) return normalize(vec3(uv.x, -1, -uv.y)); // -Y
    if (face == 4) return normalize(vec3(uv.x, -uv.y, 1));  // +Z
    return normalize(vec3(-uv.x, -uv.y, -1));               // -Z
}

void main() {
    vec3 rd = cubemapDir(pc.face, texCoord);
    float blend = smoothstep(-0.05, 0.02, rd.y);
    rd.y = max(rd.y, skyUBO.minViewCos);
    rd = normalize(mix(vec3(rd.x, skyUBO.minViewCos, rd.z), rd, blend));
    float cameraHeight = worldUBO.cameraViewMatInv[3].y;
    vec3 ro = vec3(0.0, skyUBO.Rg + cameraHeight + 70, 0.0);

    // sun and moon contributions
    vec3 sunColor = skyUBO.sunRadiance * skyUBO.envCelestial.z;
    vec3 moonColor = skyUBO.moonRadiance * skyUBO.envCelestial.w;

    vec3 L = integrateFogScattering(ro, rd, skyUBO.sunDirection, sunColor)
           + integrateFogScattering(ro, rd, skyUBO.moonDirection, moonColor);


    outColor = vec4(L, 1.0);
}
