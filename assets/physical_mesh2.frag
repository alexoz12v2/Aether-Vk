#version 450 core

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_GOOGLE_include_directive : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_memory_scope_semantics : require

#include "common.glsl"

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec3 inBitangent;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D albedoMap;
layout(binding = 1) uniform sampler2D normalMap;
layout(binding = 2) uniform sampler2D roughnessMap;
layout(binding = 3) uniform sampler2D aoMap;
layout(binding = 4) uniform sampler2D skyMap;
layout(binding = 5) uniform sampler2D emissivePaintMap; // RGBA Texture for Emissive + Paint Tool

const uint FLAG_ALBEDO    = 1u << 0;
const uint FLAG_NORMAL    = 1u << 1;
const uint FLAG_ROUGHNESS = 1u << 2;
const uint FLAG_AO        = 1u << 3;
const uint FLAG_EMISSIVE  = 1u << 4; // Gameplay mode flag for emission

const uint PAINT_MODE_NONE         = 0u;
const uint PAINT_MODE_COLOR        = 1u;
const uint PAINT_MODE_DISTRIBUTION = 2u;
const uint PAINT_MODE_SPHERICAL_GRID = 3u;

vec3 orenNayar(vec3 viewDir, vec3 lightDir, vec3 normal, vec3 albedo, float roughness) {
    float LdotN = max(dot(lightDir, normal), 0.0);
    float VdotN = max(dot(viewDir, normal), 0.0);
    if (LdotN <= 0.0) return vec3(0.0);

    float roughness2 = roughness * roughness;
    float A = 1.0 - 0.5 * (roughness2 / (roughness2 + 0.33));
    float B = 0.45 * (roughness2 / (roughness2 + 0.09));

    float LdotV = dot(lightDir, viewDir);
    float s = LdotV - LdotN * VdotN;
    float t = mix(1.0, max(LdotN, VdotN), step(0.0, s));

    return albedo * LdotN * (A + B * s / t) / 3.14159;
}

vec2 octEncode(vec3 v) {
    v /= (abs(v.x) + abs(v.y) + abs(v.z));
    vec2 uv = v.z >= 0.0 ? v.xy : (1.0 - abs(v.yx)) * sign(v.xy);
    return uv * 0.5 + 0.5;
}

void main() {
    MaterialData mat = push.material;
    SceneData scene  = push.scene;

    if (mat.emissiveColor.a < 0.0) {
        outColor = vec4(mat.emissiveColor.rgb, 1.0);
        return;
    }

    // -------------------------------------------------------------
    // Editor Visualization Tools Overrides (Bypasses Lighting entirely)
    // -------------------------------------------------------------
    if (mat.paintDisplayMode != PAINT_MODE_NONE) {
        vec4 paintSample = texture(emissivePaintMap, inUV);

        if (mat.paintDisplayMode == PAINT_MODE_COLOR) {
            // Visualizes painted RGB channels directly (Unlit)
            outColor = vec4(paintSample.rgb, 1.0);
            return;
        } else if (mat.paintDisplayMode == PAINT_MODE_DISTRIBUTION) {
            // Visualizes probability density via Alpha channel as pure Unlit grayscale
            outColor = vec4(vec3(paintSample.a), 1.0);
            return;
        } else if (mat.paintDisplayMode == PAINT_MODE_SPHERICAL_GRID) {
            vec3 localPos = inWorldPos - mat.sphereCenterRadius.xyz;
            vec3 n = normalize(localPos);
            
            float phi = atan(n.y, n.x);
            float theta = asin(n.z);
            
            float gridSpacing = 3.14159265 / 18.0; // 10 degrees
            vec2 p = vec2(phi, theta) / gridSpacing;
            
            vec2 dp = fwidth(p);
            // Fix fwidth wrap artifact on phi
            if (dp.x > 1.0) dp.x = 0.0;
            
            float minorLineWidth = 0.05 * mat.gridColorDensity.w;
            vec2 minorGrid = smoothstep(minorLineWidth + dp, max(minorLineWidth - dp, 0.0), abs(fract(p + 0.5) - 0.5));
            float minorAlpha = max(minorGrid.x, minorGrid.y);
            
            vec2 pMajor = p / 3.0; // 30 degrees
            vec2 dpMajor = dp / 3.0;
            float majorLineWidth = 0.1 * mat.gridColorDensity.w;
            vec2 majorGrid = smoothstep(majorLineWidth + dpMajor, max(majorLineWidth - dpMajor, 0.0), abs(fract(pMajor + 0.5) - 0.5));
            float majorAlpha = max(majorGrid.x, majorGrid.y);
            
            // Equator (Z=0 plane) -> Blue
            float zDist = abs(n.z);
            float zAxisAlpha = 1.0 - smoothstep(0.0, 0.015 * mat.gridColorDensity.w, zDist);
            
            // Prime Meridian (Y=0 plane) -> Red (aligns with X-axis)
            float xDist = abs(n.y);
            float xAxisAlpha = 1.0 - smoothstep(0.0, 0.015 * mat.gridColorDensity.w, xDist);
            
            vec3 color = mat.gridColorDensity.xyz;
            float alpha = max(minorAlpha * 0.3, majorAlpha * 0.8);
            
            if (zAxisAlpha > 0.1) {
                color = vec3(0.0, 0.0, 1.0);
                alpha = max(alpha, zAxisAlpha);
            }
            if (xAxisAlpha > 0.1) {
                color = vec3(1.0, 0.0, 0.0);
                alpha = max(alpha, xAxisAlpha);
            }
            
            if (alpha < 0.01) discard;
            outColor = vec4(color, alpha);
            return;
        }
    }

    // -------------------------------------------------------------
    // Standard PBR Setup
    // -------------------------------------------------------------
    bool useAlbedo    = (mat.textureFlags & FLAG_ALBEDO) != 0u;
    bool useNormal    = (mat.textureFlags & FLAG_NORMAL) != 0u;
    bool useRoughness = (mat.textureFlags & FLAG_ROUGHNESS) != 0u;
    bool useAO        = (mat.textureFlags & FLAG_AO) != 0u;
    bool useEmissive  = (mat.textureFlags & FLAG_EMISSIVE) != 0u;

    vec3 V = normalize(scene.cameraPos.xyz - inWorldPos);
    vec3 unnormalizedLightVector = scene.sunPos.xyz - inWorldPos;
    float distanceToSun = length(unnormalizedLightVector);
    vec3 lightDir = unnormalizedLightVector / distanceToSun;

    float attenuation = 1.0 / (1.0 + 0.001 * distanceToSun);
    vec3 lightColor = scene.sunColor.xyz * attenuation;

    vec3 albedo = mat.baseAlbedo.rgb;
    float roughness = mat.baseAlbedo.a;
    float ao = mat.baseAO;
    vec3 N = normalize(inNormal);

    if (useAlbedo)    albedo *= texture(albedoMap, inUV).rgb;
    if (useRoughness) roughness *= texture(roughnessMap, inUV).r;
    if (useAO)        ao *= texture(aoMap, inUV).r;

    if (useNormal) {
        vec3 T = normalize(inTangent);
        vec3 B = normalize(inBitangent);
        mat3 TBN = mat3(T, B, N);

        vec3 sampledNormal = texture(normalMap, inUV).xyz * 2.0 - 1.0;
        N = normalize(TBN * sampledNormal);
    }

    vec3 diffuse = orenNayar(V, lightDir, N, albedo, roughness);
    vec3 R = reflect(-V, N);

    vec3 irradiance = texture(skyMap, octEncode(N)).rgb * 0.02;
    vec3 diffuseIBL = irradiance * albedo;

    vec2 refUV = octEncode(R);
    float blurSpread = 0.2 * roughness + 0.05;
    vec3 radiance = (
        texture(skyMap, refUV).rgb +
        texture(skyMap, refUV + vec2(blurSpread, 0.0)).rgb +
        texture(skyMap, refUV + vec2(-blurSpread, 0.0)).rgb +
        texture(skyMap, refUV + vec2(0.0, blurSpread)).rgb +
        texture(skyMap, refUV + vec2(0.0, -blurSpread)).rgb +
        texture(skyMap, refUV + vec2(blurSpread, blurSpread)).rgb +
        texture(skyMap, refUV + vec2(-blurSpread, blurSpread)).rgb +
        texture(skyMap, refUV + vec2(blurSpread, -blurSpread)).rgb +
        texture(skyMap, refUV + vec2(-blurSpread, -blurSpread)).rgb
    ) / 9.0 * 0.01;

    vec3 F0 = vec3(0.04);
    vec3 F = F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - max(dot(N, V), 0.0), 0.0, 1.0), 5.0);
    vec3 kS = F;
    vec3 kD = 1.0 - kS;

    vec3 specularIBL = radiance * F;
    vec3 ambient = (kD * diffuseIBL + specularIBL) * ao;
    ambient += vec3(0.05) * albedo * ao;

    // Emissive Integration
    vec3 emission = mat.emissiveColor.rgb * max(mat.emissiveColor.a, 0.0);
    if (useEmissive) {
        // Only uses the RGB channels for gameplay emission glow. Alpha is safely ignored.
        emission *= texture(emissivePaintMap, inUV).rgb;
    }

    outColor = vec4(diffuse * lightColor * ao + ambient + emission, 1.0);
}
