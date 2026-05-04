#version 450 core

layout(location = 0) in vec3 inLocalPos;
layout(location = 1) in vec3 inLocalCameraPos;

layout(binding = 0) uniform sampler3D sunVolume; // The generated 3D texture

layout(location = 0) out vec4 outColor;

// Ray-Sphere intersection to bound our raymarching
vec2 intersectSphere(vec3 ro, vec3 rd, float radius) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - radius * radius;
    float h = b * b - c;
    if (h < 0.0) return vec2(-1.0); // No intersection
    h = sqrt(h);
    return vec2(-b - h, -b + h);
}

void main() {
    bool isInside = all(lessThan(abs(inLocalCameraPos), vec3(2.0001)));
    if (isInside && !gl_FrontFacing) discard;
    if (!isInside && gl_FrontFacing) discard;

    vec3 rayDir = normalize(inLocalPos - inLocalCameraPos);
    
    // Intersect with a sphere (radius 2.0) in local space
    vec2 t = intersectSphere(inLocalCameraPos, rayDir, 2.0);
    if (t.y < 0.0) discard; // Ray missed the sphere entirely
    
    // Clamp start distance to 0 if the camera is inside the sun
    float tMin = max(t.x, 0.0);
    float tMax = t.y;
    
    // Raymarching setup
    float stepSize = 0.03125; // doubled fixed step size for 2x volume
    int steps = min(int(ceil((tMax - tMin) / stepSize)), 128);
    float currentT = tMin;
    
    vec4 accumulatedColor = vec4(0.0);
    
    for (int i = 0; i < steps; i++) {
        vec3 localPos = inLocalCameraPos + rayDir * currentT;
        
        // Map local position [-2, 2] to UVW [0, 1] for texture sampling
        vec3 uvw = localPos * 0.25 + 0.5;
        
        // Sample your generated volume
        // voxel.rgb = emission, voxel.a = density/opacity
        vec4 voxel = texture(sunVolume, uvw);
        
        // Front-to-back volumetric accumulation
        // Multiply emission by step size (compute shader already pre-multiplied emission by density)
        vec3 emission = voxel.rgb * stepSize;
        float alpha = voxel.a * stepSize;
        
        // Accumulate color and alpha (standard alpha compositing)
        accumulatedColor.rgb += (1.0 - accumulatedColor.a) * emission;
        accumulatedColor.a += (1.0 - accumulatedColor.a) * alpha;
        
        // Early exit if the volume becomes completely opaque
        if (accumulatedColor.a >= 0.99) break;
        
        currentT += stepSize;
    }
    
    outColor = accumulatedColor;
}