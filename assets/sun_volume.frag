#version 450 core

layout(location = 0) in  vec3 inLocalPos;
layout(location = 1) in  vec3 inLocalCameraPos;

layout(binding = 0) uniform sampler3D sunVolume;

layout(location = 0) out vec4 outColor;

// ============================================================
// RAY-SPHERE INTERSECTION
// Returns (t_near, t_far).  Both components negative when no hit.
// ============================================================
vec2 intersectSphere(vec3 ro, vec3 rd, float radius) {
  float b = dot(ro, rd);
  float c = dot(ro, ro) - radius * radius;
  float h = b * b - c;
  if (h < 0.0) return vec2(-1.0);
  h = sqrt(h);
  return vec2(-b - h, -b + h);
}

// ============================================================
// ANALYTICAL OUTER CORONA GLOW
// Approximates K-corona (electron-scattered) brightness for rays
// that pass through -- or just outside -- the volume sphere boundary.
// Adds a soft luminous halo beyond what the 3-D texture captures.
// ============================================================
vec3 outerCoronaGlow(vec3 ro, vec3 rd, float photosphereR, float volumeR) {
  float b  = dot(ro, rd);
  float d2 = max(dot(ro, ro) - b * b, 0.0);
  float d  = sqrt(d2);
  if (d >= volumeR) return vec3(0.0);

  // Chord length through the outer corona shell beyond volumeR.
  float outerSqrt = sqrt(max(volumeR * volumeR - d2, 0.0));
  float t1 = max(-b - outerSqrt, 0.0);
  float t2 = -b + outerSqrt;

  // Exponential column density; scale height must match sungen.comp
  float H          = 0.25;
  float colDensity = max(t2 - t1, 0.0)
                   * exp(-(d - photosphereR) / H)
                   * 0.12;  // artistic brightness scale

  // Blue-white K-corona colour
  return vec3(0.72, 0.84, 1.00) * colDensity;
}

// ============================================================
// MAIN
// ============================================================
void main() {
  // Cull back-facing fragments based on whether the camera is inside or outside
  // the bounding cube (local space [-2, 2]^3).
  bool isInside = all(lessThan(abs(inLocalCameraPos), vec3(2.0001)));
  if ( isInside && !gl_FrontFacing) discard;
  if (!isInside &&  gl_FrontFacing) discard;

  vec3 rayDir = normalize(inLocalPos - inLocalCameraPos);

  // Intersect the bounding sphere (radius 2.0 in local space).
  vec2 t = intersectSphere(inLocalCameraPos, rayDir, 2.0);
  if (t.y < 0.0) discard;

  float tMin = max(t.x, 0.0);
  float tMax = t.y;

  // Adaptive step size: 256 steps distributed over the actual ray length.
  // For a ray through the sphere centre (4.0 units) this gives stepSz = 0.015625,
  // twice the quality of the original 128-step fixed sampler.
  const int MAX_STEPS = 256;
  float rayLen = tMax - tMin;
  float stepSz = max(rayLen / float(MAX_STEPS), 0.004);
  int   nSteps = min(int(ceil(rayLen / stepSz)), MAX_STEPS);

  // Stochastic sub-pixel jitter on the ray start position.
  // Breaks the regular banding pattern that appears with fixed-step marching.
  float jitter   = fract(sin(dot(inLocalPos.xy, vec2(12.9898, 78.233))) * 43758.5453);
  float currentT = tMin + stepSz * jitter;

  // Front-to-back volumetric accumulation.
  // voxel.rgb = emission pre-weighted by density (baked in sungen.comp).
  // voxel.a   = optical depth / opacity per unit length.
  vec4 accumulated = vec4(0.0);

  for (int i = 0; i < nSteps; ++i) {
    vec3 localPos = inLocalCameraPos + rayDir * currentT;

    // Map local [-2, 2] to UVW [0, 1].
    vec3 uvw = localPos * 0.25 + 0.5;

    vec4  voxel = texture(sunVolume, uvw);
    float alpha = voxel.a   * stepSz;
    vec3  emit  = voxel.rgb * stepSz;

    accumulated.rgb += (1.0 - accumulated.a) * emit;
    accumulated.a   += (1.0 - accumulated.a) * alpha;

    if (accumulated.a >= 0.99) break; // early-exit when fully opaque

    currentT += stepSz;
  }

  // Analytical outer corona halo.
  // Adds a soft blue-white glow that extends slightly beyond the volume sphere
  // (outerR = 2.4 > volume sphere radius 2.0).  Only contributes where the
  // accumulated alpha left transmittance -- i.e. corona rays, not the disk.
  vec3 glowColor = outerCoronaGlow(inLocalCameraPos, rayDir, 0.6, 2.4);
  accumulated.rgb += (1.0 - accumulated.a) * glowColor;

  outColor = accumulated;
}
