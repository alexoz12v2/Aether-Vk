$ErrorActionPreference = "Stop"

if (-not $env:VULKAN_SDK) {
    Write-Error "VULKAN_SDK environment variable is not set."
    exit 1
}

$glslc = Join-Path $env:VULKAN_SDK "bin\glslc.exe"
if (-not (Test-Path $glslc)) {
    $glslc = Join-Path $env:VULKAN_SDK "Bin\glslc.exe"
    if (-not (Test-Path $glslc)) {
        Write-Error "glslc not found in VULKAN_SDK\bin or VULKAN_SDK\Bin"
        exit 1
    }
}

$spirvVal = Join-Path $env:VULKAN_SDK "bin\spirv-val.exe"
if (-not (Test-Path $spirvVal)) {
    $spirvVal = Join-Path $env:VULKAN_SDK "Bin\spirv-val.exe"
    if (-not (Test-Path $spirvVal)) {
        Write-Error "spirv-val not found in VULKAN_SDK\bin or VULKAN_SDK\Bin"
        exit 1
    }
}

$CommonFlags = @("-x", "glsl", "--target-env=vulkan1.1", "--target-spv=spv1.3", "-std=450core", "-Os")
$WgSizes = @(4, 8, 16, 32, 64, 128, 256)

$WgVariantShaders = @(
    "integrate_bodies_p3.comp",
    "rb_force_assign.comp",
    "integrate_particles_p1_p2.comp",
    "integrate_particles_p4_5.comp",
    "apply_emitters_to_particles.comp",
    "apply_emitters_direct.comp",
    "accumulate_bvh_forces_to_particles.comp",
    "apply_impulses.comp",
    "emit_particles.comp",
    "convert_particles.comp",
    "lbvh_build.comp",
    "lbvh_build_bottomup.comp",
    "lbvh_prepass.comp",
    "lbvh_collapse.comp",
    "motion_bounds.comp",
    "motion_refit.comp",
    "bp_bounds_gen.comp",
    "bp_classify.comp",
    "bp_cross_lca.comp",
    "bp_particle_self.comp",
    "bp_scene.comp",
    "ccd.comp",
    "narrow_ccd.comp",
    "narrow_ccd_cross_lca.comp",
    "reduce_toi.comp",
    "stream_compact.comp",
    "graph_coloring.comp",
    "lcp_solver.comp",
    "barnes_hut.comp",
    "radix_sort.comp",
    # new stuff
    "apply_emitters_direct_new.comp",
    "integrate_particles_p1_p2_new.comp",
    "integrate_particles_p4_5_new.comp",
    "new_particles_compact_reset.comp",
    "new_particles_emit.comp",
    "new_particles_compact.comp",
    "new_particles_offset_particles.comp",
    "reset_particles.comp"
)

# Shaders that use float16_t / f16vec4 arithmetic (not just buffer layout).
# Compiled with NATIVE_FLOAT16=1 (native) and NATIVE_FLOAT16=0 (.nofp16 fallback).
$Float16VariantShaders = @(
    "apply_emitters_direct_new.comp",
    "integrate_particles_p1_p2_new.comp",
    "integrate_particles_p4_5_new.comp",
    "new_particles_emit.comp",
    "new_particles_offset_particles.comp"
)

function Compile-One {
    param (
        [string]$File,
        [string]$Stage,
        [string[]]$ExtraFlags,
        [string]$Out
    )
    $BaseOut = Split-Path $Out -Leaf
    $ExtraStr = if ($ExtraFlags) { $ExtraFlags -join ' ' } else { "" }
    Write-Host "  glslc $ExtraStr -> $BaseOut"

    $ArgsList = @()
    $ArgsList += $CommonFlags
    $ArgsList += "-fshader-stage=$Stage"
    if ($ExtraFlags -and $ExtraFlags.Length -gt 0) {
        $ArgsList += $ExtraFlags
    }
    $ArgsList += "-o", $Out, $File

    & $glslc $ArgsList
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to compile $File -> $Out"
        exit $LASTEXITCODE
    }

    & $spirvVal $Out
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Validation failed for $Out"
        exit $LASTEXITCODE
    }
}

Write-Host "=== Compiling vertex / fragment shaders ==="
$vfFiles = Get-ChildItem -Path "assets" -File | Where-Object { $_.Extension -match "\.(vert|frag)$" }
foreach ($file in $vfFiles) {
    $ext = $file.Extension.TrimStart('.')
    Compile-One -File $file.FullName -Stage $ext -ExtraFlags @() -Out "$($file.FullName).spv"
}

Write-Host "`n=== Compiling compute shaders ==="
$compFiles = Get-ChildItem -Path "assets", "assets/sim" -File | Where-Object { $_.Extension -eq ".comp" }
foreach ($file in $compFiles) {
    $base = $file.Name
    Write-Host "Shader: $($file.FullName)"

    $outBase = $file.FullName -replace '\.comp$', ''

    if ($WgVariantShaders -contains $base) {
        if ($Float16VariantShaders -contains $base) {
            # Produce fp16-native (NATIVE_FLOAT16=1) and fp32-fallback (NATIVE_FLOAT16=0)
            # blobs for every workgroup size. The .nofp16 infix marks the fallback blobs.
            foreach ($fp16 in @(1, 0)) {
                $fp16Flag = "-DNATIVE_FLOAT16=$fp16"
                $fp16Infix = if ($fp16 -eq 1) { "" } else { ".nofp16" }

                Compile-One -File $file.FullName -Stage "comp" `
                    -ExtraFlags @($fp16Flag) `
                    -Out "$outBase.comp$fp16Infix.spv"
                Compile-One -File $file.FullName -Stage "comp" `
                    -ExtraFlags @($fp16Flag, "-DDEBUG_SHADERS") `
                    -Out "$outBase.comp$fp16Infix.d.spv"

                foreach ($wg in $WgSizes) {
                    Compile-One -File $file.FullName -Stage "comp" `
                        -ExtraFlags @($fp16Flag, "-DLOCAL_SIZE_X=$wg") `
                        -Out "$outBase.comp$fp16Infix.wg$wg.spv"
                    Compile-One -File $file.FullName -Stage "comp" `
                        -ExtraFlags @($fp16Flag, "-DLOCAL_SIZE_X=$wg", "-DDEBUG_SHADERS") `
                        -Out "$outBase.comp$fp16Infix.wg$wg.d.spv"
                }
            }
        } else {
            # Non-float16 wg-variant: single compile pass (existing behaviour).
            Compile-One -File $file.FullName -Stage "comp" -ExtraFlags @() -Out "$($file.FullName).spv"
            Compile-One -File $file.FullName -Stage "comp" -ExtraFlags @("-DDEBUG_SHADERS") -Out "$outBase.comp.d.spv"

            foreach ($wg in $WgSizes) {
                $outWg = "$outBase.comp.wg$wg.spv"
                Compile-One -File $file.FullName -Stage "comp" -ExtraFlags @("-DLOCAL_SIZE_X=$wg") -Out $outWg

                $outWgD = "$outBase.comp.wg$wg.d.spv"
                Compile-One -File $file.FullName -Stage "comp" -ExtraFlags @("-DLOCAL_SIZE_X=$wg", "-DDEBUG_SHADERS") -Out $outWgD
            }
        }
    } else {
        Compile-One -File $file.FullName -Stage "comp" -ExtraFlags @() -Out "$($file.FullName).spv"
        Compile-One -File $file.FullName -Stage "comp" -ExtraFlags @("-DDEBUG_SHADERS") -Out "$outBase.comp.d.spv"
    }
}

Write-Host "`nAll shaders compiled successfully."