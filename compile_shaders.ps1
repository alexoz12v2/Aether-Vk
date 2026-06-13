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
    "radix_sort.comp"
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
        Compile-One -File $file.FullName -Stage "comp" -ExtraFlags @() -Out "$($file.FullName).spv"
        Compile-One -File $file.FullName -Stage "comp" -ExtraFlags @("-DDEBUG_SHADERS") -Out "$outBase.comp.d.spv"

        $bvhUtilsPath = Join-Path $PWD "assets\bvh_utils.glsl"
        $bvhUtilsBak = Join-Path $PWD "assets\bvh_utils.glsl.bak"

        foreach ($wg in $WgSizes) {
            if (Test-Path $bvhUtilsPath) {
                Copy-Item -Path $bvhUtilsPath -Destination $bvhUtilsBak -Force
                $content = [System.IO.File]::ReadAllText($bvhUtilsPath)
                $newContent = $content -replace "SUBGROUP_SIZE\s*=\s*[0-9]+", "SUBGROUP_SIZE = $wg"
                [System.IO.File]::WriteAllText($bvhUtilsPath, $newContent)
            }

            $outWg = "$outBase.comp.wg$wg.spv"
            Compile-One -File $file.FullName -Stage "comp" -ExtraFlags @("-DLOCAL_SIZE_X=$wg") -Out $outWg

            $outWgD = "$outBase.comp.wg$wg.d.spv"
            Compile-One -File $file.FullName -Stage "comp" -ExtraFlags @("-DLOCAL_SIZE_X=$wg", "-DDEBUG_SHADERS") -Out $outWgD

            if (Test-Path $bvhUtilsBak) {
                Move-Item -Path $bvhUtilsBak -Destination $bvhUtilsPath -Force
            }
        }
    } else {
        Compile-One -File $file.FullName -Stage "comp" -ExtraFlags @() -Out "$($file.FullName).spv"
        Compile-One -File $file.FullName -Stage "comp" -ExtraFlags @("-DDEBUG_SHADERS") -Out "$outBase.comp.d.spv"
    }
}

Write-Host "`nAll shaders compiled successfully."