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

$files = Get-ChildItem -Path "assets", "assets/sim" -File | Where-Object { $_.Extension -match "\.(vert|frag|comp)$" }

foreach ($file in $files) {
    $ext = $file.Extension.TrimStart('.')
    Write-Host "Compiling $($file.Name)..."
    
    $process = Start-Process -FilePath $glslc -ArgumentList "-x", "glsl", "--target-env=vulkan1.1", "--target-spv=spv1.4", "-std=450core", "-fshader-stage=$ext", "-o", "$($file.FullName).spv", "$($file.FullName)" -Wait -NoNewWindow -PassThru
    
    if ($process.ExitCode -ne 0) {
        Write-Error "Failed to compile $($file.Name)"
        exit $process.ExitCode
    }
}

Write-Host "All shaders compiled successfully."