param (
    [string]$Arch = "win-x64"
)

$AppName = "AetherVk"
$AppVersion = "1.0.1"
$PublishDir = "aethervk.ui-app\bin\Release\net10.0\$Arch\publish"
$OutZip = "bin\publish\${AppName}_${AppVersion}_${Arch}.zip"

Write-Host "=========================================="
Write-Host " Packaging Aether-Vk for Windows ($Arch)"
Write-Host "=========================================="

$CargoTarget = switch ($Arch) {
    "win-x64" { "x86_64-pc-windows-msvc" }
    "win-arm64" { "aarch64-pc-windows-msvc" }
    default { Write-Error "Unsupported architecture: $Arch"; exit 1 }
}

# Extra cargo flags for the native engine, empty by default. AETHERVK_CARGO_FLAGS=--no-default-features
# builds without CFITSIO (pure-Rust FITS reader) when vcpkg cannot build it for a target.
$ExtraCargoFlags = @()
if ($env:AETHERVK_CARGO_FLAGS) {
    $ExtraCargoFlags = @($env:AETHERVK_CARGO_FLAGS -split '\s+' | Where-Object { $_ -ne "" })
}

Write-Host "[1/4] Building the native engine ($CargoTarget) $($env:AETHERVK_CARGO_FLAGS)..."
cargo build --locked -p aethervk-core-cdylib --profile release --target $CargoTarget @ExtraCargoFlags
if ($LASTEXITCODE -ne 0) {
    Write-Error "cargo build failed ($LASTEXITCODE)"
    exit $LASTEXITCODE
}
$NativeLib = "target\$CargoTarget\release\aethervk_core_cdylib.dll"
if (-not (Test-Path $NativeLib)) {
    Write-Error "Native library missing: $NativeLib"
    exit 1
}

Write-Host "[2/4] Publishing project..."
# the engine is built above (with AETHERVK_CARGO_FLAGS), so ui-logic only copies it: a cargo build of
# its own would use the default features. ui-logic (being netstandard) does not receive -r: pass the
# target RID explicitly
dotnet publish aethervk.ui-app\aethervk.ui-app.csproj -c Release -r $Arch --self-contained true -p:SkipNativeBuild=true -p:ResolvedRID=$Arch
if ($LASTEXITCODE -ne 0) {
    Write-Error "dotnet publish failed ($LASTEXITCODE)"
    exit $LASTEXITCODE
}

# licence texts: the project's, and those of the static CFITSIO + zlib linked into the engine
Copy-Item LICENSE "$PublishDir\LICENSE" -Force
Copy-Item aethervk.cfitsio\sys\THIRD_PARTY_LICENSES.md "$PublishDir\THIRD_PARTY_LICENSES.md" -Force

Write-Host "[3/4] Creating ZIP archive directory..."
$OutDir = Split-Path $OutZip
if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir | Out-Null
}
if (Test-Path $OutZip) {
    Remove-Item $OutZip
}

Write-Host "[4/4] Compressing files to $OutZip..."
Compress-Archive -Path "$PublishDir\*" -DestinationPath $OutZip

Write-Host "=========================================="
Write-Host " Packaging complete: $OutZip"
Write-Host "=========================================="