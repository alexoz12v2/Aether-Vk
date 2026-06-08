param (
    [string]$Arch = "win-x64"
)

$AppName = "AetherVk"
$AppVersion = "1.0.0"
$PublishDir = "aethervk.ui-app\bin\Release\net10.0\$Arch\publish"
$OutZip = "bin\publish\${AppName}_${AppVersion}_${Arch}.zip"

Write-Host "=========================================="
Write-Host " Packaging Aether-Vk for Windows ($Arch)"
Write-Host "=========================================="

Write-Host "[1/3] Publishing project..."
dotnet publish aethervk.ui-app\aethervk.ui-app.csproj -c Release -r $Arch --self-contained true

Write-Host "[2/3] Creating ZIP archive directory..."
$OutDir = Split-Path $OutZip
if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir | Out-Null
}
if (Test-Path $OutZip) {
    Remove-Item $OutZip
}

Write-Host "[3/3] Compressing files to $OutZip..."
Compress-Archive -Path "$PublishDir\*" -DestinationPath $OutZip

Write-Host "=========================================="
Write-Host " Packaging complete: $OutZip"
Write-Host "=========================================="