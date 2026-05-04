param (
    [Parameter(Mandatory=$true)]
    [ValidateSet("windows", "msix")]
    [string]$Type,

    [Parameter(Mandatory=$true)]
    [string]$Path
)

Write-Host "=========================================="
Write-Host " Validating $Type package: $Path"
Write-Host "=========================================="

if (-not (Test-Path $Path)) {
    Write-Error "❌ File not found: $Path"
    exit 1
}

Add-Type -AssemblyName System.IO.Compression.FileSystem

if ($Type -eq "windows") {
    $zip = [System.IO.Compression.ZipFile]::OpenRead($Path)
    $hasDll = $false
    foreach ($entry in $zip.Entries) {
        if ($entry.FullName -match "aethervk_core_cdylib\.dll") {
            $hasDll = $true
            break
        }
    }
    $zip.Dispose()

    if ($hasDll) {
        Write-Host "✅ Native core library (.dll) found in zip."
    } else {
        Write-Error "❌ aethervk_core_cdylib.dll missing in ZIP!"
        exit 1
    }
} elseif ($Type -eq "msix") {
    $zip = [System.IO.Compression.ZipFile]::OpenRead($Path)
    $hasDll = $false
    $hasManifest = $false
    foreach ($entry in $zip.Entries) {
        if ($entry.FullName -match "aethervk_core_cdylib\.dll") {
            $hasDll = $true
        }
        if ($entry.FullName -eq "AppxManifest.xml") {
            $hasManifest = $true
        }
    }
    $zip.Dispose()

    if ($hasDll -and $hasManifest) {
        Write-Host "✅ Native core library (.dll) and AppxManifest.xml found in MSIX."
    } else {
        Write-Error "❌ Missing required files in MSIX! DLL: $hasDll, Manifest: $hasManifest"
        exit 1
    }
}

Write-Host "🎉 Windows validation passed."