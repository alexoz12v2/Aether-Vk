# Get the directory where the script is located
$ScriptDir = $PSScriptRoot
# Fallback to current directory if run directly from a console without saving
if ([string]::IsNullOrEmpty($ScriptDir)) { $ScriptDir = (Get-Location).Path }

$TargetDir = Join-Path -Path $ScriptDir -ChildPath "assets\planets"

# Create the target directory if it doesn't exist
if (!(Test-Path -Path $TargetDir)) {
    New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
}

Write-Host "Downloading SPICE kernels to $TargetDir..." -ForegroundColor Cyan

# Base URLs
$PlanetsUrl = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets"
$SatellitesUrl = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites"

# Download planet kernels
Write-Host "Downloading de442.bsp..."
Invoke-WebRequest -Uri "$PlanetsUrl/de442.bsp" -OutFile (Join-Path -Path $TargetDir -ChildPath "de442.bsp")

# Download satellite kernels
$Satellites = @(
    "jup365.bsp",
    "mar099.bsp",
    "nep105.bsp",
    "plu060.bsp",
    "sat457.bsp"
)

foreach ($file in $Satellites) {
    Write-Host "Downloading $file..."
    Invoke-WebRequest -Uri "$SatellitesUrl/$file" -OutFile (Join-Path -Path $TargetDir -ChildPath $file)
}

Write-Host "Download complete!" -ForegroundColor Green
