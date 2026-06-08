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

Write-Host "Download complete!" -ForegroundColor Green
