# tested on microsoft.windowsappsdk 1.8.2.251105000
param (
    [ValidateScript({ Test-Path -Path $_ -PathType Leaf })]
    [Parameter(Mandatory = $true, ValueFromPipeline = $false)]
    [String]$cppwinrtExe,
    [ValidateScript({
        # If the path does not exist → OK
        if (-not (Test-Path $_))
        {
            return $true
        }
        if (Test-Path -Path $_ -PathType Leaf)
        {
            throw "Path '$_' is a file"
        }
        # If it exists, ensure it is empty
        if ((Get-ChildItem -LiteralPath $_ -Force -ErrorAction Stop | Measure-Object).Count -eq 0)
        {
            return $true
        }
        throw "Path '$path' exists but is not empty."
    })]
    [Parameter(Mandatory = $true, ValueFromPipeline = $false)]
    [String]$destinationPath,
    [Parameter(Mandatory = $false)]
    [ValidateSet('x86', 'x64', 'arm', 'arm64')]
    [string]$targetArch,
    [Parameter(Mandatory = $false)]
    [ValidateSet('x86', 'x64', 'arm', 'arm64')]
    [string]$hostArch
)

$root = Join-Path $env:USERPROFILE '.nuget\packages'
$winuiPkg = "$root\microsoft.windowsappsdk.winui"
Write-Host "Checking $winuiPkg for WinUI 3 installation" -ForegroundColor Green
if (Test-Path $winuiPkg)
{
    $winuiDir = Get-ChildItem $winuiPkg | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $version = $winuiDir.Name
    Write-Host "WinUI 3 detected. Version: $version" -ForegroundColor Green
}
else
{
    throw "WinUI 3 not installed (nuget)"
}

# 1. Collect only WinUI-related package families
$packageRoots = Get-ChildItem $root -Directory |
        Where-Object { $_.Name -like "microsoft.windowsappsdk.*" }

if (-not $packageRoots)
{
    throw "No WindowsAppSDK packages found in $root"
}

# 2. Extract version directories for each package
$packageVersions = foreach ($pkg in $packageRoots)
{
    [PSCustomObject]@{
        Package = $pkg
        Versions = Get-ChildItem $pkg.FullName -Directory |
                Where-Object { $_.Name -match '^\d+\.\d+\.' } |
                Select-Object -ExpandProperty Name
    }
}

# 3. Compute the newest *major.minor* that exists in **all** packages
#    Extract major.minor from 1.8.251104000 → 1.8
$majorMinorSets = $packageVersions.Versions |
        ForEach-Object { ($_ -split '\.')[0, 1] -join '.' }

# Group by major.minor and count consistency
$grouped = $majorMinorSets | Group-Object

# Keep only major.minor values present in *every* package
$validMajorMinor = $grouped |
        Where-Object { $_.Count -eq $packageVersions.Count } |
        Select-Object -ExpandProperty Name

if (-not $validMajorMinor)
{
    throw "No consistent major.minor version exists across all packages."
}

# Pick newest major.minor
$selectedMajorMinor = $validMajorMinor |
        Sort-Object { [version]$_ } -Descending |
        Select-Object -First 1

Write-Host "Using WindowsApp SDK major.minor: $selectedMajorMinor"

# 4. For each package, pick the newest version that starts with major.minor
$resolvedPackages = foreach ($pkg in $packageVersions)
{
    $matching = $pkg.Versions |
            Where-Object { $_ -like "$selectedMajorMinor.*" }

    if (-not $matching)
    {
        throw "Package '$( $pkg.Package.Name )' does not contain a version matching $selectedMajorMinor.*"
    }

    $chosen = $matching | Sort-Object { [version]$_ } -Descending | Select-Object -First 1

    [PSCustomObject]@{
        Package = $pkg.Package
        Version = $chosen
        FullPath = Join-Path $pkg.Package.FullName $chosen
    }
}

# 5. Collect *.winmd for correct architecture + metadata folder
$pattern = "win-$TargetArch"
$results = foreach ($pkg in $resolvedPackages)
{

    # metadata folder (always architecture-independent)
    $metadataPath = Join-Path $pkg.FullPath 'metadata'
    if (Test-Path $metadataPath)
    {
        Get-ChildItem $metadataPath -Filter *.winmd -Recurse
    }

    # runtimes for architecture
    $runtimePath = Join-Path $pkg.FullPath "runtimes-framework\$pattern\native"
    if (Test-Path $runtimePath)
    {
        Get-ChildItem $runtimePath -Filter *.winmd -Recurse
    }
}

if (-not $results)
{
    throw "No .winmd files found for architecture '$targetArch' in version '$selectedMajorMinor.*'."
}

$inputs = $results.FullName | ForEach-Object { "-input", $_ }

if (-not (Test-Path -Path "$destinationPath"))
{
    New-Item -ItemType Directory -Path "$destinationPath" | Out-Null
}

Write-Host "$cppwinrtExe" -input sdk+ $inputs -output "$destinationPath" -ForegroundColor Green
& "$cppwinrtExe" -input sdk+ $inputs -output "$destinationPath"

return 0 | Out-Null
