param (
    [string]$Arch = "x64" # Allowed: x64, x86, arm64
)

$AppName = "AetherVk"
$AppVersion = "1.0.0.0"
$Publisher = "CN=AetherVkTeam"
$DotnetArch = if ($Arch -eq "x64") { "win-x64" } else { "win-$Arch" }

$PublishDir = "aethervk.ui.app\bin\Release\net10.0\$DotnetArch\publish"
$MsixDir = "bin\publish\MsixBuild_$Arch"
$OutMsix = "bin\publish\${AppName}_${Arch}.msix"

Write-Host "=========================================="
Write-Host " Packaging Aether-Vk MSIX ($Arch)"
Write-Host "=========================================="

if (-not (Test-Path $PublishDir)) {
    Write-Error "Publish directory $PublishDir does not exist. Run package_windows.ps1 first, or run dotnet publish."
    exit 1
}

if (Test-Path $MsixDir) { Remove-Item -Recurse -Force $MsixDir }
New-Item -ItemType Directory -Path $MsixDir | Out-Null

Copy-Item -Path "$PublishDir\*" -Destination $MsixDir -Recurse

# Create AppxManifest.xml
$Manifest = @"
<?xml version="1.0" encoding="utf-8"?>
<Package xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10" 
         xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10" 
         xmlns:rescap="http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities"
         IgnorableNamespaces="uap rescap">
  <Identity Name="AetherVk" Publisher="$Publisher" Version="$AppVersion" ProcessorArchitecture="$Arch" />
  <Properties>
    <DisplayName>$AppName</DisplayName>
    <PublisherDisplayName>Aether-Vk Team</PublisherDisplayName>
    <Logo>StoreLogo.png</Logo>
  </Properties>
  <Resources>
    <Resource Language="en-us" />
  </Resources>
  <Dependencies>
    <TargetDeviceFamily Name="Windows.Desktop" MinVersion="10.0.17763.0" MaxVersionTested="10.0.19041.0" />
  </Dependencies>
  <Capabilities>
    <rescap:Capability Name="runFullTrust" />
  </Capabilities>
  <Applications>
    <Application Id="AetherVkApp" Executable="$AppName.exe" EntryPoint="Windows.FullTrustApplication">
      <uap:VisualElements DisplayName="$AppName" Description="Aether-Vk Application" BackgroundColor="transparent" Square150x150Logo="Square150x150Logo.png" Square44x44Logo="Square44x44Logo.png">
        <uap:DefaultTile Wide310x310Logo="Wide310x310Logo.png" />
        <uap:SplashScreen Image="SplashScreen.png" />
      </uap:VisualElements>
    </Application>
  </Applications>
</Package>
"@
Set-Content -Path "$MsixDir\AppxManifest.xml" -Value $Manifest -Encoding UTF8

# Copy dummy images to satisfy manifest requirements
$DummyImg = "app_broken.png"
if (Test-Path $DummyImg) {
    Copy-Item $DummyImg "$MsixDir\StoreLogo.png"
    Copy-Item $DummyImg "$MsixDir\Square150x150Logo.png"
    Copy-Item $DummyImg "$MsixDir\Square44x44Logo.png"
    Copy-Item $DummyImg "$MsixDir\Wide310x310Logo.png"
    Copy-Item $DummyImg "$MsixDir\SplashScreen.png"
}

Write-Host "Attempting to locate MakeAppx.exe..."
$MakeAppxPaths = Get-ChildItem "C:\Program Files (x86)\Windows Kits\10\bin\*\x64\makeappx.exe" -ErrorAction SilentlyContinue
if ($MakeAppxPaths) {
    $MakeAppx = ($MakeAppxPaths | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
    Write-Host "Found MakeAppx at $MakeAppx. Packing..."
    & $MakeAppx pack /d $MsixDir /p $OutMsix /o
    Write-Host "Created $OutMsix"
} else {
    Write-Warning "MakeAppx.exe not found. Please install the Windows SDK, or run 'makeappx pack /d $MsixDir /p $OutMsix' manually from a Developer Command Prompt."
}
