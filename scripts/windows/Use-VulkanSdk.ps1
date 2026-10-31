#Requires -Version 7
<#
.SYNOPSIS
  Points this shell at one of the installed Vulkan SDKs (C:\VulkanSDK\<version>).

.DESCRIPTION
  Sets VULKAN_SDK, puts its Bin first on PATH and VK_ADD_LAYER_PATH at its Bin, for this session
  only (dot-source it), so a test or the app loads that SDK's validation layer without touching the
  machine-wide install. Use it to compare validation-layer versions (e.g. the GPU-AV deadlock in
  docs/vulkan_test_loader_stall.md). Without -Version it lists the installed SDKs.

.EXAMPLE
  . scripts\windows\Use-VulkanSdk.ps1                 # list
  . scripts\windows\Use-VulkanSdk.ps1 -Version latest
  . scripts\windows\Use-VulkanSdk.ps1 -Version 1.4.321.0
#>
param([string] $Version)
$ErrorActionPreference = 'Stop'

$root = 'C:\VulkanSDK'
$sdks = Get-ChildItem $root -Directory -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -match '^\d+(\.\d+)+$' } |
  Sort-Object { [version]$_.Name }
if (-not $sdks) { throw "No Vulkan SDK under $root (winget install KhronosGroup.VulkanSDK)" }

if (-not $Version) {
  $sdks | ForEach-Object { '{0} {1}' -f $(if ($_.FullName -eq $env:VULKAN_SDK) { '*' } else { ' ' }), $_.Name }
  return
}

$sdk = if ($Version -eq 'latest') { $sdks[-1] } else { $sdks | Where-Object Name -eq $Version }
if (-not $sdk) { throw "SDK $Version not found; installed: $($sdks.Name -join ', ')" }

$bin = Join-Path $sdk.FullName 'Bin'
# drop any other SDK's Bin from PATH, then put this one first
$path = ($env:PATH -split ';' | Where-Object { $_ -and $_ -notlike "$root\*" }) -join ';'
$env:PATH = "$bin;$path"
$env:VULKAN_SDK = $sdk.FullName
# the layer manifests the engine enables by name (VK_LAYER_KHRONOS_validation ...) come from here
# before the registry's explicit-layer list
$env:VK_ADD_LAYER_PATH = $bin
"VULKAN_SDK=$($sdk.FullName) (this session)"
