#Requires -Version 7
<#
.SYNOPSIS
  Points this shell at one of the installed Vulkan SDKs (C:\VulkanSDK\<version>).

.DESCRIPTION
  Sets VULKAN_SDK, puts its Bin first on PATH and VK_ADD_LAYER_PATH at its Bin, for this session
  only (dot-source it), so a test or the app loads that SDK's validation layer without touching the
  machine-wide install. Use it to compare validation-layer versions (e.g. the GPU-AV deadlock in
  docs/vulkan_test_loader_stall.md). Without -Version it selects the project's pinned SDK
  (.vulkan-sdk-version: the one compile_shaders.sh requires); -List lists the installed ones.

.EXAMPLE
  . scripts\windows\Use-VulkanSdk.ps1                 # the pinned SDK
  . scripts\windows\Use-VulkanSdk.ps1 -List
  . scripts\windows\Use-VulkanSdk.ps1 -Version latest
  . scripts\windows\Use-VulkanSdk.ps1 -Version 1.4.321.0
#>
param([string] $Version, [switch] $List)
$ErrorActionPreference = 'Stop'
$pinned = (Get-Content (Join-Path $PSScriptRoot '..\..\.vulkan-sdk-version') -Raw).Trim()

$root = 'C:\VulkanSDK'
$sdks = Get-ChildItem $root -Directory -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -match '^\d+(\.\d+)+$' } |
  Sort-Object { [version]$_.Name }
if (-not $sdks) { throw "No Vulkan SDK under $root (winget install KhronosGroup.VulkanSDK --version $pinned --force)" }

if ($List) {
  $sdks | ForEach-Object { '{0} {1}{2}' -f $(if ($_.FullName -eq $env:VULKAN_SDK) { '*' } else { ' ' }), $_.Name, $(if ($_.Name -eq $pinned) { '  (pinned)' }) }
  return
}

if (-not $Version) { $Version = $pinned }
$sdk = if ($Version -eq 'latest') { $sdks[-1] } else { $sdks | Where-Object Name -eq $Version }
if (-not $sdk) {
  $hint = if ($Version -eq $pinned) { " (the project's pinned SDK: winget install KhronosGroup.VulkanSDK --version $pinned --force)" }
  throw "SDK $Version not found$hint; installed: $($sdks.Name -join ', ')"
}

$bin = Join-Path $sdk.FullName 'Bin'
# drop any other SDK's Bin from PATH, then put this one first
$path = ($env:PATH -split ';' | Where-Object { $_ -and $_ -notlike "$root\*" }) -join ';'
$env:PATH = "$bin;$path"
$env:VULKAN_SDK = $sdk.FullName
# the layer manifests the engine enables by name (VK_LAYER_KHRONOS_validation ...) come from here
# before the registry's explicit-layer list
$env:VK_ADD_LAYER_PATH = $bin
"VULKAN_SDK=$($sdk.FullName) (this session)$(if ($sdk.Name -ne $pinned) { "; the project pins ${pinned}: do not commit shaders built with this one" })"
