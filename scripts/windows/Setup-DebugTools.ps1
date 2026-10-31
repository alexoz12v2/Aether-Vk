#Requires -Version 7
<#
.SYNOPSIS
  One-time setup and status of the Windows diagnostics tooling (cdb, symbols, crash dumps, ETW).

.DESCRIPTION
  Default (no switch): makes cdb usable and fast, then prints the status.
    - copies the WinDbg Store package's amd64 folder to %LOCALAPPDATA%\dbg\amd64 when cdb.exe is
      not there yet (the copy inside WindowsApps cannot be executed);
    - writes symsrv.ini next to cdb with the symbol-server exclusion list (vendor GPU drivers,
      Vulkan layers, our own PDBs): without it every NVIDIA ICD frame costs a server timeout;
    - creates the symbol cache (C:\symbols, or $env:AVK_SYMBOL_CACHE).
  -EnableCrashDumps / -DisableCrashDumps (elevated): Windows Error Reporting LocalDumps, so any
    process that crashes (access violation, Rust abort / __fastfail, stack overflow) leaves a dump in
    %LOCALAPPDATA%\CrashDumps without a debugger attached. The key is machine-wide (the test exes'
    names change with every build, so a per-exe key cannot follow them).
  -Status: status only, changes nothing. Also lists the Vulkan implicit layers registered on the
    machine (they are injected into every Vulkan process: OBS, Steam, Nsight, RenderDoc ...) and
    flags the ones whose manifest file is gone.
  -RemoveStaleLayers (elevated): deletes the implicit/explicit layer registry values whose manifest is gone
    (each one costs a loader error line in every instance creation).

.EXAMPLE
  scripts\windows\Setup-DebugTools.ps1
  scripts\windows\Setup-DebugTools.ps1 -Status
  scripts\windows\Setup-DebugTools.ps1 -EnableCrashDumps    # from an elevated shell
#>
param(
  [switch] $Status,
  [switch] $EnableCrashDumps,
  [switch] $DisableCrashDumps,
  [switch] $RemoveStaleLayers,
  # LocalDumps DumpType: Mini (stacks, modules; a few MB) or Full (whole memory)
  [ValidateSet('Mini', 'Full')] [string] $DumpType = 'Full',
  [int] $DumpCount = 10
)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'AvkDebug.psm1') -Force

$dbgDir = Join-Path $env:LOCALAPPDATA 'dbg\amd64'
$wer = 'HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps'
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
  [Security.Principal.WindowsBuiltInRole]::Administrator)

function Copy-CdbFromStore {
  # Get-AppxPackage does not load in pwsh 7 on every build; Windows PowerShell has it
  $loc = (& powershell.exe -NoProfile -Command "(Get-AppxPackage -Name Microsoft.WinDbg | Select-Object -First 1).InstallLocation" 2>$null)
  if (-not $loc) { throw "WinDbg is not installed. Install it with: winget install Microsoft.WinDbg  (then re-run this script)" }
  $src = Join-Path $loc.Trim() 'amd64'
  Write-Host "Copying $src -> $dbgDir"
  $null = New-Item -ItemType Directory -Force -Path $dbgDir
  Copy-Item -Path (Join-Path $src '*') -Destination $dbgDir -Recurse -Force
}

if ($EnableCrashDumps -or $DisableCrashDumps) {
  if (-not $isAdmin) {
    throw "LocalDumps lives under HKLM: run this from an elevated shell, e.g. Start-Process pwsh -Verb RunAs -ArgumentList '-NoExit','-File','$PSCommandPath','-EnableCrashDumps'"
  }
  if ($EnableCrashDumps) {
    $null = New-Item -Path $wer -Force
    # REG_EXPAND_SZ: WerFault expands it for the crashing user
    New-ItemProperty -Path $wer -Name DumpFolder -PropertyType ExpandString -Value '%LOCALAPPDATA%\CrashDumps' -Force | Out-Null
    New-ItemProperty -Path $wer -Name DumpCount -PropertyType DWord -Value $DumpCount -Force | Out-Null
    New-ItemProperty -Path $wer -Name DumpType -PropertyType DWord -Value ($(if ($DumpType -eq 'Full') { 2 } else { 1 })) -Force | Out-Null
    Write-Host "LocalDumps on: $DumpType dumps (last $DumpCount) in %LOCALAPPDATA%\CrashDumps"
  } else {
    if (Test-Path $wer) { Remove-Item -Path $wer -Recurse -Force }
    Write-Host 'LocalDumps off (key removed).'
  }
}

# Vulkan layers: HKLM / HKCU ...\Khronos\Vulkan\{Implicit,Explicit}Layers, value name = manifest
# path (the implicit ones are injected into every Vulkan process; a stale explicit one only costs a
# loader error line per instance)
$layerKeys = foreach ($hive in 'HKLM:', 'HKCU:') { foreach ($kind in 'ImplicitLayers', 'ExplicitLayers') { "$hive\SOFTWARE\Khronos\Vulkan\$kind" } }
function Get-ImplicitLayers {
  foreach ($k in $layerKeys) {
    $item = Get-Item -Path $k -ErrorAction SilentlyContinue
    if (-not $item) { continue }
    foreach ($manifest in $item.GetValueNames()) {
      $name = '?'
      if (Test-Path -LiteralPath $manifest) {
        try { $name = (Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json).layer.name } catch { $name = '(unreadable manifest)' }
      }
      $kind = if ($k -like '*Implicit*') { 'implicit' } else { 'explicit' }
      [pscustomobject]@{ Key = $k; Kind = $kind; Manifest = $manifest; Layer = $name; Stale = -not (Test-Path -LiteralPath $manifest) }
    }
  }
}

if ($RemoveStaleLayers) {
  if (-not $isAdmin) { throw "The HKLM layer list needs an elevated shell (Start-Process pwsh -Verb RunAs ...)" }
  foreach ($l in Get-ImplicitLayers | Where-Object Stale) {
    Remove-ItemProperty -Path $l.Key -Name $l.Manifest
    Write-Host "Removed stale implicit layer $($l.Manifest)"
  }
}

if (-not $Status -and -not $EnableCrashDumps -and -not $DisableCrashDumps -and -not $RemoveStaleLayers) {
  if (-not (Test-Path (Join-Path $dbgDir 'cdb.exe'))) { Copy-CdbFromStore }
  $ini = Join-Path (Split-Path (Get-AvkCdb)) 'symsrv.ini'
  $body = @('; Written by scripts\windows\Setup-DebugTools.ps1: modules the Microsoft symbol server', '; has no PDB for (asking for them only costs a timeout). Local symbol-path folders are still searched.', '[exclusions]') + $SymsrvExclusions
  Set-Content -LiteralPath $ini -Value $body -Encoding ascii
  Write-Host "Wrote $ini ($($SymsrvExclusions.Count) exclusions)"
  $cache = if ($env:AVK_SYMBOL_CACHE) { $env:AVK_SYMBOL_CACHE } else { 'C:\symbols' }
  $null = New-Item -ItemType Directory -Force -Path $cache
}

# ── status ──
$cdb = try { Get-AvkCdb } catch { $null }
$wpt = "${env:ProgramFiles(x86)}\Windows Kits\10\Windows Performance Toolkit"
$ld = Get-ItemProperty -Path $wer -ErrorAction SilentlyContinue
$rows = [ordered]@{
  'cdb.exe'               = $(if ($cdb) { $cdb } else { 'MISSING (winget install Microsoft.WinDbg, then run this script)' })
  'symsrv.ini exclusions' = $(if ($cdb -and (Test-Path (Join-Path (Split-Path $cdb) 'symsrv.ini'))) { 'present' } else { 'missing' })
  'symbol path'           = $(if ($cdb) { Get-AvkSymbolPath } else { '-' })
  'WER LocalDumps'        = $(if ($ld) { "on: type $($ld.DumpType) (1 mini, 2 full), count $($ld.DumpCount), $($ld.DumpFolder)" } else { 'off (-EnableCrashDumps, elevated)' })
  'wpr.exe'               = $(if (Get-Command wpr.exe -ErrorAction SilentlyContinue) { (Get-Command wpr.exe).Source } else { 'missing' })
  'xperf / WPA'           = $(if (Test-Path "$wpt\xperf.exe") { $wpt } else { 'missing (Windows ADK: Windows Performance Toolkit)' })
  'samply'                = $(if (Get-Command samply -ErrorAction SilentlyContinue) { (& samply --version) } else { 'missing (cargo install --locked samply)' })
  'procdump (Sysinternals)' = $(if (Get-Command procdump.exe -ErrorAction SilentlyContinue) { (Get-Command procdump.exe).Source } else { 'missing (winget install Microsoft.Sysinternals.Suite)' })
  'renderdoccmd'          = $(if (Test-Path "$env:ProgramFiles\RenderDoc\renderdoccmd.exe") { "$env:ProgramFiles\RenderDoc\renderdoccmd.exe" } else { 'missing (https://renderdoc.org)' })
  'Nsight Graphics'       = $((Get-ChildItem "$env:ProgramFiles\NVIDIA Corporation" -Directory -Filter 'Nsight Graphics*' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name) -join ', ')
  'Vulkan SDKs'           = $((Get-ChildItem 'C:\VulkanSDK' -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name) -join ', ') + "  (VULKAN_SDK=$env:VULKAN_SDK)"
  'dotnet diag tools'     = 'dotnet tool restore (dotnet-trace, dotnet-counters, dotnet-dump, dotnet-stack in .config/dotnet-tools.json)'
  'elevated shell'        = $isAdmin
}
$rows.GetEnumerator() | ForEach-Object { '{0,-24} {1}' -f $_.Key, $_.Value }

''
'Vulkan layers registered (implicit ones are injected into every Vulkan process; nextest disables them, .config/nextest.toml):'
Get-ImplicitLayers | Where-Object { $_.Kind -eq 'implicit' -or $_.Stale } | ForEach-Object {
  '  {0,-6} {1,-9} {2,-40} {3}' -f $(if ($_.Stale) { 'STALE' } else { '' }), $_.Kind, $_.Layer, $_.Manifest
}
if (Get-ImplicitLayers | Where-Object Stale) { '  -> remove the STALE entries with -RemoveStaleLayers (elevated)' }
