#Requires -Version 7
<#
.SYNOPSIS
  Application Verifier heap / handle / lock checks for one executable image (the Windows counterpart
  of running under ASan + valgrind's handle tracking on Linux).

.DESCRIPTION
  Application Verifier is per image *name*: once enabled, every process started from an exe with that
  file name is verified, until -Disable. The settings live under
  HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Image File Execution Options\<image>, so -Enable
  and -Disable need an elevated shell; -Status reads the key and works unelevated.

  Checks (-Checks, default below):
    Heaps     full page heap: a guard page after every allocation, so an overrun or a use after free
              faults on the spot (-Light: normal page heap, fill patterns checked on free; use it for
              GPU tests, where the full page heap makes the driver's allocations very slow)
    Handles   invalid / closed / double-closed handles
    Locks     critical sections: double init, release by a non-owner, freed while held
    SRWLock   the same for slim reader-writer locks (std's Mutex / RwLock on Windows)
    Memory    VirtualAlloc / VirtualFree / MapViewOfFile misuse
    TLS       TlsAlloc / TlsFree misuse
    Threadpool
  Leak and Exceptions are left out: the Rust test harness's statics and first-chance exceptions in
  the drivers report noise.

  A verifier stop breaks into the debugger. Without one, the break is an unhandled exception: the
  process dies with the stop in the event log and the app's crash handler writes
  %LOCALAPPDATA%\Aether-Vk\logs\aethervk-crash-<pid>.dmp. Under cdb, the stop shows directly:
    scripts\windows\Invoke-UnderDebugger.ps1 -- <exe> <args>     # then in the report: !avrf
  The binary log (%USERPROFILE%\AppVerifierLogs\<image>.*.dat) exports to XML with -Export.

  Test executables: the image name carries cargo's hash (aethervk_core_rlib-<hash>.exe) and stays the
  same across rebuilds of the same configuration. -Test picks the newest aethervk_core_rlib test exe.

.EXAMPLE
  scripts\windows\Enable-AppVerifier.ps1 -Status
  scripts\windows\Enable-AppVerifier.ps1 -Enable -Test             # elevated
  scripts\windows\Enable-AppVerifier.ps1 -Enable -Test -Light      # GPU tests
  scripts\windows\Enable-AppVerifier.ps1 -Enable -Image AetherVk.exe -Checks Handles,Locks
  scripts\windows\Enable-AppVerifier.ps1 -Export -Test             # the log to target\diagnostics\appverifier
  scripts\windows\Enable-AppVerifier.ps1 -Disable -Test            # elevated
#>
param(
  [switch] $Enable,
  [switch] $Disable,
  [switch] $Status,
  [switch] $Export,
  [string] $Image,
  [switch] $Test,
  [string[]] $Checks = @('Heaps', 'Handles', 'Locks', 'SRWLock', 'Memory', 'TLS', 'Threadpool'),
  [switch] $Light
)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'AvkDebug.psm1') -Force

if ($Test) {
  $exe = Get-ChildItem (Join-Path (Get-AvkRepoRoot) 'target\debug\build\aethervk-core-rlib\*\out\aethervk_core_rlib-*.exe') -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if (-not $exe) { throw 'No aethervk_core_rlib test exe under target\debug\build: cargo test -p aethervk-core-rlib --lib --no-run' }
  $Image = $exe.Name
}
if ($Image) { $Image = Split-Path -Leaf $Image }

$ifeo = 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Image File Execution Options'
# FLG_APPLICATION_VERIFIER in the image's GlobalFlag
$verifierFlag = 0x100

function Show-Status {
  $keys = Get-ChildItem $ifeo -ErrorAction SilentlyContinue | Where-Object {
    $g = (Get-ItemProperty $_.PSPath -Name GlobalFlag -ErrorAction SilentlyContinue).GlobalFlag
    $g -and (([Convert]::ToInt64("$g".Replace('0x', ''), $(if ("$g" -like '0x*') { 16 } else { 10 })) -band $verifierFlag) -ne 0)
  }
  if ($Image) { $keys = $keys | Where-Object PSChildName -eq $Image }
  if (-not $keys) {
    "Application Verifier: no image enabled$(if ($Image) { " (checked $Image)" })"
    return
  }
  foreach ($k in $keys) {
    $p = Get-ItemProperty $k.PSPath
    $dlls = if ($p.PSObject.Properties['VerifierDlls']) { $p.VerifierDlls } else { '' }
    $layers = Get-ChildItem $k.PSPath -ErrorAction SilentlyContinue | Where-Object PSChildName -like '{ApplicationVerifier*' |
      ForEach-Object { Get-ChildItem $_.PSPath -ErrorAction SilentlyContinue } | ForEach-Object PSChildName
    "{0,-48} enabled  layers: {1}  {2}" -f $k.PSChildName, ($(if ($layers) { $layers -join ',' } else { '(see appverif UI)' })), $(if ($dlls) { "dlls: $dlls" })
  }
  $logs = Join-Path $env:USERPROFILE 'AppVerifierLogs'
  if (Test-Path $logs) { "logs: $logs" }
}

function Assert-Elevated {
  $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
  if (-not $isAdmin) {
    throw "appverif needs an elevated shell: Start-Process pwsh -Verb RunAs -ArgumentList '-NoExit','-File','$PSCommandPath', ..."
  }
}

if ($Enable) {
  if (-not $Image) { throw 'Give -Image <exe name> or -Test' }
  Assert-Elevated
  $with = @()
  if ($Light -and $Checks -contains 'Heaps') { $with = @('-with', 'Heaps.Full=FALSE') }
  & appverif.exe -enable @Checks -for $Image @with
  if ($LASTEXITCODE -ne 0) { throw "appverif -enable failed ($LASTEXITCODE)" }
  "Enabled for ${Image}: $($Checks -join ', ')$(if ($Light) { ' (normal page heap)' })"
  Show-Status
} elseif ($Disable) {
  if (-not $Image) { throw 'Give -Image <exe name> or -Test' }
  Assert-Elevated
  & appverif.exe -disable * -for $Image
  if ($LASTEXITCODE -ne 0) { throw "appverif -disable failed ($LASTEXITCODE)" }
  "Disabled for $Image"
} elseif ($Export) {
  if (-not $Image) { throw 'Give -Image <exe name> or -Test' }
  Assert-Elevated
  $dir = Join-Path (Get-AvkRepoRoot) 'target\diagnostics\appverifier'
  $null = New-Item -ItemType Directory -Force -Path $dir
  $out = Join-Path $dir "$Image-$(Get-Date -Format 'yyyyMMdd-HHmmss').xml"
  & appverif.exe -export log -for $Image -with "To=$out"
  if ($LASTEXITCODE -ne 0) { throw "appverif -export failed ($LASTEXITCODE)" }
  "Log: $out"
} else {
  Show-Status
}
