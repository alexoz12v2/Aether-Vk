#Requires -Version 7
<#
.SYNOPSIS
  CPU profile of a process or a command with samply (ETW sampling; the Firefox Profiler UI, the same
  as `samply` / `perf` on Linux). The Windows counterpart of `perf record -g`.

.DESCRIPTION
  samply's ETW sampling needs an elevated shell. The profile is saved under
  target\diagnostics\profiles\ and opened in the browser unless -NoOpen (later: samply load <file>).
  For readable Rust frames in optimised code, build with the `profiling` cargo profile
  (release + line tables, not stripped: `cargo build --profile profiling ...`); debug builds
  symbolise as they are.
  -Wpr records a full ETW trace with WPR instead (CPU + context switches: who waits on whom), for
  WPA.

.EXAMPLE
  scripts\windows\Record-Cpu.ps1 -Id 1234 -Seconds 15
  scripts\windows\Record-Cpu.ps1 -- target\profiling\examples\dust_observer.exe --frames 600
  scripts\windows\Record-Cpu.ps1 -Wpr -Seconds 20       # then: wpa <etl>
#>
param(
  [int] $Id,
  [int] $Seconds = 20,
  [int] $RateHz = 1000,
  [switch] $Wpr,
  [switch] $NoOpen,
  [Parameter(ValueFromRemainingArguments = $true)] [string[]] $Command
)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'AvkDebug.psm1') -Force

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
  [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
  throw "ETW sampling needs an elevated shell: Start-Process pwsh -Verb RunAs -ArgumentList '-NoExit','-File','$PSCommandPath', ..."
}

$dir = Join-Path (Get-AvkRepoRoot) 'target\diagnostics\profiles'
$null = New-Item -ItemType Directory -Force -Path $dir
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
if ($Command -and $Command[0] -eq '--') { $Command = $Command[1..($Command.Count - 1)] }

if ($Wpr) {
  $etl = Join-Path $dir "cpu-$stamp.etl"
  & wpr -start CPU -filemode
  try {
    if ($Command) { & $Command[0] @($Command | Select-Object -Skip 1) } else { Start-Sleep -Seconds $Seconds }
  } finally {
    & wpr -stop $etl
  }
  "ETW trace: $etl  (open with wpa; Trace > Configure Symbol Paths: add the exe folder)"
  return
}

if (-not (Get-Command samply -ErrorAction SilentlyContinue)) { throw 'samply missing: cargo install --locked samply' }
$out = Join-Path $dir "cpu-$stamp.json.gz"
$sargs = @('record', '--rate', $RateHz, '--save-only', '-o', $out)
if ($Id) {
  $sargs += @('-p', $Id, '--duration', $Seconds)
} elseif ($Command) {
  $sargs += @('--') + $Command
} else {
  throw 'Give -Id <pid> or a command after --'
}
& samply @sargs
"Profile: $out"
if (-not $NoOpen) { & samply load $out }
