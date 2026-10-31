#Requires -Version 7
<#
.SYNOPSIS
  Every thread's stack of a live (typically hung) process, the Windows counterpart of
  `gdb --pid <pid> --batch -ex "thread apply all bt"`; optionally a minidump for later.

.DESCRIPTION
  Attaches cdb non-invasively (-pv): the threads are suspended while the stacks are read and
  resumed when cdb detaches, so a test run can carry on (or be killed by nextest) afterwards.
  The report (thread summary, `~*kn`, `!runaway`, `!locks`) goes to
  target\diagnostics\stacks\<label>-<name>-<pid>-<time>.txt and its summary is printed.

.EXAMPLE
  scripts\windows\Get-HungStacks.ps1 -Name aethervk_core_rlib
  scripts\windows\Get-HungStacks.ps1 -Id 1234 -Dump Full
  Open a dump later: windbgx -z <dmp>   (or: cdb -z <dmp> -c "~*kn; q")
#>
[CmdletBinding(DefaultParameterSetName = 'Name')]
param(
  [Parameter(ParameterSetName = 'Id', Mandatory, Position = 0)] [int[]] $Id,
  # Process name (wildcards allowed, no .exe), e.g. aethervk_core_rlib*, AetherVk
  [Parameter(ParameterSetName = 'Name', Position = 0)] [string] $Name = 'aethervk*',
  # None: stacks only. Mini: small dump (threads, handles, stacks). Full: .dump /ma (whole memory)
  [ValidateSet('None', 'Mini', 'Full')] [string] $Dump = 'None',
  [string] $OutDir,
  [int] $Frames = 80,
  [int] $TimeoutSec = 120,
  # Also the managed (.NET) stacks, for the AetherVk app or a dotnet test host
  [switch] $Managed
)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'AvkDebug.psm1') -Force
if (-not $OutDir) { $OutDir = Join-Path (Get-AvkRepoRoot) 'target\diagnostics\stacks' }

if ($PSCmdlet.ParameterSetName -eq 'Name') {
  $Id = @(Get-Process -Name $Name -ErrorAction SilentlyContinue | ForEach-Object Id)
  if (-not $Id) { throw "No process matches '$Name'." }
}
foreach ($p in $Id) {
  $report = Save-AvkStacks -Id $p -OutDir $OutDir -Dump $Dump -Frames $Frames -TimeoutSec $TimeoutSec -Label 'stacks' -Managed:$Managed
  Write-Host "== $report" -ForegroundColor Cyan
  $lines = @(Get-Content -LiteralPath $report)
  $end = [array]::IndexOf($lines, '== full cdb output')
  if ($end -lt 0) { $end = $lines.Count }
  $lines[1..($end - 1)]
}
