#Requires -Version 7
<#
.SYNOPSIS
  Poor man's sampling profiler: N stack snapshots of a running process via cdb, aggregated into
  the hottest leaf functions (self) and functions on stack (inclusive), plus CPU per thread. A
  quick, no-setup stand-in for `perf top -p <pid>` / `perf record -g`.

.DESCRIPTION
  Each sample is one short non-invasive attach (the threads stop for well under a second). Threads
  sitting in a kernel wait are left out unless -IncludeWaiting, so the tables show where CPU goes.
  Per-thread CPU over the whole window comes from the OS thread times (no debugger), which answers
  the first question on its own: is one thread at 100 % while the others idle?

  Resolution is coarse (a sample every ~0.5-3 s depending on PDB size). For a real profile use
  ETW: wpr -start CPU ... wpr -stop out.etl, then open it in WPA (see docs/windows_debugging.md).

.EXAMPLE
  scripts\windows\Get-StackSamples.ps1 -Name dust_observer -Samples 30
  scripts\windows\Get-StackSamples.ps1 -Id 1234 -Samples 10 -IntervalMs 1000 -Top 40
#>
[CmdletBinding(DefaultParameterSetName = 'Name')]
param(
  [Parameter(ParameterSetName = 'Id', Mandatory, Position = 0)] [int] $Id,
  [Parameter(ParameterSetName = 'Name', Mandatory, Position = 0)] [string] $Name,
  [int] $Samples = 20,
  [int] $IntervalMs = 500,
  [int] $Frames = 60,
  [int] $Top = 25,
  [switch] $IncludeWaiting,
  [string] $OutDir
)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'AvkDebug.psm1') -Force
if (-not $OutDir) { $OutDir = Join-Path (Get-AvkRepoRoot) 'target\diagnostics\samples' }
if ($PSCmdlet.ParameterSetName -eq 'Name') {
  $procs = @(Get-Process -Name $Name -ErrorAction Stop)
  if ($procs.Count -gt 1) { throw "Several processes match '$Name' ($($procs.Id -join ', ')): pass -Id." }
  $Id = $procs[0].Id
}
$proc = Get-Process -Id $Id
$exe = try { $proc.Path } catch { $null }
$sym = Get-AvkSymbolPath -Extra $(if ($exe) { @(Split-Path $exe) } else { @() })
$null = Test-AvkSymsrvExclusions

function Get-ThreadTimes { $h = @{}; foreach ($t in (Get-Process -Id $Id).Threads) { try { $h[$t.Id] = $t.TotalProcessorTime.TotalSeconds } catch {} }; $h }

$waitRegex = '!(Nt|Zw)(WaitFor|Delay|RemoveIoCompletion|UserMsgWaitFor|UserGetMessage|ReadFile)|WaitOnAddress|NtWaitForAlertByThreadId'
# thread start-up frames that are on every stack and say nothing in the inclusive table
$boilerplate = '!(BaseThreadInitThunk|RtlUserThreadStart|main|invoke_main|__scrt_common_main_seh|std::rt::lang_start.*|std::panicking::catch_unwind.*|std::panic::catch_unwind|core::panic::unwind_safe::.*|std::sys::backtrace::__rust_begin_short_backtrace.*|std::sys::thread::windows::.*thread_start|std::thread::lifecycle::.*|core::ops::function::FnOnce::call_once.*)$'
$self = @{}; $incl = @{}; $names = @{}; $perThreadLeaf = @{}
$taken = 0; $busySamples = 0
$t0 = Get-ThreadTimes; $clock = [Diagnostics.Stopwatch]::StartNew()
for ($i = 0; $i -lt $Samples; $i++) {
  if ($proc.HasExited) { break }
  $r = Invoke-AvkCdb -TimeoutSec 120 -Arguments @('-pv', '-p', "$Id", '-y', $sym, '-c', "~*kn $Frames; q")
  $threads = @(ConvertFrom-AvkStacks $r.Output)
  if (-not $threads) { continue }
  $taken++
  foreach ($t in $threads) {
    if ($t.Name) { $names[$t.Tid] = $t.Name }
    if (-not $t.Frames.Count) { continue }
    if (-not $IncludeWaiting -and $t.Frames[0] -match $waitRegex) { continue }
    $busySamples++
    $leaf = Get-AvkFunction $t.Frames[0]
    $self[$leaf] = 1 + ($self[$leaf] ?? 0)
    $key = "$($t.Tid)|$leaf"; $perThreadLeaf[$key] = 1 + ($perThreadLeaf[$key] ?? 0)
    foreach ($f in ($t.Frames | ForEach-Object { Get-AvkFunction $_ } | Where-Object { $_ -notmatch $boilerplate } | Select-Object -Unique)) { $incl[$f] = 1 + ($incl[$f] ?? 0) }
  }
  Write-Host -NoNewline "`rsample $($i + 1)/$Samples"
  Start-Sleep -Milliseconds $IntervalMs
}
Write-Host ''
$elapsed = $clock.Elapsed.TotalSeconds
$t1 = Get-ThreadTimes

$out = [System.Collections.Generic.List[string]]::new()
$out.Add("== $($proc.ProcessName) (pid $Id), $taken samples over $([math]::Round($elapsed, 1)) s; $busySamples running-thread stacks$(if ($IncludeWaiting) { ' (waiting threads included)' })")
$out.Add(''); $out.Add('== CPU per thread over the window (OS thread times)')
$t1.Keys | ForEach-Object { [pscustomobject]@{ Tid = $_; Cpu = $t1[$_] - ($t0[$_] ?? 0) } } | Sort-Object Cpu -Descending | Select-Object -First 16 | ForEach-Object {
  $out.Add(('  tid {0,6}  {1,6:N1} %  {2}' -f $_.Tid, (100 * $_.Cpu / [math]::Max($elapsed, 0.001)), ($names[$_.Tid] ?? '')))
}
$total = [math]::Max($busySamples, 1)
$out.Add(''); $out.Add("== self (leaf function), top $Top")
$self.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First $Top | ForEach-Object { $out.Add(('  {0,5:N1} %  {1,4}  {2}' -f (100 * $_.Value / $total), $_.Value, $_.Key)) }
$out.Add(''); $out.Add("== inclusive (function anywhere on the stack), top $Top")
$incl.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First $Top | ForEach-Object { $out.Add(('  {0,5:N1} %  {1,4}  {2}' -f (100 * $_.Value / $total), $_.Value, $_.Key)) }
$out.Add(''); $out.Add('== hottest leaf per thread')
$perThreadLeaf.GetEnumerator() | ForEach-Object { $tid, $fn = $_.Key -split '\|', 2; [pscustomobject]@{ Tid = [int]$tid; Fn = $fn; N = $_.Value } } |
  Group-Object Tid | ForEach-Object { $_.Group | Sort-Object N -Descending | Select-Object -First 1 } | Sort-Object N -Descending | Select-Object -First 16 | ForEach-Object {
    $out.Add(('  tid {0,6}  {1,4}  {2}  {3}' -f $_.Tid, $_.N, $_.Fn, ($names[$_.Tid] ?? '')))
  }

$null = New-Item -ItemType Directory -Force -Path $OutDir
$file = Join-Path $OutDir ('samples-{0}-{1}-{2}.txt' -f $proc.ProcessName, $Id, (Get-Date -Format 'yyyyMMdd-HHmmss'))
Set-Content -LiteralPath $file -Value $out -Encoding utf8
$out
Write-Host "== $file" -ForegroundColor Cyan
