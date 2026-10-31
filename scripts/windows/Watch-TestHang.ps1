#Requires -Version 7
<#
.SYNOPSIS
  Hang watchdog for test runs: dumps every thread's stack of a test process that stalls (0 % CPU)
  or runs too long, before nextest's slow-timeout kills it. Windows counterpart of the Linux
  "launch under gdb, SIGINT after N s, thread apply all bt" recipe and the render-stall gdb dump.

.DESCRIPTION
  With a command after `--`: runs it (output unchanged) and watches its descendant processes;
  exits with the command's exit code. Without a command: watches every matching process on the
  machine until Ctrl+C (start it in a second terminal next to a run that is already going).

  A process is caught when its image path matches -Pattern (by default the test and example exes
  under target\, in the classic deps\ layout and the nightly build\<pkg>\<hash>\out\ layout, and
  the AetherVk app; build scripts excluded) and either
    - stall: it used less than -MinCpuSec of CPU over the last -StallSec seconds, or
    - age:   it has been alive for -MaxAgeSec (a spinning hang; the default 270 s fires before
             nextest's terminate-after of 5 x 60 s in .config/nextest.toml).
  Each process is reported once per trigger, through Get-HungStacks' non-invasive attach: a report
  in -OutDir (thread summary, ~*kn, !runaway, !locks) and, with -Dump Mini|Full, a dump.

  nextest has no hook that runs before a timeout kill, so this watches from outside instead. It
  works the same for plain `cargo test` (one process for the whole suite).

.EXAMPLE
  scripts\windows\Watch-TestHang.ps1 -- cargo nextest run -p aethervk-core-rlib
  scripts\windows\Watch-TestHang.ps1 -StallSec 20 -Dump Full -- cargo test -p aethervk-core-rlib --lib -- --test-threads=1
  scripts\windows\Watch-TestHang.ps1          # watch mode, in a second terminal
#>
[CmdletBinding(PositionalBinding = $false)]
param(
  [int] $StallSec = 30,
  [double] $MinCpuSec = 0.25,
  [int] $MaxAgeSec = 270,
  [int] $PollSec = 2,
  # Test / example exes: target\<profile>\deps (classic layout) and, with the nightly cargo's new
  # build-dir layout, target\<profile>\build\<pkg>\<hash>\out\<name>-<hash>.exe; plus the app
  [string] $Pattern = '\\target\\.*\\(deps|examples|build\\[^\\]+\\[0-9a-f]{16}\\out)\\[^\\]+\.exe$|\\AetherVk\.exe$',
  # Build scripts live in the same out folders and may run long (shader compiles): never caught
  [string] $Exclude = '\\build_script_[^\\]*\.exe$',
  [ValidateSet('None', 'Mini', 'Full')] [string] $Dump = 'Mini',
  [string] $OutDir,
  [Parameter(ValueFromRemainingArguments)] [string[]] $Command
)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'AvkDebug.psm1') -Force
if (-not $OutDir) { $OutDir = Join-Path (Get-AvkRepoRoot) ('target\diagnostics\hangs\' + (Get-Date -Format 'yyyyMMdd-HHmmss')) }
if ($Command -and $Command[0] -eq '--') { $Command = $Command[1..($Command.Count - 1)] }
$null = Get-AvkCdb   # fail early when cdb is missing
$null = Test-AvkSymsrvExclusions

$root = $null
if ($Command) {
  $psi = [System.Diagnostics.ProcessStartInfo]::new($Command[0])
  foreach ($a in $Command | Select-Object -Skip 1) { $psi.ArgumentList.Add($a) }
  $psi.UseShellExecute = $false
  $psi.WorkingDirectory = (Get-Location).Path
  # cargo.exe is found on PATH like a shell would; a .cmd/.ps1 shim needs its full path
  $root = [System.Diagnostics.Process]::Start($psi)
  Write-Host "[hang-watch] watching pid $($root.Id) ($($Command -join ' ')); stall $StallSec s < $MinCpuSec s CPU, age $MaxAgeSec s -> $OutDir" -ForegroundColor DarkCyan
} else {
  Write-Host "[hang-watch] watching every process matching $Pattern (Ctrl+C to stop) -> $OutDir" -ForegroundColor DarkCyan
}

# pid -> @{ Samples = Queue[(time, cpu)]; Fired = HashSet[trigger] }
$state = @{}

function Get-Watched {
  $all = Get-CimInstance Win32_Process -Property ProcessId, ParentProcessId, ExecutablePath, CreationDate
  $pool = $all
  if ($root) {
    $children = @{}
    foreach ($p in $all) { if (-not $children.ContainsKey($p.ParentProcessId)) { $children[$p.ParentProcessId] = @() }; $children[$p.ParentProcessId] += $p }
    $pool = [System.Collections.Generic.List[object]]::new()
    $pool.AddRange([object[]]@($all | Where-Object { $_.ProcessId -eq $root.Id }))
    $queue = [System.Collections.Generic.Queue[object]]::new(); $queue.Enqueue(@($root.Id, $root.StartTime))
    while ($queue.Count) {
      $id, $born = $queue.Dequeue()
      foreach ($c in @($children[[uint32]$id])) {
        # a recycled parent pid makes an older, unrelated process look like a child: skip those
        if (-not $c -or $c.CreationDate -lt $born) { continue }
        $pool.Add($c); $queue.Enqueue(@([int]$c.ProcessId, $c.CreationDate))
      }
    }
  }
  $pool | Where-Object { $_.ExecutablePath -and $_.ExecutablePath -match $Pattern -and -not ($Exclude -and $_.ExecutablePath -match $Exclude) }
}

function Invoke-Report([int] $ProcId, [string] $Trigger, [string] $Why) {
  Write-Host "[hang-watch] $Trigger pid $ProcId : $Why; capturing stacks..." -ForegroundColor Yellow
  try {
    $report = Save-AvkStacks -Id $ProcId -OutDir $OutDir -Dump $Dump -Label "hang-$Trigger" -Note "trigger: $Trigger ($Why)" -Managed:($Why -match 'AetherVk\.exe|testhost')
    $lines = @(Get-Content -LiteralPath $report)
    $s = [array]::IndexOf($lines, '== thread summary (leaf frame / first application frame / tags)')
    $e = [array]::IndexOf($lines, '== full cdb output')
    if ($s -ge 0 -and $e -gt $s) { $lines[$s..($e - 1)] | ForEach-Object { Write-Host "[hang-watch]   $_" -ForegroundColor Yellow } }
    Write-Host "[hang-watch] report: $report" -ForegroundColor Yellow
  } catch {
    Write-Host "[hang-watch] capture failed for pid ${ProcId}: $_" -ForegroundColor Red
  }
}

try {
  while ($true) {
    if ($root -and $root.HasExited) { break }
    $now = [DateTime]::UtcNow
    $seen = @{}
    foreach ($w in @(Get-Watched)) {
      $id = [int]$w.ProcessId
      $seen[$id] = $true
      $proc = Get-Process -Id $id -ErrorAction SilentlyContinue
      if (-not $proc) { continue }
      try { $cpu = $proc.TotalProcessorTime.TotalSeconds } catch { continue }
      if (-not $state.ContainsKey($id)) {
        $state[$id] = @{ Samples = [System.Collections.Generic.Queue[object]]::new(); Fired = [System.Collections.Generic.HashSet[string]]::new(); Start = $w.CreationDate.ToUniversalTime() }
      }
      $st = $state[$id]
      $st.Samples.Enqueue(@($now, $cpu))
      while ($st.Samples.Count -gt 1 -and ($now - $st.Samples.Peek()[0]).TotalSeconds -gt $StallSec + $PollSec) { $null = $st.Samples.Dequeue() }
      $oldest = $st.Samples.Peek()
      $window = ($now - $oldest[0]).TotalSeconds
      $used = $cpu - $oldest[1]
      $age = ($now - $st.Start).TotalSeconds
      if ($window -ge $StallSec -and $used -lt $MinCpuSec -and $st.Fired.Add('stall')) {
        Invoke-Report $id 'stall' ('{0:N2} s CPU in the last {1:N0} s, alive {2:N0} s, {3}' -f $used, $window, $age, (Split-Path -Leaf $w.ExecutablePath))
      }
      # a stalled process already has its report; the age trigger is for the spinning ones
      if ($MaxAgeSec -gt 0 -and $age -ge $MaxAgeSec -and -not $st.Fired.Contains('stall') -and $st.Fired.Add('age')) {
        Invoke-Report $id 'age' ('alive {0:N0} s, {1:N1} s CPU in the last {2:N0} s, {3}' -f $age, $used, $window, (Split-Path -Leaf $w.ExecutablePath))
      }
    }
    foreach ($k in @($state.Keys)) { if (-not $seen.ContainsKey($k)) { $state.Remove($k) } }
    Start-Sleep -Seconds $PollSec
  }
} finally {
  if (Test-Path -LiteralPath $OutDir) { Write-Host "[hang-watch] reports in $OutDir" -ForegroundColor DarkCyan }
}
if ($root) {
  $root.WaitForExit()
  exit $root.ExitCode
}
