#Requires -Version 7
# Shared helpers for the Windows diagnostics scripts in this folder (the counterpart of the Linux
# gdb / perf tooling): locating cdb.exe, a fast symbol path, running cdb with a timeout, parsing
# `~*kn` output and writing a stack report (+ optional minidump) for a live process.
#
# Symbols: local folders come first (the Rust PDBs under target\, the Vulkan SDK layer PDBs), then
# the Microsoft symbol server through the cache. Loading stays deferred (no `.reload /f`), so only
# modules that appear on a stack are looked up. Vendor driver modules with no public PDB (the NVIDIA
# ICD `nvoglv64.dll` above all) are listed in `symsrv.ini` next to cdb (Setup-DebugTools.ps1 writes
# it); without that list each of them costs a symbol-server timeout.

Set-StrictMode -Version 3

$script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

# Modules that never have public symbols on the Microsoft server: the symbol server is not asked
# for them (symsrv.ini [exclusions], wildcards allowed). Local folders on the symbol path are still
# searched, so the Vulkan SDK layers and our own PDBs still resolve.
$script:SymsrvExclusions = @(
  'nvogl*.pdb', 'nvgpucomp*.pdb', 'nvwgf2um*.pdb', 'nvldumd*.pdb', 'nvd3dum*.pdb', 'nvapi*.pdb',
  'nvcuda*.pdb', 'nvumdshim*.pdb', 'nvppe*.pdb', 'nvspcap*.pdb', 'nvinit*.pdb', 'nvoptimus*.pdb',
  'amdvlk*.pdb', 'amdxc*.pdb', 'amdihk*.pdb', 'atiu*.pdb', 'atidx*.pdb', 'aticfx*.pdb',
  'igvk*.pdb', 'igc*.pdb', 'igd*.pdb', 'intelcontrollib*.pdb',
  'VkLayer_*.pdb', 'vulkan-1.pdb',
  'aethervk*.pdb', 'hangdemo*.pdb', 'libSkiaSharp.pdb', 'libHarfBuzzSharp.pdb', 'av_libglesv2.pdb',
  'gameoverlayrenderer*.pdb', 'RTSSHooks*.pdb', 'DiscordHook*.pdb', 'GfxUI*.pdb'
)

function Get-AvkRepoRoot { $script:RepoRoot }

function Get-AvkCdb {
  <# Path of cdb.exe: $env:AVK_CDB, the copy in %LOCALAPPDATA%\dbg\amd64 (the Store WinDbg's cdb
     cannot run from WindowsApps), the Windows Kits debuggers, then PATH. Throws when none is found. #>
  $candidates = @(
    $env:AVK_CDB,
    (Join-Path $env:LOCALAPPDATA 'dbg\amd64\cdb.exe'),
    "${env:ProgramFiles(x86)}\Windows Kits\10\Debuggers\x64\cdb.exe",
    "$env:ProgramFiles\Windows Kits\10\Debuggers\x64\cdb.exe"
  ) | Where-Object { $_ }
  foreach ($c in $candidates) { if (Test-Path -LiteralPath $c) { return (Resolve-Path -LiteralPath $c).Path } }
  $onPath = Get-Command cdb.exe -ErrorAction SilentlyContinue
  if ($onPath) { return $onPath.Source }
  throw "cdb.exe not found. Run scripts\windows\Setup-DebugTools.ps1 (copies it out of the WinDbg Store package) or set AVK_CDB."
}

function Get-AvkSymbolPath {
  <# Local folders first (no network, and they keep our 200 MB test PDBs out of the cache), then
     srv*<cache>*msdl. $env:AVK_SYMBOL_CACHE overrides the cache (default C:\symbols). #>
  param([string[]] $Extra = @())
  $cache = if ($env:AVK_SYMBOL_CACHE) { $env:AVK_SYMBOL_CACHE } else { 'C:\symbols' }
  $local = @($Extra) + @(
    'target\debug\deps', 'target\debug', 'target\debug\examples',
    'target\release\deps', 'target\release', 'target\release\examples'
  ).ForEach({ Join-Path $script:RepoRoot $_ })
  if ($env:VULKAN_SDK) { $local += (Join-Path $env:VULKAN_SDK 'Bin') }
  $local = $local | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -Unique
  (@($local) + "srv*$cache*https://msdl.microsoft.com/download/symbols") -join ';'
}

function Get-AvkSymsrvIniPath { Join-Path (Split-Path (Get-AvkCdb)) 'symsrv.ini' }

function Test-AvkSymsrvExclusions {
  <# Warns once when the symsrv.ini exclusion list next to cdb is missing (each vendor driver module
     then costs a symbol-server timeout). Returns $true when present. #>
  $ini = Get-AvkSymsrvIniPath
  if ((Test-Path -LiteralPath $ini) -and (Select-String -LiteralPath $ini -Pattern '^nvogl' -Quiet)) { return $true }
  Write-Warning "No symbol-server exclusion list at ${ini}: stacks through nvoglv64.dll may stall on symbol lookups. Run scripts\windows\Setup-DebugTools.ps1."
  $false
}

# cdb start-up / shutdown chatter that carries nothing for a report
$script:NoiseRegex = '^(NatVis script|\s*-+> Repository|>>>>>>>|\*{5,} (Waiting|Preparing)|Microsoft \(R\) Windows Debugger|Copyright \(c\) Microsoft|\*{5,} Path validation|Response\s+Time|Deferred\s+srv|ModLoad:|\*\*\* WARNING: Unable to verify checksum|\s*$)'

function Remove-AvkCdbNoise { param([string[]] $Lines) $Lines | Where-Object { $_ -notmatch $script:NoiseRegex } }

function Invoke-AvkCdb {
  <# Runs cdb with the given argument list, captures stdout+stderr, kills it after TimeoutSec.
     Returns [pscustomobject]@{ Output; TimedOut; ExitCode }. #>
  param(
    [Parameter(Mandatory)] [string[]] $Arguments,
    [int] $TimeoutSec = 120
  )
  $psi = [System.Diagnostics.ProcessStartInfo]::new((Get-AvkCdb))
  foreach ($a in $Arguments) { $psi.ArgumentList.Add($a) }
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  $psi.RedirectStandardInput = $true   # EOF on stdin: cdb never waits at a prompt
  $psi.UseShellExecute = $false
  $psi.CreateNoWindow = $true
  $p = [System.Diagnostics.Process]::Start($psi)
  $p.StandardInput.Close()
  $out = $p.StandardOutput.ReadToEndAsync()
  $err = $p.StandardError.ReadToEndAsync()
  $timedOut = -not $p.WaitForExit($TimeoutSec * 1000)
  if ($timedOut) { try { $p.Kill($true) } catch {} ; $p.WaitForExit() }
  $text = $out.Result + $err.Result
  [pscustomobject]@{ Output = ($text -split "`r?`n"); TimedOut = $timedOut; ExitCode = $p.ExitCode }
}

function ConvertFrom-AvkStacks {
  <# Parses `~*kn` output into threads: @{ Index; Tid; Name; Frames = @(string call sites) }. A call
     site keeps `module!function+0xoff` and drops the trailing ` [file @ line]`. #>
  param([string[]] $Lines)
  $threads = [System.Collections.Generic.List[object]]::new()
  $cur = $null
  foreach ($l in $Lines) {
    if ($l -match '^\s*[.#]?\s*(\d+)\s+Id:\s+[0-9a-f]+\.([0-9a-f]+)\s.*?(?:"([^"]*)")?\s*$') {
      $cur = [pscustomobject]@{ Index = [int]$Matches[1]; Tid = [Convert]::ToInt32($Matches[2], 16); Name = $Matches[3]; Frames = [System.Collections.Generic.List[string]]::new() }
      $threads.Add($cur)
      continue
    }
    if ($cur -and $l -match '^[0-9a-f]{2,}\s+(?:\(Inline Function\)\s+\S+|[0-9a-f`]+\s+[0-9a-f`]+)\s+(.+?)\s*$') {
      $site = $Matches[1] -replace '\s+\[.*\]$', ''
      $cur.Frames.Add($site)
    }
  }
  $threads
}

function Get-AvkFunction { param([string] $Site) ($Site -replace '\+0x[0-9a-f]+$', '') }

$script:SystemModules = '^(ntdll|KERNELBASE|KERNEL32|win32u|USER32|ucrtbase|msvcrt|msvcp\w*|vcruntime\w*|combase|RPCRT4|GDI32|gdi32full|sechost|ws2_32|mswsock|coreclr|clrjit|System_\w+)!'
$script:LibFunctions = '!(std::|core::|alloc::|parking_lot|parking_lot_core|crossbeam|rayon_core|__scrt|invoke_main|BaseThreadInitThunk|RtlUserThreadStart|thread_start|main$)'

function Get-AvkThreadSummary {
  <# One line per thread: index, tid, name, leaf frame, the first frame in application code (not a
     system module, not std/core) and a tag when the stack runs through the Vulkan loader, a layer or
     the driver. #>
  param($Threads)
  foreach ($t in $Threads) {
    $leaf = if ($t.Frames.Count) { $t.Frames[0] } else { '?' }
    $ours = $t.Frames | Where-Object { $_ -notmatch $script:SystemModules -and $_ -notmatch $script:LibFunctions } | Select-Object -First 1
    $tags = @()
    if ($t.Frames -match '^vulkan_1!') { $tags += 'loader' }
    if ($t.Frames -match '^VkLayer_') { $tags += 'layer' }
    if ($t.Frames -match '^(nvoglv64|nvgpucomp64|amdvlk64|igvk64)!') { $tags += 'driver' }
    if ($leaf -match '!(Nt|Zw)(WaitFor|Delay|RemoveIoCompletion|UserMsgWaitFor|UserGetMessage)|WaitOnAddress|NtWaitForAlertByThreadId') { $tags += 'waiting' }
    $name = if ($t.Name) { " `"$($t.Name)`"" } else { '' }
    '{0,3} tid {1,6}{2}  leaf={3}  app={4}{5}' -f $t.Index, $t.Tid, $name, (Get-AvkFunction $leaf),
      ($(if ($ours) { Get-AvkFunction $ours } else { '-' })), ($(if ($tags) { "  [" + ($tags -join ',') + "]" } else { '' }))
  }
}

function Save-AvkStacks {
  <# Non-invasive attach (-pv: the threads are suspended while cdb reads them, then resumed; the
     process is not otherwise disturbed) that writes every thread's stack, the critical-section
     owners and the per-thread CPU times to <OutDir>\<label>-<name>-<pid>-<time>.txt, plus a dump
     when Dump is Mini or Full. Returns the report path. #>
  param(
    [Parameter(Mandatory)] [int] $Id,
    [string] $OutDir = (Join-Path $script:RepoRoot 'target\diagnostics\stacks'),
    [ValidateSet('None', 'Mini', 'Full')] [string] $Dump = 'None',
    [int] $Frames = 80,
    [string] $Label = 'stacks',
    [int] $TimeoutSec = 120,
    [string] $Note = '',
    # .NET process (the Avalonia app, testhost): add the managed stacks (SOS `!clrstack -all`; the
    # first run downloads the runtime's DAC, ~40 s)
    [switch] $Managed
  )
  $proc = Get-Process -Id $Id -ErrorAction Stop
  $null = New-Item -ItemType Directory -Force -Path $OutDir
  $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
  $base = Join-Path $OutDir ('{0}-{1}-{2}-{3}' -f $Label, $proc.ProcessName, $Id, $stamp)
  $null = Test-AvkSymsrvExclusions

  $cmds = @('.echo AVK_BEGIN_STACKS', "~*kn $Frames", '.echo AVK_END_STACKS', '!runaway 3', '!locks')
  if ($Managed) { $cmds += '.echo == managed stacks (SOS)', '!clrstack -all' }
  if ($Dump -ne 'None') {
    $flag = if ($Dump -eq 'Full') { '/ma' } else { '/mhti' }
    $cmds += ".dump $flag /o `"$base.dmp`""
  }
  $cmds += 'q'
  $exe = try { $proc.Path } catch { $null }
  $extra = if ($exe) { @(Split-Path $exe) } else { @() }
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $r = Invoke-AvkCdb -TimeoutSec $TimeoutSec -Arguments @(
    '-pv', '-p', "$Id", '-y', (Get-AvkSymbolPath -Extra $extra), '-lines', '-c', ($cmds -join '; ')
  )
  $lines = @(Remove-AvkCdbNoise $r.Output)
  $stackLines = $lines
  $b = [array]::IndexOf($lines, 'AVK_BEGIN_STACKS'); $e = [array]::IndexOf($lines, 'AVK_END_STACKS')
  if ($b -ge 0 -and $e -gt $b) { $stackLines = $lines[$b..$e] }
  $threads = @(ConvertFrom-AvkStacks $stackLines)

  $head = @(
    "== $Label of $($proc.ProcessName) (pid $Id) at $(Get-Date -Format o)",
    "== exe: $exe",
    "== cpu: $([math]::Round($proc.TotalProcessorTime.TotalSeconds, 2)) s total, threads $($proc.Threads.Count), handles $($proc.HandleCount), private $([math]::Round($proc.PrivateMemorySize64 / 1MB)) MB",
    "== cdb: $([math]::Round($sw.Elapsed.TotalSeconds, 1)) s$(if ($r.TimedOut) { ' (TIMED OUT, killed)' })"
  )
  if ($Note) { $head += "== $Note" }
  $head += '', '== thread summary (leaf frame / first application frame / tags)'
  $head += @(Get-AvkThreadSummary $threads)
  $head += '', '== full cdb output'
  Set-Content -LiteralPath "$base.txt" -Value ($head + $lines) -Encoding utf8
  $base + '.txt'
}

Export-ModuleMember -Function Get-AvkRepoRoot, Get-AvkCdb, Get-AvkSymbolPath, Get-AvkSymsrvIniPath, Test-AvkSymsrvExclusions,
  Remove-AvkCdbNoise, Invoke-AvkCdb, ConvertFrom-AvkStacks, Get-AvkFunction, Get-AvkThreadSummary, Save-AvkStacks -Variable SymsrvExclusions
