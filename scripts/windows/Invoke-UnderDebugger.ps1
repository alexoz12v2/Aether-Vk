#Requires -Version 7
<#
.SYNOPSIS
  Runs a program under cdb and, if it crashes, prints the faulting stack and every thread's stack
  and writes a dump. Windows counterpart of docker-gdb-run.sh (`gdb -batch -ex run -ex "bt full"`).

.DESCRIPTION
  Usage: Invoke-UnderDebugger.ps1 <exe> [args...]
  Every argument is passed to the program untouched (there are no script parameters, so test
  harness flags like -q or --exact are safe); options come from the environment:
    AVK_CRASH_DIR    where the crash log and dump go (default target\diagnostics\crashes)
    AVK_CRASH_DUMP   Full (default, .dump /ma) | Mini | None

  Crashes caught (second chance only, so exceptions the program or a driver handles itself are
  ignored): access violation, __fastfail / Rust abort (c0000409), stack overflow, illegal
  instruction (ud2), divide by zero, in-page error, uncaught C++ / CLR exceptions, a hard-coded
  breakpoint, and any other unhandled exception. On a crash the process is ended and the script
  exits with the exception code (as the crash itself would have); otherwise with the program's exit
  code. Output that the program writes passes through unchanged.

  A `--list` invocation runs the program directly, without cdb, so the script can be a cargo /
  nextest target runner (nextest parses the list output).

.EXAMPLE
  scripts\windows\Invoke-UnderDebugger.ps1 target\debug\deps\aethervk_core_rlib-<hash>.exe --exact some::test --nocapture
  cargo nextest run -p aethervk-core-rlib -E 'test(=some::test)' --debugger 'pwsh -NoProfile -File scripts/windows/Invoke-UnderDebugger.ps1'
  # every test of a run under the debugger (cargo test or nextest), as a target runner:
  $env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUNNER = 'pwsh -NoProfile -File Y:/Aether-Vk/scripts/windows/Invoke-UnderDebugger.ps1'
#>
$ErrorActionPreference = 'Stop'
if ($args.Count -lt 1) { Write-Error 'usage: Invoke-UnderDebugger.ps1 <exe> [args...]'; exit 2 }
$exe = $args[0]
$progArgs = @($args | Select-Object -Skip 1)

if ($progArgs -contains '--list') {
  & $exe @progArgs
  exit $LASTEXITCODE
}

Import-Module (Join-Path $PSScriptRoot 'AvkDebug.psm1') -Force
$outDir = if ($env:AVK_CRASH_DIR) { $env:AVK_CRASH_DIR } else { Join-Path (Get-AvkRepoRoot) 'target\diagnostics\crashes' }
$dumpKind = if ($env:AVK_CRASH_DUMP) { $env:AVK_CRASH_DUMP } else { 'Full' }
$null = New-Item -ItemType Directory -Force -Path $outDir
$exePath = (Resolve-Path -LiteralPath $exe).Path
$base = Join-Path $outDir ('crash-{0}-{1}-{2}' -f [IO.Path]::GetFileNameWithoutExtension($exePath), $PID, (Get-Date -Format 'yyyyMMdd-HHmmss'))

# Commands run at the second-chance break. Inside the quoted `sx -c2` string a backslash escapes,
# so the dump path is written with doubled backslashes and \" quotes.
$onCrash = @('.echo ==AVK_CRASH==', '.lastevent', 'r', 'kn 100', '.echo ==AVK_ALL_THREADS==', '~*kn 60', '!locks')
if ($dumpKind -ne 'None') {
  $flag = if ($dumpKind -eq 'Mini') { '/mhti' } else { '/ma' }
  $onCrash += ".dump $flag /o \`"$($base.Replace('\', '\\')).dmp\`""
}
$onCrash += '.echo ==AVK_CRASH_END==', 'q'
$crashCmd = $onCrash -join '; '
# av: access violation, sbo: __fastfail (Rust abort), sov: stack overflow, ii: illegal instruction,
# dz: divide by zero, ip: in-page error, eh: C++ EH, clr: .NET, bpe: hard-coded breakpoint, *: the rest
$events = 'av', 'sbo', 'sov', 'ii', 'dz', 'ip', 'eh', 'clr', 'bpe', '*'
$init = @($events | ForEach-Object { "sxd -c2 `"$crashCmd`" $_" }) + 'g'

$psi = [System.Diagnostics.ProcessStartInfo]::new((Get-AvkCdb))
foreach ($a in @('-G', '-lines', '-y', (Get-AvkSymbolPath -Extra @(Split-Path $exePath)), '-c', ($init -join '; '), $exePath) + $progArgs) { $psi.ArgumentList.Add($a) }
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.RedirectStandardInput = $true   # EOF: an unexpected prompt ends the session instead of blocking
$psi.StandardOutputEncoding = [Text.Encoding]::UTF8
$psi.StandardErrorEncoding = [Text.Encoding]::UTF8
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$p = [System.Diagnostics.Process]::Start($psi)
$p.StandardInput.Close()
$errTask = $p.StandardError.ReadToEndAsync()

# cdb's banner runs until it echoes the initial command; after that, everything is the program's
# output or ours (the crash block). The banner and module-load chatter are dropped.
$inBanner = $true
$crash = [System.Collections.Generic.List[string]]::new()
$inCrash = $false
while ($null -ne ($line = $p.StandardOutput.ReadLine())) {
  if ($inBanner) {
    if ($line -match "cdb: Reading initial command") { $inBanner = $false }
    continue
  }
  if ($line -match '^ModLoad:|^NatVis script|^\*\*\* WARNING: Unable to verify checksum|^quit:$') { continue }
  if ($line -match '^\(\w+\.\w+\): .* - code [0-9a-f]+ \(first chance\)$') { continue }
  if ($line -eq '==AVK_CRASH==') { $inCrash = $true }
  if ($inCrash) { $crash.Add($line) }
  [Console]::Out.WriteLine($line)
}
$p.WaitForExit()
$stderr = $errTask.Result
if ($stderr) { [Console]::Error.Write($stderr) }

if ($crash.Count) {
  Set-Content -LiteralPath "$base.txt" -Value (@("== $exePath $($progArgs -join ' ')") + $crash) -Encoding utf8
  [Console]::Error.WriteLine("[under-debugger] crash report: $base.txt$(if ($dumpKind -ne 'None') { " (dump $base.dmp)" })")
  $code = ($crash | Select-String -Pattern 'code ([0-9a-f]{8})' | Select-Object -First 1)
  if ($code) { exit [BitConverter]::ToInt32([BitConverter]::GetBytes([Convert]::ToUInt32($code.Matches[0].Groups[1].Value, 16)), 0) }
  exit 1
}
exit $p.ExitCode
