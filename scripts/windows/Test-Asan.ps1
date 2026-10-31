#Requires -Version 7
<#
.SYNOPSIS
  Runs a crate's tests under AddressSanitizer on Windows MSVC (nightly `-Zsanitizer=address`): heap
  and stack overruns, use after free, double free in the Rust code and its C dependencies.

.DESCRIPTION
  Builds into its own target dir (target\asan), so the normal debug build is not invalidated, then
  runs the test exe with the MSVC ASan runtime (clang_rt.asan_dynamic-x86_64.dll) on PATH. The
  runtime must come from the toolset rustc linked with: the newest MSVC toolset that ships the DLL is
  picked (rustc links with the newest Visual Studio's), $env:AVK_ASAN_RUNTIME_DIR overrides it. A
  mismatched runtime fails at start (exit 127 / STATUS_DLL_NOT_FOUND, or an interception error).

  For the CPU tests only: the GPU drivers are not instrumented and their allocations confuse the
  shadow memory, so the default skips the Vulkan, simulation-API and logic-thread tests (and the
  slow scene::dust reference tests; pass -Skip @() to run them). An ASan report ends the process
  with the stack of the bad access and of the allocation; the summary line is echoed at the end.

.EXAMPLE
  scripts\windows\Test-Asan.ps1                                   # aethervk-core-rlib, CPU tests
  scripts\windows\Test-Asan.ps1 -Package aethervk-oshal-rlib -Skip @()
  scripts\windows\Test-Asan.ps1 -- scene::dust::tests::host_rasterizer_is_thread_count_independent --exact
#>
param(
  [string] $Package = 'aethervk-core-rlib',
  [string[]] $Skip = @('gpu_backends', 'simulation_api', 'logic_thread', 'scene::dust', 'test_dust_config'),
  [int] $Threads = 8,
  [string] $Toolchain = 'nightly-2026-09-26',
  [Parameter(ValueFromRemainingArguments = $true)] [string[]] $TestArgs
)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'AvkDebug.psm1') -Force
$root = Get-AvkRepoRoot

function Get-AsanRuntimeDir {
  if ($env:AVK_ASAN_RUNTIME_DIR) { return $env:AVK_ASAN_RUNTIME_DIR }
  $dlls = foreach ($vs in "$env:ProgramFiles\Microsoft Visual Studio", "${env:ProgramFiles(x86)}\Microsoft Visual Studio") {
    Get-ChildItem "$vs\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\clang_rt.asan_dynamic-x86_64.dll" -ErrorAction SilentlyContinue
  }
  $best = $dlls | Sort-Object { [version](Split-Path -Leaf (Split-Path (Split-Path (Split-Path (Split-Path $_.FullName))))) } -Descending |
    Select-Object -First 1
  if (-not $best) { throw 'clang_rt.asan_dynamic-x86_64.dll not found: install the "C++ AddressSanitizer" Visual Studio component, or set AVK_ASAN_RUNTIME_DIR' }
  $best.DirectoryName
}

if ($TestArgs -and $TestArgs[0] -eq '--') { $TestArgs = $TestArgs[1..($TestArgs.Count - 1)] }
$runtime = Get-AsanRuntimeDir
"ASan runtime: $runtime"

$env:RUSTFLAGS = '-Zsanitizer=address'
$env:CARGO_TARGET_DIR = Join-Path $root 'target\asan'
try {
  $json = & rustup run $Toolchain cargo test -p $Package --lib --target x86_64-pc-windows-msvc --no-run --message-format json 2>$null
  if ($LASTEXITCODE -ne 0) { throw "ASan build failed: rerun with RUSTFLAGS=-Zsanitizer=address CARGO_TARGET_DIR=target\asan cargo test -p $Package --lib --target x86_64-pc-windows-msvc --no-run" }
} finally {
  Remove-Item Env:RUSTFLAGS, Env:CARGO_TARGET_DIR -ErrorAction SilentlyContinue
}
$artifact = $json | ForEach-Object { try { $_ | ConvertFrom-Json } catch {} } |
  Where-Object { $_.reason -eq 'compiler-artifact' -and $_.executable -and $_.profile.test } |
  Select-Object -Last 1
if (-not $artifact) { throw 'No test executable in cargo output' }
$exe = $artifact.executable
"Test exe: $exe"

$runArgs = @("--test-threads=$Threads") + ($Skip | ForEach-Object { '--skip', $_ }) + @($TestArgs)
$env:PATH = "$runtime;$env:PATH"
# from the crate's folder, as cargo runs tests (fixtures are read relative to it)
Push-Location (Split-Path $artifact.manifest_path)
try {
  & $exe @runArgs 2>&1 | Tee-Object -Variable out | Out-Host
  $code = $LASTEXITCODE
} finally {
  Pop-Location
}
$summary = $out | Where-Object { "$_" -match 'SUMMARY: AddressSanitizer|ERROR: AddressSanitizer|test result:' }
''
$summary
exit $code
