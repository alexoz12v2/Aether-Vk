#Requires -Version 7
<#
.SYNOPSIS
  RenderDoc capture of one GPU test (or any exe), for inspecting its compute dispatches and buffers.

.DESCRIPTION
  Runs the newest aethervk-core-rlib test exe (debug build) under `renderdoccmd capture` with
  AETHERVK_RENDERDOC_CAPTURE=1: the engine then captures the test device's whole life
  (renderdoc::DeviceLifeCapture, debug builds), so a test without a swapchain still yields one
  .rdc with every submitted command buffer. Captures land in target\diagnostics\renderdoc\ and the
  newest one is opened in qrenderdoc unless -NoOpen.
  The implicit-layer disable nextest uses is cleared for this run (RenderDoc is itself an implicit
  layer). Set AETHERVK_SHADER_DEBUG_INFO=1 as well to load the -g shader twins, which RenderDoc's
  shader view and debugger show as GLSL source.

.EXAMPLE
  scripts\windows\Capture-RenderDoc.ps1 -Test gpu_backends::vulkan::device::dust::cadence::tests::gpu_dust_cadence_state_matches_reference
  scripts\windows\Capture-RenderDoc.ps1 -Exe path\to\other.exe -- --some-arg
#>
param(
  [string] $Test,
  [string] $Exe,
  [switch] $ShaderDebugInfo,
  [switch] $NoOpen,
  [Parameter(ValueFromRemainingArguments = $true)] [string[]] $ExtraArgs
)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $PSScriptRoot 'AvkDebug.psm1') -Force

$rdoc = Join-Path $env:ProgramFiles 'RenderDoc'
$cmd = Join-Path $rdoc 'renderdoccmd.exe'
if (-not (Test-Path $cmd)) { throw "RenderDoc not found in $rdoc (https://renderdoc.org)" }

$root = Get-AvkRepoRoot
if (-not $Exe) {
  if (-not $Test) { throw 'Give -Test <path::to::test> (or -Exe <program>)' }
  # the pinned nightly's build-dir layout: target\debug\build\<pkg>\<hash>\out\<name>-<hash>.exe
  $Exe = Get-ChildItem (Join-Path $root 'target\debug') -Recurse -Filter 'aethervk_core_rlib-*.exe' -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch '\\build_script_' } |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
  if (-not $Exe) { throw 'No aethervk-core-rlib test exe: build it first (cargo nextest run --no-run -p aethervk-core-rlib --lib)' }
}
if ($ExtraArgs -and $ExtraArgs[0] -eq '--') { $ExtraArgs = $ExtraArgs[1..($ExtraArgs.Count - 1)] }
$progArgs = @()
if ($Test) { $progArgs += @('--exact', $Test, '--nocapture', '--test-threads=1') }
$progArgs += $ExtraArgs

$dir = Join-Path $root 'target\diagnostics\renderdoc'
$null = New-Item -ItemType Directory -Force -Path $dir
$name = if ($Test) { ($Test -split '::')[-1] } else { [IO.Path]::GetFileNameWithoutExtension($Exe) }
$template = Join-Path $dir "$name-$(Get-Date -Format 'yyyyMMdd-HHmmss')"

$env:AETHERVK_RENDERDOC_CAPTURE = '1'
if ($ShaderDebugInfo) { $env:AETHERVK_SHADER_DEBUG_INFO = '1' }
Remove-Item Env:VK_LOADER_LAYERS_DISABLE -ErrorAction SilentlyContinue
# the asset lookup walks up from the exe; run from the repo so `assets` is found either way
& $cmd capture -w -d $root -c $template --opt-api-validation --opt-ref-all-resources $Exe @progArgs
$code = $LASTEXITCODE

$cap = Get-ChildItem $dir -Filter "$(Split-Path $template -Leaf)*.rdc" -ErrorAction SilentlyContinue |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($cap) {
  "Capture: $($cap.FullName)"
  if (-not $NoOpen) { Start-Process (Join-Path $rdoc 'qrenderdoc.exe') -ArgumentList "`"$($cap.FullName)`"" }
} else {
  "No capture written (exit $code). Is it a debug build? Did the test create a device?"
}
exit $code
