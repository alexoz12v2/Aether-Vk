param (
    [switch]$SingleThread
)

$ErrorActionPreference = "Stop"

Write-Host "========================================"
Write-Host "         Running Rust Tests             "
Write-Host "========================================"

if ($SingleThread) {
    Write-Host "Mode: Single-threaded, No Capture" -ForegroundColor Yellow
    cargo llvm-cov nextest --no-capture --test-threads 1
} else {
    Write-Host "Mode: Default"
    cargo llvm-cov nextest
}

Write-Host ""
Write-Host "========================================"
Write-Host "          Running C# Tests              "
Write-Host "========================================"

if ($SingleThread) {
    Write-Host "Mode: Single-threaded, No Capture (Verbose)" -ForegroundColor Yellow
    dotnet test -m:1 --logger "console;verbosity=normal" --collect:"XPlat Code Coverage"
} else {
    Write-Host "Mode: Default"
    dotnet test --collect:"XPlat Code Coverage"
}

Write-Host ""
Write-Host "========================================"
Write-Host "  Tests and Coverage Generation Done!   " -ForegroundColor Green
Write-Host "========================================"
