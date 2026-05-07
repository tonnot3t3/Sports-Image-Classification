# Run JMeter load test against the local uvicorn server.
# Usage:
#   .\scripts\run_jmeter_local.ps1
#   .\scripts\run_jmeter_local.ps1 -Threads 50 -Duration 120

[CmdletBinding()]
param(
    [int]    $Threads  = 30,
    [int]    $Duration = 60,
    [int]    $RampUp   = 15,
    [string] $Image    = "tests/fixtures/tennis.jpg",
    [int]    $Port     = 7860
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$outDir = "jmeter/results/local_$ts"
$reportDir = "$outDir/report"
$jtl = "$outDir/result.jtl"

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Write-Host "----------------------------------------------" -ForegroundColor DarkGray
Write-Host "JMeter LOCAL  http://localhost:$Port/predict"   -ForegroundColor Cyan
Write-Host "Threads=$Threads RampUp=${RampUp}s Duration=${Duration}s" -ForegroundColor Cyan
Write-Host "Image  = $Image"                                -ForegroundColor Cyan
Write-Host "Output = $outDir"                               -ForegroundColor Cyan
Write-Host "----------------------------------------------" -ForegroundColor DarkGray

if (-not (Test-Path $Image)) {
    Write-Host "Test image not found: $Image" -ForegroundColor Yellow
    Write-Host "Running scripts/make_test_image.py to create it..." -ForegroundColor Yellow
    python scripts/make_test_image.py
}

try {
    $null = Invoke-WebRequest "http://localhost:$Port/health" -UseBasicParsing -TimeoutSec 3
} catch {
    Write-Host "ERROR: cannot reach http://localhost:$Port/health" -ForegroundColor Red
    Write-Host "Start uvicorn first:" -ForegroundColor Red
    Write-Host "  uvicorn app.main:app --host 0.0.0.0 --port $Port" -ForegroundColor Red
    exit 1
}

jmeter -n `
       -t "jmeter/load_test.jmx" `
       -l $jtl `
       -e -o $reportDir `
       "-Jhost=localhost" `
       "-Jport=$Port" `
       "-Jscheme=http" `
       "-Jthreads=$Threads" `
       "-Jrampup=$RampUp" `
       "-Jduration=$Duration" `
       "-Jimage_path=$Image"

$dash = Join-Path $reportDir "index.html"
Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host "Raw JTL:   $jtl"
Write-Host "Dashboard: $dash"
Write-Host ""
Write-Host "Open dashboard with:" -ForegroundColor Cyan
Write-Host "  Start-Process $dash"
