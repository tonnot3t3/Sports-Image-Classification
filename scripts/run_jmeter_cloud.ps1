# Run JMeter load test against the Hugging Face Spaces deployment.
# Usage:
#   .\scripts\run_jmeter_cloud.ps1 -HFUser tonnot3t3 -HFSpace sports-vit-api
#   .\scripts\run_jmeter_cloud.ps1 -HFUser tonnot3t3 -HFSpace sports-vit-api -Threads 30 -Duration 120

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $HFUser,

    [Parameter(Mandatory = $true)]
    [string] $HFSpace,

    [int]    $Threads  = 20,
    [int]    $Duration = 60,
    [int]    $RampUp   = 15,
    [string] $Image    = "tests/fixtures/tennis.jpg"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$hfHost = "$HFUser-$HFSpace.hf.space"
$ts     = Get-Date -Format "yyyyMMdd_HHmmss"
$outDir = "jmeter/results/cloud_$ts"
$reportDir = "$outDir/report"
$jtl    = "$outDir/result.jtl"

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

Write-Host "----------------------------------------------" -ForegroundColor DarkGray
Write-Host "JMeter CLOUD  https://$hfHost/predict"          -ForegroundColor Cyan
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
    $r = Invoke-WebRequest "https://$hfHost/health" -UseBasicParsing -TimeoutSec 10
    Write-Host "Cloud /health responded: $($r.StatusCode)" -ForegroundColor Green
} catch {
    Write-Host "WARNING: could not reach https://$hfHost/health" -ForegroundColor Yellow
    Write-Host "         The Space may be sleeping (cold start ~ 30s)." -ForegroundColor Yellow
    Write-Host "         JMeter will still attempt the run." -ForegroundColor Yellow
}

jmeter -n `
       -t "jmeter/load_test.jmx" `
       -l $jtl `
       -e -o $reportDir `
       "-Jhost=$hfHost" `
       "-Jport=443" `
       "-Jscheme=https" `
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
