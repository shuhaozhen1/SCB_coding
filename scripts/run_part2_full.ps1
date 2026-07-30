Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogDir "part2_full_$Stamp.log"

Start-Transcript -Path $LogPath -Append | Out-Null
try {
    Write-Output "Part 2 full run started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Output "Working directory: $Root"
    Write-Output "Log path: $LogPath"

    Write-Output "Running proposed joint SCB over 18 settings..."
    python simulations\part2_joint.py `
        --n-list 200 300 `
        --m-list 5 10 15 `
        --noise-list gauss uniform laplace `
        --reps 1000 `
        --workers 6

    Write-Output "Running Bonferroni-Cao comparator over 18 settings..."
    python simulations\part2_bonferroni.py `
        --n-list 200 300 `
        --m-list 5 10 15 `
        --noise-list gauss uniform laplace `
        --reps 1000 `
        --workers 8

    Write-Output "Rebuilding consolidated simulation workbook..."
    python simulations\build_excel.py

    Write-Output "Part 2 full run completed: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}
finally {
    Stop-Transcript | Out-Null
}
