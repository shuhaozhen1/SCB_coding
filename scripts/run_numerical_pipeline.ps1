param(
    [switch]$Full,
    [int]$Workers = 6,
    [int]$BonferroniWorkers = 8
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$Reps = if ($Full) { 1000 } else { 20 }

Write-Output "Running numerical pipeline from $Root"
Write-Output "Replicates per simulation cell: $Reps"

python simulations\part1_vcm.py --reps $Reps --workers $Workers
python simulations\part2_joint.py --reps $Reps --workers $Workers
python simulations\part2_bonferroni.py --reps $Reps --workers $BonferroniWorkers
python simulations\part3_mean.py --reps $Reps --workers $Workers

python realdata\pbc_data_illustration.py
python realdata\pbc_mean.py
python realdata\pbc_vcm.py

python simulations\gen_plot_curves.py
python simulations\plot_bands.py
python simulations\build_excel.py

Write-Output "Numerical pipeline completed."
