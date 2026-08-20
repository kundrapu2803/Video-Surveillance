# Task runner. Every task invokes the 'cctv' conda env interpreter by ABSOLUTE PATH
# rather than relying on `conda activate`, because `conda activate` inside a script
# does not affect the caller's shell -- see plan risk table.

param(
    [Parameter(Position = 0)]
    [string]$Task = "help"
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Py = "$env:USERPROFILE\anaconda3\envs\cctv\python.exe"

function Assert-Env {
    if (-not (Test-Path $Py)) {
        Write-Error "cctv env not found at $Py. Run: .\make.ps1 setup"
        exit 2
    }
}

switch ($Task) {
    "setup" {
        & "$Root\scripts\bootstrap_env.ps1"
    }
    "verify" {
        Assert-Env
        & $Py "$Root\scripts\verify_env.py"
    }
    "test" {
        Assert-Env
        & $Py -m pytest "$Root\tests" -q
    }
    "run" {
        Assert-Env
        & $Py "$Root\run.py" --video "Dataset\mot17\MOT17-09-FRCNN\img1" --zones "configs\zones\mot17-09.json" --config "configs\sources\mot17.yaml" --output results\
    }
    "dash" {
        Assert-Env
        & $Py -m streamlit run "$Root\dashboard\app.py"
    }
    "clean" {
        Remove-Item -Recurse -Force "$Root\results\*" -ErrorAction SilentlyContinue
        Write-Host "Cleared results\."
    }
    default {
        Write-Host "Usage: .\make.ps1 <setup|verify|test|run|dash|clean>"
    }
}
