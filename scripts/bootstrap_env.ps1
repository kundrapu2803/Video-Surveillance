# Creates a fresh 'cctv' conda env and installs everything, in order.
# Idempotent: safe to re-run. The torch import gate must pass before anything
# else installs, because a broken torch (see diagnose_torch.ps1) wastes an
# entire pip resolve on top of an unusable interpreter.

$ErrorActionPreference = "Stop"

Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force

& "$env:USERPROFILE\anaconda3\shell\condabin\conda-hook.ps1"

$envExists = (conda env list) -match "^\s*cctv\s"
if (-not $envExists) {
    conda create -y -n cctv python=3.11 pip
} else {
    Write-Host "conda env 'cctv' already exists, reusing it."
}

conda activate cctv

python -m pip install --upgrade pip setuptools wheel

# CPU torch FIRST, from the CPU-only index. --index-url (not --extra-index-url)
# so PyPI can never supply a CUDA build as a fallback. torchaudio is deliberately
# NOT installed here — see diagnose_torch.ps1 for why.
python -m pip install --index-url https://download.pytorch.org/whl/cpu `
    --no-cache-dir torch==2.13.0 torchvision==0.28.0

# THE GATE. Nothing else installs until this prints a version and a float.
python -c "import torch; print(torch.__version__, torch.rand(4,4).sum().item())"
if ($LASTEXITCODE -ne 0) {
    Write-Error "TORCH GATE FAILED. Run scripts\diagnose_torch.ps1 and fix this before continuing."
    exit 1
}

python -m pip install -r requirements.txt -c constraints.txt
python -m pip install -e .

conda env config vars set -n cctv YOLO_AUTOINSTALL=false PYTHONUTF8=1 OMP_NUM_THREADS=4
conda deactivate
conda activate cctv

python scripts\verify_env.py
python scripts\fetch_weights.py

Write-Host ""
Write-Host "Setup complete. Activate this env in future shells with:"
Write-Host '  & "$env:USERPROFILE\anaconda3\shell\condabin\conda-hook.ps1"; conda activate cctv'
