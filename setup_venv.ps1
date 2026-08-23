param(
    [switch]$Deactive
)

$ErrorActionPreference = 'Stop'
$VenvDir = if ($env:VENV_DIR) { $env:VENV_DIR } else { '.venv' }
$PytorchIndex = 'https://download.pytorch.org/whl/cu128'

function Show-Usage {
    Write-Host "Usage: . .\setup_venv.ps1"
    Write-Host "Creates and activates a Python 3.12 virtual environment in '$VenvDir'."
    Write-Host "Options:"
    Write-Host "  -Deactive   Deactivate the virtual environment."
}

if ($Deactive) {
    if ($env:VIRTUAL_ENV) {
        deactivate
        Write-Host 'Virtual environment deactivated.'
    } else {
        Write-Host 'No virtual environment is currently active.'
    }
    return
}

if ($args.Count -gt 0 -and ($args[0] -eq '-h' -or $args[0] -eq '--help')) {
    Show-Usage
    return
}

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw 'Python launcher `py` was not found in PATH. Install Python 3.12 and retry.'
}

$versionCheck = (py -3.12 -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null)
if ($LASTEXITCODE -ne 0 -or $versionCheck.Trim() -ne '3.12') {
    throw 'Python 3.12 is required, but it could not be resolved with `py -3.12`.'
}

if ($env:VIRTUAL_ENV) {
    Write-Host "A virtual environment is already activated: $env:VIRTUAL_ENV"
    python --version
    return
}

if (Test-Path "$VenvDir\Scripts\Activate.ps1") {
    Write-Host "Virtual environment '$VenvDir' already exists. Activating it..."
    . "$VenvDir\Scripts\Activate.ps1"
} else {
    Write-Host "Creating virtual environment '$VenvDir' with Python 3.12..."
    py -3.12 -m venv $VenvDir
    Write-Host "Virtual environment '$VenvDir' created successfully. Activating it..."
    . "$VenvDir\Scripts\Activate.ps1"
}

if ($env:VIRTUAL_ENV) {
    Write-Host "Virtual environment '$VenvDir' is now active."
    python --version
    Write-Host 'Installing dependencies from pyproject.toml into the active environment...'
    uv sync --active --extra dev
    Write-Host "Installing CUDA-enabled PyTorch from $PytorchIndex..."
    uv pip install --reinstall 'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128' --index-url $PytorchIndex --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
} else {
    throw "Virtual environment was not activated."
}

Get-ChildItem -Directory -Filter '*.egg-info' |
    Remove-Item -Recurse -Force