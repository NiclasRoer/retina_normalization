#!/usr/bin/env bash

# Usage:
#   source new_setup_venv.sh [--help|--deactivate]
#
# Creates a Python 3.12 virtual environment for both Windows 11 and
# WSL Ubuntu using uv, then activates the matching venv layout.

set -euo pipefail

VENV_DIR="${VENV_DIR:-venv}"

print_usage() {
    echo "Usage: source setup_venv.sh"
    echo "Creates and activates a Python 3.12 virtual environment in '$VENV_DIR'."
    echo "Options:"
    echo "  --help, -h       Show this help message."
    echo "  --deactivate, -d Deactivate the virtual environment."
}

ensure_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv not found. Installing it now..."
        if command -v curl >/dev/null 2>&1; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
        else
            echo "Error: curl is required to install uv." >&2
            return 1
        fi

        export PATH="$HOME/.local/bin:$PATH"

        if ! command -v uv >/dev/null 2>&1; then
            echo "Error: uv installation failed." >&2
            return 1
        fi
    fi
}

activate_venv() {
    if [ -f "$VENV_DIR/bin/activate" ]; then
        . "$VENV_DIR/bin/activate"
    elif [ -f "$VENV_DIR/Scripts/activate" ]; then
        . "$VENV_DIR/Scripts/activate"
    else
        echo "Error: Unable to find the activation script in '$VENV_DIR'." >&2
        return 1
    fi
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    print_usage
    exit 0
fi

if [[ "${1:-}" == "--deactivate" || "${1:-}" == "-d" ]]; then
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        deactivate
        echo "Virtual environment deactivated."
    else
        echo "No virtual environment is currently active."
    fi
    exit 0
fi

ensure_uv

if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "A virtual environment is already activated: $VIRTUAL_ENV"
    echo "Python: $(python --version)"
    exit 0
fi

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' already exists. Activating it..."
    activate_venv
else
    echo "Creating virtual environment '$VENV_DIR' with Python 3.12..."
    uv venv --python 3.12 "$VENV_DIR"
    echo "Virtual environment '$VENV_DIR' created successfully. Activating it..."
    activate_venv
fi

if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "Virtual environment '$VENV_DIR' is now active."
    echo "Python: $(python --version)"
    echo "Installing dependencies from pyproject.toml into the active environment..."
    uv sync --active --extra dev
else
    echo "Error: Virtual environment was not activated." >&2
    exit 1
fi

find . -maxdepth 1 -type d -name '*.egg-info' -exec rm -rf -- {} +