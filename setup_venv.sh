#!/usr/bin/env sh
set -eu

# Install uv if it is not available
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"

    if ! command -v uv >/dev/null 2>&1; then
        echo "Failed to install uv."
        exit 1
    fi
fi

echo "Creating virtual environment..."
uv venv .venv

echo "Activating virtual environment..."
if [ -f ".venv/bin/activate" ]; then
    # Linux/macOS
    . .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    # Windows (Git Bash/MSYS/Cygwin)
    . .venv/Scripts/activate
else
    echo "Could not find an activation script."
    exit 1
fi

echo "Installing dependencies..."
uv sync

echo ""
echo "Setup complete!"
echo "Python: $(python --version)"
echo "Environment: $VIRTUAL_ENV"
