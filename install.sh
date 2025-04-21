#!/bin/bash
set -euo pipefail

# Configuration
PYTHON=${PYTHON:-python3}
MODEL_DIR=${MODEL_DIR:-model}
OS=$(uname -s)
ARCH=$(uname -m)

# ANSI color codes
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

# Logging functions
log_info()  { echo -e "${GREEN}[INFO]${RESET} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${RESET} $1" >&2; }
log_error() { echo -e "${RED}[ERROR]${RESET} $1" >&2; exit 1; }

# Dependency checks
command -v "$PYTHON" &>/dev/null || log_error "Python interpreter '$PYTHON' not found"
command -v conda &>/dev/null || log_error "conda not found in PATH"
command -v pip &>/dev/null || log_error "pip not found in PATH"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --model      Install model only"
    echo "  -h, --help   Show this help message"
    exit 0
}

install_model() {
    log_info "Installing model..."
    if ! "$PYTHON" scripts/get_model.py; then
        log_error "Model installation failed"
    fi

    [ -d "$MODEL_DIR" ] || log_error "Model directory verification failed"
}

# Argument parsing
MODE=""
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage ;;
        --model) MODE="model"; shift ;;
        *) log_error "Unknown option: $1" ;;
    esac
done

# System-specific configuration
case "$OS" in
    Linux)
        CONDA_PACKAGES=(gxx_linux-64 cudatoolkit-dev)
        DEEPSPEED_SUPPORTED=true
        FLASH_ATTN_SUPPORTED=true
        ;;
    Darwin)
        CONDA_PACKAGES=()
        DEEPSPEED_SUPPORTED=false
        FLASH_ATTN_SUPPORTED=false
        ;;
    *) log_error "Unsupported OS: $OS" ;;
esac

# Main execution
if [ "$MODE" = "model" ]; then
    install_model
else
    log_info "Detected OS: $OS ($ARCH)"

    # Conda installation
    if [ ${#CONDA_PACKAGES[@]} -gt 0 ]; then
        log_info "Installing conda packages..."
        conda install -y -c conda-forge "${CONDA_PACKAGES[@]}" || \
            log_error "Conda installation failed"
    fi

    # Python package installation
    log_info "Installing Python dependencies..."
    PYTHON_PACKAGES=(
        torch torchvision torchaudio
        transformers sentencepiece accelerate evaluate
        tensordict einops einx lightning
        axial_positional_embedding rotary-embedding-torch
        x-transformers hyper_connections pyyaml fastapi uvicorn pydantic
        trl peft vllm assoc_scan
    )

    if $FLASH_ATTN_SUPPORTED; then
        PYTHON_PACKAGES+=("flash-attn")
    else
        log_warn "Skipping flash-attn (requires Linux)"
    fi

    "$PYTHON" -m pip install --no-cache-dir "${PYTHON_PACKAGES[@]}" || \
        log_error "Python package installation failed"

    # DeepSpeed installation
    if $DEEPSPEED_SUPPORTED && [ "$ARCH" = "x86_64" ]; then
        log_info "Installing DeepSpeed..."
        "$PYTHON" -m pip install deepspeed || \
            log_warn "DeepSpeed installation failed (optional)"
    else
        log_warn "Skipping DeepSpeed (requires Linux x86_64)"
    fi

    # Final checks
    [ -d "$MODEL_DIR" ] || log_warn "Model directory not found. Run with --model to install"
fi

log_info "Installation completed successfully!"