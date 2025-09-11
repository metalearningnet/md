#!/bin/bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
LM_DIR=${LM_DIR:-models/lm}

OS=$(uname -s)
ARCH=$(uname -m)

readonly BOLD=$(tput bold 2>/dev/null || echo "")
readonly RESET=$(tput sgr0 2>/dev/null || echo "")
readonly RED=$(tput setaf 1 2>/dev/null || echo "")
readonly BLUE=$(tput setaf 4 2>/dev/null || echo "")
readonly YELLOW=$(tput setaf 3 2>/dev/null || echo "")

log_info()  { echo -e "${BLUE}${BOLD}[INFO]${RESET} $1" >&1; }
log_warn()  { echo -e "${YELLOW}${BOLD}[WARN]${RESET} $1" >&2; }
log_error() { echo -e "${RED}${BOLD}[ERROR]${RESET} $1" >&2; exit 1; }

usage() {
    cat << 'EOF'
Usage: ./install.sh [OPTIONS]

Install models or dependencies for the MD project.

Model Options:
  --model           Download the language model

Dependency Options:
  --conda           Install required conda packages
  --vllm            Install vLLM (Linux only)
  --flash-attn      Install flash-attn (Linux only)

  -h, --help        Show this help message
EOF
    exit 0
}

check_commands() {
    command -v "$PYTHON" &>/dev/null || log_error "Python interpreter '$PYTHON' not found"
    command -v pip &>/dev/null || log_error "pip not found in PATH"
    if [[ "$INSTALL_CONDA" == true ]]; then
        command -v conda &>/dev/null || log_error "conda not found in PATH"
    fi
}

install_lm() {
    log_info "Downloading language model..."
    if ! "$PYTHON" scripts/download.py --lm; then
        log_error "Language model download failed"
    fi
    if [[ ! -d "$LM_DIR" ]]; then
        log_error "Language model directory '$LM_DIR' not found after download"
    fi
}

install_conda_packages() {
    local packages=()
    case "$OS" in
        Linux)
            packages=(gxx_linux-64 cudatoolkit-dev libaio cmake sentencepiece)
            ;;
        Darwin)
            packages=(cmake sentencepiece)
            ;;
        *)
            log_error "Unsupported OS: $OS"
            ;;
    esac

    if [[ ${#packages[@]} -gt 0 ]]; then
        log_info "Installing conda packages: ${packages[*]}"
        conda install -y -c conda-forge "${packages[@]}" || \
            log_error "Conda package installation failed"
    else
        log_warn "No conda packages to install for $OS"
    fi
}

install_python_packages() {
    local py_packages=(
        torch torchvision torchaudio
        transformers sentencepiece accelerate evaluate
        tensordict einops einx lightning
        axial_positional_embedding rotary-embedding-torch
        x-transformers hyper_connections pyyaml fastapi uvicorn pydantic
        trl peft assoc_scan tensorboard wandb timm
    )

    log_info "Installing core Python dependencies..."
    pip install --no-cache-dir "${py_packages[@]}" || \
        log_error "Failed to install Python packages"

    if [[ "$INSTALL_VLLM" == true ]]; then
        if [[ "$OS" == "Linux" ]]; then
            log_info "Installing vLLM..."
            pip install vllm || log_warn "vLLM installation failed"
        else
            log_warn "vLLM is only supported on Linux. Skipping."
        fi
    fi

    if [[ "$INSTALL_FLASH_ATTN" == true ]]; then
        if [[ "$OS" == "Linux" ]]; then
            log_info "Installing flash-attn..."
            pip install flash-attn --no-build-isolation || log_warn "flash-attn installation failed"
        else
            log_warn "flash-attn is only supported on Linux. Skipping."
        fi
    fi

    if [[ "$OS" == "Linux" && "$ARCH" == "x86_64" ]]; then
        log_info "Installing DeepSpeed..."
        pip install --upgrade deepspeed || log_warn "DeepSpeed installation failed"
    elif [[ "$OS" == "Linux" ]]; then
        log_warn "DeepSpeed is only supported on x86_64. Skipping."
    else
        log_warn "DeepSpeed is not supported on $OS. Skipping."
    fi
}

INSTALL_LM=false
INSTALL_VLLM=false
INSTALL_CONDA=false
INSTALL_FLASH_ATTN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            ;;
        --model)
            INSTALL_LM=true
            ;;
        --conda)
            INSTALL_CONDA=true
            ;;
        --vllm)
            INSTALL_VLLM=true
            ;;
        --flash-attn)
            INSTALL_FLASH_ATTN=true
            ;;
        *)
            log_error "Unknown option: $1"
            ;;
    esac
    shift
done

check_commands

if [[ "$INSTALL_LM" == true ]]; then
    install_lm
fi

if [[ "$INSTALL_CONDA" == true || "$INSTALL_VLLM" == true || "$INSTALL_FLASH_ATTN"  == true || "$INSTALL_LM" == false ]]; then
    log_info "Installing dependencies..."

    if [[ "$INSTALL_CONDA" == true ]]; then
        install_conda_packages
    fi

    install_python_packages

    if [[ ! -d "$LM_DIR" ||  -z "$(ls -A "$LM_DIR")" ]]; then
        install_lm
    fi
fi
