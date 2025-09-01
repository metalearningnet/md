#!/bin/bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
MODEL_DIR=${MODEL_DIR:-model}
OS=$(uname -s)
ARCH=$(uname -m)

readonly BOLD=$(tput bold)
readonly RED=$(tput setaf 1)
readonly YELLOW=$(tput setaf 3)
readonly BLUE=$(tput setaf 4)
readonly RESET=$(tput sgr0)

log_info()  { echo -e "${BLUE}${BOLD}[INFO]${RESET} $1"; }
log_warn()  { echo -e "${YELLOW}${BOLD}[WARN]${RESET} $1" >&2; }
log_error() { echo -e "${RED}${BOLD}[ERROR]${RESET} $1" >&2; exit 1; }

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]
Options:
  --conda         Install required conda packages
  --model         Install model only (ignores other options)
  --vllm          Enable installation of vLLM
  --flash_attn    Enable installation of flash attion
  -h, --help      Show this help message
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

install_model_only() {
    log_info "Installing model..."
    if ! "$PYTHON" scripts/download.py --model; then
        log_error "Model installation failed"
    fi

    [ -d "$MODEL_DIR" ] || log_error "Model directory verification failed"
}

ARGS=()
MODE=""

ENABLE_VLLM=false
ENABLE_FLASH_ATTN=false

INSTALL_VLLM=false
INSTALL_CONDA=false
INSTALL_FLASH_ATTN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage ;;
        --conda) INSTALL_CONDA=true; ARGS+=("$1"); shift ;;
        --model) MODE="model"; shift ;;
        --vllm) ENABLE_VLLM=true; ARGS+=("$1"); shift ;;
        --flash_attn) ENABLE_FLASH_ATTN=true; ARGS+=("$1"); shift ;;
        *) log_error "Unknown option: $1" ;;
    esac
done

check_commands

if [[ "$ENABLE_VLLM" == true && "$OS" == "Linux" ]]; then
    INSTALL_VLLM=true
elif [[ "$ENABLE_VLLM" == true && "$OS" != "Linux" ]]; then
    log_warn "vLLM can only be installed on Linux. Ignoring --vllm."
fi

if [[ "$ENABLE_FLASH_ATTN" == true && "$OS" == "Linux" ]]; then
    INSTALL_FLASH_ATTN=true
elif [[ "$ENABLE_FLASH_ATTN" == true && "$OS" != "Linux" ]]; then
    log_warn "flash-attn can only be installed on Linux. Ignoring --flash_attn."
fi

case "$OS" in
    Linux)
        CONDA_PACKAGES=(gxx_linux-64 cudatoolkit-dev libaio cmake sentencepiece)
        DEEPSPEED_SUPPORTED=true
        ;;
    Darwin)
        CONDA_PACKAGES=(cmake sentencepiece)
        DEEPSPEED_SUPPORTED=false
        ;;
    *)
        log_error "Unsupported platform: $OS"
        ;;
esac

if [[ "$MODE" == "model" ]]; then
    install_model_only
else
    log_info "Platform: $OS ($ARCH)"

    if [[ "$INSTALL_CONDA" == true && ${#CONDA_PACKAGES[@]} -gt 0 ]]; then
        log_info "Installing conda packages..."
        conda install -y -c conda-forge "${CONDA_PACKAGES[@]}" || \
            log_error "Conda installation failed"
    else
        log_warn "Skipping conda packages"
    fi

    log_info "Installing Python dependencies..."
    PYTHON_PACKAGES=(
        torch torchvision torchaudio
        transformers sentencepiece accelerate evaluate
        tensordict einops einx lightning
        axial_positional_embedding rotary-embedding-torch
        x-transformers hyper_connections pyyaml fastapi uvicorn pydantic
        trl peft assoc_scan tensorboard wandb timm
    )

    pip install --no-cache-dir "${PYTHON_PACKAGES[@]}" || \
        log_error "Python package installation failed"

    if [[ "$INSTALL_VLLM" == true ]]; then
        log_info "Installing vLLM..."
        pip install vllm || log_warn "vLLM installation failed"
    else
        log_warn "Skipping vLLM"
    fi

    if [[ "$INSTALL_FLASH_ATTN" == true ]]; then
        log_info "Installing flash-attn..."
        pip install flash-attn || log_warn "flash-attn installation failed"
    else
        log_warn "Skipping flash-attn"
    fi

    if [[ "$DEEPSPEED_SUPPORTED" == true && "$ARCH" == "x86_64" ]]; then
        log_info "Installing DeepSpeed..."
        pip install --upgrade deepspeed || log_warn "DeepSpeed installation failed"
    else
        log_warn "Skipping DeepSpeed"
    fi

    if [[ ! -d "$MODEL_DIR" ]]; then
        log_warn "Model directory not found. Run with --model to install it."
    fi
fi
