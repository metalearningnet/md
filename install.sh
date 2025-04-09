#!/bin/bash

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --model      Install model only"
    echo "  -h, --help   Show this help message"
    exit 0
}

install_model() {
    echo "[INFO] Installing model..."
    if ! python3 scripts/get_model.py; then
        echo "[ERROR] Model installation failed. Exiting..."
        exit 1
    fi
}

MODEL_DIR="model"
MODE=""

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            ;;
        --model)
            MODE="model"
            shift
            ;;
        *)
            echo "[ERROR] Unknown option '$1'. Use -h or --help for usage."
            exit 1
            ;;
    esac
done

case "$MODE" in
    "model")
        install_model
        ;;
    *)
        echo "[INFO] Installing Python packages..."
        PACKAGES=(
            transformers sentencepiece accelerate evaluate tensordict einops einx
            axial_positional_embedding rotary-embedding-torch x-transformers hyper_connections
        )
        if ! pip install "${PACKAGES[@]}"; then
            echo "[ERROR] Python package installation failed. Exiting..."
            exit 1
        fi

        if [ ! -d "$MODEL_DIR" ]; then
            echo "[WARN] Model directory not found. Run '$0 --model' to install the model."
        fi
        ;;
esac

echo "[INFO] Installation completed successfully!"