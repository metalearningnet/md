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
    python3 scripts/get_model.py
    if [ $? -ne 0 ]; then
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
            usage
            exit 1
            ;;
    esac
done

case "$MODE" in
    "model")
        install_model
        ;;
    *)
        if [ ! -d $MODEL_DIR ]; then
            echo "[INFO] Model directory not found. Installing model first..."
            install_model
        fi

        echo "[INFO] Installing Python packages..."
        pip install transformers
        pip install sentencepiece
        pip install accelerate
        pip install evaluate
        pip install tensordict
        pip install einops
        pip install einx
        pip install axial_positional_embedding
        pip install rotary-embedding-torch
        pip install x-transformers
        pip install hyper_connections
        
        if [ $? -ne 0 ]; then
            echo "[ERROR] Python package installation failed. Exiting..."
            exit 1
        fi
        ;;
esac

echo "[INFO] Installation completed successfully!"
