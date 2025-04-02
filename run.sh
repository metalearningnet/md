#!/bin/bash

PROJECT_DIR=$(dirname "$(readlink -f "$0")")
PYTHON=python3

usage() {
    echo "Usage: $0 [--unittest [skill|md|loader|ckpt]] [--train] [--test]"
    echo ""
    echo "Options:"
    echo "  --unittest [skill|md|loader|ckpt]   Run unit tests for the specified category (e.g., skill, md, loader, or ckpt)."
    echo "                                   Example: $0 --unittest skill"
    echo "  --train                        Start the training process."
    echo "                                   Example: $0 --train --name ag_news"
    echo "  --test                         Start the testing process."
    echo "                                   Example: $0 --test"
    echo ""
    exit 1
}

run_unittest() {
    subcommand=$1
    case "$subcommand" in
        skill|md|loader|ckpt)
            script="test_${subcommand}.py"
            shift
            echo "[INFO] Running unit tests for category: ${subcommand}"
            if [[ -f "$PROJECT_DIR/scripts/$script" ]]; then
                $PYTHON "$PROJECT_DIR/scripts/$script" "$@"
            else
                echo "[ERROR] Unit test script '$script' not found in '$PROJECT_DIR/scripts/'."
                exit 1
            fi
            ;;
        *)
            echo "[ERROR] Unknown unit test category: '$subcommand'"
            usage
            ;;
    esac
}

case "$1" in
    --unittest)
        shift
        if [[ -z "$1" ]]; then
            echo "[ERROR] Missing subcommand for '--unittest'."
            usage
        fi
        run_unittest "$@"
        ;;

    --train)
        shift
        echo "[INFO] Starting the training process..."
        if [[ -f "$PROJECT_DIR/scripts/train.py" ]]; then
            $PYTHON "$PROJECT_DIR/scripts/train.py" "$@"
        else
            echo "[ERROR] Training script 'train.py' not found in '$PROJECT_DIR/scripts/'."
            exit 1
        fi
        ;;

    --test)
        shift
        echo "[INFO] Starting the testing process..."
        if [[ -f "$PROJECT_DIR/scripts/test.py" ]]; then
            $PYTHON "$PROJECT_DIR/scripts/test.py" "$@"
        else
            echo "[ERROR] Testing script 'test.py' not found in '$PROJECT_DIR/scripts/'."
            exit 1
        fi
        ;;

    *)
        usage
        ;;
esac