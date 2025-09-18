#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_DIR=$(dirname "$(readlink -f "$0")")
readonly PYTHON=python3

readonly BOLD=$(tput bold)
readonly RED=$(tput setaf 1)
readonly GREEN=$(tput setaf 2)
readonly YELLOW=$(tput setaf 3)
readonly BLUE=$(tput setaf 4)
readonly RESET=$(tput sgr0)

log_info() {
    echo "${BLUE}${BOLD}[INFO]${RESET} $*"
}

log_success() {
    echo "${GREEN}${BOLD}[SUCCESS]${RESET} $*"
}

log_warning() {
    echo "${YELLOW}${BOLD}[WARNING]${RESET} $*" >&2
}

log_error() {
    echo "${RED}${BOLD}[ERROR]${RESET} $*" >&2
}

show_help() {
    cat <<EOF
${BOLD}Usage:${RESET} $0 [OPTION]... [PYTHON_ARGS...]

${BOLD}Options:${RESET}
  --prepare              Prepare the training data
  --train                Start the training process
  --evaluate             Start the evaluation process
  --generate             Generate sample outputs for qualitative testing
  --test CATEGORY        Run unit tests for specified category (skill|md|ckpt|loader)
  --build ARG            Build model components (e.g., 'lm' to convert to language model)
  
  -h, --help             Show this help message

${BOLD}Examples:${RESET}
  $0 --prepare
  $0 --train --examples 1024 --epochs 2 --batch-size 4
  $0 --evaluate --examples 128
  $0 --generate --examples 16
  $0 --test md
  $0 --build lm
EOF
    exit 0
}

validate_script() {
    local script_path="$1"
    if [[ ! -f "$script_path" ]]; then
        log_error "Script not found: ${script_path}"
        exit 1
    fi
}

run_tests() {
    local category="$1"
    shift
    
    case "$category" in
        skill|md|ckpt|loader)
            local script_name="test_${category}.py"
            local script_path="${PROJECT_DIR}/scripts/${script_name}"
            
            log_info "Running unit tests for ${category}"
            validate_script "$script_path"
            
            if "$PYTHON" "$script_path" "$@"; then
                log_success "Unit tests passed for ${category}"
            else
                log_error "Unit tests failed for ${category}"
                exit 1
            fi
            ;;
        *)
            log_error "Invalid test category: '${category}'. Valid options: skill|md|ckpt|loader"
            show_help
            ;;
    esac
}

run_training() {
    local script_path="${PROJECT_DIR}/scripts/train.py"
    log_info "Starting training process..."
    validate_script "$script_path"
    
    if "$PYTHON" "$script_path" "$@"; then
        log_success "Training completed successfully"
    else
        log_error "Training failed"
        exit 1
    fi
}

run_evaluation() {
    local script_path="${PROJECT_DIR}/scripts/evaluate.py"
    log_info "Starting evaluation process..."
    validate_script "$script_path"
    
    if "$PYTHON" "$script_path" "$@"; then
        log_success "Evaluation completed successfully"
    else
        log_error "Evaluation failed"
        exit 1
    fi
}

run_build() {
    local build_arg="$1"
    shift

    if [[ -z "$build_arg" ]]; then
        log_error "Missing argument for --build. Expected: 'lm'"
        show_help
    fi

    local script_path="${PROJECT_DIR}/scripts/build.py"
    log_info "Running build process: ${build_arg}"
    validate_script "$script_path"

    if "$PYTHON" "$script_path" "$build_arg" "$@"; then
        log_success "Build completed successfully: ${build_arg}"
    else
        log_error "Build failed: ${build_arg}"
        exit 1
    fi
}

prepare_data() {
    local script_path="${PROJECT_DIR}/scripts/prepare.py"
    log_info "Preparing training data..."
    
    if "$PYTHON" "$script_path"; then
        log_success "Data preparation completed successfully"
    else
        log_error "Data preparation failed"
        exit 1
    fi
}

generate_results() {
    local script_path="${PROJECT_DIR}/scripts/generate.py"
    log_info "Generating evaluation results..."
    validate_script "$script_path"
    
    if "$PYTHON" "$script_path" "$@"; then
        log_success "Evaluation results generated successfully"
    else
        log_error "Failed to generate evaluation results"
        exit 1
    fi
}

main() {
    if [[ $# -eq 0 ]]; then
        log_error "No arguments provided"
        show_help
    fi

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --test)
                if [[ -z "${2:-}" ]]; then
                    log_error "Missing category for --test"
                    show_help
                fi
                local category="$2"
                shift 2
                run_tests "$category" "$@"
                return $?
                ;;
            --train)
                shift
                run_training "$@"
                return $?
                ;;
            --evaluate)
                shift
                run_evaluation "$@"
                return $?
                ;;
            --prepare)
                shift
                prepare_data
                return $?
                ;;
            --generate)
                shift
                generate_results "$@"
                return $?
                ;;
            --build)
                if [[ -z "${2:-}" ]]; then
                    log_error "Missing argument for --build"
                    show_help
                fi
                local build_arg="$2"
                shift 2
                run_build "$build_arg" "$@"
                return $?
                ;;
            -h|--help)
                show_help
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                ;;
        esac
    done
}

main "$@"
