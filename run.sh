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
  --unittest CATEGORY    Run unit tests for specified category (skill|md|ckpt|loader)
  --train                Start the training process
  --evaluate             Start the evaluation process
  --prepare              Prepare the training data
  --generate             Generate evaluation results
  -h, --help             Show this help message

${BOLD}Examples:${RESET}
  $0 --train --samples 1024 --epochs 2 --batch_size 4
  $0 --generate --samples 16
  $0 --unittest md
  $0 --prepare
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

run_unittests() {
    local category="$1"
    shift
    
    case "$category" in
        skill|md|ckpt|loader)
            local script_name="test_${category}.py"
            local script_path="${PROJECT_DIR}/scripts/${script_name}"
            
            log_info "Running unit tests (${category})"
            validate_script "$script_path"
            
            if "$PYTHON" "$script_path" "$@"; then
                log_success "Unit tests passed for ${category}"
            else
                log_error "Unit tests failed for ${category}"
                exit 1
            fi
            ;;
        *)
            log_error "Invalid test category: '${category}'. Valid options: skill|md|loader|ckpt"
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
            --unittest)
                if [[ -z "${2:-}" ]]; then
                    log_error "Missing category for --unittest"
                    show_help
                fi
                local category="$2"
                shift 2
                run_unittests "$category" "$@"
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
