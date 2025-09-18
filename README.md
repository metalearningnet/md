
# Memory Disentangling (MD)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A modular framework for enhancing LLMs through memory separation

## Overview

Memory Disentangling (MD) enhances large language models (LLMs) by introducing an isolated skill memory component that operates alongside the model's inherent knowledge base. As an upstream processor, MD generates specialized memory representations that combine with the original input to enable context-sensitive generation. The approach preserves the base LLM's architecture, maintaining its general knowledge and requiring only minimal parameter adjustments. This architectural separation provides adaptable skill capabilities without compromising foundational model performance.

## Architecture

MD operates as an upstream processor that concatenates with existing LLMs:

```
[Input] → [MD] → [LLM] → [Output]
           │
           └─ Skill Memory
```

## Quick Start

### Installation

#### Install dependencies
```bash
git clone https://github.com/metalearningnet/md.git
cd md
./install.sh
```

#### Install language model
```bash
./install.sh --model
```

#### Install vllm (required for data preparation)
```bash
./install.sh --vllm
```

### Basic Operations

| Command | Flags | Description |
|---------|-------|-------------|
| `./run.sh` | `--train` | Train the model using skill memory |
| | `--evaluate` | Evaluate model performance on standard datasets |
| | `--prepare` | Preprocess and prepare training dataset |
| | `--generate` | Generate sample outputs for qualitative testing |
| | `--build lm` | Convert MD checkpoint into a compatible language model format |

## License

MIT Licensed - See [LICENSE](LICENSE) for details.
