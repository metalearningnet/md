
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

```bash
git clone https://github.com/metalearningnet/md.git
cd md
./install.sh
```

### Basic Operations

| Command | Flags | Description |
|---------|-------|-------------|
| `./run.sh` | `--train` | Train with specified skills |
| | `--test` | Run benchmark evaluation |
| | `--prepare` | Preprocess dataset |
| | `--generate` | Interactive generation |

## License

MIT Licensed - See [LICENSE](LICENSE) for details.
