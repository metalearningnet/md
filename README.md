
# Memory Disentangling (MD)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

Memory Disentangling (MD) enhances large language models (LLMs) through a specialized architecture that separates skill memory (for task-specific capabilities) from knowledge memory (for general world knowledge). This modular framework operates as an upstream processor that feeds disentangled memory representations to the main LLM, enabling more focused context-aware generation. The system employs adaptive training techniques to continuously update skill memory while preserving stable knowledge access, allowing for efficient acquisition of new competencies without compromising foundational understanding. By maintaining this clear memory separation, MD provides fine-grained control over how different types of information influence model outputs.

## Quick Start

### Installation

To install dependencies:

```bash
./install.sh
```

### Usage

Run the following commands using the `run.sh` script:

```bash
# Train the model
./run.sh --train

# Test the model
./run.sh --test

# Prepare data
./run.sh --prepare

# Generate evaluation results
./run.sh --generate
```

## License

MIT Licensed - See [LICENSE](LICENSE) for details.
