
# Memory Disentangling (MD)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

Memory Disentangling (MD) is a modular framework that structurally separates long-term knowledge preservation from context-specific skill adaptation in large language models. The architecture enables:

- 🧠 **Compartmentalized Memory Systems**  
  *Persistent memory* retains fundamental world knowledge (facts, relationships) while *dynamic memory* encodes temporary task-specific operational patterns, eliminating cross-memory interference

- 🔄 **Meta-Learning**  
  Enables continuous skill refinement through localized updates to dynamic memory parameters, preserving core knowledge integrity

- 📈 **Test-Time Scaling**  
  Adapts to variable-length sequences through optimized memory utilization

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
