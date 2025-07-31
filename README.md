
# Memory Disentangling (MD)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

Memory Disentangling (MD) is a modular framework that enhances large language models (LLMs) by decoupling skill memory from knowledge memory, enabling distinct treatment of task-specific adaptation and general reasoning. The MD module serves as a complementary component integrated upstream of a downstream LLM, delivering disentangled memory representations that inform context-aware generation. Through a meta-learning approach, the skill memory is adaptively updated, facilitating rapid acquisition and refinement of new skills while maintaining stable access to foundational knowledge.

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
