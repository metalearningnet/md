# Memory Disentangling

## Overview  
MD (Memory Disentangling) is a neural network architecture designed to enhance cognitive behaviors by integrating meta-learning capabilities into a dedicated memory system. The model focuses on organizing and utilizing skill knowledge , which is established through meta-learning. It supports efficient adaptation to dynamic and complex information encountered at test time, enabled by its test-time scaling capabilities.

## Installation
To install dependencies:
```bash
./install.sh
```
## Training
To train the MD model:
```bash
./run.sh --train
```

## Testing
To run the test suite:
```bash
./run.sh --test
```

## Key Features
- 🧠 **Disentangled Memory Architecture**:  
  Separate modules for persistent memory (task knowledge) and dynamic memory (skill trajectories)
- ⚡ **Efficient Memory Updates**:  
  Momentum-based gradient updates for stable long-term memorization
- 🔄 **Modular Integration**:  
  Compatible with existing Transformer architectures via the MAC (Memory as Context) module
- 📈 **Test-Time Scaling**:  
  Performance improves with longer input sequences through adaptive memory utilization

## License
This project is licensed under the **MIT License** - see [LICENSE](LICENSE) for details.  