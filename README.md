# Memory Disentangling

## Overview  
MD (Memory Disentangling) is a neural network architecture designed to enhance cognitive behaviors by integrating meta-learning capabilities into a dedicated memory system. The model focuses on organizing and utilizing skill knowledge , which is established through meta-learning. It supports efficient adaptation to dynamic and complex information encountered at test time, enabled by its test-time scaling capabilities.

## Installation
To install dependencies, run:
```bash
./install.sh
```
## Training
To train the MD model:
```bash
./run.sh --train
```

## Evaluation
To evaluate the model using the test suite:
```bash
./run.sh --test
```

## Data Preparation 
To prepare the training data:
```bash
./run.sh --prepare
```

## Key Features  
- 🧠 **Disentangled Memory Architecture**  
  Separate modules for *persistent memory* (retains general task knowledge) and *dynamic memory* (tracks evolving skill trajectories)  

- ⚡ **Efficient Memory Updates**  
  Uses gradient descent with momentum for fast and stable updates to the neural skill memory  

- 📈 **Test-Time Scaling**  
  Adapts to longer and more complex input sequences through effective memory utilization

## License
This project is licensed under the **MIT License** - see [LICENSE](LICENSE) for details.