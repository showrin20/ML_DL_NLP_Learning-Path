# Neural Network Models for MNIST and Breast Cancer Wisconsin Datasets

This repository contains implementations and outcomes of neural network models trained on two different datasets as part of coursework. The models were developed using Python and PyTorch.

## MNIST Dataset

### Models Implemented

1. **Fully-Connected Neural Network (FCNN)**
   - Architecture:
     - Input Layer: 784 nodes (flattened MNIST images of size 28x28)
     - Hidden Layers: 256 nodes (ReLU activation), 128 nodes (ReLU activation)
     - Output Layer: 10 nodes (softmax activation for 10 digit classes)
   - Training:
     - Batch Size: 64
     - Epochs: 15
   - Performance:
     - Test Accuracy: 96.21%
     - Loss Reduction: Consistent improvement across epochs

2. **Convolutional Neural Network (CNN)**
   - Architecture:
     - Convolutional Layers: 32 filters, 64 filters (ReLU activation, followed by max-pooling)
     - Fully Connected Layer: 128 nodes (ReLU activation)
     - Output Layer: 10 nodes (softmax activation)
   - Training:
     - Batch Size: 64
     - Epochs: 15
   - Performance:
     - Test Accuracy: 99.08%
     - Loss Behavior: Rapid decrease and stabilization at lower values

### Conclusion
The CNN model outperformed the FCNN in terms of accuracy on the MNIST dataset.

## Breast Cancer Wisconsin Dataset

### Models Implemented

1. **Simple Neural Network (SNN)**
   - Architecture:
     - Input Layer: 30 features
     - Fully Connected Layers: 16 nodes (ReLU activation), 2 nodes (log-softmax activation)
   - Training:
     - Batch Size: 64
     - Epochs: 15
   - Performance:
     - Evaluation Metric: Cross-entropy loss
     - Optimizer: Adam

2. **Complex Neural Network (CNN)**
   - Architecture:
     - Fully Connected Layers: 64 nodes (ReLU activation), 32 nodes (ReLU activation), 2 nodes (log-softmax activation)
   - Training:
     - Batch Size: 64
     - Epochs: 15
   - Performance:
     - Evaluation Metric: Cross-entropy loss
     - Optimizer: Adam

### Conclusion
The performance of the neural network models varied on the Breast Cancer Wisconsin dataset, with [add conclusion here based on outcomes].

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- NumPy
- Matplotlib (for visualization, optional)

### Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/neural-network-models.git
cd neural-network-models
