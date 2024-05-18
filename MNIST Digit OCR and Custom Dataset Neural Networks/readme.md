# MNIST Digit OCR and Custom Dataset Neural Networks

This repository contains implementations of two neural network models trained on the MNIST dataset and two neural network models trained on a custom dataset. The goal is to compare the performance of different architectures on image classification tasks.

## Prerequisites

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib

## 1. MNIST Digit OCR

### Data Preparation

Download the MNIST dataset from `torchvision.datasets.MNIST` and use PyTorch's data loader to create training and test sets. The data is segmented into minibatches of size 64.

### Model 1: Fully-Connected Neural Network

- **Architecture:**
  - Input Layer: 784 input features
  - Hidden Layer 1: 256 nodes with ReLU activation
  - Hidden Layer 2: 128 nodes with ReLU activation
  - Output Layer: 10 units with softmax activation

### Model 2: Convolutional Neural Network

- **Architecture:**
  - Convolutional Layer 1: 32 filters with ReLU activation, followed by max-pooling
  - Convolutional Layer 2: 64 filters with ReLU activation, followed by max-pooling
  - Fully Connected Layer: 128 nodes with ReLU activation
  - Output Layer: 10 units with softmax activation

### Training

Both models were trained using a batch size of 64 over 15 epochs. The training process was monitored by observing the loss at each epoch, and the models' performance was evaluated using accuracy on the test dataset.

### Evaluation

- **Fully-Connected Neural Network:** Test accuracy of 96.21%. The model showed consistent improvement in loss reduction across epochs.
- **Convolutional Neural Network:** Test accuracy of 99.08%. The CNN model demonstrated rapid loss decrease, stabilizing at lower values faster than the fully-connected model.

The CNN outperformed the fully-connected network in terms of accuracy.

## 2. Custom Dataset Neural Networks

### Dataset Selection

The second part explores neural network architectures on a custom dataset, specifically the Breast Cancer Wisconsin (Diagnostic) dataset. The dataset contains 30 features and 2 classes (malignant and benign).

### Model 1: Simple Neural Network (SNN)

- **Architecture:**
  - Input Layer: 30 features
  - Hidden Layer: 16 nodes with ReLU activation
  - Output Layer: 2 nodes with log-softmax activation

### Model 2: Complex Neural Network (CNN)

- **Architecture:**
  - Hidden Layer 1: 64 nodes with ReLU activation
  - Hidden Layer 2: 32 nodes with ReLU activation
  - Output Layer: 2 nodes with log-softmax activation

### Training

The models were trained over 15 epochs with a batch size of 64, using cross-entropy loss and Adam optimizer.

### Evaluation

- **Simple Neural Network:** Accuracy of 96.49%.
- **Complex Neural Network:** Accuracy of 99.12%.

The complex model showed better performance, likely due to its increased depth and capacity to capture more complex patterns in the data. This demonstrates the potential benefit of deeper architectures in handling datasets with complex feature relationships.

