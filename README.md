# Fashion MNIST Classifier

## Overview

This project implements a simple neural network classifier for the Fashion MNIST dataset using NumPy, without relying on high-level machine learning frameworks for the model implementation. The classifier is built from scratch with basic mathematical operations to recognize clothing items from the Fashion MNIST dataset.

## Features

- Custom implementation of a softmax classifier using NumPy
- Mini-batch gradient descent optimization
- Data preprocessing and reshaping
- Visual prediction verification
- Accuracy calculation
- Interactive prediction visualization

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- Seaborn
- TensorFlow (only used for loading the Fashion MNIST dataset)

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/fashion-mnist-classifier.git
cd fashion-mnist-classifier
```

2. Install the required dependencies:
```
pip install numpy matplotlib seaborn tensorflow
```

## How It Works

The project follows these key steps:

1. **Data Loading & Preprocessing**: Loads the Fashion MNIST dataset and reshapes the images from 28x28 to 784x1 vectors
2. **Model Initialization**: Creates a simple linear model with weights and biases
3. **Forward Pass**: Implements the forward propagation to compute predictions
4. **Training**: Uses mini-batch gradient descent to train the model
5. **Prediction**: Implements functions to make predictions on test data
6. **Evaluation**: Calculates accuracy and provides visual validation of predictions

## Functions

- `forward_pass(X)`: Performs a linear forward pass
- `soft_max(Z)`: Implements the softmax activation function with numerical stability
- `compute_gradients(X, y_pred, y_true)`: Calculates gradients for backpropagation
- `update_values(Wd, bd, learning_rate)`: Updates model parameters
- `train(X, y, epochs, batch_size)`: Trains the model using mini-batch gradient descent
- `get_pred(X)`: Makes a prediction for a single sample
- `get_whole_pred(X)`: Makes predictions for a batch of samples
- `want_to_know(j, m)`: Visualizes predictions for samples from index j to m
- `accuracy_score(y_pred, y_true)`: Calculates the accuracy of predictions

## Usage

Run the main script to train the model and evaluate its performance:

```python
python fashion_mnist_classifier.py
```

To visualize predictions on specific samples:

```python
# Inside your code
want_to_know(start_index, end_index)  # e.g., want_to_know(10, 20)
```

## Dataset

The Fashion MNIST dataset consists of 70,000 grayscale images of 10 fashion categories, with 60,000 training images and 10,000 test images. Each image is 28x28 pixels.

Categories:
0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

## Performance

The current implementation achieves approximately 80-85% accuracy on the test set without complex architectures or hyperparameter tuning, demonstrating the effectiveness of even simple machine learning models on this dataset.

