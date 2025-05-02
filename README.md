# MNIST Digit Recognition with Multilayer Perceptron 🧠

This project implements a Multilayer Perceptron (MLP) neural network from scratch using PyTorch to recognize handwritten digits from the MNIST dataset.

## Project Overview 📊

The MNIST dataset is a collection of 70,000 grayscale images of handwritten digits (0-9). Each image is 28x28 pixels, resulting in 784 features when flattened. This project demonstrates how to build, train, and evaluate a simple neural network to classify these digits with impressive accuracy.

## Implementation Details ⚙️

### Data Loading and Preprocessing
- The MNIST dataset is loaded using PyTorch's `torchvision.datasets` module
- Images are transformed into tensors using `ToTensor()`
- Data is batched for efficient training using `DataLoader`

### Neural Network Architecture
- Input Layer: 784 neurons (28x28 flattened image)
- Hidden Layer: 256 neurons with ReLU activation
- Output Layer: 10 neurons (one for each digit)

```python
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_relu = F.relu(self.linear1(x))
        y_pred = self.linear2(h_relu)
        return y_pred
```

### Training Process
- Optimizer: Adam with learning rate 1e-6
- Loss Function: Cross Entropy Loss
- Training for multiple epochs to achieve convergence

## Results 📈

The model achieves over 90% accuracy on the test set after 20 epochs of training, which is impressive for such a simple architecture. The loss curve shows consistent improvement over time:

- 2 epochs: ~79.5% accuracy
- 20 epochs: ~90.9% accuracy

### Performance Analysis
- Loss decreases dramatically in the first few epochs
- The model learns more slowly in later epochs but continues to improve
- Some batches show higher loss values, indicating challenging examples

## Requirements 🔧

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- numpy

## Usage 🚀

1. Clone this repository
2. Install the required dependencies:
```
pip install torch torchvision matplotlib numpy
```
3. Run the Jupyter notebook:
```
jupyter notebook mlp.ipynb
```

## Future Improvements 🔮

Several enhancements could be made to improve model performance:

1. Add more hidden layers to create a deeper network
2. Implement dropout for regularization
3. Try different activation functions
4. Experiment with different optimizers and learning rates
5. Apply data augmentation techniques
6. Implement early stopping to prevent overfitting

## Acknowledgments 🙏

This project was developed as part of the Machine Learning course at NYU Paris, Summer 2023, under the guidance of Guillaume Staerman.

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.
