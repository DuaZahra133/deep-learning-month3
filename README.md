# MNIST Digit Recognition Neural Network

## Project Overview
This project demonstrates an **end-to-end neural network workflow** using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The project covers **data preprocessing, model building, training, evaluation, and visualization**, and is designed to showcase a strong foundational deep learning project for your portfolio.

---

## Week 1 Notebooks
| Day | Notebook | Description |
|-----|---------|------------|
| Day 49 | `day49_hello_world_nn.ipynb` | Hello World Neural Network – TensorFlow & Keras setup |
| Day 50 | `day50_mnist_basic.ipynb` | Basic MNIST model using Sequential API |
| Day 51 | `day51_activation_loss.ipynb` | Activation functions & loss functions applied |
| Day 52 | `day52_training_eval.ipynb` | Model training, evaluation, and performance visualization |
| Day 53 | `day53_mnist_digit_recognition_final.ipynb` | Final portfolio-ready MNIST model with predictions |

---

## Dataset
- **MNIST dataset**: 70,000 grayscale images of handwritten digits (0-9)  
- **Training set**: 60,000 images  
- **Test set**: 10,000 images  
- **Image size**: 28x28 pixels, flattened to 784 features for input to the neural network  

---

## Model Architecture
- **Input layer**: 784 neurons (flattened 28x28 image)  
- **Hidden layer 1**: 128 neurons, ReLU activation  
- **Hidden layer 2**: 64 neurons, ReLU activation  
- **Output layer**: 10 neurons, Softmax activation  
- **Optimizer**: Adam  
- **Loss function**: Sparse Categorical Crossentropy  
- **Metrics**: Accuracy  

---

## Training & Evaluation
- **Epochs**: 20  
- **Batch size**: 32  
- **Validation split**: 20% of training data  
- Tracked **training vs validation accuracy and loss**  
- Evaluated final model on **test set** to measure true performance  

---

## Results
- **Test accuracy**: ~95%+  
- Predictions on test images closely match true labels  
- Training and validation curves show smooth convergence  
- Overfitting/underfitting monitored and controlled  

---

