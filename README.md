# MNIST Digit Classifier with PyTorch

A lightweight, custom Convolutional Neural Network (CNN) built from scratch using PyTorch to classify handwritten digits from the classic MNIST dataset. The pipeline handles data normalization, batch training, training metric plotting, and checkpoint saving.

---

## 📌 About the Project

This project implements an end-to-end deep learning pipeline for handwriting recognition. Utilizing a custom CNN architecture, the model maps 28x28 grayscale pixel grids to their corresponding numerical digits (0–9). 

### Key Features
* **Custom CNN Architecture:** A lean network utilizing two sequential convolutional layers for fast spatial feature extraction.
* **On-the-Fly Standardization:** Normalizes pixel intensity distributions directly from the data tensors to achieve zero mean and unit variance, leading to stable gradient descent.
* **Checkpoint System:** Automatically saves model weights, optimizer states, and epoch metrics into a `checkpoint.pth` file at the end of every epoch.
* **Visual Evaluation:** Includes testing scripts paired with `matplotlib` outputs to easily verify individual test images alongside the model's predicted labels.

---

## 🏗️ Model Architecture

The model processes standard 28x28 single-channel inputs through two convolutional steps before flattening into a dense layer for classification:

* **Input Layer:** 1 x 28 x 28 (Grayscale image)
* **Convolutional Layer 1:** `nn.Conv2d(1, 20, kernel_size=5)` followed by a ReLU activation function. (Output: 20 x 24 x 24)
* **Convolutional Layer 2:** `nn.Conv2d(20, 20, kernel_size=5)` followed by a ReLU activation function. (Output: 20 x 20 x 20)
* **Flattening:** Reshapes the tensor dimensions into an 8,000-element vector.
* **Linear Output Layer:** `nn.Linear(8000, 10)` maps features directly to the 10 digit classes.

---

## 📈 Performance & Results

The network was trained for 10 epochs using the **Adam Optimizer** (Learning Rate = 0.001) and **Cross-Entropy Loss**, showing rapid and smooth convergence:

| Epoch | Average Loss |
|-------|--------------|
| 1     | 0.1567       |
| 2     | 0.0537       |
| 3     | 0.0373       |
| 4     | 0.0274       |
| 5     | 0.0189       |
| 6     | 0.0147       |
| 7     | 0.0134       |
| 8     | 0.0093       |
| 9     | 0.0098       |
| 10    | 0.0088       |

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have the required Python libraries installed:
```bash
pip install pandas torch matplotlib numpy
