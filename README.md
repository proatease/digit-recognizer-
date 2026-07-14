# Handwritten Digit Recognition using CNN (PyTorch)

A Convolutional Neural Network (CNN) built with PyTorch to classify handwritten digits (0–9) from grayscale images. The model is trained on the Kaggle Digit Recognizer dataset (MNIST format) and achieves low training loss through end-to-end supervised learning.

## Project Overview

This project demonstrates the complete deep learning workflow:

* Loading and preprocessing image data
* Normalizing pixel values
* Building a CNN architecture using PyTorch
* Training with the Adam optimizer
* Monitoring training loss
* Saving model checkpoints
* Predicting unseen test images
* Visualizing training performance

The model learns to recognize handwritten digits from 28×28 grayscale images and outputs one of 10 digit classes.

---

## Dataset

The project uses the Kaggle Digit Recognizer dataset:

* **train.csv**

  * First column: Digit label (0–9)
  * Remaining 784 columns: Pixel values

* **test.csv**

  * Contains only pixel values
  * Used for inference and prediction

Image dimensions:

```
28 × 28 × 1
```

---

## Data Preprocessing

The following preprocessing steps are performed:

1. Load data using Pandas
2. Convert data into PyTorch tensors
3. Compute dataset mean and standard deviation
4. Apply normalization:

```python
X = (X - mean) / std
```

5. Reshape flattened vectors into image format:

```python
X = X.view(count, 1, 28, 28)
```

---

## CNN Architecture

```text
Input Image (1 × 28 × 28)
        │
        ▼
Conv2D (1 → 20, kernel=5)
        │
      ReLU
        │
        ▼
Conv2D (20 → 20, kernel=5)
        │
      ReLU
        │
        ▼
Flatten
        │
        ▼
Linear (8000 → 10)
        │
        ▼
Digit Prediction
```

### Model Definition

```python
Conv2d(1, 20, 5)
Conv2d(20, 20, 5)
Linear(8000, 10)
```

---

## Training Configuration

| Parameter     | Value            |
| ------------- | ---------------- |
| Optimizer     | Adam             |
| Learning Rate | 0.001            |
| Batch Size    | 64               |
| Epochs        | 10               |
| Loss Function | CrossEntropyLoss |

---

## Training Results

| Epoch | Average Loss |
| ----- | ------------ |
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

The loss consistently decreases during training, indicating successful learning and convergence.

---

## Model Checkpointing

After every epoch, the model state and optimizer state are saved:

```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': average_loss
}, "checkpoint.pth")
```

This allows training to be resumed later without losing progress.

---

## Example Prediction

```python
Predicted label for the 99th test image: 4
```

The trained model can classify unseen handwritten digit images and return the predicted digit class.

---

## Loss Visualization

Training loss is plotted across epochs using Matplotlib:

```python
plt.plot(range(1, 11), losswatch, marker='o')
```

This provides a simple way to monitor model convergence and training performance.

---

## Technologies Used

* Python
* PyTorch
* Pandas
* NumPy
* Matplotlib

---

## Future Improvements

* Add MaxPooling layers
* Add Batch Normalization
* Introduce Dropout for regularization
* Evaluate model accuracy on a validation set
* Generate Kaggle submission files automatically
* Experiment with deeper CNN architectures
* Implement GPU training support

---

## Learning Outcomes

Through this project, I gained practical experience with:

* Convolutional Neural Networks (CNNs)
* PyTorch model development
* Image preprocessing
* Deep learning training workflows
* Model checkpointing
* Performance visualization
* Computer vision fundamentals

---

## Author

**Pratisthit Raj Baral**

Computer Engineering Student | Machine Learning & Computer Vision Enthusiast

GitHub: https://github.com/proatease
