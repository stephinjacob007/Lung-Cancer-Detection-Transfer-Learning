# 🫁 Lung Cancer Detection from CT Scan Images using Transfer Learning

## 📌 Overview

This project focuses on binary classification of lung cancer from chest CT scan images using transfer learning with multiple pre-trained Convolutional Neural Networks (CNNs).

The objective is to classify CT scan images into:

* **Positive** – Presence of lung cancer
* **Negative** – Normal CT scans

Four state-of-the-art transfer learning models are implemented and compared:

* MobileNetV2
* EfficientNetB0
* DenseNet121
* ResNet50

The project includes data augmentation, model training, performance evaluation, visualization, and model comparison.

> **Note:** This project is currently under development and model performance is being improved. Results shown in the notebook represent intermediate experiments.

---

## 📂 Dataset

### Dataset Used

**CT Scan Binary Classification Dataset**

Source:

https://www.kaggle.com/datasets/ameyac11/ct-scan-binary-classification-dataset

### Dataset Description

The dataset is derived from the IQ-OTH/NCCD Lung Cancer Dataset.

Original classes:

* Normal
* Benign
* Malignant

For binary classification:

* Benign + Malignant → **Positive**
* Normal → **Negative**

### Dataset Statistics

| Class    | Images |
| -------- | -----: |
| Positive |    681 |
| Negative |    416 |
| Total    |   1097 |

Image format:

* JPG

Dataset size:

* Approximately 158 MB

---

## 🏗 Project Workflow

### 1. Data Loading

Images are loaded directly from directory structure using:

* TensorFlow
* image_dataset_from_directory()

Dataset split:

* Training: 80%
* Validation: 10%
* Testing: 10%

---

### 2. Data Augmentation

To reduce overfitting and improve generalization, several augmentation techniques are applied:

* Random Horizontal Flip
* Random Rotation
* Random Zoom
* Random Contrast

---

### 3. Data Preprocessing

* Image resizing to 224×224
* Pixel normalization
* TensorFlow Dataset pipeline
* Prefetching for efficient training

---

### 4. Transfer Learning Models

The following ImageNet pre-trained CNN architectures are used:

#### MobileNetV2

Lightweight architecture suitable for efficient image classification.

#### EfficientNetB0

Compound-scaled CNN with improved parameter efficiency.

#### DenseNet121

Dense connectivity architecture commonly used in medical image analysis.

#### ResNet50

Residual learning-based deep neural network.

---

## 🧠 Model Architecture

Each model consists of:

Pre-trained Feature Extractor

↓

GlobalAveragePooling2D

↓

Dense Layer (256 units)

↓

Dropout (0.5)

↓

Sigmoid Output Layer

---

## ⚙ Training Configuration

### Optimizer

Adam

### Loss Function

Binary Crossentropy

### Metrics

* Accuracy

### Callbacks

* EarlyStopping
* ReduceLROnPlateau
* ModelCheckpoint

---

## 📊 Performance Evaluation

Each model is evaluated using:

### Classification Report

Includes:

* Precision
* Recall
* F1-score

### Confusion Matrix

Visualizes prediction performance.

### ROC Curve

Receiver Operating Characteristic analysis.

### AUC Score

Area Under ROC Curve.

### Training Curves

* Accuracy vs Epoch
* Loss vs Epoch

---

## 📈 Model Comparison

Models are compared using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

A comparison table is generated after training all models.

---

## 🛠 Technologies Used

### Programming Language

* Python

### Libraries

* TensorFlow
* Keras
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

### Deep Learning

* Transfer Learning
* CNN
* Image Classification

---

## 🚀 Future Improvements

Planned enhancements include:

* Fine-tuning the best-performing model
* Hyperparameter tuning
* Class imbalance handling
* Grad-CAM visualization
* Explainable AI techniques
* Model ensemble methods
* Streamlit web application
* Gradio interface
* Deployment on Hugging Face Spaces

---

## ⚠ Disclaimer

This project is intended solely for educational and research purposes.

It is not a medical device and should not be used for clinical diagnosis or treatment decisions.

---

## 🙏 Acknowledgements

### Dataset

IQ-OTH/NCCD Lung Cancer Dataset

### Kaggle Dataset

https://www.kaggle.com/datasets/ameyac11/ct-scan-binary-classification-dataset

### Frameworks

* TensorFlow
* Keras
* Scikit-learn

---

## ⭐ Current Status

🟡 Work in Progress

The current notebook contains initial experiments with multiple transfer learning models. Performance improvements and advanced explainable AI techniques are planned for future iterations.
