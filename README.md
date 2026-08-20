# Handwritten Character Recognition

## 📌 Project Overview

This project is developed as part of the **CodeAlpha Machine Learning Internship – Task 3**.

The objective of this project is to develop a machine learning system that can recognize handwritten digits using the **MNIST dataset** and a **Convolutional Neural Network (CNN)**.

## 🎯 Objective

The main objective is to train a CNN model to identify handwritten digits from **0 to 9** and evaluate its performance on unseen test images.

## 📊 Dataset

This project uses the **MNIST handwritten digit dataset**.

The dataset contains grayscale images of handwritten digits from **0 to 9**. Each image has a resolution of **28 × 28 pixels**.

The dataset is automatically downloaded using TensorFlow/Keras.

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* VS Code

## 🤖 Machine Learning Model

A **Convolutional Neural Network (CNN)** is used for handwritten digit classification.

The CNN contains:

* Convolutional layers
* Max Pooling layers
* Flatten layer
* Dense layers
* Dropout layer
* Softmax output layer

## 🔄 Project Workflow

```text
MNIST Dataset
      ↓
Load Images
      ↓
Image Preprocessing
      ↓
Normalization
      ↓
CNN Model
      ↓
Model Training
      ↓
Model Testing
      ↓
Performance Evaluation
      ↓
Digit Prediction
```

## 📈 Model Evaluation

The trained model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

The project also generates a training accuracy graph to visualize the model's learning process.

## 📁 Project Structure

```text
Handwritten_Character_Recognition/
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── training_accuracy.png
│   └── handwritten_digit_model.keras
│
├── handwritten_recognition.py
│
└── README.md
```

## ▶️ How to Run the Project

### 1. Install the required libraries

Open the VS Code terminal and run:

```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn
```

### 2. Run the Python program

```bash
python handwritten_recognition.py
```

The MNIST dataset will be downloaded automatically when the program is run for the first time.

## 📌 Results

The CNN model learns patterns from handwritten digit images and predicts the corresponding digit from 0 to 9.

The project produces:

* Model accuracy
* Classification report
* Confusion matrix
* Training accuracy graph
* Trained CNN model

## 🎓 Internship Information

**Organization:** CodeAlpha

**Domain:** Machine Learning

**Task:** Task 3 – Handwritten Character Recognition

## 👩‍💻 Author

**Anakha A**
