# Brain Tumor detaction using CNN

## Detailed Project Description

This repository contains an end-to-end deep learning pipeline for automated brain tumor classification using structural MRI images. The core idea is to use modern convolutional neural networks with transfer learning to build a robust binary classifier that flags MRI scans as either **Tumor Detected** or **Normal**.

**Problem & Motivation:** Brain tumor diagnosis from MRI scans is time-consuming and requires expert radiologist interpretation. A high-quality automated screening tool can reduce workload, provide fast triage for urgent cases, and assist in large-scale retrospective studies.

**Dataset & Preprocessing:** The model is trained on a curated dataset of axial MRI scans organized into `yes/` (tumor) and `no/` (normal) folders. Images are standardized to 224×224 RGB (duplicated channels for grayscale), normalized to [0,1], and augmented during training (rotation, shifts, flips, zoom).

**Model Architecture & Training:** We use MobileNetV2 as the feature extractor (pretrained on ImageNet) with custom dense layers on top for binary classification. MobileNetV2 is frozen initially for transfer learning; custom head layers are trained with Adam optimizer, binary crossentropy loss, and callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau). Class imbalance is handled with computed class weights.

**Evaluation & Use Cases:** The model achieves ~92% validation accuracy and ~97% ROC AUC on the validation split. While this demonstrates strong discrimination capability, the model should only be used as a secondary screening tool and always verified by clinicians. Typical uses include teleradiology triage, academic research, and educational demonstrations.

**Limitations & Future Work:** Limitations include binary-only prediction (no tumor type/segmentation), potential dataset bias, and dependency on MRI acquisition protocols. Future directions: Grad-CAM/attention explainability, multi-class tumor typing, 3D/volumetric processing, larger multi-center datasets, and clinical validation studies.

---

## ⚠️ IMPORTANT MEDICAL DISCLAIMER (Read first)

**This software is a research prototype for educational and research purposes only. It is _NOT_ approved for clinical use and should _NEVER_ be used as the sole basis for medical diagnosis or treatment decisions.**

Key points:
- Requires validation by qualified medical professionals.  
- Not FDA approved or clinically validated.  
- May produce false negatives (missed tumors) — missing tumors has serious consequences.  
- Should only be used under supervision of radiologists.  
- Not a replacement for professional medical advice.

---

## 👥 Project Team

This project was developed by a team of four students. Replace the placeholders below with actual names and contact details.

- **Member 1** — Name: _Purnesh GC_  
  Email: purneshgc@gmail.com
  GitHub: https://github.com/PURNESH2

- **Member 2** — Name: _Bikram Panda_  
  Email: bikram.panda6503@gmail.com  
  GitHub: https://github.com/bikram-0605

- **Member 3** — Name: _C Sai Krishna_    
  Email: saikrishnachakkara@gmail.com
  GitHub: https://github.com/csaikrishna04

- **Member 4** — Name: _Yuvraj V_    
  Email: yuvrajwhitefield2019@gmail.com  
  GitHub: https://github.com/Yuvraj-v15

---

## 🔍 Overview

This project implements a deep learning solution for automated brain tumor detection from MRI scans. The system uses a Convolutional Neural Network (CNN) based on the MobileNetV2 architecture with transfer learning to classify brain MRI images into two categories:

- **Tumor Detected (Yes)**: MRI scan shows presence of brain tumor  
- **Normal (No)**: MRI scan shows no tumor

### Key Highlights

- **High Accuracy**: Achieves 92.22% accuracy on validation set  
- **Transfer Learning**: Leverages MobileNetV2 pretrained on ImageNet  
- **Efficient**: Runs on CPU, optimized for resource-constrained environments  
- **Production-Ready**: Includes model saving, evaluation metrics, and inference pipeline  
- **Clinical Potential**: Suitable as a secondary screening tool for radiologists

### Real-World Applications

- Hospital radiology departments (workload reduction)  
- Teleradiology services (remote diagnosis support)  
- Medical imaging research  
- Educational tool for medical students

---

## ✨ Features

- **Automated MRI Analysis**: Classifies brain MRI images automatically  
- **Transfer Learning**: Uses pretrained MobileNetV2 for faster training and better accuracy  
- **Data Augmentation**: Implements rotation, shifting, zooming, and flipping for robust learning  
- **Class Imbalance Handling**: Automatic class weight computation  
- **Comprehensive Evaluation**: Multiple metrics (Accuracy, Precision, Recall, F1-Score, ROC AUC)  
- **Visualization**: Confusion matrix and training history plots  
- **Model Persistence**: Saves trained model for future inference  
- **Batch Inference**: Test on multiple images at once  
- **CPU Optimized**: Works on standard hardware (no GPU required)

---

## 📊 Model Performance

| Metric      | Score    | Interpretation                          |
|-------------|----------|-----------------------------------------|
| **Accuracy** | 92.22%   | Correctly classifies 92 out of 100 images |
| **Precision**| 96.26%   | When predicting tumor, correct 96% of the time |
| **Recall**   | 90.45%   | Detects 90% of actual tumors            |
| **F1-Score** | 0.9326   | Excellent balance of precision and recall |
| **ROC AUC**  | 97.46%   | Outstanding discrimination ability      |

### Confusion Matrix

```
                 Predicted
              Normal    Tumor
Actual
Normal      200 (TN)   10 (FP)
Tumor        16 (FN)  108 (TP)
```

---

## 📁 Dataset

### Dataset Structure

```
dataset/
└── brain/
    ├── yes/     # Tumor images (1,072 images)
    └── no/      # Normal images (598 images)
```

### Dataset Statistics

- **Total Images**: 1,670 MRI scans  
- **Tumor Images**: 1,072 (64%)  
- **Normal Images**: 598 (36%)  
- **Image Format**: JPEG (.jpg)  
- **Preprocessed Size**: 224×224×3 (RGB)

---

## 📂 Project Structure

```
brain-tumor-classification/
├── main.py
├── test.py
├── convert_model.py
├── requirements.txt
├── README.md
├── dataset/
└── model/
```
---

# Data Split

## Dataset Split
- **Training Set**: 80% (1,336 images)
- **Validation Set**: 20% (334 images)
- **Random Seed**: 42 (for reproducibility)

## Class Imbalance
- **Imbalance Ratio**: 1.79:1 (tumor : normal)
- **Handling**: Automatic class weight computation
  - Normal class weight: **1.44**
  - Tumor class weight: **0.77**

> **Note**: Dataset not included in this repository due to size constraints.  
> Download from the dataset source and place it in the correct directory structure.

---

# 📂 Project Structure

```
brain-tumor-classification/
│
├── main.py                           # Main training script
├── test.py                           # Inference testing script
├── convert_model.py                  # Model conversion utility
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
│
├── dataset/                          # Dataset folder
│   └── brain/
│       ├── yes/                      # Tumor images
│       └── no/                       # Normal images
│
└── model/                            # Output folder (created after training)
    ├── brain_tumor_classifier.keras  # Trained model (PRIMARY)
    ├── best_model.weights.h5         # Backup weights
    ├── training_history.csv          # Epoch-by-epoch metrics
    ├── confusion_matrix.png          # Evaluation visualization
    ├── metrics.json                  # Final evaluation scores
    └── logs/                         # TensorBoard logs
```

---

# 🏗️ Model Architecture

## Network Design

```
Input (224×224×3)
      ↓
MobileNetV2 Backbone (Pretrained, Frozen)
  - 53 convolutional layers
  - ~3.5M parameters (frozen)
  - Output: (7×7×1280) feature maps
      ↓
GlobalAveragePooling2D
  - Output: 1280-dimensional vector
      ↓
Dense Layer 1 (256 neurons, ReLU)
      ↓
Dropout (50%)
      ↓
Dense Layer 2 (128 neurons, ReLU)
      ↓
Dropout (30%)
      ↓
Output Layer (1 neuron, Sigmoid)
  - Output: Probability (0–1)
  - >0.5 = Tumor, ≤0.5 = Normal
```

## Model Summary

- **Total Parameters**: 2,618,945  
- **Trainable Parameters**: 119,425  
- **Non-trainable Parameters**: 2,499,520  
- **Model Size**: ~80 MB  
- **Input Shape**: (224, 224, 3)  
- **Output**: Binary classification  

## Why MobileNetV2?

- Pretrained on 14M+ ImageNet images  
- Lightweight & CPU-friendly  
- Strong feature extraction  
- Effective for medical imaging  
- Much smaller than VGG/ResNet  

---

# 📈 Results

## Training Performance

- **Training Time**: ~3 minutes  
- **Total Epochs**: 18  
- **Best Epoch**: 13  
- **Final Training Loss**: 0.2830  
- **Final Validation Loss**: 0.2463  

## Training Curves

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|------------|----------|-----------|---------|
| 1     | 0.6686     | 0.5012   | 0.6098    | 0.7934  |
| 5     | 0.4312     | 0.3540   | 0.8083    | 0.8503  |
| 10    | 0.3498     | 0.3364   | 0.8500    | 0.8443  |
| 13    | 0.2830     | 0.2463   | 0.8803    | 0.9222  |

---

# Evaluation Metrics Explained

### **Accuracy — 92.22%**
- Measures overall correctness.

### **Precision — 96.26%**
- Low false positives.

### **Recall — 90.45%**
- Detects most tumors but misses some.

### **F1-Score — 0.9326**
- Balance of precision & recall.

### **ROC AUC — 97.46%**
- Excellent discrimination ability.

---

# Clinical Interpretation

## ✅ Strengths
- High precision  
- High accuracy  
- Excellent ROC AUC  

## ⚠️ Concerns
- Misses ~10% of tumors  
- Not suitable for standalone diagnosis  
- Requires radiologist verification  

---

# 🔧 Technical Details

## Preprocessing Pipeline

1. Image loading (OpenCV)  
2. Grayscale/RGBA → RGB  
3. Resize to 224×224  
4. Normalize [0–1]  
5. Convert to float32  

## Data Augmentation

- Rotations ±20°  
- Shifts ±20%  
- Zoom ±20%  
- Horizontal flip  
- Fill mode: nearest  

## Training Configuration

- Optimizer: Adam  
- Loss: Binary crossentropy  
- Callbacks:  
  - ModelCheckpoint  
  - EarlyStopping  
  - ReduceLROnPlateau  

## Performance

- Data loading: 30 sec  
- Training: ~3 minutes  
- Inference: 2–5 sec  

---

# ⚠️ Limitations

## Model Limitations
1. Misses ~10% tumors  
2. Binary classifier only  
3. Limited dataset  
4. No explainability  
5. MRI-only model  

## Clinical Limitations
1. Not FDA approved  
2. Needs radiologist confirmation  
3. Not real-time  
4. Potential bias  
5. Risky false negatives  

## Technical Limitations
1. CPU slower than GPU  
2. Fixed 224×224 input  
3. High RAM requirement  
4. No multi-thread inference  

---

# 🙏 Acknowledgments

- **MobileNetV2** – Google Research  
- **TensorFlow / Keras**  
- **Dataset Provider**  
- **Medical Imaging Research Community**  

## Libraries Used

- TensorFlow / Keras  
- NumPy  
- OpenCV  
- Pandas  
- Scikit-learn  
- Matplotlib  
- tqdm  



