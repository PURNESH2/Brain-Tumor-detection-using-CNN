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
├── test_inference.py
├── convert_model.py
├── requirements.txt
├── README.md
├── LICENSE
├── dataset/
└── model/
```
---

\### Data Split

\- \*\*Training Set\*\*: 80% (1,336 images)

\- \*\*Validation Set\*\*: 20% (334 images)

\- \*\*Random Seed\*\*: 42 (for reproducibility)



\### Class Imbalance



\- Imbalance Ratio: 1.79:1 (tumor:normal)

\- Handling: Automatic class weight computation

&nbsp; - Normal class weight: 1.44

&nbsp; - Tumor class weight: 0.77



> \*\*Note\*\*: Dataset not included in repository due to size constraints. Please download from \[source](#) and place in the structure above.

---
\## 📂 Project Structure

```

brain-tumor-classification/

│

├── main.py                          # Main training script

├── test.py                          # Inference testing script

├── convert_model.py                 # Model conversion utility

├── requirements.txt                 # Python dependencies

├── README.md                        # Project documentation

├── dataset/                         # Dataset folder

│   └── brain/

│       ├── yes/                     # Tumor images

│       └── no/                      # Normal images

│

└── model/                           # Output folder (created after training)

&nbsp;   ├── brain\_tumor\_classifier.keras # Trained model (PRIMARY)

&nbsp;   ├── best\_model.weights.h5        # Backup weights

&nbsp;   ├── training\_history.csv         # Epoch-by-epoch metrics

&nbsp;   ├── confusion\_matrix.png         # Evaluation visualization

&nbsp;   ├── metrics.json                 # Final evaluation scores

&nbsp;   └── logs/                        # TensorBoard logs

```



---



\## 🏗️ Model Architecture



\### Network Design



```

Input (224×224×3)

&nbsp;   ↓

MobileNetV2 Backbone (Pretrained, Frozen)

&nbsp; - 53 convolutional layers

&nbsp; - ~3.5M parameters (frozen)

&nbsp; - Output: (7×7×1280) feature maps

&nbsp;   ↓

GlobalAveragePooling2D

&nbsp; - Output: 1280-dimensional vector

&nbsp;   ↓

Dense Layer 1 (256 neurons, ReLU)

&nbsp;   ↓

Dropout (50%)

&nbsp;   ↓

Dense Layer 2 (128 neurons, ReLU)

&nbsp;   ↓

Dropout (30%)

&nbsp;   ↓

Output Layer (1 neuron, Sigmoid)

&nbsp; - Output: Probability (0-1)

&nbsp; - >0.5 = Tumor, ≤0.5 = Normal

```

\### Model Summary


\- \*\*Total Parameters\*\*: 2,618,945

\- \*\*Trainable Parameters\*\*: 119,425 (custom layers only)

\- \*\*Non-trainable Parameters\*\*: 2,499,520 (frozen MobileNetV2)

\- \*\*Model Size\*\*: ~80 MB

\- \*\*Input Shape\*\*: (224, 224, 3)

\- \*\*Output\*\*: Binary classification (0 or 1)



\### Why MobileNetV2?



\- Pretrained on 14M ImageNet images

\- Efficient architecture (works on CPU)

\- Excellent feature extraction

\- Proven performance on medical imaging tasks

\- Lightweight (~50MB) compared to ResNet/VGG



---



\## 📈 Results



\### Training Performance



\- \*\*Training Time\*\*: ~3 minutes (CPU: Intel Core Ultra 5125H)

\- \*\*Total Epochs\*\*: 18 (stopped early)

\- \*\*Best Epoch\*\*: 13

\- \*\*Final Training Loss\*\*: 0.2830

\- \*\*Final Validation Loss\*\*: 0.2463



\### Training Curves



Training and validation loss decreased consistently, indicating proper learning without overfitting.



| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |

|-------|------------|----------|-----------|---------|

| 1     | 0.6686     | 0.5012   | 0.6098    | 0.7934  |

| 5     | 0.4312     | 0.3540   | 0.8083    | 0.8503  |

| 10    | 0.3498     | 0.3364   | 0.8500    | 0.8443  |

| 13    | 0.2830     | 0.2463   | 0.8803    | 0.9222  |



\### Evaluation Metrics Explained



\*\*Accuracy (92.22%)\*\*

\- Overall correctness of predictions

\- Good but can be misleading with imbalanced data



\*\*Precision (96.26%)\*\*

\- Of predicted tumors, how many are actually tumors?

\- Low false positive rate (only 4% false alarms)

\- Important: Reduces unnecessary patient anxiety



\*\*Recall (90.45%)\*\*

\- Of actual tumors, how many did we detect?

\- Catches 90% of tumors, misses 10%

\- Critical metric in medical imaging



\*\*F1-Score (0.9326)\*\*

\- Harmonic mean of precision and recall

\- Excellent balance between the two



\*\*ROC AUC (97.46%)\*\*

\- Threshold-independent performance measure

\- Near-perfect discrimination ability

\- 0.5 = random, 1.0 = perfect



\### Clinical Interpretation



✅ \*\*Strengths:\*\*

\- High precision = Few false alarms

\- High accuracy = Reliable overall

\- Excellent ROC AUC = Strong discriminative power



⚠️ \*\*Concerns:\*\*

\- Misses ~10% of tumors (16 out of 124)

\- Not suitable as standalone diagnostic tool

\- Requires radiologist verification



---



\## 🔧 Technical Details



\### Preprocessing Pipeline



1\. \*\*Image Loading\*\*: OpenCV (`cv2.imread`)

2\. \*\*Color Conversion\*\*: 

&nbsp;  - Grayscale → RGB (duplicate channel)

&nbsp;  - RGBA → RGB (drop alpha)

3\. \*\*Resizing\*\*: Any size → 224×224 (bilinear interpolation)

4\. \*\*Normalization\*\*: Pixel values 0-255 → 0-1 (divide by 255)

5\. \*\*Data Type\*\*: Convert to float32



\### Data Augmentation (Training Only)



Applied on-the-fly during training:

\- \*\*Rotation\*\*: ±20 degrees

\- \*\*Width Shift\*\*: ±20%

\- \*\*Height Shift\*\*: ±20%

\- \*\*Horizontal Flip\*\*: 50% probability

\- \*\*Zoom\*\*: ±20%

\- \*\*Fill Mode\*\*: Nearest neighbor



\### Training Configuration



\*\*Optimizer\*\*: Adam

\- Learning rate: 0.0001

\- Beta1: 0.9

\- Beta2: 0.999

\- Epsilon: 1e-7



\*\*Loss Function\*\*: Binary Crossentropy

\- Optimized for binary classification

\- Formula: `-\[y\*log(p) + (1-y)\*log(1-p)]`



\*\*Callbacks\*\*:



1\. \*\*ModelCheckpoint\*\*

&nbsp;  - Monitor: `val\_loss`

&nbsp;  - Save best weights only

&nbsp;  - File: `best\_model.weights.h5`



2\. \*\*EarlyStopping\*\*

&nbsp;  - Monitor: `val\_loss`

&nbsp;  - Patience: 5 epochs

&nbsp;  - Restores best weights



3\. \*\*ReduceLROnPlateau\*\*

&nbsp;  - Monitor: `val\_loss`

&nbsp;  - Factor: 0.5 (halves learning rate)

&nbsp;  - Patience: 3 epochs




\*\*Performance\*\*:

\- Data loading: 30 seconds

\- Training (18 epochs): ~3 minutes

\- Inference (single image): 2-5 seconds

\- Model loading: 3 seconds



---



\## ⚠️ Limitations



\### Model Limitations



1\. \*\*Missed Tumors\*\*: Fails to detect ~10% of tumors (false negatives)

2\. \*\*Binary Classification Only\*\*: Cannot identify tumor type, location, or size

3\. \*\*Limited Dataset\*\*: Trained on only 1,670 images

4\. \*\*No Explainability\*\*: Cannot show which image regions influenced decision

5\. \*\*Single Modality\*\*: Only works with MRI (not CT, X-ray, etc.)



\### Clinical Limitations



1\. \*\*Not FDA Approved\*\*: Research prototype only

2\. \*\*Requires Radiologist Review\*\*: Cannot replace human expertise

3\. \*\*No Real-time Processing\*\*: Takes 2-5 seconds per image

4\. \*\*Potential Bias\*\*: Dataset may not represent all demographics

5\. \*\*False Negatives Critical\*\*: Missing tumors has serious consequences



\### Technical Limitations



1\. \*\*CPU Performance\*\*: 10-100x slower than GPU

2\. \*\*Fixed Input Size\*\*: Must resize to 224×224

3\. \*\*Memory Requirements\*\*: Needs 16GB RAM for training

4\. \*\*Single-threaded Inference\*\*: No batch optimization






---



\## 🙏 Acknowledgments



\- \*\*MobileNetV2\*\*: Google Research for the pretrained architecture

\- \*\*TensorFlow/Keras\*\*: Open-source deep learning framework

\- \*\*Dataset\*\*: \[Dataset source/creator name]

\- \*\*Inspiration\*\*: Medical imaging research community

\- \*\*Mentors\*\*: \[Your teacher/advisor name]



\### Libraries Used



\- TensorFlow/Keras - Deep learning framework

\- NumPy - Numerical computing

\- OpenCV - Image processing

\- Pandas - Data manipulation

\- Scikit-learn - Machine learning utilities

\- Matplotlib - Visualization

\- tqdm - Progress bars


