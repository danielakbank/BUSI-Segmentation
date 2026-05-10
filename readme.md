# 🩺 BUSI Breast Ultrasound Image Segmentation

## 🧠 Overview

This project focuses on **medical image segmentation** for breast ultrasound scans using deep learning.

The goal is to automatically detect and segment lesion regions (benign, malignant, normal) from ultrasound images using a convolutional neural network.

---

## 🌐 Live Demo

🚀 Try the deployed app here:  
https://huggingface.co/spaces/daniel-akbank/ultrasound-segmentation

---

## 📁 Repository

GitHub Repo:  
https://github.com/danielakbank/BUSI-Segmentation

---

## 📊 Dataset

**BUSI (Breast Ultrasound Images Dataset)**

- Benign cases  
- Malignant cases  
- Normal cases  
- Each image has a corresponding segmentation mask  

---

## 🧠 Model Architecture

The model uses a **U-Net style architecture** with a pretrained encoder:

- Encoder: :contentReference[oaicite:0]{index=0}  
- Decoder: U-Net upsampling path with skip connections  
- Input size: 256 × 256 × 3  
- Output: binary segmentation mask (256 × 256 × 1)

---

## ⚙️ Training Pipeline

### 1. Data Preparation
- Image-mask pairing from BUSI dataset
- Resize to 256×256
- Normalization to [0,1]
- Binary mask conversion

### 2. Data Splitting
- Stratified train/validation split
- Class balancing via oversampling

### 3. Data Augmentation
- Horizontal & vertical flips  
- Rotation  
- Brightness/contrast adjustment  
- Elastic deformation  

### 4. Training Strategy
- Phase 1: Encoder frozen, decoder training
- Phase 2: Full model fine-tuning (low learning rate)

---

## 📈 Evaluation Metrics

- Dice Coefficient (primary metric)
- Focal Tversky Loss
- Binary Cross-Entropy
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)

---

## 📊 Results

- Final Validation Dice Score: ~0.71  
- Stable convergence across training phases  
- Strong segmentation performance on validation data  

---

## 🧪 Notebook Contents

The Jupyter notebook includes:

- Dataset loading & preprocessing  
- Exploratory data analysis  
- Data augmentation pipeline  
- U-Net model implementation  
- Training (Phase 1 + Phase 2)  
- Evaluation & visualization  
- Prediction overlays  

---

## 🛠 Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Albumentations  
- Gradio  
- Hugging Face Spaces  

---

## ⚠️ Disclaimer

This project is intended for **research and educational purposes only**.

It is **not a medical device** and should not be used for clinical diagnosis or treatment decisions.

Always consult qualified medical professionals for healthcare decisions.

---

## 👨‍💻 Author

Developed as a deep learning project for medical image segmentation using convolutional neural networks.