# 🩺 Breast Ultrasound Segmentation App

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[Live Demo (Gradio)](https://07b36d26bf766e59bf.gradio.live)

---

## 🚀 Project Overview
This project implements **breast ultrasound image segmentation** using a **U-Net model with a ResNet50V2 backbone**. The goal is to detect and segment lesions in ultrasound images with high accuracy.  

Key features:  
- U-Net architecture with ResNet50V2 encoder  
- Dice coefficient and combined loss (BCE + Focal Tversky)  
- Data augmentation for better generalization  
- Interactive **Gradio app** for testing and visualization  

---

## 📁 Folder Structure

```
BUSI_Segmentation/
│
├── README.md                          # This file
├── breast_segmentation_unet_resnet50v2.ipynb  # Training & evaluation notebook
├── Dataset_BUSI_with_GT/              # Ultrasound dataset (images + masks)
├── models/
│   └── unet_resnet50v2_final.keras    # Trained model
├── validation_images/                 # Preprocessed validation images
├── ultrasound_segmentation_app/       # Gradio app folder
│   ├── app.py
│   ├── utils.py
│   ├── custom_object.py
│   ├── unet_resnet50v2_final.keras
│   └── requirements.txt
└── venv/                               # Optional virtual environment (ignored in Git)
```

---

## ⚡ Features

- Train U-Net on BUSI dataset with augmented data  
- Stratified train/validation split and oversampling to balance classes  
- Evaluate model using Dice coefficient and visualize results  
- **Gradio app** for easy image upload and segmentation visualization:  
  - Upload an ultrasound image  
  - Output segmentation mask  
  - Overlay mask on original image  

---

## 🛠 Installation

1. Clone the repository:

```bash
git clone https://github.com/danielakbank/BUSI_Segmentation.git
cd BUSI_Segmentation
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r ultrasound_segmentation_app/requirements.txt
```

---

## ▶️ Running the Gradio App

1. Navigate to the app folder:

```bash
cd ultrasound_segmentation_app
```

2. Launch the Gradio interface:

```bash
python app.py
```

3. Open the displayed URL in your browser to test segmentation.

---

## 📊 Training & Evaluation

- Notebook: `breast_segmentation_unet_resnet50v2.ipynb`  
- Includes:  
  - Data loading and preprocessing  
  - U-Net model with ResNet50V2 backbone  
  - Training (encoder freeze + fine-tuning)  
  - Evaluation on validation images  
  - Visualizations of predicted masks vs ground truth  

- Model saved as: `models/unet_resnet50v2_final.keras`

---

## 📌 Notes

- The dataset folder `Dataset_BUSI_with_GT/` should contain all BUSI images and masks.  
- Validation images are stored in `validation_images/` for quick testing.  
- If the dataset or model is too large, consider hosting externally and linking in the README.  

---

## 📜 License

This project is open-source under the **MIT License**. See `LICENSE` for details.

---

## 🙏 Acknowledgements

- [BUSI Dataset](https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset)  
- TensorFlow and Keras libraries  
- Gradio for interactive web interface  
- Albumentations for data augmentation  

```

