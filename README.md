# 🍎 Fresh vs Rotten Fruits and Vegetables Detection using Deep Learning

This project presents an end-to-end AI system for classifying fruits and vegetables as **fresh** or **rotten**, using deep learning and computer vision. Built on the **MobileNetV2** architecture, the system provides fast, accurate, and user-friendly spoilage detection with real-world applicability.

---

## 📌 Project Highlights

- **94.8% Accuracy** on 28 classes (14 types × [healthy, spoiled])
- **<1.2s Inference Time** using optimized MobileNetV2
- **Deployment-ready** via FastAPI backend and Streamlit frontend
- Integrated **AI Chatbot** for tips and storage advice
- Uses **Grad-CAM** for explainability

---

## 🎯 Objectives

- Classify fruits and vegetables as fresh or spoiled using deep learning.
- Deploy the model using FastAPI and integrate a frontend using Streamlit.
- Provide real-time feedback, AI-generated storage tips, and chatbot-based interaction.
- Ensure high performance across accuracy, speed, and interpretability.

---

## 🧠 Model Architecture

- **Base Model:** MobileNetV2 + SE Blocks (Squeeze-and-Excitation)
- **Additional Models:** EfficientNetV2, VGG16, ResNet60 for benchmarking
- **Attention Mechanisms:** CBAM modules for enhanced focus on disease areas
- **Explainability:** Grad-CAM for visual interpretation of predictions

---

## 🧪 Methodology

### 📷 Data Preprocessing
- CLAHE for contrast enhancement
- Wavelet denoising and Macenko color normalization
- Data augmentation: rotation, flipping, gamma, HSV shifts

### 🔬 Feature Engineering
- **Handcrafted Features:** Haralick, LBP, Color Histograms, Hu Moments
- **Feature Selection:** RFE + Mutual Information → reduced 1,872 features to 342

### ⚙️ Machine Learning
- Classical Models: SVM, Random Forest
- Ensembles: XGBoost, LightGBM
- Loss Functions: Cross-Entropy + Focal Loss for class imbalance

---

## 📊 Evaluation Metrics

| Metric         | Value     |
|----------------|-----------|
| Accuracy       | 94.8%     |
| Precision      | 95.3%     |
| Recall         | 94.6%     |
| F1 Score       | 94.9%     |
| Inference Time | ~1.18s    |

Additional evaluations:
- Confusion Matrix
- ROC-AUC
- Grad-CAM Visualization

---

## 🧩 Tools & Technologies

- **Python**, **TensorFlow/Keras**, **OpenCV**, **XGBoost**, **LightGBM**
- **FastAPI** (backend), **Streamlit** (frontend), **Grad-CAM**, **SMOTE**
- Dataset: 28-class custom dataset (14 fruit/vegetable types, healthy/spoiled)

---

## 💡 Future Directions

- Deploy lightweight version on mobile using TensorFlow Lite
- Integrate **IoT data** (temp, humidity) for spoilage prediction
- Explore **Vision Transformers** and **Federated Learning**
- Make chatbot advice **season- and region-aware**

---
