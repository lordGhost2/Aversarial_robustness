# Robust Medical Image Classification Against Black-Box Adversarial Attacks

## 📌 Overview
This project implements a **robustness evaluation framework** for medical image classification models against **black-box adversarial attacks**.  
We specifically test on **MedMNIST datasets** (e.g., PathMNIST) and evaluate defenses such as **adversarial training** and visualization with **Grad-CAM**.  
The goal is to **overcome the limitations of simultaneous adversarial attacks** and ensure more reliable AI in medical diagnostics.

Key features:
- Black-box adversarial attack generation (`blackbox.py`)
- Model evaluation & confusion matrix visualization (`vizualize_confusion.py`)
- Grad-CAM heatmap generation for interpretability (`gradcam_export.py`, `grad_campdf.py`)
- Automated performance summary export to Excel (`excel_summary.py`)
- Pretrained model weights included (`resnet18_pathmnist.pth`)

---

## 🗂 Project Structure
```
final project/
│── blackbox.py               # Implements black-box adversarial attacks
│── excel_summary.py           # Generates Excel reports of model performance
│── gradcam_export.py          # Exports Grad-CAM visualizations
│── grad_campdf.py             # Creates PDF reports with Grad-CAM results
│── vizualize_confusion.py     # Plots confusion matrices
│── resnet18_pathmnist.pth     # Pre-trained ResNet18 model for PathMNIST
│── blackbox_model.pth         # Saved black-box model
│── temp.png                   # Sample visualization
│── temp_highres.png           # High-resolution visualization
│── README.md                  # Project documentation
```

---

## ⚙️ Installation

1️⃣ Clone the repository:
```bash
git clone https://github.com/your-username/medical-robustness-blackbox.git
cd medical-robustness-blackbox
```

2️⃣ Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

3️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Run Black-Box Attack
```bash
python blackbox.py
```
Generates adversarial examples and evaluates model robustness.

### 2. Visualize Confusion Matrix
```bash
python vizualize_confusion.py
```
Outputs confusion matrix plots for model performance.

### 3. Generate Grad-CAM Visualizations
```bash
python gradcam_export.py
```
Creates heatmaps highlighting important image regions.

### 4. Export Excel Summary
```bash
python excel_summary.py
```
Produces `.xlsx` performance summaries for further analysis.

---

## 📊 Dataset
This project supports **MedMNIST** datasets.  
Example:
- **PathMNIST** (colon pathology slides, 9 classes)
- Load using:
```python
from medmnist import PathMNIST
```
Dataset link: [https://medmnist.com/](https://medmnist.com/)

---

## 🧠 Model
- Base model: **ResNet-18** (PyTorch)
- Pre-trained on PathMNIST
- Fine-tuned for robustness experiments

---

## 🔍 Research Context
This project is part of an effort to:
- Evaluate **black-box attack** robustness in medical imaging
- Improve **interpretability** with Grad-CAM
- Automate reporting for **academic reproducibility**

---

## 📌 Requirements
- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- pandas
- openpyxl
- medmnist
- scikit-learn

---

## 📜 License
This project is for **academic and research purposes** only.  
Not intended for clinical deployment.
