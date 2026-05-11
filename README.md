# ❤️ DeepHealthX: Multi-Modal Explainable AI for Heart Disease Risk Assessment

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**DeepHealthX** is a sophisticated, multi-modal deep learning system designed for comprehensive heart disease risk assessment. It integrates clinical data, ECG signals, and cardiac MRI scans to provide a holistic prediction. Crucially, the system employs state-of-the-art Explainable AI (XAI) techniques to ensure that its predictions are transparent, interpretable, and actionable for medical professionals.

---

## 🌟 Key Features

### 1. Multi-Modal Fusion Architecture
DeepHealthX utilizes a robust two-stage training pipeline to integrate three distinct data modalities:
- **Clinical Branch:** A Dense Neural Network processes 13 tabular features (e.g., age, cholesterol, resting BP).
- **ECG Branch:** A 1D-CNN analyzes 187-point temporal ECG heartbeat signals.
- **MRI Branch:** A 2D-CNN (EfficientNetB0 backbone) extracts features from 224x224 RGB cardiac MRI scans.
- **Fusion Head:** Concatenates specialized embeddings (totaling 160 dimensions) from the three frozen branches into a unified risk classifier.

### 2. State-of-the-Art Explainability (XAI)
Transparent AI is critical in healthcare. DeepHealthX provides modality-specific explanations:
- **SHAP (Clinical):** Quantifies exactly how much each clinical parameter contributes to the final risk score.
- **Temporal Saliency (ECG):** Highlights critical segments (e.g., QRS complex, T-wave) in the ECG waveform that most influenced the model's decision.
- **Grad-CAM (MRI):** Generates heatmaps overlaid on cardiac MRI scans to pinpoint areas of potential pathology.

### 3. Interactive Clinical Dashboard
The project includes a sleek, production-ready **Streamlit** dashboard that allows users to:
- Input patient clinical metrics via a user-friendly interface.
- Upload ECG CSV files and MRI images (or use built-in synthetic demo data).
- View a unified risk assessment with detailed modality breakdowns.

---

## 📂 Project Structure

```text
DeepHealthX_XAI/
├── config.py                 # Centralized configuration (hyperparams, paths, architecture)
├── gradcam_mri.py            # XAI module: Grad-CAM for MRI images
├── saliency_ecg.py           # XAI module: Temporal Saliency for ECG signals
├── shap_clinical.py          # XAI module: SHAP for tabular clinical features
├── requirements.txt          # Python dependencies
├── models/                   # Pre-trained models (.h5) and scalers
├── webapp/
│   └── app.py                # Main Streamlit dashboard application
└── demo_samples/             # Sample datasets and images for testing
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+
- A CUDA-compatible GPU (Optional, but recommended for training)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/DeepHealthX_XAI.git
cd DeepHealthX_XAI
```

### Step 2: Create a Virtual Environment
```bash
python -m venv venv_app
# On Windows:
venv_app\Scripts\activate
# On macOS/Linux:
source venv_app/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Running the Web Dashboard
To launch the interactive Streamlit dashboard:

```bash
streamlit run webapp/app.py
```
The application will open automatically in your default web browser at `http://localhost:8501`. 

### Using the Application
1. **ECG & MRI Input:** Use the top panels to upload an ECG signal (CSV format) and a Cardiac MRI (JPG/PNG). If you don't have medical data on hand, select the **Demo Signal/Image** options to use built-in synthetic data.
2. **Clinical Input:** Enter the patient's vital signs and lab results in the provided grid. You can use the "Quick presets" to auto-fill high-risk or low-risk profiles.
3. **Analyze:** Click **"Run Cardiac Risk Assessment"** to view the tri-modal inference results and the clinical interpretation.

---

## 🔬 Datasets Referenced
*(Note: Raw datasets are not included in this repository to respect data licensing and size constraints. They can be obtained from their respective sources.)*
1. **Clinical:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
2. **ECG:** [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/)
3. **MRI:** [ACDC (Automated Cardiac Diagnosis Challenge) Dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/)

---

## ⚠️ Disclaimer
**For Research and Educational Purposes Only.** 
DeepHealthX is a conceptual machine learning project and is **not** a certified medical device. It should not be used as a substitute for professional medical diagnosis, advice, or treatment.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Developed by Vaka Hareesh Reddy for Portfolio V2.0*
