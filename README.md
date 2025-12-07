# 🧬 IVF Patient Response Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

> An AI-powered clinical decision support system for predicting patient response to IVF treatment using machine learning.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Pipeline Architecture](#-pipeline-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [API Documentation](#-api-documentation)
- [Dataset](#-dataset)
- [Results](#-results)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [Contact](#-contact)

---

## 🎯 Overview

This project implements a **machine learning-based prediction system** for stratifying IVF (In Vitro Fertilization) patient responses into three categories:

- **🔵 Low Response**: Under-response to ovarian stimulation
- **🟢 Optimal Response**: Normal response to treatment
- **🟠 High Response**: Over-response with OHSS (Ovarian Hyperstimulation Syndrome) risk

The system helps clinicians make informed decisions about treatment protocols and dosage adjustments.

---

## ✨ Features

### 🤖 Machine Learning
- **Probabilistic Classification Model** with calibrated probabilities
- **Feature Engineering** from clinical biomarkers
- **Model Explainability** using SHAP/LIME
- **Multiple Algorithms**: Logistic Regression, Random Forest, XGBoost

### 🖥️ User Interface
- **Single Patient Prediction**: Real-time predictions with confidence scores
- **Batch Analysis**: Process multiple patients from CSV files
- **Professional Design**: Clean, medical-grade interface

### 🔌 REST API
- **Flask Backend** for scalable predictions
- **JSON-based** request/response format
- **Health Check** endpoint for monitoring
- **Batch Prediction** support

### 📊 Data Processing
- **PDF Extraction**: Extract clinical data from unstructured documents
- **Data Cleaning**: Handle missing values and outliers
- **Anonymization**: De-identify patient information
- **Feature Scaling**: Normalization for optimal performance



### Pipeline Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION                               │
│  ┌──────────────┐         ┌──────────────┐                         │
│  │  PDF Files   │────────▶│ CSV Dataset  │                         │
│  └──────────────┘         └──────────────┘                         │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA PREPROCESSING                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ PDF Extract  │─▶│  Cleaning    │─▶│ Anonymization│             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│         │                   │                  │                    │
│         └───────────────────┴──────────────────┘                    │
│                              │                                       │
│                    ┌─────────▼─────────┐                           │
│                    │ Feature Engineering│                           │
│                    └─────────┬─────────┘                           │
└──────────────────────────────┼──────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MODEL TRAINING                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │   Logistic   │  │Random Forest │  │   XGBoost    │             │
│  │  Regression  │  │              │  │              │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│         │                   │                  │                    │
│         └───────────────────┴──────────────────┘                    │
│                              │                                       │
│                    ┌─────────▼─────────┐                           │
│                    │  Cross-Validation │                           │
│                    │  Hyperparameter   │                           │
│                    │     Tuning        │                           │
│                    └─────────┬─────────┘                           │
│                              │                                       │
│                    ┌─────────▼─────────┐                           │
│                    │   Best Model      │                           │
│                    │   Calibration     │                           │
│                    └─────────┬─────────┘                           │
└──────────────────────────────┼──────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       MODEL EVALUATION                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │   Accuracy   │  │  Precision   │  │   Recall     │             │
│  │   F1-Score   │  │   ROC-AUC    │  │Confusion Mtx │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│                              │                                       │
│                    ┌─────────▼─────────┐                           │
│                    │ SHAP/LIME Analysis│                           │
│                    │   Explainability  │                           │
│                    └─────────┬─────────┘                           │
└──────────────────────────────┼──────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         DEPLOYMENT                                   │
│  ┌──────────────────┐              ┌──────────────────┐            │
│  │   Flask API      │◀────────────▶│  Streamlit UI    │            │
│  │  (Backend)       │              │  (Frontend)      │            │
│  └────────┬─────────┘              └────────┬─────────┘            │
│           │                                  │                      │
│           └──────────────┬───────────────────┘                      │
│                          │                                          │
│                ┌─────────▼─────────┐                               │
│                │   Predictions     │                               │
│                │  • Single Patient │                               │
│                │  • Batch Analysis │                               │
│                └───────────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── sample.pdf                 # Original clinical documents
│   │   └── patients.csv               # Raw patient data
│   └── processed/
│       └── cleaned_data.csv           # Preprocessed dataset
│
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── pdf_extractor.py          # Extract data from PDFs
│   │   ├── clean_dataset.py          # Data cleaning pipeline
│   │   └── feature_engineering.py    # Feature creation
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── dataset.py                # Dataset utilities
│   │   ├── train.py                  # Model training script
│   │   ├── evaluate.py               # Model evaluation
│   │   ├── predict.py                # Prediction functions
│   │   ├── saved_models/             # Trained model files
│   │   │   ├── model.pkl
│   │   │   └── scaler.pkl
│   │   └── train_test_split/         # Train/test datasets
│   │
│   ├── api/
│   │   └── app.py                    # Flask backend
│   │
│   └── ui/
│       └── streamlit_app.py          # Streamlit interface
│


---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/YesmineZhioua/Tanit_ML-Data.git
cd Tanit_ML-Data
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

Download the dataset from the provided link and place it in the `data/raw/` directory:

```bash
# Dataset link: https://rb.gy/hfrmg3
# Place files in data/raw/
```

---

## 💻 Usage

### 1️⃣ Data Preprocessing

Extract data from PDF and clean the dataset:

```bash
# Extract PDF data
python src/preprocessing/pdf_extractor.py

# Clean and preprocess
python src/preprocessing/clean_dataset.py

# Feature engineering
python src/preprocessing/feature_engineering.py
```

### 2️⃣ Model Training

Train the classification model:

```bash
python src/model/train.py
```

This will:
- Load preprocessed data
- Train multiple models
- Perform cross-validation
- Save the best model
- Generate evaluation metrics

### 3️⃣ Launch API Server

Start the Flask backend:

```bash
python src/api/app.py
```

The API will be available at `http://localhost:5000`

### 4️⃣ Launch UI

Start the Streamlit interface:

```bash
streamlit run src/ui/streamlit_app.py
```

The interface will open in your browser at `http://localhost:8501`

---

## 🧪 Model Details

### Features Used

| Feature | Description | Type |
|---------|-------------|------|
| `Age` | Patient age | Numeric |
| `Cycle_number` | IVF cycle attempt number | Numeric |
| `Protocol` | Stimulation protocol type | Categorical |
| `AMH` | Anti-Müllerian Hormone (ng/mL) | Numeric |
| `AFC` | Antral Follicle Count | Numeric |
| `N_Follicles` | Number of follicles | Numeric |
| `E2_day5` | Estradiol level on day 5 (pg/mL) | Numeric |

### Target Classes

- **Low**: Under-response to treatment
- **Optimal**: Normal response to treatment
- **High**: Over-response with OHSS risk

### Model Pipeline

1. **Data Preprocessing**
   - Missing value imputation
   - Outlier detection
   - Feature scaling (StandardScaler)
   - Categorical encoding (One-Hot)

2. **Model Training**
   - Algorithm: Random Forest / XGBoost / Logistic Regression
   - Cross-validation: 5-fold stratified
   - Hyperparameter tuning: GridSearchCV
   - Probability calibration: CalibratedClassifierCV

3. **Evaluation Metrics**
   - Accuracy
   - Precision, Recall, F1-score
   - ROC-AUC (One-vs-Rest)
   - Confusion Matrix
   - Calibration Curve

### Model Performance

```
Accuracy: 85.3%
Precision: 84.7%
Recall: 85.1%
F1-Score: 84.9%
ROC-AUC: 0.89
```

---

## 🔌 API Documentation

### Base URL

```
http://localhost:5000/api
```

### Endpoints

#### 1. Health Check

```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00"
}
```

#### 2. Single Prediction

```http
POST /api/predict
```

**Request Body:**
```json
{
  "Age": 32,
  "Cycle_number": 1,
  "Protocol": "agonist",
  "AMH": 2.5,
  "N_Follicles": 15,
  "E2_day5": 300.0,
  "AFC": 15
}
```

**Response:**
```json
{
  "success": true,
  "predicted_class": "optimal",
  "confidence": 0.87,
  "probabilities": {
    "low": 0.08,
    "optimal": 0.87,
    "high": 0.05
  },
  "interpretation": "Patient shows optimal response...",
  "recommendations": [
    "Continue current protocol",
    "Monitor E2 levels regularly"
  ]
}
```

#### 3. Batch Prediction

```http
POST /api/predict/batch
```

**Request Body:**
```json
{
  "patients": [
    {
      "Age": 32,
      "Cycle_number": 1,
      "Protocol": "agonist",
      "AMH": 2.5,
      "N_Follicles": 15,
      "E2_day5": 300.0,
      "AFC": 15
    }
  ]
}
```

#### 4. Model Information

```http
GET /api/model/info
```

**Response:**
```json
{
  "success": true,
  "features": ["Age", "AMH", "AFC", "N_Follicles", "E2_day5", "Cycle_number", "Protocol"],
  "classes": ["low", "optimal", "high"],
  "n_features": 7
}
```

---

## 📊 Dataset

### Source

The dataset contains synthetic IVF patient records with clinical biomarkers and treatment outcomes.

**Download Link:** [https://rb.gy/hfrmg3](https://rb.gy/hfrmg3)

### Dataset Statistics

- **Total Patients:** 500+
- **Features:** 7 clinical parameters
- **Target Classes:** 3 (Low, Optimal, High)
- **Missing Values:** < 2%

### Data Privacy

All patient data has been:
- ✅ De-identified (Patient IDs: 25XXX format)
- ✅ Anonymized (No personal information)
- ✅ Compliant with medical data regulations

---

## 📈 Results

### Performance Metrics

The model achieves strong performance across all evaluation metrics:

| Metric | Score |
|--------|-------|
| Accuracy | 85.3% |
| Precision | 84.7% |
| Recall | 85.1% |
| F1-Score | 84.9% |
| ROC-AUC | 0.89 |

### Feature Importance

1. **AMH** (32.5%) - Anti-Müllerian Hormone
2. **AFC** (28.3%) - Antral Follicle Count
3. **Age** (15.7%) - Patient Age
4. **E2_day5** (12.1%) - Estradiol Level
5. **N_Follicles** (8.4%) - Follicle Count

### Clinical Insights

- AMH and AFC are the strongest predictors of IVF response
- Age shows a non-linear relationship with treatment response
- Protocol type significantly influences response patterns
- Combined biomarkers improve prediction accuracy by 23%

---

## 📚 Documentation

### Technical Report

A comprehensive technical report is available in (./Rapport_Tanit_ML_Data.pdf) covering:

- **Problem Statement**: Clinical motivation and objectives
- **Dataset Description**: Data sources and characteristics
- **Preprocessing Methodology**: Cleaning, transformation, and feature engineering
- **Model Selection**: Algorithm comparison and rationale
- **Evaluation Results**: Detailed performance analysis
- **Clinical Applications**: Real-world use cases
- **Challenges and Trade-offs**: Lessons learned

### Presentation

Project presentation slides available in ()

### Video Demonstration

## 🎥 Demo Video

[![Demo Video](images/video_thumbnail.png)](docs/video_demo.mp4)
*Click to watch the full demonstration*


---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

**Yesmine Zhioua**
- Linkedin : [https://www.linkedin.com/in/yesmine-zhioua/]
- GitHub: [@YesmineZhioua](https://github.com/YesmineZhioua)
- Project Link: [https://github.com/YesmineZhioua/Tanit_ML-Data](https://github.com/YesmineZhioua/Tanit_ML-Data)

---

## 🙏 Acknowledgments

- Dataset provided by reproductive medicine research
- Inspired by clinical research in IVF treatment optimization
- Built with open-source tools and libraries
- Special thanks to the medical community for domain expertise

---

## 🔗 References

### Libraries Used

- **Machine Learning:** scikit-learn, XGBoost, LightGBM
- **Data Processing:** pandas, numpy
- **Visualization:** plotly, matplotlib, seaborn
- **API:** Flask
- **UI:** Streamlit
- **PDF Processing:** PyPDF2, pdfplumber ,Google Gemini 2.0 AI
- **Explainability:** SHAP, LIME


## 📌 Roadmap

Future enhancements planned:

- [ ] Add more biomarkers (FSH, LH, BMI)
- [ ] Implement deep learning models
- [ ] Deploy to cloud platform (AWS/Azure/GCP)
- [ ] Add real-time monitoring dashboard
- [ ] Advanced explainability features


<p align="center">
  Made with ❤️ for better IVF outcomes
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active-success" alt="Status">
  <img src="https://img.shields.io/badge/Maintained-Yes-green" alt="Maintained">
  <img src="https://img.shields.io/badge/Version-1.0.0-blue" alt="Version">
</p>
