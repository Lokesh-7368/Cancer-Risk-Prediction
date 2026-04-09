# 🏥 Cancer Risk Level Prediction — End-to-End ML Project

> **Predict cancer risk levels (Low / Medium / High)** using an Optuna-Tuned Class-Weighted XGBoost model trained on demographic, behavioral, and health-related features.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37-red?logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![Streamlit Cloud](https://img.shields.io/badge/Deployed_on-Streamlit_Cloud-ff4b4b?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Project Overview

Advanced machine learning techniques can accurately predict cancer risk levels based on an individual's demographics and other health-related factors. This project helps identify at-risk individuals, supporting early interventions and effective healthcare strategies.

### Key Highlights
- **88% Accuracy** with strong High-risk recall (82% — 9/11 correct)
- **10-step iterative model development** with data leakage detection
- **23+ EDA visualizations** with detailed insights at every step
- **Interactive Streamlit app** with single + batch prediction
- **Deployable free** on Streamlit Community Cloud

### Live App
- **Deployed URL:** https://cancer-risk-prediction-ysqblphaxpsecircmhxktw.streamlit.app

---

## 🏗️ System Architecture

```
Cancer-Risk-Prediction/
│
├── app.py                          # Main Streamlit app (Home page)
├── pages/
│   ├── 1_📊_EDA_Dashboard.py       # 23 EDA visualizations with insights
│   ├── 2_🔮_Predict_Risk.py        # Single & batch prediction
│   └── 3_📋_Model_Performance.py   # Metrics & pipeline details
│
├── models/
│   ├── final_xgb_class_weighted.pkl  # Trained XGBoost model
│   ├── label_encoder.pkl             # Label encoder for target
│   └── feature_names.pkl             # Feature column names
│
├── data/
│   └── cancer-risk-factors.csv       # Dataset (2000 patients)
│
├── notebooks/
│   └── Cancer_Risk_Prediction_(ML).ipynb  # Full training notebook
│
├── .streamlit/
│   └── config.toml                   # Streamlit theme config
│
├── requirements.txt                  # Streamlit app runtime dependencies
├── requirements-notebook.txt         # Notebook/training-only dependencies
├── runtime.txt                       # Streamlit Cloud Python runtime
├── .gitignore
└── README.md
```

---

## 📊 Dataset Description

| Property | Value |
|----------|-------|
| **Rows** | 2,000 patients |
| **Features** | 16 input features + target |
| **Target** | Risk_Level (Low / Medium / High) |
| **Cancer Types** | Lung, Breast, Colon, Prostate, Skin |
| **Class Distribution** | Medium: 1574 (78.7%) · Low: 324 (16.2%) · High: 102 (5.1%) |

### Feature Categories

| Category | Features |
|----------|----------|
| **Demographics** | Age, Gender (0=F, 1=M), BMI |
| **Lifestyle (0-10)** | Smoking, Alcohol_Use, Obesity, Diet_Red_Meat, Diet_Salted_Processed, Fruit_Veg_Intake, Physical_Activity, Physical_Activity_Level, Calcium_Intake |
| **Environmental (0-10)** | Air_Pollution, Occupational_Hazards |
| **Genetic (0/1)** | Family_History, BRCA_Mutation, H_Pylori_Infection |

---

## 🔬 Model Development Pipeline (10 Steps)

| Step | Model | Accuracy | High-Risk Recall | Key Learning |
|------|-------|----------|-----------------|--------------|
| 1 | Logistic + RF | 99%* | — | Data leakage detected! |
| 2 | — | — | — | Removed Overall_Risk_Score |
| 3 | Random Forest | 84% | 35% (7/20) | Baseline after fix |
| 4 | Optuna RF | 83% | 45% (5/20) | Tuning alone isn't enough |
| 5 | SMOTE + Optuna RF | 83% | 45% (9/20) | SMOTE helps slightly |
| 6 | SMOTE + RF (High focus) | 80% | 60% (12/20) | Better recall, lower accuracy |
| 7 | XGBoost + SMOTE | 85% | 45% (9/11) | XGBoost more precise |
| 8 | Optuna + XGB (Recall) | 61% | 65% (13/20) | Great recall, poor accuracy |
| 9 | Class-Weighted XGB | 64% | 75% (15/20) | Best recall, worst accuracy |
| **10** | **Optuna + CW XGB ✅** | **88%** | **82% (9/11)** | **Best balance** |

---

## 📈 Key EDA Findings

1. **Air Pollution is the #1 risk factor** — High-risk patients have 3x higher exposure (8.54 vs 2.95)
2. **Smoking, Alcohol, Processed Diet** are the next strongest risk factors
3. **Fruit & Vegetable Intake is protective** — 40.1% of Low-risk vs 24.5% of High-risk have high intake
4. **BMI does NOT differentiate risk levels** — nearly identical distributions
5. **Lung cancer has the highest High-risk rate** (8.9%), Prostate the lowest (1.0%)
6. **Only 3 patients under age 30** — all female, Lung cancer
7. **Risk factors are synergistic** — multiple high factors together dramatically increase risk

---

## 🎓 Exam / Viva Presentation Notes

### 1) Problem Statement
Cancer risk screening often misses early prioritization when multiple lifestyle and exposure factors interact. This project predicts three risk tiers (Low, Medium, High) from structured patient attributes to support faster triage.

### 2) Why This Is Challenging
- Class imbalance is severe (High-risk is only ~5%).
- Medical datasets can contain leakage-like proxy variables that inflate accuracy.
- Clinical usefulness needs more than raw accuracy: minority recall matters.

### 3) What Makes This Project Strong
- End-to-end workflow from EDA to deployment.
- Explicit leakage identification and correction.
- Iterative 10-step modeling with documented trade-offs.
- Explainable outputs through probability distributions and class-wise metrics.

### 4) How To Explain Final Model Choice
The selected model is not just the one with high accuracy; it is the one with the best balance of overall correctness and meaningful High-risk detection. In real healthcare workflows, missing a High-risk patient is costlier than a false alarm.

### 5) Practical Impact
- Single-patient prediction helps clinicians in point-of-care scenarios.
- Batch prediction helps screening programs prioritize follow-up.
- Probability outputs support decision transparency instead of black-box labels.

---

## 🚀 Local Setup

### Prerequisites
- Python 3.10+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Lokesh-7368/Cancer-Risk-Prediction.git
cd Cancer-Risk-Prediction

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# Optional (for notebook training workflow only)
pip install -r requirements-notebook.txt

# 4. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🌐 Deploy on Streamlit Community Cloud 

### Step-by-Step

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Cancer Risk Prediction"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/Cancer-Risk-Prediction.git
   git push -u origin main
   ```

2. **Go to [share.streamlit.io](https://share.streamlit.io)** → Sign up / Log in

3. **Create App:**
    - Click **New app**
    - Repository: `Lokesh-7368/Cancer-Risk-Prediction`
    - Branch: `main`
    - Main file path: `app.py`
    - Click **Deploy**

4. Wait for build. Current deployed app URL:
    - `https://cancer-risk-prediction-ysqblphaxpsecircmhxktw.streamlit.app`

> **Note:** If deployment asks for Python version, this project already pins it with `runtime.txt` as Python 3.10.
> **Build-time tip:** Streamlit Cloud installs only `requirements.txt`. Notebook-only packages are kept in `requirements-notebook.txt` to speed up deploy builds.

---

## 💻 VS Code Setup Guide

### Step-by-Step

```bash
# 1. Open the project in VS Code
code Cancer-Risk-Prediction

# 2. Install Python extension (if not already)
# Ctrl+Shift+X → Search "Python" → Install

# 3. Select Python interpreter
# Ctrl+Shift+P → "Python: Select Interpreter" → Choose .venv

# 4. Open integrated terminal
# Ctrl+` (backtick)

# 5. Activate virtual environment
source .venv/bin/activate

# 6. Run the app
streamlit run app.py

# 7. For debugging, add to .vscode/launch.json:
```

### Run Notebook In VS Code (Using .venv Kernel)

1. Open `notebooks/Cancer_Risk_Prediction_(ML).ipynb`.
2. Click **Select Kernel** and choose the `.venv` Python 3.10 interpreter.
3. Run cells from top to bottom to reproduce training outputs.
4. If any package is missing, run:

```bash
pip install -r requirements.txt
pip install -r requirements-notebook.txt
```

Create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Streamlit",
            "type": "python",
            "request": "launch",
            "module": "streamlit",
            "args": ["run", "app.py"]
        }
    ]
}
```

---

## 📂 GitHub Setup

```bash
# Initialize git (if not done)
git init

# Add all files
git add .

# First commit
git commit -m "feat: Complete Cancer Risk Prediction project with EDA, model, and deployment"

# Create repo on GitHub, then:
git branch -M main
git remote add origin https://github.com/Lokesh-7368/Cancer-Risk-Prediction.git
git push -u origin main
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.10+ |
| **ML Framework** | scikit-learn, XGBoost |
| **Hyperparameter Tuning** | Optuna |
| **Class Balancing** | SMOTE (imbalanced-learn), Class Weights |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Web Framework** | Streamlit |
| **Deployment** | Streamlit Community Cloud |
| **Version Control** | Git + GitHub |

---

## 👨‍💻 Author

**Lokesh**

- GitHub: [Lokesh-7368](https://github.com/Lokesh-7368)
- LinkedIn: [7368lokesh](https://linkedin.com/in/7368lokesh)

---

## 📄 License

This project is licensed under the MIT License.
