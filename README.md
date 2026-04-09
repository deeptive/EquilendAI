# ⚖️ EquiLend AI — Open Source ML Challenge 2026

> **Bridging the credit gap with fair, explainable, alternative-data machine learning.**

---

## 🌍 The Problem

Traditional credit scoring excludes millions of "credit invisible" individuals by relying solely on historical banking data. **EquiLend AI** addresses this by leveraging **alternative data** — utility payments, cash flow consistency, and digital footprints — to assess creditworthiness fairly and transparently.

---

## 🚀 The Challenge

A Streamlit UI prototype is provided as your starting point. The core "brain" is currently a **mathematical placeholder with 4 critical logical flaws**, and the data pipeline does not yet exist.

### Your Mission

1. **Fix the Core Logic** — Resolve the 4 logical bugs in `src/app.py`
2. **Build the ML Pipeline** — Replace the formula with a robust, fair, optimized ML engine
3. **Complete All 15 Tasks** — Cover data ingestion, preprocessing, modeling, and evaluation

---

## 🐛 Task 00 — Hidden Logical Bugs (Fix First)

Before building the ML pipeline, resolve these 4 bugs in `src/app.py`:

| # | Bug | Description |
|---|-----|-------------|
| 1 | **Division by Zero** | Score crashes when `utility_bill` is entered as `0` |
| 2 | **Age Guard Bypass** | System allows scoring for users under 18 |
| 3 | **Linear Scaling Flaw** | Simple ratio formula must be replaced with a trained ML model |
| 4 | **State Persistence** | Decisions disappear on refresh — must be saved to MongoDB |

---

## 🛠 Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.10+ |
| Frontend | Streamlit |
| ML Libraries | Scikit-learn, XGBoost, LightGBM, Imbalanced-learn (SMOTE), SHAP |
| Database | MongoDB Atlas (NoSQL) |
| Testing & Data | Pytest, Faker, Pandas |

---

## 📂 Repository Structure

```text
EquiLend-AI/
├── scripts/
│   └── generate_data.py          # RUN FIRST — generates synthetic dataset
├── src/
│   ├── app.py                    # Streamlit Dashboard UI
│   ├── data_ingestion/           # Task 01: MongoDB connection
│   ├── preprocessing/            # Tasks 02, 03, 04, 07: Cleaning & Encoding
│   ├── models/                   # Tasks 05, 06, 10, 11: Training logic
│   └── evaluation/               # Tasks 08, 09, 13: SHAP & Fairness logic
├── tests/                        # Task 12: Unit Tests
├── .env.example                  # MongoDB URI template
├── requirements.txt              # Python dependencies
├── Fairness_Report.md            # Task 15: Final fairness metrics
└── README.md
```

---

## ⚡ Getting Started

### 1. Prerequisites & Virtual Environment

Ensure Python 3.10+ is installed, then set up your environment:

```bash
# Create virtual environment
python -m venv venv

# Activate — Windows
.\venv\Scripts\activate

# Activate — Mac/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the Mock Dataset

Run this **before** building any models. It creates a synthetic dataset with intentional missing values and class imbalances:

```bash
python scripts/generate_data.py
```

This generates `data/equilend_mock_data.csv`. The `data/` folder is git-ignored for security.

### 4. Setup Environment Variables

```bash
cp .env.example .env
```

Open `.env` and add your **MongoDB Atlas URI**.

### 5. Run the Dashboard

```bash
python -m streamlit run src/app.py
```

---

## 🏆 Evaluation Rubric

| Criteria | 🥇 Gold | 🥈 Silver | 🥉 Bronze |
|----------|---------|-----------|-----------|
| **Model Quality** | Optimized XGBoost/LightGBM with AUC > 0.85 | Basic Random Forest | Hard-coded logic |
| **Explainability** | Interactive SHAP plots in Streamlit | Static feature importance in console | None |
| **Fairness** | Bias detection script + `Fairness_Report.md` | Fairness mentioned in README only | No bias checking |
| **Security/Eng** | Pytest validation + secure `.env` MongoDB | Basic try-except blocks | Hard-coded credentials |

---

## 📤 Submission Guidelines

1. **Fork** this repository
2. **Complete** all 15 tasks in the designated `src/` subfolders
3. **Populate** `Fairness_Report.md` with your model's final metrics
4. **Submit a Pull Request** with a summary of your XGBoost architecture and fairness outcomes

---

