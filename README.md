# 🎓 AI-Powered Student Placement Prediction Dashboard

A production-ready machine learning web application built with **Python**, **Streamlit**, and **Scikit-learn** that predicts student campus placement outcomes using four ML algorithms, hyperparameter tuning, cross-validation, and fully interactive Plotly visualisations.

---

## ✨ Features

| Category | Details |
|---|---|
| **UI** | Modern glassmorphism dark theme · Inter font · animated gradient elements · KPI cards |
| **Data** | CSV upload · auto sample-data generation · missing-value imputation · clean CSV export |
| **Visualisations** | Feature distributions · correlation heatmap · violin/box plots · categorical cross-tabs |
| **ML Models** | Logistic Regression · KNN · Random Forest · **Gradient Boosting** |
| **Tuning** | GridSearchCV · StratifiedKFold cross-validation · expanded hyperparameter grids |
| **Evaluation** | Accuracy · Precision · Recall · F1 · **ROC-AUC** · Confusion matrices · Radar chart · CV scores |
| **Unsupervised** | PCA 2D scatter · Scree plot · KMeans clustering · Elbow method |
| **Prediction** | Single-student form · Confidence gauge · Probability bar chart · Best-model auto-selection |
| **Export** | Download prediction CSV · batch prediction CSV · model comparison CSV |
| **Persistence** | All models saved via `joblib` · auto-loaded on app restart |

---

## 📁 Project Structure

```
artifacts/student-placement/
├── app.py                  ← Main Streamlit dashboard (7 pages)
├── preprocessing.py        ← Imputation, LabelEncoder, StandardScaler
├── train_model.py          ← GridSearchCV training, evaluation, joblib save
├── train_standalone.py     ← Standalone CLI training script
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── .streamlit/
│   └── config.toml         ← Dark theme + server config
├── models/                 ← Trained .joblib files (auto-created)
│   ├── logistic_regression.joblib
│   ├── knn.joblib
│   ├── random_forest.joblib
│   ├── gradient_boosting.joblib
│   └── session_artifacts.joblib   ← Full pipeline state (auto-loaded on start)
└── data/                   ← Place your CSV files here (optional)
```

---

## 📊 Dataset Schema

| Column | Type | Range | Description |
|---|---|---|---|
| Age | Numerical | 18–30 | Student age |
| Gender | Categorical | Male / Female | — |
| Degree | Categorical | B.Tech / BCA / MCA / MBA / B.Sc | Highest qualification |
| Branch | Categorical | CS / IT / ECE / Mechanical / Civil | Specialisation |
| CGPA | Numerical | 0–10 | Cumulative Grade Point Average |
| Internships | Numerical | 0–10 | Number of internships completed |
| Projects | Numerical | 0–20 | Number of academic / personal projects |
| Coding_Skills | Numerical | 1–10 | Self-assessed coding proficiency |
| Communication | Numerical | 1–10 | Communication skills score |
| Aptitude_Test | Numerical | 0–100 | Aptitude test score |
| Soft_Skills | Numerical | 1–10 | Soft skills score |
| Certifications | Numerical | 0–20 | Number of certifications earned |
| Backlogs | Numerical | 0–20 | Active academic backlogs |
| **Placement_Status** | **Target** | Placed / Not Placed | Label to predict |

---

## ⚙️ Local Setup

```bash
# 1. Clone
git clone https://github.com/your-username/student-placement-predictor.git
cd student-placement-predictor

# 2. Virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows

# 3. Install
pip install -r requirements.txt

# 4. Pre-train models (optional — app auto-trains on first use)
python train_standalone.py

# 5. Run
streamlit run app.py
```

Opens at **http://localhost:5000**

---

## ☁️ Streamlit Cloud Deployment

1. Push the project to a **public GitHub repository**
2. Visit [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app**
4. Select your repository and set **Main file path** → `app.py`
5. Click **Deploy**

Streamlit Cloud installs `requirements.txt` automatically. The `.streamlit/config.toml` sets `headless = true` and `enableCORS = false` which are required for cloud hosting.

> **Tip:** The app auto-generates sample data and trains models on first load — no manual setup needed on the cloud.

---

## 🧠 ML Pipeline

```
CSV Upload / Sample Data
  └─► Missing Value Imputation  (median / mode)
       └─► Label Encoding       (Gender, Degree, Branch)
            └─► Standard Scaling (z-score on numerical columns)
                 └─► Stratified Train / Test Split
                      └─► GridSearchCV + StratifiedKFold (5-fold)
                           ├─► Logistic Regression  (C, solver, class_weight)
                           ├─► KNN                  (k, weights, metric)
                           ├─► Random Forest         (n_est, depth, features)
                           └─► Gradient Boosting     (lr, depth, subsample)
                                └─► Evaluate: Acc · Precision · Recall · F1 · ROC-AUC
                                     └─► Persist with joblib → models/*.joblib
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| streamlit | ≥ 1.32 | Web application framework |
| scikit-learn | ≥ 1.4 | ML algorithms, preprocessing, evaluation |
| pandas | ≥ 2.2 | Data manipulation |
| numpy | ≥ 1.26 | Numerical computation |
| plotly | ≥ 5.18 | Interactive charts |
| matplotlib | ≥ 3.8 | Static chart support |
| seaborn | ≥ 0.13 | Statistical visualisations |
| joblib | ≥ 1.3 | Model serialisation |

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first.

---

## 📄 License

[MIT](LICENSE)
