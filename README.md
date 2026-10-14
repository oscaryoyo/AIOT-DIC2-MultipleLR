# 50 Startups – CRISP-DM Profit Prediction Analysis

> **CRISP-DM** end-to-end machine learning project: 20 feature-selection methods compared on the 50 Startups dataset with an interactive results dashboard.

[![GitHub Pages](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-0cb8b6?style=for-the-badge&logo=github)](https://YOUR_USERNAME.github.io/1014/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org)

---

## 🎯 Project Goal

Predict **Profit** for a startup using R&D Spend, Administration cost, Marketing Spend and State (New York / California / Florida), following the **CRISP-DM** methodology with rigorous train / validation / test separation.

---

## 📊 Results Summary

| Metric | Value |
|---|---|
| Best method | **Permutation Importance (k=3)** |
| Selected features | R&D Spend, Marketing Spend, State_New York |
| **Test RMSE** | **8,243** |
| **Test R²** | **0.916** |
| Baseline RMSE (mean pred.) | 33,777 |
| Improvement over baseline | **−75.6 %** |

### 5-Method Comparison (best k per method)

| Method | Best k | Test RMSE | Test R² |
|---|---|---|---|
| RFE | 1 | 7,714 | 0.9265 |
| LassoCV | 1 | 7,714 | 0.9265 |
| F-test | 1 | 7,714 | 0.9265 |
| RF Importance | 1 | 7,714 | 0.9265 |
| Forward SFS | 1 | 7,714 | 0.9265 |

> All five methods agree: **R&D Spend alone** gives the lowest test RMSE.

---

## 🚀 Live Demo

👉 **[https://YOUR_USERNAME.github.io/1014/](https://YOUR_USERNAME.github.io/1014/)**

*(Replace `YOUR_USERNAME` with your GitHub username after deploying)*

---

## 🗂️ Project Structure

```
1014/
├── index.html               # Interactive results dashboard
├── index.css                # Styling (statsunlock.com-inspired)
├── main.py                  # CRISP-DM pipeline – 20 feature-selection methods
├── analysis_5methods.py     # 5-method deep dive (RMSE & R² vs k)
├── 50_Startups.csv          # Dataset (50 rows, 5 columns)
├── crisp_dm_outputs/        # Outputs from main.py
│   ├── all_20_feature_selection_comparison.csv
│   ├── final_test_metrics.csv
│   ├── prediction_results.csv
│   ├── method_comparison.png
│   └── actual_vs_predicted.png
└── 5methods_outputs/        # Outputs from analysis_5methods.py
    ├── 5methods_comparison.csv
    ├── 5methods_rmse_r2.png
    └── (per-method PNG plots)
```

---

## ⚙️ How to Run

### Install dependencies

```bash
pip install pandas numpy scipy matplotlib scikit-learn joblib
```

### Run the full 20-method CRISP-DM pipeline

```bash
python main.py
```

Outputs saved to `crisp_dm_outputs/`.

### Run the 5-method deep dive

```bash
python analysis_5methods.py
```

Outputs saved to `5methods_outputs/`.

### View the interactive dashboard

Open `index.html` directly in a browser, or serve locally:

```bash
python -m http.server 5050
# Then open http://localhost:5050
```

---

## 🌐 Deploy to GitHub Pages

1. Create a new empty repository on GitHub
2. In your terminal:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/1014.git
   git branch -M main
   git push -u origin main
   ```
3. Go to **Settings → Pages → Source: Deploy from branch → main / root**
4. Your site is live at `https://YOUR_USERNAME.github.io/1014/`

---

## 🔬 Methodology

| Phase | Detail |
|---|---|
| **Split** | 60% fit · 20% validation · 20% test (stratified random, seed=42) |
| **Encoding** | One-hot for State; California = baseline (dropped) |
| **Scaling** | StandardScaler applied inside methods that require it (RFE, Lasso, SFS) |
| **Selection** | All 20 methods trained on fit set only |
| **Comparison** | Validation RMSE used to pick the winner |
| **Final score** | Test set touched exactly once after winner is chosen |

---

## 📚 Acknowledgements

- **Design & style** inspired by [StatsUnlock.com – Multiple Linear Regression Calculator](https://statsunlock.com/multiple-linear-regression-calculator/), a free online OLS tool with coefficients, R² & VIF.
- **Charts**: [Chart.js](https://www.chartjs.org/) v4
- **ML**: [scikit-learn](https://scikit-learn.org/)
- **Dataset**: 50 Startups (public domain, widely used for regression teaching)

---

*Results are exploratory due to small sample size (n = 50). Do not use for production decisions.*
