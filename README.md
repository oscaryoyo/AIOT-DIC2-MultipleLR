# 🚀 AIOT-DIC2 — 50 Startups Profit Prediction
### Multiple Linear Regression · CRISP-DM · Feature Selection

[![GitHub Pages](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-0cb8b6?style=for-the-badge&logo=github)](https://oscaryoyo.github.io/AIOT-DIC2-MultipleLR/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Chart.js](https://img.shields.io/badge/Chart.js-4.4-FF6384?style=for-the-badge&logo=chart.js&logoColor=white)](https://chartjs.org)

---

## 📖 Project Overview

This project applies the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology to the **50 Startups dataset**, systematically comparing **20 feature-selection methods** for multiple linear regression. Results are presented through a beautifully designed interactive web dashboard inspired by [StatsUnlock.com](https://statsunlock.com/multiple-linear-regression-calculator/).

> **Goal:** Predict startup **Profit** from R&D Spend, Administration cost, Marketing Spend, and State — and determine which features matter most.

---

## 🌐 Live Demo

👉 **[https://oscaryoyo.github.io/AIOT-DIC2-MultipleLR/](https://oscaryoyo.github.io/AIOT-DIC2-MultipleLR/)**

### Dashboard Preview

| Section | Contents |
|---------|----------|
| 🎯 Business Understanding | Goal, success criteria, limitations |
| 📊 Data Understanding | Dataset stats, preview table, descriptive statistics |
| ⚙️ Data Preparation | Train/Val/Test split visualisation, encoding details |
| 🧪 Modeling – 20 Methods | Sortable, searchable comparison table for all 20 feature-selection methods |
| 🔬 5-Method Deep Dive | Interactive Chart.js RMSE & R² line charts across k=1→5 features |
| 📐 Statistical Inference | OLS equation, coefficients, SE, t-stats, p-values, 95% CIs, VIF, Adjusted R² |
| 📉 Residual Diagnostics | Residuals vs Fitted values & Normal Q-Q plots checking OLS assumptions |
| 💡 Inference vs Prediction | Deep dive comparing parameter estimation vs out-of-sample generalization |
| 🚀 Live Profit Predictor | Interactive calculator with real-time sliders & waterfall contribution breakdown |
| ✅ Evaluation | Final test metrics, actual vs predicted scatter chart, residuals bar chart |

---

## 📊 Key Results

### 1. OLS Statistical Inference (Fitted on Dev Set, N=40)

$$\widehat{\text{Profit}} = \$45,671.89 + 0.7841 \times (\text{R\&D Spend}) + 0.0391 \times (\text{Marketing Spend}) - 510.46 \times (\text{State\_New York})$$

| Term | Coef ($\beta$) | Std. Error | $t$-Statistic | $p$-Value | 95% Conf. Interval | VIF | Significance |
|------|---------------|------------|---------------|-----------|--------------------|-----|--------------|
| **Intercept** | **45,671.89** | 3,600.11 | 12.69 | < 0.0001 | [38,370.52, 52,973.26] | — | *** $p < 0.001$ |
| **R&D Spend** | **0.7841** | 0.0484 | 16.20 | < 0.0001 | [0.6859, 0.8823] | 2.28 | *** $p < 0.001$ |
| **Marketing Spend** | **0.0391** | 0.0201 | 1.95 | 0.0593 | [-0.0016, 0.0799] | 2.26 | • $p < 0.10$ |
| **State_New York** | **-510.46** | 3,260.93 | -0.16 | 0.8765 | [-7,123.94, 6,103.02] | 1.02 | ns ($p = 0.88$) |

* **Goodness-of-Fit:** $R^2 = 0.9519$, **Adjusted $R^2 = 0.9479$**, $F(3, 36) = 237.56$ ($p = 9.01 \times 10^{-24}$).
* **Multicollinearity:** All Variance Inflation Factors (VIF) are $< 2.3$, safely below the multicollinearity threshold of 5.0.
* **Parsimony Gain:** Compared to the full 5-feature baseline ($R^2 = 0.9537, \text{Adj } R^2 = 0.9469$), removing the insignificant variables (Administration $p=0.265$, Florida $p=0.811$) raised Adjusted $R^2$ to $0.9479$ and reduced test prediction error.

### 2. Final Test Results (Held-Out Test Set, N=10)

| Metric | Value |
|--------|-------|
| **Method** | Permutation Importance (k=3) |
| **Features** | R&D Spend, Marketing Spend, State_New York |
| **Test RMSE** | **8,243** |
| **Test R²** | **0.916** |
| Test MAE | 6,431 |
| Baseline RMSE (Mean Pred.) | 33,777 |
| **Improvement** | **−75.6 % vs. baseline** |

### 3. 20-Method Tournament (Validation RMSE)

| Rank | Method | Features | Val RMSE | Selected Features |
|------|--------|----------|----------|-------------------|
| 🥇 1 | Permutation Importance | 3 | **9,778** | R&D Spend, Administration, Marketing Spend |
| 2 | Random Forest Importance | 3 | 9,778 | R&D Spend, Administration, Marketing Spend |
| 3 | Gradient Boosting Importance | 3 | 9,778 | R&D Spend, Administration, Marketing Spend |
| 4 | RFECV | 1 | 10,278 | R&D Spend |
| 5 | Partial F-test | 1 | 10,278 | R&D Spend |
| … | … | … | … | … |
| 19 | All Features (Baseline) | 5 | 13,279 | All 5 predictors |

---

## 🗂️ Repository Structure

```
AIOT-DIC2-MultipleLR/
│
├── index.html                   # ✨ Interactive CRISP-DM dashboard & Live Predictor
├── index.css                    # 🎨 StatsUnlock-inspired design system
├── README.md                    # 📖 Project documentation
│
├── main.py                      # 🐍 CRISP-DM pipeline with OLS statistical inference & export
├── analysis_5methods.py         # 🐍 5-method deep dive (RMSE & R² sweep k=1→5)
├── 50_Startups.csv              # 📁 Dataset (50 obs., 5 cols.)
│
├── crisp_dm_outputs/            # 📂 Generated outputs & model artifacts
│   ├── all_20_feature_selection_comparison.csv
│   ├── final_test_metrics.csv
│   ├── ols_inference.csv        # 📐 Parameter estimates, SE, t, p-vals, 95% CIs, VIF
│   ├── prediction_results.csv
│   ├── website_data.json        # 🌐 Exported dataset powering the web UI
│   ├── startup_profit_model.joblib # 💾 Serialized model artifact for deployment
│   ├── method_comparison.png
│   ├── actual_vs_predicted.png
│   ├── residuals_vs_fitted.png  # 📉 Homoscedasticity & Linearity diagnostic plot
│   └── qq_plot.png              # 📉 Normal Q-Q residual plot
│
└── 5methods_outputs/            # 📂 Outputs from analysis_5methods.py
    ├── 5methods_comparison.csv
    ├── 5methods_best_k.csv
    ├── 5methods_rmse_r2.png
    └── 5methods_best_rmse.png
```

---

## ⚙️ How to Run Locally

### 1. Install dependencies

```bash
pip install pandas numpy scipy matplotlib scikit-learn joblib
```

### 2. Run the 20-method CRISP-DM pipeline

```bash
python main.py
```
Outputs saved to `crisp_dm_outputs/`.

### 3. Run the 5-method feature sweep

```bash
python analysis_5methods.py
```
Outputs saved to `5methods_outputs/`.

### 4. View the interactive dashboard

```bash
python -m http.server 5050
# Open: http://localhost:5050
```

Or simply open `index.html` directly in your browser.

---

## 🔬 Methods & Skills Used

### Machine Learning — Feature Selection (20 Methods)

| # | Method | Category |
|---|--------|----------|
| 00 | All Features (Baseline) | Baseline |
| 01 | RFE (Recursive Feature Elimination) | Wrapper |
| 02 | LassoCV | Penalised Regression |
| 03 | Forward Sequential Feature Selection | Wrapper |
| 04 | Backward Sequential Feature Selection | Wrapper |
| 05 | SelectKBest (F-test) | Filter |
| 06 | RFECV (with cross-validation) | Wrapper + CV |
| 07 | ElasticNetCV | Penalised Regression |
| 08 | VIF Elimination | Multicollinearity |
| 09 | Pearson Correlation | Filter |
| 10 | Mutual Information | Filter |
| 11 | Stepwise CV | Hybrid |
| 12 | Best Subset CV | Exhaustive |
| 13 | Permutation Importance | Model-Based |
| 14 | Ridge Coefficient Ranking | Penalised Regression |
| 15 | Random Forest Importance | Ensemble |
| 16 | Gradient Boosting Importance | Ensemble |
| 17 | *(reserved)* | — |
| 18 | Stability Selection (Lasso) | Randomised |
| 19 | Partial F-test | Statistical |

### ML Evaluation Skills
- **Train / Validation / Test split** — 60 / 20 / 20 (stratified, seed=42)
- Validation RMSE for method selection; test set touched once only
- Cross-validation within Stepwise CV, Best Subset CV, RFECV, ElasticNetCV, LassoCV
- One-hot encoding for categorical features; StandardScaler where required

### Python Libraries
| Library | Version | Usage |
|---------|---------|-------|
| `scikit-learn` | ≥1.9 | All feature selection & OLS regression |
| `pandas` / `numpy` | — | Data wrangling |
| `scipy` | — | Statistical tests |
| `matplotlib` | — | Output plots (PNG) |
| `joblib` | — | Model serialization |

### Web Development Skills
- **HTML5 + Vanilla CSS** — single-page app, no framework
- **CSS custom properties** — design tokens (colors, radius, shadows)
- **Sticky sidebar navigation** — active section highlight via scroll listener
- **Chart.js 4** — interactive line charts, scatter chart, bar chart
- **Inline data embedding** — all results as JS constants (no server required for GitHub Pages)
- **Sortable table** — click column headers to sort ascending/descending
- **Search + filter tabs** — real-time method filtering
- **Responsive design** — sidebar collapses on mobile
- **Google Fonts (Inter)** — modern typography
- Design inspired by **[StatsUnlock.com](https://statsunlock.com/multiple-linear-regression-calculator/)**

### DevOps / Deployment
- **Git** — version control, initial commit with 22 files
- **GitHub Pages** — static site hosting from `main` branch root

---

## � Methodology: CRISP-DM Walkthrough

### Step 1 — Business Understanding
Define the objective: predict profit, minimize RMSE, compare feature selection strategies.

### Step 2 — Data Understanding
- Dataset: 50 startups with 4 predictors + 1 target
- No missing values, no duplicates
- Strong positive correlation between R&D Spend and Profit (Pearson r ≈ 0.97)

### Step 3 — Data Preparation
- One-hot encode `State` (California = baseline dropped)
- Split: 30 fit / 10 validation / 10 test (fixed random seed=42)
- StandardScaler fitted on fit set only, applied consistently

### Step 4 — Modeling
- 20 feature-selection methods each select a feature subset from the fit set
- OLS regression trained on fit set with selected features
- Validation RMSE used to rank methods

### Step 5 — 5-Method Deep Dive
- Five key algorithms (RFE, LassoCV, F-test, RF Importance, Forward SFS) swept across k=1→5
- Test RMSE and R² computed for each (k, method) combination
- Interactive charts allow exploration of trade-offs

### Step 6 — Evaluation
- Winner (Permutation Importance, k=3) re-fitted on full dev set (fit+validation)
- Evaluated once on held-out test set: RMSE 8,243 · R² 0.916
- New startup prediction: R&D $120k · Admin $130k · Marketing $250k · CA → **$149,554**

---

## � Acknowledgements

- **Design inspiration**: [StatsUnlock.com – Multiple Linear Regression Calculator](https://statsunlock.com/multiple-linear-regression-calculator/), a free online OLS tool with coefficients, R² & VIF.
- **Charts**: [Chart.js](https://www.chartjs.org/) v4
- **ML framework**: [scikit-learn](https://scikit-learn.org/)
- **Dataset**: 50 Startups (public domain, commonly used for regression teaching)
- **Course**: AIOT-DIC2 — Data Intelligence & Computing

---

*Results are exploratory due to small sample size (n = 50). Confidence intervals would be very wide; treat all findings as illustrative.*
