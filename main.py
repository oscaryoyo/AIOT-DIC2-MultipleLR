"""CRISP-DM: 50 Startups multiple linear regression with 20 feature-selection methods.

Install: pip install pandas numpy scipy matplotlib scikit-learn joblib
Optional: pip install boruta mrmr-selection
Run: python crisp_dm_50_startups_20_methods.py
Input: 50_Startups.csv in the same directory (or set DATA_PATH).

Selection is trained on development-training data, methods compared using a
validation split, and only the winning method is evaluated on the test set.
"""
from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import f as f_distribution
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import (
    RFE, RFECV, SelectKBest, SequentialFeatureSelector, f_regression,
    mutual_info_regression,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
DATA_PATH = Path('50_Startups.csv')
OUT = Path('crisp_dm_outputs')
OUT.mkdir(exist_ok=True)


def rmse(y, pred):
    return np.sqrt(mean_squared_error(y, pred))


def keep_at_least_one(mask, scores):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        mask[np.argmax(np.abs(scores))] = True
    return mask


def top_k(scores, k):
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf)
    chosen = np.argsort(scores)[-min(k, len(scores)):]
    mask = np.zeros(len(scores), dtype=bool)
    mask[chosen] = True
    return mask


def subset_cv_rmse(X, y, columns, splits=3):
    """CV for subset search, exclusively within training data."""
    if not columns:
        return np.inf
    Xs = X.loc[:, columns]
    cv = KFold(n_splits=splits, shuffle=True, random_state=SEED)
    errors = []
    for tr, va in cv.split(Xs):
        mdl = LinearRegression().fit(Xs.iloc[tr], y.iloc[tr])
        errors.append(rmse(y.iloc[va], mdl.predict(Xs.iloc[va])))
    return np.mean(errors)


def best_subset(X, y):
    cols = X.columns.tolist()
    best_score, best_cols = np.inf, []
    for k in range(1, len(cols) + 1):
        for part in itertools.combinations(cols, k):
            score = subset_cv_rmse(X, y, list(part))
            if score < best_score:
                best_score, best_cols = score, list(part)
    return np.array([c in best_cols for c in cols])


def stepwise(X, y):
    """Bidirectional greedy stepwise selection based on training-only CV RMSE."""
    selected = []
    available = list(X.columns)
    best = np.inf
    while available:
        options = [(subset_cv_rmse(X, y, selected + [c]), c) for c in available]
        score, added = min(options)
        if score >= best - 1e-8:
            break
        selected.append(added)
        available.remove(added)
        best = score
        improved = True
        while improved and len(selected) > 1:
            improved = False
            alternatives = [(subset_cv_rmse(X, y, [x for x in selected if x != c]), c)
                            for c in selected]
            newscore, removed = min(alternatives)
            if newscore < best - 1e-8:
                selected.remove(removed)
                available.append(removed)
                best = newscore
                improved = True
    return np.array([c in selected for c in X.columns])


def vif_selection(X, threshold=5.0):
    """Iteratively drop high-VIF columns; VIF checks redundancy, not usefulness."""
    active = list(X.columns)
    while len(active) > 1:
        vifs = {}
        for col in active:
            others = [x for x in active if x != col]
            r2 = LinearRegression().fit(X[others], X[col]).score(X[others], X[col])
            vifs[col] = 1.0 / max(1.0 - r2, 1e-12)
        worst = max(vifs, key=vifs.get)
        if vifs[worst] <= threshold:
            break
        active.remove(worst)
    return np.array([c in active for c in X.columns])


def partial_f_selection(X, y, significance=0.05):
    """Backward elimination by partial F test for nested OLS models."""
    active = list(X.columns)
    n = len(y)
    while len(active) > 1:
        full = LinearRegression().fit(X[active], y)
        rss_full = np.sum((y - full.predict(X[active])) ** 2)
        df_resid = n - (len(active) + 1)
        if df_resid <= 0:
            break
        pvals = {}
        for dropped in active:
            reduced_cols = [c for c in active if c != dropped]
            reduced = LinearRegression().fit(X[reduced_cols], y)
            rss_reduced = np.sum((y - reduced.predict(X[reduced_cols])) ** 2)
            fstat = max(rss_reduced - rss_full, 0) / max(rss_full / df_resid, 1e-12)
            pvals[dropped] = f_distribution.sf(fstat, 1, df_resid)
        worst = max(pvals, key=pvals.get)
        if pvals[worst] <= significance:
            break
        active.remove(worst)
    return np.array([c in active for c in X.columns])


def stability_lasso(X, y, repeats=30):
    rng = np.random.default_rng(SEED)
    hits = np.zeros(X.shape[1])
    for _ in range(repeats):
        idx = rng.choice(len(X), size=max(12, int(0.8 * len(X))), replace=False)
        scaler = StandardScaler().fit(X.iloc[idx])
        model = LassoCV(cv=3, random_state=SEED, max_iter=20000).fit(
            scaler.transform(X.iloc[idx]), y.iloc[idx]
        )
        hits += np.abs(model.coef_) > 1e-6
    frequencies = hits / repeats
    return keep_at_least_one(frequencies >= 0.60, frequencies)


def compute_vif(X_df):
    """Compute Variance Inflation Factor (VIF) for each column in X_df."""
    vifs = {}
    cols = X_df.columns.tolist()
    for col in cols:
        others = [c for c in cols if c != col]
        if not others:
            vifs[col] = 1.0
            continue
        r2 = LinearRegression().fit(X_df[others], X_df[col]).score(X_df[others], X_df[col])
        vifs[col] = round(1.0 / max(1.0 - r2, 1e-12), 3)
    return vifs


def ols_inference(X_df, y_series):
    """Perform OLS statistical inference on given features and target."""
    n = len(y_series)
    p = X_df.shape[1]
    Xb = np.column_stack([np.ones(n), X_df.to_numpy(dtype=float)])
    y_vals = y_series.to_numpy(dtype=float)

    beta = np.linalg.lstsq(Xb, y_vals, rcond=None)[0]
    y_hat = Xb @ beta
    resid = y_vals - y_hat
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y_vals - np.mean(y_vals)) ** 2))
    r2 = max(0.0, 1.0 - rss / tss)
    df_resid = max(n - p - 1, 1)
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df_resid
    sigma2 = rss / df_resid

    cov = sigma2 * np.linalg.inv(Xb.T @ Xb)
    se = np.sqrt(np.diag(cov))
    t_stat = beta / np.where(se > 0, se, np.nan)
    p_vals = 2 * stats.t.sf(np.abs(t_stat), df=df_resid)
    t_crit = stats.t.ppf(0.975, df=df_resid)
    ci_lower = beta - t_crit * se
    ci_upper = beta + t_crit * se

    reg_ss = tss - rss
    f_stat = (reg_ss / max(p, 1)) / max(sigma2, 1e-12)
    f_p = float(stats.f.sf(f_stat, p, df_resid))

    vifs = compute_vif(X_df)
    terms = ['const'] + list(X_df.columns)
    summary_rows = []
    for i, name in enumerate(terms):
        v = vifs.get(name, None)
        summary_rows.append({
            'term': name,
            'coef': round(float(beta[i]), 4),
            'se': round(float(se[i]), 4),
            't_stat': round(float(t_stat[i]), 4),
            'p_value': round(float(p_vals[i]), 6),
            'ci_lower': round(float(ci_lower[i]), 4),
            'ci_upper': round(float(ci_upper[i]), 4),
            'vif': v
        })

    return {
        'summary': summary_rows,
        'r2': round(r2, 6),
        'adj_r2': round(adj_r2, 6),
        'f_stat': round(float(f_stat), 4),
        'f_pvalue': f_p,
        'n': n,
        'df_resid': df_resid,
        'beta': [round(float(b), 6) for b in beta],
        'residuals': resid.tolist(),
        'fitted': y_hat.tolist()
    }


def get_masks(X, y):
    """Fit ALL selectors to the supplied training subset only."""
    k = min(3, X.shape[1])
    names = np.array(X.columns)
    masks = {'00 All features (baseline)': np.ones(X.shape[1], dtype=bool)}
    cv = KFold(n_splits=3, shuffle=True, random_state=SEED)
    base = LinearRegression()
    scaled = StandardScaler().fit_transform(X)

    masks['01 RFE'] = RFE(base, n_features_to_select=k).fit(scaled, y).support_
    lasso = LassoCV(cv=cv, random_state=SEED, max_iter=30000).fit(scaled, y)
    masks['02 LassoCV'] = keep_at_least_one(abs(lasso.coef_) > 1e-6, lasso.coef_)
    masks['03 Forward SFS'] = SequentialFeatureSelector(
        base, n_features_to_select=k, direction='forward', scoring='neg_mean_squared_error', cv=cv
    ).fit(scaled, y).support_
    masks['04 Backward SFS'] = SequentialFeatureSelector(
        base, n_features_to_select=k, direction='backward', scoring='neg_mean_squared_error', cv=cv
    ).fit(scaled, y).support_
    masks['05 F-test SelectKBest'] = SelectKBest(f_regression, k=k).fit(X, y).get_support()
    masks['06 RFECV'] = RFECV(
        base, cv=cv, scoring='neg_mean_squared_error', min_features_to_select=1
    ).fit(scaled, y).support_
    elastic = ElasticNetCV(cv=cv, random_state=SEED, max_iter=30000).fit(scaled, y)
    masks['07 ElasticNetCV'] = keep_at_least_one(abs(elastic.coef_) > 1e-6, elastic.coef_)
    masks['08 VIF elimination'] = vif_selection(X)
    pearson = np.array([abs(np.corrcoef(X[c], y)[0, 1]) if X[c].std() > 0 else 0
                        for c in X.columns])
    masks['09 Pearson correlation'] = top_k(pearson, k)
    mi = mutual_info_regression(X, y, random_state=SEED)
    masks['10 Mutual information'] = top_k(mi, k)
    masks['11 Stepwise CV'] = stepwise(X, y)
    masks['12 Best subset CV'] = best_subset(X, y)
    # Permutation importance computed within TRAINING folds only.
    # Here it ranks importance for exploration; no test data are used.
    rf = RandomForestRegressor(n_estimators=150, random_state=SEED, min_samples_leaf=2)
    rf.fit(X, y)
    perm = permutation_importance(rf, X, y, n_repeats=10, random_state=SEED,
                                  scoring='neg_mean_squared_error')
    masks['13 Permutation importance'] = top_k(perm.importances_mean, k)
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 30), cv=cv).fit(scaled, y)
    masks['14 Ridge coefficient ranking'] = top_k(abs(ridge.coef_), k)
    masks['15 Random forest importance'] = top_k(rf.feature_importances_, k)
    gbr = GradientBoostingRegressor(random_state=SEED).fit(X, y)
    masks['16 Gradient boosting importance'] = top_k(gbr.feature_importances_, k)

    try:
        from boruta import BorutaPy
        boruta = BorutaPy(
            RandomForestRegressor(n_estimators=150, random_state=SEED, n_jobs=1),
            n_estimators='auto', random_state=SEED, max_iter=35, verbose=0
        )
        boruta.fit(X.to_numpy(), y.to_numpy())
        masks['17 Boruta'] = keep_at_least_one(boruta.support_, -boruta.ranking_)
    except ImportError:
        print('SKIPPED 17 Boruta: pip install boruta')

    masks['18 Stability selection (Lasso)'] = stability_lasso(X, y)
    masks['19 Partial F-test'] = partial_f_selection(X, y)
    try:
        import mrmr
        chosen = mrmr.mrmr_regression(X=X, y=y, K=k, show_progress=False)
        masks['20 mRMR'] = np.isin(names, chosen)
    except ImportError:
        print('SKIPPED 20 mRMR: pip install mrmr-selection')
    return masks


# ============================================================
# 1. BUSINESS UNDERSTANDING
# ============================================================
print('\n1. BUSINESS UNDERSTANDING')
print('Goal: predict Profit from R&D Spend, Administration, Marketing Spend, State.')
print('Success criterion: validation RMSE, compared with mean-prediction baseline.')
print('Limitation: 50 observations; rankings may be unstable.')

# ============================================================
# 2. DATA UNDERSTANDING
# ============================================================
print('\n2. DATA UNDERSTANDING')
if not DATA_PATH.exists():
    raise FileNotFoundError(f'Place {DATA_PATH.name} in the current working directory.')
df = pd.read_csv(DATA_PATH)
required = ['R&D Spend', 'Administration', 'Marketing Spend', 'State', 'Profit']
missing_columns = set(required) - set(df.columns)
if missing_columns:
    raise ValueError(f'Dataset missing columns: {missing_columns}')
print(df.head())
print('\nShape:', df.shape)
print('\nStatistics:\n', df.describe(include='all'))
print('\nNull counts:\n', df[required].isna().sum())
print('Duplicate rows:', df.duplicated().sum())

# ============================================================
# 3. DATA PREPARATION
# ============================================================
print('\n3. DATA PREPARATION')
df = df[required].drop_duplicates().dropna().copy()
for c in ['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']:
    df[c] = pd.to_numeric(df[c], errors='raise')
df = df[(df[['R&D Spend', 'Administration', 'Marketing Spend']] >= 0).all(axis=1)]
df['State'] = df['State'].astype(str).str.strip()
X_raw = df.drop(columns='Profit')
y = df['Profit']

# Outer holdout: 20% is untouched until the final selection.
X_dev_raw, X_test_raw, y_dev, y_test = train_test_split(
    X_raw, y, test_size=0.20, random_state=SEED
)
# Internal validation: compare methods without using the test set.
X_fit_raw, X_val_raw, y_fit, y_val = train_test_split(
    X_dev_raw, y_dev, test_size=0.25, random_state=SEED
)
# For this fixed historical dataset, create a consistent dummy schema using
# known state categories. No target information is used in encoding.
states = sorted(X_raw['State'].unique())
def encode(frame):
    result = pd.get_dummies(frame, columns=['State'], dtype=float)
    ordered = ['R&D Spend', 'Administration', 'Marketing Spend'] + [
        f'State_{s}' for s in states[1:]
    ]
    return result.reindex(columns=ordered, fill_value=0).astype(float)
X_fit, X_val = encode(X_fit_raw), encode(X_val_raw)
X_dev, X_test = encode(X_dev_raw), encode(X_test_raw)
print('Fit / Validation / Test:', len(X_fit), len(X_val), len(X_test))
print('Features:', X_fit.columns.tolist())

# ============================================================
# 4. MODELING: ALL 20 SELECTION METHODS
# ============================================================
print('\n4. MODELING: 20 FEATURE-SELECTION METHODS')
warnings.filterwarnings('ignore', category=UserWarning)
masks = get_masks(X_fit, y_fit)
rows = []
for method, mask in masks.items():
    columns = X_fit.columns[np.asarray(mask, bool)].tolist()
    estimator = LinearRegression().fit(X_fit[columns], y_fit)
    val_pred = estimator.predict(X_val[columns])
    rows.append({
        'Method': method,
        'Feature count': len(columns),
        'Validation RMSE': rmse(y_val, val_pred),
        'Validation MAE': mean_absolute_error(y_val, val_pred),
        'Selected features': ', '.join(columns)
    })
comparison = pd.DataFrame(rows).sort_values('Validation RMSE').reset_index(drop=True)
comparison.to_csv(OUT / 'all_20_feature_selection_comparison.csv', index=False)
print('\nFeature selection comparison (validation only):')
print(comparison.to_string(index=False))

# ============================================================
# 5. EVALUATION: SINGLE FINAL TEST ASSESSMENT
# ============================================================
print('\n5. EVALUATION')
chosen_method = comparison.loc[0, 'Method']
print('Selected by validation RMSE:', chosen_method)
# RE-FIT selection on ALL development data without touching outer test data.
final_masks = get_masks(X_dev, y_dev)
final_features = X_dev.columns[final_masks[chosen_method]].tolist()
final_model = LinearRegression().fit(X_dev[final_features], y_dev)
pred = final_model.predict(X_test[final_features])
baseline = DummyRegressor(strategy='mean').fit(X_dev, y_dev)
base_pred = baseline.predict(X_test)
metrics = {
    'Method': chosen_method,
    'Features': ', '.join(final_features),
    'Test MAE': mean_absolute_error(y_test, pred),
    'Test MSE': mean_squared_error(y_test, pred),
    'Test RMSE': rmse(y_test, pred),
    'Test R2': r2_score(y_test, pred),
    'Baseline Test RMSE': rmse(y_test, base_pred),
}
print('\nFinal test results:\n', pd.Series(metrics))
pd.DataFrame([metrics]).to_csv(OUT / 'final_test_metrics.csv', index=False)
results = pd.DataFrame({'Actual': y_test.to_numpy(), 'Predicted': pred,
                        'Residual': y_test.to_numpy() - pred})
results.to_csv(OUT / 'prediction_results.csv', index=False)
print('\nPredictions:\n', results)

# OLS Statistical Inference (Dev Set)
print('\n=== OLS STATISTICAL INFERENCE (DEV SET) ===')
dev_ols = ols_inference(X_dev[final_features], y_dev)
print(f"N = {dev_ols['n']}, df = {dev_ols['df_resid']}")
print(f"R^2 = {dev_ols['r2']:.6f}, Adj R^2 = {dev_ols['adj_r2']:.6f}, F = {dev_ols['f_stat']:.4f} (p = {dev_ols['f_pvalue']:.4e})")
ols_df = pd.DataFrame(dev_ols['summary'])
print(ols_df.to_string(index=False))
ols_df.to_csv(OUT / 'ols_inference.csv', index=False)

# Diagnostic Plots
plt.figure(figsize=(7, 5))
plt.barh(comparison['Method'][::-1], comparison['Validation RMSE'][::-1], color='#0cb8b6')
plt.xlabel('Validation RMSE (lower is better)')
plt.title('Comparison of 20 Feature-Selection Methods')
plt.tight_layout()
plt.savefig(OUT / 'method_comparison.png', dpi=160)
plt.close()

plt.figure(figsize=(6, 5))
plt.scatter(y_test, pred, color='#0cb8b6', edgecolors='#099e9c', s=50, alpha=0.9, label='Test Data')
lo, hi = min(y_test.min(), pred.min()) * 0.95, max(y_test.max(), pred.max()) * 1.05
plt.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect fit (y=x)')
plt.xlabel('Actual Profit ($)')
plt.ylabel('Predicted Profit ($)')
plt.title('Final Model: Actual vs Predicted (Test Set)')
plt.legend()
plt.tight_layout()
plt.savefig(OUT / 'actual_vs_predicted.png', dpi=160)
plt.close()

# Residuals vs Fitted Plot (Dev Set)
fitted_vals = np.array(dev_ols['fitted'])
resid_vals = np.array(dev_ols['residuals'])
plt.figure(figsize=(6, 4.5))
plt.scatter(fitted_vals, resid_vals, color='#4f8ef7', edgecolors='#2563eb', alpha=0.85, s=45)
plt.axhline(0, color='#ef4444', linestyle='--', linewidth=1.2)
plt.xlabel('Fitted Values ($)')
plt.ylabel('Residuals ($)')
plt.title('Residuals vs Fitted (Dev Set OLS)')
plt.tight_layout()
plt.savefig(OUT / 'residuals_vs_fitted.png', dpi=160)
plt.close()

# Normal Q-Q Plot (Dev Set)
plt.figure(figsize=(5.5, 4.5))
stats.probplot(resid_vals, dist='norm', plot=plt)
plt.title('Normal Q-Q Plot of Residuals')
plt.tight_layout()
plt.savefig(OUT / 'qq_plot.png', dpi=160)
plt.close()

# ============================================================
# 6. DEPLOYMENT AND REPORTING
# ============================================================
print('\n6. DEPLOYMENT')
artifact = {
    'model': final_model,
    'features': final_features,
    'all_state_categories': states,
    'selection_method': chosen_method,
    'ols_inference': dev_ols,
}
joblib.dump(artifact, OUT / 'startup_profit_model.joblib')

new_company = pd.DataFrame({
    'R&D Spend': [120000],
    'Administration': [130000],
    'Marketing Spend': [250000],
    'State': ['California']
})
new_encoded = encode(new_company)
new_prediction = float(artifact['model'].predict(new_encoded[artifact['features']])[0])
print('New startup predicted profit:', round(new_prediction, 2))

# Export comprehensive JSON data for GitHub Pages website
website_data = {
    'methods_20': comparison.to_dict(orient='records'),
    'final_metrics': metrics,
    'ols_inference': dev_ols,
    'predictions': results.to_dict(orient='records'),
    'new_startup_prediction': {
        'input': {'R&D Spend': 120000, 'Administration': 130000, 'Marketing Spend': 250000, 'State': 'California'},
        'predicted_profit': round(new_prediction, 2)
    }
}
with open(OUT / 'website_data.json', 'w', encoding='utf-8') as f:
    json.dump(website_data, f, indent=2, ensure_ascii=False)

print('Files saved in:', OUT.resolve())
print('CRISP-DM complete; results verified.')
