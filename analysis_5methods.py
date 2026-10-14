"""
5 Feature Selection Methods – RMSE & R² vs Number of Features
=============================================================
Methods compared:
  1. RFE          – Recursive Feature Elimination
  2. LassoCV      – L1-penalised regression path
  3. F-test       – SelectKBest (f_regression)
  4. RF Importance– Random Forest feature importance ranking
  5. Forward SFS  – Sequential forward selection

For every method we sweep k = 1 … n_features and record
Test RMSE and Test R² so the plots match the style of the
provided reference images.

Run: python analysis_5methods.py
Input: 50_Startups.csv in the same directory
Output: 5methods_outputs/
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    RFE, SelectKBest, SequentialFeatureSelector, f_regression,
)
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')
SEED = 42
DATA_PATH = Path('50_Startups.csv')
OUT = Path('5methods_outputs')
OUT.mkdir(exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def rmse(y, p):
    return np.sqrt(mean_squared_error(y, p))


# ── 1. Load & prepare data ────────────────────────────────────────────────────
print('Loading data …')
df = pd.read_csv(DATA_PATH)
required = ['R&D Spend', 'Administration', 'Marketing Spend', 'State', 'Profit']
df = df[required].drop_duplicates().dropna().copy()
df['State'] = df['State'].astype(str).str.strip()

states = sorted(df['State'].unique())

def encode(frame):
    result = pd.get_dummies(frame, columns=['State'], dtype=float)
    ordered = ['R&D Spend', 'Administration', 'Marketing Spend'] + [
        f'State_{s}' for s in states[1:]
    ]
    return result.reindex(columns=ordered, fill_value=0).astype(float)

X_raw = df.drop(columns='Profit')
y     = df['Profit'].reset_index(drop=True)

X_dev_raw, X_test_raw, y_dev, y_test = train_test_split(
    X_raw, y, test_size=0.20, random_state=SEED)

X_fit_raw, X_val_raw, y_fit, y_val = train_test_split(
    X_dev_raw, y_dev, test_size=0.25, random_state=SEED)

X_fit  = encode(X_fit_raw).reset_index(drop=True)
X_val  = encode(X_val_raw).reset_index(drop=True)
X_dev  = encode(X_dev_raw).reset_index(drop=True)
X_test = encode(X_test_raw).reset_index(drop=True)

n_feat   = X_fit.shape[1]          # 5 features total
feat_names = X_fit.columns.tolist()
k_range  = list(range(1, n_feat + 1))

scaler = StandardScaler()
X_fit_sc  = scaler.fit_transform(X_fit)
X_val_sc  = scaler.transform(X_val)
X_dev_sc  = scaler.fit_transform(X_dev)   # refit on full dev for final eval
X_test_sc = scaler.transform(X_test)

cv = KFold(n_splits=3, shuffle=True, random_state=SEED)

print(f'Features: {feat_names}')
print(f'Fit/Val/Test: {len(X_fit)}/{len(X_val)}/{len(X_test)}')

# ── 2. Rank features for each method ─────────────────────────────────────────
#   Each method produces a ranking (list of indices, best first).
#   We then evaluate LinearRegression using top-k features for k=1..5.

def get_ranking_rfe():
    """RFE ranking: eliminate features one at a time, record elimination order."""
    base = LinearRegression()
    rfe  = RFE(base, n_features_to_select=1, step=1).fit(X_fit_sc, y_fit)
    # ranking_: 1=selected last → most important.  Sort ascending = best first.
    order = np.argsort(rfe.ranking_)          # ascending: rank 1 first
    return order.tolist()

def get_ranking_lasso():
    """LassoCV gives coefficient magnitudes → rank descending."""
    lasso = LassoCV(cv=cv, random_state=SEED, max_iter=30000).fit(X_fit_sc, y_fit)
    return np.argsort(np.abs(lasso.coef_))[::-1].tolist()

def get_ranking_ftest():
    """F-test scores → rank descending."""
    scores = SelectKBest(f_regression, k='all').fit(X_fit, y_fit).scores_
    return np.argsort(scores)[::-1].tolist()

def get_ranking_rf():
    """Random-forest feature importance → rank descending."""
    rf = RandomForestRegressor(n_estimators=150, random_state=SEED,
                               min_samples_leaf=2).fit(X_fit, y_fit)
    return np.argsort(rf.feature_importances_)[::-1].tolist()

def get_ranking_forward_sfs():
    """
    True greedy forward selection: for k=1..n_features-1 re-run forward SFS.
    For the final k==n_features, all remaining features are added.
    """
    base = LinearRegression()
    order = []
    for k in k_range:
        if k >= n_feat:
            # Add all remaining features not yet in order
            remaining = [i for i in range(n_feat) if i not in order]
            order.extend(remaining)
            break
        sfs = SequentialFeatureSelector(
            base, n_features_to_select=k, direction='forward',
            scoring='neg_mean_squared_error', cv=cv
        ).fit(X_fit_sc, y_fit)
        selected = np.where(sfs.support_)[0].tolist()
        new = [i for i in selected if i not in order]
        order.extend(new)
    return order

# ── 3. Evaluate each method across k=1..5 ────────────────────────────────────
METHODS = {
    'RFE':           get_ranking_rfe,
    'LassoCV':       get_ranking_lasso,
    'F-test':        get_ranking_ftest,
    'RF Importance': get_ranking_rf,
    'Forward SFS':   get_ranking_forward_sfs,
}

# Decide whether to use scaled or raw X depending on the method
USE_SCALED = {'RFE', 'LassoCV', 'Forward SFS'}

results = {}   # method → {k: {'rmse': ..., 'r2': ..., 'features': [...]}}

for method_name, rank_fn in METHODS.items():
    print(f'\nRunning {method_name} …')
    ranking = rank_fn()

    # Re-fit scaler on dev, rank on dev as well for final test
    if method_name in USE_SCALED:
        Xtr, Xte = X_dev_sc, X_test_sc
    else:
        Xtr_df = X_dev
        Xte_df = X_test

    method_results = {}
    for k in k_range:
        top_k_idx  = ranking[:k]
        top_k_cols = [feat_names[i] for i in top_k_idx]

        if method_name in USE_SCALED:
            Xtr_k = Xtr[:, top_k_idx]
            Xte_k = Xte[:, top_k_idx]
        else:
            Xtr_k = Xtr_df[top_k_cols].values
            Xte_k = Xte_df[top_k_cols].values

        model = LinearRegression().fit(Xtr_k, y_dev)
        pred  = model.predict(Xte_k)

        method_results[k] = {
            'rmse':     rmse(y_test, pred),
            'r2':       r2_score(y_test, pred),
            'features': top_k_cols,
        }
        print(f'  k={k}  RMSE={method_results[k]["rmse"]:.2f}  '
              f'R2={method_results[k]["r2"]:.4f}  '
              f'Features={top_k_cols}')

    results[method_name] = method_results

# ── 4. Summary table ──────────────────────────────────────────────────────────
rows = []
for method, res in results.items():
    for k, vals in res.items():
        rows.append({
            'Method':          method,
            'Num Features':    k,
            'Selected Features': str(vals['features']),
            'RMSE':            round(vals['rmse'], 6),
            'R-squared':       round(vals['r2'],   6),
        })

summary = pd.DataFrame(rows)
summary.to_csv(OUT / '5methods_comparison.csv', index=False)

print('\n\n── Summary Table ────────────────────────────────────────────────')
print(summary.to_string(index=False))

# ── 5. Plots ──────────────────────────────────────────────────────────────────
# --- (A) Same style as reference: RMSE & R² vs #features, ONE LINE PER METHOD
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
markers = ['o', 's', '^', 'D', 'v']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('5 Feature Selection Methods – Test Performance vs. # Features',
             fontsize=13, fontweight='bold')

for ax, metric, label in [
    (axes[0], 'rmse', 'RMSE'),
    (axes[1], 'r2',   'R-squared'),
]:
    for (method, color, marker) in zip(METHODS.keys(), colors, markers):
        ys = [results[method][k][metric] for k in k_range]
        ax.plot(k_range, ys, color=color, marker=marker,
                linewidth=2, markersize=6, label=method)
    ax.set_xlabel('Number of Features', fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(f'{label} by Number of Features', fontsize=11)
    ax.set_xticks(k_range)
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(OUT / '5methods_rmse_r2.png', dpi=160)
plt.close()
print(f'\nSaved: {OUT / "5methods_rmse_r2.png"}')

# --- (B) Best-k bar chart: each method at its optimal k
best_rows = []
for method, res in results.items():
    best_k = min(res, key=lambda k: res[k]['rmse'])
    best_rows.append({
        'Method': method,
        'Best k': best_k,
        'RMSE':   res[best_k]['rmse'],
        'R2':     res[best_k]['r2'],
        'Features': ', '.join(res[best_k]['features']),
    })
best_df = pd.DataFrame(best_rows).sort_values('RMSE')
best_df.to_csv(OUT / '5methods_best_k.csv', index=False)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(best_df['Method'][::-1], best_df['RMSE'][::-1],
               color=colors[:len(best_df)], edgecolor='white')
ax.bar_label(bars, fmt='%.0f', padding=4, fontsize=9)
ax.set_xlabel('Test RMSE (lower is better)', fontsize=11)
ax.set_title('Best RMSE per Feature-Selection Method\n(at optimal # features)',
             fontsize=11, fontweight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(OUT / '5methods_best_rmse.png', dpi=160)
plt.close()
print(f'Saved: {OUT / "5methods_best_rmse.png"}')

# --- (C) Individual per-method plots (same style as reference images)
for method, color in zip(METHODS.keys(), colors):
    res = results[method]
    rmse_vals = [res[k]['rmse'] for k in k_range]
    r2_vals   = [res[k]['r2']   for k in k_range]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'{method} – RMSE & R² vs Number of Features',
                 fontsize=12, fontweight='bold')

    axes[0].plot(k_range, rmse_vals, color=color, marker='o',
                 linewidth=2, markersize=7)
    axes[0].set_xlabel('Number of Features'); axes[0].set_ylabel('RMSE')
    axes[0].set_title('RMSE by Number of Features')
    axes[0].set_xticks(k_range)
    axes[0].grid(axis='y', linestyle='--', alpha=0.4)

    axes[1].plot(k_range, r2_vals, color=color, marker='o',
                 linewidth=2, markersize=7)
    axes[1].set_xlabel('Number of Features'); axes[1].set_ylabel('R-squared')
    axes[1].set_title('R-squared by Number of Features')
    axes[1].set_xticks(k_range)
    axes[1].grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    safe = method.replace(' ', '_').lower()
    plt.savefig(OUT / f'{safe}_rmse_r2.png', dpi=160)
    plt.close()

print('\nAll individual method plots saved.')

# ── 6. Print best-k summary ───────────────────────────────────────────────────
print('\n── Best k per Method ────────────────────────────────────────────')
print(best_df.to_string(index=False))
print(f'\nAll outputs saved in: {OUT.resolve()}')
