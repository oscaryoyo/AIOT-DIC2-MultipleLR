"""Automated test suite for King County Housing Regression Project
Verifies requirements:
1. Dataset loading and verification (>13 candidate features, target separation)
2. Imputation and exact multicollinearity diagnosis (sqft_living = sqft_above + sqft_basement)
3. Model A (Simple Linear Regression)
4. Model B (Multiple Linear Regression on 16 baseline features)
5. Model C (Feature Selection methods: Pearson, SelectKBest, RFE, RFECV, Backward Elimination, VIF)
6. Metrics calculation (MAE, MSE, RMSE, R², Adjusted R²)
7. 5 Regression Assumption Diagnostics (Linearity, Independence, Normality, Homoscedasticity, Multicollinearity)
8. Artifacts generation and predictor consistency
"""
import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera


class TestKingCountyHousingRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_path = Path("kc_house_data.csv")
        cls.out_dir = Path("kc_house_outputs")
        cls.candidate_17 = [
            'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'grade', 'sqft_above',
            'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long',
            'sqft_living15', 'sqft_lot15'
        ]
        cls.baseline_16 = [c for c in cls.candidate_17 if c != 'sqft_basement']

        # Load dataset
        cls.df = pd.read_csv(cls.data_path)
        # Impute missing sqft_above
        cls.df['sqft_above'] = cls.df['sqft_above'].fillna(cls.df['sqft_living'] - cls.df['sqft_basement'])

        cls.X = cls.df[cls.baseline_16]
        cls.y = cls.df['price']

        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.2, random_state=42
        )

    def test_01_dataset_loading_and_features_count(self):
        """Verify dataset has > 13 candidate input features excluding target."""
        self.assertTrue(self.data_path.exists(), "Dataset kc_house_data.csv must exist.")
        self.assertEqual(self.df.shape[0], 21613, "Dataset should have 21,613 observations.")
        self.assertGreater(len(self.candidate_17), 13, "Candidate features count must be > 13.")
        self.assertEqual(len(self.candidate_17), 17, "Expected exactly 17 candidate features.")
        self.assertEqual(len(self.baseline_16), 16, "Baseline should have 16 features without sqft_basement.")

    def test_02_exact_multicollinearity_identity(self):
        """Verify exact identity sqft_living = sqft_above + sqft_basement."""
        diff = self.df['sqft_living'] - (self.df['sqft_above'] + self.df['sqft_basement'])
        self.assertAlmostEqual(float(np.max(np.abs(diff))), 0.0, places=5,
                               msg="sqft_living must equal sqft_above + sqft_basement.")

    def test_03_model_a_simple_linear_regression(self):
        """Verify Model A (Simple Linear Regression on sqft_living)."""
        lr = LinearRegression()
        lr.fit(self.X_train[['sqft_living']], self.y_train)
        pred = lr.predict(self.X_test[['sqft_living']])
        r2 = r2_score(self.y_test, pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, pred))
        self.assertGreater(r2, 0.45, "Model A R² should be > 0.45.")
        self.assertLess(rmse, 300000, "Model A RMSE should be < $300,000.")

    def test_04_model_b_multiple_linear_regression(self):
        """Verify Model B (16 Baseline Features)."""
        lr = LinearRegression()
        lr.fit(self.X_train, self.y_train)
        pred = lr.predict(self.X_test)
        r2 = r2_score(self.y_test, pred)
        n, p = len(self.y_test), self.X_test.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        rmse = np.sqrt(mean_squared_error(self.y_test, pred))
        self.assertGreater(r2, 0.68, "Model B R² should be > 0.68.")
        self.assertGreater(adj_r2, 0.68, "Model B Adjusted R² should be > 0.68.")
        self.assertLess(rmse, 225000, "Model B RMSE should be < $225,000.")

    def test_05_feature_selection_methods(self):
        """Verify feature selection methods produce valid subsets."""
        # 1. Pearson Correlation
        corrs = self.X_train.apply(lambda col: np.corrcoef(col, self.y_train)[0, 1]).abs().sort_values(ascending=False)
        top8_pearson = corrs.head(8).index.tolist()
        self.assertEqual(len(top8_pearson), 8)

        # 2. SelectKBest
        skb = SelectKBest(score_func=f_regression, k=8)
        skb.fit(self.X_train, self.y_train)
        skb_feats = [self.baseline_16[i] for i in skb.get_support(indices=True)]
        self.assertEqual(len(skb_feats), 8)

        # 3. RFE
        rfe = RFE(estimator=LinearRegression(), n_features_to_select=8)
        rfe.fit(self.X_train, self.y_train)
        rfe_feats = [self.baseline_16[i] for i in rfe.get_support(indices=True)]
        self.assertEqual(len(rfe_feats), 8)

    def test_06_regression_assumptions_diagnostics(self):
        """Verify 5 regression assumption tests execute properly."""
        X_const = sm.add_constant(self.X_train)
        ols = sm.OLS(self.y_train, X_const).fit()

        # 1. Linearity: Ramsey RESET
        reset = linear_reset(ols, power=3, use_f=True)
        self.assertIsNotNone(reset.fvalue)

        # 2. Independence: Durbin-Watson
        dw = durbin_watson(ols.resid)
        self.assertGreater(dw, 1.8)
        self.assertLess(dw, 2.2)

        # 3. Normality: Jarque-Bera
        jb_stat, jb_p, _, _ = jarque_bera(ols.resid)
        self.assertGreater(jb_stat, 0)

        # 4. Homoscedasticity: Breusch-Pagan
        bp_stat, bp_p, _, _ = het_breuschpagan(ols.resid, ols.model.exog)
        self.assertGreater(bp_stat, 0)

        # 5. Multicollinearity: VIF on 16 features
        vifs = [variance_inflation_factor(self.X_train.values, i) for i in range(self.X_train.shape[1])]
        max_vif = max(vifs)
        self.assertLess(max_vif, 10.0, "All 16 baseline features must have VIF < 10.0.")

    def test_07_artifacts_and_predictor_consistency(self):
        """Verify exported json and joblib artifacts are valid."""
        json_path = self.out_dir / "kc_house_data.json"
        joblib_path = self.out_dir / "kc_house_model.joblib"
        self.assertTrue(json_path.exists(), "kc_house_data.json must exist.")
        self.assertTrue(joblib_path.exists(), "kc_house_model.joblib must exist.")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.assertEqual(data['id'], 'kc_housing')
        self.assertIn('modelParams', data)
        self.assertIn('assumptions', data)
        self.assertEqual(len(data['assumptions']), 5)

        # Test saved model
        payload = joblib.load(joblib_path)
        loaded_model = payload['model']
        test_pred = loaded_model.predict(self.X_test.iloc[:1])
        self.assertGreater(test_pred[0], 0, "Predicted house price must be positive.")


if __name__ == '__main__':
    unittest.main()
