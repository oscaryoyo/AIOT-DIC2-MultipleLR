const BOSTON_RAW = {
  "name": "Boston Housing — Housing Value Prediction",
  "target": "MEDV (Median home value in ,000s)",
  "rows": 506,
  "cols": 14,
  "features_count": 13,
  "features_list": [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT"
  ],
  "preview": [
    {
      "CRIM": 0.01,
      "ZN": 18.0,
      "INDUS": 2.31,
      "CHAS": 0,
      "NOX": 0.54,
      "RM": 6.58,
      "AGE": 65.2,
      "DIS": 4.09,
      "RAD": 1,
      "TAX": 296.0,
      "PTRATIO": 15.3,
      "B": 396.9,
      "LSTAT": 4.98,
      "MEDV": 24.0
    },
    {
      "CRIM": 0.03,
      "ZN": 0.0,
      "INDUS": 7.07,
      "CHAS": 0,
      "NOX": 0.47,
      "RM": 6.42,
      "AGE": 78.9,
      "DIS": 4.97,
      "RAD": 2,
      "TAX": 242.0,
      "PTRATIO": 17.8,
      "B": 396.9,
      "LSTAT": 9.14,
      "MEDV": 21.6
    },
    {
      "CRIM": 0.03,
      "ZN": 0.0,
      "INDUS": 7.07,
      "CHAS": 0,
      "NOX": 0.47,
      "RM": 7.18,
      "AGE": 61.1,
      "DIS": 4.97,
      "RAD": 2,
      "TAX": 242.0,
      "PTRATIO": 17.8,
      "B": 392.83,
      "LSTAT": 4.03,
      "MEDV": 34.7
    },
    {
      "CRIM": 0.03,
      "ZN": 0.0,
      "INDUS": 2.18,
      "CHAS": 0,
      "NOX": 0.46,
      "RM": 7.0,
      "AGE": 45.8,
      "DIS": 6.06,
      "RAD": 3,
      "TAX": 222.0,
      "PTRATIO": 18.7,
      "B": 394.63,
      "LSTAT": 2.94,
      "MEDV": 33.4
    },
    {
      "CRIM": 0.07,
      "ZN": 0.0,
      "INDUS": 2.18,
      "CHAS": 0,
      "NOX": 0.46,
      "RM": 7.15,
      "AGE": 54.2,
      "DIS": 6.06,
      "RAD": 3,
      "TAX": 222.0,
      "PTRATIO": 18.7,
      "B": 396.9,
      "LSTAT": 5.33,
      "MEDV": 36.2
    }
  ],
  "stats": {
    "CRIM": {
      "count": 506.0,
      "mean": 3.61,
      "std": 8.6,
      "min": 0.01,
      "25%": 0.08,
      "50%": 0.26,
      "75%": 3.68,
      "max": 88.98
    },
    "ZN": {
      "count": 506.0,
      "mean": 11.36,
      "std": 23.32,
      "min": 0.0,
      "25%": 0.0,
      "50%": 0.0,
      "75%": 12.5,
      "max": 100.0
    },
    "INDUS": {
      "count": 506.0,
      "mean": 11.14,
      "std": 6.86,
      "min": 0.46,
      "25%": 5.19,
      "50%": 9.69,
      "75%": 18.1,
      "max": 27.74
    },
    "CHAS": {
      "count": 506.0,
      "mean": 0.07,
      "std": 0.25,
      "min": 0.0,
      "25%": 0.0,
      "50%": 0.0,
      "75%": 0.0,
      "max": 1.0
    },
    "NOX": {
      "count": 506.0,
      "mean": 0.55,
      "std": 0.12,
      "min": 0.38,
      "25%": 0.45,
      "50%": 0.54,
      "75%": 0.62,
      "max": 0.87
    },
    "RM": {
      "count": 506.0,
      "mean": 6.28,
      "std": 0.7,
      "min": 3.56,
      "25%": 5.89,
      "50%": 6.21,
      "75%": 6.62,
      "max": 8.78
    },
    "AGE": {
      "count": 506.0,
      "mean": 68.57,
      "std": 28.15,
      "min": 2.9,
      "25%": 45.02,
      "50%": 77.5,
      "75%": 94.07,
      "max": 100.0
    },
    "DIS": {
      "count": 506.0,
      "mean": 3.8,
      "std": 2.11,
      "min": 1.13,
      "25%": 2.1,
      "50%": 3.21,
      "75%": 5.19,
      "max": 12.13
    },
    "RAD": {
      "count": 506.0,
      "mean": 9.55,
      "std": 8.71,
      "min": 1.0,
      "25%": 4.0,
      "50%": 5.0,
      "75%": 24.0,
      "max": 24.0
    },
    "TAX": {
      "count": 506.0,
      "mean": 408.24,
      "std": 168.54,
      "min": 187.0,
      "25%": 279.0,
      "50%": 330.0,
      "75%": 666.0,
      "max": 711.0
    },
    "PTRATIO": {
      "count": 506.0,
      "mean": 18.46,
      "std": 2.16,
      "min": 12.6,
      "25%": 17.4,
      "50%": 19.05,
      "75%": 20.2,
      "max": 22.0
    },
    "B": {
      "count": 506.0,
      "mean": 356.67,
      "std": 91.29,
      "min": 0.32,
      "25%": 375.38,
      "50%": 391.44,
      "75%": 396.22,
      "max": 396.9
    },
    "LSTAT": {
      "count": 506.0,
      "mean": 12.65,
      "std": 7.14,
      "min": 1.73,
      "25%": 6.95,
      "50%": 11.36,
      "75%": 16.96,
      "max": 37.97
    },
    "MEDV": {
      "count": 506.0,
      "mean": 22.53,
      "std": 9.2,
      "min": 5.0,
      "25%": 17.02,
      "50%": 21.2,
      "75%": 25.0,
      "max": 50.0
    }
  },
  "methods_20": [
    {
      "method": "00 All features (baseline)",
      "features": 13,
      "rmse": 4.73,
      "mae": 3.2,
      "selected": "CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT",
      "rank": 1
    },
    {
      "method": "02 LassoCV",
      "features": 13,
      "rmse": 4.73,
      "mae": 3.2,
      "selected": "CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT",
      "rank": 2
    },
    {
      "method": "06 RFECV",
      "features": 13,
      "rmse": 4.73,
      "mae": 3.2,
      "selected": "CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT",
      "rank": 3
    },
    {
      "method": "07 ElasticNetCV",
      "features": 12,
      "rmse": 4.73,
      "mae": 3.2,
      "selected": "CRIM, ZN, INDUS, CHAS, NOX, RM, DIS, RAD, TAX, PTRATIO, B, LSTAT",
      "rank": 4
    },
    {
      "method": "18 Stability selection (Lasso)",
      "features": 13,
      "rmse": 4.73,
      "mae": 3.2,
      "selected": "CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT",
      "rank": 5
    },
    {
      "method": "08 VIF elimination",
      "features": 12,
      "rmse": 4.79,
      "mae": 3.23,
      "selected": "CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, PTRATIO, B, LSTAT",
      "rank": 6
    },
    {
      "method": "16 Gradient boosting importance",
      "features": 5,
      "rmse": 4.88,
      "mae": 3.25,
      "selected": "PTRATIO, CRIM, DIS, LSTAT, RM",
      "rank": 7
    },
    {
      "method": "01 RFE",
      "features": 5,
      "rmse": 4.91,
      "mae": 3.38,
      "selected": "NOX, RM, DIS, PTRATIO, LSTAT",
      "rank": 8
    },
    {
      "method": "03 Forward SFS",
      "features": 5,
      "rmse": 4.91,
      "mae": 3.38,
      "selected": "NOX, RM, DIS, PTRATIO, LSTAT",
      "rank": 9
    },
    {
      "method": "04 Backward SFS",
      "features": 5,
      "rmse": 4.91,
      "mae": 3.38,
      "selected": "NOX, RM, DIS, PTRATIO, LSTAT",
      "rank": 10
    },
    {
      "method": "14 Ridge coefficient ranking",
      "features": 5,
      "rmse": 4.91,
      "mae": 3.38,
      "selected": "NOX, PTRATIO, DIS, LSTAT, RM",
      "rank": 11
    },
    {
      "method": "11 Stepwise CV",
      "features": 5,
      "rmse": 4.91,
      "mae": 3.38,
      "selected": "NOX, RM, DIS, PTRATIO, LSTAT",
      "rank": 12
    },
    {
      "method": "12 Best subset CV",
      "features": 5,
      "rmse": 4.91,
      "mae": 3.38,
      "selected": "NOX, RM, DIS, PTRATIO, LSTAT",
      "rank": 13
    },
    {
      "method": "05 F-test SelectKBest",
      "features": 5,
      "rmse": 5.04,
      "mae": 3.3,
      "selected": "INDUS, RM, TAX, PTRATIO, LSTAT",
      "rank": 14
    },
    {
      "method": "09 Pearson correlation",
      "features": 5,
      "rmse": 5.04,
      "mae": 3.3,
      "selected": "TAX, INDUS, PTRATIO, RM, LSTAT",
      "rank": 15
    },
    {
      "method": "19 Partial F-test",
      "features": 5,
      "rmse": 5.04,
      "mae": 3.3,
      "selected": "INDUS, RM, TAX, PTRATIO, LSTAT",
      "rank": 16
    },
    {
      "method": "10 Mutual information",
      "features": 5,
      "rmse": 5.05,
      "mae": 3.32,
      "selected": "NOX, PTRATIO, INDUS, RM, LSTAT",
      "rank": 17
    },
    {
      "method": "13 Permutation importance",
      "features": 5,
      "rmse": 5.12,
      "mae": 3.45,
      "selected": "TAX, CRIM, DIS, LSTAT, RM",
      "rank": 18
    },
    {
      "method": "15 Random forest importance",
      "features": 5,
      "rmse": 5.12,
      "mae": 3.45,
      "selected": "TAX, CRIM, DIS, LSTAT, RM",
      "rank": 19
    }
  ],
  "five_methods": {
    "labels": [
      1,
      2,
      3,
      4,
      5
    ],
    "data": {
      "RFE": {
        "rmse": [
          5.79,
          5.59,
          5.21,
          5.18,
          5.02
        ],
        "r2": [
          0.5429,
          0.574,
          0.6303,
          0.6348,
          0.6564
        ],
        "features": [
          [
            "LSTAT"
          ],
          [
            "RM",
            "LSTAT"
          ],
          [
            "RM",
            "PTRATIO",
            "LSTAT"
          ],
          [
            "RM",
            "DIS",
            "PTRATIO",
            "LSTAT"
          ],
          [
            "NOX",
            "RM",
            "DIS",
            "PTRATIO",
            "LSTAT"
          ]
        ]
      },
      "LassoCV": {
        "rmse": [
          5.79,
          5.59,
          5.59,
          5.69,
          5.25
        ],
        "r2": [
          0.5429,
          0.574,
          0.5736,
          0.5581,
          0.6244
        ],
        "features": [
          [
            "LSTAT"
          ],
          [
            "RM",
            "LSTAT"
          ],
          [
            "DIS",
            "RM",
            "LSTAT"
          ],
          [
            "RAD",
            "DIS",
            "RM",
            "LSTAT"
          ],
          [
            "PTRATIO",
            "RAD",
            "DIS",
            "RM",
            "LSTAT"
          ]
        ]
      },
      "F-test": {
        "rmse": [
          5.79,
          5.59,
          5.21,
          5.22,
          5.28
        ],
        "r2": [
          0.5429,
          0.574,
          0.6303,
          0.6284,
          0.6197
        ],
        "features": [
          [
            "LSTAT"
          ],
          [
            "RM",
            "LSTAT"
          ],
          [
            "RM",
            "PTRATIO",
            "LSTAT"
          ],
          [
            "INDUS",
            "RM",
            "PTRATIO",
            "LSTAT"
          ],
          [
            "INDUS",
            "RM",
            "TAX",
            "PTRATIO",
            "LSTAT"
          ]
        ]
      },
      "RF Importance": {
        "rmse": [
          6.79,
          5.59,
          5.59,
          5.71,
          5.29
        ],
        "r2": [
          0.3708,
          0.574,
          0.5736,
          0.5547,
          0.6191
        ],
        "features": [
          [
            "RM"
          ],
          [
            "LSTAT",
            "RM"
          ],
          [
            "DIS",
            "LSTAT",
            "RM"
          ],
          [
            "CRIM",
            "DIS",
            "LSTAT",
            "RM"
          ],
          [
            "PTRATIO",
            "CRIM",
            "DIS",
            "LSTAT",
            "RM"
          ]
        ]
      },
      "Forward SFS": {
        "rmse": [
          5.79,
          5.59,
          5.21,
          5.37,
          5.37
        ],
        "r2": [
          0.5429,
          0.574,
          0.6303,
          0.6071,
          0.6061
        ],
        "features": [
          [
            "LSTAT"
          ],
          [
            "RM",
            "LSTAT"
          ],
          [
            "RM",
            "PTRATIO",
            "LSTAT"
          ],
          [
            "RM",
            "PTRATIO",
            "B",
            "LSTAT"
          ],
          [
            "RM",
            "DIS",
            "PTRATIO",
            "B",
            "LSTAT"
          ]
        ]
      }
    }
  },
  "final_metrics": {
    "method": "00 All features (baseline)",
    "features": "CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT",
    "test_rmse": 4.93,
    "test_mae": 3.19,
    "test_r2": 0.6688,
    "baseline_rmse": 8.66,
    "improvement": 43.1
  },
  "predictions": [
    {
      "actual": 23.6,
      "pred": 29.0
    },
    {
      "actual": 32.4,
      "pred": 36.03
    },
    {
      "actual": 13.6,
      "pred": 14.82
    },
    {
      "actual": 22.8,
      "pred": 25.03
    },
    {
      "actual": 16.1,
      "pred": 18.77
    },
    {
      "actual": 20.0,
      "pred": 23.25
    },
    {
      "actual": 17.8,
      "pred": 17.66
    },
    {
      "actual": 14.0,
      "pred": 14.34
    },
    {
      "actual": 19.6,
      "pred": 23.01
    },
    {
      "actual": 16.8,
      "pred": 20.63
    }
  ],
  "formula": {
    "intercept": 30.25,
    "coefs": {
      "CRIM": -0.1131,
      "ZN": 0.0301,
      "INDUS": 0.0404,
      "CHAS": 2.7844,
      "NOX": -17.2026,
      "RM": 4.4388,
      "AGE": -0.0063,
      "DIS": -1.4479,
      "RAD": 0.2624,
      "TAX": -0.0106,
      "PTRATIO": -0.9155,
      "B": 0.0124,
      "LSTAT": -0.5086
    }
  }
};

if (typeof window !== "undefined") {
  window.BOSTON_RAW = BOSTON_RAW;
}
