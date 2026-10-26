# 🏡 AIOT-DIC2 — CRISP-DM Machine Learning Platform
### Multiple Linear Regression · King County Housing · 50 Startups · Boston Housing

[![GitHub Pages](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-0cb8b6?style=for-the-badge&logo=github)](https://oscaryoyo.github.io/AIOT-DIC2-MultipleLR/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![statsmodels](https://img.shields.io/badge/statsmodels-0.15-4C72B0?style=for-the-badge&logo=python&logoColor=white)](https://www.statsmodels.org/)
[![Chart.js](https://img.shields.io/badge/Chart.js-4.4-FF6384?style=for-the-badge&logo=chart.js&logoColor=white)](https://chartjs.org)

---

## 📖 Project Overview (專案總覽)

本專案是一個遵循 **CRISP-DM (Cross-Industry Standard Process for Data Mining)** 國際標準資料探勘方法論的機器學習平台與互動式網頁應用。本系統整合了計量經濟統計推論（OLS Statistical Inference）與機器學習預測（Out-of-sample Generalization），並支援三大經典迴歸問題：

1. **🏡 King County House Price Prediction (新擴展核心專案)**:
   * **資料集**: 21,613 筆美國華盛頓州金郡（西雅圖地區）真實房屋成交數據，包含 17 項候選特徵與 16 項基準特徵（排除物理完全共線性特徵 `sqft_basement`）。
   * **功能**: 完整實作 CRISP-DM 六大階段、三種迴歸模式（Model A 單變數、Model B 16 特徵基準、Model C 6 種自動特徵選擇）、五大高斯－馬可夫迴歸統計假設檢定、以及即時房價預測模擬器。
   * **成果**: 基準模型達成 **Test RMSE $214,472.76** 與 **R² 0.6957 (Adj R² 0.6946)**，相較於均值預測基準改善 **44.8%**。
2. **🚀 50 Startups Profit Prediction**:
   * 分析 R&D、行銷、行政支出對新創企業獲利之影響，比較 20 種特徵選擇演算法，達成 Test RMSE $8,243 與 R² 0.916。
3. **🏛️ Boston Housing Value Prediction**:
   * 13 項特徵預測波士頓郊區房屋價值中位數 (MEDV)。
4. **📁 Custom Dataset / CSV Upload**:
   * 支援使用者上傳本機 CSV 或載入公開 URL 進行瀏覽器端動態多元線性迴歸訓練。

---

## 🌐 Live Demo & Web Architecture

👉 **線上展示網址**: [https://oscaryoyo.github.io/AIOT-DIC2-MultipleLR/](https://oscaryoyo.github.io/AIOT-DIC2-MultipleLR/)

### 部署架構說明 (Deployment Architecture & GitHub Pages Compatibility)
* **純前端靜態相容性**: GitHub Pages 本質為靜態網站代管服務，無法直接在伺服器端執行持久化 Python 程式。
* **本專案的高效架構**:
  1. 複雜的計算（如 21,613 筆大數據矩陣運算、5 折交叉驗證、Ramsey RESET 檢定、Breusch-Pagan 檢定與圖表產出）在離線端透過 Python (`kc_house_analysis.py`) 完成。
  2. 計算結果、估計係數向量與指標預先序列化為精簡的 JavaScript 資料產物 (`kc_house_data.js`)。
  3. 前端儀表板 (`index.html`) 與即時預測器以純 JavaScript 在使用者瀏覽器端本機毫秒級計算，**完美相容 GitHub Pages 免費靜態託管**，使用者無需建置後端伺服器即可獲得流暢體驗。

---

## 📦 Dataset Information (資料集來源與取得)

### King County House Sales Dataset
* **Kaggle 來源**: [https://www.kaggle.com/datasets/harlfoxem/housesalesprediction](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)
* **檔案名稱**: `kc_house_data.csv`
* **資料規模**: 21,613 筆觀測值，21 個原始欄位。
* **候選特徵**: 17 個（已排除 `id`, `date`, `zipcode`, `price`）。
* **完全共線性處理**: 
  由於存在結構恆等式 $\text{sqft\_living} = \text{sqft\_above} + \text{sqft\_basement}$，同時納入會導致設計矩陣秩不足與 VIF $\to \infty$。主多元線性迴歸模型移除 `sqft_basement`，保留 16 個無多重共線性特徵。
* **資料集手動下載方式**: 若本地無 `kc_house_data.csv`，請至上述 Kaggle 頁面下載並解壓縮置於專案根目錄，亦可透過網頁右上角「📁 Upload CSV」功能直接於瀏覽器載入。

---

## 🛠️ Installation & Environment Setup (環境安裝)

本專案支援 Python 3.10 及以上版本。建議使用虛擬環境：

```bash
# 1. 複製專案儲存庫
git clone https://github.com/oscaryoyo/AIOT-DIC2-MultipleLR.git
cd AIOT-DIC2-MultipleLR

# 2. 建立並啟動虛擬環境 (選用)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. 安裝必要相依套件
pip install -r requirements.txt
```

`requirements.txt` 內容清單：
```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
seaborn>=0.12.0
matplotlib>=3.7.0
joblib>=1.3.0
```

---

## 🚀 Execution & Reproduction (執行與驗證)

### 1. 執行金郡房價完整 CRISP-DM 機器學習流水線
```bash
python kc_house_analysis.py
```
執行後將自動完成：
* 載入並校驗 `kc_house_data.csv` (21,613 筆，17 候選特徵)。
* 執行 80/20 訓練/測試集切分與資料標準化。
* 擬合 Model A (單變數)、Model B (16 特徵基準)、Model C (6 種特徵篩選法與 5 折交叉驗證)。
* 執行 5 大迴歸假設檢定 (RESET, Durbin-Watson, Jarque-Bera, Breusch-Pagan, VIF)。
* 於 `kc_house_outputs/` 匯出 8 幅高解析度圖表、序列化模型 `kc_house_model.joblib` 與前台資料檔 `kc_house_data.js`。

### 2. 執行自動化單元測試 (Unit Tests)
```bash
python -m unittest test_kc_house.py
```
測試項目涵蓋：
* 資料集筆數與欄位完整性測試
* 完全共線性驗證測試
* 特徵與標的隔離測試
* 80/20 訓練測試切分一致性測試
* 三種迴歸模式執行測試
* 評估指標數學正確性測試
* 產物存在性測試

### 3. 本機啟動網頁儀表板
```bash
python -m http.server 8000
```
開啟瀏覽器訪問 `http://localhost:8000/index.html` 即可體驗完整互動功能。

---

## 📊 Summary of King County Results (金郡模型結果摘要)

### 1. 16 特徵基準多元線性迴歸 (Model B)
$$\begin{aligned}
\widehat{\text{Price}} = & -34,612,911.36 - 32,808.99(\text{bedrooms}) + 45,532.72(\text{bathrooms}) \\
& + 144.96(\text{sqft\_living}) + 0.0847(\text{sqft\_lot}) + 1,478.80(\text{floors}) \\
& + 568,137.38(\text{waterfront}) + 50,499.18(\text{view}) + 28,786.28(\text{condition}) \\
& + 95,702.36(\text{grade}) + 32.60(\text{sqft\_above}) - 2,531.97(\text{yr\_built}) \\
& + 22.07(\text{yr\_renovated}) + 556,748.03(\text{lat}) - 101,603.03(\text{long}) \\
& + 26.68(\text{sqft\_living15}) - 0.3306(\text{sqft\_lot15})
\end{aligned}$$

* **5-Fold CV RMSE**: $200,101.03
* **Test RMSE**: $214,472.76
* **Test MAE**: $128,157.32
* **Test R²**: 0.6957
* **Test Adjusted R²**: 0.6946
* **特徵影響力前三名**: `sqft_living` (生活面積), `grade` (建築等級, +$95.7k/級), `lat` (緯度區位, 北向近市區溢價顯著)。

### 2. 五大迴歸統計假設檢定摘要

| 統計假設 | 檢定工具 | 統計量與 p 值 | 結論與說明 |
|:---|:---|:---|:---|
| **1. 線性關係 (Linearity)** | Ramsey RESET Test | $F = 10,904.27, p < 0.001$ | 拒絕線性虛無假設；高價豪宅存在邊際遞增階梯。 |
| **2. 殘差獨立性 (Independence)** | Durbin-Watson | $d = 1.9960$ | 樣本排列無自我相關；但截面不動產存在客觀空間相依性。 |
| **3. 殘差常態性 (Normality)** | Jarque-Bera & Q-Q Plot | $\text{JB} = 1,716,376, p < 0.001$ | 拒絕常態分佈；龐大樣本 ($N=17,290$) 下 CLT 保證估計量漸近無偏。 |
| **4. 變異數同質性 (Homoscedasticity)** | Breusch-Pagan Test | $\text{LM} = 2,164.44, p < 0.001$ | 存在異質變異；建議以穩健標準誤進行顯著性推論。 |
| **5. 無多重共線性 (Multicollinearity)** | VIF (方差膨脹係數) | Max VIF = 8.64 (`sqft_living`) | 剔除 `sqft_basement` 後所有特徵 VIF $< 10$，共線性完全解除。 |

---

## 🗂️ Project File Structure (檔案目錄結構)

```
AIOT-DIC2-MultipleLR/
│
├── index.html                   # ✨ CRISP-DM 互動式分析儀表板與房價即時預測器
├── index.css                    # 🎨 響應式現代化設計系統樣式表
├── README.md                    # 📖 專案總說明文件
├── requirements.txt             # 🐍 Python 套件相依性清單
├── REPORT_KING_COUNTY_HOUSING.md# 🎓 大學期末專案完整學術分析報告 (Markdown 格式)
│
├── kc_house_data.csv            # 📁 金郡原始資料集 (21,613 筆，21 欄)
├── kc_house_analysis.py         # 🐍 金郡房價 CRISP-DM 完整機器學習分析腳本
├── kc_house_data.js             # 📦 金郡模型參數與預算數據 JavaScript 產物
├── test_kc_house.py             # 🧪 7 項自動化單元測試腳本
│
├── kc_house_outputs/            # 📂 金郡專案產物目錄
│   ├── kc_house_data.json       # 數據與統計指標 JSON
│   ├── kc_house_model.joblib    # 訓練完成之 scikit-learn 模型
│   ├── kc_price_distribution.png# 房價分佈圖
│   ├── kc_correlation_heatmap.png# 特徵相關係數熱圖
│   ├── kc_scatter_features_vs_price.png# 特徵與房價散佈圖
│   ├── kc_residuals_vs_fitted.png# 殘差與擬合值散佈圖
│   ├── kc_qq_plot.png           # 常態 Q-Q 圖
│   ├── kc_multicollinearity_vif.png# VIF 共線性對比圖
│   ├── kc_actual_vs_predicted.png# 實際值 vs 預測值圖
│   └── kc_model_comparison.png  # 模型特徵篩選表現比較圖
│
├── main.py                      # 50 Startups 原始分析腳本
├── analysis_5methods.py         # 50 Startups 5-method 掃描腳本
├── 50_Startups.csv              # 50 Startups 資料集
└── crisp_dm_outputs/            # 50 Startups 歷史產物目錄
```

---

## 🚢 Git & Deployment Instructions (版本控制與發布步驟)

若欲將新增之金郡房價功能更新至您的 GitHub 儲存庫與 GitHub Pages，請於本機執行下列 Git 指令：

```bash
# 1. 檢視工作目錄變更狀態
git status

# 2. 暫存所有新增與修改之檔案
git add index.html index.css README.md requirements.txt REPORT_KING_COUNTY_HOUSING.md kc_house_analysis.py test_kc_house.py kc_house_data.js kc_house_outputs/

# 3. 提交變更並撰寫具體說明
git commit -m "feat: integrate complete King County housing CRISP-DM regression project"

# 4. 推送至遠端 GitHub 儲存庫 main 分支
git push origin main
```

推送完成後，GitHub Pages 將自動偵測 `main` 分支根目錄的 `index.html` 進行靜態部署（約需 1~2 分鐘更新），即可在 [https://oscaryoyo.github.io/AIOT-DIC2-MultipleLR/](https://oscaryoyo.github.io/AIOT-DIC2-MultipleLR/) 瀏覽最新整合成果！
