# King County House Price Prediction: A CRISP-DM-Based Multiple Linear Regression Analysis
### 華盛頓州金郡（西雅圖地區）房價多元線性迴歸分析與 CRISP-DM 機器學習專案報告

---

## 1. Introduction (導言)

在現代計量經濟學與房地產大數據分析中，房屋價格預測是極具代表性的回歸問題。不動產價值受建築實體格局（如生活坪數、臥室、衛浴、樓層）、施工結構品質（等級與屋況）、歷史年代（屋齡與翻修）以及不可忽視的地理空間區位（經緯度、景觀與第一線水岸資源）等多重維度變數共同決定。

本研究依循國際標準資料探勘方法論 **CRISP-DM (Cross-Industry Standard Process for Data Mining)** 的六大階段，採用 Python 數據科學工具鏈（`pandas`, `NumPy`, `scikit-learn`, `statsmodels`, `Matplotlib`, `Seaborn`），針對美國華盛頓州金郡（King County，涵蓋西雅圖、貝爾維尤 Bellevue、麥地那 Medina 等主要城市）2014 年 5 月至 2015 年 5 月間共 **21,613 筆實際房屋成交紀錄** 進行完整的機器學習多元線性迴歸建模。

本專案旨在實現：
1. **單變數線性迴歸 (Simple Linear Regression)**、**16 特徵基準多元線性迴歸 (Multiple Linear Regression)** 與 **6 種自動化特徵選擇 (Automatic Feature Selection)** 模式之橫向對比。
2. 揭示並解決實體建築變數間的 **完全共線性問題 (Exact Multicollinearity)**。
3. 嚴格遵循機器學習驗證規範（以 80/20 訓練/測試集切分與 5 折交叉驗證防止資料外洩 Data Leakage）。
4. 進行古典高斯－馬可夫定理 (Gauss-Markov Theorem) 下的 **五大迴歸統計假設檢定 (Five Regression Assumptions)**。
5. 建立整合於現有系統的互動式即時房價模擬預測器 (Interactive Predictor) 與部署架構。

---

## 2. Dataset Source and Description (資料集來源與結構描述)

### 2.1 資料集來源
* **資料集名稱**: King County House Sales (House Sales in King County, USA)
* **資料來源**: Kaggle 競賽公開資料集
* **URL**: [https://www.kaggle.com/datasets/harlfoxem/housesalesprediction](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)
* **本地資料集檔案**: `kc_house_data.csv`
* **原始觀測值筆數 (Observations)**: 21,613 筆
* **欄位總數 (Columns)**: 21 個欄位

### 2.2 變數篩選與定義
為符合大學機器學習專案規範（自變數須超過 13 項，排除預測標的），初始建模排除下列 4 個欄位：
1. `id`: 房屋唯一識別碼（無預測意義之離散序號）。
2. `date`: 交易日期（字串格式時間戳記，避免直接作為連續數值）。
3. `zipcode`: 郵遞區號（分類型地區代碼 Categorical Variable，避免將無序之郵遞編號視為具順序尺度之數值）。
4. `price`: 本專案之連續型預測標的變數 (Target Variable, $y$)。

排除後保留 **17 個連續型與離散數值型候選特徵 (Candidate Features)**：

| 欄位名稱 | 變數型態 | 經濟意涵說明 | 單位 / 範圍 |
|:---|:---|:---|:---|
| `bedrooms` | 整數 | 臥室房間數 | 0 ~ 33 間 |
| `bathrooms` | 浮點數 | 衛浴總數（含 0.25/0.5/0.75 衛） | 0.0 ~ 8.0 間 |
| `sqft_living` | 整數 | 室內居住坪數 (Living Area) | 290 ~ 13,540 sqft |
| `sqft_lot` | 整數 | 基地佔地總坪數 (Lot Area) | 520 ~ 1,651,359 sqft |
| `floors` | 浮點數 | 建築總樓層數 | 1.0 ~ 3.5 樓 |
| `waterfront` | 二元虛擬變數 | 是否具備第一線水岸景觀 (0: 無, 1: 有) | 0 或 1 |
| `view` | 整數 | 景觀視野優良程度評級 | 0 ~ 4 級 |
| `condition` | 整數 | 屋況維護品質評級 | 1 ~ 5 級 |
| `grade` | 整數 | 金郡政府建築設計與建材等級評級 | 1 ~ 13 級 |
| `sqft_above` | 浮點數 | 地面上層室內面積 (Above Ground Area) | 290 ~ 9,410 sqft |
| `sqft_basement` | 整數 | 地下室室內面積 (Basement Area) | 0 ~ 4,820 sqft |
| `yr_built` | 整數 | 原始建造年份 | 1900 ~ 2015 年 |
| `yr_renovated` | 整數 | 最近一次翻修年份 (0 代表未曾翻修) | 0 或 1934 ~ 2015 年 |
| `lat` | 浮點數 | 房屋所在緯度 (Latitude) | 47.1559°N ~ 47.7776°N |
| `long` | 浮點數 | 房屋所在經度 (Longitude) | -122.5190°W ~ -121.3150°W |
| `sqft_living15` | 整數 | 鄰近 15 間房屋平均室內面積 | 399 ~ 6,210 sqft |
| `sqft_lot15` | 整數 | 鄰近 15 間房屋平均佔地面積 | 651 ~ 871,200 sqft |

### 2.3 完全共線性 (Exact Multicollinearity) 結構分析
在金郡資料集的原始定義中存在嚴格的物理結構約束：
$$\text{sqft\_living} = \text{sqft\_above} + \text{sqft\_basement}$$
經全體資料集逐筆驗證：
$$\sum_{i=1}^{21,613} |\text{sqft\_living}_i - (\text{sqft\_above}_i + \text{sqft\_basement}_i)| = 0$$
此一物理恆等式導致資料矩陣 $\mathbf{X}$ 出現嚴重的 **秩不足 (Rank-Deficiency)**。若同時將此 3 個欄位納入普通最小平方法 (OLS) 設計矩陣，則 $(\mathbf{X}^T\mathbf{X})$ 矩陣行列式值為 0（非可逆），方差膨脹係數 $\text{VIF} \to \infty$。

**解決策略：**
保留全部 17 個特徵作為探索性資料分析 (EDA) 與共線性診斷教學展示；而在主多元線性迴歸基準模型 (Model B) 與後續特徵選擇中，依計量經濟慣例剔除重複定義的次要變數 `sqft_basement`，保留 **16 個基準輸入特徵**，使矩陣恢復滿秩 (Full Rank)。

---

## 3. CRISP-DM Stage 1: Business Understanding (商業理解)

### 3.1 商業問題背景
房地產交易具有高金額、低交易頻率以及顯著資訊不對稱的特性。買賣雙方、估價機構、金融放貸銀行與房產經紀商皆需具備透明、客觀且具備可解釋性的估價工具。

### 3.2 專案商業目標
利用公開房屋成交特徵，建構多元線性迴歸模型以預測華盛頓州金郡住宅物業之成交售價 (Price in USD)。同時辨識各項建築特徵對售價的「邊際貢獻金額（Marginal Dollar Value）」，輔助買方合理出價、賣方設定掛牌價以及銀行進行抵押貸款風險評估。

### 3.3 迴歸問題定義與評估準則
* **預測標的 ($y$)**: 連續型房屋銷售價格 `price`（美元 USD）。
* **核心演算法**: 最小平方法多元線性迴歸 (Ordinary Least Squares Multiple Linear Regression)。
* **模型評估標準**:
  1. **Test RMSE (均方根誤差)**: 衡量未見測試集上的絕對預測金額誤差，目標相較於單純均值預測基準降低 40% 以上。
  2. **Test R² (判定係數)** 與 **Adjusted R² (調整後判定係數)**: 衡量自變數對房價總變異的解釋比例，並懲罰冗餘變數。
  3. **高斯－馬可夫假定驗證**: 確保參數估計量具備最佳線性不偏性 (BLUE)。

### 3.4 採用多元線性迴歸之學理依據
相較於複雜之黑箱模型（如神經網路或梯度提升樹），多元線性迴歸在房產估價具備不可替代的優勢：
1. **高度經濟可解釋性**: 迴歸係數 $\beta_j$ 直接代表在控制其他特徵不變時，該特徵每增加一個單位對房價的預期變動量（例如：每提升 1 級建材品質對應的溢價金額）。
2. **符合傳統特徵價格理論 (Hedonic Pricing Model)**: 將複合資產之總價格拆解為個別物理屬性與區位屬性之影子價格 (Shadow Prices)。
3. **推論統計檢定完整**: 提供標準誤 (SE)、$t$ 統計量、顯著性 $p$ 值與信賴區間，便於學術驗證與實務法規遵循。

---

## 4. CRISP-DM Stage 2: Data Understanding (資料理解)

### 4.1 描述性統計量 (Descriptive Statistics)
本專案對 21,613 筆實際樣本進行探索性統計分析：

| 變數 | 平均值 (Mean) | 標準差 (Std) | 最小值 (Min) | 中位數 (Median) | 最大值 (Max) | 偏態特性 |
|:---|:---|:---|:---|:---|:---|:---|
| **price ($)** | 540,088.14 | 367,127.20 | 75,000.00 | 450,000.00 | 7,700,000.00 | 極顯著右偏 (Right-skewed) |
| **bedrooms** | 3.37 | 0.93 | 0 | 3.00 | 33 | 存在單筆極端離群值 (33房) |
| **bathrooms** | 2.11 | 0.77 | 0.00 | 2.25 | 8.00 | 集中於 1.75 ~ 2.5 衛 |
| **sqft_living** | 2,079.90 | 918.44 | 290 | 1,910.00 | 13,540 | 中位數 1,910 坪 |
| **sqft_lot** | 15,106.97 | 41,420.51 | 520 | 7,618.00 | 1,651,359 | 郊區地坪極大，呈現厚尾分佈 |
| **floors** | 1.49 | 0.54 | 1.00 | 1.50 | 3.50 | 以 1 樓與 2 樓為主 |
| **grade** | 7.66 | 1.18 | 1 | 7.00 | 13 | 集中於 7 ~ 8 級 |
| **yr_built** | 1971.01 | 29.37 | 1900 | 1975.00 | 2015 | 橫跨百年建築歷史 |
| **lat** | 47.5601 | 0.1386 | 47.1559 | 47.5718 | 47.7776 | 緯度越高（北向近西雅圖市中心）房價越高 |
| **long** | -122.2139 | 0.1407 | -122.5190 | -122.2300 | -121.3150 | 經度集中於普吉特海灣東側 |

### 4.2 缺失值與重複值檢驗
* **重複樣本分析**: 透過 `df.duplicated().sum()` 檢核，重複資料為 **0 筆**。
* **缺失值檢核**: 
  原始資料中 `sqft_above` 存在 2 筆缺失紀錄。利用物理恆等式 $\text{sqft\_above} = \text{sqft\_living} - \text{sqft\_basement}$ 進行確定性插補 (Deterministic Imputation)，避免隨意刪減真實樣本，插補後缺失值總數為 **0**。

### 4.3 探索性資料視覺化 (EDA)
本專案產出並整合了 8 幅高解析度圖表於輸出目錄 `kc_house_outputs/`：
1. **房價分佈圖 (`kc_price_distribution.png`)**:
   直方圖與箱形圖顯示金郡房價呈現顯著右偏長尾分佈（平均數 \$540k，中位數 \$450k，最高值達 \$7.7M）。
2. **皮爾森相關係數熱圖 (`kc_correlation_heatmap.png`)**:
   與房價相關度最高的前五項特徵分別為：`sqft_living` ($r = 0.70$)、`grade` ($r = 0.67$)、`sqft_above` ($r = 0.61$)、`sqft_living15` ($r = 0.59$) 與 `bathrooms` ($r = 0.53$)。
3. **重要特徵與房價散佈圖 (`kc_scatter_features_vs_price.png`)**:
   `sqft_living`、`grade`、`bathrooms` 與 `lat` 皆與價格呈現顯著正向趨勢，且高階建築等級（Grade $\ge 11$）呈現非線性陡增之豪宅定價階梯。

---

## 5. CRISP-DM Stage 3: Data Preparation (資料準備)

### 5.1 訓練集與測試集切分 (Train/Test Split)
依據規範，採用 80% / 20% 之孤立隔離切分：
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
* **訓練集樣本數 (Train Set, 80%)**: 17,290 筆
* **測試集樣本數 (Test Set, 20%)**: 4,323 筆

### 5.2 防範資料外洩 (Data Leakage Prevention)
為防範計量經濟與機器學習中常見之資訊滲漏：
1. 特徵標準化（`StandardScaler`）之平均值與標準差 **僅自訓練集 `X_train` 計算得出**（`fit`），再變換至測試集（`transform`）。
2. 所有特徵篩選演算法（Pearson、SelectKBest、RFE、RFECV、Backward Elimination）皆僅在訓練集上執行 5 折交叉驗證，絕不以未見測試集之指標反向調校特徵集合。

### 5.3 離群值處理準則
資料集中包含 1 筆 `bedrooms = 33`（生活面積僅 1,620 sqft，售價 \$640,000）之特殊輸入紀錄，以及多筆超過 \$5,000,000 之頂級水岸豪宅。
本專案堅持 **「不隨意以自動化裁切法（如刪除 3 倍標準差外資料）粗暴抹除極端值」**。原因在於高價住宅是西雅圖地區真實存在的經濟實體，隨意剔除會低估高價市場的方差與真實殘差分佈，此點在後續迴歸假設診斷中將進一步進行嚴謹討論。

---

## 6. CRISP-DM Stage 4: Modeling (模型建置)

本專案實作三種層次的線性迴歸模式：

### 6.1 Model A: 單變數線性迴歸 (Simple Linear Regression)
* **演算法**: `sklearn.linear_model.LinearRegression`
* **模式特色**: 允許使用者自 16 項數值特徵中自由挑選單一自變數，建立一元線性方程 $y = \beta_0 + \beta_1 x$。
* **最佳單變數表現**: 選擇 `sqft_living`（室內居住坪數）
  $$\widehat{\text{Price}} = -\$41,999.19 + 279.55 \times (\text{sqft\_living})$$
  * **單變數解釋力**: $R^2 = 0.4941$
  * **未見測試集 RMSE**: \$276,559.17
  * **未見測試集 MAE**: \$177,867.54
  * **經濟意涵**: 金郡房屋在不考慮其他變數下，每增加 1 平方英尺室內面積，預期售價增加約 \$279.55 美元。

### 6.2 Model B: 16 特徵基準多元線性迴歸 (Multiple Linear Regression)
* **演算法**: `sklearn.linear_model.LinearRegression`（預測）與 `statsmodels.OLS`（推論）
* **自變數**: 排除造成共線性的 `sqft_basement` 後之 16 個特徵。
* **擬合方程式**:
  $$\begin{aligned}
  \widehat{\text{Price}} = & -34,612,911.36 - 32,808.99(\text{bedrooms}) + 45,532.72(\text{bathrooms}) \\
  & + 144.96(\text{sqft\_living}) + 0.0847(\text{sqft\_lot}) + 1,478.80(\text{floors}) \\
  & + 568,137.38(\text{waterfront}) + 50,499.18(\text{view}) + 28,786.28(\text{condition}) \\
  & + 95,702.36(\text{grade}) + 32.60(\text{sqft\_above}) - 2,531.97(\text{yr\_built}) \\
  & + 22.07(\text{yr\_renovated}) + 556,748.03(\text{lat}) - 101,603.03(\text{long}) \\
  & + 26.68(\text{sqft\_living15}) - 0.3306(\text{sqft\_lot15})
  \end{aligned}$$

#### 標準化迴歸係數與特徵重要性排序 (|Standardized Beta|)
為消除各特徵量綱（單位）差異，以 Z 分數標準化計算各特徵之相對重要性貢獻：
1. **`sqft_living`** ($\text{Std }\beta = +131,517.18$): 面積規模仍為最根本之價格基石。
2. **`grade`** ($\text{Std }\beta = +112,002.48$): 建築等級每提升 1 級，在其他條件相同下邊際溢價達 \$95,702 美元。
3. **`lat` (緯度)** ($\text{Std }\beta = +77,069.50$): 體現西雅圖市區與北向優質學區之強大地理溢價。
4. **`yr_built`** ($\text{Std }\beta = -74,528.02$): 屋齡折舊效應，每多 1 年建造歷史，殘值折損約 \$2,532 美元。
5. **`waterfront`** ($\text{Std }\beta = +47,940.63$): 第一線水岸稀缺景觀溢價高達 \$568,137 美元。
6. **`bedrooms`** ($\text{Std }\beta = -30,556.93$): 在控制總坪數 `sqft_living` 不變之前提下，臥室數量過多意味著單一房間過於狹小，因而呈現負向係數（-\$32,809 / 房）。

### 6.3 Model C: 自動化特徵選擇 (Automatic Feature Selection)
本專案實作 6 種代表性特徵篩選法，皆於訓練集上執行 5 折交叉驗證（5-Fold CV）：
1. **皮爾森相關係數法 (Pearson Correlation, Top 8)**
2. **單變量統計檢定法 (SelectKBest with f_regression, Top 8)**
3. **遞迴特徵消除法 (RFE with Linear Regression, k=8)**
4. **具交叉驗證之遞迴消除法 (RFECV, 5-Fold CV)**
5. **後向逐步淘汰法 (Backward Elimination via statsmodels, p < 0.05)**
6. **共線性方差膨脹係數修剪法 (VIF Multicollinearity Pruning, VIF < 5.0)**

---

## 7. Feature Selection Results & Evaluation (特徵選擇與模型評估)

### 7.1 特徵選擇各方法詳細比較

| 排序 | 模型名稱 | 特徵數量 $k$ | 剔除之特徵 | 5-Fold CV RMSE ($) | Test RMSE ($) | Test MAE ($) | Test R² | Test Adj R² |
|:---:|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Model B (16 特徵基準)** | 16 | (無) | **200,101.03** | **214,472.76** | **128,157.32** | **0.6957** | **0.6946** |
| 🥈 | **Model C: RFECV** | 16 | (無) | **200,101.03** | **214,472.76** | **128,157.32** | **0.6957** | **0.6946** |
| 🥉 | **Model C: Backward Elimination** | 14 | `sqft_lot`, `floors` | 200,214.39 | 214,558.87 | 128,292.83 | 0.6955 | 0.6945 |
| 4 | **Model C: VIF Pruning** | 15 | `sqft_living` | 205,482.11 | 220,177.11 | 131,980.86 | 0.6793 | 0.6782 |
| 5 | **Model C: Pearson Correlation** | 8 | 8 項次要特徵 | 217,250.21 | 233,164.12 | 139,613.79 | 0.6404 | 0.6397 |
| 6 | **Model C: SelectKBest (F-test)** | 8 | 8 項次要特徵 | 217,250.21 | 233,164.12 | 139,613.79 | 0.6404 | 0.6397 |
| 7 | **Model C: RFE (k=8)** | 8 | 8 項次要特徵 | 224,118.52 | 240,798.24 | 143,598.58 | 0.6165 | 0.6158 |
| 8 | **Model A: Simple LR (`sqft_living`)** | 1 | 15 項特徵 | 258,924.30 | 276,559.17 | 177,867.54 | 0.4941 | 0.4940 |

### 7.2 核心指標學理解析
* **Mean Absolute Error (MAE)**: 絕對誤差平均值。基準 Model B 之 MAE 為 \$128,157，代表模型預測房價平均偏離真實售價約 12.8 萬美元。
* **Root Mean Squared Error (RMSE)**: 均方根誤差。賦予極端離群誤差較重懲罰。Model B 之 Test RMSE 為 \$214,472.76。
* **Mean Prediction Baseline RMSE**: 單純以訓練集均值（\$540k）作為常數預測時，測試集 RMSE 高達 \$388,819.34。Model B 將誤差降低了 **44.8%**。
* **R-squared ($R^2$)**: 判定係數，反映模型解釋總變異之比例。Model B 之 $R^2 = 0.6957$（即解釋了約 70% 的金郡房價變異）。
* **Adjusted R-squared ($\text{Adj } R^2$)**:
  $$\text{Adj } R^2 = 1 - \left[ \frac{(1 - R^2)(n - 1)}{n - p - 1} \right]$$
  其中 $n = 4,323$ 為測試集樣本數，$p$ 為預測自變數個數。Model B 之 $\text{Adj } R^2 = 0.6946$，證明納入 16 項特徵具備高度實質訊息量，並未造成過擬合。
* **特徵精簡度權衡 (Parsimony vs Performance)**:
  後向淘汰法（Backward Elimination）以 $p < 0.05$ 為門檻剔除了統計上不顯著的 `sqft_lot` ($p=0.297$) 與 `floors` ($p=0.463$)，僅使用 **14 個特徵** 即可達到 RMSE \$214,558.87（與 16 特徵基準僅微差 \$86 美元），達成極佳之模型簡約性 (Parsimony)。

---

## 8. Five Major Regression Assumption Tests (五大迴歸假設嚴謹檢定)

在計量經濟學中，普通最小平方法 (OLS) 之估計量唯有在高斯－馬可夫假定成立時，方具備最佳線性不偏估計量 (BLUE) 性質。本專案對 16 特徵 Model B 之殘差進行完整診斷：

| 假設檢定項目 (Assumption) | 檢定方法 (Method) | 檢定統計量 (Statistic) | $p$ 值 ($p$-value) | 診斷結論 (Status) | 學術實務意涵與矯正措施 |
|:---|:---|:---:|:---:|:---:|:---|
| **1. 線性關係 (Linearity)** | Ramsey RESET Test | $F = 10,904.27$ | $< 0.0001$ | 拒絕虛無假設 (存在非線性) | 現實高價住宅存在跳躍階梯。可引進多項式項、對數變換 $\ln(\text{price})$ 或分段樣條迴歸 (Splines)。 |
| **2. 殘差獨立性 (Independence of Errors)** | Durbin-Watson (DW) | $d = 1.9960$ | — | 樣本順序無自我相關 (DW $\approx 2.0$) | **重要警語**：DW 僅能檢驗資料排列順序之相關性。房地產為截面資料，本質上存在**空間相依性 (Spatial Autocorrelation)**，需透過經緯度與鄰里變數控制。 |
| **3. 殘差常態性 (Normality of Residuals)** | Jarque-Bera (JB) 檢定與 Q-Q Plot | $\text{JB} = 1,716,376$ | $< 0.0001$ | 拒絕常態假設 (厚尾與右偏) | 受多棟數百萬美元水岸豪宅殘差拉扯。然而樣本數龐大 ($N=17,290$)，依**中央極限定理 (CLT)**，參數估計量仍具備漸近常態性與無偏性。 |
| **4. 變異數同質性 (Homoscedasticity)** | Breusch-Pagan (BP) LM 檢定 | $\text{LM} = 2,164.44$ | $< 0.0001$ | 存在異質變異 (Heteroscedasticity) | 房價越高，模型預測離散度越大。實務上應使用 **White 穩健標準誤 (HC3/HC1)** 進行推論檢定，或對應變數取對數。 |
| **5. 無嚴重共線性 (Multicollinearity)** | 方差膨脹係數 (VIF) 檢核 | $\max(\text{VIF}) = 8.64$ | — | 通過檢驗 (所有 VIF $< 10$) | 剔除 `sqft_basement` 後，原本無限大之 VIF 降至 8.64 以下（最高為 `sqft_living`），消除嚴重多重共線性。 |

### 8.1 共線性前後對比驗證 (17 特徵 vs 16 特徵)
* **包含全部 17 項特徵時**:
  由於 $\text{sqft\_living} = \text{sqft\_above} + \text{sqft\_basement}$，計算得：
  $$\text{VIF}(\text{sqft\_living}) \to \infty, \quad \text{VIF}(\text{sqft\_above}) \to \infty, \quad \text{VIF}(\text{sqft\_basement}) \to \infty$$
  矩陣不可逆，部分迴歸係數符號產生混亂顛倒。
* **剔除 `sqft_basement` 之 16 特徵基準**:
  所有特徵 VIF 皆降至 10 的學術警戒值以下（`sqft_living`: 8.64, `grade`: 3.42, `lat`: 1.18, `waterfront`: 1.21），回歸係數呈現穩健數值解。

---

## 9. CRISP-DM Stage 6: Deployment & Interactive Predictor (部署與互動系統)

### 9.1 系統架構設計
本專案採用 **「現代靜態前端 + 離線訓練預算 JSON 產物 + 瀏覽器端零延遲運算」** 之部署架構：
1. **離線訓練管道 (`kc_house_analysis.py`)**:
   執行完整的數據清理、模型擬合、5 折交叉驗證、假設檢定與圖表匯出，將最佳估計參數與診斷數據序列化為 `kc_house_outputs/kc_house_model.joblib` 與 `kc_house_data.js`。
2. **前端動態渲染器 (`index.html` & `index.css`)**:
   使用者無需伺服器端 Python 環境，即可透過純 HTML5/JavaScript 在本機或 GitHub Pages 上秒級載入 21,613 筆數據分析儀表板，並在三種迴歸模式間無縫切換。

### 9.2 互動式房價預測器 (Interactive Predictor)
使用者可透過滑桿微調或直接輸入數值：
* **輸入參數**: 涵蓋室內坪數、建材等級、臥室、衛浴、水岸景觀、屋齡、經緯度等 16 項屬性。
* **即時回傳**:
  1. 預測成交總價（美元 USD，依據基準 OLS 方程即時計算）。
  2. 核心特徵貢獻分解（瀑布式呈現建築等級溢價、坪數價值與年份折舊）。
* **內建 4 種代表性金郡房屋情境預設**:
  * **金郡中位數住宅**: 3 房 / 2 衛 / 1,910 sqft / 等級 7 / 預測約 \$45 萬美元。
  * **Medina / Bellevue 水岸豪宅**: 5 房 / 4.5 衛 / 5,500 sqft / 等級 11 / 水岸 1 / 預測約 \$280 萬美元以上。
  * **西雅圖市區緊湊別墅**: 3 房 / 2.5 衛 / 1,500 sqft / 等級 8 / 緯度 47.66 / 預測約 \$68 萬美元。
  * **南郡經濟首購屋**: 2 房 / 1 衛 / 1,050 sqft / 等級 6 / 預測約 \$28 萬美元。

---

## 10. Conclusion (結論與建議)

1. **核心驅動因素**: 實證表明金郡房價之三大決定性支柱為：**室內生活面積 (`sqft_living`)**、**建築等級 (`grade`)** 與 **地理區位緯度 (`lat`)**。
2. **共線性處理**: 證實了資料集中存在的物理共線性問題（$\text{sqft\_living} = \text{sqft\_above} + \text{sqft\_basement}$），透過剪枝成功使矩陣恢復滿秩且維持預測精度。
3. **特徵精簡可行性**: Backward Elimination 證明移除統計不顯著的 `sqft_lot` 與 `floors` 僅損失 \$86 美元 RMSE，有助於簡化實務估價流程。
4. **模型極限與展望**: 線性模型在 \$2,000,000 以上頂級豪宅市場呈現異質變異與殘差右偏。未來可進一步評估對價格取對數之半對數模型（Log-Linear Model）或空間計量迴歸（Spatial Autoregressive Model）。

---

## 11. References (參考文獻)

1. Kaggle Dataset: *House Sales in King County, USA*, uploaded by harlfoxem. URL: [https://www.kaggle.com/datasets/harlfoxem/housesalesprediction](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction).
2. Wooldridge, J. M. (2015). *Introductory Econometrics: A Modern Approach* (6th ed.). Cengage Learning.
3. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
4. Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. *Proceedings of the 9th Python in Science Conference*.
5. Wierenga, P. (2013). Cross-Industry Standard Process for Data Mining (CRISP-DM). *Data Mining and Knowledge Discovery Handbook*.

---

## 12. GPT Assistance (AI 輔助開發真實紀錄)

本專案遵循學術誠信原則，如實紀錄 AI（大型語言模型）在開發過程中的實際協同工作：

### 12.1 實際 Prompt 與任務互動紀錄
1. **探索性計量分析與共線性診斷**:
   * *開發者指令*: 「請檢查金郡資料集是否存在物理共線性，並以 statsmodels 計算 17 特徵 vs 剔除後的 VIF。」
   * *AI 協同實作*: 撰寫自動化檢驗腳本，驗證 $\text{sqft\_living} = \text{sqft\_above} + \text{sqft\_basement}$ 之數值恆等性，並建構包含 Ramsey RESET、Breusch-Pagan 與 Jarque-Bera 之診斷模組。
2. **多模式特徵選擇流水線**:
   * *開發者指令*: 「請依據 CRISP-DM 規範，建構 Model A (Simple LR)、Model B (16 特徵基準) 與 Model C (包含 Pearson, SelectKBest, RFE, RFECV, Backward Elimination, VIF) 的 5 折交叉驗證流程，確保不得洩漏測試集。」
   * *AI 協同實作*: 實作 `kc_house_analysis.py`，統整生成標準化比較數據，並輸出對應的視覺化圖表與序列化模型。
3. **前端系統整合與無障礙相容**:
   * *開發者指令*: 「請在保留現有 50 Startups 與 Boston Housing 介面與導覽的前提下，將金郡 6 大 CRISP-DM 區塊、五大假設檢定卡片與互動預測滑桿整合至 `index.html`。」
   * *AI 協同實作*: 重構 `index.html` 之 JavaScript 資料綁定邏輯，實作三種模式切換按鈕與滑桿聯動即時試算功能。

---

## 13. NotebookLM Summary (NotebookLM 研讀總結)

> **說明與填寫指引：**
> 本區塊依作業規範預留予學生在透過 Google NotebookLM 上傳本專案之資料集來源、計量經濟文獻與分析產物進行自主研讀後，填入至少 100 字之繁體中文真實研讀筆記與學習心得。
> （為遵循學術誠信，AI 系統不偽造使用者之個人 NotebookLM 互動文字，請於完成個人研讀後直接於下方框線內替換或填寫。）

```text
【使用者 NotebookLM 研讀筆記填寫處（請填入至少 100 字繁體中文）】

範例引導方向：
1. 透過 NotebookLM 針對「特徵價格模型 (Hedonic Model)」與「高斯－馬可夫定理在截面房價資料上的假設違背」所生成的摘要心得。
2. 研讀金郡房價受地理經緯度與鄰舍坪數（sqft_living15）影響的空間效應筆記。
3. 比較單純最小平方法與正則化/特徵篩選法在房地產估價上的泛化誤差差異體會。
--------------------------------------------------------------------------------
（請在此處貼上您實際使用 NotebookLM 研讀本主題之個人摘要與心得，字數需達 100 字以上）
--------------------------------------------------------------------------------
```

---
*報告完成日期：2026 年 10 月 26 日*  
*專案程式碼與網頁：已整合於既有 GitHub 儲存庫 `AIOT-DIC2-MultipleLR`*
