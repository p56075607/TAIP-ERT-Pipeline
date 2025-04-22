**TAIP‑ERT‑Pipeline**  
中文副標：**TAIP 地電阻監控與反演資料自動流程**

---

## 一、專案簡介

這是一套針對「TAIP」站址之 R2MS 資料自動下載、前處理(csv→URF)、以及基於 PyGIMLi 的 ERT（Electrical Resistivity Tomography）反演與結果視覺化的全自動化工作流。  
主要功能模組：  
1. 資料擷取 (FTP 下載 R2MS 原始 CSV/ZIP)  
2. 前處理 (解壓、CSV→URF、基本 QC)  
3. 反演 (載入 URF→轉為 ohm、去 r‑index、網格生成 → 多次迭代反演)  
4. 結果輸出 (波形圖、反演結果圖、cross‑plot、misfit 直方圖)

---

## 二、專案結構提案
TAIP‑ERT‑Pipeline/
├── README.md
├── LICENSE
├── pyproject.toml # Poetry or setuptools 設定
├── requirements.txt # 相依套件列表
├── configs/ # 參數預設檔、FTP 站點、反演參數
│ └── default.yaml
│
├── src/
│ └── taip_ert_pipeline/ # 主程式套件
│ ├── init.py
│ ├── acquisition.py # 資料下載、目錄/日誌設定
│ ├── preprocessing.py # 解壓、csv2urf、QC
│ ├── inversion.py # PyGIMLi 反演流程
│ ├── visualization.py # 繪圖、結果輸出
│ └── utils.py # ToolBox 封裝: getR2MSdata, csv2urf, convertURF…
│
├── scripts/ # command‑line 執行器
│ └── run_pipeline.py # 一鍵執行完整管線
│
├── examples/ # 範例、Jupyter Notebooks
│ └── example_TAIP_run.ipynb
│
├── docs/
│ ├── index.md # 專案總覽
│ ├── usage.md # 快速上手
│ └── architecture.md # 模組架構說明
│
└── tests/ # 單元測試
├── test_acquisition.py
├── test_preprocessing.py
├── test_inversion.py
└── test_utils.py


---

## 三、模組對應與說明

1. **acquisition.py**  
   ‑ 封裝 `getR2MSdata`、`unzip_files`、`ensure_directories`、`setup_logging` 等功能  
   ```python
   1:128:R2MS_monitor/R2MS_TAIP.py
   def getdata_and_csv2urf_main(...):
       ensure_directories(...)
       getR2MSdata(...)
       # csv→urf 主流程
   ```
2. **preprocessing.py**  
   ‑ 包含 `csv2urf`、URF 檔案檢查、初步波形繪圖  
3. **inversion.py**  
   ‑ PyGIMLi 反演主流程，承接 URF→ohm、ridx 分析、去壞點、建立網格、反演、多次迭代  
   ```python
   1:195:pyGIMLi/field data/TAIP_monitor/cgrg_test/ERT_inversion_TAIP.py
   mgr = ert.ERTManager(data)
   for i in range(repeat_times):
       mgr.invert(...)
       mgr.saveResult(...)
   ```
4. **visualization.py**  
   ‑ cross‑plot、misfit histogram、反演剖面 & 等值線圖、色彩對應  
5. **utils.py**  
   ‑ 封裝 `convertURF`, `ridx_analyse`, `data_filtering`, `create_inv_mesh` 等函式  

---

## 四、README.md 範例大綱

```markdown
# TAIP‑ERT‑Pipeline

## 1. 專案背景
- 介紹 TAIP 地電阻監測系統與 R2MS 資料格式

## 2. 核心功能
1. 自動 FTP 下載 R2MS 原始壓縮與 CSV  
2. CSV → URF 前處理並產生波形圖  
3. 使用 PyGIMLi 做 ERT 反演與多次迭代壓制壞點  
4. 輸出反演剖面、等值線圖、cross‑plot、misfit histogram

## 3. 快速上手
```bash
git clone https://github.com/yourname/TAIP-ERT-Pipeline.git
cd TAIP-ERT-Pipeline
pip install -r requirements.txt
# 編輯 configs/default.yaml 設定 FTP, 路徑, 反演參數
python scripts/run_pipeline.py --config configs/default.yaml
```

## 4. 專案結構

如上所示

## 5. 開發與貢獻
- PR、Issue、版本標記流程

## 6. 授權條款
- MIT, Apache2.0 等

---

## 五、未來擴充

- 支援多站點 (TAIP、其他測站)  
- Web UI (Flask / Dash)  
- 自動化 CI/CD (GitHub Actions)  
- Docker 容器化  
- 加入更多反演算法選項  
