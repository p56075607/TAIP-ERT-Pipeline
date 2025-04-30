**TAIP‑ERT‑Pipeline**  
中文副標：**TAIP 地電阻監控與反演資料自動流程**

---

## 一、專案簡介

這是一套針對「TAIP」站址之 R2MS 資料自動下載、前處理(csv→URF)、以及基於 PyGIMLi 的 ERT（Electrical Resistivity Tomography）反演與結果視覺化的全自動化工作流。  
主要功能模組：  
1. 資料擷取 (FTP 下載 R2MS 原始 CSV/ZIP)  
2. 前處理 (解壓、CSV→URF、基本 QC、波形圖)  
3. 反演 (載入 URF→轉為 ohm、去 r‑index、網格生成 → 多次迭代反演)  
4. 結果輸出 (反演結果圖、cross‑plot、misfit 直方圖)

---

## 二、專案結構提案
TAIP‑ERT‑Pipeline/
├── README.md
├── LICENSE
├── requirements.txt # 相依套件列表
├── configs/ # 參數預設檔、FTP 站點、反演參數
│ └── site.yaml
│
├── src/
│ └── taip_ert_pipeline/ # 主程式套件
│   ├── init.py
│   ├── pipeline.py # 處理程序
│   ├── acquisition.py # 資料下載、目錄/日誌設定
│   ├── preprocessing.py # 資料前處理類別
│   ├── inversion.py # PyGIMLi 反演類別
│   ├── utils.py # 工具程式
│   └── visualization.py # 繪圖、結果輸出
│
├── scripts/ # command‑line 執行器
│ ├── run_pipeline_R2MS.py # 一鍵執行完整流程 包含 R2MS_TAIP.py 功能 
│ └── run_pipeline_inversion.py # 一鍵執行完整流程 包含 ERT_inversion_TAIP.py 功能 
│
├── examples/ # 範例、Jupyter Notebooks
│ └── example_TAIP_run.ipynb
│
├── docs/
│ ├── index.md # 專案總覽
│ ├── usage.md # 快速上手
│ └── architecture.md # 模組架構說明
│



---

## 三、模組對應與說明

# 1. 腳本（scripts/run_pipeline.py）

- **角色**：  
  – 管道（Pipeline）的**執行入口**，負責「解析使用者輸入」並啟動整個流程。  
  – 通常只做參數（或設定檔）讀取、日誌設定、錯誤攔截、呼叫 `taip_ert_pipeline` 內的主函式。  

- **典型內容**：  
  ```python:path/to/scripts/run_pipeline.py
  import argparse
  import yaml
  from taip_ert_pipeline import pipeline

  def main():
      parser = argparse.ArgumentParser("TAIP‑ERT Pipeline")
      parser.add_argument("--config", "-c", required=True, help="設定檔路徑")
      args = parser.parse_args()

      # 讀取 YAML 設定檔
      with open(args.config) as f:
          cfg = yaml.safe_load(f)

      # 啟動管線
      pipeline.run_pipeline(cfg)

  if __name__ == "__main__":
      main()
  ```
  - 主要是「調度」與「橋接」功能，不直接實作下載、前處理、反演的細節。

---

# 2. 主程式套件（src/taip_ert_pipeline/）

- **角色**：  
  – 真正的**功能實作庫** (Library)，把整個專案的核心邏輯拆成不同模組：  
    1. `acquisition.py`   → FTP 下載、解壓、日誌與目錄管理  
    2. `preprocessing.py` → CSV→URF、波形圖、QC  
    3. `inversion.py`     → PyGIMLi 網格生成、多次迭代反演  
    4. `visualization.py` → 各種圖表輸出（剖面圖、contour、cross‑plot、misfit）  
    5. `utils.py`         → ToolBox 函式封裝：`getR2MSdata`、`csv2urf`、`convertURF`…  

- **典型結構**：  
  ```
  src/taip_ert_pipeline/
  ├── __init__.py
  ├── pipeline.py          # 主流程：串接 acquisition→preprocessing→inversion→visualization
  ├── acquisition.py
  ├── preprocessing.py
  ├── inversion.py
  ├── visualization.py
  └── utils.py
  ```
  ```python:path/to/src/taip_ert_pipeline/pipeline.py
  def run_pipeline(cfg):
      # 1. 資料擷取
      acq = acquisition.ETRAcquirer(cfg["ftp"], cfg["data"])
      acq.download_and_prepare()
      
      # 2. 前處理
      pre = preprocessing.ERTPreprocessor(cfg["data"])
      urf_files = pre.csv2urf_all()
      
      # 3. 反演
      inv = inversion.ERTInverter(cfg["inversion"])
      results = inv.run(urf_files)
      
      # 4. 可視化
      viz = visualization.ERTVisualizer(cfg["output"])
      viz.plot_all(results)
  ```

- **使用方式**：  
  – 可以被其他程式 `import`，撰寫單元測試、延伸或改寫某段流程。  
  – 維持高內聚、低耦合，方便維護與擴充。

---

## 四、快速上手

### 系統需求
- Python 3.8 或更高版本
- Windows/Linux/macOS (已在 Windows 10 上測試)

### 安裝步驟
```bash
# 1. 複製專案
git clone https://github.com/p56075607/TAIP-ERT-Pipeline.git
cd TAIP-ERT-Pipeline

# 2. 建立虛擬環境 (選用但建議)
python -m venv venv
# 在 Windows 上:
venv\Scripts\activate
# 在 Linux/macOS 上:
source venv/bin/activate

# 3. 安裝依賴套件
pip install -r requirements.txt

# 4. 安裝 PyGIMLi (如果上一步未能成功安裝)
pip install pygimli
# 或者從官方文件安裝: https://www.pygimli.org/installation.html
```

### 配置與執行
```bash
# 1. 編輯配置文件
# 複製範例配置文件
cp configs/site.yaml configs/my_site.yaml
# 編輯並設定 FTP 服務器、站點、路徑等參數

# 2. 執行 R2MS 資料擷取與前處理
python scripts/run_pipeline_R2MS.py --config configs/my_site.yaml

# 3. 執行 ERT 反演與視覺化
python scripts/run_pipeline_inversion.py --config configs/my_site.yaml
```

### 命令列參數
- 資料擷取腳本 (`run_pipeline_R2MS.py`)
  - `--config/-c`: 配置文件路徑
  - `--station/-s`: 站點名稱
  - `--line/-l`: 測線名稱
  - `--days/-d`: 處理天數
  - `--all-files/-a`: 下載所有檔案
  - `--schedule/-S`: 啟用定時排程執行

- 反演腳本 (`run_pipeline_inversion.py`)
  - `--config/-c`: 配置文件路徑
  - `--urf/-u`: 單一 URF 檔案路徑
  - `--test-dir/-t`: 測試目錄
  - `--repeat/-r`: 反演重複次數
  - `--output/-o`: 輸出目錄

---

## 五、未來擴充

- 支援多站點 (TAIP、其他測站)  
- Web UI (Flask / Dash)  
- 自動化 CI/CD (GitHub Actions)  
- Docker 容器化  
- 加入更多反演算法選項
