"""主流程：串接 acquisition → preprocessing → inversion → visualization"""

import os
import logging
from . import acquisition
from . import preprocessing
from . import inversion
from . import visualization

def setup_logging(config):
    """設置日誌配置 - 為所有模組設置控制台輸出，避免重複輸出"""
    # 獲取根目錄
    root_dir = config["data"]["root"]
    line_name = config["data"]["line_name"]
    
    # 設置根 logger
    root_logger = logging.getLogger('')
    root_logger.setLevel(logging.INFO)
    
    # 檢查全局 logger 是否已有處理器
    if not root_logger.handlers:
        # 設置控制台處理器（只需要添加一次）
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%y/%m/%d - %H:%M:%S')
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # 設置 pipeline logger 的文件處理器
    pipeline_logger = logging.getLogger('pipeline')
    pipeline_logger.setLevel(logging.INFO)
    
    # 檢查此 logger 是否已有文件處理器
    has_file_handler = False
    for handler in pipeline_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            has_file_handler = True
            break
    
    if not has_file_handler:
        log_file = os.path.join(root_dir, f'pipeline_{line_name}.log')
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%y/%m/%d - %H:%M:%S')
        file_handler.setFormatter(formatter)
        pipeline_logger.addHandler(file_handler)
    
    return pipeline_logger

def run_pipeline(config):
    """
    運行完整的 TAIP ERT 流程
    
    參數:
        config: 包含所有配置參數的字典，如 FTP 設定、資料目錄、反演參數等
    """
    # 設置日誌
    logger = setup_logging(config)
    logger.info("開始執行 TAIP ERT 流程")
    
    try:
        # 1. 資料擷取 (FTP 下載 R2MS 原始 CSV/ZIP)
        logger.info("步驟 1: 開始資料擷取")
        acq = acquisition.ERTAcquirer(config["ftp"], config["data"])
        acq.download_and_prepare()
        
        # 2. 前處理 (解壓、CSV→URF、基本 QC、波形圖)
        logger.info("步驟 2: 開始資料前處理")
        pre = preprocessing.ERTPreprocessor(config["data"])
        urf_files = pre.csv2urf_all()
        
        # 3. 反演 (載入 URF→轉為 ohm、去 r‑index、網格生成 → 多次迭代反演)
        logger.info("步驟 3: 開始資料反演處理")
        # 將 output 配置中的輸出目錄、XZV 文件等參數合併到反演配置中
        # 複製反演配置以避免修改原始配置
        inversion_config = config["inversion"].copy()
        
        # 如果 output 配置中有以下參數，則添加到反演配置中
        if "output" in config:
            # 輸出目錄
            if "output_dir" in config["output"]:
                inversion_config["output_dir"] = config["output"]["output_dir"]
            
            # XZV 文件
            if "xzv_file" in config["output"]:
                inversion_config["xzv_file"] = config["output"]["xzv_file"]
                
            # 標題格式
            if "title_verbose" in config["output"]:
                inversion_config["title_verbose"] = config["output"]["title_verbose"]
                
            # 色彩圖
            if "colormap_file" in config["output"] and "colormap_file" not in inversion_config:
                inversion_config["colormap_file"] = config["output"]["colormap_file"]
        
        inv = inversion.ERTInverter(inversion_config)
        results = inv.run_inversion(urf_files)
        
        # 4. 結果輸出 (反演結果圖、cross‑plot、misfit 直方圖)
        logger.info("步驟 4: 開始結果視覺化")
        
        # 讀取 xzv 檔案路徑
        xzv_file = config["output"].get("xzv_file", None)
        if xzv_file and os.path.exists(xzv_file):
            logger.info(f"找到 XZV 檔案: {xzv_file}，將用於繪製等值線圖")
        else:
            logger.warning(f"未找到 XZV 檔案，將使用默認方法繪製等值線圖")
            xzv_file = None
            
        # 讀取 title_verbose 選項
        title_verbose = config["output"].get("title_verbose", False)
        if title_verbose:
            logger.info("使用詳細標題格式")
        else:
            logger.info("使用簡潔標題格式")
            
        viz = visualization.ERTVisualizer(config["output"])
        
        # 對每個結果調用 plot_all，並傳遞 xzv_file 和 title_verbose
        for result in results:
            for iteration in result.get("iterations", []):
                save_path = iteration.get("save_path")
                file_name = os.path.basename(iteration.get("file", ""))
                if save_path and file_name:
                    logger.info(f"為 {file_name} 生成視覺化結果")
                    viz.plot_all(save_path, file_name, xzv_file, title_verbose)
        
        logger.info("TAIP ERT 流程執行完成")
        return True
    
    except Exception as e:
        logger.error(f"流程執行過程中發生錯誤: {str(e)}", exc_info=True)
        return False

def run_acquisition_only(config):
    """
    僅運行資料擷取和前處理階段
    
    參數:
        config: 配置字典
    """
    logger = setup_logging(config)
    logger.info("開始執行 TAIP R2MS 資料擷取流程")
    
    try:
        # 1. 資料擷取 (FTP 下載 R2MS 原始 CSV/ZIP)
        acq = acquisition.ERTAcquirer(config["ftp"], config["data"])
        acq.download_and_prepare()
        
        # 2. 前處理 (解壓、CSV→URF、基本 QC、波形圖)
        pre = preprocessing.ERTPreprocessor(config["data"])
        urf_files = pre.csv2urf_all()
        
        logger.info("R2MS 資料擷取流程執行完成")
        return urf_files
    
    except Exception as e:
        logger.error(f"資料擷取過程中發生錯誤: {str(e)}", exc_info=True)
        return []

def run_inversion_only(config, urf_files=None):
    """
    僅運行反演和視覺化階段
    
    參數:
        config: 配置字典
        urf_files: URF 檔案列表，如果為 None 則從配置中讀取
    """
    logger = setup_logging(config)
    logger.info("開始執行 TAIP ERT 反演流程")
    
    try:
        # 如果未提供 URF 檔案，嘗試從配置的目錄讀取
        if urf_files is None:
            from . import utils
            urf_dir = os.path.join(config["data"]["root"], 'urf')
            urf_files = utils.find_urf_files(urf_dir)
            
        # 將 output 配置中的輸出目錄、XZV 文件等參數合併到反演配置中
        # 複製反演配置以避免修改原始配置
        inversion_config = config["inversion"].copy()
        
        # 如果 output 配置中有以下參數，則添加到反演配置中
        if "output" in config:
            # 輸出目錄
            if "output_dir" in config["output"]:
                inversion_config["output_dir"] = config["output"]["output_dir"]
            
            # XZV 文件
            if "xzv_file" in config["output"]:
                inversion_config["xzv_file"] = config["output"]["xzv_file"]
                
            # 標題格式
            if "title_verbose" in config["output"]:
                inversion_config["title_verbose"] = config["output"]["title_verbose"]
                
            # 色彩圖
            if "colormap_file" in config["output"] and "colormap_file" not in inversion_config:
                inversion_config["colormap_file"] = config["output"]["colormap_file"]
        
        # 運行反演
        inv = inversion.ERTInverter(inversion_config)
        results = inv.run_inversion(urf_files)
        
        logger.info("ERT 反演流程執行完成")
        return True
    
    except Exception as e:
        logger.error(f"反演過程中發生錯誤: {str(e)}", exc_info=True)
        return False 