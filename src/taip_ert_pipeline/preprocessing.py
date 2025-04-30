"""前處理模組 (解壓、CSV→URF、基本 QC、波形圖)"""

import os
import glob
import logging
import datetime
import shutil
from . import utils

class ERTPreprocessor:
    """處理 ERT 資料前處理的類別，負責 CSV→URF 轉換及相關 QC"""
    
    def __init__(self, data_config):
        """
        初始化 ERT 前處理器
        
        參數:
            data_config: 包含資料相關配置的字典
        """
        self.root_dir = data_config.get("root", "")
        self.station_name = data_config.get("station_name", "TAIP")
        self.line_name = data_config.get("line_name", "T1")
        self.days_to_review = data_config.get("days_to_review", 30)
        self.plot_time = data_config.get("plot_time", [0, 4, 8, 20])
        
        # 設置 CSV→URF 參數
        self.amplitude_estimate_start_position = data_config.get("amplitude_estimate_start_position", 2)
        self.amplitude_estimate_range = data_config.get("amplitude_estimate_range", 4)
        self.png_file_first_name = f"SEE({self.amplitude_estimate_start_position}_{self.amplitude_estimate_range})"
        
        # 輸出路徑設定
        self.urf_dir = os.path.join(self.root_dir, 'urf')
        self.csv_dir = os.path.join(self.root_dir, 'csv', 'Recorder')
        self.output_png_dir = os.path.join(self.root_dir, 'Output_png')
        self.geo_file = os.path.join(self.root_dir, f'GEO_{self.line_name}.urf')
        
        # 設置日誌 - 修改為只使用已存在的logger
        self.logger = logging.getLogger(f'preprocessing_{self.line_name}')
        self.logger.setLevel(logging.INFO)
        
        # 只有在沒有任何處理器時才添加處理器
        # 這樣可以避免重複輸出
        if not self.logger.handlers:
            # 添加文件處理器
            log_file = os.path.join(self.root_dir, f'csv2urf_{self.line_name}.log')
            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%y/%m/%d - %H:%M:%S')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def csv2urf_all(self):
        """處理所有 CSV 檔案轉換為 URF 格式，並返回已處理的 URF 檔案列表"""
        processed_urf_files = []
        
        # 取得所有日期目錄
        date_dirs = sorted(glob.glob(os.path.join(self.csv_dir, '*')), 
                        key=lambda x: datetime.datetime.strptime(os.path.basename(x), '%y%m%d'), 
                        reverse=True)
        
        # 處理指定天數內的資料
        for date_path in date_dirs[:self.days_to_review]:
            date = os.path.basename(date_path)
            self.logger.info(f"處理日期: {date}")
            
            # 從最後一小時開始往前處理
            for hour in range(23, -1, -1):
                hour_str = f"{hour:02d}"
                
                # 找出該小時的 CSV 文件
                csv_pattern = os.path.join(date_path, f"{hour_str}*_E.csv")
                csv_files = glob.glob(csv_pattern)
                
                # 檢查對應的 URF 文件是否已存在
                urf_pattern = os.path.join(self.urf_dir, f"{date}{hour_str}*{self.line_name}.urf")
                urf_files = glob.glob(urf_pattern)
                
                # 如果 URF 不存在但 CSV 存在，則進行轉換
                if not urf_files and csv_files:
                    output_urf_file_name = f"{date}{hour_str}_m_{self.line_name}.urf"
                    output_urf_path = os.path.join(self.urf_dir, output_urf_file_name)
                    output_png_path = os.path.join(self.output_png_dir, f"{date}{hour_str}")
                    
                    # 判斷是否需要生成波形圖
                    plot_wave = hour in self.plot_time
                    
                    self.logger.info(f"開始處理 ERT 資料: {date}{hour_str}_E")
                    
                    # 調用 csv2urf 轉換
                    return_code = utils.csv2urf(csv_files, self.geo_file, self.urf_dir, output_urf_file_name,
                                               output_png_path, plot_wave, self.png_file_first_name, 
                                               self.amplitude_estimate_start_position, self.amplitude_estimate_range, 
                                               contain_common_N=False)
                    
                    # 記錄處理結果
                    self.logger.info(f"處理 {output_urf_file_name}: 返回碼={return_code}")
                    
                    # 如果處理成功，嘗試複製 URF 檔案到反演目錄
                    if return_code == 0:
                        # 檢查文件是否真的存在
                        if os.path.isfile(output_urf_path):
                            processed_urf_files.append(output_urf_path)
                            
                            # 複製 URF 檔案到反演測試目錄
                            inversion_urf_dir = os.path.join(self.root_dir + '_test', 'urf')
                            if not os.path.exists(inversion_urf_dir):
                                os.makedirs(inversion_urf_dir)
                                
                            target_path = os.path.join(inversion_urf_dir, output_urf_file_name)
                            if not os.path.isfile(target_path):
                                try:
                                    shutil.copy2(output_urf_path, target_path)
                                    self.logger.info(f"複製檔案到反演目錄: {target_path}")
                                except Exception as e:
                                    self.logger.warning(f"複製檔案失敗: {str(e)}")
                        else:
                            self.logger.warning(f"URF 檔案不存在: {output_urf_path}")
        
        self.logger.info(f"總共處理 {len(processed_urf_files)} 個 URF 檔案")
        return processed_urf_files
    
    def process_single_file(self, csv_file_path):
        """處理單個 CSV 檔案轉換為 URF"""
        file_name = os.path.basename(csv_file_path)
        if not file_name.endswith('_E.csv'):
            self.logger.warning(f"檔案格式不符合要求: {file_name}")
            return None
        
        # 提取日期和小時
        date_hour = file_name.split('_')[0]
        if len(date_hour) >= 8:  # 確保名稱符合格式
            date = date_hour[:6]
            hour = date_hour[6:8]
            
            output_urf_file_name = f"{date}{hour}_m_{self.line_name}.urf"
            output_png_path = os.path.join(self.output_png_dir, f"{date}{hour}")
            
            # 判斷是否需要生成波形圖
            plot_wave = int(hour) in self.plot_time
            
            self.logger.info(f"開始處理單個檔案: {file_name}")
            
            # 調用 csv2urf 轉換
            return_code = utils.csv2urf([csv_file_path], self.geo_file, self.urf_dir, output_urf_file_name,
                                       output_png_path, plot_wave, self.png_file_first_name, 
                                       self.amplitude_estimate_start_position, self.amplitude_estimate_range, 
                                       contain_common_N=False)
            
            # 記錄處理結果
            self.logger.info(f"處理 {output_urf_file_name}: 返回碼={return_code}")
            
            output_urf_path = os.path.join(self.urf_dir, output_urf_file_name)
            if return_code == 0 and os.path.isfile(output_urf_path):
                return output_urf_path
        
        return None 