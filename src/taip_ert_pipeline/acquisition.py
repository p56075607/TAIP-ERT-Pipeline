"""資料擷取模組 (FTP 下載 R2MS 原始 CSV/ZIP)"""

import os
import logging
import datetime
import ftplib
import glob
from . import utils

class ERTAcquirer:
    """處理 ERT 資料擷取的類別，負責從 FTP 下載資料"""
    
    def __init__(self, ftp_config, data_config):
        """
        初始化 ERT 資料擷取器
        
        參數:
            ftp_config: 包含 FTP 設定的字典 (host, user, password)
            data_config: 包含資料設定的字典 (root, station_name, line_name, days_to_review)
        """
        self.ftp_host = ftp_config.get("host", "")
        self.ftp_user = ftp_config.get("user", "")
        self.ftp_password = ftp_config.get("password", "")
        
        self.root_dir = data_config.get("root", "")
        self.station_name = data_config.get("station_name", "TAIP")
        self.line_name = data_config.get("line_name", "T1")
        self.days_to_review = data_config.get("days_to_review", 30)
        self.download_all_files = data_config.get("download_all_files", False)
        
        # 設置日誌 - 修改為只使用已存在的logger
        self.logger = logging.getLogger(f'acquisition_{self.line_name}')
        self.logger.setLevel(logging.INFO)
        
        # 只有在沒有任何處理器時才添加處理器
        # 這樣可以避免重複輸出
        if not self.logger.handlers:
            # 添加文件處理器
            log_file = os.path.join(self.root_dir, f'acquisition_{self.line_name}.log')
            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%y/%m/%d - %H:%M:%S')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # 確保所有必要目錄存在
        self._ensure_directories()
    
    def _ensure_directories(self):
        """確保所有需要的目錄都存在，如果不存在就創建它們"""
        directories = [
            os.path.join(self.root_dir, 'csv'),
            os.path.join(self.root_dir, 'csv', 'Recorder'),
            os.path.join(self.root_dir, 'urf'),
            os.path.join(self.root_dir, 'Output_png'),
            os.path.join(self.root_dir + '_test', 'urf')
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                self.logger.info(f"創建目錄: {directory}")
    
    def download_and_prepare(self):
        """
        從 FTP 伺服器下載 R2MS 數據並準備處理
        
        實現 getR2MSdata 功能，下載指定天數內的資料
        """
        base_folder_path = os.path.join(self.root_dir, "csv")
        self.logger.info(f"開始從 FTP 伺服器下載資料，目標目錄: {base_folder_path}")
        
        try:
            # 連接 FTP 伺服器
            ftp = ftplib.FTP(self.ftp_host)
            ftp.login(self.ftp_user, self.ftp_password)
            self.logger.info(f"成功連接到 FTP 伺服器: {self.ftp_host}")
            
            # 計算日期範圍
            current_date = datetime.datetime.now()
            date_list = [(current_date - datetime.timedelta(days=i)) for i in range(self.days_to_review)]
            
            for date in date_list:
                # 使用與 getR2MSdata.py 相同的路徑格式
                folder_name = date.strftime("%y%m%d")
                remote_directory = f"SAI1/Recorder/{folder_name}"
                
                try:
                    ftp.cwd(remote_directory)
                    self.logger.info(f"成功訪問遠端目錄: {remote_directory}")
                except ftplib.error_perm:
                    self.logger.warning(f"遠端目錄不存在: {remote_directory}")
                    continue
                
                # 確保本地目錄存在
                local_base_dir = os.path.join(base_folder_path, "Recorder", folder_name)
                if not os.path.exists(local_base_dir):
                    os.makedirs(local_base_dir)
                
                # 準備本地檔案列表進行比對
                local_file_list = []
                if os.path.exists(local_base_dir):
                    local_file_list = [f for f in os.listdir(local_base_dir) if f.endswith('.csv')]
                
                # 獲取遠端文件列表
                if self.download_all_files:
                    remote_file_list = ftp.nlst('*.zip')
                    self.logger.info(f"檢查所有壓縮檔 (*.zip)，共找到 {len(remote_file_list)} 個檔案")
                else:
                    remote_file_list = ftp.nlst('*_E.zip')
                    self.logger.info(f"只檢查電測資料檔 (*_E.zip)，共找到 {len(remote_file_list)} 個檔案")
                
                # 下載文件
                for remote_filename in remote_file_list:
                    local_filename = remote_filename.replace('.zip', '.csv')
                    # 檢查本地是否已有對應的 CSV 檔案
                    if local_filename not in local_file_list:
                        # 直接下載到 base_folder_path
                        self.logger.info(f"下載文件: {folder_name}, {remote_filename}")
                        with open(os.path.join(base_folder_path, remote_filename), 'wb') as f:
                            ftp.retrbinary(f'RETR {remote_filename}', f.write)
                
                # 回到上級目錄
                try:
                    ftp.cwd('../../..')
                    self.logger.info("返回上級目錄")
                except ftplib.error_perm as e:
                    self.logger.warning(f"無法返回上級目錄: {str(e)}")
                    # 如果無法返回上級目錄，嘗試回到根目錄
                    ftp.cwd('/')
            
                # 下載完成後解壓縮所有 ZIP 文件
                self.logger.info(f"開始解壓縮下載的文件")
                utils.unzip_files(base_folder_path)
            
            ftp.quit()
            self.logger.info("FTP 下載完成")
            
        except Exception as e:
            self.logger.error(f"FTP 下載過程中發生錯誤: {str(e)}", exc_info=True)
            raise
    
    def _check_if_file_exists(self, ftp, remote_filename, local_filepath):
        """檢查本地文件是否已存在且大小與遠端文件相同"""
        if not os.path.exists(local_filepath):
            return False
        
        # 獲取遠端文件大小
        remote_size = ftp.size(remote_filename)
        local_size = os.path.getsize(local_filepath)
        
        return remote_size == local_size 