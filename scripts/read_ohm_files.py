#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import pygimli as pg
from datetime import datetime
import re

def extract_date_from_filename(file_path):
    """
    從檔案路徑中提取日期（格式為25XXXXXX）
    返回可用於排序的字符串
    """
    file_name = os.path.basename(file_path)
    # 使用正則表達式提取日期部分
    match = re.match(r'(\d{8})_m_T1_inv\.ohm', file_name)
    if match:
        return match.group(1)
    # 如果無法匹配，返回原始檔名作為後備
    return file_name

def read_ohm_files():
    """
    讀取所有output資料夾中的25XXXXXX_m_T1資料夾中的repeat_4/ERTManager/25XXXXXX_m_T1_inv.ohm檔案
    使用pg.load加載數據並打印出來
    檔案按照名稱中的日期排序
    """
    # 外部output資料夾的路徑
    base_path = "e:/R2MSDATA/TAIP_T1_test/output"
    
    # 使用glob尋找所有符合模式的資料夾
    pattern = os.path.join(base_path, "25*_m_T1", "repeat_4", "ERTManager", "25*_m_T1_inv.ohm")
    ohm_files = glob.glob(pattern)
    
    # 按照檔案名稱中的日期排序
    ohm_files = sorted(ohm_files, key=extract_date_from_filename)
    
    print(f"=========================================")
    print(f"找到的.ohm檔案總數: {len(ohm_files)}")
    print(f"=========================================")
    print(f"檔案已按照日期排序")
    
    # 創建結果文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"scripts/ohm_files_data_{timestamp}.txt"
    
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(f"=========================================\n")
        f.write(f"找到的.ohm檔案總數: {len(ohm_files)}\n")
        f.write(f"=========================================\n")
        f.write(f"檔案已按照日期排序\n\n")
        
        # 讀取每個.ohm檔案並打印數據
        for i, file_path in enumerate(ohm_files, 1):
            folder_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
            file_name = os.path.basename(file_path)
            
            # print(f"\n處理第 {i}/{len(ohm_files)} 個檔案: {file_name}")
            f.write(f"\n檔案 {i}/{len(ohm_files)}: {folder_name}/{file_name}\n")
            
            try:
                # 使用pg.load讀取.ohm檔案
                data = pg.load(file_path)
                
                # 打印數據到控制台和檔案
                print(f"{file_name} 數據資訊: {data}")
                f.write(f"數據資訊: {data}\n")
                
                # 檢查數據類型和基本資訊
                if hasattr(data, 'shape'):
                    print(f"數據形狀: {data.shape}")
                    f.write(f"數據形狀: {data.shape}\n")
                
                if hasattr(data, 'dtype'):
                    print(f"數據類型: {data.dtype}")
                    f.write(f"數據類型: {data.dtype}\n")
                
                # 如果數據是可迭代的並且不是太大，我們可以打印部分內容
                if hasattr(data, '__iter__') and not isinstance(data, str):
                    try:
                        # 嘗試獲取前5個元素
                        data_sample = list(data)[:5] if len(data) > 5 else data
                        print(f"數據前5個元素（或全部）: {data_sample}")
                        f.write(f"數據前5個元素（或全部）: {data_sample}\n")
                    except (TypeError, AttributeError):
                        print("無法顯示數據元素")
                        f.write("無法顯示數據元素\n")
                
            except Exception as e:
                print(f"無法讀取檔案 {file_name}. 錯誤: {str(e)}")
                f.write(f"無法讀取檔案 {file_name}. 錯誤: {str(e)}\n")
    
    print(f"\n結果已保存到檔案: {result_file}")

if __name__ == "__main__":
    read_ohm_files() 