#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TAIP ERT Pipeline - R2MS Lite 資料擷取與前處理腳本

功能：
1. 從 FTP 伺服器下載 R2MS Lite 原始 CSV 資料
2. 處理 CSV 格式並轉換為 URF 格式  
3. 產生基本 QC 和波形圖

使用方式:
    python run_pipeline_R2MS_Lite.py --config configs/site_TARI_E1_Lite.yaml
"""

import os
import sys
import argparse
import yaml
import datetime
import time
import ftplib
import pandas as pd
import shutil
from pathlib import Path

# 添加 src 目錄到路徑
sys.path.append(str(Path(__file__).parent.parent))

from src.taip_ert_pipeline import pipeline, utils

def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(description="TAIP ERT Pipeline - R2MS Lite 資料擷取與前處理")
    parser.add_argument("--config", "-c", required=True, help="設定檔路徑")
    parser.add_argument("--station", "-s", help="站點名稱，會覆蓋設定檔中的值")
    parser.add_argument("--line", "-l", help="測線名稱，會覆蓋設定檔中的值")
    parser.add_argument("--days", "-d", type=int, help="處理的天數，會覆蓋設定檔中的值")
    parser.add_argument("--root", "-r", help="根目錄路徑，會覆蓋設定檔中的值")
    parser.add_argument("--schedule", "-S", action="store_true", help="啟用排程，每天特定時間執行")
    parser.add_argument("--time", "-t", default="23:45", help="排程時間 (HH:MM 格式)，預設 23:45")
    
    return parser.parse_args()

def download_csv_from_ftp(ftp_config, target_datetime, local_root_path):
    """
    從 FTP 下載指定時間的 CSV 檔案
    
    Args:
        ftp_config: FTP 配置
        target_datetime: 目標日期時間
        local_root_path: 本地根目錄路徑
    
    Returns:
        下載成功的 CSV 檔案路徑列表
    """
    downloaded_files = []
    
    try:
        # 連接 FTP
        ftp = ftplib.FTP(ftp_config['host'])
        ftp.login(ftp_config['user'], ftp_config['password'])
        
        # 構建 FTP 路徑
        year = target_datetime.strftime('%Y')
        month = target_datetime.strftime('%m')
        day = target_datetime.strftime('%d')
        hour = target_datetime.strftime('%H')
        
        # FTP 路徑格式: [20240409A]Taiwan-Taoyuan-NCUFactory(NCUF)/Recorder/Factory/2025/08/08/HH*/Part01/1/
        ftp_base_path = "[20240409A]Taiwan-Taoyuan-NCUFactory(NCUF)/Recorder/Factory"
        ftp_date_path = f"{ftp_base_path}/{year}/{month}/{day}"
        
        print(f"尋找小時目錄: {ftp_date_path}/{hour}*")
        
        try:
            # 先切換到日期目錄
            ftp.cwd("/")  # 回到根目錄，確保路徑正確
            ftp.cwd(ftp_date_path)
            
            # 列出所有目錄，找到符合 HH* 格式的目錄
            dir_list = ftp.nlst()
            matching_dirs = [d for d in dir_list if d.startswith(hour)]
            
            for hour_dir in matching_dirs:
                try:
                    # 每次都從根目錄重新導航，避免路徑累積
                    target_path = f"{ftp_base_path}/{year}/{month}/{day}/{hour_dir}/Part01/1"
                    ftp.cwd("/")  # 回到根目錄
                    ftp.cwd(target_path)
                    
                    print(f"成功進入目錄: {target_path}")
                    
                    # CSV 檔案名格式: S004YYYYMMDDHHM*.v299.csv
                    csv_pattern = f"S004{year}{month}{day}{hour}"
                    
                    # 檢查所有檔案，找到符合格式的 CSV
                    file_list = ftp.nlst()
                    matching_files = [f for f in file_list if f.startswith(csv_pattern) and f.endswith('.v299.csv')]
                    
                    for csv_filename in matching_files:
                        print(f"找到檔案: {target_path}/{csv_filename}")
                        
                        # 建立本地目錄結構
                        local_date_dir = os.path.join(local_root_path, "csv", "Recorder", f"{year[2:]}{month}{day}")
                        os.makedirs(local_date_dir, exist_ok=True)
                        
                        # 從檔案名提取完整的時間 (HHMM)
                        # S004YYYYMMDDHHmm.v299.csv -> 取得 HHmm 部分
                        file_time_part = csv_filename[12:16]  # 取得 HHMM
                        
                        # 本地檔案名格式: HHmm00_E.csv
                        local_filename = f"{file_time_part}00_E.csv"
                        local_file_path = os.path.join(local_date_dir, local_filename)
                        
                        # 下載檔案
                        with open(local_file_path, 'wb') as local_file:
                            ftp.retrbinary(f'RETR {csv_filename}', local_file.write)
                        
                        print(f"成功下載: {csv_filename} -> {local_file_path}")
                        downloaded_files.append(local_file_path)
                        
                except ftplib.error_perm as e:
                    print(f"FTP 權限錯誤 {target_path}: {e}")
                except Exception as e:
                    print(f"處理目錄 {hour_dir} 時發生錯誤: {e}")
                    
            if not matching_dirs:
                print(f"找不到符合 {hour}* 格式的目錄")
                
        except ftplib.error_perm as e:
            print(f"FTP 權限錯誤 {ftp_date_path}: {e}")
        except Exception as e:
            print(f"存取日期目錄時發生錯誤: {e}")
            
    except Exception as e:
        print(f"FTP 連接錯誤: {e}")
    finally:
        try:
            ftp.quit()
        except:
            pass
    
    return downloaded_files

def process_csv_for_lite(csv_file_path):
    """
    處理 R2MS Lite 的 CSV 檔案
    1. 用 Mode_Index 資料取代第一列
    2. 將原本的 Mode_Index 列改為全部是 '2'  
    3. 刪除標頭行
    
    Args:
        csv_file_path: CSV 檔案路徑
    """
    try:
        print(f"處理 CSV 檔案: {csv_file_path}")
        
        # 讀取 CSV 檔案
        df = pd.read_csv(csv_file_path, header=0)
        
        # 找到 Mode_Index 列
        if 'Mode_Index' not in df.columns:
            print("警告: CSV 檔案中找不到 Mode_Index 列")
            return
            
        # 找到 Mode_Index 列的位置
        mode_index_col_idx = df.columns.get_loc('Mode_Index')
        
        # 備份原始的 Mode_Index 列資料
        original_mode_index_data = df['Mode_Index'].copy()
        
        # 取得第一列的列名
        first_col_name = df.columns[0]
        
        # 用原始 Mode_Index 資料取代第一列
        df.iloc[:, 0] = original_mode_index_data
        
        # 將原本 Mode_Index 列的位置改為全部是 '2'
        df.iloc[:, mode_index_col_idx] = '2'
        
        # 在 DataFrame 最後加上一個空列，讓每行結尾都有逗點（兼容 csv2urf）
        df['empty_col'] = ''
        
        # 儲存處理後的 CSV (不含標頭)
        df.to_csv(csv_file_path, header=False, index=False)
        
        print(f"CSV 檔案處理完成: {csv_file_path}")
        print(f"  - 原始 Mode_Index 資料取代第一列 ({first_col_name})")
        print(f"  - 原本第 {mode_index_col_idx + 1} 列 (Mode_Index) 改為全部填入 '2'")
        print(f"  - 已刪除標頭行")
        print(f"  - 已在每行結尾加上逗點（兼容 csv2urf）")
        
    except Exception as e:
        print(f"處理 CSV 檔案時發生錯誤: {e}")
        import traceback
        traceback.print_exc()

def run_acquisition_lite(config):
    """執行 R2MS Lite 資料擷取流程"""
    try:
        print("開始執行 R2MS Lite 資料擷取...")
        
        # 取得配置
        root_path = config['data']['root']
        days_to_review = config['data'].get('days_to_review', 1)
        ftp_config = config['ftp']
        
        # 建立必要的目錄結構
        os.makedirs(root_path, exist_ok=True)
        
        downloaded_files = []
        
        # 處理指定天數的資料
        for day_offset in range(days_to_review):
            target_date = datetime.datetime.now() - datetime.timedelta(days=day_offset)
            
            # R2MS Lite 每小時產生檔案，檢查每個小時 (0-23)
            for hour in range(24):
                target_datetime = target_date.replace(hour=hour, minute=0, second=0, microsecond=0)
                
                # 下載 CSV 檔案
                csv_files = download_csv_from_ftp(ftp_config, target_datetime, root_path)
                
                # 處理下載的 CSV 檔案
                for csv_file in csv_files:
                    process_csv_for_lite(csv_file)
                    downloaded_files.append(csv_file)
        
        if not downloaded_files:
            print("沒有下載到任何檔案")
            return []
            
        # 使用 csv2urf 轉換檔案
        print("開始轉換 CSV 到 URF 格式...")
        
        # 取得 geo 檔案路徑 (使用動態路徑)
        line_name = config['data'].get('line_name', 'E1')
        geo_file = os.path.join(root_path, f'GEO_{line_name}.urf')
        if not os.path.exists(geo_file):
            print(f"錯誤: 找不到 geo 檔案 {geo_file}")
            return []
            
        urf_files = []
        
        # 處理每個下載的 CSV 檔案
        for csv_file in downloaded_files:
            try:
                # 建立輸出路徑
                csv_dir = os.path.dirname(csv_file)
                csv_basename = os.path.splitext(os.path.basename(csv_file))[0]
                
                # 從 CSV 檔案路徑中提取日期信息
                # csv_file 格式: root_path/csv/Recorder/YYMMDD/HHmm00_E.csv
                csv_parent_dir = os.path.basename(os.path.dirname(csv_file))  # 取得 YYMMDD
                csv_hour = csv_basename[:2]  # 從檔名中取得 HH
                
                urf_output_dir = os.path.join(root_path, "urf")
                output_png_base_dir = os.path.join(root_path, "Output_png")
                png_output_dir = os.path.join(output_png_base_dir, f"{csv_parent_dir}{csv_hour}")
                
                os.makedirs(urf_output_dir, exist_ok=True)
                os.makedirs(png_output_dir, exist_ok=True)
                
                urf_filename = f"{csv_parent_dir}{csv_hour}_m_{line_name}.urf"
                
                # 使用 csv2urf 轉換
                plot_wave = hour in config['data'].get('plot_time', [0, 4, 9, 10, 14, 20])
                result = utils.csv2urf(
                    csv_files=[csv_file],
                    one_intelligent_ERT_survey_geo_file=geo_file,
                    output_urf_path=urf_output_dir,
                    output_urf_file_name=urf_filename,
                    output_png_path=png_output_dir,
                    plot_wave=plot_wave,
                    png_file_first_name=csv_basename,
                    amplitude_estimate_start_position=config['data'].get('amplitude_estimate_start_position', 2),
                    amplitude_estimate_range=config['data'].get('amplitude_estimate_range', 4)
                )
                
                if result == 0:
                    urf_file_path = os.path.join(urf_output_dir, urf_filename)
                    urf_files.append(urf_file_path)
                    print(f"成功轉換: {csv_file} -> {urf_file_path}")
                else:
                    print(f"轉換失敗: {csv_file}")
                    
            except Exception as e:
                print(f"處理檔案 {csv_file} 時發生錯誤: {e}")
                import traceback
                traceback.print_exc()
                
        return urf_files
        
    except Exception as e:
        print(f"執行 R2MS Lite 資料擷取流程時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    """主程式"""
    print("開始執行 TAIP ERT Pipeline (R2MS Lite)...")
    args = parse_args()
    print(f"使用配置文件: {args.config}")
    
    # 讀取 YAML 設定檔，使用 UTF-8 編碼
    try:
        print(f"正在讀取配置文件...")
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("配置文件讀取成功")
    except Exception as e:
        print(f"錯誤：讀取設定檔 {args.config} 失敗: {str(e)}")
        return 1
    
    # 使用命令列參數覆蓋配置
    if args.station:
        config["data"]["station_name"] = args.station
    
    if args.line:
        config["data"]["line_name"] = args.line
    
    if args.days:
        config["data"]["days_to_review"] = args.days
    
    if args.root:
        config["data"]["root"] = args.root
    
    # 檢查必要配置
    if not config.get("data", {}).get("root"):
        root_dir = os.path.join("D:", "R2MSDATA", 
                               f"{config['data']['station_name']}_{config['data']['line_name']}")
        config["data"]["root"] = root_dir
        print(f"警告：未指定根目錄，使用預設值: {root_dir}")
    
    print(f"使用資料目錄: {config['data']['root']}")
    
    # 處理排程參數
    schedule_enabled = args.schedule
    if not schedule_enabled and config.get("data", {}).get("schedule", False):
        schedule_enabled = config["data"]["schedule"]
    
    # 處理排程時間
    schedule_time = args.time
    if schedule_time == "23:45" and "time" in config.get("data", {}):
        schedule_time = config["data"]["time"]
    
    # 設置 schedule 模式
    if schedule_enabled:
        try:
            schedule_time_parts = schedule_time.split(":")
            hour = int(schedule_time_parts[0])
            minute = int(schedule_time_parts[1])
            
            while True:
                now = datetime.datetime.now()
                next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                # 如果當前時間已經過了排定時間，則安排在明天
                if now.hour > hour or (now.hour == hour and now.minute >= minute):
                    next_run += datetime.timedelta(days=1)
                
                # 計算等待時間
                sleep_time = (next_run - now).total_seconds()
                print(f"排程模式：下次執行時間 {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"等待 {sleep_time:.0f} 秒...")
                
                time.sleep(max(0, sleep_time))
                print(f"\n開始執行排程任務，時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 執行資料擷取
                run_acquisition_lite(config)
        except KeyboardInterrupt:
            print("\n排程已被使用者中斷")
            return 0
        except Exception as e:
            print(f"排程執行錯誤: {str(e)}")
            return 1
    else:
        # 非排程模式，直接執行一次
        print("開始執行 R2MS Lite 資料擷取...")
        urf_files = run_acquisition_lite(config)
        
        # 檢查是否有處理的檔案
        if urf_files:
            print(f"成功處理 {len(urf_files)} 個 URF 檔案")
            return 0
        else:
            print("沒有新的檔案需要處理")
            return 0

if __name__ == "__main__":
    sys.exit(main()) 