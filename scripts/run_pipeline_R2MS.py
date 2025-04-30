#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TAIP ERT Pipeline - R2MS 資料擷取與前處理腳本

功能：
1. 從 FTP 伺服器下載 R2MS 原始 CSV/ZIP 資料
2. 解壓縮並轉換為 URF 格式
3. 產生基本 QC 和波形圖

使用方式:
    python run_pipeline_R2MS.py --config configs/site.yaml
"""

import os
import sys
import argparse
import yaml
import datetime
import time
from pathlib import Path

# 添加 src 目錄到路徑
sys.path.append(str(Path(__file__).parent.parent))

from src.taip_ert_pipeline import pipeline

def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(description="TAIP ERT Pipeline - R2MS 資料擷取與前處理")
    parser.add_argument("--config", "-c", required=True, help="設定檔路徑")
    parser.add_argument("--station", "-s", help="站點名稱，會覆蓋設定檔中的值")
    parser.add_argument("--line", "-l", help="測線名稱，會覆蓋設定檔中的值")
    parser.add_argument("--days", "-d", type=int, help="處理的天數，會覆蓋設定檔中的值")
    parser.add_argument("--root", "-r", help="根目錄路徑，會覆蓋設定檔中的值")
    parser.add_argument("--all-files", "-a", action="store_true", help="下載所有檔案，不只是 *_E.zip")
    parser.add_argument("--schedule", "-S", action="store_true", help="啟用排程，每天特定時間執行")
    parser.add_argument("--time", "-t", default="23:45", help="排程時間 (HH:MM 格式)，預設 23:45")
    
    return parser.parse_args()

def main():
    """主程式"""
    print("開始執行 TAIP ERT Pipeline...")
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
    
    # 更新 all_files 參數處理（明確檢查是否提供此參數）
    if args.all_files:
        config["data"]["download_all_files"] = True
    
    # 處理排程參數
    schedule_enabled = args.schedule
    if not schedule_enabled and config.get("data", {}).get("schedule", False):
        schedule_enabled = config["data"]["schedule"]
    
    # 處理排程時間
    schedule_time = args.time
    if schedule_time == "23:45" and "time" in config.get("data", {}):
        schedule_time = config["data"]["time"]
    
    # 檢查必要配置
    if not config.get("data", {}).get("root"):
        root_dir = os.path.join("D:", "R2MSDATA", 
                               f"{config['data']['station_name']}_{config['data']['line_name']}")
        config["data"]["root"] = root_dir
        print(f"警告：未指定根目錄，使用預設值: {root_dir}")
    
    print(f"使用資料目錄: {config['data']['root']}")
    
    # 設置 schedule 模式，使用修改後的變數
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
                run_acquisition(config)
        except KeyboardInterrupt:
            print("\n排程已被使用者中斷")
            return 0
        except Exception as e:
            print(f"排程執行錯誤: {str(e)}")
            return 1
    else:
        # 非排程模式，直接執行一次
        print("開始執行資料擷取...")
        return run_acquisition(config)

def run_acquisition(config):
    """執行資料擷取流程"""
    try:
        print("正在執行 pipeline 中的資料擷取功能...")
        # 執行 pipeline 中的資料擷取功能
        urf_files = pipeline.run_acquisition_only(config)
        
        # 檢查是否有處理的檔案
        if urf_files:
            print(f"成功處理 {len(urf_files)} 個 URF 檔案")
            return 0
        else:
            print("沒有新的檔案需要處理")
            return 0
            
    except Exception as e:
        print(f"錯誤：執行資料擷取流程失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 