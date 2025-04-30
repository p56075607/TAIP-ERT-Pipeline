#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TAIP ERT Pipeline - ERT 反演與視覺化腳本

功能：
1. 載入 URF 檔案，轉換為 ohm 格式
2. 進行數據過濾和 PyGIMLi 反演
3. 產生反演結果圖、cross-plot、misfit 直方圖

使用方式:
    python run_pipeline_inversion.py --config configs/site.yaml [--urf path/to/urf_file.urf]
"""

import os
import sys
import argparse
import yaml
import glob
from pathlib import Path

# 添加 src 目錄到路徑
sys.path.append(str(Path(__file__).parent.parent))

from src.taip_ert_pipeline import pipeline

def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(description="TAIP ERT Pipeline - ERT 反演處理")
    parser.add_argument("--config", "-c", required=True, help="設定檔路徑")
    parser.add_argument("--urf", "-u", help="指定單一 URF 檔案進行反演，若未指定則使用 test_dir 中的所有 URF 檔案")
    parser.add_argument("--test-dir", "-t", help="測試目錄，存放 URF 檔案的目錄")
    parser.add_argument("--repeat", "-r", type=int, help="反演重複次數，會覆蓋設定檔中的值")
    parser.add_argument("--output", "-o", help="輸出目錄，會覆蓋設定檔中的值")
    
    return parser.parse_args()

def main():
    """主程式"""
    args = parse_args()
    
    # 讀取 YAML 設定檔
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"錯誤：讀取設定檔 {args.config} 失敗: {str(e)}")
        return 1
    
    # 使用命令列參數覆蓋配置
    if args.repeat:
        config["inversion"]["repeat_times"] = args.repeat
    
    if args.output:
        config["output"]["output_dir"] = args.output
    
    if args.test_dir:
        # 覆蓋測試目錄
        test_dir = args.test_dir
    else:
        # 從配置取得測試目錄，或使用默認值
        test_dir = config.get("inversion", {}).get("test_dir", None)
        if not test_dir:
            test_dir = os.path.join(config["data"]["root"] + "_test", "urf")
    
    # 檢查必要配置
    if not config.get("inversion", {}).get("root_dir"):
        root_dir = config.get("data", {}).get("root", "")
        if root_dir:
            config["inversion"]["root_dir"] = root_dir + "_test"
            print(f"警告：未指定反演根目錄，使用值: {config['inversion']['root_dir']}")
        else:
            print("錯誤：未指定反演根目錄")
            return 1
    
    # 設置 URF 檔案
    urf_files = []
    if args.urf:
        # 指定單一 URF 檔案
        if os.path.exists(args.urf):
            urf_files = [args.urf]
        else:
            print(f"錯誤：指定的 URF 檔案不存在: {args.urf}")
            return 1
    else:
        # 使用目錄中的所有 URF 檔案
        if os.path.exists(test_dir):
            urf_files = glob.glob(os.path.join(test_dir, "*.urf"))
            if not urf_files:
                print(f"警告：在目錄 {test_dir} 中未找到 URF 檔案")
        else:
            print(f"錯誤：測試目錄不存在: {test_dir}")
            return 1
    
    # 執行反演
    if urf_files:
        print(f"將對 {len(urf_files)} 個 URF 檔案進行反演")
        return run_inversion(config, urf_files)
    else:
        print("沒有 URF 檔案可進行反演")
        return 1

def run_inversion(config, urf_files):
    """執行反演流程"""
    try:
        # 確保反演配置中有完整的配置信息
        if "inversion" in config and "output" in config:
            # 將 output 配置中的 colormap_file 複製到 inversion 配置中
            if "colormap_file" in config["output"] and "colormap_file" not in config["inversion"]:
                colormap_file = config["output"].get("colormap_file")
                if colormap_file and os.path.exists(colormap_file):
                    print(f"使用 output 配置中的色彩圖檔案: {colormap_file}")
                    config["inversion"]["colormap_file"] = colormap_file
                    
            # 檢查 colormap_file 是否存在
            if "colormap_file" in config["inversion"]:
                colormap_path = config["inversion"]["colormap_file"]
                if not os.path.exists(colormap_path):
                    print(f"警告：指定的色彩圖檔案不存在: {colormap_path}")
                    # 嘗試在 root_dir 尋找預設檔案
                    default_path = os.path.join(config["inversion"]["root_dir"], "EIclm.mat")
                    if os.path.exists(default_path):
                        print(f"使用預設色彩圖檔案: {default_path}")
                        config["inversion"]["colormap_file"] = default_path
        
        # 檢查是否有需要跳過的檔案
        output_dir = config["output"].get("output_dir", "output")
        xzv_dir = os.path.join(output_dir, "xzv")
        
        # 過濾需要處理的 URF 檔案
        filtered_urf_files = []
        skipped_files = []
        
        if os.path.exists(xzv_dir):
            for urf_file in urf_files:
                # 從 URF 檔案名稱中提取時間資訊 (YYMMDDHH)
                urf_basename = os.path.basename(urf_file).split('.')[0]
                # 假設 urf_basename 格式為 "YYMMDDHH_m_T1" 或類似格式，提取時間戳部分
                time_part = urf_basename.split('_')[0]  # 獲取 "YYMMDDHH" 部分
                
                # 檢查對應的 XZV 檔案是否存在，格式為 "YYMMDDHH.xzv"
                xzv_path = os.path.join(xzv_dir, f"{time_part}.xzv")
                
                # 如果 XZV 檔案存在，則跳過該 URF 檔案
                if os.path.exists(xzv_path):
                    print(f"跳過處理 URF 檔案 {urf_basename}，對應的 XZV 檔案已存在: {xzv_path}")
                    skipped_files.append(urf_basename)
                    continue
                
                # 如果沒有找到對應的 XZV 檔案，則加入處理列表
                filtered_urf_files.append(urf_file)
        else:
            # 如果 xzv_dir 不存在，則處理所有 URF 檔案
            filtered_urf_files = urf_files
            
        # 更新要處理的檔案列表
        if skipped_files:
            print(f"已跳過 {len(skipped_files)} 個已處理的檔案: {', '.join(skipped_files)}")
        
        if not filtered_urf_files:
            print(f"所有檔案都已處理完成，無需執行反演")
            return 0
        
        # 更新要處理的檔案列表
        urf_files = filtered_urf_files
        
        # 執行 pipeline 中的反演功能
        if pipeline.run_inversion_only(config, urf_files):
            print("反演成功完成")
            return 0
        else:
            print("反演過程出現錯誤")
            return 1
            
    except Exception as e:
        print(f"錯誤：執行反演流程失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 