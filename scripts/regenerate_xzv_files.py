#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TAIP ERT Pipeline - 重新生成 XZV 檔案腳本

功能：
遍歷所有時間資料夾，對每個 intersection 目錄重新生成 .xzv 檔案
並將生成的檔案複製到指定的 xzv_inters 目錄

使用方式:
    python regenerate_xzv_files.py
"""

import os
import sys
import glob
import shutil
import argparse
from pathlib import Path

# 添加 src 目錄到路徑
sys.path.append(str(Path(__file__).parent.parent))

# 設置 matplotlib 為非互動模式，避免彈出視窗
import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端
import matplotlib.pyplot as plt
plt.ioff()  # 關閉互動模式

from src.taip_ert_pipeline.visualization import ERTVisualizer

def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(description="TAIP ERT Pipeline - 重新生成 XZV 檔案")
    parser.add_argument("--output-root", "-o", default="E:/R2MSDATA/TAIP_T1_test/output", 
                        help="輸出根目錄路徑")
    parser.add_argument("--colormap-file", "-c", default="D:/R2MSDATA/TAIP_T1/EIclm.mat", 
                        help="色彩圖檔案路徑")
    parser.add_argument("--xzv-output-dir", "-x", default="E:/R2MSDATA/TAIP_T1_test/output/xzv_inters", 
                        help="XZV 檔案輸出目錄")
    
    return parser.parse_args()

def find_time_folders(output_root):
    """
    尋找所有時間資料夾
    
    參數:
        output_root: 輸出根目錄
        
    返回:
        time_folders: 時間資料夾列表
    """
    # 尋找符合時間格式的資料夾 (例如：25051900_m_T1)
    pattern = os.path.join(output_root, "*_m_T*")
    folders = glob.glob(pattern)
    
    # 過濾出確實是目錄的項目
    time_folders = [folder for folder in folders if os.path.isdir(folder)]
    
    print(f"找到 {len(time_folders)} 個時間資料夾:")
    for folder in time_folders:
        print(f"  - {os.path.basename(folder)}")
    
    return time_folders

def process_intersection_folder(intersection_path, visualizer, xzv_output_dir, time_folder_name):
    """
    處理單個 intersection 資料夾
    
    參數:
        intersection_path: intersection 資料夾路徑
        visualizer: ERTVisualizer 實例
        xzv_output_dir: XZV 檔案輸出目錄
        time_folder_name: 時間資料夾名稱
        
    返回:
        success: 是否成功
    """
    try:
        print(f"  處理 intersection 資料夾: {intersection_path}")
        
        # 檢查 intersection 資料夾是否存在
        if not os.path.exists(intersection_path):
            print(f"    警告：intersection 資料夾不存在，跳過")
            return False
        
        # 從時間資料夾名稱中提取時間戳 (例如：從 "25051900_m_T1" 提取 "25051900")
        time_stamp = time_folder_name.split('_')[0]
        
        # 載入反演結果
        results = visualizer.load_inversion_results(intersection_path)
        if not results:
            print(f"    錯誤：無法載入反演結果")
            return False
        
        print(f"    成功載入反演結果")
        
        # 重新生成等值線圖和 XZV 檔案
        success = visualizer.plot_inverted_contour_xzv(
            intersection_path, 
            None,  # 不使用現有的 xzv 文件，讓方法自己創建
            time_stamp
        )
        
        if not success:
            print(f"    錯誤：無法重新生成等值線圖和 XZV 檔案")
            return False
        
        print(f"    成功重新生成等值線圖和 XZV 檔案")
        
        # 尋找生成的 XZV 檔案
        generated_xzv_pattern = os.path.join(intersection_path, "*.xzv")
        generated_xzv_files = glob.glob(generated_xzv_pattern)
        
        if not generated_xzv_files:
            print(f"    警告：找不到生成的 XZV 檔案")
            return False
        
        # 確保輸出目錄存在
        if not os.path.exists(xzv_output_dir):
            os.makedirs(xzv_output_dir)
            print(f"    創建輸出目錄: {xzv_output_dir}")
        
        # 複製所有生成的 XZV 檔案到輸出目錄
        for xzv_file in generated_xzv_files:
            target_file = os.path.join(xzv_output_dir, os.path.basename(xzv_file))
            shutil.copy2(xzv_file, target_file)
            print(f"    複製 XZV 檔案: {os.path.basename(xzv_file)} -> {xzv_output_dir}")
        
        return True
        
    except Exception as e:
        print(f"    錯誤：處理 intersection 資料夾失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主程式"""
    args = parse_args()
    
    output_root = args.output_root
    colormap_file = args.colormap_file
    xzv_output_dir = args.xzv_output_dir
    
    # 檢查輸出根目錄是否存在
    if not os.path.exists(output_root):
        print(f"錯誤：輸出根目錄不存在: {output_root}")
        return 1
    
    # 檢查色彩圖檔案是否存在
    if colormap_file and not os.path.exists(colormap_file):
        print(f"警告：色彩圖檔案不存在: {colormap_file}")
        colormap_file = None
    
    print(f"開始重新生成 XZV 檔案...")
    print(f"輸出根目錄: {output_root}")
    print(f"色彩圖檔案: {colormap_file}")
    print(f"XZV 輸出目錄: {xzv_output_dir}")
    print("-" * 60)
    
    try:
        # 尋找所有時間資料夾
        time_folders = find_time_folders(output_root)
        
        if not time_folders:
            print("沒有找到任何時間資料夾")
            return 1
        
        # 初始化視覺化器
        visualization_config = {
            "root_dir": output_root,
            "colormap_file": colormap_file
        }
        
        visualizer = ERTVisualizer(visualization_config)
        
        # 處理每個時間資料夾
        processed_count = 0
        success_count = 0
        
        for time_folder in time_folders:
            time_folder_name = os.path.basename(time_folder)
            print(f"\n處理時間資料夾: {time_folder_name}")
            
            # 構建 intersection 資料夾路徑
            intersection_path = os.path.join(time_folder, "intersection")
            
            # 處理 intersection 資料夾
            success = process_intersection_folder(
                intersection_path, visualizer, xzv_output_dir, time_folder_name
            )
            
            processed_count += 1
            if success:
                success_count += 1
        
        print("-" * 60)
        print(f"處理完成！")
        print(f"總計處理: {processed_count} 個時間資料夾")
        print(f"成功處理: {success_count} 個")
        print(f"失敗處理: {processed_count - success_count} 個")
        
        if success_count == processed_count:
            print("所有 XZV 檔案重新生成成功！")
            return 0
        else:
            print("部分 XZV 檔案重新生成失敗，請查看日誌。")
            return 1
            
    except Exception as e:
        print(f"錯誤：執行重新生成過程失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 