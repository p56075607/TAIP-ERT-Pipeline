#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TAIP ERT Pipeline - 視覺化測試腳本

功能：
讀取指定路徑下的反演結果，並使用 plot_all 函數繪製所有視覺化圖表

使用方式:
    python test_visualization.py
"""

import os
import sys
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
    parser = argparse.ArgumentParser(description="TAIP ERT Pipeline - 視覺化測試")
    parser.add_argument("--result-path", "-r", default="D:/R2MSDATA/TAIP_T1_test/output/25040600_m_T1/repeat_1", 
                        help="反演結果路徑")
    parser.add_argument("--colormap-file", "-c", default="D:/R2MSDATA/TAIP_T1/EIclm.mat", 
                        help="色彩圖檔案路徑")
    parser.add_argument("--xzv-file", "-x", default="D:/R2MSDATA\TAIP_T1/25040708.xzv", 
                        help=".xzv 文件路徑")
    parser.add_argument("--file-name", "-f", default="25040600_m_T1", 
                        help="檔案名稱，用於圖表標題")
    
    return parser.parse_args()

def main():
    """主程式"""
    args = parse_args()
    
    result_path = args.result_path
    colormap_file = args.colormap_file
    xzv_file = args.xzv_file
    file_name = args.file_name
    
    # 檢查結果路徑是否存在
    if not os.path.exists(result_path):
        print(f"錯誤：結果路徑不存在: {result_path}")
        return 1
    
    # 檢查 ERTManager 目錄是否存在
    ertmanager_path = os.path.join(result_path, 'ERTManager')
    if not os.path.exists(ertmanager_path):
        print(f"錯誤：ERTManager 目錄不存在: {ertmanager_path}")
        return 1
    
    # 檢查色彩圖檔案是否存在
    if colormap_file and not os.path.exists(colormap_file):
        print(f"警告：色彩圖檔案不存在: {colormap_file}")
        colormap_file = None

    # 檢查 .xzv 文件是否存在
    if xzv_file and not os.path.exists(xzv_file):
        print(f"警告：.xzv 文件不存在: {xzv_file}")
        xzv_file = None
    
    print(f"開始繪製圖表...")
    print(f"結果路徑: {result_path}")
    print(f"色彩圖檔案: {colormap_file}")
    print(f"檔案名稱: {file_name}")
    
    try:
        # 初始化視覺化器
        visualization_config = {
            "root_dir": os.path.dirname(os.path.dirname(result_path)),
            "colormap_file": colormap_file
        }
        
        visualizer = ERTVisualizer(visualization_config)
        
        # 繪製所有圖表
        success = visualizer.plot_all(result_path, file_name, xzv_file)
        
        if success:
            print("所有圖表繪製成功！")
            return 0
        else:
            print("部分圖表繪製失敗，請查看日誌。")
            return 1
            
    except Exception as e:
        print(f"錯誤：執行視覺化過程失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 