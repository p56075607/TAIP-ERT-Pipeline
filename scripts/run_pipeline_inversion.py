#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TAIP ERT Pipeline - ERT 反演與視覺化腳本

功能：
1. 載入 URF 檔案，轉換為 ohm 格式
2. 進行數據過濾和 PyGIMLi 反演
3. 產生反演結果圖、cross-plot、misfit 直方圖
4. (可選) 執行交集反演，比較前四個時間點的資料

使用方式:
    python run_pipeline_inversion.py --config configs/site.yaml [--urf path/to/urf_file.urf] [--intersection]

注意：
    本腳本已整合了原有的 run_pipeline_intersection_inv.py 的功能，因此 run_pipeline_intersection_inv.py 可以移除
    交集反演的結果會儲存在 output/xzv_inters 和 output/profile_inters 目錄
"""

import os
import sys
from pathlib import Path

# 添加 src 目錄到路徑 - 移到最前面確保所有模組都能正確導入
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import yaml
import glob
import re
import numpy as np
import pygimli as pg
import pygimli.physics.ert as ert
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import multiprocessing
import platform
import signal
import importlib
import time

# 現在導入自定義模組
from src.taip_ert_pipeline import pipeline
from src.taip_ert_pipeline import visualization
from src.taip_ert_pipeline import result_viewer

# 全局變量
latest_result_info = None
viewer_process = None
update_queue = None

def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(description="TAIP ERT Pipeline - ERT 反演處理")
    parser.add_argument("--config", "-c", required=True, help="設定檔路徑")
    parser.add_argument("--urf", "-u", help="指定單一 URF 檔案進行反演，若未指定則使用 test_dir 中的所有 URF 檔案")
    parser.add_argument("--test-dir", "-t", help="測試目錄，存放 URF 檔案的目錄")
    parser.add_argument("--repeat", "-r", type=int, help="反演重複次數，會覆蓋設定檔中的值")
    parser.add_argument("--output", "-o", help="輸出目錄，會覆蓋設定檔中的值")
    
    # 修改 intersection 參數，設置為三態：None(默認，不覆蓋配置), True, False
    intersection_group = parser.add_mutually_exclusive_group()
    intersection_group.add_argument("--intersection", "-i", action="store_true", dest="intersection", 
                                    help="執行完反演後，執行交集反演")
    intersection_group.add_argument("--no-intersection", action="store_false", dest="intersection", 
                                    help="執行完反演後，不執行交集反演")
    parser.set_defaults(intersection=None)  # 設置為 None 表示使用配置文件的值
    
    parser.add_argument("--skip-inversion", "-s", action="store_true", help="跳過常規反演，只執行交集反演")
    
    # 添加與結果查看器相關的參數
    parser.add_argument("--viewer", action="store_true", 
                        help="啟用結果查看器，反演過程中可視化結果")
    parser.add_argument("--no-viewer", action="store_false", dest="viewer",
                        help="禁用結果查看器")
    parser.add_argument("--refresh-interval", type=int, default=5000,
                        help="查看器刷新間隔（毫秒），默認5000ms")
    parser.set_defaults(viewer=True)  # 默認啟用查看器
    
    return parser.parse_args()

def extract_datetime(urf_file):
    """從 URF 檔案名稱中提取日期時間資訊作為排序依據
    
    假設檔案名稱格式為 "YYMMDDHH_其他資訊.urf"，提取 YYMMDDHH 部分作為排序依據
    """
    basename = os.path.basename(urf_file).split('.')[0]
    # 假設時間格式在檔名的開頭部分，使用 split 提取首個部分
    time_part = basename.split('_')[0]
    
    # 確認提取的部分是否符合 YYMMDDHH 格式
    if re.match(r'^[0-9]{8}$', time_part):
        return time_part
    return basename  # 若格式不符，則返回完整檔名作為備用排序依據

def main():
    """主程式"""
    global viewer_process, update_queue
    
    args = parse_args()
    
    # 讀取 YAML 設定檔
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # 打印配置文件中的 inversion 部分，用於調試
        print("\n===== 配置文件內容 =====")
        if "inversion" in config:
            print("Inversion 配置:")
            for key, value in config["inversion"].items():
                print(f"  - {key}: {value}")
        else:
            print("警告: 配置文件中沒有 inversion 部分")
        print("=======================\n")
            
    except Exception as e:
        print(f"錯誤：讀取設定檔 {args.config} 失敗: {str(e)}")
        return 1
    
    # 使用命令列參數覆蓋配置
    if args.repeat:
        config["inversion"]["repeat_times"] = args.repeat
    
    if args.output:
        config["output"]["output_dir"] = args.output
    
    # 從配置文件中讀取 intersection 參數
    # 注意: 這裡特別檢查 inversion.intersection 字段
    do_intersection = False
    if "inversion" in config and "intersection" in config["inversion"]:
        do_intersection = config["inversion"]["intersection"]
        print(f"從配置文件讀取到 intersection 設置: {do_intersection}")
    
    # 命令行參數覆蓋配置文件
    if args.intersection is not None:  # 檢查是否明確指定了參數
        do_intersection = args.intersection
        print(f"從命令行參數覆蓋 intersection 設置: {do_intersection}")
    
    print(f"交集反演: {'啟用' if do_intersection else '停用'} (來源: {'命令行參數' if args.intersection is not None else '配置文件'})")
    
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
    
    # 根據檔案名稱中的日期時間資訊進行排序（由舊到新）
    if len(urf_files) > 1:
        urf_files.sort(key=extract_datetime)
        print("已根據檔案名稱中的日期時間資訊排序 URF 檔案（由舊到新）")
    
    # 決定是否跳過常規反演
    if args.skip_inversion:
        print("根據命令列參數，跳過常規反演階段")
        
        # 如果跳過反演但需要進行交集反演
        if do_intersection:
            print("直接進行交集反演處理")
            return process_all_intersections(config)
        else:
            print("未指定任何操作，退出程序")
            return 0
    
    # 如果沒有 URF 檔案且不需要跳過反演，則退出
    if not urf_files:
        print("沒有 URF 檔案可進行反演")
        return 1
    
    # 處理每個時段的完整流程
    # 先確保配置正確
    prepare_config(config)
    
    # 獲取輸出目錄
    output_dir = config["output"].get("output_dir", "output")
    
    # 檢查輸出目錄是否存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 創建xzv和交集反演目錄
    xzv_dir = os.path.join(output_dir, "xzv")
    os.makedirs(xzv_dir, exist_ok=True)
    
    # 如果需要交集反演，則初始化交集反演目錄
    if do_intersection:
        xzv_inters_dir = os.path.join(output_dir, "xzv_inters")
        profile_inters_dir = os.path.join(output_dir, "profile_inters")
        os.makedirs(xzv_inters_dir, exist_ok=True)
        os.makedirs(profile_inters_dir, exist_ok=True)
        print(f"已創建交集反演結果目錄: {xzv_inters_dir} 和 {profile_inters_dir}")
    
    # 配置結果查看器
    viewer_process = None
    if args.viewer:
        # 使用多進程啟動查看器，避免阻塞主流程
        if platform.system() == "Windows":
            # Windows 環境下的多進程啟動方式不同
            multiprocessing.freeze_support()
        
        # 創建並啟動查看器進程
        from multiprocessing import Queue
        
        # 創建共享隊列
        update_queue = Queue()
        
        viewer_process = multiprocessing.Process(
            target=start_viewer_process,
            args=(output_dir, args.refresh_interval)
        )
        viewer_process.daemon = True  # 設置為守護進程，主進程結束時自動終止
        viewer_process.start()
        print(f"已啟動結果查看器，輸出目錄: {output_dir}, 刷新間隔: {args.refresh_interval}ms")
    
    # 處理每個URF檔案
    process_combined_workflow(config, urf_files, do_intersection)
    
    print("\n===== 所有URF檔案處理完成 =====")
    
    # 結果查看器已經在後台運行，可以在這裡增加等待用戶關閉的提示
    if viewer_process and viewer_process.is_alive():
        # 最後通知結果查看器刷新
        notify_viewer("所有處理完成")
        
        print("反演結果查看器正在運行中，請手動關閉查看器窗口以退出程序")
        try:
            # 等待用戶按下 Ctrl+C 終止程序
            viewer_process.join()
        except KeyboardInterrupt:
            print("程序已終止")
    
    return 0

def prepare_config(config):
    """準備配置，確保所有必要的設定都正確"""
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

def process_combined_workflow(config, urf_files, do_intersection):
    """
    執行集成的反演和交集反演流程 - 完全按照時間順序處理
    
    參數:
        config: 配置字典
        urf_files: 所有URF檔案的列表
        do_intersection: 是否進行交集反演
    
    返回:
        int: 0表示成功，非0表示失敗
    """
    output_dir = config["output"].get("output_dir", "output")
    xzv_dir = os.path.join(output_dir, "xzv")
    
    # 如果需要進行交集反演，初始化目錄
    if do_intersection:
        print("交集反演已啟用 - 將在每個時段反演後檢查是否需要執行交集反演")
        xzv_inters_dir = os.path.join(output_dir, "xzv_inters")
        os.makedirs(xzv_inters_dir, exist_ok=True)
    else:
        print("交集反演已禁用 - 僅執行常規反演")
    
    # 追蹤最新處理的時段資訊
    latest_folder_name = None
    latest_repeat_or_intersection = None
    
    # 處理每個URF文件，按順序完成常規反演和交集反演
    for idx, urf_file in enumerate(urf_files):
        # 提取時間戳資訊
        urf_basename = os.path.basename(urf_file).split('.')[0]
        time_part = urf_basename.split('_')[0]  # 獲取 "YYMMDDHH" 部分
        folder_name = f"{time_part}_m_T1"  # 資料夾名稱
        
        print(f"\n===== 處理 [{idx+1}/{len(urf_files)}] {os.path.basename(urf_file)} =====")
        print(f"時間點: {time_part}, 資料夾: {folder_name}")
        
        # 更新最新時段資訊
        latest_folder_name = folder_name
        
        # 1. 檢查常規反演是否需要執行
        regular_xzv_file = os.path.join(xzv_dir, f"{time_part}.xzv")
        need_regular_inversion = not os.path.exists(regular_xzv_file)
        
        # 2. 執行常規反演（如果需要）
        if need_regular_inversion:
            print(f"1. 執行常規反演：{urf_basename}")
            result = run_inversion_single(config, urf_file)
            
            if result != 0:
                print(f"常規反演失敗，跳過後續處理")
                continue
            print(f"常規反演完成：{urf_basename}")
            
            # 常規反演成功，更新最新的反演結果為最新的 repeat 資料夾
            repeat_folders = [d for d in os.listdir(os.path.join(output_dir, folder_name)) 
                            if os.path.isdir(os.path.join(output_dir, folder_name, d)) and d.startswith("repeat_")]
            if repeat_folders:
                repeat_folders.sort(key=lambda x: int(x.split("_")[1]), reverse=True)
                latest_repeat_or_intersection = os.path.join(folder_name, repeat_folders[0])
                print(f"更新最新反演結果: {latest_repeat_or_intersection}")
        else:
            print(f"跳過常規反演：{urf_basename}（檔案已存在：{regular_xzv_file}）")
            
            # 檢查是否有已完成的反演結果
            if os.path.exists(os.path.join(output_dir, folder_name)):
                repeat_folders = [d for d in os.listdir(os.path.join(output_dir, folder_name)) 
                                if os.path.isdir(os.path.join(output_dir, folder_name, d)) and d.startswith("repeat_")]
                if repeat_folders:
                    repeat_folders.sort(key=lambda x: int(x.split("_")[1]), reverse=True)
                    latest_repeat_or_intersection = os.path.join(folder_name, repeat_folders[0])
                    print(f"已存在反演結果: {latest_repeat_or_intersection}")
        
        # 3. 如果不需要交集反演，直接跳到下一個檔案
        if not do_intersection:
            print(f"交集反演已禁用，跳過交集反演檢查")
            continue
        
        # 4. 檢查交集反演是否需要執行
        xzv_inters_dir = os.path.join(output_dir, "xzv_inters")
        intersection_xzv_file = os.path.join(xzv_inters_dir, f"{time_part}.xzv")
        need_intersection_inversion = not os.path.exists(intersection_xzv_file)
        
        if not need_intersection_inversion:
            print(f"跳過交集反演：{urf_basename}（檔案已存在：{intersection_xzv_file}）")
            
            # 檢查是否有已完成的交集反演結果
            if os.path.exists(os.path.join(output_dir, folder_name, "intersection")):
                latest_repeat_or_intersection = os.path.join(folder_name, "intersection")
                print(f"已存在交集反演結果: {latest_repeat_or_intersection}")
            
            continue
        
        print(f"2. 檢查是否可以進行交集反演：{urf_basename}")
        
        # 5. 檢查當前時段是否完成第四次反演
        repeat4_complete = check_repeat_4_complete(output_dir, folder_name)
        if not repeat4_complete:
            print(f"當前時段 {folder_name} 未完成第四次反演，無法進行交集反演")
            print(f"檢查路徑: {os.path.join(output_dir, folder_name, 'repeat_4', 'ERTManager')}")
            continue
            
        print(f"當前時段 {folder_name} 已完成第四次反演，繼續檢查")
            
        # 6. 檢查當前時段在所有時段中的位置
        sorted_folders = list_sorted_output_folders(output_dir)
        print(f"找到 {len(sorted_folders)} 個時間序列資料夾")
        
        try:
            current_index = sorted_folders.index(folder_name)
            print(f"當前資料夾 {folder_name} 在排序列表中的位置: {current_index+1}/{len(sorted_folders)}")
        except ValueError:
            print(f"無法在排序列表中找到當前資料夾 {folder_name}，無法進行交集反演")
            continue
            
        # 7. 檢查是否有足夠的先前時段
        if current_index < 3:
            print(f"當前時段 {folder_name} 前面沒有足夠的時段（需要至少3個），無法進行交集反演")
            print(f"當前索引: {current_index}，需要索引至少為 3 才有足夠的先前時段")
            continue
            
        # 8. 獲取前三個時段
        previous_folders = [sorted_folders[current_index-3], sorted_folders[current_index-2], sorted_folders[current_index-1]]
        print(f"前三個時段: {previous_folders}")
        
        # 9. 檢查前三個時段是否都完成了第四次反演
        all_previous_complete = True
        incomplete_folders = []
        for prev_folder in previous_folders:
            if not check_repeat_4_complete(output_dir, prev_folder):
                print(f"先前時段 {prev_folder} 未完成第四次反演，無法進行交集反演")
                all_previous_complete = False
                incomplete_folders.append(prev_folder)
        
        if not all_previous_complete:
            print(f"以下時段未完成第四次反演: {', '.join(incomplete_folders)}")
            continue
        
        print(f"所有前序時段已完成第四次反演，可以進行交集反演")
            
        # 10. 執行單一時段的交集反演
        print(f"執行交集反演，當前時段: {folder_name}, 前三個時段: {previous_folders}")
        result = run_intersection_inversion_single(config, folder_name, previous_folders)
        
        if result != 0:
            print(f"交集反演失敗，繼續處理下一個時段")
        else:
            print(f"交集反演成功完成: {folder_name}")
            # 交集反演成功，更新最新的反演結果為交集反演資料夾
            latest_repeat_or_intersection = os.path.join(folder_name, "intersection")
            print(f"更新最新反演結果: {latest_repeat_or_intersection}")
    
    # 在主程序中存儲最新的時段和反演結果資訊，用於後續使用
    if latest_folder_name and latest_repeat_or_intersection:
        # 生成全局變量用於保存最新結果資訊
        global latest_result_info
        latest_result_info = {
            "folder_name": latest_folder_name,
            "result_path": latest_repeat_or_intersection
        }
        print(f"完成處理，最新反演結果: {latest_repeat_or_intersection}")
        # 通知結果查看器顯示最新結果
        notify_viewer(f"處理完成，最新結果: {latest_repeat_or_intersection}")
    
    return 0

def run_inversion_single(config, urf_file):
    """
    執行單一URF檔案的反演
    
    參數:
        config: 配置字典
        urf_file: URF檔案路徑
        
    返回:
        int: 0表示成功，非0表示失敗
    """
    try:
        # 反演開始前，先通知查看器預加載當前最新的反演結果
        basename = os.path.basename(urf_file)
        output_dir = config["output"].get("output_dir", "output")
        
        # 預加載通知 - 強制刷新最新圖片
        notify_viewer(f"準備開始反演，請更新圖片: {basename}")
        
        # 等待一些時間，確保查看器有時間刷新
        time.sleep(1)
        
        print(f"開始執行反演: {basename}")
        
        # 執行 pipeline 中的反演功能
        if pipeline.run_inversion_only(config, [urf_file]):
            print(f"反演成功完成: {basename}")
            
            # 從 basename 提取時間戳部分
            time_part = basename.split('_')[0]  # 獲取 "YYMMDDHH" 部分
            folder_name = f"{time_part}_m_T1"  # 資料夾名稱
            
            # 尋找最新的 repeat 資料夾
            if os.path.exists(os.path.join(output_dir, folder_name)):
                repeat_folders = [d for d in os.listdir(os.path.join(output_dir, folder_name)) 
                                if os.path.isdir(os.path.join(output_dir, folder_name, d)) and d.startswith("repeat_")]
                if repeat_folders:
                    repeat_folders.sort(key=lambda x: int(x.split("_")[1]), reverse=True)
                    latest_repeat = repeat_folders[0]
                    latest_result_path = os.path.join(folder_name, latest_repeat)
                    
                    # 檢查資訊文件是否存在
                    info_file = os.path.join(output_dir, latest_result_path, "ERTManager/inv_info.txt")
                    if os.path.exists(info_file):
                        print(f"找到反演結果資訊檔: {info_file}")
                    else:
                        print(f"警告: 找不到反演結果資訊檔: {info_file}")
                    
                    # 通知結果查看器更新，並提供最新結果路徑
                    notify_viewer(f"反演完成，請完整更新: {basename}")
                    time.sleep(0.5)  # 確保文件已經完全寫入
                    notify_viewer(f"最新結果: {latest_result_path}")
                else:
                    print(f"警告: 在 {folder_name} 中找不到 repeat 資料夾")
                    notify_viewer(f"反演完成，請完整更新: {basename}")
            else:
                print(f"警告: 找不到資料夾 {folder_name}")
                notify_viewer(f"反演完成，請完整更新: {basename}")
            
            return 0
        else:
            print(f"反演過程出現錯誤: {os.path.basename(urf_file)}")
            return 1
            
    except Exception as e:
        print(f"錯誤：執行反演流程失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

def process_all_intersections(config):
    """
    處理所有可能的交集反演
    
    參數:
        config: 配置字典
        
    返回:
        int: 0表示成功，非0表示失敗
    """
    # 執行交集反演處理邏輯
    print("\n======= 執行所有可能的交集反演 =======")
    
    # 檢查是否有足夠的時間點進行交集反演
    output_dir = config["output"].get("output_dir", "output")
    if not os.path.exists(output_dir):
        print(f"錯誤：輸出目錄 {output_dir} 不存在，無法進行交集反演")
        return 1
        
    # 獲取並排序所有時間資料夾
    sorted_folders = list_sorted_output_folders(output_dir)
    
    if len(sorted_folders) < 4:
        print(f"錯誤：在 {output_dir} 中找到的時間點資料夾數量少於4個 ({len(sorted_folders)})，無法進行交集反演")
        return 1
        
    # 執行交集反演
    return run_intersection_inversion(config)

def run_intersection_inversion(config, current_folder=None, previous_folders=None):
    """
    執行交集反演流程
    
    參數:
        config: 配置字典
        current_folder: 當前資料夾名稱（如指定，則僅處理該時段）
        previous_folders: 前三個資料夾名稱的列表（如指定current_folder但未指定此參數，則自動選擇）
        
    返回:
        int: 0表示成功，非0表示失敗
    """
    # 取得輸出目錄
    output_dir = config["output"].get("output_dir", "output")
    
    # 檢查並創建交集反演結果目錄
    xzv_inters_dir = os.path.join(output_dir, "xzv_inters")
    profile_inters_dir = os.path.join(output_dir, "profile_inters")
    
    os.makedirs(xzv_inters_dir, exist_ok=True)
    os.makedirs(profile_inters_dir, exist_ok=True)
    
    # 單一時段模式
    if current_folder is not None:
        print(f"\n===== 執行單一時段交集反演: {current_folder} =====")
        
        # 如果沒有提供前三個時段，則自動選擇
        if previous_folders is None:
            # 獲取並排序所有時間資料夾
            sorted_folders = list_sorted_output_folders(output_dir)
            
            try:
                current_index = sorted_folders.index(current_folder)
                
                # 檢查是否有足夠的前序時段
                if current_index < 3:
                    print(f"錯誤：當前時段 {current_folder} 前面沒有足夠的時段（需要至少3個），無法進行交集反演")
                    return 1
                
                # 選擇前三個時段
                previous_folders = [
                    sorted_folders[current_index-3], 
                    sorted_folders[current_index-2], 
                    sorted_folders[current_index-1]
                ]
                print(f"自動選擇前三個時段: {', '.join(previous_folders)}")
            except ValueError:
                print(f"錯誤：在排序列表中找不到當前時段 {current_folder}，無法進行交集反演")
                return 1
        
        # 執行單一時段的交集反演
        return _process_single_intersection(config, current_folder, previous_folders)
    
    # 批量時段模式
    print(f"\n===== 執行批量交集反演 =====")
    print(f"交集反演結果將儲存至: {xzv_inters_dir} 和 {profile_inters_dir}")
    
    # 獲取並排序所有時間資料夾
    sorted_folders = list_sorted_output_folders(output_dir)
    
    print(f"找到 {len(sorted_folders)} 個時間序列資料夾")
    for i, folder in enumerate(sorted_folders[:10]):  # 只顯示前10個
        print(f"{i+1}. {folder}")
    
    if len(sorted_folders) > 10:
        print(f"... 還有 {len(sorted_folders)-10} 個資料夾")
    
    # 從第四個時間點開始處理
    if len(sorted_folders) < 4:
        print("錯誤：資料夾數量少於4個，無法繼續")
        return 1
    
    # 過濾需要處理的時間點資料夾
    filtered_folders = []
    skipped_folders = []
    
    # 只考慮從第四個時間點開始的資料夾
    for i in range(3, len(sorted_folders)):
        current_folder = sorted_folders[i]
        
        # 提取當前資料夾的時間戳
        time_part = current_folder.split('_')[0]
        
        # 檢查對應的交集反演結果是否已存在
        xzv_inters_file = os.path.join(xzv_inters_dir, f"{time_part}.xzv")
        
        # 如果 XZV 檔案存在，則跳過該時間點
        if os.path.exists(xzv_inters_file):
            print(f"跳過處理 {current_folder}，對應的交集反演 XZV 檔案已存在: {xzv_inters_file}")
            skipped_folders.append(current_folder)
            continue
            
        # 檢查是否已完成第四次反演和前三個時間點
        if not check_repeat_4_complete(output_dir, current_folder):
            print(f"警告：{current_folder} 未完成第四次反演，跳過處理")
            continue
            
        # 檢查前三個時間點是否都完成第四次反演
        previous_folders = [sorted_folders[i-3], sorted_folders[i-2], sorted_folders[i-1]]
        all_previous_complete = True
        for folder in previous_folders:
            if not check_repeat_4_complete(output_dir, folder):
                print(f"警告：{folder} 未完成第四次反演，跳過處理")
                all_previous_complete = False
                break
                
        if not all_previous_complete:
            continue
        
        # 如果通過所有檢查，則加入處理列表
        filtered_folders.append((i, current_folder, previous_folders))
    
    # 更新要處理的資料夾列表
    if skipped_folders:
        print(f"已跳過 {len(skipped_folders)} 個已處理的時間點: {', '.join(skipped_folders)}")
    
    if not filtered_folders:
        print(f"所有時間點都已處理完成，無需執行交集反演")
        return 0
    
    print(f"將對 {len(filtered_folders)} 個時間點進行交集反演")
    
    # 迭代處理已過濾的時間點
    processed_count = 0
    folder_count = len(filtered_folders)
    
    for idx, (i, current_folder, previous_folders) in enumerate(filtered_folders):
        print(f"\n===== 處理第 {idx+1}/{folder_count} 個時間點（整體進度 {i+1}/{len(sorted_folders)}）：{current_folder} =====")
        print(f"前三個時間點：{previous_folders[0]}, {previous_folders[1]}, {previous_folders[2]}")
        
        # 執行單一時段的交集反演
        result = _process_single_intersection(config, current_folder, previous_folders)
        
        if result == 0:
            print(f"成功完成 {current_folder} 的交集反演")
            processed_count += 1
        else:
            print(f"處理 {current_folder} 的交集反演失敗")
    
    if processed_count > 0:
        print(f"交集反演已完成，處理了 {processed_count} 個時間點")
        return 0
    else:
        print("沒有任何時間點需要進行交集反演，或所有時間點都已被跳過")
        return 0

def _process_single_intersection(config, current_folder, previous_folders):
    """
    處理單一時段的交集反演
    
    參數:
        config: 配置字典
        current_folder: 當前資料夾名稱
        previous_folders: 前三個資料夾名稱的列表
        
    返回:
        int: 0表示成功，非0表示失敗
    """
    # 取得輸出目錄
    output_dir = config["output"].get("output_dir", "output")
    base_path = output_dir
    
    # 提取當前資料夾的時間戳
    time_part = current_folder.split('_')[0]
    
    print(f"\n----- 開始交集反演處理: {current_folder} -----")
    print(f"時間點: {time_part}, 輸出目錄: {output_dir}")
    
    # 交集反演開始前，先通知查看器更新當前最新圖片
    notify_viewer(f"準備開始交集反演，請更新圖片: {current_folder}")
    # 等待一些時間，確保查看器有時間刷新
    time.sleep(1)
    
    # 檢查並創建交集反演結果目錄
    xzv_inters_dir = os.path.join(output_dir, "xzv_inters")
    profile_inters_dir = os.path.join(output_dir, "profile_inters")
    
    os.makedirs(xzv_inters_dir, exist_ok=True)
    os.makedirs(profile_inters_dir, exist_ok=True)
    
    # 載入配置參數
    colormap_file = config["output"].get("colormap_file")
    if colormap_file:
        print(f"使用色彩圖檔案: {colormap_file}")
    else:
        print("警告: 未指定色彩圖檔案")
        
    title_verbose = config["output"].get("title_verbose", False)
    print(f"詳細標題模式: {title_verbose}")
    
    # 反演參數
    if "inversion" in config:
        lam = config["inversion"].get("lam", 1000)
        z_weight = config["inversion"].get("z_weight", 1)
        max_iter = config["inversion"].get("max_iter", 6)
        resistivity_limits = config["inversion"].get("limits", [1, 10000])
        print(f"反演參數: lam={lam}, z_weight={z_weight}, max_iter={max_iter}, 電阻率範圍={resistivity_limits}")
    else:
        lam = 1000
        z_weight = 1
        max_iter = 6
        resistivity_limits = [1, 10000]
        print("使用默認反演參數")
    
    # 初始化視覺化器
    visualization_config = {
        "root_dir": base_path,
        "colormap_file": colormap_file,
        "title_verbose": title_verbose
    }
    
    visualizer = visualization.ERTVisualizer(visualization_config)
    print("已初始化視覺化器")
    
    # 3. 載入當前和前三個時間點的資料
    print(f"載入當前時段資料: {current_folder}")
    current_data = load_inversion_data(base_path, current_folder)
    
    print(f"載入前三個時段資料: {previous_folders}")
    previous_data_list = []
    
    for folder in previous_folders:
        print(f"處理時段: {folder}")
        data = load_inversion_data(base_path, folder)
        if data is None:
            print(f"警告: 無法載入時段 {folder} 的資料")
        else:
            print(f"成功載入時段 {folder} 的資料")
        previous_data_list.append(data)
    
    if current_data is None:
        print(f"錯誤：無法載入當前時段 {current_folder} 的資料，跳過處理")
        return 1
    
    if None in previous_data_list:
        print(f"錯誤：無法載入某些前序資料，跳過處理")
        for i, folder in enumerate(previous_folders):
            if previous_data_list[i] is None:
                print(f"  - 無法載入: {folder}")
        return 1
    
    print("所有必要資料已成功載入")
    
    # 將當前資料和前三個時間點的資料合併為一個列表
    all_data_list = previous_data_list + [current_data]
    
    # 4. 檢查資料一致性 - 資料點數量差異不應超過25%
    print("檢查資料一致性...")
    if not check_data_consistency(all_data_list):
        print(f"警告：資料點數量差異超過閾值，跳過處理 {current_folder}")
        return 1
    
    print("資料一致性檢查通過")
    
    # 5. 找出共同的電極組合
    print("尋找共同的電極組合...")
    common_quadruples = find_common_quadruples(all_data_list)
    
    if len(common_quadruples) == 0:
        print(f"錯誤：找不到共同的電極組合，跳過處理")
        return 1
    
    print(f"找到 {len(common_quadruples)} 個共同的電極組合")
    
    # 6. 過濾資料，只保留共同的電極組合
    print("過濾資料，只保留共同的電極組合...")
    filtered_data_list = filter_by_common_quadruples(all_data_list, common_quadruples)
    
    # 7. 獲取當前時間點的網格和電阻率模型
    print(f"載入當前時段的網格和電阻率模型: {current_folder}")
    mesh, initial_model = load_mesh_and_model(base_path, current_folder)
    
    if mesh is None:
        print(f"錯誤：無法載入當前時段 {current_folder} 的網格，跳過處理")
        return 1
    
    print(f"成功載入網格，節點數: {mesh.nodeCount()}, 單元數: {mesh.cellCount()}")
    if initial_model is not None:
        print(f"成功載入初始電阻率模型，長度: {len(initial_model)}")
    else:
        print("未找到初始電阻率模型，將使用默認模型")
    
    # 8. 使用當前時間點過濾後的資料進行反演
    current_filtered_data = filtered_data_list[-1]  # 當前時間點的資料是列表中的最後一個
    
    print(f"開始反演，資料點數：{len(current_filtered_data['rhoa'])}")
    
    # 創建 ERT 管理器
    try:
        mgr = ert.ERTManager(current_filtered_data)
        print("成功創建 ERT 管理器")
    except Exception as e:
        print(f"錯誤：無法創建 ERT 管理器: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 執行反演，使用初始模型（如果可用）
    try:
        if initial_model is not None:
            print(f"使用上一次反演結果作為初始模型")
            model = mgr.invert(current_filtered_data, mesh=mesh,
                             startModel=initial_model,
                             lam=lam, zWeight=z_weight,
                             maxIter=max_iter,
                             limits=resistivity_limits,
                             verbose=True)
        else:
            print(f"使用默認初始模型")
            model = mgr.invert(current_filtered_data, mesh=mesh,
                             lam=lam, zWeight=z_weight,
                             maxIter=max_iter,
                             limits=resistivity_limits,
                             verbose=True)
        
        # 計算反演結果的誤差統計
        rrms = mgr.inv.relrms()
        chi2 = mgr.inv.chi2()
        print(f"反演完成：rrms={rrms:.2f}%, chi²={chi2:.3f}")
        
        # 完成後通知查看器刷新
        notify_viewer(f"交集反演完成: {current_folder}, rrms={rrms:.2f}%, chi²={chi2:.3f}")
        
        # 在通知完成後，添加明確的結果路徑通知
        intersection_result_path = os.path.join(current_folder, "intersection")
        time.sleep(0.5)  # 確保文件已經完全寫入
        notify_viewer(f"最新結果: {intersection_result_path}")
    except Exception as e:
        print(f"錯誤：反演過程失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 9. 創建儲存路徑 - 修改為新結構
    # 先創建臨時的目錄結構
    temp_intersection_path = os.path.join(base_path, current_folder, "intersection")
    temp_intersection_ert_path = os.path.join(temp_intersection_path, "ERTManager")
    
    try:
        os.makedirs(temp_intersection_ert_path, exist_ok=True)
        print(f"創建儲存目錄: {temp_intersection_ert_path}")
    except Exception as e:
        print(f"錯誤：無法創建儲存目錄: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 10. 儲存反演結果
    try:
        # 使用 saveResult 會自動創建 ERTManager 目錄並保存多個文件
        print("儲存反演結果...")
        path, fig, ax = mgr.saveResult(temp_intersection_path) # 會自動創建 ERTManager 文件夾
        plt.close(fig)
        print(f"成功儲存反演結果到: {path}")
        
        # 額外保存資料文件
        mgr.data.save(os.path.join(temp_intersection_ert_path, f"{current_folder}_inv.ohm"))
        print(f"已保存資料文件: {os.path.join(temp_intersection_ert_path, f'{current_folder}_inv.ohm')}")
        
        # 儲存模型響應
        pg.utils.saveResult(os.path.join(temp_intersection_ert_path, 'model_response.txt'),
                         data=mgr.inv.response, mode='w')
        print(f"已保存模型響應: {os.path.join(temp_intersection_ert_path, 'model_response.txt')}")
        
        # 保存其他反演信息
        rrmsHistory = mgr.inv.rrmsHistory
        chi2History = mgr.inv.chi2History
        info_file = export_inversion_info(mgr, temp_intersection_path, lam, rrmsHistory, chi2History)
        print(f"已保存反演信息到: {info_file}")
    except Exception as e:
        print(f"錯誤：儲存反演結果失敗: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 11. 使用標準的視覺化工具生成圖表
    print(f"使用標準視覺化模組生成圖表")
    try:
        # 先讓 ERTVisualizer 加載反演結果
        print("加載反演結果...")
        results = visualizer.load_inversion_results(temp_intersection_path)
        
        # 如果視覺化器沒有找到 rrms 和 chi2，從 mgr 中取值
        if not results:
            print("無法加載反演結果")
        elif results.get("rrms") is None or results.get("chi2") is None:
            print("視覺化器未能從文件中讀取到 rrms 和 chi2 值，使用反演結果中的值")
            
            # 將值寫入到相應的文件中
            info_file = export_inversion_info(mgr, temp_intersection_path, lam, rrmsHistory, chi2History)
            
            # 重新加載反演結果
            results = visualizer.load_inversion_results(temp_intersection_path)
            if not results:
                print("警告：即使在重新導出反演信息後仍無法加載反演結果")
        else:
            print(f"成功加載反演結果，rrms={results.get('rrms')}, chi2={results.get('chi2')}")
        
        # 設置額外的可視化參數
        plot_kwargs = {
            "title_verbose": title_verbose,  # 使用配置中的設置
            'cMin': 10,                   # 最小電阻率值
            'cMax': 1000,                 # 最大電阻率值
            'label': 'Resistivity $\\Omega$m', 
        }
        
        # 從配置文件中獲取 XZV 文件路徑
        xzv_file = config.get("inversion", {}).get("xzv_file", None)
        if xzv_file and os.path.exists(xzv_file):
            print(f"使用 XZV 文件: {xzv_file}")
            # 使用 XZV 文件生成更精確的等值線圖
            try:
                visualizer.plot_all(temp_intersection_path, current_folder, xzv_file, **plot_kwargs)
                print("成功使用 XZV 文件生成視覺化結果")
            except Exception as e:
                print(f"使用 XZV 文件生成視覺化結果失敗: {str(e)}，嘗試不使用 XZV 文件")
                visualizer.plot_all(temp_intersection_path, current_folder, **plot_kwargs)
        else:
            # 不使用 XZV 文件
            print("未指定 XZV 文件或文件不存在，不使用 XZV 文件生成視覺化結果")
            visualizer.plot_all(temp_intersection_path, current_folder, **plot_kwargs)
        
        # 複製生成的檔案到新的資料夾結構
        # 1. 複製 XZV 檔案到 xzv_inters 目錄
        src_xzv = os.path.join(temp_intersection_path, f"{time_part}.xzv")
        if os.path.exists(src_xzv):
            try:
                shutil.copy2(src_xzv, xzv_inters_dir)
                print(f"已複製 XZV 檔案到: {xzv_inters_dir}/{time_part}.xzv")
            except Exception as e:
                print(f"複製 XZV 檔案失敗: {str(e)}")
        else:
            print(f"警告: 找不到 XZV 檔案: {src_xzv}")
        
        # 2. 複製 PNG 檔案到 profile_inters 目錄
        src_profile = os.path.join(temp_intersection_path, "inverted_contour_xzv.png")
        if os.path.exists(src_profile):
            try:
                dst_profile = os.path.join(profile_inters_dir, f"{time_part}.png")
                shutil.copy2(src_profile, dst_profile)
                print(f"已複製 profile 檔案到: {dst_profile}")
            except Exception as e:
                print(f"複製 profile 檔案失敗: {str(e)}")
        else:
            print(f"警告: 找不到 profile 檔案: {src_profile}")
        
        print(f"成功生成所有視覺化結果")
        return 0
    except Exception as e:
        print(f"視覺化生成過程中出現錯誤: {str(e)}")
        import traceback
        traceback.print_exc()  # 打印詳細錯誤堆疊
        return 1

def extract_date_from_filename(folder_name):
    """
    從資料夾名稱中提取日期（格式為25XXXXXX_m_T1）
    返回可用於排序的字符串
    """
    # 使用正則表達式提取日期部分
    match = re.match(r'(\d{8})_m_T1', folder_name)
    if match:
        return match.group(1)
    # 如果無法匹配，返回原始檔名作為後備
    return folder_name

def list_sorted_output_folders(base_path):
    """
    獲取並按時間排序的output資料夾
    """
    # 確保路徑存在
    if not os.path.exists(base_path):
        print(f"錯誤：路徑 {base_path} 不存在")
        return []
    
    # 獲取所有符合25XXXXXX_m_T1模式的資料夾
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and re.match(r'\d{8}_m_T1', f)]
    
    # 按日期排序
    folders = sorted(folders, key=extract_date_from_filename)
    
    return folders

def check_repeat_4_complete(base_path, folder_name):
    """
    檢查指定資料夾是否已完成第四次反演
    """
    ohm_file = os.path.join(base_path, folder_name, "repeat_4", "ERTManager", f"{folder_name}_inv.ohm")
    mesh_file = os.path.join(base_path, folder_name, "repeat_4", "ERTManager", "mesh.bms")
    
    return os.path.exists(ohm_file) and os.path.exists(mesh_file)

def load_inversion_data(base_path, folder_name):
    """
    載入指定資料夾的第四次反演數據
    """
    ohm_file = os.path.join(base_path, folder_name, "repeat_4", "ERTManager", f"{folder_name}_inv.ohm")
    if not os.path.exists(ohm_file):
        print(f"警告：找不到檔案 {ohm_file}")
        return None
    
    try:
        data = pg.load(ohm_file)
        return data
    except Exception as e:
        print(f"錯誤：無法載入 {ohm_file}. 錯誤: {str(e)}")
        return None

def load_mesh_and_model(base_path, folder_name):
    """
    載入指定資料夾的網格文件和電阻率模型
    返回網格和電阻率模型
    """
    mesh_file = os.path.join(base_path, folder_name, "repeat_4", "ERTManager", "mesh.bms")
    model_file = os.path.join(base_path, folder_name, "repeat_4", "ERTManager", "resistivity.vector")
    
    mesh = None
    model = None
    
    # 嘗試載入網格
    if not os.path.exists(mesh_file):
        print(f"警告：找不到網格檔案 {mesh_file}")
    else:
        try:
            mesh = pg.load(mesh_file)
            print(f"成功載入網格: {mesh_file}")
        except Exception as e:
            print(f"錯誤：無法載入網格檔案 {mesh_file}. 錯誤: {str(e)}")
    
    # 嘗試載入電阻率模型
    if not os.path.exists(model_file):
        print(f"警告：找不到電阻率模型檔案 {model_file}")
    else:
        try:
            model = pg.load(model_file)
            print(f"成功載入電阻率模型: {model_file}")
        except Exception as e:
            print(f"錯誤：無法載入電阻率模型檔案 {model_file}. 錯誤: {str(e)}")
    
    return mesh, model

def find_common_quadruples(data_list):
    """
    找出所有數據集中共同的電極組合(a,b,m,n)
    """
    # 為每個數據集創建電極組合集合
    sets_of_quadruples = []
    
    for data in data_list:
        if data is not None:
            quadruples = set(zip(data['a'], data['b'], data['m'], data['n']))
            sets_of_quadruples.append(quadruples)
    
    # 找出共同的電極組合
    if len(sets_of_quadruples) == 0:
        return set()
    
    common_quadruples = set.intersection(*sets_of_quadruples)
    print(f"找到共同的電極組合：{len(common_quadruples)}個")
    
    return common_quadruples

def filter_by_common_quadruples(data_list, common_quadruples):
    """
    根據共同的電極組合過濾每個數據集
    """
    filtered_data_list = []
    
    for data in data_list:
        if data is None:
            filtered_data_list.append(None)
            continue
        
        # 複製數據以避免修改原始數據
        filtered_data = data.copy()
        
        # 獲取當前數據集的電極組合
        quadruples = list(zip(data['a'], data['b'], data['m'], data['n']))
        
        # 標記不在共同電極組合中的數據
        remove_indices = [quadruple not in common_quadruples for quadruple in quadruples]
        
        # 移除不共同的數據
        if any(remove_indices):
            filtered_data.remove(remove_indices)
            print(f"從數據集中移除了 {sum(remove_indices)} 個不共同的數據點，剩餘 {len(filtered_data['rhoa'])} 個")
        
        filtered_data_list.append(filtered_data)
    
    return filtered_data_list

def export_inversion_info(mgr, save_path, lam, rrmsHistory, chi2History):
    """
    匯出反演資訊，按照標準格式
    
    參數:
        mgr: ERT管理器
        save_path: 儲存路徑
        lam: 正則化參數
        rrmsHistory: 反演過程中的相對誤差歷史記錄
        chi2History: 反演過程中的卡方值歷史記錄
    """
    # 確保ERTManager目錄存在
    mgr_dir = os.path.join(save_path, 'ERTManager')
    if not os.path.exists(mgr_dir):
        os.makedirs(mgr_dir)
    
    info_file = os.path.join(mgr_dir, 'inv_info.txt')
    with open(info_file, 'w') as f:
        f.write('## Final result ##\n')
        f.write('rrms:{}\n'.format(mgr.inv.relrms()))
        f.write('chi2:{}\n'.format(mgr.inv.chi2()))
        
        f.write('## Inversion parameters ##\n')
        f.write('use lam:{}\n'.format(lam))
        
        f.write('## Iteration ##\n')
        f.write('Iter.  rrms  chi2\n')
        for iter in range(len(rrmsHistory)):
            f.write('{:.0f},{:.2f},{:.2f}\n'.format(iter, rrmsHistory[iter], chi2History[iter]))
    
    print(f"反演信息已保存至: {info_file}")
    return info_file

def check_data_consistency(data_list):
    """
    檢查各數據集之間的資料點數量是否一致
    如果任何一個數據集的資料點數量少於其他數據集的75%（差異超過25%），則視為離群
    
    參數:
        data_list: 包含多個數據集的列表
    
    返回:
        boolean: 資料集是否通過一致性檢查
    """
    # 獲取每個數據集的資料點數量
    data_counts = []
    
    for i, data in enumerate(data_list):
        if data is None:
            print(f"數據集 {i+1} 為空，無法進行一致性檢查")
            return False
        
        count = len(data['rhoa'])
        data_counts.append(count)
        print(f"數據集 {i+1} 資料點數量: {count}")
    
    # 計算中位資料點數量
    median_count = np.median(data_counts)
    
    # 檢查每個數據集的資料點數量是否至少為中位值的75%
    for i, count in enumerate(data_counts):
        percentage = (count / median_count) * 100
        if percentage < 75:
            print(f"警告: 數據集 {i+1} 的資料點數量 ({count}) 僅為中位數量 ({median_count}) 的 {percentage:.2f}%，小於閾值 75%")
            return False
    
    print("所有數據集的資料點數量差異在允許範圍內（不超過25%）")
    return True

def run_intersection_inversion_single(config, current_folder, previous_folders):
    """
    執行單一時段的交集反演 (已棄用，請改用 run_intersection_inversion 函數)
    
    參數:
        config: 配置字典
        current_folder: 當前資料夾名稱
        previous_folders: 前三個資料夾名稱的列表
        
    返回:
        int: 0表示成功，非0表示失敗
    """
    print("警告：run_intersection_inversion_single 已棄用，請改用 run_intersection_inversion 函數")
    return run_intersection_inversion(config, current_folder, previous_folders)

def start_viewer_process(output_dir, refresh_interval):
    """啟動查看器進程的函數"""
    import sys
    from PyQt5.QtWidgets import QApplication
    import queue
    
    # 創建一個共享隊列，用於接收更新通知
    global update_queue
    update_queue = queue.Queue()
    
    app = QApplication(sys.argv)
    viewer = result_viewer.ResultMonitor(
        result_dir=output_dir, 
        output_dir=output_dir,
        refresh_interval=refresh_interval
    )
    viewer.show()
    
    # 創建一個定時器，定期檢查是否有更新通知
    from PyQt5.QtCore import QTimer
    
    # 定義檢查更新的函數
    def check_updates():
        try:
            # 非阻塞方式檢查隊列
            while not update_queue.empty():
                # 取得更新消息
                update_msg = update_queue.get_nowait()
                print(f"結果查看器收到更新通知: {update_msg}")
                
                # 根據不同的消息類型進行不同的處理
                if "最新結果:" in update_msg:
                    # 處理指定結果路徑的消息
                    try:
                        # 從消息中提取路徑信息
                        result_path = update_msg.split("最新結果:")[1].strip()
                        if result_path and os.path.exists(os.path.join(output_dir, result_path)):
                            print(f"切換到指定結果目錄: {result_path}")
                            # 直接設置查看器的最新反演結果資料夾
                            viewer.latest_repeat_folder = result_path
                            
                            # 更新 info_file 路徑 - 解決資訊未更新的問題
                            new_info_path = os.path.join(output_dir, result_path, "ERTManager/inv_info.txt")
                            if os.path.exists(new_info_path):
                                viewer.info_file = new_info_path
                                print(f"更新資訊檔案路徑: {new_info_path}")
                            else:
                                print(f"警告: 找不到資訊檔案: {new_info_path}")
                            
                            # 強制更新路徑標籤
                            viewer.update_path_label()
                            # 重新加載圖片和資訊
                            viewer.load_result_images()
                            viewer.load_inv_info()
                    except Exception as e:
                        print(f"解析結果路徑時出錯: {str(e)}")
                
                elif "準備開始反演" in update_msg:
                    # 反演開始前的預加載處理
                    print("反演準備開始，強制刷新當前圖片")
                    # 先執行一次 find_latest_result 確保顯示最新狀態
                    find_latest_result(True)
                
                elif "請完整更新" in update_msg:
                    # 反演完成後的更新
                    print("反演已完成，執行完整更新")
                    # 強制重新檢查最新結果，並清空圖片緩存
                    find_latest_result(True)
                
                else:
                    # 一般更新通知，執行標準刷新
                    viewer.force_refresh()
        except Exception as e:
            print(f"檢查更新時出錯: {str(e)}")
            import traceback
            traceback.print_exc()  # 印出詳細錯誤堆疊
    
    # 創建並啟動定時器
    update_timer = QTimer()
    update_timer.timeout.connect(check_updates)
    update_timer.start(1000)  # 每秒檢查一次
    
    # 初始檢查該輸出目錄下的最新結果
    def find_latest_result(force_reload=False):
        """
        尋找輸出目錄中最新的反演結果
        
        參數:
            force_reload: 是否強制重新載入圖片，忽略緩存
        """
        try:
            # 1. 列出並排序所有時段資料夾
            sorted_folders = list_sorted_output_folders(output_dir)
            if not sorted_folders:
                print("沒有找到任何時段資料夾")
                return
                
            # 2. 獲取最新的時段資料夾
            latest_folder = sorted_folders[-1]
            print(f"找到最新時段資料夾: {latest_folder}")
            
            # 3. 檢查是否有交集反演結果
            if os.path.exists(os.path.join(output_dir, latest_folder, "intersection")):
                latest_result = os.path.join(latest_folder, "intersection")
                print(f"找到交集反演結果: {latest_result}")
            else:
                # 4. 尋找最新的 repeat 資料夾
                repeat_folders = [d for d in os.listdir(os.path.join(output_dir, latest_folder)) 
                                if os.path.isdir(os.path.join(output_dir, latest_folder, d)) and d.startswith("repeat_")]
                if repeat_folders:
                    repeat_folders.sort(key=lambda x: int(x.split("_")[1]), reverse=True)
                    latest_result = os.path.join(latest_folder, repeat_folders[0])
                    print(f"找到最新反演結果: {latest_result}")
                else:
                    print(f"在最新時段 {latest_folder} 中沒有找到任何反演結果")
                    return
            
            # 5. 設置查看器顯示該結果
            need_reload = force_reload or viewer.latest_repeat_folder != latest_result
            
            if need_reload:
                print(f"需要重新載入結果: {latest_result}")
                viewer.latest_repeat_folder = latest_result
                
                # 更新 info_file 路徑 - 解決資訊未更新的問題
                new_info_path = os.path.join(output_dir, latest_result, "ERTManager/inv_info.txt")
                if os.path.exists(new_info_path):
                    viewer.info_file = new_info_path
                    print(f"更新資訊檔案路徑: {new_info_path}")
                else:
                    print(f"警告: 找不到資訊檔案: {new_info_path}")
                
                # 如果強制重載或路徑變更，執行完整加載
                # 清除現有圖像緩存
                viewer._clear_layout(viewer.grid_layout)
                
                # 完整重載所有內容
                viewer.load_result_images()
                viewer.load_inv_info()
                viewer.update_path_label()
                print(f"已設置查看器顯示最新結果並重新載入所有圖片: {latest_result}")
            else:
                print(f"已經顯示最新結果: {latest_result}")
                # 僅刷新顯示，不重新加載圖片
                viewer.force_refresh()
                
        except Exception as e:
            print(f"尋找最新結果時出錯: {str(e)}")
            import traceback
            traceback.print_exc()  # 印出詳細錯誤堆疊
    
    # 啟動一個延遲定時器，在界面加載後尋找最新結果
    initial_timer = QTimer()
    initial_timer.setSingleShot(True)
    initial_timer.timeout.connect(lambda: find_latest_result(True))
    initial_timer.start(2000)  # 2秒後執行一次
    
    sys.exit(app.exec_())

def notify_viewer(message):
    """通知結果查看器更新"""
    global update_queue
    if update_queue is not None:
        try:
            # 檢查是否是反演完成的消息
            if "反演完成" in message:
                # 為反演完成消息添加時間戳，確保每次通知都被視為不同的消息
                message = f"{message} (時間: {time.time()})"
                print(f"發送反演完成通知: {message}")
                
                # 等待一小段時間，確保文件已經寫入完成
                time.sleep(1)
            
            update_queue.put(message)
            print(f"已通知結果查看器更新: {message}")
        except Exception as e:
            print(f"通知結果查看器失敗: {str(e)}")
            import traceback
            traceback.print_exc()  # 印出詳細錯誤堆疊

if __name__ == "__main__":
    sys.exit(main()) 