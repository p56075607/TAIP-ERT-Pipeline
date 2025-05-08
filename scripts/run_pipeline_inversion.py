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
import argparse
import yaml
import glob
import re
from pathlib import Path
import numpy as np
import pygimli as pg
import pygimli.physics.ert as ert
import matplotlib.pyplot as plt
from datetime import datetime
import shutil

# 添加 src 目錄到路徑
sys.path.append(str(Path(__file__).parent.parent))

from src.taip_ert_pipeline import pipeline
from src.taip_ert_pipeline import visualization

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
    
    # 處理每個URF檔案
    process_combined_workflow(config, urf_files, do_intersection)
    
    print("\n===== 所有URF檔案處理完成 =====")
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
    
    # 處理每個URF文件，按順序完成常規反演和交集反演
    for idx, urf_file in enumerate(urf_files):
        # 提取時間戳資訊
        urf_basename = os.path.basename(urf_file).split('.')[0]
        time_part = urf_basename.split('_')[0]  # 獲取 "YYMMDDHH" 部分
        folder_name = f"{time_part}_m_T1"  # 資料夾名稱
        
        print(f"\n===== 處理 [{idx+1}/{len(urf_files)}] {os.path.basename(urf_file)} =====")
        print(f"時間點: {time_part}, 資料夾: {folder_name}")
        
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
        else:
            print(f"跳過常規反演：{urf_basename}（檔案已存在：{regular_xzv_file}）")
        
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
        # 執行 pipeline 中的反演功能
        if pipeline.run_inversion_only(config, [urf_file]):
            print(f"反演成功完成: {os.path.basename(urf_file)}")
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

def run_intersection_inversion_single(config, current_folder, previous_folders):
    """
    執行單一時段的交集反演
    
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

def run_intersection_inversion(config):
    """
    執行交集反演流程
    
    參數:
        config: 配置字典
    
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
    
    print(f"交集反演結果將儲存至: {xzv_inters_dir} 和 {profile_inters_dir}")
    
    # 從配置文件中讀取設置
    base_path = output_dir
    colormap_file = config["output"].get("colormap_file")
    title_verbose = config["output"].get("title_verbose", False)
    print(f"使用配置：base_path={base_path}, colormap_file={colormap_file}, title_verbose={title_verbose}")
    
    # 獲取並排序所有時間資料夾
    sorted_folders = list_sorted_output_folders(base_path)
    
    print(f"找到 {len(sorted_folders)} 個時間序列資料夾")
    for i, folder in enumerate(sorted_folders[:10]):  # 只顯示前10個
        print(f"{i+1}. {folder}")
    
    if len(sorted_folders) > 10:
        print(f"... 還有 {len(sorted_folders)-10} 個資料夾")
    
    # 從第四個時間點開始處理
    if len(sorted_folders) < 4:
        print("錯誤：資料夾數量少於4個，無法繼續")
        return 1
    
    # 過濾需要處理的時間點資料夾 - 參考 run_inversion 的邏輯
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
        if not check_repeat_4_complete(base_path, current_folder):
            print(f"警告：{current_folder} 未完成第四次反演，跳過處理")
            continue
            
        # 檢查前三個時間點是否都完成第四次反演
        previous_folders = [sorted_folders[i-3], sorted_folders[i-2], sorted_folders[i-1]]
        all_previous_complete = True
        for folder in previous_folders:
            if not check_repeat_4_complete(base_path, folder):
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
    
    # 設置反演參數
    if "inversion" in config:
        lam = config["inversion"].get("lam", 1000)
        z_weight = config["inversion"].get("z_weight", 1)
        max_iter = config["inversion"].get("max_iter", 6)
        resistivity_limits = config["inversion"].get("limits", [1, 10000])
    else:
        lam = 1000  # 正則化參數
        z_weight = 1  # 垂直權重
        max_iter = 6  # 最大迭代次數
        resistivity_limits = [1, 10000]  # 電阻率限制
    
    # 初始化視覺化器 - 使用配置文件中的色彩圖路徑
    visualization_config = {
        "root_dir": base_path,
        "colormap_file": colormap_file,
        "title_verbose": title_verbose
    }
    
    visualizer = visualization.ERTVisualizer(visualization_config)
    
    # 迭代處理已過濾的時間點
    processed_count = 0
    folder_count = len(filtered_folders)
    
    for idx, (i, current_folder, previous_folders) in enumerate(filtered_folders):
        print(f"\n===== 處理第 {idx+1}/{folder_count} 個時間點（整體進度 {i+1}/{len(sorted_folders)}）：{current_folder} =====")
        print(f"前三個時間點：{previous_folders[0]}, {previous_folders[1]}, {previous_folders[2]}")
        
        # 提取當前資料夾的時間戳
        time_part = current_folder.split('_')[0]
        
        # 3. 載入當前和前三個時間點的資料
        current_data = load_inversion_data(base_path, current_folder)
        previous_data_list = [load_inversion_data(base_path, folder) for folder in previous_folders]
        
        if current_data is None:
            print(f"錯誤：無法載入 {current_folder} 的資料，跳過處理")
            continue
        
        if None in previous_data_list:
            print(f"錯誤：無法載入某些前序資料，跳過處理")
            continue
        
        # 將當前資料和前三個時間點的資料合併為一個列表
        all_data_list = previous_data_list + [current_data]
        
        # 4. 檢查資料一致性 - 資料點數量差異不應超過25%
        if not check_data_consistency(all_data_list):
            print(f"警告：資料點數量差異超過閾值，跳過處理 {current_folder}")
            continue
        
        # 5. 找出共同的電極組合
        common_quadruples = find_common_quadruples(all_data_list)
        
        if len(common_quadruples) == 0:
            print(f"錯誤：找不到共同的電極組合，跳過處理")
            continue
        
        # 6. 過濾資料，只保留共同的電極組合
        filtered_data_list = filter_by_common_quadruples(all_data_list, common_quadruples)
        
        # 7. 獲取當前時間點的網格和電阻率模型
        mesh, initial_model = load_mesh_and_model(base_path, current_folder)
        
        if mesh is None:
            print(f"錯誤：無法載入 {current_folder} 的網格，跳過處理")
            continue
        
        # 8. 使用當前時間點過濾後的資料進行反演
        current_filtered_data = filtered_data_list[-1]  # 當前時間點的資料是列表中的最後一個
        
        print(f"開始反演，資料點數：{len(current_filtered_data['rhoa'])}")
        
        # 創建 ERT 管理器
        mgr = ert.ERTManager(current_filtered_data)
        
        # 執行反演，使用初始模型（如果可用）
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
        
        # 9. 創建儲存路徑 - 修改為新結構
        # 先創建臨時的目錄結構
        temp_intersection_path = os.path.join(base_path, current_folder, "intersection")
        temp_intersection_ert_path = os.path.join(temp_intersection_path, "ERTManager")
        os.makedirs(temp_intersection_ert_path, exist_ok=True)
        
        # 10. 儲存反演結果
        # 使用 saveResult 會自動創建 ERTManager 目錄並保存多個文件
        path, fig, ax = mgr.saveResult(temp_intersection_path) # 會自動創建 ERTManager 文件夾
        plt.close(fig)
        
        # 額外保存資料文件
        mgr.data.save(os.path.join(temp_intersection_ert_path, f"{current_folder}_inv.ohm"))
        
        # 儲存模型響應
        pg.utils.saveResult(os.path.join(temp_intersection_ert_path, 'model_response.txt'),
                          data=mgr.inv.response, mode='w')
        
        # 保存其他反演信息
        rrmsHistory = mgr.inv.rrmsHistory
        chi2History = mgr.inv.chi2History
        export_inversion_info(mgr, temp_intersection_path, lam, rrmsHistory, chi2History)
        
        # 11. 使用標準的視覺化工具生成圖表
        print(f"使用標準視覺化模組生成圖表")
        try:
            # 先讓 ERTVisualizer 加載反演結果
            results = visualizer.load_inversion_results(temp_intersection_path)
            
            # 如果視覺化器沒有找到 rrms 和 chi2，從 mgr 中取值
            if not results or results.get("rrms") is None or results.get("chi2") is None:
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
            src_xzv = os.path.join(temp_intersection_path, "ERTManager", f"{time_part}.xzv")
            if os.path.exists(src_xzv):
                try:
                    shutil.copy2(src_xzv, xzv_inters_dir)
                    print(f"已複製 XZV 檔案到: {xzv_inters_dir}/{time_part}.xzv")
                except Exception as e:
                    print(f"複製 XZV 檔案失敗: {str(e)}")
            else:
                print(f"警告: 找不到 XZV 檔案: {src_xzv}")
            
            # 2. 複製 PNG 檔案到 profile_inters 目錄
            src_profile = os.path.join(temp_intersection_path, "inverted_profile.png")
            if os.path.exists(src_profile):
                try:
                    dst_profile = os.path.join(profile_inters_dir, f"{time_part}.png")
                    shutil.copy2(src_profile, dst_profile)
                    print(f"已複製 profile 檔案到: {dst_profile}")
                except Exception as e:
                    print(f"複製 profile 檔案失敗: {str(e)}")
            else:
                print(f"警告: 找不到 profile 檔案: {src_profile}")
            
            # 3. 保留臨時目錄的完整結果，以供後續分析
            
            print(f"成功生成所有視覺化結果")
            processed_count += 1
        except Exception as e:
            print(f"視覺化生成過程中出現錯誤: {str(e)}")
            import traceback
            traceback.print_exc()  # 打印詳細錯誤堆疊
            
        print(f"成功完成 {current_folder} 的交集反演，結果已儲存到 {temp_intersection_path} 並複製到指定目錄")
    
    if processed_count > 0:
        print(f"交集反演已完成，處理了 {processed_count} 個時間點")
        return 0
    else:
        print("沒有任何時間點需要進行交集反演，或所有時間點都已被跳過")
        return 0

if __name__ == "__main__":
    sys.exit(main()) 