#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import re
import numpy as np
import pygimli as pg
import pygimli.physics.ert as ert
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import yaml
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 加入 src 目錄到 Python 路徑中，以便能夠導入 taip_ert_pipeline 模組
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from taip_ert_pipeline import visualization

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

def load_config():
    """
    載入配置文件
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "configs", "site.yaml")
    if not os.path.exists(config_path):
        print(f"錯誤：配置文件不存在 {config_path}")
        return None
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"成功載入配置文件: {config_path}")
        return config
    except Exception as e:
        print(f"載入配置文件時出錯: {str(e)}")
        return None

def run_intersection_inversion():
    """
    主函數：執行交集反演流程
    """
    # 載入配置文件
    config = load_config()
    if not config:
        print("無法載入配置文件，使用默認設置")
        base_path = "e:/R2MSDATA/TAIP_T1_test/output"
        colormap_file = None
        title_verbose = False
    else:
        # 從配置文件中讀取設置
        base_path = config["output"]["output_dir"]
        colormap_file = config["output"]["colormap_file"]
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
        return
    
    # 設置反演參數
    if config and "inversion" in config:
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
    
    # 迭代處理每個時間點
    for i in range(3, len(sorted_folders)):
        current_folder = sorted_folders[i]
        previous_folders = [sorted_folders[i-3], sorted_folders[i-2], sorted_folders[i-1]]
        
        print(f"\n===== 處理第 {i+1}/{len(sorted_folders)} 個時間點：{current_folder} =====")
        print(f"前三個時間點：{previous_folders[0]}, {previous_folders[1]}, {previous_folders[2]}")
        
        # 1. 檢查當前資料夾是否完成第四次反演
        if not check_repeat_4_complete(base_path, current_folder):
            print(f"警告：{current_folder} 未完成第四次反演，跳過處理")
            continue
        
        # 檢查前三個資料夾是否都完成第四次反演
        all_previous_complete = True
        for folder in previous_folders:
            if not check_repeat_4_complete(base_path, folder):
                print(f"警告：{folder} 未完成第四次反演，跳過處理")
                all_previous_complete = False
                break
        
        if not all_previous_complete:
            continue
        
        # 2. 檢查交集目錄是否已存在
        intersection_path = os.path.join(base_path, current_folder, "intersection")
        if os.path.exists(intersection_path):
            print(f"警告：交集目錄 {intersection_path} 已存在，跳過處理")
            continue
        
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
        
        # 9. 創建儲存路徑
        intersection_path = os.path.join(base_path, current_folder, "intersection")
        intersection_ert_path = os.path.join(intersection_path, "ERTManager")
        os.makedirs(intersection_ert_path, exist_ok=True)
        
        # 10. 儲存反演結果
        # 使用 saveResult 會自動創建 ERTManager 目錄並保存多個文件
        path, fig, ax = mgr.saveResult(intersection_path) # 會自動創建 ERTManager 文件夾
        plt.close(fig)
        
        # 額外保存資料文件
        mgr.data.save(os.path.join(intersection_ert_path, f"{current_folder}_inv.ohm"))
        
        # 儲存模型響應
        pg.utils.saveResult(os.path.join(intersection_ert_path, 'model_response.txt'),
                          data=mgr.inv.response, mode='w')
        
        # 保存其他反演信息
        rrmsHistory = mgr.inv.rrmsHistory
        chi2History = mgr.inv.chi2History
        export_inversion_info(mgr, intersection_path, lam, rrmsHistory, chi2History)
        
        # 11. 使用標準的視覺化工具生成圖表
        print(f"使用標準視覺化模組生成圖表")
        try:
            # 先讓 ERTVisualizer 加載反演結果
            results = visualizer.load_inversion_results(intersection_path)
            
            # 如果視覺化器沒有找到 rrms 和 chi2，從 mgr 中取值
            if not results or results.get("rrms") is None or results.get("chi2") is None:
                print("視覺化器未能從文件中讀取到 rrms 和 chi2 值，使用反演結果中的值")
                
                # 將值寫入到相應的文件中
                info_file = export_inversion_info(mgr, intersection_path, lam, rrmsHistory, chi2History)
                
                # 重新加載反演結果
                results = visualizer.load_inversion_results(intersection_path)
                if not results:
                    print("警告：即使在重新導出反演信息後仍無法加載反演結果")
            
            # 設置額外的可視化參數
            plot_kwargs = {
                "title_verbose": title_verbose,  # 使用配置中的設置
                'cMin': 10,                   # 最小電阻率值
                'cMax': 1000,                 # 最大電阻率值
                'label': 'Resistivity $\\Omega$m', 
            }
            
            # 從配置文件中獲取 XZV 文件路徑
            xzv_file = config.get("inversion", {}).get("xzv_file", None) if config else None
            if xzv_file and os.path.exists(xzv_file):
                print(f"使用 XZV 文件: {xzv_file}")
                # 使用 XZV 文件生成更精確的等值線圖
                visualizer.plot_all(intersection_path, current_folder, xzv_file, **plot_kwargs)
            else:
                # 不使用 XZV 文件
                visualizer.plot_all(intersection_path, current_folder, **plot_kwargs)
            
            print(f"成功生成所有視覺化結果")
        except Exception as e:
            print(f"視覺化生成過程中出現錯誤: {str(e)}")
            import traceback
            traceback.print_exc()  # 打印詳細錯誤堆疊
            
        print(f"成功完成 {current_folder} 的交集反演，結果已儲存到 {intersection_path}")

if __name__ == "__main__":
    run_intersection_inversion() 