"""視覺化模組 (反演結果圖、cross‑plot、misfit 直方圖)"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scipy.io
import json
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.tri as tri
import pygimli as pg
from pygimli.physics import ert
import glob
from . import utils
import re

class ERTVisualizer:
    """處理 ERT 反演結果視覺化的類別"""
    
    def __init__(self, visualization_config):
        """
        初始化 ERT 視覺化器
        
        參數:
            visualization_config: 包含視覺化相關配置的字典
        """
        self.config = visualization_config
        self.root_dir = visualization_config.get("root_dir", "")
        
        # 避免路徑重複包含 "output"
        if "output" in self.root_dir.split(os.path.sep)[-1]:
            self.output_dir = self.root_dir
        else:
            self.output_dir = visualization_config.get("output_dir", os.path.join(self.root_dir, "output"))
        
        self.colormap_file = visualization_config.get("colormap_file", None)
        
        # 確保輸出目錄存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 設置日誌 - 修改為只使用已存在的logger
        self.logger = logging.getLogger('visualization')
        self.logger.setLevel(logging.INFO)
        
        # 只有在沒有任何處理器時才添加處理器
        # 這樣可以避免重複輸出
        if not self.logger.handlers:
            # 添加文件處理器
            log_file = os.path.join(self.root_dir, 'visualization.log')
            file_handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%y/%m/%d - %H:%M:%S')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # 載入自定義色彩圖
        self.custom_cmap = None
        if self.colormap_file and os.path.exists(self.colormap_file):
            try:
                clm_data = scipy.io.loadmat(self.colormap_file)
                clm = clm_data['clm']
                self.custom_cmap = ListedColormap(clm)
                self.logger.info(f"已載入自定義色彩圖: {self.colormap_file}")
            except Exception as e:
                self.logger.warning(f"載入色彩圖失敗: {str(e)}")
    
    def load_inversion_results(self, result_path):
        """
        載入反演結果
        
        參數:
            result_path: 反演結果儲存路徑
            
        返回:
            results_dict: 包含反演結果的字典
        """
        try:
            # 檢查路徑是否存在
            if not os.path.exists(result_path):
                self.logger.error(f"反演結果路徑不存在: {result_path}")
                return None
                
            # 確定ERTManager目錄
            mgr_dir = os.path.join(result_path, 'ERTManager')
            if not os.path.exists(mgr_dir):
                self.logger.error(f"ERTManager目錄不存在: {mgr_dir}")
                return None
                
            self.logger.info(f"載入反演結果: {result_path}")
            
            # 載入網格 - 嘗試不同的可能的文件名
            mesh_file_options = [
                os.path.join(mgr_dir, 'resistivity-pd.bms'),  # 原始選項
                os.path.join(mgr_dir, 'paraDomain.bms'),      # 可能的替代名稱
                os.path.join(mgr_dir, 'resistivity.bms')      # 另一個可能的名稱
            ]
            
            paraDomain = None
            for mesh_file in mesh_file_options:
                if os.path.exists(mesh_file):
                    #self.logger.info(f"找到網格文件: {mesh_file}")
                    paraDomain = pg.load(mesh_file)
                    break
                    
            if paraDomain is None:
                self.logger.error(f"找不到網格文件，嘗試的路徑: {mesh_file_options}")
                return None
            
            # 載入資料
            data_files = glob.glob(os.path.join(mgr_dir, '*.ohm'))
            if not data_files:
                self.logger.error(f"找不到反演資料檔案 (.ohm)")
                return None
                
            data = pg.load(data_files[0])
            
            # 載入模型
            model_file_options = [
                os.path.join(mgr_dir, 'model.vector'),       # 原始選項
                os.path.join(mgr_dir, 'resistivity.vector')  # 可能的替代名稱
            ]
            
            model = None
            for model_file in model_file_options:
                if os.path.exists(model_file):
                    #self.logger.info(f"找到模型文件: {model_file}")
                    model = pg.load(model_file)
                    break
            
            if model is None:
                self.logger.error(f"找不到模型文件，嘗試的路徑: {model_file_options}")
                return None
            
            # 載入模型響應
            response_file_options = [
                os.path.join(mgr_dir, 'response.vector'),
                os.path.join(mgr_dir, 'model_response.txt')
            ]
            
            response = None
            for response_file in response_file_options:
                if os.path.exists(response_file):
                    #self.logger.info(f"找到響應文件: {response_file}")
                    if response_file.endswith('.txt'):
                        response = np.loadtxt(response_file)
                    else:
                        response = pg.load(response_file)
                    break
            
            if response is None:
                self.logger.error(f"找不到響應文件，嘗試的路徑: {response_file_options}")
                return None
            
            # 載入覆蓋率
            coverage_file_options = [
                os.path.join(mgr_dir, 'coverage.vector'),
                os.path.join(mgr_dir, 'resistivity-cov.vector')
            ]
            
            coverage = None
            for coverage_file in coverage_file_options:
                if os.path.exists(coverage_file):
                    #self.logger.info(f"找到覆蓋率文件: {coverage_file}")
                    coverage = pg.load(coverage_file)
                    break
            
            if coverage is None:
                self.logger.warning(f"找不到覆蓋率文件，嘗試的路徑: {coverage_file_options}")
            
            # 最大調查深度
            investg_depth = 0
            for i in range(len(paraDomain.positions())):
                if paraDomain.positions()[i][2] < investg_depth:
                    investg_depth = paraDomain.positions()[i][2]
            
            # 載入反演參數
            inv_info_file = os.path.join(mgr_dir, 'inv_info.txt')
            rrms = None
            chi2 = None
            lam = None
            rrmsHistory = []
            chi2History = []
            
            if os.path.exists(inv_info_file):
                with open(inv_info_file, 'r') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        if 'rrms:' in line:
                            rrms = float(line.split(':')[-1])
                        elif 'chi2:' in line:
                            chi2 = float(line.split(':')[-1])
                        elif 'use lam:' in line:
                            lam = float(line.split(':')[-1])
                        elif 'Iter.  rrms  chi2' in line:
                            for j in range(i+1, len(lines)):
                                if len(lines[j].strip()) > 0:
                                    parts = lines[j].strip().split(',')
                                    if len(parts) >= 3:
                                        rrmsHistory.append(float(parts[1]))
                                        chi2History.append(float(parts[2]))
            else:
                self.logger.warning(f"找不到反演資訊檔案: {inv_info_file}")
            
            # 構建結果字典
            results_dict = {
                "paraDomain": paraDomain,
                "data": data,
                "response": response,
                "model": model,
                "coverage": coverage,
                "investg_depth": investg_depth,
                "rrms": rrms,
                "chi2": chi2,
                "lam": lam,
                "rrmsHistory": rrmsHistory,
                "chi2History": chi2History,
                "result_path": result_path
            }
            
            self.logger.info(f"成功載入反演結果")
            return results_dict
            
        except Exception as e:
            self.logger.error(f"載入反演結果時發生錯誤: {str(e)}", exc_info=True)
            return None
    
    def plot_inverted_profile(self, result_path, file_name=None, **kwargs):
        """
        繪製反演結果的剖面圖
        
        參數:
            result_path: 反演結果儲存路徑
            file_name: 檔案名稱，用於圖表標題
            **kwargs: 傳給 pg.show 的參數
        """
        try:
            # 設置 matplotlib 為非交互式模式，避免彈出視窗
            plt.ioff()
            
            # 載入反演結果
            results = self.load_inversion_results(result_path)
            if not results:
                return False
                
            mesh = results["paraDomain"]
            model = results["model"]
            coverage = results["coverage"]
            rrms = results["rrms"]
            chi2 = results["chi2"]
            data = results["data"]
            
            # 建立圖表
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # 繪製模型
            if 'cMap' not in kwargs and self.colormap_file and os.path.exists(self.colormap_file):
                clm_data = scipy.io.loadmat(self.colormap_file)
                clm = clm_data['clm']
                custom_cmap = plt.cm.colors.ListedColormap(clm)
                kwargs['cMap'] = custom_cmap
                
            # 設置預設參數
            if 'cMin' not in kwargs:
                kwargs['cMin'] = 10
            if 'cMax' not in kwargs:
                kwargs['cMax'] = 1000
            if 'logScale' not in kwargs:
                kwargs['logScale'] = True
            if 'label' not in kwargs:
                kwargs['label'] = 'Resistivity $\Omega$m'
            if 'xlabel' not in kwargs:
                kwargs['xlabel'] = 'Distance [m]'
            if 'ylabel' not in kwargs:
                kwargs['ylabel'] = 'Elevation [m]'

            # 遮蓋低覆蓋率區域
            if coverage is not None:
                kwargs['coverage'] = coverage
                
            # 強制設置不顯示視窗
            kwargs['show'] = False
            
            # 繪製模型 - 設置 show=False 防止自動顯示
            pg.show(mesh, model, ax=ax, **kwargs)
            
            ax.plot(np.array(pg.x(data)), np.array(pg.y(data)), 'kv', markersize=3)
            # 添加標題
            if file_name:
                title = f"Inverted resistivity section: {file_name}\n"
            else:
                title = "Inverted resistivity section\n"
                
            title += f"rRMS = {rrms:.2f}%, $\chi^2$ = {chi2:.3f}"
            ax.set_title(title)
            
            
                        
            # 保存圖表
            save_file = os.path.join(result_path, 'inverted_profile.png')
            fig.savefig(save_file, dpi=300, bbox_inches='tight')
            plt.close(fig)  # 確保關閉圖表
            
            self.logger.info(f"反演剖面圖已保存: {save_file}")
            return True
            
        except Exception as e:
            plt.close('all')  # 確保所有圖表都被關閉
            self.logger.error(f"繪製反演剖面圖時發生錯誤: {str(e)}", exc_info=True)
            return False
    
    def plot_inverted_contour(self, result_path, file_name=None, **kwargs):
        """
        繪製反演結果的等值線圖，使用 MATLAB 風格
        
        參數:
            result_path: 反演結果儲存路徑
            file_name: 檔案名稱，用於圖表標題
            **kwargs: 繪圖參數
        """
        try:
            # 載入反演結果
            results = self.load_inversion_results(result_path)
            if not results:
                return False
                
            mesh = results["paraDomain"]
            model = results["model"]
            data = results["data"]
            rrms = results["rrms"]
            chi2 = results["chi2"]
            
            # 設置色彩映射
            custom_cmap = None
            if self.colormap_file and os.path.exists(self.colormap_file):
                try:
                    clm_data = scipy.io.loadmat(self.colormap_file)
                    clm = clm_data['clm']
                    custom_cmap = plt.cm.colors.ListedColormap(clm)
                    self.logger.info(f"已載入自定義色彩圖: {self.colormap_file}")
                except Exception as e:
                    self.logger.warning(f"載入色彩圖失敗: {str(e)}")
                    custom_cmap = None
            
            # 如果沒有提供自定義色彩圖，使用默認色彩圖
            if custom_cmap is None:
                custom_cmap = plt.cm.jet
            
            # 取得網格單元中心座標
            xc = mesh.cellCenter()[:, 0]
            yc = mesh.cellCenter()[:, 1]
            
            # 獲取模型數據的對數
            log_model = np.log10(model)
            
            # 計算網格範圍
            left = min(pg.x(data))
            right = max(pg.x(data))
            depth = (right - left) * 0.2  # 深度設為寬度的20%
            top = max(pg.y(data))
            bottom = min(pg.y(data)) - depth
            
            # 建立新的網格
            x_steps = int((right - left) / 2) + 1  # 每2米一個格點
            y_steps = int((top - bottom) / 2) + 1  # 每2米一個格點
            X, Y = np.meshgrid(np.linspace(left, right, x_steps), np.linspace(bottom, top, y_steps))
            
            # 使用 PyGIMLi 的內插函數重新內插數據到等距網格
            grid = pg.createGrid(x=np.linspace(left, right, x_steps), y=np.linspace(bottom, top, y_steps))
            grid_pos = grid.positions()
            
            # 內插到網格上
            grid_data = pg.interpolate(mesh, log_model, grid_pos)
            
            # 轉換為二維數據，用於繪製等值線
            grid_data_2d = np.reshape(grid_data, (y_steps, x_steps))
            
            # 建立 figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 設定色階與級數
            vmin = kwargs.get('cMin', 10)
            vmax = kwargs.get('cMax', 1000)
            log_vmin = np.log10(vmin)
            log_vmax = np.log10(vmax)
            levels = 32
            contour_levels = np.linspace(log_vmin, log_vmax, levels+1)
            contour_levels = np.hstack([0, contour_levels[1:]])  # 從0開始，與MATLAB一致
            
            # 將超出範圍的值裁切到極端
            grid_data_clipped = np.clip(grid_data_2d, log_vmin, log_vmax)
            
            # 繪製等值線填充，不加 extend
            cf = ax.contourf(X, Y, grid_data_clipped, contour_levels, cmap=custom_cmap)
            
            # 繪製電極位置
            ax.plot(np.array(pg.x(data)), np.array(pg.y(data)), 'sk', markersize=3, markerfacecolor='k')
            
            # 設置軸
            ax.set_aspect('equal')
            ax.set_xlim(left, right)
            ax.set_ylim(bottom, top)
            
            # 設置刻度
            x_ticks = np.arange(0, right+15, 15)
            ax.set_xticks(x_ticks)
            
            # 設置字體和線寬
            plt.rcParams["font.family"] = "Times New Roman"
            ax.tick_params(axis='both', which='major', labelsize=16, width=2)
            for spine in ax.spines.values():
                spine.set_linewidth(2)
            
            # 設置標籤
            ax.set_xlabel('Distance (m)', fontname='Times New Roman', fontsize=16, fontweight='bold')
            ax.set_ylabel('Elevation (m)', fontname='Times New Roman', fontsize=16, fontweight='bold')
            
            # 添加標題
            if file_name:
                date_part = file_name[:8]
                try:
                    # 嘗試從文件名解析日期
                    date_str = f"20{date_part[:2]}/{date_part[2:4]}/{date_part[4:6]} {date_part[6:8]}:00"
                except:
                    date_str = file_name
                title = f"Inverted Resistivity Profile at {date_str}\nnumber of data={len(data['rhoa']):.0f}, rrms={rrms:.2f}%, $\chi^2$={chi2:.3f}"
            else:
                title = f"Inverted Resistivity Profile\nrrms={rrms:.2f}%, $\chi^2$={chi2:.3f}"
                
            ax.set_title(title, fontname='Times New Roman', fontsize=16, fontweight='bold')
            
            # 添加色條
            cb = plt.colorbar(cf, ax=ax, format='%d')
            cb.ax.set_ylabel('$\Omega$-m', fontname='Times New Roman', fontsize=20)
            
            # 設置色條刻度標籤
            cb_ticks = np.linspace(log_vmin, log_vmax, 9)
            cb.set_ticks(cb_ticks)
            cb.set_ticklabels([f"{10**t:.0f}" for t in cb_ticks])
            cb.ax.tick_params(labelsize=12)
            
            # 保存圖表
            save_file = os.path.join(result_path, 'inverted_contour.png')
            fig.savefig(save_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"反演等值線圖已保存: {save_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"繪製反演等值線圖時發生錯誤: {str(e)}", exc_info=True)
            return False
    
    def plot_convergence(self, result_path):
        """
        繪製反演收斂曲線
        
        參數:
            result_path: 反演結果儲存路徑
            
        返回:
            rrmsHistory: RRMS歷史紀錄
            chi2History: Chi2歷史紀錄
        """
        try:
            # 載入反演結果
            results = self.load_inversion_results(result_path)
            if not results:
                return [], []
                
            rrmsHistory = results["rrmsHistory"]
            chi2History = results["chi2History"]
            
            # 檢查是否有收斂歷史
            if not rrmsHistory or not chi2History:
                self.logger.warning("找不到收斂歷史資料")
                return [], []
                
            # 建立圖表
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(np.linspace(1, len(rrmsHistory), len(rrmsHistory)), rrmsHistory, 
                   linestyle='-', marker='o', c='black')
            ax.set_xlabel('Iteration Number')
            ax.set_ylabel('rRMS (%)')
            ax.set_title('Convergence Curve of Resistivity Inversion')
            
            ax2 = ax.twinx()
            ax2.plot(np.linspace(1, len(rrmsHistory), len(rrmsHistory)), chi2History, 
                    linestyle='-', marker='o', c='blue')
            ax2.set_ylabel('$\chi^2$', c='blue')
            
            ax.grid()
            
            # 保存圖表
            save_file = os.path.join(result_path, 'convergence.png')
            fig.savefig(save_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"收斂曲線已保存: {save_file}")
            return rrmsHistory, chi2History
            
        except Exception as e:
            self.logger.error(f"繪製收斂曲線時發生錯誤: {str(e)}", exc_info=True)
            return [], []
    
    def plot_crossplot(self, result_path):
        """
        繪製測量值和預測值的交叉圖
        
        參數:
            result_path: 反演結果儲存路徑
        """
        try:
            # 載入反演結果
            results = self.load_inversion_results(result_path)
            if not results:
                return False
                
            data = results["data"]
            response = results["response"]
                            
            # 建立圖表
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(np.log10(data["rhoa"]), np.log10(response), s=1)
            
            # 設置圖表範圍
            xticks = ax.get_xlim()
            yticks = ax.get_ylim()
            lim = max(max(yticks), max(xticks)) + 0.5
            ax.plot([0, lim], [0, lim], 'k-', linewidth=1, alpha=0.2)
            ax.set_xlim([0, lim])
            ax.set_ylim([0, lim])
            
            # 設置標籤
            ax.set_xlabel('Log10 of Measured Apparent resistivity')
            ax.set_ylabel('Log10 of Predicted Apparent resistivity')
            ax.set_title(r'Crossplot of Measured vs Predicted Resistivity $\rho_{apparent}$')
            
            # 保存圖表
            save_file = os.path.join(result_path, 'crossplot.png')
            fig.savefig(save_file, dpi=300)
            plt.close(fig)
            
            self.logger.info(f"交叉圖已保存: {save_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"繪製交叉圖時發生錯誤: {str(e)}", exc_info=True)
            return False
    
    def plot_misfit_histogram(self, result_path):
        """
        繪製誤差直方圖
        
        參數:
            result_path: 反演結果儲存路徑
        """
        try:
            # 載入反演結果
            results = self.load_inversion_results(result_path)
            if not results:
                return False
                
            data = results["data"]
            response = results["response"]
            
            # 計算相對誤差百分比
            data['misfit'] = np.abs((response - data["rhoa"]) / data["rhoa"]) * 100
            
            # 建立圖表
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(data['misfit'], np.linspace(0, 100, 21))
            ax.set_xticks(np.linspace(0, 100, 21))
            
            # 設置標籤
            ax.set_xlabel('Relative Data Misfit (%)')
            ax.set_ylabel('Number of Data')
            ax.set_title('Data Misfit Histogram for Removal of Poorly-Fit Data')
            
            # 保存圖表
            save_file = os.path.join(result_path, 'misfit_histogram.png')
            fig.savefig(save_file, dpi=300)
            plt.close(fig)
            
            self.logger.info(f"誤差直方圖已保存: {save_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"繪製誤差直方圖時發生錯誤: {str(e)}", exc_info=True)
            return False
    
    def plot_all(self, result_path, file_name=None, xzv_file=None, **kwargs):
        """
        繪製單一結果的所有圖表
        
        參數:
            result_path: 反演結果儲存路徑
            file_name: 檔案名稱，用於圖表標題
            xzv_file: .xzv 文件路徑，用於繪製等值線圖（如果提供）
            **kwargs: 額外參數
                title_verbose: 是否顯示詳細標題，預設為 False
                output_profile_dir: 如果指定，則將等值線圖輸出到此目錄
                output_xzv_dir: 如果指定，則將 xzv 檔案輸出到此目錄
        """
        try:
            self.logger.info(f"繪製所有圖表: {result_path}")
            
            # 從 kwargs 中提取 title_verbose 參數或使用默認值
            title_verbose = kwargs.get('title_verbose', False)
            
            # 建立 kwargs 
            # 如果有自定義色彩圖，則使用
            custom_cmap = None
            if self.colormap_file and os.path.exists(self.colormap_file):
                try:
                    clm_data = scipy.io.loadmat(self.colormap_file)
                    clm = clm_data['clm']
                    custom_cmap = plt.cm.colors.ListedColormap(clm)
                    self.logger.info(f"已載入自定義色彩圖: {self.colormap_file}")
                except Exception as e:
                    self.logger.warning(f"載入色彩圖失敗: {str(e)}")
            
            # 設定繪圖參數
            plot_kwargs = {
                'cMin': 10, 
                'cMax': 1000, 
                'label': 'Resistivity $\\Omega$m', 
                'cMap': custom_cmap if custom_cmap else 'jet',
                'orientation': 'vertical',
                'logScale': True,
                'title_verbose': title_verbose
            }
            
            # 傳遞其他參數
            for key, value in kwargs.items():
                plot_kwargs[key] = value
            
            self.logger.info(f"使用繪圖參數: {plot_kwargs}")
            
            # 繪製各種圖表，傳入 plot_kwargs
            profile_success = self.plot_inverted_profile(result_path, file_name, **plot_kwargs)
            
            # 如果提供了 xzv 文件，則使用 xzv 文件繪製等值線圖
            if xzv_file and os.path.exists(xzv_file):
                self.logger.info(f"使用 XZV 文件繪製等值線圖: {xzv_file}")
                contour_success = self.plot_inverted_contour_xzv(result_path, xzv_file, file_name, **plot_kwargs)
            else:
                # 否則使用默認方法繪製等值線圖
                contour_success = self.plot_inverted_contour(result_path, file_name, **plot_kwargs)
            
            # 繪製其他圖表
            _, _ = self.plot_convergence(result_path)
            cross_success = self.plot_crossplot(result_path)
            misfit_success = self.plot_misfit_histogram(result_path)
            
            success = profile_success and contour_success and cross_success and misfit_success
            if success:
                self.logger.info(f"所有圖表繪製成功: {result_path}")
            else:
                self.logger.warning(f"部分圖表繪製失敗: {result_path}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"繪製所有圖表時發生錯誤: {str(e)}", exc_info=True)
            return False
    
    def visualize_results(self, urf_file_results):
        """
        視覺化一組 URF 檔案的反演結果
        
        參數:
            urf_file_results: 包含 URF 檔案反演結果的字典
        """
        try:
            file_name = os.path.basename(urf_file_results.get("file", ""))
            iterations = urf_file_results.get("iterations", [])
            
            self.logger.info(f"視覺化 {file_name} 的反演結果, 共 {len(iterations)} 次迭代")
            
            # 繪製所有迭代的結果
            self.plot_all(iterations[0]["save_path"], os.path.basename(iterations[0]["file"]))
            
            return True
            
        except Exception as e:
            self.logger.error(f"視覺化結果時發生錯誤: {str(e)}", exc_info=True)
            return False
    
    def export_report(self, results, output_path):
        """產生 HTML 形式的報告"""
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # 準備報告內容
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>TAIP ERT 反演結果報告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #333366; }
                .result-section { margin-bottom: 30px; border: 1px solid #ccc; padding: 15px; border-radius: 5px; }
                .iteration { margin-top: 20px; border-top: 1px dashed #999; padding-top: 10px; }
                img { max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ddd; }
                table { border-collapse: collapse; width: 100%; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>TAIP ERT 反演結果報告</h1>
            <p>生成時間: {timestamp}</p>
        """
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html_content = html_content.format(timestamp=timestamp)
        
        # 添加每個檔案的結果
        for result in results:
            file_basename = os.path.basename(result["file"]).split('.')[0]
            
            html_content += f"""
            <div class="result-section">
                <h2>檔案: {file_basename}</h2>
                <table>
                    <tr>
                        <th>迭代次數</th>
                        <th>RRMS (%)</th>
                        <th>Chi²</th>
                    </tr>
            """
            
            for iteration in result["iterations"]:
                repeat_num = iteration["repeat"]
                rrms = iteration["rrms"]
                chi2 = iteration["chi2"]
                
                html_content += f"""
                    <tr>
                        <td>{repeat_num}</td>
                        <td>{rrms:.2f}</td>
                        <td>{chi2:.3f}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            """
            
            # 添加最後一次迭代的結果圖片
            last_iteration = result["iterations"][-1]
            save_path = last_iteration["save_path"]
            
            html_content += f"""
                <div class="iteration">
                    <h3>最終反演結果 (迭代 {last_iteration['repeat']})</h3>
                    <p>RRMS: {last_iteration['rrms']:.2f}%, Chi²: {last_iteration['chi2']:.3f}</p>
                    <img src="{os.path.relpath(os.path.join(save_path, 'inverted_profile.png'), output_path)}" alt="反演剖面圖">
                    <img src="{os.path.relpath(os.path.join(save_path, 'inverted_contour.png'), output_path)}" alt="等值線圖">
                    <img src="{os.path.relpath(os.path.join(save_path, 'convergence.png'), output_path)}" alt="收斂曲線">
                    <img src="{os.path.relpath(os.path.join(save_path, 'crossplot.png'), output_path)}" alt="交叉圖">
                    <img src="{os.path.relpath(os.path.join(save_path, 'misfit_histogram.png'), output_path)}" alt="誤差直方圖">
                </div>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # 寫入 HTML 文件
        report_path = os.path.join(output_path, "report.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML 報告已生成: {report_path}")
        return report_path
    
    def plot_difference_profile(self, result_path1, result_path2, file_name1=None, file_name2=None, **kwargs):
        """
        繪製兩個反演結果的差異剖面圖
        
        參數:
            result_path1: 第一個反演結果儲存路徑
            result_path2: 第二個反演結果儲存路徑
            file_name1: 第一個檔案名稱
            file_name2: 第二個檔案名稱
            **kwargs: 繪圖參數
        """
        try:
            # 載入反演結果
            mgr1 = self.load_inversion_results(result_path1)
            mgr2 = self.load_inversion_results(result_path2)
            
            if not mgr1 or not mgr2:
                self.logger.error("無法載入反演結果")
                return False
            
            # 獲取資料
            model1 = mgr1['model']
            model2 = mgr2['model']
            data = mgr1['data']
            left = min(pg.x(data))
            right = max(pg.x(data))
            depth = mgr1['investg_depth']
            
            # 建立網格
            mesh_x = np.linspace(left, right, 250)
            mesh_y = np.linspace(-depth, 0, 150)
            grid = pg.createGrid(x=mesh_x, y=mesh_y)
            X, Y = np.meshgrid(mesh_x, mesh_y)
            
            # 計算差異百分比
            one_line_diff = (np.log10(model2) - np.log10(model1))/np.log10(model1)*100
            diff_grid = np.reshape(pg.interpolate(mgr1['paraDomain'], one_line_diff, grid.positions()), (len(mesh_y), len(mesh_x)))
            
            # 建立圖表
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # 設定色彩映射
            if 'cMap' not in kwargs and self.custom_cmap:
                kwargs['cMap'] = self.custom_cmap
                
            # 預設參數
            if 'cMin' not in kwargs:
                kwargs['cMin'] = -50
            if 'cMax' not in kwargs:
                kwargs['cMax'] = 50
                
            # 繪製等值線
            levels = 64
            if 'levels' in kwargs:
                levels = kwargs.pop('levels')
                
            contourf = ax.contourf(X, Y, diff_grid, cmap=kwargs.get('cMap', 'seismic'), 
                          levels=levels, vmin=kwargs['cMin'], vmax=kwargs['cMax'], antialiased=True)
            
            # 設置圖表參數
            ax.set_aspect('equal')
            ax.set_xlim(left, right)
            ax.set_ylim(-depth, 0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(5))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
            
            # 設置標題和標籤
            if file_name1 and file_name2:
                title = f"Difference (%) between {file_name1} and {file_name2}"
            else:
                title = "Resistivity difference profile"
                
            ax.set_title(title)
            ax.set_xlabel(kwargs.get('xlabel', 'Distance [m]'))
            ax.set_ylabel(kwargs.get('ylabel', 'Elevation [m]'))
            
            # 添加色彩條
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.15)
            cbar = plt.colorbar(contourf, cax=cax)
            cbar.ax.set_ylabel(kwargs.get('label', 'Resistivity difference (%)'))
            
            # 保存圖表
            file_basename1 = os.path.basename(result_path1)
            file_basename2 = os.path.basename(result_path2)
            save_file = os.path.join(os.path.dirname(result_path1), f'diff_profile_{file_basename1}_vs_{file_basename2}.png')
            fig.savefig(save_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"差異剖面圖已保存: {save_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"繪製差異剖面圖時發生錯誤: {str(e)}", exc_info=True)
            return False
    
    def _clip_corner(self, ax, data, left, right, depth):
        """
        將模型角落裁剪為白色
        
        參數:
            ax: matplotlib軸
            data: 資料容器
            left: 左邊界
            right: 右邊界
            depth: 深度
        """
        # 地表多邊形
        surface_x = list(pg.x(data))
        surface_y = list(pg.y(data))
        surface_polygon = np.column_stack([surface_x + [right, left], surface_y + [max(pg.y(data)), max(pg.y(data))]])
        ax.add_patch(plt.Polygon(surface_polygon, color='white'))
        
        # 深度多邊形
        depth_polygon = np.array([
            [left, pg.y(data)[0]-depth+1], 
            [left, min(pg.y(data))-depth], 
            [right, min(pg.y(data))-depth], 
            [right, pg.y(data)[-1]-depth+1], 
            [left, pg.y(data)[0]-depth+1]
        ])
        ax.add_patch(plt.Polygon(depth_polygon, color='white'))
        
        # 左右三角形
        triangle_left = np.array([
            [left, pg.y(data)[0]], 
            [left+depth, min(pg.y(data))-depth], 
            [left, min(pg.y(data))-depth], 
            [left, pg.y(data)[0]]
        ])
        triangle_right = np.array([
            [right, pg.y(data)[-1]], 
            [right-depth, min(pg.y(data))-depth], 
            [right, min(pg.y(data))-depth], 
            [right, pg.y(data)[-1]]
        ])
        ax.add_patch(plt.Polygon(triangle_left, color='white'))
        ax.add_patch(plt.Polygon(triangle_right, color='white'))
    
    def plot_difference_contour(self, result_path1, result_path2, file_name1=None, file_name2=None, **kwargs):
        """
        繪製兩個反演結果的差異等值線圖
        
        參數:
            result_path1: 第一個反演結果儲存路徑
            result_path2: 第二個反演結果儲存路徑
            file_name1: 第一個檔案名稱
            file_name2: 第二個檔案名稱
            **kwargs: 繪圖參數
        """
        try:
            # 載入反演結果
            mgr1 = self.load_inversion_results(result_path1)
            mgr2 = self.load_inversion_results(result_path2)
            
            if not mgr1 or not mgr2:
                self.logger.error("無法載入反演結果")
                return False
            
            # 獲取資料
            model1 = mgr1['model']
            model2 = mgr2['model']
            data = mgr1['data']
            left = min(pg.x(data))
            right = max(pg.x(data))
            depth = mgr1['investg_depth']
            
            # 計算差異百分比
            model_diff = (np.log10(model2) - np.log10(model1))/np.log10(model1)*100
            
            # 建立三角網格
            xc = mgr1['paraDomain'].cellCenter()[:,0]
            yc = mgr1['paraDomain'].cellCenter()[:,1]
            triang = tri.Triangulation(xc, yc)
            
            # 建立圖表
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # 設定色彩映射
            if 'cMap' not in kwargs and self.custom_cmap:
                kwargs['cMap'] = self.custom_cmap
                
            # 預設參數
            if 'cMin' not in kwargs:
                kwargs['cMin'] = -50
            if 'cMax' not in kwargs:
                kwargs['cMax'] = 50
                
            # 繪製等值線
            levels = 50
            if 'levels' in kwargs:
                levels = kwargs.pop('levels')
                
            cax = ax.tricontourf(triang, model_diff, cmap=kwargs.get('cMap', 'seismic'), 
                              levels=levels, vmin=kwargs['cMin'], vmax=kwargs['cMax'])
            
            # 設置圖表參數
            ax.set_aspect('equal')
            ax.set_xlim(left, right)
            ax.set_ylim(-depth, 0)
            ax.yaxis.set_major_locator(plt.MultipleLocator(5))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
            
            # 裁剪角落
            self._clip_corner(ax, data, left, right, depth)
            
            # 設置標題和標籤
            if file_name1 and file_name2:
                title = f"Difference (%) between {file_name1} and {file_name2}"
            else:
                title = "Resistivity difference contour"
                
            ax.set_title(title)
            ax.set_xlabel(kwargs.get('xlabel', 'Distance [m]'))
            ax.set_ylabel(kwargs.get('ylabel', 'Elevation [m]'))
            
            # 添加色彩條
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.15)
            cbar = plt.colorbar(cax, cax=cax)
            cbar.set_label(kwargs.get('label', 'Resistivity difference (%)'))
            
            # 保存圖表
            file_basename1 = os.path.basename(result_path1)
            file_basename2 = os.path.basename(result_path2)
            save_file = os.path.join(os.path.dirname(result_path1), f'diff_contour_{file_basename1}_vs_{file_basename2}.png')
            fig.savefig(save_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"差異等值線圖已保存: {save_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"繪製差異等值線圖時發生錯誤: {str(e)}", exc_info=True)
            return False
    
    def read_xzv_file(self, xzv_file_path):
        """
        讀取 .xzv 文件
        
        參數:
            xzv_file_path: .xzv 文件路徑
            
        返回:
            X: X 座標網格
            Y: Y 座標網格
            data: 數據值網格 (對數值)
            IQR: 四分位數範圍網格
        """
        try:
            if not os.path.exists(xzv_file_path):
                self.logger.error(f"XZV 文件不存在: {xzv_file_path}")
                return None, None, None, None
                
            self.logger.info(f"讀取 XZV 文件: {xzv_file_path}")
            
            # 讀取文件
            xz_data = np.loadtxt(xzv_file_path)
            
            # 提取 X 和 Y 座標
            x = xz_data[:, 0]
            y = xz_data[:, 1]
            
            # 提取數據值和 IQR 值
            data_values = xz_data[:, 2]
            iqr_values = xz_data[:, 3]
            
            # 檢查唯一的 X 和 Y 值，獲取網格維度
            unique_x = np.unique(x)
            unique_y = np.unique(y)
            nx = len(unique_x)
            ny = len(unique_y)
            
            # 創建網格
            X, Y = np.meshgrid(unique_x, unique_y)
            
            # 創建數據網格
            data = np.zeros((ny, nx))
            IQR = np.zeros((ny, nx))
            
            for i in range(len(x)):
                ix = np.where(unique_x == x[i])[0][0]
                iy = np.where(unique_y == y[i])[0][0]
                data[iy, ix] = data_values[i]
                IQR[iy, ix] = iqr_values[i]
            
            return X, Y, data, IQR
            
        except Exception as e:
            self.logger.error(f"讀取 XZV 文件時發生錯誤: {str(e)}", exc_info=True)
            return None, None, None, None
    
    def set_matlab_plot_style(self, ax, title=None, xlabel='Distance (m)', ylabel='Elevation (m)'):
        """
        設置 MATLAB 風格的圖像
        
        參數:
            ax: matplotlib 軸
            title: 標題
            xlabel: X 軸標籤
            ylabel: Y 軸標籤
        """
        # 設置字體
        plt.rcParams["font.family"] = "Times New Roman"
        
        # 設置軸線寬度和刻度大小
        ax.tick_params(axis='both', which='major', labelsize=16, width=2)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        # 設置標籤
        ax.set_xlabel(xlabel, fontname='Times New Roman', fontsize=16, fontweight='bold')
        ax.set_ylabel(ylabel, fontname='Times New Roman', fontsize=16, fontweight='bold')
        
        # 設置標題
        if title:
            ax.set_title(title, fontname='Times New Roman', fontsize=16, fontweight='bold')
            
        return ax
    
    def plot_inverted_contour_xzv(self, result_path, xzv_file_path, file_name=None, **kwargs):
        """
        使用 .xzv 文件的格點座標繪製反演結果的等值線圖
        
        參數:
            result_path: 反演結果儲存路徑
            xzv_file_path: .xzv 文件路徑
            file_name: 檔案名稱，用於圖表標題
            **kwargs: 繪圖參數
                title_verbose: 是否顯示詳細標題，預設為 False
                output_xzv_dir: 如果指定，則將 xzv 檔案輸出到此目錄
                output_profile_dir: 如果指定，則將等值線圖輸出到此目錄，並使用與 xzv 檔案相同的檔名
        """
        try:
            # 載入反演結果
            results = self.load_inversion_results(result_path)
            if not results:
                return False
                
            # 讀取 xzv 文件
            X, Y, data_xzv, IQR = self.read_xzv_file(xzv_file_path)
            if X is None:
                self.logger.warning(f"無法讀取 XZV 文件，使用默認網格生成方法")
                return self.plot_inverted_contour(result_path, file_name, **kwargs)
                
            mesh = results["paraDomain"]
            model = results["model"]
            data = results["data"]
            rrms = results["rrms"]
            chi2 = results["chi2"]
            
            # 設置色彩映射
            custom_cmap = None
            if self.colormap_file and os.path.exists(self.colormap_file):
                try:
                    clm_data = scipy.io.loadmat(self.colormap_file)
                    clm = clm_data['clm']
                    custom_cmap = plt.cm.colors.ListedColormap(clm)
                    self.logger.info(f"已載入自定義色彩圖: {self.colormap_file}")
                except Exception as e:
                    self.logger.warning(f"載入色彩圖失敗: {str(e)}")
                    custom_cmap = None
            
            if custom_cmap is None:
                custom_cmap = plt.cm.jet
            
            # 獲取網格範圍
            left = kwargs.get('xmin', np.min(X))
            right = kwargs.get('xmax', np.max(X))
            bottom = kwargs.get('ymin', np.min(Y))
            top = kwargs.get('ymax', np.max(Y))
            
            # 獲取模型數據的對數
            log_model = np.log10(model)
            
            # 直接使用 pygimli 網格和插值，避免逐點內插
            # 創建一個更簡單的網格以提高內插效率和準確性
            x_steps = int((right - left) / 2) + 1  # 每2米一個格點
            y_steps = int((top - bottom) / 2) + 1  # 每2米一個格點
            new_X, new_Y = np.meshgrid(np.linspace(left, right, x_steps), np.linspace(bottom, top, y_steps))
            
            # 創建一個簡單的等距網格用於內插
            grid = pg.createGrid(x=np.linspace(left, right, x_steps), y=np.linspace(bottom, top, y_steps))
            grid_pos = grid.positions()
            
            # 使用 PyGIMLi 的內插函數
            self.logger.info(f"進行內插計算，網格大小: {x_steps}x{y_steps}")
            grid_data_vector = pg.interpolate(mesh, log_model, grid_pos)
            
            # 將內插結果轉換為二維數組
            grid_data = np.reshape(grid_data_vector, (y_steps, x_steps))
            
            # 檢查內插結果是否有效
            valid_data = np.logical_and(np.isfinite(grid_data), grid_data != 0)
            if np.sum(valid_data) < 0.5 * grid_data.size:
                self.logger.warning(f"內插結果可能存在問題：有效數據點只有 {np.sum(valid_data)}/{grid_data.size}")
                # 檢查內插結果的範圍
                if np.any(valid_data):
                    valid_min = np.min(grid_data[valid_data])
                    valid_max = np.max(grid_data[valid_data])
                    self.logger.warning(f"有效數據範圍: [{valid_min}, {valid_max}]")
            
            # 將值為0的點設為NaN，使其顯示為白色
            grid_data = np.where(grid_data == 0, np.nan, grid_data)
            
            # 計算有效數據的範圍（非NaN值）
            valid_mask = ~np.isnan(grid_data)
            if np.any(valid_mask):
                valid_indices_y, valid_indices_x = np.where(valid_mask)
                if len(valid_indices_y) > 0 and len(valid_indices_x) > 0:
                    # 獲取有效數據的邊界
                    min_x_idx, max_x_idx = np.min(valid_indices_x), np.max(valid_indices_x)
                    min_y_idx, max_y_idx = np.min(valid_indices_y), np.max(valid_indices_y)
                    
                    # 獲取邊界對應的坐標值
                    x_min, x_max = new_X[0, min_x_idx], new_X[0, max_x_idx]
                    y_min, y_max = new_Y[min_y_idx, 0], new_Y[max_y_idx, 0]
                    
                    # 添加一點邊距
                    x_margin = (x_max - x_min) * 0.001
                    y_margin = (y_max - y_min) * 0.001
                    plot_xlim = [x_min - x_margin, x_max + x_margin]
                    plot_ylim = [y_min - y_margin, y_max + y_margin]
                else:
                    plot_xlim = [left, right]
                    plot_ylim = [bottom, top]
            else:
                plot_xlim = [left, right]
                plot_ylim = [bottom, top]
            
            # 建立 figure，背景設為白色
            fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')
            ax.set_facecolor('white')
            
            # 設定色階與級數
            vmin = kwargs.get('cMin', 10)
            vmax = kwargs.get('cMax', 1000)
            log_vmin = np.log10(vmin)
            log_vmax = np.log10(vmax)
            
            levels = kwargs.get('levels', 32)
            contour_levels = np.linspace(log_vmin, log_vmax, levels)
            
            # 將超出範圍的值裁切到極端
            grid_data_clipped = np.clip(grid_data, log_vmin, log_vmax)
            
            # 繪製等值線填充，不加 extend
            cf = ax.contourf(new_X, new_Y, grid_data_clipped, contour_levels, cmap=custom_cmap)
            
            # 繪製電極位置
            ax.plot(np.array(pg.x(data)), np.array(pg.y(data)), 'sk', markersize=3, markerfacecolor='k')
            
            # 設置字體為 Times New Roman
            plt.rcParams["font.family"] = "Times New Roman"
            
            # 設置刻度標籤字體
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontname('Times New Roman')
                label.set_fontsize(16)
            
            # 設置軸線寬度和刻度大小
            ax.tick_params(axis='both', which='major', width=2)
            for spine in ax.spines.values():
                spine.set_linewidth(2)
                
            # 設置軸
            ax.set_aspect('equal')
            ax.set_xlim(plot_xlim)
            ax.set_ylim(plot_ylim)
            
            # 設置刻度
            x_ticks = np.arange(0, right+15, 15)
            ax.set_xticks(x_ticks)
            
            # 檢查 title_verbose 選項
            title_verbose = kwargs.get('title_verbose', False)
            
            # 添加標題，根據 title_verbose 選項決定顯示方式
            if file_name:
                # 嘗試從文件名解析日期 (格式為 "yymmddhh")
                date_match = re.match(r'(\d{2})(\d{2})(\d{2})(\d{2})', file_name)
                if date_match:
                    year, month, day, hour = date_match.groups()
                    # 轉換為 datetime 物件以便格式化
                    try:
                        dt = datetime(2000 + int(year), int(month), int(day), int(hour))
                        # 格式化為 DD-MMM-YYYY HH:MM:SS
                        date_str_formatted = dt.strftime("%d-%b-%Y %H:%M:%S")
                    except ValueError:
                        # 日期解析失敗，使用原始格式
                        date_str_formatted = f"20{year}/{month}/{day} {hour}:00:00"
                else:
                    date_str_formatted = file_name
                
                if title_verbose:
                    # 使用原本的兩行標題
                    title_line1 = f"Inverted Resistivity Profile at {date_str_formatted}\n"
                    title_line2 = f"number of data={len(data['rhoa']):.0f}, rrms={rrms:.2f}%, $\chi^2$={chi2:.3f}"
                    
                    ax.set_title(title_line1, fontname='Times New Roman', fontsize=18, fontweight='bold')
                    ax.text(0.5, 1.02, title_line2, transform=ax.transAxes, ha='center', va='bottom',
                            fontname='Times New Roman', fontsize=12, fontweight='bold')
                else:
                    # 只顯示單行標題，格式為 DD-MMM-YYYY HH:MM:SS
                    ax.set_title(date_str_formatted, fontname='Times New Roman', fontsize=18, fontweight='bold')
            else:
                if title_verbose:
                    title_line1 = "Inverted Resistivity Profile\n"
                    title_line2 = f"rrms={rrms:.2f}%, $\chi^2$={chi2:.3f}"
                    
                    ax.set_title(title_line1, fontname='Times New Roman', fontsize=18, fontweight='bold')
                    ax.text(0.5, 1.02, title_line2, transform=ax.transAxes, ha='center', va='bottom',
                            fontname='Times New Roman', fontsize=12, fontweight='bold')
                else:
                    # 使用當前時間
                    now = datetime.now()
                    date_str_formatted = now.strftime("%d-%b-%Y %H:%M:%S")
                    ax.set_title(date_str_formatted, fontname='Times New Roman', fontsize=18, fontweight='bold')
            
            # 設置標籤
            ax.set_xlabel('Distance (m)', fontname='Times New Roman', fontsize=16, fontweight='bold')
            ax.set_ylabel('Elevation (m)', fontname='Times New Roman', fontsize=16, fontweight='bold')
            
            # 添加色條 - 使用 divider 設置色條與主圖等高
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.15)
            cb = plt.colorbar(cf, cax=cax)
            cb.ax.set_ylabel('$\Omega$-m', fontname='Times New Roman', fontsize=20)
            
            for label in cb.ax.get_yticklabels():
                label.set_fontname('Times New Roman')
                label.set_fontsize(12)
           
            # 生成對數刻度標籤，例如 10, 100, 1000
            if kwargs.get('logScale', True):
                ticks = np.logspace(log_vmin, log_vmax, 9)
                cb.set_ticks(np.log10(ticks))
                cb.set_ticklabels([f"{tick:.0f}" for tick in ticks])
            
            # 保存圖表
            save_file = os.path.join(result_path, 'inverted_contour_xzv.png')
            fig.savefig(save_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"等值線圖已保存: {save_file}")
            
            # 將內插結果也輸出為 xzv 文件
            xzv_output = self.export_to_xzv(result_path, new_X, new_Y, grid_data, file_name)
            self.logger.info(f"已輸出 XZV 文件: {xzv_output}")
                                 
            return True
            
        except Exception as e:
            self.logger.error(f"繪製等值線圖時發生錯誤: {str(e)}", exc_info=True)
            return False
    
    def export_to_xzv(self, result_path, X, Y, grid_data, file_name=None):
        """
        將內插後的結果輸出為 xzv 格式的檔案
        
        參數:
            result_path: 結果保存路徑
            X: X 座標網格
            Y: Y 座標網格
            grid_data: 電阻率對數值網格
            file_name: 檔案名稱，用於命名輸出檔案
        
        返回:
            輸出檔案的路徑
        """
        try:
            # 確定輸出檔案名稱，使用 YYMMDDHH.xzv 格式
            if file_name:
                # 先去除副檔名
                base_name = os.path.splitext(file_name)[0]
                
                # 如果檔名包含底線，嘗試從格式為 "YYMMDDHH_m_T1" 的檔名中提取時間戳
                if "_" in base_name:
                    time_part = base_name.split('_')[0]
                    # 檢查是否為 8 位數字 (YYMMDDHH 格式)
                    if time_part.isdigit() and len(time_part) == 8:
                        output_file = os.path.join(result_path, f"{time_part}.xzv")
                        self.logger.info(f"從檔名 '{base_name}' 提取時間戳: {time_part}")
                    else:
                        # 如果不是標準格式，使用原始正則表達式提取
                        date_match = re.match(r'(\d{2})(\d{2})(\d{2})(\d{2})', base_name)
                        if date_match:
                            year, month, day, hour = date_match.groups()
                            time_part = f"{year}{month}{day}{hour}"
                            output_file = os.path.join(result_path, f"{time_part}.xzv")
                            self.logger.info(f"使用正則表達式從檔名 '{base_name}' 提取時間戳: {time_part}")
                        else:
                            # 如果無法提取，使用原始檔名
                            output_file = os.path.join(result_path, f"{base_name}.xzv")
                            self.logger.warning(f"無法從 '{base_name}' 提取時間戳，使用完整檔名")
                else:
                    # 嘗試使用原始正則表達式提取
                    date_match = re.match(r'(\d{2})(\d{2})(\d{2})(\d{2})', base_name)
                    if date_match:
                        year, month, day, hour = date_match.groups()
                        time_part = f"{year}{month}{day}{hour}"
                        output_file = os.path.join(result_path, f"{time_part}.xzv")
                        self.logger.info(f"使用正則表達式從檔名 '{base_name}' 提取時間戳: {time_part}")
                    else:
                        # 如果無法提取，使用原始檔名
                        output_file = os.path.join(result_path, f"{base_name}.xzv")
                        self.logger.warning(f"無法從 '{base_name}' 提取時間戳，使用完整檔名")
            else:
                # 如果沒有提供文件名，使用當前時間
                now = datetime.now()
                output_file = os.path.join(result_path, f"{now.strftime('%y%m%d%H')}.xzv")
            
            # 參考文件的格式
            # 讀取參考文件以獲取網格點位置和數量
            ref_file = "src/refernce_code/25040708.xzv"
            if os.path.exists(ref_file):
                self.logger.info(f"找到參考文件 {ref_file}，將使用相同的格點配置")
                try:
                    # 讀取參考文件
                    ref_data = np.loadtxt(ref_file)
                    ref_x = ref_data[:, 0]
                    ref_y = ref_data[:, 1]
                    
                    # 使用參考文件的坐標創建相同的網格
                    unique_x = np.unique(ref_x)
                    unique_y = np.unique(ref_y)
                    
                    # 計算新網格數據 (將內插結果映射到參考網格)
                    # 確保我們的 grid_data 覆蓋參考網格的範圍
                    min_x, max_x = np.min(X), np.max(X)
                    min_y, max_y = np.min(Y), np.max(Y)
                    x_covered = np.logical_and(ref_x >= min_x, ref_x <= max_x)
                    y_covered = np.logical_and(ref_y >= min_y, ref_y <= max_y)
                    covered = np.logical_and(x_covered, y_covered)
                    
                    # 準備輸出數據
                    # 使用 "NaN" 文本而不是 -9999
                    output_data = []
                    for i in range(len(ref_x)):
                        x_val = ref_x[i]
                        y_val = ref_y[i]
                        
                        # 如果坐標在我們的網格範圍內，嘗試內插
                        if min_x <= x_val <= max_x and min_y <= y_val <= max_y:
                            # 找到最接近的網格點
                            x_idx = np.argmin(np.abs(np.linspace(min_x, max_x, X.shape[1]) - x_val))
                            y_idx = np.argmin(np.abs(np.linspace(min_y, max_y, Y.shape[0]) - y_val))
                            
                            # 獲取內插值
                            value = grid_data[y_idx, x_idx]
                            
                            # 如果是 NaN 值，使用文本 "NaN" 表示
                            if np.isnan(value):
                                line = f"{x_val:.4f} {y_val:.4f} NaN NaN"
                            else:
                                # 使用與參考文件相同的格式 (保留 IQR 值，如果原始值有的話)
                                line = f"{x_val:.4f} {y_val:.4f} {value:.6f} NaN"
                        else:
                            # 如果超出範圍，使用 NaN
                            line = f"{x_val:.4f} {y_val:.4f} NaN NaN"
                        
                        output_data.append(line)
                    
                    # 檢查是否生成了足夠的數據點
                    if len(output_data) < 4650:
                        self.logger.warning(f"生成的數據點數量 ({len(output_data)}) 少於參考文件 (4650)，將填充缺失點")
                        # 填充缺失點
                        while len(output_data) < 4650:
                            missing_idx = len(output_data)
                            if missing_idx < len(ref_data):
                                x_val = ref_data[missing_idx, 0]
                                y_val = ref_data[missing_idx, 1]
                                output_data.append(f"{x_val:.4f} {y_val:.4f} NaN NaN")
                            else:
                                # 如果超出參考數據範圍，使用最後一個點的 x+2, y 值
                                last_x, last_y = ref_data[-1, 0], ref_data[-1, 1]
                                output_data.append(f"{last_x+2:.4f} {last_y:.4f} NaN NaN")
                    
                    # 保存為 xzv 格式檔案
                    with open(output_file, 'w') as f:
                        f.write('\n'.join(output_data))
                    
                    self.logger.info(f"已將內插結果輸出為 XZV 檔案: {output_file}，共 {len(output_data)} 個數據點")
                    return output_file
                except Exception as e:
                    self.logger.error(f"處理參考文件時發生錯誤: {str(e)}")
                    # 如果參考文件處理失敗，使用默認方法繼續
            
            # 如果找不到參考文件或處理失敗，使用原有方法生成網格
            # 但確保使用 NaN 文本而不是數值
            self.logger.warning("使用默認方法生成 XZV 文件")
            
            # 生成一個固定的網格，確保有 4650 個數據點
            # 參考文件中 X 從 3.999 到 188.001，以 2.0 為步進
            # Y 從 1111.6398 到 1201.5941，共 46 個不同的值
            x_vals = np.linspace(3.999, 188.001, 93)
            y_vals = np.linspace(1111.6398, 1201.5941, 50)
            
            # 確保有 4650 個點 (93 x 50 = 4650)
            output_data = []
            for y in y_vals:
                for x in x_vals:
                    # 找到最接近的網格點
                    if X.size > 0 and Y.size > 0:  # 確保網格不為空
                        x_idx = np.argmin(np.abs(np.linspace(np.min(X), np.max(X), X.shape[1]) - x))
                        y_idx = np.argmin(np.abs(np.linspace(np.min(Y), np.max(Y), Y.shape[0]) - y))
                        
                        # 獲取內插值
                        if y_idx < grid_data.shape[0] and x_idx < grid_data.shape[1]:
                            value = grid_data[y_idx, x_idx]
                            
                            # 如果是 NaN 值，使用文本 "NaN" 表示
                            if np.isnan(value):
                                output_data.append(f"{x:.4f} {y:.4f} NaN NaN")
                            else:
                                output_data.append(f"{x:.4f} {y:.4f} {value:.6f} 0.00")
                        else:
                            output_data.append(f"{x:.4f} {y:.4f} NaN NaN")
                    else:
                        output_data.append(f"{x:.4f} {y:.4f} NaN NaN")
            
            # 保存為 xzv 格式檔案
            with open(output_file, 'w') as f:
                f.write('\n'.join(output_data))
            
            self.logger.info(f"已將內插結果輸出為 XZV 檔案: {output_file}，共 {len(output_data)} 個數據點")
            return output_file
            
        except Exception as e:
            self.logger.error(f"輸出 XZV 檔案時發生錯誤: {str(e)}", exc_info=True)
            return None 