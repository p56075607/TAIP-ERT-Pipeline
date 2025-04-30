"""反演模組 (載入 URF→轉為 ohm、去 r‑index、網格生成 → 多次迭代反演)"""

import os
import logging
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
import matplotlib.pyplot as plt
import scipy.io
from . import utils
from . import visualization

class ERTInverter:
    """處理 ERT 資料反演的類別，使用 PyGIMLi 進行反演"""
    
    def __init__(self, inversion_config):
        """
        初始化 ERT 資料反演器
        
        參數:
            inversion_config: 包含反演相關配置的字典
        """
        self.config = inversion_config
        self.root_dir = inversion_config.get("root_dir", "")
        self.repeat_times = inversion_config.get("repeat_times", 4)
        self.lam = inversion_config.get("lam", 1000)
        self.z_weight = inversion_config.get("z_weight", 1)
        self.max_iter = inversion_config.get("max_iter", 6)
        self.resistivity_limits = inversion_config.get("limits", [1, 10000])
        self.relative_error = inversion_config.get("relative_error", 0.03)
        self.remove_channels = inversion_config.get("remove_channels", [20, 36, 52])
        self.colormap_file = inversion_config.get("colormap_file", None)
        self.xzv_file = inversion_config.get("xzv_file", None)
        self.output_dir = inversion_config.get("output_dir", os.path.join(self.root_dir, "output"))
        self.title_verbose = inversion_config.get("title_verbose", False)
        
        # 設置日誌 - 只使用文件處理器，避免重複輸出到控制台
        self.logger = logging.getLogger('inversion')
        self.logger.setLevel(logging.INFO)
        
        # 只有在沒有任何處理器時才添加處理器
        if not self.logger.handlers:
            # 添加文件處理器
            log_file = os.path.join(self.root_dir, 'inversion.log')
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%y/%m/%d - %H:%M:%S')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
        # 初始化視覺化器
        visualization_config = {"root_dir": self.root_dir, "colormap_file": self.colormap_file}
        
        # 檢查 colormap_file 是否存在
        if self.colormap_file and os.path.exists(self.colormap_file):
            self.logger.info(f"使用色彩圖文件: {self.colormap_file}")
        else:
            self.logger.warning(f"色彩圖文件不存在或未指定: {self.colormap_file}")
            # 嘗試使用默認的 colormap_file
            default_colormap = os.path.join(self.root_dir, 'EIclm.mat')
            if os.path.exists(default_colormap):
                self.logger.info(f"使用默認色彩圖文件: {default_colormap}")
                visualization_config["colormap_file"] = default_colormap
        
        # 檢查 xzv_file 是否存在
        if self.xzv_file and os.path.exists(self.xzv_file):
            self.logger.info(f"使用 XZV 文件: {self.xzv_file}")
        else:
            self.logger.warning(f"XZV 文件不存在或未指定: {self.xzv_file}")
        
        self.visualizer = visualization.ERTVisualizer(visualization_config)
        
    def run_inversion(self, urf_files):
        """對所有指定的 URF 檔案進行反演"""
        results = []
        
        for urf_file in urf_files:
            self.logger.info(f"開始處理檔案: {os.path.basename(urf_file)}")
            result = self.invert_single_file(urf_file)
            if result:
                results.append(result)
        
        return results
    
    def invert_single_file(self, urf_file):
        """對單個 URF 檔案進行反演"""
        try:
            # 1. 轉換 URF 為 ohm 格式
            trn_path = self.config.get("trn_path", None)
            ohm_path = utils.convertURF(urf_file, has_trn=trn_path is not None, trn_path=trn_path)
            
            # 2. 載入資料
            data = pg.load(ohm_path)
            self.logger.info(f"載入資料: {ohm_path}")
            
            # 3. 去除 r-index (如果有提供)
            ridx_mat_path = self.config.get("ridx_mat_path", None)
            if ridx_mat_path and os.path.exists(ridx_mat_path):
                self.logger.info(f"使用 r-index 矩陣: {ridx_mat_path}")
                mat = scipy.io.loadmat(ridx_mat_path)
                ridx = np.array(mat['ridx']).T[0]
                ridx = ridx.astype(bool)
                self.logger.info(f"去除 r-index: {len(ridx[ridx == True])} 個點")
                data.remove(ridx)
            
            # 4. 去除問題通道
            for ch in self.remove_channels:
                t2 = data['a'] == ch
                index = [i for i, x in enumerate(t2) if x]
                self.logger.info(r'remove a == ch{:d} {:d}'.format(ch,len(index)))
                data.remove(data['a'] == ch)

                t2 = data['b'] == ch
                index = [i for i, x in enumerate(t2) if x]
                self.logger.info(r'remove b == ch{:d} {:d}'.format(ch,len(index)))
                data.remove(data['b'] == ch)

                t2 = data['m'] == ch
                index = [i for i, x in enumerate(t2) if x]
                self.logger.info(r'remove m == ch{:d} {:d}'.format(ch,len(index)))
                data.remove(data['m'] == ch)

                t2 = data['n'] == ch
                index = [i for i, x in enumerate(t2) if x]
                self.logger.info(r'remove n == ch{:d} {:d}'.format(ch,len(index)))
                data.remove(data['n'] == ch)
            
            # 5. 計算幾何因子與視電阻率
            data['k'] = ert.createGeometricFactors(data, numerical=True)
            data['rhoa'] = data['k'] * data['r']
            
            # 6. 過濾異常值
            # 移除負值
            t2 = data['rhoa'] < 0
            index = [i for i, x in enumerate(t2) if x]
            self.logger.info(r'remove negative rho_a: {:d}'.format(len(index)))
            data.remove(t2)
            
            # 移除過大值
            t2 = data['rhoa'] > 10000
            index = [i for i, x in enumerate(t2) if x]
            self.logger.info(r'remove rho_a > 10000: {:d}'.format(len(index)))
            data.remove(index)

            # # Skip data by a skip step
            # skip_step = 10
            # remove_index = np.array([x for x in np.arange(len(data['rhoa']))])
            # remove_index = remove_index % skip_step != 0
            # self.logger.info(f'skip data by step: {skip_step}')
            # data.remove(remove_index)
            
            # 7. 設置誤差估計
            data['err'] = ert.estimateError(data, relativeError=self.relative_error)
            self.logger.info(f"資料處理完成，剩餘資料點數: {len(data['rhoa'])}")

            # 8. 創建網格
            mesh = self._create_mesh(data)
            self.logger.info(f"網格創建完成，單元數: {len([i for i, x in enumerate(mesh.cellMarker() == 2) if x])}")
            
            # 9. 開始反演
            mgr = ert.ERTManager(data)
            
            file_basename = os.path.basename(urf_file).split('.')[0]
            result_data = {"file": urf_file, "manager": mgr, "iterations": []}
            
            # 多次迭代反演
            for i in range(self.repeat_times):
                self.logger.info(f"開始第 {i+1}/{self.repeat_times} 次反演")
                self.logger.info(f"{mesh} paraDomain cell#: {len([i for i, x in enumerate(mesh.cellMarker() == 2) if x])}")
                self.logger.info(f"{data} invert data#: {len(data['rhoa'])}")
                
                # 進行反演
                mgr.invert(data, mesh,
                          lam=self.lam, zWeight=self.z_weight,
                          maxIter=self.max_iter,
                          limits=self.resistivity_limits,
                          verbose=True)
                
                # 計算誤差統計
                rrms = mgr.inv.relrms()
                chi2 = mgr.inv.chi2()
                self.logger.info(f"反演結果: rrms={rrms:.2f}%, chi²={chi2:.3f}")
                
                # 保存本次反演結果
                save_path = os.path.join(self.root_dir, 'output', file_basename, f'repeat_{i+1}')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                # 保存反演結果
                path, fig, ax = mgr.saveResult(save_path) # 會自動創建 ERTManager 文件夾
                plt.close(fig)
                
                # 保存資料和輸出反演資訊
                mgr_dir = os.path.join(save_path, 'ERTManager')
                    
                # 保存使用的資料
                data_file_name = f"{file_basename}_inv.ohm"
                mgr.data.save(os.path.join(mgr_dir, data_file_name))
                self.logger.info(f"已儲存反演資料: {data_file_name}")
                                         
                # 輸出模型響應
                pg.utils.saveResult(os.path.join(mgr_dir, 'model_response.txt'),
                                  data=mgr.inv.response, mode='w')
                
                # 輸出反演資訊
                self._export_inversion_info(mgr, save_path, self.lam)
                self.logger.info("已儲存反演資訊")
                
                # 記錄檔案名稱和路徑，供後續處理使用
                file_name = os.path.basename(urf_file)
                
                # 立即進行視覺化處理，允許用戶查看每次反演的結果
                self.logger.info(f"正在為第 {i+1}/{self.repeat_times} 次反演生成視覺化結果")
                self.visualizer.load_inversion_results(save_path)
                # 標記這是即時視覺化，避免與最終處理重複
                # 傳遞 xzv_file 參數，使其能執行 plot_inverted_contour_xzv 方法
                plot_kwargs = {
                    'is_realtime': True,
                    'title_verbose': self.title_verbose
                }
                if self.xzv_file and os.path.exists(self.xzv_file):
                    self.logger.info(f"使用 XZV 文件進行可視化: {self.xzv_file}")
                    self.visualizer.plot_all(save_path, file_name, self.xzv_file, **plot_kwargs)
                else:
                    self.logger.info("未使用 XZV 文件，使用默認可視化方法")
                    self.visualizer.plot_all(save_path, file_name, **plot_kwargs)
                self.logger.info(f"第 {i+1}/{self.repeat_times} 次反演的視覺化結果已生成")
                
                # 保存迭代資訊
                result_data["iterations"].append({
                    "repeat": i+1,
                    "rrms": rrms,
                    "chi2": chi2,
                    "save_path": save_path,
                    "is_last": (i == self.repeat_times - 1),  # 標記是否是最後一次反演
                    "visualized": True  # 標記此迭代已經進行過視覺化
                })
                
                # 如果不是最後一次反演，移除擬合不佳的資料
                if i != self.repeat_times - 1:
                    remain_per = 0.9  # 保留90%的資料
                    # 計算實際值和模型預測值之間的相對誤差，作為誤差指標
                    # 由於沒有 'misfit' 欄位，我們使用模型回應與實際測量值的相對差異
                    actual_data = data['rhoa']
                    predicted_data = mgr.inv.response
                    rel_error = np.abs((actual_data - predicted_data) / actual_data) * 100  # 相對誤差百分比
                    
                    t1 = np.argsort(rel_error)[int(np.round(remain_per * len(data['rhoa']), 0)):]
                    remove_index = np.full((len(data['rhoa'])), False)
                    for j in range(len(t1)):
                        remove_index[t1[j]] = True
                    
                    self.logger.info(f"移除 {int(100*(1-remain_per))}% 最差的擬合資料，剩餘資料 {len(data['rhoa'])-len(t1)}")
                    self.logger.info(f"移除 {len(t1)} 個不良擬合點，誤差閾值 {rel_error[t1[0]]:.2f}%")
                    data.remove(remove_index)
                else:
                    # 如果是最後一次反演，執行檔案複製操作
                    self.logger.info("最後一次反演完成，準備複製檔案到輸出目錄")
                    
                    # 確定輸出目錄
                    if hasattr(self, 'output_dir') and self.output_dir:
                        output_dir = self.output_dir
                    else:
                        # 如果 ERTInverter 沒有這些設定，從根目錄推測
                        output_dir = os.path.join(self.root_dir, 'output')
                        
                    # 確保輸出目錄存在
                    profile_dir = os.path.join(output_dir, "profile")
                    xzv_dir = os.path.join(output_dir, "xzv")
                    os.makedirs(profile_dir, exist_ok=True)
                    os.makedirs(xzv_dir, exist_ok=True)
                    
                    # 從 URF 檔案名稱中提取時間資訊 (YYMMDDHH)
                    urf_basename = os.path.basename(urf_file).split('.')[0]
                    time_part = urf_basename.split('_')[0]  # 獲取 "YYMMDDHH" 部分
                    
                    # 複製等值線圖
                    inverted_contour_file = os.path.join(save_path, 'inverted_contour_xzv.png')
                    if os.path.exists(inverted_contour_file):
                        # 構建目標檔案路徑，使用時間戳命名
                        dst_img_path = os.path.join(profile_dir, f"{time_part}.png")
                        
                        # 複製檔案
                        import shutil
                        shutil.copy2(inverted_contour_file, dst_img_path)
                        self.logger.info(f"已將等值線圖複製到: {dst_img_path}")
                    
                    # 複製 XZV 檔案
                    # 輸出 XZV 檔案路徑
                    xzv_output_file = os.path.join(save_path, f"{time_part}.xzv")
                    if os.path.exists(xzv_output_file):
                        # 構建目標檔案路徑，使用時間戳命名
                        dst_xzv_path = os.path.join(xzv_dir, f"{time_part}.xzv")
                        
                        # 複製檔案
                        import shutil
                        shutil.copy2(xzv_output_file, dst_xzv_path)
                        self.logger.info(f"已將 XZV 檔案複製到: {dst_xzv_path}")
            
            return result_data
            
        except Exception as e:
            self.logger.error(f"反演過程中發生錯誤: {str(e)}", exc_info=True)
            return None
    
    def _create_mesh(self, data):
        """創建反演網格"""
        # 篩選高於某個高度的電極點
        elevation_threshold = self.config.get("elevation_threshold", 1195)
        mask = pg.z(data) > elevation_threshold
        data_surface = pg.DataContainerERT()
        
        # 創建傳感器
        for i in range(len(pg.x(data)[mask])):
            data_surface.createSensor(pg.Pos(pg.x(data)[mask][i], 0.0, pg.z(data)[mask][i]))
        
        # 創建網格
        left = min(pg.x(data_surface))
        right = max(pg.x(data_surface))
        length = right - left
        depth = self.config.get("mesh_depth", 90)
        
        # 建立 PLC
        para_dx = self.config.get("para_dx", 1/4)
        para_max_cell_size = self.config.get("para_max_cell_size", 5)
        
        plc = mt.createParaMeshPLC(data_surface, paraDX=para_dx, paraMaxCellSize=para_max_cell_size,
                                  paraDepth=depth, balanceDepth=False)
        
        # 添加傳感器點到 PLC
        for i in range(len(data.sensorPositions())):
            plc.createNode(pg.Pos(data.sensorPositions()[i][0], data.sensorPositions()[i][2]))
            
        # 創建網格
        mesh = mt.createMesh(plc)
        self.logger.info(f"paraDomain cell#: {len([i for i, x in enumerate(mesh.cellMarker() == 2) if x])}")
        return mesh
    
    def _export_inversion_info(self, mgr, save_path, lam):
        """
        匯出反演相關資訊到文字檔
        
        參數:
            mgr: ERT管理器
            save_path: 儲存路徑
            lam: 正則化參數
        """
        # 確保ERTManager目錄存在
        mgr_dir = os.path.join(save_path, 'ERTManager')
        if not os.path.exists(mgr_dir):
            os.makedirs(mgr_dir)
            
        information_ph = os.path.join(mgr_dir, 'inv_info.txt')
        with open(information_ph, 'w') as write_obj:
            write_obj.write('## Final result ##\n')
            write_obj.write('rrms:{}\n'.format(mgr.inv.relrms()))
            write_obj.write('chi2:{}\n'.format(mgr.inv.chi2()))

            write_obj.write('## Inversion parameters ##\n')
            write_obj.write('use lam:{}\n'.format(lam))

            write_obj.write('## Iteration ##\n')
            write_obj.write('Iter.  rrms  chi2\n')
            rrmsHistory = mgr.inv.rrmsHistory
            chi2History = mgr.inv.chi2History
            for iter in range(len(rrmsHistory)):
                write_obj.write('{:.0f},{:.2f},{:.2f}\n'.format(iter, rrmsHistory[iter], chi2History[iter])) 