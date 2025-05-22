import sys
import os
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, 
                             QVBoxLayout, QLabel, QTextEdit, QScrollArea, QSplitter,
                             QGridLayout, QFrame)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer, pyqtSlot

class ResultViewer(QMainWindow):
    def __init__(self, result_dir='.', output_dir=None, info_file=None):
        super().__init__()
        
        self.result_dir = result_dir
        self.output_dir = output_dir or result_dir
        self.info_file = info_file
        self.latest_repeat_folder = None  # 存儲找到的最新反演次數資料夾
        self.last_check_time = 0  # 上次檢查時間
        
        if not self.info_file:
            # 嘗試尋找反演信息文件
            default_info_path = os.path.join(result_dir, "ERTManager","inv_info.txt")
            
            # 先檢查是否有反演時間資料夾
            recent_folder = self._find_most_recent_folder(result_dir)
            if recent_folder:
                # 尋找當前時間點下最新的反演結果
                latest_repeat = self._find_latest_repeat(os.path.join(result_dir, recent_folder))
                if latest_repeat:
                    self.latest_repeat_folder = os.path.join(recent_folder, latest_repeat)
                    recent_info_path = os.path.join(result_dir, recent_folder, latest_repeat, "ERTManager","inv_info.txt")
                    print(f"找到最新反演結果: {self.latest_repeat_folder}")
                    self.info_file = recent_info_path if os.path.exists(recent_info_path) else default_info_path
                else:
                    print(f"找不到{recent_folder}下的反演結果")
                    self.info_file = default_info_path
            else:
                self.info_file = default_info_path
        
        self.setWindowTitle('ERT反演結果查看器')
        self.setGeometry(100, 100, 1600, 900)  # 使用更大的默認窗口尺寸
        
        # 創建主窗口部件和佈局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)
        
        # 創建標題標籤，顯示當前資料夾路徑
        self.path_label = QLabel("尚未載入反演結果")
        self.path_label.setAlignment(Qt.AlignCenter)
        self.path_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.path_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 8px; border-radius: 4px; }")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)  # 允許用戶選擇文本
        
        # 添加標題標籤到主佈局
        main_layout.addWidget(self.path_label)
        
        # 創建分割視窗：上方圖表，下方文本信息
        splitter = QSplitter(Qt.Vertical)
        
        # 創建圖片顯示區
        self.images_scroll = QScrollArea()
        self.images_scroll.setWidgetResizable(True)
        self.images_widget = QWidget()
        self.grid_layout = QGridLayout(self.images_widget)
        self.grid_layout.setSpacing(10)  # 設置網格間距
        self.images_scroll.setWidget(self.images_widget)
        
        # 創建文本框以顯示inv_info.txt內容
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setFont(QFont("Courier New", 10))
        
        # 添加元素到分割視窗
        splitter.addWidget(self.images_scroll)
        splitter.addWidget(self.info_text)
        splitter.setSizes([700, 200])  # 設置初始分割比例
        
        # 添加分割視窗到主佈局
        main_layout.addWidget(splitter)
        
        # 載入結果圖片
        self.load_result_images()
        
        # 載入反演信息
        self.load_inv_info()
        
        # 更新路徑顯示
        self.update_path_label()
    
    def _find_most_recent_folder(self, base_dir):
        """尋找最新的反演時間資料夾"""
        try:
            folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and f[0].isdigit()]
            if not folders:
                return None
            # 按照數字排序（通常是時間戳）
            folders.sort(reverse=True)  # 倒序排列，最近的排在前面
            return folders[0]
        except:
            return None
    
    def _find_latest_repeat(self, folder_path):
        """尋找指定時間點資料夾中最新的反演次數（repeat_N）"""
        try:
            # 尋找所有repeat_N資料夾和intersection資料夾
            all_folders = [f for f in os.listdir(folder_path) 
                         if os.path.isdir(os.path.join(folder_path, f))]
            
            # 分離 repeat 和 intersection 資料夾
            repeat_folders = [f for f in all_folders if f.startswith("repeat_")]
            intersection_folder = "intersection" if "intersection" in all_folders else None
            
            # 檢查是否有交集反演資料夾
            if intersection_folder:
                ert_dir = os.path.join(folder_path, intersection_folder, "ERTManager")
                if os.path.exists(ert_dir):
                    # 檢查是否有必要的檔案
                    inv_info_path = os.path.join(ert_dir, "inv_info.txt")
                    if os.path.exists(inv_info_path):
                        # 檢查檔案的修改時間是否比上次檢查更新
                        mod_time = os.path.getmtime(inv_info_path)
                        if mod_time > self.last_check_time:
                            self.last_check_time = mod_time
                            print(f"找到新的或更新的交集反演結果")
                        return intersection_folder
            
            if not repeat_folders:
                return None
                
            # 按照反演次數排序
            def extract_repeat_num(folder_name):
                try:
                    return int(folder_name.split("_")[1])
                except:
                    return 0
                    
            repeat_folders.sort(key=extract_repeat_num, reverse=True)  # 由高到低排序
            
            # 優先檢查是否有最新的反演結果
            for repeat in repeat_folders:
                ert_dir = os.path.join(folder_path, repeat, "ERTManager")
                if os.path.exists(ert_dir):
                    # 檢查是否有必要的檔案
                    inv_info_path = os.path.join(ert_dir, "inv_info.txt")
                    if os.path.exists(inv_info_path):
                        # 檢查檔案的修改時間是否比上次檢查更新
                        mod_time = os.path.getmtime(inv_info_path)
                        if mod_time > self.last_check_time:
                            self.last_check_time = mod_time
                            print(f"找到新的或更新的反演結果: {repeat}")
                        return repeat
                    
                    # 即使沒有inv_info.txt，如果有其他關鍵檔案，也可以嘗試使用
                    if os.path.exists(os.path.join(ert_dir, "resistivity.vector")) or \
                       os.path.exists(os.path.join(ert_dir, "mesh.bms")):
                        return repeat
                    
            return None
        except Exception as e:
            print(f"尋找最新反演時出錯: {str(e)}")
            return None
    
    def _check_time_folder_has_updated(self, time_folder_path):
        """檢查指定的時間資料夾是否有更新的反演結果"""
        try:
            # 檢查是否是目錄
            if not os.path.isdir(time_folder_path):
                return False
                
            # 取得上次檢查的時間戳
            current_time = os.path.getmtime(time_folder_path)
            
            # 如果資料夾修改時間比上次檢查更新，則說明有更新
            if current_time > self.last_check_time:
                self.last_check_time = current_time
                return True
                
            # 檢查repeat資料夾和intersection資料夾
            all_folders = [f for f in os.listdir(time_folder_path) 
                         if os.path.isdir(os.path.join(time_folder_path, f))]
            
            for folder in all_folders:
                folder_path = os.path.join(time_folder_path, folder)
                mod_time = os.path.getmtime(folder_path)
                if mod_time > self.last_check_time:
                    self.last_check_time = mod_time
                    return True
                    
                # 進一步檢查ERTManager資料夾
                ert_dir = os.path.join(folder_path, "ERTManager")
                if os.path.exists(ert_dir):
                    mod_time = os.path.getmtime(ert_dir)
                    if mod_time > self.last_check_time:
                        self.last_check_time = mod_time
                        return True
            
            return False
        except Exception as e:
            print(f"檢查更新時出錯: {str(e)}")
            return False
    
    def _clear_layout(self, layout):
        """清空佈局中的所有元素"""
        if layout is None:
            return
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def update_path_label(self):
        """更新路徑標籤，顯示當前載入的資料夾絕對路徑"""
        if self.latest_repeat_folder:
            # 構建完整的絕對路徑
            folder_path = os.path.abspath(os.path.join(self.result_dir, self.latest_repeat_folder))
            self.path_label.setText(f"目前查看: {folder_path}")
        elif self.info_file and os.path.exists(self.info_file):
            # 如果沒有找到資料夾但有info文件，顯示info文件的路徑
            self.path_label.setText(f"目前查看: {os.path.abspath(self.info_file)}")
        else:
            self.path_label.setText("尚未載入反演結果")
    
    def load_result_images(self):
        """載入並顯示所有反演結果圖片"""
        # 清空現有網格佈局
        self._clear_layout(self.grid_layout)
        
        image_files = [
            "inverted_contour_xzv.png", 
            "inverted_profile.png", 
            "convergence.png", 
            "crossplot.png", 
            "misfit_histogram.png"
        ]
        
        # 檢查圖像來源 - 有兩種可能的位置
        image_paths = {}  # 用於存儲找到的圖像路徑
        found_images = False
        
        # 1. 如果有已識別的反演時間點和反演次數，優先從那裡加載
        if self.latest_repeat_folder:
            repeat_path = os.path.join(self.result_dir, self.latest_repeat_folder)
            print(f"正在從 {repeat_path} 載入圖片...")
            
            # 檢查目錄是否存在
            if not os.path.exists(repeat_path):
                print(f"警告：路徑不存在 {repeat_path}")
            else:
                image_count = 0
                
                # 列出目錄中的所有檔案，檢查是否有圖片文件
                existing_files = os.listdir(repeat_path)
                png_files = [f for f in existing_files if f.endswith('.png')]
                if png_files:
                    print(f"在 {repeat_path} 中找到以下圖片文件: {', '.join(png_files)}")
                
                for img_file in image_files:
                    file_path = os.path.join(repeat_path, img_file)
                    if os.path.exists(file_path):
                        image_paths[img_file] = file_path
                        image_count += 1
                        found_images = True
                        print(f"找到圖片: {file_path}")
                    else:
                        print(f"找不到圖片: {file_path}")
                
                if image_count > 0:
                    print(f"從 {self.latest_repeat_folder} 中載入了 {image_count} 張圖片")
                else:
                    print(f"在 {self.latest_repeat_folder} 中未找到任何圖片文件")
        
        # 2. 從output_dir中加載圖片（如果從反演資料夾中未找到或找到不完整）
        if not found_images or len(image_paths) < len(image_files):
            print(f"從輸出目錄 {self.output_dir} 尋找未找到的圖片...")
            
            # 列出輸出目錄中的所有文件
            if os.path.exists(self.output_dir):
                output_files = os.listdir(self.output_dir)
                png_files = [f for f in output_files if f.endswith('.png')]
                if png_files:
                    print(f"在輸出目錄中找到以下圖片文件: {', '.join(png_files)}")
            
            for img_file in image_files:
                if img_file not in image_paths:  # 只加載還未找到的圖片
                    file_path = os.path.join(self.output_dir, img_file)
                    if os.path.exists(file_path):
                        image_paths[img_file] = file_path
                        print(f"從輸出目錄載入圖片: {file_path}")
                    else:
                        print(f"在輸出目錄中也找不到圖片: {file_path}")
        
        # 計算最佳網格佈局
        total_images = len(image_paths)
        if total_images == 0:
            print("警告: 未找到任何可顯示的圖片")
            
            # 添加一個錯誤提示標籤
            error_label = QLabel("未找到任何可顯示的圖片，請檢查反演是否已完成。")
            error_label.setAlignment(Qt.AlignCenter)
            error_label.setFont(QFont("Arial", 12, QFont.Bold))
            error_label.setStyleSheet("color: red;")
            self.grid_layout.addWidget(error_label, 0, 0)
            return
            
        # 計算最佳的網格佈局行列數
        cols = min(3, total_images)  # 最多3列
        rows = (total_images + cols - 1) // cols  # 計算需要的行數
        
        # 將圖片添加到網格佈局中
        img_index = 0
        for img_file, file_path in image_paths.items():
            row = img_index // cols
            col = img_index % cols
            
            # 創建圖片容器框架
            frame = QFrame()
            frame.setFrameShape(QFrame.StyledPanel)
            frame.setFrameShadow(QFrame.Raised)
            frame_layout = QVBoxLayout(frame)
            
            # 創建標題標籤
            title = img_file.replace('.png', '').replace('_', ' ').title()
            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setFont(QFont("Arial", 10, QFont.Bold))
            
            # 創建圖片標籤
            img_label = QLabel()
            
            # 嘗試載入圖片，強制重新讀取文件（避免緩存問題）
            try:
                # 檢查文件是否已更改
                file_modified_time = os.path.getmtime(file_path)
                
                # 載入圖片
                pixmap = QPixmap(file_path)
                
                # 檢查pixmap是否為空
                if pixmap.isNull():
                    print(f"警告: 載入圖片 {file_path} 失敗，圖片可能損壞")
                    # 創建一個顯示錯誤的pixmap
                    pixmap = QPixmap(500, 300)
                    pixmap.fill(Qt.white)
                else:
                    print(f"成功載入圖片: {file_path}")
                    if pixmap.width() > 500:  # 縮放過大的圖片
                        pixmap = pixmap.scaled(500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                
                img_label.setPixmap(pixmap)
            except Exception as e:
                print(f"載入圖片 {file_path} 時出錯: {str(e)}")
                img_label.setText(f"無法載入圖片: {str(e)}")
            
            img_label.setAlignment(Qt.AlignCenter)
            
            # 添加到框架佈局
            frame_layout.addWidget(title_label)
            frame_layout.addWidget(img_label)
            
            # 添加到網格
            self.grid_layout.addWidget(frame, row, col)
            
            img_index += 1
    
    def load_inv_info(self):
        """載入並顯示inv_info.txt的內容"""
        if not self.info_file:
            self.info_text.setText("未設置資訊檔案路徑")
            print("警告: 未設置資訊檔案路徑")
            return
            
        print(f"嘗試載入資訊檔案: {self.info_file}")
        
        if os.path.exists(self.info_file):
            try:
                # 檢查文件大小，避免讀取空文件
                file_size = os.path.getsize(self.info_file)
                if file_size == 0:
                    self.info_text.setText(f"資訊檔案為空：{self.info_file}")
                    print(f"警告: 資訊檔案為空: {self.info_file}")
                    return
                
                # 檢查文件是否可讀
                if not os.access(self.info_file, os.R_OK):
                    self.info_text.setText(f"無法讀取資訊檔案（權限問題）：{self.info_file}")
                    print(f"警告: 無法讀取資訊檔案（權限問題）: {self.info_file}")
                    return
                
                # 讀取文件內容
                with open(self.info_file, 'r') as f:
                    info_content = f.read()
                
                # 檢查讀取到的內容是否為空
                if not info_content.strip():
                    self.info_text.setText(f"資訊檔案內容為空：{self.info_file}")
                    print(f"警告: 資訊檔案內容為空: {self.info_file}")
                    return
                
                # 設置文本內容
                self.info_text.setText(info_content)
                print(f"已載入反演資訊: {self.info_file}, 內容長度: {len(info_content)} 字符")
                
                # 強制刷新UI
                self.info_text.repaint()
                
            except Exception as e:
                error_msg = f"讀取資訊檔案時出錯：{str(e)}"
                self.info_text.setText(error_msg)
                print(f"錯誤: {error_msg}")
                import traceback
                traceback.print_exc()  # 印出詳細錯誤堆疊
        else:
            error_msg = f"找不到資訊檔案：{self.info_file}"
            self.info_text.setText(error_msg)
            print(f"錯誤: {error_msg}")
            
            # 嘗試檢查目錄是否存在
            dir_path = os.path.dirname(self.info_file)
            if not os.path.exists(dir_path):
                print(f"目錄不存在: {dir_path}")
            else:
                # 列出目錄中的文件
                try:
                    files = os.listdir(dir_path)
                    print(f"目錄 {dir_path} 中的文件: {', '.join(files)}")
                except Exception as e:
                    print(f"無法列出目錄內容: {str(e)}")
    
    @pyqtSlot()
    def refresh(self, force_check=False):
        """刷新顯示的內容
        
        參數:
            force_check: 是否強制檢查最新的反演結果
        """
        # 檢查是否需要尋找新的反演結果
        need_update = force_check
        
        # 檢查當前的反演結果是否仍然有效
        if self.info_file and not os.path.exists(self.info_file):
            print(f"資訊檔案不存在，需要更新: {self.info_file}")
            need_update = True
            
        # 檢查是否有新的反演時間點資料夾
        recent_folder = self._find_most_recent_folder(self.result_dir)
        if recent_folder:
            time_folder_path = os.path.join(self.result_dir, recent_folder)
            
            # 檢查這個時間點是否有更新
            if self._check_time_folder_has_updated(time_folder_path):
                print(f"偵測到時間點資料夾有更新: {time_folder_path}")
                need_update = True
                
            # 如果需要更新，或者我們還沒有找到最新的反演結果
            if need_update or self.latest_repeat_folder is None:
                # 尋找當前時間點下最新的反演結果
                latest_repeat = self._find_latest_repeat(time_folder_path)
                if latest_repeat:
                    new_repeat_folder = os.path.join(recent_folder, latest_repeat)
                    
                    # 如果找到了新的反演結果
                    if self.latest_repeat_folder != new_repeat_folder:
                        print(f"找到新的反演結果資料夾: {new_repeat_folder}")
                        self.latest_repeat_folder = new_repeat_folder
                        new_info_path = os.path.join(self.result_dir, new_repeat_folder, "ERTManager","inv_info.txt")
                        
                        if os.path.exists(new_info_path):
                            self.info_file = new_info_path
                            print(f"發現新的反演結果資訊檔: {self.info_file}")
                            need_update = True
                        else:
                            print(f"找不到新的反演結果資訊檔: {new_info_path}")
            
            # 如果有最新的反演結果資料夾，檢查是否有新的圖片文件
            if self.latest_repeat_folder:
                repeat_path = os.path.join(self.result_dir, self.latest_repeat_folder)
                if os.path.exists(repeat_path):
                    # 檢查是否有新的圖片文件
                    png_files = [f for f in os.listdir(repeat_path) if f.endswith('.png')]
                    if png_files:
                        # 檢查圖片文件的最後修改時間
                        last_modified = max([os.path.getmtime(os.path.join(repeat_path, f)) for f in png_files])
                        if last_modified > self.last_check_time:
                            print(f"偵測到新的圖片文件或圖片文件更新: {repeat_path}")
                            need_update = True
        
        # 如果需要更新，則重新載入圖片和資訊
        if need_update:
            print("開始重新載入圖片和資訊...")
            self.load_result_images()
            self.load_inv_info()
            self.update_path_label()  # 更新路徑標籤
            # 更新最後檢查時間
            self.last_check_time = time.time()
            return True
            
        return False
        
    def force_refresh(self):
        """強制刷新，用於反演完成時調用"""
        return self.refresh(force_check=True)

class ResultMonitor(ResultViewer):
    def __init__(self, result_dir='.', output_dir=None, info_file=None, refresh_interval=5000):
        super().__init__(result_dir, output_dir, info_file)
        # 設置定時器監控結果變更
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(refresh_interval)  # 每5秒檢查一次更新

def start_viewer(result_dir='.', output_dir=None, info_file=None, refresh_interval=5000, autorefresh=True):
    """啟動反演結果查看器（非阻塞方式）"""
    # 檢查環境中是否已有正在運行的 QApplication 實例
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # 根據是否需要自動刷新選擇查看器類型
    if autorefresh:
        viewer = ResultMonitor(result_dir, output_dir, info_file, refresh_interval)
    else:
        viewer = ResultViewer(result_dir, output_dir, info_file)
    
    viewer.show()
    
    # 返回查看器實例，以便於後續操作
    return viewer

# 如果直接運行此文件，則啟動獨立的查看器
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 如果提供了命令行參數，則使用指定的路徑
    result_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    viewer = ResultMonitor(result_dir)
    viewer.show()
    
    sys.exit(app.exec_()) 