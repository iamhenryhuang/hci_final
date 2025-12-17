"""
手勢識別追蹤模組
用於追蹤用戶不雅手勢次數，並根據次數決定是否啟用臉部馬賽克
"""

import json
import os
from datetime import datetime, date


class GestureTracker:
    """追蹤不雅手勢次數的後台管理類"""
    
    def __init__(self, data_file='gesture_log.json'):
        """
        初始化追蹤器
        
        Args:
            data_file: 儲存手勢記錄的 JSON 檔案路徑
        """
        self.data_file = data_file
        self.bad_gesture_count = 0
        self.face_mosaic_enabled = False
        self.threshold = 5  # 觸發臉部馬賽克的閾值
        self.today = str(date.today())
        self.load_data()
    
    def load_data(self):
        """從檔案載入今天的手勢記錄"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 檢查是否為今天的記錄
                if data.get('date') == self.today:
                    self.bad_gesture_count = data.get('bad_gesture_count', 0)
                    self.face_mosaic_enabled = data.get('face_mosaic_enabled', False)
                    print(f"載入今日記錄: {self.bad_gesture_count} 次不雅手勢")
                else:
                    # 新的一天，重置計數器
                    print("新的一天開始，重置計數器")
                    self._reset_daily_data()
            except (json.JSONDecodeError, IOError) as e:
                print(f"載入記錄失敗: {e}")
                self._reset_daily_data()
        else:
            print("首次使用，建立新記錄檔案")
            self._reset_daily_data()
    
    def _reset_daily_data(self):
        """重置每日數據"""
        self.bad_gesture_count = 0
        self.face_mosaic_enabled = False
        self.save_data()
    
    def save_data(self):
        """儲存手勢記錄到檔案"""
        data = {
            'date': self.today,
            'bad_gesture_count': self.bad_gesture_count,
            'face_mosaic_enabled': self.face_mosaic_enabled,
            'last_update': datetime.now().isoformat()
        }
        
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"儲存記錄失敗: {e}")
    
    def add_bad_gesture(self, gesture_name):
        """
        記錄一次不雅手勢
        
        Args:
            gesture_name: 手勢名稱
            
        Returns:
            bool: 是否觸發了臉部馬賽克
        """
        self.bad_gesture_count += 1
        print(f"[警告] 偵測到不雅手勢: {gesture_name} (今日第 {self.bad_gesture_count} 次)")
        
        # 檢查是否達到閾值
        if self.bad_gesture_count >= self.threshold and not self.face_mosaic_enabled:
            self.face_mosaic_enabled = True
            print(f"\n{'='*60}")
            print(f"!!! 警告：不雅手勢次數已達 {self.threshold} 次！啟動臉部馬賽克功能 !!!")
            print(f"{'='*60}\n")
        
        self.save_data()
        return self.face_mosaic_enabled
    
    def is_face_mosaic_enabled(self):
        """檢查是否啟用臉部馬賽克"""
        return self.face_mosaic_enabled
    
    def get_statistics(self):
        """獲取統計資訊"""
        return {
            'date': self.today,
            'bad_gesture_count': self.bad_gesture_count,
            'face_mosaic_enabled': self.face_mosaic_enabled,
            'remaining_warnings': max(0, self.threshold - self.bad_gesture_count)
        }
    
    def reset(self):
        """手動重置（僅供測試或管理員使用）"""
        print("手動重置計數器")
        self._reset_daily_data()
