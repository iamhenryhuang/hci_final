"""
幾何運算模組
負責處理向量角度計算與手部關鍵點角度轉換
"""

import math

def vector_2d_angle(v1, v2):
    """
    根據兩點的座標，計算角度
    
    Args:
        v1: 第一個向量 (x, y)
        v2: 第二個向量 (x, y)

    Returns:
        angle: 兩向量之間的角度（度數）
    """
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]

    try:
        # 使用餘弦定理計算角度
        angle_ = math.degrees(
            math.acos(
                (v1_x*v2_x + v1_y*v2_y) / 
                (((v1_x**2 + v1_y**2)**0.5) * ((v2_x**2 + v2_y**2)**0.5))
            )
        )
    except:
        angle_ = 180

    return angle_


def calculate_hand_angles(landmarks):
    """
    根據傳入的 21 個手部節點座標，計算五根手指的角度
    
    Args:
        landmarks: 包含 21 個手部關鍵點的座標列表 [(x, y), ...]
    
    Returns:
        list: 五根手指的角度列表 [大拇指, 食指, 中指, 無名指, 小拇指]
    """
    angle_list = []
    
    # 大拇指角度 (Node 0-2 vs 3-4)
    angle_ = vector_2d_angle(
        ((int(landmarks[0][0]) - int(landmarks[2][0])), (int(landmarks[0][1]) - int(landmarks[2][1]))),
        ((int(landmarks[3][0]) - int(landmarks[4][0])), (int(landmarks[3][1]) - int(landmarks[4][1])))
    )
    angle_list.append(angle_)
    
    # 食指角度 (Node 0-6 vs 7-8)
    angle_ = vector_2d_angle(
        ((int(landmarks[0][0]) - int(landmarks[6][0])), (int(landmarks[0][1]) - int(landmarks[6][1]))),
        ((int(landmarks[7][0]) - int(landmarks[8][0])), (int(landmarks[7][1]) - int(landmarks[8][1])))
    )
    angle_list.append(angle_)
    
    # 中指角度 (Node 0-10 vs 11-12)
    angle_ = vector_2d_angle(
        ((int(landmarks[0][0]) - int(landmarks[10][0])), (int(landmarks[0][1]) - int(landmarks[10][1]))),
        ((int(landmarks[11][0]) - int(landmarks[12][0])), (int(landmarks[11][1]) - int(landmarks[12][1])))
    )
    angle_list.append(angle_)
    
    # 無名指角度 (Node 0-14 vs 15-16)
    angle_ = vector_2d_angle(
        ((int(landmarks[0][0]) - int(landmarks[14][0])), (int(landmarks[0][1]) - int(landmarks[14][1]))),
        ((int(landmarks[15][0]) - int(landmarks[16][0])), (int(landmarks[15][1]) - int(landmarks[16][1])))
    )
    angle_list.append(angle_)
    
    # 小拇指角度 (Node 0-18 vs 19-20)
    angle_ = vector_2d_angle(
        ((int(landmarks[0][0]) - int(landmarks[18][0])), (int(landmarks[0][1]) - int(landmarks[18][1]))),
        ((int(landmarks[19][0]) - int(landmarks[20][0])), (int(landmarks[19][1]) - int(landmarks[20][1])))
    )
    angle_list.append(angle_)
    
    return angle_list
