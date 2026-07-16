import joblib
import pandas as pd
import numpy as np
import os

# 1. 声明 V3.9 核心概率收缩层 (必须有这段，模型才能认识)
class V3_9_ProbabilityLayer:
    def __init__(self, team_weight=0.6, player_weight=0.4, soft_shrink=0.8, penalty_shrink=0.7):
        self.tw = team_weight
        self.pw = player_weight
        self.ss = soft_shrink
        self.ps = penalty_shrink
        
    def predict(self, p_team, p_player, team_diffs, player_diffs):
        p_t, p_p = np.asarray(p_team), np.asarray(p_player)
        t_d, p_d = np.asarray(team_diffs), np.asarray(player_diffs)
        
        # 0.6/0.4 基础融合
        raw = (p_t * self.tw) + (p_p * self.pw)
        # 0.8 柔性收缩
        soft = 0.5 + (raw - 0.5) * self.ss
        # 防爆冷极端降权
        final = soft.copy()
        mask_penalty = (np.abs(t_d) > 15.0) & (np.abs(p_d) < 3.0)
        final[mask_penalty] = 0.5 + (soft[mask_penalty] - 0.5) * self.ps
        return final

def run_prediction(daily_data):
    """
    专门用于每天实盘推理的函数
    """
    print("🚨 启动 V3.9 实盘预测引擎...")
    
    # 2. 直接从当前目录加载你上传好的模型文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fusion_model = joblib.load(os.path.join(current_dir, 'v3_9_fusion_model.pkl'))
    prob_layer = joblib.load(os.path.join(current_dir, 'v3_9_probability_layer.pkl'))
    
    model_team = fusion_model["team_node"]
    model_player = fusion_model["player_node"]
    
    # 3. 提取当天的特征
    X_team = daily_data[['team_strength_diff', 'home_advantage', 'rest_days_diff', 'fatigue_diff']]
    X_player = daily_data[['player_impact_diff']]
    
    # 4. 执行预测
    p_team = model_team.predict_proba(X_team)[:, 1]
    p_player = model_player.predict_proba(X_player)[:, 1]
    
    p_final = prob_layer.predict(
        p_team, p_player, 
        daily_data['team_strength_diff'], 
        daily_data['player_impact_diff']
    )
    
    # 5. 生成结果
    daily_data['final_probability'] = np.round(p_final, 4)
    daily_data['prediction_side'] = np.where(p_final >= 0.5, "HOME", "AWAY")
    
    print("✅ 预测完成！")
    return daily_data
