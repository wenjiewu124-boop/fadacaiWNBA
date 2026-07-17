import joblib
import pandas as pd
import numpy as np
import os

class V3_9_ProbabilityLayer:
    def __init__(self, team_weight=0.6, player_weight=0.4, soft_shrink=0.8, penalty_shrink=0.7):
        self.tw = team_weight
        self.pw = player_weight
        self.ss = soft_shrink
        self.ps = penalty_shrink
        
    def predict(self, p_team, p_player, team_diffs, player_diffs):
        p_t, p_p = np.asarray(p_team), np.asarray(p_player)
        t_d, p_d = np.asarray(team_diffs), np.asarray(player_diffs)
        
        raw = (p_t * self.tw) + (p_p * self.pw)
        soft = 0.5 + (raw - 0.5) * self.ss
        final = soft.copy()
        mask_penalty = (np.abs(t_d) > 15.0) & (np.abs(p_d) < 3.0)
        final[mask_penalty] = 0.5 + (soft[mask_penalty] - 0.5) * self.ps
        return final

def run_prediction(daily_data):
    print("🚨 启动 V3.9 实盘预测引擎 (深度审计模式)...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fusion_model = joblib.load(os.path.join(current_dir, 'v3_9_fusion_model.pkl'))
    prob_layer = joblib.load(os.path.join(current_dir, 'v3_9_probability_layer.pkl'))
    
    model_team = fusion_model["team_node"]
    model_player = fusion_model["player_node"]
    
    feature_cols_team = ['team_strength_diff', 'home_advantage', 'rest_days_diff', 'fatigue_diff']
    feature_cols_player = ['player_impact_diff']
    
    daily_data[feature_cols_team + feature_cols_player] = daily_data[feature_cols_team + feature_cols_player].astype(float)
    X_team = daily_data[feature_cols_team]
    X_player = daily_data[feature_cols_player]
    
    print("\n" + "="*40)
    print("🔍 [阶段 1] 模型结构与标准化审计")
    print(f"▶️ 球队节点模型类型: {type(model_team)}")
    
    if hasattr(model_team, 'steps'):
        print(f"▶️ 检测到 Pipeline，包含处理步骤: {model_team.steps}")
    else:
        print("⚠️ 警告：模型是裸奔状态（非 Pipeline），极可能缺少特征标准化！")
        
    if hasattr(model_team, 'coef_'):
        print(f"▶️ 模型权重系数: {model_team.coef_}")

    print("\n" + "="*40)
    print("🧮 [阶段 2] Sigmoid 饱和度与 Raw Score 审计")
    
    if hasattr(model_team, 'decision_function'):
        raw_score_team = model_team.decision_function(X_team)
        print(f"\n🚨 喂给 Sigmoid 函数的 raw_score: \n   {raw_score_team}")
        
        if np.any(np.abs(raw_score_team) > 5.0):
            print("\n❌ 诊断报告: 触发【Sigmoid 饱和击穿】！Raw Score 绝对值太大，输出被迫成为 1.0。")
    else:
        print("\n⚠️ 模型不支持 decision_function，无法直接截获 raw_score。")
    
    print("\n" + "="*40)
    print("🎯 [阶段 3] 最终输出对照")
    
    p_team = model_team.predict_proba(X_team)[:, 1]
    p_player = model_player.predict_proba(X_player)[:, 1]
    
    print(f"▶️ p_team (球队节点胜率): {np.round(p_team, 4)}")
    print(f"▶️ p_player (球员节点胜率): {np.round(p_player, 4)}")
    
    p_final = prob_layer.predict(
        p_team, p_player, 
        daily_data['team_strength_diff'], 
        daily_data['player_impact_diff']
    )
    
    daily_data['final_probability'] = np.round(p_final, 4)
    daily_data['prediction_side'] = np.where(p_final >= 0.5, "HOME", "AWAY")
    
    print("=======================================\n")
    return daily_data
