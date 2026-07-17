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
        
        print("\n   --- 🕵️‍♂️ [内部探针] 进入概率收缩层 ---")
        print(f"   [诊断] p_team (球队节点原始胜率): {np.round(p_t, 4)}")
        print(f"   [诊断] p_player (球员节点原始胜率): {np.round(p_p, 4)}")

        # 0.6/0.4 基础融合
        raw = (p_t * self.tw) + (p_p * self.pw)
        print(f"   [诊断] raw (基础融合得分): {np.round(raw, 4)}")
        
        # 0.8 柔性收缩
        soft = 0.5 + (raw - 0.5) * self.ss
        print(f"   [诊断] soft (0.8柔性收缩后): {np.round(soft, 4)}")
        
        # 防爆冷极端降权
        final = soft.copy()
        mask_penalty = (np.abs(t_d) > 15.0) & (np.abs(p_d) < 3.0)
        final[mask_penalty] = 0.5 + (soft[mask_penalty] - 0.5) * self.ps
        
        print(f"   [诊断] final (最终输出概率): {np.round(final, 4)}")
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
    
    # --- 🛡️ 增加数据类型强转防线 (防止数据库读出的是字符串) ---
    feature_cols_team = ['team_strength_diff', 'home_advantage', 'rest_days_diff', 'fatigue_diff']
    feature_cols_player = ['player_impact_diff']
    
    # 强制转为浮点数，防止 sklearn 底层运算崩溃或溢出
    daily_data[feature_cols_team + feature_cols_player] = daily_data[feature_cols_team + feature_cols_player].astype(float)
    
    # 3. 提取当天的特征
    X_team = daily_data[feature_cols_team]
    X_player = daily_data[feature_cols_player]
    
    print("\n===== 🔍 V3.9 预测公式白盒诊断 =====")
    print("📊 [特征快照] 喂入模型的 X_team (应为真实波动数值):")
    print(X_team.to_string(index=False))
    print("\n📊 [特征快照] 喂入模型的 X_player (应为真实波动数值):")
    print(X_player.to_string(index=False))
    
    # 4. 执行预测
    print("\n🧮 正在调用底层 sklearn 节点预测...")
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
    
    print("\n✅ 预测完成！")
    return daily_data
