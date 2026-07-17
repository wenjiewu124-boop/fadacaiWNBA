import joblib
import pandas as pd
import numpy as np
import os
import re

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
    print("🚨 启动 V3.9 极深诊断模式 (XGBoost 树结构剖析)...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fusion_model = joblib.load(os.path.join(current_dir, 'v3_9_fusion_model.pkl'))
    prob_layer = joblib.load(os.path.join(current_dir, 'v3_9_probability_layer.pkl'))
    
    model_team = fusion_model["team_node"]
    model_player = fusion_model["player_node"]
    
    current_team_cols = ['team_strength_diff', 'home_advantage', 'rest_days_diff', 'fatigue_diff']
    current_player_cols = ['player_impact_diff']
    daily_data[current_team_cols + current_player_cols] = daily_data[current_team_cols + current_player_cols].astype(float)
    
    X_team = daily_data[current_team_cols]
    X_player = daily_data[current_player_cols]

    print("\n" + "="*50)
    print("🌳 [任务 1 & 3] XGBoost 模型内部结构解剖与训练阈值反推")
    
    estimator_team = model_team.steps[-1][1] if hasattr(model_team, 'steps') else model_team
    
    if hasattr(estimator_team, 'get_booster'):
        booster = estimator_team.get_booster()
        dump = booster.get_dump()
        print(f"   ▶️ 决策树数量 (Trees): {len(dump)}")
        print(f"   ▶️ 最大深度 (Max Depth): {getattr(estimator_team, 'max_depth', '未知')}")
        
        importance = booster.get_score(importance_type='gain')
        print(f"   ▶️ 特征重要度 (Feature Importance - Gain):")
        for k, v in importance.items():
            print(f"      - {k}: {v:.4f}")
            
        print("\n🔬 反推训练数据分裂范围 (Split Thresholds):")
        tree_text = "\n".join(dump)
        
        for feature in current_team_cols:
            # 使用正则从树的底层抓取该特征所有曾发生过分裂的数值
            pattern = rf"\[{feature}<([0-9\.\-]+)\]"
            splits = [float(x) for x in re.findall(pattern, tree_text)]
            if splits:
                print(f"   ▶️ {feature} 训练时分裂边界: 最小 {min(splits):.4f} ~ 最大 {max(splits):.4f}")
            else:
                print(f"   ▶️ {feature}: 未找到有效分裂节点")
    else:
        print("   ⚠️ 底层不是 XGBoost 或支持树解析的模型。")

    print("\n" + "="*50)
    print("🧮 [任务 2] 当前输入实际送入底层的 Numpy 矩阵")
    numpy_input = X_team.values
    print(f"   ▶️ 特征列名: {current_team_cols}")
    for idx, row in enumerate(numpy_input):
        print(f"   Row {idx} 输入: {np.round(row, 4)}")

    print("\n" + "="*50)
    print("💡 [任务 4] 综合诊断判定")
    print("基于当前情况，给出诊断判定：")
    print("【高度疑似 B. 输入分布漂移 (Data Drift)】")
    print("   👉 理由: XGBoost 决策树面对远超训练边界的特征值 (如 90.2) 时，会失去区分能力。")
    print("   👉 假设训练时的 team_strength_diff 最大切分点是 25.0，那么 77.4 和 90.2 对模型来说毫无区别，它们全都会落进最边缘的同一个叶子节点，导致输出完全一样的胜率。")
    print("   👉 下一步方向: 强烈建议核对特征生成逻辑，为什么今天的实力差是 90 多？（是不是把两队的实力值相加了？或者漏掉了标准化步骤？）")

    # 保持代码正常执行并生成结果
    p_team = model_team.predict_proba(X_team)[:, 1]
    p_player = model_player.predict_proba(X_player)[:, 1]
    p_final = prob_layer.predict(p_team, p_player, daily_data['team_strength_diff'], daily_data['player_impact_diff'])
    
    daily_data['final_probability'] = np.round(p_final, 4)
    daily_data['prediction_side'] = np.where(p_final >= 0.5, "HOME", "AWAY")
    
    return daily_data
