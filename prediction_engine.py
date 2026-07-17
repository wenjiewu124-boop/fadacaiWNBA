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
    print("🚨 启动 V3.9 实盘预测引擎 (特征对齐与审计模式)...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fusion_model = joblib.load(os.path.join(current_dir, 'v3_9_fusion_model.pkl'))
    prob_layer = joblib.load(os.path.join(current_dir, 'v3_9_probability_layer.pkl'))
    
    model_team = fusion_model["team_node"]
    model_player = fusion_model["player_node"]
    
    # 原始我们以为的输入顺序
    current_team_cols = ['team_strength_diff', 'home_advantage', 'rest_days_diff', 'fatigue_diff']
    current_player_cols = ['player_impact_diff']
    
    daily_data[current_team_cols + current_player_cols] = daily_data[current_team_cols + current_player_cols].astype(float)
    
    print("\n" + "="*50)
    print("🔍 [阶段 1] 训练模型结构与特征顺序审计")

    def audit_and_align_model(model, node_name, current_cols):
        print(f"\n▶️ --- {node_name} 节点审计 ---")
        
        # 1. 检查结构与 Scaler
        is_pipeline = hasattr(model, 'steps')
        print(f"   是否为 Pipeline: {is_pipeline}")
        if is_pipeline:
            has_scaler = any('scaler' in step[0].lower() or 'standard' in str(type(step[1])).lower() for step in model.steps)
            print(f"   是否包含 StandardScaler: {has_scaler}")
            estimator = model.steps[-1][1]
        else:
            print("   是否包含 StandardScaler: False (裸奔模型)")
            estimator = model

        # 2. 提取训练时的真实特征顺序
        train_features = None
        if hasattr(estimator, 'feature_names_in_'):
            train_features = list(estimator.feature_names_in_)
        elif hasattr(model, 'feature_names_in_'):
            train_features = list(model.feature_names_in_)
        elif hasattr(estimator, 'get_booster'): # XGBoost 专属
            train_features = estimator.get_booster().feature_names
            
        print(f"   模型训练特征数量: {len(train_features) if train_features else '未知'}")
        print(f"   当前输入特征数量: {len(current_cols)}")
        print(f"   [预期] 模型训练的特征顺序: {train_features}")
        print(f"   [实际] 当前输入的特征顺序: {current_cols}")
        
        # 3. 建立兼容预测层：特征重排
        if train_features:
            if train_features == current_cols:
                print("   ✅ 诊断: 特征顺序完全一致。")
                aligned_cols = current_cols
            else:
                print("   ❌ 诊断: 特征顺序错位或缺失！已建立兼容层，强制按照训练顺序重排数据。")
                aligned_cols = train_features
        else:
            print("   ⚠️ 无法读取训练特征顺序，保持当前顺序。")
            aligned_cols = current_cols
            
        return aligned_cols

    # 获取重排后的正确列名
    aligned_team_cols = audit_and_align_model(model_team, "球队 (Team)", current_team_cols)
    aligned_player_cols = audit_and_align_model(model_player, "球员 (Player)", current_player_cols)

    print("\n" + "="*50)
    print("🧮 [阶段 2] 使用重排后数据执行预测")
    
    # 🚨 兼容层：强行按照模型要求的顺序截取和重排特征！
    # 如果发现缺少 Scaler，树模型(XGB/RF)不需要缩放；若是逻辑回归且确实漏了，后续再补。
    X_team_aligned = daily_data[aligned_team_cols]
    X_player_aligned = daily_data[aligned_player_cols]
    
    p_team = model_team.predict_proba(X_team_aligned)[:, 1]
    p_player = model_player.predict_proba(X_player_aligned)[:, 1]
    
    print(f"   修正后 p_team 概率: {np.round(p_team, 4)}")
    print(f"   修正后 p_player 概率: {np.round(p_player, 4)}")
    
    if np.std(p_team) > 0:
        print("\n   🎉 成功！概率已恢复差异！不同比赛的输入终于触发了不同的特征权重。")
    else:
        print("\n   ⚠️ 警告：概率仍然一样！可能该批次特征差异太小，未触发 XGBoost 决策树的阈值分支。")

    p_final = prob_layer.predict(
        p_team, p_player, 
        daily_data['team_strength_diff'], 
        daily_data['player_impact_diff']
    )
    
    daily_data['final_probability'] = np.round(p_final, 4)
    daily_data['prediction_side'] = np.where(p_final >= 0.5, "HOME", "AWAY")
    
    print("\n===== 🎯 最终预测结果预览 =====")
    print(daily_data[['game_id', 'home_team', 'away_team', 'final_probability']].to_string(index=False))
    
    return daily_data
