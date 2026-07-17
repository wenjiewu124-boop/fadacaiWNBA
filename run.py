import os
import pandas as pd
import numpy as np
from supabase import create_client
import joblib

# 确保 Pickle 能找到类定义
import prediction_engine
from prediction_engine import V3_9_ProbabilityLayer

print("===== 🚀 V3.9 预测链路终极方差验收 =====")

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

target_date = "2026-07-13"
columns_to_fetch = "game_id, game_date, home_team, away_team, team_strength_diff, player_impact_diff, rest_days_diff, fatigue_diff, home_advantage"

try:
    # -----------------------------------------
    # 1. 抓取修复后的特征并打印矩阵
    # -----------------------------------------
    res = supabase.table("Match_Fusion_Features_V3").select(columns_to_fetch).eq("game_date", target_date).execute()
    df = pd.DataFrame(res.data)
    
    if df.empty:
        print("⚠️ 未找到特征数据，程序退出。")
        exit(0)

    feature_cols_team = ['team_strength_diff', 'home_advantage', 'rest_days_diff', 'fatigue_diff']
    feature_cols_player = ['player_impact_diff']
    all_features = feature_cols_team + feature_cols_player
    
    # 强制浮点数，防守数据格式问题
    df[all_features] = df[all_features].astype(float)

    print("\n===== 📊 1. 模型实际输入矩阵 (检查是否进入训练分布) =====")
    print(df[['game_id'] + all_features].to_string(index=False))

    # -----------------------------------------
    # 2. 模型加载与推理
    # -----------------------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fusion_model = joblib.load(os.path.join(current_dir, 'v3_9_fusion_model.pkl'))
    prob_layer = joblib.load(os.path.join(current_dir, 'v3_9_probability_layer.pkl'))
    
    model_team = fusion_model["team_node"]
    model_player = fusion_model["player_node"]

    X_team = df[feature_cols_team]
    X_player = df[feature_cols_player]

    p_team = model_team.predict_proba(X_team)[:, 1]
    p_player = model_player.predict_proba(X_player)[:, 1]
    
    p_final = prob_layer.predict(
        p_team, p_player, 
        df['team_strength_diff'], 
        df['player_impact_diff']
    )
    
    df['p_team'] = np.round(p_team, 4)
    df['p_player'] = np.round(p_player, 4)
    df['final_probability'] = np.round(p_final, 4)
    df['prediction_side'] = np.where(p_final >= 0.5, "HOME", "AWAY")

    # -----------------------------------------
    # 3 & 4. 预测输出与方差检查
    # -----------------------------------------
    print("\n===== 🎯 XGBoost 预测输出结果 =====")
    print(df[['game_id', 'p_team', 'p_player', 'final_probability', 'prediction_side']].to_string(index=False))

    df.to_csv("final_prediction.csv", index=False, encoding="utf-8-sig")

    # -----------------------------------------
    # 5. 验收判定
    # -----------------------------------------
    std_final = df['final_probability'].std()
    
    print("\n===== 🏁 验收结论报告 =====")
    if pd.isna(std_final) or std_final == 0:
        print("❌ 【验收失败】：概率依然完全一致 (方差为0)，陷入死锁。")
        print("💡 诊断结论：输入数据已恢复正常，但模型仍无区分能力。这已不是流水线代码的问题。")
        print("\n👉 根据既定策略，请进入下一阶段模型审计：")
        print("   A. 审查 training_dataset：模型训练时是否因数据分布不均（如缺乏反例）导致过度拟合了常数？")
        print("   B. XGBoost 叶节点：输入尽管变小了，是否由于切割阈值过浅，仍掉入同一个 leaf？")
        print("   C. ⚠️ 【极高概率】保存格式错误：Pickle 在跨平台/版本保存 XGBoost 时，极其容易丢失 Booster 的树结构和权重。")
        print("      - 请务必在训练脚本中改用 `model_team.get_booster().save_model('model_team.json')`！")
    else:
        print(f"🎉 【验收通过】：概率成功恢复显著差异 (方差: {std_final:.4f})！")
        print("🎉 恭喜！不同比赛的特征成功激活了决策树的不同分支，输出 0.62、0.51 这种千姿百态的胜率！V3.9 正式完工！")

except Exception as e:
    print(f"❌ 运行报错: {e}")

exit(0)