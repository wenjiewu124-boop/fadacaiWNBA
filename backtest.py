import os
import pandas as pd
import numpy as np
from supabase import create_client
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
import prediction_engine

print("===== 🚀 V3.9 历史回测验证引擎启动 =====")

# 1. 连数据库
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

try:
    print("📡 1. 正在拉取 2023-2026 赛季历史数据...")
    
    # 拉取 V3 特征表 (最大 5000 条，覆盖近4年WNBA)
    res_v3 = supabase.table("Match_Fusion_Features_V3").select("*").limit(5000).execute()
    df_features = pd.DataFrame(res_v3.data)
    
    # 拉取基础表获取真实赛果 (is_home_win)
    res_v2 = supabase.table("WNBA_Game_Features_v2").select("match_id, is_home_win").limit(5000).execute()
    df_results = pd.DataFrame(res_v2.data)
    df_results.rename(columns={"match_id": "game_id"}, inplace=True)
    
    # 拼装回测集
    df = pd.merge(df_features, df_results, on="game_id", how="inner")
    
    if df.empty or 'is_home_win' not in df.columns:
        print("⚠️ 数据不足或缺失真实赛果(is_home_win)，请检查数据库映射。")
        exit(0)
        
    df.dropna(subset=['is_home_win'], inplace=True)
    total_games = len(df)
    print(f"✅ 成功构建回测数据集！有效比赛场次: {total_games}")

    # ========================================
    # 任务2：逐场模拟预测 (纯赛前特征)
    # ========================================
    print("⚙️ 2. 正在执行 V3.9 模型批量盲测...")
    
    # 借用 prediction_engine 的批处理能力
    df = prediction_engine.run_prediction(df)
    
    # 统一置信度指标 (预测方胜率)
    df['confidence'] = np.where(df['final_probability'] >= 0.5, df['final_probability'], 1 - df['final_probability'])
    df['pred_is_home_win'] = (df['final_probability'] >= 0.5).astype(int)
    df['is_correct'] = (df['pred_is_home_win'] == df['is_home_win']).astype(int)

    # ========================================
    # 任务3：计算模型指标与分层统计
    # ========================================
    print("🧮 3. 正在计算 Brier Score 与置信度分布...")
    
    accuracy = accuracy_score(df['is_home_win'], df['pred_is_home_win'])
    brier = brier_score_loss(df['is_home_win'], df['final_probability'])
    logloss = log_loss(df['is_home_win'], df['final_probability'])

    # 分层统计
    bins = [0.50, 0.55, 0.60, 0.65, 0.70, 1.01]
    labels = ['50%-55%', '55%-60%', '60%-65%', '65%-70%', '70%以上']
    df['prob_bin'] = pd.cut(df['confidence'], bins=bins, labels=labels, right=False)
    
    strat_stats = df.groupby('prob_bin', observed=False).agg(
        count=('game_id', 'count'),
        avg_prob=('confidence', 'mean'),
        hit_rate=('is_correct', 'mean')
    ).fillna(0)

    # ========================================
    # 任务4：重点检查高置信度 (>=65% 和 >=70%)
    # ========================================
    high_conf_65 = df[df['confidence'] >= 0.65]
    high_conf_70 = df[df['confidence'] >= 0.70]
    
    acc_65 = high_conf_65['is_correct'].mean() if not high_conf_65.empty else 0.0
    acc_70 = high_conf_70['is_correct'].mean() if not high_conf_70.empty else 0.0
    
    errors = high_conf_65[high_conf_65['is_correct'] == 0]

    # ========================================
    # 任务5：生成终极报告
    # ========================================
    print("\n\n===== 📊 V3.9 历史回测报告 =====")
    print("【数据量】")
    print(f"比赛数量: {total_games} 场 (涵盖 2023-2026 赛季)")
    
    print("\n【全局质量指标】")
    print(f"Accuracy (胜负准确率): {accuracy:.4f} (期望 > 0.60 为佳)")
    print(f"Brier Score (概率质量): {brier:.4f} (越接近0越好，0.25为瞎猜基线)")
    print(f"Log Loss (对数损失): {logloss:.4f}")
    
    print("\n【分层统计分布】")
    print("概率区间      | 比赛数量 | 预测胜率 | 实际命中率")
    print("-" * 50)
    for index, row in strat_stats.iterrows():
        print(f"{index:<13} | {int(row['count']):<8} | {row['avg_prob']:.4f}   | {row['hit_rate']:.4f}")

    print("\n【高概率区间表现】")
    print(f">=65% (场次: {len(high_conf_65)}): 命中率 {acc_65:.4f}")
    print(f">=70% (场次: {len(high_conf_70)}): 命中率 {acc_70:.4f}")
    
    print("\n【高置信度错误案例 (Top 5 爆冷)】")
    if not errors.empty:
        print(errors[['game_date', 'home_team', 'away_team', 'confidence']].head(5).to_string(index=False))
    else:
        print("未产生 >= 65% 的爆冷案例！")

    print("\n===== 🏁 最终状态判定 =====")
    if accuracy >= 0.62 and brier <= 0.23:
        status = "A. 🟢 可进入实盘阶段 (模型具备强大的区分度与概率校准能力)"
    elif accuracy >= 0.58 and brier > 0.23:
        status = "B. 🟡 需要校准概率 (胜负预测尚可，但概率偏激进或保守，建议增加 Platt Scaling)"
    else:
        status = "C. 🔴 需要重新训练 (准确率低于基线或分布崩溃，需重构模型或审视训练集)"
    
    print(f"模型状态: {status}")
    print("===================================\n")

except Exception as e:
    print(f"❌ 运行报错: {e}")

exit(0)
