import os
import pandas as pd
import numpy as np
from supabase import create_client
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
import prediction_engine
from prediction_engine import V3_9_ProbabilityLayer # 解决 pickle 依赖

print("===== 🚀 V3.9 历史回测引擎启动 =====")

# 1. 初始化数据库连接
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

start_date = "2023-01-01"
end_date = "2026-07-17"

try:
    print(f"📡 1. 正在拉取 {start_date} 至 {end_date} 历史回测数据...")
    
    # 抽取 V3 特征大表 (历史全量)
    # Supabase 单次查询限制可通过 limit 放大，假设历史比赛约 1500 场
    res_features = supabase.table("Match_Fusion_Features_V3").select(
        "game_id, game_date, home_team, away_team, team_strength_diff, player_impact_diff, rest_days_diff, fatigue_diff, home_advantage"
    ).gte("game_date", start_date).lte("game_date", end_date).limit(3000).execute()
    
    df_features = pd.DataFrame(res_features.data)
    
    # 抽取 V2 基础表获取赛后真实赛果 (is_home_win)
    res_results = supabase.table("WNBA_Game_Features_v2").select(
        "match_id, is_home_win"
    ).gte("match_date_bj", start_date).lte("match_date_bj", end_date).limit(3000).execute()
    
    df_results = pd.DataFrame(res_results.data)
    df_results.rename(columns={"match_id": "game_id"}, inplace=True)
    
    # 合并特征与赛果
    df = pd.merge(df_features, df_results, on="game_id", how="inner")
    df.dropna(subset=['is_home_win'], inplace=True) # 过滤尚未完赛或无结果的场次
    df['is_home_win'] = df['is_home_win'].astype(int)
    
    total_games = len(df)
    if total_games == 0:
        print("⚠️ 未匹配到包含真实赛果的历史数据，程序退出。")
        exit(0)
        
    print(f"✅ 成功构建回测集！有效比赛总数: {total_games} 场")

    # ==========================================
    # 2. 逐场预测：调用生产模型 (完全模拟赛前)
    # ==========================================
    print("⚙️ 2. 正在调用 V3.9 生产模型进行批量推演...")
    
    # 直接使用 prediction_engine 进行推演，保证跟实盘逻辑 100% 一致
    df = prediction_engine.run_prediction(df)
    
    # 计算预测方 (预测胜率较高的那一侧)
    df['confidence'] = np.where(df['final_probability'] >= 0.5, df['final_probability'], 1 - df['final_probability'])
    df['pred_is_home_win'] = (df['final_probability'] >= 0.5).astype(int)
    
    # 核心字段：打分对比真实结果
    df['correct'] = (df['pred_is_home_win'] == df['is_home_win']).astype(int)
    df['actual_result'] = np.where(df['is_home_win'] == 1, "HOME", "AWAY")

    # 保存逐场 CSV 报告
    report_cols = [
        "game_id", "game_date", "home_team", "away_team", 
        "final_probability", "prediction_side", "actual_result", "correct"
    ]
    df[report_cols].sort_values("game_date").to_csv("V3.9_Backtest_Report.csv", index=False, encoding="utf-8-sig")
    print("💾 成功生成单场逐日报告: V3.9_Backtest_Report.csv")

    # ==========================================
    # 3. 计算回测指标 & 分层统计
    # ==========================================
    print("🧮 3. 正在计算全局质量指标与分层统计...")
    
    accuracy = accuracy_score(df['is_home_win'], df['pred_is_home_win'])
    brier = brier_score_loss(df['is_home_win'], df['final_probability'])
    logloss = log_loss(df['is_home_win'], df['final_probability'])

    # 概率分层
    bins = [0.50, 0.55, 0.60, 0.65, 0.70, 1.01]
    labels = ['50%-55%', '55%-60%', '60%-65%', '65%-70%', '70%以上']
    df['prob_bin'] = pd.cut(df['confidence'], bins=bins, labels=labels, right=False)
    
    strat_stats = df.groupby('prob_bin', observed=False).agg(
        count=('game_id', 'count'),
        avg_prob=('confidence', 'mean'),
        hit_rate=('correct', 'mean')
    ).fillna(0)

    # 重点排查高置信度 (>=65%)
    high_conf = df[df['confidence'] >= 0.65]
    high_count = len(high_conf)
    high_hit_rate = high_conf['correct'].mean() if high_count > 0 else 0.0
    
    errors_65 = high_conf[high_conf['correct'] == 0]
    
    # ==========================================
    # 4. 生成诊断 Summary (TXT) & 终端输出
    # ==========================================
    summary_lines = []
    summary_lines.append("===== WNBA-Quant-Engine V3.9 历史回测报告 =====")
    summary_lines.append(f"\n【数据量】\n比赛数量: {total_games} 场 (2023-01-01 至 {end_date})")
    
    summary_lines.append("\n【全局概率质量指标】")
    summary_lines.append(f"Accuracy (胜负准确率): {accuracy:.4f}")
    summary_lines.append(f"Brier Score (概率得分): {brier:.4f}")
    summary_lines.append(f"Log Loss (对数损失): {logloss:.4f}")
    
    summary_lines.append("\n【分层区间表现】")
    summary_lines.append(f"{'概率区间':<10} | {'比赛数量':<6} | {'预测胜率':<8} | {'实际命中率':<8}")
    summary_lines.append("-" * 45)
    for index, row in strat_stats.iterrows():
        summary_lines.append(f"{index:<14} | {int(row['count']):<8} | {row['avg_prob']:.4f}   | {row['hit_rate']:.4f}")
        
    summary_lines.append("\n【高置信度表现 (重点检查)】")
    summary_lines.append(f">=65% 比赛数量: {high_count}")
    summary_lines.append(f">=65% 实际命中率: {high_hit_rate:.4f}")
    
    summary_lines.append("\n[ 爆冷错判列表 (>=65% 但实际亏损) ]")
    if not errors_65.empty:
        for _, err in errors_65.iterrows():
            summary_lines.append(f"[{err['game_date']}] {err['home_team']} vs {err['away_team']} | 预测: {err['prediction_side']} ({err['confidence']:.4f}) | 结果错判")
    else:
        summary_lines.append("完美！未发生 >=65% 的爆冷错判。")

    summary_lines.append("\n===== 🏁 终极状态诊断 =====")
    
    # 自动化规则判定
    if accuracy >= 0.62 and brier <= 0.23:
        status = "A. 🟢 可进入实盘阶段 (模型区分度强大，概率收缩层校准优秀)"
    elif accuracy >= 0.58 and brier > 0.23:
        status = "B. 🟡 需要概率校准 (胜负判断尚可，但 Brier 分数不及格，需外挂 Platt Scaling / Isotonic Regression)"
    else:
        status = "C. 🔴 需要重新训练 (Accuracy 过低或数据分布漂移导致特征失效)"
        
    summary_lines.append(f"模型状态判断: {status}")
    summary_lines.append("==============================================")

    summary_text = "\n".join(summary_lines)
    
    with open("V3.9_Backtest_Summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
        
    print(summary_text)
    print("\n💾 成功生成报告文件: V3.9_Backtest_Summary.txt")

except Exception as e:
    print(f"❌ 运行报错: {e}")

exit(0)
