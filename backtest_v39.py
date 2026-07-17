import os
import pandas as pd
import numpy as np
import joblib
from supabase import create_client
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

# 1. 声明 V3.9 核心概率收缩层 (确保 Pickle 正常加载依赖)
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

print("===== 🚀 V3.9 Historical Backtest Engine 启动 =====")

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

start_date = "2023-01-01"
end_date = "2026-07-17"

try:
    # ==========================================
    # 任务 1: 拉取历史数据与赛果
    # ==========================================
    print(f"📡 正在拉取 {start_date} 至 {end_date} 历史回测数据...")
    
    # 获取特征表 (假设数据量不超过单次限制，可通过分页拉取全量)
    res_features = supabase.table("Match_Fusion_Features_V3").select(
        "game_id, game_date, home_team, away_team, team_strength_diff, player_impact_diff, rest_days_diff, fatigue_diff, home_advantage"
    ).gte("game_date", start_date).lte("game_date", end_date).limit(5000).execute()
    df_features = pd.DataFrame(res_features.data)
    
    # 获取真实赛果
    res_results = supabase.table("WNBA_Game_Features_v2").select(
        "match_id, is_home_win"
    ).gte("match_date_bj", start_date).lte("match_date_bj", end_date).limit(5000).execute()
    df_results = pd.DataFrame(res_results.data).rename(columns={"match_id": "game_id"})
    
    # 合并组装
    df = pd.merge(df_features, df_results, on="game_id", how="inner")
    df.dropna(subset=['is_home_win'], inplace=True)
    df['is_home_win'] = df['is_home_win'].astype(int)
    
    total_games = len(df)
    if total_games == 0:
        print("⚠️ 未匹配到有效历史赛果，请检查数据库。")
        exit(0)
    print(f"✅ 成功构建回测集！有效比赛总数: {total_games} 场")

    # ==========================================
    # 任务 2: 加载模型与逐场预测
    # ==========================================
    print("⚙️ 正在加载当前生产模型并执行批量预测...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fusion_model = joblib.load(os.path.join(current_dir, 'v3_9_fusion_model.pkl'))
    prob_layer = joblib.load(os.path.join(current_dir, 'v3_9_probability_layer.pkl'))
    
    model_team = fusion_model["team_node"]
    model_player = fusion_model["player_node"]
    
    feature_cols_team = ['team_strength_diff', 'home_advantage', 'rest_days_diff', 'fatigue_diff']
    feature_cols_player = ['player_impact_diff']
    
    df[feature_cols_team + feature_cols_player] = df[feature_cols_team + feature_cols_player].astype(float)
    
    X_team = df[feature_cols_team]
    X_player = df[feature_cols_player]
    
    p_team = model_team.predict_proba(X_team)[:, 1]
    p_player = model_player.predict_proba(X_player)[:, 1]
    
    # 计算最终主胜概率
    df['home_win_prob'] = prob_layer.predict(p_team, p_player, df['team_strength_diff'], df['player_impact_diff'])
    
    # 组织输出字段
    df['final_probability'] = np.where(df['home_win_prob'] >= 0.5, df['home_win_prob'], 1 - df['home_win_prob'])
    df['final_probability'] = np.round(df['final_probability'], 4)
    df['prediction_side'] = np.where(df['home_win_prob'] >= 0.5, "HOME", "AWAY")
    df['actual_result'] = np.where(df['is_home_win'] == 1, "HOME", "AWAY")
    df['correct'] = (df['prediction_side'] == df['actual_result']).astype(int)

    # 💾 生成 CSV 报告
    csv_cols = ["game_id", "game_date", "home_team", "away_team", "final_probability", "prediction_side", "actual_result", "correct"]
    df[csv_cols].sort_values("game_date").to_csv("V3.9_Backtest_Report.csv", index=False, encoding="utf-8-sig")

    # ==========================================
    # 任务 3: 计算全局指标
    # ==========================================
    print("🧮 正在计算量化回测指标...")
    accuracy = accuracy_score(df['is_home_win'], (df['home_win_prob'] >= 0.5).astype(int))
    brier = brier_score_loss(df['is_home_win'], df['home_win_prob'])
    logloss = log_loss(df['is_home_win'], df['home_win_prob'])

    # ==========================================
    # 任务 4: 概率分桶统计
    # ==========================================
    bins = [0.50, 0.55, 0.60, 0.65, 0.70, 1.01]
    labels = ['50%-55%', '55%-60%', '60%-65%', '65%-70%', '70%以上']
    df['prob_bin'] = pd.cut(df['final_probability'], bins=bins, labels=labels, right=False)
    
    strat_stats = df.groupby('prob_bin', observed=False).agg(
        count=('game_id', 'count'),
        avg_prob=('final_probability', 'mean'),
        hit_rate=('correct', 'mean')
    ).fillna(0)

    # ==========================================
    # 任务 5: 高置信度排查 (>=0.65)
    # ==========================================
    high_conf = df[df['final_probability'] >= 0.65]
    hc_count = len(high_conf)
    hc_hit_rate = high_conf['correct'].mean() if hc_count > 0 else 0.0
    hc_errors = high_conf[high_conf['correct'] == 0]

    # ==========================================
    # 任务 6: 组装 TXT 报告与评级
    # ==========================================
    report = []
    report.append("===== V3.9 历史回测报告 =====")
    report.append(f"\n【回测概况】\n时间范围: {start_date} 到 {end_date}")
    report.append(f"总比赛数: {total_games} 场")
    
    report.append("\n【全局概率质量指标】")
    report.append(f"Accuracy:    {accuracy:.4f}  (>0.6为优)")
    report.append(f"Brier Score: {brier:.4f}  (越接近0越好, 盲猜基线0.25)")
    report.append(f"Log Loss:    {logloss:.4f}")
    
    report.append("\n【分层概率区间统计】")
    report.append(f"{'概率区间':<10} | {'样本数量':<6} | {'预测平均概率':<10} | {'真实命中率':<8}")
    report.append("-" * 55)
    for index, row in strat_stats.iterrows():
        report.append(f"{index:<14} | {int(row['count']):<8} | {row['avg_prob']:.4f}       | {row['hit_rate']:.4f}")
        
    report.append("\n【重点检查高置信度 (>= 65%)】")
    report.append(f"高概率样本数量: {hc_count} 场")
    report.append(f"实际命中胜率:   {hc_hit_rate:.4f}")
    report.append("\n[ 失败案例清单 (Top 10) ]")
    if not hc_errors.empty:
        for _, err in hc_errors.head(10).iterrows():
            report.append(f"- {err['game_date']} | {err['home_team']} vs {err['away_team']} | 预测: {err['prediction_side']} ({err['final_probability']:.4f}) | 结果: 错判")
    else:
        report.append("- 完美表现，无失败案例！")

    report.append("\n===== 🏁 模型终极评级 =====")
    
    # 根据数据科学通用量化基准进行自动评级
    # WNBA的商业模型及格线一般在 Acc 0.58, 优秀在 0.62+
    if accuracy >= 0.61 and brier <= 0.23:
        rating = "A. 🟢 可以进入实盘预测 (模型区分度优秀，且概率分布高度校准，符合生产环境要求)"
    elif accuracy >= 0.58 and brier > 0.23:
        rating = "B. 🟡 需要概率校准 (胜负判断在及格线之上，但概率偏向极端，建议外挂 Platt Scaling)"
    else:
        rating = "C. 🔴 需要重新训练 (Accuracy 过低或数据特征在历史回测中表现无效，存在分布漂移)"
        
    report.append(f"模型评级: {rating}")
    report.append("==================================")

    # 输出到终端并写入 TXT
    report_text = "\n".join(report)
    print(report_text)
    
    with open("V3.9_Backtest_Summary.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
        
    print("\n💾 全部报表生成完毕：")
    print("1. V3.9_Backtest_Report.csv (含逐场详细明细)")
    print("2. V3.9_Backtest_Summary.txt (含终极数据研报)")

except Exception as e:
    print(f"❌ 运行报错: {e}")

exit(0)
