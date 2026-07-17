import os
import pandas as pd
import numpy as np
import joblib
from supabase import create_client
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

# 1. 声明 V3.9 核心概率收缩层 (必须保留，用于 Pickle 加载)
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

try:
    # ==========================================
    # 1. 粗暴且安全的拉取与映射 (确保命中1000+数据)
    # ==========================================
    print("📡 正在全量加载特征库与真实赛果，准备内存联表...")
    
    # 抽取特征表全集
    res_features = supabase.table("Match_Fusion_Features_V3").select("*").limit(5000).execute()
    df_features = pd.DataFrame(res_features.data)
    
    # 抽取基础表全集 (获取真实赛果)
    res_results = supabase.table("WNBA_Game_Features_v2").select(
        "match_id, match_date_bj, home_score, away_score, is_home_win"
    ).limit(5000).execute()
    df_results = pd.DataFrame(res_results.data)
    
    if df_features.empty or df_results.empty:
        raise ValueError("数据库表为空，拉取失败！")

    # 统一主键并执行强制内存合并 (Inner Join)
    df_results.rename(columns={"match_id": "game_id"}, inplace=True)
    df = pd.merge(df_features, df_results, on="game_id", how="inner")
    
    # 统一赛果计算
    if 'is_home_win' not in df.columns or df['is_home_win'].isnull().all():
        df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
        df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')
        df['is_home_win'] = (df['home_score'] > df['away_score']).astype(int)
    else:
        # 清理可能的 None 或非 int 格式
        df['is_home_win'] = pd.to_numeric(df['is_home_win'], errors='coerce')
        
    df.dropna(subset=['is_home_win'], inplace=True)
    df['is_home_win'] = df['is_home_win'].astype(int)

    total_games = len(df)
    if total_games < 900:
        print(f"⚠️ 警告：当前合并后的有效比赛数为 {total_games}，不满足 >900 的预期要求，请检查数据回填状态。")
    else:
        print(f"✅ 数据关联完美闭环！有效比赛总数: {total_games} 场")

    # ==========================================
    # 2. V3.9 模型批量推演
    # ==========================================
    print("⚙️ 正在执行 XGBoost 矩阵推理...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fusion_model = joblib.load(os.path.join(current_dir, 'v3_9_fusion_model.pkl'))
    prob_layer = joblib.load(os.path.join(current_dir, 'v3_9_probability_layer.pkl'))
    
    X_team = df[['team_strength_diff', 'home_advantage', 'rest_days_diff', 'fatigue_diff']].astype(float)
    X_player = df[['player_impact_diff']].astype(float)
    
    p_team = fusion_model["team_node"].predict_proba(X_team)[:, 1]
    p_player = fusion_model["player_node"].predict_proba(X_player)[:, 1]
    
    # 推理最终概率
    df['home_win_prob'] = prob_layer.predict(p_team, p_player, df['team_strength_diff'], df['player_impact_diff'])
    df['final_probability'] = np.where(df['home_win_prob'] >= 0.5, df['home_win_prob'], 1 - df['home_win_prob'])
    df['prediction_side'] = np.where(df['home_win_prob'] >= 0.5, "HOME", "AWAY")
    df['actual_result'] = np.where(df['is_home_win'] == 1, "HOME", "AWAY")
    df['correct'] = (df['prediction_side'] == df['actual_result']).astype(int)

    # 导出明细
    df[['game_id', 'game_date', 'home_team', 'away_team', 'final_probability', 'prediction_side', 'actual_result', 'correct']].to_csv("V3.9_Backtest_Report.csv", index=False)

    # ==========================================
    # 3. 统计计算与终极报告生成
    # ==========================================
    acc = accuracy_score(df['is_home_win'], (df['home_win_prob'] >= 0.5).astype(int))
    br = brier_score_loss(df['is_home_win'], df['home_win_prob'])
    ll = log_loss(df['is_home_win'], df['home_win_prob'])
    
    # 概率分桶
    bins = [0.50, 0.55, 0.60, 0.65, 0.70, 1.01]
    labels = ['50%-55%', '55%-60%', '60%-65%', '65%-70%', '70%以上']
    df['prob_bin'] = pd.cut(df['final_probability'], bins=bins, labels=labels, right=False)
    
    # 分桶明细计算
    stats = df.groupby('prob_bin', observed=False).agg(
        count=('game_id', 'count'),
        avg_prob=('final_probability', 'mean'),
        hit_rate=('correct', 'mean')
    ).fillna(0)
    
    # 高置信度计算
    high_conf = df[df['final_probability'] >= 0.65]
    hc_count = len(high_conf)
    hc_hit = high_conf['correct'].mean() if hc_count > 0 else 0.0
    hc_errors = high_conf[high_conf['correct'] == 0]

    # === 生成终极研报文案 ===
    report = []
    report.append("\n===== V3.9_Backtest_Summary =====")
    report.append(f"总比赛数: {total_games}")
    report.append(f"Accuracy: {acc:.4f}")
    report.append(f"Brier Score: {br:.4f}")
    report.append(f"Log Loss: {ll:.4f}")
    
    report.append("\n[概率分桶]")
    report.append(f"{'概率区间':<10} | {'样本数量':<6} | {'平均预测概率':<10} | {'真实命中率':<8}")
    report.append("-" * 55)
    for index, row in stats.iterrows():
        report.append(f"{index:<14} | {int(row['count']):<8} | {row['avg_prob']:.4f}       | {row['hit_rate']:.4f}")
        
    report.append("\n[高置信度(>=65%)统计]")
    report.append(f"数量: {hc_count}")
    report.append(f"命中率: {hc_hit:.4f}")
    report.append("\n错误案例:")
    if not hc_errors.empty:
        report.append(hc_errors[['game_date', 'home_team', 'away_team', 'final_probability']].to_string(index=False))
    else:
        report.append("无错误案例")

    report.append("\n===== 🏁 模型评级 =====")
    # 评级判决逻辑
    if acc >= 0.61 and br <= 0.23:
        rating = "A: 可以进入实盘预测 (Accuracy >=0.61 且 Brier <=0.23，指标优秀)"
        next_step = "进入实盘预测阶段。"
    elif acc >= 0.58 and br > 0.23:
        rating = "B: 需要概率校准 (Accuracy达标但概率分布失真)"
        next_step = "进入概率校准阶段 (Platt Scaling)。"
    else:
        rating = "C: 需要重新训练 (核心指标崩塌)"
        next_step = "退回特征工程与模型训练阶段。"
        
    report.append(f"评级结果: {rating}")
    report.append(f"下一步动作: {next_step}")
    report.append("=================================")

    report_str = "\n".join(report)
    print(report_str)
    
    with open("V3.9_Backtest_Summary.txt", "w", encoding="utf-8") as f:
        f.write(report_str)
        
except Exception as e:
    print(f"❌ 运行报错: {e}")

exit(0)
