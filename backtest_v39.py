import os
import pandas as pd
import numpy as np
import joblib
from supabase import create_client
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

# 1. 声明 V3.9 核心概率收缩层 (必须保留，用于恢复 Pickle 对象)
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
    # 0. 动态嗅探数据库与赛果映射
    print("🔍 正在嗅探 WNBA_Game_Features_v2 表结构...")
    res_schema = supabase.table("WNBA_Game_Features_v2").select("*").limit(1).execute()
    db_columns = list(res_schema.data[0].keys())
    
    # 自动定位赛果字段
    result_query_cols = "match_id"
    if "is_home_win" in db_columns:
        result_query_cols += ", is_home_win"
    elif "home_score" in db_columns:
        result_query_cols += ", home_score, away_score"
    else:
        raise ValueError("无法在数据库中定位到赛果字段")

    # 1. 拉取特征与赛果
    print(f"📡 正在拉取 {start_date} 至 {end_date} 回测数据...")
    df_features = pd.DataFrame(supabase.table("Match_Fusion_Features_V3").select("*").gte("game_date", start_date).lte("game_date", end_date).limit(5000).execute().data)
    df_results = pd.DataFrame(supabase.table("WNBA_Game_Features_v2").select(result_query_cols).gte("match_date_bj", start_date).lte("match_date_bj", end_date).limit(5000).execute().data)
    df_results.rename(columns={"match_id": "game_id"}, inplace=True)
    
    # 标准化赛果处理
    if "is_home_win" not in df_results.columns:
        df_results['is_home_win'] = (df_results['home_score'] > df_results['away_score']).astype(int)
    
    df = pd.merge(df_features, df_results[['game_id', 'is_home_win']], on="game_id", how="inner")
    print(f"✅ 有效比赛总数: {len(df)} 场")

    # 2. 模型推理
    fusion_model = joblib.load('v3_9_fusion_model.pkl')
    prob_layer = joblib.load('v3_9_probability_layer.pkl')
    
    X_team = df[['team_strength_diff', 'home_advantage', 'rest_days_diff', 'fatigue_diff']]
    X_player = df[['player_impact_diff']]
    
    p_team = fusion_model["team_node"].predict_proba(X_team)[:, 1]
    p_player = fusion_model["player_node"].predict_proba(X_player)[:, 1]
    
    df['final_probability'] = prob_layer.predict(p_team, p_player, df['team_strength_diff'], df['player_impact_diff'])
    df['prediction_side'] = np.where(df['final_probability'] >= 0.5, "HOME", "AWAY")
    df['actual_result'] = np.where(df['is_home_win'] == 1, "HOME", "AWAY")
    df['correct'] = (df['prediction_side'] == df['actual_result']).astype(int)

    # 3. 生成 Report CSV
    df[['game_id', 'game_date', 'home_team', 'away_team', 'final_probability', 'prediction_side', 'actual_result', 'correct']].to_csv("V3.9_Backtest_Report.csv", index=False)

    # 4. 计算指标与分层
    acc = accuracy_score(df['is_home_win'], (df['final_probability'] >= 0.5).astype(int))
    br = brier_score_loss(df['is_home_win'], df['final_probability'])
    ll = log_loss(df['is_home_win'], df['final_probability'])
    
    df['prob_bin'] = pd.cut(df['final_probability'], bins=[0.5, 0.55, 0.6, 0.65, 0.7, 1.0], labels=['50-55%', '55-60%', '60-65%', '65-70%', '70%以上'], right=False)
    stats = df.groupby('prob_bin', observed=False).agg(count=('game_id', 'count'), avg_prob=('final_probability', 'mean'), hit_rate=('correct', 'mean'))

    # 5. 生成 Summary TXT
    high_conf = df[df['final_probability'] >= 0.65]
    summary = f"===== V3.9_Backtest_Summary =====\n总比赛数: {len(df)}\nAccuracy: {acc:.4f}\nBrier Score: {br:.4f}\nLog Loss: {ll:.4f}\n\n[分层统计]\n{stats.to_string()}\n\n[高置信度(>=65%)统计]\n数量: {len(high_conf)}\n命中率: {high_conf['correct'].mean():.4f}\n失败案例: {high_conf[high_conf['correct']==0][['game_date', 'home_team', 'away_team']].to_string(index=False)}"
    
    with open("V3.9_Backtest_Summary.txt", "w") as f: f.write(summary)
    print(summary)

except Exception as e:
    print(f"❌ 运行报错: {e}")
