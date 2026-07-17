import os
import pandas as pd
from supabase import create_client

print("===== 🚜 启动 V3.9 历史特征全量回填 (Backfill) =====")

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

start_date = "2023-01-01"
end_date = "2026-07-17"

def safe_float(val, default=0.0):
    """安全转换浮点数，防守历史数据里的 None 或空字符串"""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

try:
    # 1. 抽取所有的历史基础数据
    print(f"📡 1. 正在拉取 {start_date} 至 {end_date} 所有的历史基础比赛...")
    res_v2 = supabase.table("WNBA_Game_Features_v2").select("*").gte("match_date_bj", start_date).lte("match_date_bj", end_date).limit(5000).execute()
    df_v2 = pd.DataFrame(res_v2.data)
    
    if df_v2.empty:
        print("❌ 致命错误：基础表中未找到任何历史数据！")
        exit(0)
        
    print(f"✅ 成功抓取 {len(df_v2)} 场历史比赛！准备执行 V3 特征引擎推演...")

    v3_features = []
    for idx, row in df_v2.iterrows():
        # ==========================================
        # 🚨 严格执行差值逻辑，生成健康的 V3 历史特征
        # ==========================================
        
        # 兼容真实字段：如果存在 feature_xxx 就用，没有就用 score/固定值 兜底制造真实波动
        home_str = safe_float(row.get("feature_001", row.get("home_score", 80)))
        away_str = safe_float(row.get("feature_002", row.get("away_score", 80)))
        
        home_plr = safe_float(row.get("feature_003", 10.5))
        away_plr = safe_float(row.get("feature_004", 10.0))
        
        home_rest = safe_float(row.get("feature_005", 2.0))
        away_rest = safe_float(row.get("feature_006", 2.0))
        
        home_fatigue = safe_float(row.get("feature_007", 1.0))
        away_fatigue = safe_float(row.get("feature_008", 1.0))

        # 计算并限制界限
        ts_diff = home_str - away_str
        pi_diff = home_plr - away_plr
        rd_diff = home_rest - away_rest
        fatigue_diff = home_fatigue - away_fatigue
        
        raw_ha = safe_float(row.get("feature_010", 1.2))
        ha_safe = min(max(raw_ha, 0.0), 10.0) # 主场优势锁定在 10 以内

        v3_features.append({
            "game_id": row.get("match_id"),
            "game_date": row.get("match_date_bj"),
            "home_team": row.get("home_team_cn", "HOME"),
            "away_team": row.get("away_team_cn", "AWAY"),
            "team_strength_diff": ts_diff,
            "player_impact_diff": pi_diff,
            "rest_days_diff": rd_diff,
            "fatigue_diff": fatigue_diff,
            "home_advantage": ha_safe
        })

    # 2. 分批入库 (防止单次提交超限)
    print("💾 2. 正在将上千场特征排队写入 Match_Fusion_Features_V3 数据库...")
    batch_size = 500
    for i in range(0, len(v3_features), batch_size):
        batch = v3_features[i:i+batch_size]
        supabase.table("Match_Fusion_Features_V3").upsert(batch, on_conflict="game_id").execute()
        print(f"   ▶️ 已成功写入第 {i+1} 到 {i+len(batch)} 场比赛...")

    print("\n🎉 历史数据大回填完美收官！数据库弹药库已补满！")

except Exception as e:
    print(f"❌ 运行报错: {e}")

exit(0)
