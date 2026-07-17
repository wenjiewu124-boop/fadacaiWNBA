import os
import pandas as pd
from supabase import create_client

print("===== 🚀 V3.9 数据管道历史联调 (2026-07-13) =====")

# 1. 连数据库
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

target_date = "2026-07-13"

try:
    # --- 找基础表，改名字 ---
    res_games = supabase.table("WNBA_Game_Features_v2").select("*").eq("match_date_bj", target_date).execute()
    df_games = pd.DataFrame(res_games.data) if res_games.data else pd.DataFrame()
    
    if df_games.empty:
        print(f"⚠️ {target_date} 没数据，结束。")
        exit(0)
        
    df_games.rename(columns={"match_date_bj": "game_date", "match_id": "game_id"}, inplace=True, errors="ignore")
    
    # --- 找球员表，改名字，算回合数 ---
    res_box = supabase.table("WNBA_Player_Boxscore").select("*").eq("game_date", target_date).execute()
    df_boxscore = pd.DataFrame(res_box.data) if res_box.data else pd.DataFrame()
    
    if not df_boxscore.empty:
        df_boxscore.rename(columns={"minutes": "minutes_num"}, inplace=True, errors="ignore")
        
        if "possessions" not in df_boxscore.columns:
            fga = df_boxscore.get("field_goal_attempts", 0).fillna(0)
            fta = df_boxscore.get("free_throw_attempts", 0).fillna(0)
            tov = df_boxscore.get("turnovers", 0).fillna(0)
            df_boxscore["possessions"] = fga + 0.44 * fta + tov

    # --- 准备写入 V3 表的数据 ---
    v3_features = []
    for _, row in df_games.iterrows():
        record = {
            "game_id": str(row.get("game_id", f"TEST_{target_date}")),
            "game_date": target_date,
            "home_team": row.get("home_team_cn", "Unknown"),
            "away_team": row.get("away_team_cn", "Unknown"),
            "team_strength_diff": 5.0,   
            "player_impact_diff": 2.0,   
            "rest_days_diff": 1.0,       
            "fatigue_diff": 0.5,         
            "home_advantage": 1.2        
        }
        v3_features.append(record)

    # --- 写入数据库 ---
    write_success = False
    try:
        supabase.table("Match_Fusion_Features_V3").upsert(v3_features, on_conflict="game_id").execute()
        write_success = True
    except Exception as e:
        print(f"写入失败: {e}")

    # --- 打印结果 ---
    print("\n===== 诊断输出 =====")
    print(f"① 生成比赛数量: {len(v3_features)}")
    if v3_features:
        print(f"② 生成字段列表: {list(v3_features[0].keys())}")
    print(f"③ 是否成功 Upsert 到 Supabase: {write_success}")
    
    if write_success:
        print("\nV3.9 数据生产链路通过。")

except Exception as e:
    print(f"❌ 报错啦: {e}")

exit(0)
