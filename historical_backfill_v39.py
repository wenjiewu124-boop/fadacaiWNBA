import os
import pandas as pd
from supabase import create_client
import warnings

warnings.filterwarnings('ignore')
from src.processors.feature_generator import calculate_v39_features

print("===== 🚜 启动 V3.9 纯血历史特征重构 (True Backfill) =====")

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

try:
    print("📥 [1/4] 正在全量拉取历史基础大表 (Games, Boxscores, Ratings)...")
    res_games = supabase.table("WNBA_Game_Features_v2").select("*").limit(10000).execute()
    df_games = pd.DataFrame(res_games.data)
    
    res_box = supabase.table("WNBA_Player_Boxscore").select("*").limit(50000).execute()
    df_box = pd.DataFrame(res_box.data)
    
    res_rating = supabase.table("Player_Rating").select("*").limit(10000).execute()
    df_rating = pd.DataFrame(res_rating.data)

    if df_games.empty or df_box.empty:
        raise ValueError("基础数据表为空，无法执行重构！")

    print("⚙️ [2/4] 正在执行生产级数据预处理与字段桥接...")
    df_games['game_date'] = pd.to_datetime(df_games['match_date_bj'] if 'match_date_bj' in df_games.columns else df_games['game_date'])
    df_games.rename(columns={'match_id': 'game_id'}, errors='ignore', inplace=True)
    
    # ==========================================
    # 🚨 核心修复：桥接历史表与生产函数的字段差异
    # ==========================================
    if 'home_team_id' not in df_games.columns:
        df_games['home_team_id'] = df_games.get('home_team', '')
        df_games['away_team_id'] = df_games.get('away_team', '')
    
    if 'team_id' not in df_box.columns:
        df_box['team_id'] = df_box.get('team', df_box.get('team_name', ''))

    if 'home_team_cn' not in df_games.columns:
        df_games['home_team_cn'] = df_games.get('home_team', '')
        df_games['away_team_cn'] = df_games.get('away_team', '')
        
    if 'season' not in df_games.columns:
        df_games['season'] = df_games['game_date'].dt.year.astype(str)
    # ==========================================

    df_box['game_date'] = pd.to_datetime(df_box['game_date'])
    df_rating['game_date'] = pd.to_datetime(df_rating['game_date'])

    possible_min_cols = ['minutes_num', 'minutes', 'min', 'mp']
    matched_min = next((col for col in df_box.columns if str(col).strip().lower() in possible_min_cols), None)
    df_box['minutes_num'] = pd.to_numeric(df_box[matched_min], errors='coerce').fillna(15.0) if matched_min else 15.0

    possible_poss_cols = ['possessions', 'poss', 'poss_num']
    matched_poss = next((col for col in df_box.columns if str(col).strip().lower() in possible_poss_cols), None)
    df_box['possessions'] = pd.to_numeric(df_box[matched_poss], errors='coerce').fillna(5.0) if matched_poss else 5.0

    unique_dates = df_games['game_date'].dt.date.sort_values().unique()
    print(f"🗓️ 共计发现 {len(unique_dates)} 个历史比赛日需要重算。启动时光机...")

    all_historical_features = []
    
    print("⏳ [3/4] 开始按日穿越，调用生产计算函数...")
    for i, target_date in enumerate(unique_dates):
        date_str = target_date.strftime("%Y-%m-%d")
        
        daily_features = calculate_v39_features(df_games, df_box, df_rating, date_str)
        
        if not daily_features.empty:
            all_historical_features.append(daily_features)
        
        if (i + 1) % 50 == 0:
            print(f"   ▶️ 已处理 {i + 1} / {len(unique_dates)} 个比赛日...")

    final_df = pd.concat(all_historical_features, ignore_index=True)
    print(f"✅ 特征重构完毕！共生成 {len(final_df)} 场纯血 V3.9 比赛特征。")

    print("💾 [4/4] 正在将干净数据覆盖写入 (Upsert) 数据库...")
    records = final_df.replace({pd.NA: None}).to_dict(orient='records')
    
    batch_size = 300
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        supabase.table("Match_Fusion_Features_V3").upsert(batch, on_conflict="game_id").execute()
        print(f"   ⬆️ 成功推入第 {i+1} 到 {i+len(batch)} 条记录...")

    print("\n🎉 恭喜老大！V3.9 历史弹药库彻底洗净！现在可以去跑最终的回测了！")

except Exception as e:
    print(f"❌ 运行报错: {e}")

exit(0)
