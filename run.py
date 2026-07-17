# main.py
import os
import datetime
import pandas as pd
from supabase import create_client, Client
from src.processors.feature_generator import calculate_v39_features

# 初始化 Supabase
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
db_client: Client = create_client(url, key)

def fetch_supabase_table(table_name):
    res = db_client.table(table_name).select("*").execute()
    return pd.DataFrame(res.data)

def main():
    print("🚨 启动 WNBA-Data-Pipeline 每日增量更新作业...")
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    
    print("📡 正在调用双 API 源抓取 WNBA 增量基础数据...")
    
    print("📥 正在拉取数据库历史快照...")
    df_games = fetch_supabase_table("WNBA_Game_Features_v2")
    df_box = fetch_supabase_table("WNBA_Player_Boxscore")
    df_rating = fetch_supabase_table("Player_Rating")
    
    # 格式对齐
    df_games['game_date'] = pd.to_datetime(df_games['match_date_bj'] if 'match_date_bj' in df_games.columns else df_games['game_date'])
    df_games.rename(columns={'match_id': 'game_id'}, errors='ignore', inplace=True)
    df_box['game_date'] = pd.to_datetime(df_box['game_date'])
    
    # --- 兼容处理上场时间（minutes_num）别名或缺失 ---
    possible_min_cols = ['minutes_num', 'minutes', 'min', 'mp']
    matched_min = None
    for col in df_box.columns:
        if str(col).strip().lower() in possible_min_cols:
            matched_min = col
            break
    
    if matched_min:
        df_box['minutes_num'] = pd.to_numeric(df_box[matched_min], errors='coerce').fillna(15.0)
    else:
        df_box['minutes_num'] = 15.0

    # --- 兼容处理回合数（possessions）别名或缺失 ---
    possible_poss_cols = ['possessions', 'poss', 'poss_num']
    matched_poss = None
    for col in df_box.columns:
        if str(col).strip().lower() in possible_poss_cols:
            matched_poss = col
            break
    
    if matched_poss:
        df_box['possessions'] = pd.to_numeric(df_box[matched_poss], errors='coerce').fillna(5.0)
    else:
        df_box['possessions'] = 5.0
    
    df_rating['game_date'] = pd.to_datetime(df_rating['game_date'])
    
    # ==========================================
    # 步骤 3: 提取今日待预测赛程，计算 V3.9 特征
    # ==========================================
    today_features = calculate_v39_features(df_games, df_box, df_rating, today_str)
    
    if today_features.empty:
        print("☀️ 今日没有待计算比赛，Pipeline 任务正常结束。")
        return
        
    # ==========================================
    # 步骤 4: 将新鲜特征 Upsert 进 Match_Fusion_Features_V3
    # ==========================================
    print(f"💾 正在将新计算的 {len(today_features)} 场比赛特征写入 Match_Fusion_Features_V3...")
    
    # 强转格式规避 JSON 冲突
    records = today_features.replace({pd.NA: None}).to_dict(orient='records')
    
    try:
        db_client.table("Match_Fusion_Features_V3").upsert(records, on_conflict="game_id").execute()
        print("   ✅ 特征大宽表更新完成！已成功注入 Supabase！")
    except Exception as e:
        print(f"❌ 写入 Supabase 失败: {e}")

if __name__ == "__main__":
    main()
