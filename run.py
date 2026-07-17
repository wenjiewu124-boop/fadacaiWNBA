import os
import pandas as pd
from supabase import create_client

# 引入预测引擎模块
import prediction_engine 
from prediction_engine import V3_9_ProbabilityLayer

print("===== 🔍 V3.9 数据链路深度定位 (发货端) =====")

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

target_date = "2026-07-13"
table_name = "Match_Fusion_Features_V3"
columns_to_fetch = "game_id, game_date, home_team, away_team, team_strength_diff, player_impact_diff, rest_days_diff, fatigue_diff, home_advantage"

try:
    print(f"\n[A] 当前预测读取的数据表: {table_name}")
    print(f"    查询操作: .select('{columns_to_fetch}').eq('game_date', '{target_date}')")
    
    res = supabase.table(table_name).select(columns_to_fetch).eq("game_date", target_date).execute()
    df = pd.DataFrame(res.data)
    
    if df.empty:
        print("⚠️ 未找到特征数据，程序退出。")
        exit(0)

    print("\n[B] 当前从数据库读取的五个输入字段真实值 (发货清单):")
    check_cols = ['game_id', 'team_strength_diff', 'player_impact_diff', 'rest_days_diff', 'fatigue_diff', 'home_advantage']
    print(df[check_cols].to_string(index=False))

    # 把读出来的数据扔给 prediction_engine.py
    df_result = prediction_engine.run_prediction(df)

    output_columns = ["game_id", "game_date", "home_team", "away_team", "final_probability"]
    df_output = df_result[output_columns]
    df_output.to_csv("final_prediction.csv", index=False, encoding="utf-8-sig")
    
except Exception as e:
    print(f"❌ 报错: {e}")

exit(0)
