import os
import pandas as pd
from supabase import create_client
import prediction_engine # 引入真正的预测引擎

print("===== 🚀 V3.9 终端预测启动 (run.py) =====")

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

target_date = "2026-07-13"
columns_to_fetch = "game_id, game_date, home_team, away_team, team_strength_diff, player_impact_diff, rest_days_diff, fatigue_diff, home_advantage"

try:
    print(f"📡 1. 正在拉取 {target_date} 的特征数据...")
    res = supabase.table("Match_Fusion_Features_V3").select(columns_to_fetch).eq("game_date", target_date).execute()
    df = pd.DataFrame(res.data)
    
    if df.empty:
        print("⚠️ 未找到特征数据，程序退出。")
        exit(0)

    # 核心：把读出来的数据扔给 prediction_engine.py 算概率
    df_result = prediction_engine.run_prediction(df)

    # 挑出最终结果保存CSV
    output_columns = ["game_id", "game_date", "home_team", "away_team", "final_probability", "prediction_side"]
    df_output = df_result[output_columns]
    df_output.to_csv("final_prediction.csv", index=False, encoding="utf-8-sig")
    
    print("\n===== 🎯 最终预测结果预览 =====")
    print(df_output.to_string(index=False))

except Exception as e:
    print(f"❌ 报错: {e}")

exit(0)
