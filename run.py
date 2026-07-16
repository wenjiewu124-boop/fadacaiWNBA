import os
import pandas as pd
from supabase import create_client
import prediction_engine # 叫醒咱们刚写好的预测引擎

print("🚨 1. 拿着保险箱的钥匙，连接 Supabase 数据库...")
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

print("📡 2. 获取今日最新比赛特征...")
# 拉取最新的待预测比赛
res = supabase.table("Match_Fusion_Features_V3").select("*").order("game_date", desc=True).limit(5).execute()

if not res.data:
    print("⚠️ 今日没有比赛数据，系统自动休眠。")
    exit()

daily_data = pd.DataFrame(res.data)

print("⚙️ 3. 喂给 V3.9 引擎进行胜率推演...")
result_df = prediction_engine.run_prediction(daily_data)

print("💾 4. 保存最终预测结果单...")
# 只保留我们关心的核心字段
output_df = result_df[['game_date', 'home_team_cn', 'away_team_cn', 'team_strength_diff', 'player_impact_diff', 'final_probability', 'prediction_side']]
output_df.to_csv("final_prediction.csv", index=False, encoding='utf-8-sig')

print("✅ 任务圆满完成！今日签批单已生成: final_prediction.csv")
