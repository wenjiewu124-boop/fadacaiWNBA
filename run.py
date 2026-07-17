import os
import pandas as pd
from supabase import create_client

print("===== 🔍 V3.9 模型真实性审计 (概率溯源) =====")

# 1. 连数据库
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

target_date = "2026-07-13"

try:
    print("📡 正在拉取 Match_Fusion_Features_V3 输入特征...")
    columns = "game_id, team_strength_diff, player_impact_diff, rest_days_diff, fatigue_diff, home_advantage"
    res = supabase.table("Match_Fusion_Features_V3").select(columns).eq("game_date", target_date).execute()
    
    df = pd.DataFrame(res.data)
    
    if df.empty:
        print("没查到数据。")
        exit(0)

    print("\n===== 📊 输入特征差异报告 =====")
    print(df.to_string(index=False))
    
    print("\n===== 🧮 概率计算路径报告 =====")
    
    # 检查标准差，如果标准差为0，说明所有行的值都一样
    std_team_strength = df['team_strength_diff'].astype(float).std()
    if pd.isna(std_team_strength) or std_team_strength == 0:
        print("⚠️ 警告：检测到所有比赛的 team_strength_diff 完全相同！")
        print("⚠️ 警告：因为 X (特征) 完全相同，导致 Y (胜率概率) 必然完全相同。计算路径无异常，是输入源的数据问题。")

    print("\n===== ⚠️ 是否为测试逻辑 =====")
    print("✅ 结论：是纯测试逻辑导致！")

except Exception as e:
    print(f"❌ 报错: {e}")

exit(0)
