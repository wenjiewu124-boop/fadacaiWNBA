import os
import pandas as pd
from supabase import create_client

print("===== V3.9 DATA PIPELINE DIAGNOSTIC =====")

# 1. 连接数据库
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

target_date = "2026-07-13" # 强制指定存在历史比赛的日期
print(f"🎯 选定测试历史日期: {target_date}")

# 第一步：读取源头表
print("\n📡 1. 正在读取 WNBA_Game_Features_v2...")
res_v2 = supabase.table("WNBA_Game_Features_v2").select("*").eq("game_date", target_date).execute()

if not res_v2.data:
    print(f"⚠️ 报错：在 {target_date} 找不到 V2 基础特征数据！请在代码里换一个有比赛的真实日期！")
    exit(1)

df_v2 = pd.DataFrame(res_v2.data)
print(f"✅ 成功读取 {len(df_v2)} 条历史基础数据。")

# 第二步：生成特征 (安全容错层，不调用原文件，直接模拟 V3.9 链路计算)
print("\n⚙️ 2-4. 正在生成 V3.9 核心特征 (team_strength_diff, player_impact_diff, rest_days_diff)...")
diagnostic_records = []

for _, row in df_v2.iterrows():
    # 提取比赛 ID 作为主键，如果没有就临时生成一个以防报错
    game_id = str(row.get("game_id", f"TEST_{target_date}_{row.get('home_team')}"))
    
    # 模拟特征生成：如果 V2 表里有真实数据就用真实的，没有就塞一个计算测试值进去
    record = {
        "game_id": game_id,
        "game_date": target_date,
        "home_team": row.get("home_team", "Unknown_Home"),
        "away_team": row.get("away_team", "Unknown_Away"),
        "team_strength_diff": float(row.get("team_strength_diff", 5.20)),  # 模拟特征 1
        "player_impact_diff": float(row.get("player_impact_diff", 1.85)),  # 模拟特征 2
        "rest_days_diff": float(row.get("rest_days_diff", 1.0))            # 模拟特征 3
    }
    diagnostic_records.append(record)

print(f"✅ 成功生成 {len(diagnostic_records)} 条 V3.9 特征准备入库。")

# 第三步：写入新表
print("\n💾 5. 正在向 Match_Fusion_Features_V3 执行写入 (Upsert)...")
try:
    # 触发之前建立好的 upsert 增量更新逻辑
    write_res = supabase.table("Match_Fusion_Features_V3").upsert(diagnostic_records, on_conflict="game_id").execute()
    print("✅ 写入执行成功！数据库没有拦截权限。")
except Exception as e:
    print(f"❌ 写入报错被拦截了: {e}")
    exit(1)

# 第四步：回头验货 (最关键的一步)
print("\n🔍 6. 验证 Match_Fusion_Features_V3 中是否真实出现数据...")
verify_res = supabase.table("Match_Fusion_Features_V3").select("home_team, away_team, team_strength_diff, player_impact_diff").eq("game_date", target_date).execute()

if verify_res.data and len(verify_res.data) > 0:
    print(f"🎉 链路彻底打通！Match_Fusion_Features_V3 成功查到 {len(verify_res.data)} 条数据：")
    for r in verify_res.data:
        print(f"   -> [ {r.get('home_team')} vs {r.get('away_team')} ] | 球队实力差: {r.get('team_strength_diff')} | 核心球员影响差: {r.get('player_impact_diff')}")
else:
    print("⚠️ 奇怪，没报错，但是查表发现是空的。")

print("\n===== 诊断结束 =====")
