import os
import pandas as pd
from supabase import create_client

print("===== 🚀 WNBA V3.9 预测引擎启动 =====")

# 1. 连接数据库
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

# 这里假设 target_date 已经由你的系统自动获取，例如: target_date = "2026-07-13"
target_date = "2026-07-13" # 如果你使用的是当天自动获取，请保留你原有的时间代码

# 第一步：精准读取 (修复了字段名报错)
print(f"📡 1. 正在读取 WNBA_Game_Features_v2 (日期: {target_date})...")
try:
    # 关键修改：eq("match_date_bj", target_date)
    res_v2 = supabase.table("WNBA_Game_Features_v2").select("*").eq("match_date_bj", target_date).execute()
    
    if not res_v2.data:
        print(f"⚠️ 在 {target_date} 找不到基础特征数据，今日可能无比赛，系统安全休眠。")
        exit(0) # 优雅退出
        
    df_v2 = pd.DataFrame(res_v2.data)
    print(f"✅ 成功读取 {len(df_v2)} 条比赛数据。")
    
except Exception as e:
    print(f"❌ 读取数据库失败: {e}")
    exit(1)

# 第二步：字段适配层 (翻译给 V3.9 模型听)
print("⚙️ 2. 正在执行字段映射适配...")
df_v2.rename(
    columns={
        "match_date_bj": "game_date",
        "match_id": "game_id",
        "home_team_cn": "home_team", # 如果 V3.9 需要英文，这里可以视情况映射，如果就需要中文则保留
        "away_team_cn": "away_team"
    },
    inplace=True,
    errors="ignore" # 忽略不存在的列，防止意外报错
)
print("✅ 字段映射适配成功！已统一为 V3.9 内部标准字段。")

# --- 下面继续接你原有的 V3.9 模型推演和写入逻辑 ---
# ...
