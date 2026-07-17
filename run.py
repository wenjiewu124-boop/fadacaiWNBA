import os
from supabase import create_client

print("===== WNBA_Game_Features_v2 字段结构扫描 =====")

# 1. 连接数据库
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

print("📡 正在读取 WNBA_Game_Features_v2 字段结构...")

try:
    # 提取 1 条数据用来探测真实的 columns 结构
    res_v2 = supabase.table("WNBA_Game_Features_v2").select("*").limit(1).execute()

    if res_v2.data:
        print("✅ WNBA_Game_Features_v2真实字段:")
        # 提取并打印表头所有字段
        print(list(res_v2.data[0].keys()))
    else:
        print("⚠️ WNBA_Game_Features_v2存在，但是没有数据")

except Exception as e:
    print(f"❌ 探测失败，详细报错: {e}")

print("=============================================")
exit(0)
