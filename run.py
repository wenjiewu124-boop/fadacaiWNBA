import os
from supabase import create_client

print("===== 球员数据层 (Player Data) 字段结构扫描 =====")

# 1. 连接数据库
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

def scan_table(table_name):
    print(f"\n📡 正在读取 {table_name} 字段结构...")
    try:
        res = supabase.table(table_name).select("*").limit(1).execute()
        if res.data:
            print(f"✅ {table_name} 真实字段:")
            columns = list(res.data[0].keys())
            print(columns)
        else:
            print(f"⚠️ {table_name} 存在，但是没有数据")
    except Exception as e:
        print(f"❌ 探测失败，详细报错: {e}")

# 任务 1：扫描 WNBA_Player_Boxscore
scan_table("WNBA_Player_Boxscore")

# 任务 2：扫描 Player_Rating
scan_table("Player_Rating")

print("\n=============================================")
print("探针扫描完毕，程序安全退出。")
exit(0)
