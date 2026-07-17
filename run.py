import os
from supabase import create_client

print("===== SUPABASE TABLE AUDIT =====")

# 1. 拿钥匙连接 Supabase
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

# 我们要重点核查的 5 张表
target_tables = [
    "Match_Fusion_Features_V3",
    "team_features_v3",
    "WNBA_Game_Features_v2",
    "WNBA_Player_Boxscore",
    "Player_Rating"
]

print("\nTables:")
# 通过尝试拉取 1 条数据来真实测试表是否存在 (这是最准确的物理探测)
for table in target_tables:
    try:
        supabase.table(table).select("*").limit(1).execute()
        print(f"{table} (FOUND)")
    except Exception:
        print(f"{table} (NOT FOUND)")

print("\nMatch_Fusion_Features_V3:")
try:
    # 尝试拉取 V3 表的前 5 条数据
    res = supabase.table("Match_Fusion_Features_V3").select("*").limit(5).execute()
    print("FOUND")
    
    print("\nColumns:")
    if res.data and len(res.data) > 0:
        # 动态提取表头字段名
        for col in res.data[0].keys():
            print(col)
    else:
        print("表是空的 (Empty Table)，无法通过数据推断字段名。")
        
    print("\nSample:")
    if res.data:
        for idx, row in enumerate(res.data):
            print(f"Row {idx + 1}: {row}")
    else:
        print("无数据 (No Data Rows)")
        
except Exception as e:
    print("NOT FOUND")
    print(f"Error: {e}")

print("================================")
print("探针检测完毕，程序安全退出。")
# 强制终止程序，绝对不往下执行任何修改数据的预测逻辑
exit(0)
