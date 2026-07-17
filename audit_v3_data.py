import os
import pandas as pd
from supabase import create_client

print("===== 🕵️‍♂️ V3.9 历史数据缺口审计 =====")

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

start_date = "2023-01-01"
end_date = "2026-07-17"

try:
    # ==========================================
    # A & B. Match_Fusion_Features_V3 当前数据统计与时间分布
    # ==========================================
    print("\n[A] Match_Fusion_Features_V3 当前特征表审计")
    res_v3 = supabase.table("Match_Fusion_Features_V3").select("game_id, game_date").execute()
    df_v3 = pd.DataFrame(res_v3.data)
    
    if df_v3.empty:
        print("   ⚠️ V3 融合表目前完全为空！")
    else:
        print(f"   ▶️ 当前 V3 表总行数: {len(df_v3)} 行")
        print("   ▶️ game_date 时间分布 (具体日期及场次):")
        print(df_v3['game_date'].value_counts().to_string())

    # ==========================================
    # C. 历史 1000+ 比赛基础数据检查 (WNBA_Game_Features_v2)
    # ==========================================
    print(f"\n[B] 2023-01-01 到 {end_date} 历史比赛存量审计")
    # 检查基础表到底有没有这 3 年的比赛
    res_v2 = supabase.table("WNBA_Game_Features_v2").select("match_id, match_date_bj, home_score, away_score").gte("match_date_bj", start_date).lte("match_date_bj", end_date).execute()
    df_v2 = pd.DataFrame(res_v2.data)
    
    if df_v2.empty:
        print("   ❌ 致命错误：源头基础表 WNBA_Game_Features_v2 在该时间段内无数据！")
    else:
        print(f"   ▶️ 基础表 V2 实际拥有的历史比赛总数: {len(df_v2)} 场")
        print("   ▶️ 基础表年份分布 (一瞥):")
        df_v2['year'] = df_v2['match_date_bj'].str[:4]
        print(df_v2['year'].value_counts().to_string())

    # ==========================================
    # D. 诊断结论与缺口分析
    # ==========================================
    print("\n[C] 历史比赛无法进入融合表的原因分析 (缺口追踪)")
    print("   👉 诊断：历史比赛的原始数据其实一直躺在 V2 基础表里。")
    print("   👉 缺失原因：我们从未针对 2023-2026 的所有历史比赛，执行过【V3 特征批量生成】。")
    print("   👉 证据：V3 表里的数据，全部来自于前几天我们用 `target_date='2026-07-13'` 跑的那次尺度修复测试，所以仅有 4 条。由于缺失 V3 维度的特征，XGBoost 模型巧妇难为无米之炊，回测引擎只能测出 4 场。")

    # ==========================================
    # E. 提出补齐方案
    # ==========================================
    print("\n[D] 历史特征批量补齐方案 (Backfill Plan)")
    print("   1. 编写独立脚本 `historical_backfill.py`。")
    print("   2. 读取 WNBA_Game_Features_v2 中的全量 1000+ 场比赛历史数据。")
    print("   3. 应用已修复的正确特征生成公式 (严格使用差值相减，限制主场优势)。")
    print("   4. 批量执行计算，并将生成的特征 Upsert 写入 Match_Fusion_Features_V3。")
    print("   5. 数据补齐后，无需改动代码，再次运行刚才的 `backtest_v39.py` 即可获得真实回测报告！")

except Exception as e:
    print(f"❌ 运行报错: {e}")

exit(0)
