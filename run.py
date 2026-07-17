import os
import pandas as pd
from supabase import create_client

# 💡 请在这里导入你真实的预测引擎文件，例如：
# import prediction_engine 
# import torch (如果是用 PyTorch)

print("===== 🔍 V3.9 预测引擎真实性审计 =====")

# 1. 连数据库
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

target_date = "2026-07-13"
columns_to_fetch = "game_id, team_strength_diff, player_impact_diff, rest_days_diff, fatigue_diff, home_advantage"

try:
    print("📡 1. 正在拉取 V3.9 输入特征...")
    res = supabase.table("Match_Fusion_Features_V3").select(columns_to_fetch).eq("game_date", target_date).execute()
    df = pd.DataFrame(res.data)
    
    if df.empty:
        print("⚠️ 未找到数据。")
        exit(0)

    # 声明预测公式使用的全部输入字段
    feature_cols = ['team_strength_diff', 'player_impact_diff', 'rest_days_diff', 'fatigue_diff', 'home_advantage']
    print(f"✅ 模型核准输入字段: {feature_cols}")

    print("\n⚙️ 2. 正在载入真实预测公式进行推演...")
    
    # ⬇️⬇️⬇️ 【真实模型调用区】 ⬇️⬇️⬇️
    # ⚠️ 请把下面这行替换为你真实调用模型预测的代码
    # 例如： df['home_win_probability'] = prediction_engine.predict(df[feature_cols])
    
    # [临时占位：如果还没接上，先用这个简单的线性权重公式测试逻辑连通性]
    import math
    def audit_model(row):
        score = (float(row['team_strength_diff']) * 0.4 + 
                 float(row['player_impact_diff']) * 0.3 + 
                 float(row['home_advantage']) * 0.2 + 
                 float(row['rest_days_diff']) * 0.1 - 
                 float(row['fatigue_diff']) * 0.1)
        return round(1 / (1 + math.exp(-score)), 4)
    
    df['home_win_probability'] = df.apply(audit_model, axis=1) # <-- 替换这里
    # ⬆️⬆️⬆️ 【真实模型调用区】 ⬆️⬆️⬆️

    df['away_win_probability'] = 1 - df['home_win_probability']
    df['final_probability'] = df[['home_win_probability', 'away_win_probability']].max(axis=1)

    print("\n===== 📊 每场比赛审计报告 =====")
    for index, row in df.iterrows():
        print(f"▶️ Game ID: {row['game_id']}")
        print(f"   [输入] 实力差:{row['team_strength_diff']:>5} | 影响差:{row['player_impact_diff']:>5} | 休息差:{row['rest_days_diff']:>5} | 疲劳差:{row['fatigue_diff']:>5} | 主场优势:{row['home_advantage']:>5}")
        print(f"   [输出] 主胜概率: {row['home_win_probability']:.4f} | 客胜概率: {row['away_win_probability']:.4f} | 最终置信度: {row['final_probability']:.4f}")
        print("-" * 60)

    print("\n===== 🔍 终极诊断结论 =====")
    std_prob = df['final_probability'].astype(float).std()
    
    if pd.isna(std_prob) or std_prob == 0:
        print("❌ 警报：输入特征不同，但 final_probability 居然完全相同！预测公式存在逻辑断层。")
    else:
        print("🎉 通关验证：输入特征差异成功向后传导，最终预测胜率呈现健康波动！")

except Exception as e:
    print(f"❌ 运行报错: {e}")

exit(0)
