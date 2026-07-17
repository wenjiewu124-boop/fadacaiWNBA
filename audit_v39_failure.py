import os
import pandas as pd
import numpy as np
import joblib
from supabase import create_client

# 必须声明以保证 Pickle 正常反序列化
class V3_9_ProbabilityLayer:
    def __init__(self, team_weight=0.6, player_weight=0.4, soft_shrink=0.8, penalty_shrink=0.7):
        self.tw = team_weight
        self.pw = player_weight
        self.ss = soft_shrink
        self.ps = penalty_shrink

print("===== 🕵️‍♂️ V3.9 模型失败原因深度审计 =====")

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

try:
    # ==========================================
    # 1. 检查 v3_9_fusion_model.pkl (训练特征字段)
    # ==========================================
    print("\n[1] 检查模型训练特征字段 (Training Features)")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    fusion_model = joblib.load(os.path.join(current_dir, 'v3_9_fusion_model.pkl'))
    model_team = fusion_model["team_node"]
    model_player = fusion_model["player_node"]
    
    # 尝试提取 scikit-learn API 包装的特征名
    team_features = getattr(model_team, 'feature_names_in_', "无法从模型实例获取 (可能训练时传入的是 numpy 数组)")
    player_features = getattr(model_player, 'feature_names_in_', "无法从模型实例获取")
    
    print(f"▶️ Team 模型输入特征: {team_features}")
    print(f"▶️ Player 模型输入特征: {player_features}")
    
    # 获取决策树数量作为参考 (Pickle 不直接保存原训练样本数和时间，只能看树规模)
    try:
        print(f"▶️ Team 模型树数量: {model_team.get_booster().num_boosted_rounds()}")
    except:
        pass

    # ==========================================
    # 2. 检查当前 1000 场回测数据的分布 (Data Distribution)
    # ==========================================
    print("\n[2] 检查 Match_Fusion_Features_V3 当前1000场数据分布")
    res_features = supabase.table("Match_Fusion_Features_V3").select(
        "team_strength_diff, player_impact_diff, rest_days_diff, fatigue_diff, home_advantage"
    ).limit(2000).execute()
    
    df_features = pd.DataFrame(res_features.data)
    
    if df_features.empty:
        print("⚠️ 数据库表为空！")
    else:
        # 强制转换为 float 计算统计学特征
        for col in df_features.columns:
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
            
        desc = df_features.describe().T[['mean', 'std', 'min', 'max']]
        print("▶️ V3 表特征统计学分布 (关键看 std 是否为 0)：")
        print(desc.to_string())
        
        # 异常探针：计算有多少场的特征是完全一模一样的 (常数病)
        unique_counts = df_features.nunique()
        print("\n▶️ 唯一值数量检查 (如果某特征唯一值只有 1 或 2，说明回填数据全是常数)：")
        print(unique_counts.to_string())

    # ==========================================
    # 3. 对比假设 (Data Drift Hypothesis)
    # ==========================================
    print("\n[3] 数据漂移与特征逻辑初步诊断")
    print("👉 重点排查方向：请观察上方 `std` (标准差)。如果 std 接近 0，或者 min 和 max 极其接近，")
    print("👉 意味着我们在执行 `backfill_v3_data.py` 时，由于历史表 WNBA_Game_Features_v2 缺失原始特征，")
    print("👉 导致 fallback 逻辑 (比如默认 player_impact 都是 10.5 和 10.0) 生成了 1000 行几乎完全一样的死水数据！")

    # ==========================================
    # 4. 检查概率融合层 (Probability Layer Parameters)
    # ==========================================
    print("\n[4] 检查概率融合层 (v3_9_probability_layer.pkl)")
    prob_layer = joblib.load(os.path.join(current_dir, 'v3_9_probability_layer.pkl'))
    
    print(f"▶️ Team Weight (团队权重): {prob_layer.tw}")
    print(f"▶️ Player Weight (球员权重): {prob_layer.pw}")
    print(f"▶️ Soft Shrink (软收缩系数): {prob_layer.ss}")
    print(f"▶️ Penalty Shrink (惩罚系数): {prob_layer.ps}")
    
    print("\n[5] 审计结论准备完成")
    print("=========================================")

except Exception as e:
    print(f"❌ 运行报错: {e}")

exit(0)
