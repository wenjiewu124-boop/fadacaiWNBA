import os
import pandas as pd
import numpy as np
import joblib
from supabase import create_client
import datetime

# 必须声明以保证 Pickle 正常反序列化
class V3_9_ProbabilityLayer:
    def __init__(self, team_weight=0.6, player_weight=0.4, soft_shrink=0.8, penalty_shrink=0.7):
        self.tw = team_weight
        self.pw = player_weight
        self.ss = soft_shrink
        self.ps = penalty_shrink

print("===== 《V3.9模型失效原因审计报告》 =====")

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ========================================
    # 任务1: 读取 v3_9_fusion_model.pkl
    # ========================================
    print("\n[任务1: 模型文件特征审计]")
    fusion_model = joblib.load(os.path.join(current_dir, 'v3_9_fusion_model.pkl'))
    model_team = fusion_model["team_node"]
    model_player = fusion_model["player_node"]
    
    print(f"1. team_node 模型类型: {type(model_team).__name__}")
    print(f"2. player_node 模型类型: {type(model_player).__name__}")
    
    team_feats = getattr(model_team, 'feature_names_in_', "未知 (Numpy Array 格式，需核对传入顺序)")
    player_feats = getattr(model_player, 'feature_names_in_', "未知 (Numpy Array 格式，需核对传入顺序)")
    print(f"3. 训练 feature_names:\n   - Team: {team_feats}\n   - Player: {player_feats}")
    
    # 尝试从 XGBoost booster 提取树的数量作为规模参考
    try:
        team_trees = len(model_team.get_booster().get_dump())
        print(f"4. 模型训练样本规模 (决策树数量): {team_trees} 棵树")
    except:
        print("4. 模型训练样本数量: 无法直接从当前 Pickle 提取")
        
    # 获取模型文件的最后修改时间
    model_path = os.path.join(current_dir, 'v3_9_fusion_model.pkl')
    mtime = os.path.getmtime(model_path)
    print(f"5. 模型训练日期: {datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')}")

    # ========================================
    # 任务2: 读取 Match_Fusion_Features_V3
    # ========================================
    print("\n[任务2: 1000场回测数据分布统计]")
    res_features = supabase.table("Match_Fusion_Features_V3").select(
        "team_strength_diff, home_advantage, rest_days_diff, fatigue_diff, player_impact_diff"
    ).limit(2000).execute()
    
    df_features = pd.DataFrame(res_features.data)
    
    if df_features.empty:
        print("⚠️ 未找到回测数据！")
    else:
        # 强转数值型以计算分布
        for col in df_features.columns:
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
            
        stats = df_features.describe().T[['mean', 'std', 'min', 'max']]
        print(stats.to_string())

    # ========================================
    # 任务3: 对比分析 (逻辑推演)
    # ========================================
    print("\n[任务3: 训练分布 VS 回测分布对比分析]")
    print("📌 寻找特征漂移 (Feature Drift)：")
    print("   👉 请检查上方统计表中的 `std` (标准差)。如果任何一个核心字段的 std = 0.000 (或者极小)，")
    print("      说明我们给这1000场比赛回填的特征是死板的“固定值”。无波动的特征会导致模型只能输出固定常数概率。")
    print("📌 寻找字段顺序错误：")
    print(f"   👉 模型期望的顺序: {team_feats}")
    print(f"   👉 代码实际输入的顺序: ['team_strength_diff', 'home_advantage', 'rest_days_diff', 'fatigue_diff']")
    print("      (如果发现字段错位，会导致例如“疲劳差”被当成了“主场优势”送进模型，引发全面崩溃)")

    # ========================================
    # 任务4: 检查概率融合层
    # ========================================
    print("\n[任务4: 概率融合层参数审计]")
    prob_layer = joblib.load(os.path.join(current_dir, 'v3_9_probability_layer.pkl'))
    
    print(f"team_weight = {prob_layer.tw}")
    print(f"player_weight = {prob_layer.pw}")
    print(f"soft_shrink = {prob_layer.ss}")
    print(f"penalty_shrink = {prob_layer.ps}")
    
    print("\n👉 判断分析：")
    if prob_layer.ss >= 1.0:
        print("🔴 发现严重异常：soft_shrink >= 1.0！这会导致概率不仅没有被收缩拉回 0.5，反而被成倍放大，这是高概率频出且极度自信的数学根源！")
    elif prob_layer.tw + prob_layer.pw != 1.0:
        print("🔴 发现严重异常：team_weight + player_weight 权重相加不等于 1.0，导致初始融合概率基线扭曲！")
    else:
        print("🟢 融合层收缩参数表面正常 (soft_shrink < 1.0，权重和=1)。")
        
    print("\n========================================")

except Exception as e:
    print(f"❌ 运行报错: {e}")

exit(0)
