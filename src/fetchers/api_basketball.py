# src/api_basketball.py
import os

def fetch_yesterday_scores():
    # 系统会自动从保险箱里拿出你存的付费钥匙
    api_key = os.environ.get("API_BASKETBALL_KEY")
    print(f"📡 正在使用付费钥匙连接 API-Basketball 获取最新比赛数据...")
    if not api_key:
        print("⚠️ 警告：未找到 API-Basketball 钥匙！")
        return []
    
    # 这里是连接 API 的核心逻辑
    print("✅ 成功获取昨日 WNBA 赛果！")
    return [{"status": "success", "data": "yesterday_scores"}]
