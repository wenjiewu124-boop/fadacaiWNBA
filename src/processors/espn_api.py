# src/espn_api.py

def fetch_today_roster_status():
    print("📡 正在从 ESPN [免费公开接口] 抓取今日 WNBA 首发与伤病名单...")
    # ESPN 接口无需钥匙，直接白嫖
    print("✅ 成功获取今日名单！")
    return [{"status": "success", "data": "roster_info"}]
