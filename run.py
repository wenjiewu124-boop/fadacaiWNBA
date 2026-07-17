import os
import json
import pandas as pd
from supabase import create_client
from google.oauth2 import service_account
from googleapiclient.discovery import build
import prediction_engine

print("🚨 1. 拿着保险箱的钥匙，连接 Supabase 数据库...")
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

print("📡 2. 获取今日最新比赛特征...")
res = supabase.table("Match_Fusion_Features_V3").select("*").order("game_date", desc=True).limit(5).execute()

if not res.data:
    print("⚠️ 今日没有比赛数据，系统自动休眠。")
    exit()

daily_data = pd.DataFrame(res.data)

print("⚙️ 3. 喂给 V3.9 引擎进行胜率推演...")
result_df = prediction_engine.run_prediction(daily_data)

print("💾 4. 保存最终预测结果单...")
output_df = result_df[['game_date', 'home_team_cn', 'away_team_cn', 'team_strength_diff', 'player_impact_diff', 'final_probability', 'prediction_side']]
output_df.to_csv("final_prediction.csv", index=False, encoding='utf-8-sig')

print("✅ 任务圆满完成！今日签批单已生成: final_prediction.csv")

print("🚀 5. 正在将预测结果推送至 [全息篮球量化交割系统]...")
gcp_creds_json = os.environ.get("GCP_CREDENTIALS")

if gcp_creds_json:
    creds_dict = json.loads(gcp_creds_json)
    creds = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=['https://www.googleapis.com/auth/spreadsheets']
    )
    service = build('sheets', 'v4', credentials=creds)
    
    SPREADSHEET_ID = '12uCcuAfUCkAf3t7RiOTYO00vVAHIdUxSQ3F9j_6wU7I' 
    RANGE_NAME = '金矿!A1'

    output_df = output_df.fillna("") 
    values = [output_df.columns.values.tolist()] + output_df.values.tolist()
    body = {'values': values}

    try:
        result = service.spreadsheets().values().update(
            spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME,
            valueInputOption='RAW', body=body).execute()
        print(f"✅ 成功更新 {result.get('updatedCells')} 个单元格！『金矿』已填满！")
    except Exception as e:
        print(f"❌ 写入 Google Sheet 失败: {e}")
else:
    print("⚠️ 未找到 GCP_CREDENTIALS，跳过写入 Google Sheet 环节。")
