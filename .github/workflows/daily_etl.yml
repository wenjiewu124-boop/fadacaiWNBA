name: WNBA 数据仓库每日自动增量 ETL

on:
  schedule:
    - cron: '0 21 * * *'
  workflow_dispatch:

jobs:
  run-data-pipeline:
    runs-on: ubuntu-latest
    steps:
      - name: 📥 拉取仓库文件
        uses: actions/checkout@v3

      - name: 🐍 启动云端 Python 环境
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: 📦 安装数据库与计算依赖
        run: pip install pandas numpy supabase requests urllib3 joblib xgboost scikit-learn google-api-python-client google-auth

      - name: 🚀 执行自动更新与胜率预测
        env:
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_KEY: ${{ secrets.SUPABASE_KEY }}
          GCP_CREDENTIALS: ${{ secrets.GCP_CREDENTIALS }}
          API_BASKETBALL_KEY: ${{ secrets.API_BASKETBALL_KEY }}
        run: |
          python main.py
          python run.py
