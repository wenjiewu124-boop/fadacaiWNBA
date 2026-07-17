# src/processors/feature_generator.py
import pandas as pd
import numpy as np

def calculate_v39_features(df_games, df_box, df_rating, target_date):
    """
    完全还原 V3.9 宏微观双轨特征重构逻辑
    专门计算 target_date（例如今天）所有比赛的特征
    """
    print(f"⚙️ 正在计算 {target_date} 的 V3.9 生产级特征...")
    
    # 过滤出 target_date 当天的比赛
    today_games = df_games[df_games['game_date'].dt.date == pd.to_datetime(target_date).date()]
    if today_games.empty:
        print("⚠️ 今日无待计算比赛。")
        return pd.DataFrame()

    features_list = []
    
    # 构建临时历史净胜分日志（用于计算 team_strength_diff）
    team_game_logs = []
    df_games_past = df_games[df_games['game_date'].dt.date < pd.to_datetime(target_date).date()].copy()
    df_games_past['score_diff_raw'] = df_games_past['home_score'] - df_games_past['away_score']
    
    for _, row in df_games_past.iterrows():
        h_id, a_id, date, s_diff = row['home_team_id'], row['away_team_id'], row['game_date'], row['score_diff_raw']
        team_game_logs.append({'team_id': h_id, 'game_date': date, 'net_rtg': s_diff})
        team_game_logs.append({'team_id': a_id, 'game_date': date, 'net_rtg': -s_diff})
    df_team_logs = pd.DataFrame(team_game_logs)

    # 循环处理今天的每一场比赛
    for _, row in today_games.iterrows():
        gid = str(row['game_id'])
        h_id, a_id = row['home_team_id'], row['away_team_id']
        t_season = str(row['season'])
        
        # --- 1. 宏观特征计算 (team_strength_diff) ---
        h_past = df_team_logs[df_team_logs['team_id'] == h_id] if not df_team_logs.empty else pd.DataFrame()
        a_past = df_team_logs[df_team_logs['team_id'] == a_id] if not df_team_logs.empty else pd.DataFrame()
        
        h_macro = h_past.sort_values('game_date')['net_rtg'].tail(10).mean() if not h_past.empty else 0.0
        a_macro = a_past.sort_values('game_date')['net_rtg'].tail(10).mean() if not a_past.empty else 0.0
        
        # 休息天数计算
        h_last_date = h_past['game_date'].max() if not h_past.empty else pd.to_datetime(target_date) - pd.Timedelta(days=7)
        a_last_date = a_past['game_date'].max() if not a_past.empty else pd.to_datetime(target_date) - pd.Timedelta(days=7)
        
        h_rest = min(float((pd.to_datetime(target_date) - h_last_date).days), 7.0)
        a_rest = min(float((pd.to_datetime(target_date) - a_last_date).days), 7.0)
        
        team_strength_diff = np.clip(h_macro - a_macro, -20.0, 20.0)
        rest_days_diff = h_rest - a_rest
        fatigue_diff = (1.0 if h_rest <= 1 else 0.0) - (1.0 if a_rest <= 1 else 0.0)
        
        # --- 2. 微观特征计算 (player_impact_diff) ---
        def get_team_player_impact(team_id):
            # 获取该队本赛季 target_date 之前的历史 Boxscore
            past_box = df_box[(df_box['team_id'] == team_id) & (df_box['game_date'] < target_date) & (df_box['season'] == t_season)]
            # 模拟今日首发/伤病名单（生产环境由 ESPN API 实时抓取提供）
            # 如果暂无今日首发，则默认使用过去出场时间最长的 8 个人作为今日预计出场名单
            if past_box.empty:
                return 50.0  # 默认兜底实力中位数
            
            recent_10_dates = past_box['game_date'].drop_duplicates().nlargest(10)
            recent_box = past_box[past_box['game_date'].isin(recent_10_dates)]
            
            player_stats = recent_box.groupby('player_id').agg(
                avg_minutes=('minutes_num', 'mean'),
                avg_usg=('possessions', 'mean'),
                tot_pts=('points', 'sum'),
                tot_fga=('field_goal_attempts', 'sum'),
                tot_fta=('free_throw_attempts', 'sum')
            ).reset_index()
            
            player_stats['avg_ts'] = player_stats['tot_pts'] / (2 * (player_stats['tot_fga'] + 0.44 * player_stats['tot_fta']) + 1e-5)
            player_stats['avg_ts'] = player_stats['avg_ts'].clip(0.30, 0.75)
            player_stats = player_stats.sort_values(by='avg_minutes', ascending=False)
            
            historical_top3 = player_stats.head(3)['player_id'].tolist()
            roles = player_stats.iloc[3:8]['player_id'].tolist()
            
            def get_player_score(pid):
                pr = df_rating[(df_rating['player_id'] == pid) & (df_rating['game_date'] < target_date)]
                if pr.empty:
                    base_rtg, base_ts = 3.0, 0.50
                else:
                    latest = pr.sort_values('game_date', ascending=False).iloc[0]
                    base_rtg = latest['overall_rating']
                    base_ts = latest['efficiency_rating'] / 100.0 if latest['efficiency_rating'] > 0 else 0.50
                
                p_row = player_stats[player_stats['player_id'] == pid]
                minutes = p_row['avg_minutes'].values[0] if not p_row.empty else 15.0
                usg = p_row['avg_usg'].values[0] if not p_row.empty else 5.0
                ts = p_row['avg_ts'].values[0] if not p_row.empty else base_ts
                
                form = 0.0 # 近期状态
                return (base_rtg * 2.5) + (form * 3.5) + (ts * 12.0) + (usg * 0.4) + (minutes * 0.15)

            top1 = get_player_score(historical_top3[0]) if len(historical_top3) >= 1 else 10.0
            top2 = get_player_score(historical_top3[1]) if len(historical_top3) >= 2 else 8.0
            top3 = get_player_score(historical_top3[2]) if len(historical_top3) >= 3 else 6.0
            role_scores = [get_player_score(pid) for pid in roles]
            role_val = sum(role_scores) / len(role_scores) if role_scores else 5.0
            
            # 缺失核心惩罚项（结合 ESPN 实时伤病名单）
            # active_players = espn_get_today_active_roster(team_id)
            # absence_penalty = sum(4.0 for p in historical_top3 if p not in active_players)
            absence_penalty = 0.0 # 默认无缺失
            
            return (top1 * 0.40) + (top2 * 0.30) + (top3 * 0.20) + (role_val * 0.10) - absence_penalty

        h_player_impact = get_team_player_impact(h_id)
        a_player_impact = get_team_player_impact(a_id)
        player_impact_diff = h_player_impact - a_player_impact
        
        features_list.append({
            'game_id': gid,
            'game_date': row['game_date'],
            'season': t_season,
            'home_team_cn': row['home_team_cn'],
            'away_team_cn': row['away_team_cn'],
            'team_strength_diff': round(team_strength_diff, 3),
            'player_impact_diff': round(player_impact_diff, 3),
            'fatigue_diff': round(fatigue_diff, 1),
            'rest_days_diff': round(rest_days_diff, 1),
            'home_advantage': 1.0 # 强行灌入主场哨
        })
        
    return pd.DataFrame(features_list)
