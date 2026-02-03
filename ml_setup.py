""" Processes datasets, trains and saves the Machine Learning XGB model. """

""" ================================================================================================================ """
# IMPORTING MODULES #

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
from joblib import dump

""" ================================================================================================================ """
# FUNCTIONS TO REPLACE ONE-HOT ENCODING WITH SPECIFIC TEAMS PERFORMANCE ON EACH MAP #

def update_team_map_history(team_name, map_name, won, team_map_history):
    """ For each map, updates the team's map history (= total games played on the map and games won) """
    if team_name not in team_map_history: team_map_history[team_name] = {}
    if map_name not in team_map_history[team_name]: team_map_history[team_name][map_name] = {'wins': 0, 'total': 0}

    team_map_history[team_name][map_name]['total'] += 1
    if won: team_map_history[team_name][map_name]['wins'] += 1

    return team_map_history

def get_team_map_winrate(team_name, map_name, team_map_history):
    """ From the (wins, total) values of the team on a map, returns a win rate wins/total """
    stats = team_map_history[team_name][map_name]
    if stats['total'] == 0: return 0.5
    return stats['wins'] / stats['total']

""" ================================================================================================================ """
# ROUNDS ENCODING FUNCTION #
def extract_rounds(x): return int(str(x).strip().replace('(', '').replace(')', ''))

""" ================================================================================================================ """
# DATA PREPROCESSING FUNCTIONS #
def parse_khs(s):
    m = re.findall(r'\d+', str(s))
    if len(m) >= 2:
        return int(m[0]), int(m[1])
    return np.nan, np.nan

def parse_stat(s):
    s = str(s).replace('%', '').replace('(', '').replace(')', '')
    try:
        return float(s)
    except:
        return np.nan

""" ================================================================================================================ """
# FUNCTIONS TO COMPUTE TEAM AVERAGES #
hist_default, hist_computed = 0, 0
def calculate_team_features(team_name, team_history, HISTORY_SIZE):
    if team_name not in team_history or len(team_history[team_name]) == 0:
        return {  # average of 25th percentile values
            'win_rate': 0.5,
            'avg_rating': 1.06,
            'avg_kills': 16.8,
            'avg_deaths': 16.8,
            'avg_kast': 69.3,
            'avg_adr': 74.2,
            'avg_kddiff': 0.07,
            'avg_fkdiff': 0.03,
            'rounds_won': 13.1
        }

    recent_games = team_history[team_name][max(len(team_history[team_name]) - HISTORY_SIZE, 0):]
    return {
        'win_rate': np.mean([g['won'] for g in recent_games]),
        'avg_rating': np.mean([g['avg_rating'] for g in recent_games]),
        'avg_kills': np.mean([g['avg_kills'] for g in recent_games]),
        'avg_deaths': np.mean([g['avg_deaths'] for g in recent_games]),
        'avg_kast': np.mean([g['avg_kast'] for g in recent_games]),
        'avg_adr': np.mean([g['avg_adr'] for g in recent_games]),
        'avg_kddiff': np.mean([g['avg_kddiff'] for g in recent_games]),
        'avg_fkdiff': np.mean([g['avg_fkdiff'] for g in recent_games]),
        'rounds_won': np.mean([g['rounds'] for g in recent_games])
    }

def extract_team_avg_stats(match, team_prefix):
    """Extract team average stats (average across 5 players) for one match"""
    ratings, kills, deaths, kasts, adrs, kddiffs, fkdiffs = [], [], [], [], [], [], []
    for player_num in range(1, 6):
        r = match.get(f'{team_prefix}_p{player_num}_game_rating')
        if pd.notna(r): ratings.append(r)

        k = match.get(f'{team_prefix}_p{player_num}_kills')
        if pd.notna(k): kills.append(k)

        d = match.get(f'{team_prefix}_p{player_num}_deaths')
        if pd.notna(d): deaths.append(d)

        kast = match.get(f'{team_prefix}_p{player_num}_kast')
        if pd.notna(kast): kasts.append(kast)

        adr = match.get(f'{team_prefix}_p{player_num}_adr')
        if pd.notna(adr): adrs.append(adr)

        kd = match.get(f'{team_prefix}_p{player_num}_kddiff')
        if pd.notna(kd): kddiffs.append(kd)

        fk = match.get(f'{team_prefix}_p{player_num}_fkdiff')
        if pd.notna(fk): fkdiffs.append(fk)

    return {
        'avg_rating': np.mean(ratings) if ratings else 1.06,
        'avg_kills': np.mean(kills) if kills else 16.8,
        'avg_deaths': np.mean(deaths) if deaths else 16.8,
        'avg_kast': np.mean(kasts) if kasts else 69.3,
        'avg_adr': np.mean(adrs) if adrs else 74.2,
        'avg_kddiff': np.mean(kddiffs) if kddiffs else 0.07,
        'avg_fkdiff': np.mean(fkdiffs) if fkdiffs else 0.03,
        'rounds': match.get(f'{team_prefix}_rounds', 13.1)
    }

""" ================================================================================================================ """
# MAIN FUNCTION #

def setup_ml_model():
    # ========== READ CSV ==========
    df_games = pd.read_csv('./database/game_data_processed.csv')    # Processed was retuned manually by us.
    df_results = pd.read_csv('./database/historic_games_list.csv')

    df_games.drop(columns="Unnamed: 0", inplace=True)

    df = df_results.merge(df_games, on='game_link', how='inner')  # Merge the 2 datasets on unique variable game_link.
    df = df.drop_duplicates(subset='game_link', keep='first')
    df = df.sort_values('date_unix').reset_index(drop=True)

    # ========== ROUNDS ENCODING ==========
    df['team1_rounds'] = df['team1_rounds'].apply(extract_rounds)
    df['team2_rounds'] = df['team2_rounds'].apply(extract_rounds)
    df['team1_won'] = (df['team1_rounds'] > df['team2_rounds']).astype(int)

    print(f"Team1 wins: {df['team1_won'].sum()}, Team2 wins: {(1 - df['team1_won']).sum()}")

    # ========== DATA PREPROCESSING ==========
    df_processed = df.copy()    # Copy of the dataset to avoid fragmentation errors.

    C = list(df_processed.columns)
    D = list(df_processed.iloc[0].values)
    for i in range(len(C)): print("(" + str(i) + ") " + str(C[i]) + " : " + str(D[i]))

    # ========== TEAMS AVERAGE STATS ==========
    team_map_history = {}
    team_history = {}
    features = []
    HISTORY_SIZE = 15
    teams_avg = {}

    for idx, match in df.iterrows():    # Each match
        # Initialize variables
        team1_name = match['team1']
        team2_name = match['team2']
        map_name = match['map_name_short']

        # Calculate match-specific teams values (by averaging values of all the players on each team)
        if team1_name not in team_history: team_history[team1_name] = []
        team1_stats = extract_team_avg_stats(match, 'team1')
        team1_stats['won'] = match['team1_won']
        team_history[team1_name].append(team1_stats)

        if team2_name not in team_history: team_history[team2_name] = []
        team2_stats = extract_team_avg_stats(match, 'team2')
        team2_stats['won'] = 1 - match['team1_won']
        team_history[team2_name].append(team2_stats)

        # Updating each team's recent performances (by averaging their match-specific performance for the N latest matches)
        team1_features = calculate_team_features(team1_name, team_history, HISTORY_SIZE)
        team2_features = calculate_team_features(team2_name, team_history, HISTORY_SIZE)

        # Save features for this match
        match_features = {'game_link': match['game_link']}
        for stat_name, stat_value in team1_features.items(): match_features[f'team1_hist_{stat_name}'] = stat_value
        for stat_name, stat_value in team2_features.items(): match_features[f'team2_hist_{stat_name}'] = stat_value

        # Compute map history (wins and total games played)
        team_map_history = update_team_map_history(team1_name, map_name, match['team1_won'] == 1, team_map_history)
        team_map_history = update_team_map_history(team2_name, map_name, match['team1_won'] == 0, team_map_history)

        # Add map-specific win rate
        match_features['team1_map_winrate'] = get_team_map_winrate(team1_name, map_name, team_map_history)
        match_features['team2_map_winrate'] = get_team_map_winrate(team2_name, map_name, team_map_history)

        # Updating map averages (only used to store teams data)
        teams_avg[team1_name] = team1_features  # We do it like that as the dataset is traversed in chronological order,
        teams_avg[team2_name] = team2_features  # thus the most up-to-date features of a team are the last ones.

        if "maps_winrate" not in teams_avg[team1_name]: teams_avg[team1_name]["maps_winrate"] = {}
        if map_name not in teams_avg[team1_name]["maps_winrate"]:
            teams_avg[team1_name]["maps_winrate"][map_name] = match_features['team1_map_winrate']

        if "maps_winrate" not in teams_avg[team2_name]: teams_avg[team2_name]["maps_winrate"] = {}
        if map_name not in teams_avg[team2_name]["maps_winrate"]:
            teams_avg[team2_name]["maps_winrate"][map_name] = match_features['team2_map_winrate']

        match_features['target'] = match['team1_won']
        features.append(match_features)

        if int(idx) % 10000 == 0:
            print(f'Processed {idx}/{len(df)} matches')

    features_df = pd.DataFrame(features)

    # =========== SAVING TEAMS STATS ===========
    teams_stat = {}
    for team in teams_avg:
        true_team = team.strip()
        space_team = true_team + " "
        team_name = true_team.replace(",", "")
        if team_name in teams_stat: continue
        if team_name not in teams_avg or space_team not in teams_avg:
            teams_stat[team_name] = {}
            for stat in teams_avg[team]:
                if stat == "maps_winrate":
                    teams_stat[team_name]["maps_winrate"] = {}
                    for map in teams_avg[team]["maps_winrate"]:
                        teams_stat[team_name]["maps_winrate"][map] = round(teams_avg[team]["maps_winrate"][map], 2)
                else:
                    teams_stat[team_name][stat] = round(teams_avg[team][stat], 2)
            continue

        teams_stat[team_name] = {}
        for stat in teams_avg[true_team]:
            if stat != "maps_winrate":
                teams_stat[team_name][stat] = round((teams_avg[true_team][stat] + teams_avg[space_team][stat]) * 0.5, 2)
            else:
                teams_stat[team_name]["maps_winrate"] = {}
                for map in teams_avg[true_team]["maps_winrate"]:
                    if map in teams_avg[space_team]["maps_winrate"]: teams_stat[team_name]["maps_winrate"][map] = round((teams_avg[true_team]["maps_winrate"][map] + teams_avg[space_team]["maps_winrate"][map]) * 0.5, 2)
                    else: teams_stat[team_name]["maps_winrate"][map] = teams_avg[true_team]["maps_winrate"][map]
                for map in teams_avg[space_team]["maps_winrate"]:
                    if map not in teams_avg[true_team]["maps_winrate"]: teams_stat[team_name]["maps_winrate"][map] = teams_avg[space_team]["maps_winrate"][map]

    print(teams_stat)

    with open("team_stats.csv", "w") as f: f.write("")   # Reset the document
    with open("team_stats.csv", "a") as f:
        for team in teams_stat:
            result = "name:" + team + ","
            for stat in teams_stat[team]:
                if stat != "maps_winrate": result += stat + ":" + str(teams_stat[team][stat])
                else:
                    result += "maps_winrate:{"
                    for map in teams_stat[team]["maps_winrate"]: result += map + ":" + str(teams_stat[team]["maps_winrate"][map]) + ","
                    result = result[:-1] + "}"
                result += ","
            result = result[:-1] + "\n"
            f.write(result)


    # ========== DIFFERENCES BETWEEN TEAMS STATS ==========
    for stat in ['win_rate', 'avg_rating', 'avg_kills', 'avg_deaths', 'avg_kast', 'avg_adr', 'avg_kddiff', 'avg_fkdiff', 'rounds_won']:
        features_df[f'{stat}_diff'] = features_df[f'team1_hist_{stat}'] - features_df[f'team2_hist_{stat}']

    # Add map-specific win rate difference
    features_df['map_winrate_diff'] = features_df['team1_map_winrate'] - features_df['team2_map_winrate']


    # ========== TRAINING & TESTING SETS ==========
    useless_columns = ['team1_hist_avg_kills', 'team1_hist_avg_deaths', 'team1_hist_avg_kast', 'team1_hist_avg_adr', 'team1_hist_avg_kddiff', 'team1_hist_avg_fkdiff', 'team1_hist_rounds_won', 'team2_hist_win_rate', 'team2_hist_avg_rating', 'team2_hist_avg_kills', 'team2_hist_avg_deaths', 'team2_hist_avg_kast', 'team2_hist_avg_adr', 'team2_hist_avg_kddiff', 'team2_hist_avg_fkdiff', 'team2_hist_rounds_won', 'team1_map_winrate', 'team2_map_winrate', 'team1_hist_win_rate', 'team1_hist_avg_rating']
    X = features_df.drop(columns=['game_link', 'target'] + useless_columns)
    print(X.columns)
    y = features_df['target']

    # Chronological split, 80/20
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


    # ===== DATA SCALING =====
    X_train = X_train.apply(pd.to_numeric).fillna(0)
    X_test = X_test.apply(pd.to_numeric).fillna(0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    dump(scaler, "standard_scaler.bin")


    # ===== TRAIN MODEL =====
    model = XGBClassifier(
        n_estimators=90,
        learning_rate=0.14,
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss'
    )

    print('\nTraining model...')
    model.fit(X_train_scaled, y_train)
    print(model)
    model.save_model("xgbcmodel.json")
    print("Model saved as 'xgbcmodel.json'")

    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    train_f1 = f1_score(y_train, y_pred_train)
    test_f1 = f1_score(y_test, y_pred_test)

    print(f'Train Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}')
    print(f'Test  Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}')
    print("\n" + classification_report(y_test, y_pred_test))

    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print('Top 15 most important features:')
    print(importances.head(15))

    # ========== MODEL EVALUATION ==========
    y_pred_baseline = (features_df['team1_hist_win_rate'] > features_df['team2_hist_win_rate']).astype(int)

    # For test set only (using same chronological split)
    y_pred_baseline_test = y_pred_baseline.iloc[split_idx:]

    baseline_acc = accuracy_score(y_test, y_pred_baseline_test)
    baseline_f1 = f1_score(y_test, y_pred_baseline_test)

    print(f'Baseline Accuracy: {baseline_acc:.4f}, F1: {baseline_f1:.4f}')
    print(f'Model Test Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}')
    print(f'Improvement over baseline: {((test_acc / baseline_acc) - 1) * 100:.2f}% accuracy')
    print(X_train.columns)
