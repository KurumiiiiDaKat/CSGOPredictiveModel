import pandas as pd
from xgboost import XGBClassifier
from joblib import load

teams = {}

def available_maps():
    df = pd.read_csv('./database/historic_games_list.csv')
    maps = {}
    for id, match in df.iterrows():
        map_name = match["map_name_short"]
        if map_name not in maps: maps[map_name] = 0
        maps[map_name] += 1
    return maps

def retrieve_teams():
    with open("team_stats.csv", "r") as f:
        raw = f.readlines()
    result = []
    for line in raw:
        a, b = line.index(":"), line.index(",")
        result.append(line[a+1:b])
    return result

def retrieve_maps():
    with open("team_stats.csv", "r") as f:
        raw = f.readlines()
    mapsRaw = []
    for line in raw:
        a, b = line.index("{"), line.index("}")
        mapsRaw.append(line[a+1:b])
    mapsWithScore = []
    for r in mapsRaw:
        diffMaps = r.split(",")
        mapsWithScore += diffMaps
    result = []
    for x in mapsWithScore: result.append(x[:x.index(":")])
    result = list(set(result))
    return result

def cache_teams():
    global teams
    with open("team_stats.csv", "r") as f:
        lines = f.readlines()

    teams = {}
    for l in lines:
        startname, endname = l.index(":"), l.index(",")
        teamname = l[startname+1:endname]
        teams[teamname] = {}
        mapstart = l.index("{")
        comas = [i for i in range(len(l)) if i < mapstart and l[i] == ',']
        for i in comas:
            if i == comas[-1]:
                teams[teamname]["maps_winrate"] = {}
                a, b = l.index("{"), l.index("}")
                m = l[a+1:b]
                mapsWithStat = m.split(",")
                for map in mapsWithStat:
                    c = map.split(":")
                    teams[teamname]["maps_winrate"][c[0]] = float(c[1])
            else:
                a, b = l[i+1:].index(":"), l[i+1:].index(",")
                teams[teamname][l[i+1:i+a+1]] = float(l[i+a+2:i+b+1])
    return teams

def compare_teams(team_1, team_2, map):
    global teams
    model = XGBClassifier()
    model.load_model("xgbcmodel.json")

    t1_stats, t2_stats = teams[team_1], teams[team_2]
    input = {}

    t1_map = t1_stats["maps_winrate"][map] if map in t1_stats["maps_winrate"] else 0.5
    t2_map = t2_stats["maps_winrate"][map] if map in t2_stats["maps_winrate"] else 0.5

    input["win_rate_diff"] = t1_stats["win_rate"] - t2_stats["win_rate"]
    for stat in ["rating", "kills", "deaths", "kast", "adr", "kddiff", "fkdiff"]:
        input["avg_" + stat + "_diff"] = t1_stats["avg_" + stat] - t2_stats["avg_" + stat]
    input["rounds_won_diff"] = t1_stats["rounds_won"] - t2_stats["rounds_won"]
    input["map_winrate_diff"] = t1_map - t2_map

    for stat in input: input[stat] = [input[stat]]
    df_input = pd.DataFrame.from_dict(input)

    scaler = load("standard_scaler.bin")
    scaled_input = scaler.transform(df_input)

    # print(df_input.dtypes)
    # print(df_input.iloc[0].values)  # Affiche les vraies valeurs brutes

    # print(f"Stats T1: {t1_stats['win_rate']} | Stats T2: {t2_stats['win_rate']}")

    response = model.predict(scaled_input)
    return response
