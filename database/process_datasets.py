import pandas as pd

def game_data_dataset():
    df = pd.read_csv('./database/game_data_rh.csv')
    A = list(df.columns)
    B = list(df.iloc[0].values)
    for i in range(len(A)): print("(" + str(i) + "," + str(df[A[i]].dtype) + ") " + str(A[i]) + " : " + str(B[i]))
    print("==============================================================")


    # Drop useless columns
    df.drop(columns="Unnamed: 0", inplace=True)
    df.drop(columns="team2_half2_t", inplace=True)

    # Replace '-' with '0'.
    df.replace("-", 0.0, inplace=True)

    # Convert columns to single type.
    cols = [15,17,24,26,33,35,42,44,51,53,60,62,69,71,78,80,87,89,96,98]
    for c in cols: df[df.columns[c-1]] = df[df.columns[c-1]].astype(float)

    # Process wrongly formatted values.
    for t in [1, 2]:
        for p in range(1, 6):
            # K/HS column written : <n°kills> (<n°hs>)
            khs_col = f'team{t}_p{p}_khs'
            if khs_col in df.columns:
                # Parse kills and headshots
                kills, hs = zip(*df[khs_col].apply(str.split))
                df[f'team{t}_p{p}_kills'] = kills
                df[f'team{t}_p{p}_hs'] = tuple(map(lambda x: x.replace("(", "").replace(")", ""), hs))
                df.drop(columns=khs_col, inplace=True)    # Removing the original K/HS column.

            # KAST column written in percentages
            df[f'team{t}_p{p}_kast'] = df[f'team{t}_p{p}_kast'].apply(lambda x: round(0.01 * float(str(x).replace("%", "")), 2))

            # ASSISTS column with unexpected parentheses
            df[f'team{t}_p{p}_assists'] = df[f'team{t}_p{p}_assists'].apply(lambda x: x.split()[0])

    # Print dataset
    A = list(df.columns)
    B = list(df.iloc[0].values)
    for i in range(len(A)): print("(" + str(i) + "," + str(df[A[i]].dtype) + ") " + str(A[i]) + " : " + str(B[i]))
    #for i in range(len(A)): print("(" + str(i) + ") " + str(A[i]) + " : " + str(B[i]))

    # Saving dataset
    df.to_csv("./database/game_data_processed.csv")
