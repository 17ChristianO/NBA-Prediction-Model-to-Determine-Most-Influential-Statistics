import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#some comments are code that i used to check or visualise the data
#i kept these as comments instead of deleting so I can refer back to them if needed

nba_data = pd.read_csv("cleaned_nba_data.csv")

#convert WL into binary
nba_data["WL"] = nba_data["WL"].map({"W": 1, "L": 0})

#convert to numeric: 0 = regular, 1 = playoff
nba_data["game_type_num"] = nba_data["game_type"].map({"regular": 0, "playoff": 1})

#passing efficiency
nba_data['passing_efficiency'] = nba_data['AST'] / nba_data['TOV']
nba_data['passing_efficiency'] = nba_data['passing_efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)

#effective field goal percentage
nba_data['eFG_score'] = (nba_data['FGM'] + 0.5 * nba_data['FG3M']) / nba_data['FGA']

#performance statistic to represent multiple player features
nba_data['performance_index'] = (
    nba_data['FGM'] * 2 +          
    nba_data['FG3M'] * 1 +         
    nba_data['FTM'] * 1 +          
    nba_data['AST'] * 1.5 +        
    nba_data['REB'] * 1.2 +
    nba_data['STL'] * 1.2 -
    nba_data['TOV'] * 1.5          #penalise turnovers
)

#check if the game was home or away (home 1 away 0)
nba_data['home_game'] = nba_data['matchup'].str.contains('vs.').astype(int)

#player_efficiency to see how much they impact the court
nba_data['player_efficiency'] = (
    nba_data['points'] * 0.35 +
    nba_data['assists'] * 0.2 +
    nba_data['reboundsTotal'] * 0.2 +
    nba_data['steals'] * 0.1 +
    nba_data['blocks'] * 0.1 -
    nba_data['turnovers'] * 0.15 -
    nba_data['foulsPersonal'] * 0.05 +
    nba_data['plusMinusPoints'] * 0.3
)

nba_data['player_offensive_score'] = (
    nba_data['points'] * 0.4 +          
    nba_data['assists'] * 0.2 +         
    nba_data['reboundsOffensive'] * 0.2 +  
    nba_data['threePointersMade'] * 0.1 +  
    nba_data['freeThrowsMade'] * 0.1       #reward drawing fouls
)

#calculates a weighted player defence score
nba_data['player_defense_score'] = (
    nba_data['reboundsDefensive'] * 0.3 +
    nba_data['steals'] * 0.5 +
    nba_data['blocks'] * 0.4
)

#calculates a weighted player fault score
nba_data['player_faults_score'] = (
    nba_data['turnovers'] * 0.7 +
    nba_data['foulsPersonal'] * 0.3
)

#player passing efficiency
nba_data['player_passing_efficiency'] = nba_data['assists'] / nba_data['turnovers']
nba_data['player_passing_efficiency'] = nba_data['player_passing_efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)

#grouping by team per game
group_cols = ['gameId', 'teamName']

#best player efficiency (star player)
idx = nba_data.groupby(['gameId', 'teamName'])['player_efficiency'].idxmax()
best_player_details = nba_data.loc[idx, ['gameId', 'teamName', 'personName', 'player_efficiency']]
best_player_details.rename(columns={
    'player_efficiency': 'best_player_efficiency',
    'personName': 'best_player_name'
}, inplace=True)


#team total scores using players
offensive_score = nba_data.groupby(group_cols)['player_offensive_score'].sum().reset_index()
defense_score = nba_data.groupby(group_cols)['player_defense_score'].sum().reset_index()
fault_score = nba_data.groupby(group_cols)['player_faults_score'].sum().reset_index()
passing_efficiency_score = nba_data.groupby(group_cols)['player_passing_efficiency'].sum().reset_index()

#rename the columns appropriately
offensive_score.rename(columns={'player_offensive_score': 'team_offensive_score'}, inplace=True)
defense_score.rename(columns={'player_defense_score': 'team_defense_score'}, inplace=True)
fault_score.rename(columns={'player_faults_score': 'team_faults_score'}, inplace=True)
passing_efficiency_score.rename(columns={'player_passing_efficiency': 'team_passing_efficiency'}, inplace=True)

# ----- Merge into your final model dataset -----
#nba_data = cleaned_nba_data.copy()  # or replace with your existing final dataset

nba_data = nba_data.merge(best_player_details, on=group_cols, how='left')
nba_data = nba_data.merge(offensive_score, on=group_cols, how='left')
nba_data = nba_data.merge(defense_score, on=group_cols, how='left')
nba_data = nba_data.merge(fault_score, on=group_cols, how='left')
nba_data = nba_data.merge(passing_efficiency_score, on=group_cols, how='left')

#nba_data.to_csv("final_nba_data.csv", index=False)
