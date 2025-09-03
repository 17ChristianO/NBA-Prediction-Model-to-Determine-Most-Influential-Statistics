import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#some comments are code that i used to check or visualise the data
#i kept these as comments instead of deleting so I can refer back to them if needed

#load in cvs files and attach them to a variable
rbox_scores1 = pd.read_csv("regular_season_box_scores_2010_2024_part_1.csv")
rbox_scores2 = pd.read_csv("regular_season_box_scores_2010_2024_part_2.csv")
rbox_scores3 = pd.read_csv("regular_season_box_scores_2010_2024_part_3.csv")
playoff_totals = pd.read_csv("play_off_totals_2010_2024.csv")
playoff_box_scores = pd.read_csv("play_off_box_scores_2010_2024.csv")
regular_season_totals = pd.read_csv("regular_season_totals_2010_2024.csv")

#merge the box scores into a single dataframe
regular_season_box_scores = pd.concat([rbox_scores1, rbox_scores2, rbox_scores3], ignore_index=True)

#remove duplicates
regular_season_box_scores.drop_duplicates(inplace=True)
regular_season_totals.drop_duplicates(inplace=True)
playoff_box_scores.drop_duplicates(inplace=True)
playoff_totals.drop_duplicates(inplace=True)

drop_jersey_num = ["jerseyNum","position"]
regular_season_box_scores.drop(columns=[col for col in drop_jersey_num if col in regular_season_box_scores.columns], inplace=True)
playoff_box_scores.drop(columns=[col for col in drop_jersey_num if col in playoff_box_scores.columns], inplace=True)

#find a way to drop the rows where regular_season_box_scores["minutes"].isna() = True
drop_no_mins = regular_season_box_scores[(regular_season_box_scores["minutes"].isna())].index
drop_no_mins_bo = playoff_box_scores[(playoff_box_scores["minutes"].isna())].index
regular_season_box_scores.drop(drop_no_mins, inplace=True)
playoff_box_scores.drop(drop_no_mins_bo, inplace=True)

#remove rows with missing values
#most rows with missing values mean the player did not play in the game
#regular_season_box_scores.dropna(inplace=True)
regular_season_totals.dropna(inplace=True)
#playoff_box_scores.dropna(inplace=True)
playoff_totals.dropna(inplace=True)

#convert date columns to datetime format
regular_season_totals["GAME_DATE"] = pd.to_datetime(regular_season_totals["GAME_DATE"])
playoff_totals["GAME_DATE"] = pd.to_datetime(playoff_totals["GAME_DATE"])

#rename the season totals GAME_ID
regular_season_totals.rename(columns={"GAME_ID" : "gameId"}, inplace=True)
playoff_totals.rename(columns={"GAME_ID" : "gameId"}, inplace=True)

regular_season_data = pd.merge(regular_season_box_scores, regular_season_totals, on="gameId")
playoff_data = pd.merge(playoff_box_scores, playoff_totals, on="gameId")

'''
#print(regular_season_data.isnull().sum())
#print(playoff_data.isnull().sum())

#print(regular_season_data.duplicated().sum())
#print(playoff_data.duplicated().sum())

#print(regular_season_data.dtypes)
#print(playoff_data.dtypes)
'''

regular_season_data.rename(columns={"MATCHUP": "matchup"}, inplace=True)
regular_season_data = regular_season_data.loc[:, ~regular_season_data.columns.duplicated()]

playoff_data.rename(columns={"MATCHUP": "matchup"}, inplace=True)

#print(regular_season_data.columns)

#some columns were objects and not numerical data type so i had to change
numeric_cols = ['points', 'assists', 'turnovers', 'PLUS_MINUS', 'PTS_RANK', 'PLUS_MINUS_RANK']
regular_season_data[numeric_cols] = regular_season_data[numeric_cols].apply(pd.to_numeric, errors='coerce')
playoff_data[numeric_cols] = playoff_data[numeric_cols].apply(pd.to_numeric, errors='coerce')

#the date column was an object so changed the format
regular_season_data['game_date'] = pd.to_datetime(regular_season_data['GAME_DATE'])
playoff_data['GAME_DATE'] = pd.to_datetime(playoff_data['GAME_DATE'])

#drop unnecessary columns
columns_to_drop = ["TEAM_ID", "TEAM_ABBREVIATION", "personId", "teamSlug", "teamId"]
regular_season_data.drop(columns=[col for col in columns_to_drop if col in regular_season_data.columns], inplace=True)
playoff_data.drop(columns=[col for col in columns_to_drop if col in playoff_data.columns], inplace=True)

#regular_season_data.to_csv("cleaned_regular_season_data.csv", index = False)
#playoff_data.to_csv("cleaned_playoff_data.csv", index = False)

#add a game_type column before merging
regular_season_data["game_type"] = "regular"
playoff_data["game_type"] = "playoff"

#reset the index and drop the old one to fully clean both
regular_season_data = regular_season_data.reset_index(drop=True)
playoff_data = playoff_data.reset_index(drop=True)

#print(regular_season_data.index.is_unique)
#print(playoff_data.index.is_unique)

#accidentally had 2 matchup columns in regular season data
#print(regular_season_data.columns[regular_season_data.columns.duplicated()])
#print(playoff_data.columns[playoff_data.columns.duplicated()])


#merge playoff and regular season data
cleaned_nba_data = pd.concat([regular_season_data, playoff_data], ignore_index=True)
#cleaned_nba_data.to_csv("cleaned_nba_data.csv", index = False)


#filter to only useful stats for visualisation
my_columns = ["fieldGoalsMade", "fieldGoalsAttempted", "fieldGoalsPercentage", "threePointersMade", "threePointersAttempted", "threePointersPercentage", "freeThrowsMade", "freeThrowsAttempted", "freeThrowsPercentage", "reboundsOffensive", "reboundsDefensive", "reboundsTotal", "assists", "steals", "blocks", "turnovers", "foulsPersonal", "points", "plusMinusPoints", "WL", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "BLKA", "PF", "PFD", "PTS", "PLUS_MINUS", "GP_RANK", "W_RANK", "L_RANK", "W_PCT_RANK", "MIN_RANK", "FGM_RANK", "FGA_RANK", "FG_PCT_RANK", "FG3M_RANK", "FG3A_RANK", "FG3_PCT_RANK", "FTM_RANK", "FTA_RANK", "FT_PCT_RANK", "OREB_RANK", "DREB_RANK", "REB_RANK", "AST_RANK", "TOV_RANK", "STL_RANK", "BLK_RANK", "BLKA_RANK", "PF_RANK", "PFD_RANK", "PTS_RANK", "PLUS_MINUS_RANK"]
hm_cleaned_nba_data = cleaned_nba_data[my_columns]

#make win/loss numerical
hm_cleaned_nba_data["WL"] = hm_cleaned_nba_data["WL"].map({"W": 1, "L": 0})


'''

plt.figure(figsize=(40,30))
sns.heatmap(
    hm_cleaned_nba_data
    .corr(),
    annot=True, 
    cmap="coolwarm", 
    fmt=".2f", 
    linewidths=1,
    annot_kws={"size": 5})
plt.title("Correlation heatmap of regular season nba stats")
plt.show()

# Compute correlation matrix
regular_season_matrix = hm_cleaned_nba_data
.corr()

# Unstack correlation matrix to convert it into a long-format dataframe
corr_regular_season = regular_season_matrix.unstack()

# Remove self-correlations (where a variable is correlated with itself)
corr_regular_season = corr_regular_season[corr_regular_season.index.get_level_values(0) != corr_regular_season.index.get_level_values(1)]

# Filter for correlations greater than or equal to 0.7 (positive correlations)
filtered_corr_regular_season = corr_regular_season[corr_regular_season >= 0.7]

# Filter for correlations less than or equal to -0.7 (negative correlations)
filtered_ncorr_regular_season = corr_regular_season[corr_regular_season <= -0.7]

# Sort by correlation value in ascending order (weakest first for positive correlations)
sorted_corr_regular_season = filtered_corr_regular_season.sort_values()

# Sort by correlation value in descending order (weakest first for negative correlations)
sorted_ncorr_regular_season = filtered_ncorr_regular_season.sort_values(ascending=False)

# Display the bottom 20 correlations (weakest correlations that are still >= 0.7)
print("20 weakest strong positive correlations (closest to 0.7):")
print(sorted_corr_regular_season.head(20))

# Display the bottom 20 correlations (weakest correlations that are still <= -0.7)
print("\n20 weakest strong negative correlations (closest to -0.7):")
print(sorted_ncorr_regular_season.head(20))  # Since sorted descending, use tail()

plt.figure(figsize=(40,30))
sns.heatmap(
    hm_playoff_data.corr(),
    annot=True, 
    cmap="coolwarm", 
    fmt=".2f", 
    linewidths=1,
    annot_kws={"size": 5})
plt.title("Correlation heatmap of playoff nba stats")
plt.show()

'''

#code to show WL & Points correlation to other stats
"""
# Select only numerical columns for correlation
num_cols = hm_cleaned_nba_data.select_dtypes(include=['number'])

# Compute correlation matrix
corr_matrix = num_cols.corr()

# Focus on correlations with winning (`WL`) and final score (`PTS`)
target_corr = corr_matrix[['WL', 'PTS']].sort_values(by='PTS', ascending=False)


top_corr = corr_matrix.nlargest(1,'WL')  # Show top 20 correlated stats with WL

# Plot heatmap
plt.figure(figsize=(16, 8))

# Create the heatmap with better alignment
sns.heatmap(top_corr, annot=True, cmap="coolwarm", center=0, fmt=".2f",
            annot_kws={"size": 10}, linewidths=0.5)

plt.title("Correlation of Stats with Win/Loss and Final Score")
plt.show()
"""

#code for pairplot
'''
top_corr = ["PTS", "FG_PCT", "AST", "REB", "TOV", "WL"]  # Choose important stats
sns.pairplot(hm_cleaned_nba_data[top_corr])
plt.show()
'''

#code for box plots
'''
#key statistics to compare
key_stats = ['PTS', 'REB', 'AST', 'STL', 'TOV', 'FG_PCT']

plt.figure(figsize=(12, 8))
for i, stat in enumerate(key_stats, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x=hm_cleaned_nba_data
    ["WL"], y=hm_cleaned_nba_data
    [stat])
    plt.title(f"{stat} Distribution (Win vs. Loss)")

plt.tight_layout()
plt.show()
'''

#scatter plot
'''
plt.figure(figsize=(12, 6))

# Scatterplot for PTS vs FG_PCT
sns.scatterplot(x=hm_cleaned_nba_data["FG_PCT"], 
                y=hm_cleaned_nba_data["PTS"], 
                alpha=0.5)
plt.title("Field Goal % vs. Final Score")
plt.xlabel("Field Goal Percentage")
plt.ylabel("Final Score")
plt.show()

'''

#Top 20 correlations
'''
# Get only numerical columns
num_data = hm_cleaned_nba_data.select_dtypes(include='number')

# Calculate correlation matrix
correlation_matrix = num_data.corr()

# Get the correlation values with 'WL' (excluding WL itself)
wl_correlations = correlation_matrix['WL'].drop('WL')

# Get absolute values for sorting (to include negative correlations)
top_20 = wl_correlations.abs().sort_values(ascending=False).head(20)

# Print the top 20 most correlated stats with WL
print("Top 20 stats most correlated with Win/Loss:")
for feature in top_20.index:
    print(f"{feature}: correlation = {wl_correlations[feature]:.3f}")

'''

#heatmap of top 20 correlations to W/L and points
#"""
# Select numerical features
num_data = hm_cleaned_nba_data.select_dtypes(include='number')

# Correlation matrix
corr_matrix = num_data.corr()

#heatmap of top 20 correlations to W/L and points

#get top 20 features most correlated with WL
top_20_features = corr_matrix['WL'].drop('WL').abs().sort_values(ascending=False).head(20).index.tolist()

#make it so they can compare each other
top_features = top_20_features + ['WL', 'PTS']

#subset correlation matrix for those features only
focused_corr = corr_matrix.loc[top_features, ['WL', 'PTS']]

# Plot heatmap
plt.figure(figsize=(10, 12))
sns.heatmap(focused_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation of Top 20 Features with Win/Loss and Points", fontsize=14)
plt.tight_layout()
plt.show()

#"""
