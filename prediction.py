import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.metrics import root_mean_squared_error

from sklearn.model_selection import cross_val_score

#some comments are code that i used to check or visualise the data
#i kept these as comments instead of deleting so I can refer back to them if needed

nba_data = pd.read_csv("final_nba_data.csv")


#initial inputs for model
'''
feature_cols = ['FGM', 'FG3M', 'FTM', 'FG_PCT', 'AST', 'DREB', 'OREB', 'STL', 'PLUS_MINUS', 'PTS',
                'EFG_score', 'Performance_index', 'Best_player_efficiency', 'Team_offensive_score',
                'Team_defensive_score', 'Team_passing_efficiency', 'Team_faults_score',
                'game_type_num', 'home_game']
'''

#inputs for model
feature_cols = ['FG3M', 'FTM', 'AST', 'DREB', 'OREB', 'STL', 'best_player_efficiency', 'team_offensive_score',
                'team_defense_score', 'team_passing_efficiency', 'team_faults_score',
                'game_type_num', 'home_game']

#set inputs
X = nba_data[feature_cols]

#target for regression
y_pts = nba_data['PTS']

#train to test split
X_train_pts, X_test_pts, y_train_pts, y_test_pts = train_test_split(X, y_pts, test_size=0.2, random_state=42)

#model for regression
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)
rf_reg.fit(X_train_pts, y_train_pts)


#predict using model
y_pred_pts = rf_reg.predict(X_test_pts)

#Results from regression
print("PTS Regression")
print("MAE:", mean_absolute_error(y_test_pts, y_pred_pts))

rmse = root_mean_squared_error(y_test_pts, y_pred_pts)
print("RMSE:", rmse)
print("R²:", r2_score(y_test_pts, y_pred_pts))
cv_scores_pts = cross_val_score(rf_reg, X, y_pts, cv=5, scoring='neg_mean_squared_error')
print("CV RMSE (PTS):", np.sqrt(-cv_scores_pts).mean())

#target for classification
y_wl = nba_data['WL']

#train to test split
X_train_wl, X_test_wl, y_train_wl, y_test_wl = train_test_split(X, y_wl, test_size=0.2, random_state=42)

#model for classification
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)
rf_clf.fit(X_train_wl, y_train_wl)

#prediction
y_pred_wl = rf_clf.predict(X_test_wl)

#classification results
print("\n---- WL Classification ----")
print("Accuracy:", accuracy_score(y_test_wl, y_pred_wl))
print("Confusion Matrix:\n", confusion_matrix(y_test_wl, y_pred_wl))
cv_scores_wl = cross_val_score(rf_clf, X, y_wl, cv=5, scoring='accuracy')
print("CV Accuracy (WL):", cv_scores_wl.mean())

#features regression
importances_reg = rf_reg.feature_importances_
feat_importance_reg = pd.Series(importances_reg, index=feature_cols).sort_values(ascending=False)

#features classification
importances_clf = rf_clf.feature_importances_
feat_importance_clf = pd.Series(importances_clf, index=feature_cols).sort_values(ascending=False)

#plot them
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
feat_importance_reg.plot(kind='barh')
plt.title("Feature Importance (PTS - Regression)")
plt.gca().invert_yaxis()

plt.subplot(1, 2, 2)
feat_importance_clf.plot(kind='barh')
plt.title("Feature Importance (WL - Classification)")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()
