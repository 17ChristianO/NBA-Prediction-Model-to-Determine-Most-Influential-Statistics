# 🏀 NBA Game Outcome Prediction & Feature Importance Analysis

This project investigates which player and team statistics are most influential in determining NBA game outcomes. Using both classification and regression models, I trained predictive algorithms and analysed feature importance to uncover the stats that matter most.

## 🔍 Project Overview

- **Objective:** Determine the most influential player and team statistics in predicting NBA game outcomes.
- **Approach:** 
  - Built a **classification model** to predict win/loss outcomes.
  - Built a **regression model** to predict point differentials.
  - Used **feature importance techniques** to rank statistical impact.

## 🛠 Tools & Technologies

- Python
- Pandas & NumPy
- scikit-learn
- Jupyter Notebook
- Matplotlib / Seaborn

## 💡 Skills Demonstrated

- Data cleaning & transformation
- Exploratory data analysis (EDA)
- Classification & regression modelling
- Feature importance and interpretability
- Use of Random Forest for insight generation

## 📊 Models Used

- **Random Forest Classifier**: To predict whether the home team wins.
- **Random Forest Regressor**: To predict the point difference (margin of victory).
- Feature importances were extracted from both models to analyse the most significant factors.

## 🧠 Key Findings

The models identified several high-impact features:
- **Classification (Win/Loss):** Turnovers, field goal %, defensive rebounds.
- **Regression (Point Differential):** Assists, 3-point shooting %, opponent stats.

The results showed consistency across both models, with team efficiency and control stats often being the strongest predictors.

## 📁 Dataset Access

Due to file size limits on GitHub, datasets are hosted externally:

👉 [Access Dataset Folder]([https://drive.google.com/your-dataset-link](https://drive.google.com/file/d/1bDE9hq2ixxVyANAL6WxN-1CeXs0s4S9-/view?usp=sharing))

##🧠 Discussion & Conclusion**

This project used machine learning models to identify which team and player statistics most influence NBA game outcomes. Using historical data from 2010 to 2024, both regression and classification models were implemented to predict game scores and win-loss results.

The regression model showed that offensive performance — especially the engineered feature team_offensive_score — was the most predictive of total points scored. Supporting variables like assists, three-pointers, and free throws highlighted the importance of efficient ball movement and perimeter shooting.

The classification model, on the other hand, revealed that defensive rebounds were the strongest predictor of victory, underscoring the importance of controlling the game defensively. Additional features, such as steals and assists, also played a role, while contextual variables like home-court advantage were found to be less impactful.

Across both models, team-based statistics consistently outperformed individual metrics, reinforcing the idea that basketball outcomes are driven by collective effort rather than star power alone. The feature engineering process played a key role in preventing overfitting and improving model robustness.

These findings provide strong evidence that coordinated team play, offensive execution, and defensive control are the primary statistical drivers of NBA success.
