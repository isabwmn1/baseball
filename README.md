# baseball
Repository for CSE 163 Baseball Machine Learning Project

-- Project Overview --
This project uses the Python library pybaseball to collect and analyze Major League
Baseball roster data ove the past five seasons. The goal is to explore trends in team 
success, team roster, and usage of different statistics to quantify success. By 
combining data collection and analysis, this project provides insights into how individual
player performance affects overall team success. Additionally, two models will be built
to illustrate whether the Wins Above Replacement (WAR) statistic or individual, player-
based statistics (i.e. batting statistics for batter) develop a more accurate model.

-- Objectives --
- Collect relevant roster-related statistics for the 26-man rosters of all MLB teams across
  5 seasons, 4 season to be used as training data and 1 to be used as testing data
- Write code to obtain dataset using pybaseball, and clean and structure dataset for analysis
- Aggregate player statistics into team-level features
- Build and train a machine learning model using Scikit-learn
- Predict team success metrics
- Create visualizations to communicate findings

-- Machine Learning Approach --
- Inputs:
  Overall stat, Wins Above Replacement (WAR)
  Aggregated Batting Stats + Aggregated Pitching States based on player
- Outputs:
  Season wins out of 162 games

-- Languages/Libraries Used --
- Python
- Scikit-learn
- pybaseball
- requests (for webscraping)
- pandas, numpy, matplotlib
- Jupyter Notebook (for visualization)


-- Attached Python Files --
- get_WAR_data.py: Using pybaseball, retrieves batter and pitcher WAR data for years 2010-2019, and stores these datasets in CSV files. Retrieves the win data for each team, each year, and is reformatted and combined with the batter and pitcher WAR data. Additionally, combines this with the world_series_winners.csv dataset, which adds in a column indicating which team won the world series. Outputs the complete WAR data set as csv file WAR_2010_2019.csv.
- data_collection.py: Using the MLB stats API, pulls necessary player statistics for the mulit-statistic dataset, scraping the data one year at a time. Outputs datasets combined_labels.csv and combined_feats.csv, which contain the labels and features passed into the model for the multi-statistic dataset.
- analysis.py: Uses the outputted files from data_collection.py to train and test machine learning models on the data. This script first revises the datasets to only include desired statistics, splits the data into train and test datasets, then trains and tests 3 different machine learning models on the data. The outputs of these models' predictions vs real win data is plotted, and these plots are saved as rf_analysis.png, ridge_analysis.png, and gpr_analysis.png
- ML_WAR.py: Uses the outputted dataset from get_WAR_data.py and trains multiple different machine learning models using the data. Also, it tests the models' accuracy when using WAR data to predict the number of wins, and plots the predicted wins against the actual wins for each model. Plots are saved as decision_tree_WAR.png, linear_regression_WAR.png, and random_forest_WAR.png. Additionally, trains and tests a model to calculate how good of a metric WAR is for predicting the world series winner. A confusion matrix for this model is stored in the file 
- EDA_war_visuals.py: Used for the EDA milestone of this final project. Uses the initial batter and pitcher WAR datasets for exploratory data analysis plots.


-- Procedure to Reproduce Results --
1. First, make sure the required libraries are installed. Scikit-learn, pybaseball, pandas, numpy, matplotlib, and requests (for the webscraping data_collection.py) are necessary. Additionally, make sure the "world_series_winners.csv" file is downloaded for future use. This is the only CSV file you intially need before running code.
2. Run the get_WAR_data.py and data_collection.py scripts. These scripts will acquire the necessary datasets needed for the ML portion, and will do it automatically.
3. Run the analysis.py and ML_WAR.py. These scripts will train and test the machine learning models and output the necessary graphs showing the results, in the png files detailed above.
