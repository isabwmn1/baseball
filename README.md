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
- Build and train a machine learning model using Pytorch
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
- PyTorch
- pybaseball
- pandas, numpy, matplotlib
- Jupyter Notebook (for visualization)
