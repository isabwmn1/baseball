"""
CSE 163 Final Project
5/14/2026

This script uses the pybaseball library to pull WAR data for
pitchers and batters, along with win data for each team each year.
Also pulls win data for each team for the same year range.
This script also creates CSV files for the pitcher and batter WAR data.
"""

import pybaseball as pb
import pandas as pd
import numpy as np


def get_WAR_data(start_year: int, stop_year: int) -> tuple[pd.DataFrame,
                                                           pd.DataFrame]:
    """
    Given a starting and stopping year, uses pybaseball to return
    dataframes of the WAR data for all pitchers and batters
    in the given year range.
    """

    # Pulling batting WAR data
    batter_data = pb.bwar_bat()[['year_ID',
                                 'team_ID',
                                 'mlb_ID',
                                 'WAR']]
    batter_data = batter_data[batter_data['year_ID'].between(start_year,
                                                             stop_year)]
    # batter_data = batter_data.set_index(['year_ID', 'team_ID'])

    # Pulling pitching WAR data
    pitcher_data = pb.bwar_pitch()[['year_ID',
                                    'team_ID',
                                    'mlb_ID',
                                    'WAR']]
    pitcher_data = pitcher_data[pitcher_data['year_ID'].between(start_year,
                                                                stop_year)]
    # pitcher_data = pitcher_data.set_index(['year_ID', 'team_ID'])

    return batter_data, pitcher_data


def get_wins(year_start: int, year_end: int) -> pd.DataFrame:
    """
    Given a start and end year, returns the roster win data for the years
    between the two (inclusive).

    The season team wins data and WAR roster data use different naming schemes
    for team labels. This function formats the wins dataset's team names to be
    consistent with the WAR roster data.
    """
    # Creating array of years
    years = np.arange(year_start, year_end + 1)
    standings = pd.DataFrame()

    # Pulling the data for each year
    for year in years:
        # Getting each year's standings into DataFrame format, adding year_ID
        year_standings = pd.concat(pb.standings(year))
        year_standings['year_ID'] = year
        # Adding year's standings to complete standings array
        standings = pd.concat([standings, year_standings],
                              ignore_index=True,
                              axis=0)

    # Dropping unnecessary columns, only storing win, team, and year data.
    wins = standings.drop(labels=['L', 'W-L%', 'GB'], axis=1)

    # Dictionary used to conver team IDs
    team_name_to_abbr = {
        'Tampa Bay Rays':                  'TBR',
        'New York Yankees':                'NYY',
        'Boston Red Sox':                  'BOS',
        'Toronto Blue Jays':               'TOR',
        'Baltimore Orioles':               'BAL',
        'Minnesota Twins':                 'MIN',
        'Chicago White Sox':               'CHW',
        'Detroit Tigers':                  'DET',
        'Cleveland Indians':               'CLE',
        'Kansas City Royals':              'KCR',
        'Texas Rangers':                   'TEX',
        'Oakland Athletics':               'OAK',
        'Los Angeles Angels of Anaheim':   'LAA',
        'Seattle Mariners':                'SEA',
        'Philadelphia Phillies':           'PHI',
        'Atlanta Braves':                  'ATL',
        'Florida Marlins':                 'FLA',
        'Miami Marlins':                   'MIA',  # renamed 2012
        'New York Mets':                   'NYM',
        'Washington Nationals':            'WSN',
        'Cincinnati Reds':                 'CIN',
        'St. Louis Cardinals':             'STL',
        'Milwaukee Brewers':               'MIL',
        'Houston Astros':                  'HOU',
        'Chicago Cubs':                    'CHC',
        'Pittsburgh Pirates':              'PIT',
        'San Francisco Giants':            'SFG',
        'San Diego Padres':                'SDP',
        'Colorado Rockies':                'COL',
        'Los Angeles Dodgers':             'LAD',
        'Arizona Diamondbacks':            'ARI',
        'Cleveland Guardians':             'CLE',  # renamed 2022
        'Los Angeles Angels':              'LAA',  # renamed from 'of Anaheim'
        'Tampa Bay Devil Rays':            'TBR',  # pre-2008 name
    }

    # Converting team IDs to new column
    wins['team_ID'] = wins['Tm'].map(team_name_to_abbr)

    # Removing old wins column
    wins = wins.drop(columns='Tm')
    return wins


def get_complete_dataset(wins_df: pd.DataFrame,
                         batter_df: pd.DataFrame,
                         pitcher_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function combines the season team wins dataset and WAR roster datasets
    into one dataframe.
    The WAR data is summed for each team for each year, and is
    then combined with the wins dataset.
    """

    # Grouping data by team and year and summing WAR. NaN values are
    # skipped.
    team_batter_df = batter_df.groupby(
        ['team_ID',
         'year_ID'])['WAR'].sum(skipna=True).reset_index()
    team_pitcher_df = pitcher_df.groupby(
        ['team_ID',
         'year_ID'])['WAR'].sum(skipna=True).reset_index()

    # Merging the pitching and batting dataframes. Suffixes are created to help
    # differentiate batting vs pitching WAR.
    team_df = team_batter_df.merge(team_pitcher_df,
                                   on=['team_ID', 'year_ID'],
                                   suffixes=('_bat', '_pitch'),
                                   how='outer')

    # Merging the team WAR data and the wins to get final dataframe
    # A right merge is chosen to preserve the 'W' column that isn't present
    # in the other two datasets.
    war_df = team_df.merge(wins_df,
                           on=['team_ID', 'year_ID'],
                           how='right')

    return war_df


def main():
    # Getting win and batter data
    win_data = get_wins(2010, 2019)
    batter_data, pitcher_data = get_WAR_data(2010, 2019)

    # Getting CSV files for WAR data and wins
    batter_data.to_csv('batter_WAR_data_2010_2019.csv', index=False)
    pitcher_data.to_csv('pitcher_WAR_data_2010_2019.csv', index=False)
    win_data.to_csv('wins.csv', index=False)

    '''
    win_data = pd.read_csv('wins.csv')
    batter_data = pd.read_csv('batter_WAR_data_2010_2019.csv')
    pitcher_data = pd.read_csv('pitcher_WAR_data_2010_2019.csv')
    '''

    war_data = get_complete_dataset(win_data, batter_data, pitcher_data)

    war_data.to_csv('WAR_2010_2019.csv', index=False)


if __name__ == "__main__":
    main()
