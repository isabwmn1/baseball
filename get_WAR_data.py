"""
Isabelle Bowman, Kyle Chung, Jamie Stenwick
CSE 163 Final Project
5/14/2026

This script uses the pybaseball library to pull WAR data for
pitchers and batters.
This script also creates CSV files for the pitcher and batter WAR data.
"""

import pybaseball as pb
import pandas as pd


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
                                 'G',
                                 'WAR']]
    batter_data = batter_data[batter_data['year_ID'].between(start_year,
                                                             stop_year)]
    batter_data = batter_data.set_index(['year_ID', 'team_ID'])

    # Pulling pitching WAR data
    pitcher_data = pb.bwar_pitch()[['year_ID',
                                    'team_ID',
                                    'mlb_ID',
                                    'G',
                                    'WAR']]
    pitcher_data = pitcher_data[pitcher_data['year_ID'].between(start_year,
                                                                stop_year)]
    pitcher_data = pitcher_data.set_index(['year_ID', 'team_ID'])

    return batter_data, pitcher_data


def main():

    batter_data, pitcher_data = get_WAR_data(2010, 2019)

    print(batter_data)
    print(pitcher_data)

    # Getting CSV files for WAR data
    batter_data.to_csv('batter_WAR_data_2010_2019')
    pitcher_data.to_csv('pitcher_WAR_data_2010_2019')


if __name__ == "__main__":
    main()
