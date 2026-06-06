"""
CSE 163 Final Project
5/14/2026

This script uses the pybaseball library to pull WAR data for
pitchers and batters.
This script also creates CSV files for the pitcher and batter WAR data.
"""

from get_WAR_data import get_WAR_data
import numpy as np


def test_get_WAR_data() -> None:
    """
    Test function for the get_WAR_data function.
    This functions tests if all of the WAR data pulled is
    from the correct year, testing for the year 2010.
    Also tests for the year range 2010-2019.
    """
    batter_data, pitcher_data = get_WAR_data(2010, 2010)
    assert pitcher_data['year_ID'].unique() == [2010]
    assert batter_data['year_ID'].unique() == [2010]

    batter_data, pitcher_data = get_WAR_data(2010, 2019)
    assert (np.sort(pitcher_data['year_ID'].unique()) == np.arange(2010,
                                                                   2020)).all()
    assert (np.sort(batter_data['year_ID'].unique()) == np.arange(2010,
                                                                  2020)).all()
    # .all allows for all values in each array to be compared.


def main():
    test_get_WAR_data()


if __name__ == '__main__':

    main()
