'''
Isabelle Bowman, Kyle Chung, Jamie Stenwick
'''
import pandas as pd
import requests


TEAMS_URL = 'https://statsapi.mlb.com/api/v1/teams?sportId=1'


def get_team_ids(teams_url: str = TEAMS_URL) -> list[int]:
    '''Returns a dictionary for all the teams mapping the
    team name to their id's'''
    teams_list = []
    teams_data = requests.get((teams_url)).json()
    for team in teams_data['teams']:
        teams_list.append(team['id'])
    return teams_list


def get_player_ids(team_id: int, year: int) -> list[tuple[int, str]]:
    '''Returns a list of tuples with player ids
    and their pitching status for the specified team'''

    roster_url = (f'https://statsapi.mlb.com/api/v1/teams/{team_id}'
                  f'/roster?rosterType=active&date={year}-04-01')
    players_list = []

    roster_data = requests.get(roster_url).json()

    for person in roster_data['roster']:

        if person['position']['type'] != 'Pitcher':
            group = 'hitting'
        else:
            group = 'pitching'

        players_list.append((person['person']['id'], group))

    return players_list


def get_training_data(year: int) -> pd.DataFrame:
    '''Takes a year as input and returns the DataFrame
    that corresponds to that year's training data for our model'''

    stat_rows: list[dict[str, int | float]] = []
    team_list = get_team_ids()

    for team in team_list:
        player_list = get_player_ids(team, year)

        for player in player_list:
            player_dict = {}

            player_dict['id'] = player[0]
            player_dict['team'] = team
            player_dict['year'] = year

            api_url = (f'https://statsapi.mlb.com/api/v1/people/{player[0]}/'
                       f'stats?stats=byDateRange&group={player[1]}&startDate='
                       f'{year - 2}-04-01&endDate={year}-04-01')

            player_stats = requests.get(api_url).json()

            # Check if they have stats or not
            if player_stats['stats'][0]['splits']:

                stat_dict = player_stats['stats'][0]['splits'][0]['stat']
                player_dict.update(stat_dict)

            stat_rows.append(player_dict)

    stats_df = pd.DataFrame(stat_rows)

    return stats_df
