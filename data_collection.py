'''
Isabelle Bowman, Kyle Chung, Jamie Stenwick
'''
import pandas as pd
import requests
from pathlib import Path


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

    stat_rows: list[dict] = []
    team_list = get_team_ids()

    for i, team in enumerate(team_list):
        player_list = get_player_ids(team, year)
        print(f'{i+1}/30 teams loaded in', year)

        for player in player_list:
            player_dict = {}
            group = player[1]

            player_dict['id'] = player[0]
            player_dict['team'] = team
            player_dict['year'] = year
            player_dict['group'] = group

            api_url = (f'https://statsapi.mlb.com/api/v1/people/{player[0]}/'
                       f'stats?stats=byDateRange&group={player[1]}&startDate='
                       f'{year - 2}-04-01&endDate={year}-04-01')

            player_stats = requests.get(api_url).json()

            # Check if they have stats or not
            if player_stats['stats']:
                if player_stats['stats'][0]['splits']:

                    stat_dict = player_stats['stats'][0]['splits'][0]['stat']
                    player_dict.update(stat_dict)

            stat_rows.append(player_dict)

    stats_df = pd.DataFrame(stat_rows)

    print(stats_df.columns)
    print(stats_df[['id', 'team', 'year', 'group']].head())

    return stats_df


def get_label_data(year: int) -> pd.DataFrame:
    '''Takes a year as input and returns a dataframe with the
    year, team id, and number of wins as columns'''

    api_url = (f'https://statsapi.mlb.com/api/v1/standings?leagueId=103,104'
               f'&season={year}&standingsTypes=regularSeason')
    division_records = requests.get(api_url).json()['records']
    record_dict = {}

    for division in division_records:
        for team in division['teamRecords']:
            id = str(team['team']['id'])
            wins = team['wins']
            record_dict[id] = wins

    team_rows: list[dict] = []

    for team in record_dict:
        print(int(team))
        team_dict = {}

        team_dict['year'] = year
        team_dict['team'] = int(team)
        team_dict['wins'] = record_dict[team]

        team_rows.append(team_dict)

    labels_df = pd.DataFrame(team_rows)

    return labels_df


def main():
    '''Main function documentation'''
    '''
    years = [1998, *range(2004, 2020), *range(2023, 2026)]
    Path("features").mkdir(exist_ok=True)
    Path("labels").mkdir(exist_ok=True)

    for year in years:
        df_feat = get_training_data(year)
        df_feat.to_csv(f'features/feature_{year}.csv')

        df_label = get_label_data(year)
        df_label.to_csv(f'labels/label_{year}.csv')
    '''
    folders = ['features', 'labels']

    for data_folder in folders:
        folder = Path(data_folder)
        df = pd.concat([pd.read_csv(file) for file in folder.glob("*.csv")],
                       ignore_index=True)
        df.to_csv(data_folder + "/combined.csv", index=False)


if __name__ == '__main__':
    main()
