'''
WAR Statistic Visualizations. Use pitcher data CSV and batter data CSV,
combines them, and produces visualizations from combined data.
'''

# Imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data():
    '''
    load_data() functions reads CSV files for WAR
    pitcher and batter data and returns a combined pandas
    DataFrame.
    '''
    pitch = pd.read_csv('pitcher_WAR_data_2010_2019')
    bat = pd.read_csv('batter_WAR_data_2010_2019')

    pitch["Type"] = "Pitcher"
    bat["Type"] = "Batter"

    combined_df = pd.concat([pitch, bat], ignore_index=True)
    return combined_df


def plot_war_vs_games(df):
    '''
    Function takes in the combined DataFrame and plots
    a scatterplot of all WAR statistics across all years,
    with hue set to player Type.
    '''
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    sns.despine(fig, left=True, bottom=True)

    sns.scatterplot(x="G", y="WAR", hue="Type", data=df, ax=ax)

    ax.set_title("WAR vs. Games Played")
    ax.set_xlabel("Games Played (G)")
    ax.set_ylabel("WAR")

    plt.savefig("war_plot.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_team_war_2019(df):
    '''
    Function takes in the combined DataFrame and plots a bar
    graph of the combined WAR for all teams in 2019.
    '''
    sns.set_theme(style="whitegrid")

    war_2019 = df[df["year_ID"] == 2019]

    team_war_2019 = (
        war_2019.groupby("team_ID")["WAR"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(14, 7))

    sns.barplot(
        x="team_ID",
        y="WAR",
        data=team_war_2019,
        hue="team_ID",
        palette="viridis",
        ax=ax
    )

    ax.set_title("Combined Team WAR (2019)", fontsize=16)
    ax.set_xlabel("Teams")
    ax.set_ylabel("Total WAR")

    plt.savefig("team_war_2019_bargraph.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """Main execution function."""
    df = load_data()

    plot_war_vs_games(df)
    plot_team_war_2019(df)


if __name__ == "__main__":
    main()
