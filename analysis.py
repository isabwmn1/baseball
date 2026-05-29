'''Isabelle Bowman, Kyle Chung, Jamie Stenwick'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def prep_features(data_path: str = 'features/combined.csv') -> pd.DataFrame:
    '''Feat prep docstring'''

    # Manipulating feature data to be fed to regressor
    df_features = pd.read_csv(data_path)
    df_features = df_features.drop(columns=["Unnamed: 0"])
    cols = ["ops", 'era', 'inningsPitched', 'plateAppearances']
    df_features[cols] = df_features[cols].apply(pd.to_numeric, errors="coerce")

    # Hitter feature stats
    hit_mask = df_features['group'] == 'hitting'
    df_features['hitterPA'] = df_features[hit_mask]['plateAppearances']
    df_features['norm_ops'] = df_features['ops'] * df_features['hitterPA']

    # Pitcher feature stats
    df_features['pitcherIP'] = df_features[~hit_mask]['inningsPitched']
    df_features['norm_era'] = df_features['era'] * df_features['pitcherIP']

    # Non-rookies
    df_features['veteran'] = df_features['gamesPlayed'].notna()

    # Set MultiIndex
    df_features = df_features[["year", "team", 'veteran', 'hitterPA',
                               'norm_ops', 'norm_era', 'pitcherIP'
                               ]]
    df_features = df_features.set_index(["year", "team"])
    df_features = df_features.sort_index()

    # Make features weighted sums, as well as info about volume and rookies
    team_feats = df_features.groupby(level=["year", "team"]).agg({
        "norm_era": "sum", "norm_ops": "sum", 'hitterPA': 'sum',
        'pitcherIP': 'sum', 'veteran': 'sum'
        })
    team_feats['weight_ops'] = team_feats["norm_ops"] / team_feats["hitterPA"]
    team_feats["weight_era"] = team_feats["norm_era"] / team_feats["pitcherIP"]
    team_feats = team_feats.drop(columns=["norm_ops", "norm_era"])

    return team_feats


def prep_labels(data_path: str = 'labels/combined.csv') -> pd.DataFrame:
    '''Label prep docstring'''

    # Manipulating label data to be fed to regressor
    df_labels = pd.read_csv(data_path)
    df_labels = df_labels.drop(columns=["Unnamed: 0"])

    df_labels = df_labels.set_index(["year", "team"])
    df_labels = df_labels.sort_index()

    return df_labels


def analyze_model(feats: pd.DataFrame, labels: pd.DataFrame,
                  algo: str = 'gpr'
                  ) -> None:
    '''ML docstring'''

    feats = feats.dropna()
    labels = labels.loc[feats.index]

    # Make sure the data matches up, and define training and testing data
    assert feats.index.equals(labels.index)

    feats_arr = feats.to_numpy()
    labels_arr = labels.to_numpy().ravel()

    feats_train, feats_test, labels_train, labels_test = train_test_split(
        feats_arr, labels_arr, test_size=0.2
        )

    # Define kernel and fit GPR model
    if algo == 'gpr':
        kernel = (ConstantKernel(1.0) * RBF(length_scale=[
            1.0, 1.0, 1.0, 1.0, 1.0], length_scale_bounds=(1e-3, 1e3))
            + WhiteKernel(noise_level=0.1)
                  )

        model = make_pipeline(StandardScaler(), GaussianProcessRegressor(
            kernel=kernel, normalize_y=True,
            n_restarts_optimizer=10, random_state=0)
            )
    elif algo == 'ridge':
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    elif algo == 'rf':
        model = RandomForestRegressor(random_state=0, n_estimators=500)
    else:
        raise ValueError("Algorithm must either be 'gpr', 'ridge', or 'rf'")

    model.fit(feats_train, labels_train)

    # Predict on test points
    labels_pred = model.predict(feats_test)

    # Accuracy measures
    r_squared = r2_score(labels_test, labels_pred)
    mae = mean_absolute_error(labels_test, labels_pred)

    # Plot
    plt.scatter(labels_test, labels_pred, color='blue')
    plt.plot(labels_test, labels_test, color='orange', label='Ideal')

    text = f"$R^2$ = {r_squared:.3f}\nMAE = {mae:.2f}"

    plt.text(
        0.05, 0.95, text,
        transform=plt.gca().transAxes,
        ha='left',
        va='top',
        bbox=dict(facecolor="white", alpha=0.8)
    )

    plt.xlim(45, 110)
    plt.ylim(45, 110)
    plt.xlabel("True wins")
    plt.ylabel("Predicted wins")
    plt.title('True vs Predicted wins')
    plt.legend()
    plt.savefig(f'{algo}_analysis.png')
    plt.close()


def main():
    '''Main function documentation'''

    features = prep_features()
    labels = prep_labels()

    analyze_model(features, labels, 'gpr')
    analyze_model(features, labels, 'ridge')
    analyze_model(features, labels, 'rf')


if __name__ == '__main__':
    main()
