'''
Isabelle Bowman, Kyle Chung, Jamie Stenwick
Machine learning framework for WAR statistic.
Assumed linear regression fit based off off
data visualization in exploratory data analysis.
'''

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix


# Loading CSV file as pd.DataFrame
def load_data(filename):
    return pd.read_csv(filename)


# Training model function
def train_linear_model(df):
    '''
    train_linear_model function takes in the DataFrame of
    batting and pitching WAR statistics between 2010 and
    2019. Trains the model on 2010-2018 data and tests on
    2019. Assumes linear regression fit.
    '''

    # Split dataframe by training and testing
    train = df[df["year_ID"] < 2019]
    test = df[df["year_ID"] == 2019]

    # Setting features and labels
    train_feature = train[["WAR_bat", "WAR_pitch"]]
    train_label = train["W"]

    test_feature = test[["WAR_bat", "WAR_pitch"]]
    test_label = test["W"]

    model = LinearRegression()
    model.fit(train_feature, train_label)

    # Get model equation
    intercept = model.intercept_
    coefficients = model.coef_

    print(f"Intercept: {intercept}")
    print(f"Coefficients: {coefficients}")

    # Predictions
    test_pred = model.predict(test_feature)

    return model, test, test_label, test_pred


def train_tree_model(df):
    '''
    train_tree_model trains a Decision Tree Regressor
    using 2010–2018 data and tests on 2019 data. Allows
    for non-linear fitting.
    '''

    # Split dataframe by training and testing
    train = df[df["year_ID"] < 2019]
    test = df[df["year_ID"] == 2019]

    # Features and labels
    train_feature = train[["WAR_bat", "WAR_pitch"]]
    train_label = train["W"]

    test_feature = test[["WAR_bat", "WAR_pitch"]]
    test_label = test["W"]

    # Decision Tree model
    model = DecisionTreeRegressor(max_depth=3,
                                  random_state=42)

    # Train
    model.fit(train_feature, train_label)

    # Predictions
    test_pred = model.predict(test_feature)

    return model, test, test_label, test_pred


def train_forest_model(df):
    '''
    train_forest_model trains a Random Forest Regressor
    using 2010-2018 data and tests on 2019 data.
    '''
    # Split dataframe by training and testing
    train = df[df["year_ID"] < 2019]
    test = df[df["year_ID"] == 2019]

    # Features and labels
    train_feature = train[["WAR_bat", "WAR_pitch"]]
    train_label = train["W"]

    test_feature = test[["WAR_bat", "WAR_pitch"]]
    test_label = test["W"]

    # Random Forest model, uses 100 trees for more stable predictions
    # Max_depth = 5, prevents overfitting, overly complex trees
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    # Train model
    model.fit(train_feature, train_label)

    # Predictions
    test_pred = model.predict(test_feature)

    return model, test, test_label, test_pred


def train_world_series(df):
    '''
    train_world_series function predicts whether a team wins the
    World Series using the batting and pitching WAR. Trains on
    2010-2018 data and tests on 2019 data.

    won_world_series:
    0 = did not win World Series
    1 = won World Series
    '''

    # Training on 2010 - 2018
    train = df[df["year_ID"] < 2019]

    # Testing on 2019
    test = df[df["year_ID"] == 2019]

    # Setting features and labels
    train_feature = train[["WAR_bat", "WAR_pitch"]]
    train_label = train["won_world_series"]

    test_feature = test[["WAR_bat", "WAR_pitch"]]
    test_label = test["won_world_series"]

    # Setting model as Logistic Regression
    model = LogisticRegression(class_weight="balanced", random_state=42)

    # Training model
    model.fit(train_feature, train_label)
    test_pred = model.predict(test_feature)

    return model, test, test_label, test_pred


def evaluate_model(test_label, test_pred, model_name):
    '''
    evaluate_model function statistically tests accuracy of
    WAR model using both root mean squared error values and
    R^2 values.
    '''
    # calculated root mean squared error and R^2 values
    mse = mean_squared_error(test_label, test_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_label, test_pred)

    # print values, rounded to 3rd decimal place
    print(model_name)
    print("--------------------")
    print("R2 Score:", round(r2, 3))
    print("Mean Squared Error:", round(mse, 3))
    print("Root Mean Squared Error:", round(rmse, 3))

    return rmse, r2


def evaluate_world_series_model(test_label, test_pred):
    '''
    Evaluates logistic regression classifier.
    '''

    accuracy = accuracy_score(test_label, test_pred)

    matrix = confusion_matrix(test_label, test_pred)

    print("World Series Prediction Performance")
    print("----------------------------------")
    print("Accuracy:", round(accuracy, 3))
    print()
    print("Confusion Matrix:")
    print(matrix)

    return accuracy, matrix


def make_results_table(test, test_pred):
    '''
    make_results_tabel function creates a DataFrame displaying
    actual win game values, predicted win game values, and
    error.
    '''
    # copy test DataFrame (2010 - 2018), input column from
    # prediction, input column from predicted - test
    results = test[["team_ID", "W"]].copy()
    results["Predicted_W"] = test_pred
    results["Error"] = results["Predicted_W"] - results["W"]

    print("Prediction Results")
    print(results)

    return results


def plot_results(results):
    '''
    plot_results function plots the actual vs. predicted wins
    values with error bars.
    '''

    # set plot size
    plt.figure(figsize=(10, 6))

    # set ticks for number of teams using range of results
    x = np.arange(len(results))

    # Actual wins, s=80 sets dot size
    plt.scatter(x, results["W"], label="Actual Wins",
                color="red", s=80)

    # Predicted wins, error bars included, fmt=o makes scatter
    plt.errorbar(x, results["Predicted_W"],
                 yerr=np.abs(results["Error"]),
                 fmt="o", capsize=5,
                 label="Predicted Wins +/- Error")

    # Set x ticks with team names
    plt.xticks(x, results["team_ID"], rotation=45)

    # Labelling, etc.
    plt.xlabel("Teams")
    plt.ylabel("Wins")
    plt.title("2019 MLB Wins: Actual vs. Predicted Using WAR")
    plt.legend()
    plt.show()


def plot_linear_results(test_label, linear_pred, type, r2, rmse):
    '''
    plot_linear_results function creates a scatter plot
    of true wins vs. linear regression predicted wins with
    error bars and an ideal line y = x.
    '''

    # Create figure
    plt.figure(figsize=(8, 8))

    # Calculating absolute error
    errors = np.abs(linear_pred - test_label)
    plt.errorbar(test_label, linear_pred, yerr=errors,
                 fmt='o', capsize=5)

    # Making ideal prediction line at y=x
    min_win = min(test_label.min(), linear_pred.min())
    max_win = max(test_label.max(), linear_pred.max())

    plt.plot([min_win, max_win], [min_win, max_win],
             color="orange", linewidth=3,
             label="Ideal")

    # adding textbox for R^2 and RMSE values
    textbox = (f"R² = {r2:.3f}\n" f"RMSE = {rmse:.3f}")

    plt.text(0.05, 0.90, textbox, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top', bbox=dict(
                 boxstyle='round',
                 facecolor='white',
                 edgecolor='black',
                 linewidth=1.5,
                 alpha=0.8))

    # Labelling etc.
    plt.xlabel("True Wins")
    plt.ylabel("Predicted Wins")
    plt.title(f"{type}: True vs. WAR-Predicted Wins (2019)")
    plt.legend()
    plt.show()


def make_world_series_table(test, test_pred):
    '''
    Creates DataFrame containing actual and
    predicted World Series outcomes.
    '''

    results = test[
        ["team_ID", "won_world_series"]
    ].copy()

    results["Predicted_WS_Win"] = test_pred

    print()
    print("World Series Results")
    print("--------------------")
    print(results)


def plot_confusion_matrix(matrix):
    '''
    plot_confusion_matrix plots a matrix comparing points
    predicted labels and true labels.
    '''

    plt.figure(figsize=(5, 5))

    plt.imshow(matrix)

    plt.colorbar()

    plt.xticks([0, 1], ["No WS", "Won WS"])

    plt.yticks([0, 1], ["No WS", "Won WS"])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center",
                     color="black", fontsize=12, fontweight="bold")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.title("World Series Confusion Matrix")

    plt.tight_layout()
    plt.show()


# Main function
def main():
    filename = "WAR_2010_2019.csv"

    # Loading data
    df = load_data(filename)

    # -------------
    # LINEAR MODEL
    # -------------

    # Training model
    linear_model, test, test_label, linear_pred = train_linear_model(df)

    # Evaluating model
    rmse, r2 = evaluate_model(test_label, linear_pred,
                              "Linear Regression Performance")

    # Results table
    linear_results = make_results_table(test, linear_pred)

    # Plotting
    plot_results(linear_results)
    plot_linear_results(test_label, linear_pred, "Linear Regression", r2, rmse)

    # -------------
    # DECISION TREE MODEL
    # -------------
    tree_model, test, test_label, tree_pred = train_tree_model(df)

    rmse, r2 = evaluate_model(test_label, tree_pred,
                              "Decision Tree Performance")

    tree_results = make_results_table(test, tree_pred)

    plot_results(tree_results)
    plot_linear_results(test_label, tree_pred, "Decision Tree Regression",
                        r2, rmse)

    # -------------
    # RANDOM FOREST MODEL
    # -------------
    forest_model, test, test_label, forest_pred = train_forest_model(df)

    rmse, r2 = evaluate_model(test_label, forest_pred,
                              "Random Forest Performance")

    forest_results = make_results_table(test, forest_pred)

    plot_results(forest_results)
    plot_linear_results(test_label, forest_pred, "Random Forest Regression",
                        r2, rmse)

    # ------------------------
    # WORLD SERIES MODEL
    # ------------------------
    _, test, test_label, test_pred = train_world_series(df)

    accuracy, matrix = evaluate_world_series_model(test_label, test_pred)

    make_world_series_table(test, test_pred)

    plot_confusion_matrix(matrix)


if __name__ == "__main__":
    main()
