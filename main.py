import sys
import time
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def load_data():
    df_train = pd.read_csv('./data/cleaned_listings.csv', index_col='id')
    filter_col_prop = [col for col in df_train if col.startswith('property')]
    filter_col_amenities = [col for col in df_train if col.startswith('amenities')]
    filter_col_neighbourhood = [col for col in df_train if col.startswith('neighbourhood')]
    one_hot_cols = [filter_col_neighbourhood, filter_col_prop, filter_col_amenities]
    return df_train, len(df_train), one_hot_cols

def linear_regression(df, variables, rows, one_hot_cols, model_type):

    df = df[:rows]

    # define response variable
    y = df["price"]

    # define predictor variables
    x = list(df.columns)

    x.remove("price")

    x = x[:variables]

    if variables >= 25:
        x += one_hot_cols[0]
        if variables >= 49:
            x += one_hot_cols[1]
            if variables >= 99:
                x += one_hot_cols[2]
    x = df[x]

    # linear regressor or decision tree regressor
    if model_type == 1:
        model = RandomForestRegressor()
    else:
        model = LinearRegression()

    # start timer
    start = time.time()

    # fit the model
    model.fit(x,y)
    return time.time() - start

def plot_data(final_df):
    # plot code here
    fig = px.line(final_df, x="rows", y="duration", color="variables",
                  line_group="variables", hover_name="variables")
    fig.show()

if __name__ == "__main__":

    try:
        sys.argv[1]
    except IndexError:
        print("No model type is given, Linear regression is selected")
        model_type = 0
    else:
        model_type = int(sys.argv[1])
        print(["linear regression", "Random Forest regression"][model_type], "is selected")

    results = []
    train_df, length_rows, one_hot_cols = load_data()
    variable_list = [3, 6, 9, 11, 46, 103, 279]

    for variables in variable_list:
        for rows in range(int(length_rows/100), length_rows, int(length_rows/100)):
            duration = linear_regression(train_df, variables, rows, one_hot_cols, model_type)
            results.append([variables, rows, duration])


    data_frame_results = pd.DataFrame.from_records(results)
    data_frame_results.columns = ["variables", "rows", "duration"]
    data_frame_results.to_csv("results.csv")
    plot_data(data_frame_results)



