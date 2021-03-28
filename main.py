import sys
import time
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


def load_data():
    df_train = pd.read_csv('./data/cleaned_listings.csv', index_col='id')
    filter_col_prop = [col for col in df_train if col.startswith('property')]
    filter_col_amenities = [col for col in df_train if col.startswith('amenities')]
    filter_col_neighbourhood = [col for col in df_train if col.startswith('neighbourhood')]
    one_hot_cols = [filter_col_neighbourhood, filter_col_prop, filter_col_amenities]
    return df_train, len(df_train), one_hot_cols

def regressor(df, variables, rows, one_hot_cols, model_type):

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

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 42)

    # linear regressor or random forest regressor
    if model_type == 1:
        model = RandomForestRegressor()
    else:
        model = LinearRegression()

    # start timer
    start = time.time()

    # fit the model
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    mse = metrics.mean_squared_error(np.array(y_test), np.array(pred), squared=False)

    return time.time() - start, mse

def plot_data(final_df, log_x, log_y):
    fig = px.line(final_df,
                  x = "rows", 
                  y = "duration", 
                  color = "variables",
                  line_group = "variables", 
                  hover_name = "variables",
                  labels = {"rows" : "Number of Rows", 
                            "duration" : "Runtime in Seconds", 
                            "variables" : "Number of Variables"
                           },
                  log_x = log_x,
                  log_y = log_y
                 )
    fig.show()

if __name__ == "__main__":

    try:
        sys.argv[1]
    except IndexError:
        model_type = 0
    else:
        model_type = int(sys.argv[1])

    model_name = ["Linear-Regression", "Random-Forest-regression"][model_type]
    print(model_name, "is selected")

    results = []
    train_df, length_rows, one_hot_cols = load_data()
    variable_list = [3, 6, 9, 11, 46, 103, 279]

    for variables in variable_list:
        for rows in range(int(length_rows/100), length_rows, int(length_rows/100)):
            duration, score = regressor(train_df, variables, rows, one_hot_cols, model_type)
            results.append([variables, rows, duration, score])

    data_frame_results = pd.DataFrame.from_records(results)
    data_frame_results.columns = ["variables", "rows", "duration", "rmse"]
    data_frame_results.to_csv("results-"+str(model_name)+".csv")
    
    # multiple plots to compare regular vs logscale axes
    plot_data(data_frame_results, False, False)
    plot_data(data_frame_results, True, False)
    plot_data(data_frame_results, False, True)
    plot_data(data_frame_results, True, True)
