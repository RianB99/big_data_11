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
    df_train.pop("len_amenities")
    return df_train, len(df_train)

def regressor(df, variables, rows, model_type):

    df = df[:rows]

    # define response variable
    y = df["price"]

    # define predictor variables
    x = list(df.columns)

    x.remove("price")

    x = x[:variables]

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

def plot_data(final_df):
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
                  log_x = False,
                  log_y = True
                 )
    fig.show()
    fig = px.line(final_df,
                  x = "rows",
                  y = "rmse",
                  color = "variables",
                  line_group = "variables",
                  hover_name = "variables",
                  labels = {"rmse" : "RMSE of model",
                            "duration" : "Runtime in Seconds",
                            "variables" : "Number of Variables"
                           },
                  log_x = False,
                  log_y = False
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
    train_df, length_rows = load_data()
    variable_list = [5, 10, 45, 278]

    for variables in variable_list:
        for rows in range(int(length_rows/100), length_rows, int(length_rows/100)):
            duration, score = regressor(train_df, variables, rows, model_type)
            results.append([variables, rows, duration, score])

    data_frame_results = pd.DataFrame.from_records(results)
    data_frame_results.columns = ["variables", "rows", "duration", "rmse"]
    data_frame_results.to_csv("results-"+str(model_name)+".csv")
    
    # multiple plots to compare regular vs logscale axes
    plot_data(data_frame_results)