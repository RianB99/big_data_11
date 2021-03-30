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

def regressor(x,y, variables, rows, model_type):

    X_total = x[:rows]
    Y_total = y[:rows]

    X_train, X_test, y_train, y_test = train_test_split(X_total, Y_total, test_size= 0.2, random_state= 42)

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

def correlation(train_df):
    corr_df = train_df
    correlation = corr_df.corrwith(corr_df["price"])
    corr_df = pd.DataFrame(correlation)
    corr_df.columns = ["Correlation_with_price"]
    ordered = corr_df.sort_values(by="Correlation_with_price", ascending=False)
    order = list(ordered.index)
    ordered_df = train_df[order]
    ordered.to_csv("correlation.csv")
    return ordered_df


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
                  title="Duration in seconds for Random Forest Regressor predicting price with aggregating database",
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
                  title = "RMSE scores for Random Forest Regressor predicting price with aggregating database",
                  log_x = False,
                  log_y = True,
                 )
    fig.show()

if __name__ == "__main__":
    real_start = time.time()
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

    train_df = correlation(train_df)

    # define response variable
    y = train_df["price"]
    for variables in range(25,278,25):

        # define predictor variables
        x = list(train_df.columns)

        x.remove("price")

        x = train_df[x[:variables]]

        for rows in range(int(length_rows/100), length_rows, int(length_rows/100)):
            duration, score = regressor(x,y, variables, rows, model_type)
            results.append([variables, rows, duration, score])

    data_frame_results = pd.DataFrame.from_records(results)
    data_frame_results.columns = ["variables", "rows", "duration", "rmse"]
    data_frame_results.to_csv("results-"+str(model_name)+".csv")
    
    # multiple plots to compare regular vs logscale axes
    plot_data(data_frame_results)

    print("Duration of the the project:", round(time.time() - real_start,2), "seconds")