import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.express as px
from copy import deepcopy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression



def load_data():
    df_train = pd.read_csv('./data/cleaned_listings.csv', index_col='id')
    filter_col_prop = [col for col in df_train if col.startswith('property')]
    filter_col_amenities = [col for col in df_train if col.startswith('amenities')]
    filter_col_neighbourhood = [col for col in df_train if col.startswith('neighbourhood')]
    one_hot = len(filter_col_prop) + len(filter_col_amenities) + len(filter_col_neighbourhood)
    one_hot_cols = [filter_col_neighbourhood,filter_col_prop,filter_col_amenities]
    return df_train, len(df_train.columns) - one_hot + 3, len(df_train), one_hot_cols

def linear_regression(df, variables, rows, one_hot_cols):

    df = df[:rows]
    # define response variable
    y = df["price"]

    # define predictor variables
    x = list(df.columns)
    x.remove("price")
    if variables >= 24:
        x += one_hot_cols[0]
        if variables >= 25:
            x += one_hot_cols[1]
            if variables >= 26:
                x += one_hot_cols[2]
    x = df[x[:variables]]

    # start timer
    start = time.time()

    # fit linear regression model
    model = sm.OLS(y, x).fit()

    return time.time() - start

def plot_data(final_df):
    # plot code here
    final_df.plot()
    plt.show()

if __name__ == "__main__":
    results = {}
    train_df, length_variables, length_rows, one_hot_cols = load_data()
    for variables in range(int(length_variables/2),length_variables, 3):
        rows_duration = {}
        for rows in range(int(length_rows/100), length_rows, int(length_rows/100)):
            duration = linear_regression(train_df, variables, rows, one_hot_cols)
            rows_duration[rows] = duration
        results[variables] = rows_duration

    data_frame_results = pd.DataFrame(results)
    data_frame_results.to_csv("results_linear_regression.csv")
    plot_data(data_frame_results)



