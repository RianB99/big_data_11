import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression


def load_data():
    df_train = pd.read_csv('./data/aug_train.csv', index_col='enrollee_id')
    return df_train, len(df_train.columns), len(df_train)

def classify_data(df, variables, rows):
    start = time.time()

    # classification code here
    return time.time() - start

def plot_data(final_df):
    # plot code here
    final_df.plot()
    plt.show()

if __name__ == "__main__":
    results = {}
    #train_df, length_variables, length_rows = load_data()
    train_df, length_variables, length_rows = 0, 10, 10000
    for variables in range(int(length_variables/2),length_variables):
        rows_duration = {}
        for rows in range(int(length_rows/100), length_rows, int(length_rows/100)):
            duration = classify_data(train_df, variables, rows)
            rows_duration[rows] = duration
        results[variables] = rows_duration

    data_frame_results = pd.DataFrame(results)
    plot_data(data_frame_results)

