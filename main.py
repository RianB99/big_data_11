import pandas as pd
import numpy as np
import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

def load_data():
    df = pd.read_csv('./data/aug_train.csv', index_col='enrollee_id')
    length_variables = len(df.columns)
    length_rows = len(df)
    return df, length_variables, length_rows

def classify_data(df, variables, rows):
    start = time.time()

    return time.time() - start


if __name__ == "__main__":
    results = {}
    df, length_variables, length_rows = load_data()
    for variables in range(int(length_variables/2),length_variables):
        rows_duration = {}
        for rows in range(int(length_rows/100), length_rows, int(length_rows/100)):
            duration = classify_data(df, variables, rows)
            rows_duration[rows] = duration
        results[variables] = rows_duration

    print(results)

