import pandas as pd
import numpy as np
import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

def load_data():
    df = pd.read_csv('./data/aug_train.csv', index_col='enrollee_id')
    print(df.head(5))

    # print niformation about nans
    print("shape of dataframe:", df.shape, "\n")
    print("number of nans per column:")
    print(df.isna().sum(), "\n")
    print("number of rows with amount of nans")
    print(df.isna().sum(axis=1).value_counts())

if __name__ == "__main__":
    load_data()
