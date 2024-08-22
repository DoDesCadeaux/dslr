from sys import argv
from math import sqrt
import numpy as np
import pandas as pd

def count(data):
    columns_lengths = []
    for i in range(0, data.shape[1]):
        length = float(len(data.iloc[:, i].dropna()))
        columns_lengths.append(round(length, 6))

    return columns_lengths

def mean(data):
    columns_mean = []
    for i in range(0, data.shape[1]):
        mean = float(np.sum(data.iloc[:, i]) / len(data.iloc[:, i].dropna()))
        columns_mean.append(round(mean, 6))

    return columns_mean

def std(data):
    columns_std = []
    for i in range(0, data.shape[1]):
        col = data.iloc[:, i].dropna()
        mean = float(np.sum(data.iloc[:, i]) / len(data.iloc[:, i].dropna()))
        variance = np.sum((col - mean) ** 2) / (len(col) - 1)
        std = sqrt(variance)
        columns_std.append(round(std, 6))

    return columns_std


def calculate_statistics(data):
    stats = {
        'count': count(data),
        'mean': mean(data),
        'std': std(data)
    }

    df = pd.DataFrame(stats)
    real_df = pd.DataFrame(df.T)
    real_df.rename(columns={i: name for i, name in enumerate(data.columns)}, inplace=True)
    return real_df

if __name__ == "__main__":
    path = 'datasets/dataset_train.csv'
    data = pd.read_csv(path)

    X_train = data.select_dtypes(include='number')

    stats_df = calculate_statistics(X_train)
    print(stats_df)
    print("\n")
    print(X_train.describe())