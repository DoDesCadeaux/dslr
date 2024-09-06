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
        col = data.iloc[:, i].dropna()
        if len(col) == 0:
            columns_mean.append(np.nan)
        else:
            mean = float(np.sum(col) / len(col))
            columns_mean.append(round(mean, 6))

    return columns_mean


def std(data):
    columns_std = []

    for i in range(0, data.shape[1]):
        col = data.iloc[:, i].dropna()
        if len(col) == 0:
            columns_std.append(np.nan)
        else:
            mean = float(np.sum(col) / len(col))
            variance = np.sum((col - mean) ** 2) / (len(col) - 1)
            std = sqrt(variance)
            columns_std.append(round(std, 6))

    return columns_std


def minimum(data):
    columns_min = []

    for i in range(0, data.shape[1]):
        smallest = np.inf

        for j in data.iloc[:, i]:
            if j < smallest:
                smallest = j
        columns_min.append(smallest)

    return columns_min


def maximum(data):
    columns_max = []

    for i in range(0, data.shape[1]):
        biggest = -np.inf

        for j in data.iloc[:, i]:
            if j > biggest:
                biggest = j
        columns_max.append(biggest)

    return columns_max


def quantile(data, q):
    columns_first_quart = []

    for i in range(0, data.shape[1]):
        col = data.iloc[:, i].dropna().sort_values()
        n = len(col)

        index = (n - 1) * q
        j = int(index)
        g = index - j

        if n == 0:
            columns_first_quart.append(np.nan)
            continue

        if j < n - 1:
            q_value = (1 - g) * col.iloc[j] + g * col.iloc[j + 1]
        else:
            q_value = col.iloc[j]

        columns_first_quart.append(round(q_value, 6))

    return columns_first_quart


def calculate_statistics(data):
    stats = {
        'count': count(data),
        'mean': mean(data),
        'std': std(data),
        'min': minimum(data),
        '25%': quantile(data, 0.25),
        '50%': quantile(data, 0.50),
        '75%': quantile(data, 0.75),
        'max': maximum(data)
    }

    df = pd.DataFrame(stats)
    real_df = df.T
    real_df.rename(columns={i: name for i, name in enumerate(data.columns)}, inplace=True)
    return real_df


if __name__ == "__main__":
    folder = 'datasets/'
    try:
        print(argv[0])
        path = folder + argv[1]
        print(path)
        data = pd.read_csv(path)

    except (IndexError, FileNotFoundError) as e:
        print(e)
        exit(1)

    X_train = data.select_dtypes(include='number').dropna(axis=1, how='all')

    stats_df = calculate_statistics(X_train)
    print(stats_df)
    print(X_train.describe())
