from sys import argv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score


def min_max_scaler(X):
    X_scaled = (X - X.min()) / (X.max() - X.min())
    return X_scaled


def encode_label(y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    onehot_encoder = OneHotEncoder(sparse_output=False)
    y_onehot = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))
    return y_onehot


def sigmoid(theta, X):
    return 1 / (1 + np.exp(-np.dot(X, theta.T)))


def log_loss(A, y):
    m = len(y)
    return -1 / m * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))


def gradients(A, X, y):
    m = len(y)
    dW = 1 / m * np.dot(A.T - y.T, X)
    return dW


def artificial_neuron(X, y, learning_rate=0.01, n_iter=30000):
    m, n = X.shape
    n_classes = y.shape[1]

    theta = np.random.rand(n_classes, n)
    loss = []

    for i in range(n_iter):
        A = sigmoid(theta, X)
        loss.append(log_loss(A, y))
        dW = gradients(A, X, y)

        theta -= learning_rate * dW

    plt.plot(loss)
    plt.show()
    return theta


def predict(X, theta):
    probabilities = sigmoid(theta, X)
    return np.argmax(probabilities, axis=1)


if __name__ == '__main__':
    folder = 'datasets/'
    try:
        path = folder + argv[1]
        data = pd.read_csv(path)
    except (IndexError, FileNotFoundError) as e:
        print(e)
        exit(1)

    data = data.dropna()
    data_numeric = data.select_dtypes('number')

    X = data_numeric.iloc[:, 1:]
    y = data['Hogwarts House']

    X_scaled = min_max_scaler(X)

    y_onehot = encode_label(y)

    theta_final = artificial_neuron(X_scaled, y_onehot)

    predictions = predict(X_scaled, theta_final)
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    predicted_labels = label_encoder.inverse_transform(predictions)

    accuracy = accuracy_score(y, predicted_labels)
    print(f"Accruacy score: {accuracy}")
