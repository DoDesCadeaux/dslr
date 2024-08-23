import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sys import argv
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    dataset_folder = 'datasets/'

    try:
        path = dataset_folder + argv[1]
        data = pd.read_csv(path)
        sample_data, _ = train_test_split(data, test_size=0.9, stratify=data['Hogwarts House'], random_state=42)
    except (IndexError, FileNotFoundError) as e:
        print(e)
        exit(1)

    try:
        sns.pairplot(data=sample_data, hue='Hogwarts House', corner=True)
        plt.show()
    except KeyboardInterrupt as e:
        print(e)
        exit(2)

