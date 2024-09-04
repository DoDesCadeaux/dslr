import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sys import argv

if __name__ == '__main__':
    dataset_folder = 'datasets/'

    try:
        path = dataset_folder + argv[1]
        data = pd.read_csv(path)
    except (IndexError, FileNotFoundError) as e:
        print(e)
        print('Usage: python3 scatter_plot.py <dataset.csv> <course_1> <course_2>')
        exit(1)

    try:
        course_1 = argv[2]
        course_2 = argv[3]
    except IndexError as e:
        print(e)
        print('Usage: python3 scatter_plot.py <dataset.csv> <course_1> <course_2>')
        exit(2)

    try:
        plt.figure(figsize=(15, 6))
        sns.scatterplot(data=data, x=f'{course_1}', y=f'{course_2}', hue='Hogwarts House')
        plt.show()
    except (ValueError, KeyboardInterrupt) as e:
        print(e)
        exit(3)
