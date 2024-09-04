import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sys import argv
from tqdm import tqdm

if __name__ == '__main__':
    dataset_folder = 'datasets/'

    try:
        path = dataset_folder + argv[1]
        data = pd.read_csv(path)
    except (IndexError, FileNotFoundError) as e:
        print(e)
        exit(1)

    courses = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination',
               'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions',
               'Care of Magical Creatures', 'Charms', 'Flying']

    try:
        plt.figure(figsize=(20, 15))

        with tqdm(total=100) as pbar:
            for i, course in enumerate(courses, 1):
                plt.subplot(4, 4, i)
                sns.histplot(data=data, x=course, hue='Hogwarts House', multiple='stack')
                pbar.set_description("Processing Plots")
                pbar.update(100/13)
        plt.show()
    except KeyboardInterrupt:
        exit(0)

