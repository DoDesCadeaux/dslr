import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


path = 'datasets/dataset_train.csv'

data = pd.read_csv(path)
print(data.columns)

courses = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination',
           'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions',
           'Care of Magical Creatures', 'Charms', 'Flying']

try:
    plt.figure(figsize=(20, 15))

    for i, course in enumerate(courses, 1):
        plt.subplot(4, 4, i)
        sns.histplot(data=data, x=course, hue='Hogwarts House', multiple='stack')

    plt.show()
except KeyboardInterrupt:
    exit(0)
