from sys import argv
import pandas as pd

if __name__ == '__main__':
    folder = 'datasets/'
    try:
        print(argv[0])
        path = folder + argv[1]
        print(path)
        data = pd.read_csv(path)

    except (IndexError, FileNotFoundError) as e:
        print(e)
        exit(1)
