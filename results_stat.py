'''Scipt for checking the proportion of translated instances in (
mis)classification files.
usage:
`python3 results_stat.py path/to/model/right/or/misclassification`
'''
import argparse
import os
import pandas as pd


def show_class_distribution(df_csv_path):
    n_translated = 0

    df = pd.read_csv(df_csv_path)
    for idx, row in df.iterrows():
        if pd.isna(row.translated_from) is False:
            n_translated += 1

    total = len(df)-1
    percentage = round((n_translated/total*100),2)

    print(f'file: {os.path.basename(df_csv_path)}')
    print(f'translated: {n_translated}')
    print(f'summary: {n_translated} of {total}, {percentage}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('df_csv_path') # must be csv
    args = parser.parse_args()
    show_class_distribution(args.df_csv_path)