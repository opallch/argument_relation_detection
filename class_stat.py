'''Scipt for checking the proportion of each class of the UKP Corpus.
usage:
`python3 class_stat.py path/to/annotated/data/csv/file`
'''
import argparse
import os
import numpy as np
import pandas as pd


def show_class_distribution(df_csv_path):
    n_support = 0
    n_refute = 0
    n_comment = 0
    n_unrelated = 0

    df = pd.read_csv(df_csv_path)
    for idx, row in df.iterrows():
        if isinstance(row.raw_text, str): # if the raw text of the tweet is available
            if row.label == 'Support':
                n_support += 1
            elif row.label == 'Refute':
                n_refute += 1
            elif row.label == 'Comment':
                n_comment += 1
            elif row.label == 'Unrelated':
                n_unrelated += 1

    print(f'file: {os.path.basename(df_csv_path)}')
    print(f'support: {n_support}\nrefute: {n_refute}\ncomment: {n_comment}\nunrelated: {n_unrelated}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('df_csv_path') # must be csv
    args = parser.parse_args()
    show_class_distribution(args.df_csv_path)