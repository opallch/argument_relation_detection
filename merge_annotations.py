'''merge the expert annotated data and the student annotated data.'''
import os
import pandas as pd

CORPUS_DF_CSV_ROOT = './corpus/'

dfs_to_be_merged = list()
for root, dirs, files in os.walk(CORPUS_DF_CSV_ROOT):
    for file in files:
        if file.endswith('_column.csv'):
            dfs_to_be_merged.append(
                pd.read_csv(os.path.join(root, file))
                )
corpus_df = pd.concat(dfs_to_be_merged, ignore_index=True)
corpus_df.drop(columns=['Unnamed: 0'], inplace=True)
corpus_df.to_csv('./corpus/merged_corpus.csv')