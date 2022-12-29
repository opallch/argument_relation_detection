'''Clean the raw text of the tweets in the following ways:
url: removed 
hashtag: #-sign removed (text remains)
tagged username: 
    - removed if it is at the beginning of comment or in the case of "RT @username"
    - otherwise the usernames are replaced by the tag <USERNAME>

usage:
`python3 clean_tweets.py path/to/annotated/data/csv/file`
The new dataframe will be saved as csv file in the directory where the input dataframe reside.
`_clean` will be added as suffix to the new csv filename.
'''
import argparse
import os
import re
import pandas as pd


URL_REGEX = 'http\S+'
USER_REGEX_BEGIN = '^(RT\s)*@\S+'
USER_REGEX = '@\S+'
HASHTAG_REGEX = '#'


def clean(input_string):
    if isinstance(input_string, str): # if tweet is retrieved
        result = re.sub(URL_REGEX, '', input_string)
        result = re.sub(USER_REGEX_BEGIN, '', result)
        result = re.sub(USER_REGEX, '<USERNAME>', result)
        result = re.sub(HASHTAG_REGEX, '', result)
        result = re.sub('\s\s', ' ', result)
    else:
        result = ''
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('df_csv_path') # must be csv
    args = parser.parse_args()
    
    df = pd.read_csv(args.df_csv_path)
     
    for idx, row in df.iterrows():
        df.at[idx,'raw_text'] = clean(row.raw_text) # clean the raw_text by regex
   
    # write df to a new csv file
    dir_head_tail = os.path.split(args.df_csv_path)
    output_filename = re.sub('.csv', '_clean.csv', dir_head_tail[1])
    df.to_csv(os.path.join(dir_head_tail[0], output_filename))
    
    ## ONLY FOR TRIAL
    # example1 = 'Corona-Krise: „Das geht zu weit“ – Linke und Grüne kritisieren Reiseverbote https://t.co/22jIVuXBKE via @welt Ausgerechnet!!! Ganz ehrlich??? Ihr seid nicht mehr dicht im Kopp... Bei Stand von wievielen Toten geht euch das nicht mehr zu weit??? Flachzangen!'
    # example2 = '@nulleberg Wenn man sagen könnte, dass @jensspahn rechtzeitig wichtige Maßnahmen gegen das #coronavirus in die Wege leiten wollte, die dann an den Ländern gescheitert wären, könnte ich das nachvollziehen. So aber scheint es um ganz andere Dinge zu gehen 🤔'
    # example3 = 'RT @mattari_illust: #ドラゴンボール https://t.co/O7GBk0wNX9'
    # print(clean(example1))
    # print(clean(example2))
    # print(clean(example3))
