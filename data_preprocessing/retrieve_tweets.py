'''Scipt for retrieving tweets from the tweet-ids given in the UKP Corpus.'''
import os
import pandas as pd
import tweepy
import time
import sys

UKP_ROOT = '../ukp_covid_corpus/' # directory of the ukp covid tweet corpus
OUTPUT_ROOT = '../corpus/'
BEARER_TOKEN = '' # replace with your own token
CLIENT = tweepy.Client(bearer_token=BEARER_TOKEN)


def chunkify(my_list, n):
    """chunkify a list into n-sized sublists."""
    for i in range(0, len(my_list), n):
        yield my_list[i:i + n]


def retrieve_tweet(client, annotation_from='expert'):
    
    if annotation_from == 'expert':
        df = pd.read_csv(os.path.join(UKP_ROOT, 'expertdata.csv'))
    elif annotation_from == 'student':
        df = pd.read_csv(os.path.join(UKP_ROOT, 'studentdata.csv'))
        df = df.drop(columns=['recommendation', 'annotation_id', 'user_id', 'group_id'])

    all_tweets_in_text = list()
    n_not_found = 0
    n_retrieved_tweets = 0
    try:
        i = 0
        for chunk in chunkify(list(df.id_str.values), 300): # maybe tqdm instead
            for id in chunk:
                tweet = client.get_tweet(id)
                if tweet.data is None:
                    n_not_found += 1
                    all_tweets_in_text.append('')
                else:
                    all_tweets_in_text.append(tweet.data.text)
                
                n_retrieved_tweets += 1

            i += 1
            print(f'chunk {i} is done')
            if i < 10:
                time.sleep(930) # sleep for 15 minutes (+30 sec buffer), since max. 300 tweets can be retrieved within a 15-minute interval

    except Exception as e:
        print(e)
        sys.exit(1)
    
    finally:
        print(f'retrieved tweets: {n_retrieved_tweets}')
        print(f'{n_not_found} tweets not found.')

    df['raw_text'] = all_tweets_in_text
    df.to_csv(os.path.join(OUTPUT_ROOT, f'{annotation_from}_annotation.csv'))

if __name__ == '__main__':
    retrieve_tweet(client=CLIENT, annotation_from='student')