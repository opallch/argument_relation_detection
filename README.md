# Argument Relation Detection 
## Data Preprocessing
The preprocessed data can be found in `annotated_data/`. Files with suffix `_clean` indicates that the raw text of the tweets are already cleaned (e.g. URLs are removed). The following are the scripts to generate the data in `annotated_data/`:
- `retrieve_tweets.py`: Retreives tweets from the tweet-ids given in the UKP-Corpus. (Output: DataFrame in csv) 
- `clean_tweets.py`: Cleans the raw text of tweets in the output from `retrieve_tweets.py`.

The tweets are cleaned in the following ways:
1. URLs are removed 
2. The #-sign of the hashtags are removed, while the text remains
3. Tagged usernames: 
    - removed if it is at the beginning of comment or in the case of "RT @username"
    - otherwise the usernames are replaced by the tag `<USERNAME>`
4. Non-german tweets are translated to german by [some tool]