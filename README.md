# Argument Relation Detection
This project conducts a Supervised Classification on a corpus about the 
government measures during the Covid-19 Pandemic. It was done within the 
scope of an Argument Mining seminar in Wintersemester 2022/23 at the University of Potsdam

Link to the original corpus:   
https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2780

## Data Preprocessing
The preprocessed data can be found in `corpus/`, where **`merged_instances.csv`** is 
(and should be) used further for feature engineering, since it contains both expert and 
student annotated data, which are already preprocessed. 

Files in `./corpus/tmp/` are temporary corpus produced at different data 
preprocessing stages, e.g. Files with suffix `_clean_translated` indicates 
that the raw text of the tweets are already cleaned (e.g. URLs are removed) as follows:
    1. URLs are removed 
    2. The #-sign of the hashtags are removed, while the text remains
    3. Tagged usernames: 
        - removed if it is at the beginning of comment or in the case of "RT @username"
        - otherwise the usernames are replaced by the tag `<USERNAME>`
    4. Non-german tweets are translated to German by DeepL (where available) or 
    GoogleTranslate (for Arabic, Catalan, Korean, Persian, Urdu, Thai and 
    Telugu)

The scripts used for preprocess the data can be found in `data_preprocessing/`:
- `retrieve_tweets.py`: Retreives tweets from the tweet-ids given in the UKP-Corpus. 
(Output: DataFrame in csv) 
- `clean_tweets.py`: Cleans the raw text of tweets in the output from `retrieve_tweets.py`.
- `merge_annotations.py`: Merge student and expert annotation dataframes.

## Feature Engineering - Doc2Vec
We used `Doc2Vec` from the `gensim` package for document embedding in `create_features.py`. 
The instances are stored in `instances/merged_instances.csv` in the following format:
|`index`|n columns for document embedding vector|`label`|`original_index_in_corpus`|
- The size of the document embedding vector, n, can be defined in `instances/merged_instances.csv`
(50 was our choice).
- `original_index_in_corpus` is saved for retrieving the raw text from the corpus.

## Classification
In `classify.py` three models are set up and trained on the 
corpus. The parameters of test size, cross validation fold size and 
iterations can be modified. Furthermore, a DummyClassifier using four 
different strategies is set up. The models LearningCurves and 
the ConfusionMatrixes are plotted. All the results are stored in the 
`results` folder.
