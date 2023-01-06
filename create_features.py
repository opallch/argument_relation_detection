'''creates document embedding feature vectors for the data with Doc2Vec.'''
import os
import gensim
import pandas as pd
import spacy

CORPUS_DF_CSV_PATH = './corpus/merged_corpus.csv'
if not os.path.exists('./instances/'): os.mkdir('./instances/')
INSTANCES_OUTPUT_PATH = './instances/merged_instances.csv'
DOC2VEC_VECTOR_SIZE = 50

## (1) read the corpus (df) and tokenize the tweets
def read_corpus(corpus_df_csv_path):
    df = pd.read_csv(corpus_df_csv_path)
    nlp = spacy.load('de_core_news_sm') # install spaCy and `pip install $(spacy info de_core_news_sm --url)`
    corpus = list() # list of TaggedDocument, which is required for the Doc2Vec model
    labels = list()

    for idx, row in df.iterrows():
        if not isinstance(row.raw_text, float):
            raw_tokens = [token.text for token in nlp(row.raw_text)] # tokenize with spaCy
            corpus.append(gensim.models.doc2vec.TaggedDocument(raw_tokens, [idx]))
            labels.append(row.label)
    
    return corpus, labels

corpus, labels = read_corpus(CORPUS_DF_CSV_PATH) # we use the same data from the corpus for training and test
# print([doc[1] for doc in corpus])

## (2) train the Doc2Vec model
model = gensim.models.doc2vec.Doc2Vec(vector_size=DOC2VEC_VECTOR_SIZE, min_count=2, epochs=40)
model.build_vocab(corpus) # finding unique words, is it optional?
model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)

## (3) write vectors as df in a csv data
df = pd.DataFrame(columns=[i for i in range(0, DOC2VEC_VECTOR_SIZE)])
# generate feature vectors
for doc in corpus:
    vector = model.infer_vector(doc[0])
    df.loc[len(df)] = vector
# saves the labels and the original indices to the df
df['label'] = labels
df['original_index_in_corpus'] = [doc[1][0] for doc in corpus] # reindex() caused some problems so I add a new column instead

df.to_csv(INSTANCES_OUTPUT_PATH)