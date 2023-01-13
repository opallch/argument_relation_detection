# https://lunalux.io/creating-your-first-machine-learning-classifier-with-sklearn/
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

CSV_PATH = './instances/merged_instances.csv'

SEED = 44


def print_label_proportions(y_train, y_val):
    """Helper function to print class proportions in the dataset."""
    train_proportions = []
    val_proportions = []
    id2name = {i: name for i, name in enumerate(set(y_train.tolist()))}
    for i in range(len(id2name)):
        train_proportions.append(y_train.tolist().count(id2name[i]))
        val_proportions.append(y_val.tolist().count(id2name[i]))
    for class_id, name in id2name.items():
        train_perc = 100 * train_proportions[class_id] / sum(train_proportions)
        val_perc = 100 * val_proportions[class_id] / sum(val_proportions)

        print(
            f'The training set has {train_perc:.2f}% of datapoints with class {name}')
        print(
            f'The validation set has {val_perc:.2f}% of datapoints with class {name}')


# Read merged_instances
df = pd.read_csv(CSV_PATH)

# Set labels
labels = np.asarray(df.label)
print(labels)

# Encode labels
le = LabelEncoder()
le.fit(labels)
# apply encoding to labels
labels = le.transform(labels)
print(labels)

# print(le.inverse_transform(labels))

# Get rid of columns we don't need for classification
df_selected = df.drop(["Unnamed: 0", "original_index_in_corpus", "label"],
                      axis=1)

feature_vecs = df_selected.to_numpy() # or df_selected.values

# Set the proportion of train and test set
features_train, features_test, labels_train, labels_test = train_test_split(
 feature_vecs, labels, test_size=0.20, random_state=SEED)
print_label_proportions(labels_train, labels_test)

# Set up DummyClassifier as Baseline
for strategy in ['most_frequent', 'prior', 'stratified', 'uniform']:
    dummy_clf = DummyClassifier(strategy=strategy)
    dummy_clf.fit(features_train, labels_train)
    score = dummy_clf.score(features_test, labels_test)

    print(f'Trivial baseline accuracy with strategy {strategy} is: {score}')

# print(features_train)
# print(labels_train)

# Set up Gaussian Classifier and train the model
model = GaussianNB()
pred = model.fit(features_train, labels_train)

print(model.classes_)
print(model.class_count_)
print(model.n_features_in_)

# binary_balanc_predictions = model.predict(features_test)
# binary_balanc_probs = model.predict_proba(labels_test)

print(f'Acc on train set: {model.score(features_train, labels_train)}')
print(f'Acc on validation set: {model.score(features_test, labels_test)}')