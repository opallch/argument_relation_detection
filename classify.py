# https://lunalux.io/creating-your-first-machine-learning-classifier-with-sklearn/
import os

from sklearn import svm
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = './instances/merged_instances.csv'
RESULT_ROOT = './results/'
if not os.path.exists(RESULT_ROOT): os.mkdir(RESULT_ROOT)

MODEL_NAMES = ['svm', 'mlp', 'gaussian_nb']
TEST_SIZE = 0.2
K_CROSS_VALID = 4
ITERATIONS = 200
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


def write_baseline_results(features_train, labels_train, features_test,
                           labels_test, result_root=RESULT_ROOT):
    # Set up DummyClassifier as Baseline
    for strategy in ['most_frequent', 'prior', 'stratified', 'uniform']:
        dummy_clf = DummyClassifier(strategy=strategy)
        dummy_clf.fit(features_train, labels_train)
        score = dummy_clf.score(features_test, labels_test)

        with open(os.path.join(result_root, 'baseline.txt'), 'a') as f_out:
            print(
                f'Trivial baseline accuracy with strategy "{strategy}" is: {score}',
                file=f_out)


def write_k_cross_validation_results(features_train, labels_train, k,
                                     model_name, result_root):
    model = train_model(features_train, labels_train, model_name=model_name)
    cv_results = cross_validate(
        model,
        features_train,
        labels_train,
        cv=k,
        scoring=['f1_weighted', 'f1_micro', 'f1_macro']
    )

    with open(os.path.join(result_root, f'{k}-cross_validation.txt'),
              'a') as f_out:
        print(f'{model_name}:', file=f_out)
        for key, value in cv_results.items():
            print(value, '\t', key, file=f_out)


def train_model(features_train, labels_train, model_name='svm'):
    '''Returns a trained model upon the name given. SVM is used by default.'''
    if model_name == 'mlp':
        model = MLPClassifier(verbose=False, random_state=SEED,
                              max_iter=ITERATIONS)
    elif model_name == 'gaussian_nb':
        model = GaussianNB()
    else:
        model = svm.SVC()

    model.fit(features_train, labels_train)
    print(f'A {model_name} is trained!')
    return model


if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH)

    ## (1) Encode labels
    le = LabelEncoder()
    le.fit(["Comment", "Unrelated", "Support", "Refute"])
    labels = le.transform(df.label)  # apply encoding to labels

    ## (2) Prepare feature vectors
    # Get rid of columns we don't need for classification
    df_selected = df.drop(["Unnamed: 0", "original_index_in_corpus", "label"],
                          axis=1)
    feature_vecs = df_selected.to_numpy()  # or df_selected.values

    ## (3a) K-cross validation for having a general overview on performance
    for model_name in MODEL_NAMES:
        write_k_cross_validation_results(feature_vecs, labels, k=K_CROSS_VALID,
                                         model_name=model_name,
                                         result_root=RESULT_ROOT)

    ## (3b) Error Analysis with a fixed train-test set
    # Set the proportion of train and test set
    features_train, features_test, labels_train, labels_test, orig_train, orig_test = train_test_split(
        feature_vecs, labels, df.original_index_in_corpus.values,
        test_size=TEST_SIZE, random_state=SEED)
    # print_label_proportions(labels_train, labels_test)

    # Baseline results
    write_baseline_results(features_train, labels_train, features_test,
                           labels_test, result_root=RESULT_ROOT)

    # Real models & Error Analysis
    # for model_name in MODEL_NAMES:
    #     model = train_model(features_train, labels_train, model_name = model_name)
    #     # Scores
    #     with open(os.path.join(RESULT_ROOT, f'{model_name}.txt'), 'a') as f_out:
    #         print(f'Acc on train set: {model.score(features_train, labels_train)}', file=f_out)
    #         print(f'Acc on validation set: {model.score(features_test, labels_test)}', file=f_out)

    #     # TODO: plot ROC & AUC curves https://scikit-learn.org/0.15/auto_examples/plot_roc.html

    #     # Error Analysis
    #     # TODO: write correct classification and missclassification respectively in txt files
    #     binary_balanc_predictions = model.predict(features_test)
