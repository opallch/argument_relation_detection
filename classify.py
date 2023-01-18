# https://lunalux.io/creating-your-first-machine-learning-classifier-with-sklearn/
import os

from sklearn import svm
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import LearningCurveDisplay
from sklearn.model_selection import train_test_split, cross_validate, LearningCurveDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt
# fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), sharey=True)
import pandas as pd
pd.set_option('display.max_colwidth', None)

CORPUS_PATH = './corpus/merged_corpus.csv'
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
    model = create_model(features_train, labels_train, model_name=model_name)
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


def create_model(features_train=[], labels_train=[], model_name='svm'):
    '''Returns a trained model upon the name given. SVM is used by default.
    If `features_train` is empty, an untrained model will be returned.
    '''
    if model_name == 'mlp':
        model = MLPClassifier(verbose=False, random_state=SEED,
                              max_iter=ITERATIONS)
    elif model_name == 'gaussian_nb':
        model = GaussianNB()
    else:
        model = svm.SVC()
    
    if len(features_train) > 0 and len(labels_train) > 0:
        model.fit(features_train, labels_train)
        print(f'A {model_name} is trained!')
    
    return model


if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH)

    ## (1) Prepare feature vectors
    # Get rid of columns we don't need for classification
    labels = df.label.values
    df_selected = df.drop(["Unnamed: 0", "original_index_in_corpus", "label"],
                          axis=1)
    feature_vecs = df_selected.to_numpy()  # or df_selected.values

    ## (2) K-cross validation for having a general overview on performance
    # for model_name in MODEL_NAMES:
    #     write_k_cross_validation_results(feature_vecs, labels, k=K_CROSS_VALID,
    #                                      model_name=model_name,
    #                                      result_root=RESULT_ROOT)

    ## (3) Learning Curve Plots
    # TODO: subplot, set legend for test and training set
    for i, model_name in enumerate(MODEL_NAMES):
        model = create_model([], [], model_name = model_name)
        LearningCurveDisplay.from_estimator(model, feature_vecs, labels, train_sizes=[0.25, 0.5, 0.75, 1.0], cv=5)
        plt.savefig(os.path.join(RESULT_ROOT, f'{model_name}_learning_curve.png'))
        # handles, label = ax[i].get_legend_handles_labels()
        # ax[i].legend(handles[:2], ["Training Score", "Test Score"])
        #ax[i].set_title(f"Learning Curve for {model.__class__.__name__}")
    

    ## (4) Error Analysis with a fixed train-test set
    # Set the proportion of train and test set
    features_train, features_test, labels_train, labels_test, orig_train, orig_test = train_test_split(
        feature_vecs, labels, df.original_index_in_corpus.values,
        test_size=TEST_SIZE, random_state=SEED)

    # Baseline results
    write_baseline_results(features_train, labels_train, features_test,
                           labels_test, result_root=RESULT_ROOT)

    # Real models & Error Analysis
    for model_name in MODEL_NAMES:
        model = create_model(features_train, labels_train, model_name = model_name)

        # Error Analysis
        # (A) write correct classification and missclassification respectively in txt files
        corpus_df = pd.read_csv(CORPUS_PATH)
        predictions_test = model.predict(features_test)
        
        with open(os.path.join(RESULT_ROOT, f'{model_name}_right_classification.csv'), 'a') as f_rightclass, open(os.path.join(RESULT_ROOT, f'{model_name}_misclassification.csv'), 'a') as f_misclass:
            for i in range(0, len(predictions_test)):
                orig_idx = orig_test[i]
                raw_text = corpus_df.loc[orig_idx, 'raw_text']
                if predictions_test[i] == labels_test[i]:
                    print(f'{orig_idx},{raw_text}', file=f_rightclass)
                else:
                    print(f'{orig_idx},{raw_text}', file=f_misclass)

        # (B) Scores: on test set v.s. training set
        predictions_train = model.predict(features_train)
        with open(os.path.join(RESULT_ROOT, f'{model_name}_classification_report.txt'), 'a') as f_out:
            print("========== Test Set ==========", file=f_out)
            print(classification_report(labels_test, predictions_test), file=f_out)
            print("========== Training Set ==========", file=f_out)
            print(classification_report(labels_train, predictions_train), file=f_out)
        
        # (C) Confusion Matrix, see what was often misclassified
        # TODO: merge plots using subplot()
        ConfusionMatrixDisplay.from_predictions(labels_test, predictions_test, normalize='true',
                                        xticks_rotation='vertical')
        plt.title(f'{model_name}')
        plt.savefig(os.path.join(RESULT_ROOT, f'{model_name}_conf_mat.png'))
        