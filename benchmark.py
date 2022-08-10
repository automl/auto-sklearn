"""
# print(list(vars(cs).keys()))
# print(list(dict(cs._hyperparameters).keys()))
# print([hp for hp in dict(cs._hyperparameters).keys() if "text" in hp])
# print(cs._hyperparameters['data_preprocessor:feature_type:text_transformer:text_encoding:bag_of_word_encoding_distinct:ngram_upper_bound'])
# print(cs._hyperparameters['data_preprocessor:feature_type:text_transformer:text_encoding:bag_of_word_encoding_distinct:ngram_upper_bound'].default_value)
"""
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

import autosklearn.classification

############################################################################
# Data Loading
# ============
cats = ["comp.sys.ibm.pc.hardware", "rec.sport.baseball"]
X_train, y_train = fetch_20newsgroups(
    subset="train",  # select train set
    shuffle=True,  # shuffle the data set for unbiased validation results
    random_state=42,  # set a random seed for reproducibility
    categories=cats,  # select only 2 out of 20 labels
    return_X_y=True,  # 20NG dataset consists of 2 columns X: the text data, y: the label
)  # load this two columns separately as numpy array

X_test, y_test = fetch_20newsgroups(
    subset="test",  # select test set for unbiased evaluation
    categories=cats,  # select only 2 out of 20 labels
    return_X_y=True,  # 20NG dataset consists of 2 columns X: the text data, y: the label
)  # load this two columns separately as numpy array

############################################################################
# Creating a pandas dataframe
# ===========================
# Both categorical and text features are often strings. Python Pandas stores python stings
# in the generic `object` type. Please ensure that the correct
# `dtype <https://pandas.pydata.org/docs/user_guide/basics.html#dtypes>`_ is applied to the correct
# column.

# create a pandas dataframe for training labeling the "Text" column as sting
X_train = pd.DataFrame({"Text": pd.Series(X_train, dtype="string")})

# create a pandas dataframe for testing labeling the "Text" column as sting
X_test = pd.DataFrame({"Text": pd.Series(X_test, dtype="string")})

############################################################################
# Build and fit a classifier
# ==========================

# create an autosklearn Classifier or Regressor depending on your task at hand.
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60,
    per_run_time_limit=30,
)

cs = automl.get_configuration_space(X_train, y_train)
text_hps = [hp for hp in dict(cs._hyperparameters).keys() if "text_encoding" in hp]

hps = {
    'data_preprocessor:feature_type:text_transformer:text_encoding:tfidf_encoding:binary': [True, False],
    'data_preprocessor:feature_type:text_transformer:text_encoding:tfidf_encoding:ngram_range': ["1,1", "1,2", "1,3",
                                                                                                 "2,2", "2,3", "3,3"],
    'data_preprocessor:feature_type:text_transformer:text_encoding:tfidf_encoding:norm': ["l2", "l1"],
    'data_preprocessor:feature_type:text_transformer:text_encoding:tfidf_encoding:per_column': [True, False],
    'data_preprocessor:feature_type:text_transformer:text_encoding:tfidf_encoding:sublinear_tf': [True, False]
}

for binary in [True, False]:
    for ngram_range in ["1,1", "1,2", "1,3", "2,2", "2,3", "3,3"]:
        for norm in ["l2", "l1"]:
            for per_column in [True, False]:
                for sublinear_tf in [True, False]:
                    cs._hyperparameters[
                        'data_preprocessor:feature_type:text_transformer:text_encoding:tfidf_encoding:binary'].default_value = binary
                    cs._hyperparameters[
                        'data_preprocessor:feature_type:text_transformer:text_encoding:tfidf_encoding:ngram_range'].default_value = ngram_range
                    cs._hyperparameters[
                        'data_preprocessor:feature_type:text_transformer:text_encoding:tfidf_encoding:norm'].default_value = norm
                    cs._hyperparameters[
                        'data_preprocessor:feature_type:text_transformer:text_encoding:tfidf_encoding:per_column'].default_value = per_column
                    cs._hyperparameters[
                        'data_preprocessor:feature_type:text_transformer:text_encoding:tfidf_encoding:sublinear_tf'].default_value = sublinear_tf

                    automl.configuration_space = cs

                    print(automl.get_configuration_space(X_train, y_train).get_default_configuration())
                    print()
