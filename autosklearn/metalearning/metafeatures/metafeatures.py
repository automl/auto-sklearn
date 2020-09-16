from collections import defaultdict, OrderedDict, deque
import copy

import numpy as np
import scipy.stats
from scipy.linalg import LinAlgError
import scipy.sparse
# TODO use balanced accuracy!
from sklearn.utils import check_array
from sklearn.multiclass import OneVsRestClassifier

from autosklearn.pipeline.components.data_preprocessing.data_preprocessing \
    import DataPreprocessor
from autosklearn.util.logging_ import get_logger
from .metafeature import MetaFeature, HelperFunction, DatasetMetafeatures


# TODO Allow multiple dependencies for a metafeature
# TODO Add HelperFunction as an object
class HelperFunctions(object):
    def __init__(self):
        self.functions = OrderedDict()
        self.values = OrderedDict()

    def clear(self):
        self.values = OrderedDict()
        self.computation_time = OrderedDict()

    def __iter__(self):
        return self.functions.__iter__()

    def __getitem__(self, item):
        return self.functions.__getitem__(item)

    def __setitem__(self, key, value):
        return self.functions.__setitem__(key, value)

    def __delitem__(self, key):
        return self.functions.__delitem__(key)

    def __contains__(self, item):
        return self.functions.__contains__(item)

    def is_calculated(self, key):
        """Return if a helper function has already been executed.

        Necessary as get_value() can return None if the helper function hasn't
        been executed or if it returned None."""
        return key in self.values

    def get_value(self, key):
        return self.values.get(key).value

    def set_value(self, key, item):
        self.values[key] = item

    def define(self, name):
        """Decorator for adding helper functions to a "dictionary".
        This behaves like a function decorating a function,
        not a class decorating a function"""
        def wrapper(metafeature_class):
            instance = metafeature_class()
            self.__setitem__(name, instance)
            return instance
        return wrapper


class MetafeatureFunctions(object):
    def __init__(self):
        self.functions = OrderedDict()
        self.dependencies = OrderedDict()
        self.values = OrderedDict()

    def clear(self):
        self.values = OrderedDict()

    def __iter__(self):
        return self.functions.__iter__()

    def __getitem__(self, item):
        return self.functions.__getitem__(item)

    def __setitem__(self, key, value):
        return self.functions.__setitem__(key, value)

    def __delitem__(self, key):
        return self.functions.__delitem__(key)

    def __contains__(self, item):
        return self.functions.__contains__(item)

    def get_value(self, key):
        return self.values[key].value

    def set_value(self, key, item):
        self.values[key] = item

    def is_calculated(self, key):
        """Return if a helper function has already been executed.

        Necessary as get_value() can return None if the helper function hasn't
        been executed or if it returned None."""
        return key in self.values

    def get_dependency(self, name):
        """Return the dependency of metafeature "name".
        """
        return self.dependencies.get(name)

    def define(self, name, dependency=None):
        """Decorator for adding metafeature functions to a "dictionary" of
        metafeatures. This behaves like a function decorating a function,
        not a class decorating a function"""
        def wrapper(metafeature_class):
            instance = metafeature_class()
            self.__setitem__(name, instance)
            self.dependencies[name] = dependency
            return instance
        return wrapper


metafeatures = MetafeatureFunctions()
helper_functions = HelperFunctions()


################################################################################
# Simple features
################################################################################
@metafeatures.define("NumberOfInstances")
class NumberOfInstances(MetaFeature):
    def _calculate(self, X, y, categorical):
        return float(X.shape[0])


@metafeatures.define("LogNumberOfInstances",
                     dependency="NumberOfInstances")
class LogNumberOfInstances(MetaFeature):
    def _calculate(self, X, y, categorical):
        return np.log(metafeatures.get_value("NumberOfInstances"))


@metafeatures.define("NumberOfClasses")
class NumberOfClasses(MetaFeature):
    """
    Calculate the number of classes.

    Calls np.unique on the targets. If the dataset is a multilabel dataset,
    does this for each label seperately and returns the mean.
    """
    def _calculate(self, X, y, categorical):
        if len(y.shape) == 2:
            return np.mean([len(np.unique(y[:, i])) for i in range(y.shape[1])])
        else:
            return float(len(np.unique(y)))


@metafeatures.define("NumberOfFeatures")
class NumberOfFeatures(MetaFeature):
    def _calculate(self, X, y, categorical):
        return float(X.shape[1])


@metafeatures.define("LogNumberOfFeatures",
                     dependency="NumberOfFeatures")
class LogNumberOfFeatures(MetaFeature):
    def _calculate(self, X, y, categorical):
        return np.log(metafeatures.get_value("NumberOfFeatures"))


@helper_functions.define("MissingValues")
class MissingValues(HelperFunction):
    def _calculate(self, X, y, categorical):
        missing = ~np.isfinite(X)
        return missing

    def _calculate_sparse(self, X, y, categorical):
        data = [True if not np.isfinite(x) else False for x in X.data]
        missing = X.__class__((data, X.indices, X.indptr), shape=X.shape,
                              dtype=np.bool)
        return missing


@metafeatures.define("NumberOfInstancesWithMissingValues",
                     dependency="MissingValues")
class NumberOfInstancesWithMissingValues(MetaFeature):
    def _calculate(self, X, y, categorical):
        missing = helper_functions.get_value("MissingValues")
        num_missing = missing.sum(axis=1)
        return float(np.sum([1 if num > 0 else 0 for num in num_missing]))

    def _calculate_sparse(self, X, y, categorical):
        missing = helper_functions.get_value("MissingValues")
        new_missing = missing.tocsr()
        num_missing = [np.sum(new_missing.data[new_missing.indptr[i]:new_missing.indptr[i + 1]])
                       for i in range(new_missing.shape[0])]

        return float(np.sum([1 if num > 0 else 0 for num in num_missing]))


@metafeatures.define("PercentageOfInstancesWithMissingValues",
                     dependency="NumberOfInstancesWithMissingValues")
class PercentageOfInstancesWithMissingValues(MetaFeature):
    def _calculate(self, X, y, categorical):
        n_missing = metafeatures.get_value("NumberOfInstancesWithMissingValues")
        n_total = float(metafeatures["NumberOfInstances"](X, y).value)
        return float(n_missing / n_total)


@metafeatures.define("NumberOfFeaturesWithMissingValues",
                     dependency="MissingValues")
class NumberOfFeaturesWithMissingValues(MetaFeature):
    def _calculate(self, X, y, categorical):
        missing = helper_functions.get_value("MissingValues")
        num_missing = missing.sum(axis=0)
        return float(np.sum([1 if num > 0 else 0 for num in num_missing]))

    def _calculate_sparse(self, X, y, categorical):
        missing = helper_functions.get_value("MissingValues")
        new_missing = missing.tocsc()
        num_missing = [np.sum(
            new_missing.data[new_missing.indptr[i]:new_missing.indptr[i+1]])
                       for i in range(missing.shape[1])]

        return float(np.sum([1 if num > 0 else 0 for num in num_missing]))


@metafeatures.define("PercentageOfFeaturesWithMissingValues",
                     dependency="NumberOfFeaturesWithMissingValues")
class PercentageOfFeaturesWithMissingValues(MetaFeature):
    def _calculate(self, X, y, categorical):
        n_missing = metafeatures.get_value("NumberOfFeaturesWithMissingValues")
        n_total = float(metafeatures["NumberOfFeatures"](X, y).value)
        return float(n_missing / n_total)


@metafeatures.define("NumberOfMissingValues", dependency="MissingValues")
class NumberOfMissingValues(MetaFeature):
    def _calculate(self, X, y, categorical):
        return float(helper_functions.get_value("MissingValues").sum())


@metafeatures.define("PercentageOfMissingValues",
                     dependency="NumberOfMissingValues")
class PercentageOfMissingValues(MetaFeature):
    def _calculate(self, X, y, categorical):
        return float(metafeatures.get_value("NumberOfMissingValues")) / \
               float(X.shape[0]*X.shape[1])


# TODO: generalize this!
@metafeatures.define("NumberOfNumericFeatures")
class NumberOfNumericFeatures(MetaFeature):
    def _calculate(self, X, y, categorical):
        return len(categorical) - np.sum(categorical)


@metafeatures.define("NumberOfCategoricalFeatures")
class NumberOfCategoricalFeatures(MetaFeature):
    def _calculate(self, X, y, categorical):
        return np.sum(categorical)


@metafeatures.define("RatioNumericalToNominal")
class RatioNumericalToNominal(MetaFeature):
    def _calculate(self, X, y, categorical):
        num_categorical = float(metafeatures[
            "NumberOfCategoricalFeatures"](X, y, categorical).value)
        num_numerical = float(metafeatures[
            "NumberOfNumericFeatures"](X, y, categorical).value)
        if num_categorical == 0.0:
            return 0.
        return num_numerical / num_categorical


@metafeatures.define("RatioNominalToNumerical")
class RatioNominalToNumerical(MetaFeature):
    def _calculate(self, X, y, categorical):
        num_categorical = float(metafeatures[
            "NumberOfCategoricalFeatures"](X, y, categorical).value)
        num_numerical = float(metafeatures[
            "NumberOfNumericFeatures"](X, y, categorical).value)
        if num_numerical == 0.0:
            return 0.
        else:
            return num_categorical / num_numerical


# Number of attributes divided by number of samples
@metafeatures.define("DatasetRatio")
class DatasetRatio(MetaFeature):
    def _calculate(self, X, y, categorical):
        return float(metafeatures["NumberOfFeatures"](X, y).value) /\
            float(metafeatures["NumberOfInstances"](X, y).value)


@metafeatures.define("LogDatasetRatio", dependency="DatasetRatio")
class LogDatasetRatio(MetaFeature):
    def _calculate(self, X, y, categorical):
        return np.log(metafeatures.get_value("DatasetRatio"))


@metafeatures.define("InverseDatasetRatio")
class InverseDatasetRatio(MetaFeature):
    def _calculate(self, X, y, categorical):
        return float(metafeatures["NumberOfInstances"](X, y).value) /\
            float(metafeatures["NumberOfFeatures"](X, y).value)


@metafeatures.define("LogInverseDatasetRatio",
                     dependency="InverseDatasetRatio")
class LogInverseDatasetRatio(MetaFeature):
    def _calculate(self, X, y, categorical):
        return np.log(metafeatures.get_value("InverseDatasetRatio"))


@helper_functions.define("ClassOccurences")
class ClassOccurences(HelperFunction):
    def _calculate(self, X, y, categorical):
        if len(y.shape) == 2:
            occurences = []
            for i in range(y.shape[1]):
                occurences.append(self._calculate(X, y[:, i], categorical))
            return occurences
        else:
            occurence_dict = defaultdict(float)
            for value in y:
                occurence_dict[value] += 1
            return occurence_dict


@metafeatures.define("ClassProbabilityMin", dependency="ClassOccurences")
class ClassProbabilityMin(MetaFeature):
    def _calculate(self, X, y, categorical):
        occurences = helper_functions.get_value("ClassOccurences")

        min_value = np.iinfo(np.int64).max
        if len(y.shape) == 2:
            for i in range(y.shape[1]):
                for num_occurences in occurences[i].values():
                    if num_occurences < min_value:
                        min_value = num_occurences
        else:
            for num_occurences in occurences.values():
                if num_occurences < min_value:
                    min_value = num_occurences
        return float(min_value) / float(y.shape[0])


# aka default accuracy
@metafeatures.define("ClassProbabilityMax", dependency="ClassOccurences")
class ClassProbabilityMax(MetaFeature):
    def _calculate(self, X, y, categorical):
        occurences = helper_functions.get_value("ClassOccurences")
        max_value = -1

        if len(y.shape) == 2:
            for i in range(y.shape[1]):
                for num_occurences in occurences[i].values():
                    if num_occurences > max_value:
                        max_value = num_occurences
        else:
            for num_occurences in occurences.values():
                if num_occurences > max_value:
                    max_value = num_occurences
        return float(max_value) / float(y.shape[0])


@metafeatures.define("ClassProbabilityMean", dependency="ClassOccurences")
class ClassProbabilityMean(MetaFeature):
    def _calculate(self, X, y, categorical):
        occurence_dict = helper_functions.get_value("ClassOccurences")

        if len(y.shape) == 2:
            occurences = []
            for i in range(y.shape[1]):
                occurences.extend(
                    [occurrence for occurrence in occurence_dict[
                        i].values()])
            occurences = np.array(occurences)
        else:
            occurences = np.array([occurrence for occurrence in occurence_dict.values()],
                                  dtype=np.float64)
        return (occurences / y.shape[0]).mean()


@metafeatures.define("ClassProbabilitySTD", dependency="ClassOccurences")
class ClassProbabilitySTD(MetaFeature):
    def _calculate(self, X, y, categorical):
        occurence_dict = helper_functions.get_value("ClassOccurences")

        if len(y.shape) == 2:
            stds = []
            for i in range(y.shape[1]):
                std = np.array(
                    [occurrence for occurrence in occurence_dict[
                                                      i].values()],
                    dtype=np.float64)
                std = (std / y.shape[0]).std()
                stds.append(std)
            return np.mean(stds)
        else:
            occurences = np.array([occurrence for occurrence in occurence_dict.values()],
                                  dtype=np.float64)
            return (occurences / y.shape[0]).std()


################################################################################
# Reif, A Comprehensive Dataset for Evaluating Approaches of various Meta-Learning Tasks
# defines these five metafeatures as simple metafeatures, but they could also
#  be the counterpart for the skewness and kurtosis of the numerical features
@helper_functions.define("NumSymbols")
class NumSymbols(HelperFunction):
    def _calculate(self, X, y, categorical):
        symbols_per_column = []
        for i, column in enumerate(X.T):
            if categorical[i]:
                unique_values = np.unique(column)
                num_unique = np.sum(np.isfinite(unique_values))
                symbols_per_column.append(num_unique)
        return symbols_per_column

    def _calculate_sparse(self, X, y, categorical):
        symbols_per_column = []
        new_X = X.tocsc()
        for i in range(new_X.shape[1]):
            if categorical[i]:
                unique_values = np.unique(new_X.getcol(i).data)
                num_unique = np.sum(np.isfinite(unique_values))
                symbols_per_column.append(num_unique)
        return symbols_per_column


@metafeatures.define("SymbolsMin", dependency="NumSymbols")
class SymbolsMin(MetaFeature):
    def _calculate(self, X, y, categorical):
        # The minimum can only be zero if there are no nominal features,
        # otherwise it is at least one
        # TODO: shouldn't this rather be two?
        minimum = None
        for unique in helper_functions.get_value("NumSymbols"):
            if unique > 0 and (minimum is None or unique < minimum):
                minimum = unique
        return minimum if minimum is not None else 0


@metafeatures.define("SymbolsMax", dependency="NumSymbols")
class SymbolsMax(MetaFeature):
    def _calculate(self, X, y, categorical):
        values = helper_functions.get_value("NumSymbols")
        if len(values) == 0:
            return 0
        return max(max(values), 0)


@metafeatures.define("SymbolsMean", dependency="NumSymbols")
class SymbolsMean(MetaFeature):
    def _calculate(self, X, y, categorical):
        # TODO: categorical attributes without a symbol don't count towards this
        # measure
        values = [val for val in helper_functions.get_value("NumSymbols") if val > 0]
        mean = np.nanmean(values)
        return mean if np.isfinite(mean) else 0


@metafeatures.define("SymbolsSTD", dependency="NumSymbols")
class SymbolsSTD(MetaFeature):
    def _calculate(self, X, y, categorical):
        values = [val for val in helper_functions.get_value("NumSymbols") if val > 0]
        std = np.nanstd(values)
        return std if np.isfinite(std) else 0


@metafeatures.define("SymbolsSum", dependency="NumSymbols")
class SymbolsSum(MetaFeature):
    def _calculate(self, X, y, categorical):
        sum = np.nansum(helper_functions.get_value("NumSymbols"))
        return sum if np.isfinite(sum) else 0

################################################################################
# Statistical meta features
# Only use third and fourth statistical moment because it is common to
# standardize for the other two
# see Engels & Theusinger, 1998 - Using a Data Metric for Preprocessing Advice
# for Data Mining Applications.


@helper_functions.define("Kurtosisses")
class Kurtosisses(HelperFunction):
    def _calculate(self, X, y, categorical):
        kurts = []
        for i in range(X.shape[1]):
            if not categorical[i]:
                kurts.append(scipy.stats.kurtosis(X[:, i]))
        return kurts

    def _calculate_sparse(self, X, y, categorical):
        kurts = []
        X_new = X.tocsc()
        for i in range(X_new.shape[1]):
            if not categorical[i]:
                start = X_new.indptr[i]
                stop = X_new.indptr[i+1]
                kurts.append(scipy.stats.kurtosis(X_new.data[start:stop]))
        return kurts


@metafeatures.define("KurtosisMin", dependency="Kurtosisses")
class KurtosisMin(MetaFeature):
    def _calculate(self, X, y, categorical):
        kurts = helper_functions.get_value("Kurtosisses")
        minimum = np.nanmin(kurts) if len(kurts) > 0 else 0
        return minimum if np.isfinite(minimum) else 0


@metafeatures.define("KurtosisMax", dependency="Kurtosisses")
class KurtosisMax(MetaFeature):
    def _calculate(self, X, y, categorical):
        kurts = helper_functions.get_value("Kurtosisses")
        maximum = np.nanmax(kurts) if len(kurts) > 0 else 0
        return maximum if np.isfinite(maximum) else 0


@metafeatures.define("KurtosisMean", dependency="Kurtosisses")
class KurtosisMean(MetaFeature):
    def _calculate(self, X, y, categorical):
        kurts = helper_functions.get_value("Kurtosisses")
        mean = np.nanmean(kurts) if len(kurts) > 0 else 0
        return mean if np.isfinite(mean) else 0


@metafeatures.define("KurtosisSTD", dependency="Kurtosisses")
class KurtosisSTD(MetaFeature):
    def _calculate(self, X, y, categorical):
        kurts = helper_functions.get_value("Kurtosisses")
        std = np.nanstd(kurts) if len(kurts) > 0 else 0
        return std if np.isfinite(std) else 0


@helper_functions.define("Skewnesses")
class Skewnesses(HelperFunction):
    def _calculate(self, X, y, categorical):
        skews = []
        for i in range(X.shape[1]):
            if not categorical[i]:
                skews.append(scipy.stats.skew(X[:, i]))
        return skews

    def _calculate_sparse(self, X, y, categorical):
        skews = []
        X_new = X.tocsc()
        for i in range(X_new.shape[1]):
            if not categorical[i]:
                start = X_new.indptr[i]
                stop = X_new.indptr[i + 1]
                skews.append(scipy.stats.skew(X_new.data[start:stop]))
        return skews


@metafeatures.define("SkewnessMin", dependency="Skewnesses")
class SkewnessMin(MetaFeature):
    def _calculate(self, X, y, categorical):
        skews = helper_functions.get_value("Skewnesses")
        minimum = np.nanmin(skews) if len(skews) > 0 else 0
        return minimum if np.isfinite(minimum) else 0


@metafeatures.define("SkewnessMax", dependency="Skewnesses")
class SkewnessMax(MetaFeature):
    def _calculate(self, X, y, categorical):
        skews = helper_functions.get_value("Skewnesses")
        maximum = np.nanmax(skews) if len(skews) > 0 else 0
        return maximum if np.isfinite(maximum) else 0


@metafeatures.define("SkewnessMean", dependency="Skewnesses")
class SkewnessMean(MetaFeature):
    def _calculate(self, X, y, categorical):
        skews = helper_functions.get_value("Skewnesses")
        mean = np.nanmean(skews) if len(skews) > 0 else 0
        return mean if np.isfinite(mean) else 0


@metafeatures.define("SkewnessSTD", dependency="Skewnesses")
class SkewnessSTD(MetaFeature):
    def _calculate(self, X, y, categorical):
        skews = helper_functions.get_value("Skewnesses")
        std = np.nanstd(skews) if len(skews) > 0 else 0
        return std if np.isfinite(std) else 0


"""
@metafeatures.define("cancor1")
def cancor1(X, y):
    pass

@metafeatures.define("cancor2")
def cancor2(X, y):
    pass
"""


################################################################################
# Information-theoretic metafeatures
@metafeatures.define("ClassEntropy")
class ClassEntropy(MetaFeature):
    def _calculate(self, X, y, categorical):
        labels = 1 if len(y.shape) == 1 else y.shape[1]
        if labels == 1:
            y = y.reshape((-1, 1))

        entropies = []
        for i in range(labels):
            occurence_dict = defaultdict(float)
            for value in y[:, i]:
                occurence_dict[value] += 1
            entropies.append(scipy.stats.entropy([occurence_dict[key] for key in
                                                 occurence_dict], base=2))

        return np.mean(entropies)


"""
#@metafeatures.define("normalized_class_entropy")

#@metafeatures.define("attribute_entropy")

#@metafeatures.define("normalized_attribute_entropy")

#@metafeatures.define("joint_entropy")

#@metafeatures.define("mutual_information")

#@metafeatures.define("noise-signal-ratio")

#@metafeatures.define("signal-noise-ratio")

#@metafeatures.define("equivalent_number_of_attributes")

#@metafeatures.define("conditional_entropy")

#@metafeatures.define("average_attribute_entropy")
"""


################################################################################
# Landmarking features, computed with cross validation
# These should be invoked with the same transformations of X and y with which
# sklearn will be called later on


# from Pfahringer 2000
# Linear discriminant learner
@metafeatures.define("LandmarkLDA")
class LandmarkLDA(MetaFeature):
    def _calculate(self, X, y, categorical):
        import sklearn.discriminant_analysis
        if type(y) in ('binary', 'multiclass'):
            kf = sklearn.model_selection.StratifiedKFold(n_splits=5)
        else:
            kf = sklearn.model_selection.KFold(n_splits=5)

        accuracy = 0.
        try:
            for train, test in kf.split(X, y):
                lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()

                if len(y.shape) == 1 or y.shape[1] == 1:
                    lda.fit(X[train], y[train])
                else:
                    lda = OneVsRestClassifier(lda)
                    lda.fit(X[train], y[train])

                predictions = lda.predict(X[test])
                accuracy += sklearn.metrics.accuracy_score(predictions, y[test])
            return accuracy / 5
        except scipy.linalg.LinAlgError as e:
            self.logger.warning("LDA failed: %s Returned 0 instead!" % e)
            return np.NaN
        except ValueError as e:
            self.logger.warning("LDA failed: %s Returned 0 instead!" % e)
            return np.NaN

    def _calculate_sparse(self, X, y, categorical):
        return np.NaN


# Naive Bayes
@metafeatures.define("LandmarkNaiveBayes")
class LandmarkNaiveBayes(MetaFeature):
    def _calculate(self, X, y, categorical):
        import sklearn.naive_bayes

        if type(y) in ('binary', 'multiclass'):
            kf = sklearn.model_selection.StratifiedKFold(n_splits=5)
        else:
            kf = sklearn.model_selection.KFold(n_splits=5)

        accuracy = 0.
        for train, test in kf.split(X, y):
            nb = sklearn.naive_bayes.GaussianNB()

            if len(y.shape) == 1 or y.shape[1] == 1:
                nb.fit(X[train], y[train])
            else:
                nb = OneVsRestClassifier(nb)
                nb.fit(X[train], y[train])

            predictions = nb.predict(X[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, y[test])
        return accuracy / 5

    def _calculate_sparse(self, X, y, categorical):
        return np.NaN


# Cart learner instead of C5.0
@metafeatures.define("LandmarkDecisionTree")
class LandmarkDecisionTree(MetaFeature):
    def _calculate(self, X, y, categorical):
        import sklearn.tree

        if type(y) in ('binary', 'multiclass'):
            kf = sklearn.model_selection.StratifiedKFold(n_splits=5)
        else:
            kf = sklearn.model_selection.KFold(n_splits=5)

        accuracy = 0.
        for train, test in kf.split(X, y):
            random_state = sklearn.utils.check_random_state(42)
            tree = sklearn.tree.DecisionTreeClassifier(random_state=random_state)

            if len(y.shape) == 1 or y.shape[1] == 1:
                tree.fit(X[train], y[train])
            else:
                tree = OneVsRestClassifier(tree)
                tree.fit(X[train], y[train])

            predictions = tree.predict(X[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, y[test])
        return accuracy / 5

    def _calculate_sparse(self, X, y, categorical):
        return np.NaN


"""If there is a dataset which has OneHotEncoded features it can happend that
a node learner splits at one of the attribute encodings. This should be fine
as the dataset is later on used encoded."""


# TODO: use the same tree, this has then to be computed only once and hence
#  saves a lot of time...
@metafeatures.define("LandmarkDecisionNodeLearner")
class LandmarkDecisionNodeLearner(MetaFeature):
    def _calculate(self, X, y, categorical):
        import sklearn.tree

        if type(y) in ('binary', 'multiclass'):
            kf = sklearn.model_selection.StratifiedKFold(n_splits=5)
        else:
            kf = sklearn.model_selection.KFold(n_splits=5)

        accuracy = 0.
        for train, test in kf.split(X, y):
            random_state = sklearn.utils.check_random_state(42)
            node = sklearn.tree.DecisionTreeClassifier(
                criterion="entropy", max_depth=1, random_state=random_state,
                min_samples_split=2, min_samples_leaf=1,  max_features=None)
            if len(y.shape) == 1 or y.shape[1] == 1:
                node.fit(X[train], y[train])
            else:
                node = OneVsRestClassifier(node)
                node.fit(X[train], y[train])
            predictions = node.predict(X[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, y[test])
        return accuracy / 5

    def _calculate_sparse(self, X, y, categorical):
        return np.NaN


@metafeatures.define("LandmarkRandomNodeLearner")
class LandmarkRandomNodeLearner(MetaFeature):
    def _calculate(self, X, y, categorical):
        import sklearn.tree

        if type(y) in ('binary', 'multiclass'):
            kf = sklearn.model_selection.StratifiedKFold(n_splits=5)
        else:
            kf = sklearn.model_selection.KFold(n_splits=5)
        accuracy = 0.

        for train, test in kf.split(X, y):
            random_state = sklearn.utils.check_random_state(42)
            node = sklearn.tree.DecisionTreeClassifier(
                criterion="entropy", max_depth=1, random_state=random_state,
                min_samples_split=2, min_samples_leaf=1, max_features=1)
            node.fit(X[train], y[train])
            predictions = node.predict(X[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, y[test])
        return accuracy / 5

    def _calculate_sparse(self, X, y, categorical):
        return np.NaN


"""
This is wrong...
@metafeatures.define("landmark_worst_node_learner")
def landmark_worst_node_learner(X, y):
    # TODO: this takes more than 10 minutes on some datasets (eg mfeat-pixels)
    # which has 240*6 = 1440 discrete attributes...
    # TODO: calculate information gain instead of using the worst test result
    import sklearn.tree
    performances = []
    for attribute_idx in range(X.shape[1]):
        kf = sklearn.model_selection.StratifiedKFold(y, n_folds=5)
        accuracy = 0.
        for train, test in kf:
            node = sklearn.tree.DecisionTreeClassifier(criterion="entropy",
                max_features=None, max_depth=1, min_samples_split=1,
                min_samples_leaf=1)
            node.fit(X[train][:,attribute_idx].reshape((-1, 1)), y[train],
                     check_input=False)
            predictions = node.predict(X[test][:,attribute_idx].reshape((-1, 1)))
            accuracy += sklearn.metrics.accuracy_score(predictions, y[test])
        performances.append(1 - (accuracy / 10))
    return max(performances)
"""


# Replace the Elite 1NN with a normal 1NN, this slightly changes the
# intuition behind this landmark, but Elite 1NN is used nowhere else...
@metafeatures.define("Landmark1NN")
class Landmark1NN(MetaFeature):
    def _calculate(self, X, y, categorical):
        import sklearn.neighbors

        if type(y) in ('binary', 'multiclass'):
            kf = sklearn.model_selection.StratifiedKFold(n_splits=5)
        else:
            kf = sklearn.model_selection.KFold(n_splits=5)

        accuracy = 0.
        for train, test in kf.split(X, y):
            kNN = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
            if len(y.shape) == 1 or y.shape[1] == 1:
                kNN.fit(X[train], y[train])
            else:
                kNN = OneVsRestClassifier(kNN)
                kNN.fit(X[train], y[train])
            predictions = kNN.predict(X[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, y[test])
        return accuracy / 5


################################################################################
# Bardenet 2013 - Collaborative Hyperparameter Tuning
# K number of classes ("number_of_classes")
# log(d), log(number of attributes)
# log(n/d), log(number of training instances/number of attributes)
# p, how many principal components to keep in order to retain 95% of the
#     dataset variance
# skewness of a dataset projected onto one principal component...
# kurtosis of a dataset projected onto one principal component
@helper_functions.define("PCA")
class PCA(HelperFunction):
    def _calculate(self, X, y, categorical):
        import sklearn.decomposition
        pca = sklearn.decomposition.PCA(copy=True)
        rs = np.random.RandomState(42)
        indices = np.arange(X.shape[0])
        for i in range(10):
            try:
                rs.shuffle(indices)
                pca.fit(X[indices])
                return pca
            except LinAlgError:
                pass
        self.logger.warning("Failed to compute a Principle Component Analysis")
        return None

    def _calculate_sparse(self, X, y, categorical):
        import sklearn.decomposition
        rs = np.random.RandomState(42)
        indices = np.arange(X.shape[0])
        # This is expensive, but necessary with scikit-learn 0.15
        Xt = X.astype(np.float64)
        for i in range(10):
            try:
                rs.shuffle(indices)
                truncated_svd = sklearn.decomposition.TruncatedSVD(
                    n_components=X.shape[1]-1, random_state=i,
                    algorithm="randomized")
                truncated_svd.fit(Xt[indices])
                return truncated_svd
            except LinAlgError:
                pass
        self.logger.warning("Failed to compute a Truncated SVD")
        return None


# Maybe define some more...
@metafeatures.define("PCAFractionOfComponentsFor95PercentVariance", dependency="PCA")
class PCAFractionOfComponentsFor95PercentVariance(MetaFeature):
    def _calculate(self, X, y, categorical):
        pca_ = helper_functions.get_value("PCA")
        if pca_ is None:
            return np.NaN
        sum_ = 0.
        idx = 0
        while sum_ < 0.95 and idx < len(pca_.explained_variance_ratio_):
            sum_ += pca_.explained_variance_ratio_[idx]
            idx += 1
        return float(idx)/float(X.shape[1])


# Kurtosis of first PC
@metafeatures.define("PCAKurtosisFirstPC", dependency="PCA")
class PCAKurtosisFirstPC(MetaFeature):
    def _calculate(self, X, y, categorical):
        pca_ = helper_functions.get_value("PCA")
        if pca_ is None:
            return np.NaN
        components = pca_.components_
        pca_.components_ = components[:1]
        transformed = pca_.transform(X)
        pca_.components_ = components

        kurtosis = scipy.stats.kurtosis(transformed)
        return kurtosis[0]


# Skewness of first PC
@metafeatures.define("PCASkewnessFirstPC", dependency="PCA")
class PCASkewnessFirstPC(MetaFeature):
    def _calculate(self, X, y, categorical):
        pca_ = helper_functions.get_value("PCA")
        if pca_ is None:
            return np.NaN
        components = pca_.components_
        pca_.components_ = components[:1]
        transformed = pca_.transform(X)
        pca_.components_ = components

        skewness = scipy.stats.skew(transformed)
        return skewness[0]


def calculate_all_metafeatures_encoded_labels(X, y, categorical, dataset_name,
                                              calculate=None, dont_calculate=None):
    """
    Calculate only metafeatures for which a 1HotEncoded feature matrix is necessery.
    """

    calculate = set()
    calculate.update(npy_metafeatures)

    return calculate_all_metafeatures(X, y, categorical, dataset_name,
                                      calculate=calculate,
                                      dont_calculate=dont_calculate)


def calculate_all_metafeatures_with_labels(X, y, categorical, dataset_name, calculate=None,
                                           dont_calculate=None):
    if dont_calculate is None:
        dont_calculate = set()
    else:
        dont_calculate = copy.deepcopy(dont_calculate)
    dont_calculate.update(npy_metafeatures)
    return calculate_all_metafeatures(X, y, categorical, dataset_name,
                                      calculate=calculate,
                                      dont_calculate=dont_calculate)


def calculate_all_metafeatures(X, y, categorical, dataset_name, calculate=None, dont_calculate=None,
                               densify_threshold=1000):
    logger = get_logger(__name__)

    """Calculate all metafeatures."""
    helper_functions.clear()
    metafeatures.clear()
    mf_ = dict()

    visited = set()
    to_visit = deque()
    to_visit.extend(metafeatures)

    X_transformed = None
    y_transformed = None

    # TODO calculate the numpy metafeatures after all others to consume less
    # memory
    while len(to_visit) > 0:
        name = to_visit.pop()
        if calculate is not None and name not in calculate:
            continue
        if dont_calculate is not None and name in dont_calculate:
            continue

        if name in npy_metafeatures:
            if X_transformed is None:
                # TODO make sure this is done as efficient as possible (no copy for
                # sparse matrices because of wrong sparse format)
                sparse = scipy.sparse.issparse(X)
                DPP = DataPreprocessor(
                    categorical_features=categorical, force_sparse_output=True)
                X_transformed = DPP.fit_transform(X)
                categorical_transformed = [False] * X_transformed.shape[1]

                # Densify the transformed matrix
                if not sparse and scipy.sparse.issparse(X_transformed):
                    bytes_per_float = X_transformed.dtype.itemsize
                    num_elements = X_transformed.shape[0] * X_transformed.shape[1]
                    megabytes_required = num_elements * bytes_per_float / 1000 / 1000
                    if megabytes_required < densify_threshold:
                        X_transformed = X_transformed.todense()

                # This is not only important for datasets which are somehow
                # sorted in a strange way, but also prevents lda from failing in
                # some cases.
                # Because this is advanced indexing, a copy of the data is returned!!!
                X_transformed = check_array(X_transformed,
                                            force_all_finite=True,
                                            accept_sparse='csr')
                rs = np.random.RandomState(42)
                indices = np.arange(X_transformed.shape[0])
                rs.shuffle(indices)
                # TODO Shuffle inplace
                X_transformed = X_transformed[indices]
                y_transformed = y[indices]

            X_ = X_transformed
            y_ = y_transformed
            categorical_ = categorical_transformed
        else:
            X_ = X
            y_ = y
            categorical_ = categorical

        dependency = metafeatures.get_dependency(name)
        if dependency is not None:
            is_metafeature = dependency in metafeatures
            is_helper_function = dependency in helper_functions

            if is_metafeature and is_helper_function:
                raise NotImplementedError()
            elif not is_metafeature and not is_helper_function:
                raise ValueError(dependency)
            elif is_metafeature and not metafeatures.is_calculated(dependency):
                to_visit.appendleft(name)
                continue
            elif is_helper_function and not helper_functions.is_calculated(
                    dependency):
                logger.info("%s: Going to calculate: %s", dataset_name,
                            dependency)
                value = helper_functions[dependency](X_, y_, categorical_)
                helper_functions.set_value(dependency, value)
                mf_[dependency] = value

        logger.info("%s: Going to calculate: %s", dataset_name,
                    name)

        value = metafeatures[name](X_, y_, categorical_)
        metafeatures.set_value(name, value)
        mf_[name] = value
        visited.add(name)

    mf_ = DatasetMetafeatures(dataset_name, mf_)
    return mf_


npy_metafeatures = set(["LandmarkLDA",
                        "LandmarkNaiveBayes",
                        "LandmarkDecisionTree",
                        "LandmarkDecisionNodeLearner",
                        "LandmarkRandomNodeLearner",
                        "LandmarkWorstNodeLearner",
                        "Landmark1NN",
                        "PCAFractionOfComponentsFor95PercentVariance",
                        "PCAKurtosisFirstPC",
                        "PCASkewnessFirstPC",
                        "Skewnesses",
                        "SkewnessMin",
                        "SkewnessMax",
                        "SkewnessMean",
                        "SkewnessSTD",
                        "Kurtosisses",
                        "KurtosisMin",
                        "KurtosisMax",
                        "KurtosisMean",
                        "KurtosisSTD"])

subsets = dict()
# All implemented metafeatures
subsets["all"] = set(metafeatures.functions.keys())

# Metafeatures used by Pfahringer et al. (2000) in the first experiment
subsets["pfahringer_2000_experiment1"] = set(["number_of_features",
                                              "number_of_numeric_features",
                                              "number_of_categorical_features",
                                              "number_of_classes",
                                              "class_probability_max",
                                              "landmark_lda",
                                              "landmark_naive_bayes",
                                              "landmark_decision_tree"])

# Metafeatures used by Pfahringer et al. (2000) in the second experiment
# worst node learner not implemented yet
"""
pfahringer_2000_experiment2 = set(["landmark_decision_node_learner",
                                   "landmark_random_node_learner",
                                   "landmark_worst_node_learner",
                                   "landmark_1NN"])
"""

# Metafeatures used by Yogatama and Mann (2014)
subsets["yogotama_2014"] = set(["log_number_of_features",
                                "log_number_of_instances",
                                "number_of_classes"])

# Metafeatures used by Bardenet et al. (2013) for the AdaBoost.MH experiment
subsets["bardenet_2013_boost"] = set(["number_of_classes",
                                      "log_number_of_features",
                                      "log_inverse_dataset_ratio",
                                      "pca_95percent"])

# Metafeatures used by Bardenet et al. (2013) for the Neural Net experiment
subsets["bardenet_2013_nn"] = set(["number_of_classes",
                                   "log_number_of_features",
                                   "log_inverse_dataset_ratio",
                                   "pca_kurtosis_first_pc",
                                   "pca_skewness_first_pc"])
