from abc import ABCMeta, abstractmethod
from io import StringIO
import time
import types

import arff
import scipy.sparse

from autosklearn.util.logging_ import get_logger


class AbstractMetaFeature(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        self.logger = get_logger(__name__)

    @abstractmethod
    def _calculate(cls, X, y, categorical):
        pass

    def __call__(self, X, y, categorical=None):
        if categorical is None:
            categorical = [False for i in range(X.shape[1])]
        starttime = time.time()

        try:
            if scipy.sparse.issparse(X) and hasattr(self, "_calculate_sparse"):
                value = self._calculate_sparse(X, y, categorical)
            else:
                value = self._calculate(X, y, categorical)
            comment = ""
        except MemoryError as e:
            value = None
            comment = "Memory Error"

        endtime = time.time()
        return MetaFeatureValue(self.__class__.__name__, self.type_,
                                0, 0, value, endtime-starttime, comment=comment)


class MetaFeature(AbstractMetaFeature):
    def __init__(self):
        super(MetaFeature, self).__init__()
        self.type_ = "METAFEATURE"


class HelperFunction(AbstractMetaFeature):
    def __init__(self):
        super(HelperFunction, self).__init__()
        self.type_ = "HELPERFUNCTION"


class MetaFeatureValue(object):
    def __init__(self, name, type_, fold, repeat, value, time, comment=""):
        self.name = name
        self.type_ = type_
        self.fold = fold
        self.repeat = repeat
        self.value = value
        self.time = time
        self.comment = comment

    def to_arff_row(self):
        if self.type_ == "METAFEATURE":
            value = self.value
        else:
            value = "?"

        return [self.name, self.type_, self.fold,
                self.repeat, value, self.time, self.comment]

    def __repr__(self):
        repr = "%s (type: %s, fold: %d, repeat: %d, value: %s, time: %3.3f, " \
               "comment: %s)"
        repr = repr % tuple(self.to_arff_row()[:4] +
                            [str(self.to_arff_row()[4])] +
                            self.to_arff_row()[5:])
        return repr

class DatasetMetafeatures(object):
    def __init__(self, dataset_name, metafeature_values):
        self.dataset_name = dataset_name
        self.metafeature_values = metafeature_values

    def _get_arff(self):
        output = dict()
        output['relation'] = "metafeatures_%s" % (self.dataset_name)
        output['description'] = ""
        output['attributes'] = [('name', 'STRING'),
                                ('type', 'STRING'),
                                ('fold', 'NUMERIC'),
                                ('repeat', 'NUMERIC'),
                                ('value', 'NUMERIC'),
                                ('time', 'NUMERIC'),
                                ('comment', 'STRING')]
        output['data'] = []

        for key in sorted(self.metafeature_values):
            output['data'].append(self.metafeature_values[key].to_arff_row())
        return output

    def dumps(self):
        return self._get_arff()

    def dump(self, path_or_filehandle):
        output = self._get_arff()

        if isinstance(path_or_filehandle, str):
            with open(path_or_filehandle, "w") as fh:
                arff.dump(output, fh)
        else:
            arff.dump(output, path_or_filehandle)

    @classmethod
    def load(cls, path_or_filehandle):

        if isinstance(path_or_filehandle, str):
            with open(path_or_filehandle) as fh:
                input = arff.load(fh)
        else:
            input = arff.load(path_or_filehandle)

        dataset_name = input['relation'].replace('metafeatures_', '')
        metafeature_values = []
        for item in input['data']:
            mf = MetaFeatureValue(*item)
            metafeature_values.append(mf)

        return cls(dataset_name, metafeature_values)

    def __repr__(self, verbosity=0):
        repr = StringIO()
        repr.write("Metafeatures for dataset %s\n" % self.dataset_name)
        for name in self.metafeature_values:
            if verbosity == 0 and self.metafeature_values[name].type_ != "METAFEATURE":
                continue
            if verbosity == 0:
                repr.write("  %s: %s\n" %
                           (str(name), str(self.metafeature_values[name].value)))
            elif verbosity >= 1:
                repr.write("  %s: %10s  (%10fs)\n" %
                           (str(name), str(self.metafeature_values[
                                               name].value)[:10],
                            self.metafeature_values[name].time))

            # Add the reason for a crash if one happened!
            if verbosity > 1 and self.metafeature_values[name].comment:
                repr.write("    %s\n" % self.metafeature_values[name].comment)

        return repr.getvalue()

    def keys(self):
        return self.metafeature_values.keys()

    def __getitem__(self, item):
        return self.metafeature_values[item]
