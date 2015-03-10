from collections import OrderedDict

import unittest
import numpy

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import CategoricalHyperparameter

from ParamSklearn.components.classification.random_forest import RandomForest
from ParamSklearn.components.classification.liblinear import LibLinear_SVC

from ParamSklearn.components.preprocessing.pca import PCA
from ParamSklearn.components.preprocessing.truncatedSVD import TruncatedSVD
from ParamSklearn.components.preprocessing.no_peprocessing import NoPreprocessing

from ParamSklearn.classification import ParamSklearnClassifier

class TestCreateClassificationSearchspace(unittest.TestCase):

    def test_get_match_array(self):
        # preproc is empty
        preprocessors = OrderedDict()
        preprocessors["pca"] = PCA  # dense
        classifiers = OrderedDict()
        classifiers["random_forest"] = RandomForest
        m = ParamSklearnClassifier.get_match_array(preprocessors=preprocessors, classifiers=classifiers, sparse=True)
        self.assertEqual(numpy.sum(m), 0)

        m = ParamSklearnClassifier.get_match_array(preprocessors=preprocessors, classifiers=classifiers, sparse=False)
        self.assertEqual(m, [[1]])

        preprocessors['TSVD'] = TruncatedSVD  # sparse
        m = ParamSklearnClassifier.get_match_array(preprocessors=preprocessors, classifiers=classifiers, sparse=True)
        self.assertEqual(m[0], [0])  # pca
        self.assertEqual(m[1], [1])  # svd

        m = ParamSklearnClassifier.get_match_array(preprocessors=preprocessors, classifiers=classifiers, sparse=False)
        self.assertEqual(m[0], [1])  # pca
        self.assertEqual(m[1], [0])  # svd

        preprocessors['none'] = NoPreprocessing  # sparse + dense
        m = ParamSklearnClassifier.get_match_array(preprocessors=preprocessors, classifiers=classifiers, sparse=True)
        self.assertEqual(m[0, :], [0])  # pca
        self.assertEqual(m[1, :], [1])  # tsvd
        self.assertEqual(m[2, :], [0])  # none

        m = ParamSklearnClassifier.get_match_array(preprocessors=preprocessors, classifiers=classifiers, sparse=False)
        self.assertEqual(m[0, :], [1])  # pca
        self.assertEqual(m[1, :], [0])  # tsvd
        self.assertEqual(m[2, :], [1])  # none

        classifiers['libsvm'] = LibLinear_SVC
        m = ParamSklearnClassifier.get_match_array(preprocessors=preprocessors, classifiers=classifiers, sparse=False)
        self.assertListEqual(list(m[0, :]), [1, 1])  # pca
        self.assertListEqual(list(m[1, :]), [0, 0])  # tsvd
        self.assertListEqual(list(m[2, :]), [1, 1])  # none

        m = ParamSklearnClassifier.get_match_array(preprocessors=preprocessors, classifiers=classifiers, sparse=True)
        self.assertListEqual(list(m[0, :]), [0, 0])  # pca
        self.assertListEqual(list(m[1, :]), [1, 1])  # tsvd
        self.assertListEqual(list(m[2, :]), [0, 1])  # none

    def test_get_idx_to_keep(self):
        m = numpy.zeros([3, 4])
        col, row = ParamSklearnClassifier._get_idx_to_keep(m)
        self.assertListEqual(col, [])
        self.assertListEqual(row, [])

        m = numpy.zeros([100, 50])
        c_keep = set()
        r_keep = set()
        for i in range(20):
            col_idx = numpy.random.randint(low=0, high=50, size=1)[0]
            c_keep.add(col_idx)
            row_idx = numpy.random.randint(low=0, high=100, size=1)[0]
            r_keep.add(row_idx)
            m[row_idx, col_idx] = 1
            col, row = ParamSklearnClassifier._get_idx_to_keep(m)
            self.assertListEqual(col, sorted(c_keep))
            self.assertListEqual(row, sorted(r_keep))
            [self.assertTrue(c < m.shape[1]) for c in c_keep]
            [self.assertTrue(r < m.shape[0]) for r in r_keep]


    def test_sanitize_arrays(self):
        m = numpy.zeros([2, 3])
        preprocessors_list = ['pa', 'pb']
        preprocessors = OrderedDict([['pa', 1], ['pb', 2]])
        classifier_list = ['ca', 'cb', 'cc']
        classifiers = OrderedDict([['ca', 1], ['cb', 2], ['cc', 3]])

        # all zeros -> empty
        new_m, new_preprocessors_list, new_classifier_list, new_preproc, new_class = ParamSklearnClassifier.sanitize_arrays(m=m, preprocessors=preprocessors, preprocessors_list=preprocessors_list, classifiers=classifiers, classifiers_list=classifier_list)
        self.assertEqual(len(new_m), 0)
        self.assertTrue(len(new_classifier_list) == len(new_preprocessors_list) == 0)
        self.assertTrue(len(new_preproc) == len(new_class) == 0)

        for i in range(20):
            m = numpy.zeros([2, 3])
            class_idx = numpy.random.randint(low=0, high=m.shape[1], size=1)[0]
            pre_idx = numpy.random.randint(low=0, high=m.shape[0], size=1)[0]
            m[pre_idx, class_idx] = 1
            new_m, new_preprocessors_list, new_classifier_list, new_preproc, new_class = ParamSklearnClassifier.sanitize_arrays(m=m, preprocessors=preprocessors, preprocessors_list=preprocessors_list, classifiers=classifiers, classifiers_list=classifier_list)
            self.assertIn(preprocessors_list[pre_idx], new_preprocessors_list)
            self.assertIn(preprocessors_list[pre_idx], preprocessors)
            self.assertIn(classifier_list[class_idx], new_classifier_list)
            self.assertIn(classifier_list[class_idx], classifiers)
            self.assertTrue(new_m.shape[0] == new_m.shape[1] == 1)

        m = numpy.array([[1, 0, 0], [0, 1, 0]])
        new_m, new_preprocessors_list, new_classifier_list, new_preproc, new_class = ParamSklearnClassifier.sanitize_arrays(m=m, preprocessors=preprocessors, preprocessors_list=preprocessors_list, classifiers=classifiers, classifiers_list=classifier_list)
        self.assertListEqual(preprocessors_list, new_preprocessors_list)
        [self.assertIn(p, preprocessors) for p in preprocessors_list]
        self.assertListEqual(classifier_list[:-1], new_classifier_list)
        [self.assertIn(c, classifiers) for c in new_classifier_list]
        self.assertTrue(m.shape[0], new_m.shape[0])
        self.assertTrue(m.shape[1], new_m.shape[1])

    def test_add_forbidden(self):
        m = numpy.ones([2, 3])
        preprocessors_list = ['pa', 'pb']
        classifier_list = ['ca', 'cb', 'cc']
        cs = ConfigurationSpace()
        preprocessor = CategoricalHyperparameter(name='preprocessor', choices=preprocessors_list)
        classifier = CategoricalHyperparameter(name='classifier', choices=classifier_list)
        cs.add_hyperparameter(preprocessor)
        cs.add_hyperparameter(classifier)
        new_cs = ParamSklearnClassifier.add_forbidden(conf_space=cs, preproc_list=preprocessors_list, class_list=classifier_list, matches=m)
        self.assertEqual(len(new_cs.forbidden_clauses), 0)
        self.assertIsInstance(new_cs, ConfigurationSpace)

        m[0, 0] = 0
        new_cs = ParamSklearnClassifier.add_forbidden(conf_space=cs, preproc_list=preprocessors_list, class_list=classifier_list, matches=m)
        self.assertEqual(len(new_cs.forbidden_clauses), 1)
        self.assertEqual(new_cs.forbidden_clauses[0].components[0].value, 'ca')
        self.assertEqual(new_cs.forbidden_clauses[0].components[1].value, 'pa')
        self.assertIsInstance(new_cs, ConfigurationSpace)