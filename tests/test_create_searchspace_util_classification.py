from collections import OrderedDict

import unittest
import numpy

from HPOlibConfigSpace.configuration_space import ConfigurationSpace
from HPOlibConfigSpace.hyperparameters import CategoricalHyperparameter

from ParamSklearn.components.classification.liblinear_svc import LibLinear_SVC
from ParamSklearn.components.classification.lda import LDA

from ParamSklearn.components.preprocessing.pca import PCA
from ParamSklearn.components.preprocessing.truncatedSVD import TruncatedSVD
from ParamSklearn.components.preprocessing.no_preprocessing import NoPreprocessing
from ParamSklearn.components.preprocessing.random_trees_embedding import RandomTreesEmbedding
import ParamSklearn.create_searchspace_util

class TestCreateClassificationSearchspace(unittest.TestCase):

    def test_get_match_array(self):
        # preproc is empty
        preprocessors = OrderedDict()
        preprocessors['pca'] = PCA
        classifiers = OrderedDict()
        classifiers['rf'] = LDA
        # Sparse + dense
        class Preprocessors(object):
            @classmethod
            def get_available_components(self, *args, **kwargs):
                return preprocessors

        class Classifiers(object):
            @classmethod
            def get_available_components(self, *args, **kwargs):
                return classifiers

        # Dense
        m = ParamSklearn.create_searchspace_util.get_match_array(
            node_0=PCA, node_1=LDA, dataset_properties={'sparse': True})
        self.assertEqual(numpy.sum(m), 0)

        m = ParamSklearn.create_searchspace_util.get_match_array(
            node_0=PCA, node_1=LDA, dataset_properties={'sparse': False})
        self.assertEqual(m, [[1]])

        # Sparse
        preprocessors['tSVD'] = TruncatedSVD
        m = ParamSklearn.create_searchspace_util.get_match_array(
            node_0=Preprocessors, node_1=LDA,
            dataset_properties={'sparse': True})
        self.assertEqual(m[0], [0])  # pca
        self.assertEqual(m[1], [1])  # svd

        m = ParamSklearn.create_searchspace_util.get_match_array(
            node_0=Preprocessors, node_1=LDA,
            dataset_properties={'sparse': False})
        self.assertEqual(m[0], [1])  # pca
        self.assertEqual(m[1], [0])  # svd

        preprocessors['none'] = NoPreprocessing
        m = ParamSklearn.create_searchspace_util.get_match_array(
            node_0=Preprocessors, node_1=LDA,
            dataset_properties={'sparse': True})
        self.assertEqual(m[0, :], [0])  # pca
        self.assertEqual(m[1, :], [1])  # tsvd
        self.assertEqual(m[2, :], [0])  # none

        m = ParamSklearn.create_searchspace_util.get_match_array(
            node_0=Preprocessors, node_1=LDA,
            dataset_properties={'sparse': False})
        self.assertEqual(m[0, :], [1])  # pca
        self.assertEqual(m[1, :], [0])  # tsvd
        self.assertEqual(m[2, :], [1])  # none

        classifiers['libsvm'] = LibLinear_SVC
        m = ParamSklearn.create_searchspace_util.get_match_array(
            node_0=Preprocessors, node_1=Classifiers,
            dataset_properties={'sparse': False})
        self.assertListEqual(list(m[0, :]), [1, 1])  # pca
        self.assertListEqual(list(m[1, :]), [0, 0])  # tsvd
        self.assertListEqual(list(m[2, :]), [1, 1])  # none

        m = ParamSklearn.create_searchspace_util.get_match_array(
            node_0=Preprocessors, node_1=Classifiers,
            dataset_properties={'sparse': True})
        self.assertListEqual(list(m[0, :]), [0, 0])  # pca
        self.assertListEqual(list(m[1, :]), [1, 1])  # tsvd
        self.assertListEqual(list(m[2, :]), [0, 1])  # none

        preprocessors['rte'] = RandomTreesEmbedding
        m = ParamSklearn.create_searchspace_util.get_match_array(
            node_0=Preprocessors, node_1=Classifiers,
            dataset_properties={'sparse': False})
        self.assertListEqual(list(m[0, :]), [1, 1])  # pca
        self.assertListEqual(list(m[1, :]), [0, 0])  # tsvd
        self.assertListEqual(list(m[2, :]), [1, 1])  # none
        self.assertListEqual(list(m[3, :]), [0, 1])  # random trees embedding

        m = ParamSklearn.create_searchspace_util.get_match_array(
            node_0=Preprocessors, node_1=Classifiers,
            dataset_properties={'sparse': True})
        self.assertListEqual(list(m[0, :]), [0, 0])  # pca
        self.assertListEqual(list(m[1, :]), [1, 1])  # tsvd
        self.assertListEqual(list(m[2, :]), [0, 1])  # none
        self.assertListEqual(list(m[3, :]), [0, 0])  # random trees embedding

    def test_get_idx_to_keep(self):
        m = numpy.zeros([3, 4])
        col, row = ParamSklearn.create_searchspace_util._get_idx_to_keep(m)
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
            col, row = ParamSklearn.create_searchspace_util._get_idx_to_keep(m)
            self.assertListEqual(col, sorted(c_keep))
            self.assertListEqual(row, sorted(r_keep))
            [self.assertTrue(c < m.shape[1]) for c in c_keep]
            [self.assertTrue(r < m.shape[0]) for r in r_keep]

    def test_sanitize_arrays(self):
        class Choices(list):
            def get_available_components(self, *args, **kwargs):
                return OrderedDict(((v, v) for i, v in enumerate(self[:])))

        m = numpy.zeros([2, 3])
        preprocessors = Choices(['pa', 'pb'])
        classifiers = Choices(['ca', 'cb', 'cc'])

        # all zeros -> empty
        new_m, new_preproc_list, new_class_list = \
            ParamSklearn.create_searchspace_util.sanitize_arrays(
                matches=m, node_0=preprocessors, node_1=classifiers,
                dataset_properties={})
        self.assertEqual(len(new_m), 0)
        self.assertTrue(len(new_preproc_list) == len(new_class_list) == 0)

        for i in range(20):
            m = numpy.zeros([2, 3])
            class_idx = numpy.random.randint(low=0, high=m.shape[1], size=1)[0]
            pre_idx = numpy.random.randint(low=0, high=m.shape[0], size=1)[0]
            m[pre_idx, class_idx] = 1
            new_m, new_preproc_list, new_class_list = \
                ParamSklearn.create_searchspace_util.sanitize_arrays(
                    matches=m, node_0=preprocessors, node_1=classifiers,
                    dataset_properties={})
            print preprocessors, pre_idx, new_preproc_list
            self.assertIn(preprocessors[pre_idx], new_preproc_list)
            self.assertIn(classifiers[class_idx], new_class_list)
            self.assertTrue(new_m.shape[0] == new_m.shape[1] == 1)

        m = numpy.array([[1, 0, 0], [0, 1, 0]])
        new_m, new_preproc_list, new_class_list = \
            ParamSklearn.create_searchspace_util.sanitize_arrays(
                matches=m, node_0=preprocessors, node_1=classifiers,
                dataset_properties={})
        self.assertListEqual(preprocessors, new_preproc_list)
        [self.assertIn(p, preprocessors) for p in preprocessors]
        self.assertListEqual(classifiers[:-1], new_class_list)
        [self.assertIn(c, classifiers) for c in new_class_list]
        self.assertTrue(m.shape[0], new_m.shape[0])
        self.assertTrue(m.shape[1], new_m.shape[1])

    def test_add_forbidden(self):
        m = numpy.ones([2, 3])
        preprocessors_list = ['pa', 'pb']
        classifier_list = ['ca', 'cb', 'cc']
        cs = ConfigurationSpace()
        preprocessor = CategoricalHyperparameter(name='preprocessor',
                                                 choices=preprocessors_list)
        classifier = CategoricalHyperparameter(name='classifier',
                                               choices=classifier_list)
        cs.add_hyperparameter(preprocessor)
        cs.add_hyperparameter(classifier)
        new_cs = ParamSklearn.create_searchspace_util.add_forbidden(
            conf_space=cs, node_0_list=preprocessors_list,
            node_1_list=classifier_list, matches=m,
            node_0_name='preprocessor', node_1_name="classifier")
        self.assertEqual(len(new_cs.forbidden_clauses), 0)
        self.assertIsInstance(new_cs, ConfigurationSpace)

        m[1, 1] = 0
        new_cs = ParamSklearn.create_searchspace_util.add_forbidden(
            conf_space=cs, node_0_list=preprocessors_list,
            node_1_list=classifier_list, matches=m,
            node_0_name='preprocessor', node_1_name="classifier")
        self.assertEqual(len(new_cs.forbidden_clauses), 1)
        self.assertEqual(new_cs.forbidden_clauses[0].components[0].value, 'cb')
        self.assertEqual(new_cs.forbidden_clauses[0].components[1].value, 'pb')
        self.assertIsInstance(new_cs, ConfigurationSpace)