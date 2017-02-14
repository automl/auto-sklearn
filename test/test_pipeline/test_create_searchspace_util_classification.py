from collections import OrderedDict

import unittest
import numpy

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from autosklearn.pipeline.components.classification.liblinear_svc import LibLinear_SVC
from autosklearn.pipeline.components.classification.lda import LDA

from autosklearn.pipeline.components.feature_preprocessing.pca import PCA
from autosklearn.pipeline.components.feature_preprocessing.truncatedSVD import TruncatedSVD
from autosklearn.pipeline.components.feature_preprocessing.no_preprocessing import NoPreprocessing
from autosklearn.pipeline.components.feature_preprocessing.fast_ica import FastICA
from autosklearn.pipeline.components.feature_preprocessing.random_trees_embedding import RandomTreesEmbedding
import autosklearn.pipeline.create_searchspace_util

class TestCreateClassificationSearchspace(unittest.TestCase):
    _multiprocess_can_split_ = True

    def test_get_match_array_sparse_and_dense(self):
        # preproc is empty
        preprocessors = OrderedDict()
        preprocessors['pca'] = PCA
        classifiers = OrderedDict()
        classifiers['lda'] = LDA
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
        m = autosklearn.pipeline.create_searchspace_util.get_match_array(
            pipeline=((0, PCA), (1, LDA)), dataset_properties={'sparse': True})
        self.assertEqual(numpy.sum(m), 0)

        m = autosklearn.pipeline.create_searchspace_util.get_match_array(
            pipeline=((0, PCA), (1, LDA)), dataset_properties={'sparse': False})
        self.assertEqual(m, [[1]])

        # Sparse
        preprocessors['tSVD'] = TruncatedSVD
        m = autosklearn.pipeline.create_searchspace_util.get_match_array(
            pipeline=((0, Preprocessors), (1, LDA)),
            dataset_properties={'sparse': True})
        self.assertEqual(m[0], [0])  # pca
        self.assertEqual(m[1], [1])  # svd

        m = autosklearn.pipeline.create_searchspace_util.get_match_array(
            pipeline=((0, Preprocessors), (1, LDA)),
            dataset_properties={'sparse': False})
        self.assertEqual(m[0], [1])  # pca
        self.assertEqual(m[1], [0])  # svd

        preprocessors['none'] = NoPreprocessing
        m = autosklearn.pipeline.create_searchspace_util.get_match_array(
            pipeline=((0, Preprocessors), (1, LDA)),
            dataset_properties={'sparse': True})
        self.assertEqual(m[0, :], [0])  # pca
        self.assertEqual(m[1, :], [1])  # tsvd
        self.assertEqual(m[2, :], [0])  # none

        m = autosklearn.pipeline.create_searchspace_util.get_match_array(
            pipeline=((0, Preprocessors), (1, LDA)),
            dataset_properties={'sparse': False})
        self.assertEqual(m[0, :], [1])  # pca
        self.assertEqual(m[1, :], [0])  # tsvd
        self.assertEqual(m[2, :], [1])  # none

        classifiers['libsvm'] = LibLinear_SVC
        m = autosklearn.pipeline.create_searchspace_util.get_match_array(
            pipeline=((0, Preprocessors), (1, Classifiers)),
            dataset_properties={'sparse': False})
        self.assertListEqual(list(m[0, :]), [1, 1])  # pca
        self.assertListEqual(list(m[1, :]), [0, 0])  # tsvd
        self.assertListEqual(list(m[2, :]), [1, 1])  # none

        m = autosklearn.pipeline.create_searchspace_util.get_match_array(
            pipeline=((0, Preprocessors), (1, Classifiers)),
            dataset_properties={'sparse': True})
        self.assertListEqual(list(m[0, :]), [0, 0])  # pca
        self.assertListEqual(list(m[1, :]), [1, 1])  # tsvd
        self.assertListEqual(list(m[2, :]), [0, 1])  # none

        # Do fancy 3d stuff
        preprocessors['random_trees'] = RandomTreesEmbedding
        m = autosklearn.pipeline.create_searchspace_util.get_match_array(
            pipeline=((0, Preprocessors), (1, Preprocessors), (2, Classifiers)),
            dataset_properties={'sparse': False})
        # PCA followed by truncated SVD is forbidden
        self.assertEqual(list(m[0].flatten()), [1, 1, 0, 0, 1, 1, 0, 1])
        # Truncated SVD is forbidden
        self.assertEqual(list(m[1].flatten()), [0, 0, 0, 0, 0, 0, 0, 0])
        # Truncated SVD is forbidden after no_preprocessing
        self.assertEqual(list(m[2].flatten()), [1, 1, 0, 0, 1, 1, 0, 1])
        # PCA is forbidden, truncatedSVD allowed after random trees embedding
        # lda only allowed after truncatedSVD
        self.assertEqual(list(m[3].flatten()), [0, 0, 1, 1, 0, 1, 0, 1])

    def test_get_match_array_signed_unsigned_and_binary(self):
        pass

    @unittest.skip("Not currently working.")
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
        new_cs = autosklearn.pipeline.create_searchspace_util.add_forbidden(
            conf_space=cs, node_0_list=preprocessors_list,
            node_1_list=classifier_list, matches=m,
            node_0_name='preprocessor', node_1_name="classifier")
        self.assertEqual(len(new_cs.forbidden_clauses), 0)
        self.assertIsInstance(new_cs, ConfigurationSpace)

        m[1, 1] = 0
        new_cs = autosklearn.pipeline.create_searchspace_util.add_forbidden(
            conf_space=cs, node_0_list=preprocessors_list,
            node_1_list=classifier_list, matches=m,
            node_0_name='preprocessor', node_1_name="classifier")
        self.assertEqual(len(new_cs.forbidden_clauses), 1)
        self.assertEqual(new_cs.forbidden_clauses[0].components[0].value, 'cb')
        self.assertEqual(new_cs.forbidden_clauses[0].components[1].value, 'pb')
        self.assertIsInstance(new_cs, ConfigurationSpace)