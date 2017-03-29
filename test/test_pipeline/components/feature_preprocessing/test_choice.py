import unittest

import autosklearn.pipeline.components.feature_preprocessing as fp


class FeatureProcessingTest(unittest.TestCase):
    def test_get_available_components(self):
        # Target type
        for target_type, num_values in [('classification', 15),
                                        ('regression', 13)]:
            data_properties = {'target_type': target_type}

            available_components = fp.FeaturePreprocessorChoice(data_properties)\
                .get_available_components(data_properties)

            self.assertEqual(len(available_components), num_values)

        # Multiclass
        data_properties = {'target_type': 'classification',
                           'multiclass': True}
        available_components = fp.FeaturePreprocessorChoice(data_properties) \
            .get_available_components(data_properties)

        self.assertEqual(len(available_components), 15)

        # Multilabel
        data_properties = {'target_type': 'classification',
                           'multilabel': True}
        available_components = fp.FeaturePreprocessorChoice(data_properties) \
            .get_available_components(data_properties)

        self.assertEqual(len(available_components), 12)
