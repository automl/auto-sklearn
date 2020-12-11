import os
import unittest
import logging
import logging.config
import tempfile
import yaml


from autosklearn.util import logging_


class LoggingTest(unittest.TestCase):

    def test_setup_logger(self):
        # Test that setup_logger function correctly configures the logger
        # according to the given dictionary, and uses the default
        # logging.yaml file if logging_config is not specified.

        with open(os.path.join(os.path.dirname(__file__), 'example_config.yaml'), 'r') as fh:
            example_config = yaml.safe_load(fh)

        # Configure logger with example_config.yaml.
        logging_.setup_logger(logging_config=example_config,
                              output_dir=tempfile.gettempdir())

        # example_config sets the root logger's level to CRITICAL,
        # which corresponds to 50.
        self.assertEqual(logging.getLogger().getEffectiveLevel(), 50)

        # This time use the default configuration.
        logging_.setup_logger(logging_config=None,
                              output_dir=tempfile.gettempdir())

        # default config sets the root logger's level to DEBUG,
        # which corresponds to 10.
        self.assertEqual(logging.getLogger().getEffectiveLevel(), 10)

        # Make sure we log to the desired directory
        logging_.setup_logger(output_dir=os.path.dirname(__file__),
                              filename='test.log'
                              )
        logger = logging.getLogger()
        logger.info('test_setup_logger')

        with open(os.path.join(os.path.dirname(__file__), 'test.log')) as fh:
            self.assertIn('test_setup_logger', ''.join(fh.readlines()))
        os.remove(os.path.join(os.path.dirname(__file__), 'test.log'))
