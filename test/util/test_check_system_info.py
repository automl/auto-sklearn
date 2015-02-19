'''
Created on Dec 16, 2014

@author: Katharina Eggensperger
@projekt: AutoML2015
'''

import unittest

import AutoML2015.util.check_system_info


class TestCheckSystemInfo(unittest.TestCase):

    def test_run(self):
        AutoML2015.util.check_system_info.check_system_info()

if __name__ == "__main__":
    unittest.main()