# -*- encoding: utf-8 -*-
"""Created on Dec 16, 2014.

@author: Katharina Eggensperger
@projekt: AutoML2015

"""
import time
import unittest
import unittest.mock

from autosklearn.util import StopWatch


class Test(unittest.TestCase):
    _multiprocess_can_split_ = True

    def test_stopwatch_overhead(self):

        # Wall Overhead
        start = time.time()
        cpu_start = time.clock()
        watch = StopWatch()
        for i in range(1, 1000):
            watch.start_task('task_%d' % i)
            watch.stop_task('task_%d' % i)
        cpu_stop = time.clock()
        stop = time.time()
        dur = stop - start
        cpu_dur = cpu_stop - cpu_start
        cpu_overhead = cpu_dur - watch.cpu_sum()
        wall_overhead = dur - watch.wall_sum()

        self.assertLess(cpu_overhead, 1)
        self.assertLess(wall_overhead, 1)
        self.assertLess(watch.cpu_sum(), 2 * watch.wall_sum())


if __name__ == '__main__':
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
