# -*- encoding: utf-8 -*-
"""Created on Dec 16, 2014.

@author: Katharina Eggensperger
@projekt: AutoML2015

"""

import time
import unittest

import autosklearn.util.stopwatch


class Test(unittest.TestCase):

    def test_stopwatch_overhead(self):
        # CPU overhead
        start = time.clock()
        watch = autosklearn.util.stopwatch.StopWatch()
        for i in range(1, 100000):
            watch.start_task('task_%d' % i)
            watch.stop_task('task_%d' % i)
        stop = time.clock()
        dur = stop - start
        cpu_overhead = dur - watch.cpu_sum()
        self.assertLess(cpu_overhead, 1.5)

        # Wall Overhead
        start = time.time()
        watch = autosklearn.util.stopwatch.StopWatch()
        for i in range(1, 100000):
            watch.start_task('task_%d' % i)
            watch.stop_task('task_%d' % i)
        stop = time.time()
        dur = stop - start
        wall_overhead = dur - watch.wall_sum()

        self.assertLess(wall_overhead, 2)
        self.assertLess(cpu_overhead, wall_overhead)


if __name__ == '__main__':
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
