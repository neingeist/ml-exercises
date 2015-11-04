from __future__ import division, print_function
from numpy.testing import assert_approx_equal
import numpy as np
import unittest

from stump import Stump


class StumpTestCase(unittest.TestCase):

    def setUp(self):
        self.x = np.array(range(10))
        self.w = np.ones(10) / 10

    def testTrivial(self):
        y = np.array([1]*5 + [-1]*5)
        stump = Stump.trained(self.x, y, self.w)
        assert_approx_equal(stump.err, 0.0)
        self.assertTrue(all(stump.predict(np.array([3.5, 5.0]))
                            == np.array([1, -1])))

        y = np.array([-1]*5 + [1]*5)
        stump = Stump.trained(self.x, y, self.w)
        assert_approx_equal(stump.err, 0.0)
        self.assertTrue(all(stump.predict(np.array([3.5, 5.0]))
                            == np.array([-1, +1])))

        y = np.array([-1]*7 + [1]*3)
        stump = Stump.trained(self.x, y, self.w)
        assert_approx_equal(stump.err, 0.0)
        self.assertTrue(all(stump.predict(np.array([3.5, 5.0, 8.0]))
                            == np.array([-1, -1, +1])))

    def testEdge(self):
        y = np.array([-1]*10)
        stump = Stump.trained(self.x, y, self.w)
        assert_approx_equal(stump.err, 0.0)
        self.assertTrue(all(stump.predict(np.array([3.5, 5.0, 8.0]))
                            == np.array([-1, -1, -1])))

        y = np.array([1]*10)
        stump = Stump.trained(self.x, y, self.w)
        assert_approx_equal(stump.err, 0.0)
        self.assertTrue(all(stump.predict(np.array([3.5, 5.0, 8.0]))
                            == np.array([1, 1, 1])))

    def testWeighted(self):
        our_w = np.ones(10) / 10
        our_w[0] = 0.05
        our_w[9] = 0.15
        y = np.array([-1] + [1]*4 + [-1]*5)
        stump = Stump.trained(self.x, y, our_w)
        self.assertTrue(all(stump.predict(np.array([3.5, 5.0, 8.0]))
                            == np.array([1, -1, -1])))
        assert_approx_equal(stump.err, 0.05)


if __name__ == '__main__':
    unittest.main()
