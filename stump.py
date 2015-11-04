from __future__ import division, print_function

import numpy as np


class Stump:
    """A simple implementation of a 1D decision stump."""

    def __init__(self, err=None, threshold=None, sign=None):
        self.err = err
        self.threshold = threshold
        self.sign = sign

    def train(self, x, y, w):
        """Train the stump."""

        assert len(x) == len(y) == len(w)
        assert type(x) == type(y) == type(w) == np.ndarray
        assert all(x[:-1] < x[1:])
        assert abs(np.sum(w)-1.0) < 0.01

        err = 1.0
        threshold = None

        possible_thresholds = (x + np.append(x[1::], x[-1]+1))/2
        for t_threshold in possible_thresholds:
            t_sign = +1
            t_err = 1 - np.sum(w*((y == 1) == (x > t_threshold)))
            if t_err > 0.5:
                t_sign = -1
                t_err = 1.0 - t_err
            if t_err < err:
                err = t_err
                threshold = t_threshold
                sign = t_sign

        self.err = err
        self.threshold = threshold
        self.sign = sign

    @classmethod
    def trained(cls, x, y, w):
        """Create a trained stump."""
        stump = Stump()
        stump.train(x, y, w)
        return stump

    def predict(self, x):
        """Predict y for the given x."""
        assert type(x) == np.ndarray
        predictions = 1 * (self.sign * x > self.sign * self.threshold)
        predictions[predictions == 0] = -1
        return predictions
