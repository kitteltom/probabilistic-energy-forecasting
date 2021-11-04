import numpy as np

from distributions.distribution import Distribution


class Point(Distribution):
    """
    Provides functions for point forecasts, i.e. no distribution.
    """

    @staticmethod
    def pdf(x, x_hat):
        pass

    @staticmethod
    def cdf(x, x_hat):
        return np.array(x_hat >= x, dtype=float)

    @staticmethod
    def mean(x_hat):
        return x_hat

    @staticmethod
    def var(x_hat):
        return np.zeros_like(x_hat)

    @staticmethod
    def percentile(p, x_hat):
        return x_hat

    @staticmethod
    def crps(x, x_hat):
        return np.abs(x - x_hat)
