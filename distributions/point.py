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
        """
        Computes the value of the CDF at x, which is given by a simple step function.
        """
        return np.array(x >= x_hat, dtype=float)

    @staticmethod
    def mean(x_hat):
        """
        Returns the point forecast x_hat.
        """
        return x_hat

    @staticmethod
    def var(x_hat):
        """
        Returns zero variance for the point forecasts.
        """
        return np.zeros_like(x_hat)

    @staticmethod
    def percentile(p, x_hat):
        """
        Returns x_hat no matter which p since all the probability mass is at the point forecast.
        """
        return x_hat

    @staticmethod
    def crps(x, x_hat):
        """
        For point forecasts the CRPS reduces to the absolute error.
        """
        return np.abs(x - x_hat)
