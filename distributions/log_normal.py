import numpy as np
from scipy.special import erf, erfinv

from distributions.distribution import Distribution


class LogNormal(Distribution):
    """
    Provides functions for the log-normal forecast distribution.
    """

    @staticmethod
    def pdf(x, mu, sigma2):
        """
        Computes the value of the PDF at x for the log-normal distribution given by mu and sigma^2.
        """
        return 1 / (x * np.sqrt(sigma2 * 2 * np.pi)) * np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma2))

    @staticmethod
    def cdf(x, mu, sigma2):
        """
        Computes the value of the CDF at x for the log-normal distribution given by mu and sigma^2.
        """
        return 0.5 + 0.5 * erf((np.log(x) - mu) / np.sqrt(2 * sigma2))

    @staticmethod
    def mean(mu, sigma2):
        """
        Computes the mean of the log-normal distribution given by mu and sigma^2.
        """
        return np.exp(mu + sigma2 / 2)

    @staticmethod
    def var(mu, sigma2):
        """
        Computes the variance of the log-normal distribution given by mu and sigma^2.
        """
        return (np.exp(sigma2) - 1) * np.exp(2 * mu + sigma2)

    @staticmethod
    def mu(mean, var):
        """
        Computes distribution parameter mu for a log-normal distribution with given empirical mean and variance.
        """
        return np.log(mean ** 2 / np.sqrt(var + mean ** 2))

    @staticmethod
    def sigma2(mean, var):
        """
        Computes distribution parameter sigma^2 for a log-normal distribution with given empirical mean and variance.
        """
        return np.log(var / mean ** 2 + 1)

    @staticmethod
    def percentile(p, mu, sigma2):
        """
        Computes the p-percentile of the log-normal distribution given by mu and sigma^2.
        """
        return np.exp(mu + np.sqrt(2 * sigma2) * erfinv(2 * p / 100 - 1))

    @staticmethod
    def crps(y, mu, sigma2, tol=1e-6, num_samples=1000):
        """
        Computes the Continuous Ranked Probability Score (CRPS) for the log-normal distribution given by mu and sigma^2,
        with true value y.
        """

        # CRPS for Log-normal analytically (Jordan, 2019)
        crps = LogNormal.cdf(y, mu + sigma2, sigma2) + LogNormal.cdf(np.exp(np.sqrt(sigma2 / 2)), 0, 1) - 1
        crps *= -2 * np.exp(mu + sigma2 / 2)
        crps += y * (2 * LogNormal.cdf(y, mu, sigma2) - 1)

        return crps
