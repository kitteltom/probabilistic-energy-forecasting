import numpy as np

from distributions.distribution import Distribution


class Empirical(Distribution):
    """
    Provides functions for empirical (sampled) forecast distributions.
    """

    @staticmethod
    def pdf(x, samples):
        pass

    @staticmethod
    def cdf(x, samples):
        """
        Computes the empirical CDF of x for the distribution given by the samples.
        """
        return np.mean(samples <= x, axis=0)

    @staticmethod
    def mean(samples):
        """
        Computes the sample mean.
        """
        return np.mean(samples, axis=0)

    @staticmethod
    def var(samples):
        """
        Computes the sample variance.
        """
        return np.mean((samples - Empirical.mean(samples)[np.newaxis]) ** 2, axis=0)

    @staticmethod
    def percentile(p, samples):
        """
        Computes the p-percentile of the distribution given by the samples.
        """
        # Linear interpolation of the modes for the order statistics for the uniform distribution on [0,1]
        samples = np.sort(samples, axis=0)
        N = len(samples)
        h = (N - 1) * p / 100
        h_floor = int(np.floor(h))
        h_ceil = int(np.ceil(h))
        return samples[h_floor] + (h - h_floor) * (samples[h_ceil] - samples[h_floor])

    @staticmethod
    def crps(x, samples):
        """
        Computes the empirical Continuous Ranked Probability Score (CRPS) of the distribution given by the samples,
        with true value x.
        """
        # Calculates CRPS_F(x) = E_F|X - x| - 1/2 * E_F|X - X.T|, where X corresponds to the forecast samples
        crps = np.mean(np.abs(samples - x[np.newaxis]), axis=0)
        crps -= 0.5 * np.mean(np.abs(samples[:, np.newaxis] - samples[np.newaxis]), axis=(0, 1))

        return crps
