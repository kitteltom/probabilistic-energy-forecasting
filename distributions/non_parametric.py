import numpy as np

from distributions.distribution import Distribution


class NonParametric(Distribution):
    """
    Provides functions for a non-parametric forecast distribution.
    """

    @staticmethod
    def pdf(x, pdf_x, x_eval):
        pass

    @staticmethod
    def cdf(x, cdf_x, x_eval):
        """
        Computes the CDF of the non-parametric distribution at x given the CDF at evaluation points,
        by linear interpolation.
        """

        # Linear interpolation
        insertion_points = np.searchsorted(x_eval, x)
        r = np.minimum(insertion_points, len(x_eval) - 1)
        l = np.maximum(0, insertion_points - 1)
        idx = np.arange(len(x))
        slope = (cdf_x[r, idx] - cdf_x[l, idx]) / np.maximum(x_eval[r] - x_eval[l], 1e-6)
        return cdf_x[l, idx] + slope * (x - x_eval[l])

    @staticmethod
    def mean(pdf_x, x_eval):
        """
        Computes the mean of the non-parametric distribution by integrating the PDF at evaluation points,
        using the trapezoidal rule.
        """
        return np.trapz(
            y=x_eval[:, np.newaxis] * pdf_x,
            x=x_eval[:, np.newaxis],
            axis=0
        )

    @staticmethod
    def var(pdf_x, x_eval):
        """
        Computes the variance of the non-parametric distribution by integrating the PDF at evaluation points,
        using the trapezoidal rule.
        """
        return np.trapz(
            y=x_eval[:, np.newaxis] ** 2 * pdf_x,
            x=x_eval[:, np.newaxis],
            axis=0
        ) - np.trapz(
            y=x_eval[:, np.newaxis] * pdf_x,
            x=x_eval[:, np.newaxis],
            axis=0
        ) ** 2

    @staticmethod
    def percentile(p, cdf_x, x_eval):
        """
        Computes the p-percentile of the non-parametric distribution given the CDF at evaluation points,
        by linear interpolation.
        """

        # Linear interpolation
        insertion_points = []
        for i in range(cdf_x.shape[1]):
            insertion_points.append(np.searchsorted(cdf_x[:, i], p / 100))
        insertion_points = np.array(insertion_points)
        r = np.minimum(insertion_points, len(cdf_x) - 1)
        l = np.maximum(0, insertion_points - 1)
        idx = np.arange(cdf_x.shape[1])
        slope = (x_eval[r] - x_eval[l]) / np.maximum(cdf_x[r, idx] - cdf_x[l, idx], 1e-6)
        return x_eval[l] + slope * (p / 100 - cdf_x[l, idx])

    @staticmethod
    def crps(x, cdf_x, x_eval):
        """
        Computes the Continuous Ranked Probability Score (CRPS) of the non-parametric distribution with true value x,
        using the trapezoidal rule.
        """
        return np.trapz(
            y=(cdf_x - (x_eval[:, np.newaxis] >= x[np.newaxis, :])) ** 2,
            x=x_eval[:, np.newaxis],
            axis=0
        )
