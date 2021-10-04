import numpy as np


def half_hour(t):
    """
    Computes the half-hour of timestamp t as integer between 0 and 47.
    """
    return 2 * t.hour + t.minute // 30


def standardize(x, mean, std):
    """
    Standardizes the input x by subtracting the mean and dividing by the standard deviation.
    """
    return (x - mean) / np.maximum(1e-6, std)


def min_max_norm(x, min_x, max_x):
    """
    Computes the min-max-norm of input x with minimum min_x and maximum max_x.
    """
    return (x - min_x) / np.maximum(1e-6, (max_x - min_x))


def inv_min_max_norm(x, min_x, max_x):
    """
    Computes the inverse min-max-norm of input x with minimum min_x and maximum max_x.
    """
    return x * (max_x - min_x) + min_x


def running_nanmean(x, window_size=48):
    """
    Computes a running mean over the time series x with a window size of window_size.
    Nans are ignored for the calculation of the mean.
    """
    half_ws = window_size // 2
    N = len(x)
    x_hat = np.zeros_like(x)
    for i in range(N):
        x_hat[i] = np.nanmean(x[max(0, i - half_ws):min(N, i + half_ws + 1)])
    return x_hat


def interpolate_nans(x):
    """
    Linearly interpolates nans in the time series x and returns a time series without nans.
    """
    if x.ndim == 1:
        nans = np.isnan(x)
        x[nans] = np.interp(np.where(nans)[0], np.where(~nans)[0], x[~nans])
        return x
    else:
        for i in range(x.shape[1]):
            nans = np.isnan(x[:, i])
            x[nans, i] = np.interp(np.where(nans)[0], np.where(~nans)[0], x[~nans, i])
        return x


def weighted_nanmean(x, weights, axis=None):
    """
    Computes a weighted mean over the time series x with each entry weighted by weights.
    Nans are ignored for the calculation of the mean.
    """
    weighted_mean = np.nansum(weights * x, axis=axis)
    weighted_mean /= np.sum(weights * ~np.isnan(x), axis=axis)
    return weighted_mean


def weighted_nanvar(x, weights):
    """
    Computes a weighted variance over the time series x with each entry weighted by weights.
    Nans are ignored for the calculation of the variance.
    """
    weighted_var = np.nansum(weights * (x - weighted_nanmean(x, weights))**2)
    weighted_var /= np.sum(weights * ~np.isnan(x))
    return weighted_var


def round_floats(o):
    """
    Rounds the floating point o to 6 decimal points.
    """
    if isinstance(o, float):
        return round(o, 6)
    if isinstance(o, dict):
        return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round_floats(x) for x in o]
    return o
