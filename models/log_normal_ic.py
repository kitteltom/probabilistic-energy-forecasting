import numpy as np
from scipy.optimize import minimize
import datetime as dt
import time
import os
import json

from models.forecast_model import ForecastModel
from distributions.log_normal import LogNormal
import utils


class LogNormalIC(ForecastModel):
    """
    Implements a parametric variant of the non-parametric probabilistic forecasting model called
    KD-IC (Arora et. al., 2016). The forecasts distributions are log-normal.
    """
    def __init__(self, y, t, u=None, ID='', window_size=16):
        super().__init__(y, t, u, ID, distribution=LogNormal)

        # Default parameters [lambda]
        self.theta = np.array([0.9])
        if u is not None:
            # Default parameters [lambda, h_u]
            self.theta = np.hstack([self.theta, 0.2])

        self.day_type_t = self.get_day_type(t)

        self.window_size = window_size

        self.cnt = 0
        params_path = os.path.join(self.get_out_dir(), self.results[0]["ID"] + '.json')
        if os.path.exists(params_path):
            with open(params_path, 'r') as fp:
                res = json.load(fp)
            self.theta = np.array(res['params'])
        self.results[0]['params'] = self.theta.tolist()

    def __str__(self):
        return 'LogNormal-IC'

    @staticmethod
    def get_day_type(t):
        """
        Returns the day-type of the timestamp(s) t, which is 0 for the weekdays (Mo-Fr), 1 for Saturday and
        2 for Sunday.
        """
        if isinstance(t, dt.datetime):
            return 0 if t.weekday() < 5 else t.weekday() - 4
        else:
            return np.array([0 if tstp.weekday() < 5 else tstp.weekday() - 4 for tstp in t])

    def get_previous_idx(self, t, origin_idx):
        """
        Returns all relevant previous indices needed for the forecast of timestamps t, together with a mask,
        indicating which days correspond to the same day-type.
        """
        idx = self.idx(t, relative=False)[:, np.newaxis]
        dist_to_origin = idx - origin_idx

        # Round distance to full days
        first_idx = (dist_to_origin // self.s_d + 1) * self.s_d

        # Get all relevant previous indices
        prev_idx = idx - (first_idx + np.arange(0, self.s_w * self.window_size, self.s_d)[np.newaxis])

        # Mask the days
        day_mask = self.day_type_t[prev_idx] == self.get_day_type(t)[:, np.newaxis]

        return prev_idx, day_mask

    @staticmethod
    def k(delta, h=1., kernel_type='triangular'):
        """
        Implements the PDFs for various kernels with bandwidth h.
        """
        if kernel_type == 'gaussian':
            # Gaussian kernel
            return 1 / np.sqrt(2 * np.pi * h ** 2) * np.exp(-0.5 * delta ** 2 / h ** 2)
        elif kernel_type == 'parabolic':
            # Parabolic kernel
            return 3 / (4 * h) * np.maximum(1 - (delta / h) ** 2, 1e-6)
        else:
            # Triangular kernel
            return 1 / h * np.maximum(1 - np.abs(delta / h), 1e-6)

    def max_likelihood_prediction(self, theta, t, u=None, y=None):
        """
        Computes the distribution parameters at the timestamps t, by maximizing the likelihood using
        the Log-Normal-IC method. Optionally, the computation of the maximum likelihood is conditioned
        on the input u. The function returns the marginal likelihood estimate (MLE) for the timestamps t
        if the true observations y are known, which can be used for estimation of the parameters theta.
        """
        mu_y = np.zeros(len(t))
        sigma2_y = np.zeros(len(t))
        mle = 0

        origin_idx = self.idx(t[0], relative=False)
        prev_idx, day_mask = self.get_previous_idx(t, origin_idx)

        for i in range(len(t)):
            idx = prev_idx[i, day_mask[i, :]]

            log_y = np.log(self.y[idx])

            decay = theta[0] ** (np.floor((origin_idx - idx - 1) / self.s_w))
            temp_weight = 1
            if u is not None:
                delta = (u[i, 0] - self.u[idx, 0]) / (self.u_max[0, 0] - self.u_min[0, 0])
                temp_weight = self.k(delta, h=theta[1], kernel_type='triangular')

            mu_y[i] = utils.weighted_nanmean(log_y, decay * temp_weight)
            sigma2_y[i] = utils.weighted_nanvar(log_y, decay * temp_weight)

            if y is not None and not np.isnan(y[i]):
                v = np.log(y[i]) - mu_y[i]
                s = max(sigma2_y[i], 1e-8)
                mle += 0.5 * np.log(2 * np.pi * s) + 0.5 * v ** 2 / s

        return mu_y, sigma2_y, mle

    def objective(self, theta, val_periods=4, timer=True):
        """
        Defines the objective which can be used for parameter optimization. Here, the objective is to minimize
        the MLE in the four weeks prior to the forecast range.
        """
        start_time = time.time()
        mle = 0

        for i in range(val_periods):
            idx = np.flip(len(self.t) - 1 - np.arange(i * self.s_w, (i + 1) * self.s_w))
            if self.u is not None:
                mle += self.max_likelihood_prediction(theta, self.t[idx], self.u[idx], self.y[idx])[2]
            else:
                mle += self.max_likelihood_prediction(theta, self.t[idx], y=self.y[idx])[2]

        if timer:
            self.cnt += 1
            print(f'Iteration {self.cnt}: Time = {time.time() - start_time:.4f}s, theta = {theta}')

        return mle

    def fit(self):
        """
        Fit the parameters of the Log-Normal-IC model by minimizing the objective via the non-linear
        optimizer L-BFGS-B.
        """
        super().fit()
        start_time = time.time()

        bounds = [(0.1, 1)]
        if self.u is not None:
            bounds = [(0.1, 1), (1e-3, 1)]

        res = minimize(
            fun=self.objective,
            x0=self.theta,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-6}
        )
        self.theta = res.x

        self.results[0]['params'] = self.theta.tolist()
        print(f'{self.results[0]["ID"]} minimizer: {self.theta}')

        self.results[0]['fit_time'] = time.time() - start_time

    def add_measurements(self, y, t, u=None):
        super().add_measurements(y, t, u)

        self.day_type_t = np.hstack([self.day_type_t, self.get_day_type(t)])

    def predict(self, t, u=None):
        """
        Predicts the forecast distribution parameters for the timestamps t,
        optionally given covariates u.
        """
        if super().predict(t, u):
            return
        start_time = time.time()

        mu_y, sigma2_y, _ = self.max_likelihood_prediction(self.theta, t, u)
        self.predictions[(t[0], t[-1])] = [mu_y, sigma2_y]

        self.results[0]['prediction_time'].append(time.time() - start_time)
