import numpy as np
from scipy.special import erf
from scipy.optimize import minimize
import datetime as dt
import time
import os
import json

from models.forecast_model import ForecastModel
from distributions.non_parametric import NonParametric
import utils


class KDIC(ForecastModel):
    """
    Implements the non-parametric probabilistic forecasting model called KD-IC (Arora et. al., 2016).
    """
    def __init__(self, y, t, u=None, ID='', window_size=16, num_eval_points=100):
        super().__init__(y, t, u, ID, distribution=NonParametric)

        # Default parameters [lambda, h_y]
        self.theta = np.array([0.9, 0.015])
        if u is not None:
            # Default parameters [lambda, h_y, h_u]
            self.theta = np.hstack([self.theta, 0.2])

        self.day_type_t = self.get_day_type(t)

        self.window_size = window_size
        self.num_eval_points = num_eval_points
        delta_y = self.y_max - self.y_min
        self.y_max += 0.1 * delta_y
        self.y_min -= 0.1 * delta_y
        self.eval_points = self.get_eval_points()

        self.cnt = 0
        params_path = os.path.join(self.get_out_dir(), self.results[0]["ID"] + '.json')
        if os.path.exists(params_path):
            with open(params_path, 'r') as fp:
                res = json.load(fp)
            self.theta = np.array(res['params'])
        self.results[0]['params'] = self.theta.tolist()

    def __str__(self):
        return 'KD-IC'

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

    def get_eval_points(self, uniform_spacing=True):
        """
        Returns the evaluation points at which a density estimate should be computed. If uniform_spacing=True,
        the evaluation points are spaced uniformly. Otherwise evaluation points are spaced as in Arora et. al. (2016).
        """
        if uniform_spacing:
            return np.linspace(0, 1, self.num_eval_points)
        else:
            # Get the 90th percentile of the observation distribution
            p_90 = np.nanpercentile(utils.min_max_norm(self.y, self.y_min, self.y_max), 90)

            # Spacing as in Arora et. al. (2016)
            return np.hstack((
                np.linspace(0, p_90, int(0.9 * self.num_eval_points), endpoint=False),
                np.linspace(p_90, 1, int(0.1 * self.num_eval_points))
            ))

    @staticmethod
    def boundary_correction(h_y, y, eps=0.001):
        """
        Implements boundary correction.
        """
        # Reduce the bandwidth in the boundary area, in order to prevent the kernel
        # from allocating non-zero weights outside the upper and lower boundary limits
        if y < h_y:
            return np.maximum(y, eps)
        if y > (1 - h_y):
            return np.maximum(1 - y, eps)
        return h_y

    @staticmethod
    def k_pdf(delta, h=1., kernel_type='gaussian'):
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

    @staticmethod
    def k_cdf(delta, h=1., kernel_type='gaussian'):
        """
        Implements the CDF for the Gaussian kernel with bandwidth h.
        """
        # Gaussian kernel cdf
        return 0.5 * (1 + erf(delta / np.sqrt(2 * h ** 2)))

    def kde(self, theta, t, u=None, y=None):
        """
        Computes kernel density estimates (PDF and CDF) at the timestamps t, by using the KD-IC method.
        Optionally, the estimates are conditioned on the input u. The function returns the CRPS for the
        timestamps t if the true observations y are known, which can be used for estimation of the parameters theta.
        """
        pdf_y = np.zeros((self.num_eval_points, len(t)))
        cdf_y = np.zeros((self.num_eval_points, len(t)))

        origin_idx = self.idx(t[0], relative=False)
        prev_idx, day_mask = self.get_previous_idx(t, origin_idx)
        h_y = np.zeros(self.num_eval_points)
        for i, eval_point in enumerate(self.eval_points):
            h_y[i] = self.boundary_correction(theta[1], eval_point)

        for i in range(len(t)):
            idx = prev_idx[i, day_mask[i, :]]

            y_idx = utils.min_max_norm(self.y[idx], self.y_min, self.y_max)

            decay = (theta[0] ** (np.floor((origin_idx - idx - 1) / self.s_w)))[np.newaxis, :]
            temp_weight = 1
            if u is not None:
                delta = (u[i, 0] - self.u[idx, 0]) / (self.u_max[0, 0] - self.u_min[0, 0])
                temp_weight = self.k_pdf(delta, h=theta[2], kernel_type='triangular')[np.newaxis, :]

            k_pdf = self.k_pdf(self.eval_points[:, np.newaxis] - y_idx[np.newaxis, :], h=h_y[:, np.newaxis])
            pdf_y[:, i] = utils.weighted_nanmean(k_pdf, decay * temp_weight, axis=1)
            k_cdf = self.k_cdf(self.eval_points[:, np.newaxis] - y_idx[np.newaxis, :], h=h_y[:, np.newaxis])
            cdf_y[:, i] = utils.weighted_nanmean(k_cdf, decay * temp_weight, axis=1)

        if y is not None:
            eval_points = utils.inv_min_max_norm(self.eval_points, self.y_min, self.y_max)
            crps = np.nanmean(self.distribution.crps(y, cdf_y, eval_points))
        else:
            crps = 0

        return pdf_y, cdf_y, crps

    def objective(self, theta, val_periods=4, timer=True):
        """
        Defines the objective which can be used for parameter optimization. Here, the objective is to minimize
        the CRPS in the four weeks prior to the forecast range.
        """
        start_time = time.time()
        crps = 0

        for i in range(val_periods):
            idx = np.flip(len(self.t) - 1 - np.arange(i * self.s_w, (i + 1) * self.s_w))
            if self.u is not None:
                crps += self.kde(theta, self.t[idx], self.u[idx], self.y[idx])[2]
            else:
                crps += self.kde(theta, self.t[idx], y=self.y[idx])[2]

        if timer:
            self.cnt += 1
            print(f'Iteration {self.cnt}: Time = {time.time() - start_time:.4f}s, theta = {theta}')

        return crps / val_periods

    def fit(self):
        """
        Fit the parameters of the KD-IC model by minimizing the objective via the non-linear optimizer L-BFGS-B.
        """
        super().fit()
        start_time = time.time()

        bounds = [(0.1, 1), (1e-3, 1)]
        if self.u is not None:
            bounds = [(0.1, 1), (1e-3, 1), (1e-3, 1)]

        res = minimize(
            fun=self.objective,
            x0=self.theta,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-4}
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
        Predicts the forecast distribution at the evaluation points for the timestamps t,
        optionally given covariates u.
        """
        if super().predict(t, u):
            return
        start_time = time.time()

        pdf_y, cdf_y, _ = self.kde(self.theta, t, u)
        self.predictions[(t[0], t[-1])] = [pdf_y, cdf_y]

        self.results[0]['prediction_time'].append(time.time() - start_time)

    def mean(self, t):
        """
        Returns the mean forecasts for the timestamps t.
        """
        self.validate_timestamps(t)

        pdf_y = self.predictions[(t[0], t[-1])][0]
        mean = self.distribution.mean(pdf_y, self.eval_points)
        return utils.inv_min_max_norm(mean, self.y_min, self.y_max)

    def var(self, t):
        """
        Returns the variance forecasts for the timestamps t.
        """
        self.validate_timestamps(t)

        pdf_y = self.predictions[(t[0], t[-1])][0]
        var = self.distribution.var(pdf_y, self.eval_points)
        return (self.y_max - self.y_min) ** 2 * var

    def percentile(self, p, t):
        """
        Returns the p-percentile forecasts for the timestamps t.
        """
        self.validate_timestamps(t)

        cdf_y = self.predictions[(t[0], t[-1])][1]
        percentile = self.distribution.percentile(p, cdf_y, self.eval_points)
        return utils.inv_min_max_norm(percentile, self.y_min, self.y_max)

    def pit(self, y_true, t):
        """
        Returns the Probability Integral Transform (PIT) for the timestamps t,
        given the true observations y_true.
        """
        self.validate_timestamps(t)

        cdf_y = self.predictions[(t[0], t[-1])][1]
        eval_points = utils.inv_min_max_norm(self.eval_points, self.y_min, self.y_max)
        return self.distribution.cdf(y_true, cdf_y, eval_points)

    def crps(self, y_true, t):
        """
        Returns the Continuous Ranked Probability Score (CRPS) for the timestamps t,
        given the true observations y_true.
        """
        self.validate_timestamps(t)

        cdf_y = self.predictions[(t[0], t[-1])][1]
        eval_points = utils.inv_min_max_norm(self.eval_points, self.y_min, self.y_max)
        return self.distribution.crps(y_true, cdf_y, eval_points)
