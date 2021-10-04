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


class KDIC(ForecastModel, NonParametric):
    def __init__(self, y, t, u=None, ID='', window_size=16, num_eval_points=100):
        super().__init__(y, t, u, ID)

        # Default parameters [lambda, h_y]
        self.theta = np.array([0.9, 0.015])
        if u is not None:
            # Default parameters [lambda, h_y, h_u]
            self.theta = np.hstack([self.theta, 0.2])

        self.day_type_t = self.get_day_type(t)

        self.window_size = window_size
        self.num_eval_points = num_eval_points
        delta_y = self.y_max - self.y_min
        self.y_max += 0.05 * delta_y
        self.y_min -= 0.05 * delta_y
        self.eval_points = self.get_eval_points()

        self.pdf_y = np.zeros((num_eval_points, 0))
        self.cdf_y = np.zeros((num_eval_points, 0))
        # self.results[0]['pdf_y'] = []
        # self.results[0]['cdf_y'] = []

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
        # Mo - Fr: 0, Sa: 1, So: 2
        if isinstance(t, dt.datetime):
            return 0 if t.weekday() < 5 else t.weekday() - 4
        else:
            return np.array([0 if tstp.weekday() < 5 else tstp.weekday() - 4 for tstp in t])

    def get_previous_idx(self, t, origin_idx):
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
        if uniform_spacing:
            return np.linspace(0, 1, self.num_eval_points)
        else:
            # Get the 90th percentile of the observation distribution
            p_90 = np.nanpercentile(utils.min_max_norm(self.y, self.y_min, self.y_max), 90)

            # Spacing as in Arora, 2016
            return np.hstack((
                np.linspace(0, p_90, int(0.9 * self.num_eval_points), endpoint=False),
                np.linspace(p_90, 1, int(0.1 * self.num_eval_points))
            ))

    @staticmethod
    def boundary_correction(h_y, y, eps=0.001):
        # Reduce the bandwidth in the boundary area, in order to prevent the kernel
        # from allocating non-zero weights outside the upper and lower boundary limits
        if y < h_y:
            return np.maximum(y, eps)
        if y > (1 - h_y):
            return np.maximum(1 - y, eps)
        return h_y

    @staticmethod
    def k_pdf(delta, h=1., kernel_type='gaussian'):
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
        # Gaussian kernel cdf
        return 0.5 * (1 + erf(delta / np.sqrt(2 * h ** 2)))

    def kde(self, theta, t, u=None, y=None):
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
            crps = np.nanmean(self.crps(y, cdf_y, eval_points))
        else:
            crps = 0

        return pdf_y, cdf_y, crps

    def objective(self, theta, val_periods=4, timer=True):

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
        if super().predict(t, u):
            return
        start_time = time.time()

        pdf_y, cdf_y, _ = self.kde(self.theta, t, u)
        self.pdf_y = np.hstack([self.pdf_y, pdf_y])
        self.cdf_y = np.hstack([self.cdf_y, cdf_y])

        # self.results[0]['pdf_y'].append(pdf_y.tolist())
        # self.results[0]['cdf_y'].append(cdf_y.tolist())
        self.results[0]['prediction_time'].append(time.time() - start_time)

    def get_mean(self, t):
        super().get_mean(t)

        idx = self.idx(t)
        mean = self.mean(self.pdf_y[:, idx], self.eval_points)
        return utils.inv_min_max_norm(mean, self.y_min, self.y_max)

    def get_var(self, t):
        super().get_var(t)

        idx = self.idx(t)
        var = self.var(self.pdf_y[:, idx], self.eval_points)
        return (self.y_max - self.y_min) ** 2 * var

    def get_percentile(self, p, t):
        super().get_percentile(p, t)

        idx = self.idx(t)
        percentile = self.percentile(p, self.cdf_y[:, idx], self.eval_points)
        return utils.inv_min_max_norm(percentile, self.y_min, self.y_max)

    def get_pit(self, y_true, t):
        super().get_pit(y_true, t)

        idx = self.idx(t)
        eval_points = utils.inv_min_max_norm(self.eval_points, self.y_min, self.y_max)
        return self.cdf(y_true, self.cdf_y[:, idx], eval_points)

    def get_crps(self, y_true, t):
        super().get_crps(y_true, t)

        idx = self.idx(t)
        eval_points = utils.inv_min_max_norm(self.eval_points, self.y_min, self.y_max)
        return self.crps(y_true, self.cdf_y[:, idx], eval_points)
