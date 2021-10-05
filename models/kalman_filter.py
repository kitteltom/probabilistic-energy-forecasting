import numpy as np
from scipy.optimize import minimize
import time
import os
import json

from models.forecast_model import ForecastModel
from distributions.log_normal import LogNormal
import utils


class KalmanFilter(ForecastModel, LogNormal):
    """
    Implements the probabilistic forecasting model based on double-seasonal HWT Exponential Smoothing (Taylor, 2010)
    that estimates parameters and creates forecasts in closed form using the linear Kalman Filter (Särkkä, 2013).
    """
    def __init__(self, y, t, u=None, ID='', exp_smooth_fit=False, num_filter_weeks=52):
        super().__init__(y, t, u, ID)

        self.dim = 2 + self.s_d + self.s_w
        if u is not None:
            self.dim += u.shape[1]

        # Default parameters [alpha, delta, omega, phi, nu2]
        self.theta = np.array([0.01, 0.15, 0.15, 0.90, 1e-3])
        self.exp_smooth_fit = exp_smooth_fit
        self.filter_range = min(len(t), num_filter_weeks * self.s_w)

        # Process noise covariance matrix
        self.Q_idx = tuple(np.meshgrid(
            np.array([0, 1, 2, 2 + self.s_d]),
            np.array([0, 1, 2, 2 + self.s_d]),
            indexing='ij'
        ))
        self.Q = lambda theta: theta[4] * np.outer(
            np.array([1, theta[0], theta[1], theta[2]]),
            np.array([1, theta[0], theta[1], theta[2]])
        )
        self.q = lambda theta: np.hstack([
            1, theta[0], theta[1], np.zeros(self.s_d - 1), theta[2], np.zeros(self.dim - self.s_d - 3)
        ])

        # Measurement noise
        self.r = lambda theta: 0

        # Predicted measurement mean and variance
        self.mu_y = np.zeros(0)
        self.sigma2_y = np.zeros(0)
        # self.results[0]['mu_y'] = []
        # self.results[0]['sigma2_y'] = []

        # Mean and Variance of the prior state distribution
        self.m = self.initialize_mean(y, u)
        self.P = 1e-3 * np.eye(self.dim)

        # Negative Log Maximum Likelihood Estimate
        self.mle = 0

        self.cnt = 0
        params_path = os.path.join(self.get_out_dir(), self.results[0]["ID"] + '.json')
        if os.path.exists(params_path):
            with open(params_path, 'r') as fp:
                res = json.load(fp)
            self.theta = np.array(res['params'])
            self.results[0]['params'] = self.theta.tolist()
            _, _, self.m, self.P, self.mle = self.filter(
                self.theta,
                t[-self.filter_range:],
                u[-self.filter_range:] if u is not None else None,
                y[-self.filter_range:]
            )

    def __str__(self):
        return 'KalmanFilter'

    def fA(self, X, theta, transpose=False):
        """
        Computes the matrix-matrix (AX) or matrix-vector (Ax) multiplication of A and X efficiently. This is possible
        since A is a sparse matrix. If transpose=True XA^T is computed.
        """
        # Transition matrix
        res = np.empty_like(X)

        if X.ndim == 1:
            res[3:] = X[2:self.dim - 1]
            res[0] = theta[3] * X[0]
            res[1] = theta[0] * theta[3] * X[0] + X[1]
            res[2] = theta[1] * theta[3] * X[0] + X[1 + self.s_d]
            res[2 + self.s_d] = theta[2] * theta[3] * X[0] + X[1 + self.s_d + self.s_w]
            if self.u is not None:
                res[2 + self.s_d + self.s_w:] = X[2 + self.s_d + self.s_w:]
        elif not transpose:
            res[3:, :] = X[2:self.dim - 1, :]
            res[0, :] = theta[3] * X[0, :]
            res[1, :] = theta[0] * theta[3] * X[0, :] + X[1, :]
            res[2, :] = theta[1] * theta[3] * X[0, :] + X[1 + self.s_d, :]
            res[2 + self.s_d, :] = theta[2] * theta[3] * X[0, :] + X[1 + self.s_d + self.s_w, :]
            if self.u is not None:
                res[2 + self.s_d + self.s_w:, :] = X[2 + self.s_d + self.s_w:, :]
        else:
            res[:, 3:] = X[:, 2:self.dim - 1]
            res[:, 0] = theta[3] * X[:, 0]
            res[:, 1] = theta[0] * theta[3] * X[:, 0] + X[:, 1]
            res[:, 2] = theta[1] * theta[3] * X[:, 0] + X[:, 1 + self.s_d]
            res[:, 2 + self.s_d] = theta[2] * theta[3] * X[:, 0] + X[:, 1 + self.s_d + self.s_w]
            if self.u is not None:
                res[:, 2 + self.s_d + self.s_w:] = X[:, 2 + self.s_d + self.s_w:]

        return res

    def fh(self, X, theta, u=None):
        """
        Computes the matrix-vector multiplication (Xh) or the inner product (h^T * x) efficiently.
        """
        # Measurement model matrix
        if X.ndim == 1:
            res = (1 - theta[0] - theta[1] - theta[2]) * X[0] + X[1] + X[2] + X[2 + self.s_d]
            if u is not None:
                res += np.inner(X[2 + self.s_d + self.s_w:], u)
        else:
            res = (1 - theta[0] - theta[1] - theta[2]) * X[:, 0] + X[:, 1] + X[:, 2] + X[:, 2 + self.s_d]
            if u is not None:
                res += X[:, 2 + self.s_d + self.s_w:] @ u
        return res

    def initialize_mean(self, y, u=None):
        """
        Initializes the level, daily, weekly and weather component of the mean vector.
        """
        # Use first 3 weeks of data for initialization
        y_init = np.log(y[:3 * self.s_w])

        # Initialize the residual and the smoothed level
        e0 = 0
        l0 = np.nanmean(y_init)

        # Initialize the seasonal index for the intraday cycle
        d0 = np.nanmean(y_init.reshape(-1, self.s_d), axis=0) - l0

        # Initialize the seasonal index for the intraweek cycle
        w0 = np.nanmean(y_init.reshape(-1, self.s_w), axis=0) - np.tile(d0, int(self.s_w / self.s_d)) - l0

        # Initialize the weather regression coefficients
        if u is not None:
            b0 = -0.01 * np.log(self.y_mean) * np.ones(u.shape[1])
        else:
            b0 = np.zeros(0)

        return np.hstack((e0, l0, np.flip(d0), np.flip(w0), b0))

    def filter(self, theta, t, u=None, y=None, timer=False):
        """
        Implements the Kalman Filter prediction and update steps for the timestamps t. Additionally, the
        marginal likelihood estimate (MLE) for the parameters theta is computed in the filter recursion. The
        function returns the distribution parameter estimates mu_y and sigma^2_y for the timestamps t,
        the state vector m and covariance matrix P from the last iteration and the MLE.
        """
        start_time = time.time()

        # Initialize
        mu_y = np.zeros(len(t))
        sigma2_y = np.zeros(len(t))
        m = self.m
        P = self.P
        mle = self.mle

        # Rescale u
        if u is not None:
            u = utils.standardize(u, self.u_mean, self.u_std)

        A = lambda x: self.fA(x, theta)
        A_T = lambda x: self.fA(x, theta, transpose=True)
        Q = self.Q(theta)
        h = lambda x1, x2, x3: self.fh(x1, theta) if x2 is None else self.fh(x1, theta, x2[x3])
        r = self.r(theta)

        # Kalman filter
        for i in range(len(t)):
            # Prediction step
            m = A(m)
            P = A_T(A(P))
            P[self.Q_idx] += Q
            mu_y[i] = h(m, u, i)
            Ph = h(P, u, i)
            sigma2_y[i] = h(Ph, u, i) + r

            if y is not None and not np.isnan(y[i]):
                # Update step
                v = np.log(y[i]) - mu_y[i]
                s = max(sigma2_y[i], 1e-8)
                k = Ph / s
                m += k * v
                P -= sigma2_y[i] * k[:, np.newaxis] * k
                mle += 0.5 * np.log(2 * np.pi * s) + 0.5 * v ** 2 / s

        if timer:
            self.cnt += 1
            print(f'Iteration {self.cnt}: Time = {time.time() - start_time:.4f}s, theta = {theta}')

        return mu_y, sigma2_y, m, P, mle

    def exp_smooth(self, theta, t, u=None, y=None, timer=False):
        """
        Implements exponential smoothing for the timestamps t, which provides a fast way to estimate the parameters
        theta via MLE. However, the Kalman Filter provides better parameter estimates.
        """
        start_time = time.time()

        # Initialize
        y_hat = np.zeros(len(t))
        m = self.m

        # Rescale u
        if u is not None:
            u = utils.standardize(u, self.u_mean, self.u_std)

        A = lambda x: self.fA(x, theta)
        q = self.q(theta)
        h = lambda x1, x2, x3: self.fh(x1, theta) if x2 is None else self.fh(x1, theta, x2[x3])

        # Exponential Smoothing
        for i in range(len(t)):
            # Prediction step
            m = A(m)
            y_hat[i] = h(m, u, i)

            if y is not None and not np.isnan(y[i]):
                # Update step
                v = np.log(y[i]) - y_hat[i]
                m += q * v

        err = (np.log(y) - y_hat) ** 2
        mle = len(t) * np.log(np.nansum(err))
        eps = np.nanmean(err)

        if timer:
            self.cnt += 1
            print(f'Iteration {self.cnt}: Time = {time.time() - start_time:.4f}s, theta = {theta}')

        return mle, eps

    def fit(self):
        """
        Fits the parameters of the Kalman Filter model by minimizing the Gaussian negative log likelihood
        with the non-linear optimizer L-BFGS-B.
        """
        super().fit()
        start_time = time.time()

        # Initialize
        self.m = self.initialize_mean(self.y, self.u)
        self.P = 1e-3 * np.eye(self.dim)
        self.mle = 0

        if self.exp_smooth_fit:
            res = minimize(
                fun=lambda theta: self.exp_smooth(theta, self.t, self.u, self.y, timer=True)[0],
                x0=self.theta[:-1],
                method='L-BFGS-B',
                bounds=[(0, 1), (0, 1), (0, 1), (0, 1)],
            )
            self.theta[:-1] = res.x
            self.theta[-1] = self.exp_smooth(res.x, self.t, self.u, self.y)[1]
        else:
            res = minimize(
                fun=lambda theta: self.filter(
                    theta,
                    self.t[-self.filter_range:],
                    self.u[-self.filter_range:] if self.u is not None else None,
                    self.y[-self.filter_range:],
                    timer=True
                )[-1],
                x0=self.theta,
                method='L-BFGS-B',
                bounds=[(0, 1), (0, 1), (0, 1), (0, 1), (1e-6, 1)],
                options={'ftol': 1e-6}
            )
            self.theta = res.x

        self.results[0]['params'] = self.theta.tolist()
        print(f'{self.results[0]["ID"]} minimizer: {self.theta}')

        _, _, self.m, self.P, self.mle = self.filter(
            self.theta,
            self.t[-self.filter_range:],
            self.u[-self.filter_range:] if self.u is not None else None,
            self.y[-self.filter_range:]
        )

        self.results[0]['fit_time'] = time.time() - start_time

    def add_measurements(self, y, t, u=None):
        """
        Updates the state of the Kalman Filter after measurements are added.
        """
        super().add_measurements(y, t, u)

        _, _, self.m, self.P, self.mle = self.filter(self.theta, t, u, y)

    def predict(self, t, u=None):
        """
        Predicts the forecast distribution parameters for the timestamps t, optionally given covariates u.
        """
        if super().predict(t, u):
            return
        start_time = time.time()

        mu_y, sigma2_y, _, _, _ = self.filter(self.theta, t, u)
        self.mu_y = np.hstack([self.mu_y, mu_y])
        self.sigma2_y = np.hstack([self.sigma2_y, sigma2_y])

        # self.results[0]['mu_y'].append(mu_y.tolist())
        # self.results[0]['sigma2_y'].append(sigma2_y.tolist())
        self.results[0]['prediction_time'].append(time.time() - start_time)

    def get_mean(self, t):
        super().get_mean(t)

        idx = self.idx(t)
        return self.mean(self.mu_y[idx], self.sigma2_y[idx])

    def get_var(self, t):
        super().get_var(t)

        idx = self.idx(t)
        return self.var(self.mu_y[idx], self.sigma2_y[idx])

    def get_percentile(self, p, t):
        super().get_percentile(p, t)

        idx = self.idx(t)
        return self.percentile(p, self.mu_y[idx], self.sigma2_y[idx])

    def get_pit(self, y_true, t):
        super().get_pit(y_true, t)

        idx = self.idx(t)
        return self.cdf(y_true, self.mu_y[idx], self.sigma2_y[idx])

    def get_crps(self, y_true, t):
        super().get_crps(y_true, t)

        idx = self.idx(t)
        return self.crps(y_true, self.mu_y[idx], self.sigma2_y[idx])
