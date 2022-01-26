import numpy as np
from scipy.linalg import cho_factor, cho_solve
import datetime as dt
import time

from models.forecast_model import ForecastModel
from distributions.log_normal import LogNormal
import utils


class LastWeekRegression(ForecastModel):
    """
    Implements a simple linear regression forecast model with various optional feature vectors.
    """
    def __init__(self, y, t, u=None, ID=''):
        super().__init__(y, t, u, ID, distribution=LogNormal)

        self.use_last_week_feature = True
        self.use_weekday_bias = True
        self.num_seasonal_terms = 3
        self.weather_polynomial_degree = 2

        self.num_features = (
            1 if self.use_last_week_feature else 0
        ) + (
            7 if self.use_weekday_bias else 1
        ) + (
            2 * self.num_seasonal_terms
        ) + (
            (self.weather_polynomial_degree * u.shape[1]) if u is not None else 0
        )

        self.lam = 1e-3
        self.coefficients = np.zeros((
            self.s_d,
            self.num_features
        ))
        self.sigma2_y = np.zeros(self.s_d)

    def __str__(self):
        return 'LastWeekRegression'

    @staticmethod
    def get_weekday_mask(t, weekday):
        """
        Returns an array that masks the weekday in t.
        """
        return np.array([1.0 if tstp.weekday() == weekday else 0.0 for tstp in t])

    def phi(self, y_lw, t, u=None):
        """
        Returns the feature vector.
        """
        features = np.zeros((len(y_lw), self.num_features))
        feature_count = 0

        # Last weeks observation
        if self.use_last_week_feature:
            features[:, feature_count] = np.log(y_lw)
            feature_count += 1

        # Bias
        if self.use_weekday_bias:
            for weekday in range(7):
                features[:, feature_count] = self.get_weekday_mask(t, weekday)
                feature_count += 1
        else:
            features[:, feature_count] = 1
            feature_count += 1

        # Seasonal features
        seconds = t.map(dt.datetime.timestamp).to_numpy(float)
        seconds_per_minute = 60
        minutes_per_hour = 60
        hours_per_day = 24
        days_per_year = 365.2425
        seconds_per_year = seconds_per_minute * minutes_per_hour * hours_per_day * days_per_year
        for seasonal_factor in range(1, self.num_seasonal_terms + 1):
            features[:, feature_count] = np.sin(2 * np.pi * seasonal_factor * seconds / seconds_per_year)
            feature_count += 1
            features[:, feature_count] = np.cos(2 * np.pi * seasonal_factor * seconds / seconds_per_year)
            feature_count += 1

        # Weather polynomials
        if u is not None:
            u = utils.standardize(u, self.u_mean, self.u_std)
            for weather_var in u.T:
                start_idx = feature_count
                end_idx = feature_count + self.weather_polynomial_degree
                features[:, start_idx:end_idx] = np.power(
                    weather_var[:, np.newaxis],
                    np.arange(self.weather_polynomial_degree) + 1
                )
                feature_count += self.weather_polynomial_degree

        return features

    def lin_reg(self, Y, y_lw, t, u=None):
        """
        Return the linear regression coefficients for the features X, with labels Y.
        """
        X = self.phi(y_lw, t, u)
        I = np.eye(self.num_features)
        coefficients = cho_solve(cho_factor(X.T @ X + self.lam * I), X.T @ np.log(Y))
        sigma2_y = np.mean((np.log(Y) - X @ coefficients)**2)
        return coefficients, sigma2_y

    def fit(self):
        """
        Fits the linear regression coefficients for each hour of the day.
        """
        super().fit()
        start_time = time.time()

        y = utils.interpolate_nans(self.y)
        y_labels = y[self.s_w:]
        y_lw = y[:-self.s_w]
        t = self.t[self.s_w:]
        u = None
        if self.u is not None:
            u = self.u[self.s_w:]

        for i, t0 in enumerate(self.t[:self.s_d]):
            hh = utils.half_hour(t0)

            self.coefficients[hh], self.sigma2_y[hh] = self.lin_reg(
                y_labels[i::self.s_d],
                y_lw[i::self.s_d],
                t[i::self.s_d],
                u[i::self.s_d] if self.u is not None else None
            )

        self.results[0]['params'] = self.coefficients.tolist()
        self.results[0]['fit_time'] = time.time() - start_time

    def predict(self, t, u=None):
        """
        Predicts the observation at timestamp(s) t by linear regression.
        """
        if super().predict(t, u):
            return
        start_time = time.time()

        mu_y = np.zeros(len(t))
        sigma2_y = np.zeros(len(t))

        last_week_idx = self.idx(t, relative=False) - self.s_w
        y_lw = utils.interpolate_nans(self.y[last_week_idx])

        for i, t0 in enumerate(t[:self.s_d]):
            hh = utils.half_hour(t0)

            mu_y[i::self.s_d] = self.phi(
                y_lw[i::self.s_d],
                t[i::self.s_d],
                u[i::self.s_d] if u is not None else None
            ) @ self.coefficients[hh]
            sigma2_y[i::self.s_d] = self.sigma2_y[hh]

        self.predictions[(t[0], t[-1])] = [mu_y, sigma2_y]

        self.results[0]['prediction_time'].append(time.time() - start_time)
