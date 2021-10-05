import numpy as np
import time

from models.forecast_model import ForecastModel


class LastWeek(ForecastModel):
    """
    Implements a naive benchmark that uses the observations from the previous week as a point forecast.
    """
    def __init__(self, y, t, u=None, ID=''):
        super().__init__(y, t, u, ID)

        self.y_hat = np.zeros(0)

    def __str__(self):
        return 'LastWeek'

    def fit(self):
        super().fit()

    def predict(self, t, u=None):
        """
        Predicts the observation at timestamp(s) t by taking the observations(s) from the previous week.
        """
        if super().predict(t, u):
            return
        start_time = time.time()

        idx = self.idx(t, relative=False) - self.s_w
        self.y_hat = np.hstack([self.y_hat, self.y[idx]])

        self.results[0]['prediction_time'].append(time.time() - start_time)

    def get_mean(self, t):
        super().get_mean(t)

        idx = self.idx(t)
        return self.y_hat[idx]

    def get_var(self, t):
        super().get_var(t)

        return np.zeros(len(t))

    def get_percentile(self, p, t):
        super().get_percentile(p, t)

        idx = self.idx(t)
        return self.y_hat[idx]

    def get_pit(self, y_true, t):
        super().get_pit(y_true, t)

        return 0.5 * np.ones(len(t))

    def get_crps(self, y_true, t):
        super().get_crps(y_true, t)

        return self.ae(y_true, t)

    def plot_forecast(self, y_true, t, plot_median=False, plot_percentiles=False, save_fig=False):
        super().plot_forecast(y_true, t, plot_median=False, plot_percentiles=False, save_fig=save_fig)
