import time

from models.forecast_model import ForecastModel
from distributions.point import Point
import utils


class LastWeek(ForecastModel):
    """
    Implements a naive benchmark that uses the observations from the previous week as a point forecast.
    """
    def __init__(self, y, t, u=None, ID=''):
        super().__init__(y, t, u, ID, distribution=Point)

    def __str__(self):
        return 'LastWeek'

    def fit(self):
        super().fit()

    def predict(self, t, u=None):
        """
        Predicts the observation at timestamp(s) t by taking the observation(s) from the previous week.
        """
        if super().predict(t, u):
            return
        start_time = time.time()

        last_week_idx = self.idx(t, relative=False) - self.s_w
        y_hat = utils.interpolate_nans(self.y[last_week_idx])
        self.predictions[(t[0], t[-1])] = [y_hat]

        self.results[0]['prediction_time'].append(time.time() - start_time)

    def plot_forecast(self, y_true, t, plot_median=False, plot_percentiles=False, save_fig=False):
        super().plot_forecast(y_true, t, plot_median=False, plot_percentiles=False, save_fig=save_fig)
