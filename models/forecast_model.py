from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os
import json
import copy

import utils

OUT_PATH = './out/'


class ForecastModel(ABC):
    """
    Abstract superclass of all probabilistic forecasting models.
    """

    @abstractmethod
    def __init__(self, y, t, u=None, ID='', distribution=None, seed=0, global_model=False):
        self.distribution = distribution
        self.seed = seed
        self.global_model = global_model
        self.s_d = 48
        self.s_w = self.s_d * 7

        # Maximum forecast horizon
        self.max_horizon = self.s_w

        self.y = y
        self.t = t
        if u is not None and u.ndim == 1:
            u = u[:, np.newaxis]
        self.u = u

        # Mean, maximum and minimum value of the measurements
        self.y_mean = np.nanmean(y, axis=0)
        self.y_max = np.nanmax(y, axis=0)
        self.y_min = np.nanmin(y, axis=0)

        # Maximum, minimum, mean and std value of the input
        if u is not None:
            self.u_max = np.nanmax(u, axis=0, keepdims=True)
            self.u_min = np.nanmin(u, axis=0, keepdims=True)

            self.u_mean = np.nanmean(u, axis=0, keepdims=True)
            self.u_std = np.nanstd(u, axis=0, keepdims=True)

        # Timestamp of first forecast
        self.t_f = t[-1] + dt.timedelta(minutes=30)

        # Dictionary of predictions
        self.predictions = {}

        # Dictionary of forecast results
        results_dict = {
            't0': [],
            'fit_time': 0,
            'prediction_time': [],
            'mean': [],
            'var': [],
            'p05': [],
            'p25': [],
            'p50': [],
            'p75': [],
            'p95': [],
            'PIT': [],
            'CRPS': [],
            'mCRPS': [],
            'rCRPS': [],
            # 'AE': [],
            'MAE': [],
            'MASE': [],
            'MAPE': [],
            'rMAE': [],
            # 'SE': [],
            'RMSE': [],
            'rRMSE': [],
        }

        self.results = []
        if self.global_model:
            self.n = y.shape[1]
            for i, ts_ID in enumerate(ID):
                self.results.append(copy.deepcopy(results_dict))
                self.results[i]['ID'] = f'{self}_{ts_ID}'
        else:
            self.n = 1
            self.results.append(copy.deepcopy(results_dict))
            self.results[0]['ID'] = f'{self}_{ID}'

    @abstractmethod
    def __str__(self):
        pass

    def idx(self, t, relative=True):
        """
        Returns the index/indices of the timestamp(s) t. Either with respect to the first forecast timestamp
        (relative=True) or with respect to the timestamp t_0 (relative=False).
        """
        if isinstance(t, dt.datetime):
            return int((t - (self.t_f if relative else self.t[0])).total_seconds() / (60 * 30))
        else:
            return np.array([
                int((hh - (self.t_f if relative else self.t[0])).total_seconds() / (60 * 30)) for hh in t
            ])

    @abstractmethod
    def fit(self):
        """
        Abstract method (to be implemented by subclasses) that fits the model parameters to the data.
        """
        pass

    def validate_input(self, u):
        """
        Validates whether the input u is consistently missing or consistently not missing.
        """
        if self.u is None and u is not None:
            raise ValueError('No initial input u available.')
        if self.u is not None and u is None:
            raise ValueError('Missing input u.')

    def add_measurements(self, y, t, u=None):
        """
        Appends measurements y (and optionally covariates u) for the timestamps t. Note that the timestamps t
        must be subsequent to the timestamps self.t.
        """
        if t[0] != self.t[-1] + dt.timedelta(minutes=30):
            raise ValueError('No subsequent measurements.')

        if self.global_model:
            self.y = np.vstack([self.y, y])
        else:
            self.y = np.hstack([self.y, y])
        self.t = self.t.union(t)
        self.validate_input(u)
        if u is not None:
            if u.ndim == 1:
                u = u[:, np.newaxis]
            self.u = np.vstack([self.u, u])

    def predictions_available(self, t):
        """
        Checks if predictions are already available for the timestamps t.
        """
        return (t[0], t[-1]) in self.predictions

    @abstractmethod
    def predict(self, t, u=None):
        """
        Abstract method (to be implemented by subclasses) that predicts the distribution of observations y for
        the timestamps t, optionally given covariates u. Note that the prediction has to start right after the
        last measurement.
        """
        if self.predictions_available(t):
            return True

        if t[0] != self.t[-1] + dt.timedelta(minutes=30):
            raise ValueError('Prediction has to start right after last measurement.')
        if len(t) > self.max_horizon:
            raise ValueError(f'The maximum forecast horizon is {self.max_horizon} half-hours.')
        self.validate_input(u)

        # Execute prediction
        return False

    def validate_timestamps(self, t):
        """
        Validates whether predictions are available for the timestamps t.
        """
        if not self.predictions_available(t):
            raise ValueError('No prediction available for the timestamps t.')

    def mean(self, t):
        """
        Returns the mean forecasts for the timestamps t.
        """
        self.validate_timestamps(t)
        return self.distribution.mean(*self.predictions[(t[0], t[-1])])

    def var(self, t):
        """
        Returns the variance forecasts for the timestamps t.
        """
        self.validate_timestamps(t)
        return self.distribution.var(*self.predictions[(t[0], t[-1])])

    def percentile(self, p, t):
        """
        Returns the p-percentile forecasts for the timestamps t.
        """
        self.validate_timestamps(t)
        return self.distribution.percentile(p, *self.predictions[(t[0], t[-1])])

    def median(self, t):
        """
        Returns the median forecasts for the timestamps t.
        """
        return self.percentile(50, t)

    def pit(self, y_true, t):
        """
        Returns the Probability Integral Transform (PIT) for the timestamps t,
        given the true observations y_true.
        """
        self.validate_timestamps(t)
        return self.distribution.cdf(y_true, *self.predictions[(t[0], t[-1])])

    def crps(self, y_true, t):
        """
        Returns the Continuous Ranked Probability Score (CRPS) for the timestamps t,
        given the true observations y_true.
        """
        self.validate_timestamps(t)
        return self.distribution.crps(y_true, *self.predictions[(t[0], t[-1])])

    def mcrps(self, y_true, t):
        """
        Computes the mean CRPS for the timestamps t.
        """
        return np.nanmean(self.crps(y_true, t), axis=0)

    def rcrps(self, y_true, t):
        """
        Computes the relative CRPS for the timestamps t.
        """
        return 100 * self.mcrps(y_true, t) / self.y_mean

    def ae(self, y_true, t):
        """
        Computes the absolute error for the timestamps t. Note that the median forecast is used as a point estimate.
        """
        return np.abs(y_true - self.median(t))

    def mae(self, y_true, t):
        """
        Computes the Mean Absolute Error (MAE) for the timestamps t.
        """
        return np.nanmean(self.ae(y_true, t), axis=0)

    def mase(self, y_true, t):
        """
        Computes the Mean Absolute Scaled Error (MASE) for the timestamps t.
        """
        mae = self.mae(y_true, t)
        return mae / np.nanmean(np.abs(y_true[1:] - y_true[:-1]), axis=0)

    def mape(self, y_true, t):
        """
        Computes the Mean Absolute Percentage Error (MAPE) for the timestamps t.
        """
        ape = 100 * self.ae(y_true, t) / y_true
        return np.nanmean(ape, axis=0)

    def rmae(self, y_true, t):
        """
        Computes the relative Mean Absolute Error (rMAE) for the timestamps t.
        """
        return 100 * self.mae(y_true, t) / self.y_mean

    def se(self, y_true, t):
        """
        Computes the squared error for the timestamps t. Note that the mean forecast is used as a point estimate.
        """
        return (y_true - self.mean(t)) ** 2

    def rmse(self, y_true, t):
        """
        Computes the Root Mean Squared Error (RMSE) for the timestamps t.
        """
        return np.sqrt(np.nanmean(self.se(y_true, t), axis=0))

    def rrmse(self, y_true, t):
        """
        Computes the relative Root Mean Squared Error (rRMSE) for the timestamps t.
        """
        return 100 * self.rmse(y_true, t) / self.y_mean

    def evaluate(self, y_true, t):
        """
        Evaluates all metrics for the true observations y_true and the timestamps t and
        saves the results to a dictionary.
        """
        mean = self.mean(t)
        var = self.var(t)
        p_05 = self.percentile(5, t)
        p_25 = self.percentile(25, t)
        p_50 = self.median(t)
        p_75 = self.percentile(75, t)
        p_95 = self.percentile(95, t)
        pit = self.pit(y_true, t)
        crps = self.crps(y_true, t)
        mcrps = self.mcrps(y_true, t)
        rcrps = self.rcrps(y_true, t)
        # ae = self.ae(y_true, t)
        mae = self.mae(y_true, t)
        mase = self.mase(y_true, t)
        mape = self.mape(y_true, t)
        rmae = self.rmae(y_true, t)
        # se = self.se(y_true, t)
        rmse = self.rmse(y_true, t)
        rrmse = self.rrmse(y_true, t)

        if not self.global_model:
            mean = mean[:, np.newaxis]
            var = var[:, np.newaxis]
            p_05 = p_05[:, np.newaxis]
            p_25 = p_25[:, np.newaxis]
            p_50 = p_50[:, np.newaxis]
            p_75 = p_75[:, np.newaxis]
            p_95 = p_95[:, np.newaxis]
            pit = pit[:, np.newaxis]
            crps = crps[:, np.newaxis]
            mcrps = [mcrps]
            rcrps = [rcrps]
            # ae = ae[:, np.newaxis]
            mae = [mae]
            mase = [mase]
            mape = [mape]
            rmae = [rmae]
            # se = se[:, np.newaxis]
            rmse = [rmse]
            rrmse = [rrmse]

        for i in range(self.n):
            self.results[i]['t0'].append(t[0].strftime('%Y-%m-%d, %H:%M'))
            self.results[i]['mean'].append(mean[:, i].tolist())
            self.results[i]['var'].append(var[:, i].tolist())
            self.results[i]['p05'].append(p_05[:, i].tolist())
            self.results[i]['p25'].append(p_25[:, i].tolist())
            self.results[i]['p50'].append(p_50[:, i].tolist())
            self.results[i]['p75'].append(p_75[:, i].tolist())
            self.results[i]['p95'].append(p_95[:, i].tolist())
            self.results[i]['PIT'].append(pit[:, i].tolist())
            self.results[i]['CRPS'].append(crps[:, i].tolist())
            self.results[i]['mCRPS'].append(mcrps[i])
            self.results[i]['rCRPS'].append(rcrps[i])
            # self.results[i]['AE'].append(ae[:, i].tolist())
            self.results[i]['MAE'].append(mae[i])
            self.results[i]['MASE'].append(mase[i])
            self.results[i]['MAPE'].append(mape[i])
            self.results[i]['rMAE'].append(rmae[i])
            # self.results[i]['SE'].append(se[:, i].tolist())
            self.results[i]['RMSE'].append(rmse[i])
            self.results[i]['rRMSE'].append(rrmse[i])

    def get_out_dir(self):
        """
        Returns the directory where results will be saved to.
        """
        out_dir = os.path.join(OUT_PATH, self.__str__())
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def save_results(self):
        """
        Saves the evaluation results as a JSON file to the directory specified by get_out_dir().
        """
        out_dir = self.get_out_dir()
        for i in range(self.n):
            with open(os.path.join(out_dir, self.results[i]['ID'] + '.json'), 'w') as fp:
                json.dump(utils.round_floats(self.results[i]), fp)

    def plot_forecast(self, y_true, t, plot_median=True, plot_percentiles=True, save_fig=False):
        """
        Plots the median forecasts, the 50% confidence intervals and the 90% confidence intervals along with
        the true observations y_true for the timestamps t. If plot_median=False the mean forecast will be plotted.
        If plot_percentiles=False no confidence intervals are shown.
        """
        median = self.median(t)
        mean = self.mean(t)
        p_25 = self.percentile(25, t)
        p_75 = self.percentile(75, t)
        p_05 = self.percentile(5, t)
        p_95 = self.percentile(95, t)

        if not self.global_model:
            median = median[:, np.newaxis]
            mean = mean[:, np.newaxis]
            p_25 = p_25[:, np.newaxis]
            p_75 = p_75[:, np.newaxis]
            p_05 = p_05[:, np.newaxis]
            p_95 = p_95[:, np.newaxis]
            y_true = y_true[:, np.newaxis]

        for i in range(self.n):
            plt.figure(figsize=(10, 3.5))
            if plot_median:
                plt.plot(
                    np.arange(len(t)),
                    median[:, i],
                    color='C2',
                    linewidth=1,
                    label='Median forecast'
                )
            else:
                plt.plot(
                    np.arange(len(t)),
                    mean[:, i],
                    color='C2',
                    linewidth=1,
                    label='Mean forecast'
                )

            if plot_percentiles:
                plt.fill_between(
                    np.arange(len(t)),
                    p_25[:, i],
                    p_75[:, i],
                    alpha=0.4,
                    color='C2',
                    edgecolor='none',
                    label='50%'
                )
                plt.fill_between(
                    np.arange(len(t)),
                    p_05[:, i],
                    p_95[:, i],
                    alpha=0.25,
                    color='C2',
                    edgecolor='none',
                    label='90%'
                )

            plt.scatter(
                np.arange(len(t)),
                y_true[:, i],
                color='C7',
                label='Observations',
                s=7
            )
            plt.ylabel('Energy [kWh]')
            ticks = np.array(t[::self.s_d].map(lambda x: x.strftime('%a, %H:%M')))
            ticks[0] = t[0].strftime('%a, %H:%M\n%b %d, %Y')
            plt.xticks(np.arange(0, len(t), self.s_d), ticks, rotation=0)

            plt.title(f'{self.results[i]["ID"]}')
            plt.legend()
            if save_fig:
                start_date = t[0].strftime('%Y-%m-%d-%H-%M')
                plt.tight_layout()
                plt.savefig(
                    os.path.join(self.get_out_dir(), f'{self.results[i]["ID"]}_{start_date}.pdf'),
                    bbox_inches='tight'
                )
            else:
                plt.show()
            plt.close()
