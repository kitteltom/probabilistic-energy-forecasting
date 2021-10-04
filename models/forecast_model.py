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
    @abstractmethod
    def __init__(self, y, t, u=None, ID='', seed=0, global_model=False):
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

        # Last timestamp where a forecast is available
        self.t_l = t[-1]

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
        if isinstance(t, dt.datetime):
            return int((t - (self.t_f if relative else self.t[0])).total_seconds() / (60 * 30))
        else:
            return np.array([
                int((hh - (self.t_f if relative else self.t[0])).total_seconds() / (60 * 30)) for hh in t
            ])

    @abstractmethod
    def fit(self):
        pass

    def validate_timestamps(self, t):
        if t[0] < self.t_f or t[-1] > self.t_l:
            raise ValueError('No prediction available for the timestamps t.')

    def validate_input(self, u):
        if self.u is None and u is not None:
            raise ValueError('No initial input u available.')
        if self.u is not None and u is None:
            raise ValueError('Missing input u.')

    def add_measurements(self, y, t, u=None):
        if t[0] != self.t[-1] + dt.timedelta(minutes=30):
            raise ValueError('No subsequent measurements.')

        if self.global_model:
            self.y = np.vstack([self.y, y])
        else:
            self.y = np.hstack([self.y, y])
        self.t = self.t.union(t)
        self.t_l = t[-1]
        self.validate_input(u)
        if u is not None:
            if u.ndim == 1:
                u = u[:, np.newaxis]
            self.u = np.vstack([self.u, u])

    @abstractmethod
    def predict(self, t, u=None):
        if t[0] >= self.t_f and t[-1] <= self.t_l:
            # Prediction already available
            return True

        if t[0] != self.t[-1] + dt.timedelta(minutes=30):
            raise ValueError('Prediction has to start right after last measurement.')
        if t[0] != self.t_l + dt.timedelta(minutes=30):
            raise ValueError('Forecasts must be subsequent.')
        if len(t) > self.max_horizon:
            raise ValueError(f'The maximum forecast horizon is {self.max_horizon} half-hours.')
        self.validate_input(u)

        # Execute prediction
        self.t_l = t[-1]
        return False

    @abstractmethod
    def get_mean(self, t):
        self.validate_timestamps(t)

    @abstractmethod
    def get_var(self, t):
        self.validate_timestamps(t)

    @abstractmethod
    def get_percentile(self, p, t):
        self.validate_timestamps(t)

    def get_median(self, t):
        return self.get_percentile(50, t)

    @abstractmethod
    def get_pit(self, y_true, t):
        self.validate_timestamps(t)

    @abstractmethod
    def get_crps(self, y_true, t):
        self.validate_timestamps(t)

    def mcrps(self, y_true, t):
        return np.nanmean(self.get_crps(y_true, t), axis=0)

    def rcrps(self, y_true, t):
        return 100 * self.mcrps(y_true, t) / self.y_mean

    def ae(self, y_true, t):
        return np.abs(y_true - self.get_median(t))

    def mae(self, y_true, t):
        return np.nanmean(self.ae(y_true, t), axis=0)

    def mase(self, y_true, t):
        mae = self.mae(y_true, t)
        return mae / np.nanmean(np.abs(y_true[1:] - y_true[:-1]), axis=0)

    def mape(self, y_true, t):
        ape = 100 * self.ae(y_true, t) / y_true
        return np.nanmean(ape, axis=0)

    def rmae(self, y_true, t):
        return 100 * self.mae(y_true, t) / self.y_mean

    def se(self, y_true, t):
        return (y_true - self.get_mean(t)) ** 2

    def rmse(self, y_true, t):
        return np.sqrt(np.nanmean(self.se(y_true, t), axis=0))

    def rrmse(self, y_true, t):
        return 100 * self.rmse(y_true, t) / self.y_mean

    def evaluate(self, y_true, t):
        mean = self.get_mean(t)
        var = self.get_var(t)
        p_05 = self.get_percentile(5, t)
        p_25 = self.get_percentile(25, t)
        p_50 = self.get_median(t)
        p_75 = self.get_percentile(75, t)
        p_95 = self.get_percentile(95, t)
        pit = self.get_pit(y_true, t)
        crps = self.get_crps(y_true, t)
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
        out_dir = os.path.join(OUT_PATH, self.__str__())
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        return out_dir

    def save_results(self):
        out_dir = self.get_out_dir()
        for i in range(self.n):
            with open(os.path.join(out_dir, self.results[i]['ID'] + '.json'), 'w') as fp:
                json.dump(utils.round_floats(self.results[i]), fp)

    def plot_forecast(self, y_true, t, plot_median=True, plot_percentiles=True, save_fig=False):
        median = self.get_median(t)
        mean = self.get_mean(t)
        p_25 = self.get_percentile(25, t)
        p_75 = self.get_percentile(75, t)
        p_05 = self.get_percentile(5, t)
        p_95 = self.get_percentile(95, t)

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
