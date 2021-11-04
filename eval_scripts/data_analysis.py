import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import statsmodels.api as sm
import plotly.graph_objects as go

import utils
import main
from eval_scripts import model_eval

# LaTeX settings
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'sans-serif': ['lmodern'], 'size': 18})
plt.rc('axes', **{'titlesize': 18, 'labelsize': 18})

# Constants
DATA_PATH = './data/'
OUT_PATH = './out/'
WEATHER_VARIABLE_NAMES = {
    'temperature': 'Temperature [°C]',
    'dew_point': 'Dew point [°C]',
    'wind_speed': 'Wind speed [m/s]'
}
DECIMALS = 2
COLORS = ('#5e3c99', '#fdb863', '#e66101', '#b2abd2')
MARKERS = ('o', 'X', 'v', 'd', 'p')
S_D = 48
S_W = 7 * S_D

# Load the dataframes
energy_df = pd.read_csv(DATA_PATH + 'energy_data.csv', index_col=0, parse_dates=True)
weather_df = pd.read_csv(DATA_PATH + 'weather_reanalysis_data.csv', index_col=0, parse_dates=True)
weather_forecast4d_df = pd.read_csv(DATA_PATH + 'weather_forecast4d_data.csv', index_col=0, parse_dates=True)
demographic_df = pd.read_csv(DATA_PATH + 'demographic_data.csv', index_col=0)


def get_level_info(levels=('L0', 'L1', 'L2', 'L3')):
    level_info = {}
    for level in levels:

        # Get cluster info
        if level == 'L0':
            clusters, cardinality = ['Agg'], [demographic_df.shape[0]]
            parents = ['']
        elif level == 'L1':
            clusters, cardinality = np.unique(demographic_df.acorn_category, return_counts=True)
            clusters, cardinality = clusters[[1, 2, 0]], cardinality[[1, 2, 0]]
            parents = ['Agg'] * len(clusters)
        elif level == 'L2':
            clusters, cardinality = np.unique(demographic_df.acorn_group, return_counts=True)
            parents = []
            for cluster in clusters:
                parents.append(demographic_df.acorn_category.loc[demographic_df.acorn_group == cluster].iloc[0])
        elif level == 'L3':
            clusters, cardinality = demographic_df.index, np.ones(demographic_df.shape[0], dtype=np.int)
            parents = []
            for cluster in clusters:
                parents.append(demographic_df.acorn_group.loc[cluster])
        else:
            raise ValueError('Level not available.')

        level_info[level] = {
            'clusters': list(clusters),
            'cardinality': list(cardinality),
            'parents': list(parents)
        }

    return level_info


def create_hierarchical_sunburst():
    level_info = get_level_info(['L0', 'L1', 'L2'])

    labels = []
    ids = []
    parents = []
    values = []
    for level, info in level_info.items():
        labels += [
            f"{cluster.replace('ACORN-', '')} ({count})" for cluster, count in
            zip(info['clusters'], info['cardinality'])
        ]
        ids += info['clusters']
        parents += info['parents']
        values += info['cardinality']
    labels[0] = 'All series (2500)'
    for i, value in enumerate(values):
        if value < 30:
            labels[i] = ''

    fig = go.Figure(go.Sunburst(
        labels=labels,
        ids=ids,
        parents=parents,
        values=values,
        branchvalues='total',
        insidetextorientation='auto',
        sort=False
    ))

    fig.update_layout(
        font={'family': 'serif', 'size': 16},
        margin={'t': 5, 'l': 5, 'r': 5, 'b': 5},
        colorway=[COLORS[0], COLORS[1], COLORS[2]]
    )

    fig.write_image(OUT_PATH + 'hierarchical_sunburst.pdf')


def get_observations_at(level, cluster, t=None):
    if t is None:
        t = energy_df.index

    if level == 'L0':
        y = np.nanmean(energy_df.loc[t].to_numpy(float), axis=1) * len(demographic_df)
    elif level == 'L1':
        h_ids = demographic_df.loc[demographic_df.acorn_category == cluster].index
        y = np.nanmean(energy_df.loc[t, h_ids].to_numpy(float), axis=1) * len(h_ids)
    elif level == 'L2':
        h_ids = demographic_df.loc[demographic_df.acorn_group == cluster].index
        y = np.nanmean(energy_df.loc[t, h_ids].to_numpy(float), axis=1) * len(h_ids)
    elif level == 'L3':
        y = energy_df.loc[t, cluster].to_numpy(float)
    else:
        raise ValueError('Level not available.')
    return y


def get_weather_df(forecast):
    if forecast:
        return weather_forecast4d_df
    else:
        return weather_df


def daily(x, reduce=False):
    x = utils.interpolate_nans(x)
    x = x[:, np.newaxis]
    if x.ndim == 3:
        x_daily = x.reshape(-1, S_D, x.shape[2])
    else:
        x_daily = x.reshape(-1, S_D)
    if reduce:
        x_daily = np.nanmean(x_daily, axis=1)
    return x_daily


def rmae(y_true, y_hat, axis=1):
    return 100 * np.nanmean(np.abs(y_true - y_hat), axis=axis) / np.nanmean(y_true)


def create_weather_forecast_df(
        weather_variables=tuple(WEATHER_VARIABLE_NAMES.keys()),
        horizons=(1, 2, 3, 4),
        with_std=True,
        to_LaTeX=True
):
    t_train = main.train_val_split(energy_df.index)[0]

    row_names = [WEATHER_VARIABLE_NAMES[weather_variable].split(' [')[0] for weather_variable in weather_variables]
    col_names = [f'{horizon}-day' for horizon in horizons]

    weather_forecast_df = pd.DataFrame(index=row_names, columns=col_names, dtype=float)
    for w, weather_variable in enumerate(weather_variables):
        actual = weather_df.loc[t_train, weather_variable].to_numpy(float).reshape(-1, 4 * S_D)
        forecast = weather_forecast4d_df.loc[t_train, weather_variable].to_numpy(float).reshape(-1, 4 * S_D)
        for h, horizon in enumerate(horizons):
            idx = np.arange(0, horizon * S_D)
            mean = np.mean(rmae(actual[:, idx], forecast[:, idx]))
            weather_forecast_df.iloc[w, h] = (('%%.%sf' % DECIMALS) % mean)
            if with_std:
                std = np.std(rmae(actual[:, idx], forecast[:, idx]))
                weather_forecast_df.iloc[w, h] += (' (%%.%sf)' % DECIMALS) % std

    if to_LaTeX:
        model_eval.df_to_LaTeX(weather_forecast_df)

    return weather_forecast_df


def correlation(x, y):
    # Pearson correlation
    mean_x = np.nanmean(x)
    var_x = np.nanvar(x)
    mean_y = np.nanmean(y)
    var_y = np.nanvar(y)

    return np.nanmean((x - mean_x) * (y - mean_y)) / np.sqrt(var_x * var_y)


def autocorrelation(y, lag):
    mean_y = np.nanmean(y)
    var_y = np.nanvar(y)

    return np.nanmean((y[lag:] - mean_y) * (y[:-lag] - mean_y)) / var_y


def r_squared(y, y_hat):
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((y - y_hat) ** 2)
    return 1 - ss_res / ss_tot


def lin_reg(X, y, standardize=False):
    if X.ndim == 1:
        X = X[:, np.newaxis]
    if standardize:
        X_mean = np.mean(X, axis=0, keepdims=True)
        X_std = np.std(X, axis=0, keepdims=True)
        X = utils.standardize(X, X_mean, X_std)
    X = np.hstack([X, np.ones((len(X), 1))])
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    y_hat = X @ w
    return y_hat, w


def get_seasonal_sine(t, per_day=True, forecast=False):
    seconds = t.map(dt.datetime.timestamp).to_numpy(float)
    if per_day:
        seconds = daily(seconds, reduce=True)
    year = 24 * 60 * 60 * 365.2425
    sin_y = lambda x: np.sin((2 * np.pi / year) * seconds + x)

    # Maximize the correlation between temperature and seasonality by shifting the sine appropriately
    daily_temperature = daily(get_weather_df(forecast).temperature.loc[t].to_numpy(float), reduce=True)
    objective = lambda x: correlation(daily_temperature, sin_y(x))
    phis = np.linspace(-np.pi, np.pi, 100)
    phi = phis[np.argmax([objective(phi) for phi in phis])]
    return sin_y(phi), daily_temperature


def weather_OLS(
        level, cluster,
        weather_variables=tuple(WEATHER_VARIABLE_NAMES.keys()),
        include_seasonality=True,
        forecast=False
):
    t_train = main.train_val_split(energy_df.index)[0]
    y = daily(get_observations_at(level, cluster, t_train), reduce=True)
    y_name = f'Energy demand ({level})'
    X_name = ['Constant']
    if len(weather_variables) != 0:
        X = daily(get_weather_df(forecast).loc[t_train, weather_variables].to_numpy(float), reduce=True)
        X_name += [WEATHER_VARIABLE_NAMES[weather_variable].split(' [')[0] for weather_variable in weather_variables]
    else:
        X = np.empty((len(t_train) // S_D, 0))
    if include_seasonality:
        X = np.hstack([X, get_seasonal_sine(t_train)[0][:, np.newaxis]])
        X_name.append('Seasonality')
    X = utils.standardize(X, np.mean(X, axis=0, keepdims=True), np.std(X, axis=0, keepdims=True))
    X = sm.add_constant(X)

    ols = sm.OLS(y, X).fit()
    return ols.summary(yname=y_name, xname=X_name), ols.rsquared


def auto_OLS(level, cluster, lags=(S_D, 2 * S_D, S_W)):
    t_train = main.train_val_split(energy_df.index)[0]
    y = get_observations_at(level, cluster, t_train)
    y_name = f'Energy demand ({level})'
    X_name = ['Constant']
    X = np.empty((len(t_train), 0))
    for lag in lags:
        y_lag = np.hstack([y[:lag], y[:-lag]])[:, np.newaxis]
        X = np.hstack([X, y_lag])
        X_name.append(f'Lag {lag}')
    X = sm.add_constant(X)

    ols = sm.OLS(y, X).fit()
    return ols.summary(yname=y_name, xname=X_name), ols.rsquared


def _complete_plot(name, legend=True, grid=True):
    if legend:
        plt.legend()
    if grid:
        plt.grid()
    plt.tight_layout()
    plt.savefig(OUT_PATH + f'{name}.pdf', bbox_inches='tight')
    plt.close()


def plot_autocorrelation_over_aggregate_size(lags=(S_D, 2 * S_D, S_W)):
    t_train = main.train_val_split(energy_df.index)[0]
    level_info = get_level_info()

    plt.figure(figsize=(6, 4))
    for i, lag in enumerate(lags):
        aggregate_sizes = []
        corr = []
        for level, info in level_info.items():
            level_corr = []
            for cluster in info['clusters']:
                y = get_observations_at(level, cluster, t_train)
                level_corr.append(autocorrelation(y, lag))

            if level == 'L3':
                aggregate_sizes += [1]
                corr += [np.mean(level_corr)]
            else:
                aggregate_sizes += info['cardinality']
                corr += level_corr

        plt.scatter(
            aggregate_sizes,
            corr,
            label=f'Lag {lag}',
            marker=MARKERS[i],
            color=COLORS[i]
        )

    plt.ylabel('Autocorrelation')
    plt.xlabel('\\# aggregated meters')
    plt.xscale('log')
    plt.xticks([1, 10, 100, 1000], ['1', '10', '100', '1000'])
    _complete_plot(f'autocorrelation_over_aggregate_size', grid=False)


def plot_correlation(weather_variable, level, cluster, forecast=False):
    t_train = main.train_val_split(energy_df.index)[0]
    y = daily(get_observations_at(level, cluster, t_train), reduce=True)
    u = daily(get_weather_df(forecast).loc[t_train, weather_variable].to_numpy(float), reduce=True)

    _, w = lin_reg(u, y)
    corr = correlation(u, y)

    f = plt.figure(figsize=(3.5, 3.5))
    ax = f.add_subplot(111)
    plt.scatter(u, y, s=4, alpha=0.5, color=COLORS[0])
    u_range = np.array([np.amin(u), np.amax(u)])
    plt.plot(u_range, w[0] * u_range + w[1], color=COLORS[0], label=f'$R^2 = {corr ** 2:.3f}$')

    plt.text(0.94, 0.94, f'$R^2 = {corr ** 2:.3f}$',
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax.transAxes,
             fontsize=18,
             bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round', alpha=0.5))

    plt.ylabel('Energy [kWh]')
    plt.xlabel(WEATHER_VARIABLE_NAMES[weather_variable])
    _complete_plot(
        f'{level}_{cluster}_{weather_variable}{"_F" if forecast else ""}_correlation',
        grid=False,
        legend=False
    )


def plot_temperature_seasonality(forecast=False):
    t_train = main.train_val_split(energy_df.index)[0]
    sin_y, daily_temp = get_seasonal_sine(t_train, forecast=forecast)
    sin_y_hat = lin_reg(sin_y, daily_temp)[0]
    corr = correlation(sin_y, daily_temp)

    plt.figure(figsize=(10, 4))
    plt.plot(
        t_train[::S_D],
        daily_temp,
        label='Daily temperature average',
        color=COLORS[0]
    )
    plt.plot(
        t_train[::S_D],
        sin_y_hat,
        label=f'Yearly seasonality',
        color=COLORS[1]
    )
    print(f'R^2 = {corr ** 2:.3f}')

    plt.ylabel(WEATHER_VARIABLE_NAMES['temperature'])
    _complete_plot(f'temperature_seasonality{"_F" if forecast else ""}', grid=False)


def plot_weather_correlation_per_half_hour(
    level, cluster,
    weather_variables=tuple(WEATHER_VARIABLE_NAMES.keys()),
    forecast=False
):
    t_train = main.train_val_split(energy_df.index)[0]
    y = daily(get_observations_at(level, cluster, t_train))
    u = daily(get_weather_df(forecast).loc[t_train, weather_variables].to_numpy(float))

    plt.figure(figsize=(10, 4))
    for w, weather_variable in enumerate(weather_variables):
        corr = []
        for hh in range(S_D):
            corr.append(correlation(u[:, hh, w], y[:, hh]) ** 2)
        plt.plot(
            corr,
            label=WEATHER_VARIABLE_NAMES[weather_variable].split(' [')[0],
            marker=MARKERS[w],
            color=COLORS[w]
        )

    plt.ylabel('$R^2$')
    plt.xlabel('Time of day')
    ticks = np.array(t_train[2:S_D:6].map(lambda x: x.strftime('%H:%M')))
    plt.xticks(np.arange(2, S_D, 6), ticks, rotation=0)
    _complete_plot(f'{level}_{cluster}_weather{"_F" if forecast else ""}_correlation_per_half_hour', grid=False)


def plot_weather_correlation_over_aggregate_size(
        weather_variables=tuple(WEATHER_VARIABLE_NAMES.keys()),
        forecast=False
):
    t_train = main.train_val_split(energy_df.index)[0]
    u = daily(get_weather_df(forecast).loc[t_train, weather_variables].to_numpy(float), reduce=True)
    level_info = get_level_info()

    plt.figure(figsize=(8, 4))
    for w, weather_variable in enumerate(weather_variables):
        aggregate_sizes = []
        corr = []
        for level, info in level_info.items():
            level_corr = []
            for cluster in info['clusters']:
                y = daily(get_observations_at(level, cluster, t_train), reduce=True)
                level_corr.append(correlation(u[:, w], y) ** 2)

            if level == 'L3':
                aggregate_sizes += [1]
                corr += [np.mean(level_corr)]
            else:
                aggregate_sizes += info['cardinality']
                corr += level_corr

        plt.scatter(
            aggregate_sizes,
            corr,
            label=WEATHER_VARIABLE_NAMES[weather_variable].split(' [')[0],
            marker=MARKERS[w],
            color=COLORS[w]
        )

    plt.ylabel('$R^2$')
    plt.xlabel('\\# aggregated meters')
    plt.xscale('log')
    plt.xticks([1, 10, 100, 1000], ['1', '10', '100', '1000'])
    _complete_plot(f'weather{"_F" if forecast else ""}_correlation_over_aggregate_size', grid=False)
