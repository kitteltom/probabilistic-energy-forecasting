import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import datetime as dt

import main
from eval_scripts import data_analysis

# LaTeX settings
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'sans-serif': ['lmodern'], 'size': 18})
plt.rc('axes', **{'titlesize': 18, 'labelsize': 18})

# Constants
SEASON = 'fw'
JSON_PATH = f'/Users/kitteltom/out_{SEASON}/'
OUT_PATH = f'./out/{SEASON}/'
MODEL_NAMES = {
    'KF': ('KalmanFilter', ''),
    'KF(+W)': ('KalmanFilter', '_W'),
    'KF(+WF)': ('KalmanFilter', '_WF'),
    'KD-IC': ('KD-IC', ''),
    'KD-IC(+W)': ('KD-IC', '_W'),
    'KD-IC(+WF)': ('KD-IC', '_WF'),
    # 'LN-IC': ('LogNormal-IC', ''),
    # 'LN-IC(+W)': ('LogNormal-IC', '_W'),
    # 'LN-IC(+WF)': ('LogNormal-IC', '_WF'),
    'DeepAR': ('DeepAR', ''),
    'DeepAR(+W)': ('DeepAR', '_W'),
    'DeepAR(+WF)': ('DeepAR', '_WF'),
    'LWR': ('LastWeekRegression', ''),
    'LWR(+W)': ('LastWeekRegression', '_W'),
    'LWR(+WF)': ('LastWeekRegression', '_WF'),
    'LW': ('LastWeek', ''),
}
MAIN_SEED = '42'
FORECAST_REPS = 28
HORIZON = 192

DECIMALS = 2
COLORS = ('#5e3c99', '#fdb863', '#e66101', '#b2abd2', '#000000')
MARKERS = ('o', 'X', 'v', 'd', 'p')
LINESTYLES = ('solid', 'dashed', 'dashdot')
S_D = 48
S_W = 7 * S_D


def get_file_name(model, level, cluster, seed=''):
    return f'{MODEL_NAMES[model][0]}{seed}_{level}_{cluster}{MODEL_NAMES[model][1]}_{SEASON}'


def get_path(model, level, cluster, seed=''):
    return JSON_PATH + f'{MODEL_NAMES[model][0]}{seed}/{get_file_name(model, level, cluster, seed)}.json'


def load_res(model, level, cluster, seed=''):
    if 'DeepAR' in model and seed == '':
        seed = MAIN_SEED
    with open(get_path(model, level, cluster, seed), 'r') as fp:
        res = json.load(fp)
    return res


def collect_results(
        levels=('L0', 'L1', 'L2', 'L3'),
        metrics=('MAPE', 'rMAE', 'rRMSE', 'rCRPS'),
        models=MODEL_NAMES.keys(),
        seeds=(0, 1, 2, 3, 4),
        save_results_with_info=True
):
    results_path = os.path.join(JSON_PATH, f'results_with_info_{SEASON}.npy')
    if os.path.isfile(results_path):
        results_with_info = np.load(results_path, allow_pickle=True)
        return results_with_info[0], results_with_info[1]

    results = {}
    level_info = data_analysis.get_level_info(levels)
    for level in levels:
        clusters = level_info[level]['clusters']

        # Create results array
        results[level] = np.empty((len(metrics), len(models), len(clusters), FORECAST_REPS))
        results[level][:] = np.nan
        for m, model in enumerate(models):
            if level == 'L3' and 'KF' in model:
                # No level 3 results for the KF model
                continue

            for c, cluster in enumerate(clusters):
                if 'DeepAR' in model and level is not 'L3':
                    res_per_seed = []
                    for seed in seeds:
                        res_per_seed.append(load_res(model, level, cluster, seed))
                    for i, metric in enumerate(metrics):
                        results[level][i, m, c] = np.mean([res[metric] for res in res_per_seed], axis=0)
                else:
                    res = load_res(model, level, cluster)
                    for i, metric in enumerate(metrics):
                        if 'CRPS' in metric and model == 'LW':
                            # No distributional forecasts for LW model
                            continue

                        results[level][i, m, c] = res[metric]

    info = {
        'levels': level_info,
        'metrics': list(metrics),
        'models': list(models),
        'reps': FORECAST_REPS
    }

    if save_results_with_info:
        np.save(results_path, (results, info), allow_pickle=True)

    return results, info


def collect_results_per_tstp(
        levels=('L0', 'L1', 'L2'),
        metrics=('rMAE', 'rRMSE', 'rCRPS'),
        models=MODEL_NAMES.keys(),
        seeds=(0, 1, 2, 3, 4),
        save_results_per_tstp_with_info=True
):
    results_path = os.path.join(JSON_PATH, f'results_per_tstp_with_info_{SEASON}.npy')
    if os.path.isfile(results_path):
        results_with_info = np.load(results_path, allow_pickle=True)
        return results_with_info[0], results_with_info[1]

    results = {}
    level_info = data_analysis.get_level_info(levels)
    t_train, t_val = main.train_val_split(data_analysis.energy_df.index, winter_period=SEASON == 'fw')
    for level in levels:
        clusters = level_info[level]['clusters']

        # Create results array
        results[level] = np.empty((len(seeds), len(metrics), len(models), len(clusters), FORECAST_REPS, HORIZON))
        results[level][:] = np.nan
        level_info[level]['y_mean'] = []
        for c, cluster in enumerate(clusters):
            level_info[level]['y_mean'].append(
                np.nanmean(data_analysis.get_observations_at(level, cluster, t_train))
            )
            y_true = data_analysis.get_observations_at(level, cluster, t_val).reshape(FORECAST_REPS, -1)[:, :HORIZON]
            for m, model in enumerate(models):
                if level == 'L3' and 'KF' in model:
                    # No level 3 results for the KF model
                    continue

                if 'DeepAR' in model and level is not 'L3':
                    for s, seed in enumerate(seeds):
                        res = load_res(model, level, cluster, seed)
                        for i, metric in enumerate(metrics):
                            if metric == 'rMAE':
                                results[level][s, i, m, c] = np.abs(y_true - res['p50'])
                            elif metric == 'rRMSE':
                                results[level][s, i, m, c] = (y_true - res['mean']) ** 2
                            elif metric == 'rCRPS':
                                results[level][s, i, m, c] = res['CRPS']
                else:
                    res = load_res(model, level, cluster)
                    for i, metric in enumerate(metrics):
                        if 'CRPS' in metric and model == 'LW':
                            # No distributional forecasts for LW model
                            continue

                        if metric == 'rMAE':
                            results[level][0, i, m, c] = np.abs(y_true - res['p50'])
                        elif metric == 'rRMSE':
                            results[level][0, i, m, c] = (y_true - res['mean']) ** 2
                        elif metric == 'rCRPS':
                            results[level][0, i, m, c] = res['CRPS']

    info = {
        'levels': level_info,
        'metrics': list(metrics),
        'models': list(models),
        'reps': FORECAST_REPS,
        'horizon': HORIZON
    }

    if save_results_per_tstp_with_info:
        np.save(results_path, (results, info), allow_pickle=True)

    return results, info


def create_metric_df(metric, with_std=True, to_LaTeX=True):
    results, info = collect_results()

    i = info['metrics'].index(metric)
    row_names = info['models']
    col_names = info['levels'].keys()

    metric_df = pd.DataFrame(index=row_names, columns=col_names, dtype=float)
    for level in col_names:
        for m, model in enumerate(row_names):
            mean = np.mean(results[level][i, m])
            metric_df.loc[model, level] = (('%%.%sf' % DECIMALS) % mean) if not np.isnan(mean) else '-'
            if with_std and not np.isnan(mean):
                std = np.std(results[level][i, m])
                metric_df.loc[model, level] += (' (%%.%sf)' % DECIMALS) % std

    if to_LaTeX:
        df_to_LaTeX(metric_df)

    return metric_df


def create_level_df(level, with_std=True, to_LaTeX=True):
    results, info = collect_results()

    row_names = info['metrics']
    col_names = info['models']

    level_df = pd.DataFrame(index=row_names, columns=col_names, dtype=float)
    for i, metric in enumerate(row_names):
        for m, model in enumerate(col_names):
            mean = np.mean(results[level][i, m])
            level_df.loc[metric, model] = (('%%.%sf' % DECIMALS) % mean) if not np.isnan(mean) else '-'
            if with_std and not np.isnan(mean):
                std = np.std(results[level][i, m])
                level_df.loc[metric, model] += (' (%%.%sf)' % DECIMALS) % std

    if to_LaTeX:
        df_to_LaTeX(level_df)

    return level_df


def create_runtime_df(models=('KF', 'KD-IC', 'DeepAR', 'LWR'), with_std=False, to_LaTeX=True):
    _, info = collect_results()

    train_name = 'Avg. training time [s]'
    prediction_name = 'Avg. prediction time [s]'
    runtime_df = pd.DataFrame(index=[train_name, prediction_name], columns=models, dtype=float)
    for model in models:
        training_times = []
        prediction_times = []
        for level in info['levels'].keys():
            if level == 'L3' and 'KF' in model:
                # No level 3 results for the KF model
                continue

            for cluster in info['levels'][level]['clusters']:
                res = load_res(model, level, cluster)
                training_times.append(res['fit_time'])
                prediction_times.append(res['prediction_time'])

        decimals = DECIMALS + 1
        runtime_df.loc[train_name, model] = ('%%.%sf' % decimals) % np.mean(training_times)
        runtime_df.loc[prediction_name, model] = ('%%.%sf' % decimals) % np.mean(prediction_times)
        if with_std:
            runtime_df.loc[train_name, model] += (' (%%.%sf)' % decimals) % np.std(training_times)
            runtime_df.loc[prediction_name, model] += (' (%%.%sf)' % decimals) % np.std(prediction_times)

    if to_LaTeX:
        df_to_LaTeX(runtime_df)

    return runtime_df


def df_to_LaTeX(df):
    num_columns = len(df.columns)
    print(df.to_latex(
        float_format=f'%.{DECIMALS}f',
        na_rep='-',
        column_format='l' + ''.join('r' * num_columns)
    ))


def get_color(model):
    if 'KF' in model:
        return COLORS[0]
    elif 'KD-IC' in model:
        return COLORS[1]
    elif 'DeepAR' in model:
        return COLORS[2]
    elif 'LWR' in model:
        return COLORS[3]
    else:
        return COLORS[4]


def get_linestyle(model):
    if '(+W)' in model:
        return LINESTYLES[1]
    elif '(+WF)' in model:
        return LINESTYLES[2]
    else:
        return LINESTYLES[0]


def _complete_plot(name, legend=True, grid=True):
    if legend:
        plt.legend()
    if grid:
        plt.grid()
    plt.tight_layout()
    plt.savefig(OUT_PATH + f'{name}_{SEASON}.pdf', bbox_inches='tight')
    plt.close()


def plot_epoch_loss(model, level, cluster, seed=MAIN_SEED):
    assert 'DeepAR' in model, "Loss plot only available for deep models"

    res = load_res(model, level, cluster, seed)
    train_loss = res['train_loss']
    val_loss = res['val_loss']

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(len(train_loss)) + 1, train_loss, color=COLORS[0], label='Train')
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, color=COLORS[1], label='Validation')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    _complete_plot(f'epoch_loss_{get_file_name(model, level, cluster, seed)}', grid=False)


def plot_horizon(model, metric, horizons=(1, 2, 3, 4), levels=('L0', 'L1', 'L2')):
    results, info = collect_results_per_tstp()
    model_W = model + '(+W)'
    model_WF = model + '(+WF)'

    i = info['metrics'].index(metric)
    m = info['models'].index(model)
    m_W = info['models'].index(model_W)
    m_WF = info['models'].index(model_WF)
    score = np.empty(len(horizons))
    score_W = np.empty(len(horizons))
    score_WF = np.empty(len(horizons))

    for h, horizon in enumerate(horizons):
        idx = np.arange(0, min(horizon * S_D, HORIZON))
        res = []
        res_W = []
        res_WF = []
        for level in levels:
            for c, cluster in enumerate(info['levels'][level]['clusters']):
                y_mean = info['levels'][level]['y_mean'][c]
                if metric == 'rRMSE':
                    res.append(100 * np.sqrt(np.mean(results[level][:, i, m, c, :, idx], axis=2)) / y_mean)
                    res_W.append(100 * np.sqrt(np.mean(results[level][:, i, m_W, c, :, idx], axis=2)) / y_mean)
                    res_WF.append(100 * np.sqrt(np.mean(results[level][:, i, m_WF, c, :, idx], axis=2)) / y_mean)
                else:
                    res.append(100 * np.mean(results[level][:, i, m, c, :, idx], axis=2) / y_mean)
                    res_W.append(100 * np.mean(results[level][:, i, m_W, c, :, idx], axis=2) / y_mean)
                    res_WF.append(100 * np.mean(results[level][:, i, m_WF, c, :, idx], axis=2) / y_mean)
        score[h] = np.nanmean(res)
        score_W[h] = np.nanmean(res_W)
        score_WF[h] = np.nanmean(res_WF)

    skill_W = 100 * (1 - score_W / score)
    skill_WF = 100 * (1 - score_WF / score)
    print(f'SS_{metric} (W): {skill_W}')
    print(f'SS_{metric} (WF): {skill_WF}')

    plt.figure(figsize=(3.5, 4))
    plt.plot(
        score,
        linestyle=get_linestyle(model),
        color=get_color(model),
        marker=MARKERS[0]
    )
    plt.plot(
        score_W,
        linestyle=get_linestyle(model_W),
        color=get_color(model_W),
        marker=MARKERS[1]
    )
    plt.plot(
        score_WF,
        linestyle=get_linestyle(model_WF),
        color=get_color(model_WF),
        marker=MARKERS[2]
    )
    # plt.ylim(6.95, 8.35)
    plt.ylabel(metric)
    plt.xlabel('Horizon')
    plt.xticks(np.arange(len(horizons)), np.array(horizons))
    plt.title(model)
    _complete_plot(f"horizon_{model}_{metric}", grid=False, legend=False)


def plot_score_comparison(model, metric1, metric2, levels=('L0', 'L1', 'L2')):
    results, info = collect_results()
    model_W = model + '(+W)'
    model_WF = model + '(+WF)'

    i1 = info['metrics'].index(metric1)
    i2 = info['metrics'].index(metric2)
    m = info['models'].index(model)
    m_W = info['models'].index(model_W)
    m_WF = info['models'].index(model_WF)
    scores = ([], [])
    scores_W = ([], [])
    scores_WF = ([], [])

    for level in levels:
        if level == 'L3' and 'KF' in model:
            # No level 3 results for the KF model
            continue
        for c, cluster in enumerate(info['levels'][level]['clusters']):
            scores[0].append(np.mean(results[level][i1, m, c]))
            scores[1].append(np.mean(results[level][i2, m, c]))
            scores_W[0].append(np.mean(results[level][i1, m_W, c]))
            scores_W[1].append(np.mean(results[level][i2, m_W, c]))
            scores_WF[0].append(np.mean(results[level][i1, m_WF, c]))
            scores_WF[1].append(np.mean(results[level][i2, m_WF, c]))

    plt.figure(figsize=(3.5, 4))
    plt.scatter(
        scores[0],
        scores[1],
        color=get_color(model),
        marker=MARKERS[0],
        edgecolors='none',
        alpha=1.0
    )
    print(f'Correlation: {data_analysis.correlation(scores[0], scores[1]):.3f}')
    plt.scatter(
        scores_W[0],
        scores_W[1],
        color=get_color(model),
        marker=MARKERS[1],
        edgecolors='none',
        alpha=0.75
    )
    print(f'Correlation (W): {data_analysis.correlation(scores_W[0], scores_W[1]):.3f}')
    plt.scatter(
        scores_WF[0],
        scores_WF[1],
        color=get_color(model),
        marker=MARKERS[2],
        edgecolors='none',
        alpha=0.5
    )
    print(f'Correlation (WF): {data_analysis.correlation(scores_WF[0], scores_WF[1]):.3f}')
    plt.ylabel(metric2)
    plt.xlabel(metric1)
    _complete_plot(f"comparison_{model}_{metric1}_{metric2}", grid=False, legend=False)


# def plot_score_comparison(metric1, metric2, levels=('L0', 'L1', 'L2'), models=None, name=None):
#     results, info = collect_results()
#     models = info['models'] if models is None else models
#
#     i1 = info['metrics'].index(metric1)
#     i2 = info['metrics'].index(metric2)
#
#     plt.figure(figsize=(6, 4))
#     for j, model in enumerate(models):
#         m = info['models'].index(model)
#         metric1_mean = []
#         metric2_mean = []
#         for level in levels:
#             if level == 'L3' and 'KF' in model:
#                 # No level 3 results for the KF model
#                 continue
#             for c, cluster in enumerate(info['levels'][level]['clusters']):
#                 metric1_mean.append(np.mean(results[level][i1, m, c]))
#                 metric2_mean.append(np.mean(results[level][i2, m, c]))
#         plt.scatter(
#             metric1_mean,
#             metric2_mean,
#             label=model,
#             color=get_color(model),
#             marker=MARKERS[j]
#         )
#
#     plt.ylabel(metric2)
#     plt.xlabel(metric1)
#     _complete_plot(f"comparison_{f'{name}_' if name is not None else ''}{metric1}_{metric2}", grid=False, legend=True)


def plot_reps(metric, levels=('L0', 'L1', 'L2'), models=None, name=None):
    results, info = collect_results()
    models = info['models'] if models is None else models

    i = info['metrics'].index(metric)

    # Lines for second legend
    _, ax = plt.subplots()
    lines = ax.plot([0, 1], [0, 1], '-C7', [0, 1], [0, 2], '--C7')
    plt.close()

    plt.figure(figsize=(10, 4))
    for j, model in enumerate(models):
        m = info['models'].index(model)
        reps_mean = []
        for level in levels:
            if level == 'L3' and 'KF' in model:
                # No level 3 results for the KF model
                continue
            for c, cluster in enumerate(info['levels'][level]['clusters']):
                reps_mean.append(results[level][i, m, c])
        reps_mean = np.mean(reps_mean, axis=0)
        plt.plot(
            reps_mean,
            label=model if '(' not in model else None,
            linestyle=get_linestyle(model),
            color=get_color(model)
        )
    plt.ylabel(metric)
    plt.xlabel('Forecast origin')

    plt.yticks(np.arange(5, 17, 2.5))
    t0 = load_res('LW', 'L0', 'Agg')['t0']
    ticks = [dt.datetime.strptime(tstp, '%Y-%m-%d, %H:%M').strftime('%b, %d') for tstp in t0[1::5]]
    plt.xticks(np.arange(1, len(t0), 5), ticks, rotation=0)

    plt.grid(axis='y')
    second_legend = plt.legend(lines, ('no weather', 'actual weather'), loc='upper left')
    plt.gca().add_artist(second_legend)
    plt.legend(loc='upper right')
    _complete_plot(f"reps_{f'{name}_' if name is not None else ''}{metric}", grid=False, legend=False)


def plot_clusters(level, metric, models=None, name=None):
    results, info = collect_results()
    models = info['models'] if models is None else models

    i = info['metrics'].index(metric)
    plt.figure(figsize=(10, 4))
    for model in models:
        if level == 'L3' and 'KF' in model:
            # No level 3 results for the KF model
            continue
        m = info['models'].index(model)
        clusters_mean = np.mean(results[level][i, m], axis=1)
        plt.plot(
            clusters_mean,
            label=model,
            linestyle=get_linestyle(model),
            color=get_color(model)
        )
    plt.ylabel(metric)
    cluster_labels = [f"{cluster.replace('ACORN-', '')} ({count})" for cluster, count in zip(
        info['levels'][level]['clusters'],
        info['levels'][level]['cardinality']
    )]
    if level == 'L3':
        plt.xticks(np.arange(0, len(cluster_labels), 100), np.array(cluster_labels)[::100], rotation=90)
    elif level == 'L2':
        plt.xticks(np.arange(len(cluster_labels)), cluster_labels, rotation=90)
    else:
        plt.xticks(np.arange(len(cluster_labels)), cluster_labels)
    _complete_plot(f"clusters_{f'{name}_' if name is not None else ''}{level}_{metric}")


def plot_aggregate_size(metric, models=None, name=None):
    results, info = collect_results()
    models = info['models'] if models is None else models

    i = info['metrics'].index(metric)
    aggregate_sizes = []
    errors = {}
    bottom_level_errors = {}
    for model in models:
        errors[model] = []
        bottom_level_errors[model] = []

    for level, level_info in info['levels'].items():
        for c, agg_size in enumerate(level_info['cardinality']):

            if level != 'L3':
                aggregate_sizes.append(agg_size)
                for model in models:
                    m = info['models'].index(model)
                    errors[model].append(np.mean(results[level][i, m, c]))
            else:
                for model in models:
                    m = info['models'].index(model)
                    bottom_level_errors[model].append(np.mean(results[level][i, m, c]))

    aggregate_sizes.append(1)
    for model in models:
        errors[model].append(np.mean(bottom_level_errors[model]))

    sorted_idx = np.argsort(aggregate_sizes)
    aggregate_sizes = np.array(aggregate_sizes)[sorted_idx]

    plt.figure(figsize=(6, 4))
    for model in models:
        if 'CRPS' in metric and model == 'LW':
            # No distributional forecasts for LW model
            continue
        plt.plot(
            aggregate_sizes,
            np.array(errors[model])[sorted_idx],
            label=model,
            linestyle=get_linestyle(model),
            color=get_color(model)
        )
    plt.ylabel(metric)
    # plt.yticks(np.arange(0, 70, 20))
    plt.xlabel('\\# aggregated meters')
    plt.xscale('log')
    plt.xticks([1, 10, 100, 1000], ['1', '10', '100', '1000'])
    _complete_plot(f"aggregate_size_{f'{name}_' if name is not None else ''}{metric}", grid=False)


def get_skill_scores(model, metric, no_L3=False):
    results, info = collect_results()

    i = info['metrics'].index(metric)
    m = info['models'].index(model)
    m_W = info['models'].index(model + '(+W)')
    m_WF = info['models'].index(model + '(+WF)')
    aggregate_sizes = []
    score = []
    score_W = []
    score_WF = []
    bottom_level_score = []
    bottom_level_score_W = []
    bottom_level_score_WF = []

    t_train = main.train_val_split(data_analysis.energy_df.index, winter_period=SEASON == 'fw')[0]
    u = data_analysis.daily(
        data_analysis.get_weather_df(forecast=False).loc[t_train, 'temperature'].to_numpy(float),
        reduce=True
    )
    u_F = data_analysis.daily(
        data_analysis.get_weather_df(forecast=True).loc[t_train, 'temperature'].to_numpy(float),
        reduce=True
    )
    corr_W = []
    corr_WF = []
    bottom_level_corr_W = []
    bottom_level_corr_WF = []

    for level, level_info in info['levels'].items():
        if level == 'L3' and ('KF' in model or no_L3):
            # No level 3 results for the KF model
            continue
        for c, (cluster, agg_size) in enumerate(zip(level_info['clusters'], level_info['cardinality'])):
            y = data_analysis.daily(
                np.array(data_analysis.get_observations_at(level, cluster, t_train)),
                reduce=True
            )

            if level != 'L3':
                aggregate_sizes.append(agg_size)
                score.append(np.mean(results[level][i, m, c]))
                score_W.append(np.mean(results[level][i, m_W, c]))
                score_WF.append(np.mean(results[level][i, m_WF, c]))

                corr_W.append(data_analysis.correlation(u, y) ** 2)
                corr_WF.append(data_analysis.correlation(u_F, y) ** 2)
            else:
                bottom_level_score.append(np.mean(results[level][i, m, c]))
                bottom_level_score_W.append(np.mean(results[level][i, m_W, c]))
                bottom_level_score_WF.append(np.mean(results[level][i, m_WF, c]))

                bottom_level_corr_W.append(data_analysis.correlation(u, y) ** 2)
                bottom_level_corr_WF.append(data_analysis.correlation(u_F, y) ** 2)

    if 'KF' not in model and not no_L3:
        aggregate_sizes.append(1)
        score.append(np.mean(bottom_level_score))
        score_W.append(np.mean(bottom_level_score_W))
        score_WF.append(np.mean(bottom_level_score_WF))
        corr_W.append(np.mean(bottom_level_corr_W))
        corr_WF.append(np.mean(bottom_level_corr_WF))

    aggregate_sizes = np.array(aggregate_sizes)
    score = np.array(score)
    score_W = np.array(score_W)
    score_WF = np.array(score_WF)
    corr_W = np.array(corr_W)
    corr_WF = np.array(corr_WF)

    skill_W = 100 * (1 - score_W / score)
    skill_WF = 100 * (1 - score_WF / score)

    return skill_W, skill_WF, aggregate_sizes, corr_W, corr_WF


def plot_aggregate_size_skill(model, metric):
    skill_W, skill_WF, aggregate_sizes, _, _ = get_skill_scores(model, metric)

    # # Regression
    # x = np.logspace(np.log10(min(aggregate_sizes)), np.log10(max(aggregate_sizes)), 100)
    # X = x[:, np.newaxis]
    # X = np.hstack([np.log(X), np.ones((len(X), 1))])
    #
    # _, w_W = data_analysis.lin_reg(np.log(aggregate_sizes), skill_W, standardize=False, polynomial=False)
    # y_W = X @ w_W
    #
    # _, w_WF = data_analysis.lin_reg(np.log(aggregate_sizes), skill_WF, standardize=False, polynomial=False)
    # y_WF = X @ w_WF

    corr_W = data_analysis.correlation(np.log(aggregate_sizes), skill_W)
    corr_WF = data_analysis.correlation(np.log(aggregate_sizes), skill_WF)
    print(f'Correlation (W): {corr_W:.3f}')
    print(f'Correlation (WF): {corr_WF:.3f}')

    plt.figure(figsize=(3.5, 4))
    plt.plot([1, 2500], [0, 0], color='grey', linestyle='dashed')
    plt.scatter(
        aggregate_sizes,
        skill_W,
        marker=MARKERS[0],
        color=get_color(model),
        edgecolors='none'
    )
    # plt.plot(
    #     x,
    #     y_W,
    #     color=get_color(model),
    #     label=f'{corr_W:.2f}'
    # )
    plt.scatter(
        aggregate_sizes,
        skill_WF,
        marker=MARKERS[1],
        color=get_color(model),
        edgecolors='none',
        alpha=0.5
    )
    # plt.plot(
    #     x,
    #     y_WF,
    #     color=get_color(model),
    #     alpha=0.5,
    #     label=f'{corr_WF:.2f}'
    # )
    plt.ylabel(f'$SS_{{\\mathrm{{{metric}}}}}$')
    plt.xlabel('\\# aggregated meters')
    plt.xscale('log')
    plt.xticks([1, 10, 100, 1000], ['1', '10', '100', '1000'])
    plt.title(model)
    _complete_plot(f"aggregate_size_skill_{model}_{metric}", grid=False, legend=False)


def plot_temperature_correlation_skill(model, metric):
    skill_W, skill_WF, _, corr_W, corr_WF = get_skill_scores(model, metric, no_L3=True)
    print(f'Correlation (W): {data_analysis.correlation(corr_W, skill_W):.3f}')
    print(f'Correlation (WF): {data_analysis.correlation(corr_WF, skill_WF):.3f}')

    plt.figure(figsize=(3.5, 4))
    plt.plot(
        [min(np.amin(corr_W), np.amin(corr_WF)), max(np.amax(corr_W), np.amax(corr_WF))],
        [0, 0],
        color='grey',
        linestyle='dashed'
    )
    plt.scatter(
        corr_W,
        skill_W,
        label='W',
        marker=MARKERS[0],
        color=get_color(model),
        edgecolors='none'
    )
    plt.scatter(
        corr_WF,
        skill_WF,
        label='WF',
        marker=MARKERS[1],
        color=get_color(model),
        edgecolors='none',
        alpha=0.5
    )
    plt.ylabel(f'$SS_{{\\mathrm{{{metric}}}}}$')
    plt.xlabel('Temperature corr. [$R^2$]')
    plt.title(model)
    _complete_plot(f'temperature_correlation_skill_{model}_{metric}', grid=False, legend=False)


def get_benchmark_skill_scores(model, metric, no_L3=False):
    assert 'CRPS' not in metric
    results, info = collect_results()

    i = info['metrics'].index(metric)
    lw = info['models'].index('LW')
    m = info['models'].index(model)
    m_W = info['models'].index(model + '(+W)')
    m_WF = info['models'].index(model + '(+WF)')
    aggregate_sizes = []
    score_lw = []
    score = []
    score_W = []
    score_WF = []
    bottom_level_score_lw = []
    bottom_level_score = []
    bottom_level_score_W = []
    bottom_level_score_WF = []

    for level, level_info in info['levels'].items():
        if level == 'L3' and ('KF' in model or no_L3):
            # No level 3 results for the KF model
            continue
        for c, (cluster, agg_size) in enumerate(zip(level_info['clusters'], level_info['cardinality'])):
            if level != 'L3':
                aggregate_sizes.append(agg_size)
                score_lw.append(np.mean(results[level][i, lw, c]))
                score.append(np.mean(results[level][i, m, c]))
                score_W.append(np.mean(results[level][i, m_W, c]))
                score_WF.append(np.mean(results[level][i, m_WF, c]))
            else:
                bottom_level_score_lw.append(np.mean(results[level][i, lw, c]))
                bottom_level_score.append(np.mean(results[level][i, m, c]))
                bottom_level_score_W.append(np.mean(results[level][i, m_W, c]))
                bottom_level_score_WF.append(np.mean(results[level][i, m_WF, c]))

    if 'KF' not in model and not no_L3:
        aggregate_sizes.append(1)
        score_lw.append(np.mean(bottom_level_score_lw))
        score.append(np.mean(bottom_level_score))
        score_W.append(np.mean(bottom_level_score_W))
        score_WF.append(np.mean(bottom_level_score_WF))

    aggregate_sizes = np.array(aggregate_sizes)
    score_lw = np.array(score_lw)
    score = np.array(score)
    score_W = np.array(score_W)
    score_WF = np.array(score_WF)

    skill = 100 * (1 - score / score_lw)
    skill_W = 100 * (1 - score_W / score_lw)
    skill_WF = 100 * (1 - score_WF / score_lw)

    return skill, skill_W, skill_WF, aggregate_sizes


def plot_aggregate_size_benchmark_skill(model, metric):
    skill, skill_W, skill_WF, aggregate_sizes = get_benchmark_skill_scores(model, metric)

    plt.figure(figsize=(3.5, 4))
    plt.plot([1, 2500], [0, 0], color='grey', linestyle='dashed')
    plt.scatter(
        aggregate_sizes,
        skill,
        label='',
        marker=MARKERS[0],
        color=get_color(model),
        edgecolors='none',
        alpha=1.0
    )
    plt.scatter(
        aggregate_sizes,
        skill_W,
        label='W',
        marker=MARKERS[1],
        color=get_color(model),
        edgecolors='none',
        alpha=0.75
    )
    plt.scatter(
        aggregate_sizes,
        skill_WF,
        label='WF',
        marker=MARKERS[2],
        color=get_color(model),
        edgecolors='none',
        alpha=0.5
    )
    plt.ylabel(f'$SS_{{\\mathrm{{{metric}}}}}$')
    plt.xlabel('\\# aggregated meters')
    plt.xscale('log')
    plt.xticks([1, 10, 100, 1000], ['1', '10', '100', '1000'])
    plt.title(model)
    _complete_plot(f"aggregate_size_benchmark_skill_{model}_{metric}", grid=False, legend=False)


def plot_coverage(levels=('L0', 'L1', 'L2'), models=None, name=None):
    _, info = collect_results()
    models = info['models'] if models is None else models

    p = np.linspace(0, 1, 101)
    plt.figure(figsize=(3.5, 4))
    for j, model in enumerate(models):
        pit = []
        for level in levels:
            if level == 'L3' and 'KF' in model:
                # No level 3 results for the KF model
                continue
            for c, cluster in enumerate(info['levels'][level]['clusters']):
                res = load_res(model, level, cluster)
                pit += list(np.ravel(res['PIT']))

        coverage = np.mean(p[:, np.newaxis] > np.array(pit)[np.newaxis], axis=1)
        plt.plot(
            p,
            coverage,
            linestyle=get_linestyle(model),
            color=get_color(model)
        )

    plt.plot(p, p, linestyle=':', color='grey')
    plt.ylabel('Coverage')
    plt.xlabel('Percentile')
    plt.xticks(p[::25])
    plt.title(models[0])
    _complete_plot(f"coverage{f'_{name}' if name is not None else ''}", grid=False, legend=False)


def plot_PIT_hist(model, level, cluster):
    res = load_res(model, level, cluster)
    pit = np.ravel(res['PIT'])

    plt.figure(figsize=(3.5, 3))
    plt.hist(pit, bins=20, density=True, color=get_color(model), label=model)
    plt.plot([0, 1], [1, 1], color='grey', label='$\\mathcal{U}(0, 1)$', linestyle='dashed')
    plt.ylim((0, 2.5))
    plt.ylabel('Relative frequency')
    plt.xlabel('PIT')
    plt.title(model)
    _complete_plot(f'PIT_hist_{get_file_name(model, level, cluster)}', grid=False, legend=False)


def plot_PIT_overview(model, num_cols=2):
    level_info = data_analysis.get_level_info()

    pits = []
    titles = []
    np.random.seed(42)
    for level in level_info.keys():
        if level == 'L3' and 'KF' in model:
            # No level 3 results for the KF model
            continue
        clusters = level_info[level]['clusters']
        for cluster in clusters if level != 'L3' else np.random.choice(clusters, size=10, replace=False):
            res = load_res(model, level, cluster)
            pits.append(np.ravel(res['PIT']))
            titles.append(f'{level}: {cluster}')

    num_rows = (len(pits) + num_cols - 1) // num_cols
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 6, num_rows * 4))

    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            idx = i * len(row) + j
            if idx >= len(pits):
                break
            col.hist(pits[idx], bins=50, density=True, color=get_color(model), label=titles[idx])
            col.plot([0, 1], [1, 1], color='grey', label='$\\mathcal{U}(0, 1)$', linestyle='dashed')
            col.set_ylim((0, 3))
            col.set_ylabel('Relative frequency')
            col.set_xlabel('PIT')
            col.legend()

    plt.tight_layout()
    plt.savefig(OUT_PATH + f'PIT_overview_{MODEL_NAMES[model][0]}{MODEL_NAMES[model][1]}_{SEASON}.pdf')
    plt.close()


def get_aggregated_base_forecast(model, level, cluster, base_level='L3'):
    _, info = collect_results()

    if base_level == 'L3' and 'KF' in model:
        # No level 3 results for the KF model
        model = 'KD-IC' + model.split('KF')[1]

    base_clusters = [cluster]
    child_levels = [f'L{l}' for l in range(int(level[1]) + 1, int(base_level[1]) + 1)]
    for child_level in child_levels:
        child_info = info['levels'][child_level]

        child_idx = []
        for parent in base_clusters:
            child_idx += list(np.where(np.array(child_info['parents']) == parent)[0])

        base_clusters = list(np.array(child_info['clusters'])[child_idx])

    base_forecasts = []
    for base_cluster in base_clusters:
        res = load_res(model, base_level, base_cluster)
        base_forecasts.append(res['mean'])
    aggregated_base_forecast = np.sum(base_forecasts, axis=0)

    return aggregated_base_forecast


def plot_coherency_errors_per_half_hour(model, level, cluster, base_level='L3'):
    res = load_res(model, level, cluster)
    a_hat = np.array(res['mean'])
    b_hat = get_aggregated_base_forecast(model, level, cluster, base_level)

    # Per half hour
    a_hat = a_hat.reshape(-1, S_D)
    b_hat = b_hat.reshape(-1, S_D)

    coherency_error = 100 * (a_hat - b_hat) / a_hat

    plt.figure(figsize=(10, 4))
    plt.boxplot(
        x=coherency_error,
        whis=[0, 100],
        widths=0.65,
        patch_artist=True,
        medianprops=dict(color='grey'),
        boxprops=dict(facecolor='white', color=get_color(model)),
        whiskerprops=dict(color=get_color(model), linestyle='dashed'),
        capprops=dict(color=get_color(model)),
        flierprops=dict(markeredgecolor=get_color(model), color=get_color(model))
    )

    plt.ylabel('Coherency error [\%]')
    plt.xlabel('Time of day')
    t = pd.date_range(start=res['t0'][0], periods=S_D, freq='30min')
    ticks = np.array(t[2:S_D:6].map(lambda x: x.strftime('%H:%M')))
    plt.xticks(np.arange(2, S_D, 6) + 1, ticks, rotation=0)
    plt.title(model)
    _complete_plot(
        f'coherency_errors_per_half_hour_{get_file_name(model, level, cluster)}_{base_level}',
        legend=False,
        grid=False
    )


def plot_coherency_errors(level, cluster, base_level='L3'):
    _, info = collect_results()

    plt.figure(figsize=(10, 4))
    for m, model in enumerate(info['models']):
        if model == 'LW':
            continue

        res = load_res(model, level, cluster)
        a_hat = np.array(res['mean']).ravel()
        b_hat = get_aggregated_base_forecast(model, level, cluster, base_level).ravel()

        coherency_error = 100 * (a_hat - b_hat) / a_hat

        plt.boxplot(
            x=coherency_error,
            positions=[m + 1],
            labels=[model.split('(')[-1].split(')')[0]],
            whis=[0, 100],
            widths=0.65,
            patch_artist=True,
            medianprops=dict(color='grey'),
            boxprops=dict(facecolor='white', color=get_color(model)),
            whiskerprops=dict(color=get_color(model), linestyle='dashed'),
            capprops=dict(color=get_color(model)),
            flierprops=dict(markeredgecolor=get_color(model), color=get_color(model))
        )

    plt.ylabel('Coherency error [\%]')
    plt.xlabel('Model')
    plt.grid(axis='y')
    _complete_plot(
        f'coherency_errors_{level}_{cluster}_{base_level}',
        legend=False,
        grid=False
    )


def plot_forecast(model, level, cluster, number):
    res = load_res(model, level, cluster)

    p_05 = res['p05'][number]
    p_25 = res['p25'][number]
    p_50 = res['p50'][number]
    p_75 = res['p75'][number]
    p_95 = res['p95'][number]

    t0 = res['t0'][number]
    t = pd.date_range(start=t0, periods=len(p_50), freq='30min')
    y = data_analysis.get_observations_at(level, cluster, t)

    plt.figure(figsize=(10, 3.5))

    # Point forecast
    plt.plot(
        np.arange(len(t)),
        p_50,
        color=get_color(model),
        linewidth=1,
        label='Median forecast'
    )

    # 50% and 90% confidence intervals
    plt.fill_between(
        np.arange(len(t)),
        p_25,
        p_75,
        alpha=0.4,
        color=get_color(model),
        edgecolor='none',
        label='50\,\%'
    )
    plt.fill_between(
        np.arange(len(t)),
        p_05,
        p_95,
        alpha=0.25,
        color=get_color(model),
        edgecolor='none',
        label='90\,\%'
    )

    # Observations
    plt.scatter(
        np.arange(len(t)),
        y,
        color='grey',
        label='Observations',
        s=7
    )

    # Axes
    plt.ylabel('Energy [kWh]')
    plt.title(model)
    ticks = np.array(t[::S_D].map(lambda x: x.strftime('%a, %H:%M')))
    ticks[0] = t[0].strftime('%a, %H:%M\n%b %d, %Y')
    plt.xticks(np.arange(0, len(t), S_D), ticks, rotation=0)

    _complete_plot(f'forecast_{get_file_name(model, level, cluster)}_number{number}', legend=False, grid=False)


def plot_forecast1d(model, level, cluster, number):
    res = load_res(model, level, cluster)

    p_05 = res['p05'][number][:S_D]
    p_25 = res['p25'][number][:S_D]
    p_50 = res['p50'][number][:S_D]
    p_75 = res['p75'][number][:S_D]
    p_95 = res['p95'][number][:S_D]

    t0 = res['t0'][number][:S_D]
    t = pd.date_range(start=t0, periods=len(p_50), freq='30min')
    y = data_analysis.get_observations_at(level, cluster, t)

    plt.figure(figsize=(5, 4))

    # Point forecast
    plt.plot(
        np.arange(len(t)),
        p_50,
        color=get_color(model),
        linewidth=1,
        label='Median forecast'
    )

    # 50% and 90% confidence intervals
    plt.fill_between(
        np.arange(len(t)),
        p_25,
        p_75,
        alpha=0.4,
        color=get_color(model),
        edgecolor='none',
        label='50\,\%'
    )
    plt.fill_between(
        np.arange(len(t)),
        p_05,
        p_95,
        alpha=0.25,
        color=get_color(model),
        edgecolor='none',
        label='90\,\%'
    )

    # Observations
    plt.scatter(
        np.arange(len(t)),
        y,
        color='grey',
        label='Observations',
        s=7
    )

    # Axes
    plt.ylabel('Energy [kWh]')
    plt.title(f'{model}, {level}')
    plt.xlabel('Time of day')
    ticks = np.array(t[2:S_D:10].map(lambda x: x.strftime('%H:%M')))
    plt.xticks(np.arange(2, S_D, 10) + 1, ticks, rotation=0)
    _complete_plot(f'forecast1d_{get_file_name(model, level, cluster)}_number{number}', legend=False, grid=False)


def plot_household_level_weather_effect(
        metric, model,
        demographics=True,
        lags=None,
        weather_vars=None
):
    level = 'L3'
    results, info = collect_results()

    t = data_analysis.energy_df.index
    clusters = info['levels'][level]['clusters']

    i = info['metrics'].index(metric)

    m = info['models'].index(model)
    m_W = info['models'].index(model + '(+W)')
    m_WF = info['models'].index(model + '(+WF)')

    score = results[level][i, m]
    score_W = results[level][i, m_W]
    score_WF = results[level][i, m_WF]

    score = np.mean(score, axis=1)
    score_W = np.mean(score_W, axis=1)
    score_WF = np.mean(score_WF, axis=1)

    skill_W = 100 * (1 - score_W / score)
    skill_WF = 100 * (1 - score_WF / score)

    # Calculate overall percentage improved
    print(f'Percentage improved (W): {np.mean(score_W < score)}')
    print(f'Percentage improved (WF): {np.mean(score_WF < score)}')
    print()

    if demographics:
        # Calculate percentage improved per category/group
        parents = np.array(info['levels'][level]['parents'])
        grandparents = []
        for parent in parents:
            parent_idx = info['levels']['L2']['clusters'].index(parent)
            grandparents.append(info['levels']['L2']['parents'][parent_idx])
        grandparents = np.array(grandparents)

        p_map_W = {}
        p_map_WF = {}
        for p in np.unique(parents):
            p_idx = np.where(parents == p)[0]
            p_map_W[p[-1]] = np.mean(score_W[p_idx] < score[p_idx])
            p_map_WF[p[-1]] = np.mean(score_WF[p_idx] < score[p_idx])

        plt.figure(figsize=(6, 4))
        plt.bar(p_map_W.keys(), p_map_W.values(), color=get_color(model))
        plt.ylabel('Percentage improved')
        plt.xlabel('Acorn group')
        plt.title('Actual weather')
        _complete_plot(f'household_group_{metric}_{model}_W', legend=False, grid=False)

        plt.figure(figsize=(6, 4))
        plt.bar(p_map_WF.keys(), p_map_WF.values(), color=get_color(model))
        plt.ylabel('Percentage improved')
        plt.xlabel('Acorn group')
        plt.title('Weather forecast')
        _complete_plot(f'household_group_{metric}_{model}_WF', legend=False, grid=False)

        gp_map_W = {}
        gp_map_WF = {}
        for gp in np.unique(grandparents):
            gp_idx = np.where(grandparents == gp)[0]
            gp_map_W[gp] = np.mean(score_W[gp_idx] < score[gp_idx])
            gp_map_WF[gp] = np.mean(score_WF[gp_idx] < score[gp_idx])

        plt.figure(figsize=(6, 4))
        plt.bar(gp_map_W.keys(), gp_map_W.values(), color=get_color(model))
        plt.ylabel('Percentage improved')
        plt.xlabel('Acorn category')
        plt.title('Actual weather')
        _complete_plot(f'household_category_{metric}_{model}_W', legend=False, grid=False)

        plt.figure(figsize=(6, 4))
        plt.bar(gp_map_WF.keys(), gp_map_WF.values(), color=get_color(model))
        plt.ylabel('Percentage improved')
        plt.xlabel('Acorn category')
        plt.title('Weather forecast')
        _complete_plot(f'household_category_{metric}_{model}_WF', legend=False, grid=False)

    # Plot correlation of autocorrelation and score difference
    if lags is None:
        lags = []
    for lag in lags:
        auto_corr = []
        for cluster in clusters:
            y = data_analysis.get_observations_at(level, cluster, t)
            auto_corr.append(data_analysis.autocorrelation(y, lag=lag))

        plt.figure(figsize=(6, 4))
        plt.scatter(
            auto_corr,
            skill_W,
            color=get_color(model),
            s=1,
            alpha=0.8,
            label=f'corr = {data_analysis.correlation(auto_corr, skill_W):.3f}'
        )
        plt.ylabel(f'$SS_{{\\mathrm{{{metric}}}}}$')
        plt.xlabel(f'Autocorrelation (Lag = {lag})')
        plt.title('Actual weather')
        _complete_plot(f'household_autocorrelation{lag}_{metric}_{model}_W', legend=True, grid=True)

        plt.figure(figsize=(6, 4))
        plt.scatter(
            auto_corr,
            skill_WF,
            color=get_color(model),
            s=1,
            alpha=0.8,
            label=f'corr = {data_analysis.correlation(auto_corr, skill_WF):.3f}'
        )
        plt.ylabel(f'$SS_{{\\mathrm{{{metric}}}}}$')
        plt.xlabel(f'Autocorrelation (Lag = {lag})')
        plt.title('Weather forecast')
        _complete_plot(f'household_autocorrelation{lag}_{metric}_{model}_WF', legend=True, grid=True)

    # Plot correlation of weather variable correlation and score difference
    if weather_vars is None:
        weather_vars = []
    for var_name in weather_vars:
        var = data_analysis.daily(np.array(data_analysis.weather_df.loc[t, var_name]), reduce=True)
        var_corr = []
        for cluster in clusters:
            y = data_analysis.daily(np.array(data_analysis.get_observations_at(level, cluster, t)), reduce=True)
            var_corr.append(data_analysis.correlation(var, y) ** 2)

        plt.figure(figsize=(6, 4))
        plt.scatter(
            var_corr,
            skill_W,
            color=get_color(model),
            s=1,
            alpha=0.8,
            label=f'corr = {data_analysis.correlation(var_corr, skill_W):.3f}'
        )
        plt.ylabel(f'$SS_{{\\mathrm{{{metric}}}}}$')
        plt.xlabel(f'Squared {var_name.replace("_", " ")} correlation [$R^2$]')
        plt.title('Actual weather')
        _complete_plot(f'household_{var_name}_{metric}_{model}_W', legend=True, grid=True)

        plt.figure(figsize=(6, 4))
        plt.scatter(
            var_corr,
            skill_WF,
            color=get_color(model),
            s=1,
            alpha=0.8,
            label=f'corr = {data_analysis.correlation(var_corr, skill_WF):.3f}'
        )
        plt.ylabel(f'$SS_{{\\mathrm{{{metric}}}}}$')
        plt.xlabel(f'Squared {var_name.replace("_", " ")} correlation [$R^2$]')
        plt.title('Weather forecast')
        _complete_plot(f'household_{var_name}_{metric}_{model}_WF', legend=True, grid=True)


def post_hoc_analysis(metric, models=('KF', 'KD-IC', 'DeepAR', 'LWR'), L3=True):
    results, info = collect_results()

    i = info['metrics'].index(metric)

    better_with_W = np.zeros((len(models), 2500 if L3 else 22))
    better_with_WF = np.zeros((len(models), 2500 if L3 else 22))
    for j, model in enumerate(models):
        if L3 and 'KF' in model:
            continue

        m = info['models'].index(model)
        m_W = info['models'].index(model + '(+W)')
        m_WF = info['models'].index(model + '(+WF)')

        if L3:
            score = results['L3'][i, m]
            score_W = results['L3'][i, m_W]
            score_WF = results['L3'][i, m_WF]
        else:
            score = np.vstack([results['L0'][i, m], results['L1'][i, m], results['L2'][i, m]])
            score_W = np.vstack([results['L0'][i, m_W], results['L1'][i, m_W], results['L2'][i, m_W]])
            score_WF = np.vstack([results['L0'][i, m_WF], results['L1'][i, m_WF], results['L2'][i, m_WF]])

        # Check if the scores are consistent per week
        # (i.e whether the model with weather is consistently better or worse for each forecast per time series)
        better_with_W_per_forecast = score_W < score
        forecast_consistency_W = np.mean(better_with_W_per_forecast, axis=1)
        forecast_consistency_W = np.mean(np.maximum(1 - forecast_consistency_W, forecast_consistency_W))

        better_with_WF_per_forecast = score_WF < score
        forecast_consistency_WF = np.mean(better_with_WF_per_forecast, axis=1)
        forecast_consistency_WF = np.mean(np.maximum(1 - forecast_consistency_WF, forecast_consistency_WF))

        score = np.mean(score, axis=1)
        score_W = np.mean(score_W, axis=1)
        score_WF = np.mean(score_WF, axis=1)

        post_hoc_score_W = np.mean(np.minimum(score, score_W))
        post_hoc_score_WF = np.mean(np.minimum(score, score_WF))

        better_with_W[j] = score_W < score
        better_with_WF[j] = score_WF < score
        percentage_better_with_W = np.mean(better_with_W[j])
        percentage_better_with_WF = np.mean(better_with_WF[j])

        consistency_W_v_WF = np.mean(better_with_W[j] == better_with_WF[j])

        score = np.mean(score)
        score_W = np.mean(score_W)
        score_WF = np.mean(score_WF)

        skill_W = 100 * (1 - post_hoc_score_W / score)
        skill_WF = 100 * (1 - post_hoc_score_WF / score)

        print(model)
        print('======')
        print(f'{metric} = {score:.2f}')
        print()
        print(f'{metric}(W) = {score_W:.2f}')
        print(f'Post-hoc {metric}(W) = {post_hoc_score_W:.2f}')
        print(f'Post-hoc SS_{metric}(W) = {skill_W:.2f}')
        print(f'Percentage_improved(W) = {100 * percentage_better_with_W:.2f}')
        print(f'Consistency_across_forecasts(W): {100 * forecast_consistency_W:.2f}')
        print()
        print(f'{metric}(WF) = {score_WF:.2f}')
        print(f'Post-hoc {metric}(WF) = {post_hoc_score_WF:.2f}')
        print(f'Post-hoc SS_{metric}(WF) = {skill_WF:.2f}')
        print(f'Percentage_improved(WF) = {100 * percentage_better_with_WF:.2f}')
        print(f'Consistency_across_forecasts(WF): {100 * forecast_consistency_WF:.2f}')
        print()
        print(f'Consistency(W vs. WF): {100 * consistency_W_v_WF:.2f}')
        print()
        print()

    if L3:
        consistency_W = np.mean(better_with_W[1] == better_with_W[2])
        consistency_WF = np.mean(better_with_WF[1] == better_with_WF[2])

        print('Consistency across models')
        print('==========================')
        print(f'Consistency(W) = {100 * consistency_W:.2f}')
        print(f'Consistency(WF) = {100 * consistency_WF:.2f}')
