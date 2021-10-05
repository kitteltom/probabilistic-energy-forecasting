import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import datetime as dt

import main
from eval import data_analysis

# LaTeX settings
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'sans-serif': ['lmodern'], 'size': 18})
plt.rc('axes', **{'titlesize': 18, 'labelsize': 18})

# Constants
JSON_PATH = './out/'
OUT_PATH = './out/'
MODEL_NAMES = {
    'KF': ('KalmanFilter', ''),
    'KF(+W)': ('KalmanFilter', '_W'),
    'KF(+WF)': ('KalmanFilter', '_WF'),
    'KD-IC': ('KD-IC', ''),
    'KD-IC(+W)': ('KD-IC', '_W'),
    'KD-IC(+WF)': ('KD-IC', '_WF'),
    'LN-IC': ('LogNormal-IC', ''),
    'LN-IC(+W)': ('LogNormal-IC', '_W'),
    'LN-IC(+WF)': ('LogNormal-IC', '_WF'),
    'DeepAR': ('DeepAR', ''),
    'DeepAR(+W)': ('DeepAR', '_W'),
    'DeepAR(+WF)': ('DeepAR', '_WF'),
    'LW': ('LastWeek', '')
}
MAIN_SEED = '42'
DECIMALS = 2
COLORS = ('C0', 'C1', 'C3', 'C9', 'C7')
MARKERS = ('o', 'X', 'v', 'd', 'p')
LINESTYLES = ('solid', 'dashed', 'dashdot')
S_D = 48
S_W = 7 * S_D


def get_file_name(model, level, cluster, seed=''):
    return f'{MODEL_NAMES[model][0]}{seed}_{level}_{cluster}{MODEL_NAMES[model][1]}'


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
        models=('KF', 'KF(+W)', 'KF(+WF)',
                'KD-IC', 'KD-IC(+W)', 'KD-IC(+WF)',
                'DeepAR', 'DeepAR(+W)', 'DeepAR(+WF)',
                'LW'),
        seeds=(0, 1, 2, 3, 4),
        forecast_reps=28,
        save_results_with_info=True
):
    results_path = os.path.join(JSON_PATH, 'results_with_info.npy')
    if os.path.isfile(results_path):
        results_with_info = np.load(results_path, allow_pickle=True)
        return results_with_info[0], results_with_info[1]

    results = {}
    level_info = data_analysis.get_level_info(levels)
    for level in levels:
        clusters = level_info[level]['clusters']

        # Create results array
        results[level] = np.empty((len(metrics), len(models), len(clusters), forecast_reps))
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
        'reps': forecast_reps
    }

    if save_results_with_info:
        np.save(results_path, (results, info), allow_pickle=True)

    return results, info


def collect_results_per_tstp(
        levels=('L0', 'L1', 'L2'),
        metrics=('rMAE', 'rRMSE', 'rCRPS'),
        models=('KF', 'KF(+W)', 'KF(+WF)',
                'KD-IC', 'KD-IC(+W)', 'KD-IC(+WF)',
                'DeepAR', 'DeepAR(+W)', 'DeepAR(+WF)',
                'LW'),
        seeds=(0, 1, 2, 3, 4),
        forecast_reps=28,
        horizon=192,
        save_results_per_tstp_with_info=True
):
    results_path = os.path.join(JSON_PATH, 'results_per_tstp_with_info.npy')
    if os.path.isfile(results_path):
        results_with_info = np.load(results_path, allow_pickle=True)
        return results_with_info[0], results_with_info[1]

    results = {}
    level_info = data_analysis.get_level_info(levels)
    t_train, t_val = main.train_val_split(data_analysis.energy_df.index)
    for level in levels:
        clusters = level_info[level]['clusters']

        # Create results array
        results[level] = np.empty((len(seeds), len(metrics), len(models), len(clusters), forecast_reps, horizon))
        results[level][:] = np.nan
        level_info[level]['y_mean'] = []
        for c, cluster in enumerate(clusters):
            level_info[level]['y_mean'].append(
                np.nanmean(data_analysis.get_observations_at(level, cluster, t_train))
            )
            y_true = data_analysis.get_observations_at(level, cluster, t_val).reshape(forecast_reps, horizon)
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
        'reps': forecast_reps,
        'horizon': horizon
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


def create_runtime_df(models=('KF', 'KD-IC', 'DeepAR', 'LW'), with_std=False, to_LaTeX=True):
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
    elif 'LW' in model:
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
    plt.savefig(OUT_PATH + f'{name}.pdf', bbox_inches='tight')
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
    _complete_plot(f'{get_file_name(model, level, cluster, seed)}_epoch_loss', grid=False)


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
        idx = np.arange(0, horizon * S_D)
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
    plt.ylim(6.95, 8.35)
    plt.ylabel(metric)
    plt.xlabel('Horizon')
    plt.xticks(np.arange(len(horizons)), np.array(horizons))
    plt.title(model)
    _complete_plot(f"{model}_{metric}_horizon", grid=False, legend=False)


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
    _complete_plot(f"{f'{name}_' if name is not None else ''}{metric}_reps", grid=False)


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
    _complete_plot(f"{f'{name}_' if name is not None else ''}{level}_{metric}_clusters")


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
    plt.yticks(np.arange(0, 70, 20))
    plt.xlabel('\\# aggregated meters')
    plt.xscale('log')
    plt.xticks([1, 10, 100, 1000], ['1', '10', '100', '1000'])
    _complete_plot(f"{f'{name}_' if name is not None else ''}{metric}_aggregate_size", grid=False)


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

    t_train = main.train_val_split(data_analysis.energy_df.index)[0]
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
    print(f'Correlation (W): {data_analysis.correlation(np.log(aggregate_sizes), skill_W):.3f}')
    print(f'Correlation (WF): {data_analysis.correlation(np.log(aggregate_sizes), skill_WF):.3f}')

    plt.figure(figsize=(3.5, 4))
    plt.plot([1, 2500], [0, 0], color='grey', linestyle='dashed')
    plt.scatter(
        aggregate_sizes,
        skill_W,
        label='W',
        marker=MARKERS[0],
        color=get_color(model),
        edgecolors='none'
    )
    plt.scatter(
        aggregate_sizes,
        skill_WF,
        label='WF',
        marker=MARKERS[1],
        color=get_color(model),
        edgecolors='none',
        alpha=0.5
    )
    plt.ylabel(f'$SS_{{\\mathrm{{{metric}}}}}$')
    plt.xlabel('\\# aggregated meters')
    plt.xscale('log')
    plt.xticks([1, 10, 100, 1000], ['1', '10', '100', '1000'])
    plt.title(model)
    _complete_plot(f"{model}_{metric}_aggregate_size_skill", grid=False, legend=False)


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
    _complete_plot(f'{model}_{metric}_temperature_correlation_skill', grid=False, legend=False)


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
    _complete_plot(f"{f'{name}_' if name is not None else ''}coverage", grid=False, legend=False)


def plot_PIT_hist(model, level, cluster):
    res = load_res(model, level, cluster)
    pit = np.ravel(res['PIT'])

    plt.figure(figsize=(3.5, 3))
    plt.hist(pit, bins=20, density=True, color=get_color(model), alpha=0.8, label=model)
    plt.plot([0, 1], [1, 1], color='grey', label='$\\mathcal{U}(0, 1)$', linestyle='dashed')
    plt.ylim((0, 2.5))
    plt.ylabel('Relative frequency')
    plt.xlabel('PIT')
    plt.title(model)
    _complete_plot(f'{get_file_name(model, level, cluster)}_PIT_hist', grid=False, legend=False)


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
            col.hist(pits[idx], bins=50, density=True, color=COLORS[0], alpha=0.8, label=titles[idx])
            col.plot([0, 1], [1, 1], color=COLORS[1], label='$\\mathcal{U}(0, 1)$')
            col.set_ylim((0, 3))
            col.set_ylabel('Relative frequency')
            col.set_xlabel('PIT')
            col.legend()

    plt.tight_layout()
    plt.savefig(OUT_PATH + f'{MODEL_NAMES[model][0]}{MODEL_NAMES[model][1]}_PIT_overview.pdf')
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
        f'{get_file_name(model, level, cluster)}_{base_level}_coherency_errors_per_half_hour',
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
        f'{level}_{cluster}_{base_level}_coherency_errors',
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

    _complete_plot(f'{get_file_name(model, level, cluster)}_number{number}_forecast', legend=False, grid=False)


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
    plt.title(f'{level}: {cluster}')
    plt.xlabel('Time of day')
    ticks = np.array(t[2:S_D:10].map(lambda x: x.strftime('%H:%M')))
    plt.xticks(np.arange(2, S_D, 10) + 1, ticks, rotation=0)
    _complete_plot(f'{get_file_name(model, level, cluster)}_number{number}_forecast1d', legend=False, grid=False)


def post_hoc_analysis(metric, models=('KD-IC', 'DeepAR')):
    results, info = collect_results()

    i = info['metrics'].index(metric)

    better_with_W = np.zeros((len(models), 2500))
    better_with_WF = np.zeros((len(models), 2500))
    for j, model in enumerate(models):
        m = info['models'].index(model)
        m_W = info['models'].index(model + '(+W)')
        m_WF = info['models'].index(model + '(+WF)')

        score = results['L3'][i, m]
        score_W = results['L3'][i, m_W]
        score_WF = results['L3'][i, m_WF]

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

        consistency_W_v_WF = np.mean(better_with_W[j] == better_with_WF[j])

        score = np.mean(score)
        score_W = np.mean(score_W)
        score_WF = np.mean(score_WF)

        skill_W = 100 * (1 - post_hoc_score_W / score_W)
        skill_WF = 100 * (1 - post_hoc_score_WF / score_WF)

        print(model)
        print('======')
        print(f'{metric} = {score:.2f}')
        print()
        print(f'{metric}(W) = {score_W:.2f}')
        print(f'Post-hoc {metric}(W) = {post_hoc_score_W:.2f}')
        print(f'Post-hoc SS_{metric}(W) = {skill_W:.2f}')
        print(f'Consistency_across_forecasts(W): {100 * forecast_consistency_W:.2f}')
        print()
        print(f'{metric}(WF) = {score_WF:.2f}')
        print(f'Post-hoc {metric}(WF) = {post_hoc_score_WF:.2f}')
        print(f'Post-hoc SS_{metric}(WF) = {skill_WF:.2f}')
        print(f'Consistency_across_forecasts(WF): {100 * forecast_consistency_WF:.2f}')
        print()
        print(f'Consistency(W vs. WF): {100 * consistency_W_v_WF:.2f}')
        print()
        print()

    consistency_W = np.mean(better_with_W[0] == better_with_W[1])
    consistency_WF = np.mean(better_with_WF[0] == better_with_WF[1])

    print('Consistency across models')
    print('==========================')
    print(f'Consistency(W) = {100 * consistency_W:.2f}')
    print(f'Consistency(WF) = {100 * consistency_WF:.2f}')
