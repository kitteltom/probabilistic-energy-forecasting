import pandas as pd
import numpy as np
import datetime as dt
import argparse

from models.kalman_filter import KalmanFilter
from models.kd_ic import KDIC
from models.log_normal_ic import LogNormalIC
from models.deep_ar import DeepAR
from models.last_week import LastWeek

DATA_PATH = './data/smart_meters_london_refactored/'
TRAIN_WEEKS = 52
VAL_WEEKS = 16
TEST_WEEKS = 16


def train_val_split(t, val=False):
    """
    Splits the timestamps t into training and validation/test set, specified by TRAIN_WEEKS, VAL_WEEKS and TEST_WEEKS.
    If val is True, the train and validation timestamps are returned.
    Otherwise the train and test timestamps are returned.
    """
    t0_idx = np.where([hh.weekday() == 0 and hh.time() == dt.time(1, 0) for hh in t])[0]
    split_idx = TRAIN_WEEKS + (VAL_WEEKS if not val else 0)
    last_idx = TRAIN_WEEKS + VAL_WEEKS + (TEST_WEEKS if not val else 0)
    t_train = t[t0_idx[0]:t0_idx[split_idx]]
    t_val = t[t0_idx[split_idx]:t0_idx[last_idx]]

    return t_train, t_val


def main():
    """
    Reads the dataframes, optionally aggregates the time series, fits the specified model to the data
    and computes forecasts.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='KF',
                        help='The model to use for forecasting (KF, KD, LN, DeepAR, LW)')
    parser.add_argument("--level", nargs='+', default=[0], type=int,
                        help='List of levels in the hierarchy for which forecasts should be produced '
                             '(0 for the aggregated data, 1 for the ACORN categories, 2 for the ACORN groups, '
                             'and 3 for the household smart meter data)')
    parser.add_argument("--horizon", default=192, type=int, help='Forecast horizon in half-hours')
    parser.add_argument("--val", action="store_true", help='If set, the hold-out test set is not used')
    parser.add_argument("--fit", action="store_true", help='If set, train the parameters')
    parser.add_argument("--use_input", action="store_true", help='If set, weather input is used')
    parser.add_argument("--forecast", action="store_true",
                        help='If set, weather forecasts are used. Note: Only has an effect if --use_input is set too')
    parser.add_argument("--plot_mode", default='',
                        help="How to proceed with the figures. "
                             "Options are 'save', 'save_first', 'show', 'show_first', and '' for doing nothing")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for neural network models.")
    args = parser.parse_args()

    # Read the dataframes
    energy_df = pd.read_csv(DATA_PATH + 'energy_data.csv', index_col=0, parse_dates=True)
    demographic_df = pd.read_csv(DATA_PATH + 'demographic_data.csv', index_col=0)

    # Split the data (80% train, 20% validation)
    t_train, t_val = train_val_split(energy_df.index, args.val)
    forecast_reps = len(t_val) // args.horizon

    # Aggregate the data
    y_train = {}
    y_val = {}
    assert 0 <= min(args.level) and max(args.level) <= 3, "The level must be in range [0, 3]"
    if 0 in args.level:
        # Aggregate level
        count = len(demographic_df)
        y_train[(0, 'Agg')] = np.nanmean(energy_df.loc[t_train].to_numpy(float), axis=1) * count
        y_val[(0, 'Agg')] = np.nanmean(energy_df.loc[t_val].to_numpy(float), axis=1) * count

    if 1 in args.level:
        # Category level
        categories, cardinality = np.unique(demographic_df.acorn_category, return_counts=True)
        for category, count in zip(categories, cardinality):
            h_ids = demographic_df.loc[demographic_df.acorn_category == category].index
            y_train[(1, category)] = np.nanmean(energy_df.loc[t_train, h_ids].to_numpy(float), axis=1) * count
            y_val[(1, category)] = np.nanmean(energy_df.loc[t_val, h_ids].to_numpy(float), axis=1) * count

    if 2 in args.level:
        # Group level
        groups, cardinality = np.unique(demographic_df.acorn_group, return_counts=True)
        for group, count in zip(groups, cardinality):
            h_ids = demographic_df.loc[demographic_df.acorn_group == group].index
            y_train[(2, group)] = np.nanmean(energy_df.loc[t_train, h_ids].to_numpy(float), axis=1) * count
            y_val[(2, group)] = np.nanmean(energy_df.loc[t_val, h_ids].to_numpy(float), axis=1) * count

    if 3 in args.level:
        # Household level
        h_ids = demographic_df.index
        for h_id in h_ids:
            y_train[(3, h_id)] = energy_df.loc[t_train, h_id].to_numpy(float)
            y_val[(3, h_id)] = energy_df.loc[t_val, h_id].to_numpy(float)

    # Get weather data
    weather_variables = ['temperature', 'dew_point']
    if not args.forecast:
        weather_id = '_W'
        weather_df = pd.read_csv(DATA_PATH + 'weather_data.csv', index_col=0, parse_dates=True)
        u_train = weather_df.loc[t_train, weather_variables].to_numpy(float)
        u_val = weather_df.loc[t_val, weather_variables].to_numpy(float)
        u_val_predict = u_val
    else:
        weather_id = '_WF'
        weather_forecast1d_df = pd.read_csv(DATA_PATH + 'weather_forecast1d_data.csv', index_col=0, parse_dates=True)
        weather_forecast4d_df = pd.read_csv(DATA_PATH + 'weather_forecast4d_data.csv', index_col=0, parse_dates=True)
        u_train = weather_forecast1d_df.loc[t_train, weather_variables].to_numpy(float)
        u_val = weather_forecast1d_df.loc[t_val, weather_variables].to_numpy(float)
        if args.val:
            u_val_predict = u_val
        else:
            # Here the 4-day forecasts come into play
            u_val_predict = weather_forecast4d_df.loc[t_val, weather_variables].to_numpy(float)

    # Pick the model
    kwargs = {}
    if args.model == 'KD':
        ForecastModel = KDIC
    elif args.model == 'LN':
        ForecastModel = LogNormalIC
    elif args.model == 'DeepAR':
        ForecastModel = DeepAR
        kwargs["seed"] = args.seed
        kwargs["prediction_length"] = args.horizon
        if max(args.level) == 0:
            kwargs["num_samples"] = 200
            kwargs["num_layers"] = 2
            kwargs["num_cells"] = 20
            kwargs["batch_size"] = 64
        elif max(args.level) == 1:
            kwargs["num_samples"] = 200
            kwargs["num_layers"] = 2
            kwargs["num_cells"] = 30
            kwargs["batch_size"] = 64
        elif max(args.level) == 2:
            kwargs["num_samples"] = 200
            kwargs["num_layers"] = 2
            kwargs["num_cells"] = 40
            kwargs["batch_size"] = 64
    elif args.model == 'LW':
        ForecastModel = LastWeek
    else:
        ForecastModel = KalmanFilter

    if args.model == 'DeepAR':
        # Global model
        IDs = [f'L{ID[0]}_{ID[1]}{weather_id if args.use_input else ""}' for ID in y_train]
        kwargs['ID'] = IDs

        assert list(y_train.keys()) == list(y_val.keys())
        y_train = np.array(list(y_train.values())).T
        y_val = np.array(list(y_val.values())).T

        # Instantiate the model
        model = ForecastModel(y_train, t_train, u_train if args.use_input else None, **kwargs)

        # Train
        if args.fit:
            model.fit()

        # Forecast
        print('Evaluating...')
        for i in range(forecast_reps):
            idx = np.arange(i * args.horizon, (i + 1) * args.horizon)
            if args.use_input:
                model.predict(t_val[idx], u_val_predict[idx])
            else:
                model.predict(t_val[idx])

            model.evaluate(y_val[idx], t_val[idx])
            if args.plot_mode == 'save' or (args.plot_mode == 'save_first' and i == 0):
                model.plot_forecast(y_val[idx], t_val[idx], save_fig=True)
            elif args.plot_mode == 'show' or (args.plot_mode == 'show_first' and i == 0):
                model.plot_forecast(y_val[idx], t_val[idx], save_fig=False)

            if args.use_input:
                model.add_measurements(y_val[idx], t_val[idx], u_val[idx])
            else:
                model.add_measurements(y_val[idx], t_val[idx])

        # Save dict
        model.save_results()

    else:
        for ID in y_train:
            kwargs['ID'] = f'L{ID[0]}_{ID[1]}{weather_id if args.use_input else ""}'

            # Instantiate the model
            model = ForecastModel(y_train[ID], t_train, u_train if args.use_input else None, **kwargs)

            # Train
            if args.fit:
                model.fit()

            # Forecast
            for i in range(forecast_reps):
                idx = np.arange(i * args.horizon, (i + 1) * args.horizon)
                if args.use_input:
                    model.predict(t_val[idx], u_val_predict[idx])
                else:
                    model.predict(t_val[idx])

                model.evaluate(y_val[ID][idx], t_val[idx])
                if args.plot_mode == 'save' or (args.plot_mode == 'save_first' and i == 0):
                    model.plot_forecast(y_val[ID][idx], t_val[idx], save_fig=True)
                elif args.plot_mode == 'show' or (args.plot_mode == 'show_first' and i == 0):
                    model.plot_forecast(y_val[ID][idx], t_val[idx], save_fig=False)

                if args.use_input:
                    model.add_measurements(y_val[ID][idx], t_val[idx], u_val[idx])
                else:
                    model.add_measurements(y_val[ID][idx], t_val[idx])

            # Save dict
            model.save_results()

            # Scoring results
            print()
            print(kwargs['ID'])
            print('----------------')
            print(f'rCRPS:    {np.mean(model.results[0]["rCRPS"]):.4f}')
            print(f'MAPE:     {np.mean(model.results[0]["MAPE"]):.4f}')
            print(f'rMAE:     {np.mean(model.results[0]["rMAE"]):.4f}')
            print(f'rRMSE:    {np.mean(model.results[0]["rRMSE"]):.4f}')
            print()


if __name__ == '__main__':
    main()
