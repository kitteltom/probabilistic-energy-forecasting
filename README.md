# Probabilistic forecasting of smart meter time series

This repository contains the python code for my master thesis with the title: **Hierarchical probabilistic forecasting of smart meter time series using weather input**. 

## Abstract

The operation of sustainable energy systems requires accurate probabilistic forecasts of electricity demand on various hierarchy levels in the energy system — ranging from individual households to cities and entire countries. Weather variables, such as temperature and dew point, are strongly correlated with electricity demand on higher levels in the hierarchy (e.g. city level) but seem to have little effect on lower levels (e.g. household level). Therefore, in this thesis, we investigate at which point in the hierarchy — constructed of 2500 individual smart meter time series from London households — weather input is beneficial to four day ahead forecasts of electricity demand. More specifically, we separately analyze the influence of actual weather data and weather forecasts. To that end, we implement three probabilistic forecasting models from the literature and, if necessary, adjust the models to optionally utilize weather data. The first model is based on double seasonal Holt-Winters-Taylor Exponential Smoothing [[5]](#5), it computes forecasts with the Kalman Filter [[4]](#4) in closed form and can include weather variables through mean adjustments. The second model, called KD-IC [[1]](#1), uses Kernel Density Estimation to create non-parametric density forecasts and can be conditioned on temperature through a kernel [[2]](#2). The third model is an autoregressive Recurrent Neural Network (DeepAR [[3]](#3)) that optionally takes weather variables as network input and generates probabilistic forecasts by recursively computing sample paths. Our results indicate that actual weather data improves the forecasting performance of the Kalman Filter and DeepAR models on most hierarchy levels. In particular, our evaluation shows a strong correlation, of the score difference between the models without weather input and with actual weather input, and the number of aggregated smart meters. Nevertheless, weather forecast input rarely leads to competitive forecasting performance and is in many cases even detrimental to the electricity forecast.

## Forecast examples

The following figures contain four day ahead forecasts by the Kalman Filter, KD-IC and DeepAR model with weather input (W) for a time series consisting of 56 aggregated smart meters. The gray points are the actual observations, the solid line denotes the median forecast and the areas around the line denote the 50% and 90% confidence intervals.

![Kalman Filter](forecast_examples/KalmanFilter_W.pdf)

![KD-IC](forecast_examples/KD-IC_W.pdf)

![DeepAR](forecast_examples/DeepAR_W.pdf)

## How to use

This repository does not provide ready-to-use code for probabilistic energy forecasting. It is rather meant as 

The minimal requirements for executing the models and evaluation scripts are listed in `requirements.txt`. Everything was written using Python 3.7.9.

## Project structure

```bash
.
├── distributions
│   ├── distribution.py
│   ├── empirical.py
│   ├── log_normal.py
│   └── non_parametric.py
├── eval
│   ├── data_analysis.py
│   └── evaluation.py
├── forecast_examples
│   ├── DeepAR_W.pdf
│   ├── KD-IC_W.pdf
│   └── KalmanFilter_W.pdf
├── main.py
├── models
│   ├── deep_ar.py
│   ├── forecast_model.py
│   ├── kalman_filter.py
│   ├── kd_ic.py
│   ├── last_week.py
│   └── log_normal_ic.py
└── utils.py
```

## References

<a id="1">[1]</a> Arora, S. and Taylor, J. W. (2016). “Forecasting electricity smart meter data using conditional kernel density estimation”. In: Omega 59, pp. 47–59.

<a id="2">[2]</a> Haben, S. and Giasemidis, G. (2016). “A hybrid model of kernel density estimation and quantile regression for GEFCom2014 probabilistic load forecasting”. In: International Journal of Forecasting 32.3, pp. 1017–1022.

<a id="3">[3]</a> Salinas, D., Flunkert, V., Gasthaus, J., and Januschowski, T. (2020). “DeepAR: Probabilistic forecasting with autoregressive recurrent networks”. In: International Journal of Forecasting 36.3, pp. 1181–1191.

<a id="4">[4]</a> Särkkä, S. (2013). Bayesian filtering and smoothing. Cambridge University Press.

<a id="5">[5]</a> Taylor, J. W. (2010). “Exponentially weighted methods for forecasting intraday time series with multiple seasonal cycles”. In: International Journal of Forecasting 26.4, pp. 627–646.
