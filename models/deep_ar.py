import numpy as np
import datetime as dt
import time
from tqdm import tqdm
import os

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from models.forecast_model import ForecastModel
from distributions.empirical import Empirical
import utils


class DeepAR(nn.Module, ForecastModel, Empirical):
    def __init__(
            self, y, t, u=None, ID='', seed=0,
            prediction_length=192,
            num_samples=100,
            embedding_dim=None,
            num_layers=3,
            num_cells=40,
            epochs=50,
            batch_size=512
    ):
        nn.Module.__init__(self)
        ForecastModel.__init__(self, y, t, u, ID, seed=seed, global_model=True)

        # Fix the seed
        torch.manual_seed(seed)

        # Set the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Running on {self.device}.')

        self.seq_len = self.s_w + prediction_length
        self.seq_delta = self.s_d
        self.num_samples = num_samples
        self.embedding_dim = embedding_dim if embedding_dim is not None else int(np.sqrt(self.n))
        self.num_layers = num_layers
        self.num_cells = num_cells

        self.epochs = epochs
        self.batch_size = batch_size

        self.lags_seq = [1, 2, 3, self.s_d - 1, self.s_d, self.s_d + 1, self.s_w - 1, self.s_w, self.s_w + 1]
        num_features = 6 + len(self.lags_seq) + self.embedding_dim
        if u is not None:
            num_features += u.shape[1]

        self.embedding = nn.Embedding(self.n, self.embedding_dim)

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=num_cells,
            num_layers=num_layers,
            batch_first=True,
            # dropout=0.1
        )

        self.mu_fn = nn.Sequential(
            nn.Linear(num_cells, 1)
        )

        self.sigma2_fn = nn.Sequential(
            nn.Linear(num_cells, 1),
            nn.Softplus()
        )

        self.loss_fn = nn.GaussianNLLLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.to(self.device)

        self.X_mean = 0
        self.X_std = 1

        self.samples_y = np.zeros((self.num_samples, 0, self.n))
        # for i in range(self.n):
        #     self.results[i]['samples_y'] = []

        # Load a trained model if applicable
        self.model_path = os.path.join(self.get_out_dir(), '_state_dict_' + self.results[0]["ID"])
        if os.path.exists(self.model_path):
            self.create_features(self.t, self.u, fit=True)
            self.load_state_dict(torch.load(self.model_path, map_location=self.device))

    def __str__(self):
        return f'DeepAR{self.seed}'

    def forward(self, X, h=None):
        embeds = self.embedding(X[:, :, -1].int())
        X = torch.cat([X[:, :, :-1], embeds], dim=2)
        lstm_out, h = self.lstm(X, h)
        lstm_out = lstm_out.reshape(-1, self.num_cells)
        return self.mu_fn(lstm_out), self.sigma2_fn(lstm_out), h

    def create_features(self, t, u=None, fit=False):
        seconds = t.map(dt.datetime.timestamp).to_numpy(float)

        day = 24 * 60 * 60
        cos_d = np.cos((2 * np.pi / day) * seconds)
        sin_d = np.sin((2 * np.pi / day) * seconds)

        week = day * 7
        cos_w = np.cos((2 * np.pi / week) * seconds)
        sin_w = np.sin((2 * np.pi / week) * seconds)

        year = day * 365.2425
        cos_y = np.cos((2 * np.pi / year) * seconds)
        sin_y = np.sin((2 * np.pi / year) * seconds)

        X = np.vstack([cos_d, sin_d, cos_w, sin_w, cos_y, sin_y]).T
        if u is not None:
            X = np.hstack([X, u])

        if fit:
            self.X_mean = np.mean(X, axis=0, keepdims=True)
            self.X_std = np.std(X, axis=0, keepdims=True)
        X = utils.standardize(X, self.X_mean, self.X_std)

        return self.tensor(X)

    def create_labels(self, y):
        y = utils.interpolate_nans(y)
        return self.tensor(np.log(y / self.y_mean).T[..., np.newaxis])

    def create_input(self, t, u=None, y_lags=(), categories=None, fit=False, samples=False):
        X = self.create_features(t, u, fit=fit)

        if samples:
            X = X.repeat(self.num_samples, 1, 1)
        else:
            X = X.repeat(self.n, 1, 1)

        for y_lag in y_lags:
            X = torch.cat([X, y_lag], dim=2)

        categories = torch.unsqueeze(
            categories.repeat(len(t), self.num_samples if samples else 1).T,
            dim=2
        )
        X = torch.cat([X, categories], dim=2)

        return X

    def to_sequence(self, x):
        num_seq_per_series = (x.shape[1] - self.seq_len + self.seq_delta) // self.seq_delta
        seq = torch.zeros(self.n, num_seq_per_series, self.seq_len, x.shape[2])
        for i in range(num_seq_per_series):
            seq[:, i, :, :] = x[:, i * self.seq_delta:i * self.seq_delta + self.seq_len, :]
        return seq.reshape(-1, self.seq_len, x.shape[2])

    @staticmethod
    def tensor(x):
        return torch.from_numpy(x).float()

    @staticmethod
    def numpy(x):
        return x.cpu().detach().numpy().astype(float).squeeze()

    def train_val_split(self):
        split = int((len(self.t) * 0.2) // self.seq_delta) * self.seq_delta
        y_train, y_val = self.y[split:], self.y[:split]
        t_train, t_val = self.t[split:], self.t[:split]
        if self.u is not None:
            u_train, u_val = self.u[split:], self.u[:split]
        else:
            u_train, u_val = None, None

        return y_train, y_val, t_train, t_val, u_train, u_val

    def get_data_loader(self, y, t, u, fit=False):
        y = self.create_labels(y)
        y_lags = []
        for lag in self.lags_seq:
            y_lags.append(torch.hstack([y[:, :lag], y[:, :-lag]]))
        X = self.create_input(
            t,
            u,
            categories=self.tensor(np.arange(self.n)),
            y_lags=y_lags,
            fit=fit
        )

        data = TensorDataset(
            self.to_sequence(X),
            self.to_sequence(y)
        )

        return DataLoader(data, batch_size=self.batch_size, shuffle=fit)

    def fit(self):
        super().fit()
        start_time = time.time()

        y_train, y_val, t_train, t_val, u_train, u_val = self.train_val_split()
        train_dataloader = self.get_data_loader(y_train, t_train, u_train, fit=True)
        val_dataloader = self.get_data_loader(y_val, t_val, u_val, fit=False)

        train_loss = np.zeros(self.epochs)
        val_loss = np.zeros(self.epochs)
        best_val_loss = 0
        for epoch in range(self.epochs):
            # Train mode
            self.train()
            batch_cnt = 1
            with tqdm(train_dataloader, miniters=int(np.sqrt(len(train_dataloader)))) as batches:
                for X, y in batches:
                    X, y = X.to(self.device), y.to(self.device)
                    batches.set_description(f'Epoch {epoch + 1:>2}', refresh=False)

                    # Forward pass
                    mu_y, sigma2_y, _ = self(X)
                    loss = self.loss_fn(mu_y, y.reshape(-1, 1), sigma2_y)
                    train_loss[epoch] += (loss.item() - train_loss[epoch]) / batch_cnt
                    batch_cnt += 1

                    # Backprop
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    batches.set_postfix(loss=train_loss[epoch], refresh=False)

            val_loss[epoch] = self.val(val_dataloader)

            # Early stopping
            if val_loss[epoch] < best_val_loss:
                best_val_loss = val_loss[epoch]

                # Save the trained model
                torch.save(self.state_dict(), self.model_path)

        # After training, load the best model
        self.load_state_dict(torch.load(self.model_path, map_location=self.device))

        fit_time = time.time() - start_time
        for i in range(self.n):
            self.results[i]['fit_time'] = fit_time / self.n
            self.results[i]['train_loss'] = train_loss.tolist()
            self.results[i]['val_loss'] = val_loss.tolist()

    def val(self, val_dataloader):
        # Eval mode
        self.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_dataloader:
                X, y = X.to(self.device), y.to(self.device)

                # Forward pass
                mu_y, sigma2_y, _ = self(X)
                loss = self.loss_fn(mu_y, y.reshape(-1, 1), sigma2_y)
                val_loss += loss.item()

        return val_loss / len(val_dataloader)

    def sample_per_sample(self, h, y, t, u):
        prediction_length = len(t)
        samples_y = np.zeros((self.num_samples, prediction_length, self.n))

        for sample in tqdm(range(self.num_samples), miniters=int(np.sqrt(self.num_samples))):
            h_tilde = (
                h[0].detach().clone(),
                h[1].detach().clone()
            )
            y_tilde = y.detach().clone()

            for step in range(prediction_length):
                y_lags = []
                for lag in self.lags_seq:
                    y_lags.append(torch.unsqueeze(y_tilde[:, -lag], dim=1))
                X = self.create_input(
                    t[step:step + 1],
                    u[step:step + 1] if u is not None else None,
                    categories=self.tensor(np.arange(self.n)),
                    y_lags=y_lags
                )

                mu_y, sigma2_y, h_tilde = self(X.to(self.device), h_tilde)
                y_tilde = torch.hstack([
                    y_tilde,
                    torch.unsqueeze(torch.normal(mu_y, torch.sqrt(sigma2_y)), dim=1).cpu()
                ])

            samples_y[sample] = np.exp(self.numpy(y_tilde[:, -prediction_length:]).T) * self.y_mean[np.newaxis]

        return samples_y

    def sample_per_time_series(self, h, y, t, u):
        prediction_length = len(t)
        samples_y = np.zeros((self.num_samples, prediction_length, self.n))

        for i in tqdm(range(self.n), miniters=int(np.sqrt(self.n))):
            h_tilde = (
                h[0][:, i:i + 1].repeat(1, self.num_samples, 1),
                h[1][:, i:i + 1].repeat(1, self.num_samples, 1)
            )
            y_tilde = y[i, -self.lags_seq[-1]:].repeat(self.num_samples, 1, 1)

            for step in range(prediction_length):
                y_lags = []
                for lag in self.lags_seq:
                    y_lags.append(torch.unsqueeze(y_tilde[:, -lag], dim=1))
                X = self.create_input(
                    t[step:step + 1],
                    u[step:step + 1] if u is not None else None,
                    categories=self.tensor(np.array(i)),
                    y_lags=y_lags,
                    samples=True
                )

                mu_y, sigma2_y, h_tilde = self(X.to(self.device), h_tilde)
                y_tilde = torch.hstack([
                    y_tilde,
                    torch.unsqueeze(torch.normal(mu_y, torch.sqrt(sigma2_y)), dim=1).cpu()
                ])

            samples_y[:, :, i] = np.exp(self.numpy(y_tilde[:, -prediction_length:])) * self.y_mean[i]

        return samples_y

    def predict(self, t, u=None):
        if super().predict(t, u):
            return
        start_time = time.time()

        conditioning_length = self.seq_len - len(t)
        y = self.create_labels(self.y[-(conditioning_length + self.lags_seq[-1]):])
        y_lags = []
        for lag in self.lags_seq:
            y_lags.append(y[:, -(conditioning_length + lag):-lag])
        X = self.create_input(
            self.t[-conditioning_length:],
            self.u[-conditioning_length:] if u is not None else None,
            categories=self.tensor(np.arange(self.n)),
            y_lags=y_lags
        )

        # Eval mode
        self.eval()
        with torch.no_grad():
            _, _, h = self(X.to(self.device))

            # Decide what to put in the batch dimensions during sampling to increase parallelization
            if self.n > self.num_samples:
                samples_y = self.sample_per_sample(h, y, t, u)
            else:
                samples_y = self.sample_per_time_series(h, y, t, u)

        self.samples_y = np.hstack([self.samples_y, samples_y])

        prediction_time = time.time() - start_time
        for i in range(self.n):
            # self.results[i]['samples_y'].append(samples_y[:, :, i].tolist())
            self.results[i]['prediction_time'].append(prediction_time / self.n)

    def get_mean(self, t):
        super().get_mean(t)

        idx = self.idx(t)
        return self.mean(self.samples_y[:, idx])

    def get_var(self, t):
        super().get_var(t)

        idx = self.idx(t)
        return self.var(self.samples_y[:, idx])

    def get_percentile(self, p, t):
        super().get_percentile(p, t)

        idx = self.idx(t)
        return self.percentile(p, self.samples_y[:, idx])

    def get_pit(self, y_true, t):
        super().get_pit(y_true, t)

        idx = self.idx(t)
        return self.cdf(y_true, self.samples_y[:, idx])

    def get_crps(self, y_true, t):
        super().get_crps(y_true, t)

        idx = self.idx(t)
        return self.crps(y_true, self.samples_y[:, idx])
