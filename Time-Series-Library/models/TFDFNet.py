import torch
import torch.nn as nn


def _config_value(configs, name, fallback):
    value = getattr(configs, name, None)
    return fallback if value is None else value


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series.
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block.
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Dense(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super(Dense, self).__init__()
        self.linear1 = nn.Linear(in_dim, in_dim)
        self.drop = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(in_dim, out_dim)

    def forward(self, input):
        hidden = self.linear1(input)
        hidden = self.drop(hidden)
        return self.linear2(hidden)


class Encoder(nn.Module):
    def __init__(self, in_channels=7, out_channels=7, kernel_size=1, stride=1, dropout=0.2):
        super(Encoder, self).__init__()
        hidden_channels = max(1, in_channels // 2)
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.max_pool1 = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)
        self.drop = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.max_pool2 = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)
        self.conv3 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.max_pool3 = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.max_pool3(x)
        x = self.drop(x)
        return x


class Decoder(nn.Module):
    def __init__(self, configs):
        super(Decoder, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        dropout = getattr(configs, "dropout", 0.2)
        out_channels = int(_config_value(configs, "out_channels", getattr(configs, "c_out", 1)))
        self.dense1 = nn.Sequential(
            Dense(self.seq_len, self.pred_len, dropout=dropout),
            Dense(self.pred_len, self.pred_len, dropout=dropout),
            Dense(self.pred_len, self.pred_len, dropout=dropout),
        )
        self.flatten = nn.Flatten()
        self.dense2 = nn.Sequential(
            Dense(self.seq_len, self.pred_len, dropout=dropout),
            Dense(self.pred_len, self.pred_len, dropout=dropout),
            Dense(self.pred_len, self.pred_len, dropout=dropout),
        )
        self.out_dim = out_channels + 2
        self.dense3 = nn.Sequential(
            Dense(self.pred_len * self.out_dim, self.pred_len, dropout=dropout),
            Dense(self.pred_len, self.pred_len, dropout=dropout),
            Dense(self.pred_len, self.pred_len, dropout=dropout),
        )

    def forward(self, seasonal, trend, enc_out):
        data = torch.cat([seasonal, trend], dim=2)
        target = self.dense1(data.permute(0, 2, 1))
        features = self.dense2(enc_out)
        fea_tar = torch.cat((target, features), dim=1)
        fea_tar = self.flatten(fea_tar)
        return self.dense3(fea_tar)


class Model(nn.Module):
    """
    TFDFNet adapted to the Time-Series-Library forecasting model API.
    The original model consumes target, exogenous features, and time marks
    separately; this wrapper derives them from x_enc and x_mark_enc.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        base_channels = int(_config_value(configs, "in_channels", getattr(configs, "enc_in", 1)))
        self.base_channels = max(1, base_channels)
        self.time_feature_dim = max(0, int(_config_value(configs, "time_feature_dim", 4)))
        self.target_index = self._infer_target_index(configs)

        kernel_size = int(_config_value(configs, "kernel_size", getattr(configs, "moving_avg", 25)))
        self.decompsition = series_decomp(kernel_size)

        self.out_channels = int(_config_value(configs, "out_channels", getattr(configs, "c_out", 1)))
        self.in_channels = self.base_channels + self.time_feature_dim
        dropout = getattr(configs, "dropout", 0.2)

        self.encoder = Encoder(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            dropout=dropout,
        )
        self.decoder = Decoder(configs)

    def _infer_target_index(self, configs):
        if hasattr(configs, "target_index"):
            return int(configs.target_index)
        if getattr(configs, "features", "MS") == "MS":
            return max(0, int(getattr(configs, "enc_in", self.base_channels)) - 1)
        return 0

    def _fit_channels(self, x, channels):
        if channels == 0:
            return x.new_zeros(x.shape[0], x.shape[1], 0)
        if x.shape[-1] == channels:
            return x
        if x.shape[-1] > channels:
            return x[:, :, :channels]
        pad = x.new_zeros(x.shape[0], x.shape[1], channels - x.shape[-1])
        return torch.cat([x, pad], dim=-1)

    def _split_inputs(self, x_enc, x_mark_enc):
        if x_enc.dim() != 3:
            raise ValueError("TFDFNet expects x_enc with shape [batch, seq_len, channels]")

        channels = x_enc.shape[-1]
        target_index = min(max(0, self.target_index), channels - 1)
        true_target = x_enc[:, :, target_index]

        if channels > 1:
            true_univariate = torch.cat(
                [x_enc[:, :, :target_index], x_enc[:, :, target_index + 1:]],
                dim=-1,
            )
        else:
            true_univariate = x_enc.new_zeros(x_enc.shape[0], x_enc.shape[1], 0)
        true_univariate = self._fit_channels(true_univariate, self.base_channels - 1)

        if x_mark_enc is None:
            true_time_mask = x_enc.new_zeros(x_enc.shape[0], x_enc.shape[1], self.time_feature_dim)
        else:
            true_time_mask = x_mark_enc.to(device=x_enc.device, dtype=x_enc.dtype)
            if true_time_mask.shape[1] > x_enc.shape[1]:
                true_time_mask = true_time_mask[:, :x_enc.shape[1], :]
            elif true_time_mask.shape[1] < x_enc.shape[1]:
                pad = x_enc.new_zeros(
                    x_enc.shape[0],
                    x_enc.shape[1] - true_time_mask.shape[1],
                    true_time_mask.shape[-1],
                )
                true_time_mask = torch.cat([true_time_mask, pad], dim=1)
            true_time_mask = self._fit_channels(true_time_mask, self.time_feature_dim)

        return true_target, true_univariate, true_time_mask

    def forecast(self, x_enc, x_mark_enc=None):
        true_target, true_univariate, true_time_mask = self._split_inputs(x_enc, x_mark_enc)
        true_target = torch.unsqueeze(true_target, dim=2)
        seasonal_init, trend_init = self.decompsition(true_target)
        enc_in = torch.cat((true_target, true_univariate, true_time_mask), dim=2)
        enc_out = self.encoder(enc_in.permute(0, 2, 1))
        dec_out = self.decoder(seasonal_init, trend_init, enc_out)
        return dec_out.unsqueeze(-1)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            return self.forecast(x_enc, x_mark_enc)
        raise NotImplementedError("TFDFNet supports forecasting tasks only.")
