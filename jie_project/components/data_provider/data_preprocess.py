import datetime
import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class Dataset_Preprocess_Building_Single(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='Hog_assembly_Colette.csv', scale=True, seasonal_patterns=None):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        self.data_set_type = data_path.split('.')[0]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date).apply(str)
        self.data_stamp = df_stamp['date'].values
        self.data_stamp = [str(x) for x in self.data_stamp]
        self.data_values = df_raw.iloc[:, 1:].values.astype(float)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.token_len
        data_segment = self.data_values[s_begin:s_end, -1]
        mean_val = np.mean(data_segment)
        std_val = np.std(data_segment)
        change_val = data_segment[-1] - data_segment[0] if len(data_segment) > 0 else 0
        start = datetime.datetime.strptime(self.data_stamp[s_begin], "%Y-%m-%d %H:%M:%S")
        end = (start + datetime.timedelta(hours=self.token_len - 1)).strftime("%Y-%m-%d %H:%M:%S")
        seq_x_mark = (
            f"time range of this sequence is from {self.data_stamp[s_begin]} to {end}\n"
            f"mean of this time series is {mean_val:.4f}\n"
            f"standard deviation of this time series is {std_val:.4f}\n"
            f"changed by {change_val:.4f}"
        )
        return seq_x_mark

    def __len__(self):
        return len(self.data_stamp)


class Dataset_Preprocess_Building_Multi(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='ETTh1.csv', scale=True, seasonal_patterns=None):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len

        self.flag = flag
        self.data_set_type = data_path.split('.')[0]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date).apply(str)
        self.data_stamp = df_stamp['date'].values
        self.data_stamp = [str(x) for x in self.data_stamp]
        self.data_values = df_raw.iloc[:, 1:].values.astype(float)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.token_len
        data_segment = self.data_values[s_begin:s_end, :]
        mean_vals = np.mean(data_segment, axis=0)
        std_vals = np.std(data_segment, axis=0)
        change_vals = data_segment[-1] - data_segment[0] if len(data_segment) > 0 else 0
        start = datetime.datetime.strptime(self.data_stamp[s_begin], "%Y-%m-%d %H:%M:%S")
        end = (start + datetime.timedelta(hours=self.token_len - 1)).strftime("%Y-%m-%d %H:%M:%S")
        seq_x_mark = (
            f"time range of this sequence is from {self.data_stamp[s_begin]} to {end}\n"
            f"below is the input statistics of this time series\n"
            f"- air temperature mean is {mean_vals[0]:.4f}. standard deviation is {std_vals[0]:.4f}. changed by {change_vals[0]:.4f}\n"
            f"- cloud coverage mean is {mean_vals[1]:.4f}. standard deviation is {std_vals[1]:.4f}. changed by {change_vals[1]:.4f}\n"
            f"- dew temperature mean is {mean_vals[2]:.4f}. standard deviation is {std_vals[2]:.4f}. changed by {change_vals[2]:.4f}\n"
            f"- seaLvl pressure mean is {mean_vals[3]:.4f}. standard deviation is {std_vals[3]:.4f}. changed by {change_vals[3]:.4f}\n"
            f"- wind direction mean is {mean_vals[4]:.4f}. standard deviation is {std_vals[4]:.4f}. changed by {change_vals[4]:.4f}\n"
            f"- wind speed mean is {mean_vals[5]:.4f}. standard deviation is {std_vals[5]:.4f}. changed by {change_vals[5]:.4f}\n"
            f"- electricity mean is {mean_vals[6]:.4f}. standard deviation is {std_vals[6]:.4f}. changed by {change_vals[6]:.4f}"
        )
        return seq_x_mark

    def __len__(self):
        return len(self.data_stamp)
