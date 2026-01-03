"""
Data and Parameters
Src/paras.py
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

# Path
MODEL_NAME = "Resnet50"
RATE_CSV_PATH = f"D:/Coding/Python/DSCI/Data/{MODEL_NAME}_rates.csv"
ACC_CSV_PATH = f"D:/Coding/Python/DSCI/Data/{MODEL_NAME}_accs.csv"
LAYER_CSV_PATH = f"D:/Coding/Python/DSCI/Data/{MODEL_NAME}_layer_stats.csv"
RESULT_GA_PATH = "D:/Coding/Python/DSCI/Result/GA"
RESULT_PPO_PATH = "D:/Coding/Python/DSCI/Result/PPO"
RESULT_BF_PATH = "D:/Coding/Python/DSCI/Result/BF"

# User
NUM_USERS = 10

# Model
NUM_LAYERS = 128
EARLY_EXIT_LAYERS = [57, 103] # 1 9 12 18 9 1
NUM_EXIT_LAYERS = len(EARLY_EXIT_LAYERS)

csv_path = Path(f"D:/Coding/Python/DSCI/Data/{MODEL_NAME}_layer_stats.csv")
df = pd.read_csv(csv_path)
DATA_SIZE_LAYERS = df["num_bytes"].astype(int).tolist()
COMPUTE_SIZE_LAYERS = df["approx_flops"].astype(int).tolist()

# Compute
USER_FREQs = NUM_USERS * [2]    # 用户每人 2 GHz
EDGE_MAX_FREQ = 20.0            # 边缘服务器 10 GHz
CLOUD_MAX_FREQ = 50.0           # 云服务器 50 GHz

# Communicate
CHANNEL_GAINS_USERS = NUM_USERS * [2]   # 用户的信道增益
BANDWIDTH_EDGE = 10.0       # 边缘服务器的带宽 20 MHz
BANDWIDTH_CLOUD = 50.0      # 云服务器的带宽 50 MHz
BASE_STATION_POWER = 1.0    # 基站的发射功率 W
NOISE_POWER = 8e-11          # 高斯噪声 W


@dataclass
class Paras:
    n: int      # NUM_USERS: 终端用户数量
    m: int      # NUM_LAYERS: DNN模型层数
    E: list     # EARLY_EXIT_LAYERS: 早退层的集合
    D: list     # DATA_SIZE_LAYERS: 各层的输出数据大小
    C: list     # COMPUTE_SIZE_LAYERS: 各层的计算大小
    F_u: np.ndarray     # USER_FREQs: 每个用户的处理频率
    f_e_max: float      # EDGE_MAX_FREQ: 边缘服务器最大频率
    f_c_max: float      # CLOUD_MAX_FREQ: 云服务器最大频率
    H_u: np.ndarray     # BANDWIDTH_USERS: 每个用户的信道增益
    b_e: float          # BANDWIDTH_EDGE: 边缘服务器的带宽
    b_c: float          # BANDWIDTH_CLOUD: 云服务器的带宽
    G: float            # BASE_STATION_POWER: 基站的发射功率
    delta: float        # NOISE_POWER: 噪声功率
    alpha: float    # delay 所占权重
    beta: float     # accuracy 所占权重
