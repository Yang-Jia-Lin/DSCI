"""
Src/Experiments/Exp4_EE_Model/plot_combine_resnet.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_expectation_vs_threshold(csv_rate_path, csv_acc_path, result_folder, m=8, exit_layer=[3, 6]):
    """
    结合计算延迟期望(E_t)和精度期望(E_acc)绘制双轴曲线。

    :param csv_rate_path: 包含率(rates)数据的 CSV 路径
    :param csv_acc_path: 包含精度(accs)数据的 CSV 路径
    :param result_folder: 结果保存文件夹
    :param m: 网络总层数/阶段数
    :param exit_layer: Early Exit 所在的层索引列表
    """
    # 1. 样式配置
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False

    # 2. 数据读取
    if not os.path.exists(csv_rate_path) or not os.path.exists(csv_acc_path):
        print("错误: 请检查输入的 CSV 路径是否存在。")
        return

    df_rate = pd.read_csv(csv_rate_path)
    df_acc = pd.read_csv(csv_acc_path)

    num_thresholds = len(df_rate)
    num_exits = len(exit_layer)

    # 3. 计算 E_t (延迟期望)
    rate_matrix = np.zeros((num_thresholds, m))
    for idx, row in df_rate.iterrows():
        exit_rates = row[1:num_exits + 1].values
        for i, layer in enumerate(exit_layer):
            rate_matrix[idx, layer] = exit_rates[i]
    rate_matrix = rate_matrix * 0.01

    P = np.zeros((num_thresholds, m))
    for i in range(num_thresholds):
        for j in range(m):
            if j == 0:
                P[i, j] = rate_matrix[i, j]
            else:
                P[i, j] = rate_matrix[i, j] * np.prod(1 - rate_matrix[i, :j])
        P[i, -1] = 1 - np.sum(P[i, :])

    t_matrix = np.arange(1, m + 1).reshape(1, -1).repeat(num_thresholds, axis=0)
    E_t = np.sum(P * t_matrix, axis=1)

    # 4. 计算 E_acc (精度期望)
    acc_matrix = np.zeros((num_thresholds, m))
    for idx, row in df_acc.iterrows():
        exit_accuracies = row[1:num_exits + 2].values
        for i, layer in enumerate(exit_layer):
            acc_matrix[idx, layer] = exit_accuracies[i]
        acc_matrix[idx, m - 1] = exit_accuracies[-1]

    E_acc = np.sum(P * acc_matrix, axis=1)

    # 5. 绘图
    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=300)

    # 左轴: E_t
    lns1 = ax1.plot(df_rate['threshold'], E_t, color='black', marker='o', markersize=5,
                    label='Computation Latency Expectation')
    ax1.set_xlabel('Threshold', fontsize=20)
    ax1.set_ylabel('Computation Latency Expectation', fontsize=16)
    ax1.tick_params(axis='x', labelsize=17)
    ax1.tick_params(axis='y', labelsize=17)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(3.7, 8.1)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 右轴: E_acc
    ax2 = ax1.twinx()
    lns2 = ax2.plot(df_acc['threshold'], E_acc, color='red', marker='^', markersize=5,
                    label='Accuracy Expectation')
    ax2.set_ylabel('Accuracy Expectation (%)', fontsize=20)
    ax2.tick_params(axis='y', labelsize=17)
    ax2.set_ylim(60, 100)

    # 合并图例
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    fig.legend(lns, labs, loc='upper left', bbox_to_anchor=(0.09, 0.96), ncol=1, fontsize=16, frameon=True)

    # 6. 保存与显示
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    base_name = os.path.splitext(os.path.basename(csv_rate_path))[0]
    save_path = os.path.join(result_folder, f"{base_name}_combined_expectation.png")

    plt.tight_layout()
    plt.savefig(save_path, format='png')
    print(f"双轴期望图已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    from Src.paras import RESULT_EE_MODEL_PATH
    rate_csv = "D:\Coding\Python\DSCI\Data\Resnet50_rates.csv"
    acc_csv = "D:\Coding\Python\DSCI\Data\Resnet50_accs.csv"
    plot_expectation_vs_threshold(rate_csv, acc_csv, RESULT_EE_MODEL_PATH)