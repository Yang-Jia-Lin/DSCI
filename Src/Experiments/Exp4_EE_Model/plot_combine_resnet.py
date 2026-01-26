"""
Src/Experiments/Exp4_EE_Model/plot_combine_resnet.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from Src.Utils.plot_utils import set_ieee_style, save_fig_for_ieee


def plot_expectation_vs_threshold(csv_rate_path, csv_acc_path, result_folder):
    """
    结合计算延迟期望(E_t)和精度期望(E_acc)绘制双轴曲线。
    """
    m = 8
    exit_layer = [3, 6]
    csv_rate_path = Path(csv_rate_path)
    csv_acc_path = Path(csv_acc_path)
    result_folder = Path(result_folder)

    # 1. 一键套用模板 (自动处理字体、线宽、基础尺寸)
    set_ieee_style(mode='single')

    # 2. 数据读取与校验
    if not csv_rate_path.exists() or not csv_acc_path.exists():
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

    # 5. 绘图开始
    fig, ax1 = plt.subplots()

    # 配色方案
    color_et = 'black'
    color_acc = '#d62728'  # IEEE 常用深红色

    # 左轴: E_t (延迟)
    # 使用 markevery=10 稀释标记点，linewidth 设为 1.0 防太粗
    lns1 = ax1.plot(df_rate['threshold'], E_t, color=color_et, marker='o', markevery=3, label='Latency Expect')

    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Latency Expectation', color=color_et)
    ax1.tick_params(axis='y', labelcolor=color_et)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(3.5, 8.5)

    # 右轴: E_acc (精度)
    ax2 = ax1.twinx()
    lns2 = ax2.plot(df_acc['threshold'], E_acc, color=color_acc, marker='^', markevery=3, label='Accuracy Expect')

    ax2.set_ylabel('Accuracy Expectation (%)', color=color_acc)
    ax2.tick_params(axis='y', labelcolor=color_acc)
    ax2.set_ylim(60, 100)
    ax2.grid(False)  # 双轴图通常关闭右轴网格，防止画面太乱

    # 6. 图例合并与优化
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    # 将图例放在图表正上方，避免遮挡数据
    ax1.legend(lns, labs, loc='lower center', bbox_to_anchor=(0.5, 0.93),
               ncol=2, frameon=False, fontsize=10)

    # 7. 保存与显示
    result_folder.mkdir(parents=True, exist_ok=True)
    save_name = result_folder / f"{csv_rate_path.stem}_combined_expectation"
    save_fig_for_ieee(save_name)
    print(f"双轴期望图已保存至: {save_name}.pdf")
    plt.show()


if __name__ == "__main__":
    from Src.paras import RESULT_EE_MODEL_PATH
    rate_csv = Path(r"D:\Coding\Python\DSCI\Data\Resnet50_rates.csv")
    acc_csv = Path(r"D:\Coding\Python\DSCI\Data\Resnet50_accs.csv")
    plot_expectation_vs_threshold(rate_csv, acc_csv, RESULT_EE_MODEL_PATH)