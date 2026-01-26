"""
Src/Experiments/Exp4_EE_Model/plot_rate_resnet.py
"""
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_early_exit_probability(csv_path, result_folder):
    """
    绘制 Early Exit 概率随阈值变化的曲线图。

    :param csv_path: CSV 文件的完整路径
    :param result_folder: 结果图片保存的文件夹路径
    """
    # 1. 基础样式配置
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False

    # 2. 数据读取与校验
    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # 3. 创建画布
    plt.figure(figsize=(6, 4), dpi=300)

    # 4. 绘制数据曲线
    plt.plot(df['threshold'], df['exit1_rate'], color='blue', marker='o', markersize=5, label='Early Exit 1')
    plt.plot(df['threshold'], df['exit2_rate'], color='green', marker='^', markersize=5, label='Early Exit 2')

    # 5. 坐标轴与细节设置
    plt.xlabel('Threshold', fontsize=20)
    plt.ylabel('Early Exit Probability (%)', fontsize=20)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.xlim(0, 1)
    plt.ylim(0, 105)

    # 图例与网格
    plt.legend(loc='lower left', frameon=True, fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 6. 处理保存逻辑
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # 自动生成 PNG 文件名
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    save_path = os.path.join(result_folder, f"{base_name}_exit_probability.png")

    # 7. 保存并展示
    plt.tight_layout()
    plt.savefig(save_path, format='png')
    print(f"概率分布图已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    from Src.paras import RESULT_EE_MODEL_PATH
    output_dir = RESULT_EE_MODEL_PATH
    csv_file = r"D:\Coding\Python\DSCI\Data\Resnet50_rates.csv"
    plot_early_exit_probability(csv_file, output_dir)