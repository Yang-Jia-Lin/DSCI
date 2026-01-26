"""
Src/Experiments/Exp4_EE_Model/plot_rate_resnet.py
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from Src.Utils.plot_utils import set_ieee_style, save_fig_for_ieee


def plot_early_exit_probability(csv_path, result_folder):
    """
    绘制 Early Exit 概率随阈值变化的曲线图。
    """
    csv_path = Path(csv_path)
    result_folder = Path(result_folder)

    # 1. 一键套用模板 (设置 Times New Roman, 字体比例, 线宽等)
    set_ieee_style(mode='single')

    # 2. 数据读取与校验
    df = pd.read_csv(csv_path)

    # 3. 创建画布 (不需要手动指定 figsize)
    plt.figure()

    # 4. 绘制数据曲线
    plt.plot(df['threshold'], df['exit1_rate'],
             color='blue', marker='o', markevery=3,
             label='Early Exit 1')

    plt.plot(df['threshold'], df['exit2_rate'],
             color='green', marker='^', markevery=3,
             label='Early Exit 2')

    # 5. 坐标轴与细节设置 (移除手动指定的 fontsize，由模板统一控制)
    plt.xlabel('Threshold')
    plt.ylabel('Early Exit Probability (%)')
    plt.xlim(0, 1)
    plt.ylim(0, 105)
    plt.legend(loc='lower left', frameon=True)

    # 6. 保存
    result_folder.mkdir(parents=True, exist_ok=True)
    save_name = result_folder / f"{csv_path.stem}_exit_probability"
    save_fig_for_ieee(save_name)

    print(f"概率分布图已保存至: {save_name}.pdf")
    plt.show()


if __name__ == "__main__":
    from Src.paras import RESULT_EE_MODEL_PATH
    output_dir = RESULT_EE_MODEL_PATH
    csv_file = Path(r"D:\Coding\Python\DSCI\Data\Resnet50_rates.csv")
    plot_early_exit_probability(csv_file, output_dir)