"""
Src/Experiments/Exp4_EE_Model/plot_acc_resnet.py
"""
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_accuracy_vs_threshold(csv_path, result_folder, constant_value=87.56):
    """
    绘制模型 Early Exit 精度随阈值变化的曲线图。

    :param csv_path: CSV 文件的完整路径
    :param result_folder: 结果图片保存的文件夹路径
    :param constant_value: Main Exit 的基准精度值（默认 87.56）
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
    plt.plot(df['threshold'], df['exit1_accuracy'], color='blue', marker='o', markersize=5, label='Early Exit 1')
    plt.plot(df['threshold'], df['exit2_accuracy'], color='green', marker='s', markersize=5, label='Early Exit 2')
    plt.plot(df['threshold'], df['full_accuracy'], color='red', marker='^', markersize=5, label='Overall')

    # 绘制 Main Exit 水平线
    plt.axhline(y=constant_value, color='black', linestyle='--', label='Main Exit')

    # 5. 添加数值标注
    plt.text(
        x=-0.1,
        y=constant_value - 0.5,
        s=f'{constant_value:.2f}%',
        color='red',
        fontsize=17
    )

    # 6. 坐标轴与细节设置
    plt.xlabel('Threshold', fontsize=20)
    plt.ylabel('Accuracy (%)', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(0, 0.96)
    plt.ylim(60, 100)

    # 图例与网格
    plt.legend(loc='lower right', frameon=True, fontsize=17)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 7. 处理保存逻辑
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # 自动生成 PNG 文件名
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    save_path = os.path.join(result_folder, f"{base_name}_accuracy_threshold.png")

    plt.tight_layout()
    plt.savefig(save_path, format='png')
    print(f"精度-阈值关系图已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    from Src.paras import RESULT_EE_MODEL_PATH
    csv_file = r"D:\Coding\Python\DSCI\Data\Resnet50_accs.csv"
    output_dir = RESULT_EE_MODEL_PATH
    plot_accuracy_vs_threshold(csv_file, output_dir)