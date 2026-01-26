"""
Src/Experiments/Exp4_EE_Model/plot_train_convergence_resnet.py
"""
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_training_convergence(csv_path, result_folder):
    """
    读取训练日志并生成收敛曲线图。

    :param csv_path: CSV 文件的完整路径
    :param result_folder: 结果图片保存的文件夹路径
    """
    # 配置绘图参数
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False

    # 读取数据
    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 {csv_path}")
        return

    df = pd.read_csv(csv_path)

    val_acc1 = df['val_acc'][:50].reset_index(drop=True)
    val_acc2 = df['val_acc'][50:100].reset_index(drop=True)
    val_acc3 = df['val_acc'][100:150].reset_index(drop=True)
    epochs = list(range(1, 51))

    # 创建画布
    plt.figure(figsize=(6, 4), dpi=300)

    # 绘制曲线
    plt.plot(epochs, val_acc1, color='red', marker='o', markersize=5, label='Main Exit')
    plt.plot(epochs, val_acc2, color='blue', marker='s', markersize=5, label='Early Exit 1')
    plt.plot(epochs, val_acc3, color='green', marker='^', markersize=5, label='Early Exit 2')

    # 坐标轴设置
    plt.xlabel('Training Epochs', fontsize=20)
    plt.ylabel('Classification Accuracy (%)', fontsize=20)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.xlim(0, 50)
    plt.ylim(0, 95)

    # 图例与网格
    plt.legend(loc='lower right', frameon=True, fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 处理保存路径
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # 获取原文件名（不带后缀）作为输出文件名的基础
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    save_path = os.path.join(result_folder, f"{base_name}_convergence.png")

    # 保存并展示
    plt.tight_layout()
    plt.savefig(save_path, format='png')
    print(f"图片已保存至: {save_path}")
    plt.show()


if __name__ == '__main__':
    from Src.paras import RESULT_EE_MODEL_PATH
    csv_file = r"/Data/ResNet50_trainlog_0508_0137.csv"
    output_dir = RESULT_EE_MODEL_PATH
    plot_training_convergence(csv_file, output_dir)