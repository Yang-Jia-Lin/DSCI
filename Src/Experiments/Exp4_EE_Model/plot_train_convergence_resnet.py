"""
Src/Experiments/Exp4_EE_Model/plot_train_convergence_resnet.py
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from Src.Utils.plot_utils import set_ieee_style, save_fig_for_ieee


def plot_training_convergence(csv_path, result_folder):
    """
    读取训练日志并生成收敛曲线图。
    """
    csv_path = Path(csv_path)
    result_folder = Path(result_folder)

    # 1. 一键套用 IEEE 风格模板
    set_ieee_style(mode='single')

    # 2. 读取数据
    df = pd.read_csv(csv_path)
    val_acc1 = df['val_acc'][:50].reset_index(drop=True)
    val_acc2 = df['val_acc'][50:100].reset_index(drop=True)
    val_acc3 = df['val_acc'][100:150].reset_index(drop=True)
    epochs = list(range(1, 51))

    # 3. 创建画布
    plt.figure()

    # 4. 绘制曲线
    plt.plot(epochs, val_acc1, color='red', marker='o', linewidth=1.2, markevery=3, label='Main Exit')
    plt.plot(epochs, val_acc2, color='blue', marker='s', linewidth=1.2, markevery=3, label='Early Exit 1')
    plt.plot(epochs, val_acc3, color='green', marker='^', linewidth=1.2, markevery=3, label='Early Exit 2')

    # 5. 坐标轴设置
    plt.xlabel('Training Epochs')
    plt.ylabel('Classification Accuracy (%)')
    plt.xlim(0, 50)
    plt.ylim(45, 90)
    plt.legend(loc='lower right', frameon=True)

    # 6. 保存
    result_folder.mkdir(parents=True, exist_ok=True)
    save_name = result_folder / f"{csv_path.stem}_convergence"
    save_fig_for_ieee(save_name)
    print(f"训练收敛图已保存至: {save_name}.pdf")
    plt.show()


if __name__ == '__main__':
    from Src.paras import RESULT_EE_MODEL_PATH
    csv_file = Path(r"D:\Coding\Python\DSCI\Data\ResNet50_trainlog_0508_0137.csv")
    output_dir = RESULT_EE_MODEL_PATH
    plot_training_convergence(csv_file, output_dir)