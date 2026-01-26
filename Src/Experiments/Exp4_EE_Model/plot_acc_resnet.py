"""
Src/Experiments/Exp4_EE_Model/plot_acc_resnet.py
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from Src.Utils.plot_utils import set_ieee_style, save_fig_for_ieee


def plot_accuracy_vs_threshold(csv_path, result_folder, constant_value=87.56):
    """
    绘制模型 Early Exit 精度随阈值变化的曲线图。
    """
    csv_path = Path(csv_path)
    result_folder = Path(result_folder)

    # 1. 套用 IEEE 模板
    set_ieee_style(mode='single')

    # 2. 数据读取
    if not csv_path.exists():
        print(f"错误: 找不到文件 {csv_path}")
        return
    df = pd.read_csv(csv_path)

    # 3. 创建画布
    plt.figure()

    # 4. 绘制数据曲线
    plt.plot(df['threshold'], df['exit1_accuracy'], color='blue', marker='o', label='Early Exit 1', markevery=5)
    plt.plot(df['threshold'], df['exit2_accuracy'], color='green', marker='s', label='Early Exit 2', markevery=5)
    plt.plot(df['threshold'], df['full_accuracy'], color='red', marker='^', label='Overall', markevery=5)

    # 绘制 Main Exit 水平线
    plt.axhline(y=constant_value, color='black', linestyle='--', label='Main Exit')

    # 5. 添加数值标注
    plt.text(
        x=0.05,  # 稍微向右移一点，防止紧贴边缘
        y=constant_value + 1.0,  # 向上偏一点，防止压线
        s=f'Main Exit: {constant_value:.2f}%',
        color='black',  # 改为黑色更符合 IEEE 严肃风格
        fontweight='bold'
    )

    # 6. 坐标轴与细节设置
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy (%)')
    plt.xlim(0, 1.0)  # 阈值通常到 1.0
    plt.ylim(60, 100)

    # 图例设置
    plt.legend(loc='lower right', frameon=True)

    # 7. 处理保存逻辑
    result_folder.mkdir(parents=True, exist_ok=True)
    save_name = result_folder / f"{csv_path.stem}_accuracy_threshold"

    # 使用模板函数保存 (自动生成 pdf 和 png)
    save_fig_for_ieee(save_name)

    print(f"精度-阈值关系图已保存至: {save_name}.pdf")
    plt.show()


if __name__ == "__main__":
    from Src.paras import RESULT_EE_MODEL_PATH
    csv_file = Path(r"D:\Coding\Python\DSCI\Data\Resnet50_accs.csv")
    output_dir = Path(RESULT_EE_MODEL_PATH)
    plot_accuracy_vs_threshold(csv_file, output_dir)