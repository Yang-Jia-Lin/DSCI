"""
Src/Exp3_DSCI_Convergency/plot_convergency.py
"""
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from Src.Utils.plot_utils import save_fig_for_ieee, set_ieee_style


def plot_convergence(history, data_name: str = "DSCI Convergence", save_dir: Path = None):
    # 1. 统一风格配置
    set_ieee_style(mode='single')

    # 2. 数据处理与绘图
    history = [1.0 * x for x in history]
    plt.figure()
    plt.plot(history, label=data_name, linewidth=0.9)

    # 3. 标签与标题
    plt.xlabel('Epoch')
    plt.ylabel('Utility')
    plt.title(f'{data_name}')
    plt.grid(True, linestyle='--', alpha=0.5) # 如果 IEEE 风格函数没带 grid，可保留此项

    # 4. 统一保存逻辑
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{data_name}_{datetime.now().strftime('%m%d_%H%M')}"
        save_path = save_dir / file_name
        save_fig_for_ieee(save_path)
        print(f"收敛曲线已保存至: {save_path}")
    plt.show()


if __name__ == '__main__':
    from Src.Utils.log_function import load_and_analyze_results
    PPO_path = Path("D:\Coding\Python\DSCI\Result\Optimize\PPO\PPO_20260127_153244")
    X_opt, Y_opt, F_e, F_c, history, paras = load_and_analyze_results(
        exp_dir = PPO_path, analysis = False)
    plot_convergence(history, save_dir = PPO_path)