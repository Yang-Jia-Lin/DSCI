# Src/Utils/log_function.py

from datetime import datetime
import matplotlib.pyplot as plt
from Src.paras import *
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_convergence(history, data_name: str = "Convergence", save_dir: Path = None):
    history = [1.0 * x for x in history]

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(8, 5))
    plt.plot(history)
    plt.xlabel('迭代次数', fontsize=16)
    plt.ylabel('目标函数值', fontsize=16)
    plt.title('收敛曲线', fontsize=18, pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{data_name}_{datetime.now().strftime('%m%d_%H%M')}.svg"
        plt.savefig(save_path, format='svg')
        print(f"收敛曲线已保存至: {save_path}")

    plt.show()


def plot_decisions(X_opt: np.ndarray, Y_opt: np.ndarray, data_name: str, save_dir: Path):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    stem = Path(data_name).stem

    n, m = X_opt.shape
    E = EARLY_EXIT_LAYERS
    Y_E = Y_opt[:, E]

    # ---- X heatmap ----
    figX, axX = plt.subplots(dpi=300)
    axX.set_xlabel("DNN层", fontsize=16)  # 设置X轴标签字体大小
    axX.set_ylabel("用户", fontsize=16)  # 设置Y轴标签字体大小
    axX.set_title("切分决策 (X)", fontsize=18)  # 设置标题字体大小

    # 设置 X 轴刻度位置和标签
    axX.set_xticks(np.arange(m))  # 设置刻度位置为0到m-1
    axX.set_xticklabels([i + 1 for i in range(m)], fontsize=14)  # 设置刻度标签为 1, 2, 3, ..., m
    axX.set_yticks(range(n))  # 设置Y轴刻度位置
    axX.set_yticklabels([i + 1 for i in range(n)], fontsize=14)  # 设置Y轴刻度标签为 1, 2, 3, ..., n

    axX.matshow(X_opt, cmap='Blues', aspect='auto')  # 显示 X 的热力图
    for l in E:  # 早退层分割线
        axX.axvline(x=l, color="red", linestyle="--", linewidth=0.8)
    axX.tick_params(labelsize=14)  # 设置刻度标签字体大小
    axX.xaxis.set_ticks_position('bottom')  # 确保X轴刻度在底部

    # ---- Y heatmap ----
    figY, axY = plt.subplots(dpi=300)
    im_Y = axY.imshow(Y_E, aspect="auto", cmap="viridis", interpolation="nearest", origin='lower')
    axY.set_xlabel("DNN层", fontsize=16)  # 设置X轴标签字体大小
    axY.set_ylabel("用户", fontsize=16)  # 设置Y轴标签字体大小
    axY.set_title("早退阈值决策 (Y)", fontsize=18)

    axY.set_xticks(range(len(E)))  # 设置X轴刻度位置为早退层
    axY.set_xticklabels(E, fontsize=14)  # 设置X轴刻度标签为早退层的索引
    axY.set_yticks(range(n))  # 设置Y轴刻度位置
    axY.set_yticklabels([i + 1 for i in range(n)], fontsize=14)  # 设置Y轴刻度标签为 1, 2, 3, ..., n

    axY.tick_params(labelsize=14)  # 设置刻度标签字体大小
    axY.xaxis.set_ticks_position('bottom')  # 确保X轴刻度在底部

    # 设置 Y 轴从上到下增大
    axY.invert_yaxis()  # 反转Y轴，确保从上到下增大

    # Add color bar for Y
    cbar_Y = figY.colorbar(im_Y, ax=axY, fraction=0.046)
    cbar_Y.set_label("阈值", fontsize=14)  # 设置颜色条标签字体大小
    figY.tight_layout()
    plt.show()

    # ---- Save results ----
    if save_dir is not None:
        save_path_X = save_dir / f"{stem}_X_{datetime.now().strftime('%m%d_%H%M')}.svg"
        figX.savefig(save_path_X, format='svg')
        print(f"结果X已保存至: {save_path_X}")
        save_path_Y = save_dir / f"{stem}_Y_{datetime.now().strftime('%m%d_%H%M')}.svg"
        figY.savefig(save_path_Y, format='svg')
        print(f"结果Y已保存至: {save_path_Y}")


if __name__ == "__main__":
    # 示例数据，用于测试
    X = np.zeros([NUM_USERS, NUM_LAYERS])
    X[0, [10, 20]] = 1
    Y = np.ones([NUM_USERS, NUM_LAYERS])
    Y[2, 57] = 0.6
    Y[4, 103] = 0.8
    Y[5, 57] = 0.5
    Y[5, 103] = 0.9
    Y[7, 57] = 0.1
    Y[9, 103] = 0.
    plot_decisions(X, Y, data_name="最优结果示意图.svg", save_dir=Path(RESULT_GA_PATH))
