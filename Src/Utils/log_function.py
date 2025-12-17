import time
import json
import numpy as np
from pathlib import Path
from Src.Utils.plot_function import plot_decisions, plot_convergence


class NumpyEncoder(json.JSONEncoder):
    """
    用于解决 JSON 无法直接序列化 Numpy 数据类型的问题
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            # 如果数组太长，只保存形状信息，保持 log 简洁
            if obj.size > 20:
                return f"<numpy_array shape={obj.shape} dtype={obj.dtype}>"
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def save_experiment_results(
        save_dir: Path,
        algo_name: str,
        paras,
        best_val: float,
        best_sol: tuple,
        history: list,
        hyper_params: dict = None
):
    """
    通用 Log 保存函数：
    1. 创建独立文件夹
    2. config.json: 保存系统参数和算法超参数
    3. solution.npz: 保存最优解矩阵(X, Y, F)、最优值、历史曲线数据
    4. 保存图片
    """
    # 1. 创建 "算法名_时间戳" 的文件夹
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = save_dir / f"{algo_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 解包最优解 (根据你的代码 BF_best_sol 包含4个部分)
    X_opt, Y_opt, F_e_opt, F_c_opt = best_sol

    # ---------------------------------------------------------
    # 2. 保存参数到 config.json (替代 log.txt)
    # ---------------------------------------------------------
    # 提取 paras 对象中的属性
    paras_dict = {k: v for k, v in vars(paras).items() if not k.startswith("__")}

    config_data = {
        "Algorithm": algo_name,
        "Time": timestamp,
        "Best_Objective_Value": best_val,
        "Hyper_Parameters": hyper_params,
        "System_Parameters": paras_dict
    }

    json_path = exp_dir / "config.json"
    with open(json_path, "w", encoding="utf-8") as f:
        # 使用自定义 Encoder 处理 numpy 数组
        json.dump(config_data, f, indent=4, cls=NumpyEncoder)

    # ---------------------------------------------------------
    # 3. 保存结果数据到 solution.npz (替代 CSV 计算)
    # ---------------------------------------------------------
    # .npz 是保存多个 numpy 数组的标准格式，非常适合保存矩阵 X, Y 和 历史记录
    npz_path = exp_dir / "solution.npz"
    np.savez(
        npz_path,
        # 核心解
        X=X_opt,
        Y=Y_opt,
        F_e=F_e_opt,
        F_c=F_c_opt,
        # 标量和历史
        best_val=best_val,
        history=np.array(history)
    )

    # ---------------------------------------------------------
    # 4 & 5. 绘制并保存曲线
    # ---------------------------------------------------------
    # 图片会自动保存到 exp_dir 目录下
    plot_convergence(history, data_name=f"{algo_name}_Convergence", save_dir=exp_dir)
    plot_decisions(X_opt, Y_opt, data_name=f"{algo_name}_Decisions", save_dir=exp_dir)

    print(f"[{algo_name}] Experiment saved to: {exp_dir}")
    print(f"  - Config: config.json")
    print(f"  - Data:   solution.npz")