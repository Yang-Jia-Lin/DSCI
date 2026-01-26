"""
Src/Utils/log_function.py
"""
import os
import platform
import subprocess
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


def open_file(file_path):
    """跨平台打开文件"""
    if platform.system() == "Windows":
        os.startfile(file_path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", str(file_path)])
    else:  # Linux
        subprocess.run(["xdg-open", str(file_path)])


def save_experiment_results(
        save_dir: Path,
        algo_name: str,
        paras,
        best_val: float,
        best_sol: tuple,
        history: list,
        hyper_params: dict = None,
        extra_logs: list = None
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

    # ===== 新增：在 config.json 中记录 extra_logs 的 key 摘要（可选）=====
    if extra_logs is not None:
        config_data["Extra_Logs_Keys"] = sorted(list(extra_logs[0].keys())) if len(extra_logs) > 0 else []

    json_path = exp_dir / "config.json"
    with open(json_path, "w", encoding="utf-8") as f:
        # 使用自定义 Encoder 处理 numpy 数组
        json.dump(config_data, f, indent=4, cls=NumpyEncoder)

    # ===== 新增：保存 metrics.jsonl（可选，强烈推荐）=====
    if extra_logs is not None:
        metrics_path = exp_dir / "metrics.jsonl"
        with open(metrics_path, "w", encoding="utf-8") as f:
            for row in extra_logs:
                f.write(json.dumps(row, cls=NumpyEncoder) + "\n")

    # ---------------------------------------------------------
    # 3. 保存结果数据到 solution.npz (替代 CSV 计算)
    # ---------------------------------------------------------
    # .npz 是保存多个 numpy 数组的标准格式，非常适合保存矩阵 X, Y 和 历史记录
    npz_path = exp_dir / "solution.npz"

    # ===== 新增：把 extra_logs 的每个字段也存进 npz（可选）=====
    extra_npz = {}
    if extra_logs is not None and len(extra_logs) > 0:
        keys = extra_logs[0].keys()
        for k in keys:
            extra_npz[f"metrics_{k}"] = np.array([row.get(k) for row in extra_logs], dtype=object)

    np.savez(
        npz_path,
        # 核心解
        X=X_opt,
        Y=Y_opt,
        F_e=F_e_opt,
        F_c=F_c_opt,
        # 标量和历史
        best_val=best_val,
        history=np.array(history),
        **extra_npz
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


def load_and_analyze_results(exp_dir: Path):
    """
    加载并复现实验结果
    :param exp_dir: 实验结果文件夹路径 (例如: "Results/PPO/PPO_20260104_200958")
    """
    exp_dir = Path(exp_dir)
    json_path = exp_dir / "config.json"
    npz_path = exp_dir / "solution.npz"

    if not json_path.exists() or not npz_path.exists():
        print(f"Error: 路径 {exp_dir} 下缺少 config.json 或 solution.npz")
        return

    # 1. 加载数据
    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    data = np.load(npz_path, allow_pickle=True)

    # 2. 提取信息
    algo_name = config.get("Algorithm", "Unknown")
    timestamp = config.get("Time", "Unknown")
    hyper_params = config.get("Hyper_Parameters", {})
    sys_params = config.get("System_Parameters", {})
    best_val = config.get("Best_Objective_Value", "N/A")

    # 3. 打印
    print("=" * 50)
    print(f"实验结果: {algo_name} | {timestamp}")
    print("=" * 50)

    # 3.1 超参数
    print("\n核心超参数 (Hyper-Parameters):")
    for k, v in hyper_params.items():
        print(f"  - {k}: {v}")

    # 3.2 系统参数
    print("\n关键系统参数 (System Parameters):")
    target_sys_keys = ['F_u', 'f_e_max', 'f_c_max', 'alpha', 'beta']
    for key in target_sys_keys:
        val = sys_params.get(key, "Not Found")
        # 如果是数组，打印其形状或均值，避免刷屏
        if isinstance(val, list) and len(val) > 5:
            print(f"  - {key}: List of length {len(val)} ({val[:3]}...)")
        else:
            print(f"  - {key}: {val}")

    # 3.3 结果
    print("\n训练结果 (Results):")
    print(f"  - Best Objective Value: {best_val}")
    print(f"  - Solution Matrices in NPZ: {list(data.keys())}")

    # 4. 绘图
    print("\n生成图表...")

    # 4.1 获取数据
    X_opt = data['X']
    Y_opt = data['Y']
    history = data['history']

    # 4.2 绘图判断
    conv_svgs = list(exp_dir.glob("*Convergence*.svg"))
    decs_svgs = list(exp_dir.glob("*Decisions*.svg"))

    # 4.3 处理收敛曲线图
    if conv_svgs:
        print(f"发现收敛图，正在打开: {conv_svgs[0].name}")
        open_file(conv_svgs[0])
    else:
        print("未发现收敛图，正在重新绘制...")
        plot_convergence(history, data_name=f"{algo_name}_Convergence", save_dir=exp_dir)

    # 4.4 处理决策图
    if decs_svgs:
        print(f"发现决策图，正在打开: {decs_svgs[0].name}")
        open_file(decs_svgs[0])
    else:
        print("未发现决策图，正在重新绘制...")
        plot_decisions(X_opt, Y_opt, data_name=f"{algo_name}_Decisions", save_dir=exp_dir)


if __name__ == "__main__":
    target_path = Path("D:\Coding\Python\DSCI\Result\GA\GA_20260113_090630")
    load_and_analyze_results(target_path)