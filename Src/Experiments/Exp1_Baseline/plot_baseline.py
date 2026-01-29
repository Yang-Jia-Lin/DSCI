"""
Src/Experiments/Exp1_Baseline/plot_baseline.py
"""
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime

from Src.Utils.plot_utils import set_ieee_style, save_fig_for_ieee
from Src.paras import RESULT_BASELINE_PATH


def plot_bubble_chart(data: pd.DataFrame, save_dir=Path(RESULT_BASELINE_PATH)):
    set_ieee_style(mode='single')
    fig, ax = plt.subplots()
    colors = ['#1f77b4', '#8ca02c', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    name_map = {"仅在终端": "Device", "仅在边端": "Edge", "仅在云端": "Cloud",
                "仅协同": "Co-only", "边端 + 早退": "Edge+EE", "协同 + 早退": "Ours"}
    data['display_name'] = data['name'].map(lambda x: name_map.get(x, x))

    # 计算气泡大小
    obj_min, obj_max = data['objective'].min(), data['objective'].max()
    if obj_max != obj_min:
        # 将基础大小从 150 提至 200，映射极差提至 1500，气泡会更饱满
        data['bubble_size'] = 200 + (data['objective'] - obj_min) / (obj_max - obj_min + 1e-6) * 1500
    else:
        data['bubble_size'] = 1000
    df_sorted = data.sort_values('latency_ms').reset_index(drop=True)
    for i, row in df_sorted.iterrows():
        label = row['display_name'].replace('+', '\n+')
        ax.scatter(row['latency_ms'], row['accuracy'],
                   s=row['bubble_size'],
                   color=colors[i % len(colors)],
                   alpha=0.6, edgecolors='w', linewidth=0.5, zorder=10 + i)
        # 标签偏移逻辑
        y_offset = 14 if i % 2 == 0 else -14
        v_align = 'bottom' if i % 2 == 0 else 'top'
        ax.annotate(label,
                    xy=(row['latency_ms'], row['accuracy']),
                    xytext=(0, y_offset),
                    textcoords="offset points",
                    ha='center', va=v_align,
                    fontsize=9, weight='bold', zorder=20 + i)
    ax.set_xlabel('Inference Latency (ms)')
    ax.set_ylabel('Accuracy (%)')
    ax.margins(x=0.1, y=0.25)
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    plt.tight_layout(pad=0.15)

    save_dir.mkdir(parents=True, exist_ok=True)
    save_fig_for_ieee(save_dir / f"baseline_bubble_chart_{datetime.now().strftime('%H%M')}")
    plt.show()


def plot_utility_bar(data: pd.DataFrame, save_dir = Path(RESULT_BASELINE_PATH)):
    set_ieee_style(mode='single')
    fig, ax2 = plt.subplots()
    colors = ['#1f77b4', '#8ca02c', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    name_map = {
        "仅在终端": "Device", "仅在边端": "Edge", "仅在云端": "Cloud",
        "仅协同": "Co-only", "边端 + 早退": "Edge+EE", "协同 + 早退": "Ours"
    }
    data['display_name'] = data['name'].map(lambda x: name_map.get(x, x))
    bars = ax2.bar(data['display_name'], data['objective'], color=colors[:len(data)], width=0.7)
    for bar in bars:
        h = bar.get_height()
        va = 'bottom' if h > 0 else 'top'
        offset = 0.02 * data['objective'].max() if h > 0 else -0.05 * abs(data['objective'].min())
        ax2.text(bar.get_x() + bar.get_width() / 2., h + offset,
                 f'{h:.2f}', ha='center', va=va)
    ax2.set_ylabel('Total Utility')
    y_min = min(0, data['objective'].min() * 1.2)
    y_max = max(0, data['objective'].max() * 1.2)
    ax2.set_ylim(y_min, y_max)
    plt.xticks(rotation=15)
    plt.tight_layout(pad=0.15)

    save_dir.mkdir(parents=True, exist_ok=True)
    save_fig_for_ieee(save_dir / f"baseline_bar_chart_{datetime.now().strftime('%m%d_%H%M')}")
    plt.show()


if __name__ == "__main__":
    mock_data = pd.DataFrame({
        'name': ["仅在终端", "仅在边端", "仅在云端", "仅协同", "边端 + 早退", "协同 + 早退"],
        'latency_ms': [30, 80, 250, 150, 60, 110],
        'accuracy': [75.5, 88.2, 98.5, 92.0, 85.0, 96.5],
        'objective': [0.45, 0.62, 0.35, 0.78, 0.72, 0.95]
    })

    from Src.paras import RESULT_TEST_PATH
    plot_bubble_chart(mock_data, save_dir = Path(RESULT_TEST_PATH))
    plot_utility_bar(mock_data, save_dir = Path(RESULT_TEST_PATH))