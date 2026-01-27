"""
Src/Experiments/Exp1_Baseline/plot_baseline.py
"""
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime
from Src.Utils.plot_utils import set_ieee_style, save_fig_for_ieee
from Src.paras import RESULT_BASELINE_PATH


def plot_bubble_chart(df: pd.DataFrame, save_dir = Path(RESULT_BASELINE_PATH)):
    set_ieee_style(mode='single')
    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    colors = ['#1f77b4', '#8ca02c', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # 气泡大小映射
    bubble_size = (df['objective'] - df['objective'].min() + 1) * 500
    for i, row in df.iterrows():
        display_name = row['name'].replace('+', '\n+')
        ax.scatter(row['latency_ms'], row['accuracy'],
                   s=bubble_size[i],
                   color=colors[i % len(colors)],
                   alpha=0.8, edgecolors='w', linewidth=0.5)
        ax.annotate(display_name,
                    (row['latency_ms'], row['accuracy']),
                    ha='center', va='center',
                    fontsize=9, weight='bold')
    # 坐标轴
    ax.set_xlabel('Inference Latency (ms)')
    ax.set_ylabel('Accuracy (%)')
    ax.grid(True, linestyle='--', alpha=0.4)

    # 气泡边缘
    ax.set_xlim(df['latency_ms'].min() - 50, df['latency_ms'].max() + 80)
    ax.set_ylim(df['accuracy'].min() - 1, df['accuracy'].max() + 1)

    # 保存
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"baseline_bubble_chart_{datetime.now().strftime('%m%d_%H%M')}"
        save_path = save_dir / file_name
        save_fig_for_ieee(save_path)
        print(f"图表已保存至: {save_path}")
    plt.show()


def plot_utility_bar(df: pd.DataFrame, save_dir = Path(RESULT_BASELINE_PATH)):
    set_ieee_style(mode='single')
    colors = ['#1f77b4', '#8ca02c', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=300)
    bars = ax.bar(df['name'], df['objective'], color=colors[:len(df)], width=0.6)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Total Utility')
    ax.set_ylim(0, df['objective'].max() * 1.2)
    plt.xticks(rotation=15, ha='right')  # 稍微旋转避免重叠

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"baseline_bar_chart_{datetime.now().strftime('%m%d_%H%M')}"
        save_fig_for_ieee(save_path)
        print(f"图表已保存至: {save_path}")
    plt.show()
