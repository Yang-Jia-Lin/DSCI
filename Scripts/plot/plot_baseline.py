import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.rcParams['pdf.fonttype'] = 42  # ← 解决 Type 3 问题
mpl.rcParams['ps.fonttype'] = 42   # EPS 输出时也用 TTF
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False


# ------------------ Data ------------------
methods = ['Device', 'Edge    \n', 'Cloud', '\n  Collaborate', 'Cloud\n+EE', 'Collaborate\n+EE']
methods2 = ['Device', 'Edge', 'Cloud', 'Collaborate', 'Cloud\n+EE', 'Collaborate\n+EE']
latency_s = np.array([0.334874191214822,
                      0.000762131164417106,
                      0.13609650556441713,
                      0.02656396079555213,
                      0.09980015048201507,
                      0.03065270063597533])
latency_ms = latency_s * 1000

accuracy = np.array([5.1906,
                     5.1906,
                     5.1906,
                     5.1906,
                     5.34551197941379,
                     5.34551197941379]) * 100 / 6

# Objective function (using latency + accuracy formula, as per your approach)
obj = (np.array([4.855725808785178,
                5.189837868835583,
                5.054503494435583,
                5.164036039204448,
                5.245711828931775,
                5.314859278777815]) - 4.7)*10

# Color palette
colors = ['#1f77b4', '#8ca02c', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# =========================================================
# Figure 1: Latency-Accuracy-Utility Bubble Chart
# =========================================================
fig1, ax = plt.subplots(dpi=300)

scatter_sizes = obj*1000
for i, m in enumerate(methods):
    ax.scatter(latency_ms[i], accuracy[i],
               s=scatter_sizes[i],
               color=colors[i],
               alpha=0.8)
    # Annotate at the bubble's center
    ax.annotate(m,
                (latency_ms[i], accuracy[i]),
                ha='center', va='center',
                fontsize=16, color='black', weight='bold')

ax.set_xlim(-50, 365)
ax.set_ylim(85.5, 90)
ax.set_xlabel('Inference Latency (ms)', fontsize=20)
ax.set_ylabel('Accuracy (%)', fontsize=20)
# ax.set_title('Latency-Accuracy-Utility Bubble Chart', fontsize=18, pad=10)
ax.grid(True, linestyle='--', alpha=0.4)
# ax.invert_xaxis()           # Smaller latency should be on the right side

plt.tight_layout()
plt.savefig("4-2_baseline_bubble_chart.pdf", format='pdf')  # Saving as PDF
plt.show()

# =========================================================
# Figure 2: Objective Value Comparison Bar Chart
# =========================================================
fig2, ax2 = plt.subplots(dpi=300)

bars = ax2.bar(methods2, obj, color=colors, width=0.55)
for bar in bars:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., h + 0.01 * obj.max(),
             f'{h:.3f}',
             ha='center', va='bottom',
             fontsize=20)

ax2.set_ylabel('Total Utility', fontsize=20)
# ax2.set_title('Total Utility Comparison', fontsize=18, pad=10)
ax2.set_ylim(0, obj.max() * 1.15)
plt.xticks(rotation=30, fontsize=18)

plt.tight_layout()
plt.savefig("4-3_bar_chart.pdf", format='pdf')  # Saving as PDF
plt.show()
