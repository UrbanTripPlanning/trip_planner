import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load data
res = pd.read_csv('../astar_vs_dijkstra/computation_time.csv')

# Set seaborn theme
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)

# Define colors
color_dijkstra = "#1f77b4"  # Blue
color_astar = "#ff7f0e"     # Orange

# Create figure
plt.figure(figsize=(12, 6))

# Bars for Dijkstra
plt.bar(res.index + 1 - 0.2, res['dijkstra'], width=0.4,
        color=color_dijkstra, alpha=0.2, edgecolor='grey', label='Dijkstra')
plt.plot(res.index + 1 - 0.2, res['dijkstra'],
         color=color_dijkstra, linewidth=2.5)

# Bars for A*
plt.bar(res.index + 1 + 0.2, res['astar'], width=0.4,
        color=color_astar, alpha=0.2, edgecolor='grey', label='A*')
plt.plot(res.index + 1 + 0.2, res['astar'],
         color=color_astar, linewidth=2.5)

# Titles and labels
plt.title("Execution Time per Experiment: Dijkstra vs A*", fontsize=18, fontweight='bold')
plt.xlabel("Experiment Number", fontsize=14, labelpad=10)
plt.ylabel("Execution Time (s)", fontsize=14, labelpad=10)

# Ticks formatting
plt.xticks(ticks=res.index + 1, fontsize=12)
plt.yticks(fontsize=12)

# Legend
plt.legend(loc='upper left', frameon=False, fontsize=13)

# Grid and layout
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

# Save and show
plt.savefig("c.png", dpi=300)
plt.close()
