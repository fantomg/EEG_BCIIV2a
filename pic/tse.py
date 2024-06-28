import matplotlib.patches as patches
from matplotlib import pyplot as plt

# Setting up the figure and axes
fig, ax = plt.subplots(figsize=(10, 8))

# Create a simple flow chart
# Nodes
nodes = {
    "Data Preparation": (0.5, 0.8),
    "Phase Space Reconstruction": (0.5, 0.6),
    "Apply 2D FFT": (0.5, 0.4),
    "Extract Frequency Vectors": (0.5, 0.2),
    "Calculate TSE": (0.25, 0.0),
    "Calculate ATSE": (0.75, 0.0)
}

# Arrows and annotations
for node, (x, y) in nodes.items():
    ax.text(x, y, node, ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightblue'))

# Connecting lines
arrows = [
    ("Data Preparation", "Phase Space Reconstruction"),
    ("Phase Space Reconstruction", "Apply 2D FFT"),
    ("Apply 2D FFT", "Extract Frequency Vectors"),
    ("Extract Frequency Vectors", "Calculate TSE"),
    ("Extract Frequency Vectors", "Calculate ATSE")
]

for start, end in arrows:
    start_x, start_y = nodes[start]
    end_x, end_y = nodes[end]
    ax.annotate("", xy=(end_x, end_y + 0.05), xytext=(start_x, start_y - 0.05),
                arrowprops=dict(arrowstyle="->", lw=1.5))

# Removing axes and setting aspect
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
plt.show()
