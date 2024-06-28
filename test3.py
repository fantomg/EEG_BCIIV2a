import numpy as np

# 生成均匀分布的数据
data_uniform = np.random.uniform(-0.1, 0.1, size=100)

# 计算最大最小值的差
max_min_diff = np.max(data_uniform) - np.min(data_uniform)

# 计算标准差的两倍
std_twice = 2 * np.std(data_uniform)

print(f"Max-min difference: {max_min_diff}")
print(f"Standard deviation * 2: {std_twice}")