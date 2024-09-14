import numpy as np
import matplotlib.pyplot as plt

# 生成时间轴
t = np.linspace(0, 1, 1000)

# 生成几条带有伪影的时域信号
def generate_signal(t, frequency, noise_level):
    signal = np.sin(2 * np.pi * frequency * t)
    noise = noise_level * np.random.normal(size=t.shape)
    return signal + noise

# 设置参数
frequencies = [5, 10, 15]  # 不同的频率
noise_level = 0.2  # 噪声级别

# 创建图像
plt.figure(figsize=(10, 6))
plt.title("时域信号")

# 为每个频率生成信号并绘制
for freq in frequencies:
    signal = generate_signal(t, freq, noise_level)
    plt.plot(t, signal, label=f'频率={freq}Hz')

# 设置图像的白底和黑色线
plt.gca().set_facecolor('white')
plt.grid(True, which='both', lw=0.5)
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.xlabel('时间 (s)')
plt.ylabel('幅度')
plt.legend()
plt.show()
