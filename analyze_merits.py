import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np

matplotlib.use('TkAgg')

def calculate_power(data, sfreq, fmin=None, fmax=None):
    """
    Calculate power of a signal using its Fourier Transform.

    Parameters:
    - data: NumPy array of shape (epochs, channels, time_points)
    - sfreq: Sampling frequency of the data
    - fmin: Minimum frequency to include in power calculation
    - fmax: Maximum frequency to include in power calculation

    Returns:
    - power: Power of the signal in each channel
    """
    n_epochs, n_channels, n_times = data.shape
    # Compute the Fourier Transform
    fft_data = np.fft.rfft(data, n=n_times, axis=2)
    # Compute Power Spectral Density (PSD)
    psd = np.abs(fft_data) ** 2 / n_times
    # Calculate frequencies for PSD
    freqs = np.fft.rfftfreq(n_times, 1 / sfreq)

    # If frequency bounds are set, only sum power within those bounds
    if fmin is None:
        fmin = 0
    if fmax is None:
        fmax = sfreq / 2

    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    psd = psd[:, :, freq_mask]
    power = psd.sum(axis=2)  # Sum across frequencies to get total power

    # Return the average power across epochs
    return power.mean(axis=0)





def plt_snr(cleand_data, raw_data_selected_channels, sfreq=250, fmin=0.5, fmax=50):
    """
    绘制处理后数据的信噪比（SNR）的箱形图。

    参数：
    cleand_data : numpy.ndarray
        处理后的数据数组。
    raw_data_selected_channels : numpy.ndarray
        原始数据中选择的通道。
    sfreq : int
        采样频率。
    fmin : int
        最小频率。
    fmax : int
        最大频率。
    """
    # 计算信号功率和噪声功率
    signal_power = calculate_power(cleand_data, sfreq, fmin, fmax)
    noise_power = calculate_power(cleand_data - raw_data_selected_channels, sfreq, fmin, fmax)

    # 计算信噪比（SNR）并转换为分贝（dB）
    epsilon = 1e-10  # 防止除以零
    snr_linear = signal_power / (noise_power + epsilon)
    snr_db = 10 * np.log10(snr_linear)

    # 计算平均值和标准偏差
    mean_snr = np.mean(snr_db)
    std_snr = np.std(snr_db)

    # 创建箱形图
    fig, ax = plt.subplots(figsize=(10, 6))
    bplot = ax.boxplot(snr_db, patch_artist=True, showmeans=True, meanline=True)

    # 设置箱子的颜色
    for patch in bplot['boxes']:
        patch.set_facecolor('#1C7C54')  # 箱子填充颜色
        patch.set_edgecolor('black')  # 箱子边缘颜色

    # 在箱形图旁边显示平均值和标准偏差
    text_x_position = 1.1  # X轴上的文本位置
    ax.text(text_x_position, mean_snr, f'Mean: {mean_snr:.2f} dB', verticalalignment='center', fontsize=12)
    ax.text(text_x_position, mean_snr - std_snr, f'-STD: {std_snr:.2f} dB', verticalalignment='center', fontsize=12)
    ax.text(text_x_position, mean_snr + std_snr, f'+STD: {std_snr:.2f} dB', verticalalignment='center', fontsize=12)

    # 添加图例，表示平均值线
    ax.plot([], [], color='orange', label=f'Mean: {mean_snr:.2f} dB\n+/- STD: {std_snr:.2f} dB', linewidth=2)

    # 设置标题和坐标轴标签
    ax.set_title('Signal to Noise Ratio (SNR) Across All Channels', fontsize=18)
    ax.set_ylabel('SNR (dB)', fontsize=14)
    ax.set_xticks([1])
    ax.set_xticklabels(['All Channels'], fontsize=12)

    # 添加图例
    ax.legend()

    # 显示网格线
    ax.grid(True, linestyle='--', alpha=0.7)

    # 显示图形
    plt.show()


def calculate_nmse(data1, data2):
    """
    计算两个数据集之间的归一化均方误差（NMSE）。
    参数：
    data1 : numpy.ndarray
        第一个数据集。
    data2 : numpy.ndarray
        第二个数据集。
    返回值：
    nmse_per_channel : numpy.ndarray
        NMSE的值。
    """
    epsilon = 1e-10  # 避免除以零的小常数
    nmse_per_channel = np.mean((data1 - data2) ** 2, axis=(0, 2)) / (np.mean(data2 ** 2, axis=(0, 2)) + epsilon)
    return nmse_per_channel


def compare_nmse(cleand_data, raw_data_selected_channels, normal_asr, sfreq=250, fmin=0.5, fmax=50):
    """
    绘制处理后数据及另一个数据集的归一化均方误差（NMSE）的箱形图。
    参数：
    cleand_data : numpy.ndarray
        处理后的数据数组。
    raw_data_selected_channels : numpy.ndarray
        原始数据中选择的通道。
    normal_asr : numpy.ndarray
        另一个数据集。
    sfreq : int
        采样频率。
    fmin : int
        最小频率。
    fmax : int
        最大频率。
    """
    # 计算NMSE
    nmse_clean = calculate_nmse(cleand_data, raw_data_selected_channels)
    # print(nmse_clean)
    nmse_normal_asr = calculate_nmse(normal_asr, raw_data_selected_channels)
    # print(nmse_normal_asr)
    # 创建箱形图
    fig, ax = plt.subplots(figsize=(12, 4))
    bplot = ax.boxplot([nmse_normal_asr, nmse_clean], patch_artist=True, showmeans=True, meanline=True,
                       positions=[1, 2], widths=0.2)

    # 设置箱子的颜色和其他属性
    colors = ['#AED6F1', '#F9E79F']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)  # 箱子填充颜色
        patch.set_edgecolor('black')  # 箱子边缘颜色

    # 计算每个数据集的平均NMSE
    mean_nmse_clean = np.mean(nmse_clean)
    mean_nmse_normal_asr = np.mean(nmse_normal_asr)

    # 设置标题和坐标轴标签
    ax.set_title('Normalized Mean Squared Error (NMSE) Comparison', fontsize=18)
    ax.set_ylabel('NMSE', fontsize=14)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Normal ASR', 'MASR'], fontsize=12)

    # 显示网格线
    ax.grid(True, linestyle='--', alpha=0.7)

    # 添加图例说明
    legend_labels = [f'Normal ASR (Mean: {mean_nmse_normal_asr:.3})', f'MASR (Mean: {mean_nmse_clean:.3})']
    ax.legend(handles=[plt.Rectangle((0, 0), 0.5, 2, color=color) for color in colors], labels=legend_labels,
              framealpha=1, loc='best')

    # 显示图形
    plt.show()


def compare_snr(cleand_data, raw_data_selected_channels, normal_asr, sfreq=250, fmin=0.5, fmax=50):
    """
    绘制处理后数据及另一个数据集的信噪比（SNR）的箱形图。

    参数：
    cleand_data : numpy.ndarray
        处理后的数据数组。
    raw_data_selected_channels : numpy.ndarray
        原始数据中选择的通道。
    normal_asr : numpy.ndarray
        另一个数据集。
    sfreq : int
        采样频率。
    fmin : int
        最小频率。
    fmax : int
        最大频率。
    """
    # 计算信号功率和噪声功率
    signal_power = calculate_power(cleand_data, sfreq, fmin, fmax)
    noise_power = calculate_power(cleand_data - raw_data_selected_channels, sfreq, fmin, fmax)
    normal_asr_power = calculate_power(normal_asr - raw_data_selected_channels, sfreq, fmin, fmax)

    # 计算信噪比（SNR）并转换为分贝（dB）
    epsilon = 1e-10  # 防止除以零
    snr_linear = signal_power / (noise_power + epsilon)
    normal_asr_snr_linear = signal_power / (normal_asr_power + epsilon)

    snr_db = 10 * np.log10(snr_linear)
    normal_asr_snr_db = 10 * np.log10(normal_asr_snr_linear)

    # 创建箱形图
    fig, ax = plt.subplots(figsize=(12, 4))
    bplot = ax.boxplot([normal_asr_snr_db, snr_db], patch_artist=True, showmeans=True, meanline=True,
                       positions=[1, 2], widths=0.2)

    # 设置箱子的颜色和其他属性
    colors = ['#AED6F1', '#F9E79F']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)  # 箱子填充颜色
        patch.set_edgecolor('black')  # 箱子边缘颜色
        # 计算每个数据集的平均SNR
    mean_snr_asr = np.mean(snr_db)
    mean_snr_normal_asr = np.mean(normal_asr_snr_db)

    # 设置标题和坐标轴标签
    ax.set_title('Signal to Noise Ratio (SNR) Comparison', fontsize=18)
    ax.set_ylabel('SNR (dB)', fontsize=14)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Normal ASR', 'MASR'], fontsize=12)

    # 显示网格线
    ax.grid(True, linestyle='--', alpha=0.7)

    # 添加图例说明
    legend_labels = [f'Normal ASR (Mean: {mean_snr_normal_asr:.2f} dB)', f'MASR (Mean: {mean_snr_asr:.2f} dB)']
    ax.legend(handles=[plt.Rectangle((0, 0), 0.5, 2, color=color) for color in colors], labels=legend_labels,
              framealpha=1, loc='best')

    # 显示图形
    plt.show()
