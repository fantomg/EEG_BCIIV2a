import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

matplotlib.use('TkAgg')

from sklearn.metrics import mutual_info_score


def discretize_signal(signal, bins='auto'):
    """
    Discretize a continuous signal into bins.

    Parameters:
    - signal: NumPy array representing the continuous signal.
    - bins: The strategy to compute the bins, 'auto' will use the
            Freedman–Diaconis rule to estimate the bin size.

    Returns:
    - discretized_signal: Discretized version of the signal.
    """
    _, bin_edges = np.histogram(signal, bins=bins)
    discretized_signal = np.digitize(signal, bins=bin_edges)
    return discretized_signal


def compare_mi(cleaned_data, raw_data_selected_channels, normal_asr):
    """
    Calculate and compare the Mutual Information (MI) between cleaned EEG data and normal ASR data.

    Parameters:
    - cleaned_data: Cleaned EEG data.
    - raw_data_selected_channels: Raw EEG data from selected channels.
    - normal_asr: Normal EEG data after applying Artifact Subspace Reconstruction (ASR) or another cleaning method.

    Note: Assumes data are NumPy arrays with shape (epochs, channels, time_points)
    """
    n_epochs, n_channels, _ = cleaned_data.shape

    # Initialize an array to store MI values
    mi_values = np.zeros((n_channels, 2))  # Two columns for the two comparisons

    for i in range(n_channels):
        # Discretize the signals
        disc_cleaned = discretize_signal(cleaned_data[:, i, :].flatten())
        disc_raw = discretize_signal(raw_data_selected_channels[:, i, :].flatten())
        disc_normal_asr = discretize_signal(normal_asr[:, i, :].flatten())

        # Calculate MI between cleaned and raw, and cleaned and normal ASR
        mi_values[i, 0] = mutual_info_score(disc_normal_asr, disc_raw)
        mi_values[i, 1] = mutual_info_score(disc_cleaned, disc_raw)

    # Plotting the comparison using a boxplot
    fig, ax = plt.subplots(figsize=(12, 4))
    bplot = ax.boxplot(mi_values, patch_artist=True, showmeans=True, meanline=True,
                       positions=[1, 2], widths=0.2)

    # Set box colors and other properties
    colors = ['#AED6F1', '#F9E79F']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)  # Box fill color
        patch.set_edgecolor('black')  # Box edge color

    # Calculate mean MI for each comparison
    mean_mi_clean_raw = np.mean(mi_values[:, 0])
    mean_mi_clean_asr = np.mean(mi_values[:, 1])

    # Set title and axis labels
    ax.set_title('Mutual Information (MI) Comparison', fontsize=18)
    ax.set_ylabel('MI Value', fontsize=14)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['ASR', 'MASR'], fontsize=12)

    # Show grid lines
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    legend_labels = [f'ASR (Mean MI: {mean_mi_clean_raw:.2f})',
                     f'MASR (Mean MI: {mean_mi_clean_asr:.2f})']
    ax.legend(handles=[plt.Rectangle((0, 0), 0.5, 2, color=color) for color in colors],
              labels=legend_labels, loc='best', framealpha=1)

    # Display the plot
    plt.tight_layout()
    plt.show()


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


def calculate_rmse(data1, data2):
    """
    计算两个数据集之间的均方根误差（RMSE）。
    参数：
    data1 : numpy.ndarray
        第一个数据集。
    data2 : numpy.ndarray
        第二个数据集。
    返回值：
    rmse_per_channel : numpy.ndarray
        RMSE的值。
    """
    rmse_per_channel = np.sqrt(np.mean((data1 - data2) ** 2, axis=(0, 2)))
    return rmse_per_channel


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
    ax.set_xticklabels(['ASR', 'MASR'], fontsize=12)

    # 显示网格线
    ax.grid(True, linestyle='--', alpha=0.7)

    # 添加图例说明
    legend_labels = [f'ASR (Mean: {mean_nmse_normal_asr:.3})', f'MASR (Mean: {mean_nmse_clean:.3})']
    ax.legend(handles=[plt.Rectangle((0, 0), 0.5, 2, color=color) for color in colors], labels=legend_labels,
              framealpha=1, loc='best')

    # 显示图形
    plt.show()


def compare_rmse(cleand_data, raw_data_selected_channels, normal_asr, sfreq=250, fmin=0.5, fmax=50):
    """
    绘制处理后数据及另一个数据集的均方根误差（RMSE）的箱形图。
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
    # Calculate RMSE
    rmse_clean = calculate_rmse(cleand_data, raw_data_selected_channels)
    rmse_normal_asr = calculate_rmse(normal_asr, raw_data_selected_channels)

    # Create boxplot
    fig, ax = plt.subplots(figsize=(12, 4))
    bplot = ax.boxplot([rmse_normal_asr, rmse_clean], patch_artist=True, showmeans=True, meanline=True,
                       positions=[1, 2], widths=0.2)

    # Set box colors and other properties
    colors = ['#AED6F1', '#F9E79F']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)  # Box fill color
        patch.set_edgecolor('black')  # Box edge color

    # Calculate mean RMSE for each dataset
    mean_rmse_clean = np.mean(rmse_clean)
    mean_rmse_normal_asr = np.mean(rmse_normal_asr)

    # Set title and axis labels
    ax.set_title('Root Mean Squared Error (RMSE) Comparison', fontsize=18)
    ax.set_ylabel('RMSE', fontsize=14)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['ASR', 'MASR'], fontsize=12)

    # Show grid lines
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    legend_labels = [f'ASR (Mean: {mean_rmse_normal_asr:.3})', f'MASR (Mean: {mean_rmse_clean:.3})']
    ax.legend(handles=[plt.Rectangle((0, 0), 0.5, 2, color=color) for color in colors], labels=legend_labels,
              framealpha=1, loc='best')

    # Display the plot
    plt.show()


def compare_snr(cleand_data, raw_data_selected_channels, normal_asr, sfreq=250, fmin=0.5,
                fmax=50):
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
    signal_power1 = calculate_power(normal_asr, sfreq, fmin, fmax)
    # 计算信噪比（SNR）并转换为分贝（dB）
    epsilon = 1e-10  # 防止除以零
    snr_linear = signal_power / (noise_power + epsilon)
    normal_asr_snr_linear = signal_power1 / (normal_asr_power + epsilon)

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
    ax.set_xticklabels(['ASR', 'MASR'], fontsize=12)

    # 显示网格线
    ax.grid(True, linestyle='--', alpha=0.7)

    # 添加图例说明
    legend_labels = [f'ASR (Mean: {mean_snr_normal_asr:.2f} dB)', f'MASR (Mean: {mean_snr_asr:.2f} dB)']
    ax.legend(handles=[plt.Rectangle((0, 0), 0.5, 2, color=color) for color in colors], labels=legend_labels,
              framealpha=1, loc='best')

    # 显示图形
    plt.show()


def compare_metrics(cleand_data, raw_data_selected_channels, normal_asr, picard_eeg, SSP_eeg, sfreq=250, fmin=0.5,
                    fmax=100):
    """
    绘制处理后数据及另一个数据集的归一化均方误差（NMSE）、均方根误差（RMSE）、信噪比（SNR）和互信息（MI）的箱形图。
    参数：
    cleand_data : numpy.ndarray
        处理后的数据数组。
    raw_data_selected_channels : numpy.ndarray
        原始数据中选择的通道。
    normal_asr : numpy.ndarray
        另一个数据集。
    raw_data_eog_channels : numpy.ndarray
        原始数据中的眼动通道。
    sfreq : int
        采样频率。
    fmin : int
        最小频率。
    fmax : int
        最大频率。
    """
    # Calculate RMSE
    rmse_clean = calculate_rmse(cleand_data, raw_data_selected_channels)
    rmse_normal_asr = calculate_rmse(normal_asr, raw_data_selected_channels)
    rmse_picard = calculate_rmse(picard_eeg, raw_data_selected_channels)
    rmse_SSP = calculate_rmse(SSP_eeg, raw_data_selected_channels)
    # Calculate SNR
    signal_power = calculate_power(cleand_data, sfreq, fmin, fmax)
    noise_power = calculate_power(cleand_data - raw_data_selected_channels, sfreq, fmin, fmax)
    snr_linear = signal_power / (noise_power + 1e-10)
    snr_db = 10 * np.log10(snr_linear)
    normal_asr_power = calculate_power(normal_asr - raw_data_selected_channels, sfreq, fmin, fmax)
    normal_asr_snr_linear = signal_power / (normal_asr_power + 1e-10)
    normal_asr_snr_db = 10 * np.log10(normal_asr_snr_linear)
    picard_power = calculate_power(abs(picard_eeg - raw_data_selected_channels), sfreq, fmin, fmax)
    picard_snr_linear = signal_power / (picard_power + 1e-10)
    picard_snr_db = 10 * np.log10(picard_snr_linear)
    SSP_power = calculate_power(abs(SSP_eeg - raw_data_selected_channels), sfreq, fmin, fmax)
    SSP_snr_linear = signal_power / (SSP_power + 1e-10)
    SSP_snr_db = 10 * np.log10(SSP_snr_linear)

    # Calculate MI
    n_epochs, n_channels, _ = cleand_data.shape

    # Initialize an array to store MI values
    mi_values = np.zeros((n_channels, 4))  # Two columns for the two comparisons

    for i in range(n_channels):
        # Discretize the signals
        disc_cleaned = discretize_signal(cleand_data[:, i, :].flatten())
        disc_raw = discretize_signal(raw_data_selected_channels[:, i, :].flatten())
        disc_normal_asr = discretize_signal(normal_asr[:, i, :].flatten())
        disc_picard = discretize_signal(picard_eeg[:, i, :].flatten())
        disc_SSP = discretize_signal(SSP_eeg[:, i, :].flatten())

        # Calculate MI between cleaned and raw, and cleaned and normal ASR
        mi_values[i, 0] = mutual_info_score(disc_picard, disc_raw)
        mi_values[i, 1] = mutual_info_score(disc_SSP, disc_raw)
        mi_values[i, 2] = mutual_info_score(disc_normal_asr, disc_raw)
        mi_values[i, 3] = mutual_info_score(disc_cleaned, disc_raw)

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 6))

    # Plot RMSE
    bplot_rmse = axs[0, 0].boxplot([rmse_picard, rmse_SSP, rmse_normal_asr, rmse_clean], patch_artist=True,
                                   showmeans=True,
                                   meanline=True,
                                   positions=[1, 2, 3, 4], widths=0.15, showfliers=False)
    axs[0, 0].set_title('Root Mean Squared Error (RMSE) Comparison', fontsize=14)
    axs[0, 0].set_ylabel('RMSE', fontsize=12)
    axs[0, 0].set_xticks([1, 2, 3, 4])
    axs[0, 0].set_xticklabels(['Picard', 'SSP', 'ASR', 'MASR'], fontsize=10)
    axs[0, 0].xaxis.set_minor_locator(MultipleLocator(0.1))  # 设置x轴次要刻度间隔为0.1
    axs[0, 0].yaxis.set_minor_locator(MultipleLocator(0.1))
    # Calculate mean RMSE for each dataset
    mean_rmse_clean = np.mean(rmse_clean)
    mean_rmse_normal_asr = np.mean(rmse_normal_asr)
    mean_rmse_picard = np.mean(rmse_picard)
    mean_rmse_SSP = np.mean(rmse_SSP)
    legend_labels = [f'Picard (Mean: {mean_rmse_picard:.3})', f'SSP (Mean: {mean_rmse_SSP:.3})',
                     f'ASR (Mean: {mean_rmse_normal_asr:.3})',
                     f'MASR (Mean: {mean_rmse_clean:.3})']
    axs[0, 0].legend(
        handles=[plt.Rectangle((0, 0), 0.5, 2, color='#AED6F1'), plt.Rectangle((0, 0), 0.5, 2, color='#AED6F1'),
                 plt.Rectangle((0, 0), 0.5, 2, color='#F9E79F'),
                 plt.Rectangle((0, 0), 0.5, 2, color='#09569F')],
        labels=legend_labels, framealpha=1, loc='best')

    # Plot SNR
    bplot_snr = axs[1, 0].boxplot([picard_snr_db, SSP_snr_db, normal_asr_snr_db, snr_db], patch_artist=True,
                                  showmeans=True,
                                  meanline=True,
                                  positions=[1, 2, 3, 4], widths=0.15, showfliers=False)
    axs[1, 0].set_title('Signal to Noise Ratio (SNR) Comparison', fontsize=14)
    axs[1, 0].set_ylabel('SNR (dB)', fontsize=12)
    axs[1, 0].set_xticks([1, 2, 3, 4])
    axs[1, 0].set_xticklabels(['Picard', 'SSP', 'ASR', 'MASR'], fontsize=10)
    axs[1, 0].xaxis.set_minor_locator(MultipleLocator(0.1))  # 设置x轴次要刻度间隔为0.1
    axs[1, 0].yaxis.set_minor_locator(MultipleLocator(0.1))
    mean_snr_asr = np.mean(snr_db)
    mean_snr_normal_asr = np.mean(normal_asr_snr_db)
    mean_picard_snr_db = np.mean(picard_snr_db)
    mean_SSP_snr_db = np.mean(SSP_snr_db)
    legend_labels = [f'Picard (Mean: {mean_picard_snr_db:.3})', f'SSP (Mean: {mean_SSP_snr_db:.3})',
                     f'ASR (Mean: {mean_snr_normal_asr:.3})',
                     f'MASR (Mean: {mean_snr_asr:.3})']
    axs[1, 0].legend(
        handles=[plt.Rectangle((0, 0), 0.5, 2, color='#AED6F1'), plt.Rectangle((0, 0), 0.5, 2, color='#AED6F1'),
                 plt.Rectangle((0, 0), 0.5, 2, color='#F9E79F', ),
                 plt.Rectangle((0, 0), 0.5, 2, color='#09569F')],
        labels=legend_labels, framealpha=1, loc='best')

    # Plot NMSE
    nmse_clean = calculate_nmse(cleand_data, raw_data_selected_channels)
    nmse_normal_asr = calculate_nmse(normal_asr, raw_data_selected_channels)
    nmse_picard = calculate_nmse(picard_eeg, raw_data_selected_channels)
    nmse_SSP = calculate_nmse(SSP_eeg, raw_data_selected_channels)

    bplot_nmse = axs[0, 1].boxplot([nmse_picard, nmse_SSP, nmse_normal_asr, nmse_clean], patch_artist=True,
                                   showmeans=True,
                                   meanline=True,
                                   positions=[1, 2, 3, 4], widths=0.15, showfliers=False)
    axs[0, 1].set_title('Normalized Mean Squared Error (NMSE) Comparison', fontsize=14)
    axs[0, 1].set_ylabel('NMSE', fontsize=12)
    axs[0, 1].set_xticks([1, 2, 3, 4])
    axs[0, 1].set_xticklabels(['Picard', 'SSP', 'ASR', 'MASR'], fontsize=10)
    axs[0, 1].xaxis.set_minor_locator(MultipleLocator(0.1))  # 设置x轴次要刻度间隔为0.1
    axs[0, 1].yaxis.set_minor_locator(MultipleLocator(0.1))
    # 计算每个数据集的平均NMSE
    mean_nmse_clean = np.mean(nmse_clean)
    mean_nmse_normal_asr = np.mean(nmse_normal_asr)
    mean_picard_snr_db = np.mean(nmse_picard)
    mean_SSP_snr_db = np.mean(nmse_SSP)
    legend_labels = [f'Picard (Mean: {mean_picard_snr_db:.3})', f'SSP (Mean: {mean_SSP_snr_db:.3})',
                     f'ASR (Mean: {mean_nmse_normal_asr:.3})',
                     f'MASR (Mean: {mean_nmse_clean:.3})']
    axs[0, 1].legend(
        handles=[plt.Rectangle((0, 0), 0.5, 2, color='#AED6F1'), plt.Rectangle((0, 0), 0.5, 2, color='#AED6F1'),
                 plt.Rectangle((0, 0), 0.5, 2, color='#F9E79F', ),
                 plt.Rectangle((0, 0), 0.5, 2, color='#09569F')],
        labels=legend_labels, framealpha=1, loc='best')

    # Plot MI
    bplot_mi = axs[1, 1].boxplot(mi_values, patch_artist=True, showmeans=True, meanline=True, positions=[1, 2, 3, 4],
                                 widths=0.15, showfliers=False)
    axs[1, 1].set_title('Mutual Information (MI) Comparison', fontsize=14)
    axs[1, 1].set_ylabel('MI', fontsize=12)
    axs[1, 1].set_xticks([1, 2, 3, 4])
    axs[1, 1].set_xticklabels(['Picard', 'SSP', 'ASR', 'MASR'], fontsize=10)
    axs[1, 1].xaxis.set_minor_locator(MultipleLocator(0.1))  # 设置x轴次要刻度间隔为0.1
    axs[1, 1].yaxis.set_minor_locator(MultipleLocator(0.1))
    # Add legend
    # Calculate mean MI for each comparison
    mean_mi_picard = np.mean(mi_values[:, 0])
    mean_mi_SSP = np.mean(mi_values[:, 1])
    mean_mi_clean_raw = np.mean(mi_values[:, 2])
    mean_mi_clean_asr = np.mean(mi_values[:, 3])
    legend_labels = [f'Picard (Mean MI: {mean_mi_picard:.2f})', f'SSP (Mean MI: {mean_mi_SSP:.2f})',
                     f'ASR (Mean MI: {mean_mi_clean_raw:.2f})',
                     f'MASR (Mean MI: {mean_mi_clean_asr:.2f})']
    axs[1, 1].legend(
        handles=[plt.Rectangle((0, 0), 0.5, 2, color='#AED6F1'), plt.Rectangle((0, 0), 0.5, 2, color='#AED6F1'),
                 plt.Rectangle((0, 0), 0.5, 2, color='#F9E79F'),
                 plt.Rectangle((0, 0), 0.5, 2, color='#09569F')],
        labels=legend_labels, framealpha=1,
        loc='best')

    # Set box colors
    for bplot in [bplot_rmse, bplot_snr, bplot_nmse, bplot_mi]:
        for patch, color in zip(bplot['boxes'], ['#AED6F1', '#AED6F1', '#F9E79F', '#09569F']):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')

    # Show grid lines
    for ax in axs.flatten():
        ax.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout and display plot
    plt.tight_layout()
    plt.show()


def compare_metrics1(cleand_data, raw_data_selected_channels, normal_asr, picard_eeg, SSP_eeg, sfreq=250, fmin=0.5,
                     fmax=100):
    """
    绘制处理后数据及另一个数据集的归一化均方误差（NMSE）、均方根误差（RMSE）、信噪比（SNR）和互信息（MI）的箱形图。
    参数：
    cleand_data : numpy.ndarray
        处理后的数据数组。
    raw_data_selected_channels : numpy.ndarray
        原始数据中选择的通道。
    normal_asr : numpy.ndarray
        另一个数据集。
    raw_data_eog_channels : numpy.ndarray
        原始数据中的眼动通道。
    sfreq : int
        采样频率。
    fmin : int
        最小频率。
    fmax : int
        最大频率。
    """
    # Calculate RMSE
    rmse_clean = calculate_rmse(cleand_data, raw_data_selected_channels)
    rmse_normal_asr = calculate_rmse(normal_asr, raw_data_selected_channels)
    rmse_picard = calculate_rmse(picard_eeg, raw_data_selected_channels)
    rmse_SSP = calculate_rmse(SSP_eeg, raw_data_selected_channels)
    # Calculate SNR
    signal_power = calculate_power(cleand_data, sfreq, fmin, fmax)
    noise_power = calculate_power(cleand_data - raw_data_selected_channels, sfreq, fmin, fmax)
    snr_linear = signal_power / (noise_power + 1e-10)
    snr_db = 10 * np.log10(snr_linear)
    normal_asr_power = calculate_power(normal_asr - raw_data_selected_channels, sfreq, fmin, fmax)
    normal_asr_snr_linear = signal_power / (normal_asr_power + 1e-10)
    normal_asr_snr_db = 10 * np.log10(normal_asr_snr_linear)
    picard_power = calculate_power(abs(picard_eeg - raw_data_selected_channels), sfreq, fmin, fmax)
    picard_snr_linear = signal_power / (picard_power + 1e-10)
    picard_snr_db = 10 * np.log10(picard_snr_linear)
    SSP_power = calculate_power(abs(SSP_eeg - raw_data_selected_channels), sfreq, fmin, fmax)
    SSP_snr_linear = signal_power / (SSP_power + 1e-10)
    SSP_snr_db = 10 * np.log10(SSP_snr_linear)

    # Calculate MI
    n_epochs, n_channels, _ = cleand_data.shape

    # Initialize an array to store MI values
    mi_values = np.zeros((n_channels, 4))  # Two columns for the two comparisons

    for i in range(n_channels):
        # Discretize the signals
        disc_cleaned = discretize_signal(cleand_data[:, i, :].flatten())
        disc_raw = discretize_signal(raw_data_selected_channels[:, i, :].flatten())
        disc_normal_asr = discretize_signal(normal_asr[:, i, :].flatten())
        disc_picard = discretize_signal(picard_eeg[:, i, :].flatten())
        disc_SSP = discretize_signal(SSP_eeg[:, i, :].flatten())

        # Calculate MI between cleaned and raw, and cleaned and normal ASR
        mi_values[i, 0] = mutual_info_score(disc_picard, disc_raw)
        mi_values[i, 1] = mutual_info_score(disc_SSP, disc_raw)
        mi_values[i, 2] = mutual_info_score(disc_normal_asr, disc_raw)
        mi_values[i, 3] = mutual_info_score(disc_cleaned, disc_raw)

    # Create subplots
    fig, axs = plt.subplots(4, 1, figsize=(8, 14))

    # Plot RMSE
    bplot_rmse = axs[0].boxplot([rmse_picard, rmse_SSP, rmse_normal_asr, rmse_clean], patch_artist=True,
                                showmeans=True, meanline=True, positions=[1, 2, 3, 4], widths=0.5, showfliers=False)
    axs[0].set_title('Root Mean Squared Error (RMSE) Comparison', fontsize=14)
    axs[0].set_ylabel('RMSE', fontsize=12)
    axs[0].set_xticks([1, 2, 3, 4])
    axs[0].set_xticklabels(['Picard', 'SSP', 'ASR', 'MASR'], fontsize=10)
    axs[0].xaxis.set_minor_locator(MultipleLocator(0.1))  # 设置x轴次要刻度间隔为0.1
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.1))
    # Calculate mean RMSE for each dataset
    mean_rmse_clean = np.mean(rmse_clean)
    mean_rmse_normal_asr = np.mean(rmse_normal_asr)
    mean_rmse_picard = np.mean(rmse_picard)
    mean_rmse_SSP = np.mean(rmse_SSP)
    legend_labels = [f'Picard (Mean: {mean_rmse_picard:.3})', f'SSP (Mean: {mean_rmse_SSP:.3})',
                     f'ASR (Mean: {mean_rmse_normal_asr:.3})', f'MASR (Mean: {mean_rmse_clean:.3})']
    axs[0].legend(
        handles=[plt.Rectangle((0, 0), 0.5, 2, color='#829CBC'), plt.Rectangle((0, 0), 0.5, 2, color='#6290C8'),
                 plt.Rectangle((0, 0), 0.5, 2, color='#376996'), plt.Rectangle((0, 0), 0.5, 2, color='#1F487E')],
        labels=legend_labels, framealpha=1, loc='best')

    # Plot SNR
    bplot_snr = axs[1].boxplot([picard_snr_db, SSP_snr_db, normal_asr_snr_db, snr_db], patch_artist=True,
                               showmeans=True, meanline=True, positions=[1, 2, 3, 4], widths=0.5, showfliers=False)
    axs[1].set_title('Signal to Noise Ratio (SNR) Comparison', fontsize=14)
    axs[1].set_ylabel('SNR (dB)', fontsize=12)
    axs[1].set_xticks([1, 2, 3, 4])
    axs[1].set_xticklabels(['Picard', 'SSP', 'ASR', 'MASR'], fontsize=10)
    axs[1].xaxis.set_minor_locator(MultipleLocator(0.1))  # 设置x轴次要刻度间隔为0.1
    axs[1].yaxis.set_minor_locator(MultipleLocator(0.1))
    mean_snr_asr = np.mean(snr_db)
    mean_snr_normal_asr = np.mean(normal_asr_snr_db)
    mean_picard_snr_db = np.mean(picard_snr_db)
    mean_SSP_snr_db = np.mean(SSP_snr_db)
    legend_labels = [f'Picard (Mean: {mean_picard_snr_db:.3})', f'SSP (Mean: {mean_SSP_snr_db:.3})',
                     f'ASR (Mean: {mean_snr_normal_asr:.3})', f'MASR (Mean: {mean_snr_asr:.3})']
    axs[1].legend(
        handles=[plt.Rectangle((0, 0), 0.5, 2, color='#829CBC'), plt.Rectangle((0, 0), 0.5, 2, color='#6290C8'),
                 plt.Rectangle((0, 0), 0.5, 2, color='#376996'), plt.Rectangle((0, 0), 0.5, 2, color='#1F487E')],
        labels=legend_labels, framealpha=1, loc='best')

    # Plot NMSE
    nmse_clean = calculate_nmse(cleand_data, raw_data_selected_channels)
    nmse_normal_asr = calculate_nmse(normal_asr, raw_data_selected_channels)
    nmse_picard = calculate_nmse(picard_eeg, raw_data_selected_channels)
    nmse_SSP = calculate_nmse(SSP_eeg, raw_data_selected_channels)

    bplot_nmse = axs[2].boxplot([nmse_picard, nmse_SSP, nmse_normal_asr, nmse_clean], patch_artist=True,
                                showmeans=True, meanline=True, positions=[1, 2, 3, 4], widths=0.5, showfliers=False)
    axs[2].set_title('Normalized Mean Squared Error (NMSE) Comparison', fontsize=14)
    axs[2].set_ylabel('NMSE', fontsize=12)
    axs[2].set_xticks([1, 2, 3, 4])
    axs[2].set_xticklabels(['Picard', 'SSP', 'ASR', 'MASR'], fontsize=10)
    axs[2].xaxis.set_minor_locator(MultipleLocator(0.1))  # 设置x轴次要刻度间隔为0.1
    axs[2].yaxis.set_minor_locator(MultipleLocator(0.1))
    # 计算每个数据集的平均NMSE
    mean_nmse_clean = np.mean(nmse_clean)
    mean_nmse_normal_asr = np.mean(nmse_normal_asr)
    mean_picard_snr_db = np.mean(nmse_picard)
    mean_SSP_snr_db = np.mean(nmse_SSP)
    legend_labels = [f'Picard (Mean: {mean_picard_snr_db:.3})', f'SSP (Mean: {mean_SSP_snr_db:.3})',
                     f'ASR (Mean: {mean_nmse_normal_asr:.3})', f'MASR (Mean: {mean_nmse_clean:.3})']
    axs[2].legend(
        handles=[plt.Rectangle((0, 0), 0.5, 2, color='#829CBC'), plt.Rectangle((0, 0), 0.5, 2, color='#6290C8'),
                 plt.Rectangle((0, 0), 0.5, 2, color='#376996'), plt.Rectangle((0, 0), 0.5, 2, color='#1F487E')],
        labels=legend_labels, framealpha=1, loc='best')

    # Plot MI
    bplot_mi = axs[3].boxplot(mi_values, patch_artist=True, showmeans=True, meanline=True, positions=[1, 2, 3, 4],
                              widths=0.5, showfliers=False)
    axs[3].set_title('Mutual Information (MI) Comparison', fontsize=14)
    axs[3].set_ylabel('MI', fontsize=12)
    axs[3].set_xticks([1, 2, 3, 4])
    axs[3].set_xticklabels(['Picard', 'SSP', 'ASR', 'MASR'], fontsize=10)
    axs[3].xaxis.set_minor_locator(MultipleLocator(0.1))  # 设置x轴次要刻度间隔为0.1
    axs[3].yaxis.set_minor_locator(MultipleLocator(0.1))
    # Add legend
    # Calculate mean MI for each comparison
    mean_mi_picard = np.mean(mi_values[:, 0])
    mean_mi_SSP = np.mean(mi_values[:, 1])
    mean_mi_clean_raw = np.mean(mi_values[:, 2])
    mean_mi_clean_asr = np.mean(mi_values[:, 3])
    legend_labels = [f'Picard (Mean MI: {mean_mi_picard:.2f})', f'SSP (Mean MI: {mean_mi_SSP:.2f})',
                     f'ASR (Mean MI: {mean_mi_clean_raw:.2f})', f'MASR (Mean MI: {mean_mi_clean_asr:.2f})']
    axs[3].legend(
        handles=[plt.Rectangle((0, 0), 0.5, 2, color='#829CBC'), plt.Rectangle((0, 0), 0.5, 2, color='#6290C8'),
                 plt.Rectangle((0, 0), 0.5, 2, color='#376996'), plt.Rectangle((0, 0), 0.5, 2, color='#1F487E')],
        labels=legend_labels, framealpha=1, loc='best')

    # Set box colors
    for bplot in [bplot_rmse, bplot_snr, bplot_nmse, bplot_mi]:
        for patch, color in zip(bplot['boxes'], ['#829CBC', '#6290C8', '#376996', '#1F487E']):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')

    # Show grid lines
    for ax in axs.flatten():
        ax.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout and display plot
    plt.tight_layout()
    plt.show()
    compare_metrics3(
        np.mean(rmse_clean), np.mean(rmse_normal_asr), np.mean(rmse_picard), np.mean(rmse_SSP),
        np.mean(nmse_clean), np.mean(nmse_normal_asr), np.mean(nmse_picard), np.mean(nmse_SSP),
        np.mean(snr_db), np.mean(normal_asr_snr_db), np.mean(picard_snr_db), np.mean(SSP_snr_db),
        np.mean(mi_values[:, 3]), np.mean(mi_values[:, 2]), np.mean(mi_values[:, 0]), np.mean(mi_values[:, 1])
    )


import numpy as np
import matplotlib.pyplot as plt


def compare_metrics2(rmse_values, nmse_values, snr_values, mi_values):
    """
    绘制四种方法的雷达图，表示四个指标（1/RMSE、1/NMSE、SNR、MI）。
    参数：
    rmse_values : list or numpy.ndarray
        各方法的RMSE值列表。
    nmse_values : list or numpy.ndarray
        各方法的NMSE值列表。
    snr_values : list or numpy.ndarray
        各方法的SNR值列表。
    mi_values : list or numpy.ndarray
        各方法的MI值列表。
    """

    # 1. 计算指标的倒数或直接值
    inv_rmse = 1 / np.array(rmse_values)
    inv_nmse = 1 / np.array(nmse_values)
    snr = np.array(snr_values)
    mi = np.array(mi_values)

    # 2. 数据归一化
    all_metrics = np.array([inv_rmse, inv_nmse, snr, mi])

    # 添加偏移量以避免过小的值
    offset = 0.5  # 偏移量，可调整
    adjusted_metrics = all_metrics + offset

    # 使用非线性变换（如平方根）增强可视化
    transformed_metrics = np.sqrt(adjusted_metrics)

    # 计算最大和最小值进行归一化
    min_value = np.min(transformed_metrics)
    max_value = np.max(transformed_metrics)

    norm_metrics = (transformed_metrics - min_value) / (max_value - min_value + 1e-10)  # 正规化

    # 3. 雷达图标签设置
    methods = ['Picard', 'SSP', 'ASR', 'MASR']
    labels = ['1/RMSE', '1/NMSE', 'SNR', 'MI']
    num_vars = len(labels)

    # 4. 雷达图布局设置
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    norm_metrics = np.concatenate((norm_metrics, norm_metrics[:, [0]]), axis=1)  # 确保图形闭合
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # 5. 绘制每种方法的雷达图线
    for idx, method in enumerate(methods):
        ax.plot(angles, norm_metrics[idx], linewidth=2, label=method)
        ax.fill(angles, norm_metrics[idx], alpha=0.5)

    # 6. 添加标签和标题
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Methods Comparison on RMSE, NMSE, SNR, and MI", va='bottom')
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

    # 显示图形
    plt.show()


def compare_metrics3(rmse_clean, rmse_normal_asr, rmse_picard, rmse_SSP,
                     nmse_clean, nmse_normal_asr, nmse_picard, nmse_SSP,
                     snr_clean, snr_normal_asr, snr_picard, snr_SSP,
                     mi_clean, mi_normal_asr, mi_picard, mi_SSP):
    """
    绘制四种方法的雷达图，表示四个指标（1/RMSE、1/NMSE、SNR、MI）。
    参数：
    - rmse_clean, rmse_normal_asr, rmse_picard, rmse_SSP: RMSE值。
    - nmse_clean, nmse_normal_asr, nmse_picard, nmse_SSP: NMSE值。
    - snr_clean, snr_normal_asr, snr_picard, snr_SSP: SNR值。
    - mi_clean, mi_normal_asr, mi_picard, mi_SSP: MI值。
    """

    # 1. 将每种方法的四个指标组合在一起
    methods_metrics = {
        'Clean': [1 / rmse_clean, 1 / nmse_clean, snr_clean, mi_clean],
        'ASR': [1 / rmse_normal_asr, 1 / nmse_normal_asr, snr_normal_asr, mi_normal_asr],
        'Picard': [1 / rmse_picard, 1 / nmse_picard, snr_picard, mi_picard],
        'SSP': [1 / rmse_SSP, 1 / nmse_SSP, snr_SSP, mi_SSP]
    }

    # 2. 数据归一化
    metrics = np.array(list(methods_metrics.values()))
    norm_metrics = (metrics - metrics.min(axis=0)) / (metrics.max(axis=0) - metrics.min(axis=0) + 1e-10)

    # 3. 设置雷达图参数
    labels = ['1/RMSE', '1/NMSE', 'SNR', 'MI']
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    norm_metrics = np.concatenate((norm_metrics, norm_metrics[:, [0]]), axis=1)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # 4. 绘制每种方法的雷达图线
    for i, (method, values) in enumerate(methods_metrics.items()):
        ax.plot(angles, norm_metrics[i], linewidth=1.5, label=method)
        ax.fill(angles, norm_metrics[i], alpha=0.25)

    # 5. 添加标签和标题
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Methods Comparison on RMSE, NMSE, SNR, and MI", va='bottom')
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

    # 显示图形
    plt.show()
