import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
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


def compare_metrics_radar(rmse_picard, rmse_SSP, rmse_opt, rmse_xdawn, rmse_asr, rmse_masr,
                          nmse_picard, nmse_SSP, nmse_opt, nmse_xdawn, nmse_asr, nmse_masr,
                          snr_picard, snr_SSP, snr_opt, snr_xdawn, snr_asr, snr_masr,
                          mi_picard, mi_SSP, mi_opt, mi_xdawn, mi_asr, mi_masr):
    """更新后的雷达图数据生成函数"""
    methods_metrics = {
        'Picard': [1 / rmse_picard, 1 / nmse_picard, snr_picard, mi_picard],
        'SSP': [1 / rmse_SSP, 1 / nmse_SSP, snr_SSP, mi_SSP],
        'OTP': [1 / rmse_opt, 1 / nmse_opt, snr_opt, mi_opt],
        'XDAWN': [1 / rmse_xdawn, 1 / nmse_xdawn, snr_xdawn, mi_xdawn],  # 新增
        'ASR': [1 / rmse_asr, 1 / nmse_asr, snr_asr, mi_asr],
        'MASR': [1 / rmse_masr, 1 / nmse_masr, snr_masr, mi_masr]
    }
    return methods_metrics


def compare_metrics1(cleand_data, raw_data_selected_channels,
                     normal_asr, picard_eeg, SSP_eeg, opt_eeg, xdawn_eeg,
                     sfreq=250, fmin=0.5, fmax=100):
    """
    对比各方法（Picard-ICA、SSP、OTP、ASR、MASR）与原始数据在RMSE、NMSE、SAR、MI指标上的差异，
    绘制箱型图和雷达图进行比较。

    参数：
        cleand_data : numpy.ndarray
            MASR（清理后的数据）数组。
        raw_data_selected_channels : numpy.ndarray
            原始数据中选择的通道数据。
        normal_asr : numpy.ndarray
            ASR 处理后的数据。
        picard_eeg : numpy.ndarray
            Picard-ICA 处理后的数据。
        SSP_eeg : numpy.ndarray
            SSP 处理后的数据。
        opt_eeg : numpy.ndarray
            新增的 OTP 处理后的数据。
        sfreq : int
            采样频率。
        fmin : float
            计算功率时的最小频率。
        fmax : float
            计算功率时的最大频率。
    """
    # 计算 RMSE
    rmse_picard = calculate_rmse(picard_eeg, raw_data_selected_channels)
    rmse_SSP = calculate_rmse(SSP_eeg, raw_data_selected_channels)
    rmse_opt = calculate_rmse(opt_eeg, raw_data_selected_channels)
    rmse_xdawn = calculate_rmse(xdawn_eeg, raw_data_selected_channels)
    rmse_asr = calculate_rmse(normal_asr, raw_data_selected_channels)
    rmse_masr = calculate_rmse(cleand_data, raw_data_selected_channels)

    # 计算 SAR（信噪比）
    # 以 MASR（cleand_data）作为参考信号计算功率
    signal_power = calculate_power(cleand_data, sfreq, fmin, fmax)

    noise_power_picard = calculate_power(np.abs(picard_eeg - raw_data_selected_channels), sfreq, fmin, fmax)
    snr_picard = 10 * np.log10(signal_power / (noise_power_picard + 1e-10))

    noise_power_SSP = calculate_power(np.abs(SSP_eeg - raw_data_selected_channels), sfreq, fmin, fmax)
    snr_SSP = 10 * np.log10(signal_power / (noise_power_SSP + 1e-10))

    noise_power_opt = calculate_power(np.abs(opt_eeg - raw_data_selected_channels), sfreq, fmin, fmax)
    snr_opt = 10 * np.log10(signal_power / (noise_power_opt + 1e-10))
    noise_power_xdawn = calculate_power(np.abs(xdawn_eeg - raw_data_selected_channels), sfreq, fmin, fmax)
    snr_xdawn = 10 * np.log10(signal_power / (noise_power_xdawn + 1e-10))

    noise_power_asr = calculate_power(normal_asr - raw_data_selected_channels, sfreq, fmin, fmax)
    snr_asr = 10 * np.log10(signal_power / (noise_power_asr + 1e-10))

    noise_power_masr = calculate_power(cleand_data - raw_data_selected_channels, sfreq, fmin, fmax)
    snr_masr = 10 * np.log10(signal_power / (noise_power_masr + 1e-10))

    # 计算 NMSE
    nmse_picard = calculate_nmse(picard_eeg, raw_data_selected_channels)
    nmse_SSP = calculate_nmse(SSP_eeg, raw_data_selected_channels)
    nmse_opt = calculate_nmse(opt_eeg, raw_data_selected_channels)
    nmse_xdawn = calculate_nmse(xdawn_eeg, raw_data_selected_channels)
    nmse_asr = calculate_nmse(normal_asr, raw_data_selected_channels)
    nmse_masr = calculate_nmse(cleand_data, raw_data_selected_channels)

    # 计算互信息 (MI)
    n_epochs, n_channels, _ = cleand_data.shape
    # 原来的计算顺序为：Picard, SSP, ASR, MASR, OTP
    # 现在调整为：Picard, SSP, OTP, ASR, MASR
    mi_values = np.zeros((n_channels, 6))
    for i in range(n_channels):
        disc_raw = discretize_signal(raw_data_selected_channels[:, i, :].flatten())
        disc_picard = discretize_signal(picard_eeg[:, i, :].flatten())
        disc_SSP = discretize_signal(SSP_eeg[:, i, :].flatten())
        disc_normal = discretize_signal(normal_asr[:, i, :].flatten())
        disc_masr = discretize_signal(cleand_data[:, i, :].flatten())
        disc_opt = discretize_signal(opt_eeg[:, i, :].flatten())
        disc_xdawn = discretize_signal(xdawn_eeg[:, i, :].flatten())

        mi_values[i, 0] = mutual_info_score(disc_picard, disc_raw)
        mi_values[i, 1] = mutual_info_score(disc_SSP, disc_raw)
        mi_values[i, 2] = mutual_info_score(disc_opt, disc_raw)  # OTP
        mi_values[i, 3] = mutual_info_score(disc_xdawn, disc_raw)  # Xdawn
        mi_values[i, 4] = mutual_info_score(disc_normal, disc_raw)  # ASR
        mi_values[i, 5] = mutual_info_score(disc_masr, disc_raw)  # MASR

    # ------------------------- 绘制箱型图 -------------------------
    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(9, 14, figure=fig)

    # RMSE 箱型图
    ax1 = fig.add_subplot(gs[0:4, 0:4])
    bplot_rmse = ax1.boxplot(
        [rmse_picard, rmse_SSP, rmse_opt, rmse_xdawn, rmse_asr, rmse_masr],
        patch_artist=True, showmeans=True, meanline=True,
        positions=[1, 2, 3, 4, 5, 6], widths=0.5, showfliers=False)
    ax1.set_title('Root Mean Squared Error (RMSE)', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_xticks([1, 2, 3, 4, 5, 6])
    mean_rmse_values = [
        (np.mean(rmse_picard), 'Picard'),
        (np.mean(rmse_SSP), 'SSP'),
        (np.mean(rmse_opt), 'OTP'),
        (np.mean(rmse_xdawn), 'XDAWN'),
        (np.mean(rmse_asr), 'ASR'),
        (np.mean(rmse_masr), 'MASR')
    ]
    labels_rmse = [f'{label}\n({mean:.2g})' for mean, label in mean_rmse_values]
    ax1.set_xticklabels(labels_rmse, fontsize=9)
    ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax1.grid(True, linestyle='--', alpha=0.7)
    colors_box = ['#D85014', '#CADDE6', '#A0A0A0', '#A943A0', '#3FA0C0', '#EDB327']
    for i, patch in enumerate(bplot_rmse['boxes']):
        patch.set_facecolor(colors_box[i])
        patch.set_edgecolor('black')

    # SAR 箱型图
    ax2 = fig.add_subplot(gs[0:4, 10:14])
    bplot_snr = ax2.boxplot(
        [snr_picard, snr_SSP, snr_opt, snr_xdawn, snr_asr, snr_masr],
        patch_artist=True, showmeans=True, meanline=True,
        positions=[1, 2, 3, 4, 5, 6], widths=0.5, showfliers=False)
    ax2.set_title('Signal to Artifact Ratio (SAR)', fontsize=12)
    ax2.set_ylabel('SAR (dB)', fontsize=12)
    ax2.set_xticks([1, 2, 3, 4, 5, 6])
    mean_snr_values = [
        (np.mean(snr_picard), 'Picard'),
        (np.mean(snr_SSP), 'SSP'),
        (np.mean(snr_opt), 'OTP'),
        (np.mean(snr_xdawn), 'XDAWN'),
        (np.mean(snr_asr), 'ASR'),
        (np.mean(snr_masr), 'MASR')
    ]
    labels_snr = [f'{label}\n({mean:.2f})' for mean, label in mean_snr_values]
    ax2.set_xticklabels(labels_snr, fontsize=9)
    ax2.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax2.grid(True, linestyle='--', alpha=0.7)
    for i, patch in enumerate(bplot_snr['boxes']):
        patch.set_facecolor(colors_box[i])
        patch.set_edgecolor('black')

    # NMSE 箱型图
    ax4 = fig.add_subplot(gs[5:9, 0:4])
    bplot_nmse = ax4.boxplot(
        [nmse_picard, nmse_SSP, nmse_opt, nmse_xdawn, nmse_asr, nmse_masr],
        patch_artist=True, showmeans=True, meanline=True,
        positions=[1, 2, 3, 4, 5, 6], widths=0.5, showfliers=False)
    ax4.set_title('Normalized Mean Squared Error (NMSE)', fontsize=12)
    ax4.set_ylabel('NMSE', fontsize=12)
    ax4.set_xticks([1, 2, 3, 4, 5, 6])
    mean_nmse_values = [
        (np.mean(nmse_picard), 'Picard'),
        (np.mean(nmse_SSP), 'SSP'),
        (np.mean(nmse_opt), 'OTP'),
        (np.mean(nmse_xdawn), 'XDAWN'),
        (np.mean(nmse_asr), 'ASR'),
        (np.mean(nmse_masr), 'MASR')
    ]
    labels_nmse = [f'{label}\n({mean:.2f})' for mean, label in mean_nmse_values]
    ax4.set_xticklabels(labels_nmse, fontsize=9)
    ax4.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax4.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax4.grid(True, linestyle='--', alpha=0.7)
    for i, patch in enumerate(bplot_nmse['boxes']):
        patch.set_facecolor(colors_box[i])
        patch.set_edgecolor('black')

    # MI 箱型图
    ax5 = fig.add_subplot(gs[5:9, 10:14])
    bplot_mi = ax5.boxplot(
        [mi_values[:, 0], mi_values[:, 1], mi_values[:, 2], mi_values[:, 3], mi_values[:, 4], mi_values[:, 5]],
        patch_artist=True, showmeans=True, meanline=True,
        positions=[1, 2, 3, 4, 5, 6], widths=0.5, showfliers=False)
    ax5.set_title('Mutual Information (MI)', fontsize=12)
    ax5.set_ylabel('MI', fontsize=12)
    ax5.set_xticks([1, 2, 3, 4, 5, 6])
    mean_mi_values = [
        (np.mean(mi_values[:, 0]), 'Picard'),
        (np.mean(mi_values[:, 1]), 'SSP'),
        (np.mean(mi_values[:, 2]), 'OTP'),
        (np.mean(mi_values[:, 3]), 'XDAWN'),
        (np.mean(mi_values[:, 4]), 'ASR'),
        (np.mean(mi_values[:, 5]), 'MASR')
    ]
    labels_mi = [f'{label}\n({mean:.2f})' for mean, label in mean_mi_values]
    ax5.set_xticklabels(labels_mi, fontsize=9)
    ax5.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax5.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax5.grid(True, linestyle='--', alpha=0.7)
    for i, patch in enumerate(bplot_mi['boxes']):
        patch.set_facecolor(colors_box[i])
        patch.set_edgecolor('black')

    # ------------------------- 绘制雷达图 -------------------------
    ax3 = fig.add_subplot(gs[2:7, 5:9], polar=True)
    # 按新顺序整理指标数据
    methods_metrics = compare_metrics_radar(
        np.mean(rmse_picard), np.mean(rmse_SSP), np.mean(rmse_opt), np.mean(rmse_xdawn), np.mean(rmse_asr),
        np.mean(rmse_masr),
        np.mean(nmse_picard), np.mean(nmse_SSP), np.mean(nmse_opt), np.mean(nmse_xdawn), np.mean(nmse_asr),
        np.mean(nmse_masr),
        np.mean(snr_picard), np.mean(snr_SSP), np.mean(snr_opt), np.mean(snr_xdawn), np.mean(snr_asr),
        np.mean(snr_masr),
        np.mean(mi_values[:, 0]), np.mean(mi_values[:, 1]), np.mean(mi_values[:, 2]), np.mean(mi_values[:, 3]),
        np.mean(mi_values[:, 4]), np.mean(mi_values[:, 5])
    )

    # 提取各方法指标并增加平均值（使用副本避免原地修改）
    methods = list(methods_metrics.keys())  # 假设顺序为：Picard, SSP, OTP, ASR, MASR
    metrics_list = []
    for method in methods:
        vals = methods_metrics[method].copy()  # 复制一份数据
        avg_val = np.mean(vals)
        vals.append(avg_val)
        metrics_list.append(vals)

    metrics_array = np.array(metrics_list)

    # 定义归一化函数，包含非线性映射与区间映射
    def normalize_data(data):
        min_vals = data.min(axis=0)
        max_vals = data.max(axis=0)
        norm_data = (data - min_vals) / (max_vals - min_vals + 1e-10)
        norm_data = norm_data ** 0.5  # 非线性映射
        return norm_data * 0.68 + 0.3  # 映射到 [0.2, 0.8]

    norm_metrics = normalize_data(metrics_array)

    # 添加闭合点，使数据闭合成多边形
    norm_metrics = np.concatenate((norm_metrics, norm_metrics[:, :1]), axis=1)

    # 设置雷达图参数
    radar_labels = ['Average', '1/RMSE', '1/NMSE', 'MI', 'SAR']
    num_vars = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # 定义雷达图颜色（确保颜色数量与方法数量一致，如果方法只有5个，则删去多余的颜色）
    radar_colors = ['#D85014', '#CADDE6', '#A0A0A0', '#A943A0', '#3FA0C0', '#EDB327']

    # 隐藏极坐标框架边缘
    for spine in ax3.spines.values():
        spine.set_visible(False)

    # 自定义绘制顺序：对应的索引为 [5, 4, 0, 1, 2, 3]
    custom_order = [5, 4, 0, 1, 2, 3]

    for idx in custom_order:
        method = methods[idx]
        ax3.fill(angles, norm_metrics[idx], alpha=0.3, color=radar_colors[idx])
        ax3.plot(angles, norm_metrics[idx], linewidth=1.5, label=method, color=radar_colors[idx])

    ax3.set_ylim(0, 1)
    ax3.set_yticklabels([])

    # 设置x轴标签和对应角度，使用 tick_params 调整标签与图形之间的距离
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(radar_labels, fontsize=14)
    ax3.tick_params(axis='x', pad=15)

    # 图例设置，如果需要重新排序，可以调整 new_order 列表
    handles, labels_legend = ax3.get_legend_handles_labels()
    # 如果顺序无需调整，可直接使用默认顺序：
    ax3.legend(handles, labels_legend, loc='upper center',
               bbox_to_anchor=(0.5, -0.1), ncol=3,
               fontsize=10, handlelength=2, handleheight=1, frameon=False)

    ax3.set_theta_offset(np.pi / 2)
    plt.show()

