import cProfile
import pstats

import matplotlib
import pywt
import matplotlib.pyplot as plt
import mne
import numpy as np
from asrpy import ASR
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
from mne.preprocessing import ICA, create_eog_epochs, compute_proj_eog
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from joblib import Parallel, delayed
from scipy.stats import norm
import analyze_merits
import tse
import time
from scipy.cluster.hierarchy import dendrogram, linkage

matplotlib.use('TkAgg')


def wpd(data, max_level):
    wp = pywt.WaveletPacket(data=data, wavelet='db1', mode='symmetric', maxlevel=max_level)
    return wp


def read_raw_gdf(filename):
    raw_gdf = mne.io.read_raw_gdf(filename, stim_channel="auto", verbose='ERROR')

    # 将原通道重命名为10-20系统中，常用的64个电极位置
    raw_gdf.rename_channels(
        {'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 'EEG-3': 'FC2', 'EEG-4': 'FC4',
         'EEG-5': 'C5', 'EEG-C3': 'C3', 'EEG-6': 'C1', 'EEG-Cz': 'Cz', 'EEG-7': 'C2', 'EEG-C4': 'C4', 'EEG-8': 'C6',
         'EEG-9': 'CP3', 'EEG-10': 'CP1', 'EEG-11': 'CPz', 'EEG-12': 'CP2', 'EEG-13': 'CP4',
         'EEG-14': 'P1', 'EEG-15': 'Pz', 'EEG-16': 'P2', 'EEG-Pz': 'POz'})
    # Pre-load the dataset
    raw_gdf.load_data()
    # correct nan values
    data = raw_gdf.get_data()

    for i_chan in range(data.shape[0]):  # 遍历 22 channel
        # 将数组中的所有值设置为nan，然后将这些NaN值替换为该数组的均值。
        this_chan = data[i_chan]
        data[i_chan] = np.where(
            this_chan == np.min(this_chan), np.nan, this_chan
        )
        mask = np.isnan(data[i_chan])
        chan_mean = np.nanmean(data[i_chan])
        data[i_chan, mask] = chan_mean

    # 获取事件时间位置，返回事件和事件下标
    events, events_id = mne.events_from_annotations(raw_gdf)
    # print('Number of events:', len(events))
    # print(events_id)
    # print(events)
    # 利用mne.io.RawArray类重新创建Raw对象，已经没有nan数据了
    raw_gdf = mne.io.RawArray(data, raw_gdf.info, verbose="ERROR")
    # print(raw_gdf.info)
    # # 画出EEG通道图
    # raw_gdf.plot()
    # plt.show()
    return raw_gdf, events, events_id


def get_epochs(raw_gdf, events, events_id):
    # 选择范围为Cue后 1s - 4s 的数据
    tmin, tmax = 1.5, 5.995
    # 四类 MI 对应的 events_id
    events_id = dict({'769': 7, '770': 8})
    # events_id = dict({'769': 7, '770': 8, '771': 9, '772': 10})

    epochs = mne.Epochs(raw_gdf, events, events_id, tmin, tmax, proj=True, baseline=None, preload=True,
                        event_repeated='drop')
    # print(epochs)

    # # 画出EEG通道图
    # epochs.plot()
    # plt.show()

    # 切片，获取 events 的最后一列
    labels = epochs.events[:, -1]
    # Get all epochs as a 3D array.
    data = epochs.get_data(copy=True)
    # print(labels.shape)
    # print(data.shape)
    return epochs, events_id, data


def plot_heatmap(matrix, lable, title, num_channels, ch_names):
    """
    绘制热图的通用函数
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(matrix, interpolation='nearest', cmap='viridis', aspect='auto')
    plt.colorbar(label=lable)
    plt.title(title)
    plt.xlabel('Target Channel')
    plt.ylabel('Source Channel')
    plt.xticks(range(num_channels), ch_names[:num_channels], rotation=90)
    plt.yticks(range(num_channels), ch_names[:num_channels])
    plt.tight_layout()
    plt.show()


def wpd_plt(signal, n):
    # wpd分解
    wp = pywt.WaveletPacket(data=signal, wavelet='db1', mode='symmetric', maxlevel=n)

    # 计算每一个节点的系数，存在map中，key为'aa'等，value为列表
    map = {}
    map[1] = signal
    for row in range(1, n + 1):
        lev = []
        for i in [node.path for node in wp.get_level(row, 'freq')]:
            map[i] = wp[i].data

    # 作图
    plt.figure(figsize=(16, 12))
    plt.subplot(n + 1, 1, 1)  # 绘制第一个图
    plt.plot(map[1])
    for i in range(2, n + 2):
        level_num = pow(2, i - 1)  # 从第二行图开始，计算上一行图的2的幂次方
        # 获取每一层分解的node：比如第三层['aaa', 'aad', 'add', 'ada', 'dda', 'ddd', 'dad', 'daa']
        re = [node.path for node in wp.get_level(i - 1, 'freq')]
        for j in range(1, level_num + 1):
            plt.subplot(n + 1, level_num, level_num * (i - 1) + j)
            plt.plot(map[re[j - 1]])  # 列表从0开始


def visualize_channel_importance(transformed_features, channel_names):
    """
    可视化EEG通道的重要性。

    参数:
    transformed_features (numpy.ndarray): 经过转换的特征数组。
    channel_names (list): EEG通道的名称列表。
    """
    # Initialize MaxAbsScaler
    scaler = MaxAbsScaler()

    # Fit and transform the transformed features
    abs_scaled_features = scaler.fit_transform(transformed_features.reshape(-1, 1))

    # 线性变换到 [0, 1] 的范围
    normalized_features = (abs_scaled_features + 1) / 2

    # Reshape the normalized features back to the original shape
    normalized_features = normalized_features.reshape(transformed_features.shape)

    # Calculate the mean and standard deviation of the normalized features
    mean_value_normalized = np.mean(normalized_features)
    std_dev_normalized = np.std(normalized_features)

    # 创建渐变颜色映射
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['skyblue', 'lightgreen', 'goldenrod'])

    # 绘制图形
    fig, ax = plt.subplots(figsize=(12, 8))

    # 设置背景颜色区域
    ax.axhspan(mean_value_normalized + std_dev_normalized, 1, facecolor='peachpuff', alpha=0.5)
    ax.axhspan(mean_value_normalized - std_dev_normalized, mean_value_normalized + std_dev_normalized,
               facecolor='lightblue', alpha=0.5)
    ax.axhspan(0, mean_value_normalized - std_dev_normalized, facecolor='lightgreen', alpha=0.5)
    # 设置y轴范围
    ax.set_ylim(0, 1)
    # 绘制条形图，并设置渐变颜色
    bar_width = 0.4  # 条形宽度
    spacing = 0.1  # 条形间距
    for i in range(len(normalized_features.flatten())):
        value = normalized_features.flatten()[i]
        for y in range(int(value * 40)):
            rect = patches.Rectangle((i * (bar_width + spacing), y / 40), bar_width, 0.02, linewidth=0, edgecolor=None,
                                     facecolor=cmap(y / 40))
            ax.add_patch(rect)

    # 绘制均值和标准差线
    ax.axhline(mean_value_normalized + std_dev_normalized, color='r', linestyle='--', linewidth=2,
               label='Positive Std Dev: {:.2f}'.format(mean_value_normalized + std_dev_normalized))
    ax.axhline(mean_value_normalized - std_dev_normalized, color='r', linestyle='--', linewidth=2,
               label='Negative Std Dev: {:.2f}'.format(mean_value_normalized - std_dev_normalized))

    # 添加标签和标题
    ax.set_xlabel('EEG Channel')
    ax.set_ylabel('Channel Importance (Normalized)')
    ax.set_title('Channel Importance Influenced by EOG via PCA (Normalized)')
    ax.set_xticks([i * (bar_width + spacing) + bar_width / 2 for i in range(len(normalized_features.flatten()))])
    ax.set_xticklabels(channel_names, rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.show()


def calculate_cv(data):
    mean = np.mean(data)
    std = np.std(data)
    cv = (std / mean)
    return cv


def TPW(data, cutoff_w, mode):
    tsedata = np.mean(data, axis=0)
    num_rows = 22
    num_cols = 3
    channels = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
        'P1', 'Pz', 'P2', 'POz', 'EOG-left', 'EOG-central', 'EOG-right'
    ]
    num_channels = 25
    features_list = []

    # 根据mode值选择不同的模态组合
    if mode in [1, 4, 5, 7]:
        # 计算TSE特征
        tse_matrix = np.zeros((num_channels, num_channels))
        for i in range(num_rows):
            for j in range(num_channels - num_cols, num_channels):
                if i != j:  # 避免计算通道自身的转递熵
                    try:
                        embedding_dim = 6
                        tau = 1
                        fl1 = 0  # 频带起始索引
                        fl2 = 100  # 频带结束索引，需根据实际频率点调整
                        tse_value = tse.calculate_transfer_spectral_entropy(tsedata[i],
                                                                            tsedata[j],
                                                                            embedding_dim, tau, fl1, fl2)
                        # 存储转递熵值到矩阵中
                        tse_matrix[i, j] = tse_value
                    except Exception as e:
                        print(f"Error computing Transfer Spectral Entropy: {e}")
        # 选择需要标准化的部分
        sub_matrix = tse_matrix[:22, -3:]

        # 初始化MinMaxScaler
        scaler = MinMaxScaler()

        # 进行0-1标准化
        normalized_sub_matrix = scaler.fit_transform(sub_matrix)

        # 添加偏移量以避免0
        offset = 0.01
        normalized_sub_matrix += offset

        # 确保所有值在0到1之间
        normalized_sub_matrix = np.clip(normalized_sub_matrix, 0, 1)

        # 将标准化后的部分重新放回原矩阵中（如果需要）
        tse_matrix[:22, -3:] = normalized_sub_matrix
        features_TSE_EOG = tse_matrix[:22, -3:]
        features_list.append(features_TSE_EOG)

    if mode in [2, 4, 6, 7]:
        # 计算PSD特征
        fmin, fmax = 0.5, 100
        n_fft = 1125
        n_per_seg = 256
        psd_similarities = np.zeros((num_channels, num_channels))
        psds_dict = {}
        for ch_idx in range(num_channels):
            ch_name = channels[ch_idx]
            ch_data = data[:, ch_idx]
            psd, freqs = mne.time_frequency.psd_array_welch(ch_data, 250, fmin=fmin, fmax=fmax,
                                                            n_fft=n_fft, n_per_seg=n_per_seg, verbose=False)
            psds_dict[ch_name] = psd.mean(axis=0)
        for i in range(num_rows):
            channel_i = channels[i]
            for j in range(num_channels - num_cols, num_channels):
                channel_j = channels[j]
                corr_coef, _ = pearsonr(psds_dict[channel_i], psds_dict[channel_j])
                psd_similarities[i, j] = corr_coef


        features_PSD_EOG = psd_similarities[:22, -3:]
        # 初始化MinMaxScaler
        scaler = MinMaxScaler()

        # 进行0-1标准化
        normalized_sub_matrix = scaler.fit_transform(features_PSD_EOG)

        # 添加偏移量以避免0
        offset = 0.01
        normalized_sub_matrix += offset

        # 确保所有值在0到1之间
        normalized_sub_matrix1 = np.clip(normalized_sub_matrix, 0, 1)
        # print(features_PSD_EOG)
        features_list.append(normalized_sub_matrix1)

    if mode in [3, 5, 6, 7]:
        # 计算WPD特征
        max_level = 3
        wpd_similarities = np.zeros((num_channels, num_channels))
        for i in range(num_rows):
            for j in range(num_channels - num_cols, num_channels):
                wpd_i = wpd(data[:, i], max_level)
                wpd_j = wpd(data[:, j], max_level)
                nodes_i = [node.path for node in wpd_i.get_level(max_level, 'freq')]
                nodes_j = [node.path for node in wpd_j.get_level(max_level, 'freq')]
                sim_values = []
                for path in nodes_i:
                    if path in nodes_j:
                        num_features = wpd_i[path].data.shape[1]
                        correlations = np.zeros(num_features)
                        for feature_idx in range(num_features):
                            feature_data_i = wpd_i[path].data[:, feature_idx]
                            feature_data_j = wpd_j[path].data[:, feature_idx]
                            correlations[feature_idx], _ = pearsonr(feature_data_i, feature_data_j)
                        sim = np.mean(correlations)
                        sim_values.append(sim)
                wpd_similarities[i, j] = np.mean(sim_values)
        features_WPD_EOG = wpd_similarities[:22, -3:]
        # 初始化MinMaxScaler
        scaler = MinMaxScaler()

        # 进行0-1标准化
        normalized_sub_matrix = scaler.fit_transform(features_WPD_EOG)

        # 添加偏移量以避免0
        offset = 0.01
        normalized_sub_matrix += offset

        # 确保所有值在0到1之间
        normalized_sub_matrix2 = np.clip(normalized_sub_matrix, 0, 1)
        print(normalized_sub_matrix2)
        features_list.append(normalized_sub_matrix2)

    # 将所有选定的特征拼接成一个特征矩阵
    features = np.hstack(features_list)

    # 应用PCA进行降维
    pca = PCA(n_components=1)
    transformed_features = pca.fit_transform(features)

    # 进行 Min-Max 标准化
    scaler = MinMaxScaler()
    min_max_scaled = scaler.fit_transform(transformed_features)
    # print(min_max_scaled)
    if calculate_cv(min_max_scaled) > 0.5:
        print(calculate_cv(min_max_scaled))
        cutoff_w += 1

    return transformed_features, cutoff_w


def pca_TPW(data):
    # 将整体数据进行PCA处理
    num_splits = 5
    data_splits = np.array_split(data, num_splits)
    all_results = Parallel(n_jobs=-1)(delayed(TPW)(data_part, 0, 7) for data_part in data_splits)
    combined_features, cutoff_ws = zip(*all_results)
    plot_cutoff_w_over_iterations(combined_features)
    # 取平均特征
    combined_features = np.mean(combined_features, axis=0)

    # 累加 cutoff_w 参数
    cutoff_w = sum(cutoff_ws)
    print("Cutoff group width: ", cutoff_w)

    return combined_features, cutoff_w


def plot_cutoff_w_over_iterations(combined_features):
    # # 计算每次迭代的变异系数并存储
    cv_values = []
    for i in range(5):
        # 初始化 MinMaxScaler
        scaler = MinMaxScaler()
        min_max_scaled = scaler.fit_transform(combined_features[i])
        cv_value = calculate_cv(min_max_scaled)
        cv_values.append(cv_value)

    print(cv_values)
    # 初始化 MinMaxScaler
    scaler = MinMaxScaler()

    # 标准化 combined_features 到 [0, 1] 范围
    normalized_features = []
    for features in combined_features:
        scaled_features = scaler.fit_transform(features.reshape(-1, 1))
        # scaled_features = np.log1p(scaled_features)
        normalized_features.append(scaled_features)
        # normalized_features = np.log1p(normalized_features)

    # 转换为 numpy 数组
    normalized_features = np.array(normalized_features)
    # print("Normalized features: ", normalized_features)
    # 绘制图形
    fig, ax1 = plt.subplots()

    # 绘制蓝色折线表示 CV Value 变化
    ax1.plot(range(1, len(cv_values) + 1), cv_values, color='#fa7f6f', label='CV Value')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Coefficient of Variation', color='#fa7f6f')
    ax1.tick_params(axis='y', labelcolor='#fa7f6f')
    ax1.set_title('Coefficient of Variation and Channel Significance Over Iterations')

    # 绘制绿色虚线表示阈值为 1 的固定值
    ax1.axhline(y=0.5, color='g', linestyle='--', label='Threshold = 0.5')

    ax2 = ax1.twinx()
    for i in range(len(normalized_features)):
        for j in range(len(normalized_features[i])):
            ax2.scatter([i + 1], normalized_features[i][j], color='#0087ca', s=15,
                        label='Normalized Channel Significance' if i == 0 and j == 0 else "")

    ax2.set_ylabel('Channel Significance', color='#0087ca')
    ax2.tick_params(axis='y', labelcolor='#0087ca')

    # 设置散点图纵轴范围为 [0, 1]
    ax2.set_ylim([0, 1.2])

    # 设置横轴刻度为五段
    ax1.set_xticks(range(1, len(cv_values) + 1))

    # 添加图例
    fig.tight_layout()
    # fig.legend()

    plt.show()


def visualize_topomap(transformed_features, channel_names, sfreq=250):
    """
    可视化EEG通道的重要性地形图。

    参数:
    transformed_features (numpy.ndarray): 经过转换的特征数组。
    channel_names (list): EEG通道的名称列表。
    sfreq (int): 采样率，默认为250Hz。
    """
    # 创建MNE信息对象
    info = mne.create_info(
        ch_names=channel_names,
        ch_types=['eeg'] * len(channel_names),
        sfreq=sfreq
    )

    # 使用标准电极位置信息
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    # 创建MNE数据对象
    raw = mne.io.RawArray(transformed_features, info)

    # 绘制地形图
    mne.viz.plot_topomap(transformed_features[:, 0], raw.info, names=channel_names, sensors=True, cmap='Reds',
                         contours=4, size=3)
    plt.show()


def tpasr(transformed_features, raw_gdf, width):
    # 假设transformed_features是一个包含每个通道伪影影响程度的numpy数组
    # Sort the transformed_features array
    sorted_features = np.sort(transformed_features)
    # print(sorted_features)
    mean_value = np.mean(sorted_features)
    std_value = np.std(sorted_features)

    low_threshold = round(float(mean_value - std_value), 3)
    high_threshold = round(float(mean_value + std_value), 3)

    # 通道索引转换为通道名称
    channel_names = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
        'P1', 'Pz', 'P2', 'POz'
    ]
    montage = mne.channels.make_standard_montage('standard_1020')
    raw_gdf.info.set_montage(montage, on_missing='ignore')
    raw_gdf.set_channel_types({'EOG-left': 'eog', 'EOG-central': 'eog', 'EOG-right': 'eog'})
    # raw_gdf.compute_psd(fmin=0.5, fmax=100).plot(spatial_colors=True)
    # 绘制地形图
    # channel_names = [
    #     'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
    #     'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
    # ]
    #
    # visualize_topomap(transformed_features, channel_names)

    low_impact_channels = [channel_names[i] for i, feature in enumerate(transformed_features.flatten()) if
                           feature < low_threshold]
    medium_impact_channels = [channel_names[i] for i, feature in enumerate(transformed_features.flatten()) if
                              low_threshold <= feature <= high_threshold]
    high_impact_channels = [channel_names[i] for i, feature in enumerate(transformed_features.flatten()) if
                            feature > high_threshold]
    print(low_impact_channels)
    print(medium_impact_channels)
    print(high_impact_channels)

    # 定义每组的ASR参数
    asr_params_low = {'cutoff': 20 + width + width, 'max_bad_chans': 0.3}
    asr_params_medium = {'cutoff': 20 + width, 'max_bad_chans': 0.2}
    asr_params_high = {'cutoff': 20, 'max_bad_chans': 0.2}

    # 创建一个新的Raw对象以避免在原始数据上直接修改

    eeg_processed = raw_gdf.copy()
    # 应用ASR处理
    for group, params in zip([low_impact_channels, medium_impact_channels, high_impact_channels],
                             [asr_params_low, asr_params_medium, asr_params_high]):
        if group != []:
            # 初始化ASR实例
            asr = ASR(sfreq=raw_gdf.info['sfreq'], cutoff=params['cutoff'], max_bad_chans=params['max_bad_chans'], )
            # 训练ASR
            raw_processed = raw_gdf.copy()
            asr.fit(raw_processed)
            # 应用ASR变换
            raw_processed = asr.transform(raw_processed)
            # 替换eeg_processed中的数据
            for channel_name in group:
                channel_index = channel_names.index(channel_name)
                eeg_processed._data[channel_index] = raw_processed._data[channel_index]

    return eeg_processed


def init_asr(raw_gdf):
    # 使用ASR处理数据，这里不分组，cutoff设置为20
    asr = ASR(sfreq=raw_gdf.info['sfreq'], cutoff=20, max_bad_chans=0.2)
    # 训练ASR
    asr.fit(raw_gdf)
    # 应用ASR变换
    raw_processed = asr.transform(raw_gdf)

    return raw_processed


def plot_asr(epochs, cleaned_avg, channel_names):
    # 加载或创建通道位置信息
    montage = mne.channels.make_standard_montage('standard_1020')
    # 现在你可以重新尝试绘制，这次应该可以启用空间颜色了
    evoked1 = epochs.average(picks=channel_names)
    evoked1.set_montage(montage)
    # evoked1.plot(spatial_colors=True, titles="Before ASR")
    cleaned_avg.set_montage(montage)
    evoked2 = cleaned_avg.average()
    # evoked2.plot(spatial_colors=True, titles="After ASR")
    difference_evoked = mne.combine_evoked([evoked1, evoked2], weights=[1, -1])
    # 使用plot方法绘制差异波形
    difference_evoked.plot(spatial_colors=True, gfp=True, titles="Waveform Differences (Raw signal - After ASR)")


def asr_test(filename, training):
    if training:
        raw_gdf, events, events_id = read_raw_gdf(filename)
        channel_names = [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
            'P1', 'Pz', 'P2', 'POz'
        ]
        epochs, events_id, data = get_epochs(raw_gdf, events, events_id)
        start_time = time.time()
        transformed_features, width = pca_TPW(data)
        end_time = time.time()  # 记录结束时间
        total_time = end_time - start_time  # 计算总时间

        print(f"相关性计算用时：{total_time:.2f}秒")
        # 运行性能分析
        # profiler = cProfile.Profile()
        # profiler.enable()
        start_time = time.time()
        raw_processed = tpasr(transformed_features, raw_gdf, width)
        end_time = time.time()  # 记录结束时间
        total_time = end_time - start_time  # 计算总时间
        print(f"MASR计算用时：{total_time:.2f}秒")
        # profiler.disable()
        # # 保存性能分析结果
        # profiler.dump_stats("asr_profile.prof")
        #
        # # 分析并打印性能分析结果
        # stats = pstats.Stats("asr_profile.prof")
        # stats.sort_stats("cumulative").print_stats(10)
        start_time = time.time()
        raw_processed1 = init_asr(raw_gdf)
        end_time = time.time()  # 记录结束时间
        total_time = end_time - start_time  # 计算总时间

        print(f"ASR计算用时：{total_time:.2f}秒")
        raw_gdf.set_eeg_reference('average', projection=True)

        # 拟合 ICA 模型
        ica1 = ICA(n_components=22, method='picard')
        ica1.fit(raw_gdf)

        # 识别和标记 EOG 伪影
        eog_inds, scores = ica1.find_bads_eog(raw_gdf)  # find which ICs match the EOG pattern
        ica1.exclude = eog_inds  # 标记要排除的独立成分

        # 应用 ICA 来去除伪影成分
        raw_processed2 = ica1.apply(raw_gdf.copy())
        # # 拟合 ICA 模型
        # ica2 = ICA(n_components=22, method='fastica')
        # ica2.fit(raw_gdf)
        #
        # # 识别和标记 EOG 伪影
        # eog_inds, scores = ica2.find_bads_eog(raw_gdf)  # find which ICs match the EOG pattern
        # ica2.exclude = eog_inds  # 标记要排除的独立成分
        #
        # # 应用 ICA 来去除伪影成分
        # raw_processed3 = ica2.apply(raw_gdf.copy())

        # 创建EOG伪影事件并计算平均EOG伪影响应
        eog_epochs = mne.preprocessing.create_eog_epochs(raw_gdf)
        eog_evoked = eog_epochs.average()
        eog_evoked.plot_joint()

        raw_ssp = raw_gdf.copy()
        # 计算EOG投影向量
        eog_projs, _ = mne.preprocessing.compute_proj_eog(raw_ssp, n_grad=1, n_mag=1, n_eeg=1)

        # 检查生成的投影对象
        print(f"Type of eog_projs: {type(eog_projs)}")
        print(f"First element type: {type(eog_projs[0])}")

        # 将EOG投影向量添加到原始数据
        raw_ssp.add_proj(eog_projs)

        # 应用投影向量
        raw_processed3 = raw_ssp.copy().apply_proj()

        cleaned_avg = mne.Epochs(raw_processed, events, events_id, tmin=1.5, tmax=5.995, baseline=None, preload=True,
                                 picks=channel_names, event_repeated="drop")
        cleaned_avg1 = mne.Epochs(raw_processed1, events, events_id, tmin=1.5, tmax=5.995, baseline=None, preload=True,
                                  picks=channel_names, event_repeated="drop")
        cleaned_avg2 = mne.Epochs(raw_processed2, events, events_id, tmin=1.5, tmax=5.995, baseline=None, preload=True,
                                  picks=channel_names, event_repeated="drop")
        cleaned_avg3 = mne.Epochs(raw_processed3, events, events_id, tmin=1.5, tmax=5.995, baseline=None, preload=True,
                                  picks=channel_names, event_repeated="drop")
        plot_asr(epochs, cleaned_avg, channel_names)
        plot_asr(epochs, cleaned_avg1, channel_names)
        plot_asr(epochs, cleaned_avg2, channel_names)
        plot_asr(epochs, cleaned_avg3, channel_names)
        cleand_data = cleaned_avg.get_data(copy=True)
        # print(f"Epochs(After): {cleand_data.shape}")
        labels = cleaned_avg.events[:, -1] - 7
        # print(f"labels(After): {labels}")
        raw_data_selected_channels = epochs.get_data(copy=True)[:, :22, :]
        # plt_snr(cleand_data, raw_data_selected_channels)
        normal_asr = cleaned_avg1.get_data(copy=True)
        picard_eeg = cleaned_avg2.get_data(copy=True)
        fastica_eeg = cleaned_avg3.get_data(copy=True)
        analyze_merits.compare_metrics1(cleand_data, raw_data_selected_channels, normal_asr, picard_eeg, fastica_eeg)

    else:
        raw_gdf, events, events_id = read_raw_gdf(filename)
        events_id = dict({'1023': 1, '1072': 2, '276': 3, '277': 4, '32766': 5, '768': 6, '783': 7})
        epochs, events_id, data = get_epochs(raw_gdf, events, events_id)
        transformed_features, width = pca_TPW(data)
        raw_processed = tpasr(transformed_features, raw_gdf, width)
        # 加载或创建通道位置信息
        montage = mne.channels.make_standard_montage('standard_1020')
        channel_names = [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
            'P1', 'Pz', 'P2', 'POz'
        ]

        cleaned_avg = mne.Epochs(raw_processed, events, events_id, tmin=1.5, tmax=5.995, baseline=None, preload=True,
                                 picks=channel_names,
                                 event_repeated="merge")
        cleaned_avg.set_montage(montage)
        cleand_data = cleaned_avg.get_data(copy=True)
        labels = cleaned_avg.events[:, -1]
    return cleand_data, labels


asr_test("dataset/s5/A05T.gdf", training=True)
