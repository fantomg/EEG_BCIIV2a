import matplotlib
import pywt
import matplotlib.pyplot as plt
import mne
import numpy as np
from asrpy import ASR
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
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


def pca_TPW(data):
    tsedata = np.mean(data, axis=0)
    num_rows = 22
    num_cols = 3
    # channels = raw_gdf.ch_names
    channels = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
        'P1', 'Pz', 'P2', 'POz', 'EOG-left', 'EOG-central', 'EOG-right'
    ]
    # Dictionary to store transfer entropy results

    # Compute Transfer Entropy for all channel pairs
    num_channels = 25
    tse_matrix = np.zeros((num_channels, num_channels))

    for i, source_channel in enumerate(channels[:num_rows]):
        for j, target_channel in enumerate(channels[-num_cols:]):
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
                    print(f"Error computing Transfer Spectral Entropy from {source_channel} to {target_channel}: {e}")
    # plot_heatmap(tse_matrix, 'Transfer Spectral Entropy', 'Transfer Spectral Entropy Matrix', num_channels, channels)
    # 使用热图表示转递谱熵矩阵
    # 假设 epochs 是已经预加载的mne.Epochs对象
    # 设置感兴趣的频率范围
    fmin, fmax = 0.5, 100  # Hz
    n_fft = 1125  # FFT的窗口大小，减小以匹配数据长度
    n_per_seg = 512  # 每个段的长度，确保小于epoch的时间点数

    # 我们需要手动提取每个通道的数据
    num_channels = 25  # 假设我们有25个通道

    # 初始化一个矩阵来存储PSD相似度结果
    psd_similarities = np.zeros((num_channels, num_channels))

    # 计算每个通道的平均PSD
    psds_dict = {}
    for ch_idx in range(num_channels):
        ch_name = channels[ch_idx]
        # 提取单个通道的数据
        ch_data = data[:, ch_idx]
        # 计算PSD
        psd, freqs = mne.time_frequency.psd_array_welch(ch_data, 250, fmin=fmin, fmax=fmax,
                                                        n_fft=n_fft,
                                                        n_per_seg=n_per_seg, verbose=False)
        # 计算所有epochs的平均PSD并存储
        psds_dict[ch_name] = psd.mean(axis=0)

    # 计算所有通道对之间的PSD相似度
    for i in range(num_rows):
        channel_i = channels[i]
        for j in range(num_channels - num_cols, num_channels):  # 使用i+1开始，避免重复计算和自我比较
            channel_j = channels[j]
            # 计算两个通道之间PSD的相似度
            corr_coef, _ = pearsonr(psds_dict[channel_i], psds_dict[channel_j])
            psd_similarities[i, j] = corr_coef
            psd_similarities[j, i] = corr_coef  # 使矩阵对称
    # plot_heatmap(psd_similarities, 'PSD Similarities', 'PSD Matrix', num_channels, channels)
    # # 可视化PSD相似度矩阵
    max_level = 3  # 设置小波包分解的最大层数
    # wpd_plt(tsedata[:, 1], max_level)
    wpd_similarities = np.zeros((num_channels, num_channels))
    for i in range(num_rows):
        for j in range(num_channels - num_cols, num_channels):
            wpd_i = wpd(data[:, i], max_level)

            wpd_j = wpd(data[:, j], max_level)
            # 假设我们只对特定的节点感兴趣，例如：第三层的所有节点
            nodes_i = [node.path for node in wpd_i.get_level(max_level, 'freq')]
            nodes_j = [node.path for node in wpd_j.get_level(max_level, 'freq')]
            # 计算这些节点的特征相似度
            sim_values = []
            for path in nodes_i:
                if path in nodes_j:  # 确保两个通道都有该节点
                    # Assuming that wpd_i[path].data and wpd_j[path].data are 2D with shape (observations, features)
                    num_features = wpd_i[path].data.shape[1]
                    correlations = np.zeros(num_features)
                    for feature_idx in range(num_features):
                        feature_data_i = wpd_i[path].data[:, feature_idx]
                        feature_data_j = wpd_j[path].data[:, feature_idx]
                        correlations[feature_idx], _ = pearsonr(feature_data_i, feature_data_j)

                    # Now you have the correlation for each feature between the two channels
                    sim = np.mean(correlations)  # If you want the average correlation across all features

                    sim_values.append(sim)
            # 用平均相似度作为通道i和j之间的WPD相似度
            wpd_similarities[i, j] = np.mean(sim_values)
            wpd_similarities[j, i] = wpd_similarities[i, j]  # 保持矩阵对称性

    # 可视化WPD相似度矩阵
    # plot_heatmap(wpd_similarities, 'WPD Similarities', 'WPD Matrix', num_channels, channels)
    # 提取对22个EEG通道有影响的3个EOG通道的相关特征
    features_TSE_EOG = tse_matrix[:22, -3:]
    features_PSD_EOG = psd_similarities[:22, -3:]
    features_WPD_EOG = wpd_similarities[:22, -3:]

    # 将特征扁平化，为每个EEG通道形成一个特征向量
    features = np.hstack((features_TSE_EOG, features_PSD_EOG, features_WPD_EOG))

    # 应用PCA进行降维
    pca = PCA(n_components=1)
    transformed_features = pca.fit_transform(features)
    # print(transformed_features)
    # # 计算需要平移的量，使所有值为正
    # shift_amount = np.abs(transformed_features.min()) + 0.1  # 增加0.1来确保所有值都为正
    #
    # # 对所有值进行平移
    # A_shifted = transformed_features + shift_amount
    # channels = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    #             'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    #             'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    #             'P1', 'Pz', 'P2', 'POz']
    # # 根据脑电通道分布确定柱体的位置
    # positions = [
    #     (2.5, 5),
    #     (3.5, 4), (2.5, 4), (1.5, 4), (0.5, 4), (-0.5, 4),
    #     (4.5, 3), (3.5, 3), (2.5, 3), (1.5, 3), (0.5, 3), (-0.5, 3), (-1.5, 3),
    #     (3.5, 2), (2.5, 2), (1.5, 2), (0.5, 2), (-0.5, 2),
    #     (2.5, 1), (1.5, 1), (0.5, 1),
    #     (2.5, 0)
    # ]
    #
    # xpos = np.array([p[0] for p in positions])
    # ypos = np.array([p[1] for p in positions])
    # zpos = np.zeros(len(xpos))
    #
    # dx = dy = 0.5 * np.ones(len(xpos))
    # dz = A_shifted
    #
    # # 创建画布和坐标轴
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # 设置颜色映射
    # colors = cm.plasma_r(np.linspace(0, 1, len(A_shifted)))  # 这里的 len(A) 应该是通道的数量，即22
    #
    # # 绘制三维柱状图
    # for i in range(len(A_shifted)):
    #     ax.bar3d(xpos[i], ypos[i], zpos[i], dx[i], dy[i], dz[i], color=colors[i])
    # # 在柱体上方显示通道名称
    #
    # # 移除背景
    # ax.patch.set_alpha(0)
    #
    # # 移除坐标轴
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    # # 创建一个假的二维图以显示colorbar
    # x = np.linspace(0, 1, len(A_shifted))
    # y = np.ones(len(A_shifted))
    # c = np.linspace(0, 1, len(A_shifted))
    #
    # plt.figure()
    # sc = plt.scatter(x, y, c=c, cmap='plasma')
    # plt.colorbar(sc)
    #
    # plt.show()
    # # 可视化
    # channel_names = [
    #     'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    #     'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    #     'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    #     'P1', 'Pz', 'P2', 'POz'
    # ]
    #
    # plt.figure(figsize=(12, 8))
    # plt.bar(range(1, 23), transformed_features.flatten(), color='skyblue')
    # plt.xlabel('EEG Channel')
    # plt.ylabel('PCA Transformed Feature Value')
    # plt.title('Influence of EOG on EEG Channels via PCA')
    # plt.xticks(range(1, 23), labels=channel_names, rotation=45)
    # plt.tight_layout()
    # plt.show()
    return transformed_features


def tpasr(transformed_features, raw_gdf):
    # 假设transformed_features是一个包含每个通道伪影影响程度的numpy数组
    # Sort the transformed_features array
    sorted_features = np.sort(transformed_features)

    # # Calculate low_threshold and high_threshold based on the values at one-third and two-thirds positions
    # one_third_index = len(sorted_features) // 5
    # two_thirds_index = 4 * len(sorted_features) // 5
    #
    # # 假设 sorted_features 是一个排序后的一维数组
    # low_threshold = round(float(sorted_features[one_third_index]), 3) if np.isscalar(
    #     sorted_features[one_third_index]) else round(float(sorted_features[one_third_index][0]), 3)
    # high_threshold = round(float(sorted_features[two_thirds_index]), 3) if np.isscalar(
    #     sorted_features[two_thirds_index]) else round(float(sorted_features[two_thirds_index][0]), 3)
    # 使用numpy的percentile函数计算阈值
    # 假设 sorted_features 是一个排序后的一维数组
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
    # # 创建MNE信息对象
    # info = mne.create_info(
    #     ch_names=['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
    #               'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'],
    #     ch_types=['eeg'] * 22,
    #     sfreq=250  # 假设采样率为250Hz
    # )
    # # 使用标准电极位置信息
    # montage = mne.channels.make_standard_montage('standard_1020')
    # info.set_montage(montage)
    # # 创建MNE数据对象
    # raw = mne.io.RawArray(transformed_features, info)
    # # 绘制地形图
    # mne.viz.plot_topomap(transformed_features[:, 0], raw.info, names=channel_names, sensors=True, cmap='Reds',
    #                      contours=4, size=3)
    # plt.show()

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
    asr_params_low = {'cutoff': 30, 'max_bad_chans': 0.3}
    asr_params_medium = {'cutoff': 25, 'max_bad_chans': 0.2}
    asr_params_high = {'cutoff': 20, 'max_bad_chans': 0.1}

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
    asr = ASR(sfreq=raw_gdf.info['sfreq'], cutoff=20, max_bad_chans=0.1)
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
    difference_evoked.plot(spatial_colors=True, gfp=True, titles="Waveform Differences (Raw signal - After MASR)")


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
        transformed_features = pca_TPW(data)
        end_time = time.time()  # 记录结束时间
        total_time = end_time - start_time  # 计算总时间

        print(f"相关性计算用时：{total_time:.2f}秒")
        start_time = time.time()
        raw_processed = tpasr(transformed_features, raw_gdf)
        end_time = time.time()  # 记录结束时间
        total_time = end_time - start_time  # 计算总时间

        print(f"MASR计算用时：{total_time:.2f}秒")
        # start_time = time.time()
        raw_processed1 = init_asr(raw_gdf)
        # end_time = time.time()  # 记录结束时间
        # total_time = end_time - start_time  # 计算总时间
        #
        # print(f"ASR计算用时：{total_time:.2f}秒")
        cleaned_avg = mne.Epochs(raw_processed, events, events_id, tmin=1.5, tmax=5.995, baseline=None, preload=True,
                                 picks=channel_names, event_repeated="drop")
        cleaned_avg1 = mne.Epochs(raw_processed1, events, events_id, tmin=1.5, tmax=5.995, baseline=None, preload=True,
                                  picks=channel_names, event_repeated="drop")
        plot_asr(epochs, cleaned_avg, channel_names)
        plot_asr(epochs, cleaned_avg1, channel_names)
        cleand_data = cleaned_avg.get_data(copy=True)
        # print(f"Epochs(After): {cleand_data.shape}")
        labels = cleaned_avg.events[:, -1] - 7
        # print(f"labels(After): {labels}")
        raw_data_selected_channels = epochs.get_data(copy=True)[:, :22, :]
        # plt_snr(cleand_data, raw_data_selected_channels)
        normal_asr = cleaned_avg1.get_data(copy=True)
        analyze_merits.compare_metrics(cleand_data, raw_data_selected_channels, normal_asr)

    else:
        raw_gdf, events, events_id = read_raw_gdf(filename)
        events_id = dict({'1023': 1, '1072': 2, '276': 3, '277': 4, '32766': 5, '768': 6, '783': 7})
        epochs, events_id, data = get_epochs(raw_gdf, events, events_id)
        transformed_features = pca_TPW(data)
        raw_processed = tpasr(transformed_features, raw_gdf)
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


asr_test("dataset/s2/A02T.gdf", training=True)
