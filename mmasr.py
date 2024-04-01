import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
from asrpy import ASR
from pyinform import transfer_entropy
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import analyze_merits

matplotlib.use('TkAgg')


def bin_data(data, bins):
    hist, bin_edges = np.histogram(data, bins=bins)
    binned_data = np.digitize(data, bin_edges[:-1]) - 1  # subtract 1 to start binning from 0
    return binned_data


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
    events_id = dict({'769': 7, '770': 8, '771': 9, '772': 10})

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


def pca_TE_PSD(data, raw_gdf):
    desired_length = 255602
    num_bins = 100
    embedding_data_list_2D_trimmed = [data[:desired_length, 0] for data in data]
    # print(f" embedding_data_list_2D_trimmed: {embedding_data_list_2D_trimmed}")
    embedding_data_list_2D_binned = [bin_data(data, num_bins) for data in embedding_data_list_2D_trimmed]
    # print(f" embedding_data_list_2D_binned: {embedding_data_list_2D_binned}")

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
    TE_matrix = np.zeros((num_channels, num_channels))

    for i, source_channel in enumerate(channels[:num_channels]):
        for j, target_channel in enumerate(channels[:num_channels]):
            if i != j:  # 避免计算通道自身的转递熵
                try:
                    # 假设 transfer_entropy 函数返回的是从 source_channel 到 target_channel 的转递熵值
                    te_value = transfer_entropy(embedding_data_list_2D_binned[i], embedding_data_list_2D_binned[j], k=1)
                    # 存储转递熵值到矩阵中
                    TE_matrix[i, j] = te_value
                except Exception as e:
                    print(f"Error computing Transfer Entropy from {source_channel} to {target_channel}: {e}")

    # 使用热图表示转递熵矩阵
    # plt.figure(figsize=(12, 10))
    # plt.imshow(TE_matrix, cmap='viridis', aspect='auto')
    # plt.colorbar(label='Transfer Entropy')
    # plt.title('Transfer Entropy Matrix')
    # plt.xlabel('Target Channel')
    # plt.ylabel('Source Channel')
    # plt.xticks(range(num_channels), labels=channels[:num_channels], rotation=90)
    # plt.yticks(range(num_channels), labels=channels[:num_channels])
    # plt.tight_layout()
    # plt.show()
    # 假设 epochs 是已经预加载的mne.Epochs对象
    # 设置感兴趣的频率范围
    fmin, fmax = 0.5, 50  # Hz
    tmin, tmax = 1.5, 6.  # 选择Cue后 1s - 4s 的数据
    n_fft = 1024  # FFT的窗口大小，减小以匹配数据长度
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
    for i in range(num_channels):
        channel_i = channels[i]
        for j in range(i + 1, num_channels):  # 使用i+1开始，避免重复计算和自我比较
            channel_j = channels[j]
            # 计算两个通道之间PSD的相似度
            corr_coef, _ = pearsonr(psds_dict[channel_i], psds_dict[channel_j])
            psd_similarities[i, j] = corr_coef
            psd_similarities[j, i] = corr_coef  # 使矩阵对称

    # # 可视化PSD相似度矩阵
    # plt.figure(figsize=(10, 10))
    # plt.imshow(psd_similarities, interpolation='nearest', cmap='viridis')
    # plt.colorbar()
    # plt.title('PSD')
    # plt.xlabel('ch_names')
    # plt.ylabel('ch_names')
    # plt.xticks(range(num_channels), raw_gdf.ch_names[:num_channels], rotation=90)
    # plt.yticks(range(num_channels), raw_gdf.ch_names[:num_channels])
    # plt.show()
    # 提取对22个EEG通道有影响的3个EOG通道的相关特征
    features_TE_EOG = TE_matrix[:22, -3:]  # 最后3列代表TE值
    features_PSD_EOG = psd_similarities[:22, -3:]  # 最后3列代表PSD相似度

    # 将特征扁平化，为每个EEG通道形成一个特征向量
    features = np.hstack((features_TE_EOG, features_PSD_EOG))

    # 应用PCA进行降维，以便于可视化
    pca = PCA(n_components=1)
    transformed_features = pca.fit_transform(features)

    # 可视化
    channel_names = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
        'P1', 'Pz', 'P2', 'POz'
    ]

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

    # Calculate low_threshold and high_threshold based on the values at one-third and two-thirds positions
    one_third_index = len(sorted_features) // 3
    two_thirds_index = 2 * len(sorted_features) // 3

    # 假设 sorted_features 是一个排序后的一维数组
    low_threshold = round(float(sorted_features[one_third_index]), 3) if np.isscalar(
        sorted_features[one_third_index]) else round(float(sorted_features[one_third_index][0]), 3)
    high_threshold = round(float(sorted_features[two_thirds_index]), 3) if np.isscalar(
        sorted_features[two_thirds_index]) else round(float(sorted_features[two_thirds_index][0]), 3)

    # 通道索引转换为通道名称
    channel_names = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
        'P1', 'Pz', 'P2', 'POz'
    ]

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
    asr_params_low = {'cutoff': 20, 'max_bad_chans': 0.3}
    asr_params_medium = {'cutoff': 25, 'max_bad_chans': 0.2}
    asr_params_high = {'cutoff': 30, 'max_bad_chans': 0.1}

    # 创建一个新的Raw对象以避免在原始数据上直接修改
    raw_processed = raw_gdf.copy()

    # 应用ASR处理
    for group, params in zip([low_impact_channels, medium_impact_channels, high_impact_channels],
                             [asr_params_low, asr_params_medium, asr_params_high]):
        if group != []:
            # 初始化ASR实例
            asr = ASR(sfreq=raw_gdf.info['sfreq'], cutoff=params['cutoff'], max_bad_chans=params['max_bad_chans'], )
            # 训练ASR
            asr.fit(raw_processed, picks=group)

            # 应用ASR变换
            raw_processed = asr.transform(raw_processed, picks=group)

    return raw_processed


def init_asr(raw_gdf):
    # 使用ASR处理数据，这里不分组，cutoff设置为25
    asr = ASR(sfreq=raw_gdf.info['sfreq'], cutoff=25, max_bad_chans=0.2)
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
    difference_evoked.plot(spatial_colors=True, gfp=True, titles="Difference (Before - After ASR)")


def asr_test(filename, training):
    if training:
        raw_gdf, events, events_id = read_raw_gdf(filename)
        events_id = dict({'769': 7, '770': 8, '771': 9, '772': 10})
        channel_names = [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
            'P1', 'Pz', 'P2', 'POz'
        ]
        epochs, events_id, data = get_epochs(raw_gdf, events, events_id)
        transformed_features = pca_TE_PSD(data, raw_gdf)
        raw_processed = tpasr(transformed_features, raw_gdf)
        raw_processed1 = init_asr(raw_gdf)
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
        raw_data_eog_channels = epochs.get_data(copy=True)[:, -3:, :]
        # plt_snr(cleand_data, raw_data_selected_channels)
        normal_asr = cleaned_avg1.get_data(copy=True)
        analyze_merits.compare_nmse(cleand_data, raw_data_selected_channels, normal_asr)
        analyze_merits.compare_rmse(cleand_data, raw_data_selected_channels, normal_asr)
        analyze_merits.compare_snr(cleand_data, raw_data_selected_channels, raw_data_eog_channels, normal_asr)

    else:
        raw_gdf, events, events_id = read_raw_gdf(filename)
        events_id = dict({'1023': 1, '1072': 2, '276': 3, '277': 4, '32766': 5, '768': 6, '783': 7})
        epochs, events_id, data = get_epochs(raw_gdf, events, events_id)
        transformed_features = pca_TE_PSD(data, raw_gdf)
        raw_processed = tpasr(transformed_features, raw_gdf)
        # 加载或创建通道位置信息
        montage = mne.channels.make_standard_montage('standard_1020')
        channel_names = [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
            'P1', 'Pz', 'P2', 'POz'
        ]

        cleaned_avg = mne.Epochs(raw_processed, events, events_id, tmin=1.5, tmax=6., baseline=None, preload=True,
                                 picks=channel_names,
                                 event_repeated="merge")
        cleaned_avg.set_montage(montage)
        cleand_data = cleaned_avg.get_data(copy=True)
        labels = cleaned_avg.events[:, -1]
    return cleand_data, labels


asr_test("./dataset/s3/A03T.gdf", training=True)
