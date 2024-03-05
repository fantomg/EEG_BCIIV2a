import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
from pyinform import transfer_entropy
from BTSE import calculate_BTSE
from mne.preprocessing import ICA

desired_length = 255602
num_bins = 100
k = 1


def bin_data(data, bins):
    hist, bin_edges = np.histogram(data, bins=bins)
    binned_data = np.digitize(data, bin_edges[:-1]) - 1  # subtract 1 to start binning from 0
    return binned_data


matplotlib.use('TkAgg')

filename = "./dataset/A08T.gdf"

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
print('Number of events:', len(events))
print(events_id)
print(events)

print()
# 利用mne.io.RawArray类重新创建Raw对象，已经没有nan数据了
raw_gdf = mne.io.RawArray(data, raw_gdf.info, verbose="ERROR")
print(raw_gdf.info)

# # 画出EEG通道图
# raw_gdf.plot()
# plt.show()

# 选择范围为Cue后 1s - 4s 的数据
tmin, tmax = 1.5, 6.
# 四类 MI 对应的 events_id
event_id = dict({'769': 7, '770': 8, '771': 9, '772': 10})

epochs = mne.Epochs(raw_gdf, events, event_id, tmin, tmax, proj=True, baseline=None, preload=True)
print(epochs)

# # 画出EEG通道图
# epochs.plot()
# plt.show()

# 切片，获取 events 的最后一列
labels = epochs.events[:, -1]
# Get all epochs as a 3D array.
data = epochs.get_data(copy=True)
print(labels.shape)
print(data.shape)

embedding_data_list_2D_trimmed = [data[:desired_length, 0] for data in data]
# print(f" embedding_data_list_2D_trimmed: {embedding_data_list_2D_trimmed}")
embedding_data_list_2D_binned = [bin_data(data, num_bins) for data in embedding_data_list_2D_trimmed]
# print(f" embedding_data_list_2D_binned: {embedding_data_list_2D_binned}")

channels = raw_gdf.ch_names
# Dictionary to store transfer entropy results
TE_results = {}

# Compute Transfer Entropy for all channel pairs
for i, source_channel in enumerate(channels):
    for j, target_channel in enumerate(channels):
        if target_channel not in ["EOG-left", "EOG-central", "EOG-right"] and source_channel in ["EOG-left",
                                                                                                 "EOG-central",
                                                                                                 "EOG-right"]:  # Avoid computing Transfer Entropy for the same channel
            try:
                BTSE = transfer_entropy(embedding_data_list_2D_binned[i], embedding_data_list_2D_binned[j], 1)
                key = f"{source_channel}_to_{target_channel}"
                TE_results[key] = BTSE
                # print(f" source_channel {data[:, i, k]},target_channel {data[:, j, k]}")
                print(f"Transfer Entropy from {source_channel} to {target_channel}: {BTSE}")
            except Exception as e:
                print(f"Error computing Transfer Entropy from {source_channel} to {target_channel}: {e}")

