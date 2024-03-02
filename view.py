import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
from pyinform import transfer_entropy
from mne.preprocessing import ICA

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

# 画出EEG通道图
raw_gdf.plot()
plt.show()

# 选择范围为Cue后 1s - 4s 的数据
tmin, tmax = 1.5, 6.
# 四类 MI 对应的 events_id
event_id = dict({'769': 7, '770': 8, '771': 9, '772': 10})

epochs = mne.Epochs(raw_gdf, events, event_id, tmin, tmax, proj=True, baseline=None, preload=True)
print(epochs)

# 切片，获取 events 的最后一列
labels = epochs.events[:, -1]
# Get all epochs as a 3D array.
data = epochs.get_data(copy=True)
print(labels.shape)
print(data.shape)

# 获取 EEG 通道名称列表
eeg_channels = raw_gdf

# 定义一个字典来存储计算结果
TE_results = {}

# 设置嵌入维度
k = 1

# 遍历每个通道计算传递熵
for i, source_channel in enumerate(epochs):
    for j, target_channel in enumerate(epochs):
        if i != j:  # To avoid computing Transfer Entropy for the same channel
            try:
                TE = transfer_entropy(data[:, i, :], data[:, j, :], k)
                key = f"{source_channel}_to_{target_channel}"
                TE_results[key] = TE
                print(f"Transfer Entropy from {source_channel} to {target_channel}: {TE}")
            except Exception as e:
                print(f"Error computing Transfer Entropy from {source_channel} to {target_channel}: {e}")



# # 画出EEG通道图
# epochs.plot()
# plt.show()
# # Perform Independent Component Analysis (ICA) to remove artifacts
# ica = ICA(n_components=25, random_state=97, method="infomax")
# ica.fit(raw_gdf)
#
# # ica.exclude = []
# # Exclude components related to eye movements (EOG) or other artifacts
# # eog_indices, _ = ica.find_bads_eog(epochs)
# ica.exclude = [23, 24, 25]
#
# # plot diagnostics
# ica.plot_properties(raw_gdf, picks=[23, 24, 25])
#
# # plot ICs applied to raw data, with EOG matches highlighted
# ica.plot_sources(raw_gdf, show_scrollbars=False)

# # Apply ICA to remove artifact components
# ica.apply(raw_gdf)
#
# # Plot the cleaned EEG data
# raw_gdf.plot()
#
# plt.show()
# # Save the cleaned epochs data
# # cleaned_epochs_file = 'cleaned_epochs-epo.fif'
# # epochs.save(cleaned_epochs_file)
#
# # Further analysis or visualization can be performed on the cleaned epochs data