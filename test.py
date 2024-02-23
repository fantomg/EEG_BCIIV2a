import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.time_frequency import psd_array_multitaper, psd_array_welch

matplotlib.use('TkAgg')

filename = "dataset/A02T.gdf"

raw_gdf = mne.io.read_raw_gdf(filename, stim_channel="auto", verbose='ERROR',
                              exclude=(["EOG-left", "EOG-central", "EOG-right"]))

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

# 利用mne.io.RawArray类重新创建Raw对象，已经没有nan数据了
raw_gdf = mne.io.RawArray(data, raw_gdf.info, verbose="ERROR")
print(raw_gdf.info)

# 画出EEG通道图
raw_gdf.plot()
plt.show()

# 选择范围为Cue后 1s - 4s 的数据
tmin, tmax = 1., 4.
# 四类 MI 对应的 events_id
event_id = dict({'769': 7, '770': 8, '771': 9, '772': 10})

epochs = mne.Epochs(raw_gdf, events, event_id, tmin, tmax, proj=True, baseline=None, preload=True)
print(epochs)

# 切片，获取 events 的最后一列
labels = epochs.events[:, -1]
# Get all epochs as a 3D array.
data = epochs.get_data(copy=True)
print(labels)
print(data.shape)

# 从epochs对象中获取数据和采样频率
data = epochs.get_data()
sfreq = epochs.info['sfreq']  # 获取采样频率

# 使用Multitaper方法计算PSD
psds_multitaper, freqs_multitaper = psd_array_multitaper(data, sfreq, fmin=1, fmax=40, n_jobs=1)
psds_multitaper_log = 10 * np.log10(psds_multitaper)
print("Multitaper PSD shape:", psds_multitaper_log.shape)

# 获取数据和采样频率
data = epochs.get_data(copy=True)
sfreq = epochs.info['sfreq']

# 修正n_fft参数
n_times = data.shape[2]  # 获取数据中的时间点数
n_fft = min(2048, n_times)  # 确保n_fft不大于时间点数

# 使用Welch方法计算PSD
psds_welch, freqs_welch = psd_array_welch(data, sfreq, fmin=1, fmax=40, n_fft=n_fft, verbose=False)
psds_welch_log = 10 * np.log10(psds_welch)
psds_mean_welch = psds_welch_log.mean(axis=2)

print("Welch PSD shape:", psds_mean_welch.shape)
