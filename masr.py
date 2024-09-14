import matplotlib
import seaborn as sns
import pywt
import matplotlib.pyplot as plt
import mne
import numpy as np
from asrpy import ASR
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
from mne.preprocessing import ICA
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from joblib import Parallel, delayed
import analyze_merits
import tse
import pandas as pd
import time


def wpd(data, max_level):
    wp = pywt.WaveletPacket(data=data, wavelet='db1', mode='symmetric', maxlevel=max_level)
    return wp


def read_raw_gdf(filename):
    """
    Read and preprocess EEG data from a GDF file.

    Parameters:
    filename (str): The path to the GDF file containing the EEG data.

    Returns:
    raw_gdf (mne.io.Raw): The preprocessed EEG data.
    events (array): The event times.
    events_id (array): The event identifiers.
    """
    # Read the EEG data from the GDF file
    raw_gdf = mne.io.read_raw_gdf(filename, stim_channel="auto", verbose='ERROR')

    # Rename the channels to the 10-20 system, commonly used for 64 electrode positions
    raw_gdf.rename_channels(
        {'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 'EEG-3': 'FC2', 'EEG-4': 'FC4',
         'EEG-5': 'C5', 'EEG-C3': 'C3', 'EEG-6': 'C1', 'EEG-Cz': 'Cz', 'EEG-7': 'C2', 'EEG-C4': 'C4', 'EEG-8': 'C6',
         'EEG-9': 'CP3', 'EEG-10': 'CP1', 'EEG-11': 'CPz', 'EEG-12': 'CP2', 'EEG-13': 'CP4',
         'EEG-14': 'P1', 'EEG-15': 'Pz', 'EEG-16': 'P2', 'EEG-Pz': 'POz'})

    # Pre-load the dataset
    raw_gdf.load_data()

    # Correct NaN values in the data
    data = raw_gdf.get_data()

    for i_chan in range(data.shape[0]):  # Iterate over 22 channels
        # Set all values in the array to NaN, then replace these NaN values with the mean of the array
        this_chan = data[i_chan]
        data[i_chan] = np.where(
            this_chan == np.min(this_chan), np.nan, this_chan
        )
        mask = np.isnan(data[i_chan])
        chan_mean = np.nanmean(data[i_chan])
        data[i_chan, mask] = chan_mean

    # Get the event times and their corresponding identifiers
    events, events_id = mne.events_from_annotations(raw_gdf)

    # Recreate the Raw object using the mne.io.RawArray class, with no NaN values
    raw_gdf = mne.io.RawArray(data, raw_gdf.info, verbose="ERROR")

    # Plot the EEG channel layout
    # raw_gdf.plot()
    # plt.show()

    return raw_gdf, events, events_id


def get_epochs(raw_gdf, events):
    """
    This function selects a specific time window (1.5s to 5.995s) from the raw EEG data based on the given events and
    their corresponding IDs. It then creates an MNE Epochs object containing the selected data.

    Parameters:
    raw_gdf (mne.io.Raw): The raw EEG data to be processed.
    events (array-like): The events array containing the timing and type of each event.
    events_id (dict): A dictionary mapping event IDs to their corresponding MI class.

    Returns:
    epochs (mne.Epochs): The EEG epochs object containing the selected data.
    events_id (dict): The dictionary mapping event IDs to their corresponding MI class.
    data (array): A 3D numpy array containing the selected EEG data.
    """
    # Select a range of 1.5s to 5.995s after the Cue
    tmin, tmax = 1.5, 5.995

    # Define the event IDs corresponding to each MI class
    # events_id = dict({'769': 7, '770': 8})
    events_id = dict({'769': 7, '770': 8, '771': 9, '772': 10})

    # Create an MNE Epochs object containing the selected data
    epochs = mne.Epochs(raw_gdf, events, events_id, tmin, tmax, proj=True, baseline=None, preload=True,
                        event_repeated='drop')

    # Get all epochs as a 3D array
    data = epochs.get_data(copy=True)

    return epochs, events_id, data


def wpd_plt(signal, n):
    """
    Plots the wavelet packet decomposition (WPD) of a given signal.

    Parameters:
    signal (numpy.ndarray): The input signal for which the WPD will be calculated.
    n (int): The maximum level of decomposition.

    Returns:
    None. The function plots the WPD of the input signal.
    """
    # Perform WPD decomposition using PyWavelets library
    wp = pywt.WaveletPacket(data=signal, wavelet='db1', mode='symmetric', maxlevel=n)

    # Create a map to store the coefficients of each node
    map = {}
    map[1] = signal
    for row in range(1, n + 1):
        lev = []
        for i in [node.path for node in wp.get_level(row, 'freq')]:
            map[i] = wp[i].data

    # Plot the WPD
    plt.figure(figsize=(16, 12))
    plt.subplot(n + 1, 1, 1)  # Plot the first subplot
    plt.plot(map[1])
    for i in range(2, n + 2):
        level_num = pow(2, i - 1)  # Calculate the number of subplots in the current row
        # Get the nodes at the current level
        re = [node.path for node in wp.get_level(i - 1, 'freq')]
        for j in range(1, level_num + 1):
            plt.subplot(n + 1, level_num, level_num * (i - 1) + j)
            plt.plot(map[re[j - 1]])  # Plot the coefficients of the current node


def visualize_channel_importance(transformed_features, channel_names):
    # Initialize MaxAbsScaler
    scaler = MaxAbsScaler()
    # Fit and transform the transformed features
    abs_scaled_features = scaler.fit_transform(transformed_features.reshape(-1, 1))
    normalized_features = (abs_scaled_features + 1) / 2
    # Reshape the normalized features back to the original shape
    normalized_features = normalized_features.reshape(transformed_features.shape)
    # Calculate the mean and standard deviation of the normalized features
    mean_value_normalized = np.mean(normalized_features)
    std_dev_normalized = np.std(normalized_features)
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['skyblue', 'lightgreen', 'goldenrod'])
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axhspan(mean_value_normalized + std_dev_normalized, 1, facecolor='peachpuff', alpha=0.5)
    ax.axhspan(mean_value_normalized - std_dev_normalized, mean_value_normalized + std_dev_normalized,
               facecolor='lightblue', alpha=0.5)
    ax.axhspan(0, mean_value_normalized - std_dev_normalized, facecolor='lightgreen', alpha=0.5)
    ax.set_ylim(0, 1)
    bar_width = 0.4
    spacing = 0.1
    for i in range(len(normalized_features.flatten())):
        value = normalized_features.flatten()[i]
        for y in range(int(value * 40)):
            rect = patches.Rectangle((i * (bar_width + spacing), y / 40), bar_width, 0.02, linewidth=0, edgecolor=None,
                                     facecolor=cmap(y / 40))
            ax.add_patch(rect)
    ax.axhline(mean_value_normalized + std_dev_normalized, color='r', linestyle='--', linewidth=2,
               label='Positive Std Dev: {:.2f}'.format(mean_value_normalized + std_dev_normalized))
    ax.axhline(mean_value_normalized - std_dev_normalized, color='r', linestyle='--', linewidth=2,
               label='Negative Std Dev: {:.2f}'.format(mean_value_normalized - std_dev_normalized))
    ax.set_xlabel('EEG Channel')
    ax.set_ylabel('Channel Importance (Normalized)')
    ax.set_title('Channel Importance Influenced by EOG via PCA (Normalized)')
    ax.set_xticks([i * (bar_width + spacing) + bar_width / 2 for i in range(len(normalized_features.flatten()))])
    ax.set_xticklabels(channel_names, rotation=45)
    ax.legend()
    plt.tight_layout()
    matplotlib.use('TkAgg')
    plt.show()


def calculate_cv(data):
    """
    Calculates the coefficient of variation (CV) for a given dataset.

    Parameters:
    data (numpy.ndarray): The dataset for which to calculate the CV.

    Returns:
    float: The coefficient of variation for the given dataset.
    """
    mean = np.mean(data)
    std = np.std(data)
    cv = (std / mean)
    return cv


def TPW(data, cutoff_w):
    """
    This function calculates the Transfer Power (TPW) between different channels in the given data.
    It also performs Principal Component Analysis (PCA) on the calculated TPW values.

    Parameters:
    data (numpy.ndarray): A 2D array containing the EEG data with shape (observations, channels).
    cutoff_w (int): A parameter used in the function.

    Returns:
    transformed_features (numpy.ndarray): A 2D array containing the transformed features after PCA.
    cutoff_w (int): The updated value of cutoff_w after processing.
    """
    tsedata = np.mean(data, axis=0)
    num_channels = 25
    num_rows = 22
    num_cols = 3
    channels = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
        'P1', 'Pz', 'P2', 'POz', 'EOG-left', 'EOG-central', 'EOG-right'
    ]
    tse_matrix = np.zeros((num_channels, num_channels))

    for i in range(num_rows):
        for j in range(num_channels - num_cols, num_channels):
            if i != j:
                try:
                    embedding_dim = 6
                    tau = 1
                    fl1 = 0
                    fl2 = 100
                    tse_value = tse.calculate_transfer_spectral_entropy(tsedata[i], tsedata[j], embedding_dim, tau, fl1,
                                                                        fl2)
                    tse_matrix[i, j] = tse_value
                except Exception as e:
                    print(f"Error computing Transfer Spectral Entropy: {e}")
    sub_matrix = tse_matrix[:22, -3:]
    scaler = MinMaxScaler()
    normalized_sub_matrix = scaler.fit_transform(sub_matrix)
    offset = 0.01
    normalized_sub_matrix += offset
    normalized_sub_matrix = np.clip(normalized_sub_matrix, 0, 1)
    tse_matrix[:22, -3:] = normalized_sub_matrix
    fmin, fmax = 0.5, 100
    n_fft = 1125
    n_per_seg = 256
    num_channels = 25
    psd_similarities = np.zeros((num_channels, num_channels))
    psds_dict = {}
    for ch_idx in range(num_channels):
        ch_name = channels[ch_idx]
        ch_data = data[:, ch_idx]
        psd, freqs = mne.time_frequency.psd_array_welch(ch_data, 250, fmin=fmin, fmax=fmax, n_fft=n_fft,
                                                        n_per_seg=n_per_seg, verbose=False)
        psds_dict[ch_name] = psd.mean(axis=0)
    for i in range(num_rows):
        channel_i = channels[i]
        for j in range(num_channels - num_cols, num_channels):
            channel_j = channels[j]
            corr_coef, _ = pearsonr(psds_dict[channel_i], psds_dict[channel_j])
            psd_similarities[i, j] = corr_coef
            psd_similarities[j, i] = corr_coef
    sub_matrix = psd_similarities[:22, -3:]
    scaler = MinMaxScaler()
    normalized_sub_matrix = scaler.fit_transform(sub_matrix)
    offset = 0.01
    normalized_sub_matrix += offset
    normalized_sub_matrix = np.clip(normalized_sub_matrix, 0, 1)
    psd_similarities[:22, -3:] = normalized_sub_matrix
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
            wpd_similarities[j, i] = wpd_similarities[i, j]
    sub_matrix = wpd_similarities[:22, -3:]
    scaler = MinMaxScaler()
    normalized_sub_matrix = scaler.fit_transform(sub_matrix)
    offset = 0.01
    normalized_sub_matrix += offset
    normalized_sub_matrix = np.clip(normalized_sub_matrix, 0, 1)
    wpd_similarities[:22, -3:] = normalized_sub_matrix
    features_TSE_EOG = tse_matrix[:22, -3:]
    features_PSD_EOG = psd_similarities[:22, -3:]
    features_WPD_EOG = wpd_similarities[:22, -3:]
    features = np.hstack((features_TSE_EOG, features_PSD_EOG, features_WPD_EOG))
    pca = PCA(n_components=1)
    transformed_features = pca.fit_transform(features)
    scaler = MinMaxScaler()
    min_max_scaled = scaler.fit_transform(transformed_features)
    if calculate_cv(min_max_scaled) > 0.5:
        print(calculate_cv(min_max_scaled))
        cutoff_w += 1
    return transformed_features, cutoff_w


def pca_TPW(data):
    """
    Performs Principal Component Analysis (PCA) on the given data and applies the TPW method.

    Parameters:
    data (numpy.ndarray): The input data to be processed.

    Returns:
    combined_features (numpy.ndarray): The combined features after PCA and TPW method.
    cutoff_w (int): The sum of cutoff group widths from each iteration.
    """
    # Split the data into multiple parts for parallel processing
    num_splits = 5
    data_splits = np.array_split(data, num_splits)

    # Apply TPW method in parallel for each data split
    all_results = Parallel(n_jobs=-1)(delayed(TPW)(data_part, 0) for data_part in data_splits)

    # Unzip the results and combine the features and cutoff_w values
    combined_features, cutoff_ws = zip(*all_results)

    # Plot the cutoff group width over iterations
    plot_cutoff_w_over_iterations(combined_features)

    # Calculate the average feature
    combined_features = np.mean(combined_features, axis=0)

    # Define the channel names
    channel_names = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
        'P1', 'Pz', 'P2', 'POz'
    ]

    # Visualize the channel significance using PCA and TPW method
    visualize_channel_importance(combined_features, channel_names)

    # Calculate the sum of cutoff group widths
    cutoff_w = sum(cutoff_ws)

    # Print the cutoff group width
    print("Cutoff group width: ", cutoff_w)

    return combined_features, cutoff_w


def plot_cutoff_w_over_iterations(combined_features):
    """
    This function plots the coefficient of variation (CV) and channel significance over iterations.

    Parameters:
    combined_features (numpy.ndarray): A 3D numpy array containing the combined features from multiple iterations.
        The shape of the array is (5, n_channels, n_features), where 5 represents the number of iterations.

    Returns:
    None. The function generates a violin plot using seaborn library to visualize the CV and channel significance.
    """
    cv_values = []
    for i in range(5):
        scaler = MinMaxScaler()
        min_max_scaled = scaler.fit_transform(combined_features[i])
        cv_value = calculate_cv(min_max_scaled)
        cv_values.append(cv_value.mean())

    # Reshape the data into a DataFrame for easier plotting
    data_list = []
    for i in range(5):
        df = pd.DataFrame(combined_features[i])
        df['Iteration'] = i + 1
        data_list.append(df)
    combined_df = pd.concat(data_list, ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a violin plot using seaborn
    sns.violinplot(data=combined_df, x='Iteration', y=combined_df.columns[0], hue='Iteration',
                   inner="point", bw_adjust=cv_values[0], density_norm='width', palette="Set3", ax=ax)

    # Set the labels and title of the plot
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Channel Significance')
    ax.set_title('Coefficient of Variation and Channel Significance Over Iterations')

    # Use the TkAgg backend for matplotlib
    matplotlib.use('TkAgg')

    # Adjust the layout of the plot
    fig.tight_layout()

    # Display the plot
    plt.show()


def tpasr(transformed_features, raw_gdf, width):
    """
    This function applies the TPASR (Temporal and Pseudo-Temporal Artifact Subspace Reconstruction) algorithm to the EEG data.
    TPASR is a method for artifact removal in EEG data that combines temporal and pseudo-temporal artifact subspace
    reconstruction.

    Parameters:
    transformed_features (numpy.ndarray): A 2D numpy array containing the transformed features of each channel.
    raw_gdf (mne.io.Raw): The raw EEG data to be processed.
    width (int): The width parameter for the ASR algorithm.

    Returns:
    eeg_processed (mne.io.Raw): The processed EEG data after applying TPASR.
    """
    # Assuming transformed_features is a 2D numpy array containing the impact of each channel
    sorted_features = np.sort(transformed_features)
    mean_value = np.mean(sorted_features)
    std_value = np.std(sorted_features)

    low_threshold = round(float(mean_value - std_value), 3)
    high_threshold = round(float(mean_value + std_value), 3)

    # Convert channel indices to channel names
    channel_names = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
        'P1', 'Pz', 'P2', 'POz'
    ]
    montage = mne.channels.make_standard_montage('standard_1020')
    raw_gdf.info.set_montage(montage, on_missing='ignore')
    raw_gdf.set_channel_types({'EOG-left': 'eog', 'EOG-central': 'eog', 'EOG-right': 'eog'})

    low_impact_channels = [channel_names[i] for i, feature in enumerate(transformed_features.flatten()) if
                           feature < low_threshold]
    medium_impact_channels = [channel_names[i] for i, feature in enumerate(transformed_features.flatten()) if
                              low_threshold <= feature <= high_threshold]
    high_impact_channels = [channel_names[i] for i, feature in enumerate(transformed_features.flatten()) if
                            feature > high_threshold]
    print(low_impact_channels)
    print(medium_impact_channels)
    print(high_impact_channels)

    # Define ASR parameters for each group
    asr_params_low = {'cutoff': 20 + width + width, 'max_bad_chans': 0.3}
    asr_params_medium = {'cutoff': 20 + width, 'max_bad_chans': 0.2}
    asr_params_high = {'cutoff': 20, 'max_bad_chans': 0.2}

    # Create a new Raw object to avoid modifying the original data
    eeg_processed = raw_gdf.copy()

    # Apply ASR processing to each group
    for group, params in zip([low_impact_channels, medium_impact_channels, high_impact_channels],
                             [asr_params_low, asr_params_medium, asr_params_high]):
        if group != []:
            # Initialize ASR instance
            asr = ASR(sfreq=raw_gdf.info['sfreq'], cutoff=params['cutoff'], max_bad_chans=params['max_bad_chans'], )
            # Train ASR
            raw_processed = raw_gdf.copy()
            asr.fit(raw_processed)
            # Apply ASR transformation
            raw_processed = asr.transform(raw_processed)
            # Replace data in eeg_processed with processed data
            for channel_name in group:
                channel_index = channel_names.index(channel_name)
                eeg_processed._data[channel_index] = raw_processed._data[channel_index]

    return eeg_processed


def init_asr(raw_gdf):
    """
    Applies ASR (Artifact Subspace Reconstruction) to the raw EEG data.

    Parameters:
    raw_gdf (mne.io.Raw): The raw EEG data to be processed.

    Returns:
    raw_processed (mne.io.Raw): The processed EEG data after applying ASR.

    The function initializes an ASR instance with the specified parameters (sampling frequency, cutoff, and maximum
    allowed percentage of bad channels) and applies the ASR transformation to the raw EEG data.
    """
    # Use ASR to process data, here we don't group the channels, cutoff is set to 20
    asr = ASR(sfreq=raw_gdf.info['sfreq'], cutoff=20, max_bad_chans=0.2)
    # Train the ASR
    asr.fit(raw_gdf)
    # Apply ASR transformation
    raw_processed = asr.transform(raw_gdf)

    return raw_processed


def plot_asr(epochs, cleaned_avg, channel_names):
    """
    Plots the average EEG waveforms before and after ASR processing.

    Parameters:
    epochs (mne.Epochs): The EEG epochs object containing raw data.
    cleaned_avg (mne.Epochs): The EEG epochs object containing cleaned data after ASR processing.
    channel_names (list): The list of EEG channel names.

    Returns:
    None. The function displays the average EEG waveforms using matplotlib.
    """
    # Load or create channel position information
    montage = mne.channels.make_standard_montage('standard_1020')

    # Now you can try plotting again, this time with spatial colors enabled
    evoked1 = epochs.average(picks=channel_names)
    evoked1.set_montage(montage)
    # evoked1.plot(spatial_colors=True, titles="Before ASR")

    cleaned_avg.set_montage(montage)
    evoked2 = cleaned_avg.average()
    # evoked2.plot(spatial_colors=True, titles="After ASR")

    difference_evoked = mne.combine_evoked([evoked1, evoked2], weights=[1, -1])
    # Use plot method to plot the difference waveforms
    difference_evoked.plot(spatial_colors=True, gfp=True, titles="Waveform Differences (Raw signal - After ASR)")


def masr_test(filename):
    """
    This function performs artifact removal and artifact detection on EEG data using various methods.

    Parameters:
    filename (str): The path to the EEG data file.

    Returns:
    cleand_data (ndarray): The cleaned EEG data after artifact removal and detection.
    labels (ndarray): The labels corresponding to the events in the EEG data.
    """
    raw_gdf, events, events_id = read_raw_gdf(filename)
    channel_names = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
        'P1', 'Pz', 'P2', 'POz'
    ]
    epochs, events_id, data = get_epochs(raw_gdf, events)
    start_time = time.time()
    transformed_features, width = pca_TPW(data)
    end_time = time.time()  # Record end time
    total_time = end_time - start_time  # Calculate total time
    print(f"relation time：{total_time:.2f}秒")
    start_time = time.time()
    raw_processed = tpasr(transformed_features, raw_gdf, width)
    end_time = time.time()  # Record end time
    total_time = end_time - start_time  # Calculate total time
    print(f"MASR time：{total_time:.2f}秒")
    start_time = time.time()
    raw_processed1 = init_asr(raw_gdf)
    end_time = time.time()  # Record end time
    total_time = end_time - start_time  # Calculate total time
    print(f"ASR time：{total_time:.2f}秒")
    raw_gdf.set_eeg_reference('average', projection=True)
    # Fit ICA model
    ica1 = ICA(n_components=22, method='picard')
    ica1.fit(raw_gdf)

    # Identify and mark EOG artifacts
    eog_inds, scores = ica1.find_bads_eog(raw_gdf)  # Find which ICs match the EOG pattern
    ica1.exclude = eog_inds  # Mark components to be excluded

    # Apply ICA to remove artifact components
    raw_processed2 = ica1.apply(raw_gdf.copy())

    # Create EOG artifact events and calculate average EOG artifact response
    eog_epochs = mne.preprocessing.create_eog_epochs(raw_gdf)
    eog_evoked = eog_epochs.average()
    eog_evoked.plot_joint()

    raw_ssp = raw_gdf.copy()
    # Compute EOG projection vectors
    eog_projs, _ = mne.preprocessing.compute_proj_eog(raw_ssp, n_grad=1, n_mag=1, n_eeg=1)

    # Check generated projection objects
    print(f"Type of eog_projs: {type(eog_projs)}")
    print(f"First element type: {type(eog_projs[0])}")

    # Add EOG projection vectors to raw data
    raw_ssp.add_proj(eog_projs)

    # Apply projection vectors
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
    return cleand_data, labels


if __name__ == '__main__':
    masr_test("dataset/s6/A06T.gdf")
