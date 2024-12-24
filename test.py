import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import scipy.io as sio
from mne import create_info
from mne.io import RawArray
from mne.preprocessing import ICA

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical

import numpy as np
from BTSE import calculate_BTSE


def load_BCI2a_data(data_path, subject, training, all_trials=True):
    """ Loading and Dividing of the dataset set based on the subject-specific
    (subject-dependent) approach.
    In this approach, we used the same training and testing dataas the original
    competition, i.e., 288 x 9 trials in session 1 for training,
    and 288 x 9 trials in session 2 for testing.

        Parameters
        ----------
        data_path: string
            dataset path
            # Dataset BCI Competition IV-2a is available on
            # http://bnci-horizon-2020.eu/database/data-sets
        subject: int
            number of subject in [1, .. ,9]
        training: bool
            if True, load training dataset
            if False, load testing dataset
        all_trials: bool
            if True, load all trials
            if False, ignore trials with artifacts
    """

    # Define MI-trials parameters
    n_channels = 25
    n_tests = 12 * 48
    window_Length = 7 * 250

    # Define MI trial window
    fs = 250  # sampling rate
    t1 = int(1.5 * fs)  # start time_point
    t2 = int(6 * fs)  # end time_point

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, 22, window_Length))
    data_eog = np.zeros((n_tests, 3, window_Length))  # Store data for 3 EOG channels

    NO_valid_trial = 0
    if training:
        a = sio.loadmat(data_path + 'A0' + str(subject) + 'T.mat')
    else:
        a = sio.loadmat(data_path + 'A0' + str(subject) + 'E.mat')
    a_data = a['data']
    for ii in range(0, a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0]
        a_trial = a_data3[1]
        a_y = a_data3[2]
        a_artifacts = a_data3[5]

        for trial in range(0, a_trial.size):
            if (a_artifacts[trial] != 0 and not all_trials):
                continue
            data_return[NO_valid_trial, :, :] = np.transpose(
                a_X[int(a_trial[trial]):(int(a_trial[trial]) + window_Length), :22])
            data_eog[NO_valid_trial, :, :] = np.transpose(
                a_X[int(a_trial[trial]):(int(a_trial[trial]) + window_Length), 22:25])
            class_return[NO_valid_trial] = int(a_y[trial])
            NO_valid_trial += 1

    data_return = data_return[0:NO_valid_trial, :, t1:t2]
    data_eog = data_eog[0:NO_valid_trial, :, t1:t2]
    print("data_eeg 的形状:", data_return.shape)
    # print("data_eog 的形状:", data_eog.shape)
    print("class_return 的形状:", class_return.shape)
    # print(data_eog)
    class_return = class_return[0:NO_valid_trial]
    class_return = (class_return - 1).astype(int)

    return data_return, data_eog, class_return


def remove_artifacts_and_filter(data_return, data_eog, threshold, segment_length):
    eeg_return = np.zeros_like(data_return)  # Placeholder for processed EEG data
    num_segments = len(data_return) // segment_length
    print("num_segments:", num_segments)

    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = (i + 1) * segment_length
        segment_eeg = data_return[start_idx:end_idx]
        segment_eog = data_eog[start_idx:end_idx]

        for j in range(22):  # Iterate over each EEG channel
            for k in range(3):  # Iterate over each EOG channel
                BTSE_eeg = calculate_BTSE(segment_eeg[:, j].flatten(), segment_eog[:, k].flatten(), m=8, alpha=3)
                if -threshold < BTSE_eeg < threshold:
                    # Create MNE Info object for EEG data
                    ch_names = [f'EEG{j + 1}' for j in range(22)]
                    info = create_info(ch_names, sfreq=250, ch_types='eeg')

                    # Create RawArray object from EEG segment
                    raw_eeg = RawArray(segment_eeg.transpose(1, 0, 2).reshape(22, -1), info)

                    # Fit ICA with Raw object
                    ica = ICA(n_components=15, max_iter="auto", random_state=97)
                    ica.fit(raw_eeg)

                    # Create MNE Info object for EOG data
                    eog_ch_names = [f'EOG{k + 1}' for k in range(3)]  # Assuming channel names like EOG1, EOG2, EOG3
                    eog_info = create_info(eog_ch_names, sfreq=250, ch_types='eog')

                    # Create RawArray object from EOG segment
                    # Ensure that the shape of the EOG data is (n_channels, n_samples)
                    eog_raw = RawArray(segment_eog.transpose(1, 0, 2).reshape(3, -1), eog_info)

                    # Find EOG components and exclude them
                    eog_indices, eog_score = ica.find_bads_eog(eog_raw)
                    ica.exclude = eog_indices

                    # Apply ICA solution to remove EOG artifacts
                    cleaned_segment_eeg = ica.apply(raw_eeg.get_data().T)

                    # Replace original EEG segment with cleaned EEG segment
                    data_return[start_idx:end_idx, j] = cleaned_segment_eeg.T

    return data_return


def standardize_data(X_train, X_test, channels):
    # X_train & X_test :[Trials, MI-tasks, Channels, Time points]
    for j in range(channels):
        scaler = StandardScaler()
        scaler.fit(X_train[:, 0, j, :])
        X_train[:, 0, j, :] = scaler.transform(X_train[:, 0, j, :])
        X_test[:, 0, j, :] = scaler.transform(X_test[:, 0, j, :])

    return X_train, X_test


def get_data(path, subject, dataset='BCI2a', classes_labels='all', isStandard=True, isShuffle=True):
    # Load and split the dataset into training and testing

    """ Loading and Dividing of the dataset set based on the subject-specific
            (subject-dependent) approach.
            In this approach, we used the same training and testing dataset as the original
            competition, i.e., for BCI Competition IV-2a, 288 x 9 trials in session 1
            for training, and 288 x 9 trials in session 2 for testing.
            """
    if (dataset == 'BCI2a'):
        path = path + 's{:}/'.format(subject + 1)
        X_train, data_eogx, y_train = load_BCI2a_data(path, subject + 1, True)
        X_test, data_eogc, y_test = load_BCI2a_data(path, subject + 1, False)
    # elif (dataset == 'HGD'):
    #     X_train, y_train = load_HGD_data(path, subject+1, True)
    #     X_test, y_test = load_HGD_data(path, subject+1, False)
    else:
        raise Exception("'{}' dataset is not supported yet!".format(dataset))
    print("开始清理")
    X_train = remove_artifacts_and_filter(X_train, data_eogx, 1, 144)
    X_test = remove_artifacts_and_filter(X_test, data_eogc, 1, 144)
    # shuffle the dataset
    if isShuffle:
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_test, y_test = shuffle(X_test, y_test, random_state=42)

    # Prepare training dataset
    N_tr, N_ch, T = X_train.shape
    X_train = X_train.reshape(N_tr, 1, N_ch, T)
    y_train_onehot = to_categorical(y_train)
    # Prepare testing dataset
    N_tr, N_ch, T = X_test.shape
    X_test = X_test.reshape(N_tr, 1, N_ch, T)
    y_test_onehot = to_categorical(y_test)

    # Standardize the dataset
    if isStandard:
        X_train, X_test = standardize_data(X_train, X_test, N_ch)

    return X_train, y_train, y_train_onehot, X_test, y_test, y_test_onehot


# get_data(os.path.expanduser('~') + '/PycharmProjects/EEG_BCIIV2a/dataset/', 1)
print(tf.test.is_built_with_cuda())  # 判断CUDA是否可用
print(tf.config.list_physical_devices('GPU'))  # 查看cuda、TensorFlow_GPU和cudnn(选择下载，cuda对深度学习的补充)版本是否对应
