import scipy.io as sio

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow.python.keras.utils.np_utils import to_categorical

import numpy as np
import scipy.fftpack
import scipy

def remove_artifacts(X, Y, m, alpha, fs):
    def phase_space_reconstruction(time_series, m, tau):
        n = len(time_series)
        reconstructed = np.empty((n - (m - 1) * tau, m))
        for i in range(m):
            reconstructed[:, i] = time_series[i * tau:n - (m - 1) * tau + i * tau]
        return reconstructed

    def two_dimensional_fft(matrix):
        return scipy.fftpack.fft2(matrix)

    def escort_distribution(p_alpha):
        return p_alpha / np.sum(p_alpha)

    def renyi_transfer_entropy(Cx, Cy, alpha):
        # Placeholder for joint and conditional probabilities calculation
        p_alpha_joint = np.random.rand()  # Replace with actual joint probabilities
        p_alpha_cond = np.random.rand()  # Replace with actual conditional probabilities
        rho_alpha = escort_distribution(p_alpha_joint)
        R_alpha_joint = 1 / (1 - alpha) * np.log2(np.sum(p_alpha_joint ** alpha))
        R_alpha_cond = 1 / (1 - alpha) * np.log2(np.sum(p_alpha_cond ** alpha / np.sum(rho_alpha ** alpha)))
        return R_alpha_cond - R_alpha_joint

    def calculate_BTSE(X, Y, m, alpha, fs):
        X_reconstructed = phase_space_reconstruction(X, m, 1)
        Y_reconstructed = phase_space_reconstruction(Y, m, 1)
        X_fft = two_dimensional_fft(X_reconstructed)
        Y_fft = two_dimensional_fft(Y_reconstructed)
        # Symbolize matrices (Placeholder, replace with actual symbolization process)
        Cx = X_fft
        Cy = Y_fft
        RTE_m = renyi_transfer_entropy(Cx, Cy, alpha)
        RTE_m1 = renyi_transfer_entropy(Cx, Cy, alpha)  # Placeholder for RTE at m + 1
        BTSE = (RTE_m1 - RTE_m) / np.log((m + 1) / (m - 1))
        return BTSE

    # Iterate over trials and remove artifacts using BTSE
    for trial in range(X.shape[0]):
        # Assuming X is the input data
        X_trial = X[trial, 0, :, :]
        # Assuming Y is the target data
        Y_trial = Y[trial, 0, :, :]
        # Calculate BTSE for the trial
        btse_value = calculate_BTSE(X_trial, Y_trial, m, alpha, fs)
        # Use the btse_value to assist in removing artifacts from the trial

    # Return the modified data
    return X, Y


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
    n_channels = 22
    n_tests = 6 * 48
    window_Length = 7 * 250

    # Define MI trial window
    fs = 250  # sampling rate
    t1 = int(1.5 * fs)  # start time_point
    t2 = int(6 * fs)  # end time_point

    class_return = np.zeros(n_tests)
    data_return = np.zeros((n_tests, n_channels, window_Length))

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
            class_return[NO_valid_trial] = int(a_y[trial])
            NO_valid_trial += 1

    data_return = data_return[0:NO_valid_trial, :, t1:t2]
    class_return = class_return[0:NO_valid_trial]
    class_return = (class_return - 1).astype(int)

    return data_return, class_return


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
        X_train, y_train = load_BCI2a_data(path, subject + 1, True)
        X_test, y_test = load_BCI2a_data(path, subject + 1, False)
    # elif (dataset == 'HGD'):
    #     X_train, y_train = load_HGD_data(path, subject+1, True)
    #     X_test, y_test = load_HGD_data(path, subject+1, False)
    else:
        raise Exception("'{}' dataset is not supported yet!".format(dataset))

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
