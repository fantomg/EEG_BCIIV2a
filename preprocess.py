import os
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
import view


def load_BCI2a_data(data_path, subject, training, all_trials=True):
    if training:
        data_return, class_return = view.asr_test(data_path + 'A0' + str(subject) + 'T.gdf')
    else:
        data_return, class_return = view.asr_test(data_path + 'A0' + str(subject) + 'E.gdf')

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
