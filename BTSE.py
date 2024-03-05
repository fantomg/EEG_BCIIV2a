import numpy as np


def phase_space_reconstruction(time_series, m, tau):
    n = len(time_series)
    reconstructed = np.empty((n - (m - 1) * tau, m))
    for i in range(m):
        reconstructed[:, i] = time_series[i * tau:n - (m - 1) * tau + i * tau]
    return reconstructed


def two_dimensional_fft(matrix):
    return np.fft.fft2(matrix)


def calculate_joint_probabilities(X, Y, m, alpha):
    p_alpha_joint = np.abs(X * Y.conj()) ** alpha
    return p_alpha_joint


def calculate_conditional_probabilities(X, Y, m, alpha):
    p_alpha_cond = np.abs(Y) ** alpha
    return p_alpha_cond


def calculate_BTSE(X, Y, m, alpha):
    # Phase space reconstruction
    X_reconstructed = phase_space_reconstruction(X, m, 1)
    Y_reconstructed = phase_space_reconstruction(Y, m, 1)

    # 2D FFT
    X_fft = two_dimensional_fft(X_reconstructed)
    Y_fft = two_dimensional_fft(Y_reconstructed)

    # Calculate joint and conditional probabilities
    p_alpha_joint = calculate_joint_probabilities(X_fft, Y_fft, m, alpha)
    p_alpha_cond = calculate_conditional_probabilities(X_fft, Y_fft, m, alpha)

    # Escort distribution
    rho_alpha = p_alpha_joint / np.sum(p_alpha_joint)

    # Calculate Renyi Transfer Entropy
    R_alpha_joint = 1 / (1 - alpha) * np.log2(np.sum(p_alpha_joint))
    R_alpha_cond = 1 / (1 - alpha) * np.log2(np.sum(p_alpha_cond / rho_alpha))

    # Calculate BTSE
    BTSE = (R_alpha_cond - R_alpha_joint) * np.log((m + 1) / (m - 1))
    min_val = 0  # Minimum value for normalization
    max_val = 1  # Maximum value for normalization
    normalized_btse = (BTSE - np.min(BTSE)) / (np.max(BTSE) - np.min(BTSE)) * (max_val - min_val) + min_val
    return normalized_btse

# # Example usage
# X = np.random.rand(1000)  # Replace with actual time series dataset x(t)
# Y = np.random.rand(1000)  # Replace with actual time series dataset y(t)
# m = 3  # Example embedding dimension
# alpha = 2  # Alpha value for Renyi entropy
# fs = 100  # Example sampling frequency
#
# BTSE_value = calculate_BTSE(X, Y, m, alpha, fs)
# print(BTSE_value)
