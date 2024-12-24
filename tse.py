import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pandas.plotting import parallel_coordinates


def phase_space_reconstruction(data, embedding_dim, tau):
    n_points = len(data) - (embedding_dim - 1) * tau
    if n_points <= 0:
        raise ValueError("Time series is too short for the given dimensions and time delay.")
    reconstructed_matrix = np.empty((embedding_dim, n_points))
    for i in range(embedding_dim):
        reconstructed_matrix[i, :] = data[i * tau:i * tau + n_points]
    return reconstructed_matrix


def apply_2d_fft(S, R):
    fft_S = np.fft.fft2(S)
    fft_R = np.fft.fft2(R)
    return fft_S, fft_R


def extract_frequency_vectors(fft_S, fft_R):
    M, N = fft_S.shape
    w = np.empty((N, M))
    v = np.empty((N, M))
    for f in range(N):
        w[f] = np.abs(fft_S[:, f])
        v[f] = np.abs(fft_R[:, f])
    return w, v


def calculate_tse(w, v, phi):
    num_frequencies = v.shape[0]
    tse_values = np.zeros(num_frequencies)
    for f in range(num_frequencies):
        v_delay_vectors = [v[f, i] for i in range(phi, v.shape[1])]
        w_delay_vectors = [w[f, i] for i in range(phi, w.shape[1])]
        hist_v, bin_edges_v = np.histogram(np.hstack(v_delay_vectors), bins='auto', density=True)
        hist_wv, bin_edges_wv = np.histogram(np.vstack([np.hstack(w_delay_vectors), np.hstack(v_delay_vectors)]),
                                             bins='auto', density=True)
        p_v = hist_v * np.diff(bin_edges_v)
        p_wv = hist_wv * np.diff(bin_edges_wv)
        h_v = np.sum(p_v * np.log(p_v + np.finfo(float).eps))
        h_wv = np.sum(p_wv * np.log(p_wv + np.finfo(float).eps))
        tse_values[f] = h_v - h_wv
    return tse_values


def calculate_atse(tse_values, fl1, fl2):
    if fl1 < 0 or fl2 >= len(tse_values) or fl1 >= fl2:
        raise ValueError("Invalid frequency band indices.")
    atse = np.sum(tse_values[fl1:fl2]) / (fl2 - fl1)
    return atse


def calculate_transfer_spectral_entropy(x, y, embedding_dim, tau, fl1, fl2):
    S = phase_space_reconstruction(x, embedding_dim, tau)
    R = phase_space_reconstruction(y, embedding_dim, tau)
    fft_S, fft_R = apply_2d_fft(S, R)
    w, v = extract_frequency_vectors(fft_S, fft_R)
    tse_values = calculate_tse(w, v, tau)
    atse = calculate_atse(tse_values, fl1, fl2)
    return atse
