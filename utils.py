import numpy as np
import ot
from scipy.sparse import csr_matrix
import ipywidgets as widgets
import IPython
import matplotlib.pyplot as plt

def dirac_distribution(frequencies, values, n=1000):
    if not len(frequencies) == len(values):
        raise ValueError("Arrays should be same size")
    
    a = np.zeros(n)
    for i in range(len(frequencies)):
        f = int(frequencies[i])
        v = values[i]
        a[f] = v
    return a / np.sum(a)

def natural_dirac(f0, size, n_harmonics=None):
    f0 = int(f0)
    natural_dirac = np.zeros(size)

    if n_harmonics is None:
        n_harmonics = (size - 1) // f0

    for i in range(1, n_harmonics + 1):
        natural_dirac[f0 * i] = 1 / i**2
    return natural_dirac / np.sum(natural_dirac)

def natural_gaussian(f0, s0, size, n_harmonics=None):
    f0 = int(f0)
    s0 = int(s0)
    natural_gaussian = np.zeros(size)

    if n_harmonics is None:
        n_harmonics = (size - 1) // f0
    
    for i in range(1, n_harmonics + 1):
        natural_gaussian += ot.datasets.make_1D_gauss(size, m=f0*i, s=s0) / i**2
    
    return natural_gaussian / np.sum(natural_gaussian)

def quantile(a):
    n = len(a)
    X = np.cumsum(a)
    U = np.linspace(0, 1, n)
    return np.searchsorted(X, U, side='right')

def quantile_bary(a1, a2, alpha=0.5):
    q1 = quantile(a1)
    q2 = quantile(a2)
    return (1 - alpha) * q1 + alpha * q2

def reverse_quantile(z):
    n = len(z)
    frequencies, values = np.unique(z, return_counts=True)
    return dirac_distribution(frequencies[:-1], values[:-1], n=n)

def quantile_optimal_transport(source, target, alpha=0.5):
    z = quantile_bary(source, target, alpha)
    return reverse_quantile(z)

def get_frequency(f_i, f_j, alpha, method='int'):
    if method == 'int':
        return int((1 - alpha) * f_i + alpha * f_j)
    if method == 'round':
        return round((1 - alpha) * f_i + alpha * f_j)
    
def emd_optimal_transport(frequency, source, target, alpha=0.5, index_method='int', return_cost=False): # can i get rid of frequency?

    emd_plan = ot.emd_1d(frequency, frequency, source, target)
    emd_plan[np.isnan(emd_plan)] = 0
    emd_plan = csr_matrix(emd_plan)
    emd_interpolation = np.zeros(len(frequency))
    row, col = emd_plan.nonzero()
    for i, j in zip(row, col):
        index = get_frequency(i, j, alpha=alpha, method=index_method)
        emd_interpolation[index] += emd_plan[i, j]
    
    if return_cost:
        return emd_interpolation, emd_plan
        
    return emd_interpolation

def sinkhorn_stabilized_optimal_transport(source, target, alpha=0.5, reg=1e-2, index_method='int'):
    M = ot.utils.dist0(source.shape[0]) # cost matrix
    M /= M.max()

    ot_plan = ot.bregman.sinkhorn_stabilized(source, target, M, reg)
    ot_plan[np.isnan(ot_plan)] = 0
    ot_plan = csr_matrix(ot_plan)
    ss_interpolation = np.zeros(len(source))
    row, col = ot_plan.nonzero()
    for i, j in zip(row, col):
            index = get_frequency(i, j, alpha=alpha, method=index_method)
            ss_interpolation[index] += ot_plan[i, j]
    ss_interpolation /= np.sum(ss_interpolation)

    return ss_interpolation

# FFT

def peak_frequencies(frequency, spectrum, n_freq):
    return frequency[np.argsort(spectrum)[::-1][:n_freq]]

def invert_magnitude_fft(magnitude, source, target, alpha):
    complex_spectrum = np.zeros_like(magnitude)
    for i in range(complex_spectrum.shape[0]):
        if np.abs(source[i]) != 0:
            complex_spectrum[i] += (1 - alpha) * source[i] / np.abs(source[i])
        if np.abs(target[i]) != 0:
            complex_spectrum[i] += alpha * target[i] / np.abs(target[i])
    complex_spectrum *= magnitude
    return np.fft.ifft(complex_spectrum)    

def normalized_frame(stft, index):
    frame = stft[:, index]
    frame = np.abs(frame)
    frame_relative_amplitude = np.sum(frame)
    frame = frame / frame_relative_amplitude

    return frame, frame_relative_amplitude

# Display

def audio_widget(signal, title, sr=44100):

    audio_player = IPython.display.Audio(data=signal, rate=sr)
    out = widgets.Output()
    with out:
        display(audio_player)
    combined_widget = widgets.VBox([widgets.Label(title), out])

    return combined_widget
    
# Cost matrix exp

from math import log

def itakura_saito(f1, f2):
    if f1 == 0 or f2 == 0:
        return np.inf
    
    return f1 / f2 - log(f1 / f2) - 1

def chi_2(f1, f2):
    if f1 == 0 or f2 == 0:
        return np.inf
    
    return (f1 - f2)**2 / f1

def euclidean(f1, f2):
    return np.abs(f1 - f2)

def identity_distance(f1, f2):
    return f1

def one(f1, f2):
    return 1

def cost_matrix(support, source, target, freq_dist, amp_dist, return_distances=True):
    n_samples = len(support)
    M = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            M[i, j] = freq_dist(support[i], support[j]) * amp_dist(source[i], target[j])

    return M

def plot_submatrix(matrix, support, size_submatrix, line_index, col_index):
    sub_row_support = support[line_index - size_submatrix : line_index + size_submatrix]
    sub_col_support = support[col_index - size_submatrix : col_index + size_submatrix]

    sub_matrix = matrix[line_index - size_submatrix : line_index + size_submatrix, col_index - size_submatrix : col_index + size_submatrix]
    
    support_step = size_submatrix // 3
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(sub_matrix)
    ax.set(xticks=np.arange(0, len(sub_col_support))[::support_step], xticklabels=sub_col_support[::support_step])
    ax.set(yticks=np.arange(0, len(sub_row_support))[::support_step], yticklabels=sub_row_support[::support_step])
    ax.set_xlabel("Target", fontsize=10)
    ax.set_ylabel("Source", fontsize=10)

    cax = ax.matshow(sub_matrix, cmap='viridis')  # Use the colormap of your choice
    fig.colorbar(cax)


def submatrix(matrix, size_submatrix, line_index, col_index):
    return matrix[line_index - size_submatrix : line_index + size_submatrix, col_index - size_submatrix : col_index + size_submatrix]

def emd_custom_matrix(support, source, target, freq_dist, amp_dist, alpha=0.5):
    M = cost_matrix(support, source, target, freq_dist, amp_dist)
    max_finite_value = np.nanmax(M[np.isfinite(M)])
    
    M[np.isinf(M)] = max_finite_value
    M[np.isnan(M)] = max_finite_value

    emd_plan = ot.lp.emd(source, target, M)
    emd_plan[np.isnan(emd_plan)] = 0
    emd_plan = csr_matrix(emd_plan)
    emd_interpolation = np.zeros(len(support))
    row, col = emd_plan.nonzero()
    for i, j in zip(row, col):
        index = get_frequency(i, j, alpha=alpha, method='int')
        emd_interpolation[index] += emd_plan[i, j]
    
    return emd_interpolation, np.array(emd_plan.todense())
