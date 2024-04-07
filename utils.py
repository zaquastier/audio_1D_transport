import numpy as np
import ot
from scipy.sparse import csr_matrix
import ipywidgets as widgets
import IPython

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

# STFT    

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
    