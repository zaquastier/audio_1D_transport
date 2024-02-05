import numpy as np

def dirac_distribution(frequencies, values, n=1000):
    if not len(frequencies) == len(values):
        raise ValueError("Arrays should be same size")
    
    a = np.zeros(n)
    for i in range(len(frequencies)):
        f = int(frequencies[i])
        v = values[i]
        a[f] = v
    return a / np.sum(a)

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

def quantile_optimal_transport(a1, a2, alpha=0.5):
    z = quantile_bary(a1, a2, alpha)
    return reverse_quantile(z)

def get_frequency(f_i, f_j, alpha, method='int'):
    if method == 'int':
        return int((1 - alpha) * f_i + alpha * f_j)
    if method == 'round':
        return round((1 - alpha) * f_i + alpha * f_j)
    