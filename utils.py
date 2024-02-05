import numpy as np
import random

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

def reverse_bary(z):
    n = len(z)
    frequencies, values = np.unique(z, return_counts=True)
    return dirac_distribution(frequencies[:-1], values[:-1], n=n)

def quantile_optimal_transport(a1, a2, alpha=0.5):
    z = quantile_bary(a1, a2, alpha)
    return reverse_bary(z)

def get_frequency(f_i, f_j, alpha, method='int'):
    if method == 'int':
        return int((1 - alpha) * f_i + alpha * f_j)
    if method == 'round':
        return round((1 - alpha) * f_i + alpha * f_j)
    

## dummy functions
    
# I. Diracs
    
def get_example_diracs(n=1000, method='default'):

    if method == 'default':
        f1 = n * 0.25
        f2 = n * 0.3

        a1 = dirac_distribution([f1, f1*2, f1*3], [0.2, 1, 0.3], n=n)
        a2 = dirac_distribution([f2, f2*2, f2*3], [1, 0.5, 2], n=n)

        return f1, f2, a1, a2
    
    if method == 'random':
        r1 = random.random()
        r2 = random.random()
        f1 = int(n * r1)
        f2 = int(n * r2)

        n_harmonics = random.randint(1, n // max(f1, f2))

        f1_frequencies = [f1*i for i in range(1, n_harmonics+1)]
        f2_frequencies = [f2*i for i in range(1, n_harmonics+1)]

        value_magnitude = 5
        f1_values = [random.random() * value_magnitude for _ in range(n_harmonics)]
        f2_values = [random.random() * value_magnitude for _ in range(n_harmonics)]

        a1 = dirac_distribution(f1_frequencies, f1_values, n=n)
        a2 = dirac_distribution(f2_frequencies, f2_values, n=n)

        return f1, f2, a1, a2