import numpy as np
import scipy.stats as stats
import time
from scipy.fft import fft, ifft, fft2, fftn




def state_fft(rho, phase, hbar, m, alpha, network_type = '1D'):
    N = len(rho)
    u_0 = np.sqrt(1/N)
    u = np.sqrt(rho) - u_0
    phase = phase * u_0 
    if network_type == '1D':
        u_k = fft(u)
        phase_k = fft(phase)
        L = N * alpha
        Nx = N
    elif network_type == '2D':
        Nx = int(np.sqrt(N))
        u = u.reshape(Nx, Nx)
        phase = phase.reshape(Nx, Nx)
        u_k = fft2(u)
        phase_k = fft2(phase)
        L = Nx * alpha
    elif network_type == '3D':
        Nx = round(N**(1/3))
        u = u.reshape(Nx, Nx, Nx)
        phase = phase.reshape(Nx, Nx, Nx)
        u_k = fftn(u)
        phase_k = fftn(phase)
        L = Nx * alpha


    a_k = (u_k + phase_k / 1j) / 2
    b_k = (u_k - phase_k / 1j) / 2
    n = np.hstack(( np.arange(0, int(N/2), 1), np.arange(-int(N/2), 0, 1) ))
    k = 2 * np.pi * n / L
    omega_k = hbar * k ** 2 / m / 2
    return u_k, phase_k, a_k, b_k, L, n, k, omega_k
