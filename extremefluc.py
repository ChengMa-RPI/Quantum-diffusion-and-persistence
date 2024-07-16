import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.path.insert(1, '/home/mac6/RPI/research/')

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from cycler import cycler
import matplotlib as mpl
import itertools
import seaborn as sns
import multiprocessing as mp

from collections import Counter
import scipy.stats as stats
import time
from collections import defaultdict
from matplotlib import patches 
from calculate_fft import state_fft
from scipy import special
from scipy.fft import fft, ifft
from matplotlib.legend_handler import HandlerTuple
from scipy.signal import argrelextrema



fs = 24
ticksize = 20
labelsize = 35
anno_size = 18
subtitlesize = 15
legendsize= 20
alpha = 0.8
lw = 3
marksize = 8

colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f']

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

class extremefluc():

    def __init__(self, quantum_or_not, network_type, N, d, seed, alpha, dt, initial_setup, distribution_params, seed_initial_condition_list, rho_or_phase, full_data_t, quantum_method='CN'):
        self.quantum_or_not = quantum_or_not
        if network_type in ['2D_disorder', '3D_disorder'] and d == 1:
            self.network_type = network_type[:2]
            self.d = 4
        else:
            self.network_type = network_type
            self.d = d
        self.N = N
        self.seed = seed
        self.alpha = alpha
        self.dt = dt
        self.initial_setup = initial_setup
        self.distribution_params = distribution_params
        self.seed_initial_condition_list = seed_initial_condition_list
        self.rho_or_phase = rho_or_phase
        self.full_data_t = full_data_t
        self.quantum_method = quantum_method

    def read_phi(self, seed_initial_condition, rho_or_phase=None, full_data_t=None):
        ### rho_or_phase and full_data_t are from class parameters or from direct inputs
        if rho_or_phase is None:
            rho_or_phase = self.rho_or_phase
        if full_data_t is None:
            full_data_t = self.full_data_t
        if self.quantum_or_not == 'SE':
            if self.quantum_method == 'CN':  # Crank-Nicolson method
                if rho_or_phase == 'rho':
                    des = '../data/quantum/state/' + self.network_type + '/' 
                elif rho_or_phase == 'phase':
                    des = '../data/quantum/phase/' + self.network_type + '/' 
            elif self.quantum_method == 'eigen':  # eigenvalue decomposition
                if rho_or_phase == 'rho':
                    des = '../data/quantum/state_eigen/' + self.network_type + '/'
                elif rho_or_phase == 'phase':
                    des = '../data/quantum/phase_eigen/' + self.network_type + '/'
        elif self.quantum_or_not == 'TB':  # tight binding, eigenvalue decomposition
            if rho_or_phase == 'rho':
                des = '../data/tightbinding/state_eigen/' + self.network_type + '/'
            elif rho_or_phase == 'phase':
                des = '../data/tightbinding/phase_eigen/' + self.network_type + '/'
        else:
            des = '../data/classical/state/' + self.network_type + '/'
        if self.full_data_t:
            save_file = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_seed_initial={seed_initial_condition}_full.npy'
        else:
            save_file = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_seed_initial={seed_initial_condition}.npy'
        data = np.load(save_file)
        t, state = data[:, 0], data[:, 1:]
        return t, state

    def extract_extreme(self, ):
        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(12, 10))
        simpleaxis(ax)
        for seed_initial_condition in self.seed_initial_condition_list:
            t, state = self.read_phi(seed_initial_condition)
            state_ave = np.mean(state, 1)
            deviation = state - state_ave 
            extreme_above = np.max(deviation, 1)
            extreme_below = np.min(deviation, 1)
            ax.plot(t, extreme_above, color='tab:red', linestyle='-', linewidth=1)
            ax.plot(t, extreme_below, color='tab:blue', linestyle='-', linewidth=1)
        if self.rho_or_phase == 'rho':
            ylabel = '$\\rho_{\\Delta}$'
        elif self.rho_or_phase == 'phase':
            ylabel = '$\\theta_{\\Delta}$'
        ax.set_xlabel('t',fontsize=labelsize*0.7)
        ax.set_ylabel(ylabel, fontsize=labelsize*0.7)
        ax.tick_params(axis='both', labelsize=labelsize*0.5)
        fig.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.40, bottom=0.13, top=0.95)
        filename = f'quantum={self.quantum_or_not}_network={self.network_type}_d={self.d}_initial={self.initial_setup}_{self.distribution_params}_N={self.N}_{self.rho_or_phase}_extreme_t.png'
        save_des = '../transfer_figure/' + filename
        plt.savefig(save_des, format='png')
        plt.close()




if __name__ == '__main__':
    quantum_or_not = 'SE'
    initial_setup = 'u_normal_random'
    distribution_params = [0.2, 0]
    distribution_params = [0, 0.2]
    seed_initial_condition_list = np.arange(10)
    rho_or_phase = 'phase'
    rho_or_phase = 'rho'
    quantum_method = 'CN'
    full_data_t = True
    network_type = '1D'
    N = 10000
    d = 4
    seed = 0
    alpha = 1
    dt = 1
    ef = extremefluc(quantum_or_not, network_type, N, d, seed, alpha, dt, initial_setup, distribution_params, seed_initial_condition_list, rho_or_phase, full_data_t)
    ef.extract_extreme()
