import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.path.insert(1, '/home/mac6/RPI/research/')

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd 
from cycler import cycler
import matplotlib as mpl
import itertools
import seaborn as sns
import multiprocessing as mp

from collections import Counter
from scipy.integrate import odeint
import scipy.stats as stats
import time
import matplotlib.image as mpimg
from collections import defaultdict
from matplotlib import patches 
import json

fs = 24
ticksize = 20
labelsize = 35
anno_size = 18
subtitlesize = 15
legendsize= 20
alpha = 0.8
lw = 3
marksize = 8


mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#fc8d62',  '#66c2a5', '#e78ac3','#a6d854',  '#8da0cb', '#ffd92f','#b3b3b3', '#e5c494', '#7fc97f', '#beaed4', '#ffff99' ])  * cycler(linestyle=['-']))
color1 = ['#fc8d62',  '#66c2a5', '#e78ac3', '#a6d854',  '#8da0cb', '#ffd92f','#b3b3b3', '#e5c494', '#7fc97f', '#beaed4', '#ffff99']
color2 = ['brown', 'orange', 'lightgreen', 'steelblue','slategrey', 'violet']

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def title_name(params, quantum_or_not, d=None):

    if quantum_or_not:
        if len(params) == 4:
            rho_start, rho_end, phase_start, phase_end = params
        elif len(params) == 3:
            rho_df, phase_start, phase_end = params
        if phase_start == phase_end:
            phase_title = '$\\theta = C$'
        else:
            phase_title = '$\\theta \sim $' + f'({phase_start } $\\pi$, {phase_end } $\\pi$)'

    else:
        rho_start, rho_end = params

    if len(params) == 3:
        rho_title = f'$df = {rho_df}$'
    else:

        if rho_start == rho_end:
            rho_title = '$\\rho = C$'
        else:
            rho_title = '$\\rho \sim $' + f'({rho_start * 2}/N, {rho_end * 2}/N)'

    if quantum_or_not:
        if d:
            return f'$\\phi={d}$\n' + phase_title
        else:
            return rho_title  + '\n' + phase_title
    else:
        if d:
            return f'$\\phi={d}$'
        else:
            return rho_title  



class plotIPR():
    def __init__(self, quantum_or_not, network_type, N, d, seed, alpha, dt, initial_setup, distribution_params, seed_initial_condition_list, q_exp=2):
        self.quantum_or_not = quantum_or_not
        self.network_type = network_type
        self.N = N
        self.d = d
        self.seed = seed
        self.alpha = alpha
        self.dt = dt
        self.initial_setup = initial_setup
        self.distribution_params = distribution_params
        self.seed_initial_condition_list = seed_initial_condition_list
        self.q_exp = q_exp

    def read_phi(self, seed_initial_condition):
        if self.quantum_or_not:
            des = '../data/quantum/state/' + self.network_type + '/' 
        else:
            des = '../data/classical/state/' + self.network_type + '/' 
        save_file = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_seed_initial={seed_initial_condition}.npy'
        data = np.load(save_file)
        t, state = data[:, 0], data[:, 1:]
        return t, state
    
    def plot_ipr_t(self, ax, label):
        ipr_list = []
        for seed_initial_condition in self.seed_initial_condition_list:
            t, state = self.read_phi(seed_initial_condition)
            N_actual = len(state[0])
            ipr = np.sum(state ** self.q_exp, 1) * N_actual **(self.q_exp- 1 ) 
            ipr = np.sum(state ** self.q_exp, 1) 
            ipr_list.append(ipr)
        #ax.semilogx(t, np.vstack((ipr_list)).transpose(), color='tab:red', linewidth=1, alpha=0.6)
        #ax.semilogx(t, np.mean(np.vstack((ipr_list)), 0), color='tab:blue', linewidth = 2.5, alpha=0.8)
        #ax.semilogx(t, np.mean(np.vstack((ipr_list)), 0), linewidth = 2.5, alpha=0.8, label=f'N={self.N}')
        ax.loglog(t, np.mean(np.vstack((ipr_list)), 0), linewidth = 2.5, alpha=0.8, label=label)

        #ax.set_yscale('symlog')
        #plt.locator_params(axis='x', nbins=4)
        return None


    def plot_ipr_initial_setup(self, N_list, distribution_params_list):
        rows = 4
        cols = len(distribution_params_list) // rows
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 3.5 * rows))
        for i, distribution_params in enumerate(distribution_params_list):
            title = title_name(distribution_params, self.quantum_or_not)
            #ax = axes[i]
            ax = axes[i // cols, i % cols]
            simpleaxis(ax)
            self.distribution_params = distribution_params
            for N in N_list:
                self.N = N
                self.plot_ipr_t(ax)
            
            ax.tick_params(axis='both', which='major', labelsize=13)
            ax.set_title(title, size=labelsize*0.5, y=0.92)

        #ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.23, 0.49) ) 

        ylabel = f'$IPR ( \\times N) $'
        xlabel = '$t$'
        ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.19, 0.29) ) 
        fig.text(x=0.02, y=0.5, horizontalalignment='center', s=ylabel, size=labelsize*0.6, rotation=90)
        fig.text(x=0.5, y=0.04, horizontalalignment='center', s=xlabel, size=labelsize*0.6)
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25, hspace=0.25, bottom=0.13, top=0.90)
        filename = f'quantum={self.quantum_or_not}_network={self.network_type}_initial={self.initial_setup}_N_list={N_list}_ipr_{self.q_exp}.png'
        #filename = f'quantum={self.quantum_or_not}_network={self.network_type}_initial={self.initial_setup}_N_list={N_list}_ipr_{self.q_exp}_zoomin.png'
        save_des = '../transfer_figure/' + filename
        plt.savefig(save_des, format='png')
        plt.close()


    def plot_disorder_ipr_initial_setup(self, N_list, d_list, distribution_params_list):
        rows = len(d_list)
        cols = len(distribution_params_list)
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 3.5 * rows))
        for i, d in enumerate(d_list):
            for j, distribution_params in enumerate(distribution_params_list):
                ax = axes[i, j]
                simpleaxis(ax)
                self.distribution_params = distribution_params
                self.d = d
                title = title_name(distribution_params, self.quantum_or_not, self.d)
                if d == 1:
                    self.network_type = '2D' 
                    self.d = 4
                for N in N_list:
                    self.N = N
                    label = f'N={self.N}'
                    self.plot_ipr_t(ax, label)
                ax.tick_params(axis='both', which='major', labelsize=13)
                ax.set_title(title, size=labelsize*0.5, y=0.92)

        #ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.23, 0.09) ) 

        ylabel = f'$IPR ( \\times N) $'
        xlabel = '$t$'
        ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.19, 0.29) ) 
        fig.text(x=0.02, y=0.5, horizontalalignment='center', s=ylabel, size=labelsize*0.6, rotation=90)
        fig.text(x=0.5, y=0.04, horizontalalignment='center', s=xlabel, size=labelsize*0.6)
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25, hspace=0.25, bottom=0.13, top=0.90)
        filename = f'quantum={self.quantum_or_not}_network=2D_disorder_initial={self.initial_setup}_N_list={N_list}_ipr_{self.q_exp}.png'
        save_des = '../transfer_figure/' + filename
        plt.savefig(save_des, format='png')
        plt.close()



    def plot_disorder_ipr_phi(self, d_list):
        rows = 1
        cols = 1
        fig, ax = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(16 * cols, 12 * rows))
        simpleaxis(ax)
        for i, d in enumerate(d_list):
            self.d = d
            if d == 1:
                self.network_type = self.network_type[:2]  
                self.d = 4
                label = f'$\\phi=1$'

            else:
                self.network_type = self.network_type[:2] + '_disorder'
                self.d = d
                label = f'$\\phi={self.d}$'
            self.plot_ipr_t(ax, label)
            ax.tick_params(axis='both', which='major', labelsize=labelsize*0.7)

        self.network_type = self.network_type + '_disorder'
        ylabel = f'$IPR$'
        xlabel = '$t$'
        ax.legend(fontsize=legendsize*0.9, frameon=False, loc=4, bbox_to_anchor=(0.19, 0.01) ) 
        fig.text(x=0.03, y=0.5, horizontalalignment='center', s=ylabel, size=labelsize*0.8, rotation=90)
        fig.text(x=0.5, y=0.04, horizontalalignment='center', s=xlabel, size=labelsize*0.8)
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25, hspace=0.25, bottom=0.13, top=0.90)
        filename = f'quantum={self.quantum_or_not}_network={self.network_type}_initial={self.initial_setup}_N={self.N}_ipr_{self.q_exp}.png'
        save_des = '../transfer_figure/' + filename
        plt.savefig(save_des, format='png')
        plt.close()





if __name__ == '__main__':
    initial_setup = 'uniform_random'
    quantum_or_not = False
    quantum_or_not = True
    initial_setup = 'gaussian_wave'
    N = 1000
    d = 4
    seed = 0
    alpha = 1
    dt = 1
    seed_initial_condition_list = np.arange(0, 1, 1)
    distribution_params = [1, 1, -1, 1]
    q_exp = 2


    "uniform random for both rho and phase"
    initial_setup = 'uniform_random'
    network_type = '2D'
    rho_list = [[0, 1], [1/4, 3/4], [3/8, 5/8], [1, 1]]
    phase_list = [[-1, 1], [-1/2, 1/2], [-1/4, 1/4], [0, 0]]

    network_type = '2D_disorder'
    rho_list = [[1, 1]]
    phase_list = [[-1, 1], [-1/2, 1/2], [-1/4, 1/4], [-1/8, 1/8]]

    "chi2 for rho and uniform for phase"
    initial_setup = 'chi2_uniform'
    network_type = '1D'
    rho_list = [[1e-4], [1e-2], [1], [10]]
    phase_list = [[-1, 1], [-1/2, 1/2], [-1/4, 1/4], [0, 0]]

    distribution_params_raw = [rho + phase for rho in rho_list for phase in phase_list]
    #distribution_params_raw = [rho for rho in rho_list]

    pent = plotIPR(quantum_or_not, network_type, N, d, seed, alpha, dt, initial_setup, distribution_params, seed_initial_condition_list, q_exp)

    distribution_params_list = []
    for i in distribution_params_raw:
        distribution_params_list.append( [round(j, 5) for j in i])
    N_list = [900, 4900, 10000]
    N_list = [1000]
    #pent.plot_ipr_initial_setup(N_list, distribution_params_list)

    d_list = [0.55, 0.7, 0.9, 1]
    #pent.plot_disorder_ipr_initial_setup(N_list, d_list, distribution_params_list)
    pent.initial_setup = 'full_local'
    pent.distribution_params = [0, 0, 0]
    pent.network_type = '2D_disorder'
    pent.N = 10000
    d_list = [0.51, 0.55, 0.7, 0.9, 1]

    pent.network_type = '3D_disorder'
    pent.N = 8000
    d_list = [0.3, 0.5, 0.7, 0.9, 1]
    pent.plot_disorder_ipr_phi(d_list)
