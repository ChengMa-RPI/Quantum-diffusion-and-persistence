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
from scipy.signal import find_peaks

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
        if len(params) == 3 or len(params) == 4:
            if len(params) == 4:
                rho_start, rho_end, phase_start, phase_end = params
            elif len(params) == 3:
                rho_df, phase_start, phase_end = params
            if phase_start == phase_end:
                phase_title = '$\\theta = C$'
            else:
                phase_title = '$\\theta \sim $' + f'({phase_start } $\\pi$, {phase_end } $\\pi$)'
        elif len(params) == 5:
            rho_std, phase_x_ratio, phase_y_ratio, phase_center, phase_bound = params
            phase_title = '$\\theta_{max}=$' + f'${phase_center} \\pi$, ' +  '$\\theta_{\\delta}=$' + f'{phase_x_ratio}' 

    else:
        rho_start, rho_end = params

    if len(params) == 3:
        rho_title = f'$df = {rho_df}$'
    elif len(params) == 4:
        
        if rho_start == rho_end:
            rho_title = '$\\rho = C$'
        else:
            rho_title = '$\\rho \sim $' + f'({rho_start * 2}/N, {rho_end * 2}/N)'
    elif len(params) == 5:
        if rho_std == 0:
            rho_title = '$\\rho = C$'

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
    def __init__(self, quantum_or_not, network_type, N, d, seed, alpha, dt, initial_setup, distribution_params, seed_initial_condition_list, full_data_t, ipr_normalize, q_exp=2, quantum_method='CN'):
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
        self.ipr_normalize = ipr_normalize
        self.q_exp = q_exp
        self.full_data_t = full_data_t
        self.quantum_method = quantum_method

    def read_phi(self, seed_initial_condition):
        if self.quantum_or_not == 'SE':
            if self.quantum_method == 'CN':  # Crank-Nicolson method
                des = '../data/quantum/state/' + self.network_type + '/' 
            elif self.quantum_method == 'eigen':
                des = '../data/quantum/state_eigen/' + self.network_type + '/' 
        elif self.quantum_or_not == 'TB':
            des = '../data/tightbinding/state_eigen/' + self.network_type + '/' 

        else:
            des = '../data/classical/state/' + self.network_type + '/' 
        if self.full_data_t:
            save_file = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_seed_initial={seed_initial_condition}_full.npy'
        else:
            save_file = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_seed_initial={seed_initial_condition}.npy'
        data = np.load(save_file)
        t, state = data[:, 0], data[:, 1:]
        return t, state
    
    def cal_ipr_t(self, seed_list):
        ipr_list = []
        for seed_initial_condition in self.seed_initial_condition_list:
            for seed in seed_list:
                self.seed = seed
                t, state = self.read_phi(seed_initial_condition)
                N_actual = len(state[0])
                if self.ipr_normalize:
                    ipr = np.sum(state ** self.q_exp, 1) * N_actual **(self.q_exp- 1 ) 
                else:
                    ipr = np.sum(state ** self.q_exp, 1) 
                ipr_list.append(ipr)
        ipr_max = np.max(ipr_list, axis=1)
        ipr_min = np.min(ipr_list, axis=1)
        ipr_half = (ipr_max + ipr_min) / 2
        peak_time = []
        ipr_peak = []
        lifetime = []
        for i in range(len(ipr_list)):
            peaks = find_peaks(ipr_list[i], height=ipr_max[i] * 0.7)
            if len(peaks[0]):
                peak_i = peaks[0][0]
            else:
                peak_i = len(ipr_list[i]) - 1
            l = r = peak_i
            while l >= 0 and ipr_list[i][l] > ipr_half[i]:
                l -= 1
            while r < len(ipr_list[i]) and ipr_list[i][r] > ipr_half[i]:
                r += 1
            lifetime.append(t[r] - t[l])
            peak_time.append(t[peak_i])
            ipr_peak.append(ipr_list[i][peak_i])
        return t, ipr_list, ipr_peak, peak_time, lifetime

    def plot_ipr_collections(self, N_list, distribution_params_list, seed_list, ipr_normalize, plot_ipr_list, plot_scale, plot_t_limit):
        dict_ipr_list = dict()
        dict_ipr_peak = dict()
        dict_peak_time = dict()
        dict_lifetime = dict()
        dict_t = dict()
        for distribution_params in distribution_params_list:
            self.distribution_params = distribution_params
            for N in N_list:
                self.N = N
                t, ipr_list, ipr_peak, peak_time, lifetime = self.cal_ipr_t(seed_list)
                dict_t[(tuple(distribution_params), N)] = t
                dict_ipr_list[(tuple(distribution_params), N)] = ipr_list
                dict_ipr_peak[(tuple(distribution_params), N)] = ipr_peak
                dict_peak_time[(tuple(distribution_params), N)] = peak_time
                dict_lifetime[(tuple(distribution_params), N)] = lifetime
        for plot_ipr in plot_ipr_list:
            if plot_ipr == 'ipr_t':
                dict_data = dict_ipr_list
            elif plot_ipr == 'ipr_peak':
                dict_data = dict_ipr_peak
            elif plot_ipr == 'peak_time':
                dict_data = dict_peak_time
            elif plot_ipr == 'lifetime':
                dict_data = dict_lifetime
            else:
                print('no data available')
            self.plot_ipr_one_type(dict_t, dict_data, N_list, distribution_params_list, plot_ipr, plot_scale, plot_t_limit)

    def plot_ipr_one_type(self, dict_t, dict_data, N_list, distribution_params_list, plot_ipr, plot_scale, plot_t_limit):
        if len(N_list) == 1:
            show_all_curve = True
        else:
            show_all_curve = False
        if len(distribution_params_list) == 1:
            rows = 1
            cols = 1
            fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(12, 10))
        else:
            rows = 2
            cols = len(distribution_params_list) // rows
            fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 3.5 * rows))
        for i, distribution_params in enumerate(distribution_params_list):
            if self.initial_setup == 'full_local':
                title = 'fully localized'
            else:
                title = title_name(distribution_params, self.quantum_or_not)
            if len(distribution_params_list) == 1:
                ax = axes
            else:
                ax = axes[i // cols, i % cols]
            simpleaxis(ax)
            self.distribution_params = distribution_params
            ave_list = []
            for j, N in enumerate(N_list):
                self.N = N
                average = self.plot_ipr_ax(ax, dict_t[(tuple(distribution_params), N)], dict_data[(tuple(distribution_params), N)], color1[j], plot_ipr, plot_scale, plot_t_limit, show_all_curve)
                ave_list.append(average)
            ax.tick_params(axis='both', which='major', labelsize=13)
            ax.set_title(title, size=labelsize*0.5, y=0.94)
            if plot_ipr in ['ipr_peak', 'peak_time', 'lifetime']:
                ax.plot(N_list, ave_list, color='tab:grey', alpha=0.8)

        #ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.23, 0.49) ) 
        if plot_ipr == 'ipr_t':
            if len(distribution_params_list) == 1:
                ax.legend(fontsize=legendsize*0.8, frameon=False, loc=4, bbox_to_anchor=(1.1, 0.7) ) 
            else:
                ax.legend(fontsize=legendsize*0.6, frameon=False, loc=4, bbox_to_anchor=(1.42, 0.39) ) 

            

        fig.subplots_adjust(left=0.15, right=0.88, wspace=0.25, hspace=0.40, bottom=0.13, top=0.90)
        filename = f'quantum={self.quantum_or_not}_network={self.network_type}_d={self.d}_initial={self.initial_setup}_N_list={N_list}_ipr_{self.q_exp}_{ipr_normalize}_{plot_ipr}_{plot_scale}.png'
        #filename = f'quantum={self.quantum_or_not}_network={self.network_type}_initial={self.initial_setup}_N_list={N_list}_ipr_{self.q_exp}_zoomin.png'
        save_des = '../transfer_figure/' + filename
        plt.savefig(save_des, format='png')
        plt.close()

    def plot_ipr_ax(self, ax, t, y, color, plot_ipr, plot_scale, plot_t_limit, show_all_curve):
        if plot_ipr == 'ipr_t':
            if plot_t_limit:
                plot_index = np.where(t <= plot_t_limit)[0]
                t = t[plot_index]
                y = np.array(y)[:, plot_index]
            label = f'$N={self.N}$'
            if plot_scale == 'normal':
                if show_all_curve:
                    ax.plot(t, np.vstack((y)).transpose(), '--', color=color, linewidth=1, alpha=0.6)
                #ax.plot(t, y[0], '--', color=color, linewidth=1, alpha=0.6)
                ax.plot(t, np.mean(np.vstack((y)), 0), color=color, linewidth = 2.5, alpha=0.8, label=label)
            elif plot_scale == 'logx':
                if show_all_curve:
                    ax.semilogx(t[1:], np.vstack((y)).transpose()[1:], '--', color=color, linewidth=1, alpha=0.6)
                ax.semilogx(t[1:], np.mean(np.vstack((y)), 0)[1:], color=color, linewidth = 2.5, alpha=0.8, label=label)
            elif plot_scale == 'logy':
                if show_all_curve:
                    ax.semilogy(t, np.vstack((y)).transpose(), '--', color=color, linewidth=1, alpha=0.6)
                ax.semilogy(t, np.mean(np.vstack((y)), 0), color=color, linewidth = 2.5, alpha=0.8, label=label)
            elif plot_scale == 'loglog':
                if show_all_curve:
                    ax.loglog(t[1:], np.vstack((y)).transpose()[1:], '--', color=color, linewidth=1, alpha=0.6)
                ax.loglog(t[1:], np.mean(np.vstack((y)), 0)[1:], color=color, linewidth = 2.5, alpha=0.8, label=label)
            average = None
            xlabel = '$t$'
            if self.ipr_normalize:
                ylabel = 'IPR $( \\times N) $'
            else:
                ylabel = 'IPR'
        elif plot_ipr in ['ipr_peak', 'peak_time', 'lifetime']:
            ax.plot(np.ones(len(y)) * self.N, y, '.', color=color)
            xlabel = '$N$'
            average = np.mean(y)
            if plot_ipr == 'ipr_peak':
                ylabel = '$\mathrm{IPR_{max}}$'
            elif plot_ipr == 'peak_time':
                ylabel = '$\mathrm{t_{peak}}$'
            elif plot_ipr == 'lifetime':
                ylabel = 'lifetime'
        else:
            print('unavailable')
            return 
        ax.set_xlabel(xlabel,fontsize=labelsize*0.5)
        ax.set_ylabel(ylabel,fontsize=labelsize*0.5)
        return average






if __name__ == '__main__':
    initial_setup = 'uniform_random'
    quantum_or_not = False
    quantum_or_not = 'SE'
    quantum_or_not = 'TB'
    initial_setup = 'gaussian_wave'
    N = 1000
    d = 4
    seed = 0
    alpha = 1
    dt = 1
    seed_initial_condition_list = np.arange(0, 1, 1)
    distribution_params = [1, 1, -1, 1]
    q_exp = 2
    full_data_t = True


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

    "bowl-shaped phase"
    initial_setup = 'phase_bowl_region'
    network_type = '2D_disorder'
    d = 0.9
    rho_list = [[0]]
    phase_list = [[1, 1, 1, -1], [0.7, 0.7, 1, -1], [1, 1, 0.5, -0.5], [0.7, 0.7, 0.5, -0.5]]

    "fully local"
    initial_setup = 'full_local'
    rho_list = [[0]]
    phase_list = [[0, 0]]

    distribution_params_raw = [rho + phase for rho in rho_list for phase in phase_list]
    #distribution_params_raw = [rho for rho in rho_list]
    quantum_method = 'eigen'
    ipr_normalize = False
    pipr = plotIPR(quantum_or_not, network_type, N, d, seed, alpha, dt, initial_setup, distribution_params, seed_initial_condition_list, full_data_t, ipr_normalize, q_exp, quantum_method)

    distribution_params_list = []
    for i in distribution_params_raw:
        distribution_params_list.append( [round(j, 10) for j in i])
    N_list = [10000]
    N_list = [900, 1600, 2500, 3600, 4900, 6400, 8100, 10000]
    seed_list = np.arange(10)
    plot_ipr_list = ['ipr_t', 'ipr_peak', 'peak_time', 'lifetime']
    plot_ipr_list = ['ipr_t']
    plot_scale = 'logx'
    plot_scale = 'normal'
    plot_scale = 'loglog'
    plot_t_limit = None
    pipr.plot_ipr_collections(N_list, distribution_params_list, seed_list, ipr_normalize, plot_ipr_list, plot_scale, plot_t_limit)
