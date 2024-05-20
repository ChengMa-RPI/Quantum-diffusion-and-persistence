from functools import wraps
import inspect

def initializer(func):
    """
    Automatically assigns the parameters.
    >>> class process:
    ...     @initializer
    ...     def __init__(self, cmd, reachable=False, user='root'):
    ...         pass
    >>> p = process('halt', True)
    >>> p.cmd, p.reachable, p.user
    ('halt', True, 'root')
    """
    names, varargs, keywords, defaults = inspect.getargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper

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
from mutual_framework import network_generate, betaspace
import scipy.stats as stats
import time
import matplotlib.image as mpimg
from collections import defaultdict
from matplotlib import patches 

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
colors = ['#fc8d62',  '#66c2a5', '#e78ac3', '#a6d854',  '#8da0cb', '#ffd92f','#b3b3b3', '#e5c494', '#7fc97f', '#beaed4', '#ffff99']
color2 = ['brown', 'orange', 'lightgreen', 'steelblue','slategrey', 'violet']

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def title_name(params, quantum_or_not, initial_setup='uniform_random', d=None):

    if quantum_or_not:
        if initial_setup == 'u_uniform_random' or initial_setup == 'u_normal_random':
            rho_std, phase_std = [round(i, 4) for i in params]
            rho_title = '$\\sigma_u = $' + f'{rho_std}' 
            phase_title = '$\\sigma_{\\vartheta} = $' + f'{phase_std}'
        elif initial_setup == 'uniform_random':
            rho_start, rho_end, phase_start, phase_end = [round(i, 4) for i in params]
            if phase_start == phase_end:
                phase_title = '$\\theta = C$'
            else:
                phase_title = '$\\theta \sim $' + f'({phase_start } $\\pi$, {phase_end } $\\pi$)'
        elif initial_setup == 'u_normal_phase_uniform_random':
            rho_std, phase_range = [round(i, 4) for i in params]
            rho_title = '$\\sigma_u = $' + f'{rho_std}' 
            if phase_range == 1:
                phase_title = '$\\vartheta  \\sim $' + f'[-$\pi$, +$\pi$]'
            else:
                phase_title = '$\\vartheta  \\sim $' + f'[-{phase_range} $\pi$, + {phase_range} $\pi$]'
        elif initial_setup == 'u_normal_random_cutoff':
            rho_std, rho_cutoff, phase_std, phase_cutoff = [round(i, 4) for i in params]
            rho_title = '$\\sigma_u = $' + f'{rho_std}' 
            phase_title = '$\\sigma_{\\vartheta} = $' + f'{phase_std}'


    else:
        rho_start, rho_end = params
    if initial_setup == 'uniform_random':
        if rho_start == rho_end:
            rho_title = '$\\rho = C$'
        else:
            rho_title = '$\\rho \sim $' + f'({rho_start * 2}/N, {rho_end * 2}/N)'

    if quantum_or_not:
        if d:
            return f'$\\phi={d}$' + ', ' + phase_title
        else:
            return rho_title  + ', ' + phase_title
    else:
        if d:
            return f'$\\phi={d}$' 
        else:
            return rho_title  


class Plot_Dpp():
    def __init__(self, quantum_or_not, network_type, m, N, d, seed, alpha, dt, initial_setup, distribution_params,  seed_initial_condition_list, reference_line, rho_or_phase):
        self.quantum_or_not = quantum_or_not
        self.network_type = network_type
        self.m = m
        self.N = N
        self.d = d
        self.seed = seed
        self.alpha = alpha
        self.dt = dt
        self.initial_setup = initial_setup
        self.distribution_params = distribution_params
        self.seed_initial_condition_list = seed_initial_condition_list
        self.reference_line = reference_line
        self.rho_or_phase = rho_or_phase


    def read_dpp(self):
        """TODO: Docstring for read_dpp.

        :arg1: TODO
        :returns: TODO

        """
        if self.quantum_or_not:
            if self.rho_or_phase == 'rho':
                des = '../data/quantum/persistence/' + self.network_type + '/' 
            else:
                des = '../data/quantum/persistence_phase/' + self.network_type + '/' 
        else:
            des = '../data/classical/persistence/' + self.network_type + '/' 
        PA = []
        PB = []
        for seed_initial_condition in self.seed_initial_condition_list:
            if self.m == m_e:
                filename = f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_reference={self.reference_line}_seed_initial={seed_initial_condition}.csv'
            else:
                filename = f'm={self.m}_N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_reference={self.reference_line}_seed_initial={seed_initial_condition}.csv'
            data = np.array(pd.read_csv(des + filename, header=None))
            #t, pa, pb = data[1:, 0], data[1:, 1], data[1:, 2]
            t, pa, pb = data[:, 0], data[:, 1], data[:, 2]
            pa = pa / pa[0]
            pb = pb / pb[0]

            PA.append(pa)
            PB.append(pb)
        PA = np.vstack(( PA ))
        PB = np.vstack(( PB ))
        PA_ave = np.mean(PA, 0)
        PB_ave = np.mean(PB, 0)
        return t, PA_ave, PB_ave, PA_ave + PB_ave
        

    def plot_dpp_t(self):
        t, PA_ave, PB_ave, P_ave = self.read_dpp()
        plt.semilogy(t, PA_ave)
        plt.semilogy(t, PB_ave)
        return None

    def plot_dpp_t_N_list(self, ax, N_list, pa_or_pb):
        for N in N_list:
            self.N = N
            t, PA_ave, PB_ave, P_ave = self.read_dpp()
            index = np.where(t < 500)[0]
            t = t[index]
            PA_ave = PA_ave[index]
            PB_ave = PB_ave[index]
            P_ave = P_ave[index]
            if pa_or_pb == 'pa':
                #ax.loglog(t, PA_ave, label=f'N={N}')
                ax.semilogy(t, PA_ave, label=f'N={N}')
            elif pa_or_pb == 'pb':
                #ax.loglog(t, PB_ave, label=f'N={N}')
                ax.semilogy(t, PB_ave, label=f'N={N}')
            else:
                #ax.loglog(t, P_ave, label=f'N={N}')
                ax.semilogy(t, P_ave, label=f'N={N}')

            #plt.semilogx(t, PA_ave, label=f'N={N}')

    def plot_dpp_scaling(self, N_list, theta=0.186, z=1.99):
        fig, ax = plt.subplots(1, 1)
        simpleaxis(ax)
        for N in N_list:
            self.N = N
            t, PA_ave, PB_ave, P_ave = self.read_dpp()
            L = np.sqrt(N)
            lzt = L ** (theta * z)
            lz = L ** z
            y =  lzt * P_ave
            x = t / lz
            ax.loglog(x, y, label=f'N={N}')
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.03, 0.15) ) 
        fig.text(x=0.5, y=0.01, horizontalalignment='center', s="$t/L^{z}$", size=labelsize*0.5)
        fig.text(x=0.05, y=0.45, horizontalalignment='center', s="$L^{z\\theta}P(t, L)$", size=labelsize*0.5, rotation=90)
        fig.subplots_adjust(left=0.2, right=0.95, wspace=0.25, hspace=0.25, bottom=0.2, top=0.95)


    def plot_pa_pb_reference(self, N_list, reference_lines):
        cols = len(reference_lines)
        rows = 2
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 3.5 * rows))
        for i in range(cols):
            self.reference_line = reference_lines[i]
            for j, pa_or_pb in enumerate(['pa', 'pb']):
                if cols == 1:
                    ax = axes[j]
                else:
                    ax = axes[j, i]
                simpleaxis(ax)
                self.plot_dpp_t_N_list(ax, N_list, pa_or_pb)
                ax.tick_params(axis='both', which='major', labelsize=13)
                #ax.set_ylim(7e-2, 1)
                if j == 0:
                    if self.reference_line == 'average':
                        title = '$r = \\frac{1}{N}$'
                    else:
                        title = f'$r = {self.reference_line}\\times$' + '$\\frac{1}{N}$'

                    ax.set_title(title, size=labelsize*0.5)
        #ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.23, 0.09) ) 
        ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.03, 0.09) ) 
        fig.text(x=0.5, y=0.01, horizontalalignment='center', s="$t$", size=labelsize*0.6)
        fig.text(x=0.05, y=0.75, horizontalalignment='center', s="$P_a$", size=labelsize*0.6, rotation=90)
        fig.text(x=0.05, y=0.3, horizontalalignment='center', s="$P_b$", size=labelsize*0.6, rotation=90)
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25, hspace=0.25, bottom=0.1, top=0.95)
        #save_des = '../manuscript/dimension_reduction_v3_072422/' + self.dynamics + '_' + self.network_type + f'_tau_c_m.png'
        #plt.savefig(save_des, format='png')
        #plt.close()


    def plot_dpp_t_initial_setup(self, ax, N, pa_or_pb, label, color='', scaling=False, d=1, seed_list=[0], axis_scale='semilogy'):
        PA_ave_list = []
        PB_ave_list = []
        P_ave_list = []
        for seed in seed_list:
            self.seed = seed
            t, PA_ave, PB_ave, P_ave = self.read_dpp()
            if scaling:
                t = t/ self.alpha **2
            index = np.where((t < 900) & (t>10))[0]
            if 'cutoff' in self.initial_setup:
                index = np.where(t < 9000)[0]
            else:
                index = np.where(t < 900)[0]
            t = t[index]
            xplot = t
            PA_ave = PA_ave[index]
            PB_ave = PB_ave[index]
            P_ave = P_ave[index]

            PA_ave_list.append(PA_ave)
            PB_ave_list.append(PB_ave)
            P_ave_list.append(P_ave)

        PA_ave_mean = np.mean(np.vstack((PA_ave_list)), 0)
        PB_ave_mean = np.mean(np.vstack((PB_ave_list)), 0)
        P_ave_mean = np.mean(np.vstack((P_ave_list)), 0)
 
        if axis_scale == 'semilogy':
            if pa_or_pb == 'pa':
                if len(seed_list) > 1:
                    for PA_ave in PA_ave_list:
                        ax.semilogy(xplot, PA_ave, '--', alpha=0.5, color=color)
                if color == '':
                    ax.semilogy(xplot, PA_ave_mean, label=label)
                else:
                    ax.semilogy(xplot, PA_ave_mean, label=label, color=color)

            elif pa_or_pb == 'pb':

                if len(seed_list) > 1:
                    for PB_ave in PB_ave_list:
                        ax.semilogy(xplot, PB_ave, '--', alpha=0.5, color=color)
                if color == '':
                    ax.semilogy(xplot, PB_ave_mean, label=label)
                else:
                    ax.semilogy(xplot, PB_ave_mean, label=label, color=color)
            else:
                ax.semilogy(xplot, P_ave_mean, label=label)
        else:
            if pa_or_pb == 'pa':
                if len(seed_list) > 1:
                    for PA_ave in PA_ave_list:
                        ax.loglog(xplot, PA_ave, '--', alpha=0.5, color=color)
                if color == '':
                    ax.loglog(xplot, PA_ave_mean, label=label)
                else:
                    ax.loglog(xplot, PA_ave_mean, label=label, color=color)

            elif pa_or_pb == 'pb':

                if len(seed_list) > 1:
                    for PB_ave in PB_ave_list:
                        ax.loglog(xplot, PB_ave, '--', alpha=0.5, color=color)
                if color == '':
                    ax.loglog(xplot, PB_ave_mean, label=label)
                else:
                    ax.loglog(xplot, PB_ave_mean, label=label, color=color)
            else:
                ax.loglog(xplot, P_ave_mean, label=label)

        return None



    def plot_dpp_t_disorder_loglog(self, ax, N, pa_or_pb, label, color='', dx_scaling=False, linear_fit=True, d=1, A_as_initial=False):
        """
        plot for 1D
        """
        if not linear_fit: 
            return
        t, PA_ave, PB_ave, P_ave = self.read_dpp()
        if 'cutoff' in self.initial_setup:
            index = np.where((t < 9000) & (t>10))[0]
            index = np.where(t < 9000)[0]
        else:
            index = np.where(t < 900)[0]

        t = t[index]
        PA_ave = PA_ave[index]
        PB_ave = PB_ave[index]
        P_ave = P_ave[index]
        if A_as_initial:
            if pa_or_pb == 'pa':
                A_coeff = PA_ave[0]
            elif pa_or_pb == 'pb':
                A_coeff = PB_ave[0]
            else:
                A_coeff = P_ave[0]
        else:
            A_coeff = 1

        if dx_scaling:
            xplot = t / self.alpha**2
        else:
            xplot = t 
        if pa_or_pb == 'pa':
            p_plot = np.log(A_coeff/ PA_ave)
        elif pa_or_pb == 'pb':
            p_plot = np.log(A_coeff/ PB_ave)
        else:
            p_plot = np.log(A_coeff/ P_ave)

        """loglog
        if color == '':
            ax.loglog(xplot, p_plot, '.',  markersize=3, alpha=0.7)
        else:
            ax.loglog(xplot, p_plot, '.', color=color, markersize=3, alpha=0.7)
        """

        if linear_fit:
            #index_fit = np.where((1.1 < p_plot) & (p_plot < 9))[0] # use P_ave to determine the index for linear fit
            index_fit = np.where((1.1 < p_plot) & (p_plot < 9))[0]
            (m, b), residue, _, _, _ = np.polyfit(np.log(xplot[index_fit]), np.log(p_plot[index_fit]) , 1, full=True)
            fitted_b = np.exp(b)
            label = f'$\\beta = {round(m, 2)}$' + label
            y_fitted = A_coeff * np.exp(-fitted_b * xplot[index_fit] ** m)
            if color == '':
                ax.semilogy(xplot** m, np.exp(-p_plot), '.', markersize=3, alpha=0.7)
                ax.semilogy(xplot[index_fit] ** m, y_fitted, label=label, alpha=0.9, linewidth=2.8  )
                #ax.loglog(xplot[index_fit], np.exp(b) * xplot[index_fit]**m, label=label, alpha=0.9, linewidth=2.8)
            else:
                ax.semilogy(xplot ** m, np.exp(-p_plot), '.', color=color, markersize=3, alpha=0.7)
                ax.semilogy(xplot[index_fit] ** m, y_fitted, label=label, color=color, alpha=0.9, linewidth=2.8  )
                #ax.loglog(xplot[index_fit], np.exp(b) * xplot[index_fit]**m, label=label, color=color, alpha=0.9, linewidth=2.8)

    def plot_dpp_t_loglog_dxscaling(self, ax, N, pa_or_pb, label, color='', show_scaling=False, dx_scaling=False, linear_fit=True, d=1, A_as_initial=False):
        if not linear_fit: 
            return
        t, PA_ave, PB_ave, P_ave = self.read_dpp()
        index = np.where((t < 900) & (t>10))[0]
        index = np.where(t < 900)[0]
        t = t[index]
        PA_ave = PA_ave[index]
        PB_ave = PB_ave[index]
        P_ave = P_ave[index]
        if A_as_initial:
            if pa_or_pb == 'pa':
                A_coeff = PA_ave[0]
            elif pa_or_pb == 'pb':
                A_coeff = PB_ave[0]
            else:
                A_coeff = P_ave[0]
        else:
            A_coeff = 1

        if dx_scaling:
            xplot = t / self.alpha**2
        else:
            xplot = t 
        if pa_or_pb == 'pa':
            p_plot = np.log(A_coeff/ PA_ave)
        elif pa_or_pb == 'pb':
            p_plot = np.log(A_coeff/ PB_ave)
        else:
            p_plot = np.log(A_coeff/ P_ave)

        """loglog
        if color == '':
            ax.loglog(xplot, p_plot, '.',  markersize=3, alpha=0.7)
        else:
            ax.loglog(xplot, p_plot, '.', color=color, markersize=3, alpha=0.7)
        """

        if linear_fit:
            #index_fit = np.where((1.1 < p_plot) & (p_plot < 9))[0] # use P_ave to determine the index for linear fit
            index_fit = np.where((1.1 < p_plot) & (p_plot < 9))[0]
            (m, b), residue, _, _, _ = np.polyfit(np.log(xplot[index_fit]), np.log(p_plot[index_fit]) , 1, full=True)
            if d == 1:
                b = -2.2
                m = 0.8
            elif d == 2:
                b = -2.4
                m = 1.1
            elif d == 3:
                b = -2.1
                m = 1.2
            fitted_b = np.exp(b)
            y_fitted = A_coeff * np.exp(-fitted_b * xplot[index_fit] ** m)
            if color == '':
                ax.semilogy(xplot** m, np.exp(-p_plot), '.', markersize=3, alpha=0.7)
                ax.semilogy(xplot[index_fit] ** m, y_fitted, label=label, alpha=0.9, linewidth=2.8  )
                #ax.loglog(xplot[index_fit], np.exp(b) * xplot[index_fit]**m, label=label, alpha=0.9, linewidth=2.8)
            else:
                ax.semilogy(xplot[::3] ** m, np.exp(-p_plot)[::3], '.', color=color, label=label, markersize=3, alpha=0.6)
                if show_scaling:
                    ax.semilogy(xplot[index_fit] ** m, y_fitted, color='tab:grey', alpha=0.9, linewidth=3.  )
                #ax.loglog(xplot[index_fit], np.exp(b) * xplot[index_fit]**m, label=label, color=color, alpha=0.9, linewidth=2.8)



    def plot_pa_pb_initial_setup(self, N, distribution_params_list):
        cols = 2
        rows = 1
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 3.5 * rows))
        for j, pa_or_pb in enumerate(['pa', 'pb']):
            ax = axes[j]
            simpleaxis(ax)
            for distribution_params in distribution_params_list:
                self.distribution_params = distribution_params
                label = title_name(distribution_params, self.quantum_or_not, self.initial_setup)
                if len(distribution_params_list) > 10:
                    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
                    phase_list = set([tuple(i[2:]) for i in distribution_params_list])
                    color_index = [i for i, phase in enumerate(phase_list) if distribution_params[2:] == list(phase) ][0]
                    color = colors[color_index]
                    label = label[label.rfind('$\\vartheta'):]
                else:
                    color = ''
                if not distribution_params == [0, 0]:
                    self.plot_dpp_t_initial_setup(ax, N, pa_or_pb, label, color)
                ax.tick_params(axis='both', which='major', labelsize=13)
                ax.set_xlim(-5, 100)
                ax.set_ylim(8e-5, 1.1)

        ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(2.19, 0.53) ) 
        fig.text(x=0.02, y=0.5, horizontalalignment='center', s="$P_a$", size=labelsize*0.6, rotation=90)
        fig.text(x=0.35, y=0.5, horizontalalignment='center', s="$P_b$", size=labelsize*0.6, rotation=90)
        #fig.text(x=0.37, y=0.01, horizontalalignment='center', s="$t / (\\Delta x) ^2$", size=labelsize*0.6)
        fig.text(x=0.37, y=0.01, horizontalalignment='center', s="$t$", size=labelsize*0.6)
        fig.subplots_adjust(left=0.1, right=0.69, wspace=0.25, hspace=0.25, bottom=0.2, top=0.95)
        if self.initial_setup == 'uniform_random':
            if len(distribution_params_list) > 10:
                save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_initial_setup={self.initial_setup}_{self.rho_or_phase}_dpp_N={self.N}_all.png'
            elif distribution_params_list[0][:2] == distribution_params_list[1][:2]:
                save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_initial_setup={self.initial_setup}_{self.rho_or_phase}_dpp_N={self.N}_rho={distribution_params[:2]}.png'

            elif distribution_params_list[0][2:] == distribution_params_list[1][2:]:
                save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_initial_setup={self.initial_setup}_{self.rho_or_phase}_dpp_N={self.N}_phase={distribution_params[2:]}.png'

        elif self.initial_setup == 'u_uniform_random' or self.initial_setup == 'u_normal_random':
            if len(distribution_params_list) > 10:
                save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_initial_setup={self.initial_setup}_{self.rho_or_phase}_dpp_N={self.N}_all.png'
            elif distribution_params_list[0][:1] == distribution_params_list[1][:1]:
                save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_initial_setup={self.initial_setup}_{self.rho_or_phase}_dpp_N={self.N}_rho={distribution_params[:1]}.png'

            elif distribution_params_list[0][1:] == distribution_params_list[1][1:]:
                save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_initial_setup={self.initial_setup}_{self.rho_or_phase}_dpp_N={self.N}_phase={distribution_params[1:]}.png'


        plt.savefig(save_des, format='png')
        plt.close()
        return 


    def plot_pa_pb_initial_setup_collections(self, N, params_rho, params_phase, fixed, axis_scale):
        if fixed == 'phase':
            params_fixed = params_phase
            params_change = params_rho
        else:
            params_fixed = params_rho
            params_change = params_phase
        cols = len(params_fixed)
        rows = 2
        letters = 'abcdefghijklmn'
        if len(params_fixed) == 1:
            fig, axes = plt.subplots(rows, cols, sharex=False, sharey=False, figsize=(8 * cols, 3.5 * rows))
        else:
            fig, axes = plt.subplots(rows, cols, sharex=False, sharey=False, figsize=(4 * cols, 3.5 * rows))

        for i, param_fixed in enumerate(params_fixed):
            for j, pa_or_pb in enumerate(['pa', 'pb']):
                if len(params_fixed) == 1:
                    ax = axes[j]
                else:
                    ax = axes[j, i]
                ax.annotate(f'({letters[j*len(params_fixed) + i]})', xy=(0, 0), xytext=(-0.05, 1.02), xycoords='axes fraction', size=22)
                simpleaxis(ax)
                for  param_change in params_change:
                    if 'cutoff' in self.initial_setup:
                        if fixed == 'phase':
                            distribution_params = param_change + param_fixed
                        else:
                            distribution_params = param_fixed + param_change
                    else:
                        if fixed == 'phase':
                            distribution_params = [param_change, param_fixed]
                        else:
                            distribution_params = [param_fixed, param_change]

                    self.distribution_params = distribution_params
                
                    label = title_name(distribution_params, self.quantum_or_not, self.initial_setup)
                    separation_index = label.rfind(', $\\')
                    rho_label = label[:separation_index]
                    phase_label = label[separation_index+1:]
                    color = ''
                    if fixed == 'phase':
                        change_label = rho_label 
                        title = phase_label 
                    else:
                        change_label = phase_label
                        title = rho_label 
                    if not distribution_params == [0, 0]:
                        self.plot_dpp_t_initial_setup(ax, N, pa_or_pb, change_label, color, axis_scale=axis_scale)
                        
                #if j == 0:
                    #ax.set_title(title, size=18)
                    pass
                ax.set_title(title, size=20)
                ax.set_xlabel('$t$', size=labelsize*0.7)
                if self.rho_or_phase == 'rho':
                    if pa_or_pb == 'pa':
                        ylabel = '$P_u^a(t)$'
                    else:
                        ylabel = '$P_u^b(t)$'
                else:
                    if pa_or_pb == 'pa':
                        ylabel = '$P_{\\vartheta}^a(t)$'
                    else:
                        ylabel = '$P_{\\vartheta}^b(t)$'

                ax.set_ylabel(ylabel, size=labelsize*0.7)
                ax.tick_params(axis='both', which='major', labelsize=14)
                if 'cutoff' in self.initial_setup:
                    if self.network_type == '1D':
                        ax.set_xlim(-5, 5000)
                    elif self.network_type == '2D':
                        ax.set_xlim(-5, 1000)
                    elif self.network_type == '3D':
                        ax.set_xlim(-5, 500)
                else:
                    if self.network_type == '1D':
                        ax.set_xlim(-5, 500)
                    else:
                        if self.alpha == 1:
                            ax.set_xlim(-5, 100)
                        elif self.alpha == 0.2:
                            ax.set_xlim(-1, 10)
                if self.network_type == '3D' and 'cutoff' in self.initial_setup:
                    ax.set_ylim(5e-4, 1.1)
                else:
                    ax.set_ylim(8e-5, 1.1)

        if len(params_fixed) == 1:
            legend_loc = (0.85, 0.33)
            fig.subplots_adjust(left=0.19, right=0.93, wspace=0.25, hspace=0.55, bottom=0.1, top=0.95)
        else:
            legend_loc = (1.29, 0.33)
            fig.subplots_adjust(left=0.1, right=0.93, wspace=0.25, hspace=0.55, bottom=0.1, top=0.95)

        ax.legend(fontsize=legendsize*0.75, frameon=False, loc=4, bbox_to_anchor=legend_loc ) 
        #fig.text(x=0.02, y=0.7, horizontalalignment='center', s="$P_a$", size=labelsize*0.6, rotation=90)
        #fig.text(x=0.02, y=0.25, horizontalalignment='center', s="$P_b$", size=labelsize*0.6, rotation=90)
        #fig.text(x=0.37, y=0.01, horizontalalignment='center', s="$t / (\\Delta x) ^2$", size=labelsize*0.6)
        #fig.text(x=0.47, y=0.01, horizontalalignment='center', s="$t$", size=labelsize*0.6)
        if axis_scale == 'loglog':
            save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_initial_setup={self.initial_setup}_{self.rho_or_phase}_dpp_N={self.N}_{fixed}_fixed_{axis_scale}.png'
        else:
            save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_initial_setup={self.initial_setup}_{self.rho_or_phase}_dpp_N={self.N}_{fixed}_fixed.png'


        plt.savefig(save_des, format='png')
        plt.close()
        return 


    # deprecatd
    def plot_pa_pb_disorder(self, N, d_list, distribution_params_list):
        cols = 2
        rows = 1
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 3.5 * rows))
        for j, pa_or_pb in enumerate(['pa', 'pb']):
            ax = axes[j]
            simpleaxis(ax)
            for d, distribution_params in zip(d_list, distribution_params_list):
                self.d = d
                self.distribution_params = distribution_params
                label_all = title_name(distribution_params, self.quantum_or_not, d)
                sep_index = label_all.rfind('$\\theta')
                if d_list[0] == d_list[1]:
                    label = label_all[sep_index:]
                    title = label_all[:sep_index-2]
                else:
                    label = label_all[:sep_index-2]
                    title = label_all[sep_index:]
                if d == 1:
                    self.network_type = '2D'
                    self.d = 4
                else:
                    self.network_type = '2D_disorder'
                    self.d = d
                self.plot_dpp_t_initial_setup(ax, N, pa_or_pb, label)
                ax.tick_params(axis='both', which='major', labelsize=13)
                #ax.set_ylim(7e-2, 1)

        self.network_type = '2D_disorder'
        self.d = d
        ax.set_title(title, fontsize=16, x = 0.80, y = 0.90)
        ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(2.19, 0.3) ) 
        fig.text(x=0.02, y=0.5, horizontalalignment='center', s="$P_a$", size=labelsize*0.6, rotation=90)
        fig.text(x=0.35, y=0.5, horizontalalignment='center', s="$P_b$", size=labelsize*0.6, rotation=90)
        #fig.text(x=0.37, y=0.01, horizontalalignment='center', s="$t / (\\Delta x) ^2$", size=labelsize*0.6)
        fig.text(x=0.37, y=0.01, horizontalalignment='center', s="$t$", size=labelsize*0.6)
        fig.subplots_adjust(left=0.1, right=0.69, wspace=0.25, hspace=0.25, bottom=0.2, top=0.95)
        if d_list[0] == d_list[1]:
            save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_{self.rho_or_phase}_N={self.N}_seed={seed}_d={self.d}_dpp.png'
            save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_{self.rho_or_phase}_N={self.N}_seed={seed}_d={self.d}_dpp_loglog.png'
            save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_{self.rho_or_phase}_N={self.N}_seed={seed}_d={self.d}_dpp_loglog_zoomin.png'
        else:
            save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_{self.rho_or_phase}_N={self.N}_seed={seed}_phase={self.distribution_params[2:]}_dpp.png'
            save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_{self.rho_or_phase}_N={self.N}_seed={seed}_phase={self.distribution_params[2:]}_dpp_loglog.png'
            save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_{self.rho_or_phase}_N={self.N}_seed={seed}_phase={self.distribution_params[2:]}_dpp_loglog_zoomin.png'

        plt.savefig(save_des, format='png')
        plt.close()
        return 

    def plot_pa_pb_disorder_collections(self, N, d_list, params_rho, params_phase, pa_or_pb, linear_fit, seed_list=[0], loglog=False, A_as_initial=False):
        colors = ['#fc8d62',  '#66c2a5', '#e78ac3', '#a6d854',  '#8da0cb', '#ffd92f','#b3b3b3', '#e5c494', '#7fc97f', '#beaed4', '#ffff99']
        cols = len(params_rho)
        rows = len(params_phase)
        letters = 'abcdefghijklmnopqrst'
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 3.5 * rows), gridspec_kw={'width_ratios': [1, 1, 1, 0.5]})
        if self.rho_or_phase == 'rho':
            params_col = params_rho
            params_row = params_phase
        else:
            params_col = params_phase
            params_row = params_rho

        for j, param_row in enumerate(params_row):
            for i, param_col in enumerate(params_col):
                ax = axes[j, i]
                simpleaxis(ax)
                if self.rho_or_phase == 'rho':
                    distribution_params = [param_col, param_row]
                else:
                    distribution_params = [param_row, param_col]
                self.distribution_params = distribution_params
                for d_i, d in enumerate(d_list): 
                    if d == 1:
                        self.network_type = self.network_type[:2]
                        self.d = 4
                    else:
                        self.network_type = self.network_type[:2] + '_disorder'
                        self.d = d

                    title = title_name(distribution_params, self.quantum_or_not, self.initial_setup)
                    separation_index = title.rfind(',')
                    rho_label = title[:separation_index]
                    phase_label = title[separation_index+1:]
                    color = colors[d_i]
                    label = f'$\\phi={d}$'
                    label = ''
                    if (self.rho_or_phase == 'rho' and distribution_params[0] != 0) or (self.rho_or_phase == 'phase' and distribution_params[1] != 0):
                        ax.annotate(f'({letters[j*(cols-1) + i]})', xy=(0, 0), xytext=(-0.13, 1.04), xycoords='axes fraction', size=22)
                        ax.set_title(title, size=17, x=0.56)
                        if self.d == 4:
                            self.seed = 0 
                            if loglog:
                                self.plot_dpp_t_disorder_loglog(ax, N, pa_or_pb, label, color, linear_fit=linear_fit, d=1, A_as_initial=A_as_initial)
                                ax.legend(fontsize=legendsize*0.9, frameon=False, loc=4, bbox_to_anchor=(1.25, 0.03) ) 
                            else:
                                self.plot_dpp_t_initial_setup(ax, N, pa_or_pb, label, color, scaling=False, d=1)
                        else:
                            if loglog:
                                self.plot_dpp_t_disorder_loglog(ax, N, pa_or_pb, label, color, linear_fit=linear_fit, d=self.d, A_as_initial=A_as_initial)
                                ax.legend(fontsize=legendsize*0.9, frameon=False, loc=4, bbox_to_anchor=(1.25, 0.03) ) 
                            else:
                                self.plot_dpp_t_initial_setup(ax, N, pa_or_pb, label, color, scaling=False, d=self.d, seed_list=seed_list)

                        
                ax.tick_params(axis='both', which='major', labelsize=16)
                ax.set_xlim(-5, 100)
                ax.set_ylim(1e-4, 1.1)
                #ax.set_xlim(-5, 100)
                #ax.set_ylim(3e-1, 20)
        self.network_type = self.network_type[:2] + '_disorder'
        if distribution_params == [0, 0]:
            for d in d_list:
                if d == 4:
                    label = f'$\\phi=1$'
                else:
                    label = f'$\\phi={d}$'
                axes[-1, -1].plot([], [], label=label)
        
        for i in range(len(params_row)):
            axes[i, -1].legend(fontsize=legendsize*1.1, frameon=False, loc=4, bbox_to_anchor=(1.44, -0.06) ) 
            axes[i, -1].spines['bottom'].set_visible(False)
            axes[i, -1].spines['left'].set_visible(False)
            axes[i, -1].set_axis_off()

        if pa_or_pb == 'pa':
            if loglog:
                if A_as_initial:
                    ylabel = "$P_a/A$"
                else:
                    ylabel = '$P_a$'
                xlabel = '$t^c$'
            else:
                ylabel = '$P_a$'
                xlabel = '$t$'
        else:    
            if loglog:
                if A_as_initial:
                    ylabel = "$P_b/A$"
                else:
                    ylabel = '$P_b$'
                xlabel = '$t^c$'
            else:
                ylabel = '$P_b$'
                xlabel = '$t$'
        fig.text(x=0.03, y=0.5, horizontalalignment='center', s=ylabel, size=labelsize*0.8, rotation=90)
        fig.text(x=0.47, y=0.02, horizontalalignment='center', s=xlabel, size=labelsize*0.8)
        fig.subplots_adjust(left=0.1, right=0.93, wspace=0.25, hspace=0.25, bottom=0.1, top=0.95)
        if loglog:
            save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_d={d_list}_initial_setup={self.initial_setup}_{self.rho_or_phase}_dpp_N={self.N}_{pa_or_pb}_stretched_{A_as_initial}.png'

        else:
            save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_d={d_list}_initial_setup={self.initial_setup}_{self.rho_or_phase}_dpp_N={self.N}_{pa_or_pb}.png'

        plt.savefig(save_des, format='png')
        plt.close()
        return 


    def plot_pa_pb_stretched_exp(self, N, params_rho, params_phase, fixed):
        if fixed == 'phase':
            params_fixed = params_phase
            params_change = params_rho
        else:
            params_fixed = params_rho
            params_change = params_phase
        colors = ['#fc8d62',  '#66c2a5', '#e78ac3', '#a6d854',  '#8da0cb', '#ffd92f','#b3b3b3', '#e5c494', '#7fc97f', '#beaed4', '#ffff99']
        cols = len(params_fixed)
        rows = 2
        letters = 'abcdefghijklmn'
        fig, axes = plt.subplots(rows, cols, sharex=False, sharey=False, figsize=(4 * cols, 3.5 * rows))
        for i, param_fixed in enumerate(params_fixed):
            for j, pa_or_pb in enumerate(['pa', 'pb']):
                ax = axes[j, i]
                ax.annotate(f'({letters[j*len(params_fixed) + i]})', xy=(0, 0), xytext=(-0.05, 1.02), xycoords='axes fraction', size=22)
                simpleaxis(ax)
                for l, param_change in enumerate(params_change):
                    if 'cutoff' in self.initial_setup:
                        if fixed == 'phase':
                            distribution_params = param_change + param_fixed
                        else:
                            distribution_params = param_fixed + param_change
                    else:
                        if fixed == 'phase':
                            distribution_params = [param_change, param_fixed]
                        else:
                            distribution_params = [param_fixed, param_change]


                    self.distribution_params = distribution_params
                
                    label = title_name(distribution_params, self.quantum_or_not, self.initial_setup)
                    separation_index = label.rfind(',')
                    rho_label = label[:separation_index]
                    phase_label = label[separation_index+1:]
                    color = colors[l]
                    if fixed == 'phase':
                        change_label = rho_label 
                        title = phase_label 
                    else:
                        change_label = phase_label
                        title = rho_label 
                    if not distribution_params == [0, 0]:
                        if i < len(params_fixed) - 1 or j < 1:
                            legend_pos = (0.62, -0.06)
                            label = ''
                        else:
                            label = change_label
                            legend_pos = (1.39, 0.53)
                        self.plot_dpp_t_disorder_loglog(ax, N, pa_or_pb, label, color, linear_fit=True, d=1, A_as_initial=1)
                        ax.legend(fontsize=legendsize*0.75, frameon=False, loc=4, bbox_to_anchor=legend_pos ) 
                        
                #if j == 0:
                    #ax.set_title(title, size=18)
                    pass
                ax.set_title(title, size=20)
                ax.set_xlabel('$t^{\\beta}$', size=labelsize*0.7)
                if self.rho_or_phase == 'rho':
                    if pa_or_pb == 'pa':
                        ylabel = '$P_u^a(t)$'
                    else:
                        ylabel = '$P_u^b(t)$'
                else:
                    if pa_or_pb == 'pa':
                        ylabel = '$P_{\\vartheta}^a(t)$'
                    else:
                        ylabel = '$P_{\\vartheta}^b(t)$'

                ax.set_ylabel(ylabel, size=labelsize*0.7)
                ax.tick_params(axis='both', which='major', labelsize=14)
                if 'cutoff' in self.initial_setup:
                    if self.dt == 1:
                        if self.network_type == '1D':
                            ax.set_xlim(-5, 1000)
                        else:
                            ax.set_xlim(-5, 1000)
                    elif self.dt == 0.1:
                        ax.set_xlim(-5, 300)
                    

                else:
                    if self.network_type == '1D':
                        ax.set_xlim(-5, 100)
                    else:
                        ax.set_xlim(-5, 100)
                if self.dt == 1:
                    ax.set_ylim(8e-5, 1.1)
                elif self.dt == 0.1:
                    ax.set_ylim(8e-2, 1.1)

        #fig.text(x=0.02, y=0.7, horizontalalignment='center', s="$P_a$", size=labelsize*0.6, rotation=90)
        #fig.text(x=0.02, y=0.25, horizontalalignment='center', s="$P_b$", size=labelsize*0.6, rotation=90)
        #fig.text(x=0.37, y=0.01, horizontalalignment='center', s="$t / (\\Delta x) ^2$", size=labelsize*0.6)
        #fig.text(x=0.47, y=0.01, horizontalalignment='center', s="$t$", size=labelsize*0.6)
        fig.subplots_adjust(left=0.1, right=0.93, wspace=0.25, hspace=0.55, bottom=0.1, top=0.95)
        save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_initial_setup={self.initial_setup}_{self.rho_or_phase}_dpp_N={self.N}_{fixed}_fixed_stretched_exp.png'
        if self.dt != 1:
            save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_initial_setup={self.initial_setup}_{self.rho_or_phase}_dpp_N={self.N}_{fixed}_fixed_stretched_exp_dt={self.dt}.png'

        plt.savefig(save_des, format='png')
        plt.close()
        return 




    def plot_pa_pb_alpha_dt(self, N_list, alpha_list, dt_list, num_realization_list):
        cols = 2
        rows = 1
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 3.5 * rows))
        for j, pa_or_pb in enumerate(['pa', 'pb']):
            ax = axes[j]
            simpleaxis(ax)
            for N, alpha, dt, num_realization in zip(N_list, alpha_list, dt_list, num_realization_list):
                self.alpha = alpha
                self.dt = dt
                self.seed_initial_condition_list = np.arange(num_realization)
                self.N = N
                label = f'N={N}_$\\Delta t$={dt}'
                self.plot_dpp_t_initial_setup(ax, N, pa_or_pb, label)
                ax.tick_params(axis='both', which='major', labelsize=13)
                #ax.set_ylim(7e-2, 1)

        #ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.23, 0.09) ) 
        ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(2.09, 0.09) ) 
        fig.text(x=0.02, y=0.5, horizontalalignment='center', s="$P_a$", size=labelsize*0.6, rotation=90)
        fig.text(x=0.35, y=0.5, horizontalalignment='center', s="$P_b$", size=labelsize*0.6, rotation=90)
        #fig.text(x=0.37, y=0.01, horizontalalignment='center', s="$t / (\\Delta x) ^2$", size=labelsize*0.6)
        fig.text(x=0.37, y=0.01, horizontalalignment='center', s="$t$", size=labelsize*0.6)
        fig.subplots_adjust(left=0.1, right=0.69, wspace=0.25, hspace=0.25, bottom=0.2, top=0.95)
        save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_N={N_list}_L={N_list[0]*alpha_list[0]}_seed={seed}_d={self.d}_dpp_N_list_scale_dx2.png'
        save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_N={N_list}_L={N_list[0]*alpha_list[0]}_seed={seed}_d={self.d}_dpp_N_list.png'
        plt.savefig(save_des, format='png')
        plt.close()
        return None


    def plot_pa_pb_alpha_dt_scaling(self, N_list, alpha_list, dt_list, num_realization_list):
        cols = 2
        rows = 2
        fig, axes = plt.subplots(rows, cols, sharex=False, sharey=False, figsize=(5 * cols, 4 * rows))
        letters = 'abcdefghijk'
        for j, pa_or_pb in enumerate(['pa', 'pb']):
            for i, scaling in enumerate([False, True]):
                ax = axes[j, i]
                simpleaxis(ax)
                ax.annotate(f'({letters[j * 2 + i]})', xy=(0, 0), xytext=(-0.05, 1.02), xycoords='axes fraction', size=22)
                for N, alpha, dt, num_realization in zip(N_list, alpha_list, dt_list, num_realization_list):
                    self.alpha = alpha
                    self.dt = dt
                    self.seed_initial_condition_list = np.arange(num_realization)
                    self.N = N
                    label = f'$\\Delta x$={alpha}'
                    self.plot_dpp_t_initial_setup(ax, N, pa_or_pb, label, scaling=scaling)
                    ax.tick_params(axis='both', which='major', labelsize=13)
                    #ax.set_ylim(7e-5, 0.6)
                if scaling:
                    ax.set_xlabel("$t / (\\Delta x) ^2$", fontsize=labelsize*0.6)
                else:
                    ax.set_xlabel("$t$", fontsize=labelsize*0.6)

                if self.rho_or_phase == 'rho':
                    if pa_or_pb == 'pa':
                        ylabel = '$P_u^a(t)$'
                    else:
                        ylabel = '$P_u^b(t)$'
                else:
                    if pa_or_pb == 'pa':
                        ylabel = '$P_{\\vartheta}^a(t)$'
                    else:
                        ylabel = '$P_{\\vartheta}^b(t)$'
                ax.set_ylabel(ylabel, fontsize=labelsize*0.6)
                if self.quantum_or_not == False:
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                ax.set_ylim(8e-5, 1.05)

        ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.09, 0.09) ) 
        #ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.23, 0.09) ) 
        #fig.text(x=0.02, y=0.7, horizontalalignment='center', s="$P_a$", size=labelsize*0.6, rotation=90)
        #fig.text(x=0.02, y=0.25, horizontalalignment='center', s="$P_b$", size=labelsize*0.6, rotation=90)
        #fig.text(x=0.3, y=0.02, horizontalalignment='center', s="$t$", size=labelsize*0.6)
        #fig.text(x=0.7, y=0.02, horizontalalignment='center', s="$t / (\\Delta x) ^2$", size=labelsize*0.6)

        fig.subplots_adjust(left=0.13, right=0.95, wspace=0.25, hspace=0.25, bottom=0.1, top=0.95)
        save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_initial_setup={self.initial_setup}_params={self.distribution_params}_dx={alpha_list}_seed={seed}_d={self.d}_dpp_{self.rho_or_phase}_scale_dx2.png'
        plt.savefig(save_des, format='png')
        plt.close()
        return None

    def plot_pa_pb_alpha_dt_scaling_fitting(self, N_list, alpha_list, dt_list, num_realization_list):
        cols = 3
        rows = 2
        fig, axes = plt.subplots(rows, cols, sharex=False, sharey=False, figsize=(5 * cols, 4 * rows))
        letters = 'abcdefghijk'
        for j, pa_or_pb in enumerate(['pa', 'pb']):
            for i, scaling in enumerate([False, True, 'fitting']):
                ax = axes[j, i]
                simpleaxis(ax)
                ax.annotate(f'({letters[j * cols + i]})', xy=(0, 0), xytext=(-0.05, 1.02), xycoords='axes fraction', size=22)
                for (l, N), alpha, dt, num_realization in zip(enumerate(N_list), alpha_list, dt_list, num_realization_list):
                    if l == 0:
                        show_scaling = True
                    else:
                        show_scaling = False
                    color = colors[l]
                    self.alpha = alpha
                    self.dt = dt
                    self.seed_initial_condition_list = np.arange(num_realization)
                    self.N = N
                    label = f'$\\Delta x$={alpha}'
                    if scaling != 'fitting':
                        self.plot_dpp_t_initial_setup(ax, N, pa_or_pb, label, color=color, scaling=scaling)
                    else:
                        dim = 1 if self.network_type == '1D'  else (2 if self.network_type == '2D' else 3)
                        self.plot_dpp_t_loglog_dxscaling(ax, N, pa_or_pb, label, color=color, dx_scaling=True, show_scaling=show_scaling, linear_fit=True, d=dim, A_as_initial=False)
                    ax.tick_params(axis='both', which='major', labelsize=13)
                    #ax.set_ylim(7e-5, 0.6)
                if scaling == True:
                    ax.set_xlabel("$t / (\\Delta x) ^2$", fontsize=labelsize*0.6)
                elif scaling == False:
                    ax.set_xlabel("$t$", fontsize=labelsize*0.6)
                elif scaling == 'fitting':
                    ax.set_xlabel("$(t / (\\Delta x) ^2)^{\\beta}$", fontsize=labelsize*0.6)


                if self.rho_or_phase == 'rho':
                    if pa_or_pb == 'pa':
                        ylabel = '$P_u^a(t)$'
                    else:
                        ylabel = '$P_u^b(t)$'
                else:
                    if pa_or_pb == 'pa':
                        ylabel = '$P_{\\vartheta}^a(t)$'
                    else:
                        ylabel = '$P_{\\vartheta}^b(t)$'
                ax.set_ylabel(ylabel, fontsize=labelsize*0.6)
                if self.quantum_or_not == False:
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                ax.set_ylim(8e-5, 1.05)
                if self.network_type == '1D':
                    if scaling == 'fitting':
                        ax.set_xlim(-10, 200)
                elif self.network_type == '2D':
                    ax.set_xlim(-10, 100)
                elif self.network_type == '3D':
                    ax.set_xlim(-2, 80)
                if i == 1 and j == 1:
                    ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(0.95, 0.09) ) 
        #ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.23, 0.09) ) 
        #fig.text(x=0.02, y=0.7, horizontalalignment='center', s="$P_a$", size=labelsize*0.6, rotation=90)
        #fig.text(x=0.02, y=0.25, horizontalalignment='center', s="$P_b$", size=labelsize*0.6, rotation=90)
        #fig.text(x=0.3, y=0.02, horizontalalignment='center', s="$t$", size=labelsize*0.6)
        #fig.text(x=0.7, y=0.02, horizontalalignment='center', s="$t / (\\Delta x) ^2$", size=labelsize*0.6)

        fig.subplots_adjust(left=0.13, right=0.95, wspace=0.25, hspace=0.25, bottom=0.1, top=0.95)
        save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_initial_setup={self.initial_setup}_params={self.distribution_params}_dx={alpha_list}_seed={seed}_d={self.d}_dpp_{self.rho_or_phase}_scale_dx2_fitting.png'
        plt.savefig(save_des, format='png')
        plt.close()
        return None

    def plot_exponent_scaling(self, m_list, N_list, alpha_list, dt_list, num_realization_list):
        cols = 2
        rows = 2
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=False, figsize=(4 * cols, 3.5 * rows))
        m_a_list = []
        m_b_list = []
        b_a_list = []
        b_b_list = []
        for m, N, alpha, dt, num_realization in zip(m_list, N_list, alpha_list, dt_list, num_realization_list):
            self.m = m
            self.alpha = alpha
            self.dt = dt
            self.seed_initial_condition_list = np.arange(num_realization)
            self.N = N
            label = f'N={N}_$\\Delta t$={dt}'
            t, PA_ave, PB_ave, P_ave = self.read_dpp()
            """
            index = np.where(PA_ave < 1e-3)[0][0]
            (m_a, b_a), residue, _, _, _ = np.polyfit(t[:index], np.log(PA_ave[:index]) , 1, full=True)
            index = np.where(PB_ave < 1e-3)[0][0]
            (m_b, b_b), residue, _, _, _ = np.polyfit(t[:index], np.log(PB_ave[:index]) , 1, full=True)

            """
            index = np.where( (PA_ave < 0.2* 1e-1) & (PA_ave > 1e-3) )[0]
            (m_a, b_a), residue, _, _, _ = np.polyfit(t[index], np.log(PA_ave[index]) , 1, full=True)
            index = np.where( (PB_ave < 1e-1) & (PB_ave > 1e-3) )[0]
            (m_b, b_b), residue, _, _, _ = np.polyfit(t[index], np.log(PB_ave[index]) , 1, full=True)

            m_a_list.append(m_a)
            m_b_list.append(m_b)
            b_a_list.append(b_a)
            b_b_list.append(b_b)
        for (i, fit_data), ylabel in zip(enumerate( [m_a_list, m_b_list, b_a_list, b_b_list] ), ['$k_a$', '$k_b$', '$b_a$', '$b_b$']):
            ax = axes[i//cols, i%cols]
            simpleaxis(ax)
            x_scaling = hbar/ np.array(m_list) / np.array(alpha_list) ** 2
            ax.plot(x_scaling, fit_data, 'o')
            if ylabel in ['$k_a$', '$k_b$']:
                (fit_slope, fit_intercept), residue, _, _, _ = np.polyfit(x_scaling, fit_data, 1, full=True)
                func = f'$k = {round(fit_slope, 3)}$' + '$\\frac{\\hbar}{m \\Delta x^2}$'
                ax.plot(np.sort(x_scaling), np.sort(x_scaling) * fit_slope + fit_intercept, '--k', alpha=0.8, label = func)
                ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.05, 0.79) ) 

                
            ax.set_ylabel(ylabel, fontsize=17)
            ax.tick_params(axis='both', which='major', labelsize=13)
            #ax.set_ylim(7e-2, 1)

        #fig.text(x=0.02, y=0.5, horizontalalignment='center', s="$P_a$", size=labelsize*0.6, rotation=90)
        #fig.text(x=0.35, y=0.5, horizontalalignment='center', s="$P_b$", size=labelsize*0.6, rotation=90)
        fig.text(x=0.48, y=0.01, horizontalalignment='center', s="$\\hbar/ m \\Delta x^2$", size=labelsize*0.6)
        fig.subplots_adjust(left=0.15, right=0.95, wspace=0.25, hspace=0.25, bottom=0.2, top=0.95)
        save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_m={np.unique(m_list)}_N={N_list[0]}_d={self.d}_dpp_exponent_fit.png'
        save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_m={np.unique(m_list)}_N={N_list[0]}_d={self.d}_dpp_exponent_fit_earlycut.png'
        plt.savefig(save_des, format='png')
        plt.close()

    def plot_pa_pb_alpha_dt_linearfit(self, m_list, N_list, alpha_list, dt_list, num_realization_list):
        cols = 4
        rows = len(alpha_list) // cols
        fig, axes = plt.subplots(rows, cols, sharex=False, sharey=True, figsize=(4 * cols, 3.5 * rows))

        for (i, m), N, alpha, dt, num_realization in zip(enumerate(m_list), N_list, alpha_list, dt_list, num_realization_list):
            ax = axes[i // cols, i % cols]
            simpleaxis(ax)
            self.m = m
            self.N = N
            self.alpha = alpha
            self.dt = dt
            self.seed_initial_condition_list = np.arange(num_realization)
            t, PA_ave, PB_ave, P_ave = self.read_dpp()
            index = np.where(PA_ave < 1e-4)[0][0]
            (m_a, b_a), residue, _, _, _ = np.polyfit(t[:index], np.log(PA_ave[:index]) , 1, full=True)
            (m_b, b_b), residue, _, _, _ = np.polyfit(t[:index], np.log(PB_ave[:index]) , 1, full=True)
            ax.plot(t, np.exp(b_a + m_a * t), '--k', alpha=0.8)
            ax.semilogy(t, PA_ave, label='$P_a$')
            ax.plot(t, np.exp(b_b + m_b * t), '--k', alpha=0.8)
            ax.semilogy(t, PB_ave, label='$P_b$')
            title = f'$m={self.m},\\Delta x$={alpha},$\\Delta t$={dt}'
            ax.set_title(title, size=16)
            ax.tick_params(axis='both', which='major', labelsize=13)
            ax.set_ylim(1e-4, 1)
            ax.set_xlim(-t[index] * 0.03,  t[index] * 1.2)

        #ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.23, 0.09) ) 
        ax.legend(fontsize=legendsize*0.9, frameon=False, loc=4, bbox_to_anchor=(1.09, 0.64) ) 
        fig.text(x=0.05, y=0.5, horizontalalignment='center', s="$P$", size=labelsize*0.8, rotation=90)
        #fig.text(x=0.37, y=0.01, horizontalalignment='center', s="$t / (\\Delta x) ^2$", size=labelsize*0.6)
        fig.text(x=0.5, y=0.05, horizontalalignment='center', s="$t$", size=labelsize*0.8)
        fig.subplots_adjust(left=0.15, right=0.9, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
        save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_m={np.unique(m_list)}_N={N_list[0]}_d={self.d}_dpp_linearfit.png'
        plt.savefig(save_des, format='png')
        plt.close()

    def plot_pa_pb_exponent_disorder(self, rho_params_list, phase_params_list, d_list):
        cols = 2
        rows = 2
        fig, axes = plt.subplots(rows, cols, sharex=False, sharey=False, figsize=(8 * cols, 7 * rows))
        m_list = []
        gamma_list = []
        for axs in axes:
            for ax in axs:
                simpleaxis(ax)
                ax.tick_params(axis='both', which='major', labelsize=15)
                ax.set_xlabel('$\\phi$', size=labelsize *0.8) 

        for i, d in enumerate(d_list):
            if d == 1:
                self.d = 4
                self.network_type = self.network_type[:2] 
            else:
                self.d = d
                self.network_type = self.network_type[:2]+ '_disorder'
            for j_rho, rho_params in enumerate(rho_params_list):
                for j_phase, phase_params in enumerate(phase_params_list):
                    if rho_params == 0 and phase_params == 0:
                        continue
                    self.distribution_params = [rho_params, phase_params]

                    t, PA_ave, PB_ave, P_ave = self.read_dpp()
                    p_list = [PA_ave, PB_ave]
                    for row_i, p_i in enumerate(p_list):
                        plot_p = np.log(p_i[0] / p_i)
                        plot_p = np.log(1 / p_i)
                        index = np.where((plot_p < 9) & (plot_p >  2 ))[0]
                        (gamma, b), residue, _, _, _ = np.polyfit(np.log(t[index]), np.log(plot_p[index]) , 1, full=True)
                        if i == len(d_list)-1 and j_phase == len(phase_params_list)-1:
                            axes[row_i, 0].scatter(d, np.exp(b) * self.alpha ** 2/ (hbar/2/self.m), color=colors[j_rho], label=f'$\\sigma_u={rho_params}$' + '$r_0$')
                            axes[row_i, 1].scatter(d, gamma, color=colors[j_rho], label=f'$\\sigma_u={rho_params}$' + '$r_0$')
                        else:
                            axes[row_i, 0].scatter(d, np.exp(b) * self.alpha ** 2/ (hbar/2/self.m), color=colors[j_rho])
                            axes[row_i, 1].scatter(d, gamma, color=colors[j_rho])
                        

        axes[0, 0].set_ylabel('$k_a$', size=labelsize*0.7)
        axes[1, 0].set_ylabel('$k_b$', size=labelsize*0.7)
        axes[0, 1].set_ylabel('$\\gamma_a$', size=labelsize*0.7)
        axes[1, 1].set_ylabel('$\\gamma_b$', size=labelsize*0.7)
        self.network_type = self.network_type[:2] + '_disorder'
        #ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.23, 0.09) ) 
        axes[1, 1].legend(fontsize=legendsize*0.9, frameon=False, loc=4, bbox_to_anchor=(1.09, 0.04) ) 
        #fig.text(x=0.37, y=0.01, horizontalalignment='center', s="$t / (\\Delta x) ^2$", size=labelsize*0.6)
        fig.subplots_adjust(left=0.15, right=0.9, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
        save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_initial_setup={self.initial_setup}_rho_param={rho_params_list}_dpp_param_fitting.png'
        plt.savefig(save_des, format='png')
        plt.close()


    def plot_pa_pb_exponent_dimension(self, network_type_list, N_list, rho_params_list, phase_params):
        letters = 'abcdefghijklmn'
        cols = 4
        rows = 2
        fig, axes = plt.subplots(rows, cols, sharex=False, sharey=False, figsize=(4 * cols, 4 * rows))
        m_a_list_list = []
        m_b_list_list = []
        for (i, network_type), N in zip(enumerate(network_type_list), N_list):
            self.network_type = network_type
            self.N = N
            m_a_list = []
            m_b_list = []
            self.network_type = self.network_type[:2] 
            for j_rho, rho_params in enumerate(rho_params_list):
                self.distribution_params = [rho_params, phase_params]
                t, PA_ave, PB_ave, P_ave = self.read_dpp()
                p_list = [PA_ave, PB_ave]

                t_max = 400 / (i+1)**1.7
                if i < 2:
                    t_plot = np.arange(t[0], t_max, int(10 /(i+1)**2))
                    plot_index = [np.where(np.abs(t-t_i) < 10)[0][0] for t_i in t_plot]
                for row_i, p_i in enumerate(p_list):
                    ax = axes[row_i, i]
                    ax.tick_params(axis='both', which='major', labelsize=14)
                    if j_rho == 0:
                        simpleaxis(ax)
                    index = np.where((p_i > 1e-4) & (p_i <  4e-1 ))[0]
                    (k, b), residue, _, _, _ = np.polyfit(t[index], np.log(p_i[index]) , 1, full=True)
                    exponent = - k / (hbar / self.m / 2) * self.alpha**2
                    if row_i == 0:
                        m_a_list.append(exponent)
                        ylabel = '$P_a$'
                    else:
                        m_b_list.append(exponent)
                        ylabel = '$P_b$'
                    if i < 2:
                        ax.semilogy(t[plot_index], p_i[plot_index], '.', color=colors[j_rho], markersize=3, alpha=0.6)
                    else:
                        ax.semilogy(t, p_i, '.', color=colors[j_rho], markersize=3, alpha=0.6)
                    ax.semilogy(t[index], np.exp(b + k * t[index]), color=colors[j_rho], label='$\\sigma_{u}=$' + f'{rho_params}' )
                    ax.set_xlim(-5, t_max )
                    ax.set_ylim(1e-4, 0.7)
                    ax.set_xlabel('$t$', size=labelsize*0.6)
                    ax.set_ylabel(ylabel, size=labelsize*0.6)
                    if row_i == 0:
                        ax.set_title(f'$d={i+1}$', size=labelsize*0.5)
                    ax.annotate(f'({letters[row_i*4 + i]})', xy=(0, 0), xytext=(-0.05, 1.02), xycoords='axes fraction', size=labelsize*0.5)
                simpleaxis(ax)
            m_a_list_list.append(m_a_list)
            m_b_list_list.append(m_b_list)
                        
        simpleaxis(axes[0, -1])
        simpleaxis(axes[1, -1])
        axes[0, -1].plot([1, 2, 3], m_a_list_list, '*', markersize=10)
        axes[1, -1].plot([1, 2, 3], m_b_list_list, '*', markersize=10)
        axes[0, -1].set_xlabel('Dimension $d$', size=labelsize*0.55)
        axes[1, -1].set_xlabel('Dimension $d$', size=labelsize*0.55)
        axes[0, -1].set_title('Exponent fitting', size=labelsize*0.5)

        axes[0, -1].annotate(f'({letters[3]})', xy=(0, 0), xytext=(-0.05, 1.02), xycoords='axes fraction', size=labelsize*0.5)
        axes[1, -1].annotate(f'({letters[7]})', xy=(0, 0), xytext=(-0.05, 1.02), xycoords='axes fraction', size=labelsize*0.5)

        axes[0, -1].set_ylabel('$k_a$', size=labelsize*0.6)
        axes[1, -1].set_ylabel('$k_b$', size=labelsize*0.6)
        axes[0, -1].set_xticklabels(['1', '2', '3'])
        axes[0, -1].set_xticks([1, 2, 3])
        axes[1, -1].set_xticklabels(['1', '2', '3'])
        axes[1, -1].set_xticks([1, 2, 3])
        axes[0, -1].tick_params(axis='both', which='major', labelsize=14)
        axes[1, -1].tick_params(axis='both', which='major', labelsize=14)

        axes[-1, -2].legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.23, 0.59) ) 
        #fig.text(x=0.37, y=0.01, horizontalalignment='center', s="$t / (\\Delta x) ^2$", size=labelsize*0.6)
        fig.subplots_adjust(left=0.10, right=0.9, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
        save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network=regular_lattice_{self.rho_or_phase}_N_list={N_list}_initial_setup={self.initial_setup}_phase_param={phase_params}_dpp_param_fitting.png'
        plt.savefig(save_des, format='png')
        plt.close()



hbar = 0.6582
m_e = 5.68

m = 5.68

if __name__ == '__main__':
    quantum_or_not = True
    initial_setup = 'rho_uniform_phase_const_pi'
    initial_setup = 'rho_const_phase_uniform'
    initial_setup = 'sum_sin_inphase'
    initial_setup = 'sum_sin'
    distribution_params = [0.05, 0.05]
    network_type = '2D'
    d = 4
    network_type = '3D'
    network_type = '3D_disorder'
    N = 8000
    network_type = '2D_disorder'
    N = 10000
    network_type = '1D'
    d = 4
    #N = 1000
    #seed_initial_condition_list = np.arange(0, 100, 1)
    seed_initial_condition_list = np.arange(0, 10, 1)
    seed = 0
    alpha = 1
    reference_line = 'average'
    rho_or_phase = 'rho'
    reference_line = 0
    rho_or_phase = 'phase'

    dt = 0.2
    dt = 1
    #pdpp.plot_dpp_t()

    L_list = np.arange(10, 40, 10)
    N_list = np.power(L_list, 2)
    N_list = [100, 300, 500]

    #pdpp.plot_dpp_scaling(N_list)
    N_list = [1000]
    reference_lines = ['average']
    #pdpp.plot_pa_pb_reference(N_list, reference_lines)



    initial_setup = 'uniform_random'
    rho_list = [[0, 1], [1/4, 3/4], [3/8, 5/8], [1, 1]]
    phase_list = [[-1, 1], [-1/2, 1/2], [-1/4, 1/4], [0, 0]]

    rho_list = [[1, 1]]
    phase_list = [[-1, 1], [-1/2, 1/2], [-1/4, 1/4], [-1/8, 1/8] ]

    rho_list = [[7/16, 9/16], [15/32, 17/32], [31/64, 33/64], [63/128, 65/128]]
    phase_list = [[-1/16, 1/16], [-1/32, 1/32], [-1/64, 1/64], [-1/128, 1/128]]

    initial_setup = 'u_normal_random'
    initial_setup = 'u_normal_random_cutoff'
    rho_list = [[0], [0.05], [0.1], [0.2]]
    phase_list = [[0], [0.05], [0.1], [0.2]]
    rho_list = [[0.05]]
    phase_list = [[0.05]]

    "full phase module"
    #initial_setup = 'u_normal_phase_uniform_random'

    distribution_params_raw_list = [rho + phase for rho in rho_list for phase in phase_list]

    pdpp = Plot_Dpp(quantum_or_not, network_type, m, N, d, seed, alpha, dt, initial_setup, distribution_params, seed_initial_condition_list, reference_line, rho_or_phase)

    for i in range(4):
    #for i in range(1):
        for distribution_params_raw in (distribution_params_raw_list[i:][::4], distribution_params_raw_list[4 * i:4 * (i+1)]) :
        #for distribution_params_raw in ([distribution_params_raw_list]) :

            distribution_params_list = []
            for i in distribution_params_raw:
                distribution_params_list.append( [round(j, 10) for j in i])

            #pdpp.plot_pa_pb_initial_setup(N, distribution_params_list)
            
    """ dpp regular lattice
    """
    network_type_list = ['1D', '2D', '3D']
    N_list = [10000, 10000, 8000]
    network_type_list = ['2D']
    N_list = [10000]
    network_type_list = ['3D']
    N_list = [8000]
    #pdpp.dt = 0.2
    #pdpp.alpha = 0.2
    for network_type, N in zip(network_type_list, N_list):
        pdpp.network_type = network_type
        pdpp.N = N
        params_rho = [0.05, 0.1, 0.2]
        params_phase = [1]
        params_rho = [0.05, 0.1, 0.2]
        params_phase = [0, 0.05, 0.1, 0.2]
        params_rho = [[0.05, 0.2], [0.1, 0.2], [0.2, 0.2]]
        params_phase = [[0, 0.2], [0.05, 0.2], [0.1, 0.2], [0.2, 0.2]]
        rho_or_phase_list = ['rho']
        reference_line_list = ['average']
        for rho_or_phase, reference_line in zip(rho_or_phase_list, reference_line_list):
            pdpp.rho_or_phase = rho_or_phase
            pdpp.reference_line = reference_line
            for fixed in ['phase']:
                pdpp.plot_pa_pb_initial_setup_collections(N, params_rho, params_phase, fixed, axis_scale='semilogy')
                #pdpp.plot_pa_pb_stretched_exp(N, params_rho, params_phase, fixed)
                pass
    
        params_rho = [0, 0.05, 0.1, 0.2]
        params_phase = [0.05, 0.1, 0.2]
        params_rho = [[0, 0.2], [0.05, 0.2], [0.1, 0.2], [0.2, 0.2]]
        params_phase = [[0.05, 0.2], [0.1, 0.2], [0.2, 0.2]]
        rho_or_phase_list = ['phase']
        reference_line_list = [0]
        for rho_or_phase, reference_line in zip(rho_or_phase_list, reference_line_list):
            pdpp.rho_or_phase = rho_or_phase
            pdpp.reference_line = reference_line
            for fixed in ['rho']:
                pdpp.plot_pa_pb_initial_setup_collections(N, params_rho, params_phase, fixed, axis_scale='semilogy')
                #pdpp.plot_pa_pb_stretched_exp(N, params_rho, params_phase, fixed)
                pass
     

    


    seed_list = [i for i in range(0, 1)]



    pdpp.network_type = '3D_disorder'
    pdpp.N = 8000
    d_list = [0.3, 0.5, 0.7, 1]
    pdpp.network_type = '2D_disorder'
    pdpp.N = 10000
    d_list = [0.51, 0.55, 0.7, 0.9, 1]

    rho_or_phase_list = ['rho', 'phase']
    reference_line_list = ['average', 0]
    params_rho = [0, 0.05, 0.1, 0.2][::-1]
    params_rho = [0, 0.05, 0.1, 0.2][::-1]
    params_phase = [0, 0.05, 0.1, 0.2][::-1]
    linear_fit = 1
    loglog=False
    for rho_or_phase, reference_line in zip(rho_or_phase_list, reference_line_list):
        pdpp.rho_or_phase = rho_or_phase
        pdpp.reference_line = reference_line
        for pa_or_pb in ['pa', 'pb']:
            #pdpp.plot_pa_pb_disorder_collections(N, d_list, params_rho, params_phase, pa_or_pb, linear_fit, seed_list=seed_list, loglog=loglog)

            pass

    rho_params_list = [0, 0.05, 0.1, 0.2]
    phase_params_list = [0, 0.05, 0.1, 0.2]
    #pdpp.plot_pa_pb_exponent_disorder(rho_params_list, phase_params_list, d_list)
    """
    "disorder 2D"
    distribution_params_list = []
    for i in distribution_params_raw_list:
        distribution_params_list.append( [round(j, 3) for j in i])
    d_list = [0.5, 0.55, 0.7, 0.9, 1]
    for i, d in enumerate(d_list):
        #pdpp.plot_pa_pb_disorder(N, [d] * len(distribution_params_list), distribution_params_list)
        pass

    for i, distribution_params in enumerate(distribution_params_list):
        #pdpp.plot_pa_pb_disorder(N, d_list, [distribution_params] * len(d_list))
        pass

    """



    "test dx dt"

    N_list = [100, 100, 200, 500, 500]
    alpha_list = [2, 2, 1, 0.4, 0.4]
    dt_list = [1, 4, 1, 0.16, 1]
    m_list = [10] * 6 + [5.68] * 6
    N_list = [1000] * 12
    num_realization_list = [100] * 12
    alpha_list = [0.5, 1, 2, 3, 4, 5] * 2
    dt_list = [0.02, 0.1, 1, 1, 1, 1] * 2

    #pdpp.plot_pa_pb_alpha_dt(N_list, alpha_list, dt_list, num_realization_list)
    #pdpp.plot_exponent_scaling(m_list, N_list, alpha_list, dt_list, num_realization_list)

    #pdpp.plot_pa_pb_alpha_dt_linearfit(m_list, N_list, alpha_list, dt_list, num_realization_list)
    alpha_list = [0.5, 1, 2, 3]
    dt_list = [1] * len(alpha_list)
    num_realization_list = [100] * len(alpha_list)

    pdpp.network_type = '3D_disorder'
    pdpp.d = 0.5
    pdpp.network_type = '3D'
    pdpp.d = 4
    N_list = [1000] * len(alpha_list)
    pdpp.network_type = '2D_disorder'
    pdpp.d = 0.55
    pdpp.network_type = '1D'
    N_list = [1000] * 4

    pdpp.network_type = '3D'
    pdpp.d = 4
    N_list = [1000] * 4

    pdpp.network_type = '2D'
    pdpp.d = 4
    N_list = [900] * 4
    
    rho_or_phase_list = ['rho', 'phase']
    reference_line_list = ['average', 0]

    """
    "classical diffusion"
    pdpp.quantum_or_not = False
    pdpp.initial_setup = 'uniform_random'
    pdpp.initial_setup = 'normal_random'
    pdpp.distribution_params = 0.1
    pdpp.network_type = '1D'
    pdpp.d = 4
    alpha_list = [1, 2, 3, 4, 5] 
    """
    dt_list = [1] * len(alpha_list)
    num_realization_list = [100] * len(alpha_list)

    for rho_or_phase, reference_line  in zip(rho_or_phase_list, reference_line_list):
        pdpp.rho_or_phase = rho_or_phase
        pdpp.reference_line = reference_line
        #pdpp.plot_pa_pb_alpha_dt_scaling(N_list, alpha_list, dt_list, num_realization_list)
        #pdpp.plot_pa_pb_alpha_dt_scaling_fitting(N_list, alpha_list, dt_list, num_realization_list)

    "dimension exponent fitting"
    network_type_list = ['1D', '2D', '3D']
    N_list = [10000, 10000, 8000]
    rho_params_list = [0, 0.05, 0.1, 0.2]
    phase_params = 0.05
    pdpp.d = 4
    pdpp.rho_or_phase = 'phase'
    pdpp.reference_line = 0
    pdpp.rho_or_phase = 'rho'
    pdpp.reference_line = 'average'
    #pdpp.plot_pa_pb_exponent_dimension(network_type_list, N_list, rho_params_list, phase_params)
