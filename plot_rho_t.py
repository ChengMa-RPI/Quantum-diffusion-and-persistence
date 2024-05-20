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
import matplotlib.image as mpimg
from matplotlib import cm
from collections import defaultdict
from matplotlib import patches 
import json
from scipy.fft import fft, ifft, fft2
from calculate_fft import state_fft


fs = 24
ticksize = 20
labelsize = 35
anno_size = 18
subtitlesize = 15
legendsize= 20
alpha = 0.8
lw = 3
marksize = 8

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def title_name(params, quantum_or_not, d=None):
    # deprecated version, without support for chi distribution
    if quantum_or_not:
        rho_start, rho_end, phase_start, phase_end = [round(i, 4) for i in params]
        if phase_start == phase_end:
            phase_title = '$\\theta = C$'
        else:
            phase_title = '$\\theta \sim $' + f'({phase_start } $\\pi$, {phase_end } $\\pi$)'

    else:
        rho_start, rho_end = params

    if rho_start == rho_end:
        rho_title = '$\\rho = C$'
    else:
        rho_title = '$\\rho \sim $' + f'({rho_start * 2}/N, {rho_end * 2}/N)'

    if quantum_or_not:
        if d:
            return f'$d={d}$\n' + phase_title
        else:
            return rho_title  + '\n' + phase_title
    else:
        if d:
            return f'$d={d}$' 
        else:
            return rho_title 


class plotTemporalSpatialState():
    def __init__(self, quantum_or_not, network_type, N, d, seed, alpha, dt, initial_setup, distribution_params, seed_initial_condition_list, rho_or_phase):
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
        self.rho_or_phase = rho_or_phase

    def read_phi(self, seed_initial_condition, rho_or_phase=None, full_data_t=False):
        if rho_or_phase is None:
            rho_or_phase = self.rho_or_phase
        if self.quantum_or_not:
            if rho_or_phase == 'rho':
                des = '../data/quantum/state/' + self.network_type + '/' 
            elif rho_or_phase == 'phase':
                des = '../data/quantum/phase/' + self.network_type + '/' 

        else:
            des = '../data/classical/state/' + self.network_type + '/' 
        if full_data_t:
            save_file = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_seed_initial={seed_initial_condition}_full.npy'
        else:
            save_file = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_seed_initial={seed_initial_condition}.npy'


        data = np.load(save_file)
        t, state = data[:, 0], data[:, 1:]
        return t, state

    def plot_rho_x_compare_finite_N(self, N_list, alpha_list, dt_list, plot_t_list, seed_initial_condition):
        """ 
        plot rho vs x (position) for classical diffusion / quantum dynamics
        """
        rows = 2
        cols = len(plot_t_list) // rows
        figsize_x, figsize_y = 4 * cols, 3.5 * rows
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(figsize_x, figsize_y))

        L = int(N_list[0] * alpha_list[0])
        for (i, N), alpha, dt in zip(enumerate(N_list), alpha_list, dt_list):
            self.N = N
            self.alpha = alpha
            self.dt = dt
            t, state = self.read_phi(seed_initial_condition)
            state *= N/L
            for j, plot_t in enumerate(plot_t_list):
                ax = axes[j//cols, j% cols]
                simpleaxis(ax)
                ax.set_title(f'$t={plot_t}$')
                index = np.where(np.abs(t - plot_t) < 1e-2)[0][0]
                ax.plot(np.arange(0, L, 5), state[index][::int(5/alpha)], label=f'N={N}_dt={dt}', linewidth=1.0)
        xlabel, ylabel = '$x$', '$\\rho$'
        ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.45, 0.85) ) 
        fig.text(x=0.02, y=0.5, horizontalalignment='center', s=ylabel, size=labelsize*0.6, rotation=90)
        fig.text(x=0.5, y=0.04, horizontalalignment='center', s=xlabel, size=labelsize*0.6)
        fig.subplots_adjust(left=0.1, right=0.9, wspace=0.25, hspace=0.25, bottom=0.13, top=0.90)
        filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={N_list}_dt={dt_list}_{self.rho_or_phase}_time_space.png'
        save_des = '../transfer_figure/' + filename
        plt.savefig(save_des, format='png')
        plt.close()

    def plot_rho_x(self, plot_t_list, seed_initial_condition, full_data_t):
        """ 
        For animation.
        plot rho vs x (position) for classical diffusion / quantum dynamics
        """
        self.rho_or_phase = 'rho'
        if type(self.alpha) == int:
            dx = self.alpha
        else:
            dx = self.alpha[0]
        L = int(self.N * dx)
        t, state = self.read_phi(seed_initial_condition, full_data_t=full_data_t)
        if self.quantum_or_not:
            state /= state[0].sum() * dx 
        else:
            state = state - state[0].sum() / self.N
            state = state * 1/self.N + 1/self.N
        save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_alpha={self.alpha}_dt={self.dt}_{self.rho_or_phase}_rho_x/' 
        save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_alpha={self.alpha}_dt={self.dt}_{self.rho_or_phase}_setup={self.initial_setup}_params={self.distribution_params}_rho_x/' 
        if not os.path.exists(save_des):
            os.makedirs(save_des)

        for j, plot_t in enumerate(plot_t_list):
            fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(12, 10))
            simpleaxis(ax)
            ax.set_title(f'$t={plot_t}$', size=labelsize*1.2)
            index = np.where(np.abs(t - plot_t) < 1e-2)[0][0]
            interval = 1
            x_plot = np.arange(0, L, interval * dx)
            y_plot = state[index][::interval]
            ax.plot(x_plot, y_plot )
            ax.fill_between(x_plot, y_plot, color='tab:blue', alpha=0.4)
            if self.initial_setup == 'gaussian_wave':
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(0, 1/self.N * 2)
            ax.tick_params(axis='both', which='major', labelsize=ticksize*1.0)
            xlabel, ylabel = '$x$', '$\\rho$'
            fig.text(x=0.02, y=0.5, horizontalalignment='center', s=ylabel, size=labelsize*1, rotation=90)
            fig.text(x=0.5, y=0.04, horizontalalignment='center', s=xlabel, size=labelsize*1)

            fig.subplots_adjust(left=0.15, right=0.95, wspace=0.25, hspace=0.25, bottom=0.13, top=0.90)
            filename = save_des + f't={plot_t}.png'
            plt.savefig(filename, format='png')
            plt.close()

    def plot_rho_theta_t(self, seed_initial_condition, distribution_params_list):
        rows = 4
        cols = len(distribution_params_list) // rows
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=False, figsize=(4 * cols, 3.5 * rows))

        if self.rho_or_phase == 'rho':
            ylabel =  '$\\rho - \\langle \\rho \\rangle$'
            average = 1/ self.N
        else:
            ylabel =  '$\\theta$'
            average = 0

        for i, distribution_params in enumerate(distribution_params_list):
            ax = axes[i//cols, i% cols]
            simpleaxis(ax)
            self.distribution_params = distribution_params
            title = title_name(distribution_params, self.quantum_or_not)
            t, state = self.read_phi(seed_initial_condition)
            ax.set_title(title, size=labelsize*0.5, y = 0.92)

            index = [1, 10, 100, 500]
            t_index = np.where(t < 100)[0][-1]
            ax.plot(t[:t_index], state[:t_index, index] - average, linewidth=1.0)
        xlabel = '$t$'
        #ax.legend(fontsize=legendsize*0.7, frameon=False, loc=4, bbox_to_anchor=(1.45, 0.85) ) 
        fig.text(x=0.02, y=0.5, horizontalalignment='center', s=ylabel, size=labelsize*0.6, rotation=90)
        fig.text(x=0.5, y=0.04, horizontalalignment='center', s=xlabel, size=labelsize*0.6)
        fig.subplots_adjust(left=0.1, right=0.9, wspace=0.25, hspace=0.25, bottom=0.13, top=0.90)
        filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_{self.rho_or_phase}_t.png'
        filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_{self.rho_or_phase}_t_large.png'
        save_des = '../transfer_figure/' + filename
        plt.savefig(save_des, format='png')
        plt.close()

    def get_removal_scatter(self, n_splits=11, markersize=0.5):
        "get removed edge for 2D disordered lattice, and save it to file"

        des_remove_edge = f'../data/matrix_save/disorder_edge_remove/'
        if not os.path.exists(des_remove_edge):
            os.makedirs(des_remove_edge)
        file_remove_edge = des_remove_edge + f'network_type={self.network_type}_N={self.N}_d={self.d}_seed={self.seed}.npy'
        if not os.path.exists(file_remove_edge):
            L = int( np.sqrt(self.N) )
            G = nx.grid_graph(dim=[L, L], periodic=True)
            A = nx.to_numpy_array(G)
            modifer = 1000
            A_random = A * np.random.RandomState(self.seed + modifer).uniform(0, 1, (self.N, self.N))
            A_random = np.triu(A_random, 0) + np.triu(A_random, 1).transpose()
            A_disorder = np.array(A_random > (1-d), dtype=int)  # 1-d is the edge remove rate
            remove_matrix =  A - A_disorder
            row, column = np.where(np.triu(remove_matrix) == 1)
            r1, c1, r2, c2 = row // L, row % L, column // L, column % L
            dr, dc = r1 - r2, c1 - c2
            r_same = [r1[dr==0], c1[dr==0]]
            c_same = [r1[dc==0], c1[dc==0]]
            remove_edges = []
            scatter_x = []
            scatter_y = []
            for r1, c1 in zip(*r_same):
                scatter_x.append(np.ones(n_splits) * (c1+1))
                scatter_y.append(np.linspace(r1, r1+1, n_splits))
            for r2, c2 in zip(*c_same):
                scatter_x.append(np.linspace(c2, c2+1, n_splits))
                scatter_y.append(np.ones(n_splits) * (r2+1))
            scatter_x = np.hstack(scatter_x)
            scatter_y = np.hstack(scatter_y)
            scatters = np.vstack((scatter_x, scatter_y))

            np.save(file_remove_edge, scatters)
        else:
            scatters = np.load(file_remove_edge)
            scatter_x, scatter_y = scatters[0], scatters[1]

        plt.plot(scatter_x, scatter_y, '.', markersize=markersize, c='tab:grey')
        return None


    def heatmap(self, plot_t_list, seed_initial_condition, log_or_linear, plot_rho_or_phase, velocity=False, linewidth=0, show_remove_edge=False):
        """plot and save figure for animation

        """

        save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_initial_setup={self.initial_setup}_params={self.distribution_params}_N={self.N}_d={self.d}_dt={self.dt}_dx={self.alpha}_{rho_or_phase}_realization{seed_initial_condition}_heatmap_{plot_rho_or_phase}_{log_or_linear}_velocity={velocity}/'
        if show_remove_edge:
            save_des = save_des[:-1] + '_show_edge_removal/'
        if not os.path.exists(save_des):
            os.makedirs(save_des)

        if plot_rho_or_phase == 'rho' or plot_rho_or_phase == 'u':
            t, state = self.read_phi(seed_initial_condition, 'rho')
        else:
            t, state = self.read_phi(seed_initial_condition, 'phase')

        N_actual = len(state[0])
        r0 = np.sqrt(1/N_actual)
        if plot_rho_or_phase == 'rho':
            state = state  
        elif plot_rho_or_phase == 'u':
            state = np.sqrt(state) / r0 - 1
        if velocity:
            state = (state[1:] - state[:-1]) / (t[1] - t[0])
            
        index_plot = [np.where(np.abs(t-plot_t) < 1e-2)[0][0] for plot_t in plot_t_list]
        if log_or_linear == 'log':
            state = np.log10(state + 1e-10)
            xmin = max(np.min(state[index_plot]), -int(np.log10(self.N))-1)
            xmax = np.max(state[index_plot])
        else:
            xmin = np.min(state[index_plot])
            xmax = np.max(state[index_plot])
            if plot_rho_or_phase == 'rho':
                xmin, xmax = 0, max(np.abs(xmin), np.abs(xmax))
            else:    
                xmin, xmax = -max(np.abs(xmin), np.abs(xmax)), max(np.abs(xmin), np.abs(xmax))
        if self.network_type == '2D_disorder':
            cluster_file = f'../data/matrix_save/disorder_corresponding/network_type={self.network_type}_N={self.N}_d={self.d}_seed={self.seed}.csv'
            cluster_corresponding = pd.read_csv(cluster_file, header=None).iloc[:, -1].tolist()
            block_value = 0
            state_full = np.ones((len(t), self.N)) * block_value
            state_full[:, cluster_corresponding] = state 
            state = state_full
            block = np.setdiff1d(np.arange(self.N), cluster_corresponding)
            hole_x = block % np.sqrt(self.N)
            hole_y = block // np.sqrt(self.N)
        for i, plot_t in enumerate(plot_t_list):
            index = np.where(np.abs(t-plot_t) < 1e-2)[0][0]
            state_i = state[index]
            #xmin = np.min(state_i)
            #xmax = np.max(state_i)
            data_plot = state_i.reshape(int(np.sqrt(self.N)), int(np.sqrt(self.N)))

            #fig = sns.heatmap(data_snap, vmin=0, vmax=1, linewidths=linewidth, cbar_kws = {"orientation" : "horizontal"})
            fig = sns.heatmap(data_plot, vmin=xmin, vmax=xmax, linewidth=linewidth, cmap='bwr')
            if self.network_type == '2D_disorder':
                plt.plot(hole_x+0.46, hole_y+0.46, 's', markersize=3, color='grey')
                if show_remove_edge:
                    self.get_removal_scatter()
            cax = plt.gcf().axes[-1]
            cax.tick_params(labelsize=0.6 * fs)
            if log_or_linear == 'log':
                ave = int(np.log10(self.N))
                ticks = np.arange(-ave-1, int(xmax), 1)
                cax.set_yticks(ticks)
                cax.set_yticklabels([f'$10^{tick}$' for tick in ticks])
            fig = fig.get_figure()
            if plot_rho_or_phase == 'rho':
                #bar_title = '$\\rho / \\langle \\rho \\rangle -1$'
                bar_title = '$\\rho $'
            elif plot_rho_or_phase == 'u':
                bar_title = '$u$'
            elif plot_rho_or_phase == 'phase':
                if velocity:
                    bar_title =  '$ d\\theta / dt$'
                else:
                    bar_title = '$\\theta$'
            fig.text(x=0.95, y=0.5, horizontalalignment='center', s=bar_title, size=labelsize*0.5, rotation=90)

            plt.subplots_adjust(left=0.10, right=0.93, wspace=0.25, hspace=0.25, bottom=0.10, top=0.93)
            plt.title(f't={plot_t}', fontsize=17)
            plt.axis('off')
            #fig.patch.set_alpha(0.)
            #plt.title('time = ' + str(round(i, 2)) )
            fig.savefig(save_des + f't={plot_t}.png')
            plt.close()
        return None


    def quiver_phase(self, plot_t_list, seed_initial_condition, velocity=False, linewidth=0):
        """plot and save figure for animation

        """

        save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_initial_setup={self.initial_setup}_params={self.distribution_params}_N={self.N}_d={self.d}_dt={self.dt}_dx={self.alpha}_{rho_or_phase}_realization{seed_initial_condition}_quiver_phase_{velocity}/'
        if not os.path.exists(save_des):
            os.makedirs(save_des)

        t, state = self.read_phi(seed_initial_condition, 'phase')
        if self.network_type == '2D' or self.network_type == '2D_disorder':
            N_x = int(np.sqrt(self.N))

        if velocity:
            state = (state[1:] - state[:-1]) / (t[1]-t[0])
        index_plot = [np.where(np.abs(t-plot_t) < 1e-2)[0][0] for plot_t in plot_t_list]
        xmin = np.min(state[index_plot])
        xmax = np.max(state[index_plot])
        xmin, xmax = -max(np.abs(xmin), np.abs(xmax)), max(np.abs(xmin), np.abs(xmax))
        if self.network_type == '2D_disorder':
            des_file = f'../data/matrix_save/disorder_corresponding/network_type={self.network_type}_N={self.N}_d={self.d}_seed={self.seed}.csv'
            cluster_corresponding = pd.read_csv(des_file, header=None).iloc[:, -1].tolist()
            
            block_value = 0
            state_full = np.ones((len(t), self.N)) * block_value
            state_full[:, cluster_corresponding] = state 
            state = state_full
            block = np.setdiff1d(np.arange(self.N), cluster_corresponding)
            hole_x = block % np.sqrt(self.N)
            hole_y = block // np.sqrt(self.N)
        for i, plot_t in enumerate(plot_t_list):
            index = np.where(np.abs(t-plot_t) < 1e-2)[0][0]
            state_i = state[index]
            data_plot = state_i.reshape(N_x, N_x)
            n = int(N_x *0.1)  # how large the figure is
            data_select = data_plot[:n, :n]

            x = np.arange(0, n, 1)
            y = np.arange(0, n, 1)
            X, Y = np.meshgrid(x, y)

            u = np.cos(data_select)
            v = np.sin(data_select)
            color = data_select
            fig, ax = plt.subplots(figsize=(12, 12))
            q = ax.quiver(X, Y, u, v, color, norm=plt.Normalize(vmin=xmin, vmax=xmax), cmap='coolwarm', angles='xy', scale_units='xy', scale=1.1)
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.set_aspect('equal')
            ax.set_title(f't={plot_t}', fontsize=labelsize*0.9)
            ax.set_xlim(0, n+0.1)
            #ax.set_ylim(-0.1, n+0.3)
            ax.axis('off')

            if self.network_type == '2D_disorder':
                holes = np.vstack(([(x_i, y_i) for x_i, y_i in zip(hole_x, hole_y) if x_i < n and y_i < n]))
                ax.plot(holes[:, 0] + 0.43, holes[:, 1], '.', color='white', markersize=32 /n * 40)
            cax = fig.add_axes([0.87, 0.1, 0.02, 0.75]) 
            cax.tick_params(labelsize=0.9 * fs)

            plt.colorbar(q, ax=ax, cax=cax)
            fig = fig.get_figure()
            if velocity:
                bar_title = '$ d\\theta / dt$'
            else:
                bar_title = '$\\theta$'
            fig.text(x=0.97, y=0.5, horizontalalignment='center', s=bar_title, size=labelsize*0.9, rotation=90)


            plt.subplots_adjust(left=0.01, right=0.85, wspace=0.25, hspace=0.25, bottom=0.01, top=0.93)
            #fig.patch.set_alpha(0.)
            #plt.title('time = ' + str(round(i, 2)) )
            fig.savefig(save_des + f't={plot_t}.png')
            plt.close()
        return None

    def quiver_phase_gradient(self, plot_t_list, seed_initial_condition, linewidth=0):
        """plot and save figure for animation

        """

        save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_initial_setup={self.initial_setup}_params={self.distribution_params}_N={self.N}_d={self.d}_dt={self.dt}_dx={self.alpha}_{rho_or_phase}_realization{seed_initial_condition}_quiver_phase_gradient/'
        if not os.path.exists(save_des):
            os.makedirs(save_des)

        t, state = self.read_phi(seed_initial_condition, 'phase')
        if self.network_type == '2D' or self.network_type == '2D_disorder':
            N_x = int(np.sqrt(self.N))

        index_plot = [np.where(np.abs(t-plot_t) < 1e-2)[0][0] for plot_t in plot_t_list]
        state_transform = state.reshape(state.shape[0], N_x, N_x)
        u = (state_transform[:, 1:, :-1] - state_transform[:, :-1, :-1]) / self.alpha
        v = (state_transform[:, :-1, 1:] - state_transform[:, :-1, :-1]) / self.alpha
        color = np.sqrt((u**2 + v**2))
        xmin = np.min(color)
        xmax = np.max(color)
        if self.network_type == '2D_disorder':
            # not updated for disordered case
            des_file = f'../data/matrix_save/disorder_corresponding/network_type={self.network_type}_N={self.N}_d={self.d}_seed={self.seed}.csv'
            cluster_corresponding = pd.read_csv(des_file, header=None).iloc[:, -1].tolist()
            
            block_value = 0
            state_full = np.ones((len(t), self.N)) * block_value
            state_full[:, cluster_corresponding] = state 
            state = state_full
            block = np.setdiff1d(np.arange(self.N), cluster_corresponding)
            hole_x = block % np.sqrt(self.N)
            hole_y = block // np.sqrt(self.N)
        for i, plot_t in enumerate(plot_t_list):
            index = np.where(np.abs(t-plot_t) < 1e-2)[0][0]
            state_i = state[index]
            data_plot = state_i.reshape(N_x, N_x)
            n = int(N_x *0.2)  # how large the figure is
            data_select = data_plot[:n+1, :n+1]

            x = np.arange(0, n, 1)
            y = np.arange(0, n, 1)
            X, Y = np.meshgrid(x, y)

            u = (data_select[1:, :-1] - data_select[:-1, :-1]) / self.alpha
            v = (data_select[:-1, 1:] - data_select[:-1, :-1]) / self.alpha
            length = np.sqrt(u ** 2 + v ** 2)
            color = length
            u = u / length * np.log(length + 1.1)
            v = v / length * np.log(length + 1.1) 
            fig, ax = plt.subplots(figsize=(12, 12))
            q = ax.quiver(X, Y, u, v, color, norm=plt.Normalize(vmin=xmin, vmax=xmax), angles='xy', width=0.003, scale=5)
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.set_aspect('equal')
            ax.set_title(f't={plot_t}', fontsize=labelsize*0.9)
            ax.set_xlim(0, n+0.1)
            #ax.set_ylim(-0.1, n+0.3)
            ax.axis('off')

            if self.network_type == '2D_disorder':
                holes = np.vstack(([(x_i, y_i) for x_i, y_i in zip(hole_x, hole_y) if x_i < n and y_i < n]))
                ax.plot(holes[:, 0] + 0.43, holes[:, 1], '.', color='white', markersize=32 /n * 40)
            cax = fig.add_axes([0.87, 0.1, 0.02, 0.75]) 
            cax.tick_params(labelsize=0.9 * fs)

            plt.colorbar(q, ax=ax, cax=cax)
            fig = fig.get_figure()
            bar_title = '$\\nabla \\theta$'
            fig.text(x=0.97, y=0.5, horizontalalignment='center', s=bar_title, size=labelsize*0.9, rotation=90)


            plt.subplots_adjust(left=0.01, right=0.85, wspace=0.25, hspace=0.25, bottom=0.01, top=0.93)
            #fig.patch.set_alpha(0.)
            #plt.title('time = ' + str(round(i, 2)) )
            fig.savefig(save_des + f't={plot_t}.png')
            plt.close()
        return None

    def phase_gradient_fft(self, plot_t_list, seed_initial_condition_list, average=False):
        if average:
            save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_initial_setup={self.initial_setup}_params={self.distribution_params}_N={self.N}_d={self.d}_dt={self.dt}_dx={self.alpha}_{rho_or_phase}_realization{seed_initial_condition_list[-1]}_phase_gradient_fft_average.png'
        else:
            save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_initial_setup={self.initial_setup}_params={self.distribution_params}_N={self.N}_d={self.d}_dt={self.dt}_dx={self.alpha}_{rho_or_phase}_realization{seed_initial_condition_list[-1]}_phase_gradient_fft.png'

        if self.network_type == '2D' or self.network_type == '2D_disorder':
            N_x = int(np.sqrt(self.N))

        fig, axes = plt.subplots(nrows=2, ncols=len(plot_t_list)//2, figsize=(12, 12))
        for ax, (i, plot_t) in zip(itertools.chain(*axes), enumerate(plot_t_list)):
            k_mag_list = []
            gradient_fft_list = []
            for seed_initial_condition in seed_initial_condition_list:
                t, state = self.read_phi(seed_initial_condition, 'phase')
                index = np.where(np.abs(t-plot_t) < 1e-2)[0][0]
                state_i = state[index]
                data_plot = state_i.reshape(N_x, N_x)
                data_fft = fft2(data_plot)
                range_x = np.arange(N_x)
                dk = np.pi / (N_x//2)
                k_x = np.append(range_x[:N_x//2], -range_x[N_x//2:0:-1]) * dk
                k_y = np.append(range_x[:N_x//2], -range_x[N_x//2:0:-1]) * dk
                kx, ky = np.meshgrid(k_x, k_y) 
                k_mag = np.sqrt(kx ** 2 + ky ** 2) 
                gradient_fft = np.abs(data_fft) ** 2  * k_mag ** 2
                k_mag = k_mag.ravel()
                gradient_fft = gradient_fft.ravel()
                k_mag_list.append(k_mag)
                gradient_fft_list.append(gradient_fft)
            k_mag_list = np.hstack((k_mag_list))
            gradient_fft_list = np.hstack((gradient_fft_list))
            if average:
                k_initial = np.logspace(-2, 1, 100)
                plot_fft = []
                plot_k = []
                for k_i, k_j in zip(k_initial[:-1], k_initial[1:]):
                    index_select = np.where((k_mag_list < k_j) & (k_mag_list > k_i))[0]
                    if len(index_select):
                        plot_fft.append(np.mean(gradient_fft_list[index_select]))
                        plot_k.append(k_i)

                ax.loglog(plot_k, plot_fft, '.', label='ENI')
                ax.loglog(plot_k, [i ** 2 * 7 for i in plot_k], label='$y \\sim x^2$')
            else:
                ax.loglog(k_mag_list, gradient_fft_list, '.')

            ax.set_title(f't={plot_t}', fontsize=labelsize*0.7)
            ax.set_xlabel('|k|', size=labelsize*0.65)
            ax.set_ylabel('$|\\nabla \\theta _k|^2$',  size=labelsize*0.65)
            ax.tick_params(axis='both', which='major', labelsize=ticksize*0.7)
        ax.legend(fontsize=legendsize*1.0, frameon=False, loc=4, bbox_to_anchor=(1.0, 0.05) ) 
        plt.subplots_adjust(left=0.10, right=0.95, wspace=0.25, hspace=0.25, bottom=0.10, top=0.95)
        plt.savefig(save_des, format='png')
        plt.close()
        return None

    def phase_gradient_fft_3D(self, plot_t_list, seed_initial_condition):
        save_des = f'../transfer_figure/quantum={self.quantum_or_not}_network={self.network_type}_initial_setup={self.initial_setup}_params={self.distribution_params}_N={self.N}_d={self.d}_dt={self.dt}_dx={self.alpha}_{rho_or_phase}_realization{seed_initial_condition}_phase_gradient_fft_3D.png'

        fig, axes = plt.subplots(nrows=2, ncols=len(plot_t_list)//2, figsize=(12, 12), subplot_kw=dict(projection='3d'))
        t, state = self.read_phi(seed_initial_condition, 'phase')
        if self.network_type == '2D' or self.network_type == '2D_disorder':
            N_x = int(np.sqrt(self.N))

        index_plot = [np.where(np.abs(t-plot_t) < 1e-2)[0][0] for plot_t in plot_t_list]
        for ax, (i, plot_t) in zip(itertools.chain(*axes), enumerate(plot_t_list)):
            index = np.where(np.abs(t-plot_t) < 1e-2)[0][0]
            state_i = state[index]
            data_plot = state_i.reshape(N_x, N_x)
            data_fft = fft2(data_plot)
            range_x = np.arange(N_x)
            dk = np.pi / (N_x//2)
            k_x = np.append(range_x[:N_x//2], -range_x[N_x//2:0:-1]) * dk
            k_y = np.append(range_x[:N_x//2], -range_x[N_x//2:0:-1]) * dk
            kx, ky = np.meshgrid(k_x, k_y) 
            k_mag = np.sqrt(kx ** 2 + ky ** 2) 
            gradient_fft = np.abs(data_fft) ** 2  * k_mag ** 2
            surf = ax.plot_surface(kx, ky, gradient_fft, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, shade=False)
            ax.set_title(f't={plot_t}', fontsize=labelsize*0.7)
            #ax.set_xlabel('|k|', size=labelsize*0.65)
            #ax.set_ylabel('$|\\nabla \\theta _k|^2$',  size=labelsize*0.65)
            ax.tick_params(axis='both', which='major', labelsize=ticksize*0.7)
        plt.subplots_adjust(left=0.10, right=0.95, wspace=0.25, hspace=0.25, bottom=0.10, top=0.95)
        plt.savefig(save_des, format='png')
        plt.close()
        return None

    def state_fourier_transform(self, seed_initial_condition, t_list, x_list, plot_u, plot_t=True):

        rows = 2
        cols = len(t_list) // rows
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(6 * cols, 6 * rows))

        self.rho_or_phase = 'rho'
        t, rho_original = self.read_phi(seed_initial_condition)
        u_0 = np.sqrt(1/self.N)
        u_state = np.sqrt(rho_original) - u_0

        self.rho_or_phase = 'phase'
        t, phase_original = self.read_phi(seed_initial_condition)


        u_k, phase_k, a_k, b_k, L, n, k, omega_k = state_fft(rho_original[0], phase_original[0], hbar, m, self.alpha)
        #print(np.argsort(np.abs(np.imag(u_k)))[-10:], np.argsort(np.abs(np.real(u_k)))[-10:] )

        if plot_t:   # not complete!!!
            exponential_t = np.exp(1j * omega_k.reshape(len(omega_k), 1) * t)
            exponential_x = np.exp(1j * k * x_i)
            if plot_u:
                u_t = 1 / L * np.sum(( exponential_t * a_k.reshape(len(a_k), 1) +  exponential_t**(-1) * b_k.reshape(len(b_k), 1) ) * exponential_x.reshape(len(k), 1), axis=0 )
                ax.plot(t, u_state[:, int(x_i/self.alpha)], label='PDE')
                ax.plot(t, u_t, label='approx')
            else:
                theta_t = 1 / L * np.sum(1j * ( exponential_t * a_k.reshape(len(a_k), 1) -  exponential_t**(-1) * b_k.reshape(len(b_k), 1) ) * exponential_x.reshape(len(k), 1), axis=0 )

                ax.plot(t, u_state[:, int(x_i/self.alpha)], label='PDE')
                ax.plot(t, u_t, label='approx')
            xlabel = '$t$'

        else:
            x = np.arange(self.N) * alpha
            exponential_x = np.exp(1j * k.reshape(len(k), 1) * x)
            xlabel = '$x$'
            for i, t_i in enumerate(t_list):
                index_t = np.argmin(np.abs(t_i - t))
                t_actual = t[index_t]
                ax = axes[i//cols, i % cols]
                simpleaxis(ax)
                exponential_t = np.exp(1j * omega_k * t_actual)
                if plot_u:
                    appro_x = 1 / self.N * np.sum(( exponential_t * a_k +  exponential_t**(-1) * b_k ).reshape(len(k), 1) * exponential_x, axis=0 )
                    se_state = u_state[index_t]
                    ylabel = '$u$'
                    filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_setup={self.initial_setup}_params={self.distribution_params}_approximation_u_x.png'
                else:
                    appro_x = 1 / self.N * np.sum(1j * ( exponential_t * a_k -  exponential_t**(-1) * b_k ).reshape(len(k), 1) * exponential_x, axis=0 ) / u_0
                    filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_setup={self.initial_setup}_params={self.distribution_params}_approximation_theta_x.png'
                    se_state = phase_original[index_t]
                    ylabel = '$\\theta$'
                plot_x_index = 1000
                ax.plot(x[:plot_x_index], se_state[:plot_x_index], label='SE PDE')
                ax.plot(x[:plot_x_index], np.real(appro_x[:plot_x_index]), label='approx')
                ax.set_title(f't={int(t_actual)}', size=labelsize*0.7)
                ax.tick_params(axis='both', which='major', labelsize=ticksize*0.8)

        ax.legend(fontsize=legendsize*1.0, frameon=False, loc=4, bbox_to_anchor=(1.2, 0.85) ) 
        fig.text(x=0.02, y=0.5, horizontalalignment='center', s=ylabel, size=labelsize*1.0, rotation=90)
        fig.text(x=0.5, y=0.04, horizontalalignment='center', s=xlabel, size=labelsize*1.0)
        fig.subplots_adjust(left=0.1, right=0.9, wspace=0.25, hspace=0.25, bottom=0.13, top=0.90)
        save_des = '../transfer_figure/' + filename
        plt.savefig(save_des, format='png')
        plt.close()

        return None

    def compare_fft_oscillation(self, seed_initial_condition, n_list, t_stop):

        rows = 2
        cols = len(t_list) // rows
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(6 * cols, 6 * rows))

        t, rho_original = self.read_phi(seed_initial_condition, 'rho')
        u_0 = np.sqrt(1/self.N)
        u_state = np.sqrt(rho_original) - u_0

        t, phase_original = self.read_phi(seed_initial_condition, 'phase')
        phase_r = phase_original * u_0
        u_k_fft = fft(u_state)
        phase_k_fft = fft(phase_r)
        u_k, phase_k, a_k, b_k, L, n, k, omega_k = state_fft(rho_original[0], phase_original[0], hbar, m, self.alpha)
        u_k_oscillation = np.cos(omega_k * t.reshape(len(t), 1)) * u_k_fft[0]  +  np.sin(omega_k * t.reshape(len(t), 1)) * phase_k_fft[0] 
        stop_index = np.where(t>t_stop)[0][0]
        for i, n in enumerate(n_list):
            ax = axes[i//cols, i % cols]
            simpleaxis(ax)
            #ax.plot(t[:stop_index], np.real(u_k_oscillation[:stop_index, n]), label='linearization') 
            #ax.plot(t[:stop_index], np.real(u_k_fft[:stop_index, n]), label='fft') 

            ax.plot(t[:stop_index], np.imag(u_k_oscillation[:stop_index, n]), label='linearization') 
            ax.plot(t[:stop_index], np.imag(u_k_fft[:stop_index, n]), label='fft') 

            ax.set_title(f'n={n}', size=labelsize*0.7)
            ax.tick_params(axis='both', which='major', labelsize=ticksize*0.8)

        """
        ax = axes[-1, -1]
        ax.plot(t[:stop_index], np.sqrt(np.sum(np.abs(u_k_oscillation[:stop_index, :]) ** 2, axis=1)), label='linearization') 
        ax.plot(t[:stop_index], np.sqrt(np.sum(np.abs(u_k_fft[:stop_index, :]) ** 2, axis=1)), label='fft') 
        """
            
        xlabel = '$t$'
        ylabel = '$Re(u_k)$'
        ylabel = '$Im(u_k)$'

        ax.legend(fontsize=legendsize*1.0, frameon=False, loc=4, bbox_to_anchor=(1.5, 0.85) ) 
        filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_setup={self.initial_setup}_params={self.distribution_params}_compare_fft_linearization_imag_t_stop={t_stop}.png'
        fig.text(x=0.02, y=0.5, horizontalalignment='center', s=ylabel, size=labelsize*1.0, rotation=90)
        fig.text(x=0.5, y=0.04, horizontalalignment='center', s=xlabel, size=labelsize*1.0)
        fig.subplots_adjust(left=0.1, right=0.9, wspace=0.25, hspace=0.25, bottom=0.13, top=0.90)
        save_des = '../transfer_figure/' + filename
        plt.savefig(save_des, format='png')
        plt.close()

        return None

    def error_k_fft_linearization(self, seed_initial_condition):
        rows = 1
        cols = 2
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=False, figsize=(6 * cols, 6 * rows))

        t, rho_original = self.read_phi(seed_initial_condition, 'rho')
        u_0 = np.sqrt(1/self.N)
        u_state = np.sqrt(rho_original) - u_0

        t, phase_original = self.read_phi(seed_initial_condition, 'phase')
        phase_r = phase_original * u_0
        u_k_fft = fft(u_state)
        phase_k_fft = fft(phase_r)
        u_k, phase_k, a_k, b_k, L, n, k, omega_k = state_fft(rho_original[0], phase_original[0], hbar, m, self.alpha)
        u_k_oscillation = np.cos(omega_k * t.reshape(len(t), 1)) * u_k_fft[0]  +  np.sin(omega_k * t.reshape(len(t), 1)) * phase_k_fft[0] 
        stop_index = np.where(t>9000)[0][0]
        u_approx = np.abs(u_k_oscillation[:stop_index] )
        u_true = np.abs(u_k_fft[:stop_index])
        relative_error = np.mean( np.abs(u_approx - u_true) / u_true  ,  axis=0)
        absolute_error = np.mean( np.abs(u_approx - u_true)  ,  axis=0)
        errors = [relative_error, absolute_error]
        titles = ['Relative Error', 'Absolute Error']
        for (i, error), title_name in zip(enumerate(errors), titles):
            ax = axes[i]
            simpleaxis(ax)
            ax.plot(n[:self.N//2], error[:self.N//2], '.', alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=ticksize*0.8)
            ax.set_title(title_name, size=labelsize*0.7)
            
        xlabel = '$n$'
        ylabel = 'Error'

        #ax.legend(fontsize=legendsize*1.0, frameon=False, loc=4, bbox_to_anchor=(1.2, 0.85) ) 
        filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_setup={self.initial_setup}_params={self.distribution_params}_error_k_fft_linearization.png'
        fig.text(x=0.02, y=0.5, horizontalalignment='center', s=ylabel, size=labelsize*1.0, rotation=90)
        fig.text(x=0.5, y=0.04, horizontalalignment='center', s=xlabel, size=labelsize*1.0)
        fig.subplots_adjust(left=0.1, right=0.9, wspace=0.25, hspace=0.25, bottom=0.13, top=0.90)
        save_des = '../transfer_figure/' + filename
        plt.savefig(save_des, format='png')
        plt.close()

        return None

    def error_t_fft_linearization(self, seed_initial_condition):
        rows = 1
        cols = 2
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=False, figsize=(6 * cols, 6 * rows))

        t, rho_original = self.read_phi(seed_initial_condition, 'rho')
        u_0 = np.sqrt(1/self.N)
        u_state = np.sqrt(rho_original) - u_0

        t, phase_original = self.read_phi(seed_initial_condition, 'phase')
        phase_r = phase_original * u_0
        u_k_fft = fft(u_state)
        phase_k_fft = fft(phase_r)
        u_k, phase_k, a_k, b_k, L, n, k, omega_k = state_fft(rho_original[0], phase_original[0], hbar, m, self.alpha)
        u_k_oscillation = np.cos(omega_k * t.reshape(len(t), 1)) * u_k_fft[0]  +  np.sin(omega_k * t.reshape(len(t), 1)) * phase_k_fft[0] 
        #stop_index = np.where(t>100)[0][0]
        u_approx = np.abs(u_k_oscillation)
        u_true = np.abs(u_k_fft)
        relative_error = np.abs(u_approx - u_true) / u_true 
        absolute_error = np.abs(u_approx - u_true) 
        length = (np.arange(1, len(t)+1, 1)).reshape(len(t), 1)
        relative_error_cummean = np.cumsum(relative_error, axis=0) / length
        absolute_error_cummean = np.cumsum(absolute_error, axis=0) / length
        errors = [relative_error_cummean, absolute_error_cummean]
        titles = ['Relative Error', 'Absolute Error']
        for (i, error), title_name in zip(enumerate(errors), titles):
            ax = axes[i]
            simpleaxis(ax)
            ax.plot(t, error[:, :self.N//2][:, ::100], '-', alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=ticksize*0.8)
            ax.set_title(title_name, size=labelsize*0.7)
            
        xlabel = '$t$'
        ylabel = 'Error'

        #ax.legend(fontsize=legendsize*1.0, frameon=False, loc=4, bbox_to_anchor=(1.2, 0.85) ) 
        filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_setup={self.initial_setup}_params={self.distribution_params}_error_t_fft_linearization.png'
        fig.text(x=0.02, y=0.5, horizontalalignment='center', s=ylabel, size=labelsize*1.0, rotation=90)
        fig.text(x=0.5, y=0.04, horizontalalignment='center', s=xlabel, size=labelsize*1.0)
        fig.subplots_adjust(left=0.1, right=0.9, wspace=0.25, hspace=0.25, bottom=0.13, top=0.90)
        save_des = '../transfer_figure/' + filename
        plt.savefig(save_des, format='png')
        plt.close()

        return relative_error_cummean, absolute_error_cummean



        
hbar = 0.6582
m = 5.68

if __name__ == '__main__':
    quantum_or_not = True
    initial_setup = 'u_uniform_random'
    network_type = '1D'
    N = 10000
    d = 4
    seed = 0
    alpha = 0.1
    dt = 0.01
    alpha = 1
    dt = 1
    seed_initial_condition_list = np.arange(0, 10, 1)
    distribution_params = [7/16, 9/16, -1/16, 1/16]
    distribution_params = [63/128, 65/128, -1/128, 1/128]
    initial_setup = 'sum_sin_inphase'
    distribution_params = [round(j, 10) for j in [10, 1e-2, 0.03] ]
    rho_or_phase = 'rho'
    rho_or_phase = 'phase'
    N_list = [100, 100, 200, 200, 500]
    alpha_list = [5, 5, 2.5, 2.5, 1]
    dt_list = [1, 25, 1, 6.25, 1]

    N_list = [100, 100]
    alpha_list = [5, 5]

    N_list = [500, 200, 100]
    alpha_list = [1, 2.5, 5]
    dt_list = [1, 6.25, 25]
    plot_t_list = [0, 1, 5, 10, 25, 50]
    plot_t_list = [0, 25, 50, 100, 200, 400]
    seed_initial_condition = 0

    rho_list = [[7/16, 9/16], [15/32, 17/32], [31/64, 33/64], [63/128, 65/128]]
    phase_list = [[-1/16, 1/16], [-1/32, 1/32], [-1/64, 1/64], [-1/128, 1/128]]

    rho_list = [[0, 1], [1/4, 3/4], [3/8, 5/8], [1, 1]]
    phase_list = [[-1, 1], [-1/2, 1/2], [-1/4, 1/4], [0, 0]]

    "uniform_random"
    initial_setup = 'uniform_random'
    rho_list = [[63/128, 65/128]]
    phase_list = [ [-1/128, 1/128]]

    "sum_sin_inphase"
    initial_setup = 'sum_sin_inphase'
    rho_list = [[1], [5], [10]]
    phase_list = [[1e-3, 0.1], [1e-2, 1]]

    "u_uniform_random_cutoff"
    initial_setup = 'u_uniform_random_cutoff'
    rho_list = [[0.05, 1]]
    phase_list =[[0.05, 1]]


    "u_uniform_random"
    initial_setup = 'u_uniform_random'
    rho_list = [[0.1]]
    phase_list =[[0.1]]

    initial_setup = 'u_normal_random'
    rho_list = [[0], [0.05], [0.1]]
    phase_list =[[0] , [0.05], [0.1]]
    rho_list = [[0.05]]
    phase_list = [[0.05]]


    ptss = plotTemporalSpatialState(quantum_or_not, network_type, N, d, seed, alpha, dt, initial_setup, distribution_params, seed_initial_condition_list, rho_or_phase)


    distribution_params_raw = [rho + phase for rho in rho_list for phase in phase_list]

    distribution_params_list = []
    for i in distribution_params_raw:
        distribution_params_list.append( [round(j, 10) for j in i])


    t_i = 10
    x_i = 0
    t_list = [0, 1, 5, 10, 100, 1000]
    t_list = [i*dt for i in [0, 1000, 3000, 5000, 6000, 8000]]
    x_list = [0, 10, 50, 500]
    n_list = [0, 1, 2, 3, 4]
    n_list = [0, 1, 500, 700, 1000, 3000]
    n_list = [0, 1, 50, 70, 100, 300]
    t_stop = 500
    plot_t = False
    plot_u = True

    for distribution_params in distribution_params_list:
        ptss.distribution_params = distribution_params
        #ptss.state_fourier_transform(0, t_list, x_list, plot_u, plot_t)
        #ptss.compare_fft_oscillation(0, n_list, t_stop)
        #ptss.error_k_fft_linearization(seed_initial_condition)
        #relative_error_cummean, absolute_error_cummean  = ptss.error_t_fft_linearization(seed_initial_condition)

    """
    ptss.rho_or_phase = 'phase'
    #ptss.plot_rho_theta_t(seed_initial_condition, distribution_params_list)
    ptss.rho_or_phase = 'rho'
    #ptss.plot_rho_theta_t(seed_initial_condition, distribution_params_list)

    """
    ptss.dt = 1
    ptss.N = 10000
    ptss.alpha = 1
    ptss.initial_setup = 'u_uniform_random'
    ptss.initial_setup = 'u_normal_random'
    distribution_params_list = [[0.05, 0] ]
    plot_t_list = np.arange(0, 100, 1).tolist()
    plot_t_list = np.arange(0, 1000, 10).tolist()
    plot_t_list = np.arange(0, 10000, 100).tolist()
    plot_t_list = np.sort(np.unique(np.hstack((np.arange(0, 1000, 10).tolist(),  np.arange(0, 10000, 100).tolist() ))))
    seed_initial_condition = 0
    log_or_linear = 'linear'
    log_or_linear = 'log'
    plot_rho_or_phase_list = ['rho', 'phase', 'u']
    plot_rho_or_phase_list = ['rho']
    ptss.network_type = '2D'
    d_list = [4]
    ptss.network_type = '2D_disorder'
    d_list = [0.51, 0.55, 0.7, 0.9]
    d_list = [0.7]
    ptss.initial_setup = 'phase_multi_locals'
    distribution_params_list = [[0, 1, 1] ]
    ptss.initial_setup = 'full_local'
    distribution_params_list = [[0, 0, 0]]
    show_remove_edge = True
    """to see the diffusion / localization"""
    #ptss.initial_setup = 'full_local'
    #distribution_params_list = [[0, 0, 0]]
    for distribution_params in distribution_params_list:
        ptss.distribution_params = distribution_params
        for d in d_list:
            ptss.d = d
            for plot_rho_or_phase in plot_rho_or_phase_list:
                #ptss.heatmap(plot_t_list, seed_initial_condition, log_or_linear, plot_rho_or_phase, velocity=False, show_remove_edge=show_remove_edge)
                #ptss.heatmap(plot_t_list, seed_initial_condition, log_or_linear, plot_rho_or_phase, velocity=True)
                pass
            #ptss.quiver_phase(plot_t_list, seed_initial_condition, velocity=False)
            #ptss.quiver_phase(plot_t_list, seed_initial_condition, velocity=True)
            #ptss.quiver_phase_gradient(plot_t_list, seed_initial_condition, linewidth=0)
            #plot_t_list = [0, 10, 100, 1000]
            #seed_initial_condition_list = range(10)
            #ptss.phase_gradient_fft(plot_t_list, seed_initial_condition_list, average=True)
            #ptss.phase_gradient_fft_3D(plot_t_list, seed_initial_condition)

    "plot rho vs x for animation"
    ptss.network_type = '1D'
    ptss.d = 4
    ptss.N = 1000
    ptss.seed = 0
    ptss.initial_setup = 'gaussian_wave'
    ptss.alpha = 0.01
    ptss.dt = 0.1
    ptss.quantum_or_not = True
    ptss.distribution_params = [0.7, 0]
    ptss.quantum_or_not = False
    ptss.distribution_params = 0.7
    ptss.alpha = [0.01, 0.005]
    ptss.dt = 0.01
    full_data_t = False

    ### random distribution
    ptss.dt = 1
    ptss.quantum_or_not = True
    ptss.alpha = 1
    ptss.initial_setup = 'u_normal_random'
    ptss.distribution_params = [0.05, 0]
    ptss.distribution_params = [0, 0.05]
    ptss.quantum_or_not = False
    ptss.alpha=[1, 0.1]
    ptss.initial_setup = 'normal_random'
    ptss.distribution_params = 0.05


    ### cutoff
    ptss.dt = 1
    ptss.quantum_or_not = True
    ptss.alpha = 1
    ptss.initial_setup = 'u_normal_random_cutoff'
    ptss.distribution_params = [0, 0.2, 0.05, 0.2]
    ptss.distribution_params = [0.05, 0.2, 0, 0.2]

    plot_t_list = np.round(np.arange(0, 1000, 10), 3)
    #ptss.plot_rho_x(plot_t_list, seed_initial_condition, full_data_t)
