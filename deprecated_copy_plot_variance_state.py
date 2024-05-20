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
from collections import defaultdict
from matplotlib import patches 
import json
from calculate_fft import state_fft
from scipy import special
from scipy.fft import fft, ifft
from matplotlib.legend_handler import HandlerTuple



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

def corvar_t_scaling(cor_var, remove_bias, u_std, phase_std, alpha, d_power):
    D_u = u_std ** 2 * alpha**d_power / 2 
    D_theta = phase_std ** 2 * alpha**d_power / 2
    cor_var = cor_var - (D_u+D_theta) / alpha**d_power 
    if u_std != phase_std:
        if remove_bias == 'D':
            cor_var = cor_var / ( (D_u-D_theta) )
        elif remove_bias == 'Dx':    
            cor_var = cor_var / ( (D_u-D_theta)/alpha**d_power )
    return cor_var



class plotStateVariance():
    def __init__(self, quantum_or_not, network_type, N, d, seed, alpha, dt, initial_setup, distribution_params, seed_initial_condition_list, rho_or_phase, r_separation, t_separation):
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
        self.r_separation = r_separation
        self.t_separation = t_separation

    def read_phi(self, seed_initial_condition, rho_or_phase=None):
        if rho_or_phase is None:
            rho_or_phase = self.rho_or_phase
        if self.quantum_or_not:
            if rho_or_phase == 'rho':
                des = '../data/quantum/state/' + self.network_type + '/' 
            elif rho_or_phase == 'phase':
                des = '../data/quantum/phase/' + self.network_type + '/' 

        else:
            des = '../data/classical/state/' + self.network_type + '/' 
        save_file = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_seed_initial={seed_initial_condition}.npy'
        data = np.load(save_file)
        t, state = data[:, 0], data[:, 1:]
        return t, state

    def corvar_t(self, seed_initial_condition_list, remove_bias=False):

        if self.network_type == '1D':
            d_power = 1
        elif self.network_type == '2D':
            d_power = 2
        elif self.network_type == '3D':
            d_power = 3
        else:
            d_power = 0
        corvar_list = []
        for j, seed_initial_condition in enumerate(seed_initial_condition_list):
            t, state = self.read_phi(seed_initial_condition)
            if self.rho_or_phase == 'rho':
                N_actual = len(state[0])
                r0 = np.sqrt(1/N_actual)
                state = np.sqrt(state) - r0
                state /= r0

            cor_var = np.mean(state * state, axis=1)
            if remove_bias:
                u_std, phase_std = self.distribution_params
                cor_var = corvar_t_scaling(cor_var, remove_bias, u_std, phase_std, self.alpha, d_power)

            corvar_list.append(cor_var)
        corvar_list = np.vstack((corvar_list))
        corvar_mean = np.mean(corvar_list, axis=0)
        return corvar_list, corvar_mean, t

    def corvar_r(self, seed_initial_condition_list, t_i, remove_bias):
        corvar_list = []
        for j, seed_initial_condition in enumerate(seed_initial_condition_list):
            t, state = self.read_phi(seed_initial_condition)
            N_actual = len(state[0])

            index = np.where(np.abs(t - t_i) < 1e-10)[0][0]
            if self.rho_or_phase == 'rho':
                r0 = np.sqrt(1/N_actual)
                state = np.sqrt(state) - r0
                state /= r0
            state_i = state[index]

            if self.network_type == '2D_disorder' or self.network_type == '3D_disorder':
                des_file = f'../data/matrix_save/disorder_corresponding/network_type={self.network_type}_N={self.N}_d={self.d}_seed={self.seed}.csv'
                cluster_corresponding = pd.read_csv(des_file, header=None).iloc[:, -1].tolist()
                block_value = 0
                state_full = np.zeros(( self.N)) * block_value
                state_full[cluster_corresponding] = state_i
                state_i = state_full

            if self.network_type == '1D':
                d_power = 1
                N_x = self.N
                state_shift = np.vstack(([np.roll(state_i, shift=i) for i in range(N_x//2+1)]))
                cor_var = np.mean(state_i * state_shift, axis=1)
            elif self.network_type in ['2D', '2D_disorder'] :
                d_power = 2
                N_x = round(np.sqrt(self.N))
                state_2D = state_i.reshape(N_x, N_x)

                state_shift1 = np.stack(([np.roll(state_2D, shift=i, axis=0) for i in range(N_x//2+1)])) # shift, t, sqrt(N), sqrt(N)
                state_shift2 = np.stack(([np.roll(state_2D, shift=i, axis=1) for i in range(N_x//2+1)])) # shift, t, sqrt(N), sqrt(N)
                cor_var = 1/2*(np.mean(state_2D * state_shift1, axis=(-1, -2))  + np.mean(state_2D * state_shift2, axis=(-1, -2)) ) * self.N/N_actual

            elif self.network_type in ['3D', '3D_disorder']:
                d_power = 3
                N_x = round(self.N**(1/3))
                state_3D = state_i.reshape(N_x, N_x, N_x)
                state_shift1 = np.stack(([np.roll(state_3D, shift=i, axis=0) for i in range(N_x//2+1)])) # shift, t, sqrt(N), sqrt(N)
                state_shift2 = np.stack(([np.roll(state_3D, shift=i, axis=1) for i in range(N_x//2+1)])) # shift, t, sqrt(N), sqrt(N)
                state_shift3 = np.stack(([np.roll(state_3D, shift=i, axis=2) for i in range(N_x//2+1)])) # shift, t, sqrt(N), sqrt(N)
                cor_var = 1/3*(np.mean(state_3D * state_shift1, axis=(-1, -2, -3))  + np.mean(state_3D * state_shift2, axis=(-1, -2, -3))  + np.mean(state_3D * state_shift3, axis=(-1, -2, -3)) ) * self.N/N_actual


            if remove_bias:
                u_std, phase_std = self.distribution_params
                D_u = u_std ** 2 * self.alpha**d_power / 2 
                D_theta = phase_std ** 2 * self.alpha**d_power / 2

                cor_var[0] = cor_var[0] - (D_u+D_theta) / self.alpha**d_power 
                cor_var[1:] = cor_var[1:] 
                if u_std != phase_std:
                    if remove_bias == 'D':
                        cor_var = cor_var / ( (D_u-D_theta) )
                    elif remove_bias == 'Dx':    
                        cor_var = cor_var / ( (D_u-D_theta)/self.alpha**d_power )


            corvar_list.append(cor_var)
        corvar_list = np.vstack((corvar_list))
        corvar_mean = np.mean(corvar_list, axis=0)
        return corvar_list, corvar_mean, t



    def corvar_t_ensemble_approx(self, distribution_params, remove_bias=False):
        if self.initial_setup == 'u_uniform_random' or self.initial_setup == 'u_normal_random':
            u_std, phase_std = distribution_params
            u_cutoff = 1
            phase_cutoff = 1
        elif self.initial_setup == 'u_uniform_random_cutoff' or self.initial_setup == 'u_normal_random_cutoff':
            u_std, u_cutoff, phase_std, phase_cutoff = distribution_params
        a_r = np.pi / self.alpha * u_cutoff  # u and phase should be treated separatedly
        t, rho_state = self.read_phi(0)
        if self.network_type in ['1D', '2D', '3D']:
            _, _, _, _, L, n, k, omega_k = state_fft(rho_state[0], rho_state[0], hbar, m, self.alpha, network_type = self.network_type)
            corvar_ensemble = (u_std**2 / u_cutoff * np.sum(np.cos(t.reshape(len(t), 1) * omega_k ) **2 * np.cos(k * self.r_separation), axis=1) + phase_std**2 / phase_cutoff * np.sum(np.sin(t.reshape(len(t), 1) * omega_k ) **2 * np.cos(k * self.r_separation), axis=1) )  / np.sqrt(N) **2 
            b_r =  m * self.r_separation / 2 / hbar / t[1:]  
            c_r = m * self.r_separation ** 2 /4/ hbar / t[1:] 
            v_r = hbar * t[1:] / m 
        if self.network_type == '1D':
            d_power = 1
        elif self.network_type == '2D':
            d_power = 2
        elif self.network_type == '3D':
            d_power = 3
        else:
            d_power = 0

        cor_var = (self.r_separation==0) * ( u_std ** 2 + phase_std ** 2) / 2 + 1/ 4 / (2*np.pi/self.alpha) ** d_power * (u_std**2/u_cutoff - phase_std**2/phase_cutoff) * np.sqrt(np.pi/v_r) ** d_power * np.real( (np.exp(-1j * c_r + 1j * np.pi/4 * d_power) * (special.erf(np.sqrt(-1j * v_r) * (a_r + b_r) )    - special.erf(np.sqrt(-1j * v_r) * (-a_r + b_r) )  )) *   special.erf(np.sqrt(-1j * v_r) * a_r) **(d_power-1) )    
        corvar_limit_approx =  (self.r_separation==0) * ( u_std ** 2 + phase_std ** 2 ) / 2 + 1/2/ (2 * np.pi/self.alpha)**d_power * (u_std**2/u_cutoff- phase_std**2/phase_cutoff) * np.sqrt(np.pi/v_r)**d_power  * np.cos(c_r - np.pi/4 * d_power)   

        corvar_limit = np.hstack(([(self.r_separation==0) * ( u_std ** 2  ) ], cor_var))
        corvar_limit_approx = np.hstack(([(self.r_separation==0) * ( u_std ** 2 ) ], corvar_limit_approx))

        if remove_bias:
            corvar_limit = corvar_t_scaling(corvar_limit, remove_bias, u_std, phase_std, self.alpha, d_power)
            corvar_limit_approx = corvar_t_scaling(corvar_limit_approx, remove_bias, u_std, phase_std, self.alpha, d_power)

        return corvar_ensemble, corvar_limit, corvar_limit_approx
            
    def corvar_r_ensemble_approx(self, distribution_params, t_plot, remove_bias):
        if self.initial_setup == 'u_uniform_random' or self.initial_setup == 'u_normal_random':
            u_std, phase_std = distribution_params
            u_cutoff = 1
            phase_cutoff = 1
        elif self.initial_setup == 'u_uniform_random_cutoff' or self.initial_setup == 'u_normal_random_cutoff':
            u_std, u_cutoff, phase_std, phase_cutoff = distribution_params
        a_r = np.pi / self.alpha * u_cutoff  # u and phase should be treated separatedly
        t, rho_state = self.read_phi(0)
        if self.network_type == '1D':
            d_power = 1
            N_x = self.N
        elif self.network_type == '2D':
            d_power = 2
            N_x = round(np.sqrt(self.N))
        elif self.network_type == '3D':
            d_power = 3
            N_x = round(self.N ** ( 1/3) )
        else:
            d_power = 0
            N_x = 0

        r = np.arange(0, int(N_x//2) + 1, 1)
        b_r =  m * r / 2 / hbar / t_plot  
        c_r = m * r ** 2 /4/ hbar / t_plot 
        v_r = hbar * t_plot / m 

        corvar_limit =  1/ 4 / (2*np.pi/self.alpha) ** d_power * (u_std**2/u_cutoff - phase_std**2/phase_cutoff) * np.sqrt(np.pi/v_r) ** d_power * np.real( (np.exp(-1j * c_r + 1j * np.pi/4 * d_power) * (special.erf(np.sqrt(-1j * v_r) * (a_r + b_r) ) - special.erf(np.sqrt(-1j * v_r) * (-a_r + b_r) )  )) *   special.erf(np.sqrt(-1j * v_r) * a_r) **(d_power-1) )    

        corvar_limit_approx =  1/2/ (2 * np.pi/self.alpha)**d_power * (u_std**2/u_cutoff- phase_std**2/phase_cutoff) * np.sqrt(np.pi/v_r)**d_power  * np.cos(c_r - np.pi/4 * d_power)   

        if remove_bias:
            D_u = u_std ** 2 * self.alpha**d_power / 2 
            D_theta = phase_std ** 2 * self.alpha**d_power / 2
            if remove_bias == 'D': 
                corvar_limit = (corvar_limit  ) / ( (D_u-D_theta))
                corvar_limit_approx = (corvar_limit_approx  ) / ( (D_u-D_theta))
            else:    
                corvar_limit = (corvar_limit ) / ( (D_u-D_theta) / self.alpha**d_power)
                corvar_limit_approx = (corvar_limit_approx  ) / ( (D_u-D_thetaa / self.alpha**d_power))

        else:
            corvar_limit[0] += (u_std ** 2 + phase_std ** 2) / 2 
            corvar_limit_approx[0] += (u_std ** 2 + phase_std ** 2) / 2 

        return corvar_limit, corvar_limit_approx



    def plot_std_t_collection(self, rho_params_list, phase_params_list, seed_initial_condition_list, stop_t, remove_bias=False):
        rows = 2
        cols = 2
        fig, axes = plt.subplots(rows, cols, sharex=False, sharey=False, figsize=(16 * cols, 12 * rows))
        colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f']
        if remove_bias:
            ylabel = '$C^{\\mathrm{(scale)}}_{u} / r_0^2$'
        else:
            ylabel = '$\\sigma_{u} / r_0$'
        filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_d={self.d}_initial_setup={self.initial_setup}_seed={seed_initial_condition_list[-1]}_{self.rho_or_phase}_r={self.r_separation}_stop={stop_t}_state_variance_rmbias={remove_bias}.png'
        for i, rho_params in enumerate(rho_params_list):
            ax = axes[i//2, i%2]
            ax.annotate(f'({letters[i]})', xy=(0, 0), xytext=(-0.05, 1.02), xycoords='axes fraction', size=labelsize*1.3)

            simpleaxis(ax)
            title = '$\\sigma_u(0)=$' + f'{rho_params}'   + '$r_0$'
            for j, phase_params in enumerate(phase_params_list):
                distribution_params = [rho_params, phase_params]
                self.distribution_params = distribution_params
                color = colors[j]

                label = '$\\sigma_{\\theta}(0)=$' + f'{phase_params}'

                if rho_params == phase_params and remove_bias:
                    ax.plot([], [], label='ENI ' + label, linestyle='--', linewidth=2.0, color=color)
                    continue

                corvar_list, corvar_mean, t = self.corvar_t(seed_initial_condition_list, remove_bias)
                corvar_ensemble, corvar_limit, corvar_limit_approx = self.corvar_t_ensemble_approx(distribution_params, remove_bias )
                stop_index = np.where(t<stop_t)[0][-1] + 1
                if remove_bias:
                    plot_mean = corvar_mean
                    plot_list = corvar_list
                    plot_ensemble = corvar_ensemble
                    plot_limit = corvar_limit
                    plot_limit_approx = corvar_limit_approx
                    ax.plot(t[:stop_index], plot_mean[:stop_index], label='ENI ' + label, linestyle='--', linewidth=2.0, color=color)
                else:
                    plot_mean = np.sqrt(corvar_mean)
                    plot_list = np.sqrt(corvar_list)
                    plot_ensemble = np.sqrt(corvar_ensemble)
                    plot_limit = np.sqrt(corvar_limit)
                    plot_limit_approx = np.sqrt(corvar_limit_approx)
                    ax.plot(t[:stop_index], plot_mean[:stop_index], label='ENI ' + label, linestyle='--', linewidth=2.0, color=color)
                    #ax.plot(t[:stop_index], plot_list[:, :stop_index].transpose(), linestyle='--', linewidth=1.0, alpha=0.5, color=color)
                    ax.plot(t[:stop_index], plot_limit[:stop_index], linestyle='-', label='LLSSL', linewidth=2.0, color=color)
                    ax.plot(t[:stop_index], plot_limit_approx[:stop_index], linestyle = 'dotted', label='LLSSCL', linewidth=3.0, color=color)
                label_position = (1.48, 0.01)
            if remove_bias:
                ax.plot(t[:stop_index], plot_limit[:stop_index], linestyle='-', label='LLSSL', linewidth=2.0, color='grey')
                ax.plot(t[:stop_index], plot_limit_approx[:stop_index], linestyle = 'dotted', label='LLSSCL', linewidth=3.0, color='k')
            xlabel = '$t$'
            ax.set_xlabel(xlabel, size=labelsize*1.3)
            ax.set_ylabel(ylabel, size=labelsize*1.3)
            ax.set_title(title, size=labelsize+2)
            ax.yaxis.get_offset_text().set_fontsize(labelsize*0.87)
            ax.tick_params(axis='both', labelsize=labelsize)

        ax.legend(fontsize=labelsize*0.9, frameon=False, loc=4, bbox_to_anchor= label_position) 
        #fig.text(x=0.02, y=0.5, horizontalalignment='center', s=ylabel, size=labelsize*1.3, rotation=90)
        #fig.text(x=0.5, y=0.04, horizontalalignment='center', s=xlabel, size=labelsize*1.3)
        fig.subplots_adjust(left=0.1, right=0.85, wspace=0.25, hspace=0.25, bottom=0.13, top=0.90)
        save_des = '../transfer_figure/' + filename
        plt.savefig(save_des, format='png')
        plt.close()


    def plot_std_deltax_collection(self, alpha_list, rho_params, phase_params_list, seed_initial_condition_list, stop_t, scaling=False, D_sigma='D', remove_bias=False):
        rows = 1
        cols = 2
        fig, axes = plt.subplots(rows, cols, sharex=False, sharey=False, figsize=(16 * cols, 12 * rows))
        colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f']
        for i, phase_params in enumerate(phase_params_list):
            #ax = axes[i//2, i%2]
            ax = axes[i]
            ax.annotate(f'({letters[i]})', xy=(0, 0), xytext=(-0.05, 1.02), xycoords='axes fraction', size=labelsize*1.3)
            simpleaxis(ax)
            if D_sigma == 'D':
                title = '$\\sigma_{\\theta}(0)=$' + f'{phase_params}'  + '$/\\sqrt{\\Delta x}$'
            else:
                title = '$\\sigma_{\\theta}(0)=$' + f'{phase_params}'  
            for j, alpha in enumerate(alpha_list):
                self.alpha = alpha
                if D_sigma == 'D':
                    if phase_params == 0:
                        distribution_params = [round(rho_params / np.sqrt(self.alpha), 10), 0]
                    else:
                        distribution_params = [round(rho_params / np.sqrt(self.alpha), 10), round(phase_params / np.sqrt(self.alpha), 10)]
                else:
                    distribution_params = [rho_params, phase_params]
                self.distribution_params = distribution_params
                color = colors[j]
                label = '$\\sigma_{\\theta}(0)=$' + f'{phase_params}'
                label = '$\\Delta x=$' + f'{alpha}'
                if self.r_separation == 0:
                    ylabel = '$\\sigma^2_{u} / r_0^2$'
                else:
                    ylabel = '$C_u / r_0 ^2$'

                corvar_list, corvar_mean, t = self.corvar_t(seed_initial_condition_list, remove_bias)
                corvar_ensemble, corvar_limit, corvar_limit_approx = self.corvar_t_ensemble_approx(distribution_params, remove_bias)
                    
                filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_d={self.d}_initial_setup={self.initial_setup}_seed={seed_initial_condition_list[-1]}_{self.rho_or_phase}_r={self.r_separation}_stop={stop_t}_state_std_dx={alpha_list}_scaling={scaling}_{D_sigma}_fixed_rmbias={remove_bias}.png'
                if scaling:
                    stop_index = np.where(t/self.alpha**2<stop_t)[0][-1] + 1
                else:
                    stop_index = np.where(t<stop_t)[0][-1] + 1
                             
                plot_mean = corvar_mean
                plot_list = corvar_list
                plot_ensemble = corvar_ensemble
                plot_limit = corvar_limit
                plot_limit_approx = corvar_limit_approx
                if scaling:
                    ax.plot(t[:stop_index]/self.alpha**2, plot_mean[:stop_index], label='ENI ' + label, linestyle='--', linewidth=2.0, color=color)
                    #ax.plot(t[:stop_index], plot_list[:, :stop_index].transpose(), linestyle='--', linewidth=1.0, alpha=0.5, color=color)
                    ax.plot(t[:stop_index]/self.alpha**2, plot_limit[:stop_index], linestyle='-', label='LLSSL', linewidth=2.0, color=color)
                    ax.plot(t[:stop_index][::max(1, int(self.alpha**2))]/self.alpha**2, plot_limit_approx[:stop_index][::max(1, int(self.alpha**2))], marker = '*', markersize=20, label='LLSSCL', linewidth=1.0, color=color)
                    xlabel = '$t/\\Delta x ^2$'
                else:
                    ax.plot(t[:stop_index], plot_mean[:stop_index], label='ENI ' + label, linestyle='--', linewidth=2.0, color=color)
                    ax.plot(t[:stop_index], plot_limit[:stop_index], linestyle='-', label='LLSSL', linewidth=2.0, color=color)
                    if remove_bias == 'Dx' or remove_bias == False:
                        ax.plot(t[:stop_index], plot_limit_approx[:stop_index], marker = '*', markersize=20, label='LLSSCL', linewidth=1.0, color=color)
                    xlabel = '$t$'
                
                label_position = (1.48, 0.01)
            if remove_bias == 'D' and scaling==False:
                ax.plot(t[1:stop_index], plot_limit_approx[1:stop_index], marker = '*', markersize=20, label='LLSSCL', linewidth=1.0, color='k')
            ax.set_xlabel(xlabel, size=labelsize*1.3)
            ax.set_ylabel(ylabel, size=labelsize*1.3)
            ax.set_title(title, size=labelsize+2)
            ax.yaxis.get_offset_text().set_fontsize(labelsize*0.87)
            ax.tick_params(axis='both', labelsize=labelsize)
            if D_sigma == 'sigma':
                #ax.set_ylim(plot_limit.min() * 0.92, plot_limit.max() * 1.05)
                pass

        ax.legend(fontsize=labelsize*0.9, frameon=False, loc=4, bbox_to_anchor= label_position) 
        #fig.text(x=0.02, y=0.5, horizontalalignment='center', s=ylabel, size=labelsize*1.3, rotation=90)
        #fig.text(x=0.5, y=0.04, horizontalalignment='center', s=xlabel, size=labelsize*1.3)
        fig.subplots_adjust(left=0.1, right=0.85, wspace=0.25, hspace=0.25, bottom=0.13, top=0.90)
        save_des = '../transfer_figure/' + filename
        plt.savefig(save_des, format='png')
        plt.close()



    def plot_std_t_disorder(self, d_list, rho_params_list, phase_params_list, seed_initial_condition_list, stop_t, remove_bias):
        rows = len(rho_params_list)
        cols = len(phase_params_list)
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 3.5 * rows))
        colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f']
        for i, rho_params in enumerate(rho_params_list):
            for j, phase_params in enumerate(phase_params_list):
                ax = axes[i, j]
                ax.annotate(f'({letters[i * len(rho_params_list) + j]})', xy=(0, 0), xytext=(-0.12, 1.02), xycoords='axes fraction', size=20)
                simpleaxis(ax)
                title = '$\\sigma_u(0)=$' + f'{rho_params}'   + '$r_0$' + ', ' + '$\\sigma_{\\theta}(0)=$' + f'{phase_params}' 
                distribution_params = [rho_params, phase_params]
                self.distribution_params = distribution_params

                for k, d in enumerate(d_list):
                    color = colors[k]
                    label = f'$\\phi={d}$'
                    if d == 1:
                        self.network_type = self.network_type[:2]
                        self.d = 4
                    else:
                        self.network_type = self.network_type[:2] + '_disorder'
                        self.d = d
                    corvar_list, corvar_mean, t = self.corvar_t(seed_initial_condition_list, remove_bias)
                    stop_index = np.where(t<stop_t)[0][-1] + 1
                    if remove_bias:
                        plot_mean = corvar_mean
                        plot_list = corvar_list
                    else:
                        plot_mean = np.sqrt(corvar_mean)
                        plot_list = np.sqrt(corvar_list)
                    ax.plot(t[:stop_index], plot_mean[:stop_index], label=label, linestyle='-', linewidth=2.0, color=color)
                    #ax.plot(t[:stop_index], plot_list[:, :stop_index].transpose(), linestyle='--', linewidth=1.0, alpha=0.5, color=color)
                    label_position = (1.58, -0.11)
                ax.set_title(title, size=labelsize*0.48, x=0.56, y=1.01)
                ax.yaxis.get_offset_text().set_fontsize(labelsize*0.5)
                ax.tick_params(axis='both', labelsize=labelsize *0.5)
        if remove_bias:
            ylabel = '$C_u^{\\mathrm{(scale)}} / r_0 ^2$'
        else:
            ylabel = '$\\sigma_{u} / r_0$'
        self.network_type = self.network_type[:2] + '_disorder'
        filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_d={self.d}_initial_setup={self.initial_setup}_seed={seed_initial_condition_list[-1]}_{self.rho_or_phase}_r={self.r_separation}_stop={stop_t}_state_std_rmbias={remove_bias}.png'

        xlabel = '$t$'
        ax.legend(fontsize=labelsize*0.50, frameon=False, loc=4, bbox_to_anchor= label_position) 
        fig.text(x=0.03, y=0.5, horizontalalignment='center', s=ylabel, size=labelsize*0.8, rotation=90)
        fig.text(x=0.5, y=0.04, horizontalalignment='center', s=xlabel, size=labelsize*0.8)
        fig.subplots_adjust(left=0.1, right=0.90, wspace=0.25, hspace=0.25, bottom=0.10, top=0.90)
        save_des = '../transfer_figure/' + filename
        plt.savefig(save_des, format='png')
        plt.close()

    def plot_std_r_collection(self, rho_params, phase_params_list, seed_initial_condition_list, stop_r, t_list, remove_bias):
        rows = len(phase_params_list)
        cols = len(t_list)
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey='row', figsize=(4 * cols, 3.5 * rows))
        colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f']

        if remove_bias:
            ylabel = '$C_u^{\\mathrm{(scale)}} / r_0^2$'
        else:
            ylabel = '$C_u / r_0^2$'
        xlabel = '$r$'
        if self.network_type == '1D':
            N_x = int(self.N // 2+1)
        elif self.network_type in ['2D', '2D_disorder']:
            N_x = round(np.sqrt(self.N)//2+1)
        elif self.network_type in ['3D', '3D_disorder']:
            N_x = round(self.N**(1/3)//2+1)



        for i, phase_params in enumerate(phase_params_list):
            for j, t_i in enumerate(t_list):
                ax = axes[i, j]
                ax.annotate(f'({letters[i * cols + j]})', xy=(0, 0), xytext=(-0.12, 1.05), xycoords='axes fraction', size=labelsize*0.6)
                simpleaxis(ax)
                title = '$\\sigma_{\\theta}(0)=$' + f'{phase_params}'  +  ', ' + f'$t={t_i}$'
                distribution_params = [rho_params, phase_params]
                self.distribution_params = distribution_params
                corvar_list, corvar_mean, t = self.corvar_r(seed_initial_condition_list, t_i, remove_bias)
                corvar_limit, corvar_limit_approx = self.corvar_r_ensemble_approx(distribution_params, t_i, remove_bias)

                r = np.arange(0, N_x, 1)
                stop_index = np.where(r<stop_r)[0][-1] 
                plot_mean = corvar_mean
                plot_list = corvar_list
                plot_limit = corvar_limit
                plot_limit_approx = corvar_limit_approx
                if remove_bias:
                    ax.plot(r[:stop_index], plot_mean[:stop_index], label='ENI ' , linestyle='--', linewidth=2.0, color=colors[0])
                    #ax.plot(r[:stop_index], plot_list[:, :stop_index].transpose(), linestyle='--', linewidth=1.0, alpha=0.5, color=color)
                    ax.plot(r[:stop_index], plot_limit[:stop_index], linestyle='-', label='LLSSL', linewidth=2.0, color=colors[1])
                else:
                    ax.plot(r[:stop_index], plot_mean[:stop_index], label='ENI ' , linestyle='--', linewidth=2.0, color=colors[0])
                    #ax.plot(r[:stop_index], plot_list[:, :stop_index].transpose(), linestyle='--', linewidth=1.0, alpha=0.5, color=color)
                    ax.plot(r[:stop_index], plot_limit[:stop_index], linestyle='-', label='LLSSL', linewidth=2.0, color=colors[1])
                    #ax.plot(r[:stop_index], plot_limit_approx[:stop_index], linestyle = 'dotted', label='LLSSCL', linewidth=2.0, color=colors[2])
                label_position = (1.65, 0.05)
                xlabel = '$r$'
                #ax.set_xlabel(xlabel, size=labelsize*1.3)
                #ax.set_ylabel(ylabel, size=labelsize*1.3)
                ax.set_title(title, size=labelsize*0.48, x=0.56, y=1.02)
                ax.yaxis.get_offset_text().set_fontsize(labelsize*0.5)
                ax.tick_params(axis='both', labelsize=labelsize *0.5)




        ax.legend(fontsize=labelsize*0.48, frameon=False, loc=4, bbox_to_anchor= label_position) 

        fig.text(x=0.02, y=0.5, horizontalalignment='center', s=ylabel, size=labelsize*0.8, rotation=90)
        fig.text(x=0.5, y=0.04, horizontalalignment='center', s=xlabel, size=labelsize*0.8)
        fig.subplots_adjust(left=0.1, right=0.90, wspace=0.25, hspace=0.25, bottom=0.10, top=0.90)
        filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_d={self.d}_initial_setup={self.initial_setup}_seed={seed_initial_condition_list[-1]}_{self.rho_or_phase}_rho={rho_params}_r={stop_r}_state_variance_rmbias={remove_bias}.png'
        #filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_d={self.d}_initial_setup={self.initial_setup}_seed={seed_initial_condition_list[-1]}_{self.rho_or_phase}_rho={rho_params}_r={stop_r}_state_variance_all.png'
        save_des = '../transfer_figure/' + filename
        plt.savefig(save_des, format='png')
        plt.close()


    def plot_std_r_disorder(self, d_list, rho_params, phase_params_list, seed_initial_condition_list, stop_r, t_list, remove_bias):
        rows = len(phase_params_list)
        cols = len(t_list)
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey='row', figsize=(4 * cols, 3.5 * rows))
        colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f']
        if remove_bias:
            ylabel = '$C_u^{\\mathrm{(scale)}} / r_0^2$'
        else:
            ylabel = '$C_u / r_0^2$'
        xlabel = '$r$'
        if self.network_type == '1D':
            N_x = int(self.N // 2+1)
        elif self.network_type in ['2D', '2D_disorder']:
            N_x = round(np.sqrt(self.N)//2+1)
        elif self.network_type in ['3D', '3D_disorder']:
            N_x = round(self.N**(1/3)//2+1)

        for i, phase_params in enumerate(phase_params_list):
            for j, t_i in enumerate(t_list):
                ax = axes[i, j]
                ax.annotate(f'({letters[i * cols + j]})', xy=(0, 0), xytext=(-0.12, 1.05), xycoords='axes fraction', size=labelsize*0.6)
                simpleaxis(ax)
                title = '$\\sigma_{\\theta}(0)=$' + f'{phase_params}'  +  ', ' + f'$t={t_i}$'
                distribution_params = [rho_params, phase_params]
                self.distribution_params = distribution_params


                for k, d in enumerate(d_list):
                    color = colors[k]
                    label = f'$\\phi={d}$'
                    if d == 1:
                        self.network_type = self.network_type[:2]
                        self.d = 4
                    else:
                        self.network_type = self.network_type[:2] + '_disorder'
                        self.d = d

                    corvar_list, corvar_mean, t = self.corvar_r(seed_initial_condition_list, t_i, remove_bias)
                    r = np.arange(0, N_x, 1)
                    stop_index = np.where(r<stop_r)[0][-1] 
                    plot_mean = corvar_mean
                    plot_list = corvar_list
                    ax.plot(r[:stop_index], plot_mean[:stop_index], label=label, linestyle='--', linewidth=2.0, color=color)
                    label_position = (1.65, 0.05)

                ax.set_title(title, size=labelsize*0.48, x=0.56, y=1.02)
                ax.yaxis.get_offset_text().set_fontsize(labelsize*0.5)
                ax.tick_params(axis='both', labelsize=labelsize *0.5)

        self.network_type = self.network_type[:2] + '_disorder'
        filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_d={self.d}_initial_setup={self.initial_setup}_seed={seed_initial_condition_list[-1]}_{self.rho_or_phase}_rho={rho_params}_stop={stop_r}_variance_rmbias={remove_bias}.png'

        ax.legend(fontsize=labelsize*0.48, frameon=False, loc=4, bbox_to_anchor= label_position) 
        fig.text(x=0.02, y=0.5, horizontalalignment='center', s=ylabel, size=labelsize*0.8, rotation=90)
        fig.text(x=0.5, y=0.04, horizontalalignment='center', s=xlabel, size=labelsize*0.8)
        fig.subplots_adjust(left=0.1, right=0.90, wspace=0.25, hspace=0.25, bottom=0.10, top=0.90)
        save_des = '../transfer_figure/' + filename
        plt.savefig(save_des, format='png')
        plt.close()



    def corvar_t_separation(self, seed_initial_condition_list, t1_list, t2_list):
        corvar_list = []
        for j, seed_initial_condition in enumerate(seed_initial_condition_list):
            t, state = self.read_phi(seed_initial_condition)
            N_actual = len(state[0])
            r0 = np.sqrt(1/N_actual)
            if self.rho_or_phase == 'rho':
                state = np.sqrt(state) - r0
                state /= r0
            if len(t1_list) == len(t2_list):
                index_t1 = [np.where(np.abs(t - t1) < 1)[0][0] for t1 in t1_list]
                index_t2 = [np.where(np.abs(t - t2) < 1)[0][0] for t2 in t2_list]
                cor_var = np.mean( state[index_t1] * state[index_t2], axis=1)
            elif len(t1_list) == 1:
                index_t1 = np.where(np.abs(t - t1_list[0]) < 1)[0][0]
                index_t2 = [np.where(np.abs(t - t2) < 1)[0][0] for t2 in t2_list]
                cor_var = np.mean(state[index_t1] * state[index_t2], axis = 1)
            else:
                print('method not available')
            corvar_list.append(cor_var)
        corvar_list = np.vstack((corvar_list))
        corvar_mean = np.mean(corvar_list, axis=0)
        return corvar_list, corvar_mean

    def corvar_t_separation_ensemble_approx(self, distribution_params ):
        if self.initial_setup == 'u_uniform_random' or self.initial_setup == 'u_normal_random':
            u_std, phase_std = distribution_params
            u_cutoff = 1
            phase_cutoff = 1
        elif self.initial_setup == 'u_uniform_random_cutoff' or self.initial_setup == 'u_normal_random_cutoff':
            u_std, u_cutoff, phase_std, phase_cutoff = distribution_params
        a_t = np.pi / self.alpha * u_cutoff  # u and phase should be treated separatedly
        t, rho_state = self.read_phi(0)
        _, _, _, _, L, n, k, omega_k = state_fft(rho_state[0], rho_state[0], hbar, m, self.alpha)
        corvar_ensemble = (u_std**2 / u_cutoff * np.sum(np.cos(t.reshape(len(t), 1) * omega_k ) **2 * np.cos(k * self.r_separation), axis=1) + phase_std**2 / phase_cutoff * np.sum(np.sin(t.reshape(len(t), 1) * omega_k ) **2 * np.cos(k * self.r_separation), axis=1) )  / np.sqrt(N) **2 
        b_t = hbar / m / 2
        corvar_limit = ( u_std ** 2 + phase_std ** 2 ) / a_t / 4 * np.real(special.erf(a_t * np.sqrt(b_t * 1j * self.t_separation)) * np.sqrt(np.pi / b_t / 1j / self.t_separation)) +  (u_std**2 - phase_std**2) / a_t / 4 * np.real(special.erf(a_t * np.sqrt(b_t * 1j * (self.t_separation + 2*t) )) * np.sqrt(np.pi / b_t / 1j / (2*t + self.t_separation) )) 
        corvar_limit_approx = ( u_std ** 2 + phase_std ** 2 ) / a_t / 4 * np.real(np.sqrt(np.pi / b_t / 1j / self.t_separation)) +  (u_std**2 - phase_std**2) / a_t / 4 * np.real(np.sqrt(np.pi / b_t / 1j / (2*t + self.t_separation) )) 

        return corvar_ensemble, corvar_limit, corvar_limit_approx

    def plot_std_t_separation_collection(self, rho_params, phase_params_list, seed_initial_condition_list, t_list):
        rows = len(phase_params_list)
        cols = len(t_list)
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey='row', figsize=(8 * cols, 7 * rows))
        colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f']

        for i, phase_params in enumerate(phase_params_list):
            for j, t_i in enumerate(t_list):
                ax = axes[i, j]
                ax.annotate(f'({letters[i * cols + j]})', xy=(0, 0), xytext=(-0.12, 1.02), xycoords='axes fraction', size=labelsize*1.3)
                simpleaxis(ax)
                title = '$\\sigma_{\\theta}(0)=$' + f'{phase_params}'  +  ', ' + f'$t={t_i}$'
                distribution_params = [rho_params, phase_params]
                self.distribution_params = distribution_params
                ylabel = '$\\C_u(t) / r_0^2$'
                corvar_list, corvar_mean, t = self.corvar_r(seed_initial_condition_list, t_i)
                #corvar_limit, corvar_limit_approx = self.corvar_r_ensemble_approx(distribution_params, t_i)

                t_diff = np.arange(0, 1000, 10)
                t2 = t_i + t_diff
                t_diff_index = []
                stop_index = np.where(r<stop_r)[0][-1] 
                plot_mean = corvar_mean
                plot_list = corvar_list
                plot_limit = corvar_limit
                plot_limit_approx = corvar_limit_approx
                ax.plot(r[:stop_index], plot_mean[:stop_index], label='ENI ' , linestyle='--', linewidth=2.0, color=colors[0])
                #ax.plot(r[:stop_index], plot_list[:, :stop_index].transpose(), linestyle='--', linewidth=1.0, alpha=0.5, color=color)
                ax.plot(r[:stop_index], plot_limit[:stop_index], linestyle='-', label='LLSSL', linewidth=2.0, color=colors[1])
                #ax.plot(r[:stop_index], plot_limit_approx[:stop_index], linestyle = 'dotted', label='LLSSCL', linewidth=2.0, color=colors[2])
                label_position = (1.48, 0.21)
                xlabel = '$r$'
                #ax.set_xlabel(xlabel, size=labelsize*1.3)
                #ax.set_ylabel(ylabel, size=labelsize*1.3)
                ax.set_title(title, size=labelsize+1, x=0.56)
                ax.yaxis.get_offset_text().set_fontsize(labelsize*0.87)
                ax.tick_params(axis='both', labelsize=labelsize)

        ax.legend(fontsize=labelsize*0.9, frameon=False, loc=4, bbox_to_anchor= label_position) 
        fig.text(x=0.02, y=0.5, horizontalalignment='center', s=ylabel, size=labelsize*1.3, rotation=90)
        fig.text(x=0.5, y=0.04, horizontalalignment='center', s=xlabel, size=labelsize*1.3)
        fig.subplots_adjust(left=0.1, right=0.85, wspace=0.25, hspace=0.25, bottom=0.10, top=0.90)
        filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_d={self.d}_initial_setup={self.initial_setup}_seed={seed_initial_condition_list[-1]}_{self.rho_or_phase}_rho={rho_params}_r={stop_r}_state_variance.png'
        #filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_d={self.d}_initial_setup={self.initial_setup}_seed={seed_initial_condition_list[-1]}_{self.rho_or_phase}_rho={rho_params}_r={stop_r}_state_variance_all.png'
        save_des = '../transfer_figure/' + filename
        plt.savefig(save_des, format='png')
        plt.close()


            





    def plot_std_t_vs_initial_phase(self, rho_params_list, phase_params_list, seed_initial_condition_list, plot_std):
        rows = 1
        cols = 1
        fig, ax = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(16 * cols, 12 * rows))
        simpleaxis(ax)

        data_points = []
        data_points_label = []
        for i, rho_params in enumerate(rho_params_list):
            if self.initial_setup == 'uniform_random':
                rho_i, rho_f = rho_params
                if rho_i == rho_f:
                    label = '$\\rho \sim Const$'
                else:
                    label = '$\\rho \sim $' + f'$({2 * rho_i }/N, {2 * rho_f }/N)$'

            elif self.initial_setup == 'u_normal_random' or self.initial_setup == 'u_uniform_random':
                label = '$\\sigma_{u}(0)=$' + f'{rho_params[0]}' + '$r_0$'
            var_initial_list = []
            var_t_list = []
            for j, phase_params in enumerate(phase_params_list):
                
                self.distribution_params = [round(k, 10) for k in rho_params + phase_params]
                var_list, var_mean, _ = self.corvar_t(seed_initial_condition_list)
                if self.rho_or_phase == 'rho':
                    self.rho_or_phase = 'phase'
                    var_phase_list, var_phase_mean, _ = self.corvar_t(seed_initial_condition_list)
                    self.rho_or_phase = 'rho'
                    #var_initial = var_phase_mean[0]
                    var_initial = phase_params[0]**2 
                    if plot_std:
                        xlabel, ylabel = '$\\sigma_{\\theta}(0)$', '$\\sigma_{u}(\\infty)$'
                    else:
                        xlabel, ylabel = '$\\sigma^2_{\\theta}(0)$', '$\\sigma^2_{u}(\\infty)$'
                else:
                    var_initial = var_mean[0]
                    if plot_std :
                        xlabel, ylabel = '$\\sigma_{\\theta}(0)$', '$\\sigma_{\\theta}(\\infty)$'
                    else:
                        xlabel, ylabel = '$\\sigma^2_{\\theta}(0)$', '$\\sigma^2_{\\theta}(\\infty)$'

                var_t = np.mean(var_mean[500:])
                var_initial_list.append(var_initial)
                var_t_list.append(var_t)
            if plot_std:
                ax.plot(np.sqrt(var_initial_list), np.sqrt(var_t_list), 'o', markersize=10, label=label, linewidth=2.0)
                end_point = np.max(np.sqrt(var_t_list))
            else:
                data_point, = ax.plot(var_initial_list, var_t_list, 'o', markersize=13, label=label, linewidth=2.0, color=colors[i])
                data_points.append(data_point)
                data_points_label.append(label)
                var_t_theory = 1/2 * (rho_params[0]**2 + np.array([phase[0] for phase in phase_params_list]) **2)
                if i == len(rho_params_list)-1:  # only have label for the last curve
                    ax.plot(var_initial_list, var_t_theory, '--', linewidth=3.0, label='LLSSCL theory' , color=colors[i])
                else:
                    ax.plot(var_initial_list, var_t_theory, '--', linewidth=3.0, color=colors[i])
                end_point = np.max(var_t_list)
                if i == len(rho_params_list)-1:
                    lines = []
                    for blank_i in range(len(rho_params_list)):
                        line_i, = ax.plot([], [], color=colors[blank_i], linewidth=3)
                        lines.append(line_i)
                    ax.legend( data_points + [tuple(lines)], data_points_label + ['LLSSCL theory'], fontsize=legendsize*1.2, frameon=False, loc=4, bbox_to_anchor=(1.05,0.01), handler_map={tuple: HandlerTuple(ndivide=None)} ) 

   
        if self.rho_or_phase == 'phase':
            ax.plot([0, end_point], [0, end_point], '--', color='k', linewidth=2., alpha=0.7)
        #ax.legend(fontsize=legendsize *1.3, frameon=False, loc=4, bbox_to_anchor=(1.05, 0.01) ) 
        ax.tick_params(axis='both', labelsize=labelsize*0.8)
        fig.text(x=0.03, y=0.5, horizontalalignment='center', s=ylabel, size=labelsize*1.2, rotation=90)
        fig.text(x=0.5, y=0.04, horizontalalignment='center', s=xlabel, size=labelsize*1.2)
        fig.subplots_adjust(left=0.13, right=0.95, wspace=0.25, hspace=0.25, bottom=0.13, top=0.90)
        if plot_std:
            filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_seed={seed_initial_condition_list[-1]}_initial_setup={self.initial_setup}_{rho_or_phase}_state_std_compare_initial_theta.png'
        else:
            filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_seed={seed_initial_condition_list[-1]}_initial_setup={self.initial_setup}_{rho_or_phase}_state_varance_compare_initial_theta.png'
        save_des = '../transfer_figure/' + filename
        plt.savefig(save_des, format='png')
        plt.close()

    def plot_std_t_vs_initial_subfigure(self, rho_params_list, phase_params_list, seed_initial_condition_list, plot_std, x_rho_or_phase, ax):

        data_points = []
        data_points_label = []
        if x_rho_or_phase == 'rho':
            label_params_list = phase_params_list
            x_params_list = rho_params_list
        else:
            label_params_list = rho_params_list
            x_params_list = phase_params_list
        for i, label_params in enumerate(label_params_list):
            var_initial_list = []
            var_t_list = []
            for j, x_params in enumerate(x_params_list):
                if x_rho_or_phase == 'rho':
                    rho_params = x_params
                    phase_params = label_params
                    var_initial = rho_params[0]**2 
                    var_t_theory = 1/2 * (phase_params[0]**2 + np.array([rho[0] for rho in rho_params_list]) **2)
                    label = '$\\sigma_{\\theta}(0)=$' + f'{label_params[0]}' 
                    if plot_std:
                        xlabel, ylabel = '$\\sigma_{u}(0) / r_0^2$', '$\\sigma_{u}(\\infty)  / r_0$'
                    else:
                        xlabel, ylabel = '$\\sigma^2_{u}(0) / r_0^2$', '$\\sigma^2_{u}(\\infty) / r_0^2$'
                else:
                    rho_params = label_params
                    phase_params = x_params
                    var_initial = phase_params[0]**2 
                    var_t_theory = 1/2 * (rho_params[0]**2 + np.array([phase[0] for phase in phase_params_list]) **2)
                    label = '$\\sigma_{u}(0)=$' + f'{label_params[0]}'  + '$r_0$'
                    if plot_std:
                        xlabel, ylabel = '$\\sigma_{\\theta}(0)$', '$\\sigma_{u}(\\infty) / r_0$'
                    else:
                        xlabel, ylabel = '$\\sigma^2_{\\theta}(0)$', '$\\sigma^2_{u}(\\infty) / r_0^2$'

                self.distribution_params = [round(k, 10) for k in rho_params + phase_params]
                var_list, var_mean, _ = self.corvar_t(seed_initial_condition_list)

                var_t = np.mean(var_mean[500:])
                var_initial_list.append(var_initial)
                var_t_list.append(var_t)
            if plot_std:
                ax.plot(np.sqrt(var_initial_list), np.sqrt(var_t_list), 'o', markersize=10, label=label, linewidth=2.0)
                end_point = np.max(np.sqrt(var_t_list))
            else:
                data_point, = ax.plot(var_initial_list, var_t_list, 'o', markersize=13, label=label, linewidth=2.0, color=colors[i])
                data_points.append(data_point)
                data_points_label.append(label)
                ax.plot(var_initial_list, var_t_theory, '--', linewidth=3.0, color=colors[i])
                end_point = np.max(var_t_list)
                if i == len(rho_params_list)-1:
                    lines = []
                    for blank_i in range(len(rho_params_list)):
                        line_i, = ax.plot([], [], color=colors[blank_i], linewidth=3)
                        lines.append(line_i)
                    ax.legend( data_points + [tuple(lines)], data_points_label + ['LLSSCL theory'], fontsize=legendsize*1.2, frameon=False, loc=4, bbox_to_anchor=(1.05,0.01), handler_map={tuple: HandlerTuple(ndivide=None)} ) 

        ax.tick_params(axis='both', labelsize=labelsize*0.8)

        ax.set_ylabel(ylabel, size=labelsize*1.2)
        ax.set_xlabel(xlabel, size=labelsize*1.2)
        return ax



    def plot_std_t_vs_initial(self, rho_params_list, phase_params_list, seed_initial_condition_list, plot_std):
        rows = 1
        cols = 2
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(16 * cols, 12 * rows))
        for i, x_rho_or_phase in enumerate(['rho', 'phase']):
            ax = axes[i]
            ax.annotate(f'({letters[i]})', xy=(0, 0), xytext=(-0.05, 1.02), xycoords='axes fraction', size=labelsize*1.3)
            simpleaxis(ax)
            self.plot_std_t_vs_initial_subfigure(rho_params_list, phase_params_list, seed_initial_condition_list, plot_std, x_rho_or_phase, ax)
            
        fig.subplots_adjust(left=0.08, right=0.95, wspace=0.25, hspace=0.25, bottom=0.13, top=0.90)
        if plot_std:
            filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_seed={seed_initial_condition_list[-1]}_initial_setup={self.initial_setup}_{rho_or_phase}_state_std_compare_initial_both.png'
        else:
            filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_seed={seed_initial_condition_list[-1]}_initial_setup={self.initial_setup}_{rho_or_phase}_state_varance_compare_initial_both.png'
        save_des = '../transfer_figure/' + filename
        plt.savefig(save_des, format='png')
        plt.close()
        return False



    def plot_std_t_vs_initial_rho(self, rho_params_list, phase_params_list, seed_initial_condition_list):
        rows = 1
        cols = 1
        fig, ax = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(16 * cols, 12 * rows))
        simpleaxis(ax)

        for i, phase_params in enumerate(phase_params_list):
            if self.initial_setup == 'uniform_random':
                phase_i, phase_f = rho_params
                if phase_i == phase_f:
                    label = '$\\rho \sim Const$'
                else:
                    label = '$\\rho \sim $' + f'$({2 * rho_i }/N, {2 * rho_f }/N)$'
            elif self.initial_setup == 'u_uniform_random':
                label = '$\\sigma_{\\theta}^{(0)}=$' + f'{phase_params[0]}'

            elif self.initial_setup == 'u_normal_random':
                label = '$\\sigma_{\\theta}^{(0)}=$' + f'{phase_params[0]}'
            std_initial_list = []
            std_t_list = []
            for j, rho_params in enumerate(rho_params_list):
                
                self.distribution_params = [round(k, 10) for k in rho_params + phase_params]
                std_list, std_mean, _ = self.corvar_t(seed_initial_condition_list)
                if self.rho_or_phase == 'phase':
                    self.rho_or_phase = 'rho'
                    std_rho_list, std_rho_mean, _ = self.corvar_t(seed_initial_condition_list)
                    self.rho_or_phase = 'phase'
                    std_initial = std_rho_mean[0]
                    xlabel, ylabel = '$\\sigma_{u}^{(0)} / r_0$', '$\\sigma_{\\theta}^{(s)}$'
                else:
                    std_initial = std_mean[0]
                    xlabel, ylabel = '$\\sigma_{u}^{(0)} / r_0$', '$\\sigma_{u}^{(s)} / r_0$'

                std_t = np.mean(std_mean[500:])
                std_initial_list.append(std_initial)
                std_t_list.append(std_t)
            """ min-max scaling
            std_t_list = np.array(std_t_list)
            std_initial_list = np.array(std_initial_list)
            std_initial_list = (std_initial_list - std_initial_list.min() ) / (std_initial_list.max() - std_initial_list.min())
            std_t_list = (std_t_list - std_t_list.min()) / (std_t_list.max() - std_t_list.min()) 
            """
            ax.plot(std_initial_list, std_t_list, 'o--', label=label, linewidth=2.0)
        end_point = np.max(std_t_list)
        if self.rho_or_phase == 'rho':
            ax.plot([0, end_point], [0, end_point], '--', color='k', linewidth=2., alpha=0.7)
        ax.legend(fontsize=legendsize, frameon=False, loc=4, bbox_to_anchor=(1.0, 0.01) ) 
        ax.tick_params(axis='both', labelsize=22)
        fig.text(x=0.02, y=0.5, horizontalalignment='center', s=ylabel, size=labelsize, rotation=90)
        fig.text(x=0.5, y=0.04, horizontalalignment='center', s=xlabel, size=labelsize)
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25, hspace=0.25, bottom=0.13, top=0.90)
        filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_seed={seed_initial_condition_list[-1]}_initial_setup={self.initial_setup}_{rho_or_phase}_state_std_compare_initial_rho.png'
        save_des = '../transfer_figure/' + filename
        plt.savefig(save_des, format='png')
        plt.close()


    # deprecated
    def plot_std_t_vs_initial_rho_deprate(self, rho_params_list, phase_params_list, seed_initial_condition_list):
        rows = 1
        cols = 1
        fig, ax = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(16 * cols, 12 * rows))
        simpleaxis(ax)

        for i, phase_params in enumerate(phase_params_list):
            phase_i, phase_f = phase_params
            if phase_i == phase_f:
                label = '$\\theta \sim Const$'
            else:
                label = '$\\theta \sim $' + f'$({phase_i} \\pi, {phase_f} \\pi)$'
            std_initial_list = []
            std_t_list = []
            for j, rho_params in enumerate(rho_params_list):
                
                self.distribution_params = [round(k, 5) for k in rho_params + phase_params]
                std_list, std_mean, _ = self.corvar_t(seed_initial_condition_list)
                std_initial = std_mean[0]
                xlabel, ylabel = '$\\sigma_{\\rho}^{(0)}$', '$\\sigma_{\\rho}^{(s)}$'

                std_t = np.mean(std_mean[500:])
                std_initial_list.append(std_initial)
                std_t_list.append(std_t)
            ax.plot(std_initial_list, std_t_list, 'o--', label=label, linewidth=2.0)
        ax.legend(fontsize=legendsize, frameon=False, loc=4, bbox_to_anchor=(1.0, 0.01) ) 
        ax.tick_params(axis='both', labelsize=22)
        fig.text(x=0.02, y=0.5, horizontalalignment='center', s=ylabel, size=labelsize, rotation=90)
        fig.text(x=0.5, y=0.04, horizontalalignment='center', s=xlabel, size=labelsize)
        fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25, hspace=0.25, bottom=0.13, top=0.90)
        filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_seed={seed_initial_condition_list[-1]}_{rho_or_phase}_state_std_compare_initial_rho.png'
        save_des = '../transfer_figure/' + filename
        plt.savefig(save_des, format='png')
        plt.close()

    # deprecated
    def std_t_from_fft(self, seed_initial_condition_list):
        """ individual standard deviation from fourier transform of actual simulation data"""
        std_list = []
        for j, seed_initial_condition in enumerate(seed_initial_condition_list):
            t, rho_state = self.read_phi(seed_initial_condition, 'rho')
            r0 = 1/np.sqrt(self.N)
            u_state = np.sqrt(rho_state) - r0  
            u_k = fft(u_state) 
            std = np.sqrt(np.sum(np.abs(u_k) ** 2, axis=1) ) / self.N

            if self.rho_or_phase == 'rho':
                std /= np.sqrt(1/self.N)
            std_list.append(std)
        std_list = np.vstack((std_list))
        std_mean = np.mean(std_list, axis=0)
        return std_list, std_mean, t


    # deprecated
    def corvar_t_individual_approx(self, seed_initial_condition_list):
        """ individual standard deviation from small fluctuation approximation"""
        corvar_list = []
        for j, seed_initial_condition in enumerate(seed_initial_condition_list):
            t, rho_state = self.read_phi(seed_initial_condition, 'rho')
            N_actual = len(rho_state[0])
            t, phase_state = self.read_phi(seed_initial_condition, 'phase')
            u_k, phase_k, a_k, b_k, L, n, k, omega_k = state_fft(rho_state[0], phase_state[0], hbar, m, self.alpha)
            cor_var = np.sum( np.abs( u_k * np.cos(t.reshape(len(t), 1) * omega_k ) + phase_k * np.sin(t.reshape(len(t), 1) * omega_k ) ) **2 * np.cos(k * self.r_separation), axis=1) / self.N_actual  ** 2

            if self.rho_or_phase == 'rho':
                cor_var /= 1/self.N
            corvar_list.append(cor_var)
        corvar_list = np.vstack((corvar_list))
        corvar_mean = np.mean(corvar_list, axis=0)
        return corvar_list, corvar_mean, t
 

    # deprecated
    def plot_std_t_separation(self, distribution_params_list, seed_initial_condition_list, stop_t):
        if self.t_separation == 0:
            rows = 1
            cols = 1
        else:
            rows = 2
            cols = 2
        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=False, figsize=(16 * cols, 12 * rows))
        colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f']
        for i, distribution_params in enumerate(distribution_params_list):
            self.distribution_params = distribution_params
            color = colors[i]
            if self.initial_setup == 'uniform_random': 
                rho_i, rho_f = distribution_params[:2]
                phase_i, phase_f = distribution_params[2:]
                if self.rho_or_phase == 'rho':
                    if rho_i == rho_f:
                        label = '$\\rho \sim Const$'
                    else:
                        label = '$\\rho \sim $' + f'$({2 * rho_i}/N , {2* rho_f}/N)$'

                    if phase_i == phase_f:
                        title = '$\\theta \sim Const$'
                    else:
                        title = '$\\theta \sim $' + f'$({phase_i} \\pi, {phase_f} \\pi)$'
                    ylabel = '$\\sigma_{\\rho}$'
                    filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_phase={distribution_params[2:]}_seed={seed_initial_condition_list[-1]}_{self.rho_or_phase}_state_std.png'

                else:
                    if phase_i == phase_f:
                        label = '$\\theta \sim Const$'
                    else:
                        label = '$\\theta \sim $' + f'$({phase_i} \\pi, {phase_f} \\pi)$'
                    if rho_i == rho_f:
                        title = '$\\rho \sim Const$'
                    else:
                        title = '$\\rho \sim $' + f'$({2 * rho_i}/N , {2* rho_f}/N)$'
                    ylabel = '$\\sigma_{\\theta}$'
                    filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_initial_setup={self.initial_setup}_rho={distribution_params[:2]}_seed={seed_initial_condition_list[-1]}_{self.rho_or_phase}_state_std.png'

            elif self.initial_setup == 'u_uniform_random' or self.initial_setup == 'u_normal_random':
                u_std, phase_std = distribution_params
                label = '$\\sigma_u^{(0)}=$' + f'{u_std}' + '$r_0$, ' + '$\\sigma_{\\theta}^{(0)}=$' + f'{phase_std}' + '$r_0$'
                title = ''
                if self.r_separation == 0:
                    ylabel = '$\\sigma_{u} / r_0$'
                else:
                    ylabel = 'Cor' + '$_u / r_0 ^2$'
                filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_initial_setup={self.initial_setup}_rho={distribution_params}_seed={seed_initial_condition_list[-1]}_{self.rho_or_phase}_t={self.t_separation}_stop={stop_t}_state_std.png'

            elif self.initial_setup == 'u_uniform_random_cutoff' or self.initial_setup == 'u_normal_random_cutoff':
                u_std, u_cutoff, phase_std, phase_cutoff = distribution_params
                label = '$\\sigma_u^{(0)}=$' + f'{u_std}' + '$r_0$, ' + '$\\sigma_{\\theta}^{(0)}=$' + f'{phase_std}' + '$r_0$'
                title = ''
                if self.r_separation == 0:
                    ylabel = '$\\sigma_{u} / r_0$'
                else:
                    ylabel = 'Cor' + '$_u / r_0 ^2$'
                filename = f'quantum={self.quantum_or_not}_network={self.network_type}_N={self.N}_initial_setup={self.initial_setup}_rho={distribution_params}_seed={seed_initial_condition_list[-1]}_{self.rho_or_phase}_t={self.t_separation}_stop={stop_t}_state_std.png'


            corvar_list, corvar_mean, t, t_simulation = self.corvar_t_separation(seed_initial_condition_list)
            corvar_list_approx, corvar_mean_approx, _ = self.corvar_t_individual_approx(seed_initial_condition_list)
            corvar_ensemble, corvar_limit, corvar_limit_approx = self.corvar_t_separation_ensemble_approx(distribution_params )
            if self.t_separation == 0:
                ax = axes
                simpleaxis(ax)
                labelsize = 20
                stop_index = np.where(t>stop_t)[0][0]
                plot_mean = np.sqrt(corvar_mean)
                plot_list = np.sqrt(corvar_list)
                plot_mean_approx = np.sqrt(corvar_mean_approx)
                plot_ensemble = np.sqrt(corvar_ensemble)
                plot_limit = np.sqrt(corvar_limit)
                plot_limit_approx = np.sqrt(corvar_limit_approx)
                ax.plot(t[:stop_index], plot_mean[:stop_index], label=label, linestyle='--', linewidth=2.0, color=color)
                #ax.plot(t[:stop_index], plot_list[:, :stop_index].transpose(), linestyle='--', linewidth=1.0, alpha=0.5, color=color)
                ax.plot(t[:stop_index], plot_limit[:stop_index], linestyle='-', label='continuous limit', linewidth=2.0, color=color)
                ax.plot(t[:stop_index], plot_limit_approx[:stop_index], linestyle = 'dotted', label='infinite size', linewidth=2.0, color=color)
                ax.plot(t[:stop_index], plot_mean_approx[:stop_index], linestyle = 'dotted', label='linearization', linewidth=1.0, color='k')
                label_position = (1.29, 0.01)
            else:
                ax = axes[i//cols, i% cols]
                simpleaxis(ax)
                labelsize = 20 *1.8
                stop_index = np.where(t>stop_t)[0][0]
                plot_mean = corvar_mean
                plot_list = corvar_list
                plot_mean_approx = corvar_mean_approx
                plot_ensemble = corvar_ensemble
                plot_limit = corvar_limit
                plot_limit_approx = corvar_limit_approx
                ax.plot(t_simulation[:-1], plot_mean[:], label='simulation', linestyle='--', linewidth=2.0, color=colors[0])
                #ax.plot(t[:stop_index], plot_list[:, :stop_index].transpose(), linestyle='--', linewidth=1.0, alpha=0.5, color=colors[0])
                ax.plot(t[:], plot_limit[:], linestyle='-', label='continuous limit', linewidth=2.0, color=colors[1])
                #ax.plot(t[:stop_index+self.t_separation], plot_mean_approx[:stop_index], linestyle = '--', label='linearization', linewidth=2.0, color=colors[2])
                #ax.plot(t[:stop_index], plot_limit_approx[:stop_index], linestyle = '--', label='infinite size', linewidth=2.0, color='k')
                title = '$\\sigma_u^{(0)}=$' + f'{u_std}' + '$r_0$, ' + '$\\sigma_{\\theta}^{(0)}=$' + f'{phase_std}' + '$r_0$'
                label_position = (1.62, 0.01)
            ax.set_title(title, size=labelsize+2)
            ax.yaxis.get_offset_text().set_fontsize(labelsize*0.87)
            ax.tick_params(axis='both', labelsize=labelsize)

        xlabel = '$t$'
        ax.legend(fontsize=labelsize*1, frameon=False, loc=4, bbox_to_anchor= label_position) 
        fig.text(x=0.02, y=0.5, horizontalalignment='center', s=ylabel, size=labelsize*1, rotation=90)
        fig.text(x=0.5, y=0.04, horizontalalignment='center', s=xlabel, size=labelsize*1)
        fig.subplots_adjust(left=0.1, right=0.80, wspace=0.25, hspace=0.25, bottom=0.13, top=0.90)
        save_des = '../transfer_figure/' + filename
        plt.savefig(save_des, format='png')
        plt.close()






hbar = 0.6582
m = 5.68
letters = 'abcdefghijklmnopqrstuvwxyz'


if __name__ == '__main__':
    quantum_or_not = True
    initial_setup = 'uniform_random'

    network_type = '2D_disorder'
    N = 10000
    d = 0.51
    network_type = '3D'
    N = 8000
    d = 4
    network_type = '1D'
    N = 10000
    d = 4
    network_type = '2D'
    N = 10000
    d = 4

    seed = 0
    alpha = 1
    dt = 1
    seed_initial_condition_list = np.arange(0, 10, 1)
    distribution_params = [1, 1, -1, 1]
    rho_or_phase = 'phase'
    rho_or_phase = 'rho'

    rho_list = [[1, 1], [3/8, 5/8], [1/4, 3/4], [0, 1]]
    phase_list = [[-1, 1], [-7/8, 7/8], [-3/4, 3/4], [-5/8, 5/8], [-1/2, 1/2], [-3/8, 3/8], [-1/4, 1/4], [-1/8, 1/8], [-1/16, 1/16], [-1/32, 1/32], [0, 0]]

    initial_setup = "u_uniform_random_cutoff"
    rho_list = [ [0.05, 0.1]]
    phase_list = [[0, 0.1], [0.05, 0.1], [0.1, 0.1], [0.2, 0.1]]

    initial_setup = "u_uniform_random"
    initial_setup = "u_normal_random"
    rho_list = [ [0], [0.05], [0.1], [0.2]]
    rho_list = [ [0], [0.05], [0.1], [0.2], [0.3]]
    phase_list = [[0], [0.05], [0.1], [0.2]]
    phase_list = [[0], [0.05], [0.1], [0.2], [0.3]]


    r_separation = 10
    t_separation = 10
    psv = plotStateVariance(quantum_or_not, network_type, N, d, seed, alpha, dt, initial_setup, distribution_params, seed_initial_condition_list, rho_or_phase, r_separation, t_separation)


    stop_t_list = [100]
    phase_list_list = [phase_list]
    r_separation_list = [0]
    t_separation_list = [10, 100, 500]
    for phase_list in phase_list_list:
        distribution_params_raw = [rho + phase for rho in rho_list for phase in phase_list]
        distribution_params_list = []
        for i in distribution_params_raw:
            distribution_params_list.append( [round(j, 10) for j in i])

        for stop_t in stop_t_list:
            for r_separation in r_separation_list:
                psv.r_separation = r_separation
                #psv.plot_std_t(distribution_params_list, seed_initial_condition_list, stop_t)

            for t_separation in t_separation_list:
                psv.t_separation = t_separation
                #psv.plot_std_t_separation(distribution_params_list, seed_initial_condition_list, stop_t)
    
    psv.N = 8000
    psv.network_type = '3D_disorder'
    for plot_std in [True, False]:
        #psv.plot_std_t_vs_initial(rho_list, phase_list, seed_initial_condition_list, plot_std)
        pass

    #psv.plot_std_t_vs_initial_rho(rho_list, phase_list, seed_initial_condition_list)

    rho_list = [0, 0.05, 0.1, 0.2]
    phase_list = [0, 0.05, 0.1, 0.2]
    psv.network_type = '1D'
    psv.N = 10000
    remove_bias = False
    remove_bias = 'D'




    "variance vs r snapshots"
    psv.network_type = '1D'
    psv.d = 4
    psv.N = 10000
    network_type_list = ['1D', '2D', '3D']
    d_list = [4, 4, 4] 
    N_list = [10000, 10000, 8000]
    rho_params = 0.05
    phase_params_list =[0, 0.05, 0.1, 0.2]
    phase_params_list =[0, 0.1, 0.2]

    remove_bias = 'D'
    t_list = [1, 5, 10, 100]
    stop_r_list_list = [[50], [20], [20]]
    stop_t_list = [50]
    for network_type, N, d, stop_r_list in zip(network_type_list, N_list, d_list, stop_r_list_list):
        psv.network_type = network_type
        psv.N = N
        psv.d = d
        for stop_r in stop_r_list:
            #psv.plot_std_r_collection(rho_params, phase_params_list, seed_initial_condition_list, stop_r, t_list, remove_bias)
            pass
        for stop_t in stop_t_list:
            psv.plot_std_t_collection(rho_list, phase_list, seed_initial_condition_list, stop_t, remove_bias)
            pass

    "Correlation function of t and r for disordered lattices"
    network_type_list = ['2D_disorder', '3D_disorder']
    d_list_list = [[0.51, 0.55, 0.7, 0.9, 1], [0.3, 0.5, 0.7, 1]]
    N_list = [10000, 8000]
    stop_r_list_list = [[20], [20]]
    stop_t_list = [100]
    for network_type, N, d_list, stop_r_list in zip(network_type_list, N_list, d_list_list, stop_r_list_list):
        psv.network_type = network_type
        psv.N = N
        for stop_r in stop_r_list:
            #psv.plot_std_r_disorder(d_list, rho_params, phase_params_list, seed_initial_condition_list, stop_r, t_list, remove_bias)
            pass
        for stop_t in stop_t_list:
            #psv.plot_std_t_disorder(d_list, rho_list, phase_list, seed_initial_condition_list, stop_t, remove_bias)
            pass


    "Delta x effect"
    psv.network_type = '1D'
    psv.N = 1000
    psv.d = 4
    alpha_list = [0.5, 1, 2, 3]

    rho_params = 0.05
    phase_params_list =[0, 0.1]
    stop_t = 50
    D_sigma = 'sigma'
    for scaling in [True, False]:
        for remove_bias in ['D', 'Dx']:
            #psv.plot_std_deltax_collection(alpha_list, rho_params, phase_params_list, seed_initial_condition_list, stop_t, scaling, D_sigma, remove_bias=remove_bias)
            pass
