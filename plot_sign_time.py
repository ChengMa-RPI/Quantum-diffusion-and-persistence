import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.path.insert(1, '/home/mac/RPI/research/')

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
from netgraph import Graph
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

class plotSignTime():
    def __init__(self, quantum_or_not, network_type, N, d, seed, alpha, initial_setup, seed_initial_condition_list, reference_line):
        self.quantum_or_not = quantum_or_not
        self.network_type = network_type
        self.N = N
        self.d = d
        self.seed = seed
        self.alpha = alpha
        self.initial_setup = initial_setup
        self.seed_initial_condition_list = seed_initial_condition_list
        self.reference_line = reference_line

    def read_meta_data(self, seed_initial_condition):
        if self.quantum_or_not:
            des = '../data/quantum/meta_data/' + self.network_type + '/' 
        else:
            des = '../data/classical/meta_data/' + self.network_type + '/' 
        filename = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_setup={self.initial_setup}_reference={self.reference_line}_seed_initial={seed_initial_condition}.json'
        f = open(filename)
        meta_data = json.load(f)
        return meta_data

    def get_above_interval_time(self, seed_initial_condition, tau):
        meta_data = self.read_meta_data(seed_initial_condition)
        t_start, t_end, dt = meta_data['t']
        cross_meta_data = meta_data['meta_data']
        nodes = cross_meta_data.keys()
        N_actual = len(nodes)
        all_above_intervals = []
        all_above_time = []
        for node in nodes:
            above_start = cross_meta_data[node]['above_start']
            below_start = cross_meta_data[node]['below_start']
            above_interval = []
            above_time = 0
            if above_start[0] == 0 and len(above_start) > len(below_start):
                below_start = below_start[:-1]
            elif below_start[0] == 0:
                if len(above_start) ==  len(below_start):
                    above_start = above_start[:-1] 
                below_start = below_start[1:]

            for i, j in zip(above_start, below_start):
                if j > tau and i < tau:
                    above_interval.append([i, tau])
                    above_time +=  tau - i
                elif j < tau:
                    above_interval.append([i, j - dt])
                    above_time += j - dt - i
            all_above_intervals.append(above_interval)
            all_above_time.append(above_time)
        return all_above_intervals, all_above_time
    
    def plot_sign_time_distribution(self, tau):
        collector = []
        for seed_initial_condition in self.seed_initial_condition_list:
            all_above_intervals, all_above_time = self.get_above_interval_time(seed_initial_condition, tau)
            all_below_time = tau - np.array(all_above_time)
            collector.extend(all_below_time)
        count, bins = np.histogram(collector, bins = 100)
        y = count / self.N / len(self.seed_initial_condition_list)
        x = bins[:-1] / tau * (1 - bins[:-1] / tau)
        plt.loglog(x, y, '.')
        return collector




    def read_dpp(self):
        """TODO: Docstring for read_dpp.

        :arg1: TODO
        :returns: TODO

        """
        if self.quantum_or_not:
            des = '../data/quantum/' + self.network_type + '/' 
        else:
            des = '../data/classical/' + self.network_type + '/' 
        PA = []
        PB = []
        for seed_initial_condition in self.seed_initial_condition_list:
            filename = f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_seed_initial={seed_initial_condition}_setup={self.initial_setup}_reference={self.reference_line}.csv'
            data = np.array(pd.read_csv(des + filename, header=None))
            t, pa, pb = data[1:, 0], data[1:, 1], data[1:, 2]
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



        


if __name__ == '__main__':
    quantum_or_not = False
    network_type = '2D'
    N = 100 
    d = 4
    seed = 0
    alpha = 1
    initial_setup = 'rho_uniform_phase_uniform'
    initial_setup = 'uniform_random'
    reference_line = 0.8
    reference_line = 'average'
    seed_initial_condition_list = np.arange(0, 10, 1)
    pst = plotSignTime(quantum_or_not, network_type, N, d, seed, alpha, initial_setup, seed_initial_condition_list, reference_line)
    #pdpp.plot_dpp_t()
    L_list = np.arange(10, 40, 50)
    N_list = np.power(L_list, 2)
    #N_list = [100]

    #pdpp.plot_dpp_scaling(N_list)
    reference_lines = ['average']
    tau = 1000
    collector = pst.plot_sign_time_distribution(tau)


