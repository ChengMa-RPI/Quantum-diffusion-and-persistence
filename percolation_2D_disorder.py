import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.path.insert(1, '/home/mac6/RPI/research/')
from mutual_framework import network_generate, disorder_lattice_clusters

import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
import pandas as pd 
from scipy.linalg import inv as spinv
import networkx as nx

color_list = ['#fc8d62',  '#66c2a5', '#e78ac3', '#a6d854',  '#8da0cb', '#ffd92f','#b3b3b3', '#e5c494', '#7fc97f', '#beaed4', '#ffff99']

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def percolation_removal_rate(network_type, N_list, d_list, seed_list):
    survive_rate = np.zeros((len(N_list), len(d_list), len(seed_list) ))
    for i, N in enumerate(N_list):
        for j, d in enumerate(d_list):
            for k, seed in enumerate(seed_list):
                A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d) 
                survive_rate[i, j, k] = len(A) / N
    return survive_rate

def plot_R_phi(network_type, N_list, d_list, seed_list, color_list):
    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(4 * cols, 3.5 * rows))
    survive_rate = percolation_removal_rate(network_type, N_list, d_list, seed_list)
    for i, N in enumerate(N_list):
        ax = axes[i//cols, i%cols]
        simpleaxis(ax)
        ax.plot(d_list, survive_rate[i], color=color_list[i])
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_title(f'$N={N}$', fontsize=15)
    ax = axes[-1, -1]
    simpleaxis(ax)
    for i, N in enumerate(N_list):
        ax.plot(d_list, np.mean(survive_rate[i], axis=1), label=f'$N={N}$', color=color_list[i])
        ax.legend(fontsize=15, frameon=False, loc=4, bbox_to_anchor=(1.39, -0.05)) 
    
    ax.tick_params(axis='both', which='major', labelsize=16)
    fig.text(x=0.03, y=0.5, horizontalalignment='center', s="$R$", size=18, rotation=90)
    fig.text(x=0.5, y=0.04, horizontalalignment='center', s="$\\phi$", size=18)
    fig.subplots_adjust(left=0.15, right=0.9, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)

def find_gcc_correspondings(network_type, N, seed, d):
    cluster = disorder_lattice_clusters(network_type, N, seed, d)
    des = '../data/matrix_save/disorder_corresponding/'
    if not os.path.exists(des):
        os.makedirs(des)
    des_file = des + f'network_type={network_type}_N={N}_d={d}_seed={seed}.csv'
    df = pd.DataFrame(cluster)
    df.to_csv(des_file, header=False, index=False)
    return False





N = 10000
seed = 0

network_type = '3D_disorder'
d_list = [0.3, 0.5, 0.7, 1]
N = 8000

network_type = '2D_disorder'
d_list = [0.51, 0.55, 0.7, 0.9]
N = 100

for d in d_list:
    find_gcc_correspondings(network_type, N, seed, d)

t1 = time.time()
#plot_R_phi(network_type, N_list, d_list, seed_list, color_list)
t2 = time.time()
print(t2 - t1)
