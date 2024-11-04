import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.path.insert(1, '/home/mac6/RPI/research/')
from mutual_framework import network_generate

import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
import pandas as pd 
from scipy.linalg import inv as spinv
import networkx as nx
import json




class persistenceAnalysis:
    def __init__(self, quantum_or_not, network_type, m, N, d, seed, alpha, dt, initial_setup, distribution_params, reference_line, rho_or_phase, full_data_t=False):
        """TODO: Docstring for __init__.

        :quantum_not: TODO
        :network_type: TODO
        :N: TODO
        :d: TODO
        :seed: TODO
        :alpha: diffusion ratio
        :: TODO
        :returns: TODO

        """
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
        self.reference_line = reference_line
        self.seed_initial_condition = None
        self.rho_or_phase = rho_or_phase
        self.full_data_t = full_data_t

    def read_phi(self, seed_initial_condition):
        if self.quantum_or_not:
            if self.rho_or_phase == 'rho':
                des = '../data/quantum/state/' + self.network_type + '/' 
            elif self.rho_or_phase == 'phase':
                des = '../data/quantum/phase/' + self.network_type + '/' 

        else:
            des = '../data/classical/state/' + self.network_type + '/' 
        if self.full_data_t:    
            if self.m == m_e:    
                save_file = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_seed_initial={seed_initial_condition}_full.npy'
            else:
                save_file = des + f'm={self.m}_N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_seed_initial={seed_initial_condition}_full.npy'
        else:
            if self.m == m_e:    
                save_file = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_seed_initial={seed_initial_condition}.npy'
            else:
                save_file = des + f'm={self.m}_N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_seed_initial={seed_initial_condition}.npy'
        data = np.load(save_file)
        t, state = data[:, 0], data[:, 1:]
        return t, state

    def read_meta_data(self, seed_initial_condition):
        if self.quantum_or_not:
            if self.rho_or_phase == 'rho':
                des = '../data/quantum/meta_data/' + self.network_type + '/' 
            elif self.rho_or_phase == 'phase':
                des = '../data/quantum/meta_data_phase/' + self.network_type + '/' 
            else:
                print('rho_or_phase arg wrong')
        else:
            des = '../data/classical/meta_data/' + self.network_type + '/' 
        if self.m == m_e:
            filename = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_reference={self.reference_line}_seed_initial={seed_initial_condition}.json'
        else:
            filename = des + f'm={self.m}_N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_reference={self.reference_line}_seed_initial={seed_initial_condition}.json'
        f = open(filename)
        meta_data = json.load(f)
        return meta_data

    def get_meta_data_dt_same(self, seed_initial_condition, save_des=None):
        t, phi_state = self.read_phi(seed_initial_condition)
        t = t[2:]
        phi_state = phi_state[2:]
        t_length = len(t)
        N_actual = len(phi_state[0])
        dt = np.round(np.mean(np.diff(t)), 5)  # this is wrong after the update, as dt is not the same 
        if self.rho_or_phase == 'rho':
            reference_value = 1/ N_actual
            if reference_line != 'average':
                reference_value *= float(reference_line)
        else:
            reference_value = self.reference_line * np.pi
        mask_above = np.heaviside(phi_state - reference_value, 0)
        cross_meta_data = {}
        for node, node_mask in enumerate(mask_above.transpose()):
            transition_index = np.where(np.diff(node_mask) != 0)[0] + 1
            padded_index = np.round(np.r_[0, transition_index * dt, t[-1]], 3)  # in terms of time, but dt the not the same
            even = padded_index[::2].tolist()
            odd = padded_index[1::2].tolist()
            if node_mask[0] > 0:
                above_start = even
                below_start = odd
            else:
                above_start = odd
                below_start = even
            cross_meta_data[node] = {'above_start' : above_start, 'below_start' : below_start}
        meta_data = {'t':[t[0], t[-1], dt], 'meta_data':cross_meta_data}  # specify t_list
        if save_des:
            with open(save_des, 'w') as fp:
                json.dump(meta_data, fp)
        return meta_data

    def diffusive_persistence_dt_same(self, seed_initial_condition, save_des=None):
        """
        needs to be simplified... na, nb should be generalized
        """
        meta_data = self.read_meta_data(seed_initial_condition)
        t_start, t_end, dt = meta_data['t']  # dt not the same
        cross_meta_data = meta_data['meta_data']
        nodes = cross_meta_data.keys()
        N_actual = len(nodes)
        all_first_below = []
        all_first_above = []
        for node in nodes:
            above_start = cross_meta_data[node]['above_start']
            below_start = cross_meta_data[node]['below_start']
            if above_start[0] == 0:
                if len(above_start) > 1:
                    first_above = above_start[1]
                else:
                    first_above = -1
                first_below = below_start[0]
            elif below_start[0] == 0:
                if len(below_start) > 1:
                    first_below = below_start[1]
                else:
                    first_below = -1
                first_above = above_start[0]
            else:
                print('where is 0 in starts')
            all_first_above.append(first_above)
            all_first_below.append(first_below)

        all_first_above = np.array(all_first_above)
        all_first_below = np.array(all_first_below)
        always_above = np.sum((all_first_above < 0) & (np.abs(all_first_below - t_end) < 1e-5) )
        always_below = np.sum((all_first_below < 0) & (np.abs(all_first_above - t_end) < 1e-5) )
        t = np.arange(t_start, t_end, dt)
        na = np.zeros(( len(t) ))
        nb = np.zeros(( len(t) ))
        is_above = (all_first_above > all_first_below)
        is_below = (all_first_above < all_first_below)
        na_candidate = all_first_below[is_above]
        nb_candidate = all_first_above[is_below]
        for i, t_i in enumerate(t):
            na[i] = np.sum(na_candidate > t_i)
            nb[i] = np.sum(nb_candidate > t_i)
        # this can be improved: sort na_candidate and save cross time.
        na += always_above
        nb += always_below
        pa = na / N_actual
        pb = nb / N_actual
        df = pd.DataFrame(np.vstack(( t, pa, pb )).transpose()) 
        if save_des:
            df.to_csv(save_des, index=None, header=None)
        return df

    def get_meta_data(self, seed_initial_condition, save_des=None):
        t, phi_state = self.read_phi(seed_initial_condition)
        average_state = np.sum(phi_state[0]) / len(phi_state[0])
        if np.max(phi_state[0]) - np.min(phi_state[0]) < 1e-10:
            start_index =1
        else:
            start_index = 0
        t = t[start_index:]
        phi_state = phi_state[start_index:]
        t_length = len(t)
        N_actual = len(phi_state[0])
        #dt = np.round(np.mean(np.diff(t)), 5)  # this is wrong after the update, as dt is not the same 
        if self.reference_line == 'average':
            reference_value = average_state
        else:
            reference_value = average_state *  float(self.reference_line)
        mask_above = np.heaviside(phi_state - reference_value, 0)
        cross_meta_data = {}
        for node, node_mask in enumerate(mask_above.transpose()):
            transition_index = np.where(np.diff(node_mask) != 0)[0] + 1
            #padded_index = np.round(np.r_[0, transition_index * dt, t[-1]], 3)  # in terms of time, but dt the not the same
            padded_index = np.r_[0, transition_index, len(t)-1]
            even = padded_index[::2].tolist()
            odd = padded_index[1::2].tolist()
            if node_mask[0] > 0:
                above_start = even
                below_start = odd
            else:
                above_start = odd
                below_start = even
            cross_meta_data[node] = {'above_start' : above_start, 'below_start' : below_start}
        #meta_data = {'t':[t[0], t[-1], dt], 'meta_data':cross_meta_data}  # specify t_list
        meta_data = {'t':t.tolist(), 'meta_data':cross_meta_data}
        if save_des:
            with open(save_des, 'w') as fp:
                json.dump(meta_data, fp)
        return meta_data

    def diffusive_persistence(self, seed_initial_condition, save_des=None):
        """
        needs to be simplified... na, nb should be generalized
        """
        meta_data = self.read_meta_data(seed_initial_condition)
        #t_start, t_end, dt = meta_data['t']  # dt not the same
        t = meta_data['t']
        cross_meta_data = meta_data['meta_data']
        nodes = cross_meta_data.keys()
        N_actual = len(nodes)
        all_first_below = []
        all_first_above = []
        for node in nodes:
            above_start = cross_meta_data[node]['above_start']
            below_start = cross_meta_data[node]['below_start']
            if above_start[0] == 0:
                if len(above_start) > 1:
                    first_above = above_start[1]
                else:
                    first_above = -1
                first_below = below_start[0]
            elif below_start[0] == 0:
                if len(below_start) > 1:
                    first_below = below_start[1]
                else:
                    first_below = -1
                first_above = above_start[0]
            else:
                print('where is 0 in starts')
            all_first_above.append(first_above)
            all_first_below.append(first_below)

        all_first_above = np.array(all_first_above)
        all_first_below = np.array(all_first_below)
        always_above = np.sum((all_first_above < 0) & (np.abs(all_first_below - len(t)+1) < 1e-5) )
        always_below = np.sum((all_first_below < 0) & (np.abs(all_first_above - len(t)+1) < 1e-5) )
        #t = np.arange(t_start, t_end, dt)
        na = np.zeros(( len(t) ))
        nb = np.zeros(( len(t) ))
        is_above = (all_first_above > all_first_below)
        is_below = (all_first_above < all_first_below)
        na_candidate = all_first_below[is_above]
        nb_candidate = all_first_above[is_below]
        for i, t_i in enumerate(t):
            na[i] = np.sum(na_candidate > i)
            nb[i] = np.sum(nb_candidate > i)
        # this can be improved: sort na_candidate and save cross time.
        na += always_above
        nb += always_below
        pa = na / N_actual
        pb = nb / N_actual
        df = pd.DataFrame(np.vstack(( t, pa, pb )).transpose()) 
        if save_des:
            df.to_csv(save_des, index=None, header=None)
        return df
            
    def get_dpp_parallel(self, cpu_number, seed_initial_condition_list):
        if self.quantum_or_not:
            if self.rho_or_phase == 'rho':
                des = '../data/quantum/persistence/' + self.network_type + '/' 
            else:
                des = '../data/quantum/persistence_phase/' + self.network_type + '/' 
        else:
            des = '../data/classical/persistence/' + self.network_type + '/' 
        if self.m == m_e:
            save_file = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_reference={self.reference_line}_seed_initial='
        else:
            save_file = des + f'm={self.m}_N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_reference={self.reference_line}_seed_initial='
        if not os.path.exists(des):
            os.makedirs(des)
        p = mp.Pool(cpu_number)
        p.starmap_async(self.diffusive_persistence,  [(seed_initial_condition, save_file + f'{seed_initial_condition}.csv') for seed_initial_condition in seed_initial_condition_list]).get()
        p.close()
        p.join()
        return None

    def get_meta_data_parallel(self, cpu_number, seed_initial_condition_list):
        if self.quantum_or_not:
            if self.rho_or_phase == 'rho':
                des = '../data/quantum/meta_data/' + self.network_type + '/' 
            elif self.rho_or_phase == 'phase':
                des = '../data/quantum/meta_data_phase/' + self.network_type + '/' 
        else:
            des = '../data/classical/meta_data/' + self.network_type + '/' 
        if self.m == m_e:
            save_file = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_reference={self.reference_line}_seed_initial='
        else:
            save_file = des + f'm={self.m}_N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_reference={self.reference_line}_seed_initial='
        if not os.path.exists(des):
            os.makedirs(des)
        p = mp.Pool(cpu_number)
        p.starmap_async(self.get_meta_data,  [(seed_initial_condition, save_file + f'{seed_initial_condition}.json') for seed_initial_condition in seed_initial_condition_list]).get()
        p.close()
        p.join()
        return None

    def get_state_distribution(self, seed_initial_condition, t_list, bin_num = 100):
        if self.quantum_or_not:
            if self.rho_or_phase == 'rho':
                des = '../data/quantum/state_distribution/' + self.network_type + '/' 
            elif self.rho_or_phase == 'phase':
                des = '../data/quantum/phase_distribution/' + self.network_type + '/' 
            else:
                print('Please specify which quantity to look at!')
                return 
        else:
            des = '../data/classical/state_distribution/' + self.network_type + '/' 
        if self.m == m_e:  
            save_file = des + f'N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_reference={self.reference_line}_seed_initial={seed_initial_condition}.json'
        else:
            save_file = des + f'm={self.m}_N={self.N}_d={self.d}_seed={self.seed}_alpha={self.alpha}_dt={self.dt}_setup={self.initial_setup}_params={self.distribution_params}_reference={self.reference_line}_seed_initial={seed_initial_condition}.json'
        if not os.path.exists(des):
            os.makedirs(des)
        t, state = self.read_phi(seed_initial_condition)
        #dt = np.round(t[1] - t[0], 3)
        p_state = {} 
        for t_i in t_list:
            index = np.where(np.abs(t-t_i) < 1e-3)[0][0]
            p, bins = np.histogram(state[index], bin_num)
            p_state[t_i]  = {'p':p.tolist(), 'bins':(bins[:-1] + (bins[2]-bins[1])/2).tolist()}
        state_distribution = {'t': t[-1], 't_list': t_list, 'bin_num': bin_num, 'p_state': p_state}
        
        with open(save_file, 'w') as fp:
            json.dump(state_distribution, fp)
        return None




    

cpu_number = 40


hbar = 0.6582
m_e = 5.68


if __name__ == '__main__':
    quantum_or_not = False
    initial_setup = 'uniform_random'
    quantum_or_not = True
    N = 10000
    d = 4
    seed = 0
    alpha = 1
    reference_line = 0.5
    reference_line = 'average'
    seed_initial_condition_list = np.arange(10)


    m_list = [m_e]  * 1
    dt_list = [1] * 1
    alpha_list = [1]
    N_list = [10000] * 1
    num_realization_list = [10] * 1


    "chi2 from rho and uniform random for phase"
    initial_setup = 'chi2_uniform'
    network_type = '1D'
    d_list = [4]
    rho_list = [[1e-4], [1e-2], [1], [10] ]
    phase_list = [[-1, 1], [-1/2, 1/2], [-1/4, 1/4], [0, 0]]

    "uniform random for both rho and phase"

    network_type = '2D_disorder'
    rho_list = [[1, 1]]
    phase_list = [[-1, 1], [-1/2, 1/2], [-1/4, 1/4], [-1/8, 1/8] ]
    d_list = [0.5, 0.55, 0.6, 0.7, 0.8, 0.9]

    initial_setup = 'uniform_random'
    network_type = '1D'
    d_list = [4]
    rho_list = [[1, 1]]
    phase_list = [[-1, 1]]
    rho_list = [[7/16, 9/16], [15/32, 17/32], [31/64, 33/64], [63/128, 65/128]]
    phase_list = [[-1/16, 1/16], [-1/32, 1/32], [-1/64, 1/64], [-1/128, 1/128]]


    initial_setup = 'u_uniform_random_cutoff'
    network_type = '1D'
    d_list = [4]
    rho_list = [[0.05, 0.3], [0.05, 0.1]]
    phase_list = [[0.05, 0.3], [0.05, 0.1]]

    initial_setup = 'u_uniform_random'
    d_list = [0.51, 0.55, 0.7, 0.9]
    network_type = '1D'
    network_type = '2D'
    d_list = [4]

    network_type = '2D_disorder'
    
    initial_setup = 'u_normal_random'
    rho_list = [[0.05]]
    phase_list = [[0.05]]


    initial_setup = 'u_normal_random_cutoff'
    rho_list = [[0, 0.2], [0.05, 0.2], [0.1, 0.2], [0.2, 0.2]]
    phase_list = [[0, 0.2], [0.05, 0.2], [0.1, 0.2], [0.2, 0.2]]

    initial_setup = 'u_normal_random'
    rho_list = [[0], [0.05], [0.1], [0.2]]
    phase_list = [[0], [0.05], [0.1], [0.2]]
    rho_list = [[0.05]]
    phase_list = [[0]]


    m_list = [m_e]  * 12
    dt_list = [1] * 12
    alpha_list = [0.5, 1, 2, 3] * 3
    d_list = [0.51] * 4 + [0.55] * 4 + [0.7] * 4
    N_list = [900] * 12
    num_realization_list = [100] * 12

    distribution_params_raw = [rho + phase for rho in rho_list for phase in phase_list]


    distribution_params_list = []
    for i in distribution_params_raw:
        distribution_params_list.append( [round(j, 10) for j in i])


    rho_or_phase_list = ['rho', 'phase']
    reference_line_list = ['average', 0]

    network_type = '1D'
    N_list = [1000] * 4
    network_type = '2D'
    d_list = [4] 
    network_type = '2D_disorder'
    d_list = [0.51, 0.55, 0.7] 
    N_list = [1000] * 4
    N_list = [900] * 4
    dt_list = [1] * 4
    alpha_list = [0.5, 1, 2, 3] * 1
    num_realization_list = [100] * 4

    "3D"
    network_type = '3D'
    d_list = [4] 
    network_type = '3D_disorder'
    d_list = [0.3, 0.5, 0.7] 
    alpha_list = [0.5, 1, 2, 3] * 1
    dt_list = [1] * len(alpha_list)

    '2D disorder'
    network_type = '2D_disorder'
    d_list = [0.51, 0.55, 0.7, 0.9]
    alpha_list = [1]


    """
    "classical diffusion"
    quantum_or_not = False
    network_type = '1D'
    initial_setup = 'uniform_random'
    initial_setup = 'normal_random'
    distribution_params_list =[0.1]
    d_list = [4]
    alpha_list = [0.5, 1, 2, 3, 4, 5]
    dt_list = [0.1] * len(alpha_list) 
    """

    m_list = [m_e]  * len(alpha_list)

    seed_list = [i for i in range(1)]

    network_type = '2D'
    N_list = [10000]
    d_list = [4]
    network_type = '3D'
    N_list = [8000]
    d_list = [4]
    network_type = '1D'
    d_list = [4]
    N_list = [4000] * len(alpha_list)
    num_realization_list = [10] * len(alpha_list)
    alpha_list = [1]
    dt_list = [1]
    full_data_t = True
    #initial_setup = 'u_normal_phase_uniform_random'
    #distribution_params_list = [[0, 1], [0.05, 1], [0.1, 1], [0.2, 1]]


    for seed in seed_list:
        for d in d_list:
            for m, N, alpha, dt, num_realization in zip(m_list, N_list, alpha_list, dt_list, num_realization_list):
                seed_initial_condition_list = np.arange(num_realization)  + 10
                for distribution_params in distribution_params_list:
                    for rho_or_phase, reference_line in zip(rho_or_phase_list, reference_line_list):
                        pA = persistenceAnalysis(quantum_or_not, network_type, m, N, d, seed, alpha, dt, initial_setup, distribution_params, reference_line, rho_or_phase, full_data_t)
                        pA.get_meta_data_parallel(cpu_number, seed_initial_condition_list)
                        pA.get_dpp_parallel(cpu_number, seed_initial_condition_list)
                        t_list = np.round(np.arange(0, 100, 1), 1).tolist()
                        #t_list = np.round(np.arange(0.0, 100, 1) * dt, 2).tolist()
                        #t_list = np.round([0.0, 1, 1e1, 1e2, 1e3, 1e4, 9.99*1e4 ], 1).tolist()
                        for seed_initial_condition in seed_initial_condition_list:
                            # pA.get_state_distribution(seed_initial_condition, t_list)
                            pass
                        pass

