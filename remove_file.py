import numpy as np
import pandas as pd 
import re
import os

network_type = '2D'
des_list = [f'../data/quantum/phase/{network_type}/', f'../data/quantum/state/{network_type}/']
for des in des_list:
    files = os.listdir(des)
    for file in files:
        data = np.load(des + file)
        N = int(re.search('\d+', file).group())
        if N == 30:
            os.remove(des + file)
                
        #data = np.vstack((data[:100], data[100:][::10] ))
        #np.save(des + file, data)
        

