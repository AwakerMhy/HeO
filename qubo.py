"""Max cut"""

import numpy as np
import torch
from heo.opt import heo
from heo.log import Logger
import os
from datetime import datetime

from heo.other_methods import isingMac, Lqa
import pickle

workdir = 'qubo'
os.makedirs(workdir,exist_ok=True)
logger = Logger(f'{workdir}/log.log','qubo')


def eval(method, couplings, config):
    max_cut_s = []
    if method in ['CIM','SIM-CIM']:
        eigenvalues, _ = np.linalg.eig(couplings.cpu().numpy())
        max_eigenvalue = np.real(max(eigenvalues))
    for t in range(config['test_times']):
        if method == 'lqa':
            machine = Lqa(couplings, device='cuda:0')
            machine.to(torch.device('cuda:0'))
            energy = machine.minimise(step=1, N=config['T'], g=.1, f=0.1)
        elif method in ['CIM','SIM-CIM']:
            dt = 0.05 if method == 'CIM' else 1.
            energy = isingMac(method, couplings, times=config['T'], dt=dt, max_eigenvalue=max_eigenvalue)
        elif method in ['aSB','bSB','dSB']:
            energy = isingMac(method, couplings, times=config['T'], dt=0.5, max_eigenvalue=None) 
        elif method in ['heo']:
            def energy_func(x):
                return torch.matmul(x.reshape(1, -1), torch.matmul(couplings, x.reshape(-1, 1))).reshape(-1)*0.5            
            solver =  heo(dim=dim, energy_func=energy_func, opt_config=config)
            p = solver.solver(T=config['T'])
            result = 2 * (p > 0.5).float() - 1
            energy = energy_func(result.cuda()).cpu().numpy()

        max_cut_s.append(energy)
    return max_cut_s

config = {'device':'cuda:0',
          'test_times':10,
           'datapath': r'./example_data/rudy_all',
           'unbounded':False,
           'type':'sgd',
           'lr':2,
           'momentum':0,
           'T':5000
           }

logger.log(config)

methods = ['heo', 'lqa','CIM','SIM-CIM','aSB','bSB','dSB']

prob_list = os.listdir(config['datapath'])
instance_num = len(prob_list)

result_dict = {}
ratio_s = 0
ratio_2_s = 0
worst_12_s = 0

for method in methods:
    result_dict[method] = np.zeros([instance_num,config['test_times']])

for inst_index in np.arange(instance_num): 
    logger.log(f'problem name: {prob_list[inst_index]}')
    load_dir = r'{}/{}'.format(config['datapath'], prob_list[inst_index])
    with open(load_dir) as f:
        lines = f.readlines()

    lines[0].strip('\n').split(' ')

    dim = int(lines[0].strip('\n').split(' ')[0])
    couplings = np.zeros([dim, dim])

    for line in lines[1:]:
        data = line.strip('\n').split(' ')
        if len(data) > 1:
            u, v, w = data
            u, v, w = int(u) - 1, int(v) - 1, int(w)
            couplings[u, v] = w
            couplings[v, u] = w

    couplings = torch.from_numpy(couplings).float().cuda()

    best_of_each_methods = []

    for method in methods:
        max_cut_s = eval(method, couplings, config)
        result_dict[method][inst_index]=max_cut_s
        logger.log("{} mean: {:.4f}, std: {:.4f}, min: {:.4f}".format(method, np.mean(max_cut_s),
                                                                            np.std(max_cut_s), np.min(max_cut_s)))
        best = np.min(max_cut_s)
        best_of_each_methods.append(best)
    
    best_of_each_methods = np.array(best_of_each_methods)

    best_over_methods=np.min(best_of_each_methods)
    ratio = (np.abs(best_of_each_methods-best_over_methods)/np.abs(best_over_methods))
    ratio_s+=ratio
    ratio_2_s+=ratio**2

    worst_12 = []
    for index, method in enumerate(methods):
        if np.sum(best_of_each_methods>=best_of_each_methods[index])<=2:
            worst_12.append(1)
        else:
            worst_12.append(0)
    worst_12_s+=np.array(worst_12)

worst_12_s.astype(np.float32)
worst_12_s=worst_12_s/instance_num
ratio_s_mean =ratio_s/instance_num
ratio_2_s/=instance_num
ratio_s_std = np.sqrt(ratio_2_s-ratio_s_mean**2)
logger.log('-------------------------------------------------------------------------------')
logger.log(f'problem name: {prob_list[inst_index]}')
for index, method in enumerate(methods):
    logger.log("{} ratio: {:.4f} ({:.4f}), worst 12 frequency {:.4f}".format(
        method, ratio_s_mean[index], ratio_s_std[index], worst_12_s[index]))
logger.log('-------------------------------------------------------------------------------')

with open(r'{}/result.pickle'.format(workdir), 'wb') as f:
    pickle.dump(result_dict,f)
    
