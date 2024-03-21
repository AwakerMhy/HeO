"""3-SAT"""

import numpy as np
import torch
from heo.opt import heo
from heo.log import Logger
import os


workdir = 'sat'
os.makedirs(workdir,exist_ok=True)
logger = Logger(f'{workdir}/log.log','sat')

def SAT_original(x, index_list, factor_list):
    return (((1 - x[index_list[:, 0]] * factor_list[:, 0]) * (1 - x[index_list[:, 1]] * factor_list[:, 1]) * (
            1 - x[index_list[:, 2]] * factor_list[:, 2]) / 8)).sum()


def SAT_enegry(x, index_list, factor_list):
    return (((1 - x[index_list[:, 0]] * factor_list[:, 0]) * (1 - x[index_list[:, 1]] * factor_list[:, 1]) * (
            1 - x[index_list[:, 2]] * factor_list[:, 2]) / 8)**4).sum()


def formulate(prob):
    with open(prob) as f:
        data = f.readlines()
    max_index = 0  
    index_list = []
    factor_list = []
    for i, line in enumerate(data[8:-3]):
        if i == 0:  
            u, v, w = line.split(' ')[1:4]
        else:
            u, v, w = line.split(' ')[:3]
        u, v, w = int(u), int(v), int(w)
        max_index = np.max([np.abs(u), np.abs(v), np.abs(w), max_index])

        u_sgn, v_sgn, w_sgn = np.sign(u), np.sign(v), np.sign(w)
        u, v, w = np.abs(u) - 1, np.abs(v) - 1, np.abs(w) - 1
        index_list.append([u, v, w])
        factor_list.append(([u_sgn, v_sgn, w_sgn]))

    index_list = np.array(index_list)
    factor_list = torch.tensor(factor_list)  
    return max_index, index_list, factor_list 


SAT_original_func = lambda x: SAT_original(x, index_list, factor_list)
energy_func = lambda x: SAT_enegry(x, index_list, factor_list)


problem_dicts = {"250":['uf250-1065/uf250-0',1065]}
inst_s = np.arange(1, 11)

variable_dims = ["250"] 
config = {'device':'cuda:0',
          'type':'sgd',
          'lr':2,
          'momentum':0.9999,'T':5000,'test_times':10,
           'datapath': r'/home/mhy/SAT/'
           }


logger.log(config)


for variable_dim in variable_dims:
    logger.log("----------------------------variable_dim: {}--------------------------".format(variable_dim))
    enegry_s_s = []
    prob_path = config['datapath']+problem_dicts[variable_dim][0]
    clause_num = problem_dicts[variable_dim][1]
    for inst in inst_s:
        dim, index_list, factor_list = formulate(r'{}{}.cnf'.format(prob_path, inst))
        factor_list = factor_list.to(config['device'])

        solver =  heo(dim=dim, energy_func=energy_func, opt_config=config)

        energy_s = []
        for t in range(config['test_times']):
            p = solver.solver(T=config['T'])
            result = 2 * (p > 0.5) - 1
            enegry = SAT_original_func(result.to(config['device'])).cpu().numpy()
            energy_s.append(enegry)
            print(enegry)
        percent = 1 - np.array(energy_s) / index_list.shape[0]
        logger.log("inst: {:d}, percent mean: {:.4f}, std: {:.4f}, sat all rate: {:.4f}".format(
            inst, np.mean(percent), np.std(percent), np.mean(percent==1)))
        enegry_s_s.append(energy_s)
    
    enegry_s_s = np.array(enegry_s_s)
    mean_sat = (1-enegry_s_s/clause_num).mean()
    std_sat = (1-enegry_s_s/clause_num).std()
    mean_all_sat = np.mean(enegry_s_s==0)
    logger.log('satisfying rate: {:.4f} ({:.4f}), satisfying all rate: {}'.format(mean_sat,std_sat,mean_all_sat))

    np.save(r'{}/result_{}'.format(workdir,variable_dim), enegry_s_s)