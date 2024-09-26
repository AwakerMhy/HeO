"""
Minimum vertex cover
"""

import numpy as np
import torch
import time
import subprocess
from collections import defaultdict
import re
import torch.optim as optim

from heo.log import Logger
import os

workdir = 'mvc'
os.makedirs(workdir,exist_ok=True)
logger = Logger(f'{workdir}/log.log','mvc')

size_pattern = re.compile(r'Final vertex cover size = (\d+)')
time_pattern = re.compile(r'Time cost = (\d+\.\d+)')

def refine(x):
    for i,item in enumerate(x):
        if item and x[G_dict[i]].all():
            x[i] = False
    return x

def phi(mu, sigma):
    return torch.erf(mu / sigma)

def heo_mvc(dim, enegry_func, config):
    p = torch.ones(dim, device='cuda:0') * 0.5
    p.requires_grad = True
    if config['type'] == 'sgd':
        opt=optim.SGD([p], lr=config['lr'], momentum=config['momentum'])
    elif config['type'] == 'adam':
        opt=optim.Adam([p], lr=config['lr'])
    for t in range(config['T']):
        C = t / config['T'] * config['C_factor']
        sigma = (1 - t / config['T']) * np.sqrt(2)
        x = phi(p - torch.rand(dim, device='cuda:0'), sigma)
        energy = enegry_func((x + 1) / 2, C)
        energy.backward()
        opt.step()
        opt.zero_grad()
        p.data = torch.clamp(p.data, 0, 1)
    return p.detach().cpu()

def energy_func(x, C):
    energy = C * (torch.sparse.mm(G, 1 - x.reshape(-1, 1)).reshape(-1) * (1 - x)).sum().reshape(-1) + x.sum()
    return energy


config = {'device':'cuda:0',
          'type':'sgd',
          'lr': 2.5,
          'momentum':0.9999,
          'times':10,
          'C_factor':2.5,
          'T':50}

logger.log(config)

prob_path = r'example_data/tech-RL-caida.mtx'
dim = 190914

with open(f'{prob_path}.mtx', 'r') as f:
    prob_data = f.readlines()
rows, cols = dim, dim
u_s = []
v_s = []
G_dict = defaultdict(list)
for line in prob_data[2:]:
    u, v = line.strip('\n').split(' ')
    u, v = int(u) - 1, int(v) - 1
    u, v = min(u, v), max(u, v)
    u_s.append(u)
    v_s.append(v)
    G_dict[u].append(v)
    G_dict[v].append(u)

indices = torch.LongTensor([u_s, v_s])
values = torch.ones([len(v_s)]).float()
G = torch.sparse.FloatTensor(indices, values, torch.Size([rows, cols])).cuda()

results = []
runtimes = []
for i in range(config['times']):
    start = time.time()
    result = heo_mvc(dim, energy_func, config)
    end = time.time()
    x = (result > 0.5).int()
    e1, e2 = energy_func(x.cuda().float(), C=0), energy_func(x.cuda().float(), C=100000)
    if e2 < e1 + 1:
        x = list(x.numpy())
        with open(r'ver_cov', 'w', encoding='utf-8') as file:
            file.writelines(['p ' + str(xx) + "\n" for xx in x])
        cpp_executable = "./heo/vertex_cover_process/vertex"
        arg1 = f"example_data/tech-RL-caida.mis"
        arg2 = f"ver_cov"
        result_info = subprocess.run([cpp_executable, arg1, arg2], capture_output=True, text=True)
        size_match = size_pattern.search(result_info.stdout)
        time_match = time_pattern.search(result_info.stdout)
        final_size = int(size_match.group(1))
        time_cost = float(time_match.group(1))
        results.append((final_size))
        runtimes.append(end - start + time_cost)
results = np.array(results)
runtimes = np.array(runtimes)

logger.log('{:.1f} ({:.1f}, {:.1f}), {})'.format(results.min(), results.mean(), results.std(),runtimes.mean()))

