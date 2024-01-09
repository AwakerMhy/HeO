"""
minimum vertex cover
"""

import numpy as np
import torch
import time
import subprocess
from collections import defaultdict
import re
size_pattern = re.compile(r'Final vertex cover size = (\d+)')
time_pattern = re.compile(r'Time cost = (\d+\.\d+)')

def refine(x):
    for i,item in enumerate(x):
        if item and x[G_dict[i]].all():
            x[i] = False
    return x

def m_v(mu, sigma):
    return torch.erf(mu / sigma / np.sqrt(2))


def psi(mu, sigma):
    return np.sqrt(2 / np.pi) * torch.exp(-(mu / sigma) ** 2 / 2) / sigma

def heo_mvc(dim, enegry_func, times=10000, lr=1.):
    p = torch.ones(dim, device='cuda:0') * 0.5
    p.requires_grad = True
    for t in range(times):
        C = t / times * C_factor
        sigma = (1 - t / times)
        x = m_v(p - torch.rand(dim, device='cuda:0'), sigma)
        energy = enegry_func((x + 1) / 2, C)
        energy.backward()
        p.data = p.data - lr * p.grad
        p.data = torch.clamp(p.data, 0, 1)
    return p.detach().cpu()

def energy_func(x, C):
    energy = C * (torch.sparse.mm(G, 1 - x.reshape(-1, 1)).reshape(-1) * (1 - x)).sum().reshape(-1) + x.sum()
    return energy


prob_path = r'example_data/bio-celegans.mtx'
dim = 453

C_factor = 2.5
times = 100
iter_times = 200
lr = 2.5

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
for i in range(times):
    start = time.time()
    result = heo_mvc(dim, energy_func, times=iter_times, lr=2)
    end = time.time()
    x = (result > 0.5).int()
    e1, e2 = energy_func(x.cuda().float(), C=0), energy_func(x.cuda().float(), C=100000)
    if e2 < e1 + 1:
        x = list(x.numpy())
        with open(r'ver_cov', 'w', encoding='utf-8') as file:
            file.writelines(['p ' + str(xx) + "\n" for xx in x])
        cpp_executable = "./vertex_cover_process/vertex"
        arg1 = f"example_data/bio-celegans.mis"
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

print('{:.1f} ({:.1f}, {:.1f}), {})'.format(results.min(), results.mean(), results.std(),runtimes.mean()))

