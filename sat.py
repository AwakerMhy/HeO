"""
3-SAT
"""

import numpy as np
import torch

def m_v(mu, sigma):
    return torch.erf(mu / sigma / np.sqrt(2))


random_factor_max = 2
random_factor_min = 0.5

def HeO_SAT(dim, enegry_func, times=10000, lr=1.):
    p = torch.ones(dim, device='cuda:0') * 0.5
    p.requires_grad = True
    for t in range(times):
        sigma = (1 - t / times)
        x = m_v(p - torch.rand(dim, device='cuda:0'), sigma)

        energy = enegry_func(x)
        energy.backward()
        p.data = p.data - lr * p.grad
        p.data = torch.clamp(p.data, 0, 1)
    return p.detach().cpu()


def SAT_original(x, index_list, factor_list):
    return (((1 - x[index_list[:, 0]] * factor_list[:, 0]) * (1 - x[index_list[:, 1]] * factor_list[:, 1]) * (
            1 - x[index_list[:, 2]] * factor_list[:, 2]) / 8)).sum()
def loss_f(x):
    return x ** 4
def SAT_enegry(x, index_list, factor_list):
    return (loss_f((1 - x[index_list[:, 0]] * factor_list[:, 0]) * (1 - x[index_list[:, 1]] * factor_list[:, 1]) * (
            1 - x[index_list[:, 2]] * factor_list[:, 2]) / 8)).sum()


SAT_original_func = lambda x: SAT_original(x, index_list, factor_list)
energy_func = lambda x: SAT_enegry(x, index_list, factor_list)


prob_file = r'example_data/uf20-01.cnf'

test_times = 100

with open(f'{prob_file}') as f:
    data = f.readlines()
max_index = 0
for i, line in enumerate(data[8:-3]):
    if i == 0:  # TODO: need check
        u, v, w = line.split(' ')[1:4]
    else:
        u, v, w = line.split(' ')[:3]
    u, v, w = int(u), int(v), int(w)
    max_index = np.max([np.abs(u), np.abs(v), np.abs(w), max_index])

index_list = []
factor_list = []
for i, line in enumerate(data[8:-3]):
    if i == 0:
        u, v, w = line.split(' ')[1:4]
    else:
        u, v, w = line.split(' ')[:3]
    u, v, w = int(u), int(v), int(w)
    u_sgn, v_sgn, w_sgn = np.sign(u), np.sign(v), np.sign(w)
    u, v, w = np.abs(u) - 1, np.abs(v) - 1, np.abs(w) - 1
    index_list.append([u, v, w])
    factor_list.append(([u_sgn, v_sgn, w_sgn]))
index_list = np.array(index_list)
factor_list = torch.tensor(factor_list).to('cuda:0')

energy_s = []
for t in range(test_times):
    p = HeO_SAT(dim=max_index, enegry_func=energy_func, times=5000, lr=2)
    result = 2 * (p > 0.5) - 1
    enegry = SAT_original_func(result.cuda()).cpu().numpy()
    energy_s.append(enegry)
percent = 1 - np.array(energy_s) / index_list.shape[0]
print("percent mean: {}, std: {}".format(np.mean(percent), np.std(percent)))

