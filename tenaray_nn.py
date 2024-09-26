"""
Training neural network with tenaray-value weight
"""

import numpy as np
import torch
import torch.optim as optim
import os
from heo.log import Logger

workdir = 'nn'
os.makedirs(workdir,exist_ok=True)
logger = Logger(f'{workdir}/log.log','nn')

def phi(mu, sigma):
    return torch.erf(mu / sigma)

def heo_train(dim, config, rep_energy=False):
    p1 = torch.rand([dim2, dim], device='cuda:0')
    p2 = torch.rand([dim2, dim], device='cuda:0')
    p1.requires_grad = True
    p2.requires_grad = True

    opt = optim.SGD([p1,p2],lr=config['lr'],momentum=config['momentum'])

    if rep_energy:
        energy_s = []
        uncern_s = []
    for t in range(config['T']):
        sigma = (1 - t / config['T']) * np.sqrt(2)
        W = (phi(p1 - torch.rand([dim2, dim], device='cuda:0'), sigma) +
             phi(p2 - torch.rand([dim2, dim], device='cuda:0'), sigma)) / 2
        energy = ((new_work_ensemble(W, x) - y_real) ** 2).mean()
        energy.backward()

        opt.step()
        opt.zero_grad()

        p1.data = torch.clamp(p1.data, 0, 1)
        p2.data = torch.clamp(p2.data, 0, 1)

        if rep_energy:
            energy_s.append(energy.detach().cpu().numpy())
            uncern_s.append(((p1 * (1 - p1)).sum() + (p2 * (1 - p2)).sum()).detach().cpu().numpy())
    else:
        if rep_energy:
            return p1.detach().cpu(), p2.detach().cpu(), energy_s, uncern_s
        else:
            return p1.detach().cpu(), p2.detach().cpu()


config = {'test_times':10,'T':10000, 'lr':0.5,'momentum':0.99}

dim = 100
dim2 = 5

rep_enegry = False
batch_size_s = [10, 20, 50, 100, 200, 500, 1000, 2000]


acc_mean_s = []
acc_std_s = []
for batch_size in batch_size_s:
    acc_s = []
    for _ in range(config['test_times']):
        logger.log(f'batch size: {batch_size}')

        W_real = torch.randint(-1, 2, [dim2, dim]).float()
        W_real = W_real.cuda()

        new_work_ensemble = lambda W, x: torch.relu(torch.matmul(W, x))
        new_work_real_ensemble = lambda x: torch.relu(torch.matmul(W_real, x))

        x = (torch.rand([dim, batch_size]).cuda() > 0.5).float() + (torch.rand([dim, batch_size]).cuda() > 0.5).float() - 1
        y_real = new_work_real_ensemble(x)

        correction = False
        energy_s = []

        p1, p2 = heo_train(dim, config, rep_energy=False)
        result_W = (p1 > 0.5).float() + (p2 > 0.5).float() - 1
        acc = ((result_W.cuda() == W_real).float()).mean().cpu().numpy()
        logger.log(f'acc weight: {acc}')
        acc_s.append(acc)
    acc_mean_s.append(np.mean(acc_s))
    acc_std_s.append(np.std(acc_s))
np.savez(f'{workdir}/three_value_acc_{dim}_{dim2}', acc_mean_s=np.array(acc_mean_s), acc_std_s=np.array(acc_std_s), batch_size_s=batch_size_s)
