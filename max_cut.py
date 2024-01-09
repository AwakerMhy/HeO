"""
MAX-CUT
"""


import numpy as np
import torch
import torch.nn as nn
import time
from math import pi
import logging

level = getattr(logging, 'INFO', None)


handler1 = logging.StreamHandler()
handler2 = logging.FileHandler('output.txt')
formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
handler1.setFormatter(formatter)
handler2.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(handler1)
logger.addHandler(handler2)
logger.setLevel(level)


def CIM(couplings, times=20000, dt=0.1):
    dim = couplings.shape[0]
    x = torch.cos(2 * np.pi * torch.randn(dim, device='cuda:0')) * 0.001
    a_0 = 1
    c_0 = 1 / max_eigenvalue
    a = 0
    for t in range(times):
        a += 1.1 * a_0 / times
        x += dt * (-(x ** 2 + a_0 - a) * x - c_0 * torch.matmul(couplings, x).reshape(-1))
    x = torch.sgn(x)
    final_enegry = (torch.dot(torch.matmul(couplings, x), x)) / 2
    return final_enegry.cpu().numpy()

def SIMCIM(couplings, times=20000, dt=0.1):
    dim = couplings.shape[0]
    x = (2 * torch.randn(dim, device='cuda:0') - 1) * 0.1
    a_0 = 1
    c_0 = 1 / max_eigenvalue  # 0.7 / np.sqrt(dim)
    a = 0
    for t in range(times):
        a += a_0 / times
        x += dt * (-(a_0 - a) * x - c_0 * torch.matmul(couplings, x).reshape(-1)) + 0.25 * torch.randn_like(x)
        x[x > 1] = 1
        x[x < -1] = -1
    x = torch.sgn(x)
    final_enegry = (torch.dot(torch.matmul(couplings, x), x)) / 2
    return final_enegry.cpu().numpy()

def aSB(couplings, times=20000, dt=0.1):
    dim = couplings.shape[0]
    x = (2 * torch.randn(dim, device='cuda:0') - 1) * 0.1
    y = (2 * torch.randn(dim, device='cuda:0') - 1) * 0.1
    J = torch.sqrt((couplings * couplings).sum() / (dim * (dim - 1)))
    c_0 = 0.5 / J / np.sqrt(dim)
    a = 0
    for t in range(times):
        a += 1 / times
        y += dt * (-(x ** 2 + 1 - a) * x - c_0 * torch.matmul(couplings, x).reshape(-1))
        x += dt * y
    x = torch.sgn(x)
    final_enegry = (torch.dot(torch.matmul(couplings, x), x)) / 2
    return final_enegry.cpu().numpy()

def bSB(couplings, times=20000, dt=0.1):
    dim = couplings.shape[0]
    x = (2 * torch.randn(dim, device='cuda:0') - 1) * 0.1
    y = (2 * torch.randn(dim, device='cuda:0') - 1) * 0.1
    a_0 = 1
    J = torch.sqrt((couplings * couplings).sum() / (dim * (dim - 1)))
    c_0 = 0.5 / J / np.sqrt(dim)
    a = 0
    for t in range(times):
        a += 1 / times
        y += dt * (-(a_0 - a) * x - c_0 * torch.matmul(couplings, x).reshape(-1))
        x += dt * y
        mask = (x.abs() > 1)
        x[mask] = torch.sgn(x)[mask]
        y[mask] = 0
    x = torch.sgn(x)
    final_enegry = (torch.dot(torch.matmul(couplings, x), x)) / 2
    return final_enegry.cpu().numpy()

def dSB(couplings, times=20000, dt=0.1):
    dim = couplings.shape[0]
    x = (2 * torch.randn(dim, device='cuda:0') - 1) * 0.1
    y = (2 * torch.randn(dim, device='cuda:0') - 1) * 0.1
    a_0 = 1
    J = torch.sqrt((couplings * couplings).sum() / (dim * (dim - 1)))
    c_0 = 0.5 / J / np.sqrt(dim)
    a = 0
    for t in range(times):
        a += 1 / times
        y += dt * (-(a_0 - a) * x - c_0 * torch.matmul(couplings, torch.sgn(x).reshape(-1)))
        x += dt * y
        mask = (x.abs() > 1)
        x[mask] = torch.sgn(x)[mask]
        y[mask] = 0

    x = torch.sgn(x)
    final_enegry = (torch.dot(torch.matmul(couplings, x), x)) / 2
    return final_enegry.cpu().numpy()

class Lqa_basic():
    def __init__(self, couplings, device='cpu'):
        super(Lqa_basic, self).__init__()
        self.couplings = couplings
        self.n = couplings.shape[0]
        self.energy = 0.
        self.config = torch.zeros([self.n, 1], device=device)
        self.weights = (2 * torch.rand(self.n, device=device) - 1) * 0.1
        self.velocity = torch.zeros(self.n, device=device)
        self.grad = torch.zeros(self.n, device=device)
        self.device = device

    def forward(self, t, step, mom, g):
        # Implements momentum assisted gradient descent update
        w = torch.tanh(self.weights)
        a = 1 - torch.tanh(self.weights) ** 2
        # spin x,z values
        z = torch.sin(w * pi / 2)
        x = torch.cos(w * pi / 2)

        # gradient
        self.grad = ((1 - t) * z + 2 * t * g * torch.matmul(self.couplings, z) * x) * a * pi / 2
        # weight update
        self.velocity = mom * self.velocity - step * self.grad
        self.weights = self.weights + self.velocity

    def schedule(self, i, N):
        return i / N
    def energy_ising(self, config):
        # energy of a configuration
        return (torch.dot(torch.matmul(self.couplings, config), config)) / 2

    def minimise(self,
                 step=2,  # step size
                 g=1,  # gamma in the article
                 N=200,  # no of iterations
                 mom=0.99,  # momentum
                 f=0.1  # multiplies the weight initialisation
                 ):
        self.weights = (2 * torch.rand(self.n, device=self.device) - 1) * f
        self.velocity = torch.zeros([self.n], device=self.device)
        self.grad = torch.zeros([self.n], device=self.device)

        time0 = time.time()

        for i in range(N):
            t = self.schedule(i, N)
            self.forward(t, step, mom, g)

        self.opt_time = time.time() - time0
        self.config = torch.sign(self.weights.detach())
        self.energy = float(self.energy_ising(self.config))

        print('min energy ' + str(self.energy))

class Lqa(nn.Module):
    def __init__(self, couplings, device='cpu'):
        super(Lqa, self).__init__()

        self.couplings = couplings
        self.n = couplings.shape[0]
        self.energy = 0.
        self.config = torch.zeros([self.n, 1], device=device)
        self.min_en = 9999.
        self.min_config = torch.zeros([self.n, 1], device=device)
        self.weights = torch.zeros([self.n], device=device)
        self.device = device

    def schedule(self, i, N):
        # annealing schedule
        return i / N

    def energy_ising(self, config):
        # ising energy of a configuration
        return (torch.dot(torch.matmul(self.couplings, config), config)) / 2

    def energy_full(self, t, g):
        # cost function value
        config = torch.tanh(self.weights) * pi / 2
        ez = self.energy_ising(torch.sin(config))
        ex = torch.cos(config).sum()

        return (t * ez * g - (1 - t) * ex)

    def minimise(self,
                 step=1,  # learning rate
                 N=200,  # no of iterations
                 g=1.,
                 f=1.):

        self.weights = (2 * torch.rand([self.n], device=self.device) - 1) * f
        self.weights.requires_grad = True
        time0 = time.time()
        optimizer = torch.optim.Adam([self.weights], lr=step)

        for i in range(N):
            t = self.schedule(i, N)
            energy = self.energy_full(t, g)
            optimizer.zero_grad()
            energy.backward()
            optimizer.step()

        self.opt_time = time.time() - time0
        self.config = torch.sign(self.weights.detach())
        self.energy = float(self.energy_ising(self.config))
        if gap:
            return self.energy, self.energy - float(self.energy_ising(self.weights.detach()))
        else:
            return self.energy

def m_v(mu, sigma):
    return torch.erf(mu / sigma / np.sqrt(2))
def psi(mu, sigma):
    return np.sqrt(2 / np.pi) * torch.exp(-(mu / sigma) ** 2 / 2) / sigma

def heo(dim, W, times=10000, lr=1.):
    p = torch.ones(dim, device='cuda:0') * 0.5
    for t in range(times):
        lr_ = lr
        u = torch.rand(dim, device='cuda:0')
        sigma = (1 - t / times)
        x = m_v(p - u, sigma)
        p = p - lr_ * torch.matmul(W, x.reshape(-1, 1)).reshape(-1) * psi(p - u, sigma)
        p.data = torch.clamp(p.data, 0, 1)
    return p.cpu()

## K2000
dim = 2000
couplings = np.load(r'example_data/K2000.npy.npy')
eigenvalues, _ = np.linalg.eig(couplings)
max_eigenvalue = max(eigenvalues)
couplings = torch.from_numpy(couplings).float().cuda()

iter_num = 5000
test_times = 10

max_cut_s = []
for t in range(test_times):
    machine = Lqa(couplings, device='cuda:0')
    machine.to(torch.device('cuda:0'))
    energy = machine.minimise(step=1, N=iter_num, g=.1, f=0.1)
    max_cut = (0.5 * couplings.sum() - energy) / 2
    max_cut_s.append(max_cut.cpu().numpy())
logging.info("Lqa mean: {:.4f}, std: {:.4f}, max: {:.4f}".format(np.mean(max_cut_s),
                                                                     np.std(max_cut_s), np.max(max_cut_s)))

max_cut_s = []
for t in range(test_times):
    energy = bSB(couplings, times=iter_num, dt=0.5)
    max_cut = (0.5 * couplings.sum().cpu().numpy() - energy) / 2
    max_cut_s.append(max_cut)
logging.info("bSB mean: {:.4f}, std: {:.4f}, min: {:.4f}".format(np.mean(max_cut_s),
                                                                     np.std(max_cut_s), np.min(max_cut_s)))


max_cut_s = []
for t in range(test_times):
    energy = dSB(couplings, times=iter_num, dt=0.5)
    max_cut = (0.5 * couplings.sum().cpu().numpy() - energy) / 2
    max_cut_s.append(max_cut)
logging.info("dSB mean: {:.4f}, std: {:.4f}, min: {:.4f}".format(np.mean(max_cut_s),
                                                                     np.std(max_cut_s), np.min(max_cut_s)))

max_cut_s = []
for t in range(test_times):
    energy = aSB(couplings, times=iter_num, dt=0.5)
    max_cut = (0.5 * couplings.sum().cpu().numpy() - energy) / 2
    max_cut_s.append(max_cut)
logging.info("aSB mean: {:.4f}, std: {:.4f}, min: {:.4f}".format(np.mean(max_cut_s),np.std(max_cut_s),
                                                                 np.min(max_cut_s)))


max_cut_s = []
for t in range(test_times):
    energy = CIM(couplings, times=iter_num, dt=0.05)
    max_cut = (0.5 * couplings.sum().cpu().numpy() - energy) / 2
    max_cut_s.append(max_cut)
logging.info("CIM mean: {:.4f}, std: {:.4f}, max: {:.4f}".format(np.mean(max_cut_s),
                                                                     np.std(max_cut_s), np.max(max_cut_s)))

max_cut_s = []
for t in range(test_times):
    energy = SIMCIM(couplings, times=iter_num, dt=1)
    max_cut = (0.5 * couplings.sum().cpu().numpy() - energy) / 2
    max_cut_s.append(max_cut)
logging.info("SIM-CIM mean: {:.4f}, std: {:.4f}, max: {:.4f}".format(np.mean(max_cut_s),
                                                                        np.std(max_cut_s),
                                                                        np.max(max_cut_s)))

uncer_s = []
max_cut_s = []
for t in range(test_times):
    p = heo(dim=dim, W=couplings, times=iter_num, lr=2)
    uncer = (p * (1 - p)).sum()
    result = 2 * (p > 0.5).float() - 1
    gap = torch.matmul(result.reshape(1, -1), torch.matmul(couplings.cpu(), result)) - \
          torch.matmul((2 * p - 1).reshape(1, -1), torch.matmul(couplings.cpu(), (2 * p - 1).reshape(-1, 1)))
    max_cut = 0.25 * (
                couplings.sum().cpu() - torch.matmul(result.reshape(1, -1), torch.matmul(couplings.cpu(), result)))
    max_cut_s.append(max_cut.numpy())
    uncer_s.append(uncer.numpy())
logging.info("HeO mean: {:.4f}, std: {:.4f}, max: {:.4f}".format(np.mean(max_cut_s),
                                                                     np.std(max_cut_s), np.max(max_cut_s)))
