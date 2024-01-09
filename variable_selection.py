"""
variable selection for linear regression
"""
import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Lasso
import matplotlib.pyplot as plt
test_times = 10


def m_v(mu, sigma):
    return torch.erf(mu / sigma / np.sqrt(2))
def heo_vs(dim, x_s, y_s, times=10000, lr=1.):
    p = torch.ones(dim, device='cuda:0')
    beta = torch.zeros(dim, device='cuda:0')
    ratio_lr = times
    p.requires_grad = True
    beta.requires_grad = True
    for t in range(times):
        sigma = (1 - t / times)
        x = (m_v(p - torch.rand(dim, device='cuda:0'), sigma) + 1) / 2
        pred_y = ((x * beta).reshape(-1, 1) * x_s).sum(dim=0)
        energy = ((y_s - pred_y) ** 2).mean()
        energy.backward()
        p.data = p.data - lr * p.grad
        p.data = torch.clamp(p.data, 0, 1)
        beta.data = beta.data - lr * beta.grad / ratio_lr
    return p.detach().cpu(), beta.detach().cpu()



noise_std_s = [0.05,0.1,.2, .5, 1, 2,3,4,5,6]
K = 5  # K fold validation
ensemble = 100  # the number of versions
steps = 2000  # step number of HeO
lr = 5  # learning rate of He0

print(f'ensemble: {ensemble}, steps: {steps}, lr: {lr}')

sparse_p = 0.9

heo_acc_mean_s = []
heo_acc_std_s = []

lasso_best_acc_mean_s = []
lasso_best_acc_std_s = []

lasso_best_acc_mean_2_s = []
lasso_best_acc_std_2_s = []

hea_prob_s = []
l1_prob_s = []
l12_prob_s = []

heo_test_error_mean_s = []
heo_test_error_std_s = []

l1_test_error_mean_s = []
l1_test_error_std_s = []

l05_test_error_mean_s = []
l05_test_error_std_s = []

for noise_std in noise_std_s:
    torch.manual_seed(42)
    print(f'noise_std: {noise_std}')
    dim = 400
    batch_size_train = 1000
    batch_size_test = 1000

    heo_acc_s = []
    lasso_best_acc_s = []
    lasso_best_acc_s_2 = []

    heo_test_error_s = []
    lasso_test_error_s = []
    lasso_test_error_s_2 = []

    heo_right_s = []
    l1_right_s = []
    l12_right_s = []

    for i in range(test_times):
        mask = torch.rand([dim], device='cuda:0') > sparse_p
        beta_gt = (torch.rand([dim], device='cuda:0') + 1) * (
                (torch.rand([dim], device='cuda:0') > 0.5).float() * 2 - 1)
        beta_gt[~mask] = 0
        input_x = torch.randn([dim, batch_size_train], device='cuda:0')
        real_y = (beta_gt.reshape(-1, 1) * input_x).sum(dim=0) + noise_std * torch.randn([batch_size_train],
                                                                                         device='cuda:0')
        input_x_test = torch.randn([dim, batch_size_test], device='cuda:0')
        real_y_test = (beta_gt.reshape(-1, 1) * input_x_test).sum(dim=0)

        val_heo_error_s = []
        test_heo_error_s = []
        acc_heo_s = []

        random_index = np.arange(batch_size_train)
        np.random.shuffle(random_index)

        for e in range(ensemble):
            val_results = []
            p, beta_heo = heo_vs(dim, input_x, real_y, times=steps, lr=lr)
            for i in range(K):
                if i < K - 1:
                    random_index_train = random_index[(i + 1) * batch_size_train // K:]
                    random_index_val = random_index[i * batch_size_train // K: (i + 1) * batch_size_train // K]
                else:
                    random_index_train = random_index[:i * batch_size_train // K]
                    random_index_val = random_index[i * batch_size_train // K:]

                model = LinearRegression(fit_intercept=False)
                model.fit(input_x[p > 0.5, :].cpu().T, real_y.cpu())
                coeff_heo = model.coef_
                beta_heo = torch.zeros([dim])
                beta_heo[p > 0.5] = torch.from_numpy(coeff_heo).float()  # [p > 0.5]

                pred_y_val = (beta_heo.reshape(-1, 1) * input_x[:, random_index_val].cpu()).sum(dim=0)
                val_error_ = ((real_y[random_index_val].cpu() - pred_y_val) ** 2).mean()
                val_results.append(val_error_.numpy())

            acc_heo = (mask.cpu() == (p > 0.5)).float().mean()
            pred_y = (beta_heo.reshape(-1, 1) * input_x_test.cpu()).sum(dim=0)
            test_error_heo = ((real_y_test.cpu() - pred_y) ** 2).mean()

            var_error = np.mean(val_results)
            val_heo_error_s.append(var_error)
            test_heo_error_s.append(test_error_heo.numpy())
            acc_heo_s.append(acc_heo.numpy())

        alpha_s = [0.05, 0.1, 0.2, 0.5, 1, 2, 5]
        random_index = np.arange(batch_size_train)
        np.random.shuffle(random_index)

        # L1
        val_error_s = []
        test_error_s = []
        acc_lasso_s = []
        for alpha in alpha_s:
            val_results = []
            for i in range(K):
                if i < K - 1:
                    random_index_train = random_index[(i + 1) * batch_size_train // K:]
                    random_index_val = random_index[i * batch_size_train // K: (i + 1) * batch_size_train // K]
                else:
                    random_index_train = random_index[:i * batch_size_train // K]
                    random_index_val = random_index[i * batch_size_train // K:]
                lasso_model = Lasso(alpha=alpha, fit_intercept=False)
                lasso_model.fit(input_x[:, random_index_train].cpu().T, real_y[random_index_train].cpu())
                coeff_lasso = torch.from_numpy(lasso_model.coef_)
                pred_y_val = (coeff_lasso.reshape(-1, 1) * input_x[:, random_index_val].cpu()).sum(dim=0)
                val_error_ = ((real_y[random_index_val].cpu() - pred_y_val) ** 2).mean()
                val_results.append(val_error_.numpy())

            lasso_model = Lasso(alpha=alpha, fit_intercept=False)
            lasso_model.fit(input_x.cpu().T, real_y.cpu())
            coeff_lasso = torch.from_numpy(lasso_model.coef_)
            acc_lasso = (mask.cpu() == (coeff_lasso.abs() > 0).float()).float().mean()

            pred_y = (coeff_lasso.reshape(-1, 1) * input_x_test.cpu()).sum(dim=0)
            test_error = ((real_y_test.cpu() - pred_y) ** 2).mean()

            val_error = np.mean(val_results)
            val_error_s.append(val_error)
            test_error_s.append(test_error.numpy())
            acc_lasso_s.append(acc_lasso.numpy())

        # L0.5
        epsilon = 1e-10
        K_L05 = 10
        val_error_s_2 = []
        test_error_s_2 = []
        acc_lasso_s_2 = []
        input_x_cpu = input_x.cpu().T
        real_y_cpu = real_y.cpu()

        for alpha in alpha_s:
            val_results = []
            for i in range(K):
                if i < K - 1:
                    random_index_train = random_index[(i + 1) * batch_size_train // K:]
                    random_index_val = random_index[i * batch_size_train // K: (i + 1) * batch_size_train // K]
                else:
                    random_index_train = random_index[:i * batch_size_train // K]
                    random_index_val = random_index[i * batch_size_train // K:]
                lasso_model = Lasso(alpha=alpha, fit_intercept=False)
                lasso_model.fit(input_x_cpu, real_y_cpu)
                beta_lasso = lasso_model.coef_
                for k in range(K_L05):
                    input_x_ = input_x_cpu[random_index_train, :] * np.sqrt(np.abs(beta_lasso) + epsilon).reshape(1,
                                                                                                                  -1)
                    lasso_model = Lasso(alpha=alpha, fit_intercept=False)
                    lasso_model.fit(input_x_, real_y_cpu[random_index_train])
                    beta_lasso = lasso_model.coef_ * np.sqrt(np.abs(beta_lasso) + epsilon)
                coeff_lasso = torch.from_numpy(beta_lasso)

                pred_y_val = (coeff_lasso.reshape(-1, 1) * input_x_cpu[random_index_val, :].cpu().T).sum(dim=0)
                var_error_ = ((real_y[random_index_val].cpu() - pred_y_val) ** 2).mean()
                val_results.append(var_error_.numpy())

                pred_y = (coeff_lasso.reshape(-1, 1) * input_x_test.cpu()).sum(dim=0)
                test_error = ((real_y_test.cpu() - pred_y) ** 2).mean()
                test_error_s_2.append(test_error.numpy())

            var_error = np.mean(val_results)
            lasso_model = Lasso(alpha=alpha, fit_intercept=False)
            lasso_model.fit(input_x_cpu, real_y_cpu)
            beta_lasso = lasso_model.coef_
            for k in range(K):
                input_x_ = input_x_cpu * np.sqrt(np.abs(beta_lasso) + epsilon).reshape(1, -1)
                lasso_model = Lasso(alpha=alpha, fit_intercept=False)
                lasso_model.fit(input_x_, real_y_cpu)
                beta_lasso = lasso_model.coef_ * np.sqrt(np.abs(beta_lasso) + epsilon)
            coeff_lasso = torch.from_numpy(beta_lasso)
            acc_lasso = (mask.cpu() == (coeff_lasso.abs() > 0).float()).float().mean()
            val_error_s_2.append(val_error)
            acc_lasso_s_2.append(acc_lasso.numpy())

        heo_acc_s.append(acc_heo_s[np.argmin(val_heo_error_s)])
        heo_test_error_s.append(test_heo_error_s[np.argmin(val_heo_error_s)])
        lasso_best_acc_s.append(acc_lasso_s[np.argmin(val_error_s)])
        lasso_best_acc_s_2.append(acc_lasso_s_2[np.argmin(val_error_s_2)])
        lasso_test_error_s.append(test_error_s[np.argmin(val_error_s)])
        lasso_test_error_s_2.append(test_error_s_2[np.argmin(val_error_s_2)])

        heo_right_s.append(acc_heo_s[np.argmin(val_heo_error_s)]==1)
        l1_right_s.append(acc_lasso_s[np.argmin(val_error_s)]==1)
        l12_right_s.append(acc_lasso_s_2[np.argmin(val_error_s_2)]==1)

    print('hep acc: {}'.format(np.mean(heo_acc_s)))
    print('L1 acc: {}'.format(np.mean(lasso_best_acc_s)))
    print('L0.5 acc: {}'.format(np.mean(lasso_best_acc_s_2)))

    print('heo test error: {}'.format(np.mean(heo_test_error_s)))
    print('L1 test error: {}'.format(np.mean(lasso_test_error_s)))
    print('L0.5 test error: {}'.format(np.mean(lasso_test_error_s_2)))

    print(f'heo prob complete right: {np.mean(heo_right_s)}')
    print(f'l1 prob complete right: {np.mean(l1_right_s)}')
    print(f'L0.5 prob complete right: {np.mean(l12_right_s)}')

    heo_acc_mean_s.append(np.mean(heo_acc_s))
    lasso_best_acc_mean_s.append(np.mean(lasso_best_acc_s))
    lasso_best_acc_mean_2_s.append(np.mean(lasso_best_acc_s_2))

    heo_acc_std_s.append(np.std(heo_acc_s))
    lasso_best_acc_std_s.append(np.std(lasso_best_acc_s))
    lasso_best_acc_std_2_s.append(np.std(lasso_best_acc_s_2))

    hea_prob_s.append(np.mean(heo_right_s))
    l1_prob_s.append(np.mean(l1_right_s))
    l12_prob_s.append(np.mean(l12_right_s))

    heo_test_error_mean_s.append(np.mean(heo_test_error_s))
    heo_test_error_std_s.append(np.std(heo_test_error_s))

    l1_test_error_mean_s.append(np.mean(lasso_test_error_s))
    l1_test_error_std_s.append(np.std(lasso_test_error_s))

    l05_test_error_mean_s.append(np.mean(lasso_test_error_s_2))
    l05_test_error_std_s.append(np.std(lasso_test_error_s_2))

fontsize = 15
plt.figure(figsize=[12, 4])
plt.errorbar(noise_std_s, heo_acc_mean_s, yerr=heo_acc_std_s, c='black', marker='.', markersize=12)
plt.errorbar(noise_std_s, lasso_best_acc_mean_s, yerr=lasso_best_acc_std_s, marker='.', markersize=12)
plt.errorbar(noise_std_s, lasso_best_acc_mean_2_s, yerr=lasso_best_acc_std_2_s, marker='.', markersize=12)
plt.ylabel('accuracy', fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('noise std', fontsize=fontsize)
plt.xticks(fontsize=fontsize)

