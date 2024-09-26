"""
Variable selection for linear regression using HeO, L1 and L0.5
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LinearRegression, Lasso
import torch.optim as optim
from heo.log import Logger
from datetime import datetime
import os
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

def phi(mu, sigma):
    return torch.erf(mu / sigma)


def heo_linear_reg(dim, x_s, y_s, config):
    theta = torch.ones(dim, device='cuda:0')
    beta = torch.zeros(dim, device='cuda:0')
    ratio_lr = config['steps']
    theta.requires_grad = True
    beta.requires_grad = True
    opt1 = optim.SGD([theta],lr=config['lr'],momentum=config['momentum'])
    opt2 = optim.SGD([beta],lr=config['lr']/ratio_lr,momentum=config['momentum'])

    for t in range(config['steps']):
        sigma = (1 - t / config['steps']) * np.sqrt(2)
        x = (phi(theta - torch.rand(dim, device='cuda:0'), sigma) + 1) / 2
        pred_y = ((x * beta).reshape(-1, 1) * x_s).sum(dim=0)
        energy = ((y_s - pred_y) ** 2).mean()
        energy.backward()
        opt1.step()
        opt2.step()
        opt1.zero_grad()
        opt2.zero_grad()

        theta.data = torch.clamp(theta.data, 0, 1)
    return theta.detach().cpu(), beta.detach().cpu()


config = {'device':'cuda:0',
          'test_times':10,
          'K':5,
          'type':'sgd',
          'lr':1,
          'momentum':0.999,
          'steps':2000,
          'ensemble':100}

workdir = f'var_sel_log/{timestamp}'
os.makedirs(workdir)
logger = Logger(f'{workdir}/log.log','sat')

logger.log(config)

dim = 400 
batch_size_train = 1000 
batch_size_test = 1000

noise_std_s = [0.05, 0.1,.2, .5, 1, 2, 3, 4, 5]
for sparse_p in [0.8, 0.85, 0.9, 0.95]:
    logger.log(f'===========================sparse_p: {sparse_p}================================')
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
        logger.log(f'---------------------noise_std: {noise_std}-----------------------')

        heo_acc_s = []
        lasso_best_acc_s = []
        lasso_best_acc_s_2 = []

        heo_test_error_s = []
        lasso_test_error_s = []
        lasso_test_error_s_2 = []

        heo_right_s = []
        l1_right_s = []
        l12_right_s= []

        for i in range(config['test_times']):
            # generate dataset
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
            
            # HeO
            for e in range(config['ensemble']):
                val_results = []
                p, beta_heo = heo_linear_reg(dim, input_x, real_y, config)
                for i in range(config['K']):
                    if i < config['K']- 1:
                        random_index_train = random_index[(i + 1) * batch_size_train // config['K']:]
                        random_index_val = random_index[i * batch_size_train // config['K']: (i + 1) * batch_size_train // config['K']]
                    else:
                        random_index_train = random_index[:i * batch_size_train // config['K']]
                        random_index_val = random_index[i * batch_size_train // config['K']:]

                    model = LinearRegression(fit_intercept=False)
                    model.fit(input_x[p > 0.5, :].cpu().T, real_y.cpu())
                    coeff_heo = model.coef_
                    beta_heo = torch.zeros([dim])
                    beta_heo[p > 0.5] = torch.from_numpy(coeff_heo).float()

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

            logger.log(
                f'test_error_heo:{test_heo_error_s[np.argmin(val_heo_error_s)]}, acc: {acc_heo_s[np.argmin(val_heo_error_s)]}')

            alpha_s = [0.05, 0.1, 0.2, 0.5, 1, 2, 5]
            random_index = np.arange(batch_size_train)
            np.random.shuffle(random_index)
            
            # L1
            val_error_s = []
            test_error_s = []
            acc_lasso_s = []
            for alpha in alpha_s:
                val_results = []
                for i in range(config['K']):
                    if i < config['K'] - 1:
                        random_index_train = random_index[(i + 1) * batch_size_train // config['K']:]
                        random_index_val = random_index[i * batch_size_train // config['K']: (i + 1) * batch_size_train // config['K']]
                    else:
                        random_index_train = random_index[:i * batch_size_train // config['K']]
                        random_index_val = random_index[i * batch_size_train // config['K']:]
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
                for i in range(config['K']):
                    if i < config['K'] - 1:
                        random_index_train = random_index[(i + 1) * batch_size_train // config['K']:]
                        random_index_val = random_index[i * batch_size_train // config['K']: (i + 1) * batch_size_train // config['K']]
                    else:
                        random_index_train = random_index[:i * batch_size_train // config['K']]
                        random_index_val = random_index[i * batch_size_train // config['K']:]
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
                for k in range(config['K']):
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

        logger.log('hep acc: {}'.format(np.mean(heo_acc_s)))
        logger.log('L1 acc: {}'.format(np.mean(lasso_best_acc_s)))
        logger.log('L0.5 acc: {}'.format(np.mean(lasso_best_acc_s_2)))

        logger.log('heo test error: {}'.format(np.mean(heo_test_error_s)))
        logger.log('L1 test error: {}'.format(np.mean(lasso_test_error_s)))
        logger.log('L0.5 test error: {}'.format(np.mean(lasso_test_error_s_2)))

        logger.log(f'heo prob complete right: {np.mean(heo_right_s)}')
        logger.log(f'l1 prob complete right: {np.mean(l1_right_s)}')
        logger.log(f'L0.5 prob complete right: {np.mean(l12_right_s)}')

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


    # visualization
    fontsize = 15
    plt.figure(figsize=[12, 4])
    plt.errorbar(noise_std_s, heo_acc_mean_s, yerr=heo_acc_std_s, c='black', marker='.', markersize=12)
    plt.errorbar(noise_std_s, lasso_best_acc_mean_s, yerr=lasso_best_acc_std_s, marker='.', markersize=12)
    plt.errorbar(noise_std_s, lasso_best_acc_mean_2_s, yerr=lasso_best_acc_std_2_s, marker='.', markersize=12)
    plt.ylabel('accuracy', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('noise std', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig('{}/acc_p_{}.pdf'.format(workdir,sparse_p), format='pdf',  bbox_inches='tight',dpi=300)
    
    plt.figure(figsize=[12, 4])
    plt.errorbar(noise_std_s, heo_test_error_mean_s, yerr=heo_test_error_std_s, c='black', marker='.', markersize=12)
    plt.errorbar(noise_std_s, l1_test_error_mean_s, yerr=l1_test_error_std_s, marker='.', markersize=12)
    plt.errorbar(noise_std_s, l05_test_error_mean_s, yerr=l05_test_error_std_s, marker='.', markersize=12)
    plt.ylabel('test error', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('noise std', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig('{}/err_p_{}.pdf'.format(workdir, sparse_p), format='pdf',  bbox_inches='tight',dpi=300)


