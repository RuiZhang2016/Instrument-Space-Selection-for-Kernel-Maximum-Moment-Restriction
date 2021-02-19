import os, sys
import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
from util import get_median_inter_mnist, Kernel, load_data, ROOT_PATH, jitchol, _sqdist, \
    remove_outliers, nystrom_decomp, chol_inv
from joblib import Parallel, delayed
import time
import autograd.scipy.linalg as splg
import matplotlib.pyplot as plt

Nfeval = 1
seed = 527
np.random.seed(seed)
JITTER = 1e-7
nystr_M = 300
EYE_nystr = np.eye(nystr_M)
opt_params = None
prev_norm = None
opt_test_err = None


class autograd_data():
    def __init__(self, x, y, z):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.z = np.asarray(z)


def experiment(sname, seed, datasize, nystr=False):
    def MMR(alpha, lambdaH):
        # L = np.exp(-L0 / al / al / 2) + 1e-6 * EYEN

        alpha = alpha[:, None]
        residual = Y - L @ alpha
        w1 = residual.T @ W1 @ residual  # ((Y- L@alpha).T@W1@(Y- L@alpha))**2/np.trace(W1@W1)/np.trace(residual_mat@H@residual_mat@H)
        w0 = residual.T @ W0 @ residual  # ((Y- L@alpha).T@W0@(Y- L@alpha))**2/np.trace(W0@W0)/np.trace(residual_mat@H@residual_mat@H)
        w0, w1 = 1 / w0, 1 / w1
        W = (w1 * W1 + w0 * W0) / (w1 + w0)
        H = np.eye(W.shape[0]) - 1 / X.shape[0]
        W = H @ W @ H
        # LWL_inv = chol_inv(L @ W @ L + L / N2 + JITTER * EYEN)
        # alpha = LWL_inv @ L @ W @ Y

        return residual.T @ W @ residual + lambdaH * alpha.T @ L @ alpha


    def CV_once(train_ad, dev_ad, al, lambdaH):
        EYEN = np.eye(train_ad.x.shape[0])
        H = EYEN - 1 / train_ad.x.shape[0]
        # ak = np.array([get_median_inter_mnist(train_ad.z[:, [i]]) for i in range(train_ad.z.shape[1])])
        # W = np.array([H @ (np.exp(-_sqdist(train_ad.z[:, [i]], None) / ak[i] ** 2 / 2) + 1e-6 * EYEN) @ H / train_ad.z.shape[0]**2 for i in
        #               range(train_ad.z.shape[1])])
        ak0 = get_median_inter_mnist(train_ad.z[:, [0]])
        ak1 = get_median_inter_mnist(train_ad.z[:, [1]])
        ntrain = train_ad.z.shape[0]
        W0 = H @ (np.exp(-_sqdist(train_ad.z[:, [0]], None) / ak0 ** 2 / 2) + 1e-6 * EYEN) @ H / ntrain ** 2
        W1 = H @ (np.exp(-_sqdist(train_ad.z[:, [1]], None) / ak1 ** 2 / 2) + 1e-6 * EYEN) @ H / ntrain ** 2

        assert len(al) == train_ad.x.shape[1]
        # L = np.exp(
        #     -np.sum([_sqdist(train_ad.x[:, [i]], None) / al[i] ** 2 for i in range(train_ad.x.shape[1])]) / 2) + 1e-6 * EYEN
        L = np.exp(-_sqdist(train_ad.x, None) / al ** 2 / 2)

        def MMR(alpha):
            alpha = alpha[:, None]
            residual = train_ad.y - L @ alpha
            # w0 = (residual.T @ W0 @ residual)**2 /np.trace(W0@W0)/np.trace(residual@residual.T@H@residual@residual.T@H)
            # w1 = (residual.T @ W1 @ residual)**2 /np.trace(W1@W1)/np.trace(residual@residual.T@H@residual@residual.T@H)
            w0 = residual.T @ W0 @ residual
            w1 = residual.T @ W1 @ residual
            w0, w1 = 1 / w0, 1 / w1
            w0, w1 = w0 / (w0 + w1), w1 / (w0 + w1)
            W = w0 * W0 + w1 * W1
            train_ad_loss = lambdaH * alpha.T @ L @ alpha + residual.T @ W @ residual
            return train_ad_loss

        bounds = None
        params0 = np.random.randn(train_ad.x.shape[0]) / 10
        obj_grad = value_and_grad(lambda params: MMR(params))
        res = minimize(obj_grad, x0=params0, bounds=bounds, method='L-BFGS-B', jac=True, options={'maxiter': 500})

        alpha = res.x[:, None]
        EYEN = np.eye(dev_ad.x.shape[0])
        H = EYEN - 1 / dev_ad.x.shape[0]
        # dev_L = np.exp(-np.sum([_sqdist(dev_ad.x[:, [i]], train_ad.x[:, [i]]) / al[i] ** 2 for i in
        #                         range(dev_ad.x.shape[1])]) / 2) + 1e-6 * EYEN
        dev_L = np.exp(-_sqdist(dev_ad.x, train_ad.x) / al ** 2 / 2) + 1e-6 * EYEN
        ndev = dev_ad.z.shape[0]
        dev_W0 = H @ (np.exp(-_sqdist(dev_ad.z[:, [0]], None) / ak0 ** 2 / 2) + 1e-6 * EYEN) @ H / ndev ** 2
        dev_W1 = H @ (np.exp(-_sqdist(dev_ad.z[:, [1]], None) / ak1 ** 2 / 2) + 1e-6 * EYEN) @ H / ndev ** 2
        w0 = (dev_ad.y - dev_L @ alpha).T @ dev_W0 @ (dev_ad.y - dev_L @ alpha)
        w1 = (dev_ad.y - dev_L @ alpha).T @ dev_W1 @ (dev_ad.y - dev_L @ alpha)
        w0 = w0 / (w0 + w1)
        w1 = w1 / (w0 + w1)
        # w = np.array(
        #     [(dev_ad.y - dev_L @ alpha).T @ dev_W[i] @ (dev_ad.y - dev_L @ alpha) for i in range(train_ad.z.shape[1])])
        # w /= np.sum(w, axis=0)

        # w = np.array([(train_ad.y - L @ alpha).T @ W[i] @ (train_ad.y - L @ alpha) for i in range(train_ad.z.shape[1])])
        # w = w / np.sum(w, axis=0)
        W = w0 * W0 + w1 * W1
        dev_loss = (dev_ad.y - dev_L @ alpha).T @ W @ (dev_ad.y - dev_L @ alpha)
        return dev_loss

    def callback0(alpha):
        global Nfeval
        if Nfeval % 100 == 0:
            alpha = alpha[:, None]
            pred_mean = test_L @ alpha
            test_err = ((pred_mean - test_G) ** 2).mean()
            # print('test err, norm: ', test_err, alpha.T@L@alpha)
        Nfeval += 1

    train, dev, test = load_data(ROOT_PATH + '/data/zoo/{}_{}.npz'.format(sname, datasize))
    train_ad = autograd_data(train.x, train.y, train.z)
    dev_ad = autograd_data(dev.x, dev.y, dev.z)

    X = np.vstack((train.x, dev.x))
    Y = np.vstack((train.y, dev.y))
    Z = np.vstack((train.z, dev.z))
    test_X = test.x
    test_G = test.g

    ak0 = get_median_inter_mnist(Z[:, [0]])
    ak1 = get_median_inter_mnist(Z[:, [1]])
    al0 = get_median_inter_mnist(X)
    N2 = X.shape[0] ** 2
    W0 = _sqdist(Z[:, [0]], None)
    W1 = _sqdist(Z[:, [1]], None)
    # del W0
    L0, test_L0 = _sqdist(X, None), _sqdist(test_X, X)
    W0 = (np.exp(-W0 / ak0 ** 2 / 2) + 1e-6 * np.eye(W0.shape[0])) / N2
    W1 = (np.exp(-W1 / ak1 ** 2 / 2) + 1e-6 * np.eye(W1.shape[0])) / N2

    dev_loss_list = []
    test_err_list = []
    for al in [al0 * j for j in np.logspace(-1, 1, 10)]:
        al = np.array([al])
        for lambdaH in [10 ** (-i) for i in range(1, 10)]:
            Nfeval = 1
            print(' -------  lambda {} al {}  --------- '.format(lambdaH, al))
            dev_loss1 = CV_once(train_ad, dev_ad, al, lambdaH)
            dev_loss2 = CV_once(dev_ad, train_ad, al, lambdaH)
            # print('dev_loss:', (dev_loss1+dev_loss2)[0,0])
            dev_loss_list += [(dev_loss1+dev_loss2)[0,0]]

            L = np.exp(-L0 / al ** 2 / 2) + 1e-6 * np.eye(L0.shape[0])
            test_L = np.exp(-test_L0 / al ** 2 / 2)
            alpha0 = np.random.randn(X.shape[0])
            bounds = None
            obj_grad = value_and_grad(lambda params: MMR(params, lambdaH))
            res = minimize(obj_grad, x0=alpha0, bounds=bounds, method='L-BFGS-B', jac=True, options={'maxiter': 1000})
            alpha = res.x[:, None]
            pred_mean = test_L @ alpha
            test_err = ((pred_mean - test_G) ** 2).mean()
            test_err_list += [test_err]



    # for ite in range(10):
    #     obj_grad = value_and_grad(lambda params: MMR(paramsH,params))
    #     res = minimize(obj_grad, x0=alpha, bounds=bounds, method='L-BFGS-B', jac=True, options={'maxiter': 1000})
    #     alpha = res.x
    #     obj_grad = value_and_grad(lambda params: LMO_err(params,alpha))
    #     res = minimize(obj_grad, x0=paramsH, bounds=bounds, method='L-BFGS-B', jac=True, options={'maxiter': 2})
    #     paramsH = res.x


def summarize_res(sname, datasize):
    print(sname)
    res = []
    times = []
    for i in range(100):
        PATH = ROOT_PATH + "/results/zoo/" + sname + "/"
        filename = PATH + 'LMO_errs_{}_nystr_{}.npy'.format(i, datasize)
        if os.path.exists(filename):
            tmp_res = np.load(filename, allow_pickle=True)
            if tmp_res[-1] is not None:
                res += [tmp_res[-1]]
        time_path = PATH + '/LMO_errs_{}_nystr_{}_time.npy'.format(i, datasize)
        if os.path.exists(time_path):
            t = np.load(time_path)
            times += [t]
    res = np.array(res)
    times = np.array(times)
    res = remove_outliers(res)
    times = np.sort(times)[:80]
    print(times)
    print('mean, std: ', np.mean(res), np.std(res))
    print('time: ', np.mean(times), np.std(times))


if __name__ == '__main__':
    snames = ['step', 'sin', 'abs', 'linear']
    for datasize in [200]:  # [200,2000]
        for sname in snames[1:2]:
            print('\n############# {} ##################'.format(sname))
            for seed in range(1):  # 100
                experiment(sname, seed, datasize, False if datasize < 1000 else True)

            # summarize_res(sname, datasize)
