import os, sys
import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
from util import get_median_inter_mnist, Kernel, load_data, ROOT_PATH, jitchol, _sqdist, \
    remove_outliers, nystrom_decomp, chol_inv, Standardizer
from joblib import Parallel, delayed
import time
import autograd.scipy.linalg as splg
import matplotlib.pyplot as plt

Nfeval = 1
seed = 527
np.random.seed(seed)
JITTER = 1e-7
nystr_M = 100
EYE_nystr = np.eye(nystr_M)
opt_params = None
prev_norm = None
opt_test_err = None


def experiment(sname, seed, datasize, nystr=False):
    def LMO_err(params, M=2):
        al, bl = np.exp(params[1:]),np.exp(params[0])

        L = 0
        for i in range(len(L_init_list)):
            L = L + L_init_list[i] / al[i] / al[i] / 2
        L = bl * bl * np.exp(-L) + 1e-6 * EYEN

        if nystr:
            tmp_mat = L @ eig_vec_K
            C = L - tmp_mat @ np.linalg.inv(eig_vec_K.T @ tmp_mat / N2 + inv_eig_val_K) @ tmp_mat.T / N2
            c = C @ W_nystr_Y * N2
        else:
            LWL_inv = chol_inv(L @ W @ L + L / N2 + JITTER * EYEN)
            C = L @ LWL_inv @ L / N2
            c = C @ W @ Y * N2
        c_y = c - Y

        # if isinstance(c, np.ndarray):
        #     plt.scatter(X,c)
        #     plt.scatter(X, c+np.sqrt(np.diag(C)).reshape((-1,1)))
        #     plt.scatter(X, c - np.sqrt(np.diag(C)).reshape((-1, 1)))
        #     plt.scatter(X,Y)
        #     plt.show()
        lmo_err = 0
        N = 0
        for ii in range(1):
            for i in range(0, X.shape[0], M):
                indices = permutation[i:i + M]
                K_i = W[np.ix_(indices, indices)] * N2
                C_i = C[np.ix_(indices, indices)]
                c_y_i = c_y[indices]
                b_y = np.linalg.inv(np.eye(M) - C_i @ K_i) @ c_y_i
                lmo_err += b_y.T @ K_i @ b_y
                N += 1
        return lmo_err[0, 0] / N / M ** 2

    def callback0(params, timer=None):
        global Nfeval, prev_norm, opt_params, opt_test_err
        if Nfeval % 1 == 0:
            # if len(params)==2:
            #     al, bl = np.exp(params)
            #     W = (np.exp(-W0 / ak / ak / 2) + np.exp(-W0 / ak / ak / 200) + np.exp(-W0 / ak / ak * 50)) / 3 / N2
            # else:
            al, bl = np.exp(params[1:]), np.exp(params[0])

            L = 0
            for i in range(len(L_init_list)):
                L = L + L_init_list[i] / al[i] / al[i] / 2
            L = bl * bl * np.exp(-L) + 1e-6 * EYEN

            test_L = 0
            for i in range(len(test_init_list)):
                test_L = test_L + test_init_list[i] / al[i] / al[i] / 2
            test_L = bl * bl * np.exp(-test_L)

            if nystr:
                alpha = EYEN - eig_vec_K @ np.linalg.inv(
                    eig_vec_K.T @ L @ eig_vec_K / N2 + np.diag(1 / eig_val_K / N2)) @ eig_vec_K.T @ L / N2
                alpha = alpha @ W_nystr @ Y * N2
            else:
                LWL_inv = chol_inv(L @ W @ L + L / N2 + JITTER * EYEN)
                alpha = LWL_inv @ L @ W @ Y
                # L_W_inv = chol_inv(W*N2+L_inv)
            pred_mean = test_L @ alpha
            if timer:
                return
            test_err = ((pred_mean - test_G) ** 2).mean()  # ((pred_mean-test_G)**2/np.diag(pred_cov)).mean()+(np.log(np.diag(pred_cov))).mean()
            norm = alpha.T @ L @ alpha
            print('test_err, norm: ',test_err, norm)

        Nfeval += 1
        # if prev_norm is not None:
        #     if norm[0, 0] / prev_norm >= 3:
        #         if opt_params is None:
        #             opt_test_err = test_err
        #             opt_params = params
        #         print(True, opt_params, opt_test_err, prev_norm)
        #         raise Exception
        #
        # if prev_norm is None or norm[0, 0] <= prev_norm:
        #     prev_norm = norm[0, 0]
        # opt_test_err = test_err
        # opt_params = params
        # print('params,test_err, norm: ', opt_params, opt_test_err, prev_norm)

    Z = np.random.uniform(-3, 3, size=(200, 3))
    e = np.random.normal(0, 1, size=(200, 1))
    gamma = np.random.normal(0, 0.1, size=(200, 1))
    delta = np.random.normal(0, 0.1, size=(200, 1))
    influence = 0.4
    X = Z * (1 - influence) + influence * e + gamma
    tmp_sum = np.mean(X[:, :-1], axis=1, keepdims=True)
    G = np.sin(tmp_sum)  # (tmp_sum + np.abs(tmp_sum)) > 0
    Y = G + e + delta

    sd = Standardizer()
    X, Y, Z, G, X = sd.generate_data(X, Y, Z, G, X)

    test_G = G
    test_X = X

    EYEN = np.eye(X.shape[0])
    ak = get_median_inter_mnist(Z)
    N2 = X.shape[0] ** 2
    W0 = _sqdist(Z, None)
    # W = (np.exp(-W0 / ak / ak / 2)/np.sum(np.exp(-W0 / ak / ak / 2).flatten(),axis=0) + np.exp(-W0 / ak / ak / 200)/np.sum(np.exp(-W0 / ak / ak / 200).flatten(),axis=0)
    #      + np.exp(-W0 / ak / ak * 50)/np.sum(np.exp(-W0 / ak / ak * 50).flatten()))
    W = (np.exp(-W0 / ak / ak / 2)+ np.exp(-W0 / ak / ak / 200)+ np.exp(-W0 / ak / ak * 50) )/3/N2
    W += 1e-6*EYEN/N2
    L_init_list, test_init_list = [_sqdist(X[:, [i]], None) for i in range(X.shape[1])], [_sqdist(test_X[:, [i]], None)
                                                                                          for i in
                                                                                          range(test_X.shape[1])]

    # measure time
    # callback0(np.random.randn(2)/10,True)
    # np.save(ROOT_PATH + "/MMR_IVs/results/zoo/" + sname + '/LMO_errs_{}_nystr_{}_time.npy'.format(seed,train.x.shape[0]),time.time()-t0)
    # return

    params0 = np.random.randn(X.shape[1]+1) / 10
    bounds = None  # [[0.01,10],[0.01,5]]
    if nystr:
        for _ in range(seed + 1):
            random_indices = np.sort(np.random.choice(range(W.shape[0]), nystr_M, replace=False))
        eig_val_K, eig_vec_K = nystrom_decomp(W * N2, random_indices)
        inv_eig_val_K = np.diag(1 / eig_val_K / N2)
        W_nystr = eig_vec_K @ np.diag(eig_val_K) @ eig_vec_K.T / N2
        W_nystr_Y = W_nystr @ Y

    print(' -------  1  --------- ')
    permutation = np.random.permutation(X.shape[0])
    obj_grad = value_and_grad(lambda params: LMO_err(params))
    # res = minimize(obj_grad, x0=params0, bounds=bounds, method='L-BFGS-B', jac=True, options={'maxiter': 5000},
    #                callback=callback0)
    try:
        res = minimize(obj_grad, x0=params0, bounds=bounds, method='CG', jac=True, options={'maxiter': 50},
                       callback=callback0)
        print(res)
    except Exception as e:
        print(e)
    # PATH = ROOT_PATH + "/MMR_IVs/results/zoo/" + sname + "/"
    # os.makedirs(PATH, exist_ok=True)
    # np.save(PATH + 'LMO_errs_{}_nystr_{}.npy'.format(seed, train.x.shape[0]), [opt_params, prev_norm, opt_test_err])

    # print(' --------  2  -------- ')
    # params0 = np.random.randn(4) / 10
    # bounds = None  # [[0.01,10],[0.01,5]]
    # obj_grad = value_and_grad(lambda params: NLL(params))
    # # res = minimize(obj_grad, x0=params0, bounds=bounds, method='L-BFGS-B', jac=True,
    # #                options={'maxiter': 5000, 'disp': False}, callback=callback0)
    # try:
    #     res = minimize(obj_grad, x0=params0, bounds=bounds, method='L-BFGS-B', jac=True,
    #                    options={'maxiter': 5000,'disp':False},callback=callback0)
    # except Exception as e:
    #     print(e)

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
    for datasize in [200]: # [200,2000]
        for sname in snames:
            for seed in range(1): # 100
                opt_params = None
                prev_norm = None
                opt_test_err = None
                experiment(sname, seed, datasize, False)#False if datasize < 1000 else True)

            # summarize_res(sname, datasize)