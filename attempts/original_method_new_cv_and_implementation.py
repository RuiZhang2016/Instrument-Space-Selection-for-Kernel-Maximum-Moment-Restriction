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
nystr_M = 100
EYE_nystr = np.eye(nystr_M)
opt_params = None
prev_norm = None
opt_test_err = None


def experiment(sname, seed, datasize, nystr=False):
    def LMO_err(params, permutation, M=4):
        # al, bl,cl, r00,r01,r02,r10,r11,r12= np.exp(params)
        al, bl, cl, ak0,ak1= np.exp(params)
        cl = cl + 1e-7
        # print('r',r00,r01,r02,r10,r11)
        L = bl * np.exp(-L0 / al / al / 2) + cl*EYEN
        # W = np.exp(-W0 / ak0 / ak0 / 2)+np.exp(W1 / ak1 / ak1 / 2)
        # W = W / W.sum()
        # W = r00 * W00 + r01 * W01 + r02 * W02 + r10 * W10 + r11 * W11 + r12 * W12
        # W = W / (r00 + r01 + r02 + r10 + r11 + r12)

        if nystr:
            tmp_mat = L @ eig_vec_K
            C = L - tmp_mat @ np.linalg.inv(eig_vec_K.T @ tmp_mat / N2 + inv_eig_val_K) @ tmp_mat.T / N2
            c = C @ W_nystr_Y * N2
        else:
            # LWL_inv = # chol_inv((L @ W @ L + L/N2)/1000)/1000
            #LWL_inv = np.linalg.inv(L @ W @ L + L / N2)
            # C = L @ LWL_inv @ L / N2
            C = L @ np.linalg.solve(L @ W @ L + L / N2,L)/ N2
            c = C @ W @ Y * N2
        c_y = c - Y

        lmo_err = 0
        N = 0
        for ii in range(1):
            for i in range(0, X.shape[0], M):
                indices = permutation[i:i + M]
                K_i = W[np.ix_(indices, indices)] * N2
                C_i = C[np.ix_(indices, indices)]
                c_y_i = c_y[indices]
                # b_y = np.linalg.inv(np.eye(M) - C_i @ K_i) @ c_y_i
                b_y = np.linalg.solve(np.eye(M) - C_i @ K_i,c_y_i)
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
            # al, bl, cl, r00,r01,r02,r10,r11,r12 = np.exp(params)# al,bl,ak,sigma = np.exp(params)
            al, bl, cl, ak0, ak1 = np.exp(params)
            cl = cl + 1e-7
            L = bl * np.exp(-L0 / al / al / 2) + cl * EYEN

            #W = r00 * W00 + r01 * W01 + r02 * W02 + r10 * W10 + r11 * W11+r12*W12
            #W = W/(r00+r01+r02+r10+r11+r12)
            W = np.exp(-W0 / ak0 / ak0 / 2)# - W1 / ak1 / ak1 / 2)
            W = W/W.sum()
            if nystr:
                alpha = EYEN - eig_vec_K @ np.linalg.inv(
                    eig_vec_K.T @ L @ eig_vec_K / N2 + np.diag(1 / eig_val_K / N2)) @ eig_vec_K.T @ L / N2
                alpha = alpha @ W_nystr @ Y * N2
            else:
                # LWL_inv = chol_inv(L @ W @ L + L / N2 + JITTER * EYEN)
                # alpha = LWL_inv @ L @ W @ Y
                alpha = np.linalg.solve(L @ W @ L + L / N2,L) @ W @ Y
            test_L = bl* np.exp(-test_L0 / al / al / 2)
            pred_mean = test_L @ alpha
            test_err = ((pred_mean - test_G) ** 2).mean()  # ((pred_mean-test_G)**2/np.diag(pred_cov)).mean()+(np.log(np.diag(pred_cov))).mean()
            norm = alpha.T @ L @ alpha
            print('test_err: {} , norm: {}, al: {}, bl: {}, cl: {}, ak0,ak1: {}, {}'.format(test_err, norm, al, bl, cl, ak0,ak1),(L@alpha).T@L@(L@alpha))
            # print([r00,r01,r02]/np.sum([r00,r01,r02,r10,r11,r12]))
            # print([r10, r11, r12] / np.sum([r00, r01, r02, r10, r11, r12]))


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

    train, dev, test = load_data(ROOT_PATH + '/data/zoo/{}_{}.npz'.format(sname, datasize))

    X = np.vstack((train.x, dev.x))
    Y = np.vstack((train.y, dev.y))
    Z = np.vstack((train.z, dev.z))
    test_X = test.x
    test_G = test.g

    EYEN = np.eye(X.shape[0])
    ak0 = get_median_inter_mnist(Z[:, [0]])
    ak1 = get_median_inter_mnist(Z[:, [1]])
    al = get_median_inter_mnist(X)
    N2 = X.shape[0] ** 2
    W0 = _sqdist(Z[:,[0]], None)
    W1 = _sqdist(Z[:,[1]], None)
    # (np.exp(-W0 / ak0 / ak0 / 2)+ np.exp(-W0 / ak0 / ak0 / 200)+ np.exp(-W0 / ak0 / ak0 * 50) )/3/N2
    # W00, W01, W02 = np.exp(-W0 / ak0 / ak0 / 2), np.exp(-W0 / ak0 / ak0 / 20), np.exp(-W0 / ak0 / ak0 * 5) # (np.exp(-W0 / ak0 / ak0 / 2)+ np.exp(-W0 / ak0 / ak0 / 200)+ np.exp(-W0 / ak0 / ak0 * 50) )/3/N2
    # W00, W01, W02 = W00/W00.sum(), W01/W01.sum(), W02/W02.sum()
    # W10, W11, W12 = np.exp(-W1 / ak1 / ak1 / 2), np.exp(-W1 / ak1 / ak1 / 20), np.exp(-W1 / ak1 / ak1 * 5)
    # W10, W11, W12 = W10 / W10.sum(), W11 / W11.sum(), W12 / W12.sum()
    W = 0
    L0, test_L0 = _sqdist(X, None), _sqdist(test_X, X)

    params0 = np.random.randn(5) / 10-1
    params0[2] = -8.
    params0[0] = np.log(al)
    params0[1] = np.log(np.var(Y))
    params0[3] = np.log(ak0)
    params0[4] = np.log(ak1)

    bound_bl = [np.log(np.var(Y)/5),np.log(np.var(Y)*5)]
    bound_al = [np.log(al/10),np.log(al*10)]
    bounds = [bound_al,bound_bl,[-20,-5], [np.log(ak0/5), np.log(ak0*5)],[np.log(ak1/10), np.log(ak1*10)]]# [[np.log(1-np.exp(-1e-12)), -1e-12]]
    print(bounds)
    permutation = np.random.permutation(X.shape[0])
    if nystr:
        for _ in range(seed + 1):
            random_indices = np.sort(np.random.choice(range(W.shape[0]), nystr_M, replace=False))
        eig_val_K, eig_vec_K = nystrom_decomp(W * N2, random_indices)
        inv_eig_val_K = np.diag(1 / eig_val_K / N2)
        W_nystr = eig_vec_K @ np.diag(eig_val_K) @ eig_vec_K.T / N2
        W_nystr_Y = W_nystr @ Y

    print(' -------  1  --------- ')
    obj_grad = value_and_grad(lambda params: LMO_err(params,permutation))
    # res = minimize(obj_grad, x0=params0, bounds=bounds, method='L-BFGS-B', jac=True, options={'maxiter': 5000},
    #                callback=callback0)
    try:
        res = minimize(obj_grad, x0=params0, bounds=bounds, method='L-BFGS-B', jac=True, options={'maxiter': 5000},
                       callback=callback0)
        print(res)
    except Exception as e:
        print(e)

    # print(LMO_err(np.array([0.5,1]),permutation, M=2))
    # W = W*5
    # print(LMO_err(np.array([0.5,1-np.log(5)/2]),permutation, M=2))


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