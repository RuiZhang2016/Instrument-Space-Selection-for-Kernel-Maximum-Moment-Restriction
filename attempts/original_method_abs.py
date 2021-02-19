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
    def LMO_err(params,alpha, M=2):
        al, bl = np.exp(params)
        L = np.exp(-L0 / al / al / 2) + 1e-6 * EYEN
        alpha = alpha[:,None]
        r = np.sqrt((Y-L@alpha)**2)
        r_mat = np.exp(-_sqdist(r, None)/2/0.0001)
        return np.trace(r_mat@W)+bl*alpha.T@L@alpha

    def NLL(params):
        al, bl = np.exp(params)# al, bl, ak, sigma = np.exp(params)
        L = np.exp(-L0 / al / al / 2) + 1e-6 * EYEN
        #W = (np.exp(-(W0 / ak / ak) / 2)+sigma*EYEN)/ N2

        tri_K = np.linalg.cholesky(W*N2+1e-6*EYEN)
        tri_K_inv = splg.solve_triangular(tri_K, EYEN, lower=True)
        K_inv = np.matmul(tri_K_inv.T, tri_K_inv)
        Mat = L + K_inv*N2
        # jitter = np.diag(K).mean()*JITTER
        tri_Mat = np.linalg.cholesky(Mat+ 1e-6 * EYEN)
        logdetMat = 2 * np.sum(np.log(np.diag(tri_Mat)))
        tri_Mat_inv = splg.solve_triangular(tri_Mat, EYEN, lower=True)
        Mat_inv = np.matmul(tri_Mat_inv.T, tri_Mat_inv)
        nlm = (logdetMat + Y.T@Mat_inv@Y + np.log(2 * np.pi) * len(Y))[0, 0] / 2
        return nlm

    def callback0(params, timer=None):
        global Nfeval, prev_norm, opt_params, opt_test_err
        if Nfeval % 1 == 0:
            # if len(params)==2:
            #     al, bl = np.exp(params)
            #     W = (np.exp(-W0 / ak / ak / 2) + np.exp(-W0 / ak / ak / 200) + np.exp(-W0 / ak / ak * 50)) / 3 / N2
            # else:
            al, bl = np.exp(al0),np.exp(bl0)# al,bl,ak,sigma = np.exp(params)
            # W = (np.exp(-(W0 / ak / ak) / 2)+sigma*EYEN) / N2
            alpha = params[:,None]
            L = np.exp(-L0 / al / al / 2) + 1e-6 * EYEN
            # if nystr:
            #     alpha = EYEN - eig_vec_K @ np.linalg.inv(
            #         eig_vec_K.T @ L @ eig_vec_K / N2 + np.diag(1 / eig_val_K / N2)) @ eig_vec_K.T @ L / N2
            #     alpha = alpha @ W_nystr @ Y * N2
            # else:
            #     LWL_inv = chol_inv(L @ W @ L + L / N2 + JITTER * EYEN)
            #     alpha = LWL_inv @ L @ W @ Y
            #     # L_W_inv = chol_inv(W*N2+L_inv)\
            dev_L = np.exp(-dev_L0 / al / al / 2)
            test_L = np.exp(-test_L0 / al / al / 2)
            pred_mean = test_L @ alpha
            if timer:
                return
            test_err = ((pred_mean - test_G) ** 2).mean()  # ((pred_mean-test_G)**2/np.diag(pred_cov)).mean()+(np.log(np.diag(pred_cov))).mean()
            norm = alpha.T @ L @ alpha
            r = np.abs(Y - L @ alpha)
            r_mat = np.exp(-_sqdist(r, None) / 2 / 0.0001)
            r2 = np.abs(Y - G)
            r_mat2 = np.exp(-_sqdist(r2, None) / 2 / 0.0001)
            r3 = np.abs(dev.y-dev_L@alpha)
            r_mat3 = np.exp(-_sqdist(r3, None) / 2 / 0.0001)
            print(np.trace(r_mat @ W),np.trace( r_mat2@ W),np.trace( r_mat3@ dev_W))

            print('test_err, norm: ',test_err, norm)

            # cumulative = []
            # residual = Y-L@alpha
            # for j in range(X.shape[0]):
            #     tmp = residual[:j, [0]].T @ W[:j, :j] @ residual[:j, [0]] * N2 / (j + 1) ** 2
            #     cumulative += [np.sqrt(tmp[0,0])-np.sum(cumulative)]
            # plt.plot(cumulative)
            # plt.show()

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

    X = train.x#np.vstack((train.x, dev.x))
    Y = train.y#np.vstack((train.y, dev.y))
    Z = train.z #np.vstack((train.z, dev.z))
    G = train.g # np.vstack((train.g, dev.g))
    Z = Z[:,[0]]
    Z_ind = np.argsort(Z,axis=0).flatten()
    X, Y, Z = X[Z_ind,], Y[Z_ind,], Z[Z_ind,]
    test_X = test.x
    test_G = test.g

    t0 = time.time()
    EYEN = np.eye(X.shape[0])
    ak = get_median_inter_mnist(Z[:,[0]])
    N2 = X.shape[0] ** 2
    W0 = _sqdist(Z[:,[0]], None)
    # W = (np.exp(-W0 / ak / ak / 2)/np.sum(np.exp(-W0 / ak / ak / 2).flatten(),axis=0) + np.exp(-W0 / ak / ak / 200)/np.sum(np.exp(-W0 / ak / ak / 200).flatten(),axis=0)
    #      + np.exp(-W0 / ak / ak * 50)/np.sum(np.exp(-W0 / ak / ak * 50).flatten()))
    W = (np.exp(-W0 / ak / ak / 2)+ np.exp(-W0 / ak / ak / 200)+ np.exp(-W0 / ak / ak * 50) )/3/N2
    dev_W0 = _sqdist(dev.z[:,[0]], None)
    dev_W = (np.exp(-dev_W0 / ak / ak / 2)+ np.exp(-dev_W0 / ak / ak / 200)+ np.exp(-dev_W0 / ak / ak * 50) )/3/N2
    print(1)
    # W = np.exp(-W0 / ak / ak / 2)
    # W = W/np.sum(W.flatten())
    # print(W.flatten().sum())
    # W /= np.sum(W.flatten())
    # del W0
    L0, test_L0, dev_L0 = _sqdist(X, None), _sqdist(test_X, X),_sqdist(dev.x, X)

    # measure time
    # callback0(np.random.randn(2)/10,True)
    # np.save(ROOT_PATH + "/MMR_IVs/results/zoo/" + sname + '/LMO_errs_{}_nystr_{}_time.npy'.format(seed,train.x.shape[0]),time.time()-t0)
    # return

    params0 = np.random.randn(X.shape[0]) / 10
    bounds = None  # [[0.01,10],[0.01,5]]
    permutation = np.random.permutation(X.shape[0])
    al0, bl0 = 0.01, -1
    if nystr:
        for _ in range(seed + 1):
            random_indices = np.sort(np.random.choice(range(W.shape[0]), nystr_M, replace=False))
        eig_val_K, eig_vec_K = nystrom_decomp(W * N2, random_indices)
        inv_eig_val_K = np.diag(1 / eig_val_K / N2)
        W_nystr = eig_vec_K @ np.diag(eig_val_K) @ eig_vec_K.T / N2
        W_nystr_Y = W_nystr @ Y

    # print(' -------  1  --------- ')
    obj_grad = value_and_grad(lambda params: LMO_err(np.array([al0,bl0]),params))
    # res = minimize(obj_grad, x0=params0, bounds=bounds, method='L-BFGS-B', jac=True, options={'maxiter': 5000},
    #                callback=callback0)
    try:
        res = minimize(obj_grad, x0=params0, bounds=bounds, method='L-BFGS-B', jac=True, options={'maxiter': 5000},
                       callback=callback0)
    except Exception as e:
        print(e)
    # PATH = ROOT_PATH + "/MMR_IVs/results/zoo/" + sname + "/"
    # os.makedirs(PATH, exist_ok=True)
    # np.save(PATH + 'LMO_errs_{}_nystr_{}.npy'.format(seed, train.x.shape[0]), [opt_params, prev_norm, opt_test_err])

    print(' --------  2  -------- ')
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
        for sname in snames[:1]:
            for seed in range(1): # 100
                opt_params = None
                prev_norm = None
                opt_test_err = None
                experiment(sname, seed, datasize, False)#False if datasize < 1000 else True)

            # summarize_res(sname, datasize)