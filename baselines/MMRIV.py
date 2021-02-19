import os, sys
import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
from util import get_median_inter_mnist, Kernel, load_data, ROOT_PATH, jitchol, _sqdist, \
    remove_outliers, nystrom_decomp, chol_inv
from joblib import Parallel, delayed
from our_method.generate_data import dataset
import autograd.scipy.linalg as splg
import matplotlib.pyplot as plt
import torch

Nfeval = 1
seed = 527
np.random.seed(seed)
JITTER = 1e-7
nystr_M = 100
EYE_nystr = np.eye(nystr_M)
opt_params = None
prev_norm = None
opt_test_err = None


def experiment(jobid):
    def LMO_err(params, M=2):
        al, bl = np.exp(params)
        L = bl * bl * np.exp(-L0 / al / al / 2) + 1e-6 * EYEN
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

    def NLL(params):
        al, bl = np.exp(params)# al, bl, ak, sigma = np.exp(params)
        L = bl * bl * np.exp(-L0 / al / al / 2) + 1e-6 * EYEN
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
            al, bl = np.exp(params)# al,bl,ak,sigma = np.exp(params)
            # W = (np.exp(-(W0 / ak / ak) / 2)+sigma*EYEN) / N2
            L = bl * bl * np.exp(-L0 / al / al / 2) + 1e-6 * EYEN
            if nystr:
                alpha = EYEN - eig_vec_K @ np.linalg.inv(
                    eig_vec_K.T @ L @ eig_vec_K / N2 + np.diag(1 / eig_val_K / N2)) @ eig_vec_K.T @ L / N2
                alpha = alpha @ W_nystr @ Y * N2
            else:
                LWL_inv = chol_inv(L @ W @ L + L / N2 + JITTER * EYEN)
                alpha = LWL_inv @ L @ W @ Y
                # L_W_inv = chol_inv(W*N2+L_inv)
            test_L = bl * bl * np.exp(-test_L0 / al / al / 2)
            pred_mean = test_L @ alpha
            if timer:
                return
            test_err = ((pred_mean - test_G) ** 2).mean()  # ((pred_mean-test_G)**2/np.diag(pred_cov)).mean()+(np.log(np.diag(pred_cov))).mean()
            norm = alpha.T @ L @ alpha
            print('test_err, norm: ',test_err, norm)
            opt_test_err = test_err

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

    print('loading data')
    scenario = 'low_x_z'
    filenames = ['abs_100', 'sin_100', 'linear_100',
                 'step_100']
    rid, fid = divmod(jobid, len(filenames))
    np.random.seed(527 + rid)
    torch.manual_seed(527 + rid)
    filename = filenames[fid]
    print(scenario, filename, filenames)
    data = np.load(ROOT_PATH + '/data/{}/{}.npy'.format(scenario, filename), allow_pickle=True)
    train, dev, test = data
    X = np.vstack((train.X, dev.X))
    Z = np.vstack((train.Z, dev.Z))
    Y = np.vstack((train.Y, dev.Y))
    test_X = test.X
    test_G = test.G
    nystr = X.shape[0] >= 1000
    del data, train, dev, test

    EYEN = np.eye(X.shape[0])
    ak = get_median_inter_mnist(Z[:,[0]])
    N2 = X.shape[0] ** 2
    W0 = _sqdist(Z[:,[0]], None)
    # W = (np.exp(-W0 / ak / ak / 2)/np.sum(np.exp(-W0 / ak / ak / 2).flatten(),axis=0) + np.exp(-W0 / ak / ak / 200)/np.sum(np.exp(-W0 / ak / ak / 200).flatten(),axis=0)
    #      + np.exp(-W0 / ak / ak * 50)/np.sum(np.exp(-W0 / ak / ak * 50).flatten()))
    W = (np.exp(-W0 / ak / ak / 2)+ np.exp(-W0 / ak / ak / 200)+ np.exp(-W0 / ak / ak * 50) )/3/N2
    # W = np.exp(-W0 / ak / ak / 2)
    # W = W/np.sum(W.flatten())
    # print(W.flatten().sum())
    # W /= np.sum(W.flatten())
    # del W0
    L0, test_L0 = _sqdist(X, None), _sqdist(test_X, X)

    # measure time
    # callback0(np.random.randn(2)/10,True)
    # np.save(ROOT_PATH + "/MMR_IVs/results/zoo/" + sname + '/LMO_errs_{}_nystr_{}_time.npy'.format(seed,train.x.shape[0]),time.time()-t0)
    # return

    params0 = np.random.randn(2) / 10
    bounds = [[-3,4],[-3,4]]
    permutation = np.random.permutation(X.shape[0])
    if nystr:
        for _ in range(seed + 1):
            random_indices = np.sort(np.random.choice(range(W.shape[0]), nystr_M, replace=False))
        eig_val_K, eig_vec_K = nystrom_decomp(W * N2, random_indices)
        inv_eig_val_K = np.diag(1 / eig_val_K / N2)
        W_nystr = eig_vec_K @ np.diag(eig_val_K) @ eig_vec_K.T / N2
        W_nystr_Y = W_nystr @ Y

    # print(' -------  1  --------- ')
    obj_grad = value_and_grad(lambda params: LMO_err(params))
    res = minimize(obj_grad, x0=params0, bounds=bounds, method='L-BFGS-B', jac=True, options={'maxiter': 5000},
                   callback=callback0)
    # try:
    #     res = minimize(obj_grad, x0=params0, bounds=bounds, method='L-BFGS-B', jac=True, options={'maxiter': 5000},
    #                    callback=callback0)
    #     print(res)
    # except Exception as e:
    #     print(e)
    res_dir = ROOT_PATH + '/results/{}/{}'.format(scenario,'MMRIV')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    np.save(res_dir+ '/{}_{}.npy'.format(filename, rid),{'learnt_params_f':res.x,'weights_z':ak,'test_err':opt_test_err})

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
    for jobid in range(40): # 100
        opt_params = None
        prev_norm = None
        opt_test_err = None
        experiment(jobid)#False if datasize < 1000 else True)
