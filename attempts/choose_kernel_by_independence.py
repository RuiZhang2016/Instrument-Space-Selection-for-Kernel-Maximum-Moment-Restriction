import os, sys
import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
import autograd.scipy.linalg as sp_linalg
from util import get_median_inter_mnist, Kernel, load_data, ROOT_PATH, jitchol, _sqdist, \
    remove_outliers, nystrom_decomp, chol_inv
from joblib import Parallel, delayed
import time
import autograd.scipy.linalg as splg

Nfeval = 1
seed = 527
np.random.seed(seed)
JITTER = 1e-7
nystr_M = 300
EYE_nystr = np.eye(nystr_M)
opt_params = None
prev_norm = None
opt_test_err = None


def experiment(sname, seed, datasize, nystr=False):
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
        al, bl= np.exp(params)
        L = bl * bl * np.exp(-L0 / al / al / 2) + 1e-6 * EYEN

        Mat = L + K_inv*N2
        # jitter = np.diag(K).mean()*JITTER
        tri_Mat = np.linalg.cholesky(Mat+ 1e-6 * EYEN)
        logdetMat = 2 * np.sum(np.log(np.diag(tri_Mat)))
        tri_Mat_inv = splg.solve_triangular(tri_Mat, EYEN, lower=True)
        Mat_inv = np.matmul(tri_Mat_inv.T, tri_Mat_inv)
        nlm = (logdetMat + Y.T@Mat_inv@Y + np.log(2 * np.pi) * len(Y))[0, 0] / 2
        return nlm

    def cov_x_hz(x,params):
        ak0,ak1, sigma = np.exp(params)
        W = np.exp(-(W0 / ak0**2+W1/ak1**2) / 2)+sigma*EYEN
        # one_vec = np.ones((x.shape[0],1))
        N = x.shape[0]
        mean_xkx = np.mean((x@x.T * W).flatten())
        mean_x = np.mean(x,axis=0)
        mean_xk = np.mean(x.T@W/N,axis=1)
        mean_k = np.mean(W.flatten())
        # cov = x.T @ W @ x +(one_vec.T @ x/N)**2 * one_vec.T @ W @ one_vec - 2/N * one_vec.T @ x * x.T@W@one_vec
        cov = mean_xkx+mean_x@mean_x*mean_k-2*mean_x@mean_xk
        return cov

    def cov_xr(x,params, al,bl):
        al, bl = np.exp(al), np.exp(bl)
        ak0, ak1, sigma = np.exp(params)
        W = np.exp(-(W0 / ak0 ** 2 + W1 / ak1 ** 2) / 2) + sigma * EYEN
        L = bl * bl * np.exp(-L0 / al / al / 2) + 1e-6 * EYEN

        if nystr:
            tmp_mat = L @ eig_vec_K
            C = L - tmp_mat @ np.linalg.inv(eig_vec_K.T @ tmp_mat / N2 + inv_eig_val_K) @ tmp_mat.T / N2
            c = C @ W_nystr_Y * N2
        else:
            LWL_inv = chol_inv(L @ W @ L / N2 + L / N2 + JITTER * EYEN)  # old: no /N2
            C = L @ LWL_inv @ L / N2
            c = C @ W @ Y  # old: *N2

        r = Y - c
        return (x*r).T @ W @ (x*r)

    def HSIC_fx_hz(params,al,bl):
        al,bl = np.exp(al),np.exp(bl)
        ak0,ak1, sigma = np.exp(params)
        W = np.exp(-(W0 / ak0**2+W1/ak1**2) / 2)+sigma*EYEN
        L = np.exp(-L0 / 2)# bl * bl * np.exp(-L0 / al / al / 2) + 1e-6 * EYEN
        H = EYEN - 1/L.shape[0]*np.ones(L.shape)
        HSIC = np.trace(L@W)#np.trace(L@H@W@H)**2/np.trace(W@H@W@H)/np.trace(L@H@L@H) # normalized
        return HSIC

    def CI_xr_z(params,al,bl,ar):
        al,bl,ar = np.exp(al),np.exp(bl),np.exp(ar)
        ak0,ak1, sigma = np.exp(params)
        W = np.exp(-(W0 / ak0**2+W1/ak1**2) / 2)+sigma*EYEN
        L = bl * bl * np.exp(-L0 / al / al / 2) + 1e-6 * EYEN
        H = EYEN - 1/L.shape[0]*np.ones(L.shape)
        W_tilde = H@W@H

        if nystr:
            tmp_mat = L @ eig_vec_K
            C = L - tmp_mat @ np.linalg.inv(eig_vec_K.T @ tmp_mat / N2 + inv_eig_val_K) @ tmp_mat.T / N2
            c = C @ W_nystr_Y * N2
        else:
            LWL_inv = chol_inv(L @ W @ L/N2 + L / N2 + JITTER * EYEN) #old: no /N2
            C = L @ LWL_inv @ L / N2
            c = C @ W @ Y #old: *N2

        r = Y-c
        epsilon = 1e-6
        K_r = np.exp(-_sqdist(r, None)/ar**2/2)
        Rz = epsilon*np.linalg.inv(W_tilde + epsilon*EYEN)
        Kxz_tilde = Rz@(H@(W*L)@H)@Rz
        Ky_z = Rz@(H@K_r@H)@Rz
        CI = np.trace(Kxz_tilde@Ky_z)/H.shape[0]# **2/np.trace(Kxz_tilde@Kxz_tilde)/np.trace(Ky_z@Ky_z)
        return CI

    def HSIC_rz(params,al,bl,ar):
        al,bl,ar = np.exp(al),np.exp(bl),np.exp(ar)
        ak0,ak1, sigma = np.exp(params)
        W = np.exp(-(W0 / ak0**2+W1/ak1**2) / 2)+sigma*EYEN
        L = bl * bl * np.exp(-L0 / al / al / 2) + 1e-6 * EYEN
        H = EYEN - 1/L.shape[0]*np.ones(L.shape)

        if nystr:
            tmp_mat = L @ eig_vec_K
            C = L - tmp_mat @ np.linalg.inv(eig_vec_K.T @ tmp_mat / N2 + inv_eig_val_K) @ tmp_mat.T / N2
            c = C @ W_nystr_Y * N2
        else:
            LWL_inv = chol_inv(L @ W @ L/N2 + L / N2 + JITTER * EYEN) #old: no /N2
            C = L @ LWL_inv @ L / N2
            c = C @ W @ Y #old: *N2

        r = Y-c
        K_r = np.exp(-_sqdist(r, None)/ar**2/2)
        HSIC = np.trace(K_r@H@W@H)**2/np.trace(K_r@H@K_r@H)/np.trace(W@H@W@H)
        return HSIC

    def HSIC_rx(params,al,bl,ar):
        al,bl,ar = np.exp(al),np.exp(bl),np.exp(ar)
        ak0,ak1, sigma = np.exp(params)
        W = np.exp(-(W0 / ak0**2+W1/ak1**2) / 2)+sigma*EYEN
        L = bl * bl * np.exp(-L0 / al / al / 2) + 1e-6 * EYEN
        H = EYEN - 1/L.shape[0]*np.ones(L.shape)

        if nystr:
            tmp_mat = L @ eig_vec_K
            C = L - tmp_mat @ np.linalg.inv(eig_vec_K.T @ tmp_mat / N2 + inv_eig_val_K) @ tmp_mat.T / N2
            c = C @ W_nystr_Y * N2
        else:
            LWL_inv = chol_inv(L @ W @ L/N2 + L / N2 + JITTER * EYEN) #old: no /N2
            C = L @ LWL_inv @ L / N2
            c = C @ W @ Y #old: *N2

        r = Y-c
        K_r = np.exp(-_sqdist(r, None)/ar**2/2)
        HSIC = np.trace(K_r@H@L@H)#**2/np.trace(K_r@H@K_r@H)/np.trace(W@H@W@H)
        return HSIC

    def select_ak(x):
        params0 = np.random.randn(3) / 10
        bounds = None
        obj_grad = value_and_grad(lambda params: -cov_x_hz(x,params))
        res = minimize(obj_grad, x0=params0, bounds=bounds, method='L-BFGS-B', jac=True,
                       options={'maxiter': 5000, 'disp': False})
        return res.x

    def select_ak2(ak0,ak1,sigma, al,bl):
        bounds = None
        params0 = np.array([ak0,ak1,sigma])
        obj_grad = value_and_grad(lambda params: -HSIC_fx_hz(params,al,bl))
        res = minimize(obj_grad, x0=params0, bounds=bounds, method='L-BFGS-B', jac=True,
                       options={'maxiter': 1, 'disp': False})
        return res.x

    def select_ak3(ak0,ak1,sigma, al,bl,ar):
        bounds = None
        params0 = np.array([ak0,ak1,sigma])
        obj_grad = value_and_grad(lambda params: CI_xr_z(params,al,bl,ar))
        res = minimize(obj_grad, x0=params0, bounds=bounds, method='L-BFGS-B', jac=True,
                       options={'maxiter': 100, 'disp': False})
        return res.x

    def select_ak4(ak0,ak1,sigma, al,bl,ar):
        bounds = None
        params0 = np.array([ak0,ak1,sigma])
        obj_grad = value_and_grad(lambda params: -HSIC_rz(params,al,bl,ar))
        res = minimize(obj_grad, x0=params0, bounds=bounds, method='L-BFGS-B', jac=True,
                       options={'maxiter': 100, 'disp': False})
        return res.x

    def select_ar(ak0,ak1,sigma, al,bl,ar):
        bounds = None
        params = np.array([ak0,ak1,sigma])
        obj_grad = value_and_grad(lambda p: HSIC_rz(params,al,bl,p))
        res = minimize(obj_grad, x0=ar, bounds=bounds, method='L-BFGS-B', jac=True,
                       options={'maxiter': 1, 'disp': False})
        return res.x


    def select_ar2(ak0,ak1,sigma, al,bl,ar):
        bounds = None
        params = np.array([ak0,ak1,sigma])
        obj_grad = value_and_grad(lambda p: CI_xr_z(params,al,bl,p))
        res = minimize(obj_grad, x0=ar, bounds=bounds, method='L-BFGS-B', jac=True,
                       options={'maxiter': 100, 'disp': False})
        return res.x



    def select_ar3(ak0,ak1,sigma, al,bl):
        al, bl = np.exp(al), np.exp(bl)
        ak0, ak1, sigma = np.exp(ak0), np.exp(ak1), np.exp(sigma)
        W = np.exp(-(W0 / ak0 ** 2 + W1 / ak1 ** 2) / 2) + sigma * EYEN
        L = bl * bl * np.exp(-L0 / al / al / 2) + 1e-6 * EYEN

        if nystr:
            tmp_mat = L @ eig_vec_K
            C = L - tmp_mat @ np.linalg.inv(eig_vec_K.T @ tmp_mat / N2 + inv_eig_val_K) @ tmp_mat.T / N2
            c = C @ W_nystr_Y * N2
        else:
            LWL_inv = chol_inv(L @ W @ L / N2 + L / N2 + JITTER * EYEN)  # old: no /N2
            C = L @ LWL_inv @ L / N2
            c = C @ W @ Y  # old: *N2

        r = Y - c
        ar = get_median_inter_mnist(r)
        return np.log(ar)


    def callback0(params, timer=None):
        global Nfeval, prev_norm, opt_params, opt_test_err
        if Nfeval % 1 == 0:
            # if len(params)==2:
            #     al, bl = np.exp(params)
            #     W = (np.exp(-W0 / ak / ak / 2) + np.exp(-W0 / ak / ak / 200) + np.exp(-W0 / ak / ak * 50)) / 3 / N2
            # else:
            al,bl = np.exp(params)
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

        Nfeval += 1
        if prev_norm is not None:
            if norm[0, 0] / prev_norm >= 3:
                if opt_params is None:
                    opt_test_err = test_err
                    opt_params = params
                print(True, opt_params, opt_test_err, prev_norm)
                raise Exception

        if prev_norm is None or norm[0, 0] <= prev_norm:
            prev_norm = norm[0, 0]
        opt_test_err = test_err
        opt_params = params
        print('params,test_err, norm: ', opt_params, opt_test_err, prev_norm) #HSIC_fx_hz(np.log(np.array([ak0,ak1, sigma])),np.log(al),np.log(bl)))

    train, dev, test = load_data(ROOT_PATH + '/data/zoo/{}_{}.npz'.format(sname, datasize))

    X = np.vstack((train.x, dev.x))
    Y = np.vstack((train.y, dev.y))
    Z = np.vstack((train.z, dev.z))
    # Z = np.hstack((Z[:,[0]],Z[:,[0]]))
    test_X = test.x
    test_G = test.g

    EYEN = np.eye(X.shape[0])
    N2 = X.shape[0] ** 2
    W0 = _sqdist(Z[:,[0]], None)
    W1 = _sqdist(Z[:,[1]], None)
    # del W0
    L0, test_L0 = _sqdist(X, None), _sqdist(test_X, X)
    ak0 = get_median_inter_mnist(Z[:, [0]])
    ak1 = get_median_inter_mnist(Z[:, [1]])
    sigma = 0.1
    # measure time
    # callback0(np.random.randn(2)/10,True)
    # np.save(ROOT_PATH + "/MMR_IVs/results/zoo/" + sname + '/LMO_errs_{}_nystr_{}_time.npy'.format(seed,train.x.shape[0]),time.time()-t0)
    # return

    # params0 = np.random.randn(2) / 10
    # bounds = None  # [[0.01,10],[0.01,5]]
    # ak0, ak1, sigma = select_ak(X)
    ar = np.log(get_median_inter_mnist(Y)/10)
    ak0, ak1 = np.log(ak0), np.log(ak1)
    params0 = np.random.randn(2) / 10
    print(' -------  1  --------- ')
    permutation = np.random.permutation(X.shape[0])
    for ite in range(10):
        ak0,ak1, sigma = np.exp(ak0),np.exp(ak1),np.exp(sigma)
        print('ak0, ak1, sigma,ar: ', ak0,ak1, sigma,np.exp(ar))
        W = (np.exp(-(W0 / ak0**2+W1/ak1**2) / 2)+sigma*EYEN)/ N2
        tri_K = np.linalg.cholesky(W * N2 + 1e-6 * EYEN)
        tri_K_inv = splg.solve_triangular(tri_K, EYEN, lower=True)
        K_inv = np.matmul(tri_K_inv.T, tri_K_inv)
        if nystr:
            for _ in range(seed + 1):
                random_indices = np.sort(np.random.choice(range(W.shape[0]), nystr_M, replace=False))
            eig_val_K, eig_vec_K = nystrom_decomp(W * N2, random_indices)
            inv_eig_val_K = np.diag(1 / eig_val_K / N2)
            W_nystr = eig_vec_K @ np.diag(eig_val_K) @ eig_vec_K.T / N2
            W_nystr_Y = W_nystr @ Y

        if opt_params is not None:
            params0 = opt_params
        bounds = None
        obj_grad = value_and_grad(lambda params: LMO_err(params))
        # +1e-3*HSIC_rx(np.log([ak0,ak1,sigma]),params[0],params[1],ar))
        # +1e-3*CI_xr_z(np.log([ak0,ak1,sigma]),params[0],params[1],ar))
        # 1e-3*HSIC_rx(np.log([ak0,ak1,sigma]),params[0],params[1],ar))
        #+2*HSIC_fx_hz(np.log([ak0,ak1,sigma]),params[0],params[1]))
        try:
            res = minimize(obj_grad, x0=params0, bounds=bounds, method='L-BFGS-B', jac=True, options={'maxiter': 10},
                           callback=callback0)
        except Exception as e:
            print(e)

        # PATH = ROOT_PATH + "/MMR_IVs/results/zoo/" + sname + "/"
        # os.makedirs(PATH, exist_ok=True)
        # np.save(PATH + 'LMO_errs_{}_nystr_{}.npy'.format(seed, train.x.shape[0]), [opt_params, prev_norm, opt_test_err])



        # print(' --------  2  -------- ')
        # params0 = np.random.randn(2) / 10
        # bounds = None  # [[0.01,10],[0.01,5]]
        # obj_grad = value_and_grad(lambda params: NLL(params))
        # try:
        #     res = minimize(obj_grad, x0=params0, bounds=bounds, method='L-BFGS-B', jac=True,
        #                    options={'maxiter': 5000,'disp':False},callback=callback0)
        # except Exception as e:
        #     print(e)
        # ar = select_ar(np.log(ak0),np.log(ak1),np.log(sigma), opt_params[0], opt_params[1], ar)
        # ak0,ak1, sigma = select_ak4(np.log(ak0),np.log(ak1),np.log(sigma), opt_params[0],opt_params[1],ar)
        # ar = select_ar3(ak0, ak1, sigma, opt_params[0], opt_params[1])
        ak0,ak1, sigma = select_ak2(np.log(ak0),np.log(ak1),np.log(sigma), opt_params[0],opt_params[1])
        # ak0, ak1, sigma = select_kernel(X)

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
            print('\n###########', sname, '#############\n')
            for seed in range(1): # 100
                opt_params = None
                prev_norm = None
                opt_test_err = None
                experiment(sname, seed, datasize,False)#, False if datasize < 1000 else True)

            # summarize_res(sname, datasize)