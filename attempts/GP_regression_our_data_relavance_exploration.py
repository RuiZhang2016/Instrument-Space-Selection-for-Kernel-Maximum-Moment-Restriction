from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.numpy.linalg import solve
import autograd.scipy.stats.multivariate_normal as mvn
from autograd import value_and_grad
from scipy.optimize import minimize
from util import get_median_inter_mnist, Kernel, load_data, ROOT_PATH, jitchol, _sqdist, \
    remove_outliers, nystrom_decomp, chol_inv, Standardizer

import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
from util import _sqdist,nystrom_decomp, chol_inv
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

def make_gp_funs(cov_func, num_cov_params):
    """Functions that perform Gaussian process regression.
       cov_func has signature (cov_params, x, x')"""

    def unpack_kernel_params(params):
        mean        = params[0]
        cov_params  = params[2:]
        noise_scale = np.exp(params[1]) + 0.0001
        return mean, cov_params, noise_scale

    def predict(params, x, y, xstar):
        """Returns the predictive mean and covariance at locations xstar,
           of the latent function value f (without observation noise)."""
        mean, cov_params, noise_scale = unpack_kernel_params(params)
        cov_f_f = cov_func(cov_params, xstar, xstar)
        cov_y_f = cov_func(cov_params, x, xstar)
        cov_y_y = cov_func(cov_params, x, x) + noise_scale * np.eye(len(y))
        pred_mean = mean +   np.dot(solve(cov_y_y, cov_y_f).T, y - mean)
        pred_cov = cov_f_f - np.dot(solve(cov_y_y, cov_y_f).T, cov_y_f)
        return pred_mean, pred_cov

    def log_marginal_likelihood(params, x, y):
        mean, cov_params, noise_scale = unpack_kernel_params(params)
        cov_y_y = cov_func(cov_params, x, x) + noise_scale * np.eye(len(y))
        prior_mean = mean * np.ones(len(y))
        _,val = np.linalg.slogdet(cov_y_y)
        return -(y-prior_mean)@solve(cov_y_y,np.eye(cov_y_y.shape[0]))@(y-prior_mean)/2-1/2*val-cov_y_y.shape[0]/2*np.log(2*np.pi)
        # return mvn.logpdf(y, prior_mean, cov_y_y)

    return num_cov_params + 2, predict, log_marginal_likelihood

# Define an example covariance function.
def rbf_covariance(kernel_params, x, xp):
    output_scale = np.exp(kernel_params[0])
    lengthscales = np.exp(kernel_params[1:])
    diffs = np.expand_dims(x /lengthscales, 1)\
          - np.expand_dims(xp/lengthscales, 0)
    return output_scale * np.exp(-0.5 * np.sum(diffs**2, axis=2))


def build_toy_dataset(D=1, n_data=20, noise_std=0.1):
    rs = npr.RandomState(0)
    inputs  = np.concatenate([np.linspace(0, 3, num=n_data/2),
                              np.linspace(6, 8, num=n_data/2)])
    targets = (np.cos(inputs) + rs.randn(n_data) * noise_std) / 2.0
    inputs = (inputs - 4.0) / 2.0
    inputs  = inputs.reshape((len(inputs), D))
    return inputs, targets

def experiment(cov_params_list, seed, data, W_init_list,L_init_list, test_init_list, params0, nystr=False):
    def LMO_err(params, M=2):
        al, bl = np.exp(params[1:]),np.exp(params[0])
        assert len(al) == len(L_init_list)
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
            al, bl = np.exp(params[1:]),np.exp(params[0])
            # W = (np.exp(-(W0 / ak / ak) / 2)+sigma*EYEN) / N2
            L = 0
            for i in range(len(L_init_list)):
                L = L + L_init_list[i] / al[i] / al[i]/2
            L = bl * bl *np.exp(-L)+ 1e-6*EYEN

            test_L = 0
            for i in range(len(test_init_list)):
                test_L = test_L + test_init_list[i] / al[i] / al[i] / 2
            test_L = bl * bl *np.exp(-test_L)

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
            print('test_err, norm, params: ',test_err, norm, al)

        Nfeval += 1

    Y, test_G = data
    EYEN = np.eye(X.shape[0])
    N2 = X.shape[0] ** 2

    bounds = None  # [[0.01,10],[0.01,5]]
    permutation = np.random.permutation(X.shape[0])

    W = 0
    al = np.exp(params0[1:])
    for i in range(len(W_init_list)):
        W = W+np.exp(-W_init_list[i] / 2)
    W = W/ N2/len(W_init_list)

    # W = 0
    # al = np.exp(params0[1:])
    # for i in range(len(W_init_list)):
    #     W = W+np.exp(-W_init_list[i] / 2)/ al[i] / al[i]
    # W = (W - 1e-6 * EYEN) / N2

    print(' -------  1  --------- ')

    obj_grad = value_and_grad(lambda params: LMO_err(params))
    # res = minimize(obj_grad, x0=params0, bounds=bounds, method='L-BFGS-B', jac=True, options={'maxiter': 5000},
    #                callback=callback0)
    res = minimize(obj_grad, x0=params0, bounds=bounds, method='CG', jac=True, options={'maxiter': 10,"disp": True},
                   callback=callback0)
    params0 = res.x
    return params0

nystr = False

if __name__ == '__main__':

    for ite in range(10):

        Z = np.random.uniform(-3,3, size=(200,3))
        e = np.random.normal(0, 1, size=(200,1))
        gamma = np.random.normal(0, 0.1, size=(200, 1))
        delta = np.random.normal(0, 0.1, size=(200, 1))
        influence = 0.4
        X = Z*(1-influence) + influence*e + gamma
        tmp_sum = np.mean(X[:, :-1], axis=1, keepdims=True)
        G = tmp_sum#(tmp_sum + np.abs(tmp_sum)) > 0
        Y = G + e + delta

        sd = Standardizer()
        X,Y,Z,G,X = sd.generate_data(X,Y,Z,G,X)
        EYEN = np.eye(X.shape[0])
        N2 = X.shape[0]**2

        L_init_list = [_sqdist(X[:, [i]], None) for i in range(X.shape[1])]
        test_X = X
        test_init_list = [_sqdist(test_X[:, [i]], None) for i in range(test_X.shape[1])]
        W = 0

        if nystr:
            for _ in range(seed + 1):
                random_indices = np.sort(np.random.choice(range(W.shape[0]), nystr_M, replace=False))
            eig_val_K, eig_vec_K = nystrom_decomp(W * N2, random_indices)
            inv_eig_val_K = np.diag(1 / eig_val_K / N2)
            W_nystr = eig_vec_K @ np.diag(eig_val_K) @ eig_vec_K.T / N2
            W_nystr_Y = W_nystr @ Y

        # plt.figure(figsize=(6,12))
        # plt.subplot(3, 1, 1)
        # plt.scatter(X[:,[0]],G)
        # plt.subplot(3, 1, 2)
        # plt.scatter(X[:, [1]], G)
        # plt.subplot(3, 1, 3)
        # plt.scatter(X[:, [0]], Y)
        # plt.show()



        params0 = np.random.randn(X.shape[1] + 1) / 10
        W_init_list = None

        for ro in range(10):
            cov_params_list = []
            al, bl = np.exp(params0[1:]), np.exp(params0[0])

            if ro < 1:
                ak = get_median_inter_mnist(Z)
                W0 = _sqdist(Z, None)
                W = (np.exp(-W0 / ak / ak / 2) + np.exp(-W0 / ak / ak / 200) + np.exp(-W0 / ak / ak * 50)) / 3 / N2
                W -= 1e-6 * EYEN / N2
            else:
                W = EYEN
                al = np.exp(params0[1:])
                for i in range(len(W_init_list)):
                    W = W * np.exp(-W_init_list[i] / 2 / al[i] ** 2)
                W = W / N2

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

            for xi in range(X.shape[1]):
                y = X[:,xi]

                x = np.hstack((Z,e))
                D = x.shape[1]

                # Build model and objective function.
                num_params, predict, log_marginal_likelihood = \
                    make_gp_funs(rbf_covariance, num_cov_params=D + 1)

                objective = lambda params: -log_marginal_likelihood(params, x, y)

                def callback(params):
                    print("Log likelihood {}".format(-objective(params)))
                    plt.cla()

                    # Show posterior marginals.
                    plot_xs = np.reshape(np.linspace(-7, 7, 300), (300,1))
                    pred_mean, pred_cov = predict(params, x, y, plot_xs)
                    tmp_cov_params = params[2:]
                    print('lengthscales:', np.exp(tmp_cov_params[1:]))

                # Initialize covariance parameters
                rs = npr.RandomState(0)
                init_params = 0.1 * rs.randn(num_params)

                print("Optimizing covariance parameters...")
                cov_params = minimize(value_and_grad(objective), init_params, jac=True,
                                      method='CG',options={'maxiter':100})
                cov_params_list += [cov_params.x[3:-1]]
                print(cov_params.x[3:])

            print(cov_params_list)
            W_init_list = [_sqdist(Z / np.exp(cov_params_list[i]), None) for i in range(len(cov_params_list))]
            params0 = experiment(cov_params_list,0,[Y,G],W_init_list,L_init_list,test_init_list, params0)
