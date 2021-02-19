from autograd.numpy.linalg import solve
from util import get_median_inter_mnist, ROOT_PATH, jitchol, _sqdist, \
    remove_outliers, nystrom_decomp, chol_inv
import os, sys, argparse
import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import torch
from our_method.generate_data import dataset

Nfeval = 1
seed = 527
JITTER = 1e-7
nystr_M = 300
EYE_nystr = np.eye(nystr_M)
opt_params = None
prev_norm = None
opt_test_err = None
c_y_record = None

def make_gp_funs(cov_func, num_cov_params):
    """Functions that perform Gaussian process regression.
       cov_func has signature (cov_params, x, x')"""

    def unpack_kernel_params(params):
        mean        = params[0]
        cov_params  = params[2:]
        noise_scale = np.exp(params[1]) + 1e-7
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


def step2(data, W,L_init_list, test_init_list, params0,permutation, nystr_components,nystr=False):
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
        global c_y_record
        c_y_record = c_y

        # if isinstance(c, np.ndarray):
        #     plt.scatter(X,c)
        #     plt.scatter(X, c+np.sqrt(np.diag(C)).reshape((-1,1)))
        #     plt.scatter(X, c-np.sqrt(np.diag(C)).reshape((-1, 1)))
        #     plt.scatter(X,Y)
        #     plt.show()

        lmo_err = 0
        N = 0
        for ii in range(1):
            for i in range(0, Y.shape[0], M):
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
            al, bl = np.exp(params[1:]),np.exp(params[0])
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
                alpha = alpha @ W_nystr_Y * N2
            else:
                LWL_inv = chol_inv(L @ W @ L + L / N2 + JITTER * EYEN)
                alpha = LWL_inv @ L @ W @ Y
            pred_mean = test_L @ alpha
            if timer:
                return
            test_err = ((pred_mean - test_G) ** 2).mean()  # ((pred_mean-test_G)**2/np.diag(pred_cov)).mean()+(np.log(np.diag(pred_cov))).mean()
            norm = alpha.T @ L @ alpha
            print('test_err, norm, al: ',test_err, norm[0,0], al)
            opt_test_err = test_err
        Nfeval += 1

    if nystr:
        eig_val_K, eig_vec_K, inv_eig_val_K, W_nystr_Y = nystr_components
    Y, test_G = data
    EYEN = np.eye(Y.shape[0])
    N2 = Y.shape[0] ** 2

    up_bounds = params0.copy()+np.log(10)
    low_bounds = params0.copy()-np.log(10)
    low_bounds = [e if e<-1 else -1 for e in low_bounds]
    up_bounds = [e if e < 4 else 4 for e in up_bounds]
    bounds = list(zip(low_bounds,up_bounds))
    print('step 2')
    print('bounds:', bounds)

    obj_grad = value_and_grad(lambda params: LMO_err(params))
    res = minimize(obj_grad, x0=params0, bounds=bounds, method='L-BFGS-B', jac=True, options={'maxiter': 100},
                   callback=callback0)
    return res.x

def step1(x,y, mi_Z):

    D = x.shape[1]
    num_params, predict, log_marginal_likelihood = \
        make_gp_funs(rbf_covariance, num_cov_params=D + 1)

    init_params = 0.1 * np.random.randn(num_params)
    mean_y = np.mean(y, axis=0)
    std_y = np.std(y, axis=0)
    init_params[0] = mean_y
    init_params[1] = np.log(1e-4)
    init_params[2] = np.log(D / np.pi)
    init_params[3:-1] = [np.log(mi_Z[i]) for i in range(x.shape[1]-1)]
    init_params[-1] = np.log(get_median_inter_mnist(x[:,[-1]]))
    up_bounds = init_params.copy()
    low_bounds = init_params.copy()

    up_bounds[0] += 2 * std_y
    up_bounds[1] = np.log(std_y)
    up_bounds[2:] += np.log(10)
    low_bounds[0] -= 2 * std_y
    low_bounds[1] -= np.log(100)
    low_bounds[2:] -= np.log(10)

    objective = lambda params: -log_marginal_likelihood(params, x, y)  # np.hstack(([mean_y],params))

    def callback(params):
        print("Log likelihood {}".format(-objective(params)))
        # Show posterior marginals.
        tmp_cov_params = params[2:]
        print('lengthscales:', np.exp(tmp_cov_params[1:]))

    print("Optimizing covariance parameters (k)...")
    up_bounds[-1] = init_params[-1] + np.log(10)
    low_bounds[-1] = init_params[-1] - np.log(10)

    bounds = list(zip(low_bounds, up_bounds))
    print('step 1 ')
    print('bounds: ', bounds)
    cov_params = minimize(value_and_grad(objective), init_params, jac=True, bounds=bounds,
                          method='L-BFGS-B', options={'maxiter': 1000})
    print('optimized covariance params (k)', cov_params.x)
    return cov_params.x


def experiment(jobid):
    print('loading data')
    scenario = 'low_x_z'
    filenames = ['abs_100','sin_100','linear_100','step_100'] # os.listdir(ROOT_PATH + '/data/' + scenario)
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

    print('preparing for computation')
    permutation = np.random.permutation(Y.shape[0])
    EYEN = np.eye(X.shape[0])
    N2 = X.shape[0] ** 2
    L_init_list = [_sqdist(X[:, [i]], None) for i in range(X.shape[1])]
    test_init_list = [_sqdist(test_X[:, [i]], X[:,[i]]) for i in range(test_X.shape[1])]
    mi_Z = [get_median_inter_mnist(Z[:,[i]]) for i in range(Z.shape[1])]

    print('initialization')
    params0 = np.random.randn(X.shape[1] + 1) / 10

    params0[0] = np.log((Z.shape[1]+1) / np.pi)
    params0[1:] = [np.log(get_median_inter_mnist(X[:, [i]])) for i in range(X.shape[1])]

    learnt_params_f = None
    random_indices = None

    if nystr:
        for _ in range(rid + 1):
            random_indices = np.sort(np.random.choice(range(Z.shape[0]), nystr_M, replace=False))

    for ro in range(50):

        # cold start
        if ro < 1:
            ak = get_median_inter_mnist(Z)
            W0 = _sqdist(Z, None)
            W = (np.exp(-W0 / ak / ak / 2) + np.exp(-W0 / ak / ak / 200) + np.exp(-W0 / ak / ak * 50)) / 3 / N2
            W -= 1e-6 * EYEN / N2
        else:
            W0 = _sqdist(Z / np.exp(init_params[3:-1]), None)
            W = (np.exp(-W0 / 2) + np.exp(-W0 / 200) + np.exp(-W0 * 50)) / 3 / N2
            W -= 1e-6 * EYEN / N2

        if nystr:
            eig_val_K, eig_vec_K = nystrom_decomp(W * N2, random_indices)
            inv_eig_val_K = np.diag(1 / eig_val_K / N2)
            W_nystr = eig_vec_K @ np.diag(eig_val_K) @ eig_vec_K.T / N2
            W_nystr_Y = W_nystr @ Y

        # stage 1
        learnt_params_f = step2([Y, test_G], W, L_init_list, test_init_list, params0, permutation, [eig_val_K,eig_vec_K,inv_eig_val_K,W_nystr_Y] if nystr else None, nystr)
        print('learnt_param_f: ', learnt_params_f)

        # stage 2
        init_params = step1(np.hstack((Z, c_y_record)),Y.flatten(),mi_Z) # z weights

        print('ratios for stop criteria', abs(np.exp(params0) - np.exp(learnt_params_f)) / abs(np.exp(params0)))
        if np.max(abs(np.exp(params0) - np.exp(learnt_params_f)) / abs(np.exp(params0))) < 1e-2:
            break
        params0 = learnt_params_f

    res_dir = ROOT_PATH + '/results/{}/{}'.format(scenario,'our_method')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    np.save(res_dir+ '/{}_{}.npy'.format(filename, rid),{'learnt_params_f':learnt_params_f,'weights_z':init_params,'test_err':opt_test_err})

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--jobid', type=int, help='jobid on cluster')
    # args = parser.parse_args(sys.argv[1:])
    for jobid in range(40):
        opt_params = None
        prev_norm = None
        opt_test_err = None
        experiment(jobid)

