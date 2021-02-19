import autograd.numpy as np
import autograd.scipy.misc as sp_misc
import autograd.scipy.stats as sp_stats
import copy

from autograd import value_and_grad, grad, jacobian
from scipy.optimize import minimize
from util import get_median_inter_mnist, Kernel, load_data, ROOT_PATH, jitchol, _sqdist
import matplotlib.pyplot as plt

N = 1
learning_rate_theta = 5e-4
learning_rate_lambda = 1e-2
HVP_i = 50

# TODO: put into a function
sname = 'linear'
datasize=200
train, dev, test = load_data(ROOT_PATH + '/data/zoo/{}_{}.npz'.format(sname, datasize))
X = np.vstack((train.x, dev.x))
Y = np.vstack((train.y, dev.y))
Z = np.vstack((train.z, dev.z))
test_X = test.x
test_G = test.g

EYEN = np.eye(X.shape[0])
ak = get_median_inter_mnist(Z)
N2 = X.shape[0] ** 2
W0 = _sqdist(Z, None)
W = (np.exp(-W0 / ak / ak / 2) + np.exp(-W0 / ak / ak / 200) + np.exp(-W0 / ak / ak * 50)) /3

def log_post(x,y,theta, lambdaH):
    '''
    validation loss: log posterior distribution of f (theta)
    '''
    residual = y - x@theta.reshape((-1,1))
    rkr = (W * residual).T * residual # matrix with entries: residual * kernel * residual*lambda
    prior = sp_stats.norm.pdf(theta)
    logpost = np.log(prior)+np.sum(rkr.flatten() * lambdaH)-sp_misc.logsumexp(rkr.flatten() * lambdaH)*x.shape[0]**2
    #print(rkr.flatten() * lambdaH)
    # print(np.sum(rkr.flatten() * lambdaH),sp_misc.logsumexp(rkr.flatten() * lambdaH)*x.shape[0]**2)
    return logpost

def laplace_approx_log_post(x,y,theta,lambdaH):
    obj_grad = grad(lambda params: -log_post(x,y,params,lambdaH))
    la_mean = theta
    la_inv_cov = grad(obj_grad)(la_mean)
    return np.log(np.abs(la_inv_cov))/2-len(theta)/2*np.log(2*np.pi)

# def log_emp_lik(x,y,theta, lambdaH):
#     residual = y - x@theta
#     rkr = (W * residual).T * residual * lambdaH  # matrix with entries: residual * kernel * residual*lambda
#     return np.sum(rkr.flatten())-scipy.misc.logsumexp(rkr.flatten())*x.shape[0]**2

def lambdaH_risk(x,y, theta, lambdaH):
    '''
    training loss: argmin w.r.t lambdaH
    '''
    residual = y - x@theta.reshape((-1,1))
    rkr = (W*residual).T*residual
    loss = sp_misc.logsumexp(rkr.flatten()*lambdaH-np.log(N2))#np.mean(np.exp(rkr.flatten()*lambdaH))#
    return loss

def log_marg_lik(x,y,z, f, lambdaH):
    pass

def gradient_based_HO(x,y):
    theta, lambdaH = np.array([0.1]),np.array([-0.1])#np.random.randn(1)*10, np.random.randn(1)*10
    old_theta, old_lambdaH = None, None
    logpost = log_post(x, y, theta, lambdaH)
    old_logpost = None
    iteration = 0
    while (old_theta is None and old_lambdaH is None) or \
            (old_theta is not None and old_lambdaH is not None
        and iteration < 1000):
        if iteration % 5 == 0:
            old_logpost =  copy.copy(logpost)
        old_theta, old_lambdaH = copy.copy(theta), copy.copy(lambdaH)
        #        for i in range(N):
        d_LV_d_lambda = grad(lambda params: lambdaH_risk(x, y, theta, params))
        theta -= learning_rate_theta*hyper_gradient_theta(x, y, lambdaH, theta,d_LV_d_lambda)# d_LT_d_theta(theta)
        #print("## theta, lambdaH, logpost: ", theta, lambdaH, log_post(x, y, theta, lambdaH))
        d_LT_d_theta = grad(lambda params: -log_post(x, y, params, lambdaH))
        lambdaH -= learning_rate_lambda*hyper_gradient_lambda(x, y, lambdaH, old_theta,d_LT_d_theta)
        logpost = log_post(x, y, theta, lambdaH)
        print("## theta, lambdaH, logpost: ", theta, lambdaH, logpost)
        iteration += 1
        if iteration % 5 == 0:
            if logpost < old_logpost:
                break
    return lambdaH, theta

def hyper_gradient_lambda(x,y, lambdaH, theta, d_LT_d_theta):
    d_LV_d_theta = grad(lambda params: lambdaH_risk(x,y,params,lambdaH))
    v1 = d_LV_d_theta(theta)
    if len(theta)==1:
        v2 = v1/grad(d_LT_d_theta)(theta)
    elif len(theta)<200:
        v2 = v1 @ np.linalg.inv(grad(d_LT_d_theta)(theta))
    else:
        v2 = approx_inverse_HVP(v1,d_LT_d_theta,theta)
    d_LT_d_lambda = grad(lambda t,l: -log_post(x,y,t,l),1)
    v3 = v2@grad(lambda t:d_LT_d_lambda(t,lambdaH))(theta)
    if TST:
        numerical_v3 = (d_LT_d_lambda(theta+1e-8,lambdaH)-d_LT_d_lambda(theta,lambdaH))@v2/1e-8
        assert np.isclose(numerical_v3,v3), "numerical_v3, v3: {} {}".format(numerical_v3,v3)
    d_LV_d_lambda = grad(lambda params: lambdaH_risk(x,y,theta,params))
    print("hyper_gradient_lambda v1,v2,v3: ", v1,v2,v3)
    return d_LV_d_lambda(lambdaH)-v3

def hyper_gradient_theta(x,y, lambdaH, theta, d_LV_d_lambda):
    d_LT_d_lambda = grad(lambda params: lambdaH_risk(x,y,theta,params))
    v1 = d_LT_d_lambda(lambdaH)
    if len(lambdaH)==1:
        v2 = v1/grad(d_LV_d_lambda)(lambdaH)
    elif len(lambdaH)<200:
        v2 = v1 @ np.linalg.inv(grad(d_LV_d_lambda)(lambdaH))
    else:
        v2 = approx_inverse_HVP(v1,d_LV_d_lambda,lambdaH)
    d_LV_d_theta = grad(lambda t,l: -lambdaH_risk(x,y,t,l),0)
    v3 = v2@grad(lambda l:d_LV_d_theta(theta,l))(lambdaH)
    if TST:
        numerical_v3 = (d_LV_d_theta(theta,lambdaH+1e-8)-d_LV_d_theta(theta,lambdaH))@v2/1e-8
        assert np.isclose(numerical_v3,v3), "numerical_v3, v3: {} {}".format(numerical_v3,v3)
    d_LT_d_theta = grad(lambda params: -log_post(x,y,params,lambdaH))
    print("hyper_gradient_theta v1,v2,v3: ", v1,v2,v3)
    return d_LT_d_theta(theta)-v3

def approx_inverse_HVP(v,f,value):
    p = v
    hessian = grad(f)(value)
    for j in range(HVP_i):
        v -= hessian@v
        p += v
    return p

def tst_lambdaH_risk():
    theta, lambdaH = np.random.randn(1),np.random.randn(1)
    loss = lambdaH_risk(X,Y,theta,lambdaH)

    # re-implement
    residual = Y - X @ theta.reshape((-1, 1))
    ndata = Y.shape[0]
    tmp_loss = 0
    for i in range(ndata):
        for j in range(ndata):
            tmp_loss += np.exp(W[i,j]*residual[i]*residual[j]*lambdaH)
    tmp_loss = np.log(tmp_loss/N2)
    assert np.isclose(tmp_loss,loss), "tmp_loss, loss: {}, {} ".format(tmp_loss,loss)

    # test optimization
    theta = np.array([1.0])
    lambdaHs = []
    bounds = None
    obj_grad = value_and_grad(lambda params: lambdaH_risk(X, Y, theta, params))
    for i in range(30):
        params0 = np.random.randn(1)*10
        res = minimize(obj_grad, x0=params0, bounds=bounds, method='L-BFGS-B', jac=True,
                       options={'maxiter': 5000, 'disp': False})
        lambdaHs += [res.x]
    assert np.std(lambdaHs)**2<1e-6,('lambdaHs variance:',np.std(lambdaHs)**2)
    print("almbdaHs: ", lambdaHs)


def tst_log_post():
    '''
    validation loss: log posterior distribution of f (theta)
    '''
    theta, lambdaH = np.random.randn(1)+1, np.random.randn(1)
    logpost = log_post(X, Y, theta, lambdaH)

    # re-implement
    residual = Y - X @ theta.reshape((-1, 1))
    prior = sp_stats.norm.pdf(theta)
    sum_term = 0  # term 2
    logsumexp_term = 0 # term 3
    ndata = Y.shape[0]
    for i in range(ndata):
        for j in range(ndata):
            entry = W[i,j]*residual[i]*residual[j]*lambdaH
            sum_term += entry
            logsumexp_term += np.exp(entry)
    logsumexp_term = np.log(logsumexp_term)
    tmp_logpost = sum_term-logsumexp_term*ndata**2
    assert np.isclose(tmp_logpost,logpost), "tmp_logpost, logpost: {}, {}".format(tmp_logpost,logpost)

    # test optimization
    lambdaH = np.array([-0.15])
    thetas = []
    bounds = None
    obj_grad = value_and_grad(lambda params: -log_post(X,Y,params,lambdaH))
    for i in range(30):
        params0 = np.random.randn(1) * 10
        res = minimize(obj_grad, x0=params0, bounds=bounds, method='L-BFGS-B', jac=True,
                       options={'maxiter': 5000, 'disp': False})
        thetas += [res.x]
    print("thetas:",thetas)
    assert np.std(thetas) ** 2 < 1e-6, ('lambdaHs variance:', np.std(thetas) ** 2)

def tst_original_method():
    def original_loss(theta):
        residual = Y - X @ theta.reshape((-1, 1))
        loss = residual.T@W@residual
        return loss

    obj_grad = value_and_grad(original_loss)
    params0 = np.random.randn(1)*10
    res = minimize(obj_grad, x0=params0, bounds=None, method='L-BFGS-B', jac=True,
                   options={'maxiter': 5000, 'disp': False})

    print("original method:",res)



TST = False

if __name__ == '__main__':
    if TST:
        tst_lambdaH_risk()
        tst_log_post()
        tst_original_method()
    lambdaH, theta = gradient_based_HO(X,Y)
    # print("lambda, theta: ",lambdaH,theta)

