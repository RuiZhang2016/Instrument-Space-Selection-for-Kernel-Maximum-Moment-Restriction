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
# ak = get_median_inter_mnist(Z)*1e6
ndata = X.shape[0]
# W0 = _sqdist(Z, None)
# W = (np.exp(-W0 / ak / ak / 2) + np.exp(-W0 / ak / ak / 200) + np.exp(-W0 / ak / ak * 50)) /3

ak0 = get_median_inter_mnist(Z[:,[0]])
ak1 = get_median_inter_mnist(Z[:,[1]])
W00 = _sqdist(Z[:,[0]], None)
W01 = _sqdist(Z[:,[1]], None)
# W = np.exp(-W00 / ak0 / ak0 / 2-W01/ak1/ak1/2)
# W = np.exp(-W01 / ak1 / ak1 / 2)

def log_post(x,y,theta, c_lambda, W):
    '''
    validation loss: log posterior distribution of f (theta)
    '''
    residual = y - x@theta.reshape((-1,1))
    lambda_r = W@c_lambda.reshape((-1,1))*residual
    prior = sp_stats.norm.pdf(theta,loc=0,scale=0.5)
    return np.log(prior)+np.sum(lambda_r,axis=0)-sp_misc.logsumexp(lambda_r)*x.shape[0]

def laplace_approx_log_post(x,y,la_mean,c_lambda, W):
    obj_grad = grad(lambda params: -log_post(x,y,params,c_lambda,W))
    la_inv_cov = grad(obj_grad)(la_mean)
    print('la_inv_cov:',la_inv_cov,np.log(np.abs(la_inv_cov))/2-len(la_mean)/2*np.log(2*np.pi))
    return np.log(np.abs(la_inv_cov))/2-len(la_mean)/2*np.log(2*np.pi)

def log_marg_lik(x,y,theta,c_lambda,aks):
    aks = np.exp(aks)
    W = np.exp(-W00 / aks[0]**2 / 2 - W01 / aks[1]**2 / 2)
    lml = log_post(x, y, theta, c_lambda, W)-laplace_approx_log_post(x,y,theta,c_lambda,W)
    return lml

def lambdaH_risk(x, y, theta, c_lambda, W):
    '''
    training loss: argmin w.r.t c_lambda
    '''
    residual = y - x@theta.reshape((-1,1))
    lambda_r = W@c_lambda.reshape((-1,1))*residual
    loss = sp_misc.logsumexp(lambda_r.flatten()-np.log(x.shape[0]))#np.mean(np.exp(rkr.flatten()*c_lambda))#
    return loss

def gradient_based_HO_IFT(x,y,aks):
    aks = np.exp(aks)
    W = np.exp(-W00 / aks[0]**2 / 2 - W01 / aks[1]**2 / 2)
    theta, c_lambda = np.array([1]),np.random.randn(x.shape[0])
    old_theta, old_c_lambda = None, None
    iteration = 0
    while (old_theta is None and old_c_lambda is None) or \
            (old_theta is not None and old_c_lambda is not None
        and iteration < 2):
        if iteration % 3 == 0:
            old_theta, old_c_lambda = copy.copy(theta), copy.copy(c_lambda)
        d_LT_d_theta = grad(lambda params: -log_post(x, y, params, c_lambda, W))
        res = minimize(lambda params: -log_post(x, y, params, c_lambda, W),x0=theta,bounds=None, method='L-BFGS-B',
             options={'maxiter': 10, 'disp': False},jac=d_LT_d_theta)
        theta = res.x
        res= minimize(lambda params: lambdaH_risk(x, y, theta, params, W),x0=c_lambda,bounds=None, method='L-BFGS-B',
             options={'maxiter': 1, 'disp': False},jac=lambda params: hyper_gradient_lambda(x, y, theta, params,grad(lambda p: -log_post(x, y, p, params, W)),W))
        c_lambda = res.x
        logpost = log_post(x, y, theta, c_lambda, W)
        lambda_risk = lambdaH_risk(x, y, theta, c_lambda, W)
        print("## theta, c_lambda mean, logpost, lambda_risk: ", theta, np.mean(c_lambda), logpost, lambda_risk)
        iteration += 1
        if iteration % 3 == 0:
            if np.sqrt(np.sum((theta - old_theta)**2)/np.sum(old_theta**2))<1e-4 and \
            np.sqrt(np.sum((c_lambda - old_c_lambda)**2)/np.sum(old_c_lambda**2))<1e-4:
                break
    return c_lambda, theta


def hyper_gradient_lambda(x,y,theta,c_lambda, d_LT_d_theta, W):
    d_LV_d_theta = grad(lambda params: lambdaH_risk(x,y,params,c_lambda, W))
    v1 = d_LV_d_theta(theta)
    if len(theta)==1:
        if grad(d_LT_d_theta)(theta) == np.array([0.]):
            v2 = v1
        else:
            v2 = v1 / grad(d_LT_d_theta)(theta)
    elif len(theta)<500:
        v2 = v1 @ np.linalg.inv(grad(d_LT_d_theta)(theta))
    else:
        v2 = approx_inverse_HVP(v1,d_LT_d_theta,theta)
    d_LT_d_lambda = grad(lambda t,l: -log_post(x,y,t,l,W),1)
    dd_LT_d_tl = np.array([grad(lambda t: d_LT_d_lambda(t,c_lambda)[i])(theta) for i in range(len(c_lambda))])
    v3 = dd_LT_d_tl@v2
    if TST:
        numerical_v3 = (d_LT_d_lambda(theta+1e-8,c_lambda)-d_LT_d_lambda(theta,c_lambda)).reshape((len(c_lambda),len(theta)))@v2/1e-8
        dd_LT_d_tl_sum = grad(lambda t: d_LT_d_lambda(t, c_lambda))(theta)
        assert np.isclose(dd_LT_d_tl_sum@v2,np.sum(numerical_v3)), 'dd_LT_d_tl_sum@v2,np.sum(numerical_v3): {} {}'.format(dd_LT_d_tl_sum@v2,np.sum(numerical_v3))
        assert np.all(np.isclose(numerical_v3,v3)), "numerical_v3, v3: {} {}".format(numerical_v3,v3)
    d_LV_d_lambda = grad(lambda params: lambdaH_risk(x,y,theta,params,W))
    # print("hyper_gradient_lambda v1,v2,v3: ", v1,v2,v3)
    return d_LV_d_lambda(c_lambda)-v3


def approx_inverse_HVP(v,f,value):
    p = v
    hessian = grad(f)(value)
    for j in range(HVP_i):
        v -= hessian@v
        p += v
    return p


def tst_lambdaH_risk():
    W = np.exp(-W00 / ak0 / ak0 / 2 - W01 / ak1 / ak1 / 2)
    theta, c_lambda = np.random.randn(1),np.random.randn(X.shape[0])
    loss = lambdaH_risk(X,Y,theta,c_lambda,W)

    # re-implement
    residual = Y - X @ theta.reshape((-1, 1))
    tmp_loss = 0
    for i in range(ndata):
        tmp_loss += np.exp(W[i]@c_lambda*residual[i])
    tmp_loss = np.log(tmp_loss/ndata)
    assert np.isclose(tmp_loss,loss), "tmp_loss, loss: {}, {} ".format(tmp_loss,loss)

    # test optimization
    bounds = None
    for i in range(5):
        obj_grad = value_and_grad(lambda params: lambdaH_risk(X, Y, theta, params,W))
        theta = np.random.randn(1)
        params0 = np.random.randn(ndata)
        res = minimize(obj_grad, x0=params0, bounds=bounds, method='L-BFGS-B', jac=True,
                       options={'maxiter': 10000, 'disp': False})
        c_lambda = res.x
        lr = W@c_lambda.reshape((-1,1))*residual
        pos = sp_misc.logsumexp(lr[residual > 0] + np.log((np.diag(W).reshape((-1, 1)) * residual)[residual > 0])) \
              - sp_misc.logsumexp(lr)
        neg = sp_misc.logsumexp(lr[residual < 0] + np.log(np.abs((np.diag(W).reshape((-1, 1)) * residual)[residual < 0]))) - sp_misc.logsumexp(lr)
        print('np.exp(pos)-np.exp(neg)',np.exp(pos),np.exp(neg))
        log_prior, log_el = log_post(X,Y,theta, c_lambda,W)
        print('theta, log_prior, log_el =', theta, log_prior, log_el)

def tst_log_post():
    '''
    validation loss: log posterior distribution of f (theta)
    '''
    theta, c_lambda = np.random.randn(1)+1, np.random.randn(X.shape[0])
    W = np.exp(-W00 / ak0 / ak0 / 2 - W01 / ak1 / ak1 / 2)
    logpost = log_post(X, Y, theta, c_lambda,W)


    # re-implement
    residual = Y - X @ theta.reshape((-1, 1))
    prior = sp_stats.norm.pdf(theta,loc=0,scale=0.5)
    sum_term = 0  # term 2
    logsumexp_term = 0 # term 3
    ndata = Y.shape[0]
    for i in range(ndata):
        entry = W[i]@c_lambda.reshape((-1,1))*residual[i]
        sum_term += entry
        logsumexp_term += np.exp(entry)
    logsumexp_term = np.log(logsumexp_term)
    tmp_logpost = sum_term-logsumexp_term*ndata+np.log(prior)
    assert np.isclose(tmp_logpost,np.sum(logpost)), "tmp_logpost, logpost: {}, {}".format(tmp_logpost,logpost)

    # test optimization
    thetas = []
    bounds = None
    obj_grad = value_and_grad(lambda params: -log_post(X,Y,params,c_lambda,W))
    for i in range(30):
        params0 = np.random.randn(1)
        res = minimize(obj_grad, x0=params0, bounds=bounds, method='L-BFGS-B', jac=True,
                       options={'maxiter': 10000, 'disp': False})
        thetas += [res.x]
        print(res)
    # print("thetas:",thetas)
    assert np.std(thetas) ** 2 < 1e-6, ('thetas variance:', np.std(thetas) **2, 'thetas:',thetas)

def tst_original_method():
    W = np.exp(-W00 / ak0 / ak0 / 2 - W01 / ak1 / ak1 / 2)
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
        pass
        tst_lambdaH_risk()
        tst_log_post()
        tst_original_method()

    aks = np.log(np.abs(np.hstack((ak0,ak1))))
    c_lambda, theta = gradient_based_HO_IFT(X, Y, aks)
    lml = log_marg_lik(X, Y, theta, c_lambda, aks)
    # print('aks: ',aks)
    # for ite in range(100):
    #     c_lambda, theta = gradient_based_HO_IFT(X,Y,aks)
    #     obj_val = value_and_grad(lambda params: log_marg_lik(X, Y, theta, c_lambda, params))
    #     res = minimize(obj_val, x0=aks, bounds=None, method='L-BFGS-B',
    #              options={'maxiter': 1, 'disp': False},jac=True)
    #     aks = res.x
    #     print('aks: ',aks, 'log marg lik:', -res.fun)
    #print(lml)

