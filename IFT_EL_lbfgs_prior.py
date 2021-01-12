import autograd.numpy as np
import autograd.scipy.misc as sp_misc
import autograd.scipy.stats as sp_stats
import copy

from autograd import value_and_grad, grad, jacobian
from scipy.optimize import minimize
from util_original import get_median_inter_mnist, Kernel, load_data, ROOT_PATH, jitchol, _sqdist
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
ak0 = get_median_inter_mnist(Z[:,[0]])
ak1 = get_median_inter_mnist(Z[:,[1]])
N2 = X.shape[0] ** 2
W00 = _sqdist(Z[:,[0]], None)
W01 = _sqdist(Z[:,[1]], None)
W = np.exp(-W00 / ak0 / ak0 / 2)


def log_post(x,y,theta, lambdaH):
    '''
    validation loss: log posterior distribution of f (theta)
    '''
    residual = y - x@theta.reshape((-1,1))
    rkr = (W * residual).T * residual # matrix with entries: residual * kernel * residual*lambda
    prior = sp_stats.norm.pdf(theta,loc=prior_mean,scale=prior_std)
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

def log_marg_lik(x,y,theta,lambdaH):
    lml = log_post(x, y, theta, lambdaH)-laplace_approx_log_post(x,y,theta,lambdaH)
    return lml

def gradient_based_HO_IFT(x,y):
    theta, lambdaH = np.array([1]),np.array([-0.5])#np.random.randn(1)*10, np.random.randn(1)*10
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
        # d_LV_d_lambda = grad(lambda params: lambdaH_risk(x, y, theta, params))
        d_LT_d_theta = grad(lambda params: -log_post(x, y, params, lambdaH))
        res = minimize(lambda params: -log_post(x, y, params, lambdaH),x0=theta,bounds=None, method='L-BFGS-B',
             options={'maxiter': 2, 'disp': False},jac=d_LT_d_theta)
                       # jac=lambda params: hyper_gradient_theta(x, y, params, lambdaH,grad(lambda p: lambdaH_risk(x, y, params, p))))
        theta = res.x
        print("---------",res)
        # d_LT_d_theta = grad(lambda params: -log_post(x, y, params, lambdaH))
        # d_LV_d_lambda = grad(lambda params: lambdaH_risk(x, y, theta, params))
        res= minimize(lambda params: lambdaH_risk(x, y, theta, params),x0=lambdaH,bounds=None, method='L-BFGS-B',
             options={'maxiter': 2, 'disp': False},jac=lambda params: hyper_gradient_lambda(x, y, theta, params,grad(lambda p: -log_post(x, y, p, params))))
        print(res)
        lambdaH = res.x
        logpost = log_post(x, y, theta, lambdaH)
        print("## theta, lambdaH, logpost: ", theta, lambdaH, logpost)
        iteration += 1
        if iteration % 5 == 0:
            if logpost <= old_logpost:
                break
    return lambdaH, theta

def gradient_based_HO_gd(x,y):
    theta, lambdaH = np.array([1]),np.array([0.5])#np.random.randn(1)*10, np.random.randn(1)*10
    old_theta, old_lambdaH = None, None
    logpost = log_post(x, y, theta, lambdaH)
    old_logpost = None
    iteration = 0
    while (old_theta is None and old_lambdaH is None) or \
            (old_theta is not None and old_lambdaH is not None
        and iteration < 1000):
        if iteration % 5 == 0:
            old_logpost = copy.copy(logpost)
        old_theta, old_lambdaH = copy.copy(theta), copy.copy(lambdaH)
        d_LT_d_theta = grad(lambda p: -log_post(x, y, p, lambdaH))
        res = minimize(lambda params: -log_post(x, y, params, lambdaH),x0=theta,bounds=None, method='L-BFGS-B',
             options={'maxiter': 2, 'disp': False},
                       jac=d_LT_d_theta)
        theta = res.x
        print("---------",res)

        d_LV_d_lambda = grad(lambda params: lambdaH_risk(x, y, theta, params))
        #print('derivatives: ',d_LT_d_theta(theta),hyper_gradient_theta(x, y, theta, lambdaH,d_LV_d_lambda))
        res= minimize(lambda params: lambdaH_risk(x, y, theta, params),x0=lambdaH,bounds=None, method='L-BFGS-B',
             options={'maxiter': 2, 'disp': False},jac=d_LV_d_lambda)
        print(res)
        lambdaH = res.x
        logpost = log_post(x, y, theta, lambdaH)
        print("## theta, lambdaH, logpost: ", theta, lambdaH, logpost)
        #d_LT_d_theta = grad(lambda p: -log_post(x, y, p, lambdaH))
        # print(lambdaH,theta, d_LV_d_lambda(lambdaH),hyper_gradient_lambda(x, y, theta,lambdaH, d_LT_d_theta))
        iteration += 1
        if iteration % 5 == 0:
            if logpost <= old_logpost:
                break
    return theta, lambdaH



def hyper_gradient_lambda(x,y, theta, lambdaH, d_LT_d_theta):
    d_LV_d_theta = grad(lambda params: lambdaH_risk(x,y,params,lambdaH))
    v1 = d_LV_d_theta(theta)
    if len(theta)==1:
        if grad(d_LT_d_theta)(theta) == np.array([0.]):
            v2 = v1
        else:
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
    # print("hyper_gradient_lambda v1,v2,v3: ", v1,v2,v3)
    # print('components:',lambdaH,theta, d_LV_d_lambda(lambdaH),v3)
    return d_LV_d_lambda(lambdaH)-v3

def hyper_gradient_theta(x,y, theta,lambdaH, d_LV_d_lambda):
    d_LT_d_lambda = grad(lambda params: -log_post(x,y,theta,params))
    v1 = d_LT_d_lambda(lambdaH)
    if len(lambdaH)==1:
        if grad(d_LV_d_lambda)(lambdaH) == np.array([0.]):
            v2 = v1
        else:
            v2 = v1 / grad(d_LV_d_lambda)(lambdaH)
    elif len(lambdaH)<200:
        v2 = v1 @ np.linalg.inv(grad(d_LV_d_lambda)(lambdaH))
    else:
        v2 = approx_inverse_HVP(v1,d_LV_d_lambda,lambdaH)
    d_LV_d_theta = grad(lambda t,l: lambdaH_risk(x,y,t,l),0)
    v3 = v2@grad(lambda l:d_LV_d_theta(theta,l))(lambdaH)
    if TST:
        numerical_v3 = v2@(d_LV_d_theta(theta,lambdaH+1e-8)-d_LV_d_theta(theta,lambdaH))/1e-8
        assert np.isclose(numerical_v3,v3), "numerical_v3, v3: {} {}".format(numerical_v3,v3)
    d_LT_d_theta = grad(lambda params: -log_post(x,y,params,lambdaH))
    # print("hyper_gradient_theta v1,v2,v3: ", v1,v2,v3)
    # print('components:',theta, lambdaH, d_LT_d_theta(theta),v3)
    return d_LT_d_theta(theta)-v3

def hyper_gradient(P, HP, LT, LV, d_LT_d_P):
    d_LV_d_P = grad(lambda p: LV(p,HP))
    v1 = d_LV_d_P(P)
    if len(P) == 1:
        if grad(d_LT_d_P)(P) == np.array([0.]):
            v2 = v1
        else:
            v2 = v1 / grad(d_LT_d_P)(P)
    elif len(P) < 200:
        v2 = v1 @ np.linalg.inv(grad(d_LT_d_P)(P))
    else:
        v2 = approx_inverse_HVP(v1, d_LT_d_P, P)
    d_LT_d_HP = grad(lambda p, hp: LT(p,hp), 1)
    v3 = v2 @ grad(lambda p: d_LT_d_HP(p, HP))(P)
    if TST:
        numerical_v3 = v2@(d_LT_d_HP(P + 1e-8, HP) - d_LT_d_HP(P, HP)) / 1e-8
        assert np.isclose(numerical_v3, v3), "numerical_v3, v3: {} {}".format(numerical_v3, v3)
    d_LV_d_HP = grad(lambda hp: LV(P,hp))
    # print("hyper_gradient_lambda v1,v2,v3: ", v1,v2,v3)
    return d_LV_d_HP(HP) - v3


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
    prior = sp_stats.norm.pdf(theta,loc=prior_mean,scale=prior_std)
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

def tst_original_method(x,y):
    def original_loss(theta):
        residual = y - x @ theta.reshape((-1, 1))
        loss = residual.T@W@residual
        return loss

    obj_grad = value_and_grad(original_loss)
    params0 = np.random.randn(1)*10
    res = minimize(obj_grad, x0=params0, bounds=None, method='L-BFGS-B', jac=True,
                   options={'maxiter': 5000, 'disp': False})

    print("original method:",res)
    theta = res.x
    re = y - x @ theta.reshape((-1, 1))
    return theta, np.std(re)

def tst_hyper_gradient():
    lambdaH, theta = np.random.randn(1),np.random.randn(1)

    print("test hyper_gradient_lambda")
    LT = lambda p,hp: -log_post(X, Y, p, hp)
    LV = lambda p,hp: lambdaH_risk(X, Y, p, hp)
    d_LT_d_P = grad(lambda p: LT(p,lambdaH))
    val_hg = hyper_gradient(theta, lambdaH, LT, LV, d_LT_d_P)
    val_hgl = hyper_gradient_lambda(X, Y, theta,lambdaH, d_LT_d_P)
    print('val_hg,val_hgl:',val_hg,val_hgl)

    print("test hyper_gradient_theta")
    LV = lambda p,hp: -log_post(X, Y, hp, p)
    LT = lambda p,hp: lambdaH_risk(X, Y, hp, p)
    d_LT_d_P = grad(lambda p: LT(theta,p))
    val_hg = hyper_gradient(lambdaH, theta, LT, LV, d_LT_d_P)
    val_hgt = hyper_gradient_theta(X, Y, theta,lambdaH, d_LT_d_P)
    print('val_hg,val_hgt:',val_hg,val_hgt)




TST = False
prior_mean, prior_std = 1,0.5
if __name__ == '__main__':
    if TST:
        tst_lambdaH_risk()
        tst_log_post()
        tst_original_method()
        tst_hyper_gradient()
    #prior_mean, prior_std = tst_original_method(X,Y)
    theta, lambdaH = gradient_based_HO_gd(X,Y)
    lml = log_marg_lik(X, Y, theta, lambdaH)

    print('lml:',lml)
    # print("lambda, theta: ",lambdaH,theta)

