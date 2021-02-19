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

p_cov = np.array([[1]])
p_mean = np.array([[1]])
samples0 = np.random.randn(1000,1)

def elbo(x,y,samples0,c_lambda,params):
    q_mean, q_cov = params[:2]
    aks = np.exp(params[2:])
    W = np.exp(-W00 / aks[0] ** 2 / 2 - W01 / aks[1] ** 2 / 2)
    q_cov = np.exp(q_cov).reshape((-1,1))
    q_mean = q_mean.reshape((-1,1))
    kl = (np.log(p_cov) - np.log(q_cov)-p_cov.shape[0]+(p_mean-q_mean).T/p_cov@(p_mean-q_mean)+np.trace(1/p_cov@q_cov))/2
    samples = samples0*np.sqrt(q_cov)+q_mean
    c_lambda = c_lambda.reshape((samples0.shape[0],ndata))
    lambda_r = W @ c_lambda.T * (y - x @ samples.T)
    # ep_q = np.sum(lambdaH*(y-x@p_mean),axis=0)-ndata*np.mean([sp_misc.logsumexp(lambdaH*(y-x@samples[i])) for i in range(samples.shape[0])])
    ep_q1 = np.sum(np.mean(lambda_r ,axis=1,keepdims=True), axis=0)
    ep_q2 = ndata*np.mean(sp_misc.logsumexp(lambda_r,axis=0))
    # print('kl,ep_q1,ep_q2:',kl,ep_q1,ep_q2)
    return ep_q1-ep_q2-kl

def lambdaH_risk(x, y, samples0, c_lambda, params):
    '''
    training loss: argmin w.r.t c_lambda
    '''
    q_mean, q_cov = params[:2]
    aks = np.exp(params[2:])
    W = np.exp(-W00 / aks[0] ** 2 / 2 - W01 / aks[1] ** 2 / 2)
    q_cov = np.exp(q_cov).reshape((-1, 1))
    q_mean = q_mean.reshape((-1, 1))
    samples = samples0* np.sqrt(q_cov) + q_mean
    residual = y - x@samples.T
    c_lambda = c_lambda.reshape((samples0.shape[0], ndata))
    lambda_r = W @ c_lambda.T*residual
    loss = np.mean(np.exp(sp_misc.logsumexp(lambda_r.flatten(),axis=0)))#np.mean(np.exp(rkr.flatten()*c_lambda))#
    return loss

# def gradient_based_HO_IFT(x,y):
#     opt_params, c_lambda = np.random.randn((1,4)),np.random.randn(x.shape[0])
#     old_theta, old_c_lambda = None, None
#     iteration = 0
#     while (old_theta is None and old_c_lambda is None) or \
#             (old_theta is not None and old_c_lambda is not None
#         and iteration < 2):
#         if iteration % 3 == 0:
#             old_theta, old_c_lambda = copy.copy(theta), copy.copy(c_lambda)
#
#         obj_grad = value_and_grad(lambda params: -elbo(X, Y, c_lambda, params))
#         res = minimize(obj_grad, x0=opt_params, bounds=None, method='L-BFGS-B', jac=True,
#                        options={'maxiter': 5000, 'disp': False})
#
#         d_LT_d_theta = grad(lambda params: -elbo(x, y, c_lambda, params))
#         res = minimize(lambda params: -elbo(x, y, params, c_lambda, W),x0=theta,bounds=None, method='L-BFGS-B',
#              options={'maxiter': 10, 'disp': False},jac=d_LT_d_theta)
#         theta = res.x
#         res= minimize(lambda params: lambdaH_risk(x, y, theta, params, W),x0=c_lambda,bounds=None, method='L-BFGS-B',
#              options={'maxiter': 1, 'disp': False},jac=lambda params: hyper_gradient_lambda(x, y, theta, params,grad(lambda p: -log_post(x, y, p, params, W)),W))
#         c_lambda = res.x
#         logpost = elbo(x, y, theta, c_lambda, W)
#         lambda_risk = lambdaH_risk(x, y, theta, c_lambda, W)
#         print("## theta, c_lambda mean, logpost, lambda_risk: ", theta, np.mean(c_lambda), logpost, lambda_risk)
#         iteration += 1
#         if iteration % 3 == 0:
#             if np.sqrt(np.sum((theta - old_theta)**2)/np.sum(old_theta**2))<1e-4 and \
#             np.sqrt(np.sum((c_lambda - old_c_lambda)**2)/np.sum(old_c_lambda**2))<1e-4:
#                 break
#     return c_lambda, theta


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
    d_LT_d_lambda = grad(lambda t,l: -elbo(x,y,t,l,W),1)
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


# def tst_lambdaH_risk():
#     W = np.exp(-W00 / ak0 / ak0 / 2 - W01 / ak1 / ak1 / 2)
#     theta, c_lambda = np.random.randn(1),np.random.randn(X.shape[0])
#     loss = lambdaH_risk(X,Y,theta,c_lambda,W)
#
#     # re-implement
#     residual = Y - X @ theta.reshape((-1, 1))
#     tmp_loss = 0
#     for i in range(ndata):
#         tmp_loss += np.exp(W[i]@c_lambda*residual[i])
#     tmp_loss = np.log(tmp_loss/ndata)
#     assert np.isclose(tmp_loss,loss), "tmp_loss, loss: {}, {} ".format(tmp_loss,loss)
#
#     # test optimization
#     bounds = None
#     for i in range(5):
#         obj_grad = value_and_grad(lambda params: lambdaH_risk(X, Y, theta, params,W))
#         theta = np.random.randn(1)
#         params0 = np.random.randn(ndata)
#         res = minimize(obj_grad, x0=params0, bounds=bounds, method='L-BFGS-B', jac=True,
#                        options={'maxiter': 10000, 'disp': False})
#         c_lambda = res.x
#         lr = W@c_lambda.reshape((-1,1))*residual
#         pos = sp_misc.logsumexp(lr[residual > 0] + np.log((np.diag(W).reshape((-1, 1)) * residual)[residual > 0])) \
#               - sp_misc.logsumexp(lr)
#         neg = sp_misc.logsumexp(lr[residual < 0] + np.log(np.abs((np.diag(W).reshape((-1, 1)) * residual)[residual < 0]))) - sp_misc.logsumexp(lr)
#         print('np.exp(pos)-np.exp(neg)',np.exp(pos),np.exp(neg))
#         log_prior, log_el = log_post(X,Y,theta, c_lambda,W)
#         print('theta, log_prior, log_el =', theta, log_prior, log_el)

def tst_elbo():
    '''
    validation loss: log posterior distribution of f (theta)
    '''
    q_mean, q_cov = p_mean[0]+0.5, p_cov[0]+2
    c_lambda = np.random.randn(ndata*samples0.shape[0])*0.1
    W = np.exp(-W00 / ak0 / ak0 / 2 - W01 / ak1 / ak1 / 2)
    elbo_val = elbo(X, Y, samples0, c_lambda,np.hstack((q_mean,np.log(q_cov),np.log(ak0),np.log(ak1))))

    # test kl
    samples = np.random.randn(100000)*np.sqrt(q_cov)+q_mean
    kl = np.mean(np.log(sp_stats.norm.pdf(samples,loc=q_mean,scale = np.sqrt(q_cov))/sp_stats.norm.pdf(samples,loc=p_mean,scale = np.sqrt(p_cov))))
    print('kl',kl)

    # optimization
    model_params = np.random.randn(4)*0.1+0.4

    for ite in range(100):
        obj_grad = value_and_grad(lambda params: lambdaH_risk(X, Y, samples0, params, model_params))
        res = minimize(obj_grad, x0=c_lambda, bounds=None, method='L-BFGS-B', jac=True,
                       options={'maxiter': 2, 'disp': False})
        c_lambda = res.x
        print('model_params:',model_params,'c_lambda mean: ',np.mean(c_lambda))

        obj_grad = value_and_grad(lambda params: -elbo(X, Y, samples0, c_lambda,params))
        bounds = None # np.array([[-1,2],[-3,3],[-6,3],[-6,3]])
        res = minimize(obj_grad, x0=model_params, bounds=bounds, method='L-BFGS-B', jac=True,options={'maxiter': 10, 'disp': False})
        model_params = res.x









# def tst_original_method():
#     W = np.exp(-W00 / ak0 / ak0 / 2 - W01 / ak1 / ak1 / 2)
#     def original_loss(theta):
#         residual = Y - X @ theta.reshape((-1, 1))
#         loss = residual.T@W@residual
#         return loss
#
#     obj_grad = value_and_grad(original_loss)
#     params0 = np.random.randn(1)*10
#     res = minimize(obj_grad, x0=params0, bounds=None, method='L-BFGS-B', jac=True,
#                    options={'maxiter': 5000, 'disp': False})
#
#     print("original method:",res)


TST = True

if __name__ == '__main__':
    if TST:
        pass
        tst_elbo()

    # aks = np.log(np.abs(np.hstack((ak0,ak1))))
    # c_lambda, theta = gradient_based_HO_IFT(X, Y, aks)
    # lml = log_marg_lik(X, Y, theta, c_lambda, aks)
    # print('aks: ',aks)
    # for ite in range(100):
    #     c_lambda, theta = gradient_based_HO_IFT(X,Y,aks)
    #     obj_val = value_and_grad(lambda params: log_marg_lik(X, Y, theta, c_lambda, params))
    #     res = minimize(obj_val, x0=aks, bounds=None, method='L-BFGS-B',
    #              options={'maxiter': 1, 'disp': False},jac=True)
    #     aks = res.x
    #     print('aks: ',aks, 'log marg lik:', -res.fun)
    #print(lml)

