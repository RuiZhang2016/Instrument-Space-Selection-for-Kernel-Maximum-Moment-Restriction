from util import Kernel
import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize

kernel = Kernel('rbf')
disp = False

np.random.seed(627)

def core_quantities(X,Y,z,Z,theta_f, theta_k,name=None):
    theta_k = np.exp(theta_k)
    fX =X@(theta_f[:,None])
    E_phi_k = np.mean((Y-fX)*kernel(Z,z,theta_k,1),axis=0)[:,None]
    Kz= kernel(z,None,theta_k,1)+1e-6*np.eye(z.shape[0])
    Kz_inv=np.linalg.inv(Kz)
    mean = E_phi_k.T @ Kz_inv @ E_phi_k
    s, logdet = np.linalg.slogdet(Kz)
    log_mar_like = -mean - 0.5 * logdet
    variance = (Y - fX).T @ kernel(Z, None, theta_k, 1) @ (Y - fX) - mean
    if name == 'mean':
        return mean[0,0]
    if name == 'log_mar_like':
        return log_mar_like[0,0]
    if name == 'variance':
        return variance[0,0]

    return log_mar_like[0,0], mean[0,0], variance[0,0]


def experiments():
    # generate data
    Z = np.random.uniform(-2,2,size=(1000,1))
    e = np.random.normal(0,1,size=(1000,1))
    X = Z+e
    Y = X+e
    z = np.array([Z[i] for i in np.random.choice(Z.shape[0],50,replace=False)])

    theta_f = np.array([0.01]*X.shape[1])
    theta_k = np.array([0.01]*Z.shape[1])
    log_mar_like, mean, variance = core_quantities(X,Y,z,Z,theta_f, theta_k,None)
    print(log_mar_like, mean, variance)

    # maximize log marginal likelihood
    for ite in range(10):
        print(ite, " maximize log marginal likelihood")
        obj_grad = value_and_grad(lambda params: -core_quantities(X,Y,z,Z,theta_f,params,'log_mar_like'))
        res = minimize(obj_grad, x0=theta_k,bounds=None, method='L-BFGS-B',jac=True,options={'maxiter':5000,'disp':disp})
        theta_k = res.x
        print(res)
        print('log_mar, mean, variance:', core_quantities(X,Y,z,Z,theta_f,theta_k))

        print(ite, " minimize variance")
        obj_grad = value_and_grad(lambda params: core_quantities(X, Y, z, Z, params, theta_k, 'variance'))
        res = minimize(obj_grad, x0=theta_f, bounds=None, method='L-BFGS-B', jac=True,
                       options={'maxiter': 5000, 'disp': disp})
        theta_f = res.x
        print(res)
        print('log_mar, mean, variance:', core_quantities(X, Y, z, Z, theta_f, theta_k))

        # print(ite, " minimize mean")
        # obj_grad = value_and_grad(lambda params: core_quantities(X, Y, z, Z, params, theta_k, 'mean'))
        # res = minimize(obj_grad, x0=theta_f, bounds=None, method='L-BFGS-B', jac=True,
        #                options={'maxiter': 5000, 'disp': disp})
        # theta_f = res.x
        # print(res)
        # print('log_mar, mean, variance:', core_quantities(X, Y, z, Z, theta_f, theta_k))

if __name__ == '__main__':
    experiments()