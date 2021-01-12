# we use laplace approximation here
import autograd.numpy as np
from util import Kernel
from autograd import value_and_grad
from scipy.optimize import minimize
import matplotlib.pyplot as plt

kernel = Kernel('rbf')
Z = np.random.uniform(-2,2,size=(500,1))
e = np.random.normal(0,0.1,size=(500,1))
X = Z+e
Y = -X+e
z = np.array([Z[i] for i in np.random.choice(Z.shape[0],50,replace=False)])
theta_l = np.array([-0.1] * X.shape[1])
theta_k = np.array([-0.1] * Z.shape[1])
theta_l = np.exp(theta_l)
theta_k = np.exp(theta_k)
L = kernel(X,None,theta_l,1)
K = kernel(Z,None,theta_k,1)+1e-6*np.eye(Z.shape[0])
inv_K = np.linalg.inv(K)

def pf_h(h):
    cov = L - L @ h @h.T@L/(1+ h.T@L@h)
    mean = cov @ h @ h.T @ Y
    sample = np.random.multivariate_normal(mean.flatten(), cov=cov,size=(1))
    return sample.T

def ph_f(f):
    objective = lambda h: -np.log((h@ (f-Y))**2)+h@inv_K@h
    obj_grad = value_and_grad(objective)
    res = minimize(obj_grad, x0=np.random.uniform(0,1,size=(f.shape[0])), method='L-BFGS-B',jac=True,options={'maxiter':5000})
    m = res.x
    #S = K - K @ (f-Y) @ (f-Y).T @ K / ((m@(f-Y))**2/2+(f-Y).T@K@(f-Y))
    #sample = np.random.multivariate_normal(m, cov=S, size=(1))
    return m.reshape((-1,1)) #sample.T

class running_mean():
    def __init__(self):
        self.n = 0
        self.mean = 0

    def update(self, x):
        self.n += 1
        self.mean = self.mean*(self.n-1)/self.n+x/self.n

    def get_value(self):
        return self.mean


if __name__ == '__main__':
    sample_f = -X
    running_mean1 = running_mean()
    running_mean2 = running_mean()

    for ite in range(100):
        sample_h = ph_f(sample_f)
        sample_f = pf_h(sample_h)
        if ite > 50:
            running_mean1.update(sample_f)
            running_mean2.update(sample_f**2)



    mean = running_mean1.get_value()
    var = running_mean2.get_value() - mean**2
    plt.scatter(Z,mean)
    plt.scatter(Z,mean + var)
    plt.scatter(Z, mean-var)
    plt.show()






