import numpy as np
import matplotlib.pyplot as plt
from functools import partial

def unif_f(x,l,r):
    """
    constant function
    """
    return 1 if (l <= x <= r) else 0

def p_h_f(theta_h,theta_f,h,f,x,y,z,_lambda=1e-3):
    """
    p(h|f)
    """
    ept = (y-f(x,theta_f)).T@h(z,theta_h)/x.shape[0]
    return (ept**2+theta_h.T@theta_h*_lambda)[0,0]

def p_h_f_ratio(theta_h1,theta_h2,theta_f,h,f,x,y,z,_lambda=1e-3):
    """
    p(h|f)
    """
    ept1 = (y-f(x,theta_f)).T@h(z,theta_h1)/x.shape[0]
    ept2 = (y - f(x, theta_f)).T @ h(z, theta_h2) / x.shape[0]
    return np.exp(ept1**2+theta_h1.T@theta_h1*_lambda-ept2**2-theta_h2.T@theta_h2*_lambda)[0,0]

def p_f_h(theta_h,theta_f,h,f,x,y,z,_lambda=1e-3):
    """
    p(f|h)
    """
    ept = (y-f(x,theta_f)).T@h(z,theta_h)/x.shape[0]
    return np.exp(-ept**2+theta_f.T@theta_f*_lambda)[0,0]

def p_f_h_ratio(theta_h,theta_f1,theta_f2,h,f,x,y,z,_lambda=1e-3):
    """
    p(f|h)
    """
    ept1 = (y-f(x,theta_f1)).T@h(z,theta_h)/x.shape[0]
    ept2 = (y - f(x, theta_f2)).T @ h(z, theta_h) / x.shape[0]
    return np.exp(-ept1**2+theta_f1.T@theta_f1*_lambda+ept2**2-theta_f2.T@theta_f2*_lambda)[0,0]

def gaussian(x, Sigma, sampled=None):
    """
    multivariate Gaussian
    """
    if Sigma.shape[0] > 1:
        if sampled is None: # sampling
            L = np.linalg.cholesky(Sigma)
            z = np.random.randn(x.shape[0], 1)
            return np.dot(L, z+x)
        else: # un-normalized conditional probability
            return np.exp(-0.5*np.dot( (x-sampled).T, np.dot(np.linalg.inv(Sigma), (x-sampled))))[0,0]
    else:
        if sampled is None:
            return (np.sqrt(Sigma[0,0])) * np.random.randn(1,1)
        else:
            return np.exp(-0.5*(x - sampled) ** 2 / Sigma[0,0])

def mh_one_step(ptilde, proposal, old):
    """
    one step in the metropolis_hastings algorithm.
    """
    new = proposal(old)
    #print("new,old, ptilde(new),proposal(new, sampled = old),ptilde(old),proposal(old, sampled = new):")
    #print(new,old, ptilde(new),proposal(new, sampled = old),ptilde(old),proposal(old, sampled = new))
    print(np.hstack([new,old]))
    print(ptilde(new,old), proposal(new, sampled=old), proposal(old, sampled=new))

    # alpha = np.min([(ptilde(new)*proposal(new, sampled = old))/(ptilde(old) * proposal(old, sampled = new)), 1])
    alpha = np.min([ptilde(new,old)*proposal(new, sampled = old)/proposal(old, sampled = new),1])
    print(alpha)
    u = np.random.uniform()
    return (new, 1) if u < alpha else (old, 0)

def mcmc(chain, ptilde, proposal, init, num):
    """
    the metropolis_hastings algorithm
    """
    count = 0
    #samples = []
    for i in range(num):
        init, j = chain(ptilde, proposal, init)
        count = count + j
    return init, count

def f(x,theta):
    return abs(x)@theta

def h(z,theta):
    return np.hstack((z,z**2))@theta

def data1():
    z = np.random.uniform(-2,2,(1000,1))
    x = 0.5*z
    y = 2*x
    g = 2*x
    return z,x,y,g

def data2():
    f_star = lambda x: np.abs(x)
    delta = np.random.normal(0,0.1,(1000,1))
    gamma = np.random.normal(0,0.1,(1000,1))
    z = np.random.uniform(-3,3,(1000,2))
    e = np.random.normal(0,1,(1000,1))
    x = z[:,[0]]+e+gamma
    g = f_star(x)
    y = f_star(x)+e+delta
    return z,x,y,g

if __name__ == '__main__':
    z,x,y,g = data2()
    theta_h, theta_f = np.array([[-1],[1],[-1],[1]]), np.array([[-0.5]])
    samples_f = [theta_f[0,0]]
    for ite in range(501):
        print('sample h from phf')
        #ptilde = lambda theta_h: p_h_f(theta_h=theta_h,theta_f=theta_f,h=h, f=f, x=x, y=y, z=z)
        ptilde_ratio = lambda theta_h1, theta_h2: p_h_f_ratio(theta_h1=theta_h1, theta_h2=theta_h2,theta_f=theta_f,h=h, f=f, x=x, y=y, z=z)
        prop = partial(gaussian,Sigma=np.eye(4))
        theta_h,_ = mcmc(mh_one_step, ptilde_ratio, prop, theta_h, 1)
        print('sample f from pfh')
        # ptilde = lambda theta_f: p_f_h(theta_f=theta_f,theta_h=theta_h, h=h, f=f, x=x, y=y, z=z)
        ptilde_ratio = lambda theta_f1, theta_f2: p_f_h_ratio(theta_f1=theta_f1, theta_f2=theta_f2, theta_h=theta_h, h=h, f=f,x=x, y=y, z=z)
        prop = partial(gaussian, Sigma=np.array([[4]]))
        theta_f,_ = mcmc(mh_one_step, ptilde_ratio, prop, theta_f, 1)
        samples_f += [theta_f[0,0]]

        if ite%10 == 0 and ite > 0:
            plt.hist(samples_f)
            plt.show()

