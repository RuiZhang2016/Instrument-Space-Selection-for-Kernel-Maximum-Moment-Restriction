import autograd.numpy as np
import torch
from autograd import value_and_grad
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2
torch.manual_seed(527)
np.random.seed(527)
from util import get_median_inter_mnist,ROOT_PATH
import os,sys
import argparse
const = 1e-4
JITTER = 1e-7

fig_folder = ROOT_PATH+'/figs'

def effective_dim(Kz):

    return np.trace(Kz)/np.sqrt(np.trace(Kz@Kz))

class my_numpy_nn:
    def __init__(self, act,structure):
        if act == 'softrelu':
            self.act_fun = lambda x: np.log(1+np.exp(x))
        elif act == 'linear':
            self.act_fun = lambda x: x
        elif act == 'relu':
            self.act_fun = lambda x: np.max(x,0)
        elif act == 'sigmoid':
            self.act_fun = lambda x: 1/(1+np.exp(-x))
        else:
            raise NotImplementedError
        self.nn_params_dict = None
        self.config = structure

    def nn_params_dictize(self, params):
        nn_params_dict = dict()
        i = 0
        for name in self.config.keys():
            num = self.config[name]
            if 'w' in name:
                num_flat = num[0]*num[1]
                nn_params_dict[name] = (params[i:i+num_flat]).reshape(num)
                i += num_flat
            else:
                nn_params_dict[name] = params[i:i + num]
                i += num
        self.nn_params_dict = nn_params_dict

    def nn_params_flat(self):
        return np.hstack([v._value._value._value.flatten() for v in self.nn_params_dict.values()])


    def forward(self,x,params=None):
        if params is not None:
            self.nn_params_dictize(params) # update
        if self.nn_params_dict is None:
            raise ValueError('set nn parameters')
        output = x
        for name in self.nn_params_dict:
            if 'w' in name:
                # print(output.shape,(self.nn_params_dict[name]).shape)
                output = output@self.nn_params_dict[name]
            else:
                output = output+self.nn_params_dict[name]
                if 'b' not in name and self.nn_params_dict[name] != 1:
                    output = self.act_fun(output)
        return output

    def _warp_risk(self,x,y,Kz,params=None):
        r = risk(self.forward(x,params), y, params, Kz)
        return r



def gaussian_kernel(z,p):
    diffs = np.expand_dims(z/p, 1) - np.expand_dims(z/p, 0)
    Kz0 = np.sum(diffs ** 2, axis=2, keepdims=False)
    return np.exp(-Kz0/2)

def linear_kernel(z,p):
    return (z+p)@(z+p).T

def polynomial_kernel(z,p,d=2):
    return (z@z.T+p)**(d)

def rank_test(hessians_a, hessians_b):
    dim_root = int(np.sqrt(hessians_a.shape[1]))
    F_a = hessians_a.mean(axis=0)
    F_a = F_a.reshape((dim_root, dim_root))
    eigval, eigvec = np.linalg.eigh(F_a)
    C1 = eigvec[:, [0]]
    T_hat = (C1.T @ F_a@C1)**2
    C1C1 = (C1 @ C1.T).reshape((-1, 1))
    F_b_vec = hessians_b.mean(axis=0,keepdims=True).reshape((-1,1))
    # Omega0 = torch.stack(([h.reshape((-1, 1)) @ h.reshape((1, -1)) for h in hessians_b])).mean(axis=0) - F_b_vec @ F_b_vec.T
    Omega = hessians_b.T @ hessians_b/len(hessians_b) - F_b_vec@ F_b_vec.T
    obj = T_hat/(C1C1.T@Omega@C1C1)*len(hessians_a)
    return obj[0,0]

def silverman_ruleofthumb(z):
    # q75, q25 = np.percentile(z, [75, 25],axis=0)
    # iqr = q75 - q25
    # print(q75)
    d = z.shape[1]
    n = z.shape[0]

    #lengthscale = 0.9*min(z.std(),iqr/1.34)*np.power(len(z),-0.2)

    return (4/n/(d+2))**(1/(d+4))*z.std(axis=0)

def risk(pred,y,p,Kz):
    residual = pred - y
    output = residual.T@Kz@residual+0.001*np.sum(p*p)
    return output[0,0]


def data_gen(n_data, n_dim, func):
    e = torch.randn(n_data, 1, dtype=torch.float64)
    z = np.random.uniform(-3, 3, size=(n_data, n_dim))
    z = torch.tensor(z)
    x = z.mean(axis=1,keepdim=True)+ e  # .mean(axis=1,keepdim=True)
    g = func(x) # torch.heaviside(x,torch.tensor(0.0,dtype=torch.float64)) #torch.abs(x).sum(axis=1,keepdim=True)
    y = g+e

    mean_y, std_y = y.mean(axis=0), y.std(axis=0)
    y = (y-mean_y)/std_y
    g = (g-mean_y)/std_y

    return z.detach().numpy(), x.detach().numpy(), g.detach().numpy(), y.detach().numpy()

def main(kernel_func, kps, gen_func, n_dim, n_data, nn_type):
    if gen_func == 'linear':
        gen_func = lambda x:x
    elif gen_func == 'abs':
        gen_func = torch.abs
    elif gen_func == 'quad':
        gen_func = lambda x: x+x**2
    elif gen_func == 'sin':
        gen_func = torch.sin
    else:
        raise NotImplementedError
    train_z, train_x, train_g, train_y = data_gen(n_data, n_dim, gen_func)
    val_z, val_x, val_g, val_y = data_gen(n_data, n_dim, gen_func)
    test_z, test_x, test_g, test_y = data_gen(n_data, n_dim, gen_func)
    q9,q1 = np.percentile(test_x, [95, 5])
    test_x_flag = (q1<test_x)&(q9>test_x) #np.logic_and(q10<test_x.detach().numpy(),q10>test_x.detach().numpy())
    test_x = test_x[test_x_flag,None]
    test_g = test_g[test_x_flag,None]

    comb_y = np.vstack((train_y, val_y))
    comb_z = np.vstack((train_z, val_z))
    comb_x = np.vstack((train_x, val_x))

    if nn_type == 'NN55':
        nn_config = {'w1':[1,5],'b1':5,'w2':[5,5],'b2':5,'w3':[5,1],'b3':1}
    elif nn_type == 'NN10':
        nn_config = {'w1': [1, 10], 'b1': 10, 'w2': [10, 1], 'b2': 1}
    else:
        raise NotImplementedError
    n_params = np.sum([np.prod(e) for e in nn_config.values()])
    nn_params_flat = np.random.randn(n_params)


    l_ed = []
    l_rank_test_criterion = []
    l_error = []
    l_edc = []
    jitter_eye = JITTER * np.eye(n_data)
    jitter_eye2 = JITTER * np.eye(2*n_data)
    act = 'sigmoid'
    for kp in kps:
        if kp == -1: # silverman
            Kz = kernel_func(comb_z, silverman_ruleofthumb(comb_z))+jitter_eye2
            Kz_train = kernel_func(train_z, silverman_ruleofthumb(train_z)) + jitter_eye
            Kz_val = kernel_func(val_z, silverman_ruleofthumb(val_z)) + jitter_eye
        elif kp == -2:
            Kz = kernel_func(comb_z, get_median_inter_mnist(comb_z))+jitter_eye2
            Kz_train = kernel_func(train_z, get_median_inter_mnist(train_z)) + jitter_eye
            Kz_val = kernel_func(val_z, get_median_inter_mnist(val_z)) + jitter_eye
        else:
            Kz = kernel_func(comb_z,kp)+jitter_eye2
            Kz_train = kernel_func(train_z,kp)+jitter_eye
            Kz_val =  kernel_func(val_z,kp)+jitter_eye


        # Kz /= np.sqrt(np.trace(Kz@Kz))
        # Kz_train /= np.sqrt(np.trace(Kz_train @ Kz_train))
        # Kz_val /= np.sqrt(np.trace(Kz_val @ Kz_val))
        net = my_numpy_nn(act, nn_config)
        val_grad = value_and_grad(lambda p: net._warp_risk(train_x, train_y, Kz_train, p))
        minimize(val_grad, x0=nn_params_flat, jac=True, method='L-BFGS-B', options={'maxiter': 1000})
        risk_val = (net.forward(val_x)._value - val_y).T@Kz_val@(net.forward(val_x)._value - val_y)
        ed1 = effective_dim(Kz_val)
        edc1 = risk_val[0,0] / (n_data) + ed1 * np.log(n_data)
        net = my_numpy_nn(act, nn_config)
        val_grad = value_and_grad(lambda p: net._warp_risk(val_x, val_y, Kz_val, p))
        minimize(val_grad, x0=nn_params_flat, jac=True, method='L-BFGS-B', options={'maxiter': 1000})
        risk_val = (net.forward(train_x)._value - train_y).T @ Kz_val @ (net.forward(train_x)._value - train_y)
        ed2 = effective_dim(Kz_train)
        edc2 = risk_val[0,0] / (n_data) + ed1 * np.log(n_data)
        l_ed += [ed2 + ed1]
        l_edc += [(edc1 + edc2) / 2]

        net = my_numpy_nn(act, nn_config)
        val_grad = value_and_grad(lambda p: net._warp_risk(comb_x, comb_y, Kz, p))
        res = minimize(val_grad, x0=nn_params_flat, jac=True, method='L-BFGS-B', options={'maxiter': 1000})
        l_error += [np.mean((net.forward(test_x)._value- test_g) ** 2)]


        # grad_matrix = []
        # for id in range(comb_x.shape[0]):
        #     val_grad = value_and_grad(lambda p: net.forward(comb_x[id,None],p))
        #     _,g = val_grad(nn_params_flat)
        #     grad_matrix += [g]
        # grad_matrix = np.vstack(grad_matrix)
        if nn_type == 'NN55':
            grad_matrix = net.act_fun(net.act_fun(comb_x@net.nn_params_dict['w1']+net.nn_params_dict['b1'])@net.nn_params_dict['w2']+net.nn_params_dict['b2'])
        elif nn_type == 'NN10':
            grad_matrix = net.act_fun(comb_x @ net.nn_params_dict['w1'] + net.nn_params_dict['b1'])
        grad_matrix = np.hstack((grad_matrix._value,np.ones((grad_matrix.shape[0],1))))

        stack_train = np.einsum('nab,mbc->nmac',
                                   np.expand_dims(grad_matrix[:n_data,],2), np.expand_dims(grad_matrix[:n_data,],1)).reshape(
            (-1, (grad_matrix.shape[-1]) ** 2))
        stack_val = np.einsum('nab,mbc->nmac',
                                   np.expand_dims(grad_matrix[n_data:,], 2),
                                    np.expand_dims(grad_matrix[n_data:,], 1)).reshape((-1, (grad_matrix.shape[-1]) ** 2))
        del grad_matrix
        hess_train = Kz_train.reshape((-1, 1)) * stack_train
        del Kz_train, stack_train
        hess_val = Kz_val.reshape((-1, 1)) * stack_val
        del Kz_val, stack_val
        rank_test_criterion = (rank_test(hess_train, hess_val) + rank_test(hess_val, hess_train)) / 2
        del hess_train,hess_val
        l_rank_test_criterion += [rank_test_criterion]

    return l_ed, l_error, l_rank_test_criterion,l_edc


def run_setting(simulation,n_dim, n_data,param_id,ite_id, nn_type):
    # config = [('L', linear_kernel, np.array([0])),
    #           ('P2', polynomial_kernel, np.array([0.1, 0.2, 0.5, 1, 2, 5])),
    #           ('P4', lambda z, p: polynomial_kernel(z, p, 4), np.array([0.1, 0.2, 0.5, 1, 2, 5])),
    #           ('G', gaussian_kernel, np.array([5,2, 1,0.5, 0.2, 0.1,-1, -2]))]

    # for name, func, kps in config:
    #     for ite in range(10):
    #         results = [main(func, kps, simulation, n_dim, n_data)]
    #         np.save(ROOT_PATH+'/results/{}_{}_{}.npy'.format(simulation,name,ite),results)

    if param_id == 0:
        name = 'L'
        func = linear_kernel
        kps = np.array([0])
    elif param_id>0 and param_id<7:
        name = 'P2'
        func = polynomial_kernel
        kps = [np.array([0.1, 0.2, 0.5, 1, 2, 5])[param_id-1]]
    elif param_id>=7 and param_id<13:
        name = 'P4'
        func = lambda z, p: polynomial_kernel(z, p, 4)
        kps = [np.array([0.1, 0.2, 0.5, 1, 2, 5])[param_id - 7]]
    else:
        name = 'G'
        func = gaussian_kernel
        kps = [np.array([5,2, 1,0.5, 0.2, 0.1,-1, -2])[param_id - 13]]
    file = ROOT_PATH+'/results/{}/{}_{}_{}_{}.npy'.format(nn_type,simulation,name,kps[0],ite_id)
    # if not os.path.exists(file):
    results = [main(func, kps, simulation, n_dim, n_data, nn_type)]
    np.save(file,results)




def plot_stack_value_list(stack_value_list,ticklabels,target_file, ylabel,labels,markers):
    colors = {'G': 'r', 'L': 'g', 'P2': 'blue', 'P4': 'black'}

    val_max = np.max(np.hstack(np.hstack([list(stack_value_list[0].values()), list(stack_value_list[1].values())])))
    val_min = np.min(np.hstack(np.hstack([list(stack_value_list[0].values()), list(stack_value_list[1].values())])))
    for exp_id in range(len(stack_value_list)):
        start_id = 0
        for name in stack_value_list[exp_id].keys():
            plt.scatter(range(start_id, start_id + len(stack_value_list[exp_id][name])),
                        (stack_value_list[exp_id][name] - val_min + const) / (val_max - val_min + const),
                        marker=markers[exp_id],color=colors[name],label=labels[exp_id])

            j = 0
            for i in range(start_id, start_id + len(stack_value_list[exp_id][name])):
                print(i, (stack_value_list[exp_id][name][j]- val_min + const)/ (val_max - val_min + const), name+labels[exp_id])
                j += 1
            start_id += len(stack_value_list[exp_id][name])

    if ylabel == 'ITC':
        plt.plot([0,len(ticklabels)], (np.array([chi2.ppf(0.95, 1), chi2.ppf(0.95, 1)]) - val_min + const) / (val_max - val_min + const), 'r')
        print('KEIC threshold: ',(chi2.ppf(0.95, 1)- val_min + const)/ (val_max - val_min + const))
    plt.legend()
    plt.xticks(range(start_id), ticklabels, rotation=45)
    plt.semilogy()
    plt.grid()
    plt.xticks(range(start_id), ticklabels, rotation=45)
    plt.ylabel(ylabel)
    plt.savefig(target_file, bbox_inches='tight')
    plt.close('all')

def normalize(a):
    min_v = np.min(a[a>0])
    return (a - min_v+const)/(np.max(a)-min_v+const)

def analyze_res(nn_type):
    config = [('L', np.array([0])),
              ('P2', np.array([0.1, 0.2, 0.5, 1, 2, 5])),
              ('P4',np.array([0.1, 0.2, 0.5, 1, 2, 5])),
              ('G', np.array([5, 2, 1, 0.5, 0.2, 0.1, -1, -2]))]


    error_table = [['a' for i in range(4)] for j in range(3)]
    j = 0
    for simu in ['abs','linear','quad','sin']:
        print(simu)
        stack_error = []
        for ite in range(10):
            res = []
            for kname, kps in config:
                for kp in kps:
                    file = ROOT_PATH + '/results/{}/{}_{}_{}_{}.npy'.format(nn_type,simu,kname,kp, ite)
                    if os.path.exists(file):
                        res += [np.load(file,allow_pickle=True).flatten()]
                    else:
                        print(file,' not exists')
            res = np.array(res)
            edc = res[:,3]
            tc = res[:, 2]
            ratio = (edc/tc)[:-2]
            pratio = ratio[ratio>0]
            # ratio
            # print(normalize(tc))
            error = res[:,1]
            optid=np.where(ratio == np.min(pratio))
            # min_tc = np.min(tc[tc > 0])
            stack_error += [[ error[-1], error[-2],  error[optid[0]][0]]]

        stack_error = np.array(stack_error)
        std0 = np.std(stack_error,axis=0)
        mean0 = np.mean(stack_error,axis=0)
        for i in range(len(mean0)):
            mean = np.mean(stack_error[np.abs(stack_error[:,i] - mean0[i]) < std0[i],i], axis=0)
            std = np.std(stack_error[np.abs(stack_error[:,i] - mean0[i]) < std0[i],i], axis=0)
            print(i, np.sum(np.abs(stack_error[:,i] - mean0[i]) < std0[i]))
            error_table[i][j] = '{0:.3f}$\pm${1:.3f}'.format(mean, std)
        j += 1
    for row in error_table:
        print(' & '.join(row))


if __name__ == '__main__':


    for nn_type in ['NN10','NN55']:
        os.makedirs(ROOT_PATH + '/results/{}/'.format(nn_type), exist_ok=True)
        simus = ['abs', 'linear', 'quad', 'sin']

        #### To save time, we directly provide the results obtained from the cluster.
        #### You can directly run this script.
        #### Othereise, set run_from_scratch = True, and set cluster = True or False
        run_from_scratch = False
        cluster = True

        if run_from_scratch:
        #### we used clusters to compute the below loop; otherwise it is slow.
            if cluster:
                parser = argparse.ArgumentParser(description='Process some integers.')
                parser.add_argument('--jobid', type=int, help='jobid on cluster')
                args = parser.parse_args(sys.argv[1:])
                jobid = int(args.jobid)
                envid, pid = divmod(jobid, 210)
                ite_id,param_id = divmod(pid,21)
                np.random.seed(ite_id+527)
                simu = simus[envid]
                run_setting(simu, 2, 500, param_id,ite_id,nn_type)
            else:
        #### Squential: this is slow
                for jobid in range(840):
                    envid, pid = divmod(jobid,210)
                    ite_id,param_id = divmod(pid,21)
                    np.random.seed(ite_id+527)
                    simu = simus[envid]
                    run_setting(simu, 2, 500, param_id,ite_id,nn_type)

        # after the above loop, analyze stored results
        analyze_res(nn_type)



