import autograd.numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import chi2
torch.manual_seed(527)
np.random.seed(527)
from util import get_median_inter_mnist,ROOT_PATH
import os
const = 1e-4
JITTER = 1e-7

fig_folder = ROOT_PATH+'/figs'

def effective_dim(Kz):

    return torch.trace(Kz)/torch.trace(Kz@Kz).sqrt()

def gaussian_kernel(z,p):
    diffs = torch.unsqueeze(z/p, 1) - torch.unsqueeze(z/p, 0)
    Kz0 = torch.sum(diffs ** 2, axis=2, keepdim=False)
    return torch.exp(-Kz0/2)

def linear_kernel(z,p):
    return (z+p)@(z+p).T

def polynomial_kernel(z,p,d=2):
    return (z@z.T+p).pow(d)

def rank_test(hessians_a, hessians_b):
    dim_root = int(np.sqrt(hessians_a.shape[1]))
    F_a = hessians_a.mean(axis=0)
    F_a = F_a.reshape((dim_root, dim_root))
    eigval, eigvec = torch.symeig(F_a,eigenvectors=True)
    T_hat = eigval[0]**2
    C1 = eigvec[:, [0]]
    # T_hat = (C1.T@F_a@C1)**2
    C1C1 = (C1 @ C1.T).reshape((-1, 1))
    F_b = hessians_b.mean(axis=0)
    F_b_vec = F_b.reshape((-1,1))
    # Omega0 = torch.stack(([h.reshape((-1, 1)) @ h.reshape((1, -1)) for h in hessians_b])).mean(axis=0) - F_b_vec @ F_b_vec.T
    Omega = hessians_b.T @ hessians_b/len(hessians_b) - F_b_vec @ F_b_vec.T
    # print(Omega0-Omega)
    obj = T_hat/(C1C1.T@Omega@C1C1)*len(hessians_a)*2
    return obj[0,0]

def silverman_ruleofthumb(z):
    # q75, q25 = np.percentile(z, [75, 25],axis=0)
    # iqr = q75 - q25
    # print(q75)
    d = z.shape[1]
    n = z.shape[0]

    #lengthscale = 0.9*min(z.std(),iqr/1.34)*np.power(len(z),-0.2)

    return (4/n/(d+2))**(1/(d+4))*z.std(axis=0)

def risk(x,y,p,Kz,reg):
    residual = x@p.reshape((-1,1)) - y
    output = residual.T@Kz@residual+reg*np.sum(p*p)
    return output[0,0]


def data_gen(n_data, n_dim, func):
    e = torch.randn(n_data, 1, dtype=torch.float64)
    if n_dim >=1:
        z = np.random.uniform(-3, 3, size=(n_data, n_dim))
        z = torch.tensor(z)
        x = z.mean(axis=1,keepdim=True)+e#np.sin(z.mean(axis=1,keepdim=True))+e
    else:
        z = np.random.uniform(-3, 3, size=(n_data, 1))
        z = torch.tensor(z)
        x = np.sin(z.mean(axis=1,keepdim=True))+e
    g = func(x) # torch.heaviside(x,torch.tensor(0.0,dtype=torch.float64)) #torch.abs(x).sum(axis=1,keepdim=True)
    y = g+e

    mean_y, std_y = y.mean(axis=0), y.std(axis=0)
    y = (y-mean_y)/std_y
    g = (g-mean_y)/std_y

    return z, x, g, y

def poly_basis(x,d):
    return torch.hstack(([x.pow(i) for i in range(d+1)]))


def torch_chol_inv(A,jitter=JITTER):
    while True:
        try:
            u = torch.cholesky(A+jitter*np.eye(A.shape[0]))
            break
        except:
            jitter *=10
            print('singular and change jitter to ',jitter)
    return torch.cholesky_inverse(u)

def main(kernel_func, kps, poly_degree, gen_func, n_dim, n_data):
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
    test_x_flag = (q1<test_x.detach().numpy())&(q9>test_x.detach().numpy()) #np.logic_and(q10<test_x.detach().numpy(),q10>test_x.detach().numpy())
    test_x = test_x[test_x_flag,None]
    test_g = test_g[test_x_flag,None]

    comb_y = torch.vstack((train_y,val_y))
    comb_z = torch.vstack((train_z, val_z))
    # print(get_median_inter_mnist(comb_z))

    train_x_feature = poly_basis(train_x, poly_degree)
    test_x_feature = poly_basis(test_x, poly_degree)
    val_x_feature = poly_basis(val_x, poly_degree)
    comb_x_feature = torch.vstack((train_x_feature,val_x_feature))
    stack_train = torch.einsum('nab,mbc->nmac', (train_x_feature.unsqueeze(-1), train_x_feature.unsqueeze(-2))).reshape((-1,(train_x_feature.shape[-1])**2))
    # should be cartesian product, wrong: (train_x_feature.unsqueeze(-1)@train_x_feature.unsqueeze(-2)).reshape((-1,(train_x_feature.shape[-1])**2))
    stack_val = torch.einsum('nab,mbc->nmac', (val_x_feature.unsqueeze(-1), val_x_feature.unsqueeze(-2))).reshape((-1,(val_x_feature.shape[-1])**2))

    l_ed = []
    l_rank_test_criterion = []
    l_error = []
    l_nit = []
    l_edc = []
    jitter_eye = JITTER * torch.eye(n_data)
    jitter_eye2 = JITTER * torch.eye(2*n_data)
    for kp in kps:
        if kp == -1: # silverman
            # print('silverman kernel params:', silverman_ruleofthumb(comb_z))
            Kz = kernel_func(comb_z, silverman_ruleofthumb(comb_z))+jitter_eye2
            Kz_train = kernel_func(train_z, silverman_ruleofthumb(train_z)) + jitter_eye
            Kz_val = kernel_func(val_z, silverman_ruleofthumb(val_z)) + jitter_eye
        elif kp == -2:
            # print('heuristic kernel params:',get_median_inter_mnist(comb_z))
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

        hess_train = Kz_train.reshape((-1,1))*stack_train
        hess_val = Kz_val.reshape((-1, 1)) * stack_val
        rank_test_criterion = (rank_test(hess_train, hess_val)+rank_test(hess_val,hess_train))/2
        l_rank_test_criterion += [rank_test_criterion]

        reg = 0.001
        res_theta = torch_chol_inv(train_x_feature.T@Kz_train@train_x_feature+reg*torch.eye(train_x_feature.shape[1]))@train_x_feature.T@Kz_train@train_y
        risk_val = (val_y-val_x_feature@res_theta).T@Kz_val@(val_y-val_x_feature@res_theta)
        ed1 = effective_dim(Kz_val)
        edc1 = risk_val/(n_data)+ed1*np.log(n_data)

        res_theta = torch_chol_inv(val_x_feature.T @ Kz_val @ val_x_feature + reg * torch.eye(
            val_x_feature.shape[1])) @ val_x_feature.T @ Kz_val @ val_y
        risk_val = (train_y - train_x_feature @ res_theta).T @ Kz_train @ (train_y - train_x_feature @ res_theta)
        ed2 = effective_dim(Kz_train)
        edc2 = risk_val/(n_data) + ed2 * np.log(n_data)
        l_ed += [ed2+ed1]
        l_edc += [(edc1+edc2)/2]#[risk_val*ed] +ed*np.log(n_data*2)

        res_theta = torch_chol_inv(comb_x_feature.T @ Kz@ comb_x_feature + reg * torch.eye(
            comb_x_feature.shape[1])) @ comb_x_feature.T @ Kz @ comb_y
        l_error += [torch.mean((test_x_feature@res_theta- test_g)**2).detach().numpy()]
        # plt.scatter(x,x_1.detach().numpy()@res.x.reshape((-1,1)))
        # plt.scatter(x,g.detach().numpy())
        # plt.title(kp)
        # plt.show()

        l_nit += [0]

    return l_ed, l_error, l_rank_test_criterion,l_edc


def run_setting(simulation,poly_degree,n_dim, n_data, n_exp,config):
    stack_ed = dict()
    stack_error = dict()
    stack_rank_test_criterion = dict()
    stack_edc = dict()
    ticklabels = []
    std_error = dict()
    for name, func, kps in config:
        results = []
        for _ in range(n_exp):
            try:
                results += [main(func, kps, poly_degree, simulation,n_dim, n_data)]
            except Exception as e:
                print(_,e)
        stack_ed[name], stack_error[name], stack_rank_test_criterion[name], stack_edc[name] = [np.mean([results[j][i] for j in range(len(results))], axis=0) for i in range(len(results[0]))]
        std_error[name] = np.std([results[j][1] for j in range(len(results))], axis=0)
        ticklabels += [name] if name is 'L' else [name + '-${{{0:g}}}$'.format(e) for e in kps]




    #
    #
    # start_id = 0
    # print('error')
    # for name in stack_error.keys():
    #     plt.scatter(range(start_id, start_id + len(stack_error[name])),stack_error[name],
    #                 #(stack_error[name] - min_error + const) / (max_error - min_error + const),
    #                 color=colors[name], label=name)
    #
    #     j = 0
    #     # for i in range(start_id, start_id + len(stack_error[name])):
    #         # print(i, stack_error[name][j], name)
    #         # print(ticklabels[i])
    #         # j += 1
    #
    #     start_id += len(stack_error[name])
    #     # plt.plot((stack_ed[name]-min_ed+const)/(max_ed-min_ed+const),color=colors[name], linestyle='-',label=name+' effective dim')
    #     # plt.plot(kps, (stack_error[name]-min_error+const)/(max_error-min_error+const),'-.', label=name+' MSE')
    #     # plt.plot(kps, stack_error[name],color=colors[name], linestyle='-.', label=name + ' MSE')
    # plt.legend()
    # pick_id= np.where(stack_edc2 == np.min(stack_edc2[identifiable]))
    # error_str = ['{0:.3f}$\pm${1:.3f}'.format(stack_error2[-1],std_error2[-1]),'{0:.3f}$\pm${1:.3f}'.format(stack_error2[-2],std_error2[-2]),'{0:.3f}$\pm${1:.3f}'.format(stack_error2[pick_id[0][0]],std_error2[pick_id[0][0]])]
    # print('silverman heu our '+' '.join(error_str))
    # plt.xticks(range(start_id), ticklabels, rotation=45)
    # plt.xlabel('kernel parameter')
    # plt.ylabel('error')
    # plt.grid()
    # plt.semilogy()
    # plt.savefig(target_folder + '/error{}_{}.pdf'.format(poly_degree,n_dim), bbox_inches='tight')
    # plt.close('all')
    return stack_rank_test_criterion, stack_edc, ticklabels, stack_error,stack_ed


def plot_stack_value_list(stack_value_list,ticklabels,target_file, ylabel,labels,markers):
    colors = {'G': 'r', 'L': 'g', 'P2': 'blue', 'P4': 'black'}

    val_max = np.max(np.hstack(np.hstack([list(stack_value_list[i].values()) for i in range(len(stack_value_list))])))
    val_min = np.min(np.hstack(np.hstack([list(stack_value_list[i].values()) for i in range(len(stack_value_list))])))
    for exp_id in range(len(stack_value_list)):
        start_id = 0
        for name in stack_value_list[exp_id].keys():
            plt.scatter(range(start_id, start_id + len(stack_value_list[exp_id][name])),
                        (stack_value_list[exp_id][name] - val_min + const) / (val_max - val_min + const),
                        marker=markers[exp_id],color=colors[name],label=labels[exp_id])

            j = 0
            for i in range(start_id, start_id + len(stack_value_list[exp_id][name])):
                print(i, (stack_value_list[exp_id][name][j]- val_min + const)/ (val_max - val_min + const), 'K'+labels[exp_id])
                j += 1
            start_id += len(stack_value_list[exp_id][name])

    if ylabel == 'ITC':
        plt.plot([0,len(ticklabels)], (np.array([chi2.ppf(0.95, 1), chi2.ppf(0.95, 1)]) - val_min + const) / (val_max - val_min + const), 'r')
        print('ITC threshold: ',(chi2.ppf(0.95, 1)- val_min + const)/ (val_max - val_min + const))
    plt.legend()
    plt.xticks(range(start_id), ticklabels, rotation=45)
    plt.semilogy()
    plt.grid()
    plt.xticks(range(start_id), ticklabels, rotation=45)
    plt.ylabel(ylabel)
    plt.savefig(target_file, bbox_inches='tight')
    plt.close('all')


def experiment_strong():
    n_dim = 1

    n_exp = 10
    config = [('L', linear_kernel, torch.tensor([0])),
              ('P2', polynomial_kernel, torch.tensor([1,2])),
              ('P4', lambda z, p: polynomial_kernel(z, p, 4), torch.tensor([1,2])),
              ('G', gaussian_kernel, torch.tensor([2, 1, 0.5, 0.2, 0.1]))]
    # previously we did each experiment separately
    for f_dim in [2,4]:
        for simu in ['linear','quad']:
            if simu == 'quad':
                torch.manual_seed(528)
                np.random.seed(528)
            print('######### ' + simu + ' f{} #########'.format(f_dim))
            stack_edc_list = []
            stack_tc_list = []
            stack_ed_list = []

            stack_tc, stack_edc, ticklabels,_,stack_ed = run_setting(simu, f_dim, n_dim, 100,n_exp,config)
            stack_edc_list += [stack_edc]
            stack_tc_list += [stack_tc]
            stack_ed_list += [stack_ed]

            stack_tc, stack_edc, ticklabels,_,stack_ed = run_setting(simu, f_dim, n_dim, 500,n_exp,config)
            stack_edc_list += [stack_edc]
            stack_tc_list += [stack_tc]
            stack_ed_list += [stack_ed]

            stack_tc, stack_edc, ticklabels,_,stack_ed = run_setting(simu, f_dim, n_dim, 1000,n_exp,config)
            stack_edc_list += [stack_edc]
            stack_tc_list += [stack_tc]
            stack_ed_list += [stack_ed]

            # plot config
            target_folder = fig_folder + '/' + simu
            os.makedirs(target_folder, exist_ok=True)

            ## plot keic: different settings in one plot
            print('KEIC')
            file = target_folder + '/l_keic_{}{}.pdf'.format('s' if n_dim == 1 else 'w', f_dim)
            plot_stack_value_list(stack_edc_list, ticklabels, file, 'KEIC',['100','500','1000'],['*','.','s'])
            print('ITC')
            file = target_folder + '/l_itc_{}{}.pdf'.format('s' if n_dim == 1 else 'w', f_dim)
            plot_stack_value_list(stack_tc_list, ticklabels, file, 'ITC',['100','500','1000'],['*','.','s'])
            print('ED')
            file = target_folder + '/l_ed_{}{}.pdf'.format('s' if n_dim == 1 else 'w', f_dim)
            plot_stack_value_list(stack_ed_list, ticklabels, file, 'ED',['100','500','1000'],['*','.','s'])


def experiment_strong_weak_nonlinear(simu,f_dim=4):
    n_data = 100

    config = [('L', linear_kernel, torch.tensor([0])),
              ('P2', polynomial_kernel, torch.tensor([0.1,0.2, 0.5, 1,2, 5])),
              ('P4', lambda z, p: polynomial_kernel(z, p, 4), torch.tensor([0.1,0.2, 0.5, 1,2, 5])),
              ('G', gaussian_kernel, torch.tensor([5, 2, 1, 0.5, 0.2, 0.1,-2,-1]))]

    print('######### ' + simu + ' #########')
    stack_edc_list = []
    stack_tc_list = []
    stack_error_list = []
    stack_tc, stack_edc, ticklabels,stack_error,_ = run_setting(simu, f_dim, 1, n_data,1,config)  # f2, strong
    stack_edc_list += [stack_edc]
    stack_tc_list += [stack_tc]
    stack_error_list += [stack_error]

    stack_tc, stack_edc, ticklabels,stack_error,_ = run_setting(simu, f_dim, 6, n_data,1,config)  # f4, strong
    stack_edc_list += [stack_edc]
    stack_tc_list += [stack_tc]
    stack_error_list += [stack_error]

    stack_tc, stack_edc, ticklabels,stack_error,_ = run_setting(simu, f_dim, -1, n_data,1,config)  # f4, strong
    stack_edc_list += [stack_edc]
    stack_tc_list += [stack_tc]
    stack_error_list += [stack_error]

    # plot config
    target_folder = fig_folder + '/' + simu
    os.makedirs(target_folder, exist_ok=True)

    ##
    results = []
    for exp_id in range(len(stack_error_list)):
        stack_tc =  np.hstack(stack_tc_list[exp_id].values())[:-2]
        stack_edc = np.hstack(stack_edc_list[exp_id].values())[:-2]
        stack_error = np.hstack(stack_error_list[exp_id].values())
        id = stack_tc>= chi2.ppf(0.95, 1)
        if np.any(id):
            optid = np.where(stack_edc == min(stack_edc[id]))
            optid = optid[0]
            # print('our, heuristic, sliverman: ', stack_error[optid],stack_error[-2], stack_error[-1])
            results += [[stack_error[-1],stack_error[-2], stack_error[optid][0]]]
        else:
            print('nan')
            results += [[np.nan,np.nan, np.nan]]

    return results

    ## plot keic: different settings in one plot
    # print('KEIC')
    # file = target_folder + '/l_keic_{}{}.pdf'.format('swn', n_data)
    # plot_stack_value_list(stack_edc_list, ticklabels, file, 'KEIC',['strong','weak','nonlinear'],['*','.','o'])
    # print('ITC')
    # file = target_folder + '/l_itc_{}{}.pdf'.format('swn', n_data)
    # plot_stack_value_list(stack_tc_list, ticklabels, file, 'ITC',['strong','weak','nonlinear'],['*','.','o'])

def experiment1():
    experiment_strong()

def experiment2():
    table_error = [['a' for j in range(4)] for i in range(9)]
    simus = ['abs', 'linear', 'quad', 'sin']
    for k in range(len(simus)):
        simu = simus[k]
        stack_res = []
        for _ in range(10):
            res = experiment_strong_weak_nonlinear(simu)
            stack_res += [res]
        mean = np.nanmean(stack_res, axis=0)
        std = np.nanstd(stack_res, axis=0)
        for i in range(mean.shape[0]):
            for j in range(mean.shape[1]):
                table_error[i * 3 + j][k] = '{0:.3f} $\pm$ {1:.3f}'.format(mean[i, j], std[i, j])
    for r in table_error:
        print(' & '.join(r))

if __name__ == '__main__':
    print('running experiment 1')
    experiment1()

    print('running experiment 2')
    experiment2()

