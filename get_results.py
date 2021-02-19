from util import ROOT_PATH
import numpy as np
import os
import texttable, latextable

def get_low_x_z():
    methods = ['our_method','MMRIV']
    print_array = []
    for method in methods:
        print('method: ', method)
        test_errs = dict()
        z_weights = dict()
        res_dir = ROOT_PATH + '/results/low_x_z/{}'.format(method)
        res_files = os.listdir(res_dir)
        for f in res_files:
            res = np.load(res_dir+'/'+f,allow_pickle=True)
            # res = dict(np.ndenumerate(res))
            res=res.item()
            test_errs[f[:-6]] = test_errs.get(f[:-6],[]) + [res['test_err']]
            if method is 'our_method':
                z_weights[f[:-6]] = z_weights.get(f[:-6],[]) + [np.exp(res['weights_z'])[-3:-1]]
            else:
                z_weights[f[:-6]] = z_weights.get(f[:-6], []) + [np.exp(res['weights_z'])]
        print('dataset   size')
        tmp_array = []
        for k in test_errs.keys():
            merr = np.round(np.mean(test_errs[k]),3)
            stderr = np.round(np.std(test_errs[k]),3)
            substr_err = '$'+str(merr)+'_{\pm '+str(stderr)+'}$'
            mzw = np.round(np.mean(z_weights[k],axis=0),3)
            stdzw = np.round(np.std(z_weights[k],axis=0),3)
            substr_zw = ''
            if type(mzw) == np.float64:
                substr_zw += '$'+str(mzw)+'_{\pm '+str(stdzw)+'}$'
            else:
                for i in range(len(mzw)):
                    substr_zw += '$'+str(mzw[i])+'_{\pm '+str(stdzw[i])+'}$ '
            tmp_array += [[k[:-4],substr_err,substr_zw]]
            # print('   {}    {}'.format(k[:-4], k[-3:]), 'test errors (mean square error): ', np.mean(test_errs[k]), '+-',np.std(test_errs[k]))
            # print('                 lengthscale of Z:', np.mean(z_weights[k],axis=0),'+-',np.std(z_weights[k],axis=0))

        print_array+=(np.array(tmp_array).T).tolist()
    draw_latex(print_array,'Results on low dimensional X and Z datasets with 100 data samples.',label='tab:result_low_xz_100')





def draw_latex(table_contents, caption=None, label=None):
    ''' 
    :param table: Texttable table to be rendered in Latex.
    :param caption: A string that adds a caption to the Latex formatting.
    :param label: A string that adds a referencing label to the Latex formatting.
    :param drop_columns: A list of column names that won't be in the Latex output.
        Each column name must be in the table header.
    :return: The formatted Latex table returned as a single string.
    '''
    table = texttable.Texttable()
    table.set_cols_align(["c"]*(len(table_contents[0])))
    table.add_rows(table_contents)
    print(table.draw() + "\n")
    if label is not None:
        print('\label{'+label+'}')
    print(latextable.draw_latex(table, caption=caption),"\n")

if __name__ == '__main__':
    get_low_x_z()