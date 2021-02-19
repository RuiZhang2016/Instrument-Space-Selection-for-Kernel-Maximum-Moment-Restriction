import numpy as np
import torch
import os
from util import ROOT_PATH

np.random.seed(527)
torch.manual_seed(527)

class data_generation():

    def __init__(self):
        self.ndata = None
        self.func_name = None
        self.scenarios = {'low_x_z':self._low_x_z, 'low_x_high_z':self._low_x_high_z,'high_x_z':self._high_x_z}

    def _true_f(self,x):
        if self.func_name == 'linear':
            return x
        elif self.func_name == 'abs':
            return np.abs(x)
        elif self.func_name == 'sin':
            return np.sin(x)
        elif self.func_name == 'step':
            return 0. * (x<0) + 1. * (x>=0)
        else:
            raise NotImplementedError()

    def _low_x_z(self):
        Z = np.random.uniform(-3,3,size=(self.ndata,2))
        epsilon = np.random.normal(0,1,size=(self.ndata,1))
        gamma = np.random.normal(0,0.1,size=(self.ndata,1))
        X = Z[:,[0]] + epsilon + gamma
        delta = np.random.normal(0,0.1,size=(self.ndata,1))
        G = self._true_f(X)
        Y = G + delta + epsilon
        return dataset(Z, X, Y, G)

    def _low_x_high_z(self):
        raise NotImplementedError()

    def _high_x_z(self):
        raise NotImplementedError()

    def data_gen(self, ndata, func_name, scenario):
        self.ndata = ndata
        self.func_name = func_name
        if scenario in self.scenarios.keys():
            return (self.scenarios[scenario])()
        else:
            raise NotImplementedError()


class dataset(object):
    def __init__(self, Z,X,Y,G):
        self.X = X
        self.Z = Z
        self.Y = Y
        self.G = G
        self.size = None

    def to_tensor(self):
        self.X = torch.as_tensor(self.X).double()
        self.Z = torch.as_tensor(self.Z).double()
        self.Y = torch.as_tensor(self.Y).double()
        self.G = torch.as_tensor(self.G).double()

    def to_2d(self):
        n_data = self.Y.shape[0]
        if len(self.X.shape) > 2:
            self.X = self.X.reshape(n_data, -1)
        if len(self.z.shape) > 2:
            self.Z = self.Z.reshape(n_data, -1)

    def info(self, verbose=False):
        for name, x in [("X", self.X), ("Z", self.Z), ("Y", self.Y), ("G", self.G)]:
            print("  " + name + ":", x.__class__.__name__, "(" + str(x.dtype) + "): ",
                  "x".join([str(d) for d in x.shape]))
            if verbose:
                with np.printoptions(precision=3, suppress=True):
                    print("      min: %.2f" % x.min(), ", max: %.2f" % x.max(), ", mean: {}".format(x.mean(axis=0)))

    def as_tuple(self):
        return self.Z, self.X, self.Y, self.G

    def as_dict(self, prefix=""):
        d = {"X": self.X, "Z": self.Z, "Y": self.Y, "G": self.G}
        return {prefix + k: v for k, v in d.items()}

    def to_numpy(self):
        self.X = self.X.data.numpy()
        self.Z = self.Z.data.numpy()
        self.Y = self.Y.data.numpy()
        self.G = self.G.data.numpy()

    def to_cuda(self):
        self.X = self.X.cuda()
        self.Z = self.Z.cuda()
        self.Y = self.Y.cuda()
        self.G = self.G.cuda()

class Standardizer():

    def __init__(self):
        self._mean = None
        self._std = None

    def normalize(self, data):
        Z, X, Y, G = data.Z, data.X, data.Y, data.G
        if self._mean is None:
            self._mean = Y.mean()
            self._std = Y.std()
        Y = (Y - self._mean) / self._std
        G = (G - self._mean) / self._std
        return dataset(Z,X,Y,G)

def data_generation_test():
    sd = Standardizer()
    dg = data_generation()
    data = dg.data_gen(1000,'linear','low_x_z')
    data = sd.normalize(data)
    data.info(verbose=True)

if __name__ == '__main__':
    # data_generation_test()

    scenario = 'low_x_z'
    target_dir = ROOT_PATH+'/data/low_x_z'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for s in [100,1000]:
        for f in ['step','sin','abs','linear']:
            sd = Standardizer()
            dg = data_generation()

            data = dg.data_gen(s, f, scenario)
            train = sd.normalize(data)
            data = dg.data_gen(s, f, scenario)
            dev = sd.normalize(data)
            data = dg.data_gen(s, f, scenario)
            test = sd.normalize(data)
            np.save(target_dir+'/{}_{}'.format(f,s),[train,dev,test])

