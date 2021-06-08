import os, sys

ROOT_PATH = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
# ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_PATH)
import torch
import autograd.numpy as np

JITTER = 1e-7
nystr_M = 300
EYE_nystr = np.eye(nystr_M)


def _sqdist(x, y, Torch=False):
    if y is None:
        y = x
    if Torch:
        diffs = torch.unsqueeze(x, 1) - torch.unsqueeze(y, 0)
        sqdist = torch.sum(diffs ** 2, axis=2, keepdim=False)
    else:
        diffs = np.expand_dims(x, 1) - np.expand_dims(y, 0)
        sqdist = np.sum(diffs ** 2, axis=2)
        del diffs
    return sqdist


def get_median_inter_mnist(x):
    # x2 = np.sum(x*x,axis=1,keepdims=True)
    # sqdist = x2+x2.T-2*x@x.T
    # sqdist = (sqdist+abs(sqdist).T)/2
    if x.shape[0] < 10000:
        sqdist = _sqdist(x, None)
    else:
        M = int(x.shape[0] / 400)
        sqdist = Parallel(n_jobs=20)(delayed(_sqdist)(x[i:i + M], x) for i in range(0, x.shape[0], M))
    dist = np.sqrt(sqdist).flatten()
    return np.median(dist)


