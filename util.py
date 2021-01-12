import autograd.numpy as np
import torch

JITTER = 1e-7
nystr_M = 300
EYE_nystr = np.eye(nystr_M)

def _sqdist(x,y,Torch=False):
    if y is None:
        y = x
    if Torch:
        diffs = torch.unsqueeze(x,1)-torch.unsqueeze(y,0)
        sqdist = torch.sum(diffs**2, axis=2, keepdim=False)
    else:
        diffs = np.expand_dims(x,1)-np.expand_dims(y,0)
        sqdist = np.sum(diffs**2, axis=2)
        del diffs
    return sqdist

def Kernel(name, Torch=False):
    def poly(x, y, c, d):
        if y is None:
            y = x
            res = (x @ y.T + c * c) ** d
            res = (res + res.T) / 2
            return res
        else:
            return (x @ y.T + c * c) ** d

    def rbf(x, y, a, b, Torch=Torch):
        if y is None:
            y = x
        # sqdist = x2+y2.T-2*np.matmul(x,y.T)
        if x.shape[0] < 60000:
            sqdist = _sqdist(x, y, Torch) / a / a
        else:
            M = int(x.shape[0] / 400)
            sqdist = np.vstack([_sqdist(x[i:i + M], y, Torch) for i in range(0, x.shape[0], M)]) / a / a
        # elements can be negative due to float errors
        out = torch.exp(-sqdist / 2) if Torch else np.exp(-sqdist / 2)
        return out * b * b

    def rbf2(x, y, a, b, Torch=Torch):
        if y is None:
            y = x
        x, y = x / a, y / a
        return b * b * np.exp(-_sqdist(x, y) / 2)

    def mix_rbf(x, y, a, b, Torch=False):
        res = 0
        for i in range(len(a)):
            res += rbf(x, y, a[i], b[i], Torch)
        return res

    def laplace(x, a):
        return 0

    def quad(x, y, a, b):
        x, y = x / a, y / a
        x2, y2 = torch.sum(x * x, dim=1, keepdim=True), torch.sum(y * y, dim=1, keepdim=True)
        sqdist = x2 + y2.T - 2 * x @ y.T
        out = (sqdist + 1) ** (-b)
        return out

    def exp_sin_squared(x, y, a, b, c):
        if y is None:
            y = x
        diffs = np.expand_dims(x, 1) - np.expand_dims(y, 0)
        sqdist = np.sum(diffs ** 2, axis=2)
        assert np.all(sqdist >= 0), sqdist[sqdist < 0]
        out = b * b * np.exp(-np.sin(sqdist / c * np.pi) ** 2 / a ** 2 * 2)
        return out

    # return the kernel function
    assert isinstance(name, str), 'name should be a string'
    kernel_dict = {'rbf': rbf, 'poly': poly, 'quad': quad, 'mix_rbf': mix_rbf, 'exp_sin_squared': exp_sin_squared,
                   'rbf2': rbf2}
    return kernel_dict[name]