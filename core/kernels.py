import scipy

import sklearn.gaussian_process.kernels
from sklearn.gaussian_process.kernels import Kernel
import numpy as np
from scipy import sparse

import core.math
from core import utils

asa = np.asarray
asm = np.asmatrix

class Sparseable_kernel(Kernel):
    '''
    Writing my own function that implements sparse kernels, the idea is to also rely heavily on other libraries
    '''


class WhiteKernel(sklearn.gaussian_process.kernels.WhiteKernel, Sparseable_kernel):
    def __init__(self, noise_level=1.0, noise_level_bounds='fixed'):
        self.noise_level = noise_level
        self.noise_level_bounds = noise_level_bounds

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            K = self.noise_level * np.eye(X.shape[0])
            if eval_gradient:
                if self.hyperparameter_noise_level.fixed:
                    return K, np.empty((X.shape[0], X.shape[0], 0))
                return K, np.ones((X.shape[0], X.shape[0], 1))
            return K
        else:
            if eval_gradient:
                if self.hyperparameter_noise_level.fixed:
                    return np.zeros((X.shape[0], Y.shape[0])), np.empty((X.shape[0], Y.shape[0], 0))
                return np.zeros((X.shape[0], Y.shape[0])), np.ones((X.shape[0], Y.shape[0], 1))
            return np.zeros((X.shape[0], Y.shape[0]))


class RBF(sklearn.gaussian_process.kernels.RBF, Sparseable_kernel):
    def __call__(self, X, Y=None, eval_gradient=False):
        length_scale = sklearn.gaussian_process.kernels._check_length_scale(X, self.length_scale)
        if Y is not None:
            dists = sklearn.metrics.pairwise.euclidean_distances(X / self.length_scale,
                                                                 Y / self.length_scale,
                                                                 squared=True)
        else:
            dists = sklearn.metrics.pairwise.euclidean_distances(X / self.length_scale,
                                                                 squared=True)
        K = np.exp(-.5 * dists)
        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = \
                    (dists / self.length_scale * K)[:, :, np.newaxis]
                return K, K_gradient
            else:
                raise NotImplemented
        else:
            return K


class DotProduct(sklearn.gaussian_process.kernels.DotProduct, Sparseable_kernel):
    def __init__(self, sigma_0=0, sigma_0_bounds='fixed'):
        super().__init__(sigma_0=sigma_0, sigma_0_bounds=sigma_0_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            K = X.dot(X.T)
        else:
            K = X.dot(Y.T)
        if scipy.sparse.issparse(K):
            K = asa(K.todense())
        if eval_gradient:
            if not self.hyperparameter_sigma_0.fixed:
                gradient = np.empty((K.shape[0], K.shape[1], 1))
                gradient[..., 0] = 2 * self.sigma_0 ** 2
            else:
                gradient = np.empty((K.shape[0], K.shape[1], 0))

                return K, gradient
        else:
            return K


class ConstantKernel(sklearn.gaussian_process.kernels.ConstantKernel, Sparseable_kernel):

    def __init__(self, constant_value=1.0, constant_value_bounds=(1e-5, 1e4)):
        super().__init__(constant_value, constant_value_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        K = np.full((X.shape[0], Y.shape[0]), self.constant_value,
                    dtype=np.array(self.constant_value).dtype)
        if eval_gradient:
            if not self.hyperparameter_constant_value.fixed:
                return K, np.full((X.shape[0], Y.shape[0], 1), 1.)
            else:
                return K, np.empty((X.shape[0], Y.shape[0], 0))
        else:
            return K


class Kernel_Handler():
    def __init__(self, lin=None, rbf=None, length_scale=0, white=None, fixed_white=True,
                 lin_side=None, rbf_side=None, length_scale_side=0, **adam_args):
        self.fixed_white = fixed_white
        self.params = {'lin': lin, 'rbf': rbf, 'length_scale': length_scale, 'white': white}
        keys = sorted(self.params.keys())
        for key in keys:
            if self.params[key] is None:
                self.params.pop(key)
        self.param_names = []
        self.side_param_names = []
        self.ADAM_opt = ADAM(**adam_args)
        self.kernel = self.make_kernel(white, rbf, length_scale, lin, fixed_white=fixed_white,
                                       param_names=self.param_names)
        self.side_kernel = None
        side_params = {'lin': lin_side, 'rbf': rbf_side, 'length_scale': length_scale_side}
        if any(side_params[side] is not None for side in side_params):
            self.side_kernel = self.make_kernel(None, rbf_side, length_scale_side, lin_side, fixed_white=True,
                                                param_names=self.side_param_names)
            for name in self.side_param_names:
                self.params[name + '_side'] = side_params[name]

    def make_kernel(self, white, rbf, length_scale, lin, param_names, fixed_white=True):
        kernel = WhiteKernel(np.exp(white) if white is not None else 0,
                             noise_level_bounds='fixed' if fixed_white else (1E-3, 1E3))
        if not fixed_white and white is not None:
            param_names += ['white']
        if rbf is not None:
            kernel += ConstantKernel(np.exp(rbf)) * RBF(np.exp(length_scale))
            param_names += ['rbf', 'length_scale']
        if lin is not None:
            kernel += ConstantKernel(np.exp(lin)) * DotProduct(sigma_0_bounds='fixed')
            param_names += ['lin']
        return kernel

    def to_array(self, X,Y,side_X, side_Y):
        if not sparse.issparse(X):
            X = asa(X)
            if Y is not None: Y=asa(Y)
            if side_X is not None: side_X = asa(side_X)
            if side_Y is not None: side_Y = asa(side_Y)
        return X, Y, side_X, side_Y

    def __call__(self, X, Y=None, side_X=None, side_Y=None):
        X, Y, side_X, side_Y = self.to_array(X, Y, side_X, side_Y)
        if self.side_kernel is None or side_X is None:
            return asm(self.kernel(X, Y))
        return asm(self.kernel(X, Y)) + asm(self.side_kernel(side_X, side_Y))

    def gradients(self, X, Y=None, side_X=None, side_Y=None):
        X, Y, side_X, side_Y = self.to_array(X, Y, side_X, side_Y)
        grads = self.get_kernel_gradients(self.kernel, self.param_names, X, Y)
        if not self.side_kernel is None and not side_X is None:
            side_grads = self.get_kernel_gradients(self.side_kernel, self.side_param_names, side_X, side_Y, app='_side')
            for key in side_grads:
                grads[key + '_side'] = side_grads[key]

        return grads

    def get_kernel_gradients(self, kernel, param_names, X, Y=None, app=''):
        gradients = kernel(X, Y, eval_gradient=True)[-1]
        own_param_names = [hp for hp in kernel.hyperparameters if not hp.fixed]
        all_params = kernel.get_params()
        updateable_hps = [all_params[key[0]] for key in own_param_names]
        return {param_names[i]: np.exp(self.params[param_names[i] + app]) * gradients[..., i]
                for i, key in enumerate(updateable_hps)}

    def get_hyperparam_names(self):
        return self.param_names

    def set_hyperparam(self, name, value):
        # Sets the hyperparameter to the exponential of the given value
        assert name in self.param_names or ('_side' == name[-5:] and name[:-5] in self.side_param_names)

        # Getting the hyperparameter in the Kernel
        if '_side' in name:
            # Getting the index of the hyperparameter
            index = self.side_param_names.index(name[:-5])
            kernel = self.side_kernel
        else:
            # Getting the index of the hyperparameter
            index = self.param_names.index(name)
            kernel = self.kernel
        own_param_names = [hp for hp in kernel.hyperparameters if not hp.fixed]
        kernel.set_params(**{own_param_names[index][0]: np.exp(value)})

    def param_update(self, derivations):
        for key in derivations:
            if self.params[key] is not None:
                self.params[key] = self.params[key] + self.ADAM_opt.update(derivations[key], key)
                self.set_hyperparam(key, self.params[key])
        return self.params


class ADAM():
    def __init__(self, a=0.1, b1=0.9, b2=0.999, eps=10e-8):
        self.a = a
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.m_t = {}
        self.v_t = {}
        self.t = {}

    def update(self, g_t, key):
        # resize biased moment estimates if first iteration
        if key not in self.m_t:
            self.m_t[key] = np.zeros_like(g_t)
            self.v_t[key] = np.zeros_like(g_t)
            self.t[key] = 0

        # update timestep
        self.t[key] += 1
        # Account for the number of different parameters that are updated
        t = self.t[key]
        # update biased first moment estimate
        self.m_t[key] = self.b1 * self.m_t[key] + (1. - self.b1) * g_t

        # update biased second raw moment estimate
        self.v_t[key] = self.b2 * self.v_t[key] + (1. - self.b2) * (g_t ** 2)

        # compute bias corrected first moment estimate
        mhat_t = self.m_t[key] / (1. - self.b1 ** t)

        # compute bias corrected second raw moment estimate
        vhat_t = self.v_t[key] / (1. - self.b2 ** t)

        # apply update
        return self.a * mhat_t / (np.sqrt(vhat_t) + self.eps)


class Side_info_kernel_wrapper():
    """
    Wrapper function that calculates calls the kernel but on side info -> gets the data from the indices
    """

    def __init__(self, data, kernel_init_pars):
        self.kernel = Kernel_Handler(**kernel_init_pars)
        self.data = data
        self.sh = core.math.SparseHandler(sparse.isspmatrix(data))

    def __call__(self, batch_indices):
        batch = self.sh.take(self.data, batch_indices)
        return self.kernel(batch)

    def param_update(self, *args, **kwargs):
        return self.kernel(*args, **kwargs)

    def set_params(self, **kwargs):
        return self.kernel(**kwargs)

    def set_hyperparam(self, *args):
        return self.kernel.set_hyperparam(*args)

    def gradients(self, *args, **kwargs):
        return self.kernel(*args, **kwargs)

    def get_hyperparam_names(self):
        return self.kernel.get_hyperparam_names()


if __name__ == '__main__':
    X = np.random.uniform(-1, 1, (3, 5))
    sk_kern = sklearn.gaussian_process.kernels.RBF()
    my_kern = RBF()
    index = 1
    print(sk_kern(X, eval_gradient=True)[index] == my_kern(X, eval_gradient=True)[index])
    print()
    print(sk_kern(X, eval_gradient=True)[index] - my_kern(X, eval_gradient=True)[index])
