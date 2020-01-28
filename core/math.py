import collections
import random
from pathlib import Path

import numpy as np
import scipy
from numpy.linalg import multi_dot
from scipy import sparse as sp, integrate as integrate, interpolate
from scipy.sparse import linalg as sparselinalg
from scipy.special._ufuncs import kv as mod_bessel_2kind
from scipy.stats import norm
from sklearn import cluster
from sklearn.metrics import euclidean_distances


def get_diag_matrix(X, Y):
    """
    Naive implementation for the indicator function I guess
    :param X: input matrix DxN
    :param Y: input matrix DxM
    :return: Matrix [NxM]
    """
    out_shape = (np.shape(X)[1], np.shape(Y)[1])
    res = np.zeros(out_shape)
    Xt = X.T
    Yt = Y.T
    for x_i, x in enumerate(Xt):
        for y_i, y in enumerate(Yt):
            if (x == y).all():
                res[x_i, y_i] = 1.
    return res


def KL_div(variational_params, prior_params, sparse_handler):
    if variational_params['type'] == 'None':
        return 0
    d = len(variational_params['mean'])
    mean_diff = prior_params['mean'] - variational_params['mean']
    if variational_params['type'] == 'MV Gaussian':
        res = (- d
               + sparse_handler.trace_dot(prior_params['varinv'], variational_params['var'])
               + sparse_handler.to_dense(
                    sparse_handler.multi_dot([mean_diff.T, prior_params['varinv'], mean_diff]))
               + sparse_handler.logdet(prior_params['var']) - sparse_handler.logdet(
                    variational_params['var']))
        return float(0.5 * res)
    if variational_params['type'] == 'MF Gaussian':
        res = (- d
               + np.sum(np.multiply(prior_params['varinv'], variational_params['var']))
               + np.sum(np.multiply(mean_diff ** 2, prior_params['varinv']))
               - np.sum(np.log(variational_params['var'])))
        if isinstance(prior_params['var'], collections.Iterable):
            res += np.sum(np.log(prior_params['var']))
        else:
            res += d * np.log(prior_params['var'])
        return float(.5 * res)
    if 'Laplace MAP' in variational_params['type']:
        return np.sum(np.abs(mean_diff)) / prior_params['var'] + len(mean_diff) * np.log(2 * prior_params['var'])
    if variational_params['type'] == 'Gaussian MAP':
        # Because the variance of the prior is a diagonal matrix with the same entries,
        #   logdet(Var) is d * log(entry), etc.
        return sparse_handler.to_dense(.5 *
                                       (d * np.log(2 * np.pi)
                                        + d * np.log(prior_params['var'])
                                        + sparse_handler.multi_dot([mean_diff.T,
                                                                    1 / prior_params['var'] * sparse_handler.eye(d),
                                                                    mean_diff])))
    if 'Laplace' in variational_params['type']:
        # constants
        res = (2 * np.log(8) + 1) * d

        # E_q [log q(beta)]
        if 'MV' in variational_params['type']:
            res += sparse_handler.logdet(variational_params['var'])
        else:
            res += np.sum(np.log(variational_params['var']))
        # E_q [log q(lambda) - log(p(beta, lambda))]
        res += np.sum(.5 * np.log(variational_params['a'])
                      + 2 * np.log(mod_bessel_2kind(-.5, np.sqrt(variational_params['a']))))

        return - .5 * res
    if 'Horseshoe' in variational_params['type']:
        if 'MV' in variational_params['type']:
            res = sparse_handler.logdet(variational_params['var'])
        else:
            res = np.sum(np.log(variational_params['var']))
        res = 0.5 * res + np.sum(np.log(np.clip(variational_params['norm'], 1E-10, np.infty)))
        return - res
    return 0


def add_varinv(dist, inv_):
    if 'var' in dist:
        dist['varinv'] = inv_(dist['var'])


def gaussian(x, m, v):
    return norm.pdf(x, loc=m, scale=v)


def soft_thresh(m_b, lambda_):
    return np.sign(m_b) * np.minimum(np.abs(m_b), lambda_ * np.ones_like(m_b))


def soft_thresh_op(m_b, lambda_):
    if not np.shape(lambda_) == np.shape(m_b):
        lambda_ * np.ones_like(m_b)
    return np.where(np.abs(m_b) > lambda_, m_b - np.sign(m_b) * lambda_, np.zeros_like(m_b))


def identity(x):
    return x


def logit(x):
    return 1 / (1 + np.exp(-np.clip(x, -1E2, 1E2)))


def get_auto_length_scale(X):
    """
    Returns the log of the median euclidean distance between the datapoints.
    Because for big datasets this leads to a memory error,
    a maximum number of datapoints is randomly sampled from the dataset
    :param X: Datapoints sparse or normal numpy matrix[NxD]
    :return:
    """
    if X.shape[1] > 500:
        if sp.isspmatrix(X):
            X_ = scipy.sparse.hstack([X[:, i] for i in random.sample(range(X.shape[1]), 500)])
        else:
            X_ = np.take(X, random.sample(range(X.shape[1]), 500), axis=1)
    else:
        X_ = X
    return np.log(np.median(euclidean_distances(X_.T, X_.T)))


def make_dists(learner, qw_type):
    start_var = 1E-3
    dists = {'qu': {'type': 'MV Gaussian', 'mean': np.zeros((learner.m, 1)), 'var': start_var * np.eye(learner.m)},
             'pu': {'type': 'MV Gaussian', 'mean': np.zeros((learner.m, 1)), 'var': None,
                    'varinv': None}}

    if qw_type in ['Laplace MAP', 'Gaussian MAP', 'MF Gaussian', 'Laplace MAP iterative']:
        if 'MAP' in qw_type:
            dists['qw'] = {'type': qw_type, 'mean': np.zeros((learner.d, 1))}
            dists['pw'] = {'type': qw_type, 'mean': np.zeros((learner.d, 1)),
                           'var': learner.lambda_w}
        else:
            dists['qw'] = {'type': qw_type, 'mean': np.zeros((learner.d, 1)),
                           'var': learner.lambda_w * np.ones((learner.d,)),
                           'varinv': 1 / learner.lambda_w * np.ones((learner.d,))}
            dists['pw'] = {'type': qw_type, 'mean': np.zeros((learner.d, 1)),
                           'var': learner.lambda_w,
                           'varinv': 1 / learner.lambda_w}

    elif any(type_ in qw_type for type_ in ['Laplace', 'Horseshoe']):

        dists['qw'] = {'type': qw_type, 'mean': np.zeros((learner.d, 1)), 'tau': learner.lambda_w,
                       'var': start_var * np.ones((learner.d,))}
        dists['pw'] = {'type': qw_type, 'mean': np.zeros((learner.d, 1)), 'tau': learner.lambda_w}
        if 'Laplace' in qw_type:
            dists['qw']['a'] = (dists['qw']['mean'] ** 2 + np.ones_like(dists['qw']['mean']) * start_var) / dists['qw'][
                'tau'] ** 2
            dists['qw']['lambda'] = 1 / np.sqrt(dists['qw']['a'])
            dists['pw']['var'] = dists['qw']['tau'] ** 2 / dists['qw']['lambda']
            dists['pw']['varinv'] = 1 / dists['pw']['var']
        else:
            b = (dists['qw']['mean'] ** 2 + + np.ones_like(dists['qw']['mean']) * start_var) / dists['qw']['tau'] ** 2
            if 'MAP' in qw_type:
                dists['qw']['lambda'] = np.sqrt((b - 1 + np.sqrt(b ** 2 + 10 * b + 1)) / 6)
                dists['pw']['var'] = (dists['qw']['tau'] * dists['qw']['lambda']) ** 2
                dists['pw']['varinv'] = 1 / dists['pw']['var']
            else:
                expctd_gamma, norm = get_HS_lookup()
                dists['qw']['expected gamma'] = expctd_gamma
                dists['qw']['gamma norm'] = norm
                dists['qw']['b'] = b
                dists['qw']['norm'] = dists['qw']['gamma norm'](dists['qw']['b'])
                dists['pw']['varinv'] = expctd_gamma(b) / (dists['qw']['tau'] ** 2)
                dists['pw']['var'] = 1 / dists['pw']['varinv']

        if 'MV' in qw_type:
            dists['qw']['var'] = np.diag(dists['qw']['var'])

    elif qw_type == 'MV Gaussian':
        dists['qw'] = {'type': qw_type, 'mean': learner.sh.zeros((learner.d, 1)),
                       'var': learner.lambda_w * learner.sh.eye(learner.d),
                       'varinv': 1 / learner.lambda_w * learner.sh.eye(learner.d)}
        dists['pw'] = {'type': qw_type, 'mean': learner.sh.zeros((learner.d, 1)),
                       'var': learner.lambda_w * learner.sh.eye(learner.d),
                       'varinv': 1 / learner.lambda_w * learner.sh.eye(learner.d)}
    elif qw_type == 'None':
        dists['qw'] = {'type': qw_type}
        dists['pw'] = {'type': qw_type}
    else:
        print(qw_type)
        raise NotImplemented
    return dists


def get_HS_lookup():
    def unnorm_q_gamma(x, b):
        return 1 / (1 + x) * np.exp(-.5 * b * x)
    def exp_gamma(b):
        res = integrate.quad(lambda x: x * unnorm_q_gamma(x, b), 0, np.inf)
        return res[0]
    def log_normalizer(b):
        res = integrate.quad(lambda x: unnorm_q_gamma(x, b), 0, np.inf)
        return res[0]
    lut_fn = "HS_lookup"
    my_file = Path(lut_fn + '.npy')
    if my_file.is_file() and not debug():
        print('loading Saved Horseshoe lookup')
        expctd_gamma, norm = np.load(lut_fn + '.npy')
    else:
        print('Calculating Horseshoe lookup')
        look_up_table_x = np.geomspace(5E-5, 100000, 5000)
        prob_normalizer = np.asarray([log_normalizer(x) for x in look_up_table_x])
        prob = np.asarray([exp_gamma(x) for x in look_up_table_x]) \
               / prob_normalizer
        expctd_gamma = interpolate.interp1d(look_up_table_x, prob, fill_value='extrapolate', assume_sorted=True)
        norm = interpolate.interp1d(look_up_table_x, prob_normalizer, fill_value='extrapolate', assume_sorted=True)
        np.save(lut_fn, [expctd_gamma, norm])
    return expctd_gamma, norm


def logcosh(x):
    # log(cosh(x)) = log(1/2(e^x + e^{-x}))
    if abs(x) < 10:
        return np.log(np.cosh(x))
    return np.abs(x) - log2


class SparseHandler():
    '''
    This class is a wrapper that chooses the adequate functions for either sparse or non sparse matrices
    '''

    def __init__(self, sparse):
        self.sparse = sparse

    def eye(self, d):
        if self.sparse:
            return sp.eye(d, format='dia')
        else:
            return np.eye(d)

    def zeros(self, shape):
        if self.sparse:
            return sp.csc_matrix(shape)
        else:
            return np.zeros(shape)

    def inv(self, x):
        if self.sparse:
            if sp.isspmatrix_dia(x) and x.offsets == [0]:
                return sp.diags(1 / (x.diagonal()))
            else:
                return sparselinalg.inv(x)
        else:
            return np.linalg.inv(x)

    def multi_dot(self, param):
        if any(sp.issparse(x) for x in param):
            res = param[0]
            if not sp.issparse(res):
                res = sp.csc_matrix(res)
            # TODO maybe implement a faster function for this, supposedly sparse matrix multiplication already is quite fast, dunno
            for i in range(1, len(param)):
                res = sp.csc_matrix(res.dot(param[i]))
            return res
        else:
            return multi_dot(param)

    def take(self, X, indices):
        if sp.issparse(X):
            return sp.hstack([X[:, i] for i in indices], format='csc')
        else:
            return np.take(X, indices, axis=1)

    def get_current_batch(self, learner):
        X = self.take(learner.X, learner.current_batch)
        if learner.use_side_info:
            side_info = self.take(learner.side_info, learner.current_batch)
        else:
            side_info = None
        Y = np.expand_dims(np.take(learner.Y, learner.current_batch), axis=-1)
        return X, Y, side_info

    def join_matrices(self, A, B, axis):
        if B is None:
            return A
        if self.sparse:
            if axis == 0:
                return sp.vstack([A, B], format='csc')
            elif axis == 1:
                return sp.hstack([A, B], format='csc')
            else:
                raise NotImplemented
        else:
            return np.concatenate([A, B], axis=axis)

    def zeros_like(self, x):
        if sp.issparse(x):
            return sp.csc_matrix((x.shape))
        else:
            return np.zeros_like(x)

    def dot(self, a, b):
        return a.dot(b)

    def diag(self, param):
        if self.sparse:
            if np.ndim(param) > 1 and param.shape[-1] == 1:
                param = param[:, 0]
            if np.ndim(param) == 1:
                return sp.diags(param, format='csc')
            else:
                return np.array(param.diagonal())
        else:
            if len(np.shape(param)) == 2:
                if not isinstance(param, list) and param.size > 1 and param.shape[-1] == 1:
                    param = param[:, 0]
            return np.diag(param)

    def enforce_diagonal(self, matrix):
        if self.sparse:
            return sp.spdiags(matrix.diagonal(), [0], matrix.shape[0], matrix.shape[1])
        else:
            return np.diag(np.diag(matrix))

    def trace(self, mat):
        if self.sparse:
            return np.sum(mat.diagonal())
        else:
            return np.trace(mat)

    def trace_dot(self, A, B):
        if sp.issparse(A) or sp.issparse(B):
            return self.trace(self.dot(A, B))
        return np.einsum('ij,ji->', A, B)

    def set_inducing_points(self, X, m):
        if np.prod(X.shape) > 1E10:
            # When there are more than 10 Billion entries in the data, we won't try to use kmeans clustering
            # In that case the matrix is guaranteed to be sparse
            print('Randomly sampling inducing points')
            ind_entries = random.sample(range(X.shape[1]), m)
            Z = sp.hstack([X[:, i] for i in ind_entries], format='csc')

        else:
            Z = get_inducing_points(X, m)
            if sp.issparse(X):
                Z = sp.csc_matrix(Z)
            else:
                Z = np.asmatrix(Z)
        return Z

    def add_varinvs(self, dists):
        for name in dists:
            if name[0] == 'q':
                if 'var' in dists[name]:
                    if sp.issparse(dists[name]['var']):
                        if sp.isspmatrix_dia(dists[name]['var']):
                            return sp.dia_matrix(1 / dists[name]['var'].diagonal())
                        else:
                            return add_varinv(dists[name], sparselinalg.inv)
                else:
                    add_varinv(dists[name], np.linalg.inv)

    def to_dense(self, x):
        if np.shape(x) == (1, 1):
            return self.to_dense(x[0, 0])
        if sp.issparse(x):
            return np.asarray(x.todense())
        else:
            return x

    def logdet(self, x):
        if sp.issparse(x):
            if sp.isspmatrix_dia(x) and x.offsets == [0]:
                res = sum(np.log(x.diagonal()))
            else:
                print('Not implemented')
                res = sum(np.log(x.diagonal()))
                # factor = cholesky(x)
                # res = factor.logdet()
            return res
        else:
            return np.linalg.slogdet(x)[1]


    def woodbury_diag_A(self, A, B, C):
        """
        Implementation of the woodbury identity.
        :param A: scalar in this implementation only a * I is used
        :param B: Matrix [YxY]
        :param C: Matrix [ZxY]
        :return: (A + C B C^T)^{-1} [ZxZ]
        """

        if not isinstance(A, collections.Iterable):
            A_inv = self.eye(C.shape[0]) * 1 / A
            A = self.eye(C.shape[0]) * A
        elif A.ndim == 1 or A.shape[-1] == 1:
            A_inv = self.diag(1 / A)
            A = self.diag(A)
        else:
            A_inv = self.inv(A)
        return self.woodbury_invA(A_inv, B, C, A)

    def woodbury_invA(self, A_inv, B, C, A):
        """
        Implementation of the woodbury identity.
        :param A_inv: Inverse of A
        :param B: Matrix [YxY]
        :param C: Matrix [ZxY]
        :param A: A
        :return: (A + C B C^T)^{-1} [ZxZ]
        """
        if C.shape[0] < C.shape[1]:
            # If Z < Y, then it is not useful to do use the woodbury Identity
            return self.inv(A + self.multi_dot([C, B, C.T]))

        ctainvc = C.T.dot(A_inv.dot(C))
        res = C.dot(self.inv(self.inv(B) + ctainvc))
        res = res.dot(C.T)
        res = A_inv - A_inv.dot(res).dot(A_inv)
        # return A_inv * self.eye(C.shape[0]) - self.multi_dot([A_inv * C, self.inv(self.inv(B) + ctainvc)), A_inv * C.T])
        return res

    def multiply(self, mat, other):
        """

        :param mat: Matrix, sparse or not
        :param other: other element to multiply point-wise with (this matrix is not checked for sparsity)
        :return:
        """
        if sp.issparse(mat):
            if len(other) == mat.shape[-1]:
                return self.dot(mat, sp.csc_matrix((other, (np.array(range(len(other))), np.array(range(len(other)))))))
            else:
                return mat.multiply(other)
        return np.multiply(mat, other)

    def el_square(self, X):
        if sp.issparse(X):
            return X.power(2)
        else:
            return np.square(X)


log2 = np.log(2)


def debug(): return False


def get_inducing_points(X, m):
    """
    :param X: Data matrix DxN
    :param m: Number of inducing points
    :return: Inducing Points Dxm
    """
    kmeans = cluster.KMeans(n_clusters=m, random_state=0, n_jobs=-1, max_iter=20).fit(X.T)
    return kmeans.cluster_centers_.T
