import datetime
import os
import pickle
import random
import warnings

import numpy as np
import pandas as pd
from numpy.linalg import multi_dot
from scipy import integrate, sparse

import core.math
from core import kernels, utils
from core import plotting


class SVIinstance():
    def __init__(self, data_set, n_ind_points=49, lin_kernel=0, rbf_kernel=0, length_scale='auto', pw_var=5E-2,
                 v_coor_asc_lr=1, v_coor_asc_factor=1,
                 qw_type='Laplace MF', batch_size=500, white_kernel=0.1, hyperparameter_lr=.1,
                 epoch_hsteps=2, epoch_vsteps=3,
                 log_folder=None, epoch_lim=100, side_kernel_params=None, map_pred=False):
        """

        :param data_set: Dict with data set
        :param n_ind_points: number of inducing points
        :param lin_kernel: variance linear kernel
        :param rbf_kernel: variance rbf kernel
        :param length_scale: length scale of rbf kernel, when setting to 'auto' it is inferred automatically
        :param white_kernel: variance white (diagonal) kernel
        :param v_coor_asc_lr: learning rate for coordinate ascent, determines decay for use of mini-batches
        :param v_coor_asc_factor: constant factor for step size (equal for both mini- and full batches)
        :param batch_size: Batch size
        :param pw_var: variance of prior on weights
        :param qw_type: Distribution type for prior of weights
        :param hyperparameter_lr: Learning rate for hyperparameter learning rate (ADAM)
        :param epoch_hsteps: Number of hyperparameter optimization steps in one epoch
        :param epoch_vsteps: Number of optimization steps for variational parameters
        :param log_folder: Folder path to where run-information will be saved to
        :param epoch_lim: Maximum number of epochs
        :param side_kernel_params: Parameters for side kernel (only used when using dataset with side_information)
        :param map_pred: Boolean for predicting by marginalization (if set to False)
                or MAP (if set to true (this is faster))
        """
        print(
            'Init learner: {} on {} with s:{}, #ind:{} and epoch lim: {} data-dim:{}'.format(qw_type, data_set['name'],
                                                                                             batch_size, n_ind_points,
                                                                                             epoch_lim,
                                                                                             data_set['train'][
                                                                                                 0].shape))

        self.init_data(data_set)
        self.m = n_ind_points
        self.d, self.n = np.shape(self.X)
        if self.d > 5000 and 'MV' in qw_type:
            warnings.warn('Using a Multivariate Distribution for the weights with {} dimensions can lead '
                          'to Memory Errors and is very inefficient'.format(self.d),
                          RuntimeWarning)
        self.s = min(batch_size, self.n)
        self.local_params = None

        if n_ind_points >= self.n:
            self.Z = self.X.copy()
            if self.use_side_info:
                self.Z_side = self.side_info
            self.m = self.n
            print('Not using inducing points')
        else:
            print('finding inducing points')
            if self.use_side_info:
                self.Z = self.sh.set_inducing_points(self.sh.join_matrices(self.X, self.side_info, axis=0), self.m)
                self.Z_side = self.Z[self.d:, :]
                self.Z = self.Z[:self.d, :]
            else:
                self.Z = self.sh.set_inducing_points(self.X, self.m)

        # Automatically decide on the length scale
        if length_scale == 'auto':
            print('Automatically finding ls')
            length_scale = core.math.get_auto_length_scale(self.X)
            print('Automatic length_scale set to {}'.format(np.exp(length_scale)))
        self.kernel_init_pars = {'lin': lin_kernel, 'rbf': rbf_kernel, 'length_scale': length_scale,
                                 'white': white_kernel}

        if self.use_side_info:
            # Adding possible kernels for side
            poss_keys = ['lin_side', 'rbf_side', 'length_scale_side']
            for key in poss_keys:
                self.kernel_init_pars[key] = None
            self.kernel_init_pars['length_scale_side'] = 0
            if side_kernel_params is not None:
                for key in poss_keys:
                    if key in side_kernel_params:
                        self.kernel_init_pars[key] = side_kernel_params[key]
            else:
                self.kernel_init_pars['lin_side'] = 0
        self.kernel_params = self.kernel_init_pars.copy()
        self.kernel_init_pars['a'] = hyperparameter_lr
        self.kernel = kernels.Kernel_Handler(**self.kernel_init_pars)
        self.kp_max, self.kp_min = 100, -100
        self.max_w = 50

        self.lambda_w = pw_var

        print('Initializing dist params')
        self.dist_params = core.math.make_dists(self, qw_type)
        print('Updating kernel')
        self.update_kernel()

        print('Initializing rest of variables')
        self.v_coor_asc_lr = v_coor_asc_lr if self.use_batches() else 1
        self.v_coor_asc_step_size = self.v_coor_asc_lr
        self.v_coor_asc_factor = v_coor_asc_factor
        self.link_func = core.math.identity

        # Variables for mini-batches
        self.current_batch = None
        self.batch_counter = 0

        # Variables for hyperparamter optimization
        self.hyperparameter_lr = hyperparameter_lr

        self.kernel_param_history = {}
        self.kernel_param_log()

        # Practical Implementation stuff (possibly debug)
        self.calculated_vars = {}  # Here Constants/Variables that are possibly used again are saved
        self.log_folder = log_folder
        self.elbos = []
        self.step_types = []
        self.mchn_eps = 5E-5
        self.freq_track_elbos = True
        self.map_pred = map_pred

        # Epoch defining variables
        self.epoch_vsteps = epoch_vsteps  # the number of variational steps in one epoch
        self.epoch_hsteps = epoch_hsteps  # the number of hyperparameters steps in one epoch
        self.ls_head_start = 2  # number of epochs length_scale is optimized before optimizing the other hp's

        # Run variables
        self.epoch_count = 0
        self.conv_crit = 1E-6  # Termination criterium on ELBO/Loglikelihood
        self.epoch_lim = epoch_lim  # Termination criterium number of epochs
        self.done = False  # Boolean that decides whether terminated or not
        self.run_info = {'init': datetime.datetime.now(),
                         'data_set': {key: data_set[key] for key in data_set.keys() if key not in ['train', 'test']}}

    def init_data(self, data_set):
        """
        Initializes the data with a given dataset
        :param data_set: dict
        """
        self.X, self.Y = data_set['train']
        if not min(self.Y) == -1:
            # Ys can only be in [1, -1], not in [0,1]
            self.Y = np.where(self.Y > 0, 1, - 1)
        self.X_test, self.y_test = data_set['test']
        self.init_side_info(data_set)

        self.sh = core.math.SparseHandler(sparse.issparse(self.X))
        if not self.sh.sparse:
            self.X, self.X_test = np.asmatrix(self.X), np.asmatrix(self.X_test)

    def init_side_info(self, data_set):
        '''
        Initializes the side information
        :param data_set: dict
        '''
        # Side Info
        if 'side_info' in data_set:
            assert data_set['side_info']['train'].shape[-1] == self.X.shape[-1]
            self.side_d = data_set['side_info']['train'].shape[0]
            self.side_info_test = data_set['side_info']['test']
            self.side_info = data_set['side_info']['train']
        else:
            self.side_d = 0
            self.side_info_test = None
            self.side_info = None
        self.use_side_info = self.side_info is not None

    def update_kernel(self):
        """
        This function handles the ubiquitously used calculations of the kernels
        """
        if self.use_side_info:
            side_info_t = self.side_info.T
            Z_side_t = self.Z_side.T
        else:
            side_info_t = Z_side_t = None
        self.Kmm = self.kernel(self.Z.T, side_X=Z_side_t)
        self.Kmm_inv = np.linalg.inv(self.Kmm)

        # Upgrading the current distribution-parameters
        self.dist_params['pu']['var'] = self.Kmm
        self.dist_params['pu']['varinv'] = self.Kmm_inv
        if not self.use_batches():
            X = self.X
            self.Knn = self.kernel(X.T, side_X=side_info_t)
            self.Kmn = self.kernel(self.Z.T, X.T, side_X=Z_side_t, side_Y=side_info_t)
            self.Knm = self.Kmn.T
            self.K_tilde = self.Knn - multi_dot([self.Knm, self.Kmm_inv, self.Kmn])

            self.K_mn_prime = multi_dot([self.Kmm_inv, self.Kmn])

    def get_kmm(self):
        return self.Kmm

    def get_kmminv(self):
        return self.Kmm_inv

    def get_K_tilde(self, X, Kmn, side_info=None):
        if self.use_batches():
            if self.use_side_info:
                K_tilde = (self.kernel(X.T, side_X=side_info.T) - self.sh.multi_dot([Kmn.T, self.Kmm_inv, Kmn]))
            else:
                K_tilde = (self.kernel(X.T) - self.sh.multi_dot([Kmn.T, self.Kmm_inv, Kmn]))
        else:
            K_tilde = self.K_tilde
        return K_tilde

    def get_Kmn(self, X, side_info, force_new=False):
        if self.use_batches() or self.Kmn is None or force_new:
            if self.use_side_info:
                return self.kernel(self.Z.T, X.T, side_X=self.Z_side.T, side_Y=side_info.T)
            return self.kernel(self.Z.T, X.T)
        else:
            return self.Kmn

    def get_Kmn_prime_Kmn(self, X, side_info=None, force_new=False):
        if self.use_batches() or any(k is None for k in [self.Kmn, self.K_mn_prime]) or force_new:
            if self.Kmm is None:
                if side_info is not None:
                    self.Kmm_inv = self.sh.inv(self.kernel(self.Z.T, side_X=self.Z_side.T))
                else:
                    self.Kmm_inv = self.sh.inv(self.kernel(self.Z.T))
            Kmn = self.get_Kmn(X, side_info, force_new=force_new)
            K_mn_prime = np.dot(self.Kmm_inv, Kmn)
        else:
            Kmn = self.Kmn
            K_mn_prime = self.K_mn_prime
        return K_mn_prime, Kmn

    def get_Knn_approx(self, X=None, side_info=None):
        if X is None:
            X = self.X
            side_info = self.side_info
        K_mn_prime, Kmn = self.get_Kmn_prime_Kmn(X, side_info=side_info)
        return Kmn.T.dot(K_mn_prime)

    def variational_step(self):
        """
        Samples new batch and updates the current variational step size
        """
        self.batch_counter += 1
        batch = self.batch()

        if self.s < self.n:
            self.v_coor_asc_step_size = self.v_coor_asc_factor / (self.batch_counter ** self.v_coor_asc_lr)
        self.v_grad_ascent_step_size = 1. / (self.batch_counter ** self.v_coor_asc_lr)
        self.variational_update(batch)

        if core.math.debug():
            self.step_types.append(0)

    def epoch(self):
        """
        Goes through one loop of variational updates and hyperparameter updates
        It tracks the ELBO over time
        """
        if 'start' not in self.run_info:
            self.run_info['start'] = datetime.datetime.now()

        c_elbos = []
        print('Epoch', end='', flush=True)

        # In the first Epoch double the variational steps are taken, so that the whole system is more stable
        for _ in range(self.epoch_vsteps):
            self.variational_step()
            if self.freq_track_elbos: c_elbos.append(self.calc_ELBO())
            print('.', end='', flush=True)

        for i in range(self.epoch_hsteps):
            self.hyperparam_update()
            print('\'', end='', flush=True)
            if self.freq_track_elbos: c_elbos.append(self.calc_ELBO())

        if not self.freq_track_elbos:
            c_elbos.append(self.calc_ELBO())

        if len(self.elbos) > 0:
            self.done = self.done \
                        or abs(np.sum(c_elbos[-1]) - np.sum(self.elbos[-1])) < self.conv_crit \
                        or self.epoch_count >= self.epoch_lim
            self.run_info['end'] = datetime.datetime.now()
        self.epoch_count += 1
        self.elbos += c_elbos
        return c_elbos

    def batch(self, size=None):
        """
        This function samples a new batch
        :param size: optional: If you want a different batchsize, e.g. for testing the ELBO, default self.s
        :return: new batch [self.d x size]
        """
        if size is None:
            size = self.s
        else:
            size = min(size, self.n)

        if size < self.n or self.s < self.n:
            self.current_batch = random.sample(range(self.n), size)
        elif self.local_params is None:
            # If we want to re-use the Knm matrices etc. the dataset has to be ordered correctly
            self.current_batch = range(self.n)

        current_batch = self.get_current_batch()
        self.local_params = self.update_local_params(current_batch[0], current_batch[1],
                                                     side_info=current_batch[2])
        return current_batch

    def get_current_batch(self):
        return self.sh.get_current_batch(self)

    def variational_update(self, batch):
        for dist in sorted(self.dist_params):
            if dist in ['qu', 'qw']:
                self.variational_param_update(dist, batch)

    def coordinate_step(self, update):
        for name in update:
            if name[0] == 'q':
                for param in update[name]:
                    self.dist_params[name][param] = (1 - self.v_coor_asc_step_size) * self.dist_params[name][param] \
                                                    + self.v_coor_asc_step_size * update[name][param]

    def single_weight_coordinate_step(self, update, index):
        for name in update:
            assert name == 'qw'
            for param in update[name]:
                self.dist_params[name][param][index] = (1 - self.v_coor_asc_step_size) \
                                                       * self.dist_params[name][param][index] \
                                                       + self.v_coor_asc_step_size * update[name][param]

    def variational_param_update(self, var, batch):
        return {var: {}}

    def update_local_params(self, X, y, side_info=None, precalc_dict=None):
        return []

    def calc_ELBO(self, X=None, Y=None, side_info=None):
        self.sh.add_varinvs(self.dist_params)
        if X is None:
            X, Y, side_info = self.get_current_batch()
        return [self.calcL1(X, Y, side_info=side_info), - self.KL_omega()
            , - core.math.KL_div(self.dist_params['qu'], self.dist_params['pu'], self.sh)
            , - core.math.KL_div(self.dist_params['qw'], self.dist_params['pw'], self.sh)]

    def theta(self, c):
        return 1 / (2 * c) * np.tanh(c / 2)

    def KL_omega(self):
        # KL divergence omega
        cs = self.local_params
        theta = self.theta(cs)
        batch_factor = self.n / len(self.local_params)
        res = batch_factor * sum([core.math.logcosh(ci / 2) for ci in cs])
        res -= batch_factor * 0.5 * sum(
            [abs(theta[i]) * (ci ** 2) for i, ci in enumerate(cs)])
        return res

    def calcL1(self, X, Y, side_info=None):
        """
        Calculates the approximation to the ELBO without KL divergences
        :param X:
        :param Y:
        :return:
        """
        return 0

    def getmwSwmuSu(self):
        return [self.dist_params[d][p] for d in ['qw', 'qu'] for p in ['mean', 'var']]

    def __str__(self):
        return self.__class__.__name__

    def predict_MAP(self, X, side_info=None):
        '''
        predicts label with MAP point estimators, i.e. taking the only means
        :param X: input points [DxN]
        :return: confidence of being in class 1, [prediction based on only w, prediction based on only epsilon]
        '''
        if side_info is not None:
            K_cnm = self.kernel(X.T, self.Z.T, side_X=side_info.T, side_Y=self.Z_side.T)
        else:
            K_cnm = self.kernel(X.T, self.Z.T)
        eps_x = np.ravel(self.sh.to_dense(multi_dot([K_cnm, self.Kmm_inv, self.dist_params['qu']['mean']])))
        xw = np.ravel(self.sh.to_dense(self.sh.multi_dot([X.T, self.get_expected_w()])))
        conf = self.link_func(eps_x + xw)

        return conf, [self.link_func(xw), self.link_func(eps_x)]

    def predictive_distribution(self, X, w_MAP=False, use_eps=True, use_w=True, side_info=None):
        """
        Calculates the Normal Gaussian predictive distribution
        :param X: Points to be predicted [DxN]
        :param w_MAP: boolean that decides whether to use a MAP approximator for w or not
        :param use_w: Boolean for using w in the predictions
        :param use_eps: Boolean for using epsilon in the predictions
        :param side_info: Side information
        :return: mean, variance
        """
        mean = np.zeros((np.shape(X)[1], 1))
        var = np.zeros_like(mean)
        Xw = self.sh.to_dense(self.sh.dot(X.T, self.get_expected_w()))
        side_i = None
        sparse_ = sparse.issparse(self.Z)
        if not sparse_ and side_info is not None:
            side_info = np.asmatrix(side_info)
        for i in range(np.shape(X)[1]):
            x_i = X[:, i]
            if use_eps:
                if side_info is not None:
                    side_i = side_info[:, i]
                K_mi_prime, Kmi = self.get_Kmn_prime_Kmn(x_i, side_info=side_i, force_new=True)
                mean[i, 0] += self.sh.to_dense(np.dot(K_mi_prime.T, self.dist_params['qu']['mean']))
                cur_var = self.kernel(x_i.T, side_X=side_i.T if side_info is not None else None) \
                          + multi_dot([K_mi_prime.T, self.dist_params['qu']['var'] - self.Kmm,
                                       K_mi_prime])
                if cur_var < 0:
                    print('.... {} on sample {}'.format(cur_var, i))
                var[i, 0] += cur_var
            if use_w:
                mean[i, 0] += Xw[i]
                if not w_MAP and 'var' in self.dist_params['qw']:
                    if 'MF' in self.dist_params['qw']['type']:
                        var[i, 0] += np.sum(np.multiply(np.sum(self.sh.el_square(x_i), axis=1),
                                                        self.dist_params['qw']['var'][:, np.newaxis]))
                    else:
                        var[i, 0] += self.sh.to_dense(self.sh.multi_dot([x_i.T, self.dist_params['qw']['var'], x_i]))
        assert all(var > 0)
        return mean, var

    def predict(self, x, side_info=None, w_MAP=False, use_eps=True, use_w=True):
        '''
        Predicts the class
        :param x: point [DxN]
        :param w_MAP: w_MAP: boolean that decides whether to use a MAP approximate for w or not
        :param use_w: Boolean for using w in the predictions
        :param use_eps: Boolean for using epsilon in the predictions
        :return: confidences
        '''
        if w_MAP:
            full_conf, [w_conf, eps_conf] = self.predict_MAP(x, side_info=side_info)
            if not use_eps:
                return w_conf
            if not use_w:
                return eps_conf
            return full_conf
        mean, var = self.predictive_distribution(np.asmatrix(x), side_info=side_info, use_eps=use_eps, use_w=use_w)

        confs = []
        func = lambda x, m, v: self.link_func(x) * core.math.gaussian(x, m, v)
        for i in range(x.shape[1]):
            c_conf = integrate.quad(func, mean[i, 0] - 10 * var[i], mean[i, 0] + 10 * var[i],
                                    args=(mean[i, 0], var[i]))[0]
            confs.append(c_conf)
        return np.asarray(confs)

    def precision(self, X=None, y=None, **kwargs):
        if X is None and y is None:
            X = self.X
            y = self.Y

        confs = self.predict(X, **kwargs)
        return utils.log_confidence(y, confs)

    def score(self, X=None, y=None, side_info=None, add_info=None, print_out=True):
        """
        Goes through the possible configurations and prints scores
        :return:
        """

        if y is None:
            X = self.X_test
            y = self.y_test
            side_info = self.side_info_test
        Y_ = np.ravel(y > 0)

        scores = {}
        qw_type = self.dist_params['qw']['type']
        # Marginalizing out weights and confounder
        fb_confs = self.predict(X, side_info=side_info, w_MAP=self.map_pred)
        scores[qw_type] = utils.score(Y_, fb_confs)
        fb_confs = self.predict(X, side_info=side_info, use_w=False, w_MAP=self.map_pred)
        scores['only GP ' + qw_type] = utils.score(Y_, fb_confs)
        if 'var' in self.dist_params['qw']:
            fb_confs = self.predict(X, side_info=side_info, use_eps=False, w_MAP=self.map_pred)
            scores['only w ' + qw_type] = utils.score(Y_, fb_confs)

        if add_info is not None and 'w_true' in add_info and add_info['w_true'][0] != 0:
            y_lin = np.ravel(np.sign(X.T.dot(add_info['w_true'])) > 0)
            add_info['lin_model_score'][qw_type] = utils.score(y_lin, fb_confs)
            fb_confs = self.predict(X, side_info=side_info, use_w=False, w_MAP=self.map_pred)
            add_info['lin_model_score'][qw_type + ' only GP'] = utils.score(y_lin, fb_confs)
            if 'var' in self.dist_params['qw']:
                fb_confs = self.predict(X, side_info=side_info, use_eps=False, w_MAP=self.map_pred)
                add_info['lin_model_score'][qw_type + ' only w'] = utils.score(y_lin, fb_confs)
        if add_info is not None and side_info is not None:
            fb_confs = self.predict(X, side_info=self.sh.zeros(side_info.shape), use_w=False, w_MAP=self.map_pred)
            add_info['zero Side'][qw_type] = utils.score(Y_, fb_confs)

        self.run_info['scores'] = pd.DataFrame(scores)
        if print_out: print(self.run_info['scores'].to_string())
        return self.run_info['scores']

    def use_batches(self):
        return not self.s == self.n

    def hyperparam_update(self):
        """
        Calls the implementations of hyperparameter derivations and updates the hyperparameters
        :return:
        """
        derivations = self.hyperparam_gradients(self.batch())
        self.kernel_params = self.kernel.param_update(derivations)
        if len(derivations) > 0:
            self.update_kernel()
            self.kernel_param_log()

        self.step_types.append(1)

    def hyperparam_gradients(self, batch):
        return {}

    def kernel_param_log(self):
        # Saves the current hyperparameters
        for key in self.kernel_params:
            if key not in self.kernel_param_history:
                self.kernel_param_history[key] = []
            self.kernel_param_history[key].append(self.kernel_params[key])

    def report(self, details=False, **kwargs):
        if details:
            plotting.plot_kernel_param_hist(self)
        return self.score(**kwargs)

    def hyper_heatmap(self):
        old_params = self.kernel_params.copy()
        poss_ls = np.arange(0, 4, .15)
        poss_var = np.arange(-5, 6, .15)
        res_dict = []
        for ls_i, ls in enumerate(poss_ls):
            res_dict.append([])
            for var_i, var in enumerate(poss_var):
                new_params = {key: None for key in old_params}
                new_params['rbf'] = var
                new_params['length_scale'] = ls
                self.kernel_params = new_params
                self.update_kernel()
                res_dict[-1].append(sum(self.calc_ELBO(self.X, self.Y)))
        res_dict = np.asmatrix(res_dict)
        self.hyper_heat = {'ls': poss_ls, 'rbf': poss_var, 'data': res_dict}
        self.kernel_params = old_params
        self.update_kernel()

    def save(self, name=None):
        self.X = self.X_test = self.Y = self.y_test = self.Kmm = None
        self.current_batch = self.local_params = None
        if not self.use_batches():
            self.K_tilde = self.Kmn = self.Knm = self.K_mn_prime = None
        self.calculated_vars = None
        if name is None:
            now = datetime.datetime.now()
            name = 'D:{}xN:{}_qw_type:{}_date:{}'.format(self.d, self.n, self.dist_params['qw']['type'],
                                                         now.strftime('%d-%m-%Y_%H:%M:%S.%f'))
        path = self.log_folder
        if path is None:
            path = '../Models'
        if not utils.exists(path):
            os.mkdir(path)
        self.run_info['Saving time'] = datetime.datetime.now()
        path = os.path.join(path, name)
        os.mkdir(path)
        if not core.math.debug():
            pickle.dump(self, open(os.path.join(path, 'learner'), 'wb'))
        self.log_folder = path
        time_format = '%d.%m | %H:%M:%S'
        with open(os.path.join(path, 'log'), 'w') as f:
            if 'name' in self.run_info['data_set']:
                f.write('Dataset: {}\n'.format(self.run_info['data_set']['name']))
            f.write(
                'N inducing points: {}, batch_size: {} lambda_w: {} epochs:{}\n'.format(self.m, self.s, self.lambda_w,
                                                                                        self.epoch_count))
            if 'scores' in self.run_info:
                f.write(self.run_info['scores'].to_string())
            f.write('\nInit:         {} '
                    '\nStart:        {} '
                    '\nStop:         {} '
                    '\nTime running: {}\n'.format(self.run_info['init'].strftime(time_format),
                                                  self.run_info['start'].strftime(time_format),
                                                  self.run_info['end'].strftime(time_format),
                                                  (self.run_info['end'] - self.run_info['start'])))

        return path

    def get_expected_w(self):
        if any(type_ in self.dist_params['qw']['type'] for type_ in ['Laplace', 'Gaussian', 'Horseshoe']):
            return self.dist_params['qw']['mean']
        elif self.dist_params['qw']['type'] == 'None':
            return self.sh.zeros((self.d, 1))
        raise NotImplemented

    # The following functions are implemented for saving calculation time
    def get_expected_w_tx_x_tw(self, X, wX_t=None):
        if sparse.issparse(X):
            return self.get_expected_wXw(X.dot(X.T))
        if any(type_ in self.dist_params['qw']['type'] for type_ in ['Laplace', 'Gaussian', 'Horseshoe']):
            wX_t = wX_t if wX_t is not None else self.sh.dot(X.T, self.dist_params['qw']['mean'])
            res = self.sh.to_dense(self.sh.dot(wX_t.T, wX_t))
            if 'MV' in self.dist_params['qw']['type']:
                res += self.sh.trace_dot(X.dot(X.T), self.dist_params['qw']['var'])
            elif 'MF' in self.dist_params['qw']['type']:
                res += np.einsum('j,ji->', self.dist_params['qw']['var'], self.sh.el_square(X))
            return res
        elif self.dist_params['qw']['type'] == 'None':
            return 0
        raise NotImplemented

    def get_expected_wXw(self, X):
        if any(type_ in self.dist_params['qw']['type'] for type_ in ['Laplace', 'Gaussian', 'Horseshoe']):
            res = self.sh.to_dense(
                self.sh.multi_dot([self.dist_params['qw']['mean'].T, X, self.dist_params['qw']['mean']]))
            res += self.get_trace_Xvar(X)
            return res
        elif self.dist_params['qw']['type'] == 'None':
            return 0
        raise NotImplemented

    def get_expected_wXThetaXw(self, Xw, X, Theta):
        if self.dist_params['qw']['type'] == 'None':
            return 0
        if any(typ in self.dist_params['qw']['type'] for typ in ['MF', 'MAP']):
            res = (self.sh.multiply(self.sh.el_square(Xw), Theta[:, np.newaxis])).sum()
            if not 'MAP' in self.dist_params['qw']['type']:
                X_sq_Theta = self.sh.multiply(self.sh.el_square(X), Theta).sum(axis=1)
                res += (self.sh.multiply(X_sq_Theta, self.dist_params['qw']['var'][:, np.newaxis])).sum()
            return res
        else:
            return self.get_expected_wXw(self.sh.dot(self.sh.multiply(X, Theta), X.T))

    def get_trace_Xvar(self, X):
        if 'MV' in self.dist_params['qw']['type']:
            return self.sh.trace_dot(X.T, self.dist_params['qw']['var'])
        elif 'MF' in self.dist_params['qw']['type']:
            # Trace
            if not self.sh.sparse:
                return self.sh.trace_dot(X, self.sh.diag(self.dist_params['qw']['var']))
            else:
                return sum(np.multiply(self.dist_params['qw']['var'],
                                       self.sh.to_dense(self.sh.diag(X))))
        return 0

    def get_XvarX(self, X):
        if 'MV' in self.dist_params['qw']['type']:
            return self.sh.multi_dot([X.T, self.dist_params['qw']['var'], X])
        elif 'MF' in self.dist_params['qw']['type']:
            return self.sh.multi_dot([self.sh.multiply(X.T, self.dist_params['qw']['var']), X])
        return 0
    
    def fit(self, w_true=None, details=True, X_test=None, y_test=None, side_info_test=None, eval=True):
        """
        Fits the Classifier to the data
        :param w_true: real weights (only usable for toy dataset) [d x 1]
        :param details: Boolean 
        :param X_test: optional, if not set default test-set is used
        :param y_test: optional, if not set default test-set is used
        :param check_ELBO_bs: 
        :return: dict with info about the run
        """
        
        if X_test is None:
            X_test = self.X_test
            y_test = self.y_test
            side_info_test = self.side_info_test
        info = {}

        elbos = [[-float('inf') for _ in range(3)]]
        
        if w_true is not None:
            werrs = [utils.w_err(w_true, self.get_expected_w())]
            werrs_normalized = [utils.w_err(w_true, self.get_expected_w(), normalized=True)]
        else:
            werrs = werrs_normalized = None
        scores = []
        # get Loss
        while not self.done:
            # calculate Gradients
            # update m_w, S_w, m_u, S_u
            current_elbos = self.epoch()
            if any(np.isnan(x) or np.isinf(x) for x in current_elbos[-1]):
                print('NANs and Infs found in ELBO')
            if w_true is not None:
                werrs.append(utils.w_err(w_true, self.get_expected_w()))
                werrs_normalized.append(utils.w_err(w_true, self.get_expected_w(), normalized=True))
            if details and eval:
                confs, single_confs = self.predict_MAP(X_test, side_info=side_info_test)
                scores.append(utils.score(y_test, confs))

            # calculate Loss
            elbo_diff = sum(current_elbos[-1]) - sum(elbos[-1])
            elbos += current_elbos
            print('Epoch {:3d} || ELBO {:+.4E} ({:+.4E}) {}'.format(self.epoch_count, sum(elbos[-1]), elbo_diff,
                                                                    '' if elbo_diff >= 0 else '--'))
            if self.done:
                print('Finished')
                break
            if self.epoch_count > 5 and np.isinf(sum(elbos[-1])):
                print('Too unstable')
                break

            if w_true is not None:
                info['werrs'] = werrs
                info['werrs_normalized'] = werrs_normalized
        if 'scores_over_time' not in self.run_info:
            self.run_info['scores_over_time'] = []
        self.run_info['scores_over_time'] += scores


        return info


def load_SVI_Instance(fn, relative_path=True):
    """
    Load a SVI_Instance with a given path to the saved learner
    :param fn: path to learner
    :param relative_path:
    :return: the loaded learner
    """
    if relative_path and 'Models/' not in fn:
        fn = os.path.join('Models', fn)
    fn = os.path.join(fn, 'learner')
    learner = pickle.load(open(fn, 'rb'))
    if not 'reloaded' in learner.run_info:
        learner.run_info['reloaded'] = []
    learner.run_info['reloaded'].append(datetime.datetime.now())
    return learner


if __name__ == '__main__':
    pass
