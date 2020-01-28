import collections

import numpy as np

import core.math
from core import kernels, utils
from core.model import SVIinstance
from numpy.linalg import multi_dot, inv
import warnings


class PolyaGammaLogitClassifier(SVIinstance):

    def __init__(self, *args, omega_gradients=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.omega_gradients = omega_gradients
        self.rho = .5
        self.admm_conv_crit = 1E-5
        self.link_func = core.math.logit

    def calcL1(self, X, Y, side_info=None):
        """
        Calculates the L1 for the ELBO
        :param X:
        :param Y:
        :return: l1: lower bound for the log likelihood
        """
        # Getting the parameter values and pre-calculating values to save time
        batch_factor = self.n / np.shape(X)[-1]
        e_w = self.get_expected_w()
        m_u = self.dist_params['qu']['mean']
        S_u = self.dist_params['qu']['var']
        K_mn_prime, Kmn = self.get_Kmn_prime_Kmn(X, side_info)
        Kmnu = K_mn_prime.T.dot(m_u)
        K_tilde = self.sh.diag(self.get_K_tilde(X, Kmn, side_info=side_info))
        expctd_X_w = X.T.dot(e_w)
        cs = self.local_params
        theta_diag = self.theta(cs)
        X_sqrtT = self.sh.multiply(X, np.sqrt(theta_diag))
        Theta_Kmnu = np.multiply(Kmnu, np.expand_dims(theta_diag, -1))

        # Calculating the log likelihood
        l1 = 0.5 * batch_factor * (np.dot(Y.T, self.sh.to_dense(expctd_X_w + Kmnu))
                                   - np.sum(theta_diag * K_tilde)
                                   - self.sh.trace(
                    self.sh.multi_dot([np.multiply(K_mn_prime, theta_diag[np.newaxis, :]),
                                       K_mn_prime.T, S_u]))
                                   - self.get_expected_w_tx_x_tw(X_sqrtT)
                                   - 2 * self.sh.multi_dot([Theta_Kmnu.T, expctd_X_w])
                                   - self.sh.multi_dot([Theta_Kmnu.T, Kmnu]))
        # Assert that l1 has the correct form -> indicator that everything went well
        assert np.shape(l1) == (1, 1)
        return l1[0, 0]

    def update_local_params(self, X, y, side_info=None, precalc_dict=None):
        '''
        Calculates the optimal local parameters
        :param X: datapoints [dxs]
        :param y: labels list of length s
        :param side_info: possible side-information [d'xs]
        :param precalc_dict: dictionary with values that are already calculated, this can lead to a substantial speed-up
            these values are not checked for correctness
        :return: local parameters
        '''
        e_w = self.get_expected_w()
        m_u = self.dist_params['qu']['mean']
        S_u = self.dist_params['qu']['var']

        if precalc_dict is None: precalc_dict = {}
        if not 'Kmn' in precalc_dict: precalc_dict['Kmn'] = self.get_Kmn(X, side_info)
        Kmn = precalc_dict['Kmn']
        if not 'K_tilde' in precalc_dict: precalc_dict['K_tilde'] = np.diag(self.get_K_tilde(X, Kmn, side_info))
        K_tilde = precalc_dict['K_tilde']
        if not 'kmminvSUkmminv' in precalc_dict:
            precalc_dict['kmminvSUkmminv'] = multi_dot([self.Kmm_inv, S_u, self.Kmm_inv])
        kmminvSUkmminv = precalc_dict['kmminvSUkmminv']
        if not 'Kmnu' in precalc_dict: precalc_dict['Kmnu'] = multi_dot([Kmn.T, self.Kmm_inv, m_u])
        Kmnu = np.asarray(precalc_dict['Kmnu'])
        if not 'expected_xt_w' in precalc_dict: precalc_dict['expected_xt_w'] = X.T.dot(e_w)
        expected_xt_w = precalc_dict['expected_xt_w']
        if not 'expected_w_tx_x_tw' in precalc_dict:
            precalc_dict['expected_w_tx_x_tw'] = np.fromfunction(
                np.vectorize(lambda i:
                             self.get_expected_w_tx_x_tw(X[:, int(i)], wX_t=expected_xt_w[int(i)])), (len(y),))
        expected_w_tx_x_tw = precalc_dict['expected_w_tx_x_tw']
        if not 'knm_primeSUkmn_prime' in precalc_dict:
            precalc_dict['knm_primeSUkmn_prime'] = np.einsum('ij,ji->i', Kmn.T.dot(kmminvSUkmminv), Kmn)
        knm_primeSUkmn_prime = precalc_dict['knm_primeSUkmn_prime']

        dc = K_tilde[:, np.newaxis] + np.square(Kmnu) + 2 * self.sh.to_dense(self.sh.multiply(expected_xt_w, Kmnu)) \
             + knm_primeSUkmn_prime[:, np.newaxis] + self.sh.to_dense(expected_w_tx_x_tw)[:, np.newaxis]
        return np.asarray(np.sqrt(dc))[:, 0]

    def variational_param_update(self, var, batch):
        """
        Updates one of q(u) or q(w)
        :param var: variable that is to be updates qu or qw
        :param batch: current batch: list with entries  X, y, side_info
        :return:
        """
        X, y, side_info = batch
        batch_factor = self.n / np.shape(X)[-1]
        K_mn_prime, Kmn = self.get_Kmn_prime_Kmn(X, side_info)

        theta_diag = self.theta(self.local_params)
        Theta = self.sh.diag(theta_diag)

        # Update for qu is the same for all different priors
        if var == 'qu':
            Theta_u = np.multiply(K_mn_prime, theta_diag).dot(K_mn_prime.T)
            # Woodbury is of no use here, because it is better to invert a mxm matrix than a nxn matrix
            dS_u = inv(self.Kmm_inv + batch_factor * Theta_u)
            e_w = self.get_expected_w()
            dm_u = batch_factor * np.dot(dS_u,
                                         np.dot(K_mn_prime, .5 * y
                                                - self.sh.to_dense(self.sh.multi_dot([Theta, X.T, e_w]))))
            return self.coordinate_step({'qu': {'mean': dm_u, 'var': dS_u}})
        if var == 'qw':
            qw_type = self.dist_params['qw']['type']
            # Scale mixture model all have an underling Gaussian Distribution for q(w)
            if 'MV' in qw_type:
                S_w = self.sh.woodbury_diag_A(self.dist_params['pw']['varinv'], batch_factor * Theta, X)
                m_u = self.dist_params['qu']['mean']
                dm_w = self.sh.dot(S_w,
                                   batch_factor * self.sh.dot(X, .5 * y - self.sh.multi_dot(
                                       [Theta, K_mn_prime.T, m_u])))
                self.coordinate_step({'qw': {'mean': np.asarray(dm_w), 'var': S_w}})
                if qw_type != 'MV Gaussian': self.scale_mixture_update(qw_type)
                return
            elif 'MF' in qw_type:
                reps = 1
                if self.use_batches() and self.s < 500:
                    # For small batch-sizes the MF becomes unstable.
                    # It is better to repeat the updates a few times then
                    reps = int(np.ceil(- self.s / 100 + 6))

                for i in range(reps):
                    self.mf_gauss_CAVI(self.dist_params['qw'], self.dist_params['pw'], X=X, y=y,
                                       side_info=side_info,
                                       K_mn_prime=K_mn_prime, Kmn=Kmn, batch_factor=batch_factor)
                    if qw_type != 'MF Gaussian':
                        self.scale_mixture_update(qw_type)
                return

            if 'Laplace MAP' in qw_type:
                if qw_type == 'Laplace MAP iterative':
                    dm_w = self.iterative_lasso(K_mn_prime, theta_diag, batch_factor, X, y)
                else:
                    dm_w = self.ADMM(X, y, Theta, K_mn_prime, batch_factor)
                return self.coordinate_step({'qw': {'mean': dm_w}})

            elif qw_type == 'None':
                # XGPC implementation
                return
            print(qw_type)
        print(var)
        raise NotImplemented

    def scale_mixture_update(self, qw_type):
        """
        This function updates the estimate for the local scale parameter lambda for the Laplace and Horseshoe prior
        :param qw_type: Name of prior
        :return:
        """
        expctd_w_sqrd = self.sh.el_square(np.ravel(self.dist_params['qw']['mean']))
        if 'MV' in qw_type:
            expctd_w_sqrd += np.diag(self.dist_params['qw']['var'])
        elif 'MF' in qw_type:
            expctd_w_sqrd += self.dist_params['qw']['var']
        else:
            print('Wrong distribution')
            raise NotImplemented
        b = expctd_w_sqrd / (self.dist_params['qw']['tau'] ** 2)
        if 'Laplace' in qw_type:
            self.dist_params['qw']['a'] = b
            self.dist_params['qw']['lambda'] = 1 / np.sqrt(self.dist_params['qw']['a'])
            self.dist_params['pw']['var'] = self.dist_params['qw']['tau'] ** 2 / self.dist_params['qw'][
                'lambda']
            self.dist_params['pw']['varinv'] = 1 / self.dist_params['pw']['var']
        elif 'Horseshoe' in qw_type:
            # Warn if b gets too big or small
            with warnings.catch_warnings():
                warnings.simplefilter("once")
                if np.min(b) < self.dist_params['qw']['gamma norm'].x[0]:
                    warnings.warn('b below threshold: '
                                  '{} (thresh: {})'.format(np.min(b),
                                                           self.dist_params['qw']['gamma norm'].x[0]),
                                  RuntimeWarning)
                if np.max(b) > self.dist_params['qw']['gamma norm'].x[-1]:
                    warnings.warn('b above threshold: '
                                  '{} (thresh: {})'.format(np.max(b),
                                                           self.dist_params['qw']['gamma norm'].x[-1]),
                                  RuntimeWarning)

            self.dist_params['qw']['b'] = np.clip(b, self.dist_params['qw']['gamma norm'].x[0],
                                                  self.dist_params['qw']['gamma norm'].x[-1])
            lambda_sq_inv = self.dist_params['qw']['expected gamma'](self.dist_params['qw']['b'])
            self.dist_params['qw']['norm'] = self.dist_params['qw']['gamma norm'](self.dist_params['qw']['b'])
            self.dist_params['pw']['varinv'] = lambda_sq_inv * self.dist_params['qw']['tau'] ** (-2)
            self.dist_params['pw']['var'] = 1 / self.dist_params['pw']['varinv']
        else:
            raise NotImplemented

    def mf_gauss_CAVI(self, qw, pw, X, y, side_info,
                      K_mn_prime, Kmn, batch_factor):

        """
        Update the distributions q(w_j) for all j in {1,...,d} for a Gaussian q(w_j)
        :param qw: Variational Distribution for w
        :param pw: Prior on w
        :param X, y, side_info, K_mn_prime, Kmn, batch_factor: Parameters of the current batch and precalculated vars.
        :return:
        """
        # Predicted confounder
        m_eps = K_mn_prime.T.dot(self.dist_params['qu']['mean'])
        # old weights
        og_m_w = qw['mean'].copy()

        prior_varinv = pw['varinv'] if isinstance(pw['varinv'],
                                                  collections.Iterable) else pw['varinv'] * np.ones_like(qw['var'])
        # Get expected local parameters
        local_params = self.local_params
        theta_diag = self.theta(local_params)
        theta_eps = self.sh.multiply(m_eps, theta_diag[:, np.newaxis])

        # This is a running sum that saves us calculating X^Tw for every j
        cur_X_w = self.sh.to_dense(X.T.dot(og_m_w))
        calc_m = np.zeros_like(og_m_w)
        calc_s = np.zeros_like(og_m_w)
        # for j in np.random.permutation(self.d):
        for j in range(self.d):
            X_j = X[j, :]
            # Calculating optimal variance
            new_s_j = 1 / (prior_varinv[j] + batch_factor * self.sh.to_dense(
                self.sh.multiply(X_j, theta_diag.T).dot(X_j.T)))
            # Calculating X^Tw without dimension j
            X_w_j = self.sh.multiply(X_j, og_m_w[j]).T
            Xw_no_j = cur_X_w - X_w_j
            # Calculating the optimal weight
            new_m_j = batch_factor * new_s_j * \
                      self.sh.to_dense(X_j.dot(0.5 * y - self.sh.multiply(Xw_no_j, theta_diag[:, np.newaxis])
                                               - theta_eps))
            calc_m[j, 0] = new_m_j
            calc_s[j, 0] = new_s_j
            # Update
            self.single_weight_coordinate_step({'qw': {'mean': np.clip(new_m_j, - self.max_w, self.max_w)
                , 'var': new_s_j}}, j)

            # Updating the running sums
            delta_w_j = qw['mean'][j, 0] - og_m_w[j, 0]
            delta_X_w_j = X_j.T * delta_w_j
            cur_X_w += delta_X_w_j
            self.local_params = local_params
        return {}

    def iterative_lasso(self, K_mn_prime, theta_diag, batch_factor, X, y):
        """
        Using an active set strategy , this method applies coordinate ascent to the MAP estimates of w until convergence
        :param K_mn_prime:
        :param theta_diag:
        :param batch_factor:
        :param X:
        :param y:
        :return:
        """
        def not_optimal():
            not_optimal_weights = []
            for i in range(self.d):
                if i in safe_not_in: continue
                X_i = X[i, :]
                xiTxi = (self.sh.to_dense(self.sh.multiply(X_i, theta_diag.T).dot(X_i.T)))
                X_w_i = self.sh.multiply(X_i, dm_w[i]).T
                Xw_no_i = X_w - X_w_i
                grad = batch_factor * (0.5 * X_i.dot(y) - self.sh.to_dense(
                    X_i.dot(Theta_eps) + self.sh.multi_dot([self.sh.multiply(X_i, theta_diag.T), Xw_no_i]))
                                       - xiTxi * dm_w[i, 0])
                if dm_w[i, 0] != 0:
                    grad -= np.sign(dm_w[i, 0]) / self.dist_params['pw']['var']
                else:
                    grad -= np.sign(grad) * min(1 / self.dist_params['pw']['var'], abs(grad))
                if abs(grad) > conv_crit:
                    not_optimal_weights.append(i)
            return not_optimal_weights

        def get_safe_not_in():
            l_max = np.max(X.dot(y))
            y_ = np.asarray(.5 * y - Theta_eps[:, np.newaxis])
            l_prime = (l_max - lambda_) / l_max
            for i in range(self.d):
                X_i = X[i, :]
                if np.abs(X_i.dot(y_)) \
                        < lambda_ - np.sqrt(np.sum(np.sum(self.sh.el_square(X_i)))) \
                        * np.sqrt(np.sum(self.sh.el_square(y_))) * l_prime:
                    safe_not_in.append(i)
                    dm_w[i, 0] = 0

        conv_crit = 1E-6
        lambda_ = 1 / self.dist_params['pw']['var']
        m_eps = np.asarray(K_mn_prime.T.dot(self.dist_params['qu']['mean']))
        dm_w = self.sh.to_dense(self.dist_params['qw']['mean'])
        Theta_eps = self.sh.multiply(m_eps[:, 0], theta_diag)
        X_w = self.sh.to_dense(X.T.dot(dm_w))
        safe_not_in = []
        get_safe_not_in()
        to_be_optimized = [i for i in list(range(self.d)) if i not in safe_not_in]
        it_count = 0
        delta = 1
        while len(to_be_optimized) > 0 and delta > conv_crit:
            delta = 1
            while delta > 1E-6:
                delta = 0
                to_be_optimized = sorted(to_be_optimized, key=lambda i: abs(dm_w[i, 0]))
                for j in to_be_optimized:
                    X_j = X[j, :]
                    xjTxj = (self.sh.to_dense(self.sh.multiply(X_j, theta_diag.T).dot(X_j.T)))
                    ds_w = 1 / (xjTxj if abs(xjTxj) > 1E-10 else 1E-10)
                    X_w_j = self.sh.to_dense(self.sh.multiply(X_j, dm_w[j]).T)
                    Xw_no_j = X_w - X_w_j
                    not_regularized = self.sh.to_dense(X_j.dot(0.5 * y - Theta_eps[:, np.newaxis]
                                                               - self.sh.multiply(Xw_no_j, theta_diag[:, np.newaxis])))
                    new_w_j = core.math.soft_thresh_op(ds_w * batch_factor * not_regularized,
                                                       ds_w / self.dist_params['pw']['var']) / batch_factor

                    delta_w_j = new_w_j - dm_w[j, 0]
                    delta += abs(delta_w_j)
                    X_w += self.sh.to_dense(X_j.T * delta_w_j)
                    dm_w[j, 0] = new_w_j
                    if dm_w[j, 0] == 0 or abs(delta_w_j) < conv_crit:
                        if it_count < 10: continue
                        to_be_optimized.remove(j)

                it_count += 1
                to_be_optimized = not_optimal()
            # print('opt loop {:3d} - Delta:{:.3E} -TBO: {:3d} - SAFE: {:3d}'.format(it_count, delta,
            #                                                                       len(to_be_optimized),
            #                                                                       len(safe_not_in)))
        return dm_w

    def ADMM(self, X, y, Theta, K_mn_prime, batch_factor):
        """
        This method executes the ADMM steps iteratively for q(w)
        :param X:
        :param y:
        :return:
        """
        steps = 50
        const = batch_factor * self.sh.dot(X, 0.5 * y - self.sh.multi_dot([Theta, K_mn_prime.T,
                                                                           self.dist_params['qu']['mean']]))
        xtx_r_inv = self.sh.woodbury_diag_A(self.rho, batch_factor * Theta, X)

        # Warm start
        if self.calculated_vars is None:
            self.calculated_vars = {}
        if 'admm_vars' not in self.calculated_vars:
            self.calculated_vars['admm_vars'] = [0, 0, 0]
        w, what, u = self.calculated_vars['admm_vars']

        for _ in range(steps):
            w = self.sh.dot(xtx_r_inv, const + self.rho * (what + u))
            what = core.math.soft_thresh_op(w - u, 1 / (self.rho * self.lambda_w))
            u = u + what - w
        self.calculated_vars['admm_vars'] = w, what, u
        return np.asarray(w)

    def local_parameter_gradients(self, dKnm, dKmminv, dK_tilde, X, Kmn, Kmn_prime=None):
        """
        :param Kmn_prime: [optional] already Kmn_prime
        :param dKnm: derivation of Knm
        :param dKmminv:
        :param dK_tilde:
        :param X:
        :param Kmn:
        :return: Derivation of local parameters
        """
        c = self.local_params
        dcs = {key: [] for key in dK_tilde}
        m_w = self.get_expected_w()
        m_u = self.dist_params['qu']['mean']
        S_u = self.dist_params['qu']['var']
        dKnm_prime = {key: (dKnm[key].dot(self.Kmm_inv) + Kmn.T.dot(dKmminv[key])) for key in dKmminv}
        if Kmn_prime is None: Kmn_prime = Kmn.T.dot(self.Kmm_inv).T
        m_f = Kmn_prime.T.dot(m_u)
        Xw = X.T.dot(m_w)
        for i in range(len(c)):
            kikmminv = Kmn_prime[:, i].T
            for key in dK_tilde:
                dkikmminv = dKnm_prime[key][i, :]
                cur_c_grad = (Xw[i, 0] + m_f[i, 0]) * dkikmminv.dot(m_u) \
                             + .5 * dK_tilde[key][i] + self.sh.multi_dot([kikmminv, S_u, dkikmminv.T])
                cur_c_grad /= c[i]
                dcs[key].append(cur_c_grad[0, 0])
        return dcs

    def theta_gradients(self, cs):
        dthetas = []
        for c in cs:
            tanh = np.tanh(c / 2)
            dthetas.append(1 / (2 * c) * (.5 - tanh * (1 / c + tanh / 2)))
        return dthetas

    def hyperparam_gradients(self, batch):
        """
        This function calculates the hyperparameter gradients, given a batch
        :param batch:
        :return:
        """
        X, y, side_info = batch
        batch_factor = self.n / np.shape(X)[-1]

        m_u, S_u = [self.dist_params['qu'][key] for key in ['mean', 'var']]
        m_w = self.get_expected_w()

        def get_derivations(h):
            """
            This debug function numerically derivates the ELBO w.r.t. the hyperparameters
            with (ELBO(x + h) - ELBO(x))/h, where x is the hyperparameter
            :param h: change for the derivation
            :return: gradients w.r.t. the h.params
            """
            old_kp = self.kernel_params
            old_kernel = self.kernel
            res = {}
            for key in sorted(self.kernel_params.keys()) + [None]:
                if key is not None and self.kernel_params[key] is None:
                    # This means that this Kernel is not used
                    continue
                new_kps = old_kp.copy()

                if key is not None:
                    new_kps[key] += h
                else:
                    key = 'old'
                self.kernel_params = new_kps
                self.kernel = kernels.Kernel_Handler(**new_kps)
                self.update_kernel()
                c = {}
                if self.omega_gradients:
                    self.local_params = self.update_local_params(X, y, side_info=side_info)
                c['dc'] = np.array(self.local_params)
                Theta = np.diag(self.theta(c['dc']))
                elbo = self.calc_ELBO(X, y, side_info)
                c['L'] = sum(elbo)
                c['L1'] = elbo[0]
                c['KL_u'] = elbo[1]
                c['KL_w'] = elbo[2]
                c['Kmm'] = self.Kmm
                c['Knm'] = self.get_Kmn(X, side_info).T
                if side_info is None:
                    c['Knn'] = self.kernel(X.T)
                else:
                    c['Knn'] = self.kernel(X.T, side_X=side_info.T)
                c['Kmminv'] = self.Kmm_inv
                c['KnmKmminv'] = np.dot(c['Knm'], self.Kmm_inv)
                c['KnmKmminvKmn'] = multi_dot([c['Knm'], self.Kmm_inv, c['Knm'].T])
                c['ykmu'] = multi_dot([y.T, c['Knm'], self.Kmm_inv, m_u])
                c['trthk'] = np.trace(multi_dot([Theta, c['Knn'] - multi_dot([c['Knm'], self.Kmm_inv, c['Knm'].T])]))
                c['trkmntknmsu'] = np.trace(multi_dot([self.Kmm_inv, c['Knm'].T, Theta, c['Knm'], self.Kmm_inv, S_u]))
                c['mk+xwth'] = self.sh.multi_dot([(multi_dot([c['Knm'], self.Kmm_inv, m_u]) + self.sh.dot(X.T, m_w)).T,
                                                  Theta,
                                                  self.sh.multi_dot([c['Knm'], self.Kmm_inv, m_u]) + self.sh.dot(X.T,
                                                                                                                 m_w)])
                c['trks'] = np.trace(np.dot(self.Kmm_inv, S_u))
                c['mkm'] = multi_dot([m_u.T, self.Kmm_inv, m_u])
                c['logk'] = self.sh.logdet(self.Kmm)
                c['theta_diag'] = np.array(self.theta(c['dc']))
                c['KL_omega'] = np.array((sum(
                    [ci / 4 * np.tanh(ci / 2) for i, ci in enumerate(c['dc'])]))
                                         - sum([core.math.logcosh(ci / 2) for ci in c['dc']]))

                res[key] = c
            self.kernel = old_kernel
            return {var_key: {key: (res[var_key][key] - res['old'][key]) / h for key in res['old']} for var_key in res}

        cs = self.local_params
        Theta = self.sh.diag(self.theta(cs))

        #Getting the kernel gradients
        if side_info is not None:
            dKmms = self.kernel.gradients(self.Z.T, side_X=self.Z_side.T)
            dKnms = self.kernel.gradients(X.T, self.Z.T, side_X=side_info.T, side_Y=self.Z_side.T)
            dKnns = self.kernel.gradients(X.T, side_X=side_info.T)
            Kmn_prime, Kmn = self.get_Kmn_prime_Kmn(X, side_info=side_info)
        else:
            dKmms = self.kernel.gradients(self.Z.T)
            dKnms = self.kernel.gradients(X.T, self.Z.T)
            dKnns = self.kernel.gradients(X.T)
            Kmn_prime, Kmn = self.get_Kmn_prime_Kmn(X)
        dkmminvs = {key: - multi_dot([self.Kmm_inv, dKmms[key], self.Kmm_inv]) for key in dKnns}
        dknmkmminvs = {
            key: np.dot(dKnms[key], self.Kmm_inv) - multi_dot([Kmn.T, self.Kmm_inv, dKmms[key], self.Kmm_inv])
            for key in dKnns}
        dknmkmminvkmns = {key: 2 * multi_dot([dKnms[key], self.Kmm_inv, Kmn])
                               - multi_dot([Kmn.T, self.Kmm_inv, dKmms[key], self.Kmm_inv, Kmn])
                          for key in dKnns}
        tanhs = [np.tanh(c / 2) for c in cs]
        if self.omega_gradients:
            dcs = self.local_parameter_gradients(dKnms, dkmminvs,
                                                 {key: np.diag(dKnns[key]) - np.diag(dknmkmminvkmns[key]) for key in
                                                  dKnns},
                                                 X, Kmn, Kmn_prime=Kmn_prime)

            dtheta_diag = self.theta_gradients(cs)
            dTheta = {key: np.asarray([dcs[key][i] * dtheta_diag[i] for i in range(len(cs))]) for key in dKnns}
        K_tilde = self.get_K_tilde(X, Kmn, side_info=side_info)

        derivations = {}
        c_diffs = {}
        m_eps = multi_dot([Kmn.T, self.Kmm_inv, m_u])
        Xw = self.sh.dot(X.T, m_w)
        xwkmu = m_eps + Xw

        # Calculating the gradients, w.r.t. the different hyperparameters
        for key in dKnns:
            if self.epoch_count < self.ls_head_start and key in ['rbf', 'lin']:
                continue

            if core.math.debug():
                c = {}
                c['ykmu'] = multi_dot([y.T, dknmkmminvs[key], m_u])
                c['trthk'] = self.sh.trace(self.sh.dot(Theta, dKnns[key])) \
                             - self.sh.trace(self.sh.dot(Theta, dknmkmminvkmns[key]))
                c['trkmntknmsu'] = self.sh.trace(2 * self.sh.multi_dot([Kmn_prime, Theta, dknmkmminvs[key], S_u]))
                c['mk+xwth'] = 2 * self.sh.to_dense(self.sh.multi_dot([xwkmu.T,
                                                                       Theta, dknmkmminvs[key], m_u]))
                c['KL_omega'] = 0
                if self.omega_gradients:
                    c['trthk'] += self.sh.trace(self.sh.multiply(dTheta[key], K_tilde))
                    c['trkmntknmsu'] += self.sh.trace(
                        self.sh.multi_dot([self.sh.multiply(Kmn_prime, dTheta[key]), Kmn_prime.T, S_u]))
                    c['mk+xwth'] += self.sh.to_dense(self.sh.dot(self.sh.multiply(xwkmu.T, dTheta[key]), xwkmu))
                    c['KL_omega'] = -sum([dcs[key][i] * (.25 * tanhs[i] * (1 + ci / 2 * tanhs[i]) - ci / 8)
                                          for i, ci in enumerate(cs)])

                c['trks'] = np.trace(multi_dot([dkmminvs[key], S_u]))
                c['mkm'] = multi_dot([m_u.T, dkmminvs[key], m_u])
                c['logk'] = np.trace(np.dot(self.Kmm_inv, dKmms[key]))

            derivations[key] = multi_dot([y.T, dknmkmminvs[key], m_u])
            derivations[key] -= self.sh.trace_dot(Theta, dKnns[key])
            derivations[key] += self.sh.trace_dot(Theta, dknmkmminvkmns[key])

            derivations[key] -= self.sh.trace(2 * self.sh.multi_dot([Kmn_prime, Theta, dknmkmminvs[key], S_u]))
            derivations[key] -= (2 * self.sh.to_dense(self.sh.multi_dot([xwkmu.T,
                                                                         Theta, dknmkmminvs[key], m_u])))
            if self.omega_gradients:
                derivations[key] -= self.sh.trace(
                    self.sh.multi_dot([np.multiply(Kmn_prime, dTheta[key]), Kmn_prime.T, S_u]))
                derivations[key] -= np.sum(np.multiply(dTheta[key], np.diag(K_tilde)))
                derivations[key] -= self.sh.to_dense(self.sh.dot(self.sh.multiply(m_eps.T, dTheta[key]), m_eps))
                derivations[key] -= 2 * self.sh.to_dense(self.sh.dot(self.sh.multiply(m_eps.T, dTheta[key]), Xw))
                derivations[key] -= self.get_expected_wXThetaXw(Xw, X, dTheta[key])
            derivations[key] *= 1 / 2
            if self.omega_gradients:
                derivations[key] -= sum([dcs[key][i] * (.25 * tanhs[i] * (1 + ci / 2 * tanhs[i]) - ci / 8)
                                         for i, ci in enumerate(cs)])

            derivations[key] *= batch_factor
            if core.math.debug(): c['L1'] = derivations[key].copy()
            derivations[key] -= 1 / 2 * self.sh.trace_dot(dkmminvs[key], S_u)
            derivations[key] -= 1 / 2 * multi_dot([m_u.T, dkmminvs[key], m_u])
            derivations[key] -= 1 / 2 * self.sh.trace_dot(self.Kmm_inv, dKmms[key])
            derivations[key] = self.sh.to_dense(derivations[key])
            if core.math.debug():
                c['KL_u'] = derivations[key] - c['L1']
                c_diffs[key] = c

        if core.math.debug():
            # Checking if the calculated and numerical gradients are similar
            h = 1E-3
            real_devt = get_derivations(h)
            print()
            for key in c_diffs:
                print(key, real_devt[key]['L'], derivations[key],
                      '--------------------------------', derivations[key] / real_devt[key]['L'], 'diff:',
                      abs(derivations[key] - real_devt[key]['L']))
                if not 0.97 < derivations[key] / real_devt[key]['L'] < 1.03 \
                        and abs(derivations[key] - real_devt[key]['L']) > 1E-4:
                    print('AYAYAY calc -real - calc/real')
                    for key2 in c_diffs[key]:
                        if not (abs(real_devt[key][key2] - c_diffs[key][key2]) < 1E-10
                                or 0.95 < real_devt[key][key2] / c_diffs[key][key2] < 1.05):
                            print(key2, real_devt[key][key2], c_diffs[key][key2], c_diffs[key][key2] /
                                  self.sh.to_dense(real_devt[key][key2]))

        return derivations
