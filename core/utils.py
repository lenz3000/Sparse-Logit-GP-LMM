import datetime
import os
from pathlib import Path
import gpflow
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn

from core.math import SparseHandler, debug


def w_err(w_true, w_est, normalized=False):
    if normalized:
        norm_factor = sum(np.abs(w_true)) / max(1E-3, sum(np.abs(w_est)))
    else:
        norm_factor = 1
    return np.mean([abs(w_true[i] - w_est[i] * norm_factor) for i in range(len(w_true))])


def get_class_pred_conf(conf):
    pred = np.where(conf < 0.5, -np.ones_like(conf), np.ones_like(conf))
    conf = np.where(conf < 0.5, 1 - conf, conf)
    return pred, conf


def log_confidence(y, conf_pred):
    '''
    :param y: True labels
    :param conf_pred: predicted confidences of samples being in class 1
    :return: average confidence given for the correct classes
    '''
    score = 0
    for i in range(len(y)):
        if y[i] > 0:
            score += np.log(conf_pred[i])
        else:
            score += np.log(1 - conf_pred[i])
    return score/len(y)


def score(y_true, confs):
    """
    :param y_true: True classes (0 or 1)
    :param confs: predicted confidences of sample being in class 1 \in [0,1]
    :return: precision, ROC_AUC score, avg precision, precision
    """
    # sklearn only acepts float32
    confs = confs.astype('float32')
    if np.size(y_true) != np.size(confs):
        print('Shapes of Y and predictions are different', np.shape(y_true), np.shape(confs))
        raise AssertionError
    if np.shape(y_true) != np.shape(confs):
        y_true = np.reshape(y_true, np.shape(confs))
    if any(np.isnan(confs)) or any(np.isinf(confs)):
        print('NANs and Infs found ')
        print(confs)
        np.nan_to_num(confs, copy=False)
    if max(confs) > 1 or min(confs) < 0:
        print(max(confs), '--', min(confs), ' the confidences are not correct')
        np.clip(confs, 0, 1, out=confs)
    if min(y_true) != 0 or max(y_true) != 1:
        y_true = y_true > 0

    skm = sklearn.metrics
    preds = [confs[i] > 0.5 for i in range(len(y_true))]
    return {'Log Conf': log_confidence(y_true, confs),
            'ROC AUC': skm.roc_auc_score(y_true, confs),
            'bin. Prec.': skm.precision_score(y_true, preds),
            'bin. Recall': skm.recall_score(y_true, preds),
            'Acc.': skm.accuracy_score(y_true, preds),
            'F1': skm.f1_score(y_true, preds, average='macro')
            }


def score_baseline(dataset, kernel_params=None, Z=None, lambda_w=1, batch_size=100, add_info=None):
    """
    This function trains the baselines
    :param dataset: Dict like output of data_set_handling.reformat_data_set
    :param kernel_params: Kernel_params for SGPC
    :param Z: Inducing Points for SGPC. If side info present Z need to contain side info as well
    :param lambda_w: L1 regularization parameter for the black box approach
    :return:
    """
    scores = {}

    # Getting all the information from the dataset
    X_no_side, Y = dataset['train']
    sh = SparseHandler(sp.issparse(X_no_side))
    X_test_no_add_info, Y_test = dataset['test']
    side_train = side_test = X_zero_side = None
    if 'side_info' in dataset:
        side_train = dataset['side_info']['train']
        side_test = dataset['side_info']['test']
        X_zero_side = sh.join_matrices(X_test_no_add_info, sh.zeros(np.shape(side_test)), axis=0)

    X = sh.join_matrices(X_no_side, side_train, axis=0)

    X_test = sh.join_matrices(X_test_no_add_info, side_test, axis=0)
    Y_true = np.ravel(Y_test > 0)
    # Testing two linear classifiers, one on all data and one without side information
    y_lin = None

    if add_info is not None:
        if 'w_true' in add_info and add_info['w_true'][0] != 0:
            y_lin = np.ravel(np.sign(X_test_no_add_info.T.dot(add_info['w_true'])) > 0)

    for i in range(2):
        name = 'Log. Regression'
        if i == 0:
            X_test_ = X_test
            X_train_ = X
        else:
            if side_train is None: break
            X_test_ = X_test_no_add_info
            X_train_ = X_no_side
            name = name[:-7] + '. no side'
        clf = sklearn.linear_model.LogisticRegression(penalty='l1', max_iter=2000, tol=1E-7, C=lambda_w,
                                                      dual=False)
        print('Training: {}'.format(name))
        clf.fit(X_train_.T, np.ravel(Y > 0))
        probs = clf.predict_proba(X_test_.T)[:, -1]
        scores[name] = score(Y_true, probs)

        if add_info is not None:
            add_info['learned_weights'][name] = np.ravel(clf.coef_)
            if 'w_true' in add_info and add_info['w_true'][0] != 0:
                add_info['lin_model_score'][name] = score(y_lin, probs)
            if X_zero_side is not None and i == 0:
                add_info['zero Side'][name] = score(Y_true, clf.predict_proba(X_zero_side.T)[:, -1])

    if not sp.isspmatrix(X) and kernel_params is not None:
        print('Training: Sparse GP')
        kp = kernel_params
        d = X.shape[0]
        gp_kernel = gpflow.kernels.RBF(d, variance=np.exp(kp['rbf'] if kp['rbf'] is not None else 1E-5),
                                       lengthscales=np.exp(kp['length_scale']) if 'length_scale' in kp else 1) \
                    + gpflow.kernels.Linear(d, variance=np.exp(kp['lin']) if kp['lin'] is not None else 1E-5) \
                    + gpflow.kernels.White(d, variance=np.exp(kp['white']) if kp['white'] is not None else 0)

        gpc = gpflow.models.SVGP(
            X.T.astype(float), np.expand_dims(Y, -1).astype(float), kern=gp_kernel,
            likelihood=gpflow.likelihoods.Bernoulli(),
            Z=Z.T.astype(float), minibatch_size=batch_size)
        gpflow.train.ScipyOptimizer().minimize(gpc, maxiter=2000)
        probs = gpc.predict_y(X_test.T.astype(float))[0]
        scores['Sparse GP'] = score(Y_true, probs)
        if add_info is not None:
            if y_lin is not None:
                add_info['lin_model_score']['Sparse GP'] = score(y_lin, probs)
            if X_zero_side is not None:
                add_info['zero Side']['Sparse GP'] = score(Y_true, gpc.predict_y(X_zero_side.T.astype(float))[0])
    scores = pd.DataFrame(scores)
    return scores


def exists(fn):
    try:
        _ = Path(fn).resolve()
    except FileNotFoundError:
        return False
    return True


def hyper_step_size(epoch_count, hyperparameter_lr):
    return 1. / (1 + epoch_count ** hyperparameter_lr)

def save_add_info(add_info, data_set_name):
    now = datetime.datetime.now()
    np.save(os.path.join('../experiment_logs',
                         '{}:{}'.format(data_set_name, now.strftime('%d-%m-%Y_%H:%M:%S.%f'))), add_info)
    if debug():
        import matplotlib.pyplot as plt
        plt.figure('weights')
        if 'w_true' in add_info:
            plt.plot(add_info['w_true'])
        for key in add_info['learned_weights']:
            plt.plot(add_info['learned_weights'][key], label=key)
        plt.legend()
        plt.show()
