import collections

import numpy as np
import pandas as pd
from scipy import sparse

from core import utils
from core import model, plotting, dataset_handling
from core.PolyaGammaLogitClassifier import PolyaGammaLogitClassifier
import argparse


def learn(epoch_lim=150, n_ind_points=100, v_coor_asc_lr=1., lin_kernel=0, rbf_kernel=0,
          length_scale='auto',
          white_kernel=np.log(0.1), pw_var=5E-2, qw_type=None, batch_size=20,
          figures=True, hyperparameter_lr=0.1, **params):
    print('Dataset', end='')
    data_set = dataset_handling.load_toy_data(**params)
    print('...Done')
    SPARSE = 0
    if SPARSE:
        data_set['train'] = [sparse.csc_matrix(data_set['train'][0]), data_set['train'][1]]
        data_set['test'] = [sparse.csc_matrix(data_set['test'][0]), data_set['test'][1]]

    if isinstance(qw_type, str):
        qw_type = [qw_type]
    if isinstance(pw_var, collections.Iterable):
        assert len(pw_var) == len(qw_type)
    else:
        pw_var = [pw_var] * len(qw_type)

    learners = {}
    for learn_it, cur_qw_type in enumerate(qw_type):
        learners[cur_qw_type] = PolyaGammaLogitClassifier(data_set, n_ind_points=n_ind_points, lin_kernel=lin_kernel,
                                                          rbf_kernel=rbf_kernel,
                                                          length_scale=length_scale, pw_var=pw_var[learn_it],
                                                          v_coor_asc_lr=v_coor_asc_lr,
                                                          white_kernel=white_kernel, qw_type=cur_qw_type,
                                                          batch_size=batch_size,
                                                          epoch_lim=epoch_lim, hyperparameter_lr=hyperparameter_lr)
        learners[cur_qw_type].epoch_hsteps = 0
    return train_and_eval_on_dataset(data_set, learners, details=figures, baselines=True)


def train_and_eval_on_dataset(data_set, learner_dict, details=True, baselines=True, log=True, add_info=None, eval=True):
    """
    Runs an Experiment on a data_set, first fits all the learners from the dict, then fits baselines
    and finally prints the scores for all possible configurations
    :param data_set:
    :param learner_dict:
    :param details:
    :return:
    """
    X_test, Y_test = data_set['test']
    w_true = data_set['w_true'] if 'w_true' in data_set else None
    Y_ = np.ravel(Y_test > 0)
    all_scores = None

    if add_info is None:
        add_info = {}
    add_info['learned_weights'] = {}
    add_info['learner_paths'] = []
    if w_true is not None:
        add_info['w_true'] = w_true
        add_info['lin_model_score'] = {}
    if 'side_info' in data_set:
        add_info['zero Side'] = {}

    for learner_name in learner_dict:
        print('Training {}'.format(learner_name))
        learner = learner_dict[learner_name]
        train_info = learner.fit(w_true=w_true, details=details, X_test=X_test, y_test=Y_, eval=eval)
        print('Ind point#:{} N:({}/{}) D:{} '
              'learning rates:(v {}, h{}) lambda w:{} task:{} qwtype:{}'.format(learner.m, learner.n,
                                                                                np.shape(learner.X_test)[-1],
                                                                                learner.d, learner.v_coor_asc_lr,
                                                                                learner.hyperparameter_lr,
                                                                                learner.lambda_w, learner,
                                                                                learner.dist_params['qw']['type']))

        if eval:
            score = learner.report(details=details, add_info=add_info)
        if 'werrs' in train_info:
            learner.run_info['Werr'] = train_info['werrs'][-1]
            learner.run_info['Werr_normalized'] = train_info['werrs_normalized'][-1]
        plotting.eval_plots(learner, w_true=w_true, werrs=train_info['werrs'],
                            werrs_normalized=train_info['werrs_normalized'])

        if log:
            path = learner.save()
            plotting.save_all_figures(path)

        add_info['learned_weights'][learner.dist_params['qw']['type']] = learner.get_expected_w()
        add_info['learner_paths'].append(learner.log_folder)
        if eval:
            if all_scores is None:
                all_scores = score
            else:
                all_scores = pd.concat([all_scores, score], axis=1)

    if baselines:
        Z = learner.Z
        if learner.use_side_info:
            Z = learner.sh.join_matrices(Z, learner.Z_side, axis=0)
        all_scores = pd.concat([all_scores,
                                utils.score_baseline(data_set, learner.kernel_init_pars, Z,
                                                     learner.lambda_w, learner.s, add_info=add_info)],
                               axis=1)
    if all_scores is not None:
        add_info['scores'] = all_scores
        all_scores = all_scores.T.sort_index()
        print(all_scores.to_string())
        if w_true is not None and baselines:
            add_info['lin_model_score'] = pd.DataFrame(add_info['lin_model_score']).T
            print('\n \n --------------------------------------------\nUnderlying Model Scores:')
            print(add_info['lin_model_score'].to_string())
        if 'zero Side' in add_info:
            add_info['zero Side'] = pd.DataFrame(add_info['zero Side']).T
            print('\n \n --------------------------------------------\nSide Info Zero')
            print(add_info['zero Side'].to_string())
    if log:
        utils.save_add_info(add_info, data_set['name'])

    return all_scores


def retrain_learner(path, data, epoch_num=10, baselines=False):
    learner2 = model.load_SVI_Instance(path)
    learner2.init_data(data)
    learner2.epoch_lim += epoch_num
    learner2.done = False
    learner2.update_kernel()
    train_and_eval_on_dataset(data, {learner2.dist_params['qw']['type']: learner2}, baselines=baselines)
    return learner2.log_folder


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Loading in a dataset with given path or dataset-name')
    parser.add_argument('--d-name', '-dn', help='Name of default dataset, default "toy", overridden by "dataset-path"',
                        default='toy')
    parser.add_argument('--d-path', '-dp', help='path to dataset-dict (.npy)')
    parser.add_argument('--qw-types', '-qw', help='list of types of priors on weights, separated by comma. Implemented '
                                           'types are "Laplace MAP" [" iterative"], '
                                           '["MF"/"MV"] ["Horseshoe"/"Laplace"/"Gaussian"]'
                                           '    Default is "MF Laplace"', default=['MF Laplace', 'None'])
    parser.add_argument('--batch_size', '-b', help='batch size', default=200, type=int)
    parser.add_argument('--pw_var', help='Variance of prior on weights', default=0.05, type=float)
    parser.add_argument('--m', help='Number of Inducing points', default=150, type=int)
    parser.add_argument('--epochs', help='Number of Epochs', default=100, type=int)
    parser.add_argument('--rbf', help='rbf kernel variance', default=-1, type=float)
    parser.add_argument('--lin', help='linear kernel variance', default=-2, type=float)
    parser.add_argument('--h-steps', help='Hyper parameter update steps per epoch', default=2, type=int)
    parser.add_argument('--v-steps', help='Variational parameter update steps per epoch', default=3, type=int)
    parser.add_argument('--MAP-pred', help='Flag for predicting with MAP estimate (if set) or marginalizing (default)',
                        action='store_true')

    args = parser.parse_args()
    if args.d_path is None:
        dataset = dataset_handling.load(args.d_name)
    else:
        dataset = dataset_handling.load_dict_from_path(args.d_path)

    print('Creating Learner')
    ls = 1
    learners = {}

    for i, typ in enumerate(args.qw_types):
        print('Training ', typ)
        print(args)
        learner = PolyaGammaLogitClassifier(dataset, batch_size=args.batch_size, pw_var=args.pw_var,
                                            qw_type=typ, epoch_lim=args.epochs, n_ind_points=args.m, rbf_kernel=args.rbf,
                                            lin_kernel=args.lin, length_scale='auto', epoch_hsteps=args.h_steps,
                                            epoch_vsteps=args.v_steps, map_pred=args.MAP_pred)
        ls = learner.kernel_init_pars['length_scale']
        print('lengthscale now is', ls)
        learners[typ] = learner
    scores = train_and_eval_on_dataset(dataset, learners, details=False, baselines=True)
    print(scores.to_string())
