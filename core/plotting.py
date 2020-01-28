import seaborn as sns
import pandas as pd
import pickle
import numpy as np
from matplotlib import pyplot as plt
import os

import core.math
from core import utils

all_titles = []
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_clf = ['Horseshoe', 'Gaussian', 'Laplace', 'SGDClassifier', 'None', 'Black Box', 'Sparse GP', 'Laplace MAP',
             'other']



def get_color(clf_name):
    # Gets the adequate linestyle for a classifier
    res = {}
    if 'MF' in clf_name or 'iterative' in clf_name:
        res['linestyle'] = '--'
        if 'only w' in clf_name:
            res['linestyle'] = '-.'
    if 'Log. R' in clf_name  or 'Logistic' in clf_name:
            res['color'] = colors[color_clf.index('SGDClassifier')]

    elif 'only w' in clf_name:
        res['linestyle'] = ':'
    if not any(cclf in clf_name for cclf in color_clf):
        if any(x in clf_name for x in ['XSGPC', 'XGPC']):
            res['color'] = colors[color_clf.index('None')]
        if 'LASSO' in clf_name:
            res['color'] = colors[color_clf.index('SGDClassifier')]
    else:
        index = 0
        for el in color_clf:
            if el in clf_name:
                break
            index += 1
        res['color'] = colors[index]
    return res


def newfigure(title):
    global all_titles
    all_titles.append(title)
    fig = plt.figure(title)
    plt.clf()
    return all_titles, fig


def image_folder():
    return 'Images'


def image_path(path):
    return os.path.join(image_folder(), path)



def fill_between_errbar(ax, x,y, yerr, **kwargs):
    if 'alpha' in kwargs:
        del kwargs['alpha']
    if 'color' not in kwargs:
        print('You did not specify a color')
    assert np.size(y) == np.size(yerr)
    ax.plot(x,y, **kwargs)
    top_vals = [y[i] + np.abs(yerr[i]) for i in range(len(y))]
    bottom_vals = [y[i] - np.abs(yerr[i]) for i in range(len(y))]
    if 'label' in kwargs:
        del kwargs['label']
    ax.fill_between(x, bottom_vals, top_vals, alpha=0.15, **kwargs)

def plot_results(fn, fn2=None):
    dict_ = pickle.load(open(fn, 'rb'))
    x = sorted(dict_.keys())
    for key in sorted(dict_[x[0]].keys()):
        newfigure(key)
        plt.plot(x, [dict_[x_][key] for x_ in x], label=key)
    if fn2 is not None:
        dict_ = pickle.load(open(fn2, 'rb'))
        x = sorted(dict_.keys())
        for key in sorted(dict_[x[0]].keys()):
            newfigure(key)
            plt.plot(x, [dict_[x_][key] for x_ in x], label=key + 'l1')
            plt.ylim(0, 1.5)
            plt.legend()


def eval_plots(learner, w_true=None, werrs=None, werrs_normalized=None, werrorplot=True,
               w_eps_comparison=False, elbo_plot=True, est_vs_real_plot=True):
    elbos = learner.elbos
    if w_true is not None:
        if est_vs_real_plot:
            newfigure('Est vs Real')
            est_vs_real_weights_plot(plt.gca(), learner, w_true)
    if elbo_plot:
        title = 'cost'
        fig = newfigure(title)[-1]
        ax1 = fig.add_subplot(1, 1, 1)
        plot_elbo(ax1, elbos, learner)

    if werrs is not None:
        if werrorplot:
            newfigure('W errors')
            plt.title('mean absolute error w')
            plt.plot(werrs, label='Unnormalized')
            plt.plot(werrs_normalized, label='Normalized')
            plt.ylim(0, min(max(werrs_normalized + werrs), 5))
            plt.legend()
        if w_eps_comparison:
            newfigure('eps vs X.T w')
            pred_MAP = learner.predict_MAP(learner.X_test)
            pred_bayes = learner.predict(learner.X_test)
            for i in range(min(20, len(learner.Y_test))):
                plt.plot(i, (learner.Y_test[i] + 1) / 2, 'rx', label='real' if i == 0 else None, markersize=15)
                plt.plot(i, pred_MAP[1][1][i], 'go', label='epsilon_MAP' if i == 0 else None)
                plt.plot(i, pred_MAP[1][0][i],
                         'bo', label='X.T w_MAP' if i == 0 else None)
                plt.plot(i, pred_MAP[0][i], color='m',
                         label='prediction_MAP' if i == 0 else None, marker='8')

                plt.plot(i, pred_bayes[i], 'co', label='prediction' if i == 0 else None)

            plt.plot([-1, i + 1], [0.5, 0.5], 'black', linestyle='--')
            plt.xlim([-.5, i + 0.5])
            plt.title('Influences of eps and w')
            plt.legend()

    _, fig = newfigure('Scores')
    ax = plt.gca()
    plot_scores(learner, ax=ax)


def plot_scores(learner, ax=None, legend_=True):
    if 'scores_over_time' not in learner.run_info:
        print('NE')
        return
    scores = learner.run_info['scores_over_time']
    if len(scores) < 2:
        print('TS')
        return
    if ax is None:
        _ = plt.figure()
        ax = plt.gca()
    labels = []
    for key_i, key in enumerate(sorted(scores[0])):
        if 'bin.' in key: continue
        if key == 'Log Conf':
            continue
            ax2 = ax.twinx()
            labels.append(ax2.plot([scores[j][key] for j in range(len(scores))], label=key, color=colors[key_i])[0])
            ax2.set_ylabel(key)
        else:
            labels.append(ax.plot([scores[j][key] for j in range(len(scores))], label=key, color=colors[key_i])[0])
    labellabels = [l.get_label() for l in labels]
    labellabels[labellabels.index('F1')] = '$F_1$'
    ax.set_xlabel('Epochs')
    if legend_:
        ax.legend(labels, labellabels, fontsize=12, loc=8)


def plot_elbo(ax1, elbos, learner, no_axis_labels=False, legend_=True):
    convo_rad = 10

    def convo(arr):
        return np.convolve(arr, np.ones((convo_rad,)) / convo_rad, mode='same')[convo_rad // 2:-convo_rad // 2]

    mini_batches = learner.use_batches()
    cost_name = 'ELBO'
    if 'MAP' in learner.dist_params['qw']:
        cost_name = '$\\mathcal{L}$'
    ax1.plot([sum(pos) for pos in elbos], '-', alpha=0.4 if mini_batches else 1,
             color=colors[0], label=None if mini_batches else cost_name)
    if mini_batches:
        ax1.plot(convo([sum(pos) for pos in elbos]),
                 label=cost_name, color=colors[0])
    weight_term_label = '$- KL(q(\\beta) || p(\\beta))$'
    if 'MAP' in learner.dist_params['qw']:
        weight_term_label = '$\\log p(\\beta)$'
    for i, name in enumerate(['$ E_{q} [\\log  p( y | f, \\omega, \\beta)]$', '$- KL(q(\\omega) || p(\\omega))$',
                              '$- KL(q(u) || p(u))$', weight_term_label]):
        ax1.plot([pos[i] for pos in elbos], '--', color=colors[i + 1],
                 alpha=0.4 if mini_batches else 1, label=None if mini_batches else name)
        if mini_batches: ax1.plot(convo([pos[i] for pos in elbos]), '--', label=name, color=colors[i + 1])
    if no_axis_labels:
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])
    if legend_:
        ax1.legend(fontsize=12, loc=8)
    return i


def est_vs_real_weights_plot(ax, learner, w_true, legend_=True):
    plt.plot(w_true, label="true")
    if 'MV' in learner.dist_params['qw']['type']:
        yerr = np.diag(learner.dist_params['qw']['var'])
    elif 'MF' in learner.dist_params['qw']['type']:
        yerr = learner.dist_params['qw']['var']
    else:
        yerr = np.zeros_like(w_true)
    norma_factor = sum(np.abs(w_true)) / max(sum(np.abs(learner.get_expected_w())), 1E-3)
    for i, w in enumerate(learner.get_expected_w()):
        ax.errorbar(i, w * norma_factor, yerr=yerr[i] * norma_factor, marker='.', linestyle='None',
                     color='g', label='Normalized' if i == 0 else None)
        ax.errorbar(i, w , yerr=yerr[i], marker='o', linestyle='None',
                     color='r', label='Est' if i == 0 else None)
    title = 'True W vs est W'
    ax.set_ylim(1.5 * min(np.min(w_true), 0), 1.5 * max(np.max(w_true), 1))
    #ax.set_title(title)
    if legend_: ax.legend()
    return i


def plot_xz(X, Z, y):
    for z_ in Z.T:
        plt.plot(z_[0], z_[1], "bx")
    for xi, x_ in enumerate(X.T):
        plt.plot(x_[0], x_[1], marker="o", color="red" if y[xi] > 0 else "green", markerfacecolor='none')

    plt.show()


def plot_grid(fn):
    poss_crits, poss_rbf_ls, poss_rbf_vars, res_dict = read_in_grid(fn)

    for crit in poss_crits:
        if crit == ' AVG Prec.':
            continue
        imagename = image_path('{}:crit:{}-plot'.format(fn, crit))
        newfigure(imagename)
        print(crit)
        curr_resdict = {key:
                            {key2:
                                 res_dict[key][key2][sorted((res_dict[key][key2]).keys())[-1]][crit] for
                             key2 in poss_rbf_ls}
                        for key in poss_rbf_vars}
        df = pd.DataFrame(curr_resdict)
        sns.heatmap(df,
                    xticklabels=df.columns.values.round(3),
                    yticklabels=df.index.values.round(3),
                    annot=len(poss_rbf_ls) <= 5)
        plt.title(crit)
        plt.savefig(imagename, format='png')
    imagename = image_path('{}:crit:{}-plot'.format(fn, 'convergence at x'))
    newfigure(imagename)
    print('Plotting convergence')
    res_dict = {key:
                    {key2:
                         sorted((res_dict[key][key2]).keys())[-1] for
                     key2 in res_dict[key]}
                for key in res_dict}
    df = pd.DataFrame(res_dict)
    sns.heatmap(df,
                xticklabels=df.columns.values.round(3),
                yticklabels=df.index.values.round(3),
                annot=len(poss_rbf_ls) <= 5)
    plt.title('Convergence plot')
    plt.savefig(imagename, format='png')
    plt.show()
    return imagename


def read_in_grid(fn):
    res_dict = pickle.load(open(fn, 'rb'))
    poss_rbf_vars = sorted(res_dict.keys())
    poss_rbf_ls = sorted(res_dict[poss_rbf_vars[0]].keys())
    poss_crits = sorted((res_dict[poss_rbf_vars[0]][poss_rbf_ls[0]]
    [sorted(res_dict[poss_rbf_vars[0]][poss_rbf_ls[0]].keys())[0]]).keys())
    for key in poss_rbf_vars:
        for key2 in poss_rbf_ls:
            if 'Conf Score' in res_dict[key][key2].keys():
                res_dict[key][key2] = {0: {key: np.nan for key in poss_crits}}
    return poss_crits, poss_rbf_ls, poss_rbf_vars, res_dict


def examine_single_dot_in_grid(fn):
    poss_crits, poss_rbf_ls, poss_rbf_vars, res_dict = read_in_grid(fn)
    chosendict = res_dict
    params = []
    for keys, name in [[poss_rbf_vars, 'variance'], [poss_rbf_ls, 'lengthscale']]:
        print('Which {} do you want? Options are: \n {}'.format(name, keys))
        text = float(input("prompt"))
        params.append(text)
        print('You chose "{}"'.format(text))
        chosendict = chosendict[text]

    x_vals = sorted(chosendict.keys())
    for crit in poss_crits:
        newfigure(crit)
        title = 'Var:{}-Ls:{}'.format(params[0], params[1])
        plt.ylabel(crit)
        plt.plot(x_vals, [chosendict[pos][crit] for pos in x_vals])
        plt.title(title)

    plt.show()


def plot_kernel_param_hist(learner):
    kernel_param_history = learner.kernel_param_history
    if len(kernel_param_history.keys()) == 0: return
    title = 'Kernel parameter history'
    newfigure(title)
    fig, ax = plt.subplots(num='Kernel parameter history')
    for key in sorted(kernel_param_history.keys()):
        if kernel_param_history[key][0] is not None and not key == 'lr':
            start = kernel_param_history[key][0]
            ax.plot(np.array(kernel_param_history[key]) - start, label='{:10s} startpoint: {:.3f}'.format(key, start))

    ax.set_ylabel('Kernel parameter')
    ax.set_xlabel('Epochs')
    fig.legend()
    plt.title('Kernel parameter history')
    hyperheat = False
    if hyperheat:
        title = 'Heatmap Hyperparameter optimization'
        newfigure(title)
        hyheat = learner.hyper_heat
        df = pd.DataFrame(data=hyheat['data'], index=hyheat['ls'], columns=hyheat['rbf'])
        sns.heatmap(df, norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03))
        # plt.plot(kernel_param_history['length_scale'], kernel_param_history['rbf'])
        y_scaler = lambda x: (x - hyheat['rbf'][0]) / (hyheat['rbf'][1] - hyheat['rbf'][0])
        x_scaler = lambda x: (x - hyheat['ls'][0]) / (hyheat['ls'][1] - hyheat['ls'][0])
        for i in range(len(kernel_param_history['length_scale'])):
            plt.text(y_scaler(kernel_param_history['rbf'][i]), x_scaler(kernel_param_history['length_scale'][i]),
                     str(i))
        plt.xlabel('RBF')
        plt.ylabel('Length scale')


def save_all_figures(path=None):
    global all_titles
    if path is None: path = image_folder()
    for name in all_titles:
        plt.figure(name)
        plt.savefig(os.path.join(path, 'figure:{}.pdf'.format(name)), format='pdf')
    if not core.math.debug():
        plt.close('all')
    all_titles = []


def clean_labels(clfs):
    res = []
    for clf in clfs:
        if clf == 'None':
            clf = 'XGPC'
        if 'only w' in clf:
            clf = clf.replace('only w', 'only $\\beta$')
        if 'only eps' in clf:
            clf = clf.replace('only eps', 'only $f$')
        res.append(clf)
    return res


def plot_exp_results(fn, cost_plot=True, scores=None, do_list=None, dont_list=None):
    def score_bar_plot(df, fs=12):
        # This func makes a barplot the way I like it
        # It only plots F1 and ROC AUC Score
        curr_clfs = score_df_.axes[0]
        c_colors = [get_color(key) for key in curr_clfs]
        c_colors = [key['color'] for key in c_colors]
        ax = score_df_.plot.bar(subplots=True, rot=35, legend=False, title=['', ''], fontsize=fs,
                                color=[c_colors, c_colors])
        ax[1].set_xticklabels(clean_labels(curr_clfs))
        for a in ax:
            for p in a.patches:
                a.annotate('{:.3f}'.format(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005), fontsize=fs)
        ax[0].set_ylabel('ROC AUC', fontsize=fs + 2)
        ax[1].set_ylabel('$F_1$', fontsize=fs + 2)

    xp_rstls = np.load(os.path.join('experiment_logs', fn + '.npy')).item()
    if cost_plot:
        learners = []
        for key in xp_rstls['learner_paths']:
            learners.append(pickle.load(open(os.path.join(key, 'learner'), 'rb')))

        for l in learners:
            eval_plots(l)
            plt.savefig('cost.pdf', format='pdf')
            plt.show()

    score_df = xp_rstls['scores'].T
    clfs = score_df.axes
    score_df_ = score_df[['ROC AUC', 'F1']]
    score_df_ = score_df_.drop(
        [x for x in clfs[0] if any(y in x for y in ['only', 'SGDClassifier', 'Black Box', 'Sparse'])])
    clfs = score_df_.axes
    score_df_ = score_df_.drop([x for x in clfs[0] if 'MAP' == x[:3]])
    score_df_.sort_index(axis=0, inplace=True, ascending=False)
    score_bar_plot(score_df_)
    plt.show()


if __name__ == '__main__':
    fig, ax = plt.subplots(1, 1, sharex=True)
    x = range(20)
    y = np.sin(x)
    yerrs = np.random.uniform(0.1, 0.5, np.shape(x))
    fill_between_errbar(ax, x, y, yerrs)
    plt.show()
