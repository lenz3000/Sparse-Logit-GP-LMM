import csv
import datetime

import numpy as np
import pickle
from sklearn.datasets import load_svmlight_file
import random
from scipy import sparse
import scipy.io
from sklearn.preprocessing import StandardScaler
import os

import core.math
from core import kernels, utils


cur_dir = os.path.dirname(__file__)
def dataset_folder():
    return os.path.join(cur_dir,'../datasets')


def dataset_path(fn):
    return os.path.join(dataset_folder(), fn)


def toy_data_folder():
    return 'toy'


def get_toy_data_fn(n, d, non_zeros, rand, lin, rbf, l_scale, white, n_test, extra_noise=0, confounder=False):
    lin, rbf, l_scale, white = [x if x is not None else 0 for x in [lin, rbf, l_scale, white]]
    fn = '({}x{}(+{}))_{}non_zero_{}weights_kernel:lin:{:.3E}_rbf:({:.3E},{:.3E})_noise:{:.3E}_classification{}'.format(
        d, n, n_test, non_zeros, 'rand_' if rand else '', lin, rbf, l_scale, white,'' if not confounder else 'confounder{}'.format(confounder)
    )
    if extra_noise > 0:
        fn += 'extra_noise:{}'.format(extra_noise)
    return fn


def load_toy_data(n=200, d=50, non_zeros=10, rand=False, lin=-3, rbf=-2, l_scale=-1.5, white=np.log(0.01),
                  n_test=200, w_true=None, load_saved=True, extra_noise=0, abs_confounder=False):
    """
    function that checks if this data-sets exists already,
                         if yes load that dataset
                         if not create it, save it and return it
    :param w_true: If set those weights are used to create the dataset, in this case no data-set is loaded,
                                                    but a new one is created and not saved
    :param n: Number of Samples
    :param d: Number of dimensions
    :param non_zeros: Number of the first entries in w that are non-zero
    :param rand: Boolean that decides if non-zero entries are randomly set or ones
    :param l_scale: length_scale for rbf kernel
    :param lin: Factor for linear cov-matrix
    :param rbf: Factor for uniform cov-matrix
    :param n_test: Number of test-points
    :return: Datapoints [DxN], labels [Nx1], true weights [Dx1], Covariance Matrix [NxN]

    """

    def create_dataset():
        return create_toy_data(n=n, d=d, non_zeros=non_zeros, rand=rand, lin=lin, rbf=rbf, l_scale=l_scale, white=white,
                               classes=True, n_test=n_test, w_true=w_true, extra_noise=extra_noise, abs_confounder=abs_confounder)

    if w_true is None and load_saved:
        fn = os.path.join(dataset_folder(), toy_data_folder(), get_toy_data_fn(n, d, non_zeros, rand, lin, rbf, l_scale,
                                                                               white, n_test,
                                                                               extra_noise=extra_noise,
                                                                               confounder=abs_confounder))
        if not utils.exists(fn):
            # doesn't exist
            data_set = create_dataset()
            data_set['data_path'] = fn
            pickle.dump(data_set, open(fn, 'wb'))
        else:
            data_set = pickle.load(open(fn, 'rb'))
    else:
        data_set = create_dataset()
    return data_set


def create_toy_data(n, d, non_zeros, rand, lin, rbf, l_scale, white, classes=True,
                    n_test=200, w_true=None, extra_noise=0, tries=3, abs_confounder=0):
    """
    Creates Toy dataset
    :param w_true: If set those weights are used to create the dataset
    :param classes: Bool for classification or regression
    :param n: Number of Samples
    :param d: Number of dimensions
    :param non_zeros: Number of the first entries in w that are non-zero
    :param rand: Boolean that decides if non-zero entries are randomly set or ones
    :param l_scale: length_scale for rbf kernel
    :param lin: Factor for linear cov-matrix
    :param rbf: Factor for uniform cov-matrix
    :param n_test: Number of test-poiunts
    :return: Datapoints [DxN], labels [Nx1], true weights [Dx1], Covariance Matrix [NxN]

    """
    try:
        X = np.random.uniform(-1, 1, (d, n + n_test))
        kernel = kernels.Kernel_Handler(lin=lin, rbf=rbf, length_scale=l_scale, white=white)
        print('Creating Data with Kernel_params: lin:{}, rbf_var:{}, l_s:{}, white:{}'.format(lin, rbf, l_scale, white))
        sigma = kernel(X.T)
        if extra_noise > 0:
            sigma += np.random.uniform(-extra_noise, extra_noise, sigma.shape)
        errs = np.reshape(np.random.multivariate_normal(np.zeros(n + n_test), sigma), (n + n_test, 1))
        if w_true is None:
            if rand:
                w = np.random.uniform(-2, 2, (d, 1))
            else:
                w = np.ones((d, 1))
            w[non_zeros:, :] = np.zeros((d - non_zeros, 1))
        else:
            w = w_true
        Y = np.dot(X.T, w) + errs
        if classes:
            Y = np.sign(Y)
        data_dict = {}
        train = [X[:, :-n_test], Y[:-n_test]]
        test = [X[:, -n_test:], Y[-n_test:]]
        if abs_confounder:
            mask = np.random.choice([True, False], n)
            train[0][:, mask] -= abs_confounder
            data_dict['side_info'] = {'train': mask[np.newaxis, :].astype(float), 'test':np.zeros_like(test[1]).T}

        data_dict['train'] = train
        data_dict['test'] = test
        data_dict['w_true'] = w
        data_dict['kernel'] = kernel
        data_dict['errors'] = errs
        data_dict['name'] = 'Toy'
        return data_dict
    except np.linalg.linalg.LinAlgError:
        if tries <= 0:
            print('I\'ve tried it 3 times... SVD does not converge')
            raise np.linalg.linalg.LinAlgError
        else:
            return create_toy_data(n, d, non_zeros, rand, lin, rbf, l_scale, white, classes=classes,
                                   n_test=n_test, w_true=w_true, extra_noise=extra_noise, tries=tries - 1)


def handle_multiple_classes(class_dict, name):
    """
    This function gets
    :param class_dict: a dict of the samples for every class with the class being the key,
    the outlier class needs to be "0"
    :param name: Name of the dataset
    :return: Train and dataset, where none of the classes in training are in the test set (TODO?)
    """
    # Getting all classes
    classes = list(class_dict.keys())
    # Remove the 0 class
    classes = [class_ for class_ in classes if not class_ == 0]

    # Randomly sample three quarters of the classes for training
    training_classes = random.sample(classes, (len(classes) * 9) // 10)
    test_classes = [class_ for class_ in classes if class_ not in training_classes]
    no_class = class_dict[0]
    no_class_n = no_class.shape[0]

    # Separating the 0 class data in the test/train ratio of the classes
    test_indices = random.sample(range(no_class_n),
                                 int(0.5 * no_class_n * sum([class_dict[class_].shape[0] for class_ in test_classes])
                                     / sum([class_dict[class_].shape[0] for class_ in classes])))


    sparse_bool = sparse.issparse(no_class[0, :])

    def vstack(x):
        if sparse_bool:
            return sparse.vstack(x, format='csr')
        return np.vstack(x)

    def hstack(x):
        if sparse_bool:
            return sparse.hstack(x, format='csr')
        return np.hstack(x)

    def zeros(shape):
        if sparse_bool:
            return sparse.lil_matrix(shape)
        return np.zeros(shape)

    # Putting the training set together and out in the 0/1 labels
    train_zeros = [i for i in range(no_class_n) if i not in test_indices]
    train_zeros = random.sample(train_zeros, len(train_zeros))
    train_set = vstack([no_class[i, :] for i in range(len(train_zeros))])
    train_set_y = np.zeros((len(train_zeros), 1), dtype=train_set.dtype)
    class_num = len(classes)
    train_set_side_info = zeros((class_num, len(train_set_y)))
    for class_ in training_classes:
        train_set = vstack([train_set, class_dict[class_]])
        train_set_y = np.concatenate([train_set_y, np.ones((class_dict[class_].shape[0], 1), dtype=train_set.dtype)],
                                     axis=0)
        side_info = zeros((class_num, class_dict[class_].shape[0]))
        side_info[classes.index(class_), :] = 1
        train_set_side_info = hstack([train_set_side_info, side_info])
    data_set = {'train': [train_set.T, train_set_y]}  # , 'side_info': {'train': train_set_side_info}}
    test_set = vstack([no_class[i, :] for i in test_indices])
    test_set_y = np.zeros((len(test_indices), 1), dtype=train_set.dtype)
    test_side_info = zeros((class_num, len(test_indices)))
    for class_ in test_classes:
        test_set = vstack([test_set, class_dict[class_]])
        test_set_y = np.concatenate([test_set_y, np.ones((class_dict[class_].shape[0], 1), dtype=train_set.dtype)],
                                    axis=0)
        side_info = zeros((class_num, class_dict[class_].shape[0]))
        side_info[classes.index(class_), :] = 1
        test_side_info = hstack([test_side_info, side_info])
    data_set['test'] = [test_set.T, test_set_y]
    # data_set['side_info']['test'] = test_side_info
    # if not sparse_bool:
    #    for key in data_set['side_info']: data_set['side_info'][key] = np.asmatrix(data_set['side_info'][key])
    scale_data_set(data_set)
    data_set['name'] = name
    return data_set


def scale_data_set(data_set):
    if not sparse.issparse(data_set['train'][0]):
        # Scaling leads to non-sparse entries
        scaler = StandardScaler()
        data_set['train'][0] = scaler.fit_transform(data_set['train'][0].T).T
        if len(data_set['test'][1]) > 0:
            data_set['test'][0] = scaler.transform(data_set['test'][0].T).T
        data_set['scaler'] = scaler
        if 'side_info' in data_set:
            side_scaler = StandardScaler()
            data_set['side_info']['train'] = side_scaler.fit_transform(data_set['side_info']['train'].T).T
            if len(data_set['test'][1]) > 0:
                data_set['side_info']['test'] = side_scaler.transform(data_set['side_info']['test'].T).T
            data_set['side_info']['scaler'] = side_scaler

    else:
        # Adding one columns for intercept weights
        data_set['test'][0] = sparse.vstack([np.ones((1, data_set['test'][0].shape[-1])), data_set['test'][0]],
                                            format='csr')
        data_set['train'][0] = sparse.vstack([np.ones((1, data_set['train'][0].shape[-1])),
                                              data_set['train'][0]], format='csr')


def load_drebin(class_info=False, **kwargs):
    data = load_svmlight_file(dataset_path('drebin/drebin.libsvm'))
    if core.math.debug():
        for i in range(5): print('DEBUG')
        limit = 10000
        data = list(data)
        data[0] = data[0][:limit, :2000]
    else:
        limit = len(data[1])
    data_dict = {'X': data[0], 'y': data[1]}
    if not class_info:
        return reformat_data_set(data_dict, 'drebin', **kwargs)
    # This reads in the files from idx_sets, which are the multiclass labels

    class_ids = {}
    for dirpath, dirnames, filenames in os.walk(dataset_path('drebin/idx_sets')):
        for filename in [f for f in filenames if f.endswith(".txt")]:
            class_members = list(open(os.path.join(dirpath, filename), 'r'))
            # For some text-files the indices are only seperated by ' ' instead of '/n'
            if len(class_members) == 1: class_members = class_members[0].split(' ')
            class_ids[filename[:-4]] = np.asarray(class_members, dtype=int)
            class_ids[filename[:-4]] = np.asarray([i for i in class_ids[filename[:-4]] if i < limit], dtype=int)
    data_dict['classes'] = class_ids
    class_dict = {}
    for class_ in data_dict['classes']:
        class_dict[class_] = sparse.vstack([data[0][i, :] for i in data_dict['classes'][class_]], format='csr')
    class_dict[0] = class_dict.pop('benign_IDX')
    np.save('drebin_families', class_dict)
    return handle_multiple_classes(class_dict, 'drebin', **kwargs)


def load_tbc(**kwargs):
    data = scipy.io.loadmat(dataset_path('tbc/tbc_raw.mat'))
    data_dict = {'X': data['X'].T.astype(float), 'y': data['y']}
    data_dict['side_info'] = data['Z'].T.astype(float)
    data_dict['side dimension names'] = ['age', 'sex', 'race black', 'race white', 'race s. asian', 'race other asian',
                                         'other']
    return reformat_data_set(data_dict, 'tbc', **kwargs)


def load_arabidopsis():
    data = scipy.io.loadmat(dataset_path('arabidopsis/arabidopsis.mat'))
    return reformat_data_set({'X': sparse.csc_matrix(data['X'].astype(float)), 'y': data['y'].astype(float)},
                             'arabidopsis')


def load_cancer():
    data = scipy.io.loadmat(dataset_path('ASU/kProstate_GE.mat'))
    return reformat_data_set({'X': data['X'], 'y': data['Y'].astype(float) - 1},
                             'Prostate Cancer')


def get_dataset(data):
    if isinstance(data, str):
        return pickle.load(open(data, 'rb'))
    else:
        return data()


def reformat_data_set(data, name, ratio=1, seed=True, test_ratio=.05, **kwargs):
    '''
    Separates training und test-set, takes care of side info
    :param data: dict with "X" [NxD] and "y" entries, optionally "side_info" [NxD']
    :param name: dataset name
    :param kwargs: further arguments thar are passed to the classifier
    :return:
    '''
    data_set = {}
    if kwargs is not None:
        data_set = kwargs

    X = data['X']
    y = np.ravel(data['y'])
    if core.math.debug():
        for i in range(5): print('DEBUG')
        X = X[:200, :100]
        y = y[:200]
    if seed:
        random.seed(0)
    else:
        random.seed(datetime.datetime.now())
    if ratio < 1:
        selected = random.sample(range(np.shape(X)[0]), int(ratio * np.shape(X)[0]))
        X = X[selected, :]
        y = y[selected]

    n = np.shape(X)[0]

    test_indices = random.sample(range(n), int(test_ratio * n))
    all_side_info = None
    if 'side_info' in data:
        all_side_info = data['side_info'].copy()

    if sparse.isspmatrix(X):
        test = [sparse.vstack([X[i, :] for i in test_indices], format='csc').T, np.take(y, test_indices)]

        train = [sparse.vstack([X[i, :] for i in range(n) if i not in test_indices], format='csc').T,
                 np.delete(y, test_indices)]

        if all_side_info is not None:
            data_set['side_info'] = {'train':
                                         sparse.vstack([all_side_info[i, :] for i in range(n) if i not in test_indices],
                                                       format='csc').T,
                                     'test':
                                         sparse.vstack([all_side_info[i, :] for i in test_indices], format='csc').T}
    else:
        test = [np.take(X, test_indices, axis=0).T, np.take(y, test_indices)]
        train = [np.take(X, [i for i in range(n) if i not in test_indices], axis=0).T, np.delete(y, test_indices)]
        if np.isnan(train[0]).any() or np.isnan(test[0]).any():
            # Replacing NAN with mean of respective dimension
            means = np.nanmean(train[0], axis=1)
            train_inds = np.where(np.isnan(train[0]))
            test_inds = np.where(np.isnan(test[0]))
            train[0][train_inds] = np.take(means, train_inds[0])
            test[0][test_inds] = np.take(means, test_inds[0])

        if all_side_info is not None:
            data_set['side_info'] = {'train':
                                         np.take(all_side_info, [i for i in range(n) if i not in test_indices],
                                                 axis=0).T,
                                     'test': np.take(all_side_info, test_indices, axis=0).T}
    if test_ratio > 0 and np.var(test[1]) == 0:
        print('No variance in test class repeating sampling ')
        print(test[1])
        return reformat_data_set(data, name, ratio, seed=seed + 1, **kwargs)
    data_set['train'] = train
    data_set['test'] = test
    scale_data_set(data_set)
    data_set['name'] = name
    for key in ['dimension names','side dimension names']:
        if key in data:
            data_set[key] = data[key]
    print('Loading of Data-set {} completed'.format(name))
    return data_set


def load_german():
    data = []
    with open(dataset_path('UCI/german.data-numeric'), newline='') as file:
        for line in file:
            while '  ' in line:
                line = line.replace('  ', ' ')
            line_ = line.split(' ')[1:-1]
            data.append(np.asarray(line_, 'int'))
    data = np.vstack(data)
    y = data[:, -1] - 1
    X = data[:, :-1]
    data_dict = {'X': X, 'y': y}
    return reformat_data_set(data_dict, 'German')


def load_microarray():
    X = []
    with open(dataset_path('UCI/micromass/pure_spectra_matrix.csv'), 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        for row in spamreader:
            X.append(np.asarray(row, dtype=float))
    X = np.vstack(X)
    Y = []
    with open(dataset_path('UCI/micromass/pure_spectra_metadata.csv'), 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        for row in spamreader:
            Y.append(row[0])
    data_set = {}
    for i, y in enumerate(Y[1:]):
        cur_class = 0
        if not any(n_c_name in y for n_c_name in ['JNH', 'NYV', 'BUT', 'EMD', 'AUG']):
            cur_class = y
        if cur_class not in data_set:
            data_set[cur_class] = [X[i, :]]
        else:
            data_set[cur_class].append(X[i, :])
    for key in data_set:
        data_set[key] = np.vstack(data_set[key])
    return handle_multiple_classes(data_set, 'microarray')


def load_SECOM(**kwargs):
    X = []
    with open(dataset_path('UCI/secom/secom.data'), 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ')
        for row in spamreader:
            X.append(np.asarray(row, dtype=float))
    X = np.vstack(X)
    y = []
    with open(dataset_path('UCI/secom/secom_labels.data'), 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ')
        for row in spamreader:
            y.append(-1 * int(row[0]))

    return reformat_data_set({'X': X, 'y': y}, 'SECOM', **kwargs)


def get_part(data, ratio):
    selected = random.sample(range(np.shape(data['train'][0])[1]), int(ratio * np.shape(data['train'][0])[1]))
    data['train'][0] = data['train'][0][:, selected]
    data['train'][1] = data['train'][1][selected]
    if 'side_info' in data:
        data['side_info']['train'] = data['side_info']['train'][:, selected]


def load_CINA():
    X = scipy.io.loadmat(dataset_path('CINA0/cina0_train.mat'))['X']
    with open(dataset_path('CINA0/cina0_train.targets'), 'r') as file:
        Y = [(int(f[:-1]) + 1) // 2 for f in file]

    return reformat_data_set({'X': X.astype(np.float), 'y': Y}, 'CINA')

def load_clean_adult(sensible=1, **kwargs):
    data = np.load(dataset_path('UCI/adult/clean_adult.npy')).item()
    dims = data['dimensions'][:-1]
    dat = data['data']
    Y = dat[:, -1]
    X = dat[:, :-1]
    set_name = 'Adult-' + str(sensible)
    nono_list = []
    if sensible > 0:
        nono_list += ['race', 'sex']
    if sensible > 1:
        nono_list.append('nati-')
    # Separating side info and normal info and dimension names
    clean_dim_mask = [not any(nono in dims[d] for nono in nono_list) for d in range(X.shape[1])]
    X_clean = np.vstack([X[:,d].T for d in range(X.shape[1]) if clean_dim_mask[d]]).T
    clean_dimnames = np.asarray(dims)[clean_dim_mask]
    res_dict = {'X': X_clean, 'y': Y, 'dimension names': clean_dimnames}
    if sensible > 0:
        side_info = np.vstack([X[:,d].T for d in range(X.shape[1]) if not clean_dim_mask[d]]).T
        side_dimnames = [d for d in dims if d not in clean_dimnames]
        res_dict['side_info'] = side_info
        res_dict['side dimension names'] = side_dimnames
    return reformat_data_set(res_dict, set_name, **kwargs)


def load(name, *args, **kwargs):
    if name.lower() == 'adult':
        return load_clean_adult(**kwargs)
    if name.lower() == 'ad_extra_clean':
        return load_clean_adult(sensible=1, **kwargs)
    if name.lower() == 'drebin':
        return load_drebin(*args, **kwargs)
    if name.lower() == 'tbc':
        return load_tbc(**kwargs)
    if name.lower() == 'toy':
        return load_toy_data(*args, **kwargs)

if __name__ == '__main__':
    data = load_clean_adult(sensible=0)


def load_dict_from_path(d_path):
    data = np.load(d_path)
    assert all(x in data for x in ['train', 'test']), \
        'Dataset has to be dict with entries "train" and "test"'
    assert all(len(data[x]) == 2 for x in ['train', 'test']), 'test and train must be lists of length 2 with X,y'
    if 'name' not in data:
        print('You did not specify a name for the dataset, setting to "Custom"')
        data['name'] = 'custom'
    return data
