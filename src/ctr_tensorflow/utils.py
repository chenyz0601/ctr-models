import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
from datetime import datetime

def restore_from_scope(scope):
    print("building saver to restore {0}".format(scope))
    restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    saver = tf.train.Saver(restore_vars)
    return saver

def get_init_opt(sess):
    print("getting uninitialized variables")
    uninit_vars = []
    for var in tf.all_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninit_vars.append(var)
    return tf.variables_initializer(uninit_vars)

def normalize_data(feats):
    """
    input:
        feats: np array, obs*feats
    output:
        np array
    """
    return (feats - feats.mean(axis=0)) / feats.std(axis=0)
    
def get_unique_random_idx(in_a, size, remove=True):
    if isinstance(in_a, int):
        a = np.arange(in_a)
    else:
        a = in_a
        if not isinstance(in_a, np.ndarray):
            raise ValueError("in_a should be either int or a numpy array")

    if len(a) <= size:
        if remove:
            return a, None
        else:
            return a
    else:
        idx = np.random.choice(a, size, replace=False)
        if remove:
            a_set = set(list(a))
            idx_set = set(list(idx))
            out_a = np.asarray(list(a_set-idx_set), dtype=np.int64)
            return idx, out_a
        else:
            return idx

def delete_rows_mask(high, indices):
    if high < len(indices):
        raise ValueError("high must larger than num of indices")
    else:
        mask = np.ones(high, dtype=bool)
        mask[indices] = False
        return mask

def get_raw_data(path=None, nrows=None):
    """
    input:
        path: path to raw data
        nrows: number of lines want to load, None means all
    output:
        data: dataframe without the feature names
        feature2idx: a dict from feature to idx
    """
    data = pd.read_csv(filepath_or_buffer=path,
                       sep='\s',
                       nrows=nrows,
                       header=None)
    feat2idx = {data.loc[0][i]:i for i in range(data.shape[1])}
    data = data.drop(0, axis=0)
    
    data[data.isnull()] = '-1'
    
    return data, feat2idx


def get_int_histograms(data=None, col=None, is_int=False, do_plot=False):
    """
    input:
        data: pandas dataframe, to be processed
        col: int, column to be counted
        is_numerical: bool, if the col data is numerical or not
        is_int: bool, if the col data is double
        acc: int, set the accuracy of double data
        do_plot: bool
    output:
        a dict of feature vs counts
        bad data rate
    """
    bad_rate = 0.
            
    m, n = data.shape
        
    data[[n-1]] = data[[n-1]].apply(pd.to_numeric)
    if is_int:
        positive = 1
        bad_data = -1
        data[[col]] = data[[col]].apply(pd.to_numeric)
    else:
        positive = '1'
        bad_data = '-1'
            
    id_set = pd.Series.unique(data.loc[:][col])        
    
    if is_int:
        sorted(id_set)

    count_list = []
    
    for item in id_set:
        tmp_data = data.loc[data[col] == item]
        try:
            count_list.append(tmp_data[[n-1]].apply(pd.value_counts).loc[positive].values[0] / tmp_data.shape[0]) 
        except:
            count_list.append(0.)
        
    out_dict = dict(zip(id_set, count_list))
    
    if do_plot:
        pd.DataFrame(out_dict, index=['feature {0}'.format(col)]).plot(kind='bar')
        plt.show()
        
    br = data[[col]].apply(pd.value_counts)/len(data[[col]])
    
    try:
        br = br.loc[bad_data].values[0]
    except:
        br = 0.
    
    print('rate of bad feature 14 is: ', br)
    
    return out_dict, br

"""
pred1 = csv1['predicted_score']
pred2 = csv2['predicted_score']
pred3 = csv3['predicted_score']
list_of_preds = np.array([pred1, pred2, pred3]).T

def esemble(list_of_preds):
    x = np.mean(-np.log(1/list_of_preds - 1), axis=1)
    return 1 / (1 + np.exp(-x))

res = esemble(list_of_preds)
print(res)

csv_esemble['predicted_score'] = res
csv_esemble.to_csv('/Users/yz/esemble_2.txt', index=False, sep=' ')
"""
def esemble(list_of_preds):
    x = np.mean(-np.log(1/list_of_preds - 1), axis=1)
    return 1 / (1 + np.exp(-x))
