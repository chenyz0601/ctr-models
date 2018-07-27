import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import sparse as sp
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2
from datetime import datetime
import itertools
from .utils import get_unique_random_idx, delete_rows_mask, normalize_data

class RawData:
    data_clean=False
    def __init__(self):
        self.user = ["user_id", "user_gender_id", "user_occupation_id", "user_age_level", "user_star_level"]
        self.context = ["day", "hour", "day_hour", "context_page_id"]
        self.item = ["item_id", "item_brand_id", "item_city_id", "item_category_id", "item_price_level",
                "item_sales_level", "item_collected_level", "item_pv_level"]
        self.shop = ["shop_id", "shop_review_num_level", "shop_review_positive_rate",
                "shop_star_level", "shop_score_service", "shop_score_delivery", "shop_score_description"]
        self.shop_cat = ["shop_id", "shop_star_level"]
        """
        self.list_numeric = ["shop_review_num_level", "shop_review_positive_rate", "shop_star_level", 
                             "shop_score_service", "shop_score_delivery", "shop_score_description", 
                             "item_price_level", "item_sales_level", "item_collected_level", "item_pv_level",
                             "user_star_level"]
        """
        self.list_numeric = ["shop_review_positive_rate", "shop_score_service", "shop_score_delivery", "shop_score_description"]
        self.list_cat = ["user_id", "user_gender_id", "user_occupation_id", "user_age_level", "user_star_level", 
                         "day", "hour", "day_hour", "context_page_id", "item_id", "item_brand_id", "item_city_id",
                         "item_category_id", "item_price_level", "item_sales_level", "item_collected_level",
                         "item_pv_level", "shop_id", "shop_review_num_level", "shop_star_level", 
                         "int_shop_review_positive_rate", "int_shop_score_delivery",  "int_shop_score_description", "int_shop_score_service"]
    
    def load_data(self, path_train, path_test):
        X_test = pd.read_csv(path_test, compression="zip", header=0, sep=" ")
        self.instance_id = X_test[["instance_id"]]
        X_train = pd.read_csv(path_train, compression="zip", header=0, sep=" ").drop_duplicates(keep=False)
        # cancat the data in form of train, valid, test
        tmp_t = pd.to_datetime(X_train["context_timestamp"], unit="s") + pd.Timedelta(hours=8)
        X_valid = X_train[tmp_t.dt.day==24]
        X_train = X_train[tmp_t.dt.day<24]
        self.num_train = X_train.shape[0]
        self.num_valid = X_valid.shape[0]
        self.num_train_all = self.num_train + self.num_valid
        Y_train = X_train.pop("is_trade")
        Y_valid = X_valid.pop("is_trade")
        self.Y = pd.concat([Y_train, Y_valid])
        
        self.X_raw = pd.concat([X_train, X_valid, X_test])
        self.N = self.X_raw.shape[0]
        
        t = pd.to_datetime(self.X_raw.pop("context_timestamp"), unit="s")
        self.X_raw["day"] = t.dt.day
        self.X_raw["hour"] = t.dt.hour
        self.X_raw["day_hour"] = pd.Series((self.X_raw.day - 18) * 24 + self.X_raw.hour)
        self.X_raw = self.X_raw.drop("instance_id", axis=1)
        self.X_raw = self.X_raw.drop("context_id", axis=1)
        
        ### TODU ###
        ### deal with ipl and pcp ###
        self.ipl = self.X_raw.pop("item_property_list")
        self.pcp = self.X_raw.pop("predict_category_property")
        return 
    
    def wash_data(self):
        # split item category list
        # and create a new column "item_category_id" in X_raw
        self.process_icl()

        # fill missing data in double feats
        self.fill_missing()
        # discetize double feats
        # add to X_raw as column "int_"+name
        self.discretize_double()

        # hash all cat feats to int by take module of num of unique
        self.hash_ids()
        self.data_clean = True
        
    def get_dense_feats(self,
                        with_counts=False,
                        with_ndprior=False,
                        rate_chi2=0.2,
                        rate_corr=0.9,
                        ndprior_list=["user_id", "item_id", "shop_id",
                                     ["user_id", "item_id"], ["user_id", "item_brand_id"],
                                     ["user_gender_id", "item_brand_id"], ["user_occupation_id", "item_brand_id"], 
                                     ["user_age_level", "item_price_level"], ["user_id", "shop_id"],
                                     ["item_id", "shop_id"]]):
        # first clean data
        if not self.data_clean:
            self.wash_data()
            self.data_clean = True
            
        # get pure numeric feats
        X_numeric = self.normalize(self._numeric_feats())
        
        # get general prob feats and keep some of them by chi2 check
        if with_counts:
            X_prob = self._prob_feats()
            k_chi2 = int(X_prob.shape[1] * rate_chi2)
            X_prob = self.normalize(self.Chi2_selector(X_prob, k_chi2))
        
        # get daily convert rate for each feature
        if with_ndprior:
            X_prior = self.normalize(self._ndprior_convert_count(_list=ndprior_list))
            # concat to sparse feats, normalize and select by corr check
            
        if with_counts and with_ndprior:
            X = pd.concat([X_numeric, X_prob, X_prior], axis=1)
        elif with_counts:
            X = pd.concat([X_numeric, X_prob], axis=1)
        elif with_ndprior:
            X = pd.concat([X_numeric, X_prior], axis=1)
        else:
            X = X_numeric

#         X = self.normalize(X)
#         k_corr = int(rate_corr * X.shape[1])
#         X = self.NLargestCorr_selector(X, k_corr)
        return X
    
    def get_sparse_feats(self):
        if not self.data_clean:
            self.wash_data()
        out = self._one_hot_feats()
        out.update({"ipl": self._n_hot_ipl()})
        return out
                    

    def get_split_X(self, X, with_valid=True):
        if isinstance(X, dict):
            if with_valid:
                train_ = {}
                valid_ = {}
                test_ = {}
                for key, x in X.items():
                    train_.update({key: x[:self.num_train]})
                    valid_.update({key: x[self.num_train: self.num_train_all]})
                    test_.update({key: x[self.num_train_all:]})
                return train_, valid_, test_
            else:
                train_ = {}
                test_ = {}
                for key, x in X.items():
                    train_.update({key: x[:self.num_train_all]})
                    test_.update({key: x[self.num_train_all:]})
                return train_, test_
        if with_valid:
            X_train = X[:self.num_train].values
            X_valid = X[self.num_train: self.num_train_all].values
            X_test = X[self.num_train_all:].values
            return X_train, X_valid, X_test
        else:
            X_test = X[self.num_train_all:].values
            X_train = X[:self.num_train_all].values
            return X_train, X_test
        
    def get_split_Y(self):
        return self.Y[:self.num_train], self.Y[self.num_train:]
        
    def _numeric_feats(self):
        out = self.X_raw[self.list_numeric]
        self.X_raw = self.X_raw.drop(self.list_numeric, axis=0)
        return out
    
    def _prob_feats(self):
        prob_feats = pd.DataFrame({"dummy": ()})
        cat_list = list(set().union(self.user, self.context, self.shop, self.item))
        for cat_feat in cat_list:
            prob_feats["count_"+cat_feat] = self.X_raw.groupby(cat_feat)[cat_feat].transform("size")
            
        for l1, l2 in itertools.combinations([self.user, self.context, self.shop, self.item], 2):
            for name1, name2 in itertools.product(l1, l2):
                name = "joint_"+name1+"_"+name2
                prob_feats[name] = self.X_raw.groupby([name1, name2])[name1].transform("size")
        return prob_feats.drop("dummy", axis=1)

    def _ndprior_convert_count(self, _list):
        out = pd.DataFrame()
        trn = self.X_raw.iloc[:self.num_train]
        trn["is_trade"] = self.Y
        for day in range(18, 24):
            _trn = trn[trn["day"]<=day]
            for feat in _list:
                group = _trn.groupby(feat)
                count = group["is_trade"].agg(np.size)
                convert = group["is_trade"].agg(np.sum)
                if not isinstance(feat, list):
                    index = self.X_raw[feat].values
                    _count = pd.Series(count[index].values)
                    _count = _count.fillna(0)
                    _convert = pd.Series(convert[index].values)
                    _convert = _convert.fillna(0)
                    out["count_{0}_{1}".format(day, feat)] = _count
                    out["convert_{0}_{1}".format(day, feat)] = _convert
                else:
                    tuple_ = list(zip(*self.X_raw[feat].values.T))
                    index = pd.MultiIndex.from_tuples(tuple_, names=feat)
                    _count = pd.Series(count[index].values)
                    _count = _count.fillna(0)
                    _convert = pd.Series(convert[index].values)
                    _convert = _convert.fillna(0)
                    out["count_{0}_{1}_{2}".format(day, feat[0], feat[1])] = _count
                    out["convert_{0}_{1}_{2}".format(day, feat[0], feat[1])] = _convert
        return out.reindex(self.X_raw.index)
    
    def _n_hot_ipl(self, save_path=None):
        ipl = self.ipl.str.split(pat=";", expand=True)
        ipl = ipl.apply(lambda x: x.str[-3:]).fillna("1001")
        ipl = np.asarray(ipl.astype(int)).flatten("F")
        ipl = ipl.reshape([ipl.shape[0], 1])
        raw_encoded = OneHotEncoder(sparse=True).fit_transform(ipl)
        encoded = 0
        for i in range(100):
            encoded = encoded + raw_encoded[i*self.N: (i+1)*self.N]
            
        value = encoded.data
        value[value>1.] = 1.
        n_hot = sp.csr_matrix((value, encoded.indices, encoded.indptr), encoded.shape)
        if save_path is None:
            return n_hot
        else:
            sp.save_npz(save_path, n_hot)
            
    def _one_hot_feats(self):
        if not self.data_clean:
            self.wash_data()
            self.data_clean = True
        encoder = OneHotEncoder(sparse=True)
        out = {}
        for feat in self.list_cat:
            tmp = self.X_raw.pop(feat).values
            out.update({feat: encoder.fit_transform(np.reshape(tmp, [tmp.shape[0],1]))})
        
        return out
    
    def normalize(self, X):
        return X.apply(lambda x: (x-x.mean())/x.std())
    
    def VarianceThreshold_selector(self, X, threshold):
        ### select feats with variance > threshold, feats should be normalized
        ### X is train+test
        mask = X.apply(lambda x: x.var())>=threshold
        return X.iloc[:, mask.values]
    
    def NLargestCorr_selector(self, X, k=200):
        ### select k largest correlation of feats between labels, feats should be normalized
        ### X is train+test
        f = lambda X, Y: X.apply(lambda x: x.corr(Y))  
        return X[f(X[:self.num_train_all], self.Y).nlargest(k).keys()]
    
    def Chi2_selector(self, X, k=150):
        ### select k best chi2 feats w.r.t. labels
        ### feats have to be postive now, like frequency
        selector = SelectKBest(chi2, k=k)
        _ = selector.fit_transform(X[:self.num_train_all], self.Y)
        return X[X.columns[selector.get_support(indices=True)]]
                        
    def hash_ids(self,
                 num_digits=5,
                 feats_list=None):
        feats_list = list(set(self.list_cat)-set(["day", "hour", "day_hour"]))
        N = int(10**num_digits)
        self.X_raw[feats_list] = self.X_raw[feats_list].apply(lambda x: x % min(x.unique().shape[0], N))
        
    def fill_missing(self,
                     feats_list=["shop_review_positive_rate", "shop_score_delivery",
                                    "shop_score_description", "shop_score_service"],
                     value_list=[1.0, 0.97, 0.98, 0.97]):
        if value_list is None:
            pass
        else:
            for i in range(len(feats_list)):
                self.X_raw[feats_list[i]].replace(-1, value_list[i])
                
    def discretize_double(self, num_digits=3):
        double_list = ["shop_review_positive_rate", "shop_score_delivery", 
                       "shop_score_description", "shop_score_service"]
        num = 10**num_digits
        for name in double_list:
            self.X_raw["int_"+name] = self.X_raw[name].apply(lambda x: int(x*num))
        
    def process_icl(self):
        icl = self.X_raw["item_category_list"].str.split(pat=";", expand=True)
        self.X_raw["item_category_id"] = icl[[1]].astype(int)
        self.X_raw = self.X_raw.drop("item_category_list", axis=1)
        
    def get_missing_rate(self, data):
        for name in data.keys():
            group = data.groupby(name)[name]
            try:
                miss_rate = group.count()["-1"]/group.count().sum()
                max_rate = group.count().max()/group.count().sum()
                min_rate = group.count().min()/group.count().sum()
                print("{0}: miss rate is {1:5f}, max rate is {2:5f}, min rate is {3:5f}".format(name, miss_rate, max_rate, min_rate))
            except KeyError:
                print("no missing data in {0}".format(name))

class Data:
    num_valid = None
    num_train = None
    num_test = None
    feats_valid_dict = {}
    feats_train_dict = {}
    feats_test_dict = {}
    labels_valid = None
    labels_train = None
    labels_holder = None
    feats_holder_dict = {}
    sparse_feats_list = ["user_id", "user_gender_id", "user_occupation_id", "user_age_level", "user_star_level", 
                         "day", "hour", "day_hour", "context_page_id", "item_id", "item_brand_id", "item_city_id",
                         "item_category_id", "item_price_level", "item_sales_level", "item_collected_level",
                         "item_pv_level", "ipl", "shop_id", "shop_review_num_level", "shop_star_level", 
                         "int_shop_review_positive_rate", "int_shop_score_delivery",  "int_shop_score_description", "int_shop_score_service"]
    dense_dim = None
    sparse_dim_dict={}
    idx_train = None
    idx_valid = None
    idx_test = None

    def __init__(self,
                 with_test,
                 with_valid,
                 with_train,
                 with_dense,
                 with_saprse,
                 with_labels):
        ### all inputs are bool
        self.with_test = with_test
        self.with_valid = with_valid
        self.with_train = with_train
        self.with_dense = with_dense
        self.with_sparse = with_saprse
        self.with_labels = with_labels
    
    def load_data(self,
                  dense_path_valid=None,
                  label_path_valid=None,
                  dense_path_train=None,
                  label_path_train=None,
                  dense_path_test=None,
                  sparse_path_train=None,
                  sparse_path_valid=None,
                  sparse_path_test=None,
                  suffix='.npz'):
        if self.with_train:
            try:
                dfeats_train = np.load(dense_path_train)
            except:
                dfeats_train = dense_path_train
            self.dense_dim = dfeats_train.shape[1]
            self.num_train = dfeats_train.shape[0]

        if self.with_valid:
            try:
                dfeats_valid = np.load(dense_path_valid)
            except:
                dfeats_valid = dense_path_valid
            self.dense_dim = dfeats_valid.shape[1]
            self.num_valid = dfeats_valid.shape[0]
            
        if self.with_test:
            try:
                dfeats_test = np.load(dense_path_test)
            except:
                dfeats_test = dense_path_test
            self.dense_dim = dfeats_test.shape[1]
            self.num_test = dfeats_test.shape[0]
            
        # try to deal with sparse feats
        if self.with_sparse:
            try:
                if self.with_train:
                    for key in self.sparse_feats_list:
                        tmp = sp.load_npz(sparse_path_train+key+suffix)
                        self.feats_train_dict.update({key: tmp})
                        self.sparse_dim_dict.update({key: tmp.shape[1]})
                        if not isinstance(tmp, sp.csr_matrix):
                            raise ValueError("works only for CSR format -- use .tocsr() first")
                            
                if self.with_valid:
                    for key in self.sparse_feats_list:
                        tmp = sp.load_npz(sparse_path_valid+key+suffix)
                        self.feats_valid_dict.update({key: tmp})
                        if not isinstance(tmp, sp.csr_matrix):
                            raise ValueError("works only for CSR format -- use .tocsr() first")

                if self.with_test:
                    for key in self.sparse_feats_list:
                        tmp = sp.load_npz(sparse_path_test+key+suffix)
                        self.feats_test_dict.update({key: tmp})
                        if not isinstance(tmp, sp.csr_matrix):
                            raise ValueError("works only for CSR format -- use .tocsr() first")
                        
            except:
                if self.with_train:
                    self.feats_train_dict = sparse_path_train
                    self.sparse_feats_list = list(sparse_path_train.keys())
                    self.sparse_dim_dict = {key: x.shape[1] for key, x in self.feats_train_dict.items()}
                if self.with_valid:
                    self.feats_valid_dict = sparse_path_valid
                if self.with_test:
                    self.feats_test_dict = sparse_path_test
            

        # get train and sparse data
        if self.with_train:
            if self.with_dense:
                self.feats_train_dict.update({"dense": dfeats_train})
            if self.with_labels:
                try:
                    self.labels_train = np.load(label_path_train).reshape((self.num_train, 1))
                except:
                    self.labels_train = label_path_train.reshape(self.num_train, 1)

        if self.with_valid:
            if self.with_dense:
                self.feats_valid_dict.update({"dense": dfeats_valid})
            if self.with_labels:
                try:
                    self.labels_valid = np.load(label_path_valid).reshape((self.num_valid, 1)) 
                except:
                    self.labels_valid = label_path_valid.reshape(self.num_valid, 1)

        if self.with_test:
            if self.with_dense:
                self.feats_test_dict.update({"dense": dfeats_test})

    def define_holders(self, from_restore, dense_dim=None):
        if dense_dim is None:
            dense_dim = self.dense_dim
        if not from_restore:
            if self.with_sparse:
                # define sparse placeholders
                with tf.name_scope('sparse_input'):
                    if self.sparse_feats_list is None:
                        raise ValueError("sparse feat dict should not be None, call load data")
                    for key in self.sparse_feats_list:
                        self.feats_holder_dict.update({key: tf.sparse_placeholder(dtype=tf.float32, 
                                                                                  name="sparse_holder_of_{0}".format(key))})
            # define dense placeholders
            with tf.name_scope('dense_input'):
                self.feats_holder_dict.update({"dense":
                                               tf.placeholder(dtype=tf.float32,
                                                              shape=[None, dense_dim],
                                                              name="dense_holder")})
            
            with tf.name_scope('label_input'):
                self.labels_holder = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="labels_holder")
        else:
            if self.with_sparse: 
                if self.sparse_feats_list is None:
                    raise ValueError("sparse feat dict should not be None, call load data")
                for key in self.sparse_feats_list:
                    idx = tf.get_default_graph().get_tensor_by_name("sparse_input/sparse_holder_of_{0}/indices:0".format(key))
                    values = tf.get_default_graph().get_tensor_by_name("sparse_input/sparse_holder_of_{0}/values:0".format(key))
                    shape = tf.get_default_graph().get_tensor_by_name("sparse_input/sparse_holder_of_{0}/shape:0".format(key))
                    self.feats_holder_dict.update({key: tf.SparseTensor(idx, values, shape)})

            self.feats_holder_dict.update({"dense": tf.get_default_graph().get_tensor_by_name("dense_input/dense_holder:0")})
            if self.with_labels:
                self.labels_holder = tf.get_default_graph().get_tensor_by_name("label_input/labels_holder:0")

            print("place holders are ready!")

        
    def get_sparse_feats_list(self, path):
        f = open(path, 'r')
        self.sparse_feats_list = f.read().splitlines()
        
    def get_sparse_values(self, data):
        tmp = data.tocoo()
        indices = np.asarray(tmp.nonzero(), dtype=np.int64).T
        values = tmp.data 
        shape = tmp.shape
        return tf.SparseTensorValue(indices, values, shape)
    
    def get_batch(self,
                  feats_dict,
                  batch_size,
                  phase,
                  start=None,
                  end=None):
        """
        get a mini batch of feats and labels randomly
        or from start to end
        """
        if start == None and end == None:
            if phase is "train":
                # get the indices and remove form idx_train
                if self.idx_train is None:
                    self.idx_train = self.num_train
                idx, idx_ = get_unique_random_idx(self.idx_train, size=batch_size, remove=True)
                self.idx_train = idx_
            elif phase is "valid":
                # get the indices and remove form idx_valid
                if self.idx_valid is None:
                    self.idx_valid = self.num_valid
                idx, idx_ = get_unique_random_idx(self.idx_valid, size=batch_size, remove=True)
                self.idx_valid = idx_

            elif phase is "test":
                # get the indices and remove form idx_valid
                if self.idx_test is None:
                    self.idx_test = self.num_test
                idx, idx_ = get_unique_random_idx(self.idx_test, size=batch_size, remove=True)
                self.idx_test = idx_

            out_dict = {}
            for key, value in feats_dict.items():
                try:
                    tmp = self.get_sparse_values(data=value[idx])
                    out_dict.update({key: tmp})
                except:
                    out_dict.update({key: value[idx, :]})
            return out_dict, idx
        else:
            out_dict = {}
            for key, value in feats_dict.items():
                try:
                    tmp = self.get_sparse_values(data=value[start:end])
                    out_dict.update({key: tmp})
                except:
                    out_dict.update({key: value[start:end, :]})
            return out_dict, np.arange(start, end)
            
    def get_batch_feed_dict(self,
                            batch_size=512,
                            start=None,
                            end=None):

        feats_dict, idx = self.get_batch(feats_dict=self.feats_train_dict,
                                            batch_size=batch_size,
                                            phase="train",
                                            start=start,
                                            end=end)
        out_dict = {}
        for key in feats_dict.keys():
            out_dict.update({self.feats_holder_dict[key]: feats_dict[key]})
        if self.with_labels:
            out_dict.update({self.labels_holder: self.labels_train[idx, :]})
        return out_dict
            
    def get_valid_feed_dict(self,
                            batch_size=512,
                            start=None,
                            end=None):

        feats_dict, idx = self.get_batch(feats_dict=self.feats_valid_dict,
                                         batch_size=batch_size,
                                         phase="valid",
                                         start=start,
                                         end=end)
        out_dict = {}
        for key in feats_dict.keys():
            out_dict.update({self.feats_holder_dict[key]: feats_dict[key]})
        if self.with_labels:
            out_dict.update({self.labels_holder: self.labels_valid[idx, :]})
        return out_dict
            
    def get_test_feed_dict(self,
                           batch_size=512,
                           start=None,
                           end=None):

        feats_dict, idx = self.get_batch(feats_dict=self.feats_test_dict,
                                         batch_size=batch_size,
                                         phase="test",
                                         start=start,
                                         end=end)
        out_dict = {}
        for key in feats_dict.keys():
            out_dict.update({self.feats_holder_dict[key]: feats_dict[key]})

        return out_dict
        
    def split(self, data, idx):
        try:
            return np.copy(data[idx, :]), np.delete(data, idx, axis=0)
        except:
            tmp = np.delete(data, idx)
            return np.reshape(np.copy(data[idx]), [len(idx), 1]), np.reshape(tmp, [tmp.shape[0], 1])
        
    def sparse_split(self, data, idx):
        if not isinstance(data, sp.csr_matrix):
            raise ValueError("works only for CSR format -- use .tocsr() first")
        return data[idx], self.delete_rows_csr(mat=data, indices=idx)
        
    def delete_rows_csr(self, mat, indices):
        """
        Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
        """
        if not isinstance(mat, sp.csr_matrix):
            raise ValueError("works only for CSR format -- use .tocsr() first")
        indices = list(indices)
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[indices] = False
        return mat[mask]
"""
class RawData:
    
    def __init__(self):
        pass
    
    def preprocess(self, train_path, test_path, header=0, nrows=None, valid_date=24, with_cats=False, save_path=None):
        
        double_list=['shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description']
        int_list = ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'user_age_level', 'user_star_level', 'shop_review_num_level', 'shop_star_level', 'day', 'hour']
        cat_list = ["user_id", "user_gender_id", "user_occupation_id", "context_page_id", "item_id", "item_brand_id", "item_city_id", "item_category_id", "shop_id"]

        train = pd.read_csv(train_path, compression="zip", header=0, sep=" ", nrows=nrows)
        test = pd.read_csv(test_path, compression="zip", header=0, sep=" ", nrows=nrows)

        labels = train["is_trade"]

        ## deal with the time information
        t = pd.to_datetime(test['context_timestamp'], unit='s')
        test["hour"] = t.dt.hour
        test["day"] = t.dt.day
        t = pd.to_datetime(train['context_timestamp'], unit='s')
        train["hour"] = t.dt.hour
        train["day"] = t.dt.day
        train = train.fillna(-1)
        test = test.fillna(-1)

        data = pd.concat([train, test])
        
        ##### TODO #####
        if with_cats:
            data_cat = data[cat_list]
    
        ## deal with item catrgory list
        icl = data["item_category_list"].str.split(pat=";", expand=True)
        data["item_category_id"] = icl[1]

        ## make double into darts 0.01
        double_level_list = []
        for name in double_list:
            new_name = name+"_level"
            double_level_list.append(new_name)
            data[new_name] = data[name].round(2)

        ## count feats list is int + cat
        _list = list(set().union(double_level_list, int_list, cat_list))

        data_new = pd.DataFrame({"A": []})
        ## get count feats
        count_list = []
        for count_ in _list:
            name = "count_"+count_
            count_list.append(name)
            data_new[name] = data.groupby(count_)[count_].transform("size")
        print("have {0} count feats".format(len(count_list)))

        ## get 2 bag feats
        _2bag_list = []
        for name1, name2 in itertools.combinations(_list, 2):
            name = "_2bag_"+name1+"_"+name2
            _2bag_list.append(name)
            data_new[name] = data.groupby([name1, name2])["instance_id"].transform("size")
        print("have {0} 2-bag feats".format(len(_2bag_list)))

        ## get 3 bag feats
        shop = ['shop_review_positive_rate_level', 'shop_score_service_level', 'shop_score_delivery_level', 'shop_score_description_level']
        item = ["item_brand_id", "item_city_id", "item_category_id", "shop_id", 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']
        user = ['user_age_level', 'user_star_level', "user_gender_id", "user_occupation_id"]
        context = ["context_page_id", 'day', 'hour']
        _3bag_list = []
        iter_list = [shop, item, user, context]
        ## 3 feats from 3 different cats
        for main1, main2, main3 in itertools.combinations(iter_list, 3):
            for name1, name2, name3 in itertools.product(main1, main2, main3):
                name = "_3bag_"+name1+"_"+name2+"_"+name3
                _3bag_list.append(name)
                data_new[name] = data.groupby([name1, name2, name3])["instance_id"].transform("size")
        print("have {0} 3-bag feats from 3 diff cats".format(len(_3bag_list)))

        ## 3 feats from 2 different cats
        for main1, main2 in itertools.combinations(iter_list, 2):
            for name1 in main1:
                for name2, name3 in itertools.combinations(main2, 2):
                    name = "_3bag_"+name1+"_"+name2+"_"+name3
                    _3bag_list.append(name)
                    data_new[name] = data.groupby([name1, name2, name3])["instance_id"].transform("size")
        print("have {0} 3-bag feats in total".format(len(_3bag_list)))

        ## 4 feats from 4 diff cats
        _4bag_list = []
        for name1, name2, name3, name4 in itertools.product(shop, item, user, context):
            name = "_4bag_"+name1+"_"+name2+"_"+name3+"_"+name4
            _4bag_list.append(name)
            data[name] = data.groupby([name1, name2, name3, name4])["instance_id"].transform("size")
        print("have {0} 4-bag feats".format(len(_4bag_list)))
        feats_list = list(set().union(double_list, int_list, count_list, _2bag_list, _3bag_list))
        data = pd.concat([data, data_new], axis=0, ignore_index=True)
        data = data[feats_list]
        data = (data-data.mean())/data.std()
        # data = data.dropna(axis=1, how="any")
        data = data.fillna(0)

        _train = data.iloc[:train.shape[0], :]
        _valid = _train[_train["day"]==24]
        _train = _train[_train["day"]<24]
        labels_train = _train["is_trade"]
        labels_valid = _valid["is_trade"]

        _test = data.iloc[train.shape[0]:, :]
        _test = _test[feats_list]

        labels_train = labels[_train.shape[0]:]
        labels_valid = labels[_train.shape[0]:train.shape[0]]

        np_train = np.array(_train.apply(pd.to_numeric))
        np_labels_train = np.array(labels_train.apply(pd.to_numeric))
        np_valid = np.array(_valid.apply(pd.to_numeric))
        np_labels_valid = np.array(labels_valid.apply(pd.to_numeric))
        np_test = np.array(_test.apply(pd.to_numeric))

        np.save(save_path+"train", np_train)
        np.save(save_path+"labels_train", np_labels_train)
        np.save(save_path+"valid", np_valid)
        np.save(save_path+"labels_valid", np_labels_valid)
        np.save(save_path+"test", np_test)

    def print_daily_trade_rate(self, train):
        labels = train["is_trade"]
        for day in train["day"].unique():
            cur_labels = labels[train["day"]==day]
            print("day: {0}, trade rate is: {1:5f}".format(day, cur_labels[cur_labels==1].shape[0] / cur_labels.shape[0]))

    def write_sparse_list(self, write_path):
        with open(write_path, 'w') as file_handler:
            for item in self.sparse_list:
                file_handler.write("{}\n".format(item))
    
    def save_labels(self, train_path=None, valid_path=None):

        self.labels_train = self.labels_train.fillna(value='-1')
        labels_train = np.asarray(self.labels_train.apply(pd.to_numeric))
        #########################################
        ### to do: try other fill-in strategy ###
        #########################################
        labels_train[np.where(labels_train == -1.)] = 0.

        if train_path is None:
            return labels_train
        else:
            np.save(train_path+'labels', labels_train)

        self.labels_valid = self.labels_valid.fillna(value='-1')
        labels_valid = np.asarray(self.labels_valid.apply(pd.to_numeric))
        #########################################
        ### to do: try other fill-in strategy ###
        #########################################
        labels_valid[np.where(labels_valid == -1.)] = 0.

        if valid_path is None:
            return labels_valid
        else:
            np.save(valid_path+'labels', labels_valid)

    def process_feats(self,
                      dtype="dense",
                      method="freq",
                      save_train=None,
                      save_valid=None,
                      save_test=None,
                      whitening=True,
                      stack=False):
        input:
            feat_list: list, feats need to be processed
            dtype: str, type of the feats, could be 'dense', 'sparse'
            save_path: str, path to save prcessed features
        output:
            np array of processed features
        if dtype is 'dense':
            dense_feats = np.asarray(self.raw_data[self.dense_list].apply(pd.to_numeric))
            if whitening:
                dense_feats = normalize_data(feats=dense_feats)
            if save_train is None:
                return dense_feats[:self.n_tr, :], dense_feats[self.n_tr : self.n_tr+self.n_valid, :], dense_feats[self.n_tr+self.n_valid:, :]
            else:
                np.save(save_train+dtype+'_feats',
                        dense_feats[:self.n_tr, :])
                
                np.save(save_valid+dtype+'_feats',
                        dense_feats[self.n_tr : self.n_tr + self.n_valid, :])
                if save_test is not None:
                    np.save(save_test+dtype+'_feats',
                            dense_feats[self.n_tr + self.n_valid :, :])
        elif dtype is 'sparse':
            self.sparse2vec(save_train=save_train,
                            save_valid=save_valid,
                            save_test=save_test,
                            method=method,
                            stack=stack)
        else:
            raise ValueError('feature type should be dense or sparse')
                
    def sparse2vec(self,
                   save_train=None,
                   save_valid=None,
                   save_test=None, 
                   method="freq",
                   stack=False):
        
        sp_list = []
        if method is "raw":
            for key in self.sparse_list:
                sp_list.append(self.do_sparse2vec(np.copy(self.raw_data[key])))
        elif method is "freq":
            for key in self.sparse_list:
                freq = self.raw_data.groupby(key)[key].transform('size')
                sp_list.append(self.do_sparse2vec(freq))
        else:
            raise ValueError("method should be raw or freq")
        
        if stack:
            hot_feats = sp.hstack(sp_list)

            if save_train is None:
                return hot_feats.tocsr()[:self.n_tr], hot_feats.tocsr()[self.n_tr:self.n_tr + self.n_valid], hot_feats.tocsr()[self.n_tr + self.n_valid :]
            else:
                sp.save_npz(save_train+'stacked_one_hot_feats', hot_feats.tocsr()[:self.n_tr])
                sp.save_npz(save_valid+'stacked_one_hot_feats', hot_feats.tocsr()[self.n_tr:self.n_tr + self.n_valid])
                if save_test is not None:
                    sp.save_npz(save_test+'stacked_one_hot_feats', hot_feats.tocsr()[self.n_tr + self.n_valid:])
        else:
            if save_train is None:
                return sp_list
            else:
                for i, key in enumerate(self.sparse_list):
                    sp.save_npz(save_train+key+'_one_hot_feats', sp_list[i].tocsr()[:self.n_tr])
                    sp.save_npz(save_valid+key+'_one_hot_feats', sp_list[i].tocsr()[self.n_tr:self.n_tr + self.n_valid])
                    if save_test is not None:
                        sp.save_npz(save_test+key+'_one_hot_feats', sp_list[i].tocsr()[self.n_tr + self.n_valid:])
    
    def do_sparse2vec(self, column):
        values = np.array(column)
        # integer encode
        
        integer_encoded = label_encoder.fit_transform(values)
        # binary encode
        onehot_encoder = OneHotEncoder(sparse=True)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        return onehot_encoded

class TestData:
    feats_dict = {}
    feats_holder_dict = {}

    def __init__(self,
                 dense_path,
                 sparse_path=None,
                 sparse_feats_list_path=None,
                 suffix='_one_hot_feats.npz'):
        dfeats = np.load(dense_path)
        dense_dim = dfeats.shape[1]

        try:
            # try to deal with sparse feats
            sparse_feats_list = self.get_sparse_feats_list(sparse_feats_list_path)
            for key in sparse_feats_list:
                tmp = sp.load_npz(sparse_path+key+suffix)
                self.feats_dict.update({key: tmp})
                if not isinstance(tmp, sp.csr_matrix):
                    raise ValueError("works only for CSR format -- use .tocsr() first")

            # get sparse placeholders
            for key in sparse_feats_list:
                idx = tf.get_default_graph().get_tensor_by_name("sparse_input/sparse_holder_of_{0}/indices:0".format(key))
                values = tf.get_default_graph().get_tensor_by_name("sparse_input/sparse_holder_of_{0}/values:0".format(key))
                shape = tf.get_default_graph().get_tensor_by_name("sparse_input/sparse_holder_of_{0}/shape:0".format(key))
                self.feats_holder_dict.update({key: tf.SparseTensor(idx, values, shape)})
        except:
            pass
                
        self.feats_dict.update({"dense": dfeats})
        # define dense placeholders
        self.feats_holder_dict.update({"dense": tf.get_default_graph().get_tensor_by_name("dense_input/dense_holder:0")})
        
    def get_sparse_feats_list(self, path):
        f = open(path, 'r')
        return f.read().splitlines()
        
    def get_sparse_values(self, data):
        tmp = data.tocoo()
        indices = np.asarray(tmp.nonzero()).T
        values = tmp.data 
        shape = tmp.shape 
        return indices, values, shape
    
    def get_batch_feed_dict(self, start, end):
        if type(start) is int and type(end) is int:
            out_dict = {}
            for key, value in self.feats_dict.items():
                try:
                    tmp1, tmp2, tmp3 = self.get_sparse_values(data=value[start:end])
                    out_dict.update({self.feats_holder_dict[key]: (tmp1, tmp2, tmp3)})
                except:
                    out_dict.update({self.feats_holder_dict[key]: value[start:end, :]})
            return out_dict
        else:
            raise TypeError('type of batch_size should be int')
"""

"""
input:
    train_path: path to raw train data
    test_path: path to raw test data
    header: int or None, col index of pandas dataframe
    nrows: number of lines want to load, None means all
    list_max: int, max cardinality of list features
    drop_list: list of col names, which will be replaced by some detailed features

self.dense_list = dense_list
train_data = pd.read_csv(filepath_or_buffer=train_path,
                         sep='\s',
                         nrows=nrows,
                         header=header)
test_data = pd.read_csv(filepath_or_buffer=test_path,
                        sep='\s',
                        nrows=nrows,
                        header=header)

# split valid from train
if not rand_valid:
    valid_time = datetime(2018, 9, valid_date).timestamp()
    valid_data = train_data[train_data['context_timestamp']>valid_time]
    train_data = train_data[train_data['context_timestamp']<=valid_time]
else:
    if isinstance(valid_rate, float):
        tmp = train_data.shape[0]
        valid_size = int(tmp * valid_rate)
        idx, idx_ = get_unique_random_idx(tmp, valid_size, remove=True) 
        valid_data = train_data.loc[idx, :]
        train_data = train_data.loc[idx_, :]
    else:
        raise ValueError("valid_rate should be float between (0, 1)")

# get the num of train and valid instance
self.n_tr = train_data.shape[0]
self.n_valid = valid_data.shape[0]
print("train num is {0}, valid num is {1}".format(self.n_tr, self.n_valid))

# copy  the train valid labels out
self.labels_train = pd.DataFrame.copy(train_data['is_trade'])
self.labels_valid = pd.DataFrame.copy(valid_data['is_trade'])

# drop labels from dataframe and concat with test data
# prepare for one hot encoding
data = pd.concat([train_data.drop('is_trade', axis=1), valid_data.drop('is_trade', axis=1), test_data]) 
print("concat_shape is {0}".format(data.shape))

# span timestamps to 'day', 'hour', 'minute'
t = pd.to_datetime(data['context_timestamp'], unit='s')
data['day'] = t.dt.day
data['hour'] = t.dt.hour
data['minute'] = t.dt.minute

# span list features with n_col cardinality
tmp1 = data['item_category_list'].str.split(pat=';', expand=True, n=-1)
n_col = min(list_max, tmp1.shape[1])
tmp11 = pd.DataFrame.copy(tmp1[list(range(n_col))])
tmp111 = pd.DataFrame(tmp11.values,
                      columns=['icl{0}'.format(i)
                               for i in range(tmp1.shape[1])])

tmp2 = data['item_property_list'].str.split(pat=';', expand=True, n=-1)
n_col = min(list_max, tmp2.shape[1])
tmp22 = pd.DataFrame.copy(tmp2[list(range(n_col))])
tmp222 = pd.DataFrame(tmp22.values,
                      columns=['ipl{0}'.format(i)
                               for i in range(n_col)])

tmp3 = data['predict_category_property'].str.split(pat=';', expand=True, n=-1)
n_col = min(list_max, tmp3.shape[1])
tmp33 = pd.DataFrame.copy(tmp3[list(range(n_col))])
tmp333 = pd.DataFrame(tmp33.values,
                      columns=['pcp{0}'.format(i)
                               for i in range(n_col)])

# drop redundent features from dataframe
data = data.drop(drop_list, axis=1)
# concat with spaned list features
data = pd.concat([data.reset_index(drop=True),
                  tmp111.reset_index(drop=True), 
                  tmp222.reset_index(drop=True),
                  tmp333.reset_index(drop=True)],
                  axis=1)

# fill nan to -1
data = data.fillna(value='-1')

# get sparse feats list
self.sparse_list = list(set(data.keys()) - (set(self.dense_list) - set(int_list)))
self.raw_data = data  ## self.raw_data is feats of train and test
"""
