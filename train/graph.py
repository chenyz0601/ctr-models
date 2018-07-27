import pandas as pd
import numpy as np
from ctr import Data, CtrModel, Solver, RawData
import tensorflow as tf

def prepare_data():
    print("preparing data")
    d = Data(with_labels=True, with_dense=True, with_saprse=True, with_train=True, with_valid=True, with_test=True)

    d.load_data(dense_path_train="../data/prep/24/train/dense.npy",
                dense_path_valid="../data/prep/24/valid/dense.npy", 
                dense_path_test="../data/prep/24/test/dense.npy", 
                label_path_train="../data/prep/24/train/labels.npy",
                label_path_valid="../data/prep/24/valid/labels.npy",
                sparse_path_train="../data/prep/24/train/",
                sparse_path_valid="../data/prep/24/valid/",
                sparse_path_test="../data/prep/24/test/")
    return d

def run(name, d):

    print("building {0} ...".format(name))

    tf.reset_default_graph()

    d.define_holders(from_restore=False)

    model = CtrModel(holder_dict=d.feats_holder_dict,
            sparse_dim_dict=d.sparse_dim_dict,
            methods_list=name,
            drop_rate=0.5,
            dnn_h_list=[[1024, 1024, 1024, 1024], [512, 512, 512]],
            embed_base=40.,
            embed_exp=1/4.)

    loss, opt = model.get_loss_and_minimizer(labels=d.labels_holder, clip_by=100.)

    solver = Solver()

    lr_assign = tf.assign(model.learning_rate_tensor, 1e-4)

    with tf.Session() as sess:
        # write graph
        writer = tf.summary.FileWriter('./logs/'+name, sess.graph)
        
if __name__ == "__main__":
    d = prepare_data()
    run("FFM", d)
