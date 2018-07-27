from ctr.data import Data
from ctr.solver import Solver
import numpy as np
import tensorflow as tf
import pandas as pd
import json, ast

def run(name):
    config = json.loads(open("../config.json").read())
    conf_param = config["param"]
    conf_path = config["path"]

    i_id = pd.read_csv(filepath_or_buffer='/u/halle/chenyua/home_at/IJCAI2018/IJCAI2018-backup/data/raw_data/round1_ijcai_18_test_a_20180301.zip', compression="zip", sep='\s', header=0)[["instance_id"]]
    N = i_id.shape[0]

    solver = Solver()

    pred_list = []

    session_conf = tf.ConfigProto(device_count={'CPU' : 1, 'GPU' : 0}, allow_soft_placement=True, log_device_placement=False)

    with tf.Session(config=session_conf) as sess:
        print("restoring trained {0} model".format(name))
        saver = tf.train.import_meta_graph("../sessions/" + name + "/" + name + '-83100.meta')
        saver.restore(sess, "../sessions/" + name + "/" + name + '-83100')

        loss = tf.get_collection("loss")[0]

        valid = Data(with_valid=True,
                     with_train=False,
                     with_test=False,
                     with_saprse=False,
                     with_labels=True)

        valid.load_data(dense_path_train=conf_path["dense_path_train"],
                        sparse_path_train=conf_path["sparse_path_train"],
                        label_path_train=conf_path["label_path_train"],
                        dense_path_valid=conf_path["dense_path_valid"],
                        sparse_path_valid=conf_path["sparse_path_valid"],
                        label_path_valid=conf_path["label_path_valid"],
                        sparse_feats_list_path=conf_path["sparse_feats_list_path"])

        valid.define_holders(from_restore=True)

        valid_loss = solver.get_global_loss(loss=loss,
                                            batch_size=ast.literal_eval(conf_param["valid_batch_size"]),
                                            data_size=valid.num_valid,
                                            get_feed_fn=valid.get_valid_feed_dict)
        print("model {0} with valid loss {1}".format(name, valid_loss))

        test = Data(with_valid=False,
                    with_train=False,
                    with_test=True,
                    with_saprse=False,
                    with_labels=False)
        
        test.load_data(dense_path_test=conf_path["test_dense_path"],
                       sparse_path_test=conf_path["test_sparse_path"],
                       sparse_feats_list_path=conf_path["sparse_feats_list_path"])

        test.define_holders(from_restore=True)

        pred = tf.get_default_graph().get_tensor_by_name("predicts/predicts:0")

        preds = solver.get_predicts(pred, 512, test.num_test, test.get_test_feed_dict)

    print("pred length is {0}, num test instance is {1}".format(len(preds), N))
    i_id["predicted_score"] = pd.Series(data=preds.flatten())
    i_id.to_csv(path_or_buf='./test_a/' + name + '_predicts_{0:5f}.txt'.format(valid_loss), sep=" ", index=False, line_terminator='\n')

if __name__ == "__main__":
    run("CN_3")
