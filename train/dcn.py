import numpy as np
from ctr import Data, CtrModel, Solver
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

def run(name, d, _list):
    for idx, struct in enumerate(_list):
        print("building model {0} with {1}".format(name, struct))

        tf.reset_default_graph()

        d.define_holders(from_restore=False)

        model = CtrModel(holder_dict=d.feats_holder_dict,
                sparse_dim_dict=d.sparse_dim_dict,
                methods_list=["CN", "DNN"],
                dcn_degree=3,
                drop_rate=None,
                dnn_h_list=struct,
                dnn_act_fn=tf.nn.relu,
                embed_w_scale=1e2,
                nn_w_scale=1e2,
                # nn_b_scale=1e2,
                lr_w_scale=1e2,
                # lr_b_scale=1e2,
                embed_base=10.,
                embed_exp=0.25)

        loss, opt = model.get_loss_and_minimizer(labels=d.labels_holder, method="momentum", clip_by=50.)

        solver = Solver()

        d.with_train = False

        with tf.Session() as sess:
        # write graph
        # writer = tf.summary.FileWriter('./logs/'+name, sess.graph)
            sess.run(tf.global_variables_initializer())
            train_loss_hist, valid_loss_hist = solver.train(sess=sess,
                                                            loss=loss,
                                                            logits=model.logits,
                                                            pred=model.predicts,
                                                            learning_rate_tensor=model.learning_rate_tensor,
                                                            data=d,
                                                            minimizer=opt,
                                                            auto_stop=False,
                                                            phase=model.phase_tensor,
                                                            global_step_tensor=model.global_step_tensor,
                                                            learning_rate_init=1e-4,
                                                            learning_rate_min=1e-6,
                                                            save_path="./results/0402/{0}/".format(name),
                                                            save_valid_loss=0.0820,
                                                            with_ema=False,
                                                            max_train_step=10000,
                                                            tol=1,
                                                            log_period=100)
        
if __name__ == "__main__":
    # st = [[512,512],[1024, 1024, 1024],[512, 512, 512, 512],[1024, 1024, 1024, 1024],[512, 512, 512, 512, 512, 512]]
    # st = [[1024, 1024, 1024], [1024, 1024, 1024]]
    # st = [1, 2, 3, 4, 5, 6, 7, 8]
    st = [[128, 128, 128]]
    d = prepare_data()
    run("DCN", d, st)
