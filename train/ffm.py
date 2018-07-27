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
        print("building model {0}".format(struct))

        tf.reset_default_graph()

        d.define_holders(from_restore=False)

        model = CtrModel(holder_dict=d.feats_holder_dict,
                sparse_dim_dict=d.sparse_dim_dict,
                methods_list=name,
                ffm_nv=20,
                drop_rate=None,
                dnn_h_list=struct,
                dnn_act_fn=tf.nn.relu,
                embed_w_scale=1e2,
                nn_w_scale=1e2,
                # nn_b_scale=1e2,
                lr_w_scale=1e2,
                # lr_b_scale=1e2,
                embed_base=10.,
                embed_exp=1/4.)

        loss, opt = model.get_loss_and_minimizer(labels=d.labels_holder, method="adam", clip_by=100.)

        solver = Solver()

        d.with_train = False

        lr_assign = tf.assign(model.learning_rate_tensor, 1e-5)

        with tf.Session() as sess:
        # write graph
        # writer = tf.summary.FileWriter('./logs/'+name, sess.graph)
            sess.run(tf.global_variables_initializer())
            sess.run(lr_assign)
            train_loss_hist, valid_loss_hist = solver.train(sess=sess,
                                                            loss=loss,
                                                            logits=model.logits,
                                                            pred=model.predicts,
                                                            data=d,
                                                            minimizer=opt,
                                                            auto_stop=False,
                                                            phase=model.phase_tensor,
                                                            global_step_tensor=model.global_step_tensor,
                                                            save_path="./results/0401/{0}/".format(name),
                                                            save_valid_loss=0.0825,
                                                            batch_size=256,
                                                            with_ema=False,
                                                            max_train_step=20000,
                                                            log_period=500)
        
if __name__ == "__main__":
    # st = [[512,512],[1024, 1024, 1024],[512, 512, 512, 512],[1024, 1024, 1024, 1024],[512, 512, 512, 512, 512, 512]]
    # st = [[1024, 1024, 1024], [1024, 1024, 1024]]
    # st = [1, 2, 3, 4, 5, 6, 7, 8]
    st = [[216, 128, 64]]
    d = prepare_data()
    run("FFM", d, st)
