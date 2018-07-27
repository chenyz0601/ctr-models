import numpy as np
import pandas as pd
import tensorflow as tf
from ctr.data import Data
from tqdm import tqdm

class Solver:
    def __init__(self):
        self.loss_train_history = []
        self.loss_valid_history = []
        self.list_logits = []
        self.phase = None
        self.sess = None
        self.logits_avg=None
        self.tst_logits_avg=None
        return
    
    def train(self,
              sess,
              loss,
              data,
              minimizer,
              phase,
              learning_rate_tensor,
              global_step_tensor,
              logits=None,
              pred=None,
              write_summary=False,
              train_summary=None,
              val_summary=None,
              writer=None,
              saver=None,
              save_path=None,
              save_valid_loss=0.0822,
              save_valid_loss_avg=0.0825,
              save_period=None,
              auto_stop=True,
              batch_size=512,
              valid_batch_size=512,
              learning_rate_init=1e-5,
              learning_rate_min=1e-5,
              learning_rate_decay=0.1,
              decay=0.99,
              log_period=50, 
              max_train_step=np.inf,
              with_ema=False,
              tol=2, 
              verbose=True):
        
        assert isinstance(data, Data)

        self.phase = phase
        self.sess = sess

        last_v_loss = np.inf
        patients = 0

        loss_train = 0.
        loss_valid = 0.

        # Training cycle
        go_out = False
        learning_rate = max(learning_rate_init, learning_rate_min)
        print("start training")
        self.sess.run(tf.assign(learning_rate_tensor, learning_rate))
        print("assign learning rate with {0:5f}".format(learning_rate_tensor.eval()))
        for step in tqdm(range(max_train_step)):
#         for step in range(max_train_step):
            if go_out:
                break
            # Loop over all batches
            # for i in tqdm(range(num_batches)):
            if global_step_tensor.eval() > max_train_step:
                go_out = True
                break
            # Run optimization op (backprop) and cost op (to get loss value)
            f_dict = data.get_batch_feed_dict(batch_size)
            # set to train phase
            self.sess.run(tf.assign(self.phase, True))
            self.sess.run(minimizer, feed_dict=f_dict)
            
            # write into summary
            """
            if (1+global_step_tensor.eval()) % log_period == 0:
                if write_summary:
                    train_summ = self.sess.run(train_summary, feed_dict=f_dict)
                    val_f_dict = data.get_valid_feed_dict(valid_batch_size)
                    val_summ = self.sess.run(val_summary, feed_dict=val_f_dict)
                    writer.add_summary(train_summ, global_step_tensor.eval())
                    writer.add_summary(val_summ, global_step_tensor.eval())
            """
            # compute loss for train and valid data
            if (1+global_step_tensor.eval()) % log_period == 0:
                if data.with_train:
                    loss_train = self.get_global_loss(loss, batch_size, data.num_train, data.get_batch_feed_dict)
                    if loss_train < save_valid_loss:
                        print("maybe overfit")
                        break
                    self.loss_train_history.append(loss_train)
                if data.with_valid:
                    # get the loss on valid and on one of its random subset
                    loss_valid = self.get_global_loss(loss, valid_batch_size, 0, data.num_valid, data.get_valid_feed_dict)
                    
                    # check valid loss to decide change of learning rate
                    if loss_valid < last_v_loss:
                        patients = 0
                    else:
                        patients += 1
                    last_v_loss = loss_valid
                    if patients > tol:
                        if learning_rate <= 2*learning_rate_min:
                            if auto_stop:
                                print("maybe overfit!")
                                predicts = self.get_predicts(pred=pred, batch_size=valid_batch_size, get_feed_fn=data.get_test_feed_dict, data_size=data.num_test)
                                np.save(save_path+"pred_{0:5f}".format(loss_valid), predicts)
                                break
                            else:
                                pass
                        else:
                            learning_rate = max(learning_rate_min, learning_rate*learning_rate_decay)
                            self.sess.run(tf.assign(learning_rate_tensor, learning_rate))
                            print("assign learning rate to {0:5f}".format(learning_rate_tensor.eval()))
                        
                    if with_ema:
                        # compute current valid logits and update the exponential moving average
                        cur_logits = self.get_logits(logits, valid_batch_size, data.num_valid, data.get_valid_feed_dict)
                        if self.logits_avg is None:
                            self.logits_avg = cur_logits
                        else:
                            self.logits_avg = np.add(np.multiply(decay, self.logits_avg), np.multiply((1.-decay), cur_logits))
                        # compute current test logits and update the exponential moving average
                        cur_tst_logits = self.get_logits(logits, valid_batch_size, data.num_test, data.get_test_feed_dict)
                        if self.tst_logits_avg is None:
                            self.tst_logits_avg = cur_tst_logits
                        else:
                            self.tst_logits_avg = np.add(np.multiply(decay, self.tst_logits_avg), np.multiply((1.-decay), cur_tst_logits))
                        loss_valid_avg = self.get_loss_from_logits(self.logits_avg, data.labels_valid)
                        
                        if loss_valid_avg < save_valid_loss_avg:
                            print("saving tst pred with avg valid loss {0:5f}".format(loss_valid_avg))
                            tst_predicts = self.get_predicts_from_logits(self.tst_logits_avg)
                            np.save(save_path+"avg_pred_{0:5f}".format(loss_valid_avg), tst_predicts)
                    
                    if verbose:
                        try:
                            print('step: {0}, valid loss: {1:5f}, valid loss avg: {2:5f}'.format(global_step_tensor.eval(), loss_valid, loss_valid_avg))
                        except:
                            print('step: {0}, valid loss: {1:5f}'.format(global_step_tensor.eval(), loss_valid))
                    
                    # set threshold to write predicts and logits out
                    if loss_valid < save_valid_loss:
                        print("saving pred and logits with valid loss {0:5f}".format(loss_valid))
                        predicts = self.get_predicts(pred=pred, batch_size=valid_batch_size, get_feed_fn=data.get_test_feed_dict, data_size=data.num_test)
                        cur_logits = self.get_logits(logits=logits, batch_size=valid_batch_size, data_size=data.num_valid, get_feed_fn=data.get_valid_feed_dict)
                        # s_path = saver.save(self.sess, save_path+"_{0:5f}".format(loss_valid), global_step=global_step_tensor)
                        np.save(save_path+"pred_{0:5f}".format(loss_valid), predicts)
                        np.save(save_path+"logits_{0:5f}".format(loss_valid), cur_logits)  
                     
                    # conpute average loss
                    """
                    if logits is not None:
                        cur_logits = self.get_logits(logits, valid_batch_size, data.num_valid, data.get_valid_feed_dict)
                        self.list_logits.append(cur_logits)
                        self.list_logits = self.list_logits[-num_avg:]
                        loss_valid_avg = self.get_loss_from_logits(self.list_logits, data.labels_valid)
                    """

        return self.loss_train_history, self.loss_valid_history

    def get_global_loss(self, 
                        loss,
                        batch_size,
                        start,
                        data_size, 
                        get_feed_fn):
        # set to test phase
        loss_ = 0.
        self.sess.run(tf.assign(self.phase, False))
        num_batches = int(data_size / batch_size)
        for i in range(num_batches):
            _dict = get_feed_fn(batch_size=batch_size, 
                                 start=start+i*batch_size,
                                 end=start+(i+1)*batch_size)
            loss_ +=loss.eval(feed_dict=_dict)

        _dict = get_feed_fn(batch_size=batch_size, 
                            start=start+num_batches*batch_size,
                            end=start+data_size)
        loss_ += loss.eval(feed_dict=_dict)

        loss_ = loss_/data_size

        return loss_
    
    def get_logits(self, 
                   logits, 
                   batch_size, 
                   data_size, 
                   get_feed_fn):
        self.sess.run(tf.assign(self.phase, False))
        logits_list = []
        num_batches = int(data_size / batch_size)
        for i in range(num_batches):
            _dict = get_feed_fn(batch_size=batch_size, 
                                 start=i*batch_size,
                                 end=(i+1)*batch_size)
            logits_list.append(logits.eval(feed_dict=_dict))

        _dict = get_feed_fn(batch_size=batch_size, 
                            start=num_batches*batch_size,
                            end=data_size)
        logits_list.append(logits.eval(feed_dict=_dict))
        _logits = np.concatenate(logits_list)
        return _logits
    
    def get_loss_from_logits(self, x, z):
        out = x.copy()
        out = out.reshape([out.shape[0], 1])
        z = np.reshape(np.asarray(z, dtype=np.float), out.shape)
        out[out<0.] = 0.
        out = out - x*z + np.log(1. + np.exp(-np.abs(x)))
        return np.mean(out)

    def get_predicts_from_logits(self, x):
        return 1./(1.+np.exp(-x))

    def get_predicts(self, 
                     pred,
                     batch_size,
                     get_feed_fn,
                     instance_id=None,
                     data_size=None,
                     path=None):
        # set to test phase
        self.sess.run(tf.assign(self.phase, False))
        if instance_id is None:
            if not isinstance(data_size, int):
                raise ValueError("miss input of data_size or instance_id")
        else:
            data_size = instance_id.shape[0]
        pred_list = []
        num_batches = int(data_size / batch_size)
        for i in range(num_batches):
            _dict = get_feed_fn(batch_size=batch_size, 
                                 start=i*batch_size,
                                 end=(i+1)*batch_size)
            pred_list.append(pred.eval(feed_dict=_dict))

        _dict = get_feed_fn(batch_size=batch_size, 
                            start=num_batches*batch_size,
                            end=data_size)
        pred_list.append(pred.eval(feed_dict=_dict))
        _pred = np.concatenate(pred_list)
        print("pred length is {0}, num test instance is {1}".format(len(_pred), data_size))

        if instance_id is None:
            return _pred
        elif path is not None:
            instance_id["predicted_score"] = pd.Series(data=_pred.flatten())
            instance_id.to_csv(path_or_buf=path, sep=" ", index=False, line_terminator='\n')
