import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, l2_regularizer, l1_regularizer, xavier_initializer, dropout, batch_norm

class CtrModel:
    logits = None
    predicts = None
    learning_rate_tensor = None
    global_step_tensor = None
    # indicate is train or not
    phase_tensor = None 
    def __init__(self,
                 holder_dict,
                 sparse_dim_dict=None,
                 embed_base=6,
                 embed_exp=1./4.,
                 methods_list=['lr'],
                 drop_rate=None,
                 dnn_h_list=None,
                 dnn_act_fn=tf.nn.relu,
                 dcn_degree=4,
                 ffm_nv=10,
                 embed_reg="l2",
                 embed_w_scale=0.0,
                 nn_reg='l2',
                 nn_w_scale=0.0,
                 nn_b_scale=0.0,
                 lr_reg='l2',
                 lr_w_scale=0.0,
                 lr_b_scale=0.0):
        """
        d_holder: place holder of dense feats
        sp_holder: list of place holders of sparse feats
        methods_list: a list of strs indicate the type of feature
        """
        self.phase_tensor = tf.Variable(True, dtype=tf.bool, name="train_test_phase")
        ### embedding one hot feats
        embed_list = []
        with tf.variable_scope("embedding"):
            for key, holder in holder_dict.items():
                if key != "dense":
                    if embed_reg == 'l2':
                        w_reg = l2_regularizer(scale=embed_w_scale)
                    if lr_reg == 'l1':
                        w_reg = l1_regularizer(scale=embed_w_scale)
                    embed_list.append(self.sparse2embed(key=key,
                                                        n_v=sparse_dim_dict[key],
                                                        sp_holder=holder,
                                                        embed_base=embed_base,
                                                        embed_reg=w_reg,
                                                        embed_exp=embed_exp))
        embed_list.append(holder_dict["dense"])
        
        if lr_reg == 'l2':
            w_reg = l2_regularizer(scale=lr_w_scale)
            b_reg = l2_regularizer(scale=lr_b_scale)
        if lr_reg == 'l1':
            w_reg = l1_regularizer(scale=lr_w_scale)
            b_reg = l1_regularizer(scale=lr_b_scale)    


        ### concat dense and embedded feats
        with tf.name_scope('in_concat'):
            feats = tf.concat(embed_list, axis=1, name='in_concat')
        
        if isinstance(methods_list, list):
            last_feats_list = [self.extract_feats(feats=feats,
                                                  method=method,
                                                  drop_rate=drop_rate,
                                                  dnn_h_list=dnn_h_list,
                                                  dnn_act_fn=dnn_act_fn,
                                                  dcn_degree=dcn_degree,
                                                  nn_reg=nn_reg,
                                                  nn_w_scale=nn_w_scale,
                                                  nn_b_scale=nn_b_scale)
                               for method in methods_list]

            with tf.name_scope('out_concat'):
                self.last_feats = tf.concat(last_feats_list, axis=1, name='out_concat')
            
            self.logits = fully_connected(inputs=self.last_feats,
                                         activation_fn=None,
                                         num_outputs=1,
                                         weights_regularizer=w_reg,
                                         biases_regularizer=b_reg,
                                         scope='feats2logits')
        elif methods_list is "FFM":
            self.logits = self.extract_feats(feats, "FFM", n_v=ffm_nv)
        elif methods_list is "DeepFM":
            ffm_ = self.extract_feats(feats,
                                      "FFM",
                                      nn_reg=nn_reg,
                                      nn_w_scale=nn_w_scale,
                                      nn_b_scale=nn_b_scale,
                                      n_v = ffm_nv)
            dnn_ = self.extract_feats(feats=feats,
                                      method="DNN",
                                      drop_rate=drop_rate,
                                      dnn_h_list=dnn_h_list,
                                      dnn_act_fn=dnn_act_fn,
                                      nn_reg=nn_reg,
                                      nn_w_scale=nn_w_scale,
                                      nn_b_scale=nn_b_scale)
            dnn_ = fully_connected(inputs=dnn_,
                                         activation_fn=None,
                                         num_outputs=1,
                                         weights_regularizer=w_reg,
                                         biases_regularizer=b_reg,
                                         scope='feats2logits')
            self.logits = tf.add(ffm_, dnn_)
        else:
            raise ValueError("unrecogonizable methods!")

        with tf.name_scope('predicts'):
            self.predicts = tf.sigmoid(self.logits, name='predicts')

    def sparse2embed(self,
                     key,
                     n_v,
                     sp_holder,
                     embed_base,
                     embed_exp,
                     embed_reg=None,
                     trainable=True):
        ################################
        ######    to do init      ######
        ################################
        """
        input:
            key: int, indicate id of sparse feats
            sp_holder: sparse holder for sparse batch data
        output:
            embed_mat: matrix batch_size*n_e
        """
        with tf.variable_scope("field_{0}".format(key), reuse=tf.AUTO_REUSE):
            n_e = int(embed_base * (n_v ** embed_exp))
            W_e = tf.get_variable('W_e_{0}'.format(key),
                                  initializer=self.get_W_embedding_initializer(),
                                  regularizer=embed_reg,
                                  shape=[n_v, n_e],
                                  dtype=tf.float32,
                                  trainable=trainable)

            out = tf.sparse_tensor_dense_matmul(sp_holder, W_e)

        return out

    def get_W_embedding_initializer(self, value=None):
        if value == None:
            init = xavier_initializer()
        else:
            init = tf.constant(value)
        return init
    
    def get_loss_and_minimizer(self,
                               labels,
                               method="adam",
                               clip_by=None):
        """
        labels: a tf placeholder of all labels
        """
        with tf.name_scope('logistic_regression'):
            tmp = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=self.logits)
            loss_eval = tf.reduce_sum(tmp, name='lr_loss')
            loss = tf.reduce_sum(tmp)

        with tf.name_scope('train'):
            self.learning_rate_tensor = tf.Variable(1e-5, dtype=tf.float32, trainable=False, name="learning_rate")
            self.global_step_tensor = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
            
            if method == "adam":
                optimizer = tf.train.AdamOptimizer(self.learning_rate_tensor, name="adam_optimizer")
            elif method == "momentum":
                self.momentum_tensor = tf.Variable(0.9, dtype=tf.float32, trainable=False, name="monmentum")
                optimizer = tf.train.MomentumOptimizer(self.learning_rate_tensor, self.momentum_tensor, name="momt_optimizer")
    
            if clip_by is not None:
                self.clip_tensor = tf.Variable(clip_by, dtype=tf.float32, trainable=False, name="clip_norm")
                gradients, variables = zip(*optimizer.compute_gradients(loss))
                gradients, _ = tf.clip_by_global_norm(gradients, self.clip_tensor)
                mini_opt = optimizer.apply_gradients(zip(gradients, variables))
                increase_global_step_opt = tf.assign_add(self.global_step_tensor, 1)
                minimizer = [mini_opt, increase_global_step_opt]
                ### to do: add gradients clip ###
            else:
                minimizer = optimizer.minimize(loss, global_step=self.global_step_tensor, name="minimizer")

        return loss_eval, minimizer
        
    def extract_feats(self,
                       feats,
                       method,
                       drop_rate=None,
                       dnn_h_list=None,
                       dnn_act_fn=tf.nn.relu,
                       n_v=10,
                       dcn_degree=6,
                       nn_reg='l2',
                       nn_w_scale=0.0,
                       nn_b_scale=0.0):
        """
        feats, a tf placeholder in shape [batch_size, num_feats]
        """
        if nn_reg == 'l2':
            w_reg = l2_regularizer(scale=nn_w_scale)
            b_reg = l2_regularizer(scale=nn_b_scale)
        if nn_reg == 'l1':
            w_reg = l1_regularizer(scale=nn_w_scale)
            b_reg = l1_regularizer(scale=nn_b_scale)
        if method == 'lr':
            last_feats = feats
        elif method == 'DNN':
            last_feats = self.DNN(feats=feats,
                                  drop_rate=drop_rate,
                                  h_list=dnn_h_list,
                                  w_reg=w_reg,
                                  b_reg=b_reg,
                                  act_fn=dnn_act_fn)
        elif method == 'FullRes':
            last_feats = self.FullRes(feats=feats,
                                  drop_rate=drop_rate,
                                  h_list=dnn_h_list,
                                  w_reg=w_reg,
                                  b_reg=b_reg,
                                  act_fn=dnn_act_fn)
            
        elif method == 'ResNN':
            last_feats = self.ResNN(feats=feats,
                                  drop_rate=drop_rate,
                                  h_list=dnn_h_list,
                                  w_reg=w_reg,
                                  b_reg=b_reg,
                                  act_fn=dnn_act_fn)
            
        elif method == 'DenseNN':
            last_feats = self.DenseNN(feats=feats,
                                  drop_rate=drop_rate,
                                  h_list=dnn_h_list,
                                  w_reg=w_reg,
                                  b_reg=b_reg,
                                  act_fn=dnn_act_fn)
        
        elif method == 'CN':
            last_feats = self.CN(feats=feats,
                                 degree=dcn_degree,
                                 w_reg=w_reg,
                                 b_reg=b_reg)
        elif method == "FFM":
            last_feats = self.FFM(feats,
                                  n_v,
                                  w_reg=w_reg,
                                  b_reg=b_reg)
        else:
            raise TypeError('method should be lr, DNN, CN, FFM or FullRes')
        
        return last_feats

    def FFM(self,
            feats,
            n_v,
            w_reg,
            b_reg):
        with tf.variable_scope("FFM_linear"):
            linear = tf.layers.dense(feats,
                                units=1,
                                activation=None,
                                use_bias=True,
                                kernel_initializer=xavier_initializer(),
                                bias_initializer=tf.zeros_initializer(),
                                kernel_regularizer=w_reg,
                                bias_regularizer=b_reg,
                                trainable=True,
                                reuse=tf.AUTO_REUSE,
                                name="FFM_linear")
        with tf.variable_scope("FFM_quadratic"):
            quad1 = tf.layers.dense(feats,
                                units=n_v,
                                activation=None,
                                use_bias=False,
                                kernel_initializer=xavier_initializer(),
                                kernel_regularizer=w_reg,
                                trainable=True,
                                reuse=tf.AUTO_REUSE,
                                name="FFM_quad")
            quad2 = tf.layers.dense(feats,
                                units=n_v,
                                activation=None,
                                use_bias=False,
                                kernel_initializer=xavier_initializer(),
                                kernel_regularizer=w_reg,
                                trainable=True,
                                reuse=tf.AUTO_REUSE,
                                name="FFM_quad")
            quad = tf.expand_dims(tf.reduce_sum(tf.multiply(quad1, quad2), axis=1), 1)

        return tf.add(linear, quad)
    
    def DNN(self,
            feats,
            h_list,
            drop_rate,
            w_reg,
            b_reg,
            act_fn=tf.nn.relu):

        hidden_layer = feats

        for i, h in enumerate(h_list):
            with tf.variable_scope("dense_layer_{0}".format(i+1)):
                hidden_layer = tf.layers.dense(hidden_layer,
                                    units=h,
                                    activation=act_fn,
                                    use_bias=True,
                                    kernel_initializer=xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer(),
                                    kernel_regularizer=w_reg,
                                    bias_regularizer=b_reg,
                                    trainable=True,
                                    reuse=tf.AUTO_REUSE,
                                    name="size_{0}".format(h))
                # hidden_layer = batch_norm(hidden_layer, activation_fn=act_fn, is_training=self.phase_tensor, scope="batch_norm_{0}".format(i+1))
                if drop_rate is not None:
                    hidden_layer = dropout(hidden_layer,
                                           drop_rate,
                                           is_training=self.phase_tensor,
                                           scope="dropout_{0}".format(i+1))
        return hidden_layer
    
    def ResNN(self,
            feats,
            h_list,
            drop_rate,
            w_reg,
            b_reg,
            act_fn=tf.nn.relu):

        hidden_layer = feats
        for j, block in enumerate(h_list):
            for i, h in enumerate(block):
                with tf.variable_scope("block_{0}_layer_{1}".format(j+1, i+1)):
                    if i == 0:
                        hidden_layer = tf.layers.dense(hidden_layer,
                                            units=h,
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer=xavier_initializer(),
                                            bias_initializer=tf.zeros_initializer(),
                                            kernel_regularizer=w_reg,
                                            bias_regularizer=b_reg,
                                            trainable=True,
                                            reuse=tf.AUTO_REUSE,
                                            name="block_{0}_parm_{1}".format(j+1, h))
                        # hidden_layer = batch_norm(hidden_layer, activation_fn=None, is_training=self.phase_tensor, scope="batch_norm_{0}".format(i+1))
                    else:
                        _layer = tf.layers.dense(hidden_layer,
                                            units=h,
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer=xavier_initializer(),
                                            bias_initializer=tf.zeros_initializer(),
                                            kernel_regularizer=w_reg,
                                            bias_regularizer=b_reg,
                                            trainable=True,
                                            reuse=tf.AUTO_REUSE,
                                            name="block_{0}_parm_{1}_unit_1".format(j+1, h))
                        if act_fn is not None:
                            _layer = act_fn(_layer, name="block_{0}_act_fn_mid_{1}".format(j+1, i+1))
                        _layer = tf.layers.dense(_layer,
                                            units=h,
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer=xavier_initializer(),
                                            bias_initializer=tf.zeros_initializer(),
                                            kernel_regularizer=w_reg,
                                            bias_regularizer=b_reg,
                                            trainable=True,
                                            reuse=tf.AUTO_REUSE,
                                            name="block_{0}_parm_{1}_unit_2".format(j+1, h))
                        # _layer = batch_norm(_layer, activation_fn=None, is_training=self.phase_tensor, scope="batch_norm_{0}".format(i+1))
                        # add previous layers to make it learning residual
                        hidden_layer = tf.add(hidden_layer, _layer)
                    
                    if act_fn is not None:
                        hidden_layer = act_fn(hidden_layer, name="block_{0}_act_fn_{1}".format(j+1, i+1))
                    # hidden_layer = batch_norm(hidden_layer, activation_fn=act_fn, is_training=self.phase_tensor, scope="batch_norm_{0}".format(i))
                    if drop_rate is not None:
                        hidden_layer = dropout(hidden_layer,
                                               drop_rate,
                                               is_training=self.phase_tensor,
                                               scope="block_{0}_dropout_{1}".format(j+1, i+1))
        return hidden_layer

    def FullRes(self,
            feats,
            h_list,
            drop_rate,
            w_reg,
            b_reg,
            act_fn=tf.nn.relu):

        hidden_layer = feats
        sum_layer = 0.

        for i, h in enumerate(h_list):
            with tf.variable_scope("dense_layer_{0}".format(i+1)):
                hidden_layer = tf.layers.dense(hidden_layer,
                                    units=h,
                                    activation=None,
                                    use_bias=True,
                                    kernel_initializer=xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer(),
                                    kernel_regularizer=w_reg,
                                    bias_regularizer=b_reg,
                                    trainable=True,
                                    reuse=tf.AUTO_REUSE,
                                    name="size_{0}".format(h))
                # add all previous layers to make it dense
                hidden_layer = tf.add(hidden_layer, sum_layer, name="sum_with_front_layers")
                if act_fn is not None:
                    hidden_layer = act_fn(hidden_layer, name="act_fn_{0}".format(i))
                # hidden_layer = batch_norm(hidden_layer, activation_fn=act_fn, is_training=self.phase_tensor, scope="batch_norm_{0}".format(i))
                if drop_rate is not None:
                    hidden_layer = dropout(hidden_layer,
                                           drop_rate,
                                           is_training=self.phase_tensor,
                                           scope="dropout_{0}".format(i))
                sum_layer = tf.add(sum_layer, hidden_layer, "add_current_layer_to_sum")

        return hidden_layer
    
    def DenseNN(self,
            feats,
            h_list,
            drop_rate,
            w_reg,
            b_reg,
            act_fn=tf.nn.relu):

        hidden_layer = feats
        prior_layer_list = []

        for i, h in enumerate(h_list):
            with tf.variable_scope("dense_layer_{0}".format(i+1)):
                hidden_layer = tf.layers.dense(hidden_layer,
                                    units=h,
                                    activation=None,
                                    use_bias=True,
                                    kernel_initializer=xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer(),
                                    kernel_regularizer=w_reg,
                                    bias_regularizer=b_reg,
                                    trainable=True,
                                    reuse=tf.AUTO_REUSE,
                                    name="size_{0}".format(h))
                # add all previous layers to make it dense
                prior_layer_list.append(hidden_layer)
                hidden_layer = tf.concat(prior_layer_list, axis=1, name='prior_layers_concat')
                if act_fn is not None:
                    hidden_layer = act_fn(hidden_layer, name="act_fn_{0}".format(i+1))
                # hidden_layer = batch_norm(hidden_layer, activation_fn=act_fn, is_training=self.phase_tensor, scope="batch_norm_{0}".format(i))
                if drop_rate is not None:
                    hidden_layer = dropout(hidden_layer,
                                           drop_rate,
                                           is_training=self.phase_tensor,
                                           scope="dropout_{0}".format(i+1))
                prior_layer_list.pop()
                prior_layer_list.append(hidden_layer)

        return hidden_layer

    def CN(self,
            feats,
            degree,
            w_reg=None,
            b_reg=None):
        # the model is: x_l+1 = x_0 elementwise multipy (x_l dot w_l) + b_l + x_l
        num_feats = feats.get_shape()[1]
        x0 = feats
        x = feats
        for i in range(degree):
            with tf.variable_scope('CN_layer_{0}'.format(i+1)):
                x = tf.layers.dense(x,
                                    units=1,
                                    activation=None,
                                    use_bias=False,
                                    kernel_initializer=xavier_initializer(),
                                    kernel_regularizer=w_reg,
                                    trainable=True,
                                    reuse=tf.AUTO_REUSE,
                                    name='W_{0}'.format(i+1, i+1))

                b = tf.get_variable("b_{0}".format(i+1),
                                    shape=[1, num_feats],
                                    dtype=tf.float32,
                                    initializer=xavier_initializer(),
                                    regularizer=b_reg,
                                    trainable=True)

                x0x = tf.add(tf.multiply(x0, x), b)

                x = tf.add(x0x, x)

        return x 
    """
    def CN(self,
            feats,
            degree,
            w_reg=None,
            b_reg=None):
        # the model is: x_l+1 = x_0 outer_product x_l dot w_l + b_l + x_l
        with tf.name_scope('x_0'):
            x0 = tf.expand_dims(feats, 2, name="concat_embedded_sparse_dense")
            x = tf.expand_dims(feats, 2, name="concat_embedded_sparse_dense_transpose")
        for i in range(degree):
            with tf.variable_scope('CN_layer_{0}'.format(i+1)):
                x_ = tf.transpose(x, [0, 2, 1])
                xx = tf.einsum('aij,ajk->aik', x0, x_)
                x = tf.layers.dense(xx,
                                    units=1,
                                    activation=None,
                                    use_bias=True,
                                    kernel_initializer=xavier_initializer(),
                                    bias_initializer=tf.zeros_initializer(),
                                    kernel_regularizer=w_reg,
                                    bias_regularizer=b_reg,
                                    trainable=True,
                                    reuse=tf.AUTO_REUSE,
                                    name='W_{0}_b_{1}'.format(i+1, i+1)) + x

        out = tf.contrib.layers.flatten(x, scope='x_{0}'.format(degree)) 
                
        return out 
        """
