import numpy as np
import os
import tensorflow as tf
import shutil
from networks.keras_progbar import Progbar
from functools import reduce
from random import shuffle
from tensorflow.python import debug as tf_debug

# from utilities.keras_progbar import Progbar

"""
This class instantiates a neural network for regression on a specific dataset
"""
class DQN():

    def __init__(self, n_actions, history_len, screen_width, screen_height, gamma=0.99, debug=False, pred_before_train=True):
        self.n_actions = n_actions
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.gamma = gamma
        self.debug = debug
        self.history_len = history_len
        self.pred_before_train= pred_before_train

        self.initializer = tf.truncated_normal_initializer(0, 0.02)
        # self.initializer = tf.xavier_initializer()
        # self.debug = True
        self.nepochs = 30
        self.keep_prob = 0.8
        self.batch_size = 32
        self.lr_method = "rmsprop"
        self.learning_rate = 0.00025
        self.lr_decay = .96
        self.clip = -1  # if negative, no clipping
        self.nepoch_no_imprv = 5
        self.sess = None
        self.saver = None
        # delete ./out
        if os.path.isdir("./out"):
            shutil.rmtree("./out")
        self.dir_output = "./out"
        self.dir_model = "./model"
        self.train_steps = 0
        self.is_training = False

    def reinitialize_weights(self, scope_name):
        variables = tf.contrib.framework.get_variables(scope_init)
        name = tf.variables_initializer(variables)
        self.sess.run(init)

    def add_train_op(self, lr_method, lr, loss, clip=-1):
        _lr_m = lr_method.lower()  # lower to make sure

        with tf.variable_scope("train_step"):
            if _lr_m == 'adam':  # sgd method
                optimizer = tf.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr, momentum=0.95, epsilon=0.01)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            if clip > 0:  # gradient clipping if clip is positive
                grads, vs = zip(*optimizer.compute_gradients(loss))
                grads, gnorm = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)

    def initialize_session(self):
        print("Initializing tf session")
        self.sess = tf.Session()
        if self.debug:
            self.sess = tf_debug.TensorBoardDebugWrapperSession(self.sess, "localhost:6064")
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def close_session(self):
        self.sess.close()

    def add_summary(self, summary_tags):
        self.summary_placeholders = {}
        self.summary_ops = {}
        for tag in summary_tags:
            self.summary_placeholders[tag] = tf.placeholder(tf.float32, None, name=tag)
            self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
        self.file_writer = tf.summary.FileWriter(self.dir_output+"/train",
                                                 self.sess.graph)

    def inject_summary(self, tag_dict):
        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict],{
            self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
        })
        for summ in summary_str_lists:
            self.file_writer.add_summary(summ, self.train_steps)


    def train_on_batch_target(self, state, action, reward, state_, terminal):
        self.is_training = True
        target_val = self.q_target_out.eval({self.state_target : state_}, session=self.sess)
        terminal = np.array(terminal) + 0.
        max_target = np.max(target_val, axis=1)
        target = (1. - terminal) * self.gamma * max_target + reward
        _, train_loss, loss_summary = self.sess.run(
            [self.train_op, self.loss, self.loss_summary], feed_dict={
                self.state : state,
                self.action : action,
                self.target_val: target,
                self.lr : self.learning_rate,
                self.dropout : self.keep_prob
            }
        )
        self.file_writer.add_summary(loss_summary, self.train_steps)
        if self.train_steps % 10000 == 0 and self.train_steps != 0:
            self.learning_rate *= self.lr_decay  # decay learning rate
        self.train_steps += 1

    def add_placeholders(self):
        self.w = {}
        self.w_target = {}
        self.state = tf.placeholder(tf.float32, shape=[None, self.history_len, self.screen_height, self.screen_width],
                                       name="input_state")
        self.action = tf.placeholder(tf.int32, shape=[None], name="action_input")
        self.reward = tf.placeholder(tf.int32, shape=[None], name="reward")

        self.state_target = tf.placeholder(tf.float32, shape=[None,self.history_len, self.screen_height, self.screen_width],
                                       name="input_target")
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                                 name="lr")
        self.terminal = tf.placeholder(dtype=tf.uint8, shape=[None], name="terminal")

        self.target_val = tf.placeholder(dtype=tf.float32, shape=[None])


    def get_feed_dict(self, state_inputs, action_inputs=None,reward=None, state_target=None,terminal=None, lr=None, dropout=None):
        feed = {
            self.state: state_inputs
        }
        if action_inputs is not None:
            feed[self.action] = action_inputs

        if reward is not None:
            feed[self.reward] = reward

        if state_target is not None:
            feed[self.state_target] = state_target

        if lr is not None:
            feed[self.lr] = lr
        if dropout is not None:
            feed[self.dropout] = dropout
        if terminal is not None:
            feed[self.terminal] = terminal
        return feed


    def add_logits_op_train(self):
        self.state = tf.divide(self.state, 255)
        with tf.variable_scope("conv1_train"):
            w = tf.get_variable("wc1", (8, 8, self.state.get_shape()[1], 32), dtype=tf.float32, initializer=self.initializer)
            conv = tf.nn.conv2d(self.state, w, [1, 1, 4, 4], padding='VALID', data_format='NCHW')
            b = tf.get_variable('bc1', [32], initializer=self.initializer)
            self.w['wc1'] = w
            self.w['bc1'] = b
            out = tf.nn.bias_add(conv, b, "NCHW")
            out = tf.layers.batch_normalization(out, training=self.is_training)
            out = tf.nn.relu(out)

        with tf.variable_scope("conv2_train"):
            w = tf.get_variable("wc2", (4, 4, 32, 64), dtype=tf.float32, initializer=self.initializer)
            conv = tf.nn.conv2d(out, w, [1, 1, 2, 2], padding='VALID', data_format='NCHW')
            b = tf.get_variable('bc2', [64], initializer=self.initializer)
            self.w['wc2'] = w
            self.w['bc2'] = b
            out = tf.nn.bias_add(conv, b, "NCHW")
            out = tf.layers.batch_normalization(out, training=self.is_training)
            out = tf.nn.relu(out)

        with tf.variable_scope("conv3_train"):
            w = tf.get_variable("wc3", (3, 3, 64, 64), dtype=tf.float32, initializer=self.initializer)
            conv = tf.nn.conv2d(out, w, [1, 1, 1, 1], padding='VALID', data_format='NCHW')
            b = tf.get_variable('bc3', [64], initializer=self.initializer)
            self.w['wc3'] = w
            self.w['bc3'] = b
            out = tf.nn.bias_add(conv, b, "NCHW")
            out = tf.layers.batch_normalization(out, training=self.is_training)
            out = tf.nn.relu(out)

            shape = out.get_shape().as_list()
            out_flat = tf.reshape(out, [-1, reduce(lambda x,y: x * y, shape[1:])])
            shape = out_flat.get_shape().as_list()

        with tf.variable_scope("fully1_train"):
            w = tf.get_variable('wf1', [shape[1], 512], dtype=tf.float32, initializer=self.initializer)
            b = tf.get_variable('bf1', [512], dtype=tf.float32, initializer=self.initializer)
            self.w["wf1"] = w
            self.w["bf1"] = b
            out = tf.nn.xw_plus_b(out_flat, w, b)
            out = tf.nn.relu(out)
            out = tf.nn.dropout(out, self.dropout)

        with tf.variable_scope("out_train"):
            w = tf.get_variable('wout', [512, self.n_actions], dtype=tf.float32, initializer=self.initializer)
            b = tf.get_variable('bout', [self.n_actions], dtype=tf.float32, initializer=self.initializer)
            self.w["wout"] = w
            self.w["bout"] = b
            out = tf.nn.xw_plus_b(out, w, b)
            out = tf.nn.relu(out)
            self.q_out = out
            self.q_action = tf.argmax(self.q_out, axis=1)

    def add_logits_op_target(self):
        self.state_target = tf.divide(self.state_target, 255)
        with tf.variable_scope("conv1_target"):
            w = tf.get_variable("wc1", (8, 8, self.state.get_shape()[1], 32), dtype=tf.float32, initializer=self.initializer, trainable=False)
            conv = tf.nn.conv2d(self.state_target, w, [1, 1, 4, 4], padding='VALID', data_format='NCHW')
            b = tf.get_variable('bc1', [32], initializer=self.initializer, trainable=False)
            self.w_target['wc1'] = w
            self.w_target['bc1'] = b
            out = tf.nn.bias_add(conv, b, "NCHW")
            out = tf.layers.batch_normalization(out, training=False)
            out = tf.nn.relu(out)

        with tf.variable_scope("conv2_target"):
            w = tf.get_variable("wc2", (4, 4, 32, 64), dtype=tf.float32, initializer=self.initializer, trainable=False)
            conv = tf.nn.conv2d(out, w, [1, 1, 2, 2], padding='VALID', data_format='NCHW')
            b = tf.get_variable('bc2', [64], initializer=self.initializer, trainable=False)
            self.w_target['wc2'] = w
            self.w_target['bc2'] = b
            out = tf.nn.bias_add(conv, b, "NCHW")
            out = tf.layers.batch_normalization(out, training=False)
            out = tf.nn.relu(out)

        with tf.variable_scope("conv3_target"):
            w = tf.get_variable("wc3", (3, 3, 64, 64), dtype=tf.float32, initializer=self.initializer, trainable=False)
            conv = tf.nn.conv2d(out, w, [1, 1, 1, 1], padding='VALID', data_format='NCHW')
            b = tf.get_variable('bc3', [64], initializer=self.initializer, trainable=False)
            self.w_target['wc3'] = w
            self.w_target['bc3'] = b
            out = tf.nn.bias_add(conv, b, "NCHW")
            out = tf.layers.batch_normalization(out, training=False)
            out = tf.nn.relu(out)

            shape = out.get_shape().as_list()
            out_flat = tf.reshape(out, [-1, reduce(lambda x,y: x * y, shape[1:])])
            shape = out_flat.get_shape().as_list()

        with tf.variable_scope("fully1_target"):
            w = tf.get_variable('wf1', [shape[1], 512], dtype=tf.float32, initializer=self.initializer, trainable=False)
            b = tf.get_variable('bf1', [512], dtype=tf.float32, initializer=self.initializer, trainable=False)
            self.w_target["wf1"] = w
            self.w_target["bf1"] = b
            out = tf.nn.xw_plus_b(out_flat, w, b)
            out = tf.nn.relu(out)

        with tf.variable_scope("out_target"):
            w = tf.get_variable('wout', [512, self.n_actions], dtype=tf.float32, initializer=self.initializer, trainable=False)
            b = tf.get_variable('bout', [self.n_actions], dtype=tf.float32, initializer=self.initializer, trainable=False)
            self.w_target["wout"] = w
            self.w_target["bout"] = b
            out = tf.nn.xw_plus_b(out, w, b)
            out = tf.nn.relu(out)
            self.q_target_out = out
            self.q_target_action = tf.argmax(self.q_target_out, axis=1)


    def init_update(self):
        self.target_w_in = {}
        self.target_w_assign = {}
        for name in self.w:
            self.target_w_assign[name] = self.w_target[name].assign(self.w[name].eval(session=self.sess))


    def add_loss_op_target(self):
        action_one_hot = tf.one_hot(self.action, self.n_actions, 1.0, 0.0, name='action_one_hot')
        train = tf.reduce_sum(self.q_out * action_one_hot, reduction_indices=1, name='action_one_hot')
        delta = self.target_val - train
        self.loss = tf.reduce_mean(self.clipping(delta))
        self.loss_summary = tf.summary.scalar("loss", self.loss)


    def clipping(self, x):
        try:
            return tf.select(tf.abs(x)< 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
        except:
            return tf.where(tf.abs(x)< 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

    def build(self):
        self.add_placeholders()
        self.add_logits_op_train()
        self.add_logits_op_target()
        self.add_loss_op_target()
        self.add_train_op(self.lr_method, self.lr, self.loss,
                          self.clip)
        self.initialize_session()
        self.init_update()

    def predict_batch(self, state_input):
        self.is_training = False
        fd = self.get_feed_dict(state_input, dropout=1.0)
        q, a = self.sess.run([self.q_out, self.q_action], feed_dict=fd)
        return q, a

    def predict_batch_target(self, state_input):
        self.is_training = False
        fd = self.get_feed_dict(state_input, dropout=1.0)
        q = self.sess.run([self.q_target_out], feed_dict=fd)
        return q



    def save_session(self):
        """Saves session = weights"""
        if not os.path.exists(self.dir_model):
            os.makedirs(self.dir_model)
        self.saver.save(self.sess, self.dir_model)


    def update_target(self):
        for name in self.w:
            self.sess.run(self.target_w_assign[name])
