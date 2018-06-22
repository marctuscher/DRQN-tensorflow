




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

    def __init__(self, n_actions, screen_width, screen_height, gamma=0.99, debug=False):
        self.n_actions = n_actions
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.history_len=1
        self.gamma = gamma
        self.debug = debug

        self.initializer = tf.truncated_normal_initializer(0, 0.02)
        # self.debug = True
        self.nepochs =5
        self.keep_prob = 0.8
        self.batch_size = 64
        self.lr_method = "adam"
        self.learning_rate = 0.01
        self.lr_decay = 0.99
        self.clip = 1  # if negative, no clipping
        self.nepoch_no_imprv = 5
        self.sess = None
        self.saver = None
        # delete ./out
        if os.path.isdir("./out"):
            shutil.rmtree("./out")
        self.dir_output = "./out"
        # self.dir_model = os.getenv("HOME") + str("/tmp/btcmodel/model.ckpt")

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
                optimizer = tf.train.RMSPropOptimizer(lr)
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

    def add_summary(self):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.dir_output+"/train",
                                                 self.sess.graph)

    def train(self, train):
        best_score = 0
        nepoch_no_imprv = 0  # for early stopping
        # self.add_summary()  # tensorboard

        for epoch in range(self.nepochs):
            print("Epoch {:} out of {:}\n".format(epoch + 1,
                                                self.nepochs))
            self.run_epoch(train, epoch)
            self.learning_rate *= self.lr_decay  # decay learning rate
            self.update_target()


    def evaluate(self, test):
        print("Testing model over test set")
        metrics = self.run_evaluate(test)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                          for k, v in metrics.items()])
        print(msg)

    def add_placeholders(self):
        self.w = {}
        self.w_target = {}
        self.state = tf.placeholder(tf.float32, shape=[None, self.history_len, self.screen_width, self.screen_height],
                                       name="input_state")
        self.action = tf.placeholder(tf.int32, shape=[None], name="action_input")
        self.reward = tf.placeholder(tf.int32, shape=[None], name="reward")

        self.state_target = tf.placeholder(tf.float32, shape=[None,self.history_len, 80,112],
                                       name="input_target")
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                                 name="lr")

    def get_feed_dict(self, state_inputs, action_inputs=None,reward=None, state_target=None, lr=None, dropout=None):
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

        return feed


    def add_logits_op_train(self):
        with tf.variable_scope("conv1_train"):
            w = tf.get_variable("wc1", (8, 8, self.state.get_shape()[1], 32), dtype=tf.float32, initializer=self.initializer)
            conv = tf.nn.conv2d(self.state, w, [1, 1, 4, 4], padding='VALID', data_format='NCHW')
            b = tf.get_variable('bc1', [32], initializer=self.initializer)
            self.w['wc1'] = w
            self.w['bc1'] = b
            out = tf.nn.bias_add(conv, b, "NCHW")
            out = tf.nn.relu(out)

        with tf.variable_scope("conv2_train"):
            w = tf.get_variable("wc2", (4, 4, 32, 64), dtype=tf.float32, initializer=self.initializer)
            conv = tf.nn.conv2d(out, w, [1, 1, 2, 2], padding='VALID', data_format='NCHW')
            b = tf.get_variable('bc2', [64], initializer=self.initializer)
            self.w['wc2'] = w
            self.w['bc2'] = b
            out = tf.nn.bias_add(conv, b, "NCHW")
            out = tf.nn.relu(out)

        with tf.variable_scope("conv3_train"):
            w = tf.get_variable("wc3", (3, 3, 64, 64), dtype=tf.float32, initializer=self.initializer)
            conv = tf.nn.conv2d(out, w, [1, 1, 1, 1], padding='VALID', data_format='NCHW')
            b = tf.get_variable('bc3', [64], initializer=self.initializer)
            self.w['wc3'] = w
            self.w['bc3'] = b
            out = tf.nn.bias_add(conv, b, "NCHW")
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
        with tf.variable_scope("conv1_target"):
            w = tf.get_variable("wc1", (8, 8, self.state.get_shape()[1], 32), dtype=tf.float32, initializer=self.initializer)
            conv = tf.nn.conv2d(self.state, w, [1, 1, 4, 4], padding='VALID', data_format='NCHW')
            b = tf.get_variable('bc1', [32], initializer=self.initializer)
            self.w_target['wc1'] = w
            self.w_target['bc1'] = b
            out = tf.nn.bias_add(conv, b, "NCHW")
            out = tf.nn.relu(out)

        with tf.variable_scope("conv2_target"):
            w = tf.get_variable("wc2", (4, 4, 32, 64), dtype=tf.float32, initializer=self.initializer)
            conv = tf.nn.conv2d(out, w, [1, 1, 2, 2], padding='VALID', data_format='NCHW')
            b = tf.get_variable('bc2', [64], initializer=self.initializer)
            self.w_target['wc2'] = w
            self.w_target['bc2'] = b
            out = tf.nn.bias_add(conv, b, "NCHW")
            out = tf.nn.relu(out)

        with tf.variable_scope("conv3_target"):
            w = tf.get_variable("wc3", (3, 3, 64, 64), dtype=tf.float32, initializer=self.initializer)
            conv = tf.nn.conv2d(out, w, [1, 1, 1, 1], padding='VALID', data_format='NCHW')
            b = tf.get_variable('bc3', [64], initializer=self.initializer)
            self.w_target['wc3'] = w
            self.w_target['bc3'] = b
            out = tf.nn.bias_add(conv, b, "NCHW")
            out = tf.nn.relu(out)
            shape = out.get_shape().as_list()
            out_flat = tf.reshape(out, [-1, reduce(lambda x,y: x * y, shape[1:])])
            shape = out_flat.get_shape().as_list()

        with tf.variable_scope("fully1_target"):
            w = tf.get_variable('wf1', [shape[1], 512], dtype=tf.float32, initializer=self.initializer)
            b = tf.get_variable('bf1', [512], dtype=tf.float32, initializer=self.initializer)
            self.w_target["wf1"] = w
            self.w_target["bf1"] = b
            out = tf.nn.xw_plus_b(out_flat, w, b)
            out = tf.nn.relu(out)

        with tf.variable_scope("out_target"):
            w = tf.get_variable('wout', [512, self.n_actions], dtype=tf.float32, initializer=self.initializer)
            b = tf.get_variable('bout', [self.n_actions], dtype=tf.float32, initializer=self.initializer)
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

    def add_loss_op(self):
        action_one_hot = tf.one_hot(self.action, self.n_actions, 1.0, 0.0, name='action_one_hot')
        target_one_hot = tf.one_hot(self.q_target_action, self.n_actions, 1.0, 0.0, name='target_action_one_hot')
        target = tf.reduce_sum(self.q_target_out * target_one_hot, reduction_indices=1, name='target')
        train = tf.reduce_sum(self.q_out * action_one_hot, reduction_indices=1, name='action_one_hot')
        loss = tf.square(tf.subtract(tf.add(tf.cast(self.reward, dtype=tf.float32), self.gamma * target), train))
        self.loss = tf.reduce_mean(loss)
        tf.summary.scalar("loss", self.loss)

    def build(self):
        self.add_placeholders()
        self.add_logits_op_train()
        self.add_logits_op_target()
        self.add_loss_op()
        self.add_train_op(self.lr_method, self.lr, self.loss,
                          self.clip)
        self.initialize_session()
        self.init_update()
        self.add_summary()

    def predict_batch(self, state_input):
        fd = self.get_feed_dict(state_input, dropout=1.0)
        q, a = self.sess.run([self.q_out, self.q_action], feed_dict=fd)
        return q, a

    def run_epoch(self, train, epoch):
        # progbar stuff for logging
        batch_size = self.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)
        shuffle(train)
        for i, (state, action, reward, state_) in enumerate(self.minibatches(train, batch_size)):
            fd = self.get_feed_dict(state, action, reward, state_, self.learning_rate,
                                       self.keep_prob)
            _, train_loss, summary = self.sess.run(
                [self.train_op, self.loss, self.merged], feed_dict=fd
            )
            prog.update(i + 1, [("train loss", train_loss)])
            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * nbatches + i)


    def save_session(self):
        """Saves session = weights"""
        if not os.path.exists(self.dir_model):
            os.makedirs(self.dir_model)
        self.saver.save(self.sess, self.dir_model)


    def update_target(self):
        for name in self.w:
            self.sess.run(self.target_w_assign[name])

    def minibatches(self, data, minibatch_size):
        """
        Args:
            data: generator of (sentence, tags) tuples
            minibatch_size: (int)
        Yields:
            list of tuples
        """
        action_batch, state_batch, reward_batch, state_batch_ = [], [], [], []

        for (state, action, reward, state_) in data:
            if len(state_batch) == minibatch_size:
                yield state_batch, action_batch, reward_batch, state_batch_
                action_batch, state_batch, reward_batch, state_batch_ = [], [], [], []

            action_batch += [action]
            reward_batch += [reward]
            state_batch += [state]
            state_batch_ += [state_]

        if len(state_batch) != 0:
            yield state_batch, action_batch, reward_batch, state_batch_
