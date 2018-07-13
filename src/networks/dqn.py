import numpy as np
import os
import tensorflow as tf
import shutil
from functools import reduce
from tensorflow.python import debug as tf_debug
from src.utils import conv2d_layer, fully_connected_layer

# from utilities.keras_progbar import Progbar

"""
This class instantiates a neural network for regression on a specific dataset
"""


class DQN():

    def __init__(self, n_actions, config):
        self.n_actions = n_actions
        self.screen_width = config.screen_width
        self.screen_height = config.screen_height
        self.gamma = config.gamma
        self.history_len = config.history_len
        self.dir_save = config.dir_save
        self.learning_rate_minimum = config.learning_rate_minimum

        # self.initializer = tf.truncated_normal_initializer(0, 0.01)
        # self.initializer = tf.contrib.layers.xavier_initializer()
        self.initializer = tf.zeros_initializer()
        self.debug = not True
        self.keep_prob = config.keep_prob
        self.batch_size = config.batch_size
        self.lr_method = config.lr_method
        self.learning_rate = config.learning_rate
        self.lr_decay = config.lr_decay
        self.sess = None
        self.saver = None
        self.all_tf = not True
        # delete ./out
        if os.path.isdir("./out"):
            shutil.rmtree("./out")
        self.dir_output = "./out"
        self.dir_model = self.dir_save + "/net/"
        self.train_steps = 0
        self.is_training = False

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
        self.file_writer = tf.summary.FileWriter(self.dir_output + "/train",
                                                 self.sess.graph)

    def inject_summary(self, tag_dict, step):
        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict], {
            self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
        })
        for summ in summary_str_lists:
            self.file_writer.add_summary(summ, step)

    def train_on_batch_target(self, state, action, reward, state_, terminal, steps):
        self.is_training = True
        state_ = state_ / 255.0
        state = state / 255.0
        target_val = self.q_target_out.eval({self.state_target: state_}, session=self.sess)
        max_target = np.max(target_val, axis=1)
        target = (1. - terminal) * self.gamma * max_target + reward
        _, q, train_loss, loss_summary, q_summary, image_summary = self.sess.run(
            [self.train_op, self.q_out, self.loss, self.loss_summary, self.avg_q_summary, self.merged_image_sum],
            feed_dict={
                self.state: state,
                self.action: action,
                self.target_val: target,
                self.lr: self.learning_rate,
                self.dropout: self.keep_prob
            }
        )
        if self.train_steps % 1000 == 0:
            self.file_writer.add_summary(loss_summary, self.train_steps)
            self.file_writer.add_summary(q_summary, self.train_steps)
            self.file_writer.add_summary(image_summary, self.train_steps)
        if steps % 20000 == 0 and steps > 50000:
            self.learning_rate *= self.lr_decay  # decay learning rate
            if self.learning_rate < self.learning_rate_minimum:
                self.learning_rate = self.learning_rate_minimum
        self.train_steps += 1
        return q.mean()

    def train_on_batch_all_tf(self, state, action, reward, state_, terminal, steps):
        self.is_training = True
        terminal = (terminal == True)
        target_val_tf = self.q_target_out.eval({self.state_target: state_}, session=self.sess)
        _, q, train_loss, loss_summary, q_summary = self.sess.run(
            [self.train_op, self.q_out, self.loss, self.loss_summary, self.avg_q_summary], feed_dict={
                self.state: state,
                self.action: action,
                self.target_val_tf: target_val_tf,
                self.reward: reward,
                self.terminal: terminal,
                self.lr: self.learning_rate,
                self.dropout: self.keep_prob
            }
        )
        if self.train_steps % 1000 == 0:
            self.file_writer.add_summary(loss_summary, self.train_steps)
            self.file_writer.add_summary(q_summary, self.train_steps)
        if self.train_steps % 10000 == 0 and steps > 1000000:
            self.learning_rate *= self.lr_decay  # decay learning rate
            if self.learning_rate < self.learning_rate_minimum:
                self.learning_rate = self.learning_rate_minimum
        self.train_steps += 1
        return q.mean()

    def add_placeholders(self):
        self.w = {}
        self.w_target = {}
        self.state = tf.placeholder(tf.float32, shape=[None, self.history_len, self.screen_height, self.screen_width],
                                    name="input_state")
        self.action = tf.placeholder(tf.int32, shape=[None], name="action_input")
        self.reward = tf.placeholder(tf.int32, shape=[None], name="reward")

        self.state_target = tf.placeholder(tf.float32,
                                           shape=[None, self.history_len, self.screen_height, self.screen_width],
                                           name="input_target")
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                                 name="lr")
        self.terminal = tf.placeholder(dtype=tf.float16, shape=[None], name="terminal")

        self.target_val = tf.placeholder(dtype=tf.float32, shape=[None], name="target_val")
        self.target_val_tf = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions])

        self.learning_rate_step = tf.placeholder("int64", None, name="learning_rate_step")

    def add_logits_op_train(self):
        self.image_summary = []
        input = tf.transpose(self.state, [0, 2, 3, 1])
        w, b, out, summary = conv2d_layer(input, 32, [8, 8], [4, 4], scope_name="conv1_train", summary_tag="conv1_out",
                                          activation=tf.nn.leaky_relu)
        self.w["wc1"] = w
        self.w["bc1"] = b
        self.image_summary.append(summary)

        w, b, out, summary = conv2d_layer(out, 64, [4, 4], [2, 2], scope_name="conv2_train", summary_tag="conv2_out",
                                          activation=tf.nn.leaky_relu)
        self.w["wc2"] = w
        self.w["bc2"] = b
        self.image_summary.append(summary)

        w, b, out, summary = conv2d_layer(out, 64, [3, 3], [1, 1], scope_name="conv3_train", summary_tag="conv3_out",
                                          activation=tf.nn.leaky_relu)
        self.w["wc3"] = w
        self.w["bc3"] = b
        self.image_summary.append(summary)

        shape = out.get_shape().as_list()
        out_flat = tf.reshape(out, [-1, reduce(lambda x, y: x * y, shape[1:])])
        shape = out_flat.get_shape().as_list()

        w, b, out = fully_connected_layer(out_flat, shape[1], 512, scope_name="fully1_train")

        self.w["wf1"] = w
        self.w["bf1"] = b

        w, b, out = fully_connected_layer(out, 512, self.n_actions, scope_name="out_train", activation=None)

        self.w["wout"] = w
        self.w["bout"] = b

        self.q_out = out
        self.q_action = tf.argmax(self.q_out, axis=1)

    def add_logits_op_target(self):
        input = tf.transpose(self.state_target, [0, 2, 3, 1])

        w, b, out, _ = conv2d_layer(input, 32, [8, 8], [4, 4], scope_name="conv1_target", summary_tag=None,
                                    activation=tf.nn.leaky_relu)
        self.w_target["wc1"] = w
        self.w_target["bc1"] = b

        w, b, out, _ = conv2d_layer(out, 64, [4, 4], [2, 2], scope_name="conv2_target", summary_tag=None,
                                    activation=tf.nn.leaky_relu)
        self.w_target["wc2"] = w
        self.w_target["bc2"] = b

        w, b, out, _ = conv2d_layer(out, 64, [3, 3], [1, 1], scope_name="conv3_target", summary_tag=None,
                                    activation=tf.nn.leaky_relu)
        self.w_target["wc3"] = w
        self.w_target["bc3"] = b

        shape = out.get_shape().as_list()
        out_flat = tf.reshape(out, [-1, reduce(lambda x, y: x * y, shape[1:])])
        shape = out_flat.get_shape().as_list()

        w, b, out = fully_connected_layer(out_flat, shape[1], 512, scope_name="fully1_target")

        self.w_target["wf1"] = w
        self.w_target["bf1"] = b

        w, b, out = fully_connected_layer(out, 512, self.n_actions, scope_name="out_target", activation=None)

        self.w_target["wout"] = w
        self.w_target["bout"] = b

        self.q_target_out = out
        self.q_target_action = tf.argmax(self.q_target_out, axis=1)

    def init_update(self):
        self.target_w_in = {}
        self.target_w_assign = {}
        for name in self.w:
            self.target_w_in[name] = tf.placeholder(tf.float32, self.w_target[name].get_shape().as_list(), name=name)
            self.target_w_assign[name] = self.w_target[name].assign(self.target_w_in[name])

    def add_loss_op_target(self):
        action_one_hot = tf.one_hot(self.action, self.n_actions, 1.0, 0.0, name='action_one_hot')
        train = tf.reduce_sum(self.q_out * action_one_hot, reduction_indices=1, name='action_one_hot')
        self.delta = train - tf.stop_gradient(self.target_val)
        self.loss = tf.reduce_mean(self.clipping(self.delta))

        optimizer = tf.train.RMSPropOptimizer(self.lr, momentum=0.95, epsilon=0.01)
        self.train_op = optimizer.minimize(self.loss)
        avg_q = tf.reduce_mean(self.q_out, 0)
        q_summary = []
        for i in range(self.n_actions):
            q_summary.append(tf.summary.histogram('q/{}'.format(i), avg_q[i]))
        self.merged_image_sum = tf.summary.merge(self.image_summary, "images")
        self.avg_q_summary = tf.summary.merge(q_summary, 'q_summary')
        self.loss_summary = tf.summary.scalar("loss", self.loss)

    def add_loss_op_target_tf(self):
        target_argmax = tf.argmax(self.target_val_tf, axis=1)
        target_one_hot = tf.one_hot(target_argmax, self.n_actions, 1.0, 0.0, name="target_one_ho")
        target_val = tf.reduce_sum(self.target_val_tf * target_one_hot, reduction_indices=1)
        target = tf.add(tf.multiply((tf.subtract(1.0, tf.cast(self.terminal, dtype=tf.float32))),
                                    tf.multiply(self.gamma, target_val)), tf.cast(self.reward, dtype=tf.float32))
        action_one_hot = tf.one_hot(self.action, self.n_actions, 1.0, 0.0, name='action_one_hot')
        train = tf.reduce_sum(self.q_out * action_one_hot, reduction_indices=1)
        delta = target - train
        self.loss = tf.reduce_mean(self.clipping(delta))
        avg_q = tf.reduce_mean(self.q_out, 0)
        q_summary = []
        for i in range(self.n_actions):
            q_summary.append(tf.summary.histogram('q/{}'.format(i), avg_q[i]))
        self.avg_q_summary = tf.summary.merge(q_summary, 'q_summary')
        self.loss_summary = tf.summary.scalar("loss", self.loss)

    def clipping(self, x):
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

    def build(self):
        self.add_placeholders()
        self.add_logits_op_train()
        self.add_logits_op_target()
        if self.all_tf:
            self.add_loss_op_target_tf()
        else:
            self.add_loss_op_target()
        # self.add_train_op(self.lr_method, self.lr, self.loss)
        # self.preprocess_func()
        self.initialize_session()
        self.init_update()

    def save_session(self):
        """Saves session = weights"""
        if not os.path.exists(self.dir_model):
            os.makedirs(self.dir_model)
        self.saver.save(self.sess, self.dir_model)

    def update_target(self):
        for name in self.w:
            self.target_w_assign[name].eval({self.target_w_in[name]: self.w[name].eval(session=self.sess)},
                                            session=self.sess)

    def restore_session(self):
        self.saver.restore(self.sess, self.dir_model)

    def integer_product(self, x):
        return int(np.prod(x))

    def initializer_bounds_filter(self, filter_shape):
        fan_in = self.integer_product(filter_shape[:3])
        fan_out = self.integer_product(filter_shape[:2]) * filter_shape[3]
        return np.sqrt(6. / (fan_in + fan_out))
