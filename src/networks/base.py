import numpy as np
import os
import time
import tensorflow as tf
import datetime
import shutil
from functools import reduce
from tensorflow.python import debug as tf_debug
from src.utils import conv2d_layer, fully_connected_layer

# from utilities.keras_progbar import Progbar

"""
This class instantiates a neural network for regression on a specific dataset
"""


class BaseModel():
    """
    Base class for deep Q learning
    """

    def __init__(self, config, network_type):

        self.screen_width = config.screen_width
        self.screen_height = config.screen_height
        self.gamma = config.gamma
        self.dir_save = config.dir_save
        self.learning_rate_minimum = config.learning_rate_minimum
        self.debug = not True
        self.keep_prob = config.keep_prob
        self.batch_size = config.batch_size
        self.lr_method = config.lr_method
        self.learning_rate = config.learning_rate
        self.lr_decay = config.lr_decay
        self.sess = None
        self.saver = None
        # delete ./out
        self.dir_output = "./out/"+network_type+"/"+ str(datetime.datetime.utcnow()) + "/"
        self.dir_model = self.dir_save + "/net/" + str(time.time()) + "/"
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

    def add_summary(self, summary_tags, histogram_tags):
        self.summary_placeholders = {}
        self.summary_ops = {}
        for tag in summary_tags:
            self.summary_placeholders[tag] = tf.placeholder(tf.float32, None, name=tag)
            self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
        for tag in histogram_tags:
            self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
            self.summary_ops[tag] = tf.summary.histogram(tag, self.summary_placeholders[tag])
        self.file_writer = tf.summary.FileWriter(self.dir_output + "/train",
                                                 self.sess.graph)

    def inject_summary(self, tag_dict, step):
        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict], {
            self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
        })
        for summ in summary_str_lists:
            self.file_writer.add_summary(summ, step)


    def save_session(self):
        """Saves session = weights"""
        if not os.path.exists(self.dir_model):
            os.makedirs(self.dir_model)
        self.saver.save(self.sess, self.dir_model)

    def restore_session(self):
        self.saver.restore(self.sess, self.dir_model)
