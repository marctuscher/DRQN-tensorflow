




import numpy as np
import os
import tensorflow as tf
import shutil
from random import shuffle

from utilities.keras_progbar import Progbar

"""
This class instantiates a neural network for regression on a specific dataset
"""
class DQN():

    def __init__(self, n_actions):
        """
        Defines the hyperparameters
        """
        # training
        self.utils = utils
        self.n_actions = n_actions
        self.nepochs =100
        self.keep_prob = 0.5
        self.batch_size = 512
        self.lr_method = "adam"
        self.learning_rate = 0.001
        self.lr_decay = 0.9
        self.clip = -1  # if negative, no clipping
        self.nepoch_no_imprv = 5
        # model hyperparameters
        self.hidden_size_lstm = 300  # lstm on word embeddings
        self.sess = None
        self.saver = None
        # delete ./out so
        if os.path.isdir("./out"):
            shutil.rmtree("./out")
        self.dir_output = "./out"
        self.dir_model = os.getenv("HOME") + str("/tmp/btcmodel/model.ckpt")
        self.acc = 0
        self.seq_len=60

    def reinitialize_weights(self, scope_name):
        variables = tf.contrib.framework.get_variables(scope_name)
        init = tf.variables_initializer(variables)
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
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def close_session(self):
        self.sess.close()

    def add_summary(self):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.dir_output+"/train",
                                                 self.sess.graph)

    def train(self, train, dev):
        best_score = 0
        nepoch_no_imprv = 0  # for early stopping
        self.add_summary()  # tensorboard

        for epoch in range(self.nepochs):
            print("Epoch {:} out of {:}\n".format(epoch + 1,
                                                self.nepochs))
            self.run_epoch(train, dev, epoch)
            self.learning_rate *= self.lr_decay  # decay learning rate

            if epoch % 2 == 0:

                metrics = self.run_evaluate(dev)
                msg = " - ".join(["{} {:04.2f}".format(k, v)
                          for k, v in metrics.items()])
                print(msg)
                self.file_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="accuracy", simple_value=metrics["acc"])]), epoch)
                # early stopping and saving best parameters
                if metrics["acc"] < best_score:
                    nepoch_no_imprv = 0
                    self.save_session()
                    best_score = metrics["acc"]
                    print("- new best score!")
                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= self.nepoch_no_imprv:
                        print("- early stopping {} epochs without " \
                              "improvement".format(nepoch_no_imprv))
                        break

    def evaluate(self, test):
        print("Testing model over test set")
        metrics = self.run_evaluate(test)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                          for k, v in metrics.items()])
        print(msg)

    def add_placeholders(self):
        self.weights = {}
        self.target_weights = {}
        self.state = tf.placeholder(tf.float32, shape=[None, self.history_len, self.screen_width, self.screen_height],
                                       name="input")

        self.state_target = tf.placeholder(tf.float32, shape=[None, 80,112],
                                       name="input_target")
        self.y = tf.placeholder(tf.float32, shape=[self.batch_size, self.action_shape[0], self.action_shape[1]],
                                     name="y")
        self.y_target = tf.placeholder(tf.float32, shape=[self.batch_size, self.action_shape[0], self.action_shape[1]],
                                     name="y_target")
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                                 name="lr")

    def get_feed_dict(self, x_inputs, y=None, lr=None, dropout=None):

        feed = {
            self.x_inputs: x_inputs
        }

        if y is not None:
            feed[self.y] = y

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed


    def add_logits_op_train(self):
        with tf.variable_scope("conv1_train"):
            w = tf.get_variable("wc1", (8, 8, self.state.get_shape()[1], 32), dtype=tf.float32, initializer=tf.zeros_initializer())
            conv = tf.nn.conv2d(self.state, w, [1, 1, 4, 4], padding='VALID', data_format='NHWC')
            b = tf.get_variable('bc1', [32], initializer=tf.zeros_initializer())
            self.w['wc1'] = w
            self.w['bc1'] = b
            out = tf.nn.bias_add(conv, b, "NHWC")
            out = tf.nn.relu(out)

        with tf.variable_scope("conv2_train"):
            w = tf.get_variable("wc2", (4, 4, 32, 64), dtype=tf.float32, initializer=tf.zeros_initializer())
            conv = tf.nn.conv2d(out, w, [1, 1, 2, 2], padding='VALID', data_format='NHWC')
            b = tf.get_variable('bc2', [64], initializer=tf.zeros_initializer())
            self.w['wc2'] = w
            self.w['bc2'] = b
            out = tf.nn.bias_add(conv, b, "NHWC")
            out = tf.nn.relu(out)

        with tf.variable_scope("conv3_train"):
            w = tf.get_variable("wc3", (3, 3, 64, 64), dtype=tf.float32, initializer=tf.zeros_initializer())
            conv = tf.nn.conv2d(out, w, [1, 1, 1, 1], padding='VALID', data_format='NHWC')
            b = tf.get_variable('bc3', [64], initializer=tf.zeros_initializer())
            self.w['wc3'] = w
            self.w['bc3'] = b
            out = tf.nn.bias_add(conv, b, "NHWC")
            out = tf.nn.relu(out)

            shape = out.get_shape().as_list()
            out_flat = tf.reshape(out, [-1, reduce(lambda x,y: x * y, shape[1:])])
            shape = out_flat.get_shape().as_list()

        with tf.variable_scope("fully1_train"):
            w = tf.get_variable('wf1', [shape[1], 512], dtype=tf.float32, initializer=tf.zeros_initializer())
            b = tf.get_variable('bf1', [512], dtype=tf.float32, initializer=tf.zeros_initializer())
            self.w["wf1"] = w
            self.w["bf1"] = b
            out = tf.nn.xw_plus_b(out_flat, w, b)
            out = tf.nn.relu(out)

        with tf.variable_scope("out_train"):
            w = tf.get_variable('wout', [512, self.n_actions], dtype=tf.float32, initializer=tf.zeros_initializer())
            b = tf.get_variable('bout', [self.n_actions], dtype=tf.float32, initializer=tf.zeros_initializer())
            self.w["wout"] = w
            self.w["bout"] = b
            out = tf.nn.xw_plus_b(out_flat, w, b)
            out = tf.nn.relu(out)
            self.q_out = out

    def add_logits_op_target(self):
        with tf.variable_scope("conv1_target"):
            w = tf.get_variable("wc1", (8, 8, self.state.get_shape()[1], 32), dtype=tf.float32, initializer=tf.zeros_initializer())
            conv = tf.nn.conv2d(self.state, w, [1, 1, 4, 4], padding='VALID', data_format='NHWC')
            b = tf.get_variable('bc1', [32], initializer=tf.zeros_initializer())
            self.w_target['wc1'] = w
            self.w_target['bc1'] = b
            out = tf.nn.bias_add(conv, b, "NHWC")
            out = tf.nn.relu(out)

        with tf.variable_scope("conv2_target"):
            w = tf.get_variable("wc2", (4, 4, 32, 64), dtype=tf.float32, initializer=tf.zeros_initializer())
            conv = tf.nn.conv2d(out, w, [1, 1, 2, 2], padding='VALID', data_format='NHWC')
            b = tf.get_variable('bc2', [64], initializer=tf.zeros_initializer())
            self.w_target['wc2'] = w
            self.w_target['bc2'] = b
            out = tf.nn.bias_add(conv, b, "NHWC")
            out = tf.nn.relu(out)

        with tf.variable_scope("conv3_target"):
            w = tf.get_variable("wc3", (3, 3, 64, 64), dtype=tf.float32, initializer=tf.zeros_initializer())
            conv = tf.nn.conv2d(out, w, [1, 1, 1, 1], padding='VALID', data_format='NHWC')
            b = tf.get_variable('bc3', [64], initializer=tf.zeros_initializer())
            self.w_target['wc3'] = w
            self.w_target['bc3'] = b
            out = tf.nn.bias_add(conv, b, "NHWC")
            out = tf.nn.relu(out)

            shape = out.get_shape().as_list()
            out_flat = tf.reshape(out, [-1, reduce(lambda x,y: x * y, shape[1:])])
            shape = out_flat.get_shape().as_list()

        with tf.variable_scope("fully1_target"):
            w = tf.get_variable('wf1', [shape[1], 512], dtype=tf.float32, initializer=tf.zeros_initializer())
            b = tf.get_variable('bf1', [512], dtype=tf.float32, initializer=tf.zeros_initializer())
            self.w_target["wf1"] = w
            self.w_target["bf1"] = b
            out = tf.nn.xw_plus_b(out_flat, w, b)
            out = tf.nn.relu(out)

        with tf.variable_scope("out_target"):
            w = tf.get_variable('wout', [512, self.n_actions], dtype=tf.float32, initializer=tf.zeros_initializer())
            b = tf.get_variable('bout', [self.n_actions], dtype=tf.float32, initializer=tf.zeros_initializer())
            self.w_target["wout"] = w
            self.w_target["bout"] = b
            out = tf.nn.xw_plus_b(out_flat, w, b)
            out = tf.nn.relu(out)
            self.q_target_out = out


    def add_pred_op(self):
        """Defines self.labels_pred
        Gets int labels from the output of the softmax layer. The predicted label is
        the argmax of this layer
        """
        self.y_pred = self.logits

    def add_loss_op(self):
        """Losses for training"""
        loss = tf.losses.mean_squared_error(
            predictions=self.logits, labels=self.y)
        self.loss = tf.reduce_mean(loss)

        # Scalars for tensorboard
        tf.summary.scalar("loss", self.loss)
    def build(self):
        """
        Build the computational graph with functions defined earlier
        """
        self.add_placeholders()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op(self.lr_method, self.lr, self.loss,
                          self.clip)
        self.initialize_session()

    def predict_batch(self, x):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length
        Predict a batch of sentences (list of word_ids)
        """
        fd = self.get_feed_dict(x, dropout=1.0)
        y_pred = self.sess.run(self.y_pred, feed_dict=fd)
        return y_pred

    def run_epoch(self, train, dev, epoch):
        """Performs one complete epoch over the dataset

        Args:
            train: dataset for training that yields tuple of sentences, tags
            dev: dataset for evaluation that yields tuple of sentences, tags
            epoch: (int) index of the current epoch

        Returns:
            acc: (float) current accuracy score over evaluation dataset

        """
        # progbar stuff for logging
        batch_size = self.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)
        shuffle(train)
        for i, (x, y) in enumerate(self.utils.minibatches(train, batch_size)):
            fd = self.get_feed_dict(x, y, self.learning_rate,
                                       self.keep_prob)

            _, train_loss, summary = self.sess.run(
                [self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * nbatches + i)


    def run_evaluate(self, test):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        accs = []
        for x, y in self.utils.minibatches(test, self.batch_size):
            y_pred= self.predict_batch(x)
            for y, y_pred in zip(y, y_pred):
                print("gt: ", y, " pred ", y_pred)
                accs += [(y-y_pred)**2]
        acc = np.mean(accs)
        # set self.acc for Tensorboard visualization
        return {"acc": acc}


    def save_session(self):
        """Saves session = weights"""
        if not os.path.exists(self.dir_model):
            os.makedirs(self.dir_model)
        self.saver.save(self.sess, self.dir_model)
