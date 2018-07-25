from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from config import Config
from early_stopping_hook import EarlyStoppingHook

from data import Data, DataSet


class Model(object):
    def __init__(self, config):
        self.config = config
        self.logger = config.logger
        self.data = Data(self.config)
        self.data.load_vocab()
        self.data.load_shrunk_embedding()

        self.dataset = DataSet(self.config)

    def _train_input_fn(self):
        return self.dataset.train()

    def _valid_input_fn(self):
        return self.dataset.valid()

    def _model_fn(self, features, mode, params):
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.config.dropout_ph = self.config.dropout
            self.config.lr = self.config.lr_decay * self.config.lr
        if mode == tf.estimator.ModeKeys.EVAL:
            self.config.dropout_ph = 1.0

        self._create_model(features)

        if mode == tf.estimator.ModeKeys.TRAIN:
            logging_hook = tf.train.LoggingTensorHook({"loss": self.loss}, every_n_iter=10)
            return tf.estimator.EstimatorSpec(mode, loss=self.loss, train_op=self.train_op,
                                              training_hooks=[logging_hook])
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=self.loss, eval_metric_ops={
                'accuracy': self.acc
            })

    def _create_model(self, features):
        self._add_variables(features)
        self._add_embedding_op()
        self._add_logits_op()
        self._add_loss_op()
        self._add_accuracy_op()
        self._add_train_op()

    def _add_variables(self, features):
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = features['word_ids']

        # shape = (batch size)
        self.sequence_lengths = tf.count_nonzero(self.word_ids, axis=1, name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = features['char_ids']

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.count_nonzero(self.char_ids, axis=2, name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = features['tag_ids']

        # ??
        self.lr = tf.Variable(self.config.lr, dtype=tf.float32, trainable=False, name='learning_rate')
        # ??
        self.dropout = tf.Variable(self.config.dropout_ph, dtype=tf.float32, trainable=False, name='dropout')

    def _add_embedding_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(
                self.data.word_embeddings,
                name="_word_embeddings",
                dtype=tf.float32,
                trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            _char_embeddings = tf.Variable(  # ?? get_variable
                self.data.char_embeddings,
                name="_char_embeddings",
                dtype=tf.float32,
                trainable=self.config.train_embeddings)

            char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.char_ids, name="char_embeddings")

            # put the time dimension on axis=1
            s = tf.shape(char_embeddings)
            char_embeddings = tf.reshape(char_embeddings, shape=[s[0] * s[1], s[-2], self.config.dim_char])
            word_lengths = tf.reshape(self.word_lengths, shape=[s[0] * s[1]])

            # bi lstm on chars
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char, state_is_tuple=True)
            _output = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, char_embeddings,
                sequence_length=word_lengths, dtype=tf.float32)

            # read and concat output
            _, ((_, output_fw), (_, output_bw)) = _output
            output = tf.concat([output_fw, output_bw], axis=-1)

            # shape = (batch size, max sentence length, char hidden size)
            output = tf.reshape(output, shape=[s[0], s[1], 2 * self.config.hidden_size_char])
            word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

    def _add_logits_op(self):
        """
        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.word_embeddings,
                                                                        sequence_length=self.sequence_lengths,
                                                                        dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                                shape=[2 * self.config.hidden_size_lstm, len(self.data.tag_vocab)])

            b = tf.get_variable("b", shape=[len(self.data.tag_vocab)],
                                dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2 * self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, len(self.data.tag_vocab)])

    def _add_accuracy_op(self):
        self.acc = tf.metrics.accuracy(labels=self.labels, predictions=tf.argmax(self.logits, axis=2))

    def _add_loss_op(self):
        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
            self.logits, self.labels, self.sequence_lengths)
        self.trans_params = trans_params  # need to evaluate it for decoding
        self.loss = tf.reduce_mean(-log_likelihood, name='loss')

        # for tensorboard
        tf.summary.scalar("loss", self.loss)

    def _add_train_op(self):
        with tf.variable_scope("train_step"):
            optimizer = tf.train.AdamOptimizer(self.lr)
            if self.config.clip > 0:  # gradient clipping if clip is positive
                grads, vs = zip(*optimizer.compute_gradients(self.loss))
                grads, gnorm = tf.clip_by_global_norm(grads, self.config.clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss=self.loss, global_step=tf.train.get_or_create_global_step())

    def run(self, _):
        # run_config = tf.estimator.RunConfig()

        predictor = tf.estimator.Estimator(
            model_fn=self._model_fn,
            model_dir=self.config.output_dir_root,
            # warm_start_from=tf.estimator.WarmStartSettings(ckpt_to_initialize_from=self.config.output_dir_root)
        )

        early_stopping_hook = EarlyStoppingHook(estimator=predictor, input_fn=self._valid_input_fn,
                                                patience=self.config.num_epoch_no_imprv)

        for _ in range(self.config.num_epoch):
            predictor.train(input_fn=self._train_input_fn, hooks=[early_stopping_hook])


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    model = Model(Config())

    tf.app.run(main=model.run)
