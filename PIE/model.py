from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf
from config import Config
from tensorflow.python.training import session_run_hook

from data import Data, DataSet


class Model(object):
    def __init__(self, config):
        self.config = config
        self.logger = config.logger
        self.data = Data(self.config)
        self.data.load_vocab()
        self.data.load_shrunk_embedding()

        self.dataset = DataSet(self.config)

        self.eval_hook = None

    def _train_input_fn(self):
        return self.dataset.train()

    def _valid_input_fn(self):
        return self.dataset.valid()

    def _create_serving_input_receiver(self):
        inputs = {'word_ids': tf.placeholder(dtype=tf.int64, shape=[None, None], name="word_ids"),
                  'char_ids': tf.placeholder(dtype=tf.int64, shape=[None, None, None], name="char_ids")}
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    def _model_fn(self, features, labels, mode):
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.config.dropout_ph = self.config.dropout
            self.config.lr = self.config.lr_decay * self.config.lr
        if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
            self.config.dropout_ph = 1.0

        self._create_model(features, labels, mode)

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode, loss=self.loss, train_op=self.train_op,
                                              training_chief_hooks=[_CPSaverHook(
                                                  checkpoint_dir=self.config.output_dir_root,
                                                  save_steps=sys.maxsize // 2)])
        if mode == tf.estimator.ModeKeys.EVAL:
            if self.eval_hook is None:
                self.eval_hook = _EvaluationHook(model=self)

            return tf.estimator.EstimatorSpec(mode, loss=-self.f1[1], eval_metric_ops={
                'accuracy': self.accuracy,
                'f1': self.f1
            }, evaluation_hooks=[self.eval_hook])

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'viterbi_sequence': self.viterbi_sequence,
                'logits': self.logits,
                "tp": self.trans_params_v
            }
            export_outputs = {
                'prediction': tf.estimator.export.PredictOutput(predictions)
            }
            return tf.estimator.EstimatorSpec(
                mode,
                predictions=predictions,
                export_outputs=export_outputs)

    def _create_model(self, features, labels, mode):
        self._add_variables(features, labels, mode)
        self._add_embedding_op()
        self._add_logits_op()
        if mode in [tf.estimator.ModeKeys.TRAIN]:
            self._add_loss_op()
            self._add_train_op()
        if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
            self._add_transition_parameter()
            self._add_prediction_op()
        if mode in [tf.estimator.ModeKeys.EVAL]:
            self._add_accuracy_op()

    def _add_variables(self, features, labels, mode):
        with tf.variable_scope("variable"):
            # shape = (batch size, max length of sentence in batch)
            self.word_ids = tf.cast(features['word_ids'], dtype=tf.int32, name='word_ids')

            # shape = (batch size)
            self.sequence_lengths = tf.count_nonzero(self.word_ids, axis=1, name="sequence_lengths", dtype=tf.int32)

            # shape = (batch size, max length of sentence, max length of word)
            self.char_ids = tf.cast(features['char_ids'], dtype=tf.int32, name='char_ids')

            # shape = (batch_size, max_length of sentence)
            self.word_lengths = tf.count_nonzero(self.char_ids, axis=2, name="word_lengths", dtype=tf.int32)

            # shape = (batch size, max length of sentence in batch)
            if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
                self.labels = tf.cast(labels, dtype=tf.int32, name='labels')

            self.lr = tf.Variable(self.config.lr, dtype=tf.float32, trainable=False, name='learning_rate')
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

    def _add_loss_op(self):
        with tf.variable_scope("loss_op"):
            log_likelihood, self.trans_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.labels, self.sequence_lengths)
            self.loss = tf.reduce_mean(-log_likelihood, name='loss')

            self.trans_params_v = tf.convert_to_tensor(self.trans_params)

    def _add_transition_parameter(self):
        with tf.variable_scope("loss_op"):
            self.trans_params = tf.get_variable('transitions', [len(self.data.tag_vocab), len(self.data.tag_vocab)])

    def _add_train_op(self):
        with tf.variable_scope("train_op"):
            optimizer = tf.train.AdamOptimizer(self.lr)
            if self.config.clip > 0:  # gradient clipping if clip is positive
                grads, vs = zip(*optimizer.compute_gradients(self.loss))
                grads, gnorm = tf.clip_by_global_norm(grads, self.config.clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss=self.loss, global_step=tf.train.get_or_create_global_step())

    def _add_prediction_op(self):
        with tf.variable_scope("prediction_op"):
            self.viterbi_sequence, self.viterbi_score = tf.contrib.crf.crf_decode(self.logits, self.trans_params,
                                                                                  self.sequence_lengths)

    def _add_accuracy_op(self):
        with tf.variable_scope('accuracy_op'):
            self.accuracy = tf.metrics.accuracy(self.labels, self.viterbi_sequence)
            self.precision = tf.metrics.precision_at_top_k(tf.cast(self.labels, tf.int64), self.viterbi_sequence,
                                                           len(self.data.tag_vocab))
            self.recall = tf.metrics.recall_at_top_k(tf.cast(self.labels, tf.int64), self.viterbi_sequence,
                                                     len(self.data.tag_vocab))

            self.f1 = (2.0 * self.precision[0] * self.recall[0] / (self.precision[0] + self.recall[0]),
                       2.0 * self.precision[1] * self.recall[1] / (self.precision[1] + self.recall[1]))

    def run(self, _):
        self.run_config = tf.estimator.RunConfig(keep_checkpoint_max=3)

        self.predictor = tf.estimator.Estimator(
            model_fn=self._model_fn,
            model_dir=self.config.output_dir_root,
            config=self.run_config
        )

        # if there is checkpoint already, need to evaluate first then train
        # update self.best in EvalHook
        # if self.predictor.latest_checkpoint() is not None:
        #     self.predictor.evaluate(input_fn=self._valid_input_fn)

        tf.gfile.MakeDirs(self.config.output_dir_savedmodel)

        def _f1_bigger(best_eval_result, current_eval_result):
            return best_eval_result['f1'] < current_eval_result['f1']

        try:
            tf.estimator.train_and_evaluate(estimator=self.predictor,
                                            train_spec=tf.estimator.TrainSpec(input_fn=self._train_input_fn),
                                            eval_spec=tf.estimator.EvalSpec(input_fn=self._valid_input_fn,
                                                                            start_delay_secs=0,
                                                                            exporters=tf.estimator.BestExporter(
                                                                                name=self.config.exporter_name,
                                                                                serving_input_receiver_fn=self._create_serving_input_receiver,
                                                                                exports_to_keep=2,
                                                                                compare_fn=_f1_bigger)))
        except RuntimeError:
            # workaround to exit training loop when no evaluation performance improvement after long epochs.
            pass
        # estimator.train does not work in distributed training
        # predictor.train(input_fn=self._train_input_fn)
        # predictor.evaluate(input_fn=self._valid_input_fn)


class _EvaluationHook(session_run_hook.SessionRunHook):
    def __init__(self, model):
        self.wait = 0
        self.best = -np.Inf

        self.model = model

        self.f1 = 0

        self.epoch = 0

    def begin(self):
        self.f1 = 0

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return session_run_hook.SessionRunArgs([self.model.f1[1]])

    def after_run(self, run_context, run_values):
        self.f1 = run_values.results[0]

    def end(self, session):
        self.epoch += 1

        print('======================Evaluation Result===========================')
        print('F1: ', 100 * self.f1, '\tEpoch: ', self.epoch)

        if self.f1 > self.best:
            self.best = self.f1
            self.wait = 0
            print('New Best F1 Score!')
            print('======================Evaluation Result===========================')
        else:
            self.wait += 1
            print('# epochs with no improvement: ', self.wait)
            print('======================Evaluation Result===========================')
            if self.wait >= self.model.config.patience:
                raise RuntimeError('Can not make progress!')


class _CPSaverHook(tf.train.CheckpointSaverHook):
    def after_create_session(self, session, coord):
        # override parent class to disable checkpoint file writing
        pass

    def after_run(self, run_context, run_values):
        # override parent class to disable checkpoint file writing
        pass


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    # tf.enable_eager_execution()

    model = Model(Config())

    tf.app.run(main=model.run)
