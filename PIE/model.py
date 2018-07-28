from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf
from config import Config
from tensorflow.python.training import session_run_hook

from data import Data, DataSet, Postprocessor


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

    def _model_fn(self, features, labels, mode, params, config):
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.config.dropout_ph = self.config.dropout
            self.config.lr = self.config.lr_decay * self.config.lr
        if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
            self.config.dropout_ph = 1.0

        self._create_model(features, labels, mode)

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode, loss=self.loss, train_op=self.train_op,
                                              training_chief_hooks=[
                                                  CPSaverHook(checkpoint_dir=self.config.output_dir_root,
                                                              save_steps=sys.maxsize // 2)])
        if mode == tf.estimator.ModeKeys.EVAL:
            if self.eval_hook is None:
                self.eval_hook = EvaluationHook(data=self.data, patience=self.config.num_epoch_no_imprv,
                                                config=self.config)

            self.eval_hook.set_fetches(logits=self.logits, trans_params=self.trans_params,
                                       sequence_lengths=self.sequence_lengths, labels=self.labels)

            return tf.estimator.EstimatorSpec(mode, loss=self.loss, evaluation_hooks=[self.eval_hook])

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'logits': self.logits,
                'trans_params': self.trans_params_tensor,
                'sequence_lengths': self.sequence_lengths
            }
            export_outputs = {
                'prediction': tf.estimator.export.PredictOutput(predictions)
            }
            return tf.estimator.EstimatorSpec(
                mode,
                predictions=predictions,
                export_outputs=export_outputs)

    def _create_model(self, features, labels, mode):
        if mode == tf.estimator.ModeKeys.PREDICT:
            labels = tf.zeros_like(features['word_ids'])

        self._add_variables(features, labels)
        self._add_embedding_op()
        self._add_logits_op()
        self._add_loss_op()

        if mode != tf.estimator.ModeKeys.PREDICT:
            self._add_train_op()

    def _add_variables(self, features, labels):
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = features['word_ids']

        # shape = (batch size)
        self.sequence_lengths = tf.count_nonzero(self.word_ids, axis=1, name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = features['char_ids']

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.count_nonzero(self.char_ids, axis=2, name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = labels

        # ??
        self.lr = tf.Variable(self.config.lr, dtype=tf.float32, trainable=False, name='learning_rate')
        tf.summary.scalar('lr', self.lr)
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

    def _add_loss_op(self):
        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
            self.logits, self.labels, self.sequence_lengths)
        self.trans_params = trans_params  # need to evaluate it for decoding
        self.trans_params_tensor = tf.convert_to_tensor(self.trans_params)
        self.loss = tf.reduce_mean(-log_likelihood, name='loss')

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
        # tf.gfile.DeleteRecursively(self.config.output_dir_root)

        run_config = tf.estimator.RunConfig()

        predictor = tf.estimator.Estimator(
            model_fn=self._model_fn,
            model_dir=self.config.output_dir_root,
            config=run_config
        )

        # if there is checkpoint already, need to evaluate first then train
        # update self.best in EvalHook
        if predictor.latest_checkpoint() is not None:
            predictor.evaluate(input_fn=self._valid_input_fn)

        try:
            tf.estimator.train_and_evaluate(estimator=predictor,
                                            train_spec=tf.estimator.TrainSpec(input_fn=self._train_input_fn),
                                            eval_spec=tf.estimator.EvalSpec(input_fn=self._valid_input_fn,
                                                                            start_delay_secs=0))
        except RuntimeError:
            # workaround to exit training loop when no evaluation performance improvement after long epochs.
            pass
        # estimator.train does not work in distributed training
        # predictor.train(input_fn=self._train_input_fn)
        # predictor.evaluate(input_fn=self._valid_input_fn)

        if run_config.is_chief and self.config.should_export_savedmodel:
            predictor.export_savedmodel(self.config.output_dir_root, self._create_serving_input_receiver)

        # predictions = list(predictor.predict(input_fn=self._valid_input_fn, yield_single_examples=False))
        # predicted_classes = [p["classes"] for p in predictions]
        # print("New Samples, Class Predictions:    {}\n".format(predicted_classes))


class EvaluationHook(session_run_hook.SessionRunHook):
    def __init__(self, data, patience, config):
        self.data = data

        self.patience = patience
        self.wait = 0
        self.best = -np.Inf

        self.config = config

        self.accs = []
        self.correct_preds, self.total_correct, self.total_preds = 0., 0., 0.

        self.epoch = 0

        self.last_cp = None

    def set_fetches(self, logits, trans_params, sequence_lengths, labels):
        self.logits = logits
        self.trans_params = trans_params
        self.sequence_lengths = sequence_lengths
        self.labels = labels

    def begin(self):
        self.accs = []
        self.correct_preds, self.total_correct, self.total_preds = 0., 0., 0.

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return session_run_hook.SessionRunArgs([self.logits, self.trans_params, self.sequence_lengths, self.labels])

    def after_run(self, run_context, run_values):
        logits, trans_params, sequence_lengths, labels = run_values.results

        labels_pred = []
        # iterate over the sentences because no batching in vitervi_decode
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
            labels_pred += [viterbi_seq]

        for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
            lab = lab[:length]
            lab_pred = lab_pred[:length]
            self.accs += [a == b for (a, b) in zip(lab, lab_pred)]

            lab_chunks = set(Postprocessor.get_chunks(lab, self.data.tag_vocab))
            lab_pred_chunks = set(Postprocessor.get_chunks(lab_pred, self.data.tag_vocab))

            self.correct_preds += len(lab_chunks & lab_pred_chunks)
            self.total_preds += len(lab_pred_chunks)
            self.total_correct += len(lab_chunks)

    def end(self, session):
        self.epoch += 1

        p = self.correct_preds / self.total_preds if self.correct_preds > 0 else 0
        r = self.correct_preds / self.total_correct if self.correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if self.correct_preds > 0 else 0
        acc = np.mean(self.accs)

        eval_result = {"acc": 100 * acc, "f1": 100 * f1}
        print('======================Evaluation Result===========================')
        print(eval_result, 'Epoch: ', self.epoch)

        if f1 > self.best:
            self.best = f1
            self.wait = 0
            self.last_cp = tf.train.latest_checkpoint(self.config.output_dir_root)
            self.config.should_export_savedmodel = True
            print('New Best F1 Score: ', 100 * f1)

            # clear previous checkpoint files
            latest_cp = tf.train.latest_checkpoint(self.config.output_dir_root)
            for file in set(tf.gfile.Glob(self.config.output_dir_root + 'model.ckpt' + '*')) - set(
                    tf.gfile.Glob(latest_cp + '*')):
                tf.gfile.Remove(file)
            tf.train.update_checkpoint_state(save_dir=self.config.output_dir_root,
                                             model_checkpoint_path=latest_cp,
                                             all_model_checkpoint_paths=[latest_cp])
        else:
            # clear last checkpoint file
            latest_cp = tf.train.latest_checkpoint(self.config.output_dir_root)
            if self.last_cp != latest_cp:
                for file in tf.gfile.Glob(latest_cp + '*'):
                    tf.gfile.Remove(file)

                # workaround to override evaluation checking: no new checkpoint, no evaluation
                # increase number by 1
                old_global_step = self.last_cp.split('-')[-1]
                new_global_step = int(old_global_step) + 1
                for file in tf.gfile.Glob(self.last_cp + '*'):
                    tf.gfile.Rename(file, file.replace('-' + old_global_step + '.', '-' + new_global_step + '.'))

                self.last_cp = self.last_cp.replace('-' + old_global_step, '-' + new_global_step)

                tf.train.update_checkpoint_state(save_dir=self.config.output_dir_root,
                                                 model_checkpoint_path=self.last_cp,
                                                 all_model_checkpoint_paths=[self.last_cp])

            # increase global step by 1 to override evaluation checking: no new checkpoint, no evaluation
            with tf.Session() as sess:
                sess.run(tf.add(tf.train.get_global_step, tf.constant(1)))

            self.wait += 1
            print('# epochs with no improvement: ', self.wait)
            if self.wait >= self.patience:
                raise RuntimeError('Can not make progress!')

        print('======================Evaluation Result===========================')


class CPSaverHook(tf.train.CheckpointSaverHook):
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
