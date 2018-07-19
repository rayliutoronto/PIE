from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import numpy as np
import tensorflow as tf
from conf import Conf
from progress_bar import ProgressBar

from data import Data, Preprocessor, Postprocessor, DataSet


class Model(object):
    def __init__(self, config):
        self.config = config
        self.logger = config.logger
        self.data = Data(self.config)
        self.data.load_vocab()
        self.data.load_shrunk_embedding()

    def __build(self):
        self.__add_placeholders()
        self.__add_embedding_op()
        self.__add_logits_op()
        self.__add_loss_op()

        self.__add_train_op(self.lr, self.loss, self.config.clip)

    def __add_placeholders(self):
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(dtype=tf.int64, shape=[None, None], name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.count_nonzero(self.word_ids, axis=1, name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(dtype=tf.int64, shape=[None, None, None], name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(dtype=tf.int32, shape=[None, None], name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(dtype=tf.int64, shape=[None, None], name='labels')

        self.lr = tf.placeholder(dtype=tf.float32, name='learning_rate')

        self.dropout = tf.placeholder(dtype=tf.float32, name='dropout')

    def __add_embedding_op(self):
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

    def __add_logits_op(self):
        """
        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, self.word_embeddings,
                sequence_length=self.sequence_lengths, dtype=tf.float32)
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

    def __add_loss_op(self):
        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
            self.logits, self.labels, self.sequence_lengths)
        self.trans_params = trans_params  # need to evaluate it for decoding
        self.loss = tf.reduce_mean(-log_likelihood)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)

    def __add_train_op(self, lr, loss, clip):
        with tf.variable_scope("train_step"):
            optimizer = tf.train.AdamOptimizer(lr)
            if clip > 0:  # gradient clipping if clip is positive
                grads, vs = zip(*optimizer.compute_gradients(loss))
                grads, gnorm = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)

    def __add_summary(self, session):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_dir_root, session.graph)

    def __count_train_batchs(self, session):
        self.train_batchs = 0

        session.run(self.train_ds_op)
        while True:
            try:
                batch = session.run(self.batch)
                self.train_batchs += 1
            except tf.errors.OutOfRangeError:
                break

    def __run_epoch(self, session, epoch):
        session.run(self.train_ds_op)

        progress = ProgressBar(target=self.train_batchs)
        prog_count = 0

        while True:
            try:
                batch = session.run(self.batch)

                char_ids, word_lengths = Preprocessor.pad_sequences(
                    np.array([pickle.loads(x[0]) for x in batch['char_ids']]), pad_tok=0, nlevels=2)

                _, train_loss, summary = session.run([self.train_op, self.loss, self.merged],
                                                     feed_dict={self.word_ids: batch['word_ids'],
                                                                self.char_ids: char_ids,
                                                                self.word_lengths: word_lengths,
                                                                self.labels: batch['tag_ids'],
                                                                self.lr: self.config.lr,
                                                                self.dropout: self.config.dropout})

                prog_count += 1
                progress.update(prog_count, [("train loss", train_loss)])

                # tensorboard
                if prog_count % 10 == 0:
                    self.file_writer.add_summary(summary, epoch * self.train_batchs + prog_count)

            except tf.errors.OutOfRangeError:
                break

        metrics = self.__run_evaluate(session)
        msg = " - ".join(["{} {:04.2f}".format(k, v) for k, v in metrics.items()])
        self.logger.info(msg)
        print(msg)

        return metrics["f1"]

    def train(self):
        self.train_ds_op, self.valid_ds_op, self.batch = DataSet(self.config).load()

        self.__build()

        self.logger.info("Initializing tensorflow session")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            best_score = 0
            num_epoch_no_imprv = 0
            self.__add_summary(sess)

            self.__count_train_batchs(sess)

            for epoch in range(self.config.num_epoch):
                self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.config.num_epoch))

                score = self.__run_epoch(sess, epoch)
                self.config.lr *= self.config.lr_decay  # decay learning rate

                # early stopping and saving best parameters
                if score >= best_score:
                    num_epoch_no_imprv = 0
                    self.__save_session(sess)
                    best_score = score
                    self.logger.info("- new best score!")
                else:
                    num_epoch_no_imprv += 1
                    if num_epoch_no_imprv >= self.config.num_epoch_no_imprv:
                        self.logger.info("- early stopping {} epochs without improvement".format(num_epoch_no_imprv))
                        break

    def __run_evaluate(self, session):
        session.run(self.valid_ds_op)

        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.

        while True:
            try:
                batch = session.run(self.batch)

                char_ids, word_lengths = Preprocessor.pad_sequences(
                    np.array([pickle.loads(x[0]) for x in batch['char_ids']]), pad_tok=0, nlevels=2)

                labels_pred = []
                logits, trans_params, sequence_lengths = session.run(
                    [self.logits, self.trans_params, self.sequence_lengths],
                    feed_dict={self.word_ids: batch['word_ids'],
                               self.char_ids: char_ids,
                               self.word_lengths: word_lengths,
                               self.dropout: 1.0})

                # iterate over the sentences because no batching in vitervi_decode
                for logit, sequence_length in zip(logits, sequence_lengths):
                    logit = logit[:sequence_length]  # keep only the valid steps
                    viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
                    labels_pred += [viterbi_seq]

                for lab, lab_pred, length in zip(batch['tag_ids'], labels_pred, sequence_lengths):
                    lab = lab[:length]
                    lab_pred = lab_pred[:length]
                    accs += [a == b for (a, b) in zip(lab, lab_pred)]

                    lab_chunks = set(Postprocessor.get_chunks(lab, self.data.tag_vocab))
                    lab_pred_chunks = set(Postprocessor.get_chunks(lab_pred, self.data.tag_vocab))

                    correct_preds += len(lab_chunks & lab_pred_chunks)
                    total_preds += len(lab_pred_chunks)
                    total_correct += len(lab_chunks)
            except tf.errors.OutOfRangeError:
                break

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return {"acc": 100 * acc, "f1": 100 * f1}

    def __export_saved_model(self, session):
        os.makedirs(self.config.saved_model_dir, exist_ok=True)

        builder = tf.saved_model.builder.SavedModelBuilder(self.config.saved_model_dir)

        tensor_info_word_ids = tf.saved_model.utils.build_tensor_info(self.word_ids)
        tensor_info_char_ids = tf.saved_model.utils.build_tensor_info(self.char_ids)
        # tensor_info_word_lengths = tf.saved_model.utils.build_tensor_info(self.word_lengths)

        tensor_info_y = tf.saved_model.utils.build_tensor_info(self.logits)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'word_ids': tensor_info_word_ids, 'char_ids': tensor_info_char_ids},
                outputs={'tag': tensor_info_y},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        builder.add_meta_graph_and_variables(
            session, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict':
                    prediction_signature
            },
            legacy_init_op=legacy_init_op)

        builder.save()

    def __save_session(self, session):
        os.makedirs(self.config.saved_session_dir, exist_ok=True)

        tf.train.Saver().save(session, self.config.saved_session_dir)

    def __restore_session(self, session):
        tf.train.Saver().restore(session, self.config.saved_session_dir)


# def predict(self, words_raw):
#     """Returns list of tags
#
#     Args:
#         words_raw: list of words (string), just one sentence (no batch)
#
#     Returns:
#         preds: list of tags (string), one for each word in the sentence
#
#     """
#     words = [self.config.preprocessor.word(w) for w in words_raw]
#     if type(words[0]) == tuple:
#         words = zip(*words)
#     pred_ids, _ = self.predict_batch([words])
#     preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]
#
#     return preds


if __name__ == '__main__':
    model = Model(Conf())
    model.train()
