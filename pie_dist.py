from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def print_log(worker_num, arg):
    print("%d: " % worker_num, end=" ")
    print(arg)


def map_fun(args, ctx):
    from datetime import datetime
    import numpy as np
    import tensorflow as tf
    import time

    worker_num = ctx.worker_num
    job_name = ctx.job_name
    task_index = ctx.task_index
    cluster_spec = ctx.cluster_spec
    num_workers = len(cluster_spec['worker'])

    # Delay PS nodes a bit, since workers seem to reserve GPUs more quickly/reliably (w/o conflict)
    if job_name == "ps":
        time.sleep((worker_num + 1) * 5)

    # Parameters
    IMAGE_PIXELS = 28
    hidden_units = 128

    # Get TF cluster and server instances
    cluster, server = ctx.start_cluster_server(1, args.rdma)

    def get_chunk_type(tok, idx_to_tag):
        """
        Args:
            tok: id of token, ex 4
            idx_to_tag: dictionary {4: "B-PER", ...}

        Returns:
            tuple: "B", "PER"

        """
        tag_name = idx_to_tag[tok]
        tag_class = tag_name.split('-')[0]
        tag_type = tag_name.split('-')[-1]
        return tag_class, tag_type

    def get_chunks(seq, tags):
        """Given a sequence of tags, group entities and their position

        Args:
            seq: [4, 4, 0, 0, ...] sequence of labels
            tags: dict["O"] = 4

        Returns:
            list of (chunk_type, chunk_start, chunk_end)

        Example:
            seq = [4, 5, 0, 3]
            tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
            result = [("PER", 0, 2), ("LOC", 3, 4)]

        """
        default = tags["O"]
        idx_to_tag = {idx: tag for tag, idx in tags.items()}
        chunks = []
        chunk_type, chunk_start = None, None
        for i, tok in enumerate(seq):
            # End of a chunk 1
            if tok == default and chunk_type is not None:
                # Add a chunk.
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = None, None

            # End of a chunk + start of a chunk!
            elif tok != default:
                tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
                if chunk_type is None:
                    chunk_type, chunk_start = tok_chunk_type, i
                elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                    chunk = (chunk_type, chunk_start, i)
                    chunks.append(chunk)
                    chunk_type, chunk_start = tok_chunk_type, i
            else:
                pass

        # end condition
        if chunk_type is not None:
            chunk = (chunk_type, chunk_start, len(seq))
            chunks.append(chunk)

        return chunks

    def _pad_sequences(sequences, pad_tok, max_length):
        """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the char to pad with

        Returns:
            a list of list where each sublist has same length
        """
        sequence_padded, sequence_length = [], []

        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
            sequence_padded += [seq_]
            sequence_length += [min(len(seq), max_length)]

        return sequence_padded, sequence_length

    def pad_sequences(sequences, pad_tok, nlevels=1):
        """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the char to pad with
            nlevels: "depth" of padding, for the case where we have characters ids

        Returns:
            a list of list where each sublist has same length

        """
        if nlevels == 1:
            max_length = max(map(lambda x: len(x), sequences))
            sequence_padded, sequence_length = _pad_sequences(sequences,
                                                              pad_tok, max_length)

        elif nlevels == 2:
            max_length_word = max([max(map(lambda x: len(x), seq))
                                   for seq in sequences])
            sequence_padded, sequence_length = [], []
            for seq in sequences:
                # all words are same length now
                sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
                sequence_padded += [sp]
                sequence_length += [sl]

            max_length_sentence = max(map(lambda x: len(x), sequences))
            sequence_padded, _ = _pad_sequences(sequence_padded,
                                                [pad_tok] * max_length_word, max_length_sentence)
            sequence_length, _ = _pad_sequences(sequence_length, 0,
                                                max_length_sentence)

        return sequence_padded, sequence_length

    def feed_dict(batch):
        words, tags = [], []
        for item in batch:
            words.append(item[0])
            tags.append(item[1])

        char_ids, word_ids = zip(*words)
        word_ids, sequence_lengths = pad_sequences(word_ids, 0)
        char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                                               nlevels=2)
        labels, _ = pad_sequences(tags, 0)

        return (word_ids, sequence_lengths, char_ids, word_lengths, labels, 0.005, 0.68), sequence_lengths

    if job_name == "ps":
        server.join()
    elif job_name == "worker":
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % task_index,
                cluster=cluster)):

            # shape = (batch size, max length of sentence in batch)
            word_ids = tf.placeholder(tf.int32, shape=[None, None],
                                      name="word_ids")

            # shape = (batch size)
            sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                                              name="sequence_lengths")

            # shape = (batch size, max length of sentence, max length of word)
            char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                                      name="char_ids")

            # shape = (batch_size, max_length of sentence)
            word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                                          name="word_lengths")

            # shape = (batch size, max length of sentence in batch)
            labels = tf.placeholder(tf.int32, shape=[None, None],
                                    name="labels")

            # hyper parameters
            dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                     name="dropout")
            lr = tf.placeholder(dtype=tf.float32, shape=[],
                                name="lr")

            # read word_embeding from file
            with np.load('/vagrant/data/embedding.npz') as f:
                word_embeddings = f['word_embeddings']
                char_embeddings = f['char_embeddings']
            dim_char = 50
            hidden_size_char = 100
            hidden_size_lstm = 100
            ntags = 18  # rneed to read tags.txt

            with tf.variable_scope("words"):
                _word_embeddings = tf.Variable(
                    word_embeddings,
                    name="_word_embeddings",
                    dtype=tf.float32,
                    trainable=False)

                word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                                                         word_ids, name="word_embeddings")

            with tf.variable_scope("chars"):
                # get char embeddings matrix
                _char_embeddings = tf.Variable(
                    char_embeddings,
                    name="_char_embeddings",
                    dtype=tf.float32,
                    trainable=False)

                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                         char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                                             shape=[s[0] * s[1], s[-2], dim_char], name="char_embeddubg_reshape")
                _word_lengths = tf.reshape(word_lengths, shape=[s[0] * s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(hidden_size_char,
                                                  state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(hidden_size_char,
                                                  state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, char_embeddings,
                    sequence_length=_word_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output,
                                    shape=[s[0], s[1], 2 * hidden_size_char])
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

            word_embeddings = tf.nn.dropout(word_embeddings, dropout)

            with tf.variable_scope("bi-lstm"):
                cell_fw = tf.contrib.rnn.LSTMCell(hidden_size_lstm)
                cell_bw = tf.contrib.rnn.LSTMCell(hidden_size_lstm)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, word_embeddings,
                    sequence_length=sequence_lengths, dtype=tf.float32)
                output = tf.concat([output_fw, output_bw], axis=-1)
                output = tf.nn.dropout(output, dropout)

            with tf.variable_scope("proj"):
                W = tf.get_variable("W", dtype=tf.float32,
                                    shape=[2 * hidden_size_lstm, ntags])

                b = tf.get_variable("b", shape=[ntags],
                                    dtype=tf.float32, initializer=tf.zeros_initializer())

                nsteps = tf.shape(output)[1]
                output = tf.reshape(output, [-1, 2 * hidden_size_lstm])
                pred = tf.matmul(output, W) + b
                logits = tf.reshape(pred, [-1, nsteps, ntags])

            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                logits, labels, sequence_lengths)
            loss = tf.reduce_mean(-log_likelihood)
            tf.summary.scalar("loss", loss)

            global_step = tf.Variable(0)
            train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)

            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

        # Create a "supervisor", which oversees the training process and stores model state into HDFS
        logdir = ctx.absolute_path(args.model)
        print("tensorflow model path: {0}".format(logdir))

        if job_name == "worker" and task_index == 0:
            summary_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())

        if args.mode == "train":
            sv = tf.train.Supervisor(is_chief=(task_index == 0),
                                     logdir=logdir,
                                     init_op=init_op,
                                     summary_op=None,
                                     summary_writer=None,
                                     saver=saver,
                                     global_step=global_step,
                                     stop_grace_secs=300,
                                     save_model_secs=10)
        else:
            sv = tf.train.Supervisor(is_chief=(task_index == 0),
                                     logdir=logdir,
                                     summary_op=None,
                                     saver=saver,
                                     global_step=global_step,
                                     stop_grace_secs=300,
                                     save_model_secs=0)
            output_dir = ctx.absolute_path(args.output)
            output_file = tf.gfile.Open("{0}/part-{1:05d}".format(output_dir, worker_num), mode='w')

        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.
        with sv.managed_session(server.target) as sess:
            print("{0} session ready".format(datetime.now().isoformat()))

            # Loop until the supervisor shuts down or 1000000 steps have completed.
            step = 0
            count = 0
            tf_feed = ctx.get_data_feed(args.mode == "train")

            accs = []
            correct_preds, total_correct, total_preds = 0., 0., 0.
            with open('/vagrant/data/tags.txt', mode='r', encoding='UTF-8') as f:
                vocab_tags = {tag.strip(): idx for idx, tag in enumerate(f)}

            while not sv.should_stop() and step < args.steps:
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.

                feed, _ = feed_dict(tf_feed.next_batch(args.batch_size))
                feeeed = {
                    word_ids: feed[0],
                    sequence_lengths: feed[1],
                    char_ids: feed[2],
                    word_lengths: feed[3],
                    labels: feed[4],
                    lr: 0.005,
                    dropout: 0.68
                }
                # Test trained model






                # using QueueRunners/Readers
                if args.mode == "train":
                    if (step % 100 == 0):
                        print(
                            "{0} step: {1}".format(datetime.now().isoformat(), step))
                    _, summary, step, _logits, _trans_params= sess.run([train_op, summary_op, global_step, logits, trans_params], feed_dict=feeeed)

                    viterbi_sequences = []
                    for logit, sequence_length in zip(_logits, feed[1]):
                        logit = logit[:sequence_length]  # keep only the valid steps
                        viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                            logit, _trans_params)
                        viterbi_sequences += [viterbi_seq]

                    for lab, lab_pred, length in zip(feed[4], viterbi_sequences,
                                                     feed[1]):
                        lab = lab[:length]
                        lab_pred = lab_pred[:length]
                        accs += [a == b for (a, b) in zip(lab, lab_pred)]

                        lab_chunks = set(get_chunks(lab, vocab_tags))
                        lab_pred_chunks = set(get_chunks(lab_pred,
                                                         vocab_tags))

                        correct_preds += len(lab_chunks & lab_pred_chunks)
                        total_preds += len(lab_pred_chunks)
                        total_correct += len(lab_chunks)

                    if sv.is_chief:
                        summary_writer.add_summary(summary, step)
                else:  # args.mode == "inference"
                    feed['dropout'] = 1.0
                    #labels, preds, acc = sess.run([label, prediction, accuracy], feed_dict=feeeed)
                    # results = ["{0} Label: {1}, Prediction: {2}".format(datetime.now().isoformat(), l, p) for l, p in
                    #            zip(labels, preds)]
                    # tf_feed.batch_results(results)
                    # print("results: {0}, acc: {1}".format(results, acc))

            if sess.should_stop() or step >= args.steps:
                tf_feed.terminate()

                p = correct_preds / total_preds if correct_preds > 0 else 0
                r = correct_preds / total_correct if correct_preds > 0 else 0
                f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
                acc = np.mean(accs)
                print(
                    "accuracy: {}".format(acc))
                tf.summary.scalar("acc", acc)

        # Ask for all the services to stop.
        print("{0} stopping supervisor".format(datetime.now().isoformat()))
        sv.stop()

    if job_name == "worker" and task_index == 0:
        summary_writer.close()
