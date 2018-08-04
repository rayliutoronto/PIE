from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


# def _bytes_feature(value):
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#
#
# def _int64_feature(value):
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def map_fn(records):
    context_features = {
        "tag": tf.VarLenFeature(tf.int64),
        "word": tf.VarLenFeature(tf.int64)
    }
    sequence_features = {
        "char": tf.VarLenFeature(tf.int64)
    }

    tags, chars, words = [], [], []

    # def manipulate_record(self,record):
    #     parsed_features = tf.parse_single_sequence_example(record, context_features, sequence_features)
    #     if self.tags is None:
    #         self.tags = tf.contrib.layers.dense_to_sparse(tf.expand_dims(tf.sparse_tensor_to_dense(parsed_features[0]["tag"]), 0))
    #     else:
    #         self.tags = tf.sparse_concat(axis=1, sp_inputs=[self.tags, tf.contrib.layers.dense_to_sparse(tf.expand_dims(tf.sparse_tensor_to_dense(parsed_features[0]["tag"]), 0))], expand_nonconcat_dim=True)
    #     tags.append(tf.sparse_tensor_to_dense(parsed_features[0]["tag"]))
    #     chars.append(tf.sparse_tensor_to_dense(parsed_features[1]["char"]))
    #     words.append(tf.sparse_tensor_to_dense(parsed_features[0]["word"]))

    parsed_features = tf.parse_single_sequence_example(records, context_features, sequence_features)
    # tags.append(tf.sparse_tensor_to_dense(parsed_features[0]["tag"]))
    # chars.append(tf.sparse_tensor_to_dense(parsed_features[1]["char"]))
    # words.append(tf.sparse_tensor_to_dense(parsed_features[0]["word"]))

    # tf.map_fn(manipulate_record, records)

    # tf.map_fn((lambda x:pickle.loads(x[0])), chars)
    return parsed_features[0]["tag"], parsed_features[0]["word"], parsed_features[1]["char"]


def map_fn2(x, y, z):
    return tf.sparse_tensor_to_dense(x), tf.sparse_tensor_to_dense(y), tf.sparse_tensor_to_dense(z)


if __name__ == '__main__':
    # tf.enable_eager_execution()

    import tensorflow as tf

    labels = tf.constant([[0, 0],[5, 7]], tf.int64)
    predictions = tf.constant([[0, 7],[6, 7]])

    metric = tf.metrics.precision_at_top_k(labels, predictions, 7)

    sess = tf.Session()
    sess.run(tf.local_variables_initializer())

    precision, update = sess.run(metric)
    print(update)  # 0.5

    print("done")
