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

    writer = tf.python_io.TFRecordWriter("1.tfrecords")

    req_ex = tf.train.SequenceExample()
    req_ex.context.feature["tag"].int64_list.value.append(1)
    req_ex.context.feature["tag"].int64_list.value.append(3)

    req_ex.context.feature["word"].int64_list.value.append(1)
    req_ex.context.feature["word"].int64_list.value.append(2)

    req_ex.feature_lists.feature_list["char"].feature.add().int64_list.value.extend([4, 5])
    req_ex.feature_lists.feature_list["char"].feature.add().int64_list.value.extend([6, 5, 9])

    writer.write(req_ex.SerializeToString())

    req_ex = tf.train.SequenceExample()
    req_ex.context.feature["tag"].int64_list.value.append(3)
    req_ex.context.feature["tag"].int64_list.value.append(3)
    req_ex.context.feature["tag"].int64_list.value.append(6)

    req_ex.context.feature["word"].int64_list.value.append(1)
    req_ex.context.feature["word"].int64_list.value.append(5)
    req_ex.context.feature["word"].int64_list.value.append(5)

    req_ex.feature_lists.feature_list["char"].feature.add().int64_list.value.extend([4, 5, 9, 6])
    req_ex.feature_lists.feature_list["char"].feature.add().int64_list.value.extend([1, 5, 9])
    req_ex.feature_lists.feature_list["char"].feature.add().int64_list.value.extend([7, 8])

    writer.write(req_ex.SerializeToString())

    # example = tf.train.Example(features=tf.train.Features(feature={
    #     'tag': _int64_feature([3, 3, 6]),
    #     'chars': _bytes_feature(pickle.dumps([[4, 5, 9, 6], [7, 8]])),
    #     'words': _int64_feature([1, 5, 5, 9])}))
    # writer.write(example.SerializeToString())

    writer.close()

    dataset = tf.data.TFRecordDataset(['1.tfrecords']).map(map_fn).batch(2).map(map_fn2)
    it = dataset.make_initializable_iterator()
    next = it.get_next()

    p = tf.placeholder(dtype=tf.int64, shape=[None, None], name='word_ids')
    z = tf.count_nonzero(p, axis=1)
    z_1 = tf.add(z, 0)

    c = tf.placeholder(dtype=tf.int64, shape=[None, None, None], name='char_ids')
    cc = tf.count_nonzero(c, axis=2)

    with tf.Session() as sess:

        for _ in range(2):
            sess.run(it.initializer)
            while True:
                try:
                    n = sess.run(next)

                    # xxx = np.array([pickle.loads(x[0]) for x in n[1]])
                    # #mapfn = lambda x: pickle.loads(x)
                    #
                    # print(xxx)
                    # print(np.array(xxx).shape)

                    _p, _z1, _c, _cc = sess.run([p, z_1, c, cc], feed_dict={p: n[1], c: n[2]})

                    print(_c, _cc)
                except tf.errors.OutOfRangeError:
                    break

        # print("ddd", pickle.loads(n[1][1]))

    print("done")
