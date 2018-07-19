from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

import tensorflow as tf


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def map_fn(record):
    features = {"tag": tf.VarLenFeature(tf.int64),
                "chars": tf.VarLenFeature(tf.string),
                "words": tf.VarLenFeature(tf.int64)}
    parsed_features = tf.parse_example(record, features)
    chars = tf.sparse_tensor_to_dense(parsed_features["chars"], default_value='')
    # tf.map_fn((lambda x:pickle.loads(x.eval())), chars)
    return tf.sparse_tensor_to_dense(parsed_features["tag"]), chars, tf.sparse_tensor_to_dense(parsed_features["words"])


if __name__ == '__main__':
    # tf.enable_eager_execution()

    writer = tf.python_io.TFRecordWriter("1.tfrecords")

    example = tf.train.Example(features=tf.train.Features(feature={
        'tag': _int64_feature([0, 3]),
        'chars': _bytes_feature(pickle.dumps([[4, 5], [7, 8, 9]])),
        'words': _int64_feature([1, 2])}))
    writer.write(example.SerializeToString())

    example = tf.train.Example(features=tf.train.Features(feature={
        'tag': _int64_feature([3, 3, 6]),
        'chars': _bytes_feature(pickle.dumps([[4, 5, 9, 6], [7, 8]])),
        'words': _int64_feature([1, 5, 5])}))
    writer.write(example.SerializeToString())

    # example = tf.train.Example(features=tf.train.Features(feature={
    #     'tag': _int64_feature([3, 3, 6]),
    #     'chars': _bytes_feature(pickle.dumps([[4, 5, 9, 6], [7, 8]])),
    #     'words': _int64_feature([1, 5, 5, 9])}))
    # writer.write(example.SerializeToString())

    writer.close()

    dataset = tf.data.TFRecordDataset(['1.tfrecords']).batch(2).map(map_fn)
    it = dataset.make_initializable_iterator()
    next = it.get_next()

    p = tf.placeholder(dtype=tf.int64, shape=[None, None], name='word_ids')
    z = tf.count_nonzero(p, axis=1)
    z_1 = tf.add(z, 0)

    c = tf.placeholder(dtype=tf.int64, shape=[None, None, None], name='char_ids')

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

                    xxx = [[[1], [2]], [[3], [4]]]
                    _p, _z1, _c = sess.run([p, z_1, c], feed_dict={p: n[0], c: xxx})

                    print(_p, _z1)
                except tf.errors.OutOfRangeError:
                    break

        # print("ddd", pickle.loads(n[1][1]))

    print("done")
