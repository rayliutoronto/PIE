from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from config import Config
from grpc.beta import implementations
from tensorflow_serving.apis import prediction_service_pb2, predict_pb2
from tokenizer import Tokenizer

from data import Data, Preprocessor, TFRecordManager


class Prediction(object):
    def __init__(self, config, host='localhost', port=9000):
        self.config = config
        self.data = Data(self.config)
        self.data.load_vocab()
        self.preprocessor = Preprocessor(self.data.word_vocab, self.data.tag_vocab, self.data.char_vocab)
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(
            implementations.insecure_channel(host, port))

        self.tokenizer = Tokenizer('en')

    def predict(self, sentence):
        words_raw = [token.text for doc in self.tokenizer.split(sentence.strip()) for token in doc if
                     not token.is_space]
        word_ids = []
        char_ids = []
        for word_raw in words_raw:
            wc = self.preprocessor.word(word_raw)
            word_ids += [wc[1]]
            char_ids += [wc[0]]

        seq_example = tf.train.SequenceExample()
        for word, char in zip(word_ids, char_ids):
            seq_example.context.feature["words"].int64_list.value.append(word)
            seq_example.feature_lists.feature_list["chars"].feature.add().int64_list.value.extend(char)

        word_ids, char_ids, _ = TFRecordManager.map_fn_to_sparse(seq_example.SerializeToString())
        word_ids, char_ids = tf.expand_dims(tf.sparse_tensor_to_dense(word_ids), 0), tf.expand_dims(
            tf.sparse_tensor_to_dense(char_ids), 0)
        with tf.Session() as sess:
            word_ids, char_ids = sess.run([word_ids, char_ids])

        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'pie'
        request.inputs['word_ids'].CopyFrom(tf.make_tensor_proto(word_ids, dtype=tf.int64))
        request.inputs['char_ids'].CopyFrom(tf.make_tensor_proto(char_ids, dtype=tf.int64))

        response = self.stub.Predict(request, 10.0)  # 10 seconds timeout

        labels_pred = tf.contrib.util.make_ndarray(response.outputs['viterbi_sequence'])
        logits = tf.contrib.util.make_ndarray(response.outputs['logits'])
        tp = tf.contrib.util.make_ndarray(response.outputs['tp'])

        return [self.data.idx_tag_vocab[x] for x in labels_pred[0]]


if __name__ == '__main__':
    prediction = Prediction(Config(), '192.168.99.100', 9000)

    while True:
        sentence = input("input> ")

        if sentence == 'exit':
            break

        result = prediction.predict(sentence)

        print(result)
