from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from config import Config
from grpc.beta import implementations
from tensorflow_serving.apis import prediction_service_pb2, predict_pb2
from tokenizer import Tokenizer

from data import Data, Preprocessor


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
        words_raw = [token.text for doc in self.tokenizer.split(sentence.strip()) for token in doc]
        word_ids = []
        char_ids = []
        for word_raw in words_raw:
            wc = self.preprocessor.word(word_raw)
            word_ids += [wc[1]]
            char_ids += [wc[0]]

        exmaples = []
        for word, char in zip(word_ids, char_ids):
            seq_example = tf.train.SequenceExample()
            seq_example.context.feature["word_ids"].int64_list.value.append(word)
            seq_example.feature_lists.feature_list["char_ids"].feature.add().int64_list.value.extend(char)
            exmaples.append(seq_example)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'pie'
        request.inputs['word_ids'].CopyFrom(tf.contrib.util.make_tensor_proto([word_ids], dtype=tf.int64))
        request.inputs['char_ids'].CopyFrom(tf.contrib.util.make_tensor_proto([char_ids], dtype=tf.int64))

        response = self.stub.Predict(request, 10.0)  # 10 seconds timeout

        logits = tf.contrib.util.make_ndarray(response.outputs['logits'])
        trans_params = tf.contrib.util.make_ndarray(response.outputs['trans_params'])
        sequence_lengths = tf.contrib.util.make_ndarray(response.outputs['sequence_lengths'])

        labels_pred = []
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit = logit[:sequence_length]  # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
            labels_pred += [viterbi_seq]

        # convert id to tag

        return labels_pred


if __name__ == '__main__':
    prediction = Prediction(Config(), 'localhost', 9000)

    while True:
        sentence = input("input> ")

        if sentence == 'exit':
            break

        result = prediction.predict(sentence)

        print(result)
