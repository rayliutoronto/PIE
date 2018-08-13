from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import prediction_service_pb2, predict_pb2

from PIE.config import Config
from PIE.data import Data, Preprocessor, TFRecordManager
from PIE.tokenizer import Tokenizer


class Prediction(object):
    def __init__(self, config, host='localhost', port=9000):
        self.config = config
        self.data = Data(self.config)
        self.data.load_vocab()
        self.preprocessor = Preprocessor(self.data.word_vocab, self.data.tag_vocab, self.data.char_vocab)
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(
            implementations.insecure_channel(host, port))

        self.tokenizer = Tokenizer('en')

    def predict_json_object(self, json_object):
        # TODO support depth > 1, no inner structure, array, etc is supported yet
        if isinstance(json_object, dict):
            json_object = [json_object]

        doc_field_value_list = []
        doc_field_name_list = []
        for list_item in json_object:
            doc_field_value_list.append([str(list_item[x]).strip() for x in list_item])
            doc_field_name_list.extend([x.strip() for x in list_item])

        return self._predict(doc_field_value_list, doc_field_name_list)

    def predict_json_string(self, json_string):
        json_object = json.loads(json_string, encoding='UTF-8')
        return self.predict_json_object(json_object)

    def predict_sentence(self, sentence):
        return self._predict([[sentence]])

    def _predict(self, documents, headers=None):
        """

        :param header:
        :param document: 2-D list is expected
        :return:
        """
        documents_words_raw = [
            [[token.text for token in doc if not token.is_space] for doc in self.tokenizer.split(document)] for document
            in documents]

        documents_word_ids = []
        documents_char_ids = []
        for document_word_raw in documents_words_raw:
            for fields in document_word_raw:
                field_word_ids = []
                field_char_ids = []
                for word in fields:
                    wc = self.preprocessor.word(word)
                    field_word_ids += [wc[1]]
                    field_char_ids += [wc[0]]

                documents_word_ids.append(field_word_ids)
                documents_char_ids.append(field_char_ids)

        word_ids = []
        char_ids = []
        for words, chars in zip(documents_word_ids, documents_char_ids):
            seq_example = tf.train.SequenceExample()
            for word, char in zip(words, chars):
                seq_example.context.feature["words"].int64_list.value.append(word)
                seq_example.feature_lists.feature_list["chars"].feature.add().int64_list.value.extend(char)

            word_id, char_id, _ = TFRecordManager.map_fn_to_sparse(seq_example.SerializeToString())
            word_id, char_id = self._expand_dim_sparse(word_id), self._expand_dim_sparse(char_id)
            word_ids.append(word_id)
            char_ids.append(char_id)
        word_ids, char_ids = tf.sparse_tensor_to_dense(
            tf.sparse_concat(0, word_ids, expand_nonconcat_dim=True)), tf.sparse_tensor_to_dense(
            tf.sparse_concat(0, char_ids, expand_nonconcat_dim=True))

        with tf.Session() as sess:
            word_ids, char_ids = sess.run([word_ids, char_ids])

        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'pie'
        request.inputs['word_ids'].CopyFrom(tf.make_tensor_proto(word_ids, dtype=tf.int64))
        request.inputs['char_ids'].CopyFrom(tf.make_tensor_proto(char_ids, dtype=tf.int64))

        response = self.stub.Predict(request, 10.0)  # 10 seconds timeout
        # TODO raise error message in case of timeout

        labels_pred = tf.contrib.util.make_ndarray(response.outputs['viterbi_sequence'])
        logits = tf.contrib.util.make_ndarray(response.outputs['logits'])
        # tp = tf.contrib.util.make_ndarray(response.outputs['tp'])

        if headers is None:
            return [self.data.idx_tag_vocab[x] for labels in labels_pred for x in labels]
        else:
            label_texts = [[self.data.idx_tag_vocab[x] for x in labels] for labels in labels_pred]
            header_dict = {}
            for i, label_text in enumerate(label_texts):
                field_tag = set()
                for l in label_text:
                    if l not in ['O', ' ']:
                        field_tag.add(l.split('-')[1])
                if len(field_tag) > 0:
                    if headers[i] in header_dict:
                        field_tag.update(header_dict[headers[i]])

                    header_dict[headers[i]] = list(field_tag)

            return json.dumps(header_dict)

    def _expand_dim_sparse(self, sparse):
        indices = tf.map_fn(lambda x: tf.concat([[0], x], axis=0), sparse.indices)
        dense_shape = tf.concat([[1], sparse.dense_shape], axis=0)

        return tf.SparseTensor.from_value(tf.SparseTensor(indices, sparse.values, dense_shape))


if __name__ == '__main__':
    prediction = Prediction(Config(), '192.168.99.101', 9000)

    while True:
        sentence = input("input> ")

        if sentence == 'exit':
            break
        try:
            result = prediction.predict_json_string(sentence)
        except ValueError:
            result = prediction.predict_sentence(sentence)

        print(result)
