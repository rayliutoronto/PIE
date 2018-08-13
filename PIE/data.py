from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
import re
import string

import numpy as np
import tensorflow as tf

from PIE.config import Config


class Data(object):
    def __init__(self, config):
        self.config = config

    def __build_word_vocab(self):
        if os.path.exists(self.config.word_vocab_filename):
            return
        else:
            os.makedirs(os.path.dirname(self.config.word_vocab_filename), exist_ok=True)

        word_vocab = [' ']  # ' ' is placeholder for index 0
        print("abs path: ", os.path.abspath(self.config.glove_filename))
        with open(self.config.glove_filename, mode='r', encoding='UTF-8') as f:
            for line in f:
                word = line.strip().split(' ')[0]
                word_vocab.append(word)

        word_vocab.append(self.config.UNKNOWN)
        word_vocab.append(self.config.NUM)

        with open(self.config.word_vocab_filename, mode="w", encoding='UTF-8') as f:
            for i, word in enumerate(word_vocab):
                if i != len(word_vocab) - 1:
                    f.write("{}\n".format(word))
                else:
                    f.write(word)

    def __build_char_vocab(self):
        if os.path.exists(self.config.char_vocab_filename):
            return

        char_vocab = {char: idx for idx, char in
                      enumerate([x for x in list(
                          ' ' + string.digits + string.ascii_lowercase + string.punctuation + '\0')])}  # ' ' is placeholder for index 0, \0 is for unknown char

        with open(self.config.char_vocab_filename, mode="w", encoding='UTF-8') as f:
            for i, char in enumerate(char_vocab):
                if i != len(char_vocab) - 1:
                    f.write("{}\n".format(char))
                else:
                    f.write(char)

    def __build_tag_vocab(self):
        existing_tags = set()
        if os.path.exists(self.config.tag_vocab_filename):
            with open(self.config.tag_vocab_filename, mode='r', encoding='UTF-8') as f:
                for line in f:
                    existing_tags.update([line.strip('\n')])  # strip \n but keep space

        tag_vocab = set(' ')  # ' ' is placeholder for index 0
        _, _, tags = self.__transfer_raw('train.txt')
        tag_vocab.update([e for x in tags for e in x])

        with open(self.config.tag_vocab_filename, mode="a+", encoding='UTF-8') as f:
            tag_list = list(tag_vocab - existing_tags)
            tag_list.sort()
            for i, tag in enumerate(tag_list):
                if i != len(tag_list) - 1:
                    f.write("{}\n".format(tag))
                else:
                    f.write(tag)

    def __shrink_embedding(self):
        if os.path.exists(self.config.embedding_filename):
            return

        self.load_vocab()

        word_embeddings = np.zeros([len(self.word_vocab), self.config.dim_word])
        char_embeddings = np.zeros([len(self.char_vocab), self.config.dim_char])
        with open(self.config.glove_filename, mode='r', encoding='UTF-8') as f:
            for line in f:
                line = line.strip().split(' ')
                embedding = [float(x) for x in line[1:]]
                word_embeddings[self.word_vocab[line[0]]] = np.asarray(embedding)

                if line[0] in self.char_vocab:
                    char_embeddings[self.char_vocab[line[0]]] = np.asarray(embedding)

        # set random embedding for UNKNOWN and NUM, \0
        word_embeddings[len(self.word_vocab) - 2] = np.random.randn(1, self.config.dim_word)[0]  # $UNKNOWN$
        word_embeddings[len(self.word_vocab) - 1] = np.random.randn(1, self.config.dim_word)[0]  # $NUM$
        char_embeddings[len(self.char_vocab) - 1] = np.random.randn(1, self.config.dim_char)[0]  # \0

        np.savez_compressed(self.config.embedding_filename, word_embeddings=word_embeddings,
                            char_embeddings=char_embeddings)

    def load_vocab(self):
        with open(self.config.word_vocab_filename, mode='r', encoding='UTF-8') as f:
            self.word_vocab = {word.strip('\n'): idx for idx, word in enumerate(f)}

        with open(self.config.word_vocab_filename, mode='r', encoding='UTF-8') as f:
            self.idx_word_vocab = {idx: word.strip('\n') for idx, word in enumerate(f)}

        with open(self.config.tag_vocab_filename, mode='r', encoding='UTF-8') as f:
            self.tag_vocab = {tag.strip('\n'): idx for idx, tag in enumerate(f)}

        with open(self.config.tag_vocab_filename, mode='r', encoding='UTF-8') as f:
            self.idx_tag_vocab = {idx: tag.strip('\n') for idx, tag in enumerate(f)}

        with open(self.config.char_vocab_filename, mode='r', encoding='UTF-8') as f:
            self.char_vocab = {char.strip('\n'): idx for idx, char in enumerate(f)}

    def __transfer_raw(self, file_suffix='train.txt', preprocessor=None, tfrecord=None):
        train_files = []
        for root, dirs, files in os.walk(self.config.data_dir_raw):
            for file in files:
                if file.endswith(file_suffix):
                    train_files.append(os.path.join(root, file))

        words_list, chars_list, tags_list = [], [], []
        for train_file in train_files:
            with open(train_file) as f:
                words, chars, tags = [], [], []
                for line in f:
                    line = line.strip()
                    if len(line) == 0 or line.startswith("-DOCSTART-"):
                        if len(words) != 0:
                            words_list.append(words)
                            chars_list.append(chars)
                            tags_list.append(tags)

                            words, chars, tags = [], [], []
                    else:
                        ls = line.split(' ')
                        word, tag = ls[0], ls[-1]
                        if preprocessor is not None:
                            word = preprocessor.word(word)
                            tag = preprocessor.tag(tag)
                            words += [word[1]]
                            chars += [word[0]]
                        else:
                            words += [word]
                        tags += [tag]

            if tfrecord is not None:
                tfrecord_filename = re.sub(r'[\/\\\.]', '_', train_file)
                tfrecord.write(tfrecord_filename, words_list, chars_list, tags_list)
                words_list, chars_list, tags_list = [], [], []

        return words_list, chars_list, tags_list

    def generate_train_tfrecords(self):
        self.__build_word_vocab()
        self.__build_char_vocab()
        self.__build_tag_vocab()
        self.__shrink_embedding()

        self.load_vocab()

        self.__transfer_raw('train.txt', Preprocessor(self.word_vocab, self.tag_vocab, self.char_vocab),
                            TFRecordManager(self.config, True))

    def generate_valid_tfrecords(self):
        self.load_vocab()

        self.__transfer_raw('valid.txt', Preprocessor(self.word_vocab, self.tag_vocab, self.char_vocab),
                            TFRecordManager(self.config, False))

    def load_shrunk_embedding(self):
        with np.load(self.config.embedding_filename) as f:
            self.word_embeddings = f['word_embeddings']
            self.char_embeddings = f['char_embeddings']


class Preprocessor(object):
    def __init__(self, word_vocab=None, tag_vocab=None, char_vocab=None):
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab
        self.char_vocab = char_vocab

    def word(self, word):
        char_ids = []
        for char in word:
            char_ids.append(self.char_vocab[char.lower()] if char.lower() in self.char_vocab else self.char_vocab['\0'])

            # if self.char_vocab[char.lower()] == 0:
            #     pass

        word = word.lower()
        if word.isdigit():
            word = Config.NUM

        if word in self.word_vocab:
            word = self.word_vocab[word]

            # if word == 24177:
            #     pass
        else:
            word = self.word_vocab[Config.UNKNOWN]

        return char_ids, word

    def tag(self, tag):
        if tag in self.tag_vocab:
            tag = self.tag_vocab[tag]
        else:
            raise Exception("Found unknown tag in vocabulary: {}".format(tag))

        return tag


# class Postprocessor(object):
#     @staticmethod
#     def __get_chunk_type(tok, idx_to_tag):
#         """
#         Args:
#             tok: id of token, ex 4
#             idx_to_tag: dictionary {4: "B-PER", ...}
#
#         Returns:
#             tuple: "B", "PER"
#
#         """
#         tag_name = idx_to_tag[tok]
#         tag_class = tag_name.split('-')[0]
#         tag_type = tag_name.split('-')[-1]
#         return tag_class, tag_type
#
#     @staticmethod
#     def get_chunks(seq, tags):
#         """Given a sequence of tags, group entities and their position
#
#         Args:
#             seq: [4, 4, 0, 0, ...] sequence of labels
#             tags: dict["O"] = 4
#
#         Returns:
#             list of (chunk_type, chunk_start, chunk_end)
#
#         Example:
#             seq = [4, 5, 0, 3]
#             tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
#             result = [("PER", 0, 2), ("LOC", 3, 4)]
#
#         """
#         default = tags["O"]
#         idx_to_tag = {idx: tag for tag, idx in tags.items()}
#         chunks = []
#         chunk_type, chunk_start = None, None
#         for i, tok in enumerate(seq):
#             # End of a chunk 1
#             if tok == default and chunk_type is not None:
#                 # Add a chunk.
#                 chunk = (chunk_type, chunk_start, i)
#                 chunks.append(chunk)
#                 chunk_type, chunk_start = None, None
#
#             # End of a chunk + start of a chunk!
#             elif tok != default:
#                 tok_chunk_class, tok_chunk_type = Postprocessor.__get_chunk_type(tok, idx_to_tag)
#                 if chunk_type is None:
#                     chunk_type, chunk_start = tok_chunk_type, i
#                 elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
#                     chunk = (chunk_type, chunk_start, i)
#                     chunks.append(chunk)
#                     chunk_type, chunk_start = tok_chunk_type, i
#             else:
#                 pass
#
#         # end condition
#         if chunk_type is not None:
#             chunk = (chunk_type, chunk_start, len(seq))
#             chunks.append(chunk)
#
#         return chunks


class TFRecordManager(object):
    def __init__(self, config, train=True):
        self.config = config
        self.train = train

    def write(self, filename, words_list, chars_list, tags_list):
        file = (
                   self.config.dataset_dir_train if self.train else self.config.dataset_dir_valid) + filename + ".tfrecords"
        os.makedirs(os.path.dirname(file), exist_ok=True)
        writer = tf.python_io.TFRecordWriter(file)

        for words, chars, tags in zip(words_list, chars_list, tags_list):
            seq_example = tf.train.SequenceExample()

            for word in words:
                seq_example.context.feature["words"].int64_list.value.append(word)

            for char in chars:
                seq_example.feature_lists.feature_list["chars"].feature.add().int64_list.value.extend(char)

            for tag in tags:
                seq_example.context.feature["tags"].int64_list.value.append(tag)

            writer.write(seq_example.SerializeToString())

        writer.close()

    @staticmethod
    def map_fn_to_sparse(record):
        context_feature = {
            "tags": tf.VarLenFeature(tf.int64),
            "words": tf.VarLenFeature(tf.int64)
        }
        sequence_feature = {
            "chars": tf.VarLenFeature(tf.int64)
        }

        context_features, sequence_features = tf.parse_single_sequence_example(record, context_feature,
                                                                               sequence_feature)

        return context_features['words'], sequence_features['chars'], context_features['tags']

    @staticmethod
    def map_fn_to_dense(words, chars, tags):
        return {'char_ids': tf.sparse_tensor_to_dense(chars),
                'word_ids': tf.sparse_tensor_to_dense(words)}, tf.sparse_tensor_to_dense(tags)


class DataSet(object):
    def __init__(self, config):
        self.config = config

    def train(self):
        train_tfrecord_files = []
        for root, dirs, files in os.walk(self.config.dataset_dir_train):
            for file in files:
                if file.endswith(".tfrecords"):
                    train_tfrecord_files.append(os.path.join(root, file))
        # TODO shuffle
        return tf.data.TFRecordDataset(train_tfrecord_files, buffer_size=64000,
                                       num_parallel_reads=multiprocessing.cpu_count()).prefetch(
            self.config.batch_size).map(TFRecordManager.map_fn_to_sparse, multiprocessing.cpu_count()).batch(
            self.config.batch_size).map(TFRecordManager.map_fn_to_dense, multiprocessing.cpu_count()).cache()

    def valid(self):
        valid_tfrecord_files = []
        for root, dirs, files in os.walk(self.config.dataset_dir_valid):
            for file in files:
                if file.endswith(".tfrecords"):
                    valid_tfrecord_files.append(os.path.join(root, file))

        return tf.data.TFRecordDataset(valid_tfrecord_files, buffer_size=64000,
                                       num_parallel_reads=multiprocessing.cpu_count()).prefetch(
            self.config.batch_size).map(TFRecordManager.map_fn_to_sparse, multiprocessing.cpu_count()).batch(
            self.config.batch_size).map(TFRecordManager.map_fn_to_dense, multiprocessing.cpu_count()).cache()


if __name__ == '__main__':
    data = Data(Config())
    data.generate_train_tfrecords()
    data.generate_valid_tfrecords()
