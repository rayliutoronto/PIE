from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

import tensorflow as tf


class Config(object):
    def __get_logger(log_filename):
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)

        logger = logging.getLogger('logger')
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(format='%(message)s', level=logging.DEBUG)
        handler = logging.FileHandler(log_filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)

        return logger

    NUM = '$NUM$'
    UNKNOWN = '$UNKNOWN$'

    dim_word = dim_char = 50

    data_dir_root = '../data/'

    data_dir_wordvector = data_dir_root + 'word_vectors/'
    glove_filename = data_dir_wordvector + 'glove.6B.{}d.txt'.format(dim_word)

    data_dir_raw = data_dir_root + 'raw/'

    dataset_dir_root = '../dataset/'
    dataset_dir_train = dataset_dir_root + 'train/'
    dataset_dir_valid = dataset_dir_root + 'valid/'
    dataset_dir_vocab = dataset_dir_root + 'vocab/'
    embedding_filename = dataset_dir_vocab + 'embedding.npz'

    word_vocab_filename = dataset_dir_vocab + 'word.txt'
    char_vocab_filename = dataset_dir_vocab + 'char.txt'
    tag_vocab_filename = dataset_dir_vocab + 'tag.txt'

    num_epoch = 100
    lr = 0.005
    lr_decay = 0.98
    dropout = 0.3
    batch_size = 64 if tf.test.is_gpu_available() else 32
    clip = 0.001  # if negative, no clipping
    patience = 6  # early stop

    hidden_size_char = 100  # lstm on chars
    hidden_size_lstm = 100  # lstm on word embeddings

    train_embeddings = False

    output_dir_root = '../output/'
    log_filename = output_dir_root + 'logs/log.txt'
    logger = __get_logger(log_filename)

    exporter_name = 'BestExport'
    output_dir_savedmodel = output_dir_root + 'export/' + exporter_name
