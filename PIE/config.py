import os
import string
import numpy as np
from .utils import get_logger
from .data_set import DataSet
from .preprocessor import Preprocessor, NUM, UNKNOWN


class Config():
    def __init__(self, build=True, load=True):
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        if build:
            self.build()

        # load if requested (default)
        if load:
            self.load()

    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        with open(self.filename_words, mode='r', encoding='UTF-8') as f:
            self.vocab_words = {word.strip(): idx for idx, word in enumerate(f)}

        with open(self.filename_tags, mode='r', encoding='UTF-8') as f:
            self.vocab_tags = {tag.strip(): idx for idx, tag in enumerate(f)}

        with open(self.filename_chars, mode='r', encoding='UTF-8') as f:
            self.vocab_chars = {char.strip(): idx for idx, char in enumerate(f)}

        self.vocab_chars = {char: idx for idx, char in
                            enumerate([x for x in list(string.printable) if x not in list(string.ascii_uppercase)])}

        self.nwords = len(self.vocab_words)
        self.nchars = len(self.vocab_chars)
        self.ntags = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.preprocessor = Preprocessor(self.vocab_words, self.vocab_tags,
                                         self.vocab_chars)

        # 3. get pre-trained embeddings
        with np.load(self.filename_embedding) as f:
            self.word_embeddings = f['word_embeddings']
            self.char_embeddings = f['char_embeddings']

    def build(self):
        word_vocab = []
        with open(self.filename_glove, mode='r', encoding='UTF-8') as f:
            for line in f:
                word = line.strip().split(' ')[0]
                word_vocab.append(word)

        word_vocab.append(UNKNOWN)
        word_vocab.append(NUM)

        with open(self.filename_words, mode="w", encoding='UTF-8') as f:
            for i, word in enumerate(word_vocab):
                if i != len(word_vocab) - 1:
                    f.write("{}\n".format(word))
                else:
                    f.write(word)

        tag_vocab = set()
        for _, tags in DataSet(self.filename_train):
            tag_vocab.update(tags)

        with open(self.filename_tags, mode="w", encoding='UTF-8') as f:
            for i, tag in enumerate(tag_vocab):
                if i != len(tag_vocab) - 1:
                    f.write("{}\n".format(tag))
                else:
                    f.write(tag)

        char_vocab = {char: idx for idx, char in
                      enumerate([x for x in list(string.printable) if x not in list(string.ascii_uppercase)])}

        with open(self.filename_chars, mode="w", encoding='UTF-8') as f:
            for i, char in enumerate(char_vocab):
                if i != len(char_vocab) - 1:
                    f.write("{}\n".format(char))
                else:
                    f.write(char)

        word_embeddings = np.zeros([len(word_vocab), self.dim_word]) if self.use_pretrained else None
        char_embeddings = np.zeros([len(char_vocab), self.dim_char]) if self.use_pretrained else None
        if self.use_pretrained:
            word_idx = 0
            with open(self.filename_glove, mode='r', encoding='UTF-8') as f:
                for line in f:
                    line = line.strip().split(' ')
                    embedding = [float(x) for x in line[1:]]
                    word_embeddings[word_idx] = np.asarray(embedding)
                    word_idx += 1

                    if line[0] in char_vocab:
                        char_embeddings[char_vocab[line[0]]] = np.asarray(embedding)

        np.savez_compressed(self.filename_embedding, word_embeddings=word_embeddings, char_embeddings=char_embeddings)

    # general config
    dir_output = "output/"
    dir_model = dir_output + "PIE.weights/"
    path_log = dir_output + "log.txt"

    # embeddings
    dim_word = 50
    dim_char = dim_word

    # glove files
    filename_glove = "data/word_vectors/glove.6B.{}d.txt".format(dim_word)
    use_pretrained = True

    filename_dev = "data/conll2003/en/valid.txt"
    filename_test = "data/conll2003/en/test.txt"
    filename_train = "data/conll2003/en/train.txt"

    max_iter = None  # if not None, max number of examples in Dataset

    # vocab
    filename_words = "data/words.txt"
    filename_tags = "data/tags.txt"
    filename_chars = "data/chars.txt"
    # embedding
    filename_embedding = "data/embedding.npz"

    # training
    train_embeddings = False
    nepochs = 100
    dropout = 0.68
    batch_size = 32
    lr_method = "adam"
    lr = 0.005
    lr_decay = 0.9
    clip = -1  # if negative, no clipping
    nepoch_no_imprv = 10

    # PIE hyperparameters
    hidden_size_char = 100  # lstm on chars
    hidden_size_lstm = 100  # lstm on word embeddings

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True  # if crf, training is 1.7x slower on CPU
    use_chars = True  # if char embedding, training is 3.5x slower on CPU
