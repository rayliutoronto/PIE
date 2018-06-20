NUM = '$NUM$'
UNKNOWN = '$UNKNOWN'

class Preprocessor(object):
    def __init__(self, word_vocab=None, tag_vocab=None, char_vocab=None):
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab
        self.char_vocab = char_vocab

    def word(self, word, use_chars=True, lowercase=True, allow_unknow=True):
        if use_chars:
            char_ids = []
            for char in word:
                char_ids.append(self.char_vocab[char.lower() if lowercase else char])

        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        if word in self.word_vocab:
            word = self.word_vocab[word]
        else:
            if allow_unknow:
                word = self.word_vocab[UNKNOWN]
            else:
                raise Exception("Found unknown word in vocabulary: {}".format(word))

        if use_chars:
            return char_ids, word
        else:
            return word

    def tag(self, tag):
        if tag in self.tag_vocab:
            tag = self.tag_vocab[tag]
        else:
            raise Exception("Found unknown tag in vocabulary: {}".format(tag))

        return tag
