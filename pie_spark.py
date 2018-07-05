from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime

from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from tensorflowonspark import TFCluster

import pie_dist

sc = SparkContext(conf=SparkConf().setAppName("pie_tf"))
executors = sc._conf.get("spark.executor.instances")
num_executors = int(executors) if executors is not None else 1
num_ps = 1

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", help="number of records per batch", type=int, default=32)
parser.add_argument("--epochs", help="number of epochs", type=int, default=0)
parser.add_argument("--words", help="HDFS path to MNIST images in parallelized format")
parser.add_argument("--tags", help="HDFS path to MNIST labels in parallelized format")
parser.add_argument("--model", help="HDFS path to save/load model during train/test", default="pie_model")
parser.add_argument("--cluster_size", help="number of nodes in the cluster (for Spark Standalone)", type=int,
                    default=num_executors)
parser.add_argument("--output", help="HDFS path to save test/inference output", default="predictions")
parser.add_argument("--readers", help="number of reader/enqueue threads", type=int, default=1)
parser.add_argument("--steps", help="maximum number of steps", type=int, default=1000000)
parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")
parser.add_argument("--mode", help="train|inference", default="train")
parser.add_argument("--rdma", help="use rdma connection", default=False)
args = parser.parse_args()
print("args:", args)


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


def load_dataset(filename, preprocessor):
    with open(filename) as f:
        words, tags = [], []
        words_list, tags_list = [], []
        for line in f:
            line = line.strip()
            if (len(line) == 0 or line.startswith("-DOCSTART-")):
                if len(words) != 0:
                    words_list += [zip(*words)]
                    tags_list += [tags]
                    words, tags = [], []
            else:
                ls = line.split(' ')
                word, tag = ls[0], ls[-1]
                word = preprocessor.word(word)
                tag = preprocessor.tag(tag)
                words += [word]
                tags += [tag]

    return sc.parallelize(words_list), sc.parallelize(tags_list)




print("{0} ===== Start".format(datetime.now().isoformat()))

vocab_words = {word.strip(): idx for idx, word in enumerate(sc.textFile('output/words.txt').collect())}
vocab_tags = {tag.strip(): idx for idx, tag in enumerate(sc.textFile('output/tags.txt').collect())}
vocab_chars = {char.strip(): idx for idx, char in enumerate(sc.textFile('output/chars.txt').collect())}
preprocessor = Preprocessor(vocab_words, vocab_tags, vocab_chars)
train_x, train_y = load_dataset('data/conll2003/en/train.txt', preprocessor)
valid_x, valid_y = load_dataset('data/conll2003/en/test.txt', preprocessor)

cluster = TFCluster.run(sc, pie_dist.map_fun, args, args.cluster_size, num_ps, args.tensorboard,
                        TFCluster.InputMode.SPARK, log_dir=args.model)
if args.mode == "train":
    cluster.train(train_x.zip(train_y), args.epochs)
    # labelRDD = cluster.inference(valid_x.zip(valid_y))
    # labelRDD.saveAsTextFile(args.output)
else:
    labelRDD = cluster.inference(valid_x.zip(valid_y))
    labelRDD.saveAsTextFile(args.output)

cluster.shutdown()

print("{0} ===== Stop".format(datetime.now().isoformat()))
