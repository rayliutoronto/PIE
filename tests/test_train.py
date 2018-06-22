import os

from PIE.config import Config
from PIE.data_set import DataSet
from PIE.model import Model


def train(config):
    model = Model(config)
    model.build()

    model.train(DataSet(config.filename_train, config.preprocessor), DataSet(config.filename_dev, config.preprocessor))


def eval(config):
    # build PIE
    model = Model(config)
    model.build()
    model.restore_session(config.dir_model)

    # create dataset
    test = DataSet(config.filename_test, config.preprocessor)
    # evaluate
    return model.run_evaluate(test)


def test_train():
    # config = Config(build=True, load=True)
    #
    # train(config)

    # assert os.path.exists(Config.dir_output + 'tags.txt')
    # assert os.path.exists(Config.dir_output + 'chars.txt')
    # assert os.path.exists(Config.dir_output + 'words.txt')
    # assert os.path.exists(Config.dir_output + 'embedding.npz')

    config = Config(build=False, load=True)
    metrics = eval(config)

    assert metrics.get("f1") > 90


if __name__ == '__main__':
    config = Config(build=True, load=True)
    train(config)
