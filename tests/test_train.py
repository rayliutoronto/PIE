import os

from PIE.model import Model
from PIE.config import Config
from PIE.data_set import DataSet


def train():
    config = Config(build=True, load=True)

    model = Model(config)
    model.build()

    model.train(DataSet(config.filename_train, config.preprocessor), DataSet(config.filename_dev, config.preprocessor))

def eval():
    # create instance of config
    config = Config(build=False, load=True)

    # build PIE
    model = Model(config)
    model.build()
    model.restore_session(config.dir_model)

    # create dataset
    test = DataSet(config.filename_test, config.preprocessor)
    # evaluate
    return model.run_evaluate(test)

def test_train():
    train()

    assert os.path.exists('data/tags.txt')
    assert os.path.exists('data/chars.txt')
    assert os.path.exists('data/words.txt')
    assert os.path.exists('data/embedding.npz')

    metrics = eval()

    assert metrics.get["f1"] > 90

if __name__ == '__main__':
    train()
