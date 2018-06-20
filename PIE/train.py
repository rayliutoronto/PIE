from PIE.model import Model
from PIE.config import Config
from PIE.data_set import DataSet


def main():
    config = Config(build=True, load=True)

    model = Model(config)
    model.build()

    model.train(DataSet(config.filename_train, config.preprocessor), DataSet(config.filename_dev, config.preprocessor))


if __name__ == '__main__':
    main()
