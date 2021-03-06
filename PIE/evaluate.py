from PIE.data_set import DataSet
from PIE.model import Model
from PIE.config import Config
from PIE.tokenizer import Tokenizer

def align_data(data):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned


def interactive_shell(model):
    """Creates interactive shell to play with PIE

    Args:
        model: instance of NERModel

    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a sentence like
input> 416-234-0973""")

    tokenizer = Tokenizer('en')

    while True:
        sentence = input("input> ")

        words_raw = [token.text for doc in tokenizer.split(sentence.strip()) for token in doc]
        # words_raw = sentence.strip().split(" ")

        if words_raw == ["exit"]:
            break

        preds = model.predict(words_raw)
        to_print = align_data({"input": words_raw, "output": preds})

        for key, seq in to_print.items():
            model.logger.info(seq)


def main():
    # create instance of config
    config = Config(build=False, load=True)

    # build PIE
    model = Model(config)
    model.build()
    model.restore_session(config.dir_model)

    # create dataset
    #test = DataSet(config.filename_test, config.preprocessor)
    # evaluate and interact
    #model.evaluate(test)

    interactive_shell(model)


if __name__ == "__main__":
    main()
