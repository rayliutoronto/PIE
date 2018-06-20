import spacy


class Tokenizer(object):
    class __Tokenizer:
        def __init__(self, language='en'):
            self.__nlp = spacy.load(language)
            self.__nlp.tokenizer = self.__create_custom_tokenizer()

        def split(self, text):
            if type(text) == str:
                text = [text]

            return [self.__nlp(x) for x in text]

        def __create_custom_tokenizer(self):
            my_prefix = []

            all_prefixes_re = spacy.util.compile_prefix_regex(tuple(list(self.__nlp.Defaults.prefixes) + my_prefix))

            custom_infixes = ['@', '</', '<', '>']
            infix_re = spacy.util.compile_infix_regex(tuple(list(self.__nlp.Defaults.infixes) + custom_infixes))

            suffix_re = spacy.util.compile_suffix_regex(self.__nlp.Defaults.suffixes)

            return spacy.tokenizer.Tokenizer(self.__nlp.vocab, self.__nlp.Defaults.tokenizer_exceptions,
                                             prefix_search=all_prefixes_re.search,
                                             infix_finditer=infix_re.finditer, suffix_search=suffix_re.search)

    __instance = {}

    def __new__(cls, language='en'):
        if Tokenizer.__instance.get(language) is None:
            Tokenizer.__instance.update({language: Tokenizer.__Tokenizer(language)})

        return Tokenizer.__instance.get(language)


if __name__ == '__main__':
    t = Tokenizer()

    tokens = [token.text for doc in t.split(
        ['abc@toronto.ca', '<email>general.info@toronto.ca</email><email>general.info@toronto.ca</email>',
         'x@y.z, 532 234 098, 416-123-0001', '[{"email": \'don\'t@json.ai\'}, {"phone": "(437) 000-1234"}]']) for token
              in doc]
    for token in tokens:
        print(token, end='  ')
