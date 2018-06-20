class DataSet(object):
    def __init__(self, filename, preprocessor=None, max_iter=None):
        self.filename = filename
        self.preprocessor = preprocessor
        self.max_iter = max_iter
        self.length = None

    def __iter__(self):
        niter = 0
        with open(self.filename) as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if (len(line) == 0 or line.startswith("-DOCSTART-")):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags
                        words, tags = [], []
                else:
                    ls = line.split(' ')
                    word, tag = ls[0], ls[-1]
                    if self.preprocessor is not None:
                        word = self.preprocessor.word(word)
                        tag = self.preprocessor.tag(tag)
                    words += [word]
                    tags += [tag]

    def __len__(self):
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length
