from gensim.models import Word2Vec, KeyedVectors
from forbiddenfruit import curse
import logging
import numpy as np

from tst.preprocess.helper import tokenize


def create_and_train(self, corpus, vsize=100, window=5, epochs=5):
    logging.info("W2V: Create corpus")
    w2v = Word2Vec(size=vsize, window=window, min_count=1, workers=4)
    logging.info("W2V: Train model")
    w2v.train(corpus, total_examples=w2v.corpus_count, epochs=epochs)
    return w2v


def evaluate_analogies(self,
                       analogies_filepath="https://raw.githubusercontent.com/"
                                          "RaRe-Technologies/gensim/develop/gensim/test/test_data/questions-words.txt"):
    testresults = self.wv.evaluate_word_analogies(analogies_filepath, case_insensitive=True, restrict_vocab=1000000)
    testresult_scores = {d["section"]: len(d["correct"]) / (len(d["correct"]) + len(d["incorrect"])) for d in
                         testresults[1]}
    logging.info("Analogy Test Results:")
    for measure, score in testresult_scores.items():
        logging.info("{}: {:.3f}".format(measure, score))
    return testresult_scores


def save_word2vec_format_reduced(self, fname, words=None, topn=None):
    if words is None:
        vocab = {k: v.count for k, v in self.vocab.items()}
        words = sorted(vocab, key=vocab.get, reverse=True)
    else:
        if isinstance(words, str):
            with open(words, 'r') as f:
                words = [line.split()[0] for line in f]
        words = [w for w in words if w in self]
    
    words = words[:topn or len(words)]
    with open(fname, 'wt') as out:
        out.write(f'{len(words)} {self.vector_size}\n')
        for word in words:
            out.write('{} {}\n'.format(word, ' '.join(map(str, self[word]))))

            
def vector_or_zeros(self, word):
    if word in self:
        return self[word]
    else:
        return np.zeros(self.vector_size)


def vectorize(self, text=None, tokens=None, ignore_missing=True):
    if text:
        tokens = tokenize(text)
        
    return np.array([[self.vector_or_zeros(w) for w in s if w in self or not ignore_missing] for s in tokens])

            
curse(Word2Vec, "create_and_train", classmethod(create_and_train))
curse(Word2Vec, "evaluate_analogies", evaluate_analogies)

curse(KeyedVectors, "save_word2vec_format_reduced", save_word2vec_format_reduced)
curse(KeyedVectors, "vector_or_zeros", vector_or_zeros)
curse(KeyedVectors, "vectorize", vectorize)