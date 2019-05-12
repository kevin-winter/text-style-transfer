from gensim.models import Word2Vec
from forbiddenfruit import curse
import logging


def create_and_train(self, corpus, vsize=100, window=5, epochs=5):
    logging.info("W2V: Create corpus")
    w2v = Word2Vec(corpus, size=vsize, window=window, min_count=1, workers=4)
    logging.info("W2V: Train model")
    w2v.train(corpus, total_examples=w2v.corpus_count, epochs=epochs)
    return w2v


def evaluate_analogies(self,
                       analogies_filepath="https://raw.githubusercontent.com/"
                                          "RaRe-Technologies/gensim/develop/gensim/test/test_data/questions-words.txt"):
    testresults = self.wv.evaluate_word_analogies(analogies_filepath, case_insensitive=True, restrict_vocab=1000000000)
    testresult_scores = {d["section"]: len(d["correct"]) / (len(d["correct"]) + len(d["incorrect"])) for d in
                         testresults[1]}
    logging.info("Analogy Test Results:")
    for measure, score in testresult_scores.items():
        logging.info("{}: {:.3f}".format(measure, score))
    return testresult_scores


curse(Word2Vec, "create_and_train", classmethod(create_and_train))
curse(Word2Vec, "evaluate_analogies", evaluate_analogies)
