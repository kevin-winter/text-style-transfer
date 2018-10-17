from gensim.models import Word2Vec
from preprocess import tokenize

def trainW2V(corpus, vsize=100, window=5, epochs=10):
    print("W2V: Create corpus")
    w2v = Word2Vec(corpus, size=vsize, window=window, min_count=1, workers=4)
    print("W2V: Train model")
    w2v.train(corpus, total_examples=w2v.corpus_count, epochs=epochs)
    return w2v

def tokenizeAndTrainW2V(corpus, vsize=100, window=5, epochs=10):
    tokens = tokenize(corpus)
    return trainW2V(corpus, vsize, window, epochs)

def evaluate_w2v(w2v, analogies_filepath="https://raw.githubusercontent.com/RaRe-Technologies/gensim/develop/gensim/test/test_data/questions-words.txt"):
    testresults = w2v.wv.evaluate_word_analogies(analogies_filepath, case_insensitive=True, restrict_vocab=1000000000)
    testresult_scores = {d["section"]: len(d["correct"]) / (len(d["correct"]) + len(d["incorrect"])) for d in testresults[1]}
    for measure, score in testresult_scores.items():
        print("{}: {:.3f}".format(measure, score))
    return testresult_scores