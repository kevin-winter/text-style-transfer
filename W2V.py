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