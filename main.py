from W2V import *
from CorpusStreamer import *

configure_logging()

xmlfolderpath = ""

#tokenpath = ""
stream = CorpusStreamer(xmlfolderpath)
w2v = Word2Vec.create_and_train(stream, epochs=1)
w2v.evaluate_analogies()