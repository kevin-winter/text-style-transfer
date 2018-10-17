from xml_to_tokens import xml_to_tokens
from CorpusStreamer import CorpusStreamer
from W2V import trainW2V

xmlfolderpath = ""
xml_to_tokens(xmlfolderpath)

tokenpath = ""
stream = CorpusStreamer(tokenpath)
w2v = trainW2V(stream, epochs=1)
evaluate_w2v(w2v2)