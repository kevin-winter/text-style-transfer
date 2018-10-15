import os
import pickle as pkl

class CorpusStreamer:
    def __init__(self, path):
        self.path = path
    
    def __iter__(self):
        for fname in os.listdir(self.path):
            print("W2V: Loading: " + fname)
            with open(self.path + fname, "rb") as file:
                sents = pkl.load(file)
                for sent in sents:
                    yield sent
