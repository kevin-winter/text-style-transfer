from CorpusFileHandler import *

class CorpusStreamer:
    def __init__(self, path):
        self.path = path
  
    def __iter__(self):
        files = os.listdir(self.path)
        nr_files = len(files)

        for i, fname in enumerate(files):
            print("CorpusStreamer: Loading: " + fname)
            yield from CorpusFileHandler(self.path + fname).as_token_stream()