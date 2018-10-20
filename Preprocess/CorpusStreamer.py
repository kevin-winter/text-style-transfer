from CorpusFileHandler import *
import os

class CorpusStreamer:
    def __init__(self, path):
        self._path = path
        self._store = False
        self._buffer = []
        self._save_percent = 5
  
    def __iter__(self):
        files = os.listdir(self._path)
        nr_files = len(files)
        logging.info("CorpusStreamer: Loading {} files.".format(nr_files))
        
        for i, fname in enumerate(files):
            logging.debug("CorpusStreamer: Loading: " + fname)
            yield from CorpusFileHandler(os.path.join(self._path, fname)).as_token_stream()

            percentile_action(i+1, nr_files, 5, report, None)
            
            if self._store:
                if percentile_action(i+1, nr_files, self._save_percent, save_sents, *(self._buffer, self._save_path)):
                    self._buffer = []
            
    def store(self, path, portion_in_percent):
        self._save_path = path
        
        if int(portion_in_percent) > 0 and int(portion_in_percent) <= 100:
            self._save_percent = portion_in_percent
        
        self._store = True
        for sent in self:
            self._buffer.append(sent)
        
        self._store = False
            