from CorpusFileHandler import *

class CorpusStreamer:
    def __init__(self, path):
        self.path = path
  
    def __iter__(self):
        files = os.listdir(self.path)
        nr_files = len(files)
        logging.info("CorpusStreamer: Loading {} files.".format(nr_files))
        
        for i, fname in enumerate(files):
            logging.debug("CorpusStreamer: Loading: " + fname)
            yield from CorpusFileHandler(self.path + fname).as_token_stream()

            percentile_action(i+1, nr_files, 5, report, None)