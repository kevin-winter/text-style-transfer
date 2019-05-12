import os
import pickle as pkl
from helper import *

class CorpusFileHandler:
    def __init__(self, path):
        self.path = path
        
        self.__depickle = path.endswith(".pkl")
        self.__isXML = path.endswith(".xml")
        self.__isTXT = path.endswith(".txt")

    def as_token_stream(self):
        yield from self.read(True)
        
    def as_string_stream(self):
        yield from self.read(False)
    
    def as_stream(self, tokenize):
        yield from self.read(tokenize)

    def read(self, as_tokens=True):
        try:
            with open(self.path, 
                mode = "rb" if self.__depickle else "r", 
                encoding = None if self.__depickle else "utf8") as file:

                if self.__depickle:
                    content = pkl.load(file)

                elif self.__isXML:
                    content = get_text(file)
                
                elif self.__isTXT:
                    content = crop_body(file.read())

                else:
                    raise AttributeError("File type not supported yes")
            
            if as_tokens:
                content = tokenize(content)
            
            return content

        except Exception as e:
            logging.error("Could not parse '{}' ({})".format(self.path, e))
            return []
        
        
class CorpusStreamer:
    def __init__(self, path, tokenize=True):
        self._path = path
        self._store = False
        self._buffer = []
        self._save_percent = 5
        self._tokenize = tokenize
  
    def __iter__(self):
        files = os.listdir(self._path)
        nr_files = len(files)
        logging.info("CorpusStreamer: Loading {} files.".format(nr_files))
        
        for i, fname in enumerate(files):
            if os.path.isdir(os.path.join(self._path, fname)):
                continue
                
            logging.debug("CorpusStreamer: Loading: " + fname)
            yield from CorpusFileHandler(os.path.join(self._path, fname)).as_stream(self._tokenize)

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
        
    def toString(self):
        tmp = self._tokenize
        buffer = ""
        
        self._tokenize = False
        
        for doc in self:
            buffer += doc
        
        self._tokenize = tmp
        return buffer