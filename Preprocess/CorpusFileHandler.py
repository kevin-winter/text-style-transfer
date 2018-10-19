import os
import pickle as pkl
from helper import *

class CorpusFileHandler():
    def __init__(self, path):
        self.path = path
        
        self.__depickle = path.endswith(".pkl")
        self.__isXML = path.endswith(".xml")
        self.__isTXT = path.endswith(".txt")

    def as_token_stream(self):
        yield from self.read()

    def read(self):
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
                  
            if type(content) != list:
                content = tokenize(content)
            
            return content

        except Exception as e:
            logging.error("Could not parse '{}' ({})".format(self.path, e))
            return []