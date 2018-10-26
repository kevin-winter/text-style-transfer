from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict
from pandas import DataFrame

from helper import *

## SELECTORS

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns, byname=False, inverse=False):
        self.byname = byname
        self.columns = columns
        self.inverse = inverse
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        cols = self.columns if self.byname else X.columns[self.columns]
        mask = X.columns.isin(cols)
        return X.loc[:, ~mask] if self.inverse else X.loc[:, mask]
    

class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtypes):
        self.dtypes = dtypes
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.select_dtypes(include=self.dtypes)    
    
## TRANSFORMERS


class NamedEntityMasker(BaseEstimator, TransformerMixin):
    def __init__(self, filtered_types=["PERSON","ORG", "NORP","GPE"], maskwith=""):
        self.filtered_types = filtered_types
        self.maskwith = maskwith
        
    def fit(self, X, y=None):
        self.entities = defaultdict(set)
        return self
    
    def transform(self, X):
        doc = nlp()(X, disable=["tagger","parser"])

        for ent in doc.ents:
            if ent.label_ in self.filtered_types:
                self.entities[ent.label_].add(ent.text)
                
        for ent_type, ents in self.entities.items():
            for i, ent in enumerate(ents):
                replacement = self.maskwith if self.maskwith else "{}_{}".format(ent_type, i)
                X = re.sub(" {} ".format(ent), " {} ".format(replacement), X)

        return X

    
class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        doc = nlp()(X.lower(), disable=["ner"])
        feature_map = lambda x: (x.text, x.i - x.sent.start, x.sent.end - x.i, x.pos_, x.tag_)
        return DataFrame(list(map(feature_map, doc)))


class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = re.sub("[^A-Za-z,.?! ]","", X)
        X = re.sub("(\r?\n)+"," ", X)
        X = re.sub(" +"," ", X)
        return X

    
class TextVectorizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return w2v_word_mapper(X.iloc[:,0])


class SpacyAnalyser(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return nlp()(X)
    
