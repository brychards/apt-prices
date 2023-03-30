
import re

import pandas as pd

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import TruncatedSVD




WORDS_TO_DROP = set(['a', 'on', 'our', 'all', 'at', 'and', '$250', 'with', 'the', 'is', 'or', 'this', 'that', 'of', 'to', 'in', 'for', 'you', 'as', '-'])

# Removes punctuation, numbers, and the words in WORDS_TO_DROP
class BlurbFeatures:
  def __init__(self, num_components = 5):
    self.count_vectorizer = None
    self.tfidf_transformer = None
    self.training_tfidf_matrix = None 
    self.svd_transformer = None
    self.training_svd_df = None
    self.num_components = num_components

  @staticmethod
  def _clean_text(text):
    punc_removed = re.sub('[\.\,\:\$\!]', '', text)
    words = punc_removed.split()
    words_to_keep = [w.lower() for w in words if not (w in WORDS_TO_DROP or w.isdigit())]
    cleaned = " ".join(words_to_keep)
    return cleaned

  ''' Computesnum_components and returns the TD-IDF matrix, where each row is a document, and each column is a word in the corpus.
  inputs:
    blurbs: a pandas series
    min_df: the min document frequency of words to keep
    max_df: the max document frequency of words to keep
  '''
  def compute_training_tfidf_matrix(self, training_blurbs, min_df=0.04, max_df=0.5, n_gram_range=(2,3)):
    self.count_vectorizer = CountVectorizer(ngram_range=n_gram_range, min_df=min_df, max_df=max_df)
    cleaned_blurbs = training_blurbs.map(BlurbFeatures._clean_text)
    Counts = self.count_vectorizer.fit_transform(cleaned_blurbs)
    self.tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    self.training_tfidf_matrix = self.tfidf_transformer.fit_transform(Counts)
  
  def get_training_tfidf_matrix(self):
    if self.training_tfidf_matrix is None:
      print("Must first call fit_dfidf_transformer()!")
      return None
    return self.training_tfidf_matrix

  def compute_testing_tfidf_matrix(self, testing_blurbs):
    if self.tfidf_transformer is None:
      print("Must first call fit_tfidf_transformer()!")
      return None
    cleaned_blurbs = testing_blurbs.map(BlurbFeatures._clean_text)
    Counts = self.count_vectorizer.transform(cleaned_blurbs)
    Tfidfs = self.tfidf_transformer.transform(Counts)
    return Tfidfs


  '''
  Computes the SVD DataFrame from the TDIDF matrix.
  Used to reduce the number of dimensions of our feature space.
  Note that we save it as a DataFrame instead of a numpy 2-d array, because this is more likely to be used as features than the high-dimension TD-IDF matrix.
  '''
  def compute_training_svd_df(self, training_tfidf_matrix = None):
    if training_tfidf_matrix is None:
      if self.training_tfidf_matrix is None:
        print("Must either supply a training_tfidf_matrix, or call compute_training_tfidf_matrix() before calling compute_training_svd_df()")
        return
      training_tfidf_matrix = self.training_tfidf_matrix
    self.svd_transformer = TruncatedSVD(n_components=self.num_components)
    training_svd_matrix = self.svd_transformer.fit_transform(training_tfidf_matrix)
    print('training svd matrix shape ', training_svd_matrix.shape)
    col_names = ["Blurb_svd_" + str(i) for i in range(1, self.num_components + 1)]
    self.traning_svd_df = pd.DataFrame(training_svd_matrix, columns=col_names)
    print('about to return the training_svd_df, with shape', self.traning_svd_df.shape)
    return self.traning_svd_df


  def compute_testing_svd_df(self, testing_dfidf_matrix):
    if self.svd_transformer is None:
      print("Must first fit the TruncatedSVD on training data by calling compute_training_svd_df()")
      return
    testing_svd_matrix = self.svd_transformer.transform(testing_dfidf_matrix)
    col_names = ["Blurb_svd_" + str(i) for i in range(1, self.num_components + 1)]
    return pd.DataFrame(testing_svd_matrix, columns=col_names)
  
  def compute_testing_svd_df_from_blurbs(self, testing_blurbs):
    testing_tdidf_matrix = self.compute_testing_tfidf_matrix(testing_blurbs)
    return self.compute_testing_svd_df(testing_tdidf_matrix)


