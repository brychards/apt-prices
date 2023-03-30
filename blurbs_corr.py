
import re

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import TruncatedSVD




WORDS_TO_DROP = set(['a', 'on', 'our', 'all', 'at', 'and', '$250', 'with', 'the', 'is', 'or', 'this', 'that', 'of', 'to', 'in', 'for', 'you', 'as', '-'])

# Removes punctuation, numbers, and the words in WORDS_TO_DROP
class BlurbFeatures:
  def __init__(self, num_components = 5):
    self.training_vocabulary = set()
    self.count_vectorizer = None

  @staticmethod
  def _clean_text(text):
    punc_removed = re.sub('[\.\,\:\$\!]', '', text)
    words = punc_removed.split()
    words_to_keep = [w.lower() for w in words if not (w in WORDS_TO_DROP or w.isdigit())]
    cleaned = " ".join(words_to_keep)
    return cleaned

def compute_training_vocabulary(self, blurbs, min_doc_freq=.02, max_doc_freq=0.4, ngram_range=(2,3)):
    blurbs_df = pd.DataFrame(blurbs, columns=['blurb'])
    blurbs_df['cleaned_blurbs'] = blurbs_df.blurbs.map(BlurbFeatures._clean_text)
    self.count_vectorizer = CountVectorizer(
        ngram_range=ngram_range, min_df=min_doc_freq, max_df=max_doc_freq)
    counts = self.count_vectorizer.fit_transform(blurbs_df.cleaned_blurbs)


