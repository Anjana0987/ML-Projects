import re
import pandas as pd
import numpy as np
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

class TextCleaning:

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.cv = CountVectorizer(min_df=0., max_df=1.)
        
    # Import data from csv into dataframe
    def read_csv_in_dataframe(self, filepath):
        df = pd.read_csv(filepath)
        return df

    # Core cleaning processes for user text
    def core_cleaning(self, text):
        # Remove new lines with regular expression
        text = re.sub(r'\n', ' ', text)

        # Remove digits with regular expression
        text = re.sub(r'\d', '', text)

        # Remove patterns matching url format
        url_pattern = r'((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?'
        text = re.sub(url_pattern, ' ', text)

        # Remove non-ascii characters
        text = ''.join(character for character in text if ord(character) < 128)

        # Remove punctuations
        text = ''.join(character for character in text if character not in set(string.punctuation))

        # Standardize white space
        text = re.sub(r'\s+', ' ', text)

        # Drop capitalization
        text = text.lower()

        # Remove white space
        text = text.strip()

        #word tokens
        text = word_tokenize(text)

        #remove stopwords
        text = [w for w in text if w not in stopwords.words('English')]

        #lemmatization
        text = [self.lemmatizer.lemmatize(i, 'v') for i in text]
        text = ' '.join(text)
        return text
    

    # Bag of words
    def count_vectorizer(self, text):
        cv_matrix = self.cv.fit_transform(text)
        cv_matrix = cv_matrix.toarray()

        # get all unique words in the corpus
        vocabulary = self.cv.get_feature_names()

        # show document feature vectors
        cv_df = pd.DataFrame(cv_matrix, columns=vocabulary)
        return cv_df