from gensim import corpora
from nltk.tokenize import word_tokenize

class WordDictionary:

    def __init__(self):
        pass

    # Create Gensim dictionary from series of text
    def create_dictionary(self, text_series):
        text_series = text_series.tolist()
        for i in range(len(text_series)):
            text_series[i] = text_series[i].split(' ')
        text_list = list(text_series)
        text_doc2w = []
        dictionary = corpora.Dictionary(text_list)
        #dictionary.save_as_text('data/output/consolidated_data/corpus_dict.txt')
        for i in text_series:
            text_doc2w.append(dictionary.doc2bow(i))
        return text_doc2w