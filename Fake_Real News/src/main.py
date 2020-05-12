from text_cleaning import TextCleaning
from word_dictionary import WordDictionary
import numpy as np


#Set data file path
data_dirpath = '../Data/'
data_filepath = 'Data/Fake.csv'

#Initialize TextDataCleaning class
o = TextCleaning()

# Initialize WordDictionary class
p = WordDictionary()

#Read data from csv file to panda dataframe
df = o.read_csv_in_dataframe(data_filepath)

#Print dataframe head
print(df.head())

df["text"].dropna(inplace = True)

#Run text cleaning pipeline
normalize_corpus = np.vectorize(o.core_cleaning)
text_series = normalize_corpus(df['text'])

#Print first 5 rows
print('\n', text_series[:5])

#Run Bag of Words pipeline
words_df = o.count_vectorizer(text_series)
print(words_df.head())

# Run word dictionary 
word_dic = p.create_dictionary(text_series)
print(word_dic[1:10])

words_df['text'] = text_series.tolist()


#words_df.to_csv(r'data/output.csv', index = False, header = True)