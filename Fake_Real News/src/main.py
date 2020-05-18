from text_cleaning import TextCleaning
from word_dictionary import WordDictionary
import numpy as np
import pandas as pd 


#Set data file path
data_dirpath = '../Data/'
data_filepath = 'Data/Fake.csv'
true_data_filepath = 'Data/True.csv'

#Initialize TextDataCleaning class
o = TextCleaning()

# Initialize WordDictionary class
p = WordDictionary()

#Read data from csv file to panda dataframe
fake_df = o.read_csv_in_dataframe(data_filepath)
fake_df['class'] = 0
true_df = o.read_csv_in_dataframe(true_data_filepath)
true_df['class'] = 1
frames = [fake_df, true_df]
df = pd.concat(frames)

#Print dataframe head
print(df.head())

df["text"].dropna(inplace = True)

#Run text cleaning pipeline
normalize_corpus = np.vectorize(o.core_cleaning)
text_series = normalize_corpus(df['text'])

#Print first 5 rows
#print('\n', text_series[:5])

#Run Bag of Words pipeline
words_df = o.count_vectorizer(text_series)
print(words_df.head())

# Run word dictionary 
word_dic = p.create_dictionary(text_series)

words_df['class'] = df['class']


words_df.to_csv(r'Data/output/output.csv', index = False, header = True)