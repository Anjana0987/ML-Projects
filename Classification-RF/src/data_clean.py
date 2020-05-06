import pandas as pd 
import numpy as np

class CleaningData:

    def cleaning_columns(self, df):
        #Total number of Null values in Age column = 177/891
        df['Age'] = df['Age'].fillna(df['Age'].mean()) #check mean or median
        # Extracting Title from Names
        df['Title'] = df.Name.str.extract(r',\s*([^\.]*)\s*\.', expand=False)
        # Converting age to age groups
        bins = [0, 5, 15, 25, 35, 45, 55, 65, 80, np.inf]
        #names = ['<2', '5-15', '15-25', '25-35', '35-45', '45-55', '55-65', '65-80', '80+']
        names = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        df['AgeRange'] = pd.cut(df['Age'], bins, labels=names)
        print(df['Age'], df['AgeRange'])
        return df

    def drop_columns(self, df):

        return df