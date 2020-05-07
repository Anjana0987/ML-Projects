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
        # Range: ['<2', '5-15', '15-25', '25-35', '35-45', '45-55', '55-65', '65-80', '80+']
        names = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        df['AgeRange'] = pd.cut(df['Age'], bins, labels=names)
        bin = [0, 10, 30, 50, 70, 90, 110, 130, np.inf]
        name = ['1', '2', '3', '4', '5', '6', '7', '8']
        df['FareRange'] = pd.cut(df['Fare'], bin, labels=name)
        # Replacing Categorical data to numerical categories: Sex Column
        df.Sex.replace(['male', 'female'], [1, 0], inplace=True)
        # Replacing Categorial data to numerical categories: Embark Column
        df.Embarked.replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)
        # Replacing Categorical data to numerical categories: Titke Column
        df.Title.replace(['Mr','Mrs', 'Miss', 'Master', 'Don', 'Rev', 'Dr', 'Mme', 'Ms', 'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'the Countess', 'Jonkheer'], [1, 2, 3, 4,5,6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], inplace = True)
        return df

    def drop_columns(self, df):
        # Drop the PassengerID column
        del df['PassengerId']
        # Drop the Name column
        del df['Name']
        # Drop the Age column
        del df['Age']
        # Drop the ticked column
        del df['Ticket']
        # Drop the Fare column
        del df['Fare']
        # Drop the Cabin column
        del df['Cabin']
        return df