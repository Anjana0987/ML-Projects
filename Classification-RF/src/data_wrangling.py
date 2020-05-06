# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

class DataWrangling:

    def read_csv_in_dataframe(self, filepath):
        df = pd.read_csv(filepath)
        return df

    def data_wrangle(self, train_df):
        # Print the Data columns
        print(train_df.columns.values)
        print('----------------------------')

        # Check the survival rate for categorical data
        print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
        print('----------------------------')
        print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
        print('----------------------------')
        print(train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
        print('----------------------------')
        print(train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))
        print('----------------------------')

        # Check survival rate for non-categorical data
        g = sns.FacetGrid(train_df, col='Survived')
        g.map(plt.hist, 'Age', bins=20)

        # Combining Age and Pclass to see the correlation
        grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
        grid.map(plt.hist, 'Age', alpha=.5, bins=20)
        grid.add_legend()

        # Check correlation between embark point, sex, and Pclass
        grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
        grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
        grid.add_legend()

        # Check correlation between embark point, sex and fare
        grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
        grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
        grid.add_legend()
        return 0
