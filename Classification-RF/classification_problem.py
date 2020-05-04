# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

train_df = pd.read_csv('Data/train.csv')
test_df = pd.read_csv('Data/test.csv')

print('Data columns:', train_df.columns.values)



