from data_wrangling import DataWrangling
from data_clean import CleaningData

# Set data file path
data_dirpath = '../Data/'
train_data_filepath = 'Data/train.csv'
test_data_filepath = 'Data/test.csv'

# Initialise DataWrangling
dw = DataWrangling()

# Initialize CleaningData
cd = CleaningData()

# Read data from csv file to panda dataframe
train_df = dw.read_csv_in_dataframe(train_data_filepath)
test_df = dw.read_csv_in_dataframe(test_data_filepath)

# Replace Missing values
train_df = cd.cleaning_columns(train_df)
test_df = cd.cleaning_columns(test_df)

#Data Wrangling
train_data_study = dw.data_wrangle(train_df)


