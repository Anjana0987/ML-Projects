from data_wrangling import DataWrangling

# Set data file path
data_dirpath = '../Data/'
train_data_filepath = 'Data/train.csv'
test_data_filepath = 'Data/test.csv'

# Initialise DataWrangling
dw = DataWrangling()

#Read data from csv file to panda dataframe
train_df = dw.read_csv_in_dataframe(train_data_filepath)
test_df = dw.read_csv_in_dataframe(test_data_filepath)

#Data Wrangling
train_data_study = dw.data_wrangle(train_df)


