from data_wrangling import DataWrangling
from data_clean import CleaningData
from model_training import RandomForestModel

# Set data file path
data_dirpath = '../Data/'
train_data_filepath = 'Data/train.csv'
test_data_filepath = 'Data/test.csv'
ytest_data_filepath = 'Data/gender_submission.csv'

# Initialise DataWrangling
dw = DataWrangling()

# Initialize CleaningData
cd = CleaningData()

# Initialize Random Forest Model
rf = RandomForestModel()

# Read data from csv file to panda dataframe
train_df = dw.read_csv_in_dataframe(train_data_filepath)
test_df = dw.read_csv_in_dataframe(test_data_filepath)
yTest = dw.read_csv_in_dataframe(ytest_data_filepath)

#Data Wrangling
train_data_study = dw.data_wrangle(train_df)

# Replace Missing values and Cleaning columns
train_df = cd.cleaning_columns(train_df)
test_df = cd.cleaning_columns(test_df)

# Drop columns
train_df = cd.drop_columns(train_df)
test_df = cd.drop_columns(test_df)
train_df.to_csv('Train_d.csv')
test_df.to_csv('test.csv')
print(train_df.head())

# Train the models
metrics_values = rf.training(train_df, test_df, yTest)
evaluation = rf.evaluation_metrics(metrics_values[0], metrics_values[1], metrics_values[2])




