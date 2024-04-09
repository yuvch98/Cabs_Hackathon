import pandas as pd
import numpy as np
import Constants
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
def create_submission(model):
    features = ['day_of_week', 'month_bin', 'pickup_bin', 'holiday']

    # bin the hours and assign labels to each bin

    new_inputs = pd.read_csv(r'TestDataForStudents.csv')
    bins_days = [0, 6, 12, 18, 23]
    labels_days = [0, 1, 2, 3]
    bins_months = [1, 4, 8, 12]
    labels_months = [1, 0, 2]
    checking_df = pd.DataFrame()
    print(new_inputs.head(5))
    date = []
    time = []
    for row in new_inputs.values.tolist():
        date.append(row[0])
        time.append(row[1])

    print(time)
    print(date)
    checking_df['tpep_pickup_datetime'] = pd.to_datetime(new_inputs['Dates'])
    checking_df['day_of_week'] = checking_df['tpep_pickup_datetime'].dt.dayofweek
    checking_df['pickup_bin'] = pd.cut(time, bins=bins_days, labels=labels_days, include_lowest=True)
    checking_df['month_bin'] = pd.cut(date, bins=bins_months, labels=labels_months, include_lowest=True)
    checking_df['holiday'] = [1 if date.strftime('%m-%d') in Constants.holidays else 0 for date in pd.to_datetime(date)]
    checking_df.set_index(checking_df['tpep_pickup_datetime'], inplace=True)

    predictions = model.predict(checking_df[features])
    print(predictions)

    # uploading submissions
    new_inputs['predictions'] = predictions
    new_inputs.to_csv('SubmissionTestDataForStudents.csv')