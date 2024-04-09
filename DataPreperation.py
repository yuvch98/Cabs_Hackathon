import Constants
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def data_creation():
    use_cols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.read_csv(Constants.file_path, usecols=use_cols)
    df.to_csv('new_Cabs.csv', index=False)
    file_path = "new_Cabs.csv"
    df = pd.read_csv(file_path)
    #e copying the file to work on a safe file
    work_df = df.copy()
    #e print(f"shape = {work_df.shape}")
    #e print(f"Checking for nulls: {work_df.isnull().sum()}")
    # adjusting the data
    work_df['tpep_pickup_datetime'] = pd.to_datetime(work_df['tpep_pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
    #e extract date and time into separate columns
    work_df['tpep_pickup_date'] = work_df['tpep_pickup_datetime'].dt.date
    work_df['hour_pickup'] = work_df['tpep_pickup_datetime'].dt.hour
    work_df.drop(['tpep_dropoff_datetime'],axis=1,inplace=True)
    table = pd.crosstab(index=[work_df['tpep_pickup_date'], work_df['hour_pickup']], columns=['count'])
    #e in order to work better on less time per refresh
    table.to_csv(Constants.final_path, index=True)
    final_df = pd.read_csv(Constants.final_path)
    holidays = ['12-31', '03-17', '07-04', '10-31', '12-24']
    final_df.set_index(['tpep_pickup_date','hour_pickup'],inplace=True)
    #e NOTICE that the numbers represented are given from monday where monday is 0  to sunday where sunday is 6 in day_of_week
    final_df['day_of_week']= pd.to_datetime(final_df.index.get_level_values(0).astype(str))
    final_df['day_of_week'] = final_df['day_of_week'].dt.weekday
    final_df['month_bin'] = pd.to_datetime(final_df.index.get_level_values(0).astype(str))
    final_df['month_bin'] = final_df['month_bin'].dt.month
    #e extract the hour component from the datetime format
    # #define the bins and labels
    bins_days = [0, 6, 12, 18, 23]
    labels_days = [0, 1, 2, 3]
    bins_months = [1, 4, 8, 12]
    labels_months = [1, 0, 2]
    #e bin the hours and assign labels to each bin
    final_df['pickup_bin'] = pd.cut(final_df.index.get_level_values(1), bins=bins_days, labels=labels_days, include_lowest=True)
    final_df['month_bin'] = pd.cut(pd.to_datetime((final_df.index.get_level_values(0))).month, bins=bins_months, labels=labels_months, include_lowest=True)
    final_df.to_csv(Constants.model_file_path,index=True)
    return final_df, final_df['Count']

def get_outliers(df, series):
  q1 = series.quantile(0.25)
  q3 = series.quantile(0.75)

  if q1*q3 == 0:
    iqr = abs(2*(q1+q3))
    toprange = iqr
    botrange = -toprange
  else:
    iqr = q3-q1
    toprange = q3 + iqr * 1.5
    botrange = q1 - iqr * 1.5

  outliers_top=df[series > toprange]
  outliers_bot= df[series < botrange]
  outliers = pd.concat([outliers_bot, outliers_top], axis=0)

  return (botrange, toprange, outliers)

def check_outliers(df,series):
    bot_range, top_range, outliers = get_outliers(df, series)
    print(bot_range)
    print(top_range)
    print(outliers)
def getInputModel():
    final_df = pd.read_csv(Constants.model_file_path)

    final_df.set_index(['tpep_pickup_date', 'hour_pickup'], inplace=True)
    final_df['holiday'] = [1 if date.strftime('%m-%d') in Constants.holidays else 0 for date in
                           pd.to_datetime(final_df.index.get_level_values(0))]
    print(final_df.columns)
    X = final_df.drop('count', axis=1)
    y = final_df['count']
    return X, y