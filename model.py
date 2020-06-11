"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2020

Author(s): Nitsan, Shahar, Gal, Noa

===================================================
"""
import random
import pandas as pd
import numpy as np
from plotnine import *
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar


CSV_PATH = "data/train_data.csv"


class FlightPredictor:
    def __init__(self, path_to_weather=''):
        """
        Initialize an object from this class.
        @param path_to_weather: The path to a csv file containing weather data.
        """
        random.seed(5)
        raw_data = pd.read_csv(CSV_PATH)
        X = raw_data.drop(["ArrDelay", "DelayFactor"], axis=1)
        y = raw_data[["ArrDelay", "DelayFactor"]]
        x_temp, self.x_test, y_temp, self.y_test = train_test_split(X, y, test_size=0.2)
        self.x_train, self.x_validate, self.y_train, self.y_validate = train_test_split(x_temp, y_temp, test_size=0.2)

        # Only use x_train and y_train!!!!!!!!!!!!!!!
        self.clean_up_data(path_to_weather)

    def predict(self, x):
        """
        Recieves a pandas DataFrame of shape (m, 15) with m flight features, and predicts their
        delay at arrival and the main factor for the delay.
        @param x: A pandas DataFrame with shape (m, 15)
        @return: A pandas DataFrame with shape (m, 2) with your prediction
        """
        raise NotImplementedError

    def clean_up_data(self, path_to_weather):
        jointDf = self.x_train.join(self.y_train)
        jointDf = get_dummies(jointDf)
        add_arrival_departure_bins(jointDf)
        make_flight_date_canonical(jointDf)
        factorize_delay(jointDf)
        add_is_same_state(jointDf)
        if path_to_weather:
            add_weather_data(jointDf, path_to_weather)
        cross_holidays(jointDf)
        print(jointDf.head())
        # TODO keep goin'


    def visualize(self, df):
        # Airport vs. delay
        df["is_delayed"] = (df['DelayFactor'] != -1).astype(int)
        print(df["is_delayed"])

        df_dest_delayed = df.groupby('Dest').agg({'is_delayed': ['mean']})
        df_origin_delayed = df.groupby('Origin').agg({'is_delayed': ['mean']})

        print(df_dest_delayed)


def add_arrival_departure_bins(jointDf):
    two_hour_bins = np.linspace(0, 2400, num=13)
    two_hour_labels = np.rint(np.linspace(0, 22, 12))
    jointDf["DepBin"] = pd.cut(jointDf['CRSDepTime'], bins=two_hour_bins,
                               labels=two_hour_labels)
    jointDf["ArrBin"] = pd.cut(jointDf['CRSArrTime'], bins=two_hour_bins,
                               labels=two_hour_labels)
    jointDf.drop(['CRSDepTime', 'CRSArrTime'], axis=1)


def get_dummies(jointDf):
    return pd.get_dummies(jointDf, columns=['DayOfWeek', 'Reporting_Airline'],
                          prefix=['weekday', 'airline'])


def make_flight_date_canonical(jointDf):
    jointDf["CanonicalFlightDate"] = pd.DataFrame(pd.to_datetime(jointDf["FlightDate"], errors="coerce")).values.astype(
        float) / 10 ** 10
    jointDf["CanonicalFlightDate"] = jointDf["CanonicalFlightDate"] - np.min(jointDf["CanonicalFlightDate"])



def add_is_same_state(jointDf):
    jointDf["SameState"] = jointDf["OriginState"] == jointDf["DestState"]


def add_weather_data(jointDf, path_to_weather):
    weather_data = pd.read_csv(path_to_weather)

    #TODO match jointDf["FlightDate"] to weather_data["date"]
    #TODO match jointDf["Origin" or "Dest"] to weather_data["station"]
    #TODO create columns in jointDf based on weather_data columns:

    # snow_in -> snowed (boolean) (could be None / -99 / 0 if data is missing)
    # precip_in -> precipitation (float) (check possible nan types)
    # avg_wind_speed_kts -> avg_wing_spd (float)
    # avg_wind_drct -> avg_wind_dir (float)
    # min_temp_f -> min_temp (float)



def factorize_delay(jointDf):
    # fix DelayFactor to numeric
    delay_factor = jointDf['DelayFactor'].factorize()
    jointDf['DelayFactor'] = delay_factor[0]
    print(delay_factor[1])


def cross_holidays(jointDf):
    jointDf['FlightDate'] = pd.to_datetime(jointDf['FlightDate'])
    cal = calendar()
    holidays = cal.holidays(start=jointDf['FlightDate'].min(), end=jointDf[
        'FlightDate'].max())
    jointDf["is_holiday"] = jointDf["FlightDate"].isin(holidays)
    return
