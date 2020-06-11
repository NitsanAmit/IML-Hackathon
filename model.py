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
        self.clean_up_data(self.x_train, self.y_train)

    def predict(self, x):
        """
        Recieves a pandas DataFrame of shape (m, 15) with m flight features, and predicts their
        delay at arrival and the main factor for the delay.
        @param x: A pandas DataFrame with shape (m, 15)
        @return: A pandas DataFrame with shape (m, 2) with your prediction
        """
        raise NotImplementedError

    def clean_up_data(self, X, y):
        jointDf = X.join(y)
        jointDf = get_dummies(jointDf)
        add_arrival_departure_bins(jointDf)
        make_flight_date_canonical(jointDf)
        factorize_delay(jointDf)
        add_is_same_state(jointDf)
        print(jointDf.head())
        # TODO keep goin'

    def add_holidays_column(self):
        pass

    def cross_holidays(self):
        self.['Date'] = pd.to_datetime()
        dr = pd.date_range(start='2010-01-01', end='2019-12-31')
        # TODO add
        df = pd.DataFrame()
        df['Date'] = dr
        cal = calendar()
        holidays = cal.holidays(start=dr.min(), end=dr.max())
        df['Holiday'] = df['Date'].isin(holidays)
        print(df)

    def visualize(self, df):
        # Airport vs. delay
        df["is_delayed"] = df['DelayFactor'].astype(int) + 1
        print(df["is_delayed"])

        p = (ggplot(df, aes(x='Dest', y='is_delayed')) + geom_line(size=2, group=1) +
         geom_point(size=6, shape='.', fill="white"))  #dest

        print(p)

        # ggplot(df, aes(x='BP')) + geom_histogram(binwidth=5) #origin


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
    pass


def add_is_same_state(jointDf):
    jointDf["SameState"] = np.rint(jointDf["OriginState"] == jointDf["DestState"])


def factorize_delay(jointDf):
    # fix DelayFactor to numeric
    delay_factor = jointDf['DelayFactor'].factorize()
    jointDf['DelayFactor'] = delay_factor[0]
    print(delay_factor[1])

def add_holidays_column(self):
    pass

def cross_holidays(df):
    df['Date'] = pd.to_datetime()
    dr = pd.date_range(start='2010-01-01', end='2019-12-31')
    # TODO add
    df = pd.DataFrame()
    df['Date'] = dr
    cal = calendar()
    holidays = cal.holidays(start=dr.min(), end=dr.max())
    df['Holiday'] = df['Date'].isin(holidays)
    print(df)