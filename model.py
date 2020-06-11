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
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt



class FlightPredictor:
    def __init__(self, path_to_weather=''):
        """
        Initialize an object from this class.
        @param path_to_weather: The path to a csv file containing weather data.
        """
        random.seed(5)
        raw_data = pd.read_csv('data/train_data.csv')
        X = raw_data.drop(["ArrDelay", "DelayFactor"], axis=1)
        y = raw_data[["ArrDelay", "DelayFactor"]]
        x_temp, x_test, y_temp, y_test = train_test_split(X, y, test_size=0.2)
        x_train, x_validate, y_train, y_validate = train_test_split(x_temp, y_temp, test_size=0.2)

        # Only use x_train and y_train!!!!!!!!!!!!!!!
        self.clean_up_data(x_train, y_train)

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
        print(jointDf.head())

    def visualize(self, X, y):
        pass


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
