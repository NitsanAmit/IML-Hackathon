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
        joint_df = self.x_train.join(self.y_train)
        remove_outliers(joint_df)
        joint_df = get_dummies(joint_df)
        add_arrival_departure_bins(joint_df)
        make_flight_date_canonical(joint_df)
        factorize_delay(joint_df)
        add_is_same_state(joint_df)

        if path_to_weather:
            add_weather_data(joint_df, path_to_weather)
        cross_holidays(joint_df)
        add_weather_data(joint_df, path_to_weather)
        joint_df = drop_features(joint_df)
        # if path_to_weather:
        #     add_weather_data(joint_df, path_to_weather)
        # cross_holidays(joint_df)
        self.visualize(joint_df)
        print(joint_df.head())
        # TODO keep goin'

    def visualize(self, df):
        # Airport vs. delay
        df["is_delayed"] = (df['DelayFactor'] != -1).astype(int)
        print(df["is_delayed"])

        # Distance vs. delay time
        p = (ggplot(df, aes(x='Distance', y='ArrDelay', color='is_delayed')) + geom_point())
        print(p)

        # df_dest_delayed = df.groupby('Dest').agg({'is_delayed': ['mean']})


def remove_outliers(jointDf):
    entries_before = jointDf.shape[0]
    jointDf = jointDf[abs(jointDf["ArrDelay"] - np.mean(jointDf["ArrDelay"])) <
                      3.5 * np.std(jointDf["ArrDelay"])]
    jointDf = jointDf[~jointDf["ArrDelay"].isna()]
    removed = entries_before - jointDf.shape[0]
    print("Removed " + str(removed) + " rows in cleanup")


def add_arrival_departure_bins(jointDf):
    two_hour_bins = np.linspace(0, 2400, num=13)
    two_hour_labels = np.rint(np.linspace(0, 22, 12))
    jointDf["DepBin"] = pd.cut(jointDf['CRSDepTime'], bins=two_hour_bins,
                               labels=two_hour_labels)
    jointDf["ArrBin"] = pd.cut(jointDf['CRSArrTime'], bins=two_hour_bins,
                               labels=two_hour_labels)
    jointDf.drop(['CRSDepTime', 'CRSArrTime'], axis=1)


def get_dummies(jointDf):
    return pd.get_dummies(jointDf, columns=['DayOfWeek', 'Reporting_Airline', 'Dest', 'Origin'],
                          prefix=['weekday', 'airline', 'DestAirport', 'OriginAirport'])

def drop_features(jointDf):
    return jointDf.drop(['OriginCityName', 'OriginState', 'DestCityName', 'DestState'], axis=1)


def make_flight_date_canonical(jointDf):
    jointDf["CanonicalFlightDate"] = pd.DataFrame(pd.to_datetime(jointDf["FlightDate"], errors="coerce")).values.astype(
        float) / 10 ** 10
    jointDf["CanonicalFlightDate"] = jointDf["CanonicalFlightDate"] - np.min(jointDf["CanonicalFlightDate"])



def add_is_same_state(jointDf):
    jointDf["SameState"] = jointDf["OriginState"] == jointDf["DestState"]


def add_weather_data(jointDf, path_to_weather):
    weather_data = pd.read_csv(path_to_weather)
    weather_data['day'] = pd.to_datetime(weather_data['day'])
    #TODO match jointDf["FlightDate"] to weather_data["date"]
    #TODO match jointDf["Origin" or "Dest"] to weather_data["station"]
    #TODO create columns in jointDf based on weather_data columns:
    pd.merge(jointDf, weather_data, left_on="FlightDate", right_on="day")

    # jointDf["snowed"]

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
