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
from sklearn.model_selection import train_test_split
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.linear_model import LassoCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

CSV_PATH = "data/train_data.csv"
disqualify_weather = False


class FlightPredictor:
    def __init__(self, path_to_weather=None):
        """
        Initialize an object from this class.
        @param path_to_weather: The path to a csv file containing weather data.
        """
        random.seed(5)
        raw_data = pd.read_csv(CSV_PATH)
        self.path_to_weather = path_to_weather
        X = raw_data.drop(["ArrDelay", "DelayFactor"], axis=1)
        y = raw_data[["ArrDelay", "DelayFactor"]]
        x_train, self.x_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2)

        train = self.clean_up_train(path_to_weather, x_train, y_train)
        self._cols_name = train.columns.values.tolist()
        self.y_train_regression = train.loc[:, "ArrDelay"].to_frame()
        self.y_train_classification = train.loc[:, "DelayFactor"].to_frame()
        self.x_train = train.drop(["ArrDelay", "DelayFactor"], axis=1)
        self._cols_name = self.x_train.columns.values.tolist()

        # regression
        self._lasso_regression = LassoCV(cv=5, random_state=0).fit(self.x_train, self.y_train_regression.values.ravel())

        # classification
        self._classification_model = OneVsRestClassifier(DecisionTreeClassifier(max_depth=11))
        self._classification_model.fit(self.x_train, self.y_train_classification)

    def predict(self, x):
        """
        Recieves a pandas DataFrame of shape (m, 15) with m flight features, and predicts their
        delay at arrival and the main factor for the delay.
        @param x: A pandas DataFrame with shape (m, 15)
        @return: A pandas DataFrame with shape (m, 2) with your prediction
        """
        df = self.clean_up_test(x)

        # regression
        regression = self._lasso_regression.predict(df)
        # classification
        classification = self._classification_model.predict(df)
        prediction = pd.DataFrame({'PredArrDelay': regression, 'PredDelayFactor': classification})
        prediction['PredDelayFactor'] = prediction['PredDelayFactor'].apply(self.get_label)
        return prediction

    def score(self):
        m = len(self.y_test)
        y_hat = self._classification_model.predict(self.x_test)
        accuracy = np.sum(self.y_test == y_hat) / m
        print(accuracy)
        print(self._lasso_regression.score(self.x_test,
                                           self.y_test.values.reval()))

    def clean_up_train(self, path_to_weather, X, y):
        joint_df = X.join(y)
        joint_df = self.factorize_delay(joint_df)
        joint_df = remove_outliers(joint_df)
        joint_df = clean_up_data(joint_df, path_to_weather)
        return joint_df

    def clean_up_test(self, X):
        joint_df = clean_up_data(X, self.path_to_weather)
        test_col_names = joint_df.columns.values.tolist()
        len_col = len(self._cols_name)
        for i in range(len_col):
            if self._cols_name[i] not in test_col_names:
                joint_df.insert(loc=i, column=self._cols_name[i], value=np.zeros(X.shape[0]))
        return joint_df

    def factorize_delay(self, joint_df):
        # fix DelayFactor to numeric
        self.delay_factor = joint_df['DelayFactor'].factorize()
        joint_df['DelayFactor'] = self.delay_factor[0]
        return joint_df

    def get_label(self, row):
        if row != -1:
            return self.delay_factor[1][row]
        return np.nan

def cv_dt(X, y):
    depth = []
    for i in range(3, 20):
        clf = DecisionTreeClassifier(max_depth=i)
        scores = cross_val_score(estimator=clf, X=X, y=y, cv=7, n_jobs=4)
        depth.append(scores.mean())
    idx = np.argmax(np.asarray(depth))
    return idx + 3


def clean_up_data(joint_df, path_to_weather):
    joint_df = add_arrival_departure_bins(joint_df)
    joint_df = make_flight_date_canonical(joint_df)
    joint_df = add_is_same_state(joint_df)
    if path_to_weather and not disqualify_weather:
        add_weather_data(joint_df, path_to_weather)
    joint_df = get_dummies(joint_df)
    joint_df = cross_holidays(joint_df)
    joint_df = drop_features(joint_df)
    return joint_df


def visualize(df):
    # Airport vs. delay
    df["is_delayed"] = (df['DelayFactor'] != -1).astype(int)

    p1 = (ggplot(df, aes(x='Distance', y='ArrDelay', color='is_delayed')) + geom_point() +
          labs(title="Distance vs. delay time, correlation: {}".format(df['ArrDelay'].corr(df['Distance']))))
    print(p1)

    p2 = (ggplot(df, aes(x='CRSElapsedTime', y='ArrDelay', color='is_delayed')) + geom_point() +
          labs(
              title="CRSElapsedTime vs. delay time, correlation: {}".format(df['ArrDelay'].corr(df['CRSElapsedTime']))))
    print(p2)

    p3 = (ggplot(df, aes(x='CanonicalFlightDate', y='ArrDelay', color='is_delayed')) + geom_point() +
          labs(title="CanonicalFlightDate vs. delay time, correlation: {}".format(
              df['ArrDelay'].corr(df['CanonicalFlightDate']))))
    print(p3)

    p4 = (ggplot(df, aes(x='DepBin', y='ArrDelay', color='is_delayed')) + geom_point() +
          labs(title="DepBin vs. delay time, correlation: {}".format(df['ArrDelay'].corr(df['DepBin']))))
    print(p4)

    p5 = (ggplot(df, aes(x='ArrBin', y='ArrDelay', color='is_delayed')) + geom_point() +
          labs(title="ArrBin vs. delay time, correlation: {}".format(df['ArrDelay'].corr(df['ArrBin']))))
    print(p5)

    numeric_columns = pd.concat([df['ArrBin'], df['DepBin'], df['CanonicalFlightDate'], df['CRSElapsedTime'],
                                 df['Distance'], df['ArrDelay']], axis=1)
    cormat = numeric_columns.corr(method='pearson').round(2)
    cormat.index.name = 'variable2'
    cormat.reset_index(inplace=True)
    melted_cormat = pd.melt(cormat, id_vars=['variable2'])
    p_corr = ggplot(melted_cormat, aes(x='variable', y='variable2', fill='value')) + geom_tile() \
             + labs(title="Numeric Columns Correlation Table")
    print(p_corr)

    # df_dest_delayed = df.groupby('Dest').agg({'is_delayed': ['mean']})


def remove_outliers(joint_df):
    entries_before = joint_df.shape[0]
    joint_df = joint_df[abs(joint_df["ArrDelay"] - np.mean(joint_df["ArrDelay"])) <
                        3.5 * np.std(joint_df["ArrDelay"])]
    joint_df = joint_df[~joint_df["ArrDelay"].isna()]
    removed = entries_before - joint_df.shape[0]
    print("Removed " + str(removed) + " rows in cleanup")
    return joint_df


def add_arrival_departure_bins(joint_df):
    hour_bins = np.linspace(0, 2400, num=25)
    hour_labels = np.rint(np.linspace(0, 23, 24))
    joint_df["DepBin"] = pd.cut(joint_df['CRSDepTime'], bins=hour_bins, labels=hour_labels)
    joint_df["ArrBin"] = pd.cut(joint_df['CRSArrTime'], bins=hour_bins, labels=hour_labels)
    joint_df.drop(['CRSDepTime', 'CRSArrTime'], axis=1)
    return joint_df


def get_dummies(joint_df):
    # return pd.get_dummies(joint_df, columns=['DayOfWeek', 'Reporting_Airline', 'Dest', 'Origin'],
    #                       prefix=['weekday', 'airline', 'DestAirport', 'OriginAirport'])
    return pd.get_dummies(joint_df, columns=['DayOfWeek'], prefix=['weekday'])


def drop_features(joint_df):
    # return joint_df.drop(['Tail_Number', 'OriginCityName', 'OriginState', 'DestCityName',
    #                       'DestState', 'Flight_Number_Reporting_Airline', 'CRSElapsedTime', 'FlightDate'], axis=1)
    return joint_df.drop(['Tail_Number', 'OriginCityName', 'OriginState', 'DestCityName',
                          'DestState', 'Flight_Number_Reporting_Airline', 'CRSElapsedTime', 'FlightDate',
                          'CanonicalFlightDate', 'Reporting_Airline', 'Dest', 'Origin'], axis=1)


def make_flight_date_canonical(joint_df):
    joint_df["CanonicalFlightDate"] = pd.DataFrame(
        pd.to_datetime(joint_df["FlightDate"], errors="coerce")).values.astype(
        float) / 10 ** 10
    joint_df["CanonicalFlightDate"] = joint_df["CanonicalFlightDate"] - np.min(joint_df["CanonicalFlightDate"])
    return joint_df


def add_is_same_state(joint_df):
    joint_df["SameState"] = joint_df["OriginState"] == joint_df["DestState"]
    return joint_df


def add_weather_data(joint_df, path_to_weather):
    weather_data = pd.read_csv(path_to_weather,
                               index_col=["day", "station"],
                               usecols=["day", "station", "precip_in", "avg_wind_speed_kts",
                                        "avg_wind_drct", "snow_in"])
    joint_df["FlightDate"] = joint_df["FlightDate"].apply(convert_date)  # change the date of our
    # df to match the weather data

    # create a list of tuples to be used as multi-index keys for weather_data
    keys = [tuple(x) for x in joint_df[["FlightDate", "Origin"]].to_numpy()]

    rows_with_weather_data = pd.Series(keys).isin(weather_data.index).array
    percentage = 1 - (rows_with_weather_data.sum() / joint_df.shape[0])
    if percentage > 0.05:
        global disqualify_weather
        disqualify_weather = True
        return
    joint_df = joint_df[rows_with_weather_data]
    keys = pd.Series(keys)[rows_with_weather_data].tolist()
    weather_data_rows = weather_data.loc[keys]
    weather_data_rows = weather_data_rows.reset_index(drop=True)
    p = pd.concat([joint_df, weather_data_rows], axis=1)
    p.dropna(axis=0, inplace=True)
    # TODO clean outlier afterwards
    return remove_weather_outliers(p)


def remove_weather_outliers(joint_df):
    for col_name in ["snow_in", "precip_in", "avg_wind_speed_kts"]:
        mean = joint_df[((joint_df[col_name] != 'None') & (
                joint_df[col_name] != '-99') & ~joint_df[
            col_name].isna())][col_name].astype('float').mean()

        joint_df[col_name] = joint_df[col_name].astype('string')
        joint_df[col_name].loc[joint_df[col_name] == 'None'] = str(mean)
        joint_df[col_name].loc[joint_df[col_name].isna()] = str(mean)

        joint_df[col_name] = joint_df[col_name].astype('float')
        joint_df[col_name].loc[joint_df[col_name].astype('float') < 0] = mean
        joint_df[col_name].loc[abs(joint_df[col_name].astype('float') - mean) <
                               3.5 * np.std(joint_df[col_name].astype('float'))] = mean
        joint_df[col_name][joint_df['FlightDate'].str.contains("-0[3-9]-", na=False)] = 0

    return joint_df


def cross_holidays(joint_df):
    joint_df['FlightDate'] = pd.to_datetime(joint_df['FlightDate'], infer_datetime_format=True)
    cal = calendar()
    holidays = cal.holidays(start=joint_df['FlightDate'].min(), end=joint_df[
        'FlightDate'].max())
    joint_df["is_holiday"] = joint_df["FlightDate"].isin(holidays)
    return joint_df


def convert_date(str):
    strList = str.split("-")
    strList[0] = strList[0][2:4]
    return strList[2] + "-" + strList[1] + "-" + strList[0]
