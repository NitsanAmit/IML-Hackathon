from model import FlightPredictor

if __name__ == '__main__':
    # 'data/all_weather_data.csv'
    f = FlightPredictor('data/all_weather_data.csv')
    print("end init")
    print(f.predict(f.data_to_pred))