import pandas as pd


def forecast_sales(model, date):

    date = pd.to_datetime(date)

    data = {
        "day": [date.day],
        "month": [date.month],
        "year": [date.year],
        "day_of_week": [date.dayofweek]
    }

    df = pd.DataFrame(data)

    prediction = model.predict(df)

    return prediction[0]