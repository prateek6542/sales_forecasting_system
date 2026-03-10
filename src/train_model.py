from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def train_sales_model(df):
    
    features = ['day', 'month', 'year', 'day_of_week']
    target = 'sales'
    
    x = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse
    }

    return model, metrics, X_test, y_test, predictions