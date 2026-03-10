import matplotlib.pyplot as plt

from src.preprocess import load_data, preprocess_data
from src.train_model import train_sales_model
from src.forecast import forecast_sales


def main():

    # Load data
    df = load_data("data/sales_data.csv")

    # Preprocess data
    df = preprocess_data(df)

    # Train model
    model, metrics, X_test, y_test, predictions = train_sales_model(df)

    print("Model Evaluation Metrics")
    print(metrics)

    # Plot results
    plt.figure(figsize=(8,5))
    plt.scatter(range(len(y_test)), y_test, label="Actual Sales")
    plt.scatter(range(len(predictions)), predictions, label="Predicted Sales")
    plt.legend()
    plt.title("Sales Prediction vs Actual")
    plt.show()

    # Forecast future sales
    future_date = "2023-02-01"

    predicted_sales = forecast_sales(model, future_date)

    print(f"Predicted sales for {future_date}: {predicted_sales}")


if __name__ == "__main__":
    main()