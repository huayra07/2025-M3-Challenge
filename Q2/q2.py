import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def main():
    # 1. Read CSV and check columns
    df = pd.read_csv('Copy of for_all2 - Shelby County Annual Electricity Consumption.csv')
    print("Columns in the dataset:", df.columns)

    # 2. Convert 'Year' to datetime, then sort chronologically
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    df = df.sort_values('Year')

    # 3. Set 'Year' as the index
    df.set_index('Year', inplace=True)

    # 4. Provide an annual frequency to the DateTimeIndex
    #    This helps the model interpret the data as yearly.
    df = df.asfreq('YS')  # 'YS' = Year Start

    # 5. Convert consumption column to numeric
    #    Replace commas if necessary, then convert to float.
    df['Consumption (kWh)'] = pd.to_numeric(
        df['Consumption (kWh)'].replace({',': ''}, regex=True),
        errors='coerce'
    )

    # 6. Drop rows where the numeric conversion failed (NaN)
    df.dropna(inplace=True)

    # 7. Extract the time series
    ts = df['Consumption (kWh)']

    # Quick check of the time series after cleaning
    print("\nFirst few rows of cleaned data:\n", ts.head())
    print("\nLast few rows of cleaned data:\n", ts.tail())

    # 8. Fit Holt's Exponential Smoothing with a damped trend
    model = ExponentialSmoothing(
        ts,
        trend='add',
        seasonal=None,
        damped_trend=True
    ).fit(optimized=True)

    print("\nFitted model parameters:", model.params)

    # 9. Forecast the next 20 years
    forecast_steps = 20
    forecast_values = model.forecast(forecast_steps)

    # Because we forced an annual frequency, we can build a date range accordingly.
    # We'll start from one year after the last date in ts.
    last_date = ts.index[-1]
    forecast_years = pd.date_range(start=last_date + pd.offsets.YearBegin(1),
                                   periods=forecast_steps,
                                   freq='YS')

    # 10. Plot historical data + forecast
    plt.figure(figsize=(10, 6))

    # Plot the historical series
    plt.plot(ts.index.year, ts.values, marker='o', label='Historical Consumption')

    # Plot the forecast
    plt.plot(forecast_years.year, forecast_values, marker='o',
             linestyle='--', color='red', label='Forecasted Consumption')

    plt.title('Forecast of Electricity Consumption for the Next 20 Years')
    plt.xlabel('Year')
    plt.ylabel('Electricity Consumption (kWh)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 11. Print forecasted values
    forecast_df = pd.DataFrame({
        'Year': forecast_years.year,
        'Forecasted Consumption': forecast_values.values
    })
    print("\nForecasted Values:\n", forecast_df)

if __name__ == "__main__":
    main()
