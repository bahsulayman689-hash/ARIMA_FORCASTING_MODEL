import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings as wr
import itertools
wr.filterwarnings("ignore")

data = pd.read_csv("daily-min-temperatures.csv")
print(data.head())
data["Temp"] = pd.to_numeric(data['Temp'], errors='coerce')
ts = data["Temp"]
ts.plot(title="Daily minimum temperatures in Melbourne")
plt.show()
"""
dropna(): Removes the missing values created after differencing.
If the p-value is greater than 0.05, the series is non-stationary and differencing is needed.


"""
ts = ts.dropna()
result = adfuller(ts)
print("ADF Statistic: %f" % result[0])
print("p-value: %f" % result[1])
if result[1] > 0.05:
     print("Series is non stationary; differencing is needed.")
else:
    print("Series is stationary; no differencing needed.")
    """
    ts.diff(): Computes the first-order difference of the series to remove trends.
dropna(): Removes the missing values created after differencing.
plot(): Visualizes the differenced series
    
    """
ts_diff = ts.diff().dropna()

ts_diff.plot(title='Differenced Series')
plt.show()

result_diff = adfuller(ts_diff)
print('ADF Statistic (differenced): %f' % result_diff[0])
print('p-value (differenced): %f' % result_diff[1])
plot_acf(ts_diff)
plt.show()

plot_pacf(ts_diff)
plt.show()

p = range(0, 4)
d = range(0, 3)
q = range(0, 4)
pdq = list(itertools.product(p, d, q))

best_aic = np.inf
best_order = None
best_model = None

for order in pdq:
    try:
        model = ARIMA(ts, order=order)
        results = model.fit()
        if results.aic < best_aic:
            best_aic = results.aic
            best_order = order
            best_model = results
    except:
        continue

print(f'Best ARIMA order: {best_order} with AIC: {best_aic}')
final_model = ARIMA(ts, order=best_order)
results = final_model.fit()

forecast_values = results.forecast(steps=10)
plt.plot(ts[-10:].index, ts[-10:], label='Actual', color='blue', linestyle='-', marker='o')
plt.plot(forecast_values.index, forecast_values, label='Forecast', color='red', linestyle='--', marker='x')

plt.title('ARIMA Forecast: Actual vs Forecast', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
