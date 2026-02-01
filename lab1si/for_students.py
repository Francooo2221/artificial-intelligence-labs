import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

X_train = np.vstack([np.ones(len(x_train)), x_train]).T
theta_best = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
print("Theta:", theta_best)

y_pred = theta_best[0] + theta_best[1] * x_test
squared_errors = (y_test - y_pred) ** 2
mse = np.mean(squared_errors)
print("MSE:", mse)

x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

meanX = np.mean(x_train)
std_devX = np.std(x_train)
meanY = np.mean(y_train)
std_devY = np.std(y_train)

x_train_standardized = (x_train - meanX) / std_devX
y_train_standardized = (y_train - meanY) / std_devY
x_test_standardized = (x_test - meanX) / std_devX
y_test_standardized = (x_test - meanY) / std_devY

X_train_standardized = np.vstack([np.ones(len(x_train_standardized)), x_train_standardized]).T
alpha = 0.001
iterations = 10000
m = len(y_train)
theta = np.zeros(2)
def compute_cost(X, y, theta):
    predictions = X.dot(theta)
    cost = (1 / (2*m)) * np.sum((predictions - y) ** 2)
    return cost
for i in range(iterations):
    predictions = X_train_standardized.dot(theta)
    gradient = (1 / m) * X_train_standardized.T.dot(predictions - y_train_standardized)
    theta -= alpha * gradient
    cost = compute_cost(X_train_standardized, y_train_standardized, theta)
    if i % 100 == 0:
        print(f"Iteracja {i}, koszt: {cost}")
print("Theta:", theta)

y_pred = theta[0] + theta[1] * x_train_standardized
squared_errors = (y_train_standardized - y_pred) ** 2
mse = np.mean(squared_errors)
print("MSE:", mse)

x = np.linspace(min(x_train_standardized), max(x_train_standardized), 100)
y = float(theta[0]) + float(theta[1]) * x
plt.plot(x, y)
plt.scatter(x_train_standardized, y_train_standardized)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

