import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

#For datas
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#Create a new object and use it
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x, y)

#Predict new value
z = regressor.predict([[6.5]])
print(z)

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('True or false old salary')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()