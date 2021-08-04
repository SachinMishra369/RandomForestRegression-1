#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading the dataset

dataset=pd.read_csv('salary.csv')
#reading the dataset
x=dataset.iloc[:,1:-1].values

#reading the dataset
y=dataset.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
#3 Fitting the Random Forest Regression Model to the dataset
# Create RF regressor here
from sklearn.ensemble import RandomForestRegressor 
#Put 10 for the n_estimators argument. n_estimators mean the number #of trees in the forest.
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(x,y)
print("Salary of employee having 6.5 years of experience is ",regressor.predict([[6.5]]))

#4 Visualising the Regression results (for higher resolution and #smoother curve)
X_grid = np.arange(min(x), max(x), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Check It (Random Forest Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()