import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import metrics


### DATA LOADING ###

DATA_URL = "./admissiondata.csv"

dataset = pd.read_csv(DATA_URL)

x = dataset.iloc[:, 0:6].values
y = dataset.iloc[:, 7].values


### SPLIT DATA ###
# No x-validation?
# 500 entries -> 100 are test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

### POLY TRAINING ###
multi_poly = PolynomialFeatures(degree=2)
x_poly = multi_poly.fit_transform(x_train)
multi_poly.fit(x_poly, y_train)

lin_reg_multi = LinearRegression()
lin_reg_multi.fit(x_poly, y_train)


### FIND MEAN SQUARED ERROR ###
y_predictions = lin_reg_multi.predict(multi_poly.fit_transform(x_test))

print(metrics.mean_squared_error(y_test, y_predictions))
