from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import matplotlib.pyplot as plt

DATA_URL = "./fathersonheight.csv"

if __name__ == "__main__":
    ### DATA LOADING ###
    # Already preprocessed!
    dataset = pd.read_csv(DATA_URL)
    
    x = dataset['Father'].values.reshape(-1, 1) # This is important for the matrix math
    y = dataset['Son'].values
    
    
    ### LINEAR REGRESSION ###
    lin_reg = LinearRegression()
    lin_reg.fit(x, y)
    
    ### PLOTTING DATA ###
    # Orig data
    plt.scatter(x, y, color="blue")
    # Predicted results
    plt.plot(x, lin_reg.predict(x), color="red", lineWidth=4)
    
    
    ### POLYNOMIAL REGRESSION ###
    poly = PolynomialFeatures(degree=4)
    x_poly = poly.fit_transform(x)
    poly.fit(x_poly, y)
    
    lin_reg_poly = LinearRegression()
    lin_reg_poly.fit(x_poly, y)
    
    plt.plot(x, lin_reg_poly.predict(poly.fit_transform(x)), color="yellow", lineWidth=2)
    
    
    plt.show()