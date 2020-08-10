import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier


DATA_URL = "./cancer.csv"

### Preprocessing
dataframe = pd.read_csv(DATA_URL)
x = dataframe.iloc[:, 2:].values
y = dataframe.iloc[:, 1].values

x_train, x_test, y_train, y_test = train_test_split(x, y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


### PCA; principal component analysis
# With highly correlated variables, turn them into 1, scalar instead of x,y
# Reduce dimensions!
pca = PCA(n_components=1) 
# Hm, shouldn't this not be based on what we want the number of components to be?
# Surely we don't know how many dimensions can be reduced, which is the whole point of this?
# I suppose its more like pick the 1 most important, but eeeeeeeeh.
x_train_scale = pca.fit_transform(x_train)

plt.scatter(x_train_scale, y_train)
plt.show()


### Gradient boosting
# Make many models, use GradDesc to find best model, take features and combine to god-model
gradient_boost = GradientBoostingClassifier()
gradient_boost.fit(x_train, y_train)

y_preds = gradient_boost.predict(x_test)
print(confusion_matrix(y_test, y_preds))

# Extreme gradient boosting! Woo! Wins kaggle competitions! That's important, apparently!
# I guess it must also be good.
xgboost = XGBClassifier()
xgboost.fit(x_train, y_train)

y_preds = xgboost.predict(x_test)
print(confusion_matrix(y_test, y_preds))

# Kind of a marginal improvement, but an improvement nonetheless
