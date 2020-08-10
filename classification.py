import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

DATA_URL = "./cancer.csv"
LABELS = "id,diagnosis,radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave points_worst,symmetry_worst,fractal_dimension_worst".split(",")

# 32 labels. LABELS[0] is ID, which is irrelevant, and LABELS[1] is diagnosis, which is what we're looking to predict
dataset = pd.read_csv(DATA_URL)

x = dataset.iloc[:, 2:].values
y = dataset.iloc[:, 1].values

# Train/Test split
# Default test_size = 0.25
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Scale data to same...uh, scale. Z-score it. X_new = X_i - X_mean / stdev
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# Logistic regression. Binary value that effectively classifies.
logistic_classifier = LogisticRegression(solver='lbfgs')
logistic_classifier.fit(x_train, y_train)

y_preds = logistic_classifier.predict(x_test)
# Confusion matrix will show accuracy, but also which kinds of misclassifications there are.
# [[TRUE_POS, FALSE_POS],
#  [FALSE_NEG, TRUE_NEG]]
print(confusion_matrix(y_test, y_preds))

#y_preds2 = logistic_classifier.predict(x_train) # How overfitted are we?
#print(confusion_matrix(y_train, y_preds2))


# Support vector machines. Basically maximize distance between groupings of data.
# Usable for classification or regression, more popularly the former.
# Typically work better than LogReg, but slower, so for less data, use LogReg

svm = SVC(kernel='rbf') # Is already the default kernel.
svm.fit(x_train, y_train)

y_preds = svm.predict(x_test)
print(confusion_matrix(y_test, y_preds))


# Decision trees. Basically a Y/N flowchart to finally classify something. Can get complex.
# criterion is how model self-evaluates
tree = DecisionTreeClassifier(criterion="entropy")
tree.fit(x_train, y_train)

y_preds = tree.predict(x_test)
print(confusion_matrix(y_test, y_preds))

# Wew, very inaccurate. One tree alone is imprecise, but want some 'Wisdom of the Crowd'
# Make a random forest; ensemble learning. We Are Legion! Sweet. I'm hype.
# Use 100 trees. These are still prone to overfitting, so keep the test set handy.
forest = RandomForestClassifier(n_estimators=100, criterion='entropy')
forest.fit(x_train, y_train)

y_preds = forest.predict(x_test)
print(confusion_matrix(y_test, y_preds))

# Can see that its still not great; SVMs win out for the largest array of problems.
