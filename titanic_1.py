# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 14:59:46 2022

@author: abhishek
"""

"""
--> Each row represents the data of 1 passenger.
-->Columns represent the features. We have 10 features/ variables in this dataset.

-----> Survival: This variable shows whether the person survived or not. This is our target variable & we have to predict its value. It’s a binary variable. 0 means not survived and 1 means survived.
-----> pclass: The ticket class of passengers. 1st  (upper class), 2nd (middle), or 3rd (lower).
-----> Sex: Gender of passenger
-----> Age: Age (in years) of a passenger
-----> sibsp: The no. of siblings/spouses of a particular passenger who were there on the ship.
-----> parch: The no. of parents/children of a particular passenger who were there on the ship.
-----> ticket: Ticket Number
-----> fare: Passenger fare (like 1st class ticket fare must be greater than 2nd pr 3rd class ticket right)
-----> cabin: Cabin Number
-----> embarked: Port of Embarkation; From where that passenger took the ship. ( C = Cherbourg, Q = Queenstown, S = Southampton)
"""


"""
Importing Libraries & Loading Dataset
"""

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree,svm
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Importing Dataset
train_csv = pd.read_csv("F:/ML-DS-PYTHON/My_projects/Titanic/dataset/train.csv")
test_csv = pd.read_csv("F:/ML-DS-PYTHON/My_projects/Titanic/dataset/test.csv")
gendersubmission_csv = pd.read_csv("F:/ML-DS-PYTHON/My_projects/Titanic/dataset/gender_submission.csv")

# Printing first 10 rows of the dataset
train_csv.head(10)
print('The shape of our training set: %s passengers and %s features'%(train_csv.shape[0],train_csv.shape[1]))
train_csv.info()

# Checking Null Values
train_csv.isnull().sum()


"""
Exploratory Data Analysis
"""

# Creating heatmap
heatmap = sns.heatmap(train_csv[["Survived", "SibSp", "Parch", "Age", "Fare"]].corr(), annot = True)
sns.set(rc={'figure.figsize':(12,10)})

# SibSp – Number of Siblings / Spouses aboard the Titanic
# Finding unique values
train_csv['SibSp'].unique()
bargraph_sibsp = sns.catplot(x = "SibSp", y = "Survived", data = train_csv, kind="bar", height = 8)

# Age Column
ageplot = sns.FacetGrid(train_csv, col="Survived", height = 7)
ageplot = ageplot.map(sns.distplot, "Age")
ageplot = ageplot.set_ylabels("Survival Probability")

# Gender Column
genderplot = sns.barplot(x = "Sex", y = "Survived", data = train_csv)

# Pclass Column
pclassplot = sns.catplot(x = "Pclass", y="Survived", data = train_csv, kind="bar", height = 6)


"""
Data Preprocessing
"""

# Checking null values
train_csv.isnull().sum()

# Age Columns
# Handling Missing Values of Age Column
mean = train_csv["Age"].mean()
std = train_csv["Age"].std()

rand_age = np.random.randint( mean - std, mean + std, size = 177)
age_slice = train_csv["Age"].copy()
age_slice[np.isnan(age_slice)] = rand_age
train_csv["Age"] = age_slice

# Again checking for null values
train_csv.isnull().sum()

# Dropping Columns
col_to_drop = ["PassengerId", "Ticket", "Cabin", "Name"]
train_csv.drop(col_to_drop, axis=1, inplace=True)
train_csv.head(10)

# Converting Categorical Variables to Numeric
genders = {"male":0, "female":1}
train_csv["Sex"] = train_csv["Sex"].map(genders)
ports = {"S":0, "C":1, "Q":2}
train_csv["Embarked"] = train_csv["Embarked"].map(ports)
train_csv.head()

train_csv["Embarked"].fillna(train_csv["Embarked"].value_counts().index[0],inplace=True)

train_csv.isnull().sum()



"""
Building Machine Learning Model
"""


df_train_x = train_csv[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# Target variable column
df_train_y = train_csv[['Survived']]

# Train Test Splitting
x_train, x_test, y_train, y_test = train_test_split(df_train_x, df_train_y, test_size=0.20, random_state=42)


# Random Forest Classifier
print("Random Forest Classifier : \n")
# Creating alias for Classifier
RFC = RandomForestClassifier()
# Fitting the model using training data
RFC.fit(x_train, y_train)
# Predicting on test data
rfc_y_pred = RFC.predict(x_test)
# Calculating Accuracy to compare all models
rfc_accuracy = accuracy_score(y_test,rfc_y_pred) * 100
print("accuracy=",rfc_accuracy)


# Logistic Regression
print("Logistic Regression : \n")
LR = LogisticRegression()
LR.fit(x_train, y_train)
lr_y_pred = LR.predict(x_test)
lr_accuracy = accuracy_score(y_test,lr_y_pred)*100
print("accuracy=",lr_accuracy)


# K-Neighbor Classifier
print("K-Neighbor Classifier : \n")
KNC = KNeighborsClassifier(5)
KNC.fit(x_train, y_train)
knc_y_pred = KNC.predict(x_test)
knc_accuracy = accuracy_score(y_test,knc_y_pred)*100
print("accuracy=",knc_accuracy)


# Decision Tree Classifier
print("Decision Tree Classifier : \n")
DTC = tree.DecisionTreeClassifier()
DTC = DTC.fit(x_train, y_train)
dtc_y_pred = DTC.predict(x_test)
dtc_accuracy = accuracy_score(y_test,dtc_y_pred)*100
print("accuracy=",dtc_accuracy)


# Support Vector Machine
print("Support Vector Machine : \n")
SVM = svm.SVC()
SVM.fit(x_train, y_train)
svm_y_pred = SVM.predict(x_test)
svm_accuracy = accuracy_score(y_test,svm_y_pred)*100
print("accuracy=",svm_accuracy)


# Gaussian Naive Bayes
print("Gaussian Naive Bayes : \n")
GNB = GaussianNB()
GNB.fit(x_train, y_train)
GNB_y_pred = GNB.predict(x_test)
gnb_accuracy = accuracy_score(y_test, GNB_y_pred)*100
print("accuracy=",gnb_accuracy)

# Linear SVC
print("Linear SVC : \n")
linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
linear_svc_y_pred = linear_svc.predict(x_test)
linear_svc_accuracy = accuracy_score(y_test, linear_svc_y_pred)*100
print("accuracy=",linear_svc_accuracy)

# Perceptron
print("Perceptron : \n")
perceptron = Perceptron()
perceptron.fit(x_train, y_train)
perceptron_y_pred = perceptron.predict(x_test)
perceptron_accuracy = accuracy_score(y_test, perceptron_y_pred)*100
print("accuracy=",perceptron_accuracy)


# Stochastic Gradient Descent
print("Stochastic Gradient Descent : \n")
SGD = SGDClassifier()
SGD.fit(x_train, y_train)
SGD_y_pred = SGD.predict(x_test)
sgd_accuracy = accuracy_score(y_test, SGD_y_pred)*100
print("accuracy=",sgd_accuracy)

# Gradient Boosting Classifier
print("Gradient Boosting Classifier : \n")
GBC = GradientBoostingClassifier()
GBC.fit(x_train, y_train)
GBC_y_pred = GBC.predict(x_test)
gbc_accuracy = accuracy_score(y_test, GBC_y_pred)*100
print("accuracy=",gbc_accuracy)


# Accuracy Scores of All Classifiers
print("Accuracy of Random Forest Classifier =",rfc_accuracy)
print("Accuracy of Logistic Regressor =",lr_accuracy)
print("Accuracy of K-Neighbor Classifier =",knc_accuracy)
print("Accuracy of Decision Tree Classifier = ",dtc_accuracy)
print("Accuracy of Support Vector Machine Classifier = ",svm_accuracy)
print("Accuracy of Gaussian Naive Bayes Classifier = ",gnb_accuracy)
print("Accuracy of Linear SVC Classifier = ",linear_svc_accuracy)
print("Accuracy of Perceptron Classifier = ",perceptron_accuracy)
print("Accuracy of Stochastic Gradient Descent Classifier = ",sgd_accuracy)
print("Accuracy of Gradient Boosting Classifier = ",gbc_accuracy)


# Let's compare the accuracies of each model!

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [svm_accuracy, knc_accuracy, lr_accuracy, 
              rfc_accuracy, gnb_accuracy, perceptron_accuracy, linear_svc_accuracy, dtc_accuracy,
              sgd_accuracy, gbc_accuracy]})
models.sort_values(by='Score', ascending=False)

"""
Final Prediction with Machine Learning Model
"""


# predicting using test data set
test_csv.head(10)

test_csv.info()

# checking nullvalues in test data
test_csv.isnull().sum()

# handling Missing values in test data

# Replacing missing values of age column
mean = test_csv["Age"].mean()
std = test_csv["Age"].std()
rand_age = np.random.randint(mean-std, mean+std, size = 86)
age_slice = test_csv["Age"].copy()
age_slice[np.isnan(age_slice)] = rand_age
test_csv["Age"] = age_slice

# Replacing missing value of Fare column
test_csv['Fare'].fillna(test_csv['Fare'].mean(), inplace=True)

test_csv.isnull().sum()
col_to_drop = ["PassengerId", "Ticket", "Cabin", "Name"]
test_csv.drop(col_to_drop, axis=1, inplace=True)
test_csv.head(10)
genders = {"male":0, "female":1}
test_csv["Sex"] = test_csv["Sex"].map(genders)

ports = {"S":0, "C":1, "Q":2}
test_csv["Embarked"] = test_csv["Embarked"].map(ports)

test_csv.head()



"""
Machine Learning Project Submission File

"""
x_test = test_csv
y_pred = RFC.predict(x_test)
originaltest_data = pd.read_csv('F:/ML-DS-PYTHON/My_projects/Titanic/dataset/test.csv')
submission = pd.DataFrame({"PassengerId": originaltest_data["PassengerId"],"Survived": y_pred})
submission.to_csv('submission.csv', index=False) 
submission.head(20)