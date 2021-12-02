from sklearn.preprocessing import StandardScaler
from numpy.core.numeric import full
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


titanic = pd.read_csv('3章/datasets/train.csv')

titanic = titanic.drop(['Cabin', 'Name', 'Ticket'], axis=1)

for i, ind in enumerate(titanic['Embarked'].isnull()):
    if ind:
        titanic = titanic.drop(i)

data = titanic.drop('Survived', axis=1)
target = titanic['Survived']

data_num = data[['Age', 'SibSp', 'Parch', 'Fare']]
data_cat = data.drop(['Age', 'SibSp', 'Parch', 'Fare', 'PassengerId'], axis=1)

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder())
])

num_attributes = list(data_num)
cat_attributes = list(data_cat)

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attributes),
    ('cat', cat_pipeline, cat_attributes)
])

prepared_data = full_pipeline.fit_transform(data)

param_grid = [{'kernel': ['rbf', 'linear'],
               'C': [0.001, 0.01, 0.1, 1, 10],
               'gamma': [0.01, 0.1, 1, 10, 100]}]

grid = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-2)
grid.fit(prepared_data, target)
print(grid.best_score_)
print(grid.best_params_)