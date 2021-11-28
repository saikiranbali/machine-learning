import pandas as pd
#I am importin pandas

from sklearn.tree import DecisionTreeClassifier
# I am importing decision tree from scikit learn package


import joblib
#importing joblib to dump and load our models so that we dont need to create again and train our models. we can use existing pre-trained model


joblib.load('music-suggestion.joblib')
#loads my AI or previously created model and predicts my values.

pred = model.predict([[21,1],[22,0]])
#adding new columns ad 21 age and 1 as male
#adding new columns as 22 age and 0 as female to get some sample predictions
pred
#gives me the values like previously how I used to get the output from my code